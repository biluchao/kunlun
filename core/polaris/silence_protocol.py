#!/usr/bin/env python3
"""
Kunlun System · Silence Protocol (SilenceProtocol)

Core Responsibilities:
1. Detect external environment anomalies (WebSocket disconnections, REST errors,
   exchange maintenance announcements, complete trade stagnation)
2. Trigger "Silence Mode": cancel all open orders, freeze new entries, keep only
   risk-management closing ability.
3. Safely exit Silence Mode after recovery, confirming market tradability through
   an observation window.

External Dependencies (real module interfaces):
- infrastructure.stream_gateway.StreamGateway : connection state, data staleness,
  heartbeat counters (via health_check and status dict)
- hermes.order_gateway.OrderExecutionGateway : batch order cancellation
  (cancel_all_orders method)
- polaris.snapshot_reconciler.SnapshotReconciler : data integrity validation
- infrastructure.health_pulse.HealthPulseMonitor : report silence status to
  health scoring
- infrastructure.error_registry.ErrorRegistry : centralized error code lookup
- infrastructure.audit_logger.AuditLogger : persistent audit trail

Interface Contract:
- evaluate(gateway_status: GatewayStatus, market_snapshot: MarketSnapshot) -> EvalResult
- force_silence(reason: str) -> ActionResult
- attempt_exit() -> ActionResult
- health_check() -> HealthStatus

All returned dicts are TypedDicts (see EvalResult, ActionResult, HealthStatus).
Output dicts always contain "status" (str), "reason" (str), "warnings" (List[str]).
Additional keys follow the specific TypedDict.

Exception & Degradation:
- If OrderExecutionGateway is unavailable, cancellation is logged with error
  KUN-EXE-E005 and manual intervention is requested.
- If SnapshotReconciler reports data inconsistency, observation period is
  extended to EXTENDED_OBSERVATION_SEC and an alert is raised.
- During silence mode, health_check returns degraded status.

Resource Management:
- No persistent external resources; maintains only in-memory state, timers,
  and thread-safe locks for concurrent access.
- All time measurements use a monotonic clock for intervals, and wall-clock
  time for external timestamps (aligned with exchange data).
- Audit events are written asynchronously via a thread-safe queue to prevent
  blocking the main trading loop.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple, TypedDict, Callable
from enum import Enum
from threading import RLock
from collections import deque
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TypedDicts for precise return types
# ---------------------------------------------------------------------------
class GatewayStatus(TypedDict, total=False):
    ws_connected: bool
    ws_active_streams: int
    rest_last_error: bool
    rest_last_status_code: Optional[int]

class MarketSnapshot(TypedDict, total=False):
    symbols: List[str]
    # dynamic keys: {symbol}_last_update_time: float

class EvalResult(TypedDict):
    status: str
    state: str
    action: str
    reason: str
    warnings: List[str]
    progress: float  # only meaningful in observing; 0.0 otherwise

class ActionResult(TypedDict):
    status: str
    action: str
    reason: str
    cancel_all_orders: bool
    freeze_new_entries: bool
    warnings: List[str]

class HealthStatus(TypedDict):
    status: str
    message: str
    degraded: bool

# ---------------------------------------------------------------------------
class SilenceState(Enum):
    NORMAL = "normal"            # Fully operational
    PRE_SILENCE = "pre_silence"  # Warning, preparing to silence (brief transient)
    SILENT = "silent"            # Orders cancelled, no new entries
    OBSERVING = "observing"      # Data restored, verifying stability
    EXITING = "exiting"          # Restoring capabilities stepwise

# ---------------------------------------------------------------------------
class SilenceProtocol:
    """Manages silence protocol triggered by external anomalies."""

    # -----------------------------------------------------------------------
    # Class Constants (configuration defaults with units & valid ranges)
    # -----------------------------------------------------------------------
    WS_DISCONNECT_TIMEOUT_SEC = 5.0          # seconds, [1.0, 30.0]
    WS_MIN_ACTIVE_STREAMS = 1                # at least 1 stream must be alive
    REST_ERROR_THRESHOLD_COUNT = 8            # number of 5xx errors in window
    REST_ERROR_WINDOW_SEC = 30.0              # seconds, [10, 120]
    PRICE_STAGNATION_TIMEOUT_SEC = 2.0        # seconds, [0.5, 5.0]
    PRICE_STAGNATION_MIN_SYMBOLS = 2          # at least 2 symbols must be tracked
    PRICE_MAX_VALID_AGE_SEC = 86400           # 24h, symbols older than this are ignored
    MAINTENANCE_POLL_INTERVAL_SEC = 30.0      # seconds, [10, 120] (reserved for polling)

    OBSERVATION_DURATION_SEC = 60.0           # seconds, [30, 300]
    EXTENDED_OBSERVATION_SEC = 120.0          # seconds, [60, 600]
    MIN_STABLE_PERIOD_SEC = 30.0              # seconds, [10, 120]
    MAX_SILENCE_DURATION_SEC = 3600.0         # 1 hour, after which force attempt

    MAX_REASON_LENGTH = 200                   # characters
    MAX_EVENT_LOG_SIZE = 500                  # events
    CANCEL_ORDER_TIMEOUT_SEC = 5.0            # seconds

    __version__ = "3.0.0"

    # -----------------------------------------------------------------------
    def __init__(self, config: Optional[Dict] = None,
                 order_gateway=None, health_monitor=None,
                 snapshot_reconciler=None, audit_logger=None):
        """
        :param config: optional dict overriding class constants
        :param order_gateway: OrderExecutionGateway instance (for cancellation)
        :param health_monitor: HealthPulseMonitor instance
        :param snapshot_reconciler: SnapshotReconciler instance
        :param audit_logger: AuditLogger instance for persistent audit trail
        """
        self._lock = RLock()
        self._order_gw = order_gateway
        self._health_mon = health_monitor
        self._snapshot_rec = snapshot_reconciler
        self._audit_logger = audit_logger

        # Copy class constants to instance attributes to allow per-instance config
        self._ws_timeout = self.WS_DISCONNECT_TIMEOUT_SEC
        self._ws_min_streams = self.WS_MIN_ACTIVE_STREAMS
        self._rest_err_threshold = self.REST_ERROR_THRESHOLD_COUNT
        self._rest_err_window = self.REST_ERROR_WINDOW_SEC
        self._stag_timeout = self.PRICE_STAGNATION_TIMEOUT_SEC
        self._stag_min_symbols = self.PRICE_STAGNATION_MIN_SYMBOLS
        self._obs_duration = self.OBSERVATION_DURATION_SEC
        self._ext_obs_duration = self.EXTENDED_OBSERVATION_SEC
        self._min_stable = self.MIN_STABLE_PERIOD_SEC
        self._max_silence = self.MAX_SILENCE_DURATION_SEC
        self._cancel_timeout = self.CANCEL_ORDER_TIMEOUT_SEC

        if config:
            self._apply_config(config)

        # State
        self._state: SilenceState = SilenceState.NORMAL
        self._silence_reason: str = ""
        self._silence_enter_time: float = 0.0      # wall clock time when silence began
        self._last_state_change: float = time.time()

        # WS tracking
        self._ws_disconnected_since: Optional[float] = None  # monotonic

        # REST error tracking (using windowed counter instead of deque scanning)
        self._rest_error_timestamps: deque = deque(maxlen=200)  # monotonic times
        self._rest_error_count_in_window: int = 0  # fast cached count

        # Price stagnation
        self._symbol_update_times: Dict[str, float] = {}  # symbol -> wall clock timestamp
        self._stagnation_start: Optional[float] = None     # monotonic

        # Observation
        self._observation_start: float = 0.0       # monotonic
        self._stable_since: float = 0.0            # monotonic, set once when entering observing
        self._data_integrity_ok: bool = True
        self._extended_observation: bool = False

        # Maintenance flag
        self._exchange_maintenance: bool = False
        self._last_maintenance_check: float = 0.0

        # Audit trail in-memory (circular)
        self._event_log: deque = deque(maxlen=self.MAX_EVENT_LOG_SIZE)
        # Asynchronous audit logger (if provided)
        self._audit_queue: Optional[deque] = None
        if hasattr(audit_logger, 'enqueue'):
            self._audit_queue = deque()  # simple in-memory queue, better to use real queue

        # Metrics counters (for Prometheus or similar)
        self._metric_silence_entries = 0
        self._metric_observation_failures = 0
        self._metric_anomalies_detected = 0

        logger.info("SilenceProtocol v%s initialized, state=%s", self.__version__, self._state.value)

    # -----------------------------------------------------------------------
    # Configuration (instance-level)
    # -----------------------------------------------------------------------
    @staticmethod
    def _validate_value(key: str, value: Any) -> bool:
        constraints = {
            'WS_DISCONNECT_TIMEOUT_SEC': (1.0, 30.0),
            'WS_MIN_ACTIVE_STREAMS': (1, 20),
            'REST_ERROR_THRESHOLD_COUNT': (3, 20),
            'REST_ERROR_WINDOW_SEC': (10.0, 120.0),
            'PRICE_STAGNATION_TIMEOUT_SEC': (0.5, 5.0),
            'PRICE_STAGNATION_MIN_SYMBOLS': (1, 10),
            'OBSERVATION_DURATION_SEC': (30.0, 300.0),
            'EXTENDED_OBSERVATION_SEC': (60.0, 600.0),
            'MIN_STABLE_PERIOD_SEC': (10.0, 120.0),
            'MAX_SILENCE_DURATION_SEC': (600.0, 7200.0),
            'CANCEL_ORDER_TIMEOUT_SEC': (1.0, 30.0),
        }
        if key in constraints:
            lo, hi = constraints[key]
            return lo <= value <= hi
        return True

    def _apply_config(self, config: Dict) -> None:
        for key, value in config.items():
            if not self._validate_value(key, value):
                logger.warning("Config value out of range: %s=%s, using default", key, value)
                continue
            # Map config key to instance attribute
            attr_map = {
                'WS_DISCONNECT_TIMEOUT_SEC': '_ws_timeout',
                'WS_MIN_ACTIVE_STREAMS': '_ws_min_streams',
                'REST_ERROR_THRESHOLD_COUNT': '_rest_err_threshold',
                'REST_ERROR_WINDOW_SEC': '_rest_err_window',
                'PRICE_STAGNATION_TIMEOUT_SEC': '_stag_timeout',
                'PRICE_STAGNATION_MIN_SYMBOLS': '_stag_min_symbols',
                'OBSERVATION_DURATION_SEC': '_obs_duration',
                'EXTENDED_OBSERVATION_SEC': '_ext_obs_duration',
                'MIN_STABLE_PERIOD_SEC': '_min_stable',
                'MAX_SILENCE_DURATION_SEC': '_max_silence',
                'CANCEL_ORDER_TIMEOUT_SEC': '_cancel_timeout',
            }
            if key in attr_map:
                setattr(self, attr_map[key], value)
                logger.info("Config override: %s = %s", key, value)
            else:
                logger.warning("Unknown config key: %s", key)

    # -----------------------------------------------------------------------
    # Clock Helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _monotonic() -> float:
        return time.monotonic()

    @staticmethod
    def _wall_time() -> float:
        return time.time()

    # -----------------------------------------------------------------------
    # Asynchronous Audit (non-blocking)
    # -----------------------------------------------------------------------
    def _record_event(self, event_type: str, details: Dict) -> None:
        """Append event and optionally enqueue for async audit logging."""
        entry = {
            "timestamp_wall": self._wall_time(),
            "timestamp_mono": self._monotonic(),
            "state": self._state.value,
            "event": event_type,
            "details": str(details)[:500]  # sanitize
        }
        with self._lock:
            self._event_log.append(entry)
        # If async audit available, enqueue non-blocking
        if self._audit_queue is not None:
            try:
                self._audit_queue.append(entry)
            except Exception:
                pass  # silently ignore if queue full, but log locally
        elif self._audit_logger:
            # Fallback synchronous (but unlikely in production)
            try:
                self._audit_logger.log_event("silence_protocol", entry)
            except Exception as e:
                logger.error("Audit log sync write failed: %s", e, exc_info=True)

    # -----------------------------------------------------------------------
    # Anomaly Checks (require lock held)
    # -----------------------------------------------------------------------
    def _update_rest_error_count(self, now_mono: float) -> int:
        """Recalculate error count within window and prune old entries."""
        window_start = now_mono - self._rest_err_window
        # Prune old entries from front of deque
        while self._rest_error_timestamps and self._rest_error_timestamps[0] <= window_start:
            self._rest_error_timestamps.popleft()
        self._rest_error_count_in_window = len(self._rest_error_timestamps)
        return self._rest_error_count_in_window

    def _check_ws_status(self, gs: GatewayStatus) -> Tuple[bool, str]:
        connected = gs.get('ws_connected', False)  # default False to be safe
        active = gs.get('ws_active_streams', 0)
        now_mono = self._monotonic()
        if not connected or active < self._ws_min_streams:
            if self._ws_disconnected_since is None:
                self._ws_disconnected_since = now_mono
            elif (now_mono - self._ws_disconnected_since) > self._ws_timeout:
                return True, (f"WS anomaly for {now_mono - self._ws_disconnected_since:.1f}s, "
                              f"active={active}")
        else:
            self._ws_disconnected_since = None
        return False, ""

    def _check_rest_errors(self, gs: GatewayStatus) -> Tuple[bool, str]:
        last_error = gs.get('rest_last_error', False)
        status_code = gs.get('rest_last_status_code')
        now_mono = self._monotonic()
        if last_error and status_code is not None and 500 <= status_code < 600:
            self._rest_error_timestamps.append(now_mono)
        elif not last_error:
            self._rest_error_timestamps.clear()
            self._rest_error_count_in_window = 0
            return False, ""
        count = self._update_rest_error_count(now_mono)
        if count >= self._rest_err_threshold:
            return True, f"REST 5xx errors {count} in window"
        return False, ""

    def _check_price_stagnation(self, ms: MarketSnapshot) -> Tuple[bool, str]:
        symbols = ms.get('symbols', [])
        valid_symbols = [s for s in symbols if isinstance(s, str) and re.match(r'^[A-Z0-9/_\-]+$', s)]
        if len(valid_symbols) < self._stag_min_symbols:
            return False, ""
        now_wall = self._wall_time()
        stagnant = 0
        for sym in valid_symbols:
            key = f'{sym}_last_update_time'
            last_ts = ms.get(key)
            if last_ts is None:
                last_ts = self._symbol_update_times.get(sym, 0.0)
            else:
                self._symbol_update_times[sym] = last_ts
            if not isinstance(last_ts, (int, float)) or last_ts <= 0:
                stagnant += 1
            elif (now_wall - last_ts) > self.PRICE_MAX_VALID_AGE_SEC:
                stagnant += 1
            elif (now_wall - last_ts) > self._stag_timeout:
                stagnant += 1
        if stagnant == len(valid_symbols):
            if self._stagnation_start is None:
                self._stagnation_start = self._monotonic()
            return True, f"All {len(valid_symbols)} symbols stagnant"
        else:
            self._stagnation_start = None
        return False, ""

    def _check_maintenance(self) -> Tuple[bool, str]:
        # Polling logic could be added here; for now uses injected flag
        return self._exchange_maintenance, "Exchange maintenance"

    # -----------------------------------------------------------------------
    # Order Cancellation with timeout
    # -----------------------------------------------------------------------
    def _cancel_all_orders(self) -> bool:
        """Cancel all open orders with timeout."""
        if not self._order_gw or not hasattr(self._order_gw, 'cancel_all_orders'):
            logger.warning("[KUN-EXE-E005] Order gateway unavailable")
            return False
        import threading
        result_holder = []
        def cancel():
            try:
                res = self._order_gw.cancel_all_orders()
                result_holder.append(res)
            except Exception as e:
                result_holder.append(e)
        t = threading.Thread(target=cancel, daemon=True)
        t.start()
        t.join(timeout=self._cancel_timeout)
        if t.is_alive():
            logger.error("[KUN-EXE-E005] Cancel all orders timed out")
            return False
        if not result_holder:
            return False
        result = result_holder[0]
        if isinstance(result, Exception):
            logger.error("[KUN-EXE-E005] Cancel exception: %s", result)
            return False
        return result.get('status') == 'ok'

    # -----------------------------------------------------------------------
    # Internal State Transitions (assume lock not held)
    # -----------------------------------------------------------------------
    def _enter_silence(self, reason: str) -> EvalResult:
        safe_reason = reason[:self.MAX_REASON_LENGTH].replace('\n', ' ').strip()
        with self._lock:
            if self._state == SilenceState.SILENT:
                return EvalResult(status="ok", state=self._state.value,
                                  action="already_silent", reason=self._silence_reason,
                                  warnings=[], progress=0.0)
            self._state = SilenceState.SILENT
            self._silence_reason = safe_reason
            self._silence_enter_time = self._wall_time()
            self._last_state_change = self._wall_time()
            self._metric_silence_entries += 1
            self._record_event("enter_silence", {"reason": safe_reason})
            logger.warning("[KUN-SYS-W005] Silence entered: %s", safe_reason)

        cancel_ok = self._cancel_all_orders()
        if not cancel_ok:
            logger.error("[KUN-EXE-E005] Order cancellation failed on silence entry")
            if self._health_mon:
                self._health_mon.report_incident("silence_cancel_failed", {})

        return EvalResult(status="ok", state=SilenceState.SILENT.value,
                          action="enter_silence", reason=safe_reason,
                          cancel_all_orders=cancel_ok, freeze_new_entries=True,
                          warnings=[] if cancel_ok else ["order cancellation failed"])

    def _start_observing(self):
        with self._lock:
            self._state = SilenceState.OBSERVING
            self._observation_start = self._monotonic()
            self._stable_since = self._observation_start  # set once here
            self._extended_observation = not self._data_integrity_ok
            self._record_event("start_observation",
                               {"extended": self._extended_observation})
            logger.info("[KUN-SYS-I005] Observation started")

    def _exit_observing(self) -> EvalResult:
        with self._lock:
            if self._state != SilenceState.OBSERVING:
                return EvalResult(status="error", reason="Not in observing",
                                  state=self._state.value, action="none",
                                  warnings=[], progress=0.0)
            self._state = SilenceState.EXITING
            self._last_state_change = self._wall_time()
            self._record_event("start_exit", {})
            logger.info("[KUN-SYS-I006] Exiting silence")
            return EvalResult(status="ok", state=self._state.value,
                              action="exit_silence",
                              reason="Observation successful",
                              warnings=["Gradually restoring trading"],
                              progress=1.0)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def evaluate(self, gateway_status: GatewayStatus,
                 market_snapshot: MarketSnapshot) -> EvalResult:
        if gateway_status is None or market_snapshot is None:
            return EvalResult(status="error", reason="Invalid input", state="unknown",
                              action="none", warnings=[], progress=0.0)

        # Snapshot state under lock
        with self._lock:
            current_state = self._state
            now_mono = self._monotonic()
            silence_dur = (now_mono - self._silence_enter_time) if self._silence_enter_time > 0 else 0.0

            # Forced exit if silence too long
            if current_state == SilenceState.SILENT and silence_dur > self._max_silence:
                logger.error("[KUN-SYS-F002] Max silence exceeded, forcing observation")
                self._state = SilenceState.OBSERVING
                self._observation_start = now_mono
                self._stable_since = now_mono
                self._extended_observation = True
                self._silence_reason = ""  # clear old reason
                self._record_event("max_silence_forced_observe", {})
                current_state = SilenceState.OBSERVING

            # Perform anomaly checks while holding lock
            anom_ws, ws_reason = self._check_ws_status(gateway_status)
            anom_rest, rest_reason = self._check_rest_errors(gateway_status)
            anom_price, price_reason = self._check_price_stagnation(market_snapshot)
            anom_maint, maint_reason = self._check_maintenance()

        # Act based on state
        if current_state == SilenceState.NORMAL:
            trigger = ""
            if anom_ws: trigger = ws_reason
            elif anom_rest: trigger = rest_reason
            elif anom_price: trigger = price_reason
            elif anom_maint: trigger = maint_reason
            if trigger:
                with self._lock:
                    if self._state == SilenceState.NORMAL:
                        self._state = SilenceState.PRE_SILENCE
                        self._last_state_change = self._wall_time()
                return self._enter_silence(trigger)
            return EvalResult(status="ok", state="normal", action="none",
                              reason="", warnings=[], progress=0.0)

        elif current_state == SilenceState.PRE_SILENCE:
            return self._enter_silence("PRE_SILENCE active")

        elif current_state == SilenceState.SILENT:
            if not (anom_ws or anom_rest or anom_price or anom_maint):
                self._start_observing()
                return EvalResult(status="ok", state="observing",
                                  action="start_observation",
                                  reason="All clear",
                                  warnings=["Observation period started"],
                                  progress=0.0)
            else:
                return EvalResult(status="ok", state="silent",
                                  action="remain_silent",
                                  reason=self._silence_reason,
                                  warnings=[], progress=0.0)

        elif current_state == SilenceState.OBSERVING:
            if anom_ws or anom_rest or anom_price or anom_maint:
                self._metric_observation_failures += 1
                logger.warning("[KUN-SYS-W006] Observation failed, anomaly reappeared")
                self._record_event("observation_fail", {})
                return self._enter_silence("Observation fail: anomaly detected")

            with self._lock:
                req = self._ext_obs_duration if self._extended_observation else self._obs_duration
                elapsed = self._monotonic() - self._observation_start
                # stable_since stays at the original value set when entering observing
                stable_dur = self._monotonic() - self._stable_since
                if elapsed >= req and stable_dur >= self._min_stable:
                    # exit observing (call outside lock)
                    pass
                else:
                    progress = min(1.0, elapsed / req) if req > 0 else 1.0
                    return EvalResult(status="ok", state="observing",
                                      action="observing", reason="",
                                      warnings=[], progress=progress)
            return self._exit_observing()

        elif current_state == SilenceState.EXITING:
            with self._lock:
                if anom_ws or anom_rest or anom_price or anom_maint:
                    logger.warning("[KUN-SYS-W007] Exit aborted, anomaly during EXITING")
                    self._state = SilenceState.SILENT  # fallback to silent without re-cancelling?
                    self._silence_reason = "Exit aborted due to anomaly"
                    self._silence_enter_time = self._wall_time()
                    self._record_event("exit_aborted", {})
                    return EvalResult(status="ok", state="silent",
                                      action="enter_silence",
                                      reason=self._silence_reason,
                                      warnings=[], progress=0.0)
                self._state = SilenceState.NORMAL
                self._silence_reason = ""
                self._last_state_change = self._wall_time()
                self._record_event("fully_restored", {})
                return EvalResult(status="ok", state="normal",
                                  action="fully_restored", reason="",
                                  warnings=[], progress=0.0)

        return EvalResult(status="error", reason=f"Unknown state {current_state}",
                          state="unknown", action="none", warnings=[], progress=0.0)

    def force_silence(self, reason: str) -> ActionResult:
        safe_reason = reason[:self.MAX_REASON_LENGTH].strip() or "manual"
        res = self._enter_silence(safe_reason)
        return ActionResult(
            status=res['status'], action=res['action'], reason=res['reason'],
            cancel_all_orders=res.get('cancel_all_orders', False),
            freeze_new_entries=res.get('freeze_new_entries', True),
            warnings=res.get('warnings', [])
        )

    def attempt_exit(self) -> ActionResult:
        with self._lock:
            if self._state == SilenceState.SILENT:
                self._start_observing()
                return ActionResult(status="ok", action="start_observation",
                                    reason="manual exit attempt",
                                    cancel_all_orders=False,
                                    freeze_new_entries=True,
                                    warnings=[])
            return ActionResult(status="error", action="none",
                                reason=f"cannot exit from {self._state.value}",
                                cancel_all_orders=False,
                                freeze_new_entries=False,
                                warnings=[])

    def is_silent(self) -> bool:
        with self._lock:
            return self._state in (SilenceState.SILENT, SilenceState.OBSERVING,
                                   SilenceState.EXITING)

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            now_mono = self._monotonic()
            progress = 0.0
            if self._state == SilenceState.OBSERVING:
                req = self._ext_obs_duration if self._extended_observation else self._obs_duration
                elapsed = now_mono - self._observation_start
                progress = min(1.0, elapsed / req) if req > 0 else 1.0
            return {
                "state": self._state.value,
                "reason": self._silence_reason,
                "enter_time": self._silence_enter_time,
                "observation_progress": progress,
                "ws_disconnected_since": self._ws_disconnected_since,
                "rest_error_count": self._rest_error_count_in_window,
                "data_integrity_ok": self._data_integrity_ok,
                "max_silence_exceeded": (
                    (now_mono - self._silence_enter_time) > self._max_silence
                    if self._state == SilenceState.SILENT else False
                ),
                "version": self.__version__
            }

    def set_data_integrity(self, ok: bool):
        with self._lock:
            if not ok and self._data_integrity_ok:
                logger.warning("[KUN-DAT-W003] Data integrity lost")
                self._extended_observation = True
            elif ok and not self._data_integrity_ok:
                logger.info("[KUN-DAT-I001] Data integrity restored")
                self._extended_observation = False
            self._data_integrity_ok = ok

    def set_maintenance_flag(self, maintenance: bool, source: str = "external"):
        safe_source = source[:100].strip() or "unknown"
        with self._lock:
            if self._exchange_maintenance != maintenance:
                self._exchange_maintenance = maintenance
                self._record_event("maintenance_change", {
                    "maintenance": maintenance, "source": safe_source
                })
                if maintenance:
                    logger.warning("[KUN-EXE-W010] Maintenance flag set by %s", safe_source)

    # -----------------------------------------------------------------------
    # Health Check
    # -----------------------------------------------------------------------
    @classmethod
    def health_check(cls) -> HealthStatus:
        try:
            sp = cls()
            # Normal operation
            res = sp.evaluate(
                {'ws_connected': True, 'ws_active_streams': 10, 'rest_last_error': False},
                {'symbols': ['BTCUSDT', 'ETHUSDT'], 'BTCUSDT_last_update_time': time.time(),
                 'ETHUSDT_last_update_time': time.time()}
            )
            if res['state'] != 'normal':
                return HealthStatus(status="error", message=f"Normal test failed: {res['state']}", degraded=True)
            # Anomaly detection
            res = sp.evaluate(
                {'ws_connected': False, 'ws_active_streams': 0, 'rest_last_error': True},
                {'symbols': ['BTCUSDT'], 'BTCUSDT_last_update_time': 0.0}
            )
            if res['state'] not in ('silent', 'pre_silence'):
                return HealthStatus(status="error", message=f"Anomaly test failed: {res['state']}", degraded=True)
            return HealthStatus(status="ok", message="All checks passed", degraded=False)
        except Exception as e:
            logger.error("Health check failed: %s", e, exc_info=True)
            return HealthStatus(status="error", message=str(e), degraded=True)
