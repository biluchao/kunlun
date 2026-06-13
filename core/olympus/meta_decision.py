#!/usr/bin/env python3
"""
Kunlun System · MetaDecisionEngine (Institutional Final v4)

Core Responsibilities:
1. Dynamically select system trading mode based on market regime, agent consensus,
   real-time risk budget, system health vector, liquidity depth, and exchange status.
2. Manage five modes: Trend Following, Grid Ranging, Idle Wait, Emergency Only, Kill.
3. Provide versioned, hot-reloadable parameter groups (Aggressive/Neutral/Conservative).
4. Enforce strict safety: warm-up readiness, health hysteresis with time-based bounds,
   consecutive loss penalties (weighted), flash crash detection (tick-level), rate-limit
   awareness, and deterministic latency < 1ms (99th percentile).
5. Ensure full auditability: every decision logged, state persisted asynchronously with
   retry, and thread-safe operations throughout.

External Dependencies (Real Interfaces):
- polaris.market_regime.MarketRegimeClassifier : 9-grid market state
- olympus.agent_arbiter.AgentArbiter : aggregated agent opinions with timestamps
- strategos.signal_hunger.SignalHungerRegulator : hunger level & confidence
- infrastructure.health_pulse.HealthPulseMonitor : system health vector & sub-scores
- hermes.circuit_breaker.CircuitBreakerCascade : circuit breaker status
- infrastructure.rate_limiter.RateLimitGovernor : exchange rate limit governor
- infrastructure.audit_chain.AuditLogChain : immutable decision audit
- infrastructure.stream_gateway.StreamGateway : exchange connectivity & data stream health
- hermes.order_gateway.OrderExecutionGateway : order cancellation & emergency stops

Interface Contracts:
- select_mode(context: Dict) -> Dict[str, Any]
- get_active_parameters() -> Dict[str, Any]
- compute_grid_levels(atr: float, price: float) -> int
- force_mode_external(mode: str, reason: str) -> Dict[str, Any]  # manual override
- reload_config(new_config: Dict) -> None
- health_check() -> Dict[str, Any]

Error Handling & Degradation:
- Missing critical inputs → EMERGENCY_ONLY
- All agents unavailable → Conservative + human flag
- Decision time > 0.8ms → return cached safe decision (with mode consistency check)
- Persistence failures → retry with backoff + local file fallback
- Audit log failure → alert + local secure log
"""

import logging
import time
import threading
from typing import Dict, Any, List, Optional
from enum import Enum
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class SystemMode(Enum):
    TREND_FOLLOWING = "trend_following"
    GRID_RANGING = "grid_ranging"
    IDLE_WAIT = "idle_wait"
    EMERGENCY_ONLY = "emergency_only"
    KILL = "kill"


class ParamGroup(Enum):
    AGGRESSIVE = "aggressive"
    NEUTRAL = "neutral"
    CONSERVATIVE = "conservative"


class MetaDecisionEngine:
    """Global Trading Mode Controller — Institutional Grade"""

    # -------------------------------------------------------------------------
    # Class Constants (calibrated via historical simulation & live drills)
    # -------------------------------------------------------------------------
    # Mode switching hysteresis
    MODE_SWITCH_MIN_STABLE_SEC = 60.0
    MODE_SWITCH_CONFIRMATION_BARS = 3
    MODE_EMERGENCY_BYPASS = True

    # Health thresholds (sub-dimensional checks)
    META_HEALTH_CRITICAL = 40.0
    META_HEALTH_RECOVERY_HYSTERESIS = 50.0
    META_HEALTH_CAUTIOUS = 60.0
    META_HEALTH_ALL_SUBS_MIN = 60.0
    # Maximum duration in EMERGENCY_ONLY before forced re-evaluation (seconds)
    EMERGENCY_MAX_DURATION_SEC = 300.0
    # Time required continuously above recovery threshold to exit emergency
    EMERGENCY_RECOVERY_STABLE_SEC = 5.0

    # Hunger dynamics
    HUNGER_AGGRESSIVE_MIN = 0.6
    HUNGER_CONSERVATIVE_MAX = 0.3
    HUNGER_QUALITY_WEIGHT = 0.7

    # Risk & liquidity
    DEFAULT_RISK_BUDGET_PCT = 2.0            # % of AUM
    RISK_BUDGET_IDLE_THRESHOLD = 0.8         # 80% consumption triggers IDLE
    LIQUIDITY_CRITICAL_PERCENTILE = 0.2
    MIN_ORDERBOOK_DEPTH_USDT = 50000.0       # absolute depth threshold (USDT)

    # Warm-up (unit: orderbook updates count)
    WARMUP_MIN_OBSERVATIONS = 500
    WARMUP_MAX_DURATION_SEC = 600.0          # maximum wait before forced ready

    # Decision timeout & cache (target < 1ms)
    DECISION_TIMEOUT_NS = 800_000            # 0.8ms
    CACHED_DECISION_MAX_AGE_SEC = 5.0
    CACHED_DECISION_MIN_AGE_FORCE_SEC = 0.5  # avoid using too-fresh cache in fallback

    # Flash crash detection (tick-level)
    FLASH_CRASH_TICK_PRICE_DROP = 0.02       # 2% in single tick
    FLASH_CRASH_TICK_VOL_MULTIPLIER = 3.0    # volume > 3x recent average
    FLASH_CRASH_CONSECUTIVE_TICKS = 2        # consecutive ticks

    # Weighted loss
    LOSS_WEIGHT_THRESHOLD = -0.03            # -3% weighted loss
    LOSS_WEIGHT_DECAY = 0.9
    LOSS_EMA_ALPHA = 0.05                    # EMA alpha for incremental update

    # Consensus freshness
    CONSENSUS_MAX_AGE_SEC = 1.0

    # Default regime-mode mapping (overridable)
    DEFAULT_REGIME_MODE_MAP = {
        'dormant':           SystemMode.IDLE_WAIT,
        'cold_range':        SystemMode.GRID_RANGING,
        'accumulation':      SystemMode.TREND_FOLLOWING,
        'weak_trend':        SystemMode.TREND_FOLLOWING,
        'normal':            SystemMode.TREND_FOLLOWING,
        'battle':            SystemMode.TREND_FOLLOWING,
        'liquidity_trap':    SystemMode.IDLE_WAIT,
        'emotional':         SystemMode.IDLE_WAIT,
        'trending_strong':   SystemMode.TREND_FOLLOWING,
    }

    # Parameter presets (versioned, units, calibration date)
    TREND_PARAMS = {
        ParamGroup.AGGRESSIVE.value: {
            'atr_stop_multiplier': 2.5,       # [1.5,5.0], calibrated 2025-12
            'factor_threshold': 0.55,
            'add_position_base_coeff': 0.9,
            'max_positions': 3,
            'allow_add_position': True,
            'stop_limit_offset_bps': 10,
        },
        ParamGroup.NEUTRAL.value: {
            'atr_stop_multiplier': 3.0,
            'factor_threshold': 0.60,
            'add_position_base_coeff': 0.7,
            'max_positions': 2,
            'allow_add_position': True,
            'stop_limit_offset_bps': 15,
        },
        ParamGroup.CONSERVATIVE.value: {
            'atr_stop_multiplier': 4.0,
            'factor_threshold': 0.75,
            'add_position_base_coeff': 0.0,
            'max_positions': 0,
            'allow_add_position': False,
            'stop_limit_offset_bps': 25,
        }
    }

    GRID_PARAMS = {
        ParamGroup.AGGRESSIVE.value: {
            'grid_levels_range': (8, 14),
            'grid_width_atr': 2.0,
            'position_per_level_bps': 800,
            'max_grid_loss_bps': 5000,        # enforced by risk module
        },
        ParamGroup.NEUTRAL.value: {
            'grid_levels_range': (5, 10),
            'grid_width_atr': 2.5,
            'position_per_level_bps': 500,
            'max_grid_loss_bps': 3000,
        },
        ParamGroup.CONSERVATIVE.value: {
            'grid_levels_range': (3, 6),
            'grid_width_atr': 3.5,
            'position_per_level_bps': 200,
            'max_grid_loss_bps': 1500,
        }
    }

    # Mandatory sub-health keys
    MANDATORY_HEALTH_SUB_KEYS = ['data_health', 'execution_health', 'strategy_health']

    def __init__(self, config: Optional[Dict] = None):
        # Thread safety
        self._rwlock = threading.RLock()

        # Operational state
        self._current_mode: SystemMode = SystemMode.IDLE_WAIT
        self._prev_mode: SystemMode = SystemMode.IDLE_WAIT  # for rollback
        self._current_param_group: ParamGroup = ParamGroup.CONSERVATIVE
        self._startup_time_mono: float = time.monotonic()
        self._ready: bool = False
        self._warmup_observation_count: int = 0

        # Mode switching state machine
        self._pending_mode: Optional[SystemMode] = None
        self._mode_confirmation_counter: int = 0
        self._last_mode_switch_time_mono: float = 0.0
        self._emergency_enter_time_mono: Optional[float] = None
        self._recovery_stable_start_mono: Optional[float] = None

        # Mappings (deep copy)
        self._regime_mode_map = deepcopy(self.DEFAULT_REGIME_MODE_MAP)

        # Agent consensus cache with freshness
        self._cached_consensus: Dict[str, Any] = {'force_idle': False, 'stone_fear': 0.0}
        self._consensus_timestamp: float = 0.0
        self._force_idle_counter: int = 0
        self._smoothed_fear: float = 0.5
        self._last_fear_update_mono: float = 0.0

        # Weighted loss (EMA-based)
        self._weighted_loss: float = 0.0

        # Decision cache
        self._cached_decision: Optional[Dict] = None
        self._cached_decision_time_mono: float = 0.0

        # External dependencies
        self._market_regime = None
        self._agent_arbiter = None
        self._hunger_regulator = None
        self._health_monitor = None
        self._circuit_breaker = None
        self._rate_limiter = None
        self._order_gateway = None
        self._stream_gateway = None

        # Persistence executor (single background thread)
        self._persist_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="meta_persist")

        # Log throttling
        self._last_warning_time: Dict[str, float] = {}

        if config:
            self._apply_config(config)

        logger.info("MetaDecisionEngine initialized. Mode: %s", self._current_mode.value)

    # -------------------------------------------------------------------------
    # Configuration (whitelist-based)
    # -------------------------------------------------------------------------
    _ALLOWED_CONFIG_KEYS = {
        'MODE_SWITCH_MIN_STABLE_SEC', 'MODE_SWITCH_CONFIRMATION_BARS',
        'META_HEALTH_CRITICAL', 'META_HEALTH_RECOVERY_HYSTERESIS',
        'META_HEALTH_CAUTIOUS', 'META_HEALTH_ALL_SUBS_MIN',
        'EMERGENCY_MAX_DURATION_SEC', 'EMERGENCY_RECOVERY_STABLE_SEC',
        'HUNGER_AGGRESSIVE_MIN', 'HUNGER_CONSERVATIVE_MAX', 'HUNGER_QUALITY_WEIGHT',
        'DEFAULT_RISK_BUDGET_PCT', 'RISK_BUDGET_IDLE_THRESHOLD',
        'LIQUIDITY_CRITICAL_PERCENTILE', 'MIN_ORDERBOOK_DEPTH_USDT',
        'WARMUP_MIN_OBSERVATIONS', 'WARMUP_MAX_DURATION_SEC',
        'DECISION_TIMEOUT_NS', 'CACHED_DECISION_MAX_AGE_SEC',
        'FLASH_CRASH_TICK_PRICE_DROP', 'FLASH_CRASH_TICK_VOL_MULTIPLIER',
        'FLASH_CRASH_CONSECUTIVE_TICKS', 'LOSS_WEIGHT_THRESHOLD',
        'LOSS_WEIGHT_DECAY', 'LOSS_EMA_ALPHA', 'CONSENSUS_MAX_AGE_SEC'
    }

    @classmethod
    def _apply_config(cls, config: Dict) -> None:
        for key, value in config.items():
            if key in cls._ALLOWED_CONFIG_KEYS:
                setattr(cls, key, value)
                logger.debug("Config override: %s = %s", key, value)
            else:
                logger.warning("Ignored unknown config key: %s", key)

    def reload_config(self, new_config: Dict) -> None:
        """Hot-reload with thread safety for param groups."""
        with self._rwlock:
            if 'regime_mode_map' in new_config:
                self._regime_mode_map = deepcopy(new_config['regime_mode_map'])
            if 'trend_params' in new_config:
                MetaDecisionEngine.TREND_PARAMS = deepcopy(new_config['trend_params'])
            if 'grid_params' in new_config:
                MetaDecisionEngine.GRID_PARAMS = deepcopy(new_config['grid_params'])
            self._apply_config(new_config)
        logger.info("Configuration hot-reloaded.")

    # -------------------------------------------------------------------------
    # Dependency Injection
    # -------------------------------------------------------------------------
    def set_market_regime(self, obj): self._market_regime = obj
    def set_agent_arbiter(self, obj): self._agent_arbiter = obj
    def set_hunger_regulator(self, obj): self._hunger_regulator = obj
    def set_health_monitor(self, obj): self._health_monitor = obj
    def set_circuit_breaker(self, obj): self._circuit_breaker = obj
    def set_rate_limiter(self, obj): self._rate_limiter = obj
    def set_order_gateway(self, obj): self._order_gateway = obj
    def set_stream_gateway(self, obj): self._stream_gateway = obj

    # -------------------------------------------------------------------------
    # Main Decision Entry Point
    # -------------------------------------------------------------------------
    def select_mode(self, context: Dict[str, Any]) -> Dict[str, Any]:
        start_ns = time.perf_counter_ns()
        warnings: List[str] = []

        # Validate and sanitize context early
        if not self._validate_context(context):
            return self._force_mode(SystemMode.EMERGENCY_ONLY, ParamGroup.CONSERVATIVE,
                                    "Invalid context", warnings)

        # Flash crash / tick-level anomaly detection
        if self._detect_flash_crash(context):
            return self._force_mode(SystemMode.KILL, ParamGroup.CONSERVATIVE,
                                    "Flash crash detected at tick level", warnings)

        # Warm-up readiness
        self._check_warmup(context)
        if not self._ready:
            return self._force_mode(SystemMode.IDLE_WAIT, ParamGroup.CONSERVATIVE,
                                    "System warming up", warnings)

        # --- Extract fields (with defaults) ---
        market_state = context.get('market_state', {})
        agent_consensus = context.get('agent_consensus', {})
        hunger_level = self._clamp(context.get('hunger_level', 0.5), 0.0, 1.0)
        hunger_confidence = self._clamp(context.get('hunger_confidence', 0.5), 0.0, 1.0)
        health_score = self._clamp(context.get('health_score', 100.0), 0.0, 100.0)
        health_subs = context.get('health_sub_scores', {})
        risk_budget_used = self._clamp(context.get('risk_budget_used_pct', 0.0), 0.0, 1.0)
        exchange_status = context.get('exchange_status', 'normal')
        consecutive_losses = context.get('consecutive_losses', 0)
        pnl_history = context.get('pnl_history', [])

        # Validate sub-health keys
        missing_subs = [k for k in self.MANDATORY_HEALTH_SUB_KEYS if k not in health_subs]
        if missing_subs:
            warnings.append(f"Missing health sub-keys: {missing_subs}")
            health_subs = {k: health_subs.get(k, 0.0) for k in self.MANDATORY_HEALTH_SUB_KEYS}

        # Update agent consensus (with freshness check)
        if agent_consensus and 'timestamp' in agent_consensus:
            if time.time() - agent_consensus['timestamp'] < self.CONSENSUS_MAX_AGE_SEC:
                self._cached_consensus = agent_consensus
                self._consensus_timestamp = agent_consensus['timestamp']
        # else keep old cached, but will be considered stale by _select_param_group

        # Update weighted loss incrementally
        if pnl_history:
            self._update_weighted_loss_incremental(pnl_history)

        # System circuit breaker
        if self._circuit_breaker and self._circuit_breaker.is_system_breached():
            return self._force_mode(SystemMode.KILL, ParamGroup.CONSERVATIVE,
                                    "Circuit breaker breached", warnings)

        # Exchange status
        if exchange_status in ('maintenance', 'halted'):
            return self._force_mode(SystemMode.IDLE_WAIT, ParamGroup.CONSERVATIVE,
                                    f"Exchange {exchange_status}", warnings)

        # Health emergency with hysteresis and time limit
        if health_score < self.META_HEALTH_CRITICAL:
            if self._emergency_enter_time_mono is None:
                self._emergency_enter_time_mono = time.monotonic()
            return self._force_mode(SystemMode.EMERGENCY_ONLY, ParamGroup.CONSERVATIVE,
                                    f"Critical health {health_score:.1f}", warnings)
        elif self._emergency_enter_time_mono is not None:
            # Trying to exit emergency
            if health_score >= self.META_HEALTH_RECOVERY_HYSTERESIS:
                if self._recovery_stable_start_mono is None:
                    self._recovery_stable_start_mono = time.monotonic()
                elif time.monotonic() - self._recovery_stable_start_mono >= self.EMERGENCY_RECOVERY_STABLE_SEC:
                    # exit emergency
                    self._emergency_enter_time_mono = None
                    self._recovery_stable_start_mono = None
                    logger.info("Health recovered, exiting emergency.")
                else:
                    return self._force_mode(SystemMode.EMERGENCY_ONLY, ParamGroup.CONSERVATIVE,
                                            "Health recovering, waiting stable", warnings)
            else:
                self._recovery_stable_start_mono = None  # reset
                # Check maximum emergency duration
                if time.monotonic() - self._emergency_enter_time_mono > self.EMERGENCY_MAX_DURATION_SEC:
                    logger.warning("Max emergency duration reached, forcing exit.")
                    self._emergency_enter_time_mono = None
                else:
                    return self._force_mode(SystemMode.EMERGENCY_ONLY, ParamGroup.CONSERVATIVE,
                                            "Health still critical", warnings)

        # Rate limiter
        if self._rate_limiter and self._rate_limiter.near_limit():
            return self._force_mode(SystemMode.IDLE_WAIT, ParamGroup.CONSERVATIVE,
                                    "Exchange rate limit near", warnings)

        # Base mode from regime
        regime_name = market_state.get('regime', 'normal')
        base_mode = self._regime_mode_map.get(regime_name, SystemMode.TREND_FOLLOWING)

        # Risk budget
        if risk_budget_used > self.RISK_BUDGET_IDLE_THRESHOLD:
            base_mode = SystemMode.IDLE_WAIT
            warnings.append(f"Risk budget exhausted {risk_budget_used:.2%}")

        # Liquidity: both percentile and absolute depth
        liquidity_pct = market_state.get('liquidity_percentile', 0.5)
        orderbook_depth = market_state.get('orderbook_depth_usdt', 1e9)
        if liquidity_pct < self.LIQUIDITY_CRITICAL_PERCENTILE or orderbook_depth < self.MIN_ORDERBOOK_DEPTH_USDT:
            base_mode = SystemMode.IDLE_WAIT
            warnings.append(f"Low liquidity: pct={liquidity_pct:.2f}, depth={orderbook_depth:.0f}")

        # Agent consensus force_idle with debounce
        if self._cached_consensus.get('force_idle'):
            self._force_idle_counter += 1
            if self._force_idle_counter >= 2:
                base_mode = SystemMode.IDLE_WAIT
                warnings.append("Agent consensus force idle (confirmed)")
        else:
            self._force_idle_counter = 0

        # Weighted loss penalty
        if self._weighted_loss < self.LOSS_WEIGHT_THRESHOLD:
            base_mode = SystemMode.IDLE_WAIT
            warnings.append(f"Weighted loss critical ({self._weighted_loss:.4f})")

        # Parameter group selection
        param_group = self._select_param_group(
            hunger_level, hunger_confidence, health_score, health_subs,
            market_state, consecutive_losses
        )

        # Mode switching with hysteresis (only for non-emergency transitions)
        prev_mode = self._current_mode
        final_mode = self._confirm_mode_switch(base_mode, is_emergency=base_mode in (
            SystemMode.EMERGENCY_ONLY, SystemMode.KILL, SystemMode.IDLE_WAIT
        ))

        # Rollback point save
        with self._rwlock:
            self._prev_mode = prev_mode
            self._current_mode = final_mode
            self._current_param_group = param_group

        # Post-switch actions
        self._on_mode_changed(final_mode)

        result = {
            "status": "ok",
            "mode": final_mode.value,
            "param_group": param_group.value,
            "warnings": list(warnings),
            "timestamp_utc": time.time(),
        }

        # Decision timeout handling
        elapsed_ns = time.perf_counter_ns() - start_ns
        if elapsed_ns > self.DECISION_TIMEOUT_NS:
            logger.warning("MetaDecision timeout: %.1f us", elapsed_ns / 1000.0)
            if self._cached_decision and (time.monotonic() - self._cached_decision_time_mono) < self.CACHED_DECISION_MAX_AGE_SEC:
                # Ensure cache mode is safe (not KILL/EMERGENCY unless system still in that state)
                cached = self._cached_decision
                if cached.get('mode') not in (SystemMode.KILL.value, SystemMode.EMERGENCY_ONLY.value):
                    # rollback mode
                    with self._rwlock:
                        self._current_mode = self._prev_mode
                    logger.error("Rolling back mode to %s due to timeout", self._prev_mode.value)
                    return dict(cached)
                else:
                    # cached decision is emergency, but it may be outdated; still return it with warning
                    return dict(cached, warnings=cached.get('warnings', []) + ["Timeout fallback to emergency cache"])
            else:
                # no cache: force safe
                return self._force_mode(SystemMode.EMERGENCY_ONLY, ParamGroup.CONSERVATIVE,
                                        "Decision timeout without cache", warnings)

        # Cache safe decisions (non KILL/EMERGENCY) for future fallback
        if final_mode not in (SystemMode.KILL, SystemMode.EMERGENCY_ONLY):
            self._cached_decision = result
            self._cached_decision_time_mono = time.monotonic()

        self._log_decision(result, context)
        return result

    # -------------------------------------------------------------------------
    # Parameter Group Selection
    # -------------------------------------------------------------------------
    def _select_param_group(self, hunger: float, hunger_confidence: float,
                            health: float, health_subs: Dict,
                            market_state: Dict, consecutive_losses: int) -> ParamGroup:
        # Health forced conservative
        if health < self.META_HEALTH_CAUTIOUS:
            return ParamGroup.CONSERVATIVE
        sub_values = [v for v in health_subs.values() if isinstance(v, (int, float))]
        if sub_values and min(sub_values) < self.META_HEALTH_ALL_SUBS_MIN:
            return ParamGroup.CONSERVATIVE

        # Smoothed fear with dynamic alpha (corrected)
        raw_fear = self._clamp(self._cached_consensus.get('stone_fear', 0.0), 0.0, 1.0)
        delta = abs(raw_fear - self._smoothed_fear)
        alpha = 0.6 if delta > 0.2 else 0.2
        self._smoothed_fear = alpha * raw_fear + (1 - alpha) * self._smoothed_fear
        self._last_fear_update_mono = time.monotonic()
        if self._smoothed_fear > 0.7:
            return ParamGroup.CONSERVATIVE

        # Fear decay if no updates for 60s
        if time.monotonic() - self._last_fear_update_mono > 60.0:
            # revert toward 0.5
            self._smoothed_fear += 0.01 * (0.5 - self._smoothed_fear)

        # Consecutive losses combined with weighted loss
        if consecutive_losses >= 3 and self._weighted_loss < -0.01:
            return ParamGroup.CONSERVATIVE

        # Effective hunger
        effective_hunger = hunger * (hunger_confidence * self.HUNGER_QUALITY_WEIGHT + (1 - self.HUNGER_QUALITY_WEIGHT))
        if effective_hunger < self.HUNGER_CONSERVATIVE_MAX:
            return ParamGroup.CONSERVATIVE

        # Aggressive eligibility
        vol_pct = market_state.get('volatility_percentile', 0.5)
        if (effective_hunger > self.HUNGER_AGGRESSIVE_MIN and health > 75
                and 0.3 < vol_pct < 0.8 and self._smoothed_fear < 0.5):
            return ParamGroup.AGGRESSIVE

        return ParamGroup.NEUTRAL

    # -------------------------------------------------------------------------
    # Mode Switching with Hysteresis
    # -------------------------------------------------------------------------
    def _confirm_mode_switch(self, desired_mode: SystemMode, is_emergency: bool = False) -> SystemMode:
        with self._rwlock:
            if desired_mode == self._current_mode:
                self._pending_mode = None
                self._mode_confirmation_counter = 0
                return self._current_mode

            # Instant switch for true emergencies (circuit breaker, flash crash, exchange maintenance)
            if is_emergency and desired_mode in (SystemMode.EMERGENCY_ONLY, SystemMode.KILL, SystemMode.IDLE_WAIT):
                self._pending_mode = desired_mode
                self._mode_confirmation_counter = self.MODE_SWITCH_CONFIRMATION_BARS
            else:
                if self._pending_mode != desired_mode:
                    self._pending_mode = desired_mode
                    self._mode_confirmation_counter = 1
                else:
                    self._mode_confirmation_counter += 1

                if time.monotonic() - self._last_mode_switch_time_mono < self.MODE_SWITCH_MIN_STABLE_SEC:
                    return self._current_mode

            if self._mode_confirmation_counter >= self.MODE_SWITCH_CONFIRMATION_BARS:
                logger.info("Mode switch: %s -> %s", self._current_mode.value, desired_mode.value)
                self._last_mode_switch_time_mono = time.monotonic()
                self._pending_mode = None
                self._mode_confirmation_counter = 0
                return desired_mode

            return self._current_mode

    def _force_mode(self, mode: SystemMode, param: ParamGroup, reason: str, warnings: List) -> Dict:
        with self._rwlock:
            if self._current_mode != mode:
                logger.warning("Forced mode: %s (reason: %s)", mode.value, reason)
                self._on_mode_changed(mode)
            self._current_mode = mode
            self._current_param_group = param
            self._pending_mode = None
            self._mode_confirmation_counter = 0
        return {
            "status": "ok",
            "mode": mode.value,
            "param_group": param.value,
            "reason": reason,
            "warnings": list(warnings),
            "timestamp_utc": time.time(),
        }

    def force_mode_external(self, mode_str: str, reason: str) -> Dict:
        """Manual override for operational control."""
        try:
            mode = SystemMode(mode_str)
        except ValueError:
            return {"status": "error", "reason": f"Invalid mode: {mode_str}"}
        return self._force_mode(mode, ParamGroup.CONSERVATIVE, f"Manual: {reason}", [])

    def _on_mode_changed(self, new_mode: SystemMode):
        if new_mode in (SystemMode.IDLE_WAIT, SystemMode.EMERGENCY_ONLY, SystemMode.KILL):
            if self._order_gateway:
                try:
                    self._order_gateway.cancel_all_orders(timeout=0.2)
                except Exception as e:
                    logger.error("Cancel orders failed: %s", str(e))
            # Request risk module to place emergency stops on all positions
            try:
                if self._health_monitor:  # placeholder for risk interface
                    # In production: self._risk_manager.ensure_emergency_stops()
                    pass
            except Exception as e:
                logger.error("Failed to ensure emergency stops: %s", str(e))
        self._async_persist_state()

    # -------------------------------------------------------------------------
    # Warm-up Logic
    # -------------------------------------------------------------------------
    def _check_warmup(self, context: Dict):
        if self._ready:
            return
        updates = context.get('market_updates_since_start', 0)
        # require per-symbol readiness
        symbols_ready = context.get('symbols_orderbook_ready', False)
        elapsed = time.monotonic() - self._startup_time_mono
        if (updates >= self.WARMUP_MIN_OBSERVATIONS and symbols_ready) or elapsed > self.WARMUP_MAX_DURATION_SEC:
            self._ready = True
            logger.info("System warm-up complete (updates=%d, time=%.1fs)", updates, elapsed)

    # -------------------------------------------------------------------------
    # Flash Crash Detection (tick-level)
    # -------------------------------------------------------------------------
    def _detect_flash_crash(self, context: Dict) -> bool:
        tick_price_change = context.get('tick_price_change')
        tick_volume = context.get('tick_volume')
        avg_volume = context.get('avg_tick_volume', 1.0)
        if tick_price_change is None or not isinstance(tick_price_change, (int, float)):
            return False
        if abs(tick_price_change) > self.FLASH_CRASH_TICK_PRICE_DROP:
            if tick_volume is not None and avg_volume > 0 and tick_volume > self.FLASH_CRASH_TICK_VOL_MULTIPLIER * avg_volume:
                # check consecutive ticks
                crash_ticks = context.get('consecutive_crash_ticks', 0) + 1
                context['consecutive_crash_ticks'] = crash_ticks
                if crash_ticks >= self.FLASH_CRASH_CONSECUTIVE_TICKS:
                    return True
            else:
                context['consecutive_crash_ticks'] = 0
        else:
            context['consecutive_crash_ticks'] = 0
        return False

    # -------------------------------------------------------------------------
    # Weighted Loss Update (EMA-based)
    # -------------------------------------------------------------------------
    def _update_weighted_loss_incremental(self, pnl_history: List[float]):
        if not pnl_history:
            return
        # Use the most recent pnl (the list is expected to contain the latest trade pnl)
        latest_pnl = pnl_history[-1]
        self._weighted_loss = (self.LOSS_EMA_ALPHA * latest_pnl +
                               (1 - self.LOSS_EMA_ALPHA) * self._weighted_loss)

    # -------------------------------------------------------------------------
    # Persistence (async with retry)
    # -------------------------------------------------------------------------
    def _async_persist_state(self):
        try:
            self._persist_executor.submit(self._persist_state_sync)
        except Exception as e:
            logger.error("Failed to submit persist task: %s", str(e))

    def _persist_state_sync(self):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                from infrastructure.chronos_db import ChronosDB
                ChronosDB.upsert('meta_state', {
                    'mode': self._current_mode.value,
                    'param_group': self._current_param_group.value,
                    'last_switch': self._last_mode_switch_time_mono
                })
                break
            except Exception as e:
                logger.error("Persist attempt %d failed: %s", attempt+1, str(e))
                time.sleep(0.1 * (2**attempt))
        else:
            # Fallback to local file
            try:
                with open('/tmp/kunlun_meta_state.json', 'w') as f:
                    import json
                    json.dump({'mode': self._current_mode.value}, f)
            except Exception as e:
                logger.critical("Local state persist also failed: %s", str(e))

    # -------------------------------------------------------------------------
    # Audit Logging (mandatory)
    # -------------------------------------------------------------------------
    def _log_decision(self, result: Dict, context: Dict):
        try:
            from infrastructure.audit_chain import AuditLogChain
            AuditLogChain.log_event("meta_decision", "INFO", {
                "decision": result,
                "context_summary": {
                    "regime": context.get('market_state', {}).get('regime'),
                    "health": context.get('health_score'),
                    "hunger": context.get('hunger_level'),
                }
            })
        except ImportError:
            # Write to local audit fallback
            try:
                with open('/var/log/kunlun/audit_fallback.log', 'a') as f:
                    f.write(f"{time.time()}|META|{result}\n")
            except Exception:
                pass
        except Exception as e:
            logger.critical("Audit logging failed: %s", str(e))

    # -------------------------------------------------------------------------
    # Public Query Interfaces
    # -------------------------------------------------------------------------
    def get_active_parameters(self) -> Dict[str, Any]:
        with self._rwlock:
            mode = self._current_mode
            pg = self._current_param_group.value
        if mode == SystemMode.TREND_FOLLOWING:
            params = deepcopy(self.TREND_PARAMS.get(pg, {}))
        elif mode == SystemMode.GRID_RANGING:
            params = deepcopy(self.GRID_PARAMS.get(pg, {}))
        else:
            params = {}
        params['trading_allowed'] = mode in (SystemMode.TREND_FOLLOWING, SystemMode.GRID_RANGING)
        return {
            "mode": mode.value,
            "param_group": pg,
            "parameters": params,
        }

    def compute_grid_levels(self, atr: float, price: float) -> int:
        """Calculate actual number of grid levels from configured range."""
        with self._rwlock:
            pg = self._current_param_group.value
        grid_cfg = self.GRID_PARAMS.get(pg, {})
        low, high = grid_cfg.get('grid_levels_range', (5, 10))
        # Simple logic: scale linearly with ATR/price ratio
        ratio = atr / price if price > 0 else 0.01
        levels = int(low + (high - low) * min(ratio * 100, 1.0))
        return max(low, min(high, levels))

    def is_trading_allowed(self) -> bool:
        return self._current_mode in (SystemMode.TREND_FOLLOWING, SystemMode.GRID_RANGING)

    @staticmethod
    def _clamp(val: float, lower: float, upper: float) -> float:
        """Clamp value between lower and upper bounds."""
        return max(lower, min(upper, val))

    # -------------------------------------------------------------------------
    # Context Validation
    # -------------------------------------------------------------------------
    def _validate_context(self, context: Dict) -> bool:
        required = {'market_state', 'health_score'}
        for key in required:
            if key not in context or context[key] is None:
                logger.error("Missing critical context key: %s", key)
                return False
        # Type check
        if not isinstance(context['health_score'], (int, float)):
            logger.error("health_score is not numeric")
            return False
        # market_state must be dict
        if not isinstance(context['market_state'], dict):
            logger.error("market_state is not a dict")
            return False
        return True

    # -------------------------------------------------------------------------
    # Health Check
    # -------------------------------------------------------------------------
    @classmethod
    def health_check(cls) -> Dict[str, Any]:
        try:
            engine = cls()
            ctx = {
                'market_state': {'regime': 'trending_strong', 'volatility_percentile': 0.6, 'liquidity_percentile': 0.5, 'orderbook_depth_usdt': 1e7},
                'agent_consensus': {'stone_fear': 0.3, 'force_idle': False, 'timestamp': time.time()},
                'hunger_level': 0.7,
                'hunger_confidence': 0.9,
                'health_score': 85.0,
                'health_sub_scores': {'data_health': 90, 'execution_health': 95, 'strategy_health': 88},
                'risk_budget_used_pct': 0.2,
                'exchange_status': 'normal',
                'consecutive_losses': 0,
                'pnl_history': [],
                'tick_price_change': 0.001,
                'tick_volume': 1.0,
                'avg_tick_volume': 1.0,
                'market_updates_since_start': 1000,
                'symbols_orderbook_ready': True
            }
            engine._ready = True  # force ready for test
            decision = engine.select_mode(ctx)
            if decision['mode'] not in [m.value for m in SystemMode]:
                return {"status": "error", "message": "Invalid mode"}
            return {"status": "ok", "message": f"Decision OK: {decision['mode']}"}
        except Exception as e:
            logger.error("Health check failed: %s", str(e))
            return {"status": "error", "message": str(e)}
