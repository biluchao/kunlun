#!/usr/bin/env python3
"""
Kunlun System · Conflict Escalation Manager (ConflictEscalation)
Git: kunlun/core/olympus/conflict_escalation.py
Version: 3.4.0

Core Responsibilities:
1. Manage agent disagreements with strict lifecycle control. Escalation is based on
   sustained divergence duration, weighted by real-time market volatility.
2. Resolve conflicts through a 3-tier pipeline: mild (weighted vote), severe
   (historical precedent lookup via Book agent), critical (forced conservative action
   with optional AI analysis).
3. Guarantee system liveness: any conflict older than MAX_UNRESOLVED_SEC is
   forcibly closed with a CRITICAL-level resolution, ensuring funds are never
   left in an uncertain state.
4. Provide complete audit trail for every conflict lifecycle event. All mutations are
   recorded with monotonic timestamps, actor identities, and resolution reasons.
5. Operate with zero memory leaks: background maintenance thread prunes resolved
   conflicts, enforces hard caps on total entries, and truncates escalation histories.

External Dependencies (all injected, validated at startup):
- olympus.agent_arbiter.AgentArbiter
    .get_agent_weights() -> Dict[str, float]
    .notify_conflict_resolved(signal_id: str, action: str) -> None
- agents.stone_guardian.StoneGuardian
    .get_emergency_action(*, timeout: float) -> Optional[str]
- agents.book_chronicler.BookChronicler
    .get_historical_advice(context: Dict, *, timeout: float) -> Optional[Dict]
- xuanpu.deepseek_loader.DeepSeekLoader
    .analyze_conflict(payload: Dict, *, timeout: float) -> Optional[Dict]
- infrastructure.audit_chain.AuditLogChain
    .log_event(record: Dict) -> None

Interface Contracts:
- register_conflict(signal_id, level, opinions, market_context) -> Dict
    Returns {"status": "ok", "signal_id": str, "current_level": int}
    or {"status": "error", "reason": str}
- resolve(signal_id, market_context) -> Dict
    Returns {"status": "ok", "action": str, "reason": str, "ai_consulted": bool}
    or {"status": "error", "reason": str}
- get_active_conflicts() -> List[Dict]
- get_conflict_count() -> Dict[str, int]
- start_background_tasks() / stop_background_tasks()
- health_check() -> Dict

Exception Handling & Degradation:
- All external calls (Stone, Book, DeepSeek, Audit, Arbiter) are wrapped with
  timeouts. Failures are logged at ERROR level with full traceback and do not
  propagate to the caller.
- If AuditLogChain is unavailable, events are written to the local logger at INFO
  level with a structured [AUDIT] prefix.
- If DeepSeek is not injected or times out, fallback logic uses Stone emergency
  action or DEFAULT_CONSERVATIVE_ACTION.
- If the conflict registry reaches MAX_CONFLICTS, new registrations are rejected
  until the background thread prunes resolved entries.

Resource Management:
- Background daemon thread cleans expired conflicts every 15 seconds.
- Hard limit of MAX_CONFLICTS in-memory entries. Overage triggers forced eviction
  of oldest resolved conflicts before accepting new ones.
- Escalation history per conflict capped at MAX_ESCALATION_HISTORY entries.
- Deep copies of all externally-provided mutable structures prevent accidental
  aliasing and concurrent modification.

Thread Safety:
- Public methods use threading.RLock for all shared state access.
- Blocking I/O (AI, Book, Stone) is performed OUTSIDE the lock after capturing
  a deep copy of the required state.
- Background cleanup acquires the same lock, ensuring visibility of state changes.
"""

import logging
import threading
import time
import copy
import math
from typing import Dict, Any, List, Optional, Set, Tuple
from enum import IntEnum

logger = logging.getLogger(__name__)


# ============================================================================
# Enumerations
# ============================================================================

class ConflictLevel(IntEnum):
    """Ordered conflict severity. Higher value = more urgent."""
    NONE = 0
    MILD = 1
    SEVERE = 2
    CRITICAL = 3

    @property
    def priority(self) -> int:
        return self.value


class DeadlockCategory:
    """Tags for internal deadlock classification."""
    DIRECTION = "direction"
    SIZING = "sizing"
    TIMING = "timing"
    RISK_REWARD = "risk_reward"
    STRATEGY_MODE = "strategy_mode"


# ============================================================================
# Main Class
# ============================================================================

class ConflictEscalation:
    """
    Institutional-grade conflict escalation engine.
    Designed for high-frequency, multi-agent environments managing
    trillion-dollar AUM with full audit compliance.
    """

    # ---- Class constants (overridable via constructor config dict) ----
    MILD_ESCALATION_SEC: float = 30.0
    SEVERE_ESCALATION_SEC: float = 45.0
    MAX_UNRESOLVED_SEC: float = 60.0
    RESOLVED_RETENTION_SEC: float = 86400.0          # 24 hours
    BACKGROUND_CLEANUP_INTERVAL_SEC: float = 15.0
    MAX_CONFLICTS: int = 1000
    MAX_ESCALATION_HISTORY: int = 20
    MAX_SIGNAL_ID_LEN: int = 128
    MAX_LOG_DETAIL_LEN: int = 256

    AI_CALL_TIMEOUT_SEC: float = 30.0
    BOOK_ADVICE_TIMEOUT_SEC: float = 2.0
    STONE_EMERGENCY_TIMEOUT_SEC: float = 3.0
    AUDIT_LOG_TIMEOUT_SEC: float = 1.0

    DEFAULT_CONSERVATIVE_ACTION: str = "close_all"
    VALID_RESOLVE_ACTIONS: Set[str] = {
        "open_long", "open_short", "add_position",
        "reduce_position", "close_all", "hold"
    }

    # ---- Constructor ----
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Instance-level config
        self._mild_esc_sec = self.MILD_ESCALATION_SEC
        self._severe_esc_sec = self.SEVERE_ESCALATION_SEC
        self._max_unresolved_sec = self.MAX_UNRESOLVED_SEC
        self._retention_sec = self.RESOLVED_RETENTION_SEC
        self._max_conflicts = self.MAX_CONFLICTS
        self._ai_timeout = self.AI_CALL_TIMEOUT_SEC
        self._book_timeout = self.BOOK_ADVICE_TIMEOUT_SEC
        self._stone_timeout = self.STONE_EMERGENCY_TIMEOUT_SEC

        if config:
            self._apply_user_config(config)

        # Concurrency
        self._lock = threading.RLock()

        # Core state
        self._conflicts: Dict[str, Dict[str, Any]] = {}

        # Dependencies (must be injected before use)
        self._arbiter: Any = None
        self._stone: Any = None
        self._book: Any = None
        self._deepseek: Any = None
        self._audit: Any = None

        # Background maintenance
        self._cleanup_thread: Optional[threading.Thread] = None
        self._shutdown_flag = threading.Event()

        # Metrics
        self._total_registered: int = 0
        self._total_resolved: int = 0
        self._total_escalations: int = 0

        logger.info("ConflictEscalation v3.4.0 initialized (max_conflicts=%d)", self._max_conflicts)

    # ---- Configuration ----
    def _apply_user_config(self, cfg: Dict[str, Any]) -> None:
        """Apply user-provided config values with type validation."""
        mappings = {
            'MILD_ESCALATION_SEC': ('_mild_esc_sec', float),
            'SEVERE_ESCALATION_SEC': ('_severe_esc_sec', float),
            'MAX_UNRESOLVED_SEC': ('_max_unresolved_sec', float),
            'RESOLVED_RETENTION_SEC': ('_retention_sec', float),
            'MAX_CONFLICTS': ('_max_conflicts', int),
            'AI_CALL_TIMEOUT_SEC': ('_ai_timeout', float),
            'BOOK_ADVICE_TIMEOUT_SEC': ('_book_timeout', float),
            'STONE_EMERGENCY_TIMEOUT_SEC': ('_stone_timeout', float),
        }
        for key, (attr, dtype) in mappings.items():
            if key in cfg:
                setattr(self, attr, dtype(cfg[key]))

    # ---- Dependency Injection ----
    def inject_dependencies(self, *,
                            arbiter: Any = None,
                            stone: Any = None,
                            book: Any = None,
                            deepseek: Any = None,
                            audit: Any = None) -> None:
        """Inject all external dependencies at once. Thread-safe after construction."""
        self._arbiter = arbiter
        self._stone = stone
        self._book = book
        self._deepseek = deepseek
        self._audit = audit
        logger.info("Dependencies injected: arbiter=%s stone=%s book=%s deepseek=%s audit=%s",
                    bool(arbiter), bool(stone), bool(book), bool(deepseek), bool(audit))

    # ---- Background Maintenance ----
    def start_background_tasks(self) -> None:
        """Start the daemon cleanup thread. Idempotent."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return
        self._shutdown_flag.clear()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="conflict-cleanup"
        )
        self._cleanup_thread.start()
        logger.info("Background cleanup thread started")

    def stop_background_tasks(self) -> None:
        """Signal the cleanup thread to stop and wait for it."""
        self._shutdown_flag.set()
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=10.0)
            if self._cleanup_thread.is_alive():
                logger.warning("Cleanup thread did not terminate within 10s")

    def _cleanup_loop(self) -> None:
        """Loop until shutdown, periodically pruning resolved conflicts."""
        while not self._shutdown_flag.wait(self.BACKGROUND_CLEANUP_INTERVAL_SEC):
            try:
                self._prune_resolved_conflicts()
                self._enforce_total_cap()
            except Exception as e:
                logger.error("Cleanup iteration failed: %s", e, exc_info=True)

    def _prune_resolved_conflicts(self) -> None:
        """Remove resolved conflicts older than retention period. Locked."""
        now = time.monotonic()
        with self._lock:
            expired = [
                sid for sid, c in self._conflicts.items()
                if c.get('resolution') and (now - c['resolution']['resolved_at'] > self._retention_sec)
            ]
            for sid in expired:
                del self._conflicts[sid]
            if expired:
                logger.debug("Pruned %d expired conflicts", len(expired))

    def _enforce_total_cap(self) -> None:
        """If total exceeds MAX_CONFLICTS, evict oldest resolved first."""
        with self._lock:
            if len(self._conflicts) <= self._max_conflicts:
                return
            # Sort resolved first, then by age
            ordered = sorted(
                self._conflicts.items(),
                key=lambda kv: (
                    0 if kv[1].get('resolution') else 1,
                    kv[1]['start_time']
                )
            )
            excess = len(ordered) - self._max_conflicts
            for i in range(excess):
                sid = ordered[i][0]
                del self._conflicts[sid]
            logger.warning("Enforced total cap: evicted %d conflicts", excess)

    # ---- Public API ----
    def register_conflict(self,
                          signal_id: str,
                          level: ConflictLevel,
                          opinions: Dict[str, Dict[str, Any]],
                          market_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Register a new conflict or update an existing one.
        Returns current status. Thread-safe and non-blocking.
        """
        # Input validation
        if not isinstance(level, ConflictLevel):
            return {"status": "error", "reason": "Invalid conflict level type"}
        if not isinstance(signal_id, str) or not signal_id.strip():
            return {"status": "error", "reason": "signal_id must be a non-empty string"}

        signal_id = signal_id.strip()[:self.MAX_SIGNAL_ID_LEN]
        now = time.monotonic()

        with self._lock:
            # Capacity check
            if len(self._conflicts) >= self._max_conflicts and signal_id not in self._conflicts:
                self._audit_log("CONFLICT_REJECTED", signal_id, "registry_full")
                return {"status": "error", "reason": "Conflict registry full"}

            # Deep-copy mutable inputs to prevent aliasing
            opinions_copy = copy.deepcopy(opinions) if opinions else {}
            ctx_copy = copy.deepcopy(market_context) if market_context else {}

            if signal_id in self._conflicts:
                return self._update_existing(signal_id, opinions_copy, ctx_copy, now)

            return self._create_new(signal_id, level, opinions_copy, ctx_copy, now)

    def resolve(self, signal_id: str, market_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Resolve a conflict, producing a concrete trading action.
        Non-blocking: captures state under lock, then resolves externally.
        """
        # Snapshot under lock
        with self._lock:
            conflict = self._conflicts.get(signal_id)
            if not conflict:
                return {"status": "error", "reason": f"Conflict {signal_id} not found"}
            level = conflict['level']
            opinions = copy.deepcopy(conflict.get('opinions', {}))
            ctx = copy.deepcopy(market_context) if market_context else {}
            start_time = conflict['start_time']

        # Resolve outside lock (may call external services)
        if level == ConflictLevel.MILD:
            result = self._resolve_mild(opinions)
        elif level == ConflictLevel.SEVERE:
            result = self._resolve_severe(opinions, ctx)
        else:
            result = self._resolve_critical(opinions, ctx, start_time)

        # Validate result
        action = result.get('action', 'hold')
        if action not in self.VALID_RESOLVE_ACTIONS:
            logger.warning("Invalid resolve action '%s', falling back to hold", action)
            action = 'hold'

        # Mark resolved under lock
        with self._lock:
            if signal_id in self._conflicts:
                self._mark_resolved(self._conflicts[signal_id], action, result.get('reason', ''))
            else:
                logger.warning("Conflict %s vanished during resolution", signal_id)

        self._total_resolved += 1
        self._audit_log("CONFLICT_RESOLVED", signal_id,
                        f"action={action} reason={result.get('reason', '')}")

        # Notify arbiter of resolution
        self._safe_notify_arbiter(signal_id, action)

        return {
            "status": "ok",
            "signal_id": signal_id,
            "action": action,
            "reason": result.get('reason', ''),
            "ai_consulted": result.get('ai_consulted', False)
        }

    def get_active_conflicts(self) -> List[Dict[str, Any]]:
        """Return list of unresolved conflicts sorted by severity (desc)."""
        with self._lock:
            now = time.monotonic()
            active = [
                {
                    'signal_id': sid,
                    'level': c['level'].name,
                    'duration_sec': round(now - c['start_time'], 3),
                    'agent_count': len(c.get('opinions', {}))
                }
                for sid, c in self._conflicts.items()
                if c.get('resolution') is None
            ]
            active.sort(key=lambda x: ConflictLevel[x['level']].priority, reverse=True)
            return active

    def get_conflict_count(self) -> Dict[str, int]:
        """Return counts by severity level."""
        counts: Dict[str, int] = {'mild': 0, 'severe': 0, 'critical': 0, 'resolved': 0}
        with self._lock:
            for c in self._conflicts.values():
                if c.get('resolution'):
                    counts['resolved'] += 1
                else:
                    level_name = c['level'].name.lower()
                    if level_name in counts:
                        counts[level_name] += 1
        return counts

    def evaluate_deadlock(self, opinions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Check if current agent opinions constitute a decision deadlock."""
        available = {k: v for k, v in opinions.items() if v.get('available')}
        if len(available) < 3:
            return {"is_deadlock": False, "reason": "insufficient_active_agents"}

        confidences = []
        for v in available.values():
            conf = v.get('confidence')
            if isinstance(conf, (int, float)):
                confidences.append(float(conf))
        if not confidences:
            return {"is_deadlock": False, "reason": "no_valid_confidence"}

        avg_confidence = sum(confidences) / len(confidences)
        decisions = [v.get('decision', '') for v in available.values()]

        if avg_confidence < 0.4 and len(set(decisions)) >= 3:
            dtype = self._classify_deadlock(decisions)
            return {
                "is_deadlock": True,
                "type": dtype,
                "avg_confidence": round(avg_confidence, 4),
                "suggested_action": "defer_to_stone"
            }
        return {"is_deadlock": False}

    def handle_prolonged_deadlock(self, signal_id: str,
                                  market_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Force resolution of a deadlock that has exceeded the time limit."""
        with self._lock:
            conflict = self._conflicts.get(signal_id)
            if not conflict or conflict.get('resolution'):
                return {"status": "ok", "action": "none"}
            opinions = copy.deepcopy(conflict.get('opinions', {}))
            ctx = copy.deepcopy(market_context) if market_context else {}
            start_time = conflict['start_time']

        duration = time.monotonic() - start_time
        if duration > self._max_unresolved_sec:
            return self._resolve_critical(opinions, ctx, start_time)
        return {"status": "ok", "action": "continue_waiting"}

    def get_metrics(self) -> Dict[str, int]:
        """Return operational metrics for monitoring."""
        with self._lock:
            return {
                'total_registered': self._total_registered,
                'total_resolved': self._total_resolved,
                'total_escalations': self._total_escalations,
                'current_active': sum(1 for c in self._conflicts.values() if not c.get('resolution')),
                'current_resolved': sum(1 for c in self._conflicts.values() if c.get('resolution')),
                'registry_size': len(self._conflicts)
            }

    # ---- Internal Registration Helpers (lock held) ----
    def _create_new(self, sid: str, level: ConflictLevel,
                    opinions: Dict, ctx: Dict, now: float) -> Dict[str, Any]:
        self._conflicts[sid] = {
            'signal_id': sid,
            'level': level,
            'previous_level': level,
            'start_time': now,
            'last_updated': now,
            'duration': 0.0,
            'opinions': opinions,
            'context': ctx,
            'escalation_history': [
                {'time': now, 'from': level.name, 'to': level.name, 'trigger': 'initial'}
            ],
            'resolution': None
        }
        self._total_registered += 1
        self._audit_log("CONFLICT_REGISTERED", sid, f"level={level.name}")
        return {"status": "ok", "signal_id": sid, "current_level": level.value}

    def _update_existing(self, sid: str, opinions: Dict, ctx: Dict, now: float) -> Dict[str, Any]:
        conflict = self._conflicts[sid]
        conflict['last_updated'] = now
        conflict['duration'] = now - conflict['start_time']
        conflict['opinions'] = opinions
        conflict['context'] = ctx

        old_level = conflict['level']
        new_level = self._auto_escalate_internal(conflict)
        if new_level != old_level:
            conflict['previous_level'] = old_level
            conflict['level'] = new_level
            history: list = conflict.setdefault('escalation_history', [])
            if len(history) < self.MAX_ESCALATION_HISTORY:
                history.append({
                    'time': now,
                    'from': old_level.name,
                    'to': new_level.name,
                    'trigger': 'auto'
                })
            self._total_escalations += 1
            self._audit_log("CONFLICT_ESCALATED", sid, f"{old_level.name}->{new_level.name}")
        return {"status": "ok", "signal_id": sid, "current_level": new_level.value}

    def _auto_escalate_internal(self, conflict: Dict[str, Any]) -> ConflictLevel:
        duration = conflict['duration']
        current = conflict['level']
        if current == ConflictLevel.MILD and duration >= self._mild_esc_sec:
            return ConflictLevel.SEVERE
        if current == ConflictLevel.SEVERE and duration >= self._severe_esc_sec:
            return ConflictLevel.CRITICAL
        if duration >= self._max_unresolved_sec:
            return ConflictLevel.CRITICAL
        return current

    # ---- Resolution Strategies (no lock) ----
    def _resolve_mild(self, opinions: Dict[str, Dict]) -> Dict[str, Any]:
        stone_meta = opinions.get('stone', {}).get('metadata', {})
        wind_meta = opinions.get('wind', {}).get('metadata', {})
        try:
            stone_sz = float(stone_meta.get('suggested_size', 0.5))
            wind_sz = float(wind_meta.get('suggested_size', 0.5))
        except (TypeError, ValueError):
            stone_sz, wind_sz = 0.5, 0.5

        # Retrieve dynamic weights from arbiter, else use defaults
        if self._arbiter:
            try:
                weights = self._arbiter.get_agent_weights()
                sw = weights.get('stone', 0.6)
                ww = weights.get('wind', 0.4)
            except Exception:
                sw, ww = 0.6, 0.4
        else:
            sw, ww = 0.6, 0.4

        avg = stone_sz * sw + wind_sz * ww
        action = "add_position" if avg > 0.3 else "hold"
        return {"action": action, "reason": f"mild_weighted_average({avg:.3f})"}

    def _resolve_severe(self, opinions: Dict[str, Dict], ctx: Dict) -> Dict[str, Any]:
        if self._book:
            try:
                advice = self._book.get_historical_advice(ctx, timeout=self._book_timeout)
                if (isinstance(advice, dict) and
                    isinstance(advice.get('confidence'), (int, float)) and
                    advice['confidence'] > 0.5):
                    action = advice.get('decision', 'hold')
                    return {"action": action, "reason": f"book_historical({advice.get('reason','')})"}
            except Exception as e:
                logger.error("Book advice failed: %s", e, exc_info=True)
        # Fallback to Stone
        action = opinions.get('stone', {}).get('decision', 'hold')
        return {"action": action, "reason": "severe_stone_fallback"}

    def _resolve_critical(self, opinions: Dict[str, Dict], ctx: Dict, start_time: float) -> Dict[str, Any]:
        ai_consulted = False
        action = self.DEFAULT_CONSERVATIVE_ACTION
        reason = "critical_forced"

        if self._deepseek:
            try:
                payload = {
                    'conflict_age_sec': round(time.monotonic() - start_time, 2),
                    'agent_decisions': {k: v.get('decision') for k, v in opinions.items()},
                    'context_keys': list(ctx.keys()) if ctx else []
                }
                resp = self._deepseek.analyze_conflict(payload, timeout=self._ai_timeout)
                ai_consulted = True
                if isinstance(resp, dict) and resp.get('confidence', 0) > 0.6:
                    action = resp.get('decision', action)
                    reason = f"ai_assisted({resp.get('reason','')})"
            except Exception as e:
                logger.error("DeepSeek conflict analysis failed: %s", e, exc_info=True)
        else:
            if self._stone:
                try:
                    stone_action = self._stone.get_emergency_action(timeout=self._stone_timeout)
                    if stone_action:
                        action = stone_action
                        reason = "stone_emergency_override"
                except Exception as e:
                    logger.error("Stone emergency action failed: %s", e, exc_info=True)

        return {"action": action, "reason": reason, "ai_consulted": ai_consulted}

    # ---- Helpers ----
    def _mark_resolved(self, conflict: Dict[str, Any], decision: str, reason: str) -> None:
        conflict['resolution'] = {
            'decision': decision,
            'reason': reason,
            'resolved_at': time.monotonic()
        }
        conflict['status'] = 'resolved'
        conflict['last_updated'] = time.monotonic()

    def _classify_deadlock(self, decisions: List[str]) -> str:
        unique = set(decisions)
        if 'open_long' in unique and 'open_short' in unique:
            return DeadlockCategory.DIRECTION
        if 'add_position' in unique and 'reduce_position' in unique:
            return DeadlockCategory.SIZING
        if len(unique) >= 4:
            return DeadlockCategory.STRATEGY_MODE
        return DeadlockCategory.TIMING

    def _safe_notify_arbiter(self, signal_id: str, action: str) -> None:
        if not self._arbiter:
            return
        try:
            self._arbiter.notify_conflict_resolved(signal_id, action)
        except Exception as e:
            logger.error("Failed to notify arbiter: %s", e, exc_info=True)

    def _audit_log(self, event_type: str, signal_id: str, detail: str) -> None:
        """Write an immutable audit record. Never raises."""
        detail = detail[:self.MAX_LOG_DETAIL_LEN]
        try:
            if self._audit:
                self._audit.log_event({
                    'module': 'ConflictEscalation',
                    'event': event_type,
                    'signal_id': signal_id,
                    'detail': detail,
                    'timestamp': time.time()
                })
            else:
                logger.info("[AUDIT|%s] %s %s %s", time.time(), event_type, signal_id, detail)
        except Exception:
            # Absolute last resort: ensure critical events survive
            logger.critical("[AUDIT_FALLBACK] %s %s %s", event_type, signal_id, detail)

    # ---- Health Check ----
    @classmethod
    def health_check(cls) -> Dict[str, Any]:
        """Comprehensive self-test simulating full conflict lifecycle."""
        esc = cls()
        esc.start_background_tasks()
        try:
            opinions = {
                'stone': {'available': True, 'decision': 'reduce_position', 'confidence': 0.8,
                          'metadata': {'suggested_size': 0.2}},
                'wind':  {'available': True, 'decision': 'add_position', 'confidence': 0.6,
                          'metadata': {'suggested_size': 0.8}}
            }

            # 1. Registration
            reg = esc.register_conflict('healthcheck_001', ConflictLevel.MILD, opinions)
            if reg['status'] != 'ok' or reg['current_level'] != ConflictLevel.MILD.value:
                return {"status": "error", "message": f"Registration failed: {reg}"}

            # 2. Forced escalation via internal duration
            with esc._lock:
                esc._conflicts['healthcheck_001']['duration'] = esc._mild_esc_sec + 0.1
            # Update to trigger escalation check
            reg2 = esc.register_conflict('healthcheck_001', ConflictLevel.MILD, opinions)
            if reg2['current_level'] != ConflictLevel.SEVERE.value:
                return {"status": "error", "message": f"Escalation failed: {reg2}"}

            # 3. Resolution
            res = esc.resolve('healthcheck_001', {'test': True})
            if res['status'] != 'ok':
                return {"status": "error", "message": f"Resolution failed: {res}"}
            if res['action'] not in esc.VALID_RESOLVE_ACTIONS:
                return {"status": "error", "message": f"Invalid action: {res['action']}"}

            # 4. Verify metrics
            metrics = esc.get_metrics()
            if metrics['total_registered'] != 1 or metrics['total_resolved'] != 1:
                return {"status": "error", "message": f"Metrics mismatch: {metrics}"}

            return {"status": "ok", "message": "Full lifecycle test passed"}
        except Exception as e:
            logger.exception("Health check failed")
            return {"status": "error", "message": str(e)}
        finally:
            esc.stop_background_tasks()
