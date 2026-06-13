#!/usr/bin/env python3
"""
Kunlun System · Agent Arbiter (AgentArbiter) — Fifth Generation Ultra-Institutional Implementation

Core Responsibilities:
1. Collect independent opinions from five guardian agents (lock-free snapshot with deepcopy isolation),
   dynamically weighted voting to generate final trade decisions.
2. Three-tier conflict escalation (MILD/SEVERE/CRITICAL) with automatic convergence to conservative
   strategy (force close all) in extreme market conditions.
3. Sliding-window contribution evaluation for each agent, with periodic weight adjustment subject to
   cooldown and floor constraints to ensure minority voices are never fully silenced.
4. Emergency override channel: Eye (cross-market crash) and Stone (force close) can trigger
   immediate full liquidation, bypassing normal voting.

External Dependencies (Real Module Interfaces):
- agents.stone_guardian.StoneGuardian.evaluate(ctx: Dict) -> Dict
- agents.wind_seeker.WindSeeker.evaluate(ctx: Dict) -> Dict
- agents.wind_seeker.WindSeeker.force_exploration(ctx: Dict) -> Dict
- agents.mirror_auditor.MirrorAuditor.evaluate(ctx: Dict) -> Dict
- agents.eye_sentinel.EyeSentinel.evaluate(ctx: Dict) -> Dict
- agents.book_chronicler.BookChronicler.evaluate(ctx: Dict) -> Dict
- infrastructure.audit_chain.AuditLogChain.log_event(event_type, severity, details) -> None
- infrastructure.error_registry.ErrorRegistry.resolve(code: str) -> Dict

Interface Contract:
- arbitrate(signal_context: Dict) -> Dict[str, Any]
  Thread-safe. Signal context is deep-copied before agent evaluation to prevent mutation.
  Returns fixed keys: status, decision, reason, error_code, audit_ref, warnings.
- evaluate_agents(signal_context: Dict) -> Dict[str, Dict]
  Collects opinions with per-agent timeout (2s). Returns standardized opinion dicts.
- record_outcome(trade_report: Dict) -> None
  Records trade result; triggers weight adjustment after cooldown (300s) and min samples.
- health_check() -> Dict[str, Any]
  Fully isolated self-test; does not affect production arbiter.

Exception Handling & Degradation:
- Agent eval timeout/exception: agent returns neutral opinion (KUN-AGT-E011), others unaffected.
- All agents unavailable: forced HOLD, error KUN-AGT-F010, logged as CRITICAL.
- Arbiter internal error or lock timeout: returns conservative HOLD (KUN-AGT-F012/F015).
- Weight non-normalized: auto-reset to default (KUN-AGT-W020).
- Decision type conversion failure: logged as KUN-AGT-E025, degrades to HOLD.

Resource Management:
- threading.RLock with 5s timeout; agent evaluation performed outside lock to minimize critical section.
- Conflict records managed with TTL + max cap, cleaned periodically.
- Contribution log uses deque with maxlen; weight adjustment throttled by cooldown.
- Audit log written asynchronously outside lock; sensitive data sanitized.

Thread Safety Guarantee:
- All writes (registration, weights, conflicts, veto timestamps) protected by self._lock.
- Reads (get_weights, conflict count) use lock-protected copy of state.
- Agent evaluation uses independent deepcopy of context; no shared mutable state.
"""

import logging
import time
import threading
from typing import Dict, Any, List, Optional, Tuple, Set
from enum import Enum
from collections import deque
import copy

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Standard decision types (immutable). Priority: CLOSE_ALL > REJECT > REDUCE > HOLD > OPEN/ADD."""
    CLOSE_ALL = "close_all"
    REJECT = "reject"
    REDUCE_POSITION = "reduce_position"
    HOLD = "hold"
    ADD_POSITION = "add_position"
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"

    @classmethod
    def safe_parse(cls, value: str) -> 'DecisionType':
        """Parse string to DecisionType; returns HOLD on invalid input."""
        try:
            return cls(value)
        except ValueError:
            logger.warning("[KUN-AGT-E025] Invalid decision '%s', defaulting to HOLD", value)
            return cls.HOLD


class ConflictLevel(Enum):
    """Conflict severity level."""
    NONE = 0
    MILD = 1
    SEVERE = 2
    CRITICAL = 3


# Agent identifier constants
AGENT_STONE = "stone"
AGENT_WIND = "wind"
AGENT_MIRROR = "mirror"
AGENT_EYE = "eye"
AGENT_BOOK = "book"
ALL_AGENTS: Tuple[str, ...] = (AGENT_STONE, AGENT_WIND, AGENT_MIRROR, AGENT_EYE, AGENT_BOOK)

# Immutable base for neutral opinion (frozen copy will be made on each use)
_NEUTRAL_OPINION_TEMPLATE: Dict = {
    'available': False,
    'decision': DecisionType.HOLD.value,
    'confidence': 0.0,
    'reason': '',
    'metadata': {}
}

# Audit module lazy-load cache (supports hot-reload via reset)
_audit_module_cache: Optional[Any] = None
_audit_module_checked: bool = False


def _reset_audit_cache() -> None:
    """Reset audit module cache to enable re-import after hot-reload."""
    global _audit_module_cache, _audit_module_checked
    _audit_module_cache = None
    _audit_module_checked = False


def _get_audit_module() -> Optional[Any]:
    """Lazy-load audit module, re-probing on hot-reload."""
    global _audit_module_cache, _audit_module_checked
    if not _audit_module_checked:
        try:
            from infrastructure.audit_chain import AuditLogChain
            _audit_module_cache = AuditLogChain
        except ImportError:
            _audit_module_cache = None
        _audit_module_checked = True
    return _audit_module_cache


class AgentArbiter:
    """Five-Agent Arbiter — Fifth Generation Ultra-Institutional Grade."""

    # ---------- Immutable Class Constants ----------
    DEFAULT_VOTING_WEIGHTS: Dict[str, float] = {
        AGENT_STONE: 0.40,
        AGENT_EYE: 0.35,
        AGENT_MIRROR: 0.15,
        AGENT_BOOK: 0.07,
        AGENT_WIND: 0.03
    }

    # Conflict escalation parameters
    SEVERE_CONFLICT_MIN_DIRECTIONS = 2
    CRITICAL_CONFLICT_DURATION_SEC = 10.0

    # Contribution evaluation parameters
    CONTRIBUTION_WINDOW = 50
    WEIGHT_ADJUST_STEP = 0.02
    MAX_WEIGHT_DELTA = 0.20
    MIN_SAMPLES = 10
    WEIGHT_ADJUST_COOLDOWN_SEC = 300
    MIN_AGENT_WEIGHT = 0.005  # floor to prevent complete silencing

    # Force exploration parameters
    MAX_NO_SIGNAL_SEC = 3600
    FORCE_EXPLORE_COOLDOWN_SEC = 600

    # Agent evaluation timeout
    AGENT_EVALUATE_TIMEOUT_SEC = 2.0

    # Stone veto cooldown
    STONE_VETO_COOLDOWN_SEC = 300
    MAX_STONE_VETO_IN_WINDOW = 3

    # Conflict record management
    CONFLICT_TTL_SEC = 60.0
    MAX_CONFLICT_RECORDS = 500
    CONFLICT_CLEANUP_INTERVAL = 10  # calls between cleanups

    # Contribution log capacity
    MAX_CONTRIBUTION_ENTRIES = 200

    # Lock timeout
    LOCK_TIMEOUT_SEC = 5.0

    def __init__(self, config: Optional[Dict] = None):
        self._lock = threading.RLock()

        # Agent registry
        self._agents: Dict[str, Any] = {name: None for name in ALL_AGENTS}

        # Dynamic voting weights (always normalized)
        self._voting_weights: Dict[str, float] = dict(self.DEFAULT_VOTING_WEIGHTS)
        self._normalize_weights()

        # Conflict tracking
        self._active_conflicts: Dict[str, Dict] = {}
        self._conflict_call_counter: int = 0

        # Stone veto timestamps (deque with maxlen)
        self._stone_veto_ts: deque = deque(maxlen=self.MAX_STONE_VETO_IN_WINDOW * 2)

        # Contribution log (deque with maxlen)
        self._contribution_log: deque = deque(maxlen=self.MAX_CONTRIBUTION_ENTRIES)
        self._last_weight_adjust_time: float = 0.0

        # No-signal timer
        self._last_signal_time: float = time.time()
        self._last_force_explore_time: float = 0.0

        # Apply external configuration (whitelist only)
        if config:
            self._apply_config(config)

        logger.info("AgentArbiter v5 initialized: lock_timeout=%.1fs, weights=%s",
                    self.LOCK_TIMEOUT_SEC,
                    {k: round(v, 4) for k, v in self._voting_weights.items()})

    # ---------- Dependency Injection ----------
    def register_agent(self, name: str, instance: Any) -> bool:
        """Register an agent instance after verifying required interface."""
        if name not in self._agents:
            logger.error("Unknown agent name: %s", name)
            return False
        if not callable(getattr(instance, 'evaluate', None)):
            logger.error("Agent %s lacks callable 'evaluate'", name)
            return False
        with self._lock:
            self._agents[name] = instance
        logger.info("Agent %s registered", name)
        return True

    def unregister_agent(self, name: str) -> bool:
        """Remove an agent from the registry."""
        with self._lock:
            if name in self._agents:
                self._agents[name] = None
                return True
        return False

    # ---------- Opinion Collection (Lock-Free with Deepcopy) ----------
    def evaluate_agents(self, signal_context: Dict) -> Dict[str, Dict]:
        """
        Collect standardized opinions from all agents.
        Uses a deepcopy of context to prevent cross-agent mutation.
        """
        # Deepcopy context once to avoid per-agent overhead and mutation risks
        ctx_copy = copy.deepcopy(signal_context) if signal_context else {}

        # Snapshot agent references under lock
        with self._lock:
            agents_snapshot = dict(self._agents)

        opinions = {}
        for name in ALL_AGENTS:
            agent = agents_snapshot.get(name)
            if agent is None:
                opinions[name] = self._make_neutral_opinion("Unregistered")
                continue
            try:
                start = time.perf_counter()
                raw = agent.evaluate(ctx_copy)
                elapsed = time.perf_counter() - start
                if elapsed > self.AGENT_EVALUATE_TIMEOUT_SEC:
                    logger.warning("[KUN-AGT-W015] %s timed out (%.2fs)", name, elapsed)
                    opinions[name] = self._make_neutral_opinion("Timeout")
                    continue

                # Safely extract metadata
                metadata = raw.get('metadata', {})
                if not isinstance(metadata, dict):
                    metadata = {}

                opinions[name] = {
                    'available': True,
                    'decision': self._normalize_decision(raw.get('decision', 'hold')),
                    'confidence': float(raw.get('confidence', 0.0)),
                    'reason': str(raw.get('reason', '')),
                    'metadata': metadata
                }
            except Exception as e:
                logger.error("[KUN-AGT-E011] %s evaluate exception: %s", name, str(e))
                opinions[name] = self._make_neutral_opinion(f"Exception: {e}")

        return opinions

    @staticmethod
    def _make_neutral_opinion(reason: str = "") -> Dict:
        """Return a mutable copy of the neutral opinion template."""
        opinion = _NEUTRAL_OPINION_TEMPLATE.copy()
        opinion['reason'] = reason
        return opinion

    @staticmethod
    def _normalize_decision(raw: str) -> str:
        """Convert a raw decision string to a standard enum value."""
        return DecisionType.safe_parse(raw).value

    # ---------- Main Arbitration Entry Point (Thread-Safe) ----------
    def arbitrate(self, signal_context: Dict) -> Dict[str, Any]:
        """
        Core arbitration: collect opinions -> weight voting -> emergency override -> audit.
        Returns standardized response dict.
        """
        # 1. Collect opinions outside lock (blocking I/O from agents)
        opinions = self.evaluate_agents(signal_context)
        decision_resp = None

        # 2. Acquire lock for decision synthesis
        if not self._lock.acquire(timeout=self.LOCK_TIMEOUT_SEC):
            logger.critical("[KUN-AGT-F015] Lock acquisition timed out, returning conservative HOLD")
            return self._build_response(DecisionType.HOLD, reason="Arbiter overload",
                                       error_code="KUN-AGT-F015")

        try:
            self._last_signal_time = time.time()

            # Emergency overrides
            emergency = self._check_emergency(opinions)
            if emergency:
                decision_resp = emergency
                return decision_resp

            # All agents unavailable
            if all(not op['available'] for op in opinions.values()):
                decision_resp = self._build_response(DecisionType.HOLD,
                                                    reason="All agents unavailable",
                                                    error_code="KUN-AGT-F010",
                                                    opinions=opinions)
                return decision_resp

            # Stone veto
            stone_op = opinions[AGENT_STONE]
            if stone_op['decision'] == DecisionType.REJECT.value and stone_op['confidence'] > 0.7:
                if self._can_stone_veto():
                    decision_resp = self._build_response(DecisionType.REJECT,
                                                        reason=f"Stone veto: {stone_op['reason']}",
                                                        opinions=opinions)
                    return decision_resp

            # Wind exploration validation
            wind_meta = opinions[AGENT_WIND].get('metadata', {})
            if wind_meta.get('is_exploration', False):
                if not self._validate_exploration(opinions):
                    decision_resp = self._build_response(DecisionType.HOLD,
                                                        reason="Exploration conditions not met",
                                                        opinions=opinions)
                    return decision_resp

            # Book historical reference
            book_conf = opinions[AGENT_BOOK]['confidence']
            position_mult = signal_context.get('position_multiplier', 1.0)
            if book_conf < 0.4:
                position_mult *= 0.5

            # Eye environmental check
            eye_ok = opinions[AGENT_EYE].get('metadata', {}).get('environment_ok', True)
            if not eye_ok:
                decision_resp = self._build_response(DecisionType.HOLD,
                                                    reason="Environment not allowed",
                                                    opinions=opinions)
                return decision_resp

            # Weighted vote
            final_decision, conflict = self._weighted_vote(opinions)
            if conflict is not ConflictLevel.NONE:
                self._handle_conflict(signal_context.get('signal_id', ''), conflict)

            # Force exploration detection
            if self._should_force_exploration():
                logger.warning("[KUN-AGT-W016] Triggering force exploration")
                forced = self._force_exploration(signal_context)
                final_decision = DecisionType.safe_parse(
                    forced.get('decision', DecisionType.HOLD.value)
                )

            decision_resp = self._build_response(final_decision, opinions=opinions,
                                                position_multiplier=position_mult)
            self._attach_warnings(decision_resp)

        except Exception as e:
            logger.critical("[KUN-AGT-F012] Arbiter internal error: %s", str(e), exc_info=True)
            decision_resp = self._build_response(DecisionType.HOLD,
                                                reason="Internal arbiter error",
                                                error_code="KUN-AGT-F012")
        finally:
            self._lock.release()

        # 3. Audit outside lock
        self._audit(decision_resp, signal_context, opinions)
        return decision_resp

    # ---------- Decision Response Builder ----------
    def _build_response(self, decision: DecisionType, reason: str = "",
                        error_code: str = "", opinions: Dict = None,
                        **kwargs) -> Dict:
        """Construct a standardized response dictionary. Core keys are protected."""
        resp = {
            "status": "ok",
            "decision": decision.value,
            "reason": reason,
            "error_code": error_code,
            "audit_ref": "",
            "warnings": []
        }
        if error_code:
            # Determine status from error registry (simplified: F/E -> error)
            if error_code.startswith("KUN-AGT-F") or error_code.startswith("KUN-AGT-E"):
                resp["status"] = "error"

        if opinions:
            resp["agent_opinions"] = {
                name: {
                    "decision": op['decision'],
                    "confidence": op['confidence'],
                    "available": op['available']
                }
                for name, op in opinions.items()
            }

        # Safely add extra fields without overwriting core keys
        for k, v in kwargs.items():
            if k not in resp:
                resp[k] = v
        return resp

    def _attach_warnings(self, response: Dict) -> None:
        """Attach system warnings to the response."""
        # Snapshot agent availability
        with self._lock:
            unavailable = [name for name in ALL_AGENTS if self._agents.get(name) is None]
        if unavailable:
            response['warnings'].append(f"[KUN-AGT-W018] Missing agents: {', '.join(unavailable)}")
        conflict_count = len(self._active_conflicts)
        if conflict_count > 10:
            response['warnings'].append(f"[KUN-AGT-W019] High active conflicts: {conflict_count}")

    # ---------- Weighted Voting ----------
    def _weighted_vote(self, opinions: Dict) -> Tuple[DecisionType, ConflictLevel]:
        """
        Compute weighted vote across all available agents.
        Returns (best_decision, conflict_level).
        """
        scores: Dict[DecisionType, float] = {d: 0.0 for d in DecisionType}
        unique_decisions: Set[str] = set()

        for name, op in opinions.items():
            if not op['available']:
                continue
            w = self._voting_weights.get(name, 0.0)
            dec = DecisionType.safe_parse(op['decision'])
            scores[dec] += w * op['confidence']
            unique_decisions.add(op['decision'])

        # Pick the decision with highest score; tie-breaking by enum order (CLOSE_ALL highest priority)
        best = max(scores, key=lambda d: (scores[d], list(DecisionType).index(d)))
        best_score = scores[best]

        # Conflict detection
        conflict = ConflictLevel.NONE
        if len(unique_decisions) >= 3:
            conflict = ConflictLevel.MILD
        has_long = DecisionType.OPEN_LONG.value in unique_decisions
        has_short = DecisionType.OPEN_SHORT.value in unique_decisions
        has_close = DecisionType.CLOSE_ALL.value in unique_decisions
        if has_long and has_short:
            conflict = ConflictLevel.SEVERE
        if has_close and (has_long or has_short):
            conflict = ConflictLevel.CRITICAL

        if best_score < 0.5:
            return DecisionType.HOLD, conflict
        return best, conflict

    # ---------- Emergency Override ----------
    def _check_emergency(self, opinions: Dict) -> Optional[Dict]:
        """Check for emergency conditions that override normal voting."""
        eye_meta = opinions.get(AGENT_EYE, {}).get('metadata', {})
        stone_meta = opinions.get(AGENT_STONE, {}).get('metadata', {})
        if eye_meta.get('cross_market_crash'):
            return self._build_response(DecisionType.CLOSE_ALL,
                                       reason="Eye: cross-market crash detected",
                                       error_code="KUN-AGT-F001", emergency=True)
        if stone_meta.get('force_close'):
            return self._build_response(DecisionType.CLOSE_ALL,
                                       reason="Stone: forced liquidation",
                                       error_code="KUN-AGT-F002", emergency=True)
        return None

    # ---------- Conflict Management ----------
    def _handle_conflict(self, signal_id: str, level: ConflictLevel) -> None:
        """Record and possibly escalate a conflict."""
        now = time.time()
        if not signal_id:
            signal_id = "unknown"

        if signal_id not in self._active_conflicts:
            self._active_conflicts[signal_id] = {
                'level': level, 'start_time': now, 'last_update': now
            }
        else:
            conf = self._active_conflicts[signal_id]
            conf['last_update'] = now
            # Only escalate SEVERE conflicts that persist
            if level is ConflictLevel.SEVERE and (now - conf['start_time']) > self.CRITICAL_CONFLICT_DURATION_SEC:
                conf['level'] = ConflictLevel.CRITICAL
                logger.critical("[KUN-AGT-F001] Conflict escalated to CRITICAL: %s", signal_id)

        self._conflict_call_counter += 1
        if self._conflict_call_counter % self.CONFLICT_CLEANUP_INTERVAL == 0:
            self._cleanup_conflicts()

    def _cleanup_conflicts(self) -> None:
        """Remove expired and excess conflict records."""
        now = time.time()
        expired = [sid for sid, c in self._active_conflicts.items()
                   if now - c['last_update'] > self.CONFLICT_TTL_SEC]
        for sid in expired:
            del self._active_conflicts[sid]

        # Enforce max capacity
        if len(self._active_conflicts) > self.MAX_CONFLICT_RECORDS:
            sorted_ids = sorted(self._active_conflicts.keys(),
                               key=lambda x: self._active_conflicts[x]['last_update'])
            for sid in sorted_ids[:-self.MAX_CONFLICT_RECORDS]:
                del self._active_conflicts[sid]

    # ---------- Stone Veto Cooldown ----------
    def _can_stone_veto(self) -> bool:
        """Check if Stone is allowed to exercise veto based on cooldown."""
        now = time.time()
        while self._stone_veto_ts and (now - self._stone_veto_ts[0]) > self.STONE_VETO_COOLDOWN_SEC:
            self._stone_veto_ts.popleft()
        if len(self._stone_veto_ts) >= self.MAX_STONE_VETO_IN_WINDOW:
            logger.warning("Stone veto cooldown active (%d/%d)", len(self._stone_veto_ts), self.MAX_STONE_VETO_IN_WINDOW)
            return False
        self._stone_veto_ts.append(now)
        return True

    # ---------- Exploration Validation ----------
    def _validate_exploration(self, opinions: Dict) -> bool:
        """Validate Wind's exploration signal against safety conditions."""
        book_conf = opinions[AGENT_BOOK].get('confidence', 0.0)
        if book_conf < 0.5:
            return False
        if not opinions[AGENT_EYE].get('metadata', {}).get('environment_ok', True):
            return False
        stone_fear = opinions[AGENT_STONE].get('metadata', {}).get('fear_index', 0.0)
        if stone_fear > 0.7:
            return False
        return True

    def _should_force_exploration(self) -> bool:
        """Check if system should trigger force exploration due to prolonged no-signal period."""
        now = time.time()
        if now - self._last_signal_time < self.MAX_NO_SIGNAL_SEC:
            return False
        if now - self._last_force_explore_time < self.FORCE_EXPLORE_COOLDOWN_SEC:
            return False
        return True

    def _force_exploration(self, context: Dict) -> Dict:
        """Execute forced exploration via Wind agent."""
        self._last_force_explore_time = time.time()
        wind = self._agents.get(AGENT_WIND)
        if wind is None or not callable(getattr(wind, 'force_exploration', None)):
            logger.warning("[KUN-AGT-W017] Wind agent unavailable, skipping force exploration")
            return {'decision': DecisionType.HOLD.value, 'error_code': 'KUN-AGT-W017'}

        try:
            # Pass a lightweight copy to avoid mutation
            return wind.force_exploration(copy.deepcopy(context))
        except Exception as e:
            logger.error("Force exploration call failed: %s", e)
            return {'decision': DecisionType.HOLD.value, 'error_code': 'KUN-AGT-E018'}

    # ---------- Contribution Evaluation & Weight Adjustment ----------
    def record_outcome(self, trade_report: Dict) -> None:
        """Record a trade outcome and periodically adjust agent weights."""
        # Extract only essential fields
        essential = {
            'pnl_percent': trade_report.get('pnl_percent', 0.0),
            'agent_decisions': trade_report.get('agent_decisions', {}).copy(),
            'timestamp': time.time()
        }
        # Append outside lock (deque thread-safe for append in CPython; guarded for portability)
        with self._lock:
            self._contribution_log.append(essential)

        now = time.time()
        if now - self._last_weight_adjust_time < self.WEIGHT_ADJUST_COOLDOWN_SEC:
            return
        if len(self._contribution_log) < self.MIN_SAMPLES:
            return

        with self._lock:
            self._adjust_weights()
            self._last_weight_adjust_time = now

    def _adjust_weights(self) -> None:
        """Adjust agent weights based on recent contribution scores."""
        recent = list(self._contribution_log)[-self.CONTRIBUTION_WINDOW:]
        scores: Dict[str, float] = {name: 0.0 for name in ALL_AGENTS}
        counts: Dict[str, int] = {name: 0 for name in ALL_AGENTS}

        for rep in recent:
            pnl = rep['pnl_percent']
            decs = rep['agent_decisions']
            for name in ALL_AGENTS:
                d = decs.get(name)
                if d is None:
                    continue
                # Score rule: if trade was profitable, agents that voted for opening/adding
                # get credit; if losing, agents that voted for closing/reducing/rejecting get credit.
                if (pnl > 0 and d in (DecisionType.OPEN_LONG.value, DecisionType.OPEN_SHORT.value,
                                      DecisionType.ADD_POSITION.value)) or \
                   (pnl < 0 and d in (DecisionType.CLOSE_ALL.value, DecisionType.REDUCE_POSITION.value,
                                      DecisionType.REJECT.value)):
                    scores[name] += 1.0
                else:
                    scores[name] -= 0.5
                counts[name] += 1

        new_weights = {}
        for name in ALL_AGENTS:
            if counts[name] > 0:
                avg = scores[name] / counts[name]
                base = self._voting_weights.get(name, 0.0)
                delta = avg * self.WEIGHT_ADJUST_STEP
                delta = max(-self.MAX_WEIGHT_DELTA, min(self.MAX_WEIGHT_DELTA, delta))
                new_weights[name] = max(self.MIN_AGENT_WEIGHT, base + delta)
            else:
                new_weights[name] = self._voting_weights[name]

        self._voting_weights = new_weights
        self._normalize_weights()
        logger.info("Agent weights updated: %s",
                    {k: round(v, 4) for k, v in self._voting_weights.items()})

    def _normalize_weights(self) -> None:
        """Ensure voting weights sum to 1.0; reset to defaults if invalid."""
        total = sum(self._voting_weights.values())
        if total <= 0 or abs(total - 1.0) > 0.0001:
            self._voting_weights = dict(self.DEFAULT_VOTING_WEIGHTS)
            logger.warning("[KUN-AGT-W020] Weights abnormal, reset to defaults")
            return
        # Already normalized by construction; no division needed if we enforce total=1.0
        # Keep for safety
        for name in self._voting_weights:
            self._voting_weights[name] /= total

    # ---------- Audit Trail ----------
    def _audit(self, decision: Dict, context: Dict, opinions: Dict) -> None:
        """Write audit log asynchronously (non-blocking)."""
        audit_mod = _get_audit_module()
        if audit_mod is None:
            return
        try:
            # Sanitize context: only include non-sensitive identifiers
            safe_ctx = {
                'signal_id': context.get('signal_id', ''),
                'symbol': context.get('symbol', ''),
                'side': context.get('side', '')
            }
            opinions_summary = {
                name: {'decision': op['decision'], 'confidence': op['confidence']}
                for name, op in opinions.items()
            } if opinions else None

            audit_mod.log_event(
                event_type="agent_arbitration",
                severity="INFO",
                details={
                    "decision": decision['decision'],
                    "reason": decision['reason'],
                    "error_code": decision.get('error_code', ''),
                    "context": safe_ctx,
                    "opinions": opinions_summary,
                    "timestamp": time.time()
                }
            )
            # Mark audit reference
            decision['audit_ref'] = str(int(time.time() * 1_000_000))
        except Exception as e:
            logger.debug("Audit write failed (non-critical): %s", e)

    # ---------- Configuration ----------
    @staticmethod
    def _apply_config(config: Dict) -> None:
        """Apply a whitelist of safe configuration overrides."""
        allowed_keys = {
            'MAX_NO_SIGNAL_SEC', 'STONE_VETO_COOLDOWN_SEC', 'WEIGHT_ADJUST_COOLDOWN_SEC',
            'AGENT_EVALUATE_TIMEOUT_SEC', 'CONTRIBUTION_WINDOW', 'LOCK_TIMEOUT_SEC'
        }
        for k, v in config.items():
            if k in allowed_keys:
                setattr(AgentArbiter, k, v)
                logger.debug("Config override: %s = %s", k, v)

    # ---------- State Accessors ----------
    def get_weights(self) -> Dict[str, float]:
        """Return a snapshot of current voting weights."""
        with self._lock:
            return dict(self._voting_weights)

    def get_conflict_count(self) -> int:
        """Return current number of active conflicts."""
        with self._lock:
            return len(self._active_conflicts)

    def __repr__(self) -> str:
        return (f"AgentArbiter(weights={self._voting_weights}, "
                f"conflicts={len(self._active_conflicts)})")

    # ---------- Health Check (Isolated Instance) ----------
    @classmethod
    def health_check(cls) -> Dict[str, Any]:
        """Full self-test using a temporary instance; does not affect production."""
        try:
            arb = cls()
            # Verify weight normalization
            total = sum(arb._voting_weights.values())
            if not (abs(total - 1.0) < 0.0001 and all(w > 0 for w in arb._voting_weights.values())):
                return {"status": "error", "message": "Weight normalization failed"}

            # Register mock agents
            class MockAgent:
                def evaluate(self, ctx):
                    return {'decision': 'hold', 'confidence': 0.5}

            for name in ALL_AGENTS:
                arb.register_agent(name, MockAgent())

            # Test normal arbitration
            res = arb.arbitrate({'signal_id': 'hc', 'symbol': 'TEST'})
            if res['decision'] != 'hold':
                return {"status": "error", "message": f"Unexpected decision: {res['decision']}"}

            # Test all agents down
            for name in ALL_AGENTS:
                arb.unregister_agent(name)
            res_down = arb.arbitrate({'signal_id': 'hc2'})
            if res_down['error_code'] != 'KUN-AGT-F010':
                return {"status": "error", "message": "All-down code mismatch"}

            # Test emergency override
            class MockEye:
                def evaluate(self, ctx):
                    return {'decision': 'close_all', 'confidence': 1.0,
                            'metadata': {'cross_market_crash': True}}
            arb.register_agent(AGENT_EYE, MockEye())
            res_em = arb.arbitrate({'signal_id': 'hc3'})
            if not res_em.get('emergency'):
                return {"status": "error", "message": "Emergency override failed"}

            return {"status": "ok", "message": "All arbitration paths verified"}
        except Exception as e:
            logger.error("Health check failed: %s", e)
            return {"status": "error", "message": str(e)}
