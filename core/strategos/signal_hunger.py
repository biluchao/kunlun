#!/usr/bin/env python3
"""
Kunlun System · Signal Hunger Regulator (SHR) v4.0 — Institutional Grade

Core Responsibilities:
1. Dynamically adjust the system's "hunger" for trading signals based on a
   risk‑adjusted rolling performance window.  The window considers PnL,
   notional exposure, win rate, and consecutive streaks.
2. Provide periodic evaluation to gradually increase hunger when no signals
   have been generated for an extended period (combating strategy dormancy).
3. Detect when recent hunger adjustments have harmed performance (regret
   mechanism) and rollback, while penalising the adjustment direction on an
   instance level.  Gradual recovery of penalised step sizes is supported.
4. Enforce strict upper bounds on hunger under adverse market regimes, high
   Stone Fear Index, extreme volatility, and systemic drawdowns.

External Dependencies (real module interfaces):
- olympus.agent_arbiter.AgentArbiter : provides Stone Fear Index [0.0, 1.0]
- polaris.market_regime.MarketRegimeClassifier : market state & vol percentile

Interface Contract (every public method returns Dict[str, Any]):
- record_trade(trade_result: Dict) -> Dict[str, Any]
- periodic_update(env: Dict) -> Dict[str, Any]
- get_effective_hunger(env: Dict) -> Dict[str, Any]
- export_state() -> Dict[str, Any]
- import_state(state: Dict) -> None
- health_check() -> Dict[str, Any]

All dictionaries contain at least "status", "reason", "warnings".
"""

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Deque
from collections import deque

logger = logging.getLogger(__name__)

# ============================================================================
# Immutable default configuration
# ============================================================================
class _Defaults:
    """Class‑level defaults; instance configs are deep‑copied from these."""
    # Window & update
    EVAL_WINDOW_TRADES: int = 30
    MIN_TRADES_FOR_UPDATE: int = 5
    MIN_WINDOW_FILL_RATIO: float = 0.5         # require at least 50% full window

    # Hunger bounds
    INITIAL_HUNGER: float = 0.5
    MIN_HUNGER: float = 0.05
    MAX_HUNGER: float = 0.95

    # Threshold mapping
    BASE_THRESHOLD: float = 0.6
    THRESHOLD_RANGE: float = 0.25
    MIN_THRESHOLD: float = 0.35
    MAX_THRESHOLD: float = 0.85

    # Momentum smoothing
    MOMENTUM_DECAY: float = 0.7               # weight on previous momentum
    MAX_SINGLE_ADJUSTMENT: float = 0.10       # clip per‑update adjustment

    # Step sizes (direction‑aware, instance copies may be penalised)
    STEP_POSITIVE: float = 0.015
    STEP_CONSEC_WIN: float = 0.02
    STEP_NEGATIVE: float = -0.04
    STEP_CONSEC_LOSS: float = -0.06
    CONSEC_COUNT: int = 4

    # Signal starvation
    NO_SIGNAL_TIMEOUT_SEC: float = 1800.0
    NO_SIGNAL_ADJ_PER_HOUR: float = 0.02
    NO_SIGNAL_MAX_INCREMENT: float = 0.05

    # Drawdown guard
    MAX_DD_HUNGER_CAP: float = 0.20
    MAX_DD_THRESHOLD: float = 0.10

    # Volatility caps
    VOL_HIGH_PERCENTILE: float = 0.80
    VOL_EXTREME_PERCENTILE: float = 0.90
    VOL_HIGH_CAP: float = 0.40
    VOL_EXTREME_CAP: float = 0.25

    # Market state caps (deep‑copied per instance)
    MARKET_CAPS: Dict[str, float] = {
        "normal": 0.95,
        "trending_strong": 0.95,
        "cold_range": 0.70,
        "emotional": 0.50,
        "dormant": 0.30,
        "liquidity_trap": 0.20,
        "chaotic": 0.30,
        "unknown": 0.50
    }

    # Fear index dampening factor: fear_cap = raw * (1.0 - fear * FEAR_DAMPEN)
    FEAR_DAMPEN: float = 0.5                 # 0.5 → moderate dampening
    FEAR_HIGH_THRESHOLD: float = 0.6         # only apply when fear > this

    # Regret mechanism
    REGRET_WINDOW_TRADES: int = 10
    REGRET_LOSS_COUNT_THRESHOLD: int = 3
    REGRET_MIN_LOSS_MAGNITUDE: float = -0.005  # -0.5% – only meaningful losses
    REGRET_STEP_PENALTY: float = 0.5
    REGRET_MIN_STEP_ABS: float = 0.001        # floor for absolute step size
    REGRET_RECOVERY_FACTOR: float = 1.01      # geometric recovery toward original
    REGRET_COOLDOWN_TRADES: int = 5           # ignore further rollbacks for N trades

    # Position sizing / stop multipliers (hunger→ parameter)
    POSITION_MULT_BASE: float = 1.0
    POSITION_MULT_RANGE: float = 0.15         # max deviation from base
    POSITION_MULT_MIN: float = 0.80
    POSITION_MULT_MAX: float = 1.20
    STOP_MULT_BASE: float = 2.0
    STOP_MULT_RANGE: float = 0.8
    STOP_MULT_MIN: float = 1.5
    STOP_MULT_MAX: float = 3.0

    # Misc
    MICRO_PNL: float = 0.0005                 # 0.05% – considered flat
    EPS: float = 1e-12
    CLOCK = time.time                         # injectable for testing

# ============================================================================
# SignalHungerRegulator
# ============================================================================
class SignalHungerRegulator:
    """Adaptive signal hunger regulator – institutional production version."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Deep‑copy defaults (avoid shared mutable objects)
        self.cfg: Dict[str, Any] = {}
        for k, v in _Defaults.__dict__.items():
            if k.startswith('_') or callable(v):
                continue
            if isinstance(v, dict):
                self.cfg[k] = v.copy()
            else:
                self.cfg[k] = v
        # Additional aliases for fast access
        self.cfg['CLOCK'] = _Defaults.CLOCK

        if config:
            self._apply_config(config)

        # ---- State ----
        self._trades: Deque[Dict[str, Any]] = deque(
            maxlen=self.cfg['EVAL_WINDOW_TRADES'])
        self._raw_hunger: float = self.cfg['INITIAL_HUNGER']
        self._hunger_momentum: float = 0.0

        # Timestamps (epoch seconds)
        self._last_trade_time: float = 0.0
        self._last_signal_time: float = 0.0

        # Instance‑level step sizes (signed, may be penalised)
        self._step = {
            'positive': self.cfg['STEP_POSITIVE'],
            'consec_win': self.cfg['STEP_CONSEC_WIN'],
            'negative': self.cfg['STEP_NEGATIVE'],
            'consec_loss': self.cfg['STEP_CONSEC_LOSS']
        }

        # Regret history & cooldown
        self._regret_log: Deque[Dict[str, Any]] = deque(maxlen=20)
        self._regret_cooldown_remaining: int = 0

        # External risk injection
        self._external_max_dd: Optional[float] = None

        logger.info("SHR v4.0 initialized, hunger=%.3f", self._raw_hunger)

    # ======================================================================
    # Configuration
    # ======================================================================
    def _apply_config(self, user: Dict[str, Any]) -> None:
        """Merge user config with validation."""
        for key, value in user.items():
            if key not in self.cfg:
                logger.warning("Unknown config key: %s", key)
                continue
            if key == 'MARKET_CAPS' and isinstance(value, dict):
                # Merge instead of replace
                self.cfg['MARKET_CAPS'].update(value)
            else:
                # Basic type check / clamping
                if isinstance(self.cfg[key], (int, float)):
                    value = float(value)
                    if 'HUNGER' in key or 'THRESHOLD' in key or 'MULT' in key:
                        value = max(0.0, min(1.0, value))
                self.cfg[key] = value
        # Resync step sizes (if user changed the defaults)
        self._step['positive'] = self.cfg['STEP_POSITIVE']
        self._step['consec_win'] = self.cfg['STEP_CONSEC_WIN']
        self._step['negative'] = self.cfg['STEP_NEGATIVE']
        self._step['consec_loss'] = self.cfg['STEP_CONSEC_LOSS']

    # ======================================================================
    # State injection (setters return Dict to honour contract)
    # ======================================================================
    def set_external_drawdown(self, dd: Optional[float]) -> Dict[str, Any]:
        """Update systemic max drawdown (0.0‑1.0)."""
        if dd is not None and not (0.0 <= dd <= 1.0):
            return {"status": "error", "reason": f"Invalid drawdown: {dd}", "warnings": []}
        self._external_max_dd = dd
        return {"status": "ok", "reason": f"Drawdown set to {dd}", "warnings": []}

    def set_last_signal_time(self, ts: float) -> Dict[str, Any]:
        """Record when a valid signal was last generated."""
        if ts <= 0:
            return {"status": "error", "reason": "Timestamp must be > 0", "warnings": []}
        self._last_signal_time = ts
        return {"status": "ok", "reason": "Signal time updated", "warnings": []}

    # ======================================================================
    # Core update methods
    # ======================================================================
    def record_trade(self, trade_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a completed trade.
        Required: 'pnl_percent' (e.g. 0.005). Optional: 'notional_weight' (0‑1,
        fraction of account equity; default 1.0 if omitted but a warning is raised).
        """
        if 'pnl_percent' not in trade_result:
            return {"status": "error", "reason": "Missing 'pnl_percent'", "warnings": []}
        pnl = float(trade_result['pnl_percent'])
        if not (-1.0 <= pnl <= 1.0):
            return {"status": "error", "reason": f"pnl_percent out of range: {pnl}", "warnings": []}

        nw = trade_result.get('notional_weight')
        if nw is None:
            logger.warning("[KUN-SHR-W011] notional_weight missing, using 1.0")
            nw = 1.0
        else:
            nw = float(nw)
            if nw <= 0:
                return {"status": "error", "reason": "notional_weight must be > 0", "warnings": []}

        now = self.cfg['CLOCK']()
        self._last_trade_time = now

        self._trades.append({
            'pnl_percent': pnl,
            'notional_weight': nw,
            'timestamp': now
        })

        old_hunger = self._raw_hunger
        adjustment = self._compute_adjustment()
        # Clip per‑update adjustment
        max_adj = self.cfg['MAX_SINGLE_ADJUSTMENT']
        adjustment = max(-max_adj, min(max_adj, adjustment))

        decay = self.cfg['MOMENTUM_DECAY']
        self._hunger_momentum = decay * self._hunger_momentum + (1.0 - decay) * adjustment
        self._raw_hunger += self._hunger_momentum
        self._raw_hunger = max(self.cfg['MIN_HUNGER'],
                               min(self.cfg['MAX_HUNGER'], self._raw_hunger))

        # Regret evaluation (with cooldown)
        rollback = False
        if self._regret_cooldown_remaining <= 0:
            self._regret_log.append({
                "old_hunger": old_hunger,
                "new_hunger": self._raw_hunger,
                "adjustment": adjustment,
                "trade_count": len(self._trades)
            })
            reg = self._evaluate_regret()
            if reg["rollback"]:
                logger.warning("[KUN-SHR-W001] Regret rollback: %.3f -> %.3f",
                               self._raw_hunger, reg["restored_hunger"])
                self._raw_hunger = reg["restored_hunger"]
                self._hunger_momentum = 0.0
                self._regret_cooldown_remaining = self.cfg['REGRET_COOLDOWN_TRADES']
                rollback = True
        else:
            self._regret_cooldown_remaining -= 1

        # Gradual step‑size recovery
        self._recover_steps()

        return {
            "status": "ok",
            "reason": "Trade recorded",
            "raw_hunger": self._raw_hunger,
            "adjustment": adjustment,
            "regret_rollback": rollback,
            "warnings": []
        }

    def periodic_update(self, env: Dict[str, Any]) -> Dict[str, Any]:
        """
        Time‑based update (call every ~60s).
        env may contain:
            time_since_last_signal_sec: float (optional, if not provided we use
                                         internal _last_signal_time)
        """
        # Determine time since last signal
        if 'time_since_last_signal_sec' in env:
            t_since = float(env['time_since_last_signal_sec'])
        elif self._last_signal_time > 0:
            t_since = self.cfg['CLOCK']() - self._last_signal_time
        else:
            t_since = 0.0

        adjustment = 0.0
        timeout = self.cfg['NO_SIGNAL_TIMEOUT_SEC']
        if t_since > timeout and self._raw_hunger < 0.6:
            ms = env.get('market_state', 'normal')
            if ms not in ('dormant', 'liquidity_trap', 'chaotic'):
                hours = t_since / 3600.0
                inc = min(self.cfg['NO_SIGNAL_MAX_INCREMENT'],
                          self.cfg['NO_SIGNAL_ADJ_PER_HOUR'] * hours)
                # additionally cap to prevent overshooting
                inc = min(inc, 0.6 - self._raw_hunger)
                adjustment += inc
                logger.debug("Starvation boost: +%.4f", inc)

        if abs(adjustment) > 1e-9:
            self._raw_hunger += adjustment
            self._raw_hunger = max(self.cfg['MIN_HUNGER'],
                                   min(self.cfg['MAX_HUNGER'], self._raw_hunger))
            # do NOT change momentum for time‑based adjustment

        return {
            "status": "ok",
            "reason": "Periodic update",
            "raw_hunger": self._raw_hunger,
            "adjustment_applied": adjustment,
            "warnings": []
        }

    # ======================================================================
    # Effective hunger (with all environmental constraints)
    # ======================================================================
    def get_effective_hunger(self, env: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return final hunger and derived trading parameters.
        env: market_state (str), fear_index (float), volatility_percentile (float).
        """
        raw = self._raw_hunger
        caps_desc = []

        # 1. Volatility percentile caps
        vol_pct = float(env.get('volatility_percentile', 0.5))
        if vol_pct > self.cfg['VOL_EXTREME_PERCENTILE']:
            caps_desc.append(('vol_extreme', self.cfg['VOL_EXTREME_CAP']))
        elif vol_pct > self.cfg['VOL_HIGH_PERCENTILE']:
            caps_desc.append(('vol_high', self.cfg['VOL_HIGH_CAP']))

        # 2. Market state cap
        ms = env.get('market_state', 'normal')
        ms_cap = self.cfg['MARKET_CAPS'].get(ms, self.cfg['MARKET_CAPS']['unknown'])
        caps_desc.append(('market_state', ms_cap))

        # 3. Stone Fear dampening (applied as multiplicative cap on raw)
        fear = float(env.get('fear_index', 0.0))
        if fear > self.cfg['FEAR_HIGH_THRESHOLD']:
            factor = max(0.1, 1.0 - fear * self.cfg['FEAR_DAMPEN'])
            fear_cap = raw * factor
            caps_desc.append(('fear', fear_cap))
        else:
            fear_cap = None

        # 4. Drawdown cap
        if self._external_max_dd is not None and self._external_max_dd > self.cfg['MAX_DD_THRESHOLD']:
            caps_desc.append(('drawdown', self.cfg['MAX_DD_HUNGER_CAP']))

        # The effective hunger is the minimum of raw and all caps
        effective = raw
        for _, cap in caps_desc:
            if cap < effective:
                effective = cap

        effective = max(self.cfg['MIN_HUNGER'],
                        min(self.cfg['MAX_HUNGER'], effective))

        # Derived parameters
        base_th = self.cfg['BASE_THRESHOLD']
        rng = self.cfg['THRESHOLD_RANGE']
        threshold = base_th + rng * (1.0 - 2.0 * effective)
        threshold = max(self.cfg['MIN_THRESHOLD'],
                        min(self.cfg['MAX_THRESHOLD'], threshold))

        pos_mult = (self.cfg['POSITION_MULT_BASE'] +
                    self.cfg['POSITION_MULT_RANGE'] * (effective - 0.5))
        pos_mult = max(self.cfg['POSITION_MULT_MIN'],
                       min(self.cfg['POSITION_MULT_MAX'], pos_mult))

        stop_mult = (self.cfg['STOP_MULT_BASE'] +
                     self.cfg['STOP_MULT_RANGE'] * (effective - 0.5))
        stop_mult = max(self.cfg['STOP_MULT_MIN'],
                        min(self.cfg['STOP_MULT_MAX'], stop_mult))

        warnings = []
        if effective < 0.2:
            warnings.append("[KUN-SHR-W002] Hunger critically low")
        if fear > 0.8:
            warnings.append("[KUN-SHR-W003] High Stone Fear constraining hunger")

        return {
            "status": "ok",
            "reason": f"Effective hunger computed with {len(caps_desc)} caps",
            "hunger_level": effective,
            "raw_hunger": raw,
            "factor_threshold": threshold,
            "position_multiplier": pos_mult,
            "stop_multiplier": stop_mult,
            "applied_caps": [desc for desc, _ in caps_desc],
            "warnings": warnings
        }

    # ======================================================================
    # Internal analytics
    # ======================================================================
    def _compute_adjustment(self) -> float:
        """Compute hunger adjustment from the current trade window."""
        min_req = self.cfg['MIN_TRADES_FOR_UPDATE']
        if len(self._trades) < min_req:
            return 0.0
        # Also require window to be sufficiently filled
        fill_ratio = len(self._trades) / self.cfg['EVAL_WINDOW_TRADES']
        if fill_ratio < self.cfg['MIN_WINDOW_FILL_RATIO']:
            return 0.0

        window = list(self._trades)
        n = len(window)
        half_life = max(1, n // 2)
        decay = math.exp(-math.log(2) / half_life)
        weights = [decay ** (n - 1 - i) for i in range(n)]
        total_w = sum(weights) + self.cfg['EPS']

        micro = self.cfg['MICRO_PNL']
        weighted_pnl = 0.0
        pnl_list = []
        wins = 0
        for i, t in enumerate(window):
            # PnL contribution = percentage × notional weight
            contrib = t['pnl_percent'] * t['notional_weight']
            weighted_pnl += contrib * weights[i]
            pnl_list.append(contrib)
            if contrib > micro:
                wins += 1
        weighted_pnl /= total_w
        win_rate = wins / n if n > 0 else 0.0

        # Consecutive streaks (most recent first)
        consec_wins = 0
        for val in reversed(pnl_list):
            if val > micro:
                consec_wins += 1
            else:
                break
        consec_losses = 0
        for val in reversed(pnl_list):
            if val < -micro:
                consec_losses += 1
            else:
                break

        adj = 0.0
        cc = self.cfg['CONSEC_COUNT']
        if weighted_pnl > 0.002 and win_rate > 0.5:
            adj += self._step['positive']
        if consec_wins >= cc:
            adj += self._step['consec_win']
        if weighted_pnl < -0.002 or win_rate < 0.35:
            adj += self._step['negative']
        if consec_losses >= cc:
            adj += self._step['consec_loss']

        return adj

    def _evaluate_regret(self) -> Dict[str, Any]:
        """Check if recent trades indicate the last adjustment was harmful."""
        if len(self._regret_log) < 2 or len(self._trades) < self.cfg['REGRET_WINDOW_TRADES']:
            return {"rollback": False}

        recent = list(self._trades)[-self.cfg['REGRET_WINDOW_TRADES']:]
        loss_threshold = self.cfg['REGRET_MIN_LOSS_MAGNITUDE']
        loss_count = sum(1 for t in recent
                         if t['pnl_percent'] * t['notional_weight'] < loss_threshold)

        if loss_count >= self.cfg['REGRET_LOSS_COUNT_THRESHOLD']:
            last = self._regret_log[-1]
            # Penalise the direction
            if last["adjustment"] > 0:
                self._step['positive'] *= self.cfg['REGRET_STEP_PENALTY']
                self._step['consec_win'] *= self.cfg['REGRET_STEP_PENALTY']
            else:
                self._step['negative'] *= self.cfg['REGRET_STEP_PENALTY']
                self._step['consec_loss'] *= self.cfg['REGRET_STEP_PENALTY']
            # Enforce minimum absolute value
            for k in self._step:
                self._step[k] = (math.copysign(
                    max(self.cfg['REGRET_MIN_STEP_ABS'], abs(self._step[k])),
                    self._step[k]))
            return {"rollback": True, "restored_hunger": last["old_hunger"]}
        return {"rollback": False}

    def _recover_steps(self) -> None:
        """Gradually move penalised step sizes back to their original values."""
        orig = {
            'positive': self.cfg['STEP_POSITIVE'],
            'consec_win': self.cfg['STEP_CONSEC_WIN'],
            'negative': self.cfg['STEP_NEGATIVE'],
            'consec_loss': self.cfg['STEP_CONSEC_LOSS']
        }
        factor = self.cfg['REGRET_RECOVERY_FACTOR']
        for k in self._step:
            target = orig[k]
            cur = self._step[k]
            if abs(cur) < abs(target):
                # Increase absolute value while preserving sign
                new_abs = min(abs(target), abs(cur) * factor)
                self._step[k] = math.copysign(new_abs, target)

    # ======================================================================
    # State persistence
    # ======================================================================
    def export_state(self) -> Dict[str, Any]:
        """Export full state for hot‑reload. Includes complete trade window."""
        return {
            "raw_hunger": self._raw_hunger,
            "hunger_momentum": self._hunger_momentum,
            "last_trade_time": self._last_trade_time,
            "last_signal_time": self._last_signal_time,
            "step_sizes": dict(self._step),
            "regret_cooldown": self._regret_cooldown_remaining,
            "trades": list(self._trades)  # full window
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        """Import state and validate."""
        self._raw_hunger = float(state.get("raw_hunger", self.cfg['INITIAL_HUNGER']))
        self._raw_hunger = max(self.cfg['MIN_HUNGER'],
                               min(self.cfg['MAX_HUNGER'], self._raw_hunger))
        self._hunger_momentum = float(state.get("hunger_momentum", 0.0))
        self._last_trade_time = float(state.get("last_trade_time", 0.0))
        self._last_signal_time = float(state.get("last_signal_time", 0.0))
        self._regret_cooldown_remaining = int(state.get("regret_cooldown", 0))

        steps = state.get("step_sizes", {})
        for k in self._step:
            if k in steps:
                self._step[k] = math.copysign(
                    max(self.cfg['REGRET_MIN_STEP_ABS'], abs(float(steps[k]))),
                    self._step[k])  # preserve original sign direction

        self._trades.clear()
        for t in state.get("trades", []):
            self._trades.append(t)
        logger.info("State imported, hunger=%.3f, trades restored=%d",
                    self._raw_hunger, len(self._trades))

    # ======================================================================
    # Health check
    # ======================================================================
    @classmethod
    def health_check(cls) -> Dict[str, Any]:
        """Comprehensive self‑test."""
        try:
            shr = cls()
            # 1. Initial state
            assert abs(shr._raw_hunger - 0.5) < 1e-9

            # 2. Record trades
            shr.record_trade({"pnl_percent": 0.02, "notional_weight": 0.5})
            shr.record_trade({"pnl_percent": -0.01, "notional_weight": 0.5})
            assert 0.0 < shr._raw_hunger < 1.0

            # 3. Periodic update
            shr.set_last_signal_time(time.time() - 3600)
            shr.periodic_update({})
            # Not much change expected

            # 4. Effective hunger under extreme conditions
            env = {"market_state": "chaotic", "fear_index": 0.9,
                   "volatility_percentile": 0.95}
            res = shr.get_effective_hunger(env)
            assert res["hunger_level"] <= 0.3, "Chaotic environment not capped"

            # 5. State export/import round‑trip
            state = shr.export_state()
            shr2 = cls()
            shr2.import_state(state)
            assert abs(shr2._raw_hunger - shr._raw_hunger) < 1e-9

            # 6. Regret & recovery
            for _ in range(15):
                shr.record_trade({"pnl_percent": -0.02, "notional_weight": 1.0})
            # Hunger should be reduced
            assert shr._raw_hunger < 0.5

            return {"status": "ok", "reason": "All checks passed", "warnings": []}
        except Exception as e:
            logger.error("Health check failed: %s", e)
            return {"status": "error", "reason": str(e), "warnings": []}
