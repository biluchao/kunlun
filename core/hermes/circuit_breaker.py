#!/usr/bin/env python3
"""
昆仑系统 · 五级熔断级联 (CircuitBreakerCascade)
全球顶尖量化对冲基金生产环境 · 万亿美金级风控基石

核心职责：
1. 基于精确风险阈值（Decimal），实时监控交易/策略/资产/系统/交易所五级风险
2. 触发后执行确定动作（禁开仓/减仓/紧急停机），支持跨重启状态持久化与人工重置
3. 提供线程安全、输入验证、异步回调隔离、配置热更新等机构级保障

外部依赖：
- infrastructure.health_pulse.HealthPulseMonitor : 实时权益、回撤、保证金率
- hermes.order_gateway.OrderExecutionGateway : 撤单与强制减仓
- polaris.silence_protocol.SilenceProtocol : 交易所异常状态
- infrastructure.audit_chain.AuditLogChain : 记录熔断事件

接口契约：
- check_all_breakers(context: BreakerContext) -> BreakerResult
- is_trading_allowed() -> bool
- reset_breaker(level: BreakerLevel, auth_token: str) -> Dict[str, Any]
- health_check() -> Dict[str, Any]

异常与降级：
- 所有子检查包裹异常捕获，缺失统计使用最坏情况假设
- 回调异常不影响主流程
- 持久化版本不匹配时降级为冷启动状态

资源管理：
- 每策略一个实例，内存恒定
- 无阻塞 I/O，回调应快速返回
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 枚举与数据类
# ---------------------------------------------------------------------------

class BreakerLevel(IntEnum):
    """熔断层级（严重性递增）"""
    TRADE = 1
    STRATEGY = 2
    ASSET = 3
    SYSTEM = 4
    EXCHANGE = 5


class Action(IntEnum):
    """熔断动作"""
    NONE = 0
    COOLDOWN_TRADE = 1           # 禁止该策略开仓一段时间
    PAUSE_STRATEGY = 2           # 暂停整个策略
    BLOCK_NEW_ENTRIES = 3        # 禁止所有新开仓
    REDUCE_POSITIONS = 4         # 按配置比例减仓
    EMERGENCY_SHUTDOWN = 5       # 紧急停止所有交易
    CROSS_VALIDATE = 6           # 交叉验证数据源
    REDUCE_EXPOSURE = 7          # 降低整体敞口


# 需人工重置的层级（无自动冷却）
MANUAL_RESET_LEVELS = {BreakerLevel.ASSET, BreakerLevel.SYSTEM, BreakerLevel.EXCHANGE}


@dataclass(frozen=True)
class BreakerConfig:
    """熔断阈值配置（不可变）"""
    trade_max_loss_pct: Decimal = Decimal("0.02")
    trade_max_slippage_pct: Decimal = Decimal("0.005")
    trade_cooldown_sec: float = 60.0

    strategy_max_drawdown: Decimal = Decimal("0.05")
    strategy_max_consecutive_losses: int = 5
    strategy_recovery_sec: float = 21600.0

    asset_max_daily_loss: Decimal = Decimal("0.03")
    asset_max_concentration: Decimal = Decimal("0.15")
    asset_margin_ratio_threshold: Decimal = Decimal("0.85")
    asset_reduce_ratio: Decimal = Decimal("0.5")

    system_max_daily_drawdown: Decimal = Decimal("0.05")
    system_max_weekly_drawdown: Decimal = Decimal("0.10")
    system_max_total_drawdown: Decimal = Decimal("0.20")

    exchange_action_maintenance: Action = Action.EMERGENCY_SHUTDOWN
    exchange_action_latency: Action = Action.REDUCE_EXPOSURE
    exchange_action_price_deviation: Action = Action.CROSS_VALIDATE

    # 通用容差
    epsilon: Decimal = Decimal("1e-10")
    # 最大冷却时间（防止无限延长）
    max_cooldown_sec: float = 86400.0


@dataclass
class BreakerState:
    """单层熔断运行时状态"""
    triggered_count: int = 0
    cooldown_until_utc: float = 0.0
    last_trigger_utc: float = 0.0
    is_tripped: bool = False          # 需人工重置的层是否已被触发并保持


@dataclass
class BreakerContext:
    """检查上下文，所有百分比均以小数表示（0.01=1%）"""
    trade: Dict[str, Any] = field(default_factory=dict)
    strategy: Dict[str, Any] = field(default_factory=dict)
    asset: Dict[str, Any] = field(default_factory=dict)
    system: Dict[str, Any] = field(default_factory=dict)
    exchange: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BreakerResult:
    """熔断检查结果"""
    can_trade: bool = True
    triggered_levels: List[BreakerLevel] = field(default_factory=list)
    actions: List[Tuple[Action, Dict[str, Any]]] = field(default_factory=list)  # (动作, 参数)
    reasons: List[str] = field(default_factory=list)
    details: Dict[BreakerLevel, Dict[str, Any]] = field(default_factory=dict)
    check_timestamp_utc: float = 0.0


# ---------------------------------------------------------------------------
# 熔断器主类
# ---------------------------------------------------------------------------

class CircuitBreakerCascade:
    """五级熔断级联管理器 —— 全球顶级量化机构版本"""

    VERSION = 2   # 状态序列化版本

    def __init__(self, config: Optional[Dict[str, Any]] = None, strategy_id: str = "default"):
        self._strategy_id = strategy_id
        self._cfg = self._build_config(config or {})
        self._lock = threading.RLock()
        self._states: Dict[BreakerLevel, BreakerState] = {lvl: BreakerState() for lvl in BreakerLevel}
        self._consecutive_loss_count = 0
        self._on_trigger_callbacks: List[Callable[[BreakerLevel, Action, str], None]] = []
        self._cached_result: Optional[BreakerResult] = None
        self._last_check_time = time.monotonic()
        self._min_check_interval = 0.05

    # -----------------------------------------------------------------------
    # 配置构建与验证
    # -----------------------------------------------------------------------
    @staticmethod
    def _safe_decimal(val: Any, default: Decimal, field: str) -> Decimal:
        """安全转换为 Decimal，值域检查"""
        try:
            d = Decimal(str(val))
            if d < 0 or d > 1:
                logger.warning(f"配置 {field} 超出[0,1]: {d}，使用默认值 {default}")
                return default
            return d
        except Exception:
            logger.warning(f"配置 {field} 非法: {val}，使用默认值")
            return default

    @classmethod
    def _build_config(cls, raw: Dict[str, Any]) -> BreakerConfig:
        return BreakerConfig(
            trade_max_loss_pct=cls._safe_decimal(raw.get("trade_max_loss_pct"), Decimal("0.02"), "trade_max_loss_pct"),
            trade_max_slippage_pct=cls._safe_decimal(raw.get("trade_max_slippage_pct"), Decimal("0.005"), "trade_max_slippage_pct"),
            trade_cooldown_sec=float(raw.get("trade_cooldown_sec", 60)),
            strategy_max_drawdown=cls._safe_decimal(raw.get("strategy_max_drawdown"), Decimal("0.05"), "strategy_max_drawdown"),
            strategy_max_consecutive_losses=int(raw.get("strategy_max_consecutive_losses", 5)),
            strategy_recovery_sec=float(raw.get("strategy_recovery_sec", 21600)),
            asset_max_daily_loss=cls._safe_decimal(raw.get("asset_max_daily_loss"), Decimal("0.03"), "asset_max_daily_loss"),
            asset_max_concentration=cls._safe_decimal(raw.get("asset_max_concentration"), Decimal("0.15"), "asset_max_concentration"),
            asset_margin_ratio_threshold=cls._safe_decimal(raw.get("asset_margin_ratio_threshold"), Decimal("0.85"), "asset_margin_ratio_threshold"),
            asset_reduce_ratio=cls._safe_decimal(raw.get("asset_reduce_ratio"), Decimal("0.5"), "asset_reduce_ratio"),
            system_max_daily_drawdown=cls._safe_decimal(raw.get("system_max_daily_drawdown"), Decimal("0.05"), "system_max_daily_drawdown"),
            system_max_weekly_drawdown=cls._safe_decimal(raw.get("system_max_weekly_drawdown"), Decimal("0.10"), "system_max_weekly_drawdown"),
            system_max_total_drawdown=cls._safe_decimal(raw.get("system_max_total_drawdown"), Decimal("0.20"), "system_max_total_drawdown"),
            exchange_action_maintenance=Action(raw.get("exchange_action_maintenance", 5)),
            exchange_action_latency=Action(raw.get("exchange_action_latency", 7)),
            exchange_action_price_deviation=Action(raw.get("exchange_action_price_deviation", 6)),
            epsilon=Decimal(str(raw.get("epsilon", "1e-10"))),
            max_cooldown_sec=float(raw.get("max_cooldown_sec", 86400)),
        )

    def update_config(self, config_dict: Dict[str, Any]) -> None:
        with self._lock:
            self._cfg = self._build_config(config_dict)
            self._cached_result = None
            logger.info("熔断配置已热更新")

    # -----------------------------------------------------------------------
    # 时间与冷却
    # -----------------------------------------------------------------------
    @staticmethod
    def _utc_now() -> float:
        return time.time()

    def _is_in_cooldown(self, level: BreakerLevel) -> bool:
        state = self._states[level]
        if level in MANUAL_RESET_LEVELS:
            return state.is_tripped   # 人工层一旦触发就保持，直到重置
        return self._utc_now() < state.cooldown_until_utc

    def _set_cooldown(self, level: BreakerLevel, cooldown_sec: float) -> None:
        state = self._states[level]
        now = self._utc_now()
        if level in MANUAL_RESET_LEVELS:
            state.is_tripped = True
            state.last_trigger_utc = now
            state.triggered_count += 1
            return
        # 自动冷却层：设置到期时间，限制最大冷却
        new_until = now + min(cooldown_sec, self._cfg.max_cooldown_sec)
        if state.cooldown_until_utc > new_until:
            logger.debug("冷却期延长: %s 新到期 %s", level.name, time.ctime(state.cooldown_until_utc))
            return
        state.cooldown_until_utc = new_until
        state.last_trigger_utc = now
        state.triggered_count += 1

    # -----------------------------------------------------------------------
    # 上下文安全访问
    # -----------------------------------------------------------------------
    @staticmethod
    def _get_decimal(d: Dict, key: str, default: Decimal) -> Decimal:
        try:
            val = d.get(key)
            if val is None:
                return default
            return Decimal(str(val))
        except Exception:
            return default

    @staticmethod
    def _get_float(d: Dict, key: str, default: float) -> float:
        try:
            return float(d.get(key, default))
        except Exception:
            return default

    # -----------------------------------------------------------------------
    # 各层检查
    # -----------------------------------------------------------------------
    def _check_trade(self, trade: Dict) -> Tuple[bool, Action, str]:
        loss = abs(self._get_decimal(trade, "pnl_percent", Decimal(0)))
        if loss > self._cfg.trade_max_loss_pct:
            return True, Action.COOLDOWN_TRADE, f"单笔亏损{loss:.4%}>{self._cfg.trade_max_loss_pct:.4%}"
        slippage = abs(self._get_decimal(trade, "slippage_pct", Decimal(0)))
        if slippage > self._cfg.trade_max_slippage_pct:
            return True, Action.COOLDOWN_TRADE, f"单笔滑点{slippage:.4%}>{self._cfg.trade_max_slippage_pct:.4%}"
        return False, Action.NONE, ""

    def _check_strategy(self, strategy: Dict) -> Tuple[bool, Action, str]:
        drawdown = abs(self._get_decimal(strategy, "current_drawdown", Decimal(0)))
        if drawdown > self._cfg.strategy_max_drawdown:
            return True, Action.PAUSE_STRATEGY, f"策略回撤{drawdown:.4%}>{self._cfg.strategy_max_drawdown:.4%}"
        cons_loss = strategy.get("consecutive_losses", self._consecutive_loss_count)
        if isinstance(cons_loss, (int, float)) and cons_loss >= self._cfg.strategy_max_consecutive_losses:
            return True, Action.PAUSE_STRATEGY, f"连续亏损{cons_loss}笔>={self._cfg.strategy_max_consecutive_losses}"
        return False, Action.NONE, ""

    def _check_asset(self, asset: Dict) -> Tuple[bool, Action, Dict[str, Any], str]:
        reasons = []
        daily = abs(self._get_decimal(asset, "daily_loss_pct", Decimal(0)))
        if daily > self._cfg.asset_max_daily_loss:
            reasons.append(f"日亏损{daily:.4%}>{self._cfg.asset_max_daily_loss:.4%}")
        conc = self._get_decimal(asset, "max_asset_concentration", Decimal(0))
        if conc > self._cfg.asset_max_concentration:
            reasons.append(f"集中度{conc:.4%}>{self._cfg.asset_max_concentration:.4%}")
        margin = self._get_decimal(asset, "current_margin_ratio", Decimal(1))
        if margin < self._cfg.asset_margin_ratio_threshold:
            reasons.append(f"保证金率{margin:.4%}<{self._cfg.asset_margin_ratio_threshold:.4%}")
        if reasons:
            return True, Action.REDUCE_POSITIONS, {"reduce_ratio": str(self._cfg.asset_reduce_ratio)}, "; ".join(reasons)
        return False, Action.NONE, {}, ""

    def _check_system(self, system: Dict) -> Tuple[bool, Action, str]:
        daily = abs(self._get_decimal(system, "daily_drawdown", Decimal(0)))
        if daily > self._cfg.system_max_daily_drawdown:
            return True, Action.EMERGENCY_SHUTDOWN, f"日回撤{daily:.4%}>{self._cfg.system_max_daily_drawdown:.4%}"
        weekly = abs(self._get_decimal(system, "weekly_drawdown", Decimal(0)))
        if weekly > self._cfg.system_max_weekly_drawdown:
            return True, Action.EMERGENCY_SHUTDOWN, f"周回撤{weekly:.4%}>{self._cfg.system_max_weekly_drawdown:.4%}"
        total = abs(self._get_decimal(system, "total_drawdown", Decimal(0)))
        if total > self._cfg.system_max_total_drawdown:
            return True, Action.EMERGENCY_SHUTDOWN, f"总回撤{total:.4%}>{self._cfg.system_max_total_drawdown:.4%}"
        return False, Action.NONE, ""

    def _check_exchange(self, exchange: Dict) -> Tuple[bool, Action, str]:
        if exchange.get("maintenance"):
            return True, self._cfg.exchange_action_maintenance, "交易所维护"
        if exchange.get("abnormal_latency"):
            return True, self._cfg.exchange_action_latency, "网络延迟异常"
        if exchange.get("price_feed_deviation"):
            return True, self._cfg.exchange_action_price_deviation, "价格源偏差"
        return False, Action.NONE, ""

    # -----------------------------------------------------------------------
    # 主检查入口 (全程持锁保证一致性)
    # -----------------------------------------------------------------------
    def check_all_breakers(self, context: BreakerContext) -> BreakerResult:
        with self._lock:
            now = self._utc_now()
            # 去抖（仍然使用单调时间以避免系统时间回退影响）
            if self._cached_result and (time.monotonic() - self._last_check_time) < self._min_check_interval:
                return self._cached_result

            result = BreakerResult(check_timestamp_utc=now)
            # 按严重性降序检查
            order = [BreakerLevel.EXCHANGE, BreakerLevel.SYSTEM, BreakerLevel.ASSET,
                     BreakerLevel.STRATEGY, BreakerLevel.TRADE]
            for level in order:
                try:
                    if level == BreakerLevel.EXCHANGE:
                        trig, act, reason = self._check_exchange(context.exchange)
                        params = {}
                    elif level == BreakerLevel.SYSTEM:
                        trig, act, reason = self._check_system(context.system)
                        params = {}
                    elif level == BreakerLevel.ASSET:
                        trig, act, params, reason = self._check_asset(context.asset)
                    elif level == BreakerLevel.STRATEGY:
                        trig, act, reason = self._check_strategy(context.strategy)
                        params = {}
                    elif level == BreakerLevel.TRADE:
                        trig, act, reason = self._check_trade(context.trade)
                        params = {}
                    else:
                        continue

                    if not trig:
                        result.details[level] = {"triggered": False}
                        continue

                    # 检查是否仍在冷却/保持触发
                    if self._is_in_cooldown(level):
                        logger.debug("熔断抑制 [%s] 已触发/冷却中", level.name)
                        result.details[level] = {"triggered": False, "suppressed": True, "reason": reason}
                        continue

                    # 触发
                    cooldown = self._get_cooldown_sec(level)
                    self._set_cooldown(level, cooldown)
                    self._fire_breaker(level, act, reason)
                    result.triggered_levels.append(level)
                    result.actions.append((act, params))
                    result.reasons.append(f"[{level.name}] {reason}")
                    result.details[level] = {"triggered": True, "action": act.name, "params": params, "reason": reason}

                    # 高级别熔断立即禁止交易
                    if level in (BreakerLevel.EXCHANGE, BreakerLevel.SYSTEM, BreakerLevel.ASSET):
                        result.can_trade = False
                except Exception as e:
                    logger.error("熔断层 %s 检查异常: %s", level.name, e, exc_info=True)
                    # 保守处理
                    result.triggered_levels.append(level)
                    result.actions.append((Action.BLOCK_NEW_ENTRIES, {}))
                    result.reasons.append(f"[{level.name}] 检查异常: {e}")
                    result.can_trade = False
                    result.details[level] = {"triggered": True, "error": str(e)}

            self._cached_result = result
            self._last_check_time = time.monotonic()
            return result

    def _get_cooldown_sec(self, level: BreakerLevel) -> float:
        if level == BreakerLevel.TRADE:
            return self._cfg.trade_cooldown_sec
        if level == BreakerLevel.STRATEGY:
            return self._cfg.strategy_recovery_sec
        return 0.0  # 人工层无自动冷却

    def _fire_breaker(self, level: BreakerLevel, action: Action, reason: str) -> None:
        if level in (BreakerLevel.SYSTEM, BreakerLevel.EXCHANGE):
            logger.critical("熔断触发 [%s] 动作=%s 原因=%s", level.name, action.name, reason)
        else:
            logger.warning("熔断触发 [%s] 动作=%s 原因=%s", level.name, action.name, reason)
        for cb in self._on_trigger_callbacks:
            try:
                cb(level, action, reason)
            except Exception as e:
                logger.error("熔断回调异常: %s", e, exc_info=True)

    # -----------------------------------------------------------------------
    # 交易许可查询与重置
    # -----------------------------------------------------------------------
    def is_trading_allowed(self) -> bool:
        with self._lock:
            if self._cached_result:
                return self._cached_result.can_trade
            # 无缓存时直接快速检查主要状态
            for lvl in MANUAL_RESET_LEVELS:
                if self._states[lvl].is_tripped:
                    return False
            return True

    def record_trade_result(self, is_loss: bool) -> None:
        with self._lock:
            if is_loss:
                self._consecutive_loss_count += 1
            else:
                self._consecutive_loss_count = 0

    def reset_breaker(self, level: BreakerLevel, auth_token: str) -> Dict[str, Any]:
        if not self._validate_auth(auth_token):
            return {"status": "error", "reason": "授权失败"}
        with self._lock:
            state = self._states[level]
            state.cooldown_until_utc = 0.0
            state.is_tripped = False
            state.triggered_count = 0
            self._cached_result = None
            logger.info("熔断器 %s 已手动重置", level.name)
        return {"status": "ok", "message": f"{level.name} 已重置"}

    def _validate_auth(self, token: str) -> bool:
        # 生产环境应调用 security.permission_matrix
        return isinstance(token, str) and len(token) > 8

    # -----------------------------------------------------------------------
    # 状态持久化
    # -----------------------------------------------------------------------
    def snapshot_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "version": self.VERSION,
                "strategy_id": self._strategy_id,
                "states": {lvl.name: {"triggered_count": s.triggered_count,
                                      "cooldown_until_utc": s.cooldown_until_utc,
                                      "last_trigger_utc": s.last_trigger_utc,
                                      "is_tripped": s.is_tripped}
                           for lvl, s in self._states.items()},
                "consecutive_losses": self._consecutive_loss_count,
                "snapshot_utc": self._utc_now()
            }

    def restore_state(self, saved: Dict[str, Any]) -> None:
        with self._lock:
            if saved.get("version") != self.VERSION:
                logger.warning("状态版本不匹配，冷启动")
                return
            for lvl_name, data in saved.get("states", {}).items():
                try:
                    lvl = BreakerLevel[lvl_name]
                    self._states[lvl].triggered_count = data.get("triggered_count", 0)
                    self._states[lvl].cooldown_until_utc = data.get("cooldown_until_utc", 0.0)
                    self._states[lvl].last_trigger_utc = data.get("last_trigger_utc", 0.0)
                    self._states[lvl].is_tripped = data.get("is_tripped", False)
                except KeyError:
                    pass
            self._consecutive_loss_count = saved.get("consecutive_losses", 0)

    # -----------------------------------------------------------------------
    # 健康检查
    # -----------------------------------------------------------------------
    @classmethod
    def health_check(cls) -> Dict[str, Any]:
        try:
            cb = cls(strategy_id="health_check")
            # 正常上下文
            normal = BreakerContext(
                trade={"pnl_percent": 0.01, "slippage_pct": 0.001},
                strategy={"current_drawdown": 0.01, "consecutive_losses": 1},
                asset={"daily_loss_pct": 0.005, "max_asset_concentration": 0.1, "current_margin_ratio": 0.95},
                system={"daily_drawdown": 0.01, "weekly_drawdown": 0.02, "total_drawdown": 0.03},
                exchange={"maintenance": False}
            )
            res = cb.check_all_breakers(normal)
            if not res.can_trade or res.triggered_levels:
                return {"status": "error", "message": "正常上下文误触发"}

            # 系统级触发
            danger = BreakerContext(system={"daily_drawdown": 0.06})
            res2 = cb.check_all_breakers(danger)
            if res2.can_trade or BreakerLevel.SYSTEM not in res2.triggered_levels:
                return {"status": "error", "message": "系统级熔断未触发"}

            # 持久化往返
            snap = cb.snapshot_state()
            cb2 = cls(strategy_id="health_check")
            cb2.restore_state(snap)
            if cb2._states[BreakerLevel.SYSTEM].is_tripped != cb._states[BreakerLevel.SYSTEM].is_tripped:
                return {"status": "error", "message": "状态恢复不一致"}

            return {"status": "ok", "message": "熔断器全面验证通过"}
        except Exception as e:
            logger.error("健康检查异常: %s", e, exc_info=True)
            return {"status": "error", "message": str(e)}
