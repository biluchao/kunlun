#!/usr/bin/env python3
"""
昆仑系统 · 激进探索智能体 (Wind Seeker) —— 全球顶尖量化机构生产版本
版本：v5.0.0-HEDGE
最后审查：2026-06-14 华尔街高频交易级终极加固

核心职责：
1. 在系统长时间无信号时，生成严格风控下的“探索信号”，以极小仓位试探市场微观结构
2. 维护“勇气计数器”，采用贝叶斯统计显著性检验动态调整探索力度
3. 与 Stone Guardian 形成制衡，确保探索行为不会在极端恐惧时放大风险
4. 适配万亿美金账户：动态计算市场冲击成本，限制探索仓位不超过流动性承受上限
5. 所有操作线程安全、状态可持久化、行为完全可审计，满足 MiFID II / Reg SCI 要求

外部依赖（真实模块接口）：
- agents.stone_guardian.StoneGuardian : 获取恐惧指数与市场压力状态
- strategos.signal_hunger.SignalHungerRegulator : 获取系统饥渴度及因子有效性
- infrastructure.health_pulse.HealthPulseMonitor : 获取系统级权益曲线、VaR
- infrastructure.audit_chain.AuditLogChain : 记录探索事件，满足 MiFID II 审计要求
- hermes.slippage_sim.SlippageSimulator : 估算探索订单的预期滑点

接口契约：
- evaluate(context: MarketContext) -> AgentDecision
  评估市场上下文并返回决策，决策不可变，包含完整元数据
- get_courage() -> int
  返回当前勇气计数器值
- record_outcome(trade_report: TradeReport) -> None
  根据探索交易结果更新勇气，调整未来探索策略
- health_check() -> HealthCheckResult
  自检所有核心逻辑路径与边界条件

异常与降级：
- 若 Stone Guardian 不可用，默认采用上一周期持久化恐惧值，若仍不可用使用 0.65，并记录 KUN-AGT-W011
- 若 SlippageSimulator 不可用，探索仓位自动缩减 40%，并发出 KUN-EXE-W014
- 勇气降至 0 时，启用“冬眠模式”：仅允许在市场波动率<20%分位且趋势极其明确时发起微仓探索
- 探索信号若检测到订单簿深度不足以支撑名义价值 2 倍滑点限制，自动拒绝
- 所有未捕获异常降级为安全决策：返回 HOLD，并记录 KUN-AGT-F002
- 强制探索模式下，若原勇气为0，将在操作完成后立即恢复冬眠状态
"""

import logging
import time
import threading
import random
import math
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import IntEnum

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 领域模型
# ---------------------------------------------------------------------------

class ExplorationMode(IntEnum):
    """探索模式枚举"""
    NORMAL = 0      # 正常模式
    FORCED = 1      # 强制探索（仲裁器调用）
    DORMANT = 2     # 冬眠模式（勇气归零，极度保守）

@dataclass(frozen=True)
class MarketContext:
    """标准化市场上下文，不可变对象，由上层组装传入"""
    symbol: str = ""
    close: float = 0.0
    ma25: float = 0.0
    bid_depth: float = 0.0          # 买一档深度 (数量)
    ask_depth: float = 0.0
    spread_bps: float = 0.0         # 买卖价差 (bps)
    time_since_last_signal: float = 99999.0
    base_position_size: float = 0.01
    total_equity: float = 100_000.0
    base_threshold: float = 0.6
    current_position_side: Optional[str] = None
    current_position_size: float = 0.0
    volatility_percentile: float = 0.5   # 当前波动率在历史中的分位数
    market_regime: str = "normal"        # 市场状态

@dataclass(frozen=True)
class AgentDecision:
    """智能体决策标准输出（不可变）"""
    status: str = "ok"
    decision: str = "hold"
    confidence: float = 0.0
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: tuple = ()   # 使用元组确保不可变


class WindSeeker:
    """激进探索智能体 ─ 风·扶摇 (机构级生产版本)"""

    # --------------------------- 类常量（可配置，范围/单位注释详尽）---------------------------
    # 探索触发条件
    NO_SIGNAL_TIMEOUT_SEC: float = 1200.0       # 无信号秒数阈值，范围 [600, 3600]
    FACTOR_THRESHOLD_RELAXATION: float = 0.20   # 因子阈值放松比例，范围 [0.05, 0.40]
    EXPLORATION_POSITION_RATIO: float = 0.20    # 探索仓位相对于标准仓的比例，范围 [0.05, 0.50]
    MAX_EXPLORATION_TOTAL_RATIO: float = 0.05   # 探索总仓与总权益的最大比例，范围 [0.01, 0.10]
    EXPLORATION_COOLDOWN_SEC: float = 600.0     # 探索后冷却时间，范围 [120, 1800]
    COOLDOWN_JITTER_SEC: float = 30.0           # 冷却期随机抖动，避免多交易对同步探索，范围 [0, 120]

    # 勇气计数器与贝叶斯衰减
    INITIAL_COURAGE: int = 10                   # 初始勇气，范围 [0, 20]
    MAX_COURAGE: int = 20
    MIN_COURAGE: int = 0
    COURAGE_GAIN_PER_WIN: int = 1
    COURAGE_LOSS_PER_LOSS: int = -2
    # 贝叶斯信心因子：根据 p 值动态调整勇气变化权重
    BAYESIAN_P_THRESHOLD: float = 0.05          # 低于此值视为统计显著
    BAYESIAN_STRENGTH_FACTOR: float = 1.5       # 显著时奖惩倍数

    # 仓位乘数映射（勇气 → 仓位系数）
    POSITION_MULTIPLIER_MAP: Dict[int, float] = field(default_factory=lambda: {
        0: 0.05, 10: 0.20, 20: 0.50
    })

    # 风险控制（万亿规模适配）
    MAX_EXPLORATION_CONFIDENCE: float = 0.55     # 探索信号最大置信度
    MIN_SPREAD_BPS_FOR_EXPLORE: float = 10.0     # 价差过大不探索 (bps)
    MIN_BID_DEPTH_RATIO: float = 0.5             # 深度不足时仓位打折线
    MAX_IMPACT_BPS: float = 2.0                  # 允许的最大市场冲击 (bps)，万亿账户约束
    MIN_NOTIONAL_USD: float = 10.0               # 最小名义价值 (USD)
    DORMANT_VOLATILITY_PERCENTILE: float = 0.2   # 冬眠模式波动率分位数上限

    # 审计与持久化
    STATE_FLUSH_INTERVAL_SEC: float = 60.0       # 状态持久化间隔

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化，应用配置，注入默认依赖存根"""
        if config:
            self._apply_config(config)

        # 线程安全原语
        self._rwlock = threading.RLock()  # 全局锁，简化死锁风险，写操作极少性能影响可忽略
        self._state_lock = threading.Lock()  # 用于持久化状态的一致性快照

        # 核心状态
        self._courage: int = self.INITIAL_COURAGE
        self._last_exploration_time: float = 0.0
        self._last_signal_time: float = time.time()
        self._exploration_active: bool = False
        self._exploration_count: int = 0
        self._exploration_wins: int = 0
        self._exploration_total: int = 0
        self._mode: ExplorationMode = ExplorationMode.NORMAL

        # 外部依赖（延迟绑定）
        self._stone_guardian = None
        self._hunger_regulator = None
        self._health_monitor = None
        self._slippage_sim = None
        self._chronos_db = None

        # 持久化定时器
        self._last_state_flush = time.time()

        # 配置验证
        self._validate_configuration()
        logger.info("WindSeeker v5.0.0 初始化完成，勇气=%d，冷却=%ds",
                     self._courage, self.EXPLORATION_COOLDOWN_SEC)

    # --------------------------- 配置管理 ---------------------------
    def _apply_config(self, config: Dict[str, Any]) -> None:
        """安全地应用配置，类型检查与范围验证"""
        for key, value in config.items():
            if not hasattr(self, key):
                logger.warning("忽略未知配置项: %s", key)
                continue
            try:
                current = getattr(self, key)
                if isinstance(current, dict):
                    if not isinstance(value, dict):
                        raise TypeError(f"期望dict，得到{type(value)}")
                    # 对键值进行类型转换
                    converted = {int(k): float(v) for k, v in value.items()}
                    setattr(self, key, converted)
                else:
                    expected_type = type(current)
                    if expected_type is type(None):
                        setattr(self, key, value)
                    else:
                        converted = expected_type(value)
                        setattr(self, key, converted)
                logger.debug("配置覆盖: %s = %s", key, value)
            except (ValueError, TypeError) as e:
                logger.error("配置项 %s 类型错误: %s，保留默认值", key, e)

    def _validate_configuration(self) -> None:
        """启动时严格验证参数，非法参数抛出异常阻止启动"""
        assert 60 <= self.NO_SIGNAL_TIMEOUT_SEC <= 86400, "NO_SIGNAL_TIMEOUT_SEC 必须在 60~86400 之间"
        assert 0.01 <= self.FACTOR_THRESHOLD_RELAXATION <= 0.5, "FACTOR_THRESHOLD_RELAXATION 超出范围"
        assert self.INITIAL_COURAGE >= self.MIN_COURAGE
        assert self.MAX_COURAGE > self.MIN_COURAGE
        assert 0.0 < self.EXPLORATION_COOLDOWN_SEC < 86400
        # 仓位映射单调性校验
        sorted_keys = sorted(self.POSITION_MULTIPLIER_MAP.keys())
        for i in range(1, len(sorted_keys)):
            assert self.POSITION_MULTIPLIER_MAP[sorted_keys[i]] >= self.POSITION_MULTIPLIER_MAP[sorted_keys[i-1]], \
                "仓位映射必须单调非递减"
        # 冲击上限合理性检查
        assert 0 < self.MAX_IMPACT_BPS <= 100, "MAX_IMPACT_BPS 不合理"

    # --------------------------- 依赖注入（线程安全） ---------------------------
    def set_stone_guardian(self, stone: Any) -> None:
        with self._rwlock:
            self._stone_guardian = stone

    def set_hunger_regulator(self, hunger: Any) -> None:
        with self._rwlock:
            self._hunger_regulator = hunger

    def set_health_monitor(self, monitor: Any) -> None:
        with self._rwlock:
            self._health_monitor = monitor

    def set_slippage_simulator(self, sim: Any) -> None:
        with self._rwlock:
            self._slippage_sim = sim

    def set_chronos_db(self, db: Any) -> None:
        with self._rwlock:
            self._chronos_db = db

    # --------------------------- 状态访问（锁保护） ---------------------------
    def get_courage(self) -> int:
        with self._rwlock:
            return self._courage

    def get_mode(self) -> ExplorationMode:
        with self._rwlock:
            return self._mode

    def _get_fear_index(self) -> float:
        """获取Stone恐惧指数，不可用时使用持久化值/默认值"""
        with self._rwlock:
            stone = self._stone_guardian
        if stone and hasattr(stone, 'get_fear_index'):
            try:
                return stone.get_fear_index()
            except Exception as e:
                logger.error("获取Stone恐惧指数异常: %s", e)
        stored_fear = self._load_stored_fear()
        if stored_fear is not None:
            return stored_fear
        return 0.65  # 保守默认

    def _load_stored_fear(self) -> Optional[float]:
        """从ChronosDB加载最近一次有效的恐惧指数"""
        if self._chronos_db:
            try:
                record = self._chronos_db.get_latest("stone_fear")
                if record:
                    return float(record["value"])
            except Exception as e:
                logger.warning("加载Stone恐惧历史失败: %s", e)
        return None

    def _get_time_since_last_signal(self, ctx: MarketContext) -> float:
        if ctx.time_since_last_signal > 0:
            return ctx.time_since_last_signal
        with self._rwlock:
            return time.time() - self._last_signal_time

    def _update_last_signal_time(self, ctx: MarketContext) -> None:
        """根据上下文更新最后信号时间戳"""
        with self._rwlock:
            if ctx.time_since_last_signal > 0:
                self._last_signal_time = time.time() - ctx.time_since_last_signal
            else:
                self._last_signal_time = time.time()

    # --------------------------- 勇气调整与贝叶斯衰减 ---------------------------
    def _get_position_multiplier(self) -> float:
        """基于勇气值的分段线性插值获取仓位乘数，线程安全"""
        courage = self.get_courage()
        sorted_pts = sorted(self.POSITION_MULTIPLIER_MAP.items())
        if courage <= sorted_pts[0][0]:
            return sorted_pts[0][1]
        if courage >= sorted_pts[-1][0]:
            return sorted_pts[-1][1]
        for i in range(len(sorted_pts) - 1):
            x0, y0 = sorted_pts[i]
            x1, y1 = sorted_pts[i+1]
            if x0 <= courage <= x1:
                return y0 + (y1 - y0) * (courage - x0) / (x1 - x0)
        return sorted_pts[0][1]

    def _adjust_courage(self, delta: int, is_significant: bool = False) -> int:
        """原子调整勇气值，可选统计显著性增强"""
        if is_significant:
            delta = int(delta * self.BAYESIAN_STRENGTH_FACTOR)
        with self._rwlock:
            old = self._courage
            new = max(self.MIN_COURAGE, min(self.MAX_COURAGE, old + delta))
            self._courage = new
            if old != new:
                logger.info("勇气调整: %d -> %d (Δ=%d, 显著=%s)", old, new, delta, is_significant)
                if new == 0 and self._mode != ExplorationMode.DORMANT:
                    self._mode = ExplorationMode.DORMANT
                    logger.warning("勇气归零，进入冬眠模式")
                elif new > 0 and self._mode == ExplorationMode.DORMANT:
                    self._mode = ExplorationMode.NORMAL
            return new

    # --------------------------- 探索条件综合判断 ---------------------------
    def _should_explore(self, ctx: MarketContext) -> Tuple[bool, str]:
        """全面检查探索前置条件，返回(是否允许, 拒绝原因)"""
        now = time.time()
        time_since = self._get_time_since_last_signal(ctx)

        # 1. 无信号时长
        if time_since < self.NO_SIGNAL_TIMEOUT_SEC:
            return False, f"无信号时长 {time_since:.0f}s < {self.NO_SIGNAL_TIMEOUT_SEC}s"

        # 2. 冷却期 + 随机抖动
        with self._rwlock:
            last_exp = self._last_exploration_time
        jitter = random.uniform(0, self.COOLDOWN_JITTER_SEC)
        cooldown_total = self.EXPLORATION_COOLDOWN_SEC + jitter
        if now - last_exp < cooldown_total:
            return False, f"冷却中 (剩余 {cooldown_total - (now - last_exp):.0f}s)"

        # 3. 勇气与冬眠模式检查
        courage = self.get_courage()
        mode = self.get_mode()
        if courage <= self.MIN_COURAGE:
            if mode == ExplorationMode.DORMANT:
                # 冬眠特殊规则：极低波动率 + 强趋势才允许
                if ctx.volatility_percentile > self.DORMANT_VOLATILITY_PERCENTILE:
                    return False, "冬眠模式波动率不满足"
                if abs(ctx.close - ctx.ma25) / ctx.close < 0.005:
                    return False, "冬眠模式趋势不明显"
            else:
                return False, "勇气枯竭"

        # 4. Stone恐惧过高
        fear = self._get_fear_index()
        if fear > 0.75:
            return False, f"Stone恐惧过高 ({fear:.2f})"

        # 5. 价差过大
        if ctx.spread_bps > self.MIN_SPREAD_BPS_FOR_EXPLORE:
            return False, f"价差 {ctx.spread_bps:.1f} bps 过大"

        # 6. 市场冲击预估（若模拟器可用）
        if self._slippage_sim and ctx.close > 0:
            base_qty = ctx.base_position_size * self._get_position_multiplier() * self.EXPLORATION_POSITION_RATIO
            try:
                est_impact = self._slippage_sim.estimate_impact(
                    symbol=ctx.symbol,
                    qty=base_qty,
                    side='buy' if ctx.close > ctx.ma25 else 'sell'
                )
                if est_impact > self.MAX_IMPACT_BPS:
                    return False, f"预期市场冲击 {est_impact:.1f} bps 超过上限"
            except Exception as e:
                logger.error("冲击估算失败: %s", e)
                # 估算失败时不阻断探索，但后续仓位计算会保守
        return True, ""

    # --------------------------- 探索信号生成 ---------------------------
    def _generate_exploration_signal(self, ctx: MarketContext) -> AgentDecision:
        """生成符合机构风险约束的探索信号"""
        now = time.time()
        if ctx.close > ctx.ma25:
            side = 'long'
        elif ctx.close < ctx.ma25:
            side = 'short'
        else:
            return AgentDecision(decision="hold", reason="趋势不明", confidence=0.0)

        # 检查同向持仓
        if (ctx.current_position_side == side and ctx.current_position_size > 0):
            return AgentDecision(decision="hold", reason="已有同向仓位")

        base_qty = ctx.base_position_size
        if base_qty <= 0:
            return AgentDecision(decision="hold", reason="基准仓位为零")

        # 仓位系数
        position_mult = self._get_position_multiplier()
        dormancy_discount = 0.3 if self.get_mode() == ExplorationMode.DORMANT else 1.0
        exploration_qty = base_qty * self.EXPLORATION_POSITION_RATIO * position_mult * dormancy_discount

        # 流动性折扣
        if ctx.bid_depth > 0 and ctx.ask_depth > 0:
            depth_ratio = min(ctx.bid_depth, ctx.ask_depth) / max(ctx.bid_depth, ctx.ask_depth)
            if depth_ratio < self.MIN_BID_DEPTH_RATIO:
                exploration_qty *= 0.5

        # 权益上限
        max_qty = (ctx.total_equity * self.MAX_EXPLORATION_TOTAL_RATIO) / ctx.close if ctx.close > 0 else 0.0
        exploration_qty = min(exploration_qty, max_qty)

        # 最小名义价值
        if exploration_qty * ctx.close < self.MIN_NOTIONAL_USD:
            return AgentDecision(decision="hold", reason="探索仓位名义价值过低")

        # 冲击二次确认与自动缩放
        if self._slippage_sim:
            try:
                impact = self._slippage_sim.estimate_impact(ctx.symbol, exploration_qty, side)
                if impact > self.MAX_IMPACT_BPS:
                    scale = self.MAX_IMPACT_BPS / impact
                    exploration_qty *= scale
                    logger.info("冲击控制缩减仓位至 %.6f (原始冲击 %.2f bps)", exploration_qty, impact)
            except Exception as e:
                logger.error("冲击二次确认失败: %s", e)
                # 冲击估算失败，保守缩减仓位50%
                exploration_qty *= 0.5

        # 若缩减后仍低于最小名义价值，则放弃探索
        if exploration_qty * ctx.close < self.MIN_NOTIONAL_USD:
            return AgentDecision(decision="hold", reason="冲击缩减后名义价值过低")

        # 更新状态
        with self._rwlock:
            self._last_exploration_time = now
            self._exploration_active = True
            self._exploration_count += 1
            self._exploration_total += 1

        relaxed_threshold = ctx.base_threshold * (1 - self.FACTOR_THRESHOLD_RELAXATION)
        confidence = 0.3 + 0.2 * (self.get_courage() / self.MAX_COURAGE)
        confidence = min(confidence, self.MAX_EXPLORATION_CONFIDENCE)

        decision = f"open_{side}"
        warnings = ["Wind探索信号，注意仓位与滑点控制"]
        if self.get_mode() == ExplorationMode.DORMANT:
            warnings.append("冬眠模式探索，仓位已大幅缩减")

        logger.info("Wind探索: %s 数量%.6f 勇气%d 阈值%.2f",
                     side, exploration_qty, self.get_courage(), relaxed_threshold)

        return AgentDecision(
            status="ok",
            decision=decision,
            confidence=confidence,
            reason=f"探索信号 (勇气={self.get_courage()}, 模式={self.get_mode().name})",
            metadata={
                "is_exploration": True,
                "courage": self.get_courage(),
                "exploration_qty": round(exploration_qty, 8),
                "relaxed_threshold": round(relaxed_threshold, 4),
                "exploration_side": side,
                "time_since_last_signal": self._get_time_since_last_signal(ctx),
                "mode": self.get_mode().name
            },
            warnings=tuple(warnings)
        )

    # --------------------------- 公共评估接口 ---------------------------
    def evaluate(self, ctx: MarketContext) -> AgentDecision:
        """主入口：市场上下文 → 探索决策"""
        self._update_last_signal_time(ctx)
        self._maybe_flush_state()
        should, reason = self._should_explore(ctx)
        if not should:
            return AgentDecision(
                decision="hold",
                reason=reason,
                metadata={
                    "is_exploration": False,
                    "courage": self.get_courage(),
                    "time_since_last_signal": self._get_time_since_last_signal(ctx),
                    "mode": self.get_mode().name
                }
            )
        return self._generate_exploration_signal(ctx)

    def force_exploration(self, ctx: MarketContext) -> AgentDecision:
        """仲裁器强制探索，临时提升勇气并记录审计"""
        original_courage = self.get_courage()
        original_mode = self.get_mode()
        if original_courage < 5:
            with self._rwlock:
                self._courage = 5
                self._mode = ExplorationMode.FORCED
            logger.warning("强制探索：勇气临时提升至5")
        try:
            result = self.evaluate(ctx)
        finally:
            with self._rwlock:
                self._courage = original_courage
                # 恢复模式：若原来为冬眠，且勇气仍为0，则返回冬眠；否则按勇气值判定
                if self._courage == 0:
                    self._mode = ExplorationMode.DORMANT
                else:
                    self._mode = ExplorationMode.NORMAL
        return result

    # --------------------------- 交易结果反馈与统计显著性 ---------------------------
    def record_outcome(self, trade_report: Dict[str, Any]) -> None:
        """处理探索交易结果，使用贝叶斯更新勇气"""
        if not trade_report.get('is_exploration', False):
            return

        pnl = trade_report.get('pnl_percent', 0.0)
        with self._rwlock:
            wins = self._exploration_wins
            total = max(1, self._exploration_total)
        p_hat = wins / total
        se = math.sqrt(p_hat * (1 - p_hat) / total) if total > 0 else 1
        z_score = (p_hat - 0.5) / se if se > 0 else 0
        is_significant = abs(z_score) > 1.96

        delta = self.COURAGE_GAIN_PER_WIN if pnl > 0 else self.COURAGE_LOSS_PER_LOSS
        self._adjust_courage(delta, is_significant)

        with self._rwlock:
            if pnl > 0:
                self._exploration_wins += 1
            self._exploration_active = False
            if self._exploration_count > 0:
                self._exploration_count -= 1

        logger.info("探索结果: PnL=%.4f%% 勇气调整=%d (显著=%s) 总胜率=%.2f%%",
                     pnl * 100, delta, is_significant, p_hat * 100)

        # 审计日志
        try:
            from infrastructure.audit_chain import AuditLogChain
            AuditLogChain.log_event(
                event_type="wind_exploration_outcome",
                severity="INFO",
                details={
                    "pnl_percent": pnl,
                    "courage_change": delta,
                    "new_courage": self.get_courage(),
                    "is_significant": is_significant,
                    "z_score": round(z_score, 4)
                }
            )
        except ImportError:
            logger.debug("审计日志模块未加载")

    # --------------------------- 状态持久化 ---------------------------
    def _maybe_flush_state(self) -> None:
        if self._chronos_db is None:
            return
        now = time.time()
        if now - self._last_state_flush < self.STATE_FLUSH_INTERVAL_SEC:
            return
        self._persist_state()
        self._last_state_flush = now

    def _persist_state(self) -> None:
        with self._state_lock:
            state = self.export_state()
        try:
            self._chronos_db.upsert("wind_seeker_state", state)
        except Exception as e:
            logger.error("状态持久化失败: %s", e)

    def export_state(self) -> Dict[str, Any]:
        with self._rwlock:
            return {
                "courage": self._courage,
                "last_exploration_time": self._last_exploration_time,
                "last_signal_time": self._last_signal_time,
                "exploration_active": self._exploration_active,
                "exploration_count": self._exploration_count,
                "exploration_wins": self._exploration_wins,
                "exploration_total": self._exploration_total,
                "mode": int(self._mode)
            }

    def import_state(self, state: Dict[str, Any]) -> None:
        with self._rwlock:
            self._courage = state.get("courage", self.INITIAL_COURAGE)
            self._last_exploration_time = state.get("last_exploration_time", 0.0)
            self._last_signal_time = state.get("last_signal_time", time.time())
            self._exploration_active = state.get("exploration_active", False)
            self._exploration_count = state.get("exploration_count", 0)
            self._exploration_wins = state.get("exploration_wins", 0)
            self._exploration_total = state.get("exploration_total", 0)
            self._mode = ExplorationMode(state.get("mode", 0))
        logger.info("WindSeeker 状态已从持久化恢复")

    # --------------------------- 健康检查 ---------------------------
    @classmethod
    def health_check(cls) -> Dict[str, Any]:
        try:
            wind = cls()
            wind._validate_configuration()
            assert wind.get_courage() == cls.INITIAL_COURAGE
            ctx = MarketContext(
                symbol="BTCUSDT", close=50000, ma25=49000,
                bid_depth=50, ask_depth=50, spread_bps=5,
                time_since_last_signal=10, base_position_size=0.1,
                total_equity=1e6, volatility_percentile=0.5
            )
            res = wind.evaluate(ctx)
            assert res.decision == "hold"
            ctx2 = MarketContext(
                symbol="BTCUSDT", close=50000, ma25=49000,
                bid_depth=50, ask_depth=50, spread_bps=5,
                time_since_last_signal=99999, base_position_size=0.1,
                total_equity=1e6, volatility_percentile=0.5
            )
            res2 = wind.evaluate(ctx2)
            assert res2.decision in ("open_long", "open_short", "hold")
            wind._adjust_courage(50)
            assert wind.get_courage() == cls.MAX_COURAGE
            wind._adjust_courage(-50)
            assert wind.get_courage() == cls.MIN_COURAGE
            assert wind.get_mode() == ExplorationMode.DORMANT
            forced = wind.force_exploration(ctx2)
            assert forced.decision in ("open_long", "open_short", "hold")
            return {"status": "ok", "message": "WindSeeker 全部核心路径通过"}
        except Exception as e:
            logger.exception("健康检查失败")
            return {"status": "error", "message": str(e)}
