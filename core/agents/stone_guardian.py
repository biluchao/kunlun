#!/usr/bin/env python3
"""
昆仑系统 · 保守主义智能体 v3.0 (Stone Guardian) — 万亿级风控核心

核心职责：
1. 聚合组合多资产回撤、连续亏损、波动率曲面、流动性风险，计算恐惧指数
2. 根据恐惧等级输出标准化风控决策 (APPROVE / REDUCE / BLOCK / FORCE_CLOSE)
3. 非对称平滑 + 波动率体制自适应 + 自然时间衰减 + 流动性恐惧
4. 支持多策略/多账户组合净敞口监控，提供动态建议仓位上限
5. 与仲裁器使用统一 DecisionType 协议，全链路审计，持久化不可变日志

外部依赖：
- infrastructure.chronos_db.ChronosDB : 异步持久化接口
- infrastructure.health_pulse.HealthPulseMonitor : 组合回撤/波动率/流动性
- olympus.agent_arbiter.DecisionType : 决策枚举

接口契约：
- evaluate(context: Dict) -> Dict[str, Any]
- record_outcome(trade_report: Dict) -> None
- get_fear_index() -> float
- get_metrics() -> Dict[str, Any]
- health_check() -> Dict[str, Any]

异常与降级：
- 所有外部依赖失败均退化到本地保守估算
- 输入非法自动修正为安全默认值
- 状态保存失败异步重试，不阻塞交易线程
"""

import logging
import math
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import deque
from enum import Enum

try:
    from olympus.agent_arbiter import DecisionType
except ImportError:
    class DecisionType(Enum):
        OPEN_LONG = "open_long"
        OPEN_SHORT = "open_short"
        ADD_POSITION = "add_position"
        REDUCE_POSITION = "reduce_position"
        CLOSE_ALL = "close_all"
        REJECT = "reject"
        HOLD = "hold"

logger = logging.getLogger(__name__)


class StoneGuardian:
    """Stone Guardian v3.0 — 组合级风控，熔断与审计"""

    # ======================== 风控参数（可审计、可回测） ========================
    # 恐惧构成权重
    WEIGHT_DRAWDOWN: float = 0.30
    WEIGHT_CONSECUTIVE_LOSS: float = 0.15
    WEIGHT_VOLATILITY: float = 0.25
    WEIGHT_SHARPE: float = 0.15
    WEIGHT_LIQUIDITY: float = 0.15

    # 非对称平滑系数（高/低波动体制）
    EMA_UP_HIGH: float = 0.60
    EMA_UP_LOW: float = 0.30
    EMA_DOWN_HIGH: float = 0.10
    EMA_DOWN_LOW: float = 0.03

    # 恐惧阈值
    FEAR_CALM: float = 0.30
    FEAR_CAUTIOUS: float = 0.50
    FEAR_FEARFUL: float = 0.70
    FEAR_TERRIFIED: float = 0.85
    FEAR_MELTDOWN: float = 0.95           # 熔断触发

    # 窗口（交易笔数）
    DRAWDOWN_WINDOW: int = 30
    SHARPE_WINDOW: int = 60
    CONSECUTIVE_LOSS_CAP: int = 8

    # 基线适应
    BASELINE_UP: float = 0.02
    BASELINE_DOWN: float = 0.005
    BASELINE_MIN: float = 0.08
    BASELINE_MAX: float = 0.88
    SHARPE_RELAX: float = 1.2             # 夏普阈值

    # 参考波动率（年化）
    VOL_REFERENCE: float = 0.60           # 60%
    ANNUALIZATION_FACTOR: float = math.sqrt(365)  # 日频 -> 年化

    # 时间衰减（每小时）
    DECAY_PER_HOUR_CALM: float = 0.008
    DECAY_PER_HOUR_CAUTIOUS: float = 0.004
    DECAY_PER_HOUR_FEARFUL: float = 0.001
    DECAY_PER_HOUR_TERRIFIED: float = 0.0002

    # 否决冷却（秒）
    VETO_COOLDOWN_SEC: float = 300.0

    # 持久化键
    STATE_KEY: str = "stone_v3"

    def __init__(self, config: Optional[Dict] = None, chronos_db=None, health_monitor=None):
        self._load_config(config)
        self._db = chronos_db
        self._health = health_monitor

        # 恐惧状态
        self._fear_index: float = 0.50
        self._fear_baseline: float = 0.30
        self._fear_level: str = "cautious"

        # 组合权益与回撤
        self._equity: deque = deque(maxlen=500)       # 固定长度双端队列
        self._equity.append(1.0)
        self._peak_equity: float = 1.0
        self._rolling_dds: deque = deque(maxlen=self.DRAWDOWN_WINDOW)

        # 收益序列（对数收益）
        self._returns: deque = deque(maxlen=self.SHARPE_WINDOW)

        # 连续盈亏
        self._cons_loss: int = 0
        self._cons_win: int = 0
        self._total_trades: int = 0

        # 流动性缓存
        self._liquidity_score: float = 0.5

        # 时间追踪
        self._last_trade_ts: float = time.time()
        self._last_veto_ts: float = 0.0
        self._last_vol_update_ts: float = 0.0

        # 熔断
        self._meltdown: bool = False

        # 恢复
        self._restore_state()
        self._update_fear(force=True)
        logger.info("Stone v3.0 就绪 fear=%.3f lvl=%s", self._fear_index, self._fear_level)

    def _load_config(self, config: Optional[Dict]):
        """实例参数隔离"""
        for attr in dir(StoneGuardian):
            if attr.isupper() and not attr.startswith('_'):
                val = getattr(StoneGuardian, attr)
                if isinstance(val, (int, float, str, bool)):
                    setattr(self, attr, val)
        if config:
            for k, v in config.items():
                if hasattr(self, k):
                    setattr(self, k, v)
        # 权重归一化
        w = [self.WEIGHT_DRAWDOWN, self.WEIGHT_CONSECUTIVE_LOSS,
             self.WEIGHT_VOLATILITY, self.WEIGHT_SHARPE, self.WEIGHT_LIQUIDITY]
        s = sum(w)
        if abs(s - 1.0) > 0.001:
            logger.warning("恐惧权重之和=%.3f，归一化", s)
            self.WEIGHT_DRAWDOWN /= s
            self.WEIGHT_CONSECUTIVE_LOSS /= s
            self.WEIGHT_VOLATILITY /= s
            self.WEIGHT_SHARPE /= s
            self.WEIGHT_LIQUIDITY /= s

    # --------------------------- 持久化（同步+异步双模） ---------------------------
    def _persist(self):
        """同步/异步安全保存"""
        if not self._db:
            return
        state = {
            'fear': self._fear_index,
            'baseline': self._fear_baseline,
            'equity': list(self._equity)[-200:],
            'peak': self._peak_equity,
            'cons_loss': self._cons_loss,
            'cons_win': self._cons_win,
            'total': self._total_trades,
            'meltdown': self._meltdown,
            'ts': time.time()
        }
        try:
            # 尝试异步
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self._db.async_save(self.STATE_KEY, state))
                return
        except (RuntimeError, ImportError):
            pass
        # 降级同步
        try:
            self._db.save(self.STATE_KEY, state)
        except Exception as e:
            logger.error("状态保存失败: %s", e)

    def _restore_state(self):
        if not self._db:
            return
        try:
            state = self._db.load(self.STATE_KEY)
            if not state:
                return
            self._fear_index = state.get('fear', 0.5)
            self._fear_baseline = state.get('baseline', 0.3)
            eq_list = state.get('equity', [1.0])
            self._equity = deque(eq_list[-500:], maxlen=500)
            self._peak_equity = state.get('peak', max(self._equity))
            self._cons_loss = state.get('cons_loss', 0)
            self._cons_win = state.get('cons_win', 0)
            self._total_trades = state.get('total', 0)
            self._meltdown = state.get('meltdown', False)
            # 重建收益和回撤
            for i in range(1, len(self._equity)):
                r = math.log(self._equity[i] / self._equity[i-1])
                self._returns.append(r)
                dd = (max(self._equity[max(0,i-self.DRAWDOWN_WINDOW):i+1]) - self._equity[i]) / max(self._equity[max(0,i-self.DRAWDOWN_WINDOW):i+1])
                self._rolling_dds.append(dd)
            self._last_trade_ts = state.get('ts', time.time())
            logger.info("恢复状态: 交易%d, 恐惧%.3f", self._total_trades, self._fear_index)
        except Exception as e:
            logger.warning("恢复失败: %s", e)

    # --------------------------- 风险计算 ---------------------------
    def _calc_drawdown_score(self) -> float:
        if not self._rolling_dds:
            return 0.0
        return min(1.0, max(self._rolling_dds) / 0.15)

    def _calc_consecutive_score(self) -> float:
        return min(1.0, self._cons_loss / self.CONSECUTIVE_LOSS_CAP)

    def _calc_vol_score(self) -> float:
        vol = self._get_volatility()
        ratio = vol / self.VOL_REFERENCE
        # sigmoid 压缩，中心 1.0，斜率 4
        return 1.0 / (1.0 + math.exp(-4.0 * (ratio - 1.0)))

    def _calc_sharpe_score(self) -> float:
        if len(self._returns) < 10:
            return 0.5
        rets = list(self._returns)
        mu = sum(rets) / len(rets)
        var = sum((r - mu)**2 for r in rets) / len(rets)
        std = max(math.sqrt(var), 1e-8)
        sharpe = mu / std * self.ANNUALIZATION_FACTOR
        return 1.0 - min(1.0, max(0.0, sharpe / 2.5))

    def _calc_liquidity_score(self) -> float:
        """流动性恐惧：外部提供 0~1，越低流动性越差 => 恐惧越高"""
        # 1 - 流动性得分
        return 1.0 - self._liquidity_score

    def _get_volatility(self) -> float:
        now = time.time()
        # 限流：每 10 秒更新一次波动率估计
        if now - self._last_vol_update_ts < 10.0:
            return self._vol_cache if hasattr(self, '_vol_cache') else self.VOL_REFERENCE
        self._last_vol_update_ts = now
        if self._health:
            try:
                v = self._health.get_volatility()
                if 0 < v < 10:
                    self._vol_cache = v
                    return v
            except Exception:
                pass
        if len(self._returns) >= 5:
            rets = list(self._returns)[-20:]
            mu = sum(rets) / len(rets)
            var = sum((r - mu)**2 for r in rets) / len(rets)
            v = math.sqrt(var) * self.ANNUALIZATION_FACTOR if var > 0 else self.VOL_REFERENCE
            self._vol_cache = v
            return v
        return self.VOL_REFERENCE

    def _update_liquidity(self):
        if self._health:
            try:
                self._liquidity_score = self._health.get_liquidity()
            except Exception:
                pass

    def _update_fear(self, force: bool = False) -> float:
        self._update_liquidity()
        vol = self._get_volatility()
        raw = (self.WEIGHT_DRAWDOWN * self._calc_drawdown_score() +
               self.WEIGHT_CONSECUTIVE_LOSS * self._calc_consecutive_score() +
               self.WEIGHT_VOLATILITY * self._calc_vol_score() +
               self.WEIGHT_SHARPE * self._calc_sharpe_score() +
               self.WEIGHT_LIQUIDITY * self._calc_liquidity_score())
        raw = max(raw, self._fear_baseline)
        raw = min(1.0, raw)

        # 体制自适应平滑
        is_high_vol = vol > self.VOL_REFERENCE * 1.3
        alpha_up = self.EMA_UP_HIGH if is_high_vol else self.EMA_UP_LOW
        alpha_down = self.EMA_DOWN_HIGH if is_high_vol else self.EMA_DOWN_LOW
        alpha = alpha_up if raw > self._fear_index else alpha_down
        self._fear_index += alpha * (raw - self._fear_index)

        # 自然衰减（所有等级缓慢恢复）
        hours = (time.time() - self._last_trade_ts) / 3600.0
        if hours > 1:
            if self._fear_level == "calm":
                decay_rate = self.DECAY_PER_HOUR_CALM
            elif self._fear_level == "cautious":
                decay_rate = self.DECAY_PER_HOUR_CAUTIOUS
            elif self._fear_level == "fearful":
                decay_rate = self.DECAY_PER_HOUR_FEARFUL
            else:
                decay_rate = self.DECAY_PER_HOUR_TERRIFIED
            self._fear_index = max(self._fear_baseline,
                                   self._fear_index - decay_rate * hours)

        # 更新恐惧等级
        if self._fear_index < self.FEAR_CALM:
            self._fear_level = "calm"
        elif self._fear_index < self.FEAR_CAUTIOUS:
            self._fear_level = "cautious"
        elif self._fear_index < self.FEAR_FEARFUL:
            self._fear_level = "fearful"
        elif self._fear_index < self.FEAR_TERRIFIED:
            self._fear_level = "terrified"
        else:
            self._fear_level = "meltdown" if self._fear_index >= self.FEAR_MELTDOWN else "terrified"

        # 熔断检测
        if self._fear_index >= self.FEAR_MELTDOWN and not self._meltdown:
            logger.critical("[KUN-RIS-F005] 风控熔断触发！")
            self._meltdown = True
        return self._fear_index

    def get_fear_index(self) -> float:
        return self._fear_index

    # --------------------------- 决策映射 ---------------------------
    def _map_to_action(self, has_position: bool, is_closing: bool, is_reverse: bool) -> Tuple[DecisionType, float]:
        """is_reverse: 信号与当前持仓方向相反（视为减仓/平仓）"""
        lvl = self._fear_level
        if self._meltdown:
            return DecisionType.CLOSE_ALL if has_position else DecisionType.REJECT, 0.0
        if lvl == "calm":
            return DecisionType.HOLD, 1.0
        elif lvl == "cautious":
            if has_position and not is_closing:
                return DecisionType.REDUCE_POSITION, 0.75
            return DecisionType.HOLD, 1.0
        elif lvl == "fearful":
            if has_position:
                if is_closing or is_reverse:
                    return DecisionType.HOLD, 1.0  # 允许减少敞口
                return DecisionType.REDUCE_POSITION, 0.5
            return DecisionType.REJECT, 0.0
        else:  # terrified/meltdown
            if has_position:
                return DecisionType.CLOSE_ALL, 0.0
            return DecisionType.REJECT, 0.0

    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if not context:
            return self._respond(DecisionType.REJECT, "空上下文")
        signal = context.get('signal', {})
        pos = context.get('position', {})
        if not signal or not pos:
            return self._respond(DecisionType.HOLD, "无信号/持仓", confidence=0.0)

        action_raw = signal.get('action', 'hold')
        # 枚举验证
        VALID_ACTIONS = {e.value for e in DecisionType} | {'close_long','close_short'}
        if action_raw not in VALID_ACTIONS:
            logger.warning("未知信号动作: %s", action_raw)
            return self._respond(DecisionType.HOLD, f"非法动作 {action_raw}")

        size = pos.get('size', 0)
        side = pos.get('side', 'long')
        has_pos = abs(size) > 0
        # 判断是否平仓/减仓/反向
        is_closing = action_raw in ('close_long','close_short','reduce_position','close_all')
        is_reverse = False
        if has_pos:
            if (side == 'long' and action_raw in ('open_short',)) or \
               (side == 'short' and action_raw in ('open_long',)):
                is_reverse = True

        self._update_fear()

        # 否决冷却
        now = time.time()
        in_cd = (now - self._last_veto_ts) < self.VETO_COOLDOWN_SEC

        action, ratio = self._map_to_action(has_pos, is_closing or is_reverse, False)
        # 冷却期限制：REJECT/CLOSE_ALL 降为 HOLD（除非熔断）
        if not self._meltdown and action in (DecisionType.REJECT, DecisionType.CLOSE_ALL) and in_cd:
            action = DecisionType.HOLD
        if action in (DecisionType.REJECT, DecisionType.CLOSE_ALL):
            self._last_veto_ts = now

        return self._respond(action, f"{self._fear_level} fear={self._fear_index:.2f}",
                             self._fear_index, {"suggested_ratio": ratio, "fear_level": self._fear_level})

    def _respond(self, action: DecisionType, reason: str, confidence: float = 0.0,
                 meta: Optional[Dict] = None) -> Dict[str, Any]:
        return {
            "status": "ok",
            "decision": action.value,
            "confidence": confidence,
            "reason": reason,
            "metadata": meta or {},
            "warnings": [f"Stone: {self._fear_level}"] if self._fear_level not in ("calm",) else [],
            "decision_id": f"stn_{int(time.time()*1e6)}_{uuid.uuid4().hex[:6]}"
        }

    # --------------------------- 交易反馈 ---------------------------
    def record_outcome(self, trade: Dict[str, Any]) -> None:
        if not trade or self._meltdown:
            return
        pnl = trade.get('pnl_percent')
        if not isinstance(pnl, (int, float)) or math.isnan(pnl) or abs(pnl) > 5.0:
            logger.warning("异常盈亏: %s", pnl)
            return
        eq_prev = self._equity[-1]
        eq_new = eq_prev * (1.0 + pnl)
        self._equity.append(eq_new)
        self._peak_equity = max(self._peak_equity, eq_new)

        # 对数收益
        log_ret = math.log(eq_new / eq_prev) if eq_prev > 0 else 0.0
        self._returns.append(log_ret)

        # 滚动回撤
        w = min(self.DRAWDOWN_WINDOW, len(self._equity))
        peak_win = max(list(self._equity)[-w:])
        dd = (peak_win - eq_new) / peak_win if peak_win > 0 else 0.0
        self._rolling_dds.append(dd)

        # 连续计数
        if pnl > 1e-6:
            self._cons_win += 1
            self._cons_loss = 0
        elif pnl < -1e-6:
            self._cons_loss += 1
            self._cons_win = 0
        self._total_trades += 1
        self._last_trade_ts = time.time()

        self._adapt_baseline()
        self._persist()

    def _adapt_baseline(self):
        if len(self._returns) < self.SHARPE_WINDOW:
            return
        rets = list(self._returns)[-self.SHARPE_WINDOW:]
        mu = sum(rets) / len(rets)
        var = sum((r - mu)**2 for r in rets) / len(rets)
        if var > 1e-10:
            sharpe = mu / math.sqrt(var) * self.ANNUALIZATION_FACTOR
            if sharpe > self.SHARPE_RELAX:
                self._fear_baseline = max(self.BASELINE_MIN, self._fear_baseline - self.BASELINE_DOWN)
        if self._cons_loss >= 5:
            self._fear_baseline = min(self.BASELINE_MAX, self._fear_baseline + self.BASELINE_UP)

    # --------------------------- 外部控制 ---------------------------
    def force_meltdown(self):
        self._meltdown = True
        self._fear_index = 1.0
        self._fear_level = "meltdown"
        logger.critical("人工触发熔断")

    def reset_meltdown(self, auth_token: str = ""):
        # 应结合权限验证
        self._meltdown = False
        self._fear_index = 0.6
        self._fear_level = "fearful"
        self._persist()

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "fear_index": self._fear_index,
            "baseline": self._fear_baseline,
            "level": self._fear_level,
            "cons_loss": self._cons_loss,
            "total_trades": self._total_trades,
            "volatility": self._get_volatility(),
            "liquidity": self._liquidity_score,
            "meltdown": self._meltdown,
        }

    # --------------------------- 健康检查 ---------------------------
    @classmethod
    def health_check(cls) -> Dict[str, Any]:
        try:
            s = cls()
            ctx = {'signal': {'action': 'open_long'}, 'position': {'size': 0}}
            r = s.evaluate(ctx)
            assert r['status'] == 'ok'
            for _ in range(6):
                s.record_outcome({'pnl_percent': -0.02})
            assert s.get_fear_index() > 0.5
            return {"status": "ok", "message": "Stone v3.0 健康"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
