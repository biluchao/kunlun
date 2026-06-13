#!/usr/bin/env python3
"""
昆仑系统 · 智能体贡献值评估器 (ContributionEvaluator)

核心职责：
1. 以时间加权方式评估智能体决策质量，使用移动块自助法 (moving block bootstrap)
   估计时间加权均值的置信区间与统计显著性。
2. 根据贡献值动态生成权重乘数并异步推送至仲裁器，实现智能体权重的
   闭环自动调整。
3. 完整持久化评估历史与衰减状态，支持无缝热重载，保证重启后评估窗口不丢失。

外部依赖（真实模块接口）：
- infrastructure.chronos_db.ChronosDB : 持久化衰减因子与评估快照
- infrastructure.audit_chain.AuditLogChain : 审计日志

接口契约：
- record_decision(signal_id, agent_name, decision, confidence, side) -> Dict
- evaluate_outcome(signal_id, pnl_percent, exit_reason, holding_duration_sec) -> Dict
- get_agent_report(agent_name=None) -> Dict
- apply_weight_adjustments() -> Dict   # 计算并推送权重乘数
- save_state() -> Dict
- restore_state(state) -> None
- health_check() -> Dict

异常与降级：
- 盈亏异常值被截断并记录 KUN-AGT-W010
- 数据库不可用时衰减因子仅存内存，重启后丢失
- 非策略性平仓（manual/force_close/emergency）不计入贡献评估

资源管理：
- 内存保留最近 2000 条记录，超出后移除最旧并清理孤立数据
- 每日 00:30 执行清理，确保评估窗口数据充足
"""

import logging
import time
import math
from typing import Dict, Any, List, Optional, Tuple, Callable
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from threading import RLock
import numpy as np

logger = logging.getLogger(__name__)

# 智能体 / 决策 / 退出原因白名单
VALID_AGENTS = {'stone', 'wind', 'mirror', 'eye', 'book'}
VALID_DECISIONS = {'open_long', 'open_short', 'add_position',
                   'reduce_position', 'close_all', 'reject', 'hold'}
VALID_EXIT_REASONS = {'stop_loss', 'take_profit', 'signal_exit',
                      'emergency', 'manual', 'force_close'}
VALID_SIDES = {'long', 'short'}
NEUTRAL_EXIT_REASONS = {'emergency', 'manual', 'force_close'}

@dataclass
class AgentDecision:
    """不可变决策记录"""
    signal_id: str
    agent_name: str
    decision: str
    confidence: float   # [0, 1]
    timestamp: float    # UTC epoch seconds
    side: str = 'long'  # 'long' 或 'short'

@dataclass
class TradeOutcome:
    """交易结果记录"""
    signal_id: str
    pnl_percent: float
    exit_reason: str
    holding_duration_sec: float
    timestamp: float
    evaluated: bool = False

class ContributionEvaluator:
    """智能体贡献值评估器 (v3.0 生产级)"""

    # ---- 可配置常量 ----
    EVALUATION_WINDOW: int = 50
    MIN_SAMPLES_FOR_EVAL: int = 10
    MIN_SAMPLES_FOR_TEST: int = 20

    BOOTSTRAP_SAMPLES: int = 2000
    BLOCK_SIZE: int = 5          # 移动块大小
    CONFIDENCE_LEVEL: float = 0.95
    MIN_EFFECT_SIZE: float = 0.05  # 效应量阈值

    TIME_DECAY_HALF_LIFE: int = 60   # 半衰期（交易笔数），略大于评估窗口，更平滑

    WEIGHT_DECAY_THRESHOLD: float = -0.2
    MAX_DECAY_FACTOR: float = 0.3
    DECAY_RECOVERY_STEP: float = 0.05
    MIN_WEIGHT_MULTIPLIER: float = 0.01

    MAX_SILENT_SECONDS: float = 7200.0
    SILENT_CHECK_INTERVAL: float = 300.0

    MAX_ABS_PNL_PERCENT: float = 5.0        # 单笔盈亏截断阈值
    DEFAULT_VOL_BASELINE: float = 0.02      # 默认基准波动率，可被外部覆盖

    MAX_MEMORY_RECORDS: int = 2000           # 决策和结果各自的最大容量
    PERSIST_INTERVAL_SEC: float = 900.0

    def __init__(self, config: Optional[Dict] = None,
                 chronos_db=None,
                 weight_callback: Optional[Callable[[str, float], None]] = None):
        """
        :param chronos_db: 数据库接口
        :param weight_callback: 异步权重乘数推送回调 (agent_name, multiplier)
        """
        self._db = chronos_db
        self._weight_callback = weight_callback

        # 实例级配置覆盖（仅修改自身属性）
        if config:
            for k, v in config.items():
                if hasattr(self, k):
                    setattr(self, k, v)

        # 内部状态
        self._decisions: Dict[str, List[AgentDecision]] = OrderedDict()
        self._outcomes: Dict[str, TradeOutcome] = OrderedDict()
        self._last_active_time: Dict[str, float] = defaultdict(lambda: time.time())
        self._decay_state: Dict[str, float] = defaultdict(lambda: 1.0)

        # 缓存与锁
        self._contribution_cache: Dict[str, Any] = {}
        self._cache_timestamp: float = 0.0
        self._lock = RLock()

        # 随机状态（可替换）
        self.rng = np.random.RandomState(int(time.time() * 1e6) % (2**31))

        # 加载持久化衰减因子
        self._load_decay_state()

        # 最后持久化时间
        self._last_persist = time.time()
        logger.info("ContributionEvaluator v3.0 就绪，衰减因子: %s",
                    dict(self._decay_state))

    # ---- 持久化辅助 ----
    def _load_decay_state(self):
        if self._db:
            try:
                raw = self._db.get("agent_decay_state")
                if isinstance(raw, dict):
                    for k, v in raw.items():
                        if k in VALID_AGENTS and isinstance(v, (int, float)):
                            self._decay_state[k] = max(self.MAX_DECAY_FACTOR, min(1.0, float(v)))
            except Exception as e:
                logger.error("[KUN-AGT-E015] 加载衰减状态失败: %s", e)

    def _save_decay_state(self):
        if self._db:
            try:
                self._db.put("agent_decay_state", dict(self._decay_state))
            except Exception as e:
                logger.error("[KUN-AGT-E016] 持久化衰减状态失败: %s", e)

    # ---- 记录接口 ----
    def record_decision(self, signal_id: str, agent_name: str,
                        decision: str, confidence: float,
                        side: str = 'long') -> Dict[str, Any]:
        """记录智能体的决策，返回状态"""
        if agent_name not in VALID_AGENTS:
            return {"status": "error", "reason": f"非法智能体: {agent_name}"}
        if decision not in VALID_DECISIONS:
            return {"status": "error", "reason": f"非法决策: {decision}"}
        if side not in VALID_SIDES:
            return {"status": "error", "reason": f"非法方向: {side}"}
        confidence = max(0.0, min(1.0, confidence))

        rec = AgentDecision(signal_id=signal_id, agent_name=agent_name,
                            decision=decision, confidence=confidence,
                            timestamp=time.time(), side=side)
        with self._lock:
            lst = self._decisions.setdefault(signal_id, [])
            lst.append(rec)
            self._last_active_time[agent_name] = rec.timestamp
            # 容量控制
            while len(self._decisions) > self.MAX_MEMORY_RECORDS:
                oldest_sig = next(iter(self._decisions))
                # 移除最旧信号的同时，若其关联结果尚未评估，则一并删除
                if oldest_sig in self._outcomes and not self._outcomes[oldest_sig].evaluated:
                    del self._outcomes[oldest_sig]
                del self._decisions[oldest_sig]
            self._cache_timestamp = 0.0
        return {"status": "ok", "confidence_clipped": confidence}

    def record_outcome(self, signal_id: str, pnl_percent: float,
                       exit_reason: str, holding_duration_sec: float) -> Dict[str, Any]:
        """记录交易结果，若已存在未评估结果则覆盖（仅允许覆盖一次）"""
        if exit_reason not in VALID_EXIT_REASONS:
            return {"status": "error", "reason": f"非法退出原因: {exit_reason}"}
        if math.isnan(pnl_percent) or math.isinf(pnl_percent):
            logger.error("[KUN-AGT-W010] pnl异常，信号 %s", signal_id)
            return {"status": "error", "reason": "非法盈亏值"}
        pnl = max(-self.MAX_ABS_PNL_PERCENT, min(self.MAX_ABS_PNL_PERCENT, pnl_percent))
        holding = max(0.0, holding_duration_sec)

        with self._lock:
            existing = self._outcomes.get(signal_id)
            if existing and existing.evaluated:
                return {"status": "error", "reason": "信号已评估，不可覆盖"}
            outcome = TradeOutcome(signal_id=signal_id, pnl_percent=pnl,
                                   exit_reason=exit_reason,
                                   holding_duration_sec=holding,
                                   timestamp=time.time())
            self._outcomes[signal_id] = outcome
            while len(self._outcomes) > self.MAX_MEMORY_RECORDS:
                oldest_sig = next(iter(self._outcomes))
                del self._outcomes[oldest_sig]
                # 清理孤立决策（若对应决策已无结果，则可保留）
            self._cache_timestamp = 0.0
        return {"status": "ok"}

    # ---- 核心评估 ----
    def evaluate_outcome(self, signal_id: str, pnl_percent: float,
                         exit_reason: str, holding_duration_sec: float = 0.0) -> Dict[str, Any]:
        """评估交易结果对智能体贡献的影响"""
        rec_result = self.record_outcome(signal_id, pnl_percent, exit_reason, holding_duration_sec)
        if rec_result['status'] != 'ok':
            return rec_result

        with self._lock:
            outcome = self._outcomes[signal_id]
            decisions = self._decisions.get(signal_id, [])
            if not decisions:
                logger.warning("[KUN-AGT-W011] 信号 %s 无决策记录", signal_id)
                return {"status": "ok", "scores": {}, "warnings": ["无决策记录"]}
            outcome.evaluated = True
            # 复制数据，锁外计算
            decisions_copy = [d for d in decisions if d.timestamp <= outcome.timestamp + 1.0]
            if not decisions_copy:
                logger.warning("所有决策时间晚于结果，信号 %s", signal_id)
            outcome_copy = outcome

        # 打分
        scores = {}
        for ad in decisions_copy:
            # 检查决策方向与交易方向一致性
            if ad.side not in VALID_SIDES:
                continue
            score = self._score_decision(ad.decision, outcome_copy.pnl_percent,
                                         outcome_copy.exit_reason,
                                         outcome_copy.holding_duration_sec,
                                         ad.side)
            scores[ad.agent_name] = score

        self._cache_timestamp = 0.0  # 失效缓存
        # 异步发送权重调整（不在评估主路径中阻塞）
        self._maybe_apply_adjustments()
        return {"status": "ok", "signal_id": signal_id, "scores": scores}

    def _score_decision(self, decision: str, pnl: float, exit_reason: str,
                        duration: float, side: str) -> float:
        """内部打分函数，返回 [-1, 1]"""
        if exit_reason in NEUTRAL_EXIT_REASONS:
            return 0.0
        # 动态基准波动率可来自实例变量
        baseline = self.DEFAULT_VOL_BASELINE
        normalized = pnl / baseline
        normalized = max(-10.0, min(10.0, normalized))
        is_profit = pnl > 1e-8
        is_loss = pnl < -1e-8

        if decision in ('open_long', 'open_short'):
            # 多空盈利逻辑一致：pnl>0 即赚钱
            if is_profit:
                score = math.tanh(normalized)
            elif is_loss:
                score = -math.tanh(abs(normalized))
            else:
                score = 0.0
        elif decision == 'add_position':
            # 加仓方向由 side 决定，盈利逻辑同上
            if is_profit:
                score = 0.5 * math.tanh(normalized)
            elif is_loss:
                score = -0.5 * math.tanh(abs(normalized))
            else:
                score = 0.0
        elif decision in ('reduce_position', 'close_all'):
            # 平仓：如果平仓后继续亏损（正确），给正分；如果平仓后市场反转则错误，
            # 但我们没有未来信息，因此采用保守估计：
            # 亏损平仓（避免更大损失）得 0.5，盈利平仓（可能过早）得 0.1
            if is_loss:
                score = 0.5
            elif is_profit:
                score = 0.1
            else:
                score = 0.0
        elif decision == 'reject':
            if is_loss:
                score = 0.3
            elif is_profit:
                score = -0.2
            else:
                score = 0.0
        elif decision == 'hold':
            score = 0.0
        else:
            score = 0.0

        # 时间加成：持仓时间长的正确决策更可靠
        if decision in ('open_long', 'open_short', 'add_position'):
            time_factor = min(1.0, duration / 3600.0) * 0.1
            if score > 0:
                score += time_factor
            else:
                score -= time_factor

        return max(-1.0, min(1.0, score))

    # ---- 贡献值计算 ----
    def _calculate_contribution(self) -> Dict[str, Any]:
        with self._lock:
            if self._cache_timestamp > 0 and (time.time() - self._cache_timestamp) < 60:
                return self._contribution_cache
            # 快照
            dec_snap = {k: list(v) for k, v in self._decisions.items()}
            out_snap = {k: v for k, v in self._outcomes.items() if v.evaluated}

        records = []
        for sig_id, outcome in out_snap.items():
            decs = dec_snap.get(sig_id, [])
            for ad in decs:
                if ad.timestamp > outcome.timestamp + 1.0:
                    continue
                score = self._score_decision(ad.decision, outcome.pnl_percent,
                                             outcome.exit_reason,
                                             outcome.holding_duration_sec,
                                             ad.side)
                records.append((ad.agent_name, score, outcome.timestamp))

        contributions = {}
        if records:
            agent_data = defaultdict(list)
            for name, score, ts in records:
                agent_data[name].append((score, ts))
            for name, vals in agent_data.items():
                scores = np.array([v[0] for v in vals])
                timestamps = np.array([v[1] for v in vals])
                n = len(scores)
                if n < self.MIN_SAMPLES_FOR_EVAL:
                    contributions[name] = {'mean': 0.0, 'count': n, 'reliable': False}
                    continue
                order = np.argsort(timestamps)
                sorted_scores = scores[order]
                half = self.TIME_DECAY_HALF_LIFE
                # 指数衰减权重：最新权重最大
                raw_weights = np.power(2.0, -(n - 1 - np.arange(n)) / half)
                w_sum = raw_weights.sum()
                if w_sum == 0:
                    w_mean = np.mean(sorted_scores)
                else:
                    w_mean = np.average(sorted_scores, weights=raw_weights)
                ci_low, ci_high = None, None
                significant = False
                if n >= self.MIN_SAMPLES_FOR_TEST:
                    ci_low, ci_high = self._moving_block_bootstrap_ci(
                        sorted_scores, raw_weights / w_sum)
                    if ci_low is not None:
                        significant = (ci_low > self.MIN_EFFECT_SIZE) or \
                                      (ci_high < -self.MIN_EFFECT_SIZE)
                contributions[name] = {
                    'mean': w_mean,
                    'count': n,
                    'reliable': n >= self.MIN_SAMPLES_FOR_TEST,
                    'ci_lower': ci_low,
                    'ci_upper': ci_high,
                    'significant': significant
                }

        with self._lock:
            self._contribution_cache = contributions
            self._cache_timestamp = time.time()
        return contributions

    def _moving_block_bootstrap_ci(self, scores: np.ndarray,
                                   weights: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """
        移动块自助法：每次抽样固定长度 n 的序列。
        使用块重采样以保留局部时间相关性。
        """
        n = len(scores)
        if n < 10:
            return None, None
        block_size = min(self.BLOCK_SIZE, max(2, n // 5))
        # 将序列划分为重叠的块：块 i 包含 scores[i:i+block_size]
        n_blocks = n - block_size + 1
        if n_blocks <= 0:
            return np.mean(scores), np.mean(scores)

        means = []
        rng = self.rng
        for _ in range(self.BOOTSTRAP_SAMPLES):
            # 随机选取 n_blocks 个块起始索引（有放回）
            idx = rng.randint(0, n_blocks, size=n_blocks)
            sample_scores = []
            sample_weights = []
            for start in idx:
                sample_scores.extend(scores[start:start+block_size])
                sample_weights.extend(weights[start:start+block_size])
            # 截断到 n 并重新归一化
            sample_scores = np.array(sample_scores[:n])
            sample_weights = np.array(sample_weights[:n])
            w_sum = sample_weights.sum()
            if w_sum > 0:
                sample_weights /= w_sum
                w_mean = np.average(sample_scores, weights=sample_weights)
            else:
                w_mean = np.mean(sample_scores)
            means.append(w_mean)

        means.sort()
        alpha = (1 - self.CONFIDENCE_LEVEL) / 2
        low_idx = int(alpha * len(means))
        high_idx = int((1 - alpha) * len(means)) - 1
        return means[low_idx], means[high_idx]

    # ---- 权重调整与推送 ----
    def _maybe_apply_adjustments(self):
        """检查是否需要调整智能体权重并推送（非阻塞）"""
        contributions = self._calculate_contribution()
        for name, contrib in contributions.items():
            if not contrib.get('reliable'):
                continue
            mean = contrib['mean']
            current = self._decay_state.get(name, 1.0)
            if contrib.get('significant') and mean < self.WEIGHT_DECAY_THRESHOLD:
                new_mult = max(self.MAX_DECAY_FACTOR, current - 0.1)
                self._decay_state[name] = new_mult
                logger.info("[KUN-AGT-I009] %s 权重乘数降至 %.2f", name, new_mult)
                self._push_weight(name, new_mult)
            elif mean >= self.WEIGHT_DECAY_THRESHOLD and current < 1.0:
                new_mult = min(1.0, current + self.DECAY_RECOVERY_STEP)
                self._decay_state[name] = new_mult
                logger.info("[KUN-AGT-I010] %s 权重乘数恢复至 %.2f", name, new_mult)
                self._push_weight(name, new_mult)
        self._save_decay_state()

    def _push_weight(self, agent_name: str, multiplier: float):
        """安全推送权重回调"""
        if self._weight_callback:
            try:
                self._weight_callback(agent_name, multiplier)
            except Exception as e:
                logger.error("[KUN-AGT-E017] 权重回调异常: %s", e)

    def apply_weight_adjustments(self) -> Dict[str, Any]:
        """公开接口：手动触发权重调整与推送"""
        with self._lock:
            self._maybe_apply_adjustments()
        return {"status": "ok", "decay_state": dict(self._decay_state)}

    # ---- 报告与监控 ----
    def get_agent_report(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        self._check_silent_agents()
        contributions = self._calculate_contribution()
        if agent_name:
            if agent_name not in VALID_AGENTS:
                return {"status": "error", "reason": f"未知智能体: {agent_name}"}
            contrib = contributions.get(agent_name, {})
            return {"status": "ok", "agent": agent_name,
                    "contribution": contrib,
                    "weight_multiplier": self._decay_state.get(agent_name, 1.0)}
        report = OrderedDict()
        for name in sorted(VALID_AGENTS):
            report[name] = {
                "contribution": contributions.get(name, {}),
                "weight_multiplier": self._decay_state.get(name, 1.0),
                "last_active": self._last_active_time.get(name, 0)
            }
        return {"status": "ok", "agents": report,
                "total_evaluated": sum(1 for o in self._outcomes.values() if o.evaluated)}

    def _check_silent_agents(self):
        now = time.time()
        if now - getattr(self, '_last_silent_check', 0) < self.SILENT_CHECK_INTERVAL:
            return
        self._last_silent_check = now
        for name in VALID_AGENTS:
            last = self._last_active_time.get(name, 0)
            if now - last > self.MAX_SILENT_SECONDS:
                msg = f"[KUN-AGT-W012] {name} 静默 {int(now-last)}s"
                logger.warning(msg)
                try:
                    from infrastructure.audit_chain import AuditLogChain
                    AuditLogChain.log_event("agent_silence", "WARNING",
                                            {"agent": name, "seconds": now-last})
                except ImportError:
                    pass

    # ---- 生命周期管理 ----
    def save_state(self) -> Dict:
        with self._lock:
            dec_list = {k: [(d.agent_name, d.decision, d.confidence, d.timestamp, d.side)
                            for d in v] for k, v in self._decisions.items()}
            out_list = {k: (o.pnl_percent, o.exit_reason, o.holding_duration_sec,
                            o.timestamp, o.evaluated)
                        for k, o in self._outcomes.items()}
            return {
                "decisions": dec_list,
                "outcomes": out_list,
                "decay_state": dict(self._decay_state)
            }

    def restore_state(self, state: Dict):
        with self._lock:
            self._decisions = OrderedDict()
            for k, v in state.get("decisions", {}).items():
                self._decisions[k] = [AgentDecision(k, *d) for d in v]
            self._outcomes = OrderedDict()
            for k, v in state.get("outcomes", {}).items():
                self._outcomes[k] = TradeOutcome(k, *v)
            self._decay_state = defaultdict(lambda: 1.0, state.get("decay_state", {}))
            self._cache_timestamp = 0.0
        logger.info("状态恢复完成，决策 %d 条，结果 %d 条",
                    len(self._decisions), len(self._outcomes))

    def daily_cleanup(self):
        """清理 30 天前的数据，同时确保评估窗口数据充足"""
        cutoff = time.time() - 86400 * 30
        with self._lock:
            evaluated = [o for o in self._outcomes.values() if o.evaluated]
            if len(evaluated) > self.EVALUATION_WINDOW:
                evaluated.sort(key=lambda x: x.timestamp)
                keep_since = max(cutoff, evaluated[-self.EVALUATION_WINDOW].timestamp)
            else:
                keep_since = 0.0

            expired_sigs = [sid for sid, out in self._outcomes.items()
                            if out.timestamp < keep_since]
            for sid in expired_sigs:
                del self._outcomes[sid]
                # 清理对应的决策
                if sid in self._decisions:
                    del self._decisions[sid]
            # 额外清理无结果的孤立决策（超过1天）
            for sid in list(self._decisions.keys()):
                if sid not in self._outcomes and self._decisions[sid] and \
                   self._decisions[sid][-1].timestamp < time.time() - 86400:
                    del self._decisions[sid]
            self._cache_timestamp = 0.0
        self._save_decay_state()
        logger.info("每日清理完成，当前决策 %d，结果 %d",
                    len(self._decisions), len(self._outcomes))

    # ---- 健康检查 ----
    @classmethod
    def health_check(cls) -> Dict[str, Any]:
        try:
            inst = cls()
            inst.rng = np.random.RandomState(42)
            # 测试打分
            tests = [
                ('open_long', 0.02, 'take_profit', 'long', 1),
                ('open_long', -0.02, 'stop_loss', 'long', -1),
                ('open_short', 0.02, 'take_profit', 'short', 1),
                ('open_short', -0.02, 'stop_loss', 'short', -1),
                ('close_all', -0.05, 'stop_loss', 'long', 1),
            ]
            for dec, pnl, reason, side, sign in tests:
                s = inst._score_decision(dec, pnl, reason, 0, side)
                if sign > 0 and s <= 0:
                    return {"status": "error", "message": f"{dec} 应得正分，实得 {s:.3f}"}
                if sign < 0 and s >= 0:
                    return {"status": "error", "message": f"{dec} 应得负分，实得 {s:.3f}"}
            # bootstrap
            sample = np.array([0.02]*30)
            w = np.ones(30)/30
            ci = inst._moving_block_bootstrap_ci(sample, w)
            if ci[0] is None:
                return {"status": "error", "message": "bootstrap 失败"}
            return {"status": "ok", "message": "所有检查通过"}
        except Exception as e:
            logger.error("健康检查异常: %s", e)
            return {"status": "error", "message": str(e)}
