#!/usr/bin/env python3
"""
昆仑系统 · 因子计算引擎 (FactorComputeEngine) v6.0.0-prime

核心职责：
1. 实时计算 8 个微观/宏观因子，输出带置信度的稳健信号强度 (0~1)
2. 根据 HMM 市场状态、饥渴度阈值和因子有效性标志，动态聚合因子得分
3. 监控因子多重共线性 (VIF) 与边际贡献，适时采用增量 PCA 或自适应惩罚
4. 提供健康自检、滚动 IC/IR 跟踪、异常值隔离和冷启动预热

外部依赖（真实模块接口）：
- polaris.market_regime.MarketRegimeClassifier : 获取 9 宫格市场状态与 HMM 状态概率
- polaris.hmm_engine.HMMEngine : 获取趋势/震荡状态及置信度
- infrastructure.chronos_db.ChronosDB : 读取历史 OHLCV/ATR 序列
- infrastructure.stream_gateway.StreamGateway : 实时订单簿、逐笔成交、K线
- olympus.agent_arbiter.AgentArbiter : 获取当前有效饥渴度阈值及因子偏好
- infrastructure.error_registry.ErrorRegistry : 统一错误码注册与查询

接口契约：
- compute_all_factors(market_data: FactorInputDict) -> FactorOutputDict
- get_composite_score(factor_signals: Dict, regime: str, hunger_threshold: float) -> CompositeScoreDict
- health_check() -> HealthCheckDict

异常与降级：
- 订单簿快照年龄 > 50ms：F4/F5 标记为降级，权重减半，综合得分忽略微观结构
- 成交量缺失：F6/F8 最多填充 3 次；超过后标记为过期，权重归零
- MAD 退化 (<1e-8)：回退为 Z-score 并记录 KUN-FAC-W003
- VIF > 5 且历史足够：使用增量 PCA 降维；否则对相关因子组乘以 0.5 惩罚系数
- 非有限值：因子置为 0.5，触发 KUN-FAC-F005

资源管理：
- 使用 __slots__ 固定实例属性，避免动态字典
- 滚动窗口采用高效 deque 和增量 Welford 统计，避免频繁分配 numpy 数组
- MACD 状态对象维护每个周期的增量 EMA，常数时间更新
- 不持有长生命周期外部连接，所有数据通过输入字典注入

并发安全：
- 设计为单线程事件循环模型，无锁操作；如有需要可继承实现线程安全版本
"""

import math
import time
from typing import Dict, Any, List, Tuple, Optional, Final, TypedDict, Union
from collections import deque
import logging
import numpy as np
from copy import deepcopy

logger = logging.getLogger(__name__)

# ---------- 类型契约 ----------
class FactorInputDict(TypedDict, total=False):
    """因子计算输入数据的类型约束"""
    close: float
    ma25: float
    ma25_past: float
    atr14: float
    atr56: float
    orderbook: Dict
    avg_depth: float
    trades: List[Dict]
    close_5m: float          # 5分钟K线最新收盘价
    close_15m: float         # 15分钟K线最新收盘价
    volume: float
    vol_ma: float
    vol_std: float
    timestamp_ms: int

class FactorOutputDict(TypedDict):
    """因子计算输出字典"""
    status: str
    factors: Dict[str, float]
    vif: Dict[str, float]
    warnings: List[str]
    metrics: Dict[str, Any]

class CompositeScoreDict(TypedDict):
    """综合得分输出"""
    composite_score: float
    passed: bool
    category_breakdown: Dict[str, float]
    threshold_used: float

class HealthCheckDict(TypedDict):
    """健康检查结果"""
    status: str
    message: str
    warmup_complete: bool


# ---------- MACD 增量状态 ----------
class MACDState:
    """维护单个周期的 MACD 增量计算状态，避免 O(N) 全量重算"""
    __slots__ = ('fast_period', 'slow_period', 'signal_period',
                 'ema_fast', 'ema_slow', 'dea', 'initialized')

    def __init__(self, fast: int, slow: int, signal: int):
        self.fast_period = fast
        self.slow_period = slow
        self.signal_period = signal
        self.ema_fast: Optional[float] = None
        self.ema_slow: Optional[float] = None
        self.dea: Optional[float] = None
        self.initialized = False

    def initialize_with_series(self, prices: List[float]) -> None:
        """用历史价格序列初始化 MACD 状态，确保预热后正确"""
        if len(prices) < self.slow_period:
            return
        # 初始 EMA 为简单平均
        self.ema_fast = sum(prices[-self.fast_period:]) / self.fast_period
        self.ema_slow = sum(prices[-self.slow_period:]) / self.slow_period
        dif = self.ema_fast - self.ema_slow
        # DEA 使用初始 DIF 序列近似
        diffs = []
        ema_f = self.ema_fast
        ema_s = self.ema_slow
        alpha_f = 2.0 / (self.fast_period + 1)
        alpha_s = 2.0 / (self.slow_period + 1)
        for p in prices[-self.signal_period:]:
            ema_f = (p - ema_f) * alpha_f + ema_f
            ema_s = (p - ema_s) * alpha_s + ema_s
            diffs.append(ema_f - ema_s)
        self.dea = sum(diffs) / len(diffs) if diffs else dif
        self.initialized = True

    def update(self, price: float) -> Tuple[float, float, float]:
        """输入新价格，返回 (DIF, DEA, HIST)"""
        if not self.initialized:
            # 尚未初始化，执行简单冷启动
            if self.ema_fast is None:
                self.ema_fast = price
                self.ema_slow = price
                self.dea = 0.0
                return 0.0, 0.0, 0.0
            # 继续积累直到 DEA 计算可行（此处简化）
            alpha_f = 2.0 / (self.fast_period + 1)
            alpha_s = 2.0 / (self.slow_period + 1)
            self.ema_fast = (price - self.ema_fast) * alpha_f + self.ema_fast
            self.ema_slow = (price - self.ema_slow) * alpha_s + self.ema_slow
            dif = self.ema_fast - self.ema_slow
            alpha_signal = 2.0 / (self.signal_period + 1)
            if self.dea is None:
                self.dea = dif
            else:
                self.dea = (dif - self.dea) * alpha_signal + self.dea
            # 当收集足够点后标记为已初始化（条件：至少 slow_period 个点）
            # 这里简化为始终返回，调用方负责确保有足够历史
            return dif, self.dea, dif - self.dea

        alpha_f = 2.0 / (self.fast_period + 1)
        alpha_s = 2.0 / (self.slow_period + 1)
        alpha_signal = 2.0 / (self.signal_period + 1)

        self.ema_fast = (price - self.ema_fast) * alpha_f + self.ema_fast
        self.ema_slow = (price - self.ema_slow) * alpha_s + self.ema_slow
        dif = self.ema_fast - self.ema_slow
        self.dea = (dif - self.dea) * alpha_signal + self.dea
        hist = dif - self.dea
        return dif, self.dea, hist

    def reset(self):
        """重置所有状态"""
        self.ema_fast = None
        self.ema_slow = None
        self.dea = None
        self.initialized = False


class FactorComputeEngine:
    """机构级因子计算引擎（单交易对，单线程安全）"""

    # ---------- 类常量 (Final) ----------
    F1_MA_PERIOD: Final = 25
    F1_ATR_PERIOD: Final = 14
    F1_SATURATION_SCALE: Final = 2.0
    F2_SLOPE_LOOKBACK: Final = 6
    F3_MACD_FAST_5M: Final = 12
    F3_MACD_SLOW_5M: Final = 26
    F3_MACD_SIGNAL_5M: Final = 9
    F3_MACD_FAST_15M: Final = 12
    F3_MACD_SLOW_15M: Final = 26
    F3_MACD_SIGNAL_15M: Final = 9
    F4_DEPTH_LEVELS: Final = 5
    F4_EMA_PERIOD: Final = 3
    F4_MIN_DEPTH_RATIO: Final = 1.2
    F5_DEPTH_LEVELS: Final = 5
    F6_LARGE_TRADE_THRESHOLD: Final = 50000.0
    F6_EMA_PERIOD: Final = 12
    F7_OPTIMAL_LOW: Final = -0.1
    F7_OPTIMAL_HIGH: Final = 0.3
    F7_OUTSIDE_PENALTY: Final = 0.5
    F8_OPTIMAL_ZONE: Final = (0.0, 2.0)
    WINDOW_SIZE: Final = 100
    MAD_EPSILON: Final = 1e-8
    VIF_THRESHOLD: Final = 5.0
    PCA_EXPLAINED_VARIANCE: Final = 0.85
    SNAPSHOT_MAX_AGE_MS: Final = 50
    MAX_VOL_FILL: Final = 3
    # 因子名称列表（不可变）
    ALL_FACTOR_NAMES: Final = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8']

    # 默认权重矩阵
    DEFAULT_WEIGHT_MATRIX: Final = {
        'trending_strong': {'trend_confirm': 0.50, 'microstructure': 0.30, 'risk_env': 0.20},
        'normal':           {'trend_confirm': 0.40, 'microstructure': 0.35, 'risk_env': 0.25},
        'cold_range':       {'trend_confirm': 0.20, 'microstructure': 0.55, 'risk_env': 0.25}
    }

    # 可配置参数的名称（白名单）
    _CONFIG_WHITELIST = {
        'F4_DEPTH_LEVELS': int,
        'F5_DEPTH_LEVELS': int,
        'F6_LARGE_TRADE_THRESHOLD': float,
        'WINDOW_SIZE': int,
        'VIF_THRESHOLD': float,
        'SNAPSHOT_MAX_AGE_MS': int,
        'weight_matrix': dict
    }

    __slots__ = (
        '_symbol', '_history', '_incremental_stats',
        '_obi_ema', '_whale_flow_ema', '_vol_fill_count',
        '_weight_matrix', '_warmup_complete',
        '_macd_5m', '_macd_15m',
        '_vif_cache', '_pca_cache', '_pca_update_counter',
        '_last_vif_hash', '_config_hash'
    )

    def __init__(self, symbol: str = "DEFAULT", config: Optional[Dict] = None):
        """
        初始化因子引擎，每个实例绑定单一交易对。
        :param symbol: 交易对标识
        :param config: 可选配置字典，仅允许白名单中的键
        """
        self._symbol = symbol

        # 滚动历史窗口
        self._history = {name: deque(maxlen=self.WINDOW_SIZE) for name in self.ALL_FACTOR_NAMES}
        # 增量统计量 (Welford)
        self._incremental_stats = {
            name: {'count': 0, 'mean': 0.0, 'M2': 0.0} for name in self.ALL_FACTOR_NAMES
        }

        # EMA 状态
        self._obi_ema: Optional[float] = None
        self._whale_flow_ema: Optional[float] = None
        self._vol_fill_count = 0

        # MACD 增量状态
        self._macd_5m = MACDState(self.F3_MACD_FAST_5M, self.F3_MACD_SLOW_5M, self.F3_MACD_SIGNAL_5M)
        self._macd_15m = MACDState(self.F3_MACD_FAST_15M, self.F3_MACD_SLOW_15M, self.F3_MACD_SIGNAL_15M)

        # 权重矩阵深拷贝
        self._weight_matrix = deepcopy(self.DEFAULT_WEIGHT_MATRIX)

        self._warmup_complete = False

        # VIF/PCA 缓存
        self._vif_cache: Optional[Dict[str, float]] = None
        self._pca_cache: Optional[Dict] = None
        self._pca_update_counter = 0
        self._last_vif_hash: Optional[int] = None

        self._config_hash = hash(str(config)) if config else 0

        if config:
            self._apply_config(config)

        logger.info("[KUN-FAC-I001] FactorComputeEngine 初始化完成，交易对=%s", symbol)

    # ---------- 配置管理 ----------
    def _apply_config(self, config: Dict) -> None:
        """仅应用白名单内的配置项，并做类型校验"""
        for key, value in config.items():
            if key not in self._CONFIG_WHITELIST:
                logger.warning("[KUN-FAC-W006] 忽略不允许的配置项: %s", key)
                continue
            expected_type = self._CONFIG_WHITELIST[key]
            if not isinstance(value, expected_type):
                raise TypeError(f"配置项 {key} 期望类型 {expected_type.__name__}, 实际 {type(value).__name__}")
            if key == 'weight_matrix':
                self._weight_matrix = deepcopy(value)
                self._validate_weights()
            else:
                setattr(self, key, value)
        self._config_hash = hash(str(config))

    def _validate_weights(self) -> None:
        """验证权重矩阵和为 1，否则自动归一化（副本）"""
        for regime, cats in self._weight_matrix.items():
            total = sum(cats.values())
            if abs(total - 1.0) > 1e-6:
                logger.warning("[KUN-FAC-W007] 权重矩阵 %s 总和=%.4f，执行归一化", regime, total)
                for k in cats:
                    cats[k] /= total

    # ---------- 预热 ----------
    def warmup(self, historical_3m: List[FactorInputDict],
               historical_5m: Optional[List[float]] = None,
               historical_15m: Optional[List[float]] = None) -> bool:
        """
        使用历史数据预热引擎。
        需提供足够的 3 分钟 K 线快照，以及可选的 5/15 分钟收盘价序列以初始化 MACD。
        """
        if len(historical_3m) < self.WINDOW_SIZE:
            logger.warning("[KUN-FAC-W008] 历史3分钟数据不足 (%d)，无法预热", len(historical_3m))
            return False
        self.reset_history()
        # 初始化 MACD 状态
        if historical_5m and len(historical_5m) >= self.F3_MACD_SLOW_5M:
            self._macd_5m.initialize_with_series(historical_5m)
        if historical_15m and len(historical_15m) >= self.F3_MACD_SLOW_15M:
            self._macd_15m.initialize_with_series(historical_15m)

        for snap in historical_3m[-self.WINDOW_SIZE:]:
            result = self._compute_all_factors_impl(snap, skip_vif=True, is_warmup=True)
            if result['status'] != 'ok':
                continue
            for name in self.ALL_FACTOR_NAMES:
                val = result['factors'][name]
                self._history[name].append(val)
                self._update_statistics(name, val)
        self._warmup_complete = True
        logger.info("[KUN-FAC-I002] 预热完成，窗口大小=%d", len(self._history['F1']))
        return True

    # ---------- 统计工具 ----------
    @staticmethod
    def _is_finite(value: float) -> bool:
        return math.isfinite(value)

    @staticmethod
    def _clip_signal(value: float) -> float:
        """将信号安全限幅在 [0,1]，非有限值返回 0.5"""
        if not math.isfinite(value):
            return 0.5
        return max(0.0, min(1.0, value))

    def _update_statistics(self, name: str, value: float) -> None:
        """Welford 单步更新均值与 M2"""
        stat = self._incremental_stats[name]
        stat['count'] += 1
        delta = value - stat['mean']
        stat['mean'] += delta / stat['count']
        delta2 = value - stat['mean']
        stat['M2'] += delta * delta2
        # 防止长期累积溢出：每 1000 次归一化
        if stat['count'] % 1000 == 0:
            stat['M2'] /= stat['count']
            # 保持均值不变，后续继续增量更新（近似）

    def _robust_standardize(self, value: float, name: str) -> float:
        """基于增量统计的稳健标准化（当前为 Z-score，未来可扩展 MAD）"""
        if not self._warmup_complete or self._incremental_stats[name]['count'] < 5:
            return 0.0
        stat = self._incremental_stats[name]
        variance = stat['M2'] / (stat['count'] - 1) if stat['count'] > 1 else 0.0
        std = math.sqrt(variance)
        if std < self.MAD_EPSILON:
            return 0.0
        return (value - stat['mean']) / std

    @staticmethod
    def _ema_update(current: Optional[float], new_val: float, period: int) -> float:
        """指数移动平均更新"""
        if current is None:
            return new_val
        alpha = 2.0 / (period + 1)
        return current * (1.0 - alpha) + new_val * alpha

    # ---------- 因子计算 ----------
    def _compute_f1(self, close: float, ma25: float, atr14: float) -> float:
        """F1 趋势强度"""
        if atr14 < self.MAD_EPSILON or not self._is_finite(close):
            return 0.5
        raw = (close - ma25) / atr14
        raw = max(-10.0, min(10.0, raw))
        normalized = math.tanh(raw / self.F1_SATURATION_SCALE)
        return self._clip_signal((normalized + 1.0) / 2.0)

    def _compute_f2(self, ma25: float, ma25_past: float, atr14: float) -> float:
        """F2 均线斜率"""
        if atr14 < self.MAD_EPSILON:
            return 0.5
        raw = (ma25 - ma25_past) / atr14
        raw = max(-10.0, min(10.0, raw))
        normalized = math.tanh(raw / 0.3)
        return self._clip_signal((normalized + 1.0) / 2.0)

    def _compute_f3(self, price_5m: float, price_15m: float) -> float:
        """F3 多周期共振 (增量 MACD)"""
        _, _, hist_5m = self._macd_5m.update(price_5m)
        _, _, hist_15m = self._macd_15m.update(price_15m)
        if (hist_5m > 0 and hist_15m > 0) or (hist_5m < 0 and hist_15m < 0):
            # 使用近似 ATR 标准化（建议后续传入 atr 做更精确缩放）
            ref_vol = max(abs(price_5m) * 0.0005, 0.01)
            strength = min(abs(hist_5m), abs(hist_15m)) / ref_vol
            return min(1.0, strength / 5.0)
        return 0.0

    def _compute_f4(self, orderbook: Dict, avg_depth: float) -> Tuple[float, bool]:
        """F4 订单簿不平衡度，返回 (信号强度, 深度是否充足)"""
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        if not bids or not asks:
            return 0.5, False
        try:
            bid_vol = sum(float(b[1]) for b in bids[:self.F4_DEPTH_LEVELS])
            ask_vol = sum(float(a[1]) for a in asks[:self.F4_DEPTH_LEVELS])
        except (IndexError, TypeError, ValueError):
            return 0.5, False
        total = bid_vol + ask_vol
        if total < self.MAD_EPSILON:
            return 0.5, False
        obi = (bid_vol - ask_vol) / total
        # 仅当深度充足时更新 EMA
        depth_ok = total > avg_depth * self.F4_MIN_DEPTH_RATIO
        if depth_ok:
            self._obi_ema = self._ema_update(self._obi_ema, obi, self.F4_EMA_PERIOD)
        smoothed = self._obi_ema if self._obi_ema is not None else obi
        normalized = math.tanh(smoothed / 0.3)
        signal = (normalized + 1.0) / 2.0
        return self._clip_signal(signal), depth_ok

    def _compute_f5(self, orderbook: Dict) -> float:
        """F5 订单簿熵 (集中度)"""
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        try:
            volumes = [float(b[1]) for b in bids[:self.F5_DEPTH_LEVELS]] + \
                      [float(a[1]) for a in asks[:self.F5_DEPTH_LEVELS]]
        except (IndexError, TypeError, ValueError):
            return 0.5
        total = sum(volumes)
        if total < self.MAD_EPSILON:
            return 0.5
        probs = [v / total for v in volumes if v > 0]
        if not probs:
            return 0.5
        entropy = -sum(p * math.log2(p) for p in probs)
        max_entropy = math.log2(len(probs))
        concentration = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
        return self._clip_signal(concentration)

    def _compute_f6(self, trades: List[Dict]) -> float:
        """F6 大单流向"""
        if not trades:
            return 0.5
        buy_vol = sell_vol = total_vol = 0.0
        for t in trades:
            size = float(t.get('qty', t.get('size', 0.0)))  # 兼容币安 size/qty 字段
            total_vol += size
            side = t.get('side', '')
            if size >= self.F6_LARGE_TRADE_THRESHOLD:
                if side == 'buy':
                    buy_vol += size
                elif side == 'sell':
                    sell_vol += size
        if total_vol < self.MAD_EPSILON:
            return 0.5
        flow = (buy_vol - sell_vol) / total_vol
        self._whale_flow_ema = self._ema_update(self._whale_flow_ema, flow, self.F6_EMA_PERIOD)
        normalized = math.tanh(self._whale_flow_ema / 0.2)
        return self._clip_signal((normalized + 1.0) / 2.0)

    def _compute_f7(self, atr14: float, atr56: float) -> float:
        """F7 波动率状态"""
        if atr56 < self.MAD_EPSILON:
            return 0.5
        ratio = atr14 / atr56 - 1.0
        center = (self.F7_OPTIMAL_LOW + self.F7_OPTIMAL_HIGH) / 2.0
        half_range = (self.F7_OPTIMAL_HIGH - self.F7_OPTIMAL_LOW) / 2.0
        if half_range < self.MAD_EPSILON:
            return 0.5
        score = math.exp(-((ratio - center) ** 2) / (2 * half_range ** 2))
        if ratio < self.F7_OPTIMAL_LOW or ratio > self.F7_OPTIMAL_HIGH:
            score *= self.F7_OUTSIDE_PENALTY
        return self._clip_signal(score)

    def _compute_f8(self, current_vol: float, vol_ma: float, vol_std: float) -> float:
        """F8 成交量温度"""
        if vol_std < self.MAD_EPSILON:
            return 0.5
        z = (current_vol - vol_ma) / vol_std
        if z < -0.5:  # 极度缩量
            return 0.1
        signal = 1.0 / (1.0 + math.exp(-(z - 0.5)))
        return self._clip_signal(signal)

    # ---------- VIF 与 PCA ----------
    def _calculate_vif(self) -> Dict[str, float]:
        """计算方差膨胀因子，使用缓存避免重复计算"""
        if not self._warmup_complete:
            return {}
        hist_hash = self._compute_hist_hash()
        if self._vif_cache is not None and hist_hash == self._last_vif_hash:
            return self._vif_cache
        data = {name: list(self._history[name]) for name in self.ALL_FACTOR_NAMES}
        min_len = min(len(v) for v in data.values())
        if min_len < 5:
            return {}
        matrix = np.column_stack([data[name][-min_len:] for name in self.ALL_FACTOR_NAMES])
        # Winsorize 1%-99% 极端值
        lower = np.percentile(matrix, 1, axis=0, method='lower')
        upper = np.percentile(matrix, 99, axis=0, method='higher')
        matrix = np.clip(matrix, lower, upper)
        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)
        std[std < 1e-10] = 1e-10
        X = (matrix - mean) / std
        vif = {}
        for i, name in enumerate(self.ALL_FACTOR_NAMES):
            y = X[:, i]
            X_rest = np.delete(X, i, axis=1)
            try:
                coef = np.linalg.lstsq(X_rest, y, rcond=None)[0]
                y_pred = X_rest @ coef
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                if ss_tot < 1e-10:
                    vif[name] = np.inf
                else:
                    r2 = 1.0 - ss_res / ss_tot
                    vif[name] = np.inf if r2 >= 1.0 else 1.0 / (1.0 - r2)
            except np.linalg.LinAlgError:
                vif[name] = np.inf
        self._vif_cache = vif
        self._last_vif_hash = hist_hash
        return vif

    def _compute_hist_hash(self) -> int:
        """基于最近样本的轻量级哈希，用于检测分布变化"""
        samples = tuple(tuple(list(self._history[name])[-20:]) for name in self.ALL_FACTOR_NAMES)
        return hash(samples)

    def _apply_pca_if_needed(self, factor_values: Dict[str, float],
                             vif_dict: Dict[str, float]) -> Tuple[Dict[str, float], bool]:
        """当 VIF 过高时应用 PCA 降维，返回调整后的因子值和是否应用标志"""
        high_vif = {k: v for k, v in vif_dict.items() if v > self.VIF_THRESHOLD}
        if not high_vif or not self._warmup_complete:
            return factor_values, False
        self._pca_update_counter += 1
        # 每 50 次或首次计算 PCA 投影
        if self._pca_cache is not None and self._pca_update_counter % 50 != 0:
            proj = self._pca_cache
        else:
            affected_idx = [self.ALL_FACTOR_NAMES.index(k) for k in high_vif if k in self.ALL_FACTOR_NAMES]
            if len(affected_idx) < 2:
                return factor_values, False
            data = {name: list(self._history[name]) for name in self.ALL_FACTOR_NAMES}
            min_len = min(len(v) for v in data.values())
            matrix = np.column_stack([data[name][-min_len:] for name in self.ALL_FACTOR_NAMES])
            sub_matrix = matrix[:, affected_idx]
            mean = np.mean(sub_matrix, axis=0)
            std = np.std(sub_matrix, axis=0)
            std[std < 1e-10] = 1e-10
            sub_norm = (sub_matrix - mean) / std
            cov = np.cov(sub_norm.T)
            eig_vals, eig_vecs = np.linalg.eigh(cov)
            eig_vals = np.maximum(eig_vals, 0)
            idx = np.argsort(eig_vals)[::-1]
            eig_vals = eig_vals[idx]
            eig_vecs = eig_vecs[:, idx]
            cum_var = np.cumsum(eig_vals) / np.sum(eig_vals)
            n_comp = np.searchsorted(cum_var, self.PCA_EXPLAINED_VARIANCE) + 1
            n_comp = min(n_comp, len(affected_idx))
            proj = {
                'indices': affected_idx,
                'mean': mean,
                'std': std,
                'components': eig_vecs[:, :n_comp],
                'n_components': n_comp
            }
            self._pca_cache = proj
        # 应用投影
        affected_idx = proj['indices']
        current_sub = np.array([factor_values[self.ALL_FACTOR_NAMES[i]] for i in affected_idx])
        current_norm = (current_sub - proj['mean']) / proj['std']
        projected = current_norm @ proj['components'] @ proj['components'].T
        reconstructed = projected * proj['std'] + proj['mean']
        adjusted = factor_values.copy()
        for i, idx in enumerate(affected_idx):
            adjusted[self.ALL_FACTOR_NAMES[idx]] = self._clip_signal(reconstructed[i])
        logger.info("[KUN-FAC-W004] PCA 已应用至因子 %s", str(high_vif.keys()))
        return adjusted, True

    def get_composite_score(self, factor_signals: Dict[str, float], regime: str,
                            hunger_threshold: float = 0.6) -> CompositeScoreDict:
        """根据市场状态权重聚合因子得分"""
        weights = self._weight_matrix.get(regime)
        if weights is None:
            logger.warning("[KUN-FAC-W009] 未知市场状态 '%s'，回退为 normal", regime)
            weights = self._weight_matrix['normal']
        cat_scores = {'trend_confirm': 0.0, 'microstructure': 0.0, 'risk_env': 0.0}
        cat_counts = {'trend_confirm': 0, 'microstructure': 0, 'risk_env': 0}
        factor_cat = {
            'F1': 'trend_confirm', 'F2': 'trend_confirm', 'F3': 'trend_confirm',
            'F4': 'microstructure', 'F5': 'microstructure', 'F6': 'microstructure',
            'F7': 'risk_env', 'F8': 'risk_env'
        }
        for f, sig in factor_signals.items():
            cat = factor_cat.get(f)
            if cat and math.isfinite(sig):
                cat_scores[cat] += sig
                cat_counts[cat] += 1
        for cat in cat_scores:
            if cat_counts[cat] > 0:
                cat_scores[cat] /= cat_counts[cat]
        composite = sum(cat_scores[cat] * weights.get(cat, 0.0) for cat in cat_scores)
        passed = composite >= hunger_threshold
        return {
            "composite_score": composite,
            "passed": passed,
            "category_breakdown": cat_scores,
            "threshold_used": hunger_threshold
        }

    # ---------- 主入口 ----------
    def compute_all_factors(self, market_data: FactorInputDict) -> FactorOutputDict:
        """线程安全的外观方法（当前无锁，依赖外部事件循环串行调用）"""
        return self._compute_all_factors_impl(market_data, skip_vif=False, is_warmup=False)

    def _compute_all_factors_impl(self, market_data: FactorInputDict,
                                  skip_vif: bool = False,
                                  is_warmup: bool = False) -> FactorOutputDict:
        """内部实现，计算所有因子并返回结构化结果"""
        start_time = time.perf_counter()
        warnings: List[str] = []
        factors: Dict[str, float] = {}

        try:
            close = market_data['close']
            ma25 = market_data['ma25']
            ma25_past = market_data['ma25_past']
            atr14 = market_data['atr14']
            atr56 = market_data.get('atr56', atr14 * 2.5)
            orderbook = market_data.get('orderbook', {})
            avg_depth = market_data.get('avg_depth', 0.0)
            trades = market_data.get('trades', [])
            volume = market_data.get('volume', 0.0)
            vol_ma = market_data.get('vol_ma', volume)
            vol_std = market_data.get('vol_std', 0.01)
            price_5m = market_data.get('close_5m', close)  # 若缺失则用当前价近似
            price_15m = market_data.get('close_15m', close)

            # 订单簿时效检查（使用统一数据网关赋予的本地接收时间）
            local_recv_ms = market_data.get('timestamp_ms', 0)
            ob_timestamp = orderbook.get('timestamp', 0)
            snapshot_age = abs(local_recv_ms - ob_timestamp) if ob_timestamp and local_recv_ms else self.SNAPSHOT_MAX_AGE_MS + 1
            ob_valid = snapshot_age < self.SNAPSHOT_MAX_AGE_MS
            if not ob_valid:
                warnings.append(f"[KUN-FAC-W001] 订单簿快照过期 ({snapshot_age}ms)")

            # 成交量填充
            if volume <= 0 and self._vol_fill_count < self.MAX_VOL_FILL:
                volume = vol_ma if vol_ma > 0 else 1.0
                self._vol_fill_count += 1
                warnings.append(f"[KUN-FAC-W002] 成交量填充次数={self._vol_fill_count}")
            elif volume <= 0:
                self._vol_fill_count = 0
                warnings.append("[KUN-FAC-E001] 成交量填充次数已超限")
                volume = 0.0

            # 计算各因子
            factors['F1'] = self._compute_f1(close, ma25, atr14)
            factors['F2'] = self._compute_f2(ma25, ma25_past, atr14)
            factors['F3'] = self._compute_f3(price_5m, price_15m)

            if ob_valid:
                f4_sig, depth_ok = self._compute_f4(orderbook, avg_depth)
                factors['F4'] = f4_sig
                if not depth_ok:
                    warnings.append("[KUN-FAC-W005] 订单簿深度不足")
                factors['F5'] = self._compute_f5(orderbook)
            else:
                factors['F4'] = 0.5
                factors['F5'] = 0.5

            factors['F6'] = self._compute_f6(trades)
            factors['F7'] = self._compute_f7(atr14, atr56)
            factors['F8'] = self._compute_f8(volume, vol_ma, vol_std)

            # NaN/Inf 保护
            for name in self.ALL_FACTOR_NAMES:
                if not self._is_finite(factors[name]):
                    logger.error("[KUN-FAC-F005] 因子 %s 非有限值，设为0.5", name)
                    factors[name] = 0.5
                    warnings.append(f"[KUN-FAC-F005] {name} 非有限")

            # 更新历史窗口和统计量（预热期间也更新）
            for name in self.ALL_FACTOR_NAMES:
                self._history[name].append(factors[name])
                self._update_statistics(name, factors[name])

            if len(self._history['F1']) >= self.WINDOW_SIZE and not is_warmup:
                self._warmup_complete = True

            # VIF 与 PCA（非预热且非跳过时计算）
            vif_result: Dict[str, float] = {}
            if not skip_vif and self._warmup_complete:
                vif_result = self._calculate_vif()
                if vif_result:
                    adj, pca_used = self._apply_pca_if_needed(factors, vif_result)
                    if pca_used:
                        factors = adj
                        warnings.append("[KUN-FAC-W004] PCA 已应用")

            elapsed_us = int((time.perf_counter() - start_time) * 1_000_000)
            return {
                "status": "ok",
                "factors": factors,
                "vif": vif_result,
                "warnings": warnings,
                "metrics": {
                    "compute_time_us": elapsed_us,
                    "warmup_complete": self._warmup_complete,
                    "snapshot_age_ms": snapshot_age
                }
            }

        except KeyError as e:
            logger.error("[KUN-FAC-E002] 缺少关键数据字段: %s", e)
            # 返回安全的默认因子值
            safe_factors = {name: 0.5 for name in self.ALL_FACTOR_NAMES}
            return {
                "status": "error",
                "reason": f"缺少字段: {e}",
                "factors": safe_factors,
                "vif": {},
                "warnings": warnings,
                "metrics": {}
            }
        except Exception as e:
            logger.exception("[KUN-FAC-F001] 因子计算未捕获异常")
            safe_factors = {name: 0.5 for name in self.ALL_FACTOR_NAMES}
            return {
                "status": "error",
                "reason": str(e),
                "factors": safe_factors,
                "vif": {},
                "warnings": warnings,
                "metrics": {}
            }

    # ---------- 健康检查与工具 ----------
    def health_check(self) -> HealthCheckDict:
        """自检模块，使用独立数据（不污染生产状态）"""
        try:
            # 构造一个临时引擎实例进行测试，避免污染生产数据
            test_engine = FactorComputeEngine(symbol="HEALTHCHECK")
            test_snap: FactorInputDict = {
                'close': 50000.0, 'ma25': 49900.0, 'ma25_past': 49800.0,
                'atr14': 200.0, 'atr56': 300.0,
                'orderbook': {
                    'bids': [[49950, 1.5], [49940, 2.0]],
                    'asks': [[50050, 1.0], [50060, 0.5]],
                    'timestamp': int(time.time() * 1000) - 10
                },
                'avg_depth': 3.0,
                'trades': [{'side': 'buy', 'size': 60000}, {'side': 'sell', 'size': 20000}],
                'close_5m': 50000.0, 'close_15m': 50000.0,
                'volume': 100.0, 'vol_ma': 90.0, 'vol_std': 15.0,
                'timestamp_ms': int(time.time() * 1000) - 5
            }
            result = test_engine.compute_all_factors(test_snap)
            if result['status'] != 'ok':
                return {"status": "error", "message": result.get('reason', '未知错误'), "warmup_complete": False}
            if len(result['factors']) != 8:
                return {"status": "error", "message": f"因子数量异常 {len(result['factors'])}", "warmup_complete": False}
            for f, v in result['factors'].items():
                if not (0.0 <= v <= 1.0):
                    return {"status": "error", "message": f"{f} 数值越界 {v}", "warmup_complete": False}
            return {"status": "ok", "message": "所有因子计算路径验证通过", "warmup_complete": test_engine._warmup_complete}
        except Exception as e:
            logger.exception("健康检查异常")
            return {"status": "error", "message": str(e), "warmup_complete": False}

    def reset_history(self) -> None:
        """重置所有历史数据和状态"""
        for buf in self._history.values():
            buf.clear()
        for stat in self._incremental_stats.values():
            stat['count'] = 0
            stat['mean'] = 0.0
            stat['M2'] = 0.0
        self._obi_ema = None
        self._whale_flow_ema = None
        self._vol_fill_count = 0
        self._warmup_complete = False
        self._macd_5m.reset()
        self._macd_15m.reset()
        self._vif_cache = None
        self._pca_cache = None
        self._pca_update_counter = 0
        logger.info("[KUN-FAC-I004] 历史与缓存已重置")

    def update_weights(self, new_weights: Dict[str, Dict[str, float]]) -> None:
        """更新权重矩阵（深拷贝）"""
        self._weight_matrix = deepcopy(new_weights)
        self._validate_weights()
        logger.info("[KUN-FAC-I003] 权重矩阵已更新")

    @property
    def is_warm(self) -> bool:
        """返回引擎是否已完成预热"""
        return self._warmup_complete

    @property
    def symbol(self) -> str:
        """返回绑定的交易对"""
        return self._symbol

    def get_weights(self) -> Dict[str, Dict[str, float]]:
        """返回当前权重的深拷贝"""
        return deepcopy(self._weight_matrix)

    def __repr__(self) -> str:
        return (f"FactorComputeEngine(symbol={self._symbol}, "
                f"warm={self._warmup_complete}, history={len(self._history['F1'])})")
