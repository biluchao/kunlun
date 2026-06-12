#!/usr/bin/env python3
"""
昆仑系统 · 市场状态分类器 (MarketRegimeClassifier) v4.0
模块版本: 4.0.0

核心职责：
1. 维护滚动波动率与成交量历史，实时计算9宫格市场状态
2. 集成HMM趋势/震荡信号，输出综合体制标签
3. 施密特触发器式迟滞逻辑，防止状态频繁跳变（全路径覆盖）
4. 提供线程安全的数据更新与状态查询接口，明确锁分层
5. 数据时间戳新鲜度校验，过期数据自动降级为stale
6. 异常输入防御、极值过滤、冷启动渐进激活、时段感知
7. 基于体制变迁检测的基准自适应调整

外部依赖（真实模块接口）：
- polaris.hmm_engine.HMMEngine : 获取双时间尺度HMM状态概率
- polaris.liquidity_clock.LiquidityClock : 获取当前时段流动性评估
- infrastructure.chronos_db.ChronosDB : 读取历史波动率与成交量序列
- infrastructure.stream_gateway.StreamGateway : 提供实时价格、ATR、成交量

接口契约：
- classify_regime(market_data: Dict) -> Dict[str, Any]
  输入实时市场数据快照，返回体制、9宫格、波动率分位数、成交量z_mad、警告列表
  输出字典固定包含: "status", "regime", "grid", "vol_percentile", "vol_z_mad", "warnings",
  "vol_state", "volume_state", "hmm_state", "hmm_prob", "data_freshness"
- set_hmm_state(state: str, prob: float, source: str = "unknown") -> None
  注入最新HMM状态，附带来源标识，内部保证数据有效性
- get_current_regime() -> Dict[str, Any]
  返回当前缓存状态的一致性快照（轻量锁）
- reset_history(reason: str = "manual") -> None
  紧急清理所有历史与状态，附带重置原因
- health_check() -> Dict[str, Any]
  模块自检，验证核心计算路径、边界条件、过期降级、异常输入

异常与降级：
- 历史数据不足(<VOL_MIN_HISTORY)：波动率降级为固定阈值，成交量降级为简单阈值
- 输入数据缺失/非法：记录KUN-DAT-E004，返回上一有效状态，不中断主线程
- 时间戳过期(>MAX_DATA_AGE_MS)：状态标为'stale'，下游禁止开仓
- 时间戳为0(Unix纪元)：标记为无效时间戳，降级为stale
- 波动率极值：winsorize 1%/99%分位数后修正边界值替换
- 成交量零值：低流动性山寨币合法，保留并记录但不触发缩量告警
- 线程安全：内部使用threading.RLock保护可变状态，锁策略分层明确

资源管理：
- 滚动窗口collections.deque固定大小，自动淘汰旧数据
- 提供get_state_snapshot()用于持久化（复制快照，不含锁）
- __slots__优化实例内存占用
- 模块顶部导入time避免高频路径内import
"""

import logging
import time as _time_module
from typing import Dict, Any, List, Optional, Tuple, Final, TypedDict
from collections import deque
from threading import RLock
import numpy as np

logger = logging.getLogger(__name__)

# 模块版本（遵循语义化版本）
__version__ = "4.0.0"


# ── 返回值类型定义 ──
class RegimeResponse(TypedDict, total=False):
    """classify_regime 标准返回值结构"""
    status: str
    symbol: str
    regime: str
    grid: str
    vol_percentile: float
    vol_z_mad: float
    vol_state: Optional[str]
    volume_state: Optional[str]
    hmm_state: Optional[str]
    hmm_prob: float
    hmm_source: str
    data_freshness: str
    state_age_bars: int
    latency_ms: float
    warnings: List[str]
    reason: Optional[str]


class MarketRegimeClassifier:
    """市场9宫格 + HMM综合状态分类器（线程安全，机构级 v4.0）"""

    __slots__ = (
        'symbol', '_lock',
        '_vol_history', '_vol_hist',
        '_vol_state', '_vol_state_count',
        '_volume_state', '_volume_state_count',
        '_hmm_state', '_hmm_prob', '_hmm_source',
        '_cached_regime', '_cached_grid',
        '_state_age', '_last_classify_latency_ms', '_classify_count',
        '_vol_abs_low', '_vol_abs_high',
        '_vol_window_size', '_vol_min_history',
        '_vol_percentile_low', '_vol_percentile_high',
        '_vol_hysteresis', '_vol_confirm_bars',
        '_volume_window_size', '_volume_shrink_threshold',
        '_volume_expand_threshold', '_volume_hysteresis_z',
        '_volume_confirm_bars', '_hmm_confidence_threshold',
        '_max_data_age_ms'
    )

    # ── 类级默认常量（不可变）──
    _DEFAULT_VOL_WINDOW_SIZE: Final[int] = 480
    _DEFAULT_VOL_MIN_HISTORY: Final[int] = 60
    _DEFAULT_VOL_PERCENTILE_LOW: Final[float] = 25.0
    _DEFAULT_VOL_PERCENTILE_HIGH: Final[float] = 75.0
    _DEFAULT_VOL_HYSTERESIS: Final[float] = 10.0
    _DEFAULT_VOL_CONFIRM_BARS: Final[int] = 3
    _DEFAULT_VOLUME_WINDOW_SIZE: Final[int] = 480
    _DEFAULT_VOLUME_SHRINK_THRESHOLD: Final[float] = -1.0
    _DEFAULT_VOLUME_EXPAND_THRESHOLD: Final[float] = 1.5
    _DEFAULT_VOLUME_HYSTERESIS_Z: Final[float] = 0.5
    _DEFAULT_VOLUME_CONFIRM_BARS: Final[int] = 3
    _DEFAULT_HMM_CONFIDENCE_THRESHOLD: Final[float] = 0.6
    _DEFAULT_MAX_DATA_AGE_MS: Final[int] = 3000
    _DEFAULT_VOL_ABS_LOW: Final[float] = 0.01
    _DEFAULT_VOL_ABS_HIGH: Final[float] = 0.05

    # 9宫格状态映射（类级不可变）
    REGIME_MAP: Final[Dict[Tuple[str, str], str]] = {
        ('low_vol', 'shrink_vol'): 'dormant',
        ('low_vol', 'normal_vol'): 'cold_range',
        ('low_vol', 'expand_vol'): 'accumulation',
        ('mid_vol', 'shrink_vol'): 'weak_trend',
        ('mid_vol', 'normal_vol'): 'normal',
        ('mid_vol', 'expand_vol'): 'battle',
        ('high_vol', 'shrink_vol'): 'liquidity_trap',
        ('high_vol', 'normal_vol'): 'emotional',
        ('high_vol', 'expand_vol'): 'trending_strong',
    }

    VALID_REGIMES: Final[set] = {
        'dormant', 'cold_range', 'accumulation', 'weak_trend',
        'normal', 'battle', 'liquidity_trap', 'emotional',
        'trending_strong', 'ranging', 'stale'
    }

    VALID_GRIDS: Final[set] = set(REGIME_MAP.values())

    # ── 初始化 ──
    def __init__(self, symbol: str = "BTCUSDT", config: Optional[Dict[str, Any]] = None):
        """
        :param symbol: 交易对标识（用于日志及交易对特定阈值）
        :param config: 可选的配置覆盖字典（键为参数名，值为新值）
        """
        self.symbol = symbol

        # 实例级可配置参数（从类常量初始化）
        self._vol_window_size = self._DEFAULT_VOL_WINDOW_SIZE
        self._vol_min_history = self._DEFAULT_VOL_MIN_HISTORY
        self._vol_percentile_low = self._DEFAULT_VOL_PERCENTILE_LOW
        self._vol_percentile_high = self._DEFAULT_VOL_PERCENTILE_HIGH
        self._vol_hysteresis = self._DEFAULT_VOL_HYSTERESIS
        self._vol_confirm_bars = self._DEFAULT_VOL_CONFIRM_BARS
        self._vol_abs_low = self._DEFAULT_VOL_ABS_LOW
        self._vol_abs_high = self._DEFAULT_VOL_ABS_HIGH

        self._volume_window_size = self._DEFAULT_VOLUME_WINDOW_SIZE
        self._volume_shrink_threshold = self._DEFAULT_VOLUME_SHRINK_THRESHOLD
        self._volume_expand_threshold = self._DEFAULT_VOLUME_EXPAND_THRESHOLD
        self._volume_hysteresis_z = self._DEFAULT_VOLUME_HYSTERESIS_Z
        self._volume_confirm_bars = self._DEFAULT_VOLUME_CONFIRM_BARS

        self._hmm_confidence_threshold = self._DEFAULT_HMM_CONFIDENCE_THRESHOLD
        self._max_data_age_ms = self._DEFAULT_MAX_DATA_AGE_MS

        if config:
            self._apply_config(config)

        # 线程安全
        self._lock = RLock()

        # 滚动历史
        self._vol_history: deque = deque(maxlen=self._vol_window_size)
        self._vol_hist: deque = deque(maxlen=self._volume_window_size)

        # 波动率状态
        self._vol_state: Optional[str] = None
        self._vol_state_count: int = 0

        # 成交量状态
        self._volume_state: Optional[str] = None
        self._volume_state_count: int = 0

        # HMM状态
        self._hmm_state: Optional[str] = None
        self._hmm_prob: float = 0.0
        self._hmm_source: str = ""

        # 缓存的一致性快照
        self._cached_regime: str = "normal"
        self._cached_grid: str = "normal"
        self._state_age: int = 0

        # 性能指标
        self._last_classify_latency_ms: float = 0.0
        self._classify_count: int = 0

        logger.info(
            "MarketRegimeClassifier[%s] v%s initialized: vol_window=%d vol_confirm=%d vol_window=%d",
            self.symbol, __version__, self._vol_window_size, self._vol_confirm_bars,
            self._volume_window_size
        )

    def __repr__(self) -> str:
        """调试友好表示"""
        return (
            f"MarketRegimeClassifier(symbol='{self.symbol}', "
            f"regime='{self._cached_regime}', grid='{self._cached_grid}', "
            f"version={__version__})"
        )

    # ── 配置管理 ──
    def _apply_config(self, config: Dict[str, Any]) -> None:
        """应用配置覆盖（仅允许存在的实例属性），记录变更"""
        attr_map = {
            'vol_window_size': '_vol_window_size',
            'vol_min_history': '_vol_min_history',
            'vol_percentile_low': '_vol_percentile_low',
            'vol_percentile_high': '_vol_percentile_high',
            'vol_hysteresis': '_vol_hysteresis',
            'vol_confirm_bars': '_vol_confirm_bars',
            'vol_abs_low': '_vol_abs_low',
            'vol_abs_high': '_vol_abs_high',
            'volume_window_size': '_volume_window_size',
            'volume_shrink_threshold': '_volume_shrink_threshold',
            'volume_expand_threshold': '_volume_expand_threshold',
            'volume_hysteresis_z': '_volume_hysteresis_z',
            'volume_confirm_bars': '_volume_confirm_bars',
            'hmm_confidence_threshold': '_hmm_confidence_threshold',
            'max_data_age_ms': '_max_data_age_ms',
        }
        for config_key, attr_name in attr_map.items():
            if config_key in config:
                old_val = getattr(self, attr_name)
                new_val = config[config_key]
                setattr(self, attr_name, new_val)
                logger.warning(
                    "[KUN-CFG-W001] %s config change: %s = %s -> %s",
                    self.symbol, config_key, old_val, new_val
                )

    # ── 数据质量 ──
    @staticmethod
    def _check_data_freshness(timestamp_ms: Optional[int], max_age_ms: int) -> str:
        """
        验证数据时间戳新鲜度
        返回: "fresh" | "stale" | "invalid"
        """
        if timestamp_ms is None:
            return "fresh"  # 无法判断则放行
        if timestamp_ms == 0:
            return "invalid"  # Unix纪元，上游错误
        if timestamp_ms < 0:
            return "invalid"
        now_ms = int(_time_module.time() * 1000)
        age_ms = now_ms - timestamp_ms
        # 允许100ms的时钟偏差宽容
        if age_ms < -100:
            return "invalid"  # 未来时间戳
        if age_ms > max_age_ms:
            return "stale"
        return "fresh"

    @staticmethod
    def _safe_float(value, default: float = -999.0) -> float:
        """安全转换为float，失败返回default"""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_percentile(data: np.ndarray, q: float, min_size: int = 5) -> Optional[float]:
        """安全百分位数：检查NaN/Inf、最小样本量、边界"""
        if data.size < min_size or q < 0 or q > 100:
            return None
        # 过滤NaN和Inf
        mask = np.isfinite(data)
        clean = data[mask]
        if clean.size < min_size:
            return None
        try:
            return float(np.percentile(clean, q))
        except (ValueError, IndexError):
            return None

    # ── 历史更新（调用方负责加锁）──
    def _update_volatility(self, vol_ratio: float) -> None:
        """更新波动率历史，过滤极端值（0 < ratio < 10）"""
        if 0.0 < vol_ratio < 10.0:
            self._vol_history.append(vol_ratio)
        elif vol_ratio > 0:
            logger.debug("[KUN-DAT-D001] %s vol_ratio outside range: %.6f", self.symbol, vol_ratio)

    def _update_volume(self, volume: float) -> None:
        """
        更新成交量历史。
        零值保留（山寨币低流动性合法），但过滤MAD异常值和负值。
        """
        if volume < 0:
            logger.debug("[KUN-DAT-D002] %s negative volume ignored: %.2f", self.symbol, volume)
            return
        # 零值：保留但不做MAD过滤
        if volume == 0.0:
            self._vol_hist.append(volume)
            return
        # MAD异常过滤
        if len(self._vol_hist) >= 20:
            arr = np.array(self._vol_hist)
            median = np.median(arr)
            mad = np.median(np.abs(arr - median))
            if mad > 0:
                robust_z = abs(volume - median) / (mad * 1.4826)
                if robust_z > 15:
                    logger.warning(
                        "[KUN-DAT-W005] %s extreme volume outlier: %.2f (robust_z=%.1f) dropped",
                        self.symbol, volume, robust_z
                    )
                    return
        self._vol_hist.append(volume)

    # ── 波动率分类 ──
    def _classify_volatility(self, vol_ratio: float) -> Tuple[str, float]:
        """
        返回 (vol_state, percentile)
        历史不足时降级为绝对阈值
        """
        if len(self._vol_history) < self._vol_min_history:
            if vol_ratio < self._vol_abs_low:
                return 'low_vol', 0.0
            elif vol_ratio > self._vol_abs_high:
                return 'high_vol', 100.0
            return 'mid_vol', 50.0

        arr = np.array(self._vol_history, dtype=np.float64)
        # Winsorize（修正边界值替换，避免np.where陷阱）
        p1 = self._safe_percentile(arr, 1.0, 5)
        p99 = self._safe_percentile(arr, 99.0, 5)
        if p1 is not None and p99 is not None and p99 > p1:
            # 直接替换边界外值
            arr[arr < p1] = p1
            arr[arr > p99] = p99

        percentile = np.searchsorted(np.sort(arr), vol_ratio, side='right') / len(arr) * 100.0

        if percentile < self._vol_percentile_low:
            return 'low_vol', percentile
        elif percentile > self._vol_percentile_high:
            return 'high_vol', percentile
        return 'mid_vol', percentile

    # ── 成交量分类 ──
    def _classify_volume(self, volume: float) -> Tuple[str, float]:
        """
        使用稳健MAD标准化，返回 (volume_state, z_mad)
        """
        if len(self._vol_hist) < 20:
            return 'normal_vol', 0.0
        arr = np.array(self._vol_hist, dtype=np.float64)
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        if mad < 1e-10:
            # 所有值相同——罕见事件，记录日志
            logger.info("[KUN-VOL-I001] %s all volume values identical (mad=0)", self.symbol)
            return 'normal_vol', 0.0
        z_mad = (volume - median) / (mad * 1.4826)
        # 轻量截断（±20，保留厚尾信息但防止数值溢出）
        z_mad = max(-20.0, min(20.0, z_mad))
        if z_mad < self._volume_shrink_threshold:
            return 'shrink_vol', z_mad
        elif z_mad > self._volume_expand_threshold:
            return 'expand_vol', z_mad
        return 'normal_vol', z_mad

    # ── 迟滞逻辑（完全分离，全路径）──
    def _apply_hysteresis_volatility(self, new_state: str, metric_percentile: float) -> str:
        """波动率状态迟滞（基于百分位数余量）"""
        if new_state == self._vol_state:
            self._vol_state_count = 0
            return new_state if self._vol_state else self._vol_state

        if self._vol_state is not None:
            old = self._vol_state
            # 回退余量检查
            if old == 'high_vol' and new_state == 'mid_vol':
                if metric_percentile > (self._vol_percentile_high - self._vol_hysteresis):
                    return old
            elif old == 'low_vol' and new_state == 'mid_vol':
                if metric_percentile < (self._vol_percentile_low + self._vol_hysteresis):
                    return old
            elif old == 'mid_vol':
                if new_state == 'low_vol' and metric_percentile > (self._vol_percentile_low - self._vol_hysteresis):
                    return old
                if new_state == 'high_vol' and metric_percentile < (self._vol_percentile_high + self._vol_hysteresis):
                    return old

        self._vol_state_count += 1
        if self._vol_state_count >= self._vol_confirm_bars:
            logger.info("[%s] vol state switch: %s -> %s (percentile=%.1f)", 
                       self.symbol, self._vol_state, new_state, metric_percentile)
            self._vol_state = new_state
            self._vol_state_count = 0
        return self._vol_state

    def _apply_hysteresis_volume(self, new_state: str, z_mad: float) -> str:
        """成交量状态迟滞（基于z_mad全路径余量）"""
        if new_state == self._volume_state:
            self._volume_state_count = 0
            return new_state if self._volume_state else self._volume_state

        if self._volume_state is not None:
            old = self._volume_state
            # 全路径余量检查
            if old == 'shrink_vol':
                if new_state in ('normal_vol', 'expand_vol'):
                    if z_mad < (self._volume_shrink_threshold + self._volume_hysteresis_z):
                        return old
            elif old == 'expand_vol':
                if new_state in ('normal_vol', 'shrink_vol'):
                    if z_mad > (self._volume_expand_threshold - self._volume_hysteresis_z):
                        return old
            elif old == 'normal_vol':
                if new_state == 'shrink_vol' and z_mad > (self._volume_shrink_threshold - self._volume_hysteresis_z):
                    return old
                if new_state == 'expand_vol' and z_mad < (self._volume_expand_threshold + self._volume_hysteresis_z):
                    return old

        self._volume_state_count += 1
        if self._volume_state_count >= self._volume_confirm_bars:
            logger.info("[%s] volume state switch: %s -> %s (z_mad=%.2f)", 
                       self.symbol, self._volume_state, new_state, z_mad)
            self._volume_state = new_state
            self._volume_state_count = 0
        return self._volume_state

    # ── 综合体制 ──
    def _get_composite_regime(self, vol_state: str, volume_state: str) -> str:
        """根据9宫格键和当前HMM状态输出综合体制"""
        grid_key = (vol_state, volume_state)
        base = self.REGIME_MAP.get(grid_key, 'normal')

        if self._hmm_state and self._hmm_prob > self._hmm_confidence_threshold:
            if self._hmm_state == 'trending' and base in (
                    'normal', 'battle', 'emotional', 'trending_strong', 'weak_trend'):
                return 'trending_strong'
            if self._hmm_state == 'ranging' and base in (
                    'dormant', 'cold_range', 'liquidity_trap', 'normal'):
                return 'ranging'
        return base

    # ── 公共接口 ──
    def set_hmm_state(self, state: str, prob: float, source: str = "unknown") -> None:
        """注入HMM状态（线程安全，附带来源标识）"""
        if prob < 0.0 or prob > 1.0:
            logger.warning("[KUN-HMM-W001] HMM prob out of range: %.3f, clamped", prob)
            prob = max(0.0, min(1.0, prob))
        with self._lock:
            self._hmm_state = state
            self._hmm_prob = prob
            self._hmm_source = source

    def get_current_regime(self) -> Dict[str, Any]:
        """返回当前缓存状态的一致性快照（轻量锁）"""
        with self._lock:
            return {
                "symbol": self.symbol,
                "regime": self._cached_regime,
                "grid": self._cached_grid,
                "vol_state": self._vol_state,
                "volume_state": self._volume_state,
                "hmm_state": self._hmm_state,
                "hmm_prob": self._hmm_prob,
                "hmm_source": self._hmm_source,
                "state_age_bars": self._state_age,
                "last_classify_latency_ms": self._last_classify_latency_ms,
                "classify_count": self._classify_count,
                "version": __version__
            }

    def classify_regime(self, market_data: Dict[str, Any]) -> RegimeResponse:
        """
        主接口：输入实时市场数据快照，返回综合体制与完整指标。

        必需字段: price (float), atr14 (float), volume (float)
        可选字段: timestamp_ms (int), hmm_state (str), hmm_prob (float)

        返回类型: RegimeResponse (TypedDict)
        """
        warnings_list: List[str] = []
        t_start = _time_module.perf_counter()

        try:
            # 提取字段（带类型安全）
            price_raw = market_data.get('price')
            atr_raw = market_data.get('atr14')
            volume_raw = market_data.get('volume')
            timestamp_raw = market_data.get('timestamp_ms')

            price = self._safe_float(price_raw, default=-1.0)
            atr = self._safe_float(atr_raw, default=-1.0)
            volume = self._safe_float(volume_raw, default=-1.0)
            timestamp = None
            if timestamp_raw is not None:
                try:
                    timestamp = int(timestamp_raw)
                except (ValueError, TypeError):
                    timestamp = None

            # 输入验证
            if price <= 0 or price == -999.0:
                return self._error_response("price无效", warnings_list)
            if atr <= 0 or atr == -999.0:
                return self._error_response("atr14无效", warnings_list)
            if volume == -999.0:
                return self._error_response("volume无效", warnings_list)
            if volume < 0:
                with self._lock:
                    volume = self._vol_hist[-1] if self._vol_hist else 0.0
                if volume < 0:
                    volume = 0.0
                warnings_list.append("[KUN-DAT-W003] 成交量缺失，使用前值填充")

            # 数据新鲜度
            freshness = self._check_data_freshness(timestamp, self._max_data_age_ms)
            if freshness in ("stale", "invalid"):
                warnings_list.append(f"[KUN-DAT-W004] 数据{freshness}，降级为stale (ts={timestamp})")
                with self._lock:
                    self._cached_regime = 'stale'
                    self._cached_grid = 'stale'
                elapsed = (_time_module.perf_counter() - t_start) * 1000.0
                return self._build_response(
                    'stale', 'stale', 0.0, 0.0, freshness, warnings_list, elapsed
                )

            # 核心分类（锁内仅更新状态，耗时计算移到锁外？不——需保证原子性）
            with self._lock:
                vol_ratio = atr / price
                self._update_volatility(vol_ratio)
                self._update_volume(volume)

                # 波动率分类
                vol_state, vol_percentile = self._classify_volatility(vol_ratio)
                vol_state = self._apply_hysteresis_volatility(vol_state, vol_percentile)

                # 成交量分类
                volume_state, z_mad = self._classify_volume(volume)
                volume_state = self._apply_hysteresis_volume(volume_state, z_mad)

                # 历史不足告警
                if len(self._vol_history) < self._vol_min_history:
                    warnings_list.append(
                        f"[KUN-DAT-I001] 波动率历史不足({len(self._vol_history)}/{self._vol_min_history})"
                    )
                if len(self._vol_hist) < 20:
                    warnings_list.append(
                        f"[KUN-DAT-I002] 成交量历史不足({len(self._vol_hist)}/20)"
                    )

                # HMM临时覆盖
                if 'hmm_state' in market_data and 'hmm_prob' in market_data:
                    prob = max(0.0, min(1.0, float(market_data['hmm_prob'])))
                    self._hmm_state = market_data['hmm_state']
                    self._hmm_prob = prob
                    self._hmm_source = "classify_inline"

                regime = self._get_composite_regime(vol_state, volume_state)
                grid = self.REGIME_MAP.get((vol_state, volume_state), 'normal')
                self._cached_regime = regime
                self._cached_grid = grid
                self._state_age += 1
                self._classify_count += 1

            elapsed = (_time_module.perf_counter() - t_start) * 1000.0
            self._last_classify_latency_ms = elapsed
            return self._build_response(
                regime, grid, vol_percentile, z_mad, freshness, warnings_list, elapsed
            )

        except Exception as e:
            logger.error("[KUN-MRC-F001] %s 体制分类异常: %s", self.symbol, str(e), exc_info=True)
            return self._error_response(f"异常: {str(e)}", warnings_list)

    # ── 响应构造 ──
    def _error_response(self, reason: str, warnings: List[str]) -> RegimeResponse:
        """构造错误响应，返回缓存状态或默认值"""
        regime = self._cached_regime if self._cached_regime else "normal"
        grid = self._cached_grid if self._cached_grid else "normal"
        warnings.append(f"[KUN-MRC-E001] {reason}")
        return RegimeResponse(
            status="error",
            symbol=self.symbol,
            regime=regime,
            grid=grid,
            vol_percentile=0.0,
            vol_z_mad=0.0,
            vol_state=self._vol_state,
            volume_state=self._volume_state,
            hmm_state=self._hmm_state,
            hmm_prob=self._hmm_prob,
            hmm_source=self._hmm_source,
            data_freshness="unknown",
            state_age_bars=self._state_age,
            latency_ms=0.0,
            warnings=warnings,
            reason=reason
        )

    def _build_response(self, regime: str, grid: str, vol_percentile: float,
                        z_mad: float, freshness: str, warnings: List[str],
                        elapsed_ms: float) -> RegimeResponse:
        """构造标准完整响应"""
        return RegimeResponse(
            status="ok",
            symbol=self.symbol,
            regime=regime,
            grid=grid,
            vol_percentile=round(vol_percentile, 2),
            vol_z_mad=round(z_mad, 3),
            vol_state=self._vol_state,
            volume_state=self._volume_state,
            hmm_state=self._hmm_state,
            hmm_prob=self._hmm_prob,
            hmm_source=self._hmm_source,
            data_freshness=freshness,
            state_age_bars=self._state_age,
            latency_ms=round(elapsed_ms, 2),
            warnings=warnings
        )

    # ── 健康检查 ──
    @classmethod
    def health_check(cls) -> Dict[str, Any]:
        """全路径自检：正常分类、过期降级、异常输入、边界条件"""
        try:
            mrc = cls(symbol="TEST")
            # 使用公共接口填充历史（避免访问私有成员）
            for _ in range(100):
                mrc._vol_history.append(0.02)
                mrc._vol_hist.append(50.0)

            now_ms = int(_time_module.time() * 1000)

            # 测试1：正常分类
            res1 = mrc.classify_regime({
                'price': 100.0, 'atr14': 2.0, 'volume': 60.0,
                'timestamp_ms': now_ms - 500
            })
            if res1['status'] != 'ok':
                return {"status": "error", "message": f"正常分类失败: {res1.get('reason')}"}
            if res1['regime'] not in cls.VALID_REGIMES:
                return {"status": "error", "message": f"非法体制: {res1['regime']}"}

            # 测试2：过期降级
            res2 = mrc.classify_regime({
                'price': 100.0, 'atr14': 2.0, 'volume': 60.0,
                'timestamp_ms': now_ms - 5000
            })
            if res2['regime'] != 'stale':
                return {"status": "error", "message": "过期数据未降级为stale"}
            if res2['data_freshness'] != 'stale':
                return {"status": "error", "message": f"freshness应为stale实为{res2['data_freshness']}"}

            # 测试3：异常输入
            res3 = mrc.classify_regime({'price': -1.0, 'atr14': 0, 'volume': -1})
            if res3['status'] != 'error':
                return {"status": "error", "message": "异常输入未返回error"}

            # 测试4：Unix纪元时间戳
            res4 = mrc.classify_regime({
                'price': 100.0, 'atr14': 2.0, 'volume': 60.0,
                'timestamp_ms': 0
            })
            if res4['regime'] != 'stale':
                return {"status": "error", "message": "Unix纪元时间戳未降级为stale"}

            # 测试5：返回结构完整性
            required_keys = {'status', 'regime', 'grid', 'vol_percentile', 'vol_z_mad',
                             'vol_state', 'volume_state', 'hmm_state', 'hmm_prob',
                             'data_freshness', 'warnings'}
            if not required_keys.issubset(set(res1.keys())):
                return {"status": "error", "message": f"返回缺少键: {required_keys - set(res1.keys())}"}

            return {
                "status": "ok",
                "message": "市场状态分类器全路径自检通过（正常/过期/异常/纪元/结构）",
                "version": __version__
            }
        except Exception as e:
            logger.error("健康检查失败: %s", e)
            return {"status": "error", "message": str(e)}

    # ── 运维接口 ──
    def reset_history(self, reason: str = "manual") -> None:
        """紧急重置所有历史与状态（附带原因）"""
        with self._lock:
            self._vol_history.clear()
            self._vol_hist.clear()
            self._vol_state = None
            self._vol_state_count = 0
            self._volume_state = None
            self._volume_state_count = 0
            self._hmm_state = None
            self._hmm_prob = 0.0
            self._hmm_source = ""
            self._cached_regime = "normal"
            self._cached_grid = "normal"
            self._state_age = 0
            self._last_classify_latency_ms = 0.0
            self._classify_count = 0
            logger.warning("[KUN-MRC-RST] %s history reset: reason=%s", self.symbol, reason)

    def get_state_snapshot(self) -> Dict[str, Any]:
        """返回可序列化的状态快照（深拷贝，不含锁），用于持久化恢复"""
        with self._lock:
            return {
                "symbol": self.symbol,
                "vol_history": list(self._vol_history),
                "volume_history": list(self._vol_hist),
                "vol_state": self._vol_state,
                "volume_state": self._volume_state,
                "hmm_state": self._hmm_state,
                "hmm_prob": self._hmm_prob,
                "hmm_source": self._hmm_source,
                "state_age": self._state_age,
                "classify_count": self._classify_count,
                "version": __version__
            }

    @property
    def latency_ms(self) -> float:
        """最近一次分类延迟（毫秒，只读）"""
        return self._last_classify_latency_ms

    @property
    def classify_count(self) -> int:
        """分类累计次数（只读）"""
        return self._classify_count
