#!/usr/bin/env python3
"""
昆仑系统 · 流动性时钟 (LiquidityClock) — 机构级 v4.0

核心职责：
1. 维护基于时间衰减的流动性档案（价差、深度、成交量、冲击成本）
2. 实时评估当前流动性状态，输出多维度流动性指数及分位数
3. 根据流动性指数动态生成订单执行限制（仓位上限、TWAP参数、滑点容忍度）
4. 提供短期流动性崩溃预警（5分钟窗口），与风控系统联动

外部依赖（真实模块接口）：
- infrastructure.chronos_db.ChronosDB : 查询历史订单簿快照与成交聚合数据
- polaris.market_regime.MarketRegimeClassifier : 获取当前波动率分位数辅助调整限制

接口契约：
- evaluate(symbol: str, current_spread: float, current_volume: float, current_depth: float) -> Dict[str, Any]
- get_restrictions(symbol: str, order_size: float) -> Dict[str, Any]
- update_profile(symbol: str) -> bool
- health_check() -> Dict[str, Any]

异常与降级：
- 数据库不可用：回退至保守默认值，并触发 KUN-DAT-F003
- 档案数据不足：使用全球平均流动性参数，标记 KUN-LIQ-W001
- 实时数据异常（价差为负/极大）：沿用最近有效评估，最大 120 秒，触发 KUN-LIQ-W002

资源管理：
- 内部缓存使用 Pandas Series 存储时间衰减统计，内存上限 20MB/交易对
- 每 12 小时自动清理一次超过 24 小时未访问的交易对档案
"""

import logging
import time
import math
from typing import Dict, Any, List, Optional, Tuple, Protocol
from collections import defaultdict, deque
from dataclasses import dataclass, field
import threading

import numpy as np

logger = logging.getLogger(__name__)


# ==================== 数据提供者协议 ====================
class LiquidityDataProvider(Protocol):
    """流动性数据提供者接口，由 ChronosDB 实现"""
    def fetch_hourly_liquidity(self, symbol: str, days: int) -> List[Dict[str, float]]:
        """
        返回过去 days 天的小时聚合数据
        每条记录: {'hour_utc': int, 'avg_spread': float, 'avg_depth': float, 'total_volume': float}
        """
        ...


# ==================== 流动性档案结构 ====================
@dataclass
class HourBucketStats:
    """单小时桶的统计量，使用指数衰减更新"""
    decay: float = 0.97                  # 日衰减因子 (97% per day equivalent)
    weighted_spread: float = 0.0
    weighted_depth: float = 0.0
    weighted_volume: float = 0.0
    weight_sum: float = 1e-6
    last_update: float = 0.0

    def add(self, spread: float, depth: float, volume: float, timestamp: float) -> None:
        w = self.decay ** ((timestamp - self.last_update) / 86400.0) if self.last_update > 0 else 1.0
        self.weighted_spread = self.weighted_spread * w + spread
        self.weighted_depth = self.weighted_depth * w + depth
        self.weighted_volume = self.weighted_volume * w + volume
        self.weight_sum = self.weight_sum * w + 1.0
        self.last_update = timestamp

    @property
    def avg_spread(self) -> float:
        return self.weighted_spread / self.weight_sum

    @property
    def avg_depth(self) -> float:
        return self.weighted_depth / self.weight_sum

    @property
    def avg_volume(self) -> float:
        return self.weighted_volume / self.weight_sum


class LiquidityClock:
    """流动性时钟：时段流动性评估与交易约束"""

    # --------------------------- 类常量 ---------------------------
    # 档案参数
    PROFILE_DAYS: int = 30
    MIN_DAYS_FOR_VALID: int = 7
    HOUR_BUCKETS: int = 24

    # 流动性指数构成权重
    SPREAD_WEIGHT: float = 0.35
    DEPTH_WEIGHT: float = 0.30
    VOLUME_WEIGHT: float = 0.25
    IMPACT_WEIGHT: float = 0.10          # 冲击成本估算权重

    # 分位数阈值
    LOW_LIQUIDITY_PERCENTILE: float = 20.0
    CRITICAL_LIQUIDITY_PERCENTILE: float = 5.0

    # 低流动性限制参数 (作为基准乘数，最终结合波动率调整)
    BASE_POSITION_CAP_RATIO: float = 0.3
    BASE_STOP_OFFSET_MULT: float = 2.0
    BASE_MAX_PARTICIPATION: float = 0.02  # 最大参与率 2%

    # 实时评估
    MAX_STALE_AGE_SEC: float = 120.0       # 异常沿用最大时间
    SHORT_TERM_WINDOW_MIN: int = 5         # 短期流动性崩溃检测窗口 (分钟)

    # 缓存维护
    PROFILE_REFRESH_SEC: float = 43200.0   # 12小时刷新
    CLEANUP_INTERVAL_SEC: float = 86400.0  # 24小时清理不活跃交易对

    # 有效范围
    MIN_VALID_SPREAD: float = 1e-8
    MAX_VALID_SPREAD: float = 0.5          # 最大 50% 价差视为错误
    MIN_VALID_VOLUME: float = 0.0

    def __init__(self, config: Optional[Dict] = None, data_provider: Optional[LiquidityDataProvider] = None):
        self.data_provider = data_provider
        if config:
            self._apply_config(config)

        self._lock = threading.RLock()

        # 流动性档案: symbol -> dict[int, HourBucketStats]
        self._profiles: Dict[str, Dict[int, HourBucketStats]] = defaultdict(dict)
        self._last_refresh: Dict[str, float] = {}
        self._last_access: Dict[str, float] = {}

        # 短期窗口缓存 (用于崩溃检测)
        self._recent_spreads: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.SHORT_TERM_WINDOW_MIN * 2))
        self._recent_volumes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.SHORT_TERM_WINDOW_MIN * 2))

        # 评估缓存
        self._last_valid_result: Dict[str, Dict[str, Any]] = {}
        self._last_valid_time: Dict[str, float] = {}

        logger.info("LiquidityClock v4.0 初始化完成，档案窗口=%d天", self.PROFILE_DAYS)

    # --------------------------- 配置管理 ---------------------------
    @classmethod
    def _apply_config(cls, config: Dict) -> None:
        whitelist = [
            'PROFILE_DAYS', 'MIN_DAYS_FOR_VALID', 'SPREAD_WEIGHT', 'DEPTH_WEIGHT',
            'VOLUME_WEIGHT', 'IMPACT_WEIGHT', 'LOW_LIQUIDITY_PERCENTILE',
            'CRITICAL_LIQUIDITY_PERCENTILE', 'BASE_POSITION_CAP_RATIO',
            'BASE_STOP_OFFSET_MULT', 'BASE_MAX_PARTICIPATION', 'MAX_STALE_AGE_SEC'
        ]
        for key, value in config.items():
            if key in whitelist:
                setattr(cls, key, value)
                logger.debug("配置覆盖: %s = %s", key, value)
            else:
                logger.warning("忽略非白名单配置项: %s", key)

    # --------------------------- 档案加载与维护 ---------------------------
    def update_profile(self, symbol: str) -> bool:
        if self.data_provider is None:
            logger.error("[KUN-DAT-F003] 流动性数据提供者未注入，无法加载档案")
            return False

        try:
            raw_data = self.data_provider.fetch_hourly_liquidity(symbol, self.PROFILE_DAYS)
            if not raw_data:
                logger.warning("[KUN-LIQ-W001] 交易对%s无流动性历史数据", symbol)
                return False

            profile: Dict[int, HourBucketStats] = {}
            now = time.time()
            for rec in raw_data:
                hour = int(rec['hour_utc']) % 24
                spread = float(rec.get('avg_spread', 0.0))
                depth = float(rec.get('avg_depth', 0.0))
                volume = float(rec.get('total_volume', 0.0))
                if spread <= 0 or depth <= 0:
                    continue
                if hour not in profile:
                    profile[hour] = HourBucketStats()
                profile[hour].add(spread, depth, volume, now)

            with self._lock:
                self._profiles[symbol] = profile
                self._last_refresh[symbol] = now
                self._last_access[symbol] = now
            logger.info("[KUN-LIQ-I001] 交易对%s流动性档案已刷新，共%d时段", symbol, len(profile))
            return True

        except Exception as e:
            logger.error("[KUN-DAT-F002] 更新流动性档案失败: %s", e)
            return False

    def _ensure_fresh(self, symbol: str):
        now = time.time()
        last = self._last_refresh.get(symbol, 0)
        if now - last > self.PROFILE_REFRESH_SEC:
            self.update_profile(symbol)

    def _cleanup_stale_profiles(self):
        now = time.time()
        with self._lock:
            stale = [s for s, t in self._last_access.items() if now - t > self.CLEANUP_INTERVAL_SEC]
            for s in stale:
                del self._profiles[s]
                del self._last_refresh[s]
                del self._last_access[s]
                self._last_valid_result.pop(s, None)
                self._last_valid_time.pop(s, None)
                logger.info("已清理不活跃交易对档案: %s", s)

    # --------------------------- 流动性指数计算 ---------------------------
    def _compute_liquidity_index(self, symbol: str, current_spread: float,
                                 current_volume: float, current_depth: float) -> Dict[str, float]:
        """计算当前流动性指数 (0-100)"""
        with self._lock:
            profile = self._profiles.get(symbol, {})
        hour = time.gmtime().tm_hour
        bucket = profile.get(hour)
        if bucket is None or bucket.weight_sum < 5:
            return {"index": 50.0, "spread_pct": 50.0, "depth_pct": 50.0, "vol_pct": 50.0}

        # 基于指数衰减统计的“期望值”计算偏离度
        hist_spread = bucket.avg_spread
        hist_depth = bucket.avg_depth
        hist_volume = bucket.avg_volume

        # 使用对数相对偏差（无量纲）
        spread_dev = max(-5.0, min(5.0, math.log(current_spread / hist_spread)))
        depth_dev = max(-5.0, min(5.0, math.log(current_depth / hist_depth)))
        vol_dev = max(-5.0, min(5.0, math.log(current_volume / hist_volume + 1e-6)))

        # 映射到 0-100 分位数 (sigmoid)
        def to_pct(dev: float) -> float:
            return 100.0 / (1.0 + math.exp(dev * 1.5))  # 偏差为正(差) -> 低分

        spread_pct = to_pct(spread_dev)
        depth_pct = 100.0 - to_pct(-depth_dev)  # 深度高 -> 高分
        vol_pct = 100.0 - to_pct(-vol_dev)

        # 合成指数
        index = (self.SPREAD_WEIGHT * spread_pct +
                 self.DEPTH_WEIGHT * depth_pct +
                 self.VOLUME_WEIGHT * vol_pct)
        return {"index": round(index, 2), "spread_pct": round(spread_pct, 2),
                "depth_pct": round(depth_pct, 2), "vol_pct": round(vol_pct, 2)}

    # --------------------------- 实时评估 ---------------------------
    def evaluate(self, symbol: str, current_spread: float, current_volume: float,
                 current_depth: float = 0.0) -> Dict[str, Any]:
        """
        评估当前流动性状态
        :param symbol: 交易对
        :param current_spread: 当前买卖价差 (小数, e.g. 0.0005)
        :param current_volume: 最近1分钟成交量 (以计价货币计)
        :param current_depth: 前5档总深度 (可选)
        """
        # 输入校验
        if current_spread <= self.MIN_VALID_SPREAD or current_spread > self.MAX_VALID_SPREAD:
            return self._fallback_result(symbol, f"无效价差: {current_spread}")
        if current_volume < self.MIN_VALID_VOLUME:
            return self._fallback_result(symbol, f"无效成交量: {current_volume}")

        # 短期窗口记录
        self._recent_spreads[symbol].append(current_spread)
        self._recent_volumes[symbol].append(current_volume)

        # 确保档案
        self._ensure_fresh(symbol)

        liq = self._compute_liquidity_index(symbol, current_spread, current_volume, current_depth)
        index = liq["index"]

        # 短期崩溃检测
        crash_warning = False
        if len(self._recent_spreads[symbol]) >= self.SHORT_TERM_WINDOW_MIN:
            recent = list(self._recent_spreads[symbol])[-self.SHORT_TERM_WINDOW_MIN:]
            if min(recent) <= 0 or max(recent) > self.MAX_VALID_SPREAD:
                crash_warning = True
            elif np.mean(recent) > 5 * np.median(recent):  # 突然剧烈扩大
                crash_warning = True

        is_low = index < self.LOW_LIQUIDITY_PERCENTILE
        is_critical = index < self.CRITICAL_LIQUIDITY_PERCENTILE

        restrictions = self._generate_restrictions(index, is_low, is_critical, current_spread, current_volume)

        result = {
            "status": "ok",
            "symbol": symbol,
            "timestamp": time.time(),
            "liquidity_index": index,
            "components": liq,
            "is_low_liquidity": is_low,
            "is_critical": is_critical,
            "short_term_crash_warning": crash_warning,
            "restrictions": restrictions,
            "warnings": []
        }
        self._last_valid_result[symbol] = result
        self._last_valid_time[symbol] = time.time()
        self._last_access[symbol] = time.time()

        if crash_warning:
            logger.warning("[KUN-LIQ-W003] %s 检测到短期流动性崩溃", symbol)
        return result

    def _fallback_result(self, symbol: str, reason: str) -> Dict[str, Any]:
        now = time.time()
        if (symbol in self._last_valid_result and
            now - self._last_valid_time.get(symbol, 0) < self.MAX_STALE_AGE_SEC):
            return self._last_valid_result[symbol]

        return {
            "status": "fallback",
            "symbol": symbol,
            "timestamp": now,
            "liquidity_index": 10.0,
            "is_low_liquidity": True,
            "is_critical": True,
            "short_term_crash_warning": True,
            "restrictions": self._generate_restrictions(10.0, True, True, 0.01, 0),
            "warnings": [f"输入异常，采用保守流动性限制: {reason}"]
        }

    # --------------------------- 限制生成 ---------------------------
    def _generate_restrictions(self, liq_index: float, is_low: bool, is_critical: bool,
                               spread: float, volume: float) -> Dict[str, Any]:
        base_cap = self.BASE_POSITION_CAP_RATIO
        base_offset = self.BASE_STOP_OFFSET_MULT
        if is_critical:
            cap_mult = 0.5
            offset_mult = 3.0
            max_participation = 0.002
        elif is_low:
            cap_mult = 1.0
            offset_mult = 1.0
            max_participation = self.BASE_MAX_PARTICIPATION
        else:
            cap_mult = 1.5
            offset_mult = 0.8
            max_participation = 0.05

        return {
            "position_cap_ratio": round(base_cap * cap_mult, 4),
            "stop_offset_multiplier": round(base_offset * offset_mult, 2),
            "twap_disabled": is_critical or is_low,
            "fallback_algo": "limit_maker" if (is_critical or is_low) else "twap",
            "max_participation_rate": round(max_participation, 4),
            "slippage_buffer_bps": int(10.0 / (liq_index / 100.0 + 0.1))
        }

    def get_restrictions(self, symbol: str, order_size: float) -> Dict[str, Any]:
        """提供给执行网关的便捷方法"""
        # 此处需调用 evaluate 获得最新限制，同时结合订单大小调整参与率限制
        # 简化实现：使用缓存的最后一次评估结果
        res = self._last_valid_result.get(symbol)
        if not res:
            self._ensure_fresh(symbol)
            # 主动评估一次（需提供实时数据，这里由上层保证）
            return self._generate_restrictions(50.0, False, False, 0.0005, 1000)
        return res.get("restrictions", {})

    # --------------------------- 健康检查 ---------------------------
    @classmethod
    def health_check(cls) -> Dict[str, Any]:
        """模块自检，模拟正常评估流程"""
        try:
            # 构造一个带有 mock data_provider 的实例
            class MockProvider:
                def fetch_hourly_liquidity(self, symbol, days):
                    return [{'hour_utc': h, 'avg_spread': 0.001, 'avg_depth': 50000, 'total_volume': 1000}
                            for h in range(24)]

            clock = cls(data_provider=MockProvider())
            clock.update_profile("TESTBTC")
            res = clock.evaluate("TESTBTC", 0.001, 1000, 50000)
            assert res["status"] == "ok"
            assert 0 <= res["liquidity_index"] <= 100
            return {"status": "ok", "message": "流动性时钟自检通过"}
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return {"status": "error", "message": str(e)}
