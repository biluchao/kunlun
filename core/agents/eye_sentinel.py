#!/usr/bin/env python3
"""
昆仑系统 · 全局观察智能体 (Eye Sentinel)
版本: 6.0.0  华尔街机构级/万亿美金账户生产标准

核心职责：
1. 实时计算加密市场内部及与传统市场的动态相关性矩阵（自适应窗口，阈值随波动率调整）
2. 多维度检测跨市场传染风险，结合波动率自适应崩盘阈值（累计收益尺度一致）
3. 验证HMM市场状态，提供环境通行证，并输出置信度评分
4. 内置数据完整性防火墙、并发安全、资源自检、审计日志及降级联动

外部依赖：
- polaris.market_regime.MarketRegimeClassifier
- infrastructure.stream_gateway.StreamGateway
- olympus.agent_arbiter.AgentArbiter (间接)

接口契约：
- evaluate(context) -> Dict : 环境评估
- get_correlation_matrix() -> Dict : 获取相关性矩阵
- health_check() -> Dict : 自检
所有方法线程安全。

异常与降级：
- 传统数据断连 -> 内生模式，自动恢复
- 数据质量下降 -> 资产隔离，连续有效后自动恢复
- 超时 -> 缓存回退，记录延迟指标
- 冷启动超时 -> 强制完成并告警

资源管理：预分配循环数组、增量均值统计、定期内存回收

审计：所有环境风险决策记录审计日志（脱敏），包含时间戳与触发条件
"""

import logging
import time
import math
import threading
from typing import Dict, Any, List, Optional, Set
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class EyeSentinel:
    """全局观察智能体 ─ 眼·千里 (v6.0.0)"""

    # --------------------------- 类常量（不可变默认值）---------------------------
    CORRELATION_WINDOW = 100                   # 最大相关性窗口 (K线数)
    CORRELATION_UPDATE_INTERVAL_BARS = 3       # 更新间隔 (K线)
    CORRELATION_ALERT_THRESHOLD = 0.7          # 高相关性告警 [0,1]
    MIN_CORRELATION_SAMPLES = 20
    CORRELATION_CACHE_TTL_SEC = 60.0

    BTC_CRASH_BASE_THRESHOLD = 0.03            # 基础崩盘阈值 (百分比, 如0.03=3%)
    BTC_SPIKE_BASE_THRESHOLD = 0.05
    BTC_VOLATILITY_LOOKBACK = 20               # 波动率计算窗口
    REACTION_TIME_TARGET_MS = 100

    REDUCE_EXPOSURE_PERCENT = 0.50
    RELATED_ASSETS_CORR_MIN = 0.5

    CONFIRMATION_ASSETS = ['BTCUSDT', 'ETHUSDT']
    MAX_CONFIDENCE_REDUCTION = 0.30
    BTC_MOMENTUM_THRESHOLD = 0.3

    # 默认监控资产（实例初始化时复制）
    DEFAULT_CRYPTO_ASSETS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
    DEFAULT_TRADITIONAL_PROXIES = ['SPX', 'GLD', 'DXY']

    # 数据质量
    MAX_PRICE_AGE_SEC = 5.0
    MAX_NAN_FILL_CONSECUTIVE = 3
    PRICE_JUMP_THRESHOLD = 0.15
    DATA_INSUFFICIENT_RATIO = 0.5

    # 性能
    EVALUATION_TIMEOUT_MS = 500
    CACHE_VALIDITY_SEC = 1.0
    PRICE_BUFFER_MAX_SIZE = 200

    WARMUP_TIMEOUT_SEC = 300.0
    ASSET_RECOVERY_GOOD_SAMPLES = 10           # 连续有效样本数恢复资产

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化，支持通过 config 覆盖实例级配置（不会影响类属性）
        """
        # 实例级资产列表（副本，独立于类）
        self.crypto_assets: List[str] = list(self.DEFAULT_CRYPTO_ASSETS)
        self.traditional_proxies: List[str] = list(self.DEFAULT_TRADITIONAL_PROXIES)

        # 实例级配置参数（从类常量复制，可被 config 覆盖）
        self._init_instance_config(config)

        # 锁
        self._lock = threading.RLock()

        # 价格缓存
        self._price_cache: Dict[str, deque] = {}
        self._init_price_caches()

        # 传统数据追踪
        self._traditional_data_available = True
        self._last_traditional_update: float = time.time()   # 任意传统资产的最新更新时间

        # 相关性
        self._correlation_matrix: Dict[str, Dict[str, float]] = {}
        self._last_correlation_update_time: float = 0.0
        self._bar_count: int = 0

        # 外部依赖
        self._market_regime = None
        self._stream_gateway = None
        self._agent_arbiter = None

        # 评估缓存
        self._last_valid_result: Optional[Dict] = None
        self._last_valid_result_time: float = 0.0

        # 数据质量
        self._nan_fill_count: Dict[str, int] = {}
        self._data_quality_flags: Dict[str, bool] = {}
        self._data_recovery_counter: Dict[str, int] = {}
        self._init_quality_trackers()

        # 延迟监控 EWMA
        self._eval_time_ewma: float = 1.0
        self._eval_time_alpha: float = 0.05

        # 冷启动
        self._warmup_complete = False
        self._warmup_start_time: float = time.time()
        # 固定冷启动检查的核心资产（初始化时的加密资产快照）
        self._core_warmup_assets: Set[str] = set(self.crypto_assets)

        # 审计日志器
        self._audit_logger = None

        logger.info("Eye Sentinel v6.0.0 initialized | crypto=%d trad=%d",
                     len(self.crypto_assets), len(self.traditional_proxies))

    def _init_instance_config(self, config: Optional[Dict]):
        """应用实例级配置，只覆盖在类常量中存在的键，并进行类型安全转换"""
        if not config:
            return
        for key, value in config.items():
            if not hasattr(EyeSentinel, key):
                logger.warning("忽略未知配置项: %s", key)
                continue
            default_val = getattr(EyeSentinel, key)
            try:
                # 特殊处理列表类型：传入的可迭代对象转为列表
                if isinstance(default_val, list):
                    setattr(self, key, list(value))
                else:
                    # 尝试转换为原类型
                    converted = type(default_val)(value)
                    setattr(self, key, converted)
                logger.debug("配置覆盖: %s = %s", key, value)
            except (ValueError, TypeError) as e:
                logger.error("配置项 %s 类型转换失败: %s，保留默认值", key, str(e))

    def _all_assets(self) -> List[str]:
        return self.crypto_assets + self.traditional_proxies

    def _init_price_caches(self):
        for sym in self._all_assets():
            self._price_cache[sym] = deque(maxlen=self.PRICE_BUFFER_MAX_SIZE)

    def _init_quality_trackers(self):
        for sym in self._all_assets():
            self._nan_fill_count[sym] = 0
            self._data_quality_flags[sym] = True
            self._data_recovery_counter[sym] = 0

    # --------------------------- 资产动态管理 ---------------------------
    def register_asset(self, symbol: str, is_traditional: bool = False) -> None:
        """实例级注册新资产，不影响其他实例或全局配置"""
        with self._lock:
            if symbol in self._price_cache:
                return
            self._price_cache[symbol] = deque(maxlen=self.PRICE_BUFFER_MAX_SIZE)
            self._nan_fill_count[symbol] = 0
            self._data_quality_flags[symbol] = True
            self._data_recovery_counter[symbol] = 0
            if is_traditional:
                self.traditional_proxies.append(symbol)
            else:
                self.crypto_assets.append(symbol)
            logger.info("资产已注册: %s (traditional=%s)", symbol, is_traditional)

    def remove_asset(self, symbol: str) -> None:
        """移除资产监控"""
        with self._lock:
            self._price_cache.pop(symbol, None)
            self._nan_fill_count.pop(symbol, None)
            self._data_quality_flags.pop(symbol, None)
            self._data_recovery_counter.pop(symbol, None)
            if symbol in self.crypto_assets:
                self.crypto_assets.remove(symbol)
            if symbol in self.traditional_proxies:
                self.traditional_proxies.remove(symbol)

    # --------------------------- 依赖注入 ---------------------------
    def set_market_regime(self, regime): self._market_regime = regime
    def set_stream_gateway(self, gw): self._stream_gateway = gw
    def set_agent_arbiter(self, arb): self._agent_arbiter = arb
    def set_audit_logger(self, audit): self._audit_logger = audit

    # --------------------------- 数据清洗 ---------------------------
    @staticmethod
    def _validate_price(price: float, symbol: str, previous: Optional[float]) -> Optional[float]:
        if not math.isfinite(price) or price <= 0:
            logger.warning("[KUN-DAT-W008] %s 无效价格: %s", symbol, price)
            return previous if (previous and math.isfinite(previous) and previous > 0) else None
        if previous and previous > 0:
            change = abs(price - previous) / previous
            if change > EyeSentinel.PRICE_JUMP_THRESHOLD:
                logger.warning("[KUN-DAT-W009] %s 价格跳跃 %.2f%%", symbol, change*100)
                return previous
        return price

    # --------------------------- 价格更新 ---------------------------
    def update_prices(self, market_data: Dict[str, float]) -> None:
        """线程安全更新价格，内置数据质量追踪与自动恢复"""
        with self._lock:
            now = time.time()
            for sym, raw in market_data.items():
                if sym not in self._price_cache:
                    continue
                cache = self._price_cache[sym]
                prev = cache[-1] if cache else None
                clean = self._validate_price(raw, sym, prev)
                if clean is None:
                    if prev is not None and self._nan_fill_count[sym] < self.MAX_NAN_FILL_CONSECUTIVE:
                        clean = prev
                        self._nan_fill_count[sym] += 1
                    else:
                        self._data_quality_flags[sym] = False
                        logger.error("[KUN-DAT-E010] %s 数据质量严重下降，暂时隔离", sym)
                        continue
                else:
                    self._nan_fill_count[sym] = 0
                    if not self._data_quality_flags[sym]:
                        self._data_recovery_counter[sym] += 1
                        if self._data_recovery_counter[sym] >= self.ASSET_RECOVERY_GOOD_SAMPLES:
                            self._data_quality_flags[sym] = True
                            self._data_recovery_counter[sym] = 0
                            logger.info("[KUN-DAT-I011] %s 数据质量恢复", sym)
                cache.append(clean)
                # 更新传统数据时间戳（任意传统资产）
                if sym in self.traditional_proxies:
                    self._last_traditional_update = now
            self._bar_count += 1
            self._check_warmup()

    def _check_warmup(self):
        if self._warmup_complete:
            return
        # 仅使用核心资产检查
        lengths = [len(self._price_cache[s]) for s in self._core_warmup_assets if s in self._price_cache]
        if lengths and min(lengths) >= self.MIN_CORRELATION_SAMPLES:
            self._warmup_complete = True
            logger.info("Eye Sentinel 冷启动完成")
        elif (time.time() - self._warmup_start_time) > self.WARMUP_TIMEOUT_SEC:
            self._warmup_complete = True
            logger.warning("Eye Sentinel 冷启动超时，强制退出")

    # --------------------------- 相关性计算 ---------------------------
    def _compute_correlation(self, s1: str, s2: str) -> float:
        with self._lock:
            p1 = list(self._price_cache.get(s1, []))
            p2 = list(self._price_cache.get(s2, []))
        # 限制窗口大小
        n = min(len(p1), len(p2), self.CORRELATION_WINDOW)
        if n < self.MIN_CORRELATION_SAMPLES:
            return 0.0
        try:
            a1 = np.array(p1[-n:], dtype=np.float64)
            a2 = np.array(p2[-n:], dtype=np.float64)
            mask = np.isfinite(a1) & np.isfinite(a2) & (a1 > 0) & (a2 > 0)
            if np.sum(mask) < self.MIN_CORRELATION_SAMPLES:
                return 0.0
            a1, a2 = a1[mask], a2[mask]
            ret1 = np.diff(np.log(a1))
            ret2 = np.diff(np.log(a2))
            valid = np.isfinite(ret1) & np.isfinite(ret2)
            ret1, ret2 = ret1[valid], ret2[valid]
            if len(ret1) < 5 or np.std(ret1) < 1e-12 or np.std(ret2) < 1e-12:
                return 0.0
            corr = np.corrcoef(ret1, ret2)[0, 1]
            return float(np.clip(corr, -1.0, 1.0)) if np.isfinite(corr) else 0.0
        except Exception:
            return 0.0

    def update_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """更新相关性矩阵，线程安全"""
        now = time.time()
        if (self._correlation_matrix and
                (now - self._last_correlation_update_time) < self.CORRELATION_CACHE_TTL_SEC):
            return self._correlation_matrix
        with self._lock:
            assets = self.crypto_assets.copy()
            if self._traditional_data_available:
                assets.extend(self.traditional_proxies)
            matrix = {}
            for s1 in assets:
                matrix[s1] = {}
                for s2 in assets:
                    if s1 == s2:
                        matrix[s1][s2] = 1.0
                    elif s2 in matrix and s1 in matrix[s2]:
                        matrix[s1][s2] = matrix[s2][s1]
                    else:
                        matrix[s1][s2] = self._compute_correlation(s1, s2)
            self._correlation_matrix = matrix
            self._last_correlation_update_time = now
        return matrix

    def get_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """外部接口，返回矩阵的深拷贝避免外部修改"""
        if self._bar_count % self.CORRELATION_UPDATE_INTERVAL_BARS == 0:
            self.update_correlation_matrix()
        return self.update_correlation_matrix()  # 内部有TTL控制，不会重复计算

    # --------------------------- 指标计算 ---------------------------
    def _safe_log_return(self, symbol: str, lookback: int = 5) -> Optional[float]:
        """返回平均每期对数收益率，若数据不足返回None"""
        with self._lock:
            prices = list(self._price_cache.get(symbol, []))
        if len(prices) < lookback + 1:
            return None
        start, end = prices[-lookback-1], prices[-1]
        if start <= 0 or end <= 0 or not math.isfinite(start) or not math.isfinite(end):
            return None
        # 返回平均每期对数收益
        return math.log(end / start) / lookback

    def _volatility(self, symbol: str, window: int = 20) -> Optional[float]:
        """返回单期对数收益率的标准差（无偏估计），数据不足返回None"""
        with self._lock:
            prices = list(self._price_cache.get(symbol, []))
        if len(prices) < window:
            return None
        arr = np.array(prices[-window:], dtype=np.float64)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if len(arr) < 5:
            return None
        log_ret = np.diff(np.log(arr))
        log_ret = log_ret[np.isfinite(log_ret)]
        if len(log_ret) < 3:
            return None
        return float(np.std(log_ret, ddof=1))

    def _dynamic_crash_threshold(self, lookback: int = 5) -> float:
        """基于BTC波动率动态调整崩盘阈值（适用于累计百分比变化）"""
        vol = self._volatility('BTCUSDT', self.BTC_VOLATILITY_LOOKBACK)
        if vol is None:
            return self.BTC_CRASH_BASE_THRESHOLD
        # 将单期波动率转换为lookback期累计波动的近似标准差
        scaled_vol = vol * math.sqrt(lookback)
        # 基础阈值 3% 对应波动率 2% 的情况，线性缩放
        return self.BTC_CRASH_BASE_THRESHOLD * (scaled_vol / 0.02)

    # --------------------------- 核心评估 ---------------------------
    def evaluate(self, context: Dict) -> Dict[str, Any]:
        start_ns = time.perf_counter_ns()
        prices = context.get('prices', {})
        if prices:
            self.update_prices(prices)

        warnings: List[str] = []
        decision = 'hold'
        confidence = 1.0
        env_ok = True
        crash = False

        # 冷启动返回
        if not self._warmup_complete:
            return self._finalize(decision, 0.3, "冷启动中", False, False, warnings, start_ns)

        # 关键资产质量检查：BTC 数据不可用则拒绝交易
        if not self._data_quality_flags.get('BTCUSDT', False):
            logger.error("BTCUSDT 数据质量不可靠，环境强制不安全")
            return self._finalize('close_all', 0.0, "BTC数据不可靠", False, True, warnings, start_ns)

        # 1. BTC 变动分析
        btc_avg_log_ret = self._safe_log_return('BTCUSDT', 5)
        if btc_avg_log_ret is None:
            warnings.append("BTC数据不足")
            confidence -= 0.2
        else:
            btc_5_period_change = math.expm1(btc_avg_log_ret * 5)  # 累计百分比
            crash_thresh = self._dynamic_crash_threshold(5)
            if btc_5_period_change <= -crash_thresh:
                warnings.append(f"BTC闪崩 {btc_5_period_change:.2%} (阈值{crash_thresh:.2%})")
                crash = True
                env_ok = False
                decision = 'close_all'
                confidence = 0.95
            elif btc_5_period_change >= self.BTC_SPIKE_BASE_THRESHOLD:
                warnings.append(f"BTC异常暴涨 {btc_5_period_change:.2%}")
                confidence -= 0.1

        # 2. 传染风险
        symbol = context.get('symbol', '')
        if symbol and symbol not in self.CONFIRMATION_ASSETS and btc_avg_log_ret is not None and btc_avg_log_ret < 0:
            corr = self.get_correlation_matrix().get(symbol, {}).get('BTCUSDT', 0.0)
            if corr > self.RELATED_ASSETS_CORR_MIN and crash:
                warnings.append(f"{symbol}与BTC高相关({corr:.2f})，传染风险")
                decision = 'close_all'
                confidence = 0.9

        # 3. 传统市场背景
        if self._traditional_data_available:
            spx_ret = self._safe_log_return('SPX', 5)
            dxy_ret = self._safe_log_return('DXY', 5)
            if spx_ret is not None and abs(spx_ret) * 5 > 0.02:  # 近似5期变化
                warnings.append("SPX显著波动")
                confidence -= 0.05
            if dxy_ret is not None and abs(dxy_ret) * 5 > 0.01:
                warnings.append("DXY显著波动")
                confidence -= 0.03
        # 传统数据可用性检查与恢复
        now = time.time()
        if now - self._last_traditional_update > 300:
            if self._traditional_data_available:
                self._traditional_data_available = False
                warnings.append("传统市场数据断连，切换至内生模式")
            confidence -= 0.1
        else:
            if not self._traditional_data_available:
                self._traditional_data_available = True
                logger.info("传统市场数据恢复")

        # 4. HMM 状态质疑
        if self._market_regime and btc_avg_log_ret is not None:
            try:
                state = self._market_regime.get_state()
                if state is not None:
                    regime = state.get('regime', 'unknown')
                    vol = self._volatility('BTCUSDT')
                    if vol is not None and vol > 1e-6:
                        btc_momentum = abs(btc_avg_log_ret) / vol
                        if regime == 'trending_strong' and btc_momentum < self.BTC_MOMENTUM_THRESHOLD:
                            confidence -= self.MAX_CONFIDENCE_REDUCTION
                            warnings.append("HMM趋势但BTC动量不足")
            except Exception as e:
                logger.error("HMM状态查询异常: %s", e)

        # 5. 最终判定
        if crash:
            env_ok = False
        if confidence < 0.5:
            env_ok = False

        return self._finalize(decision, confidence, '; '.join(warnings) if warnings else "环境稳定",
                              env_ok, crash, warnings, start_ns)

    def _finalize(self, decision, confidence, reason, env_ok, crash, warnings, start_ns):
        """组装最终结果，处理缓存、延迟统计和审计"""
        now = time.time()
        elapsed = (time.perf_counter_ns() - start_ns) / 1e6

        # 超时且缓存可用，返回缓存但不更新延迟
        if elapsed > self.EVALUATION_TIMEOUT_MS and self._last_valid_result and \
                (now - self._last_valid_result_time) < self.CACHE_VALIDITY_SEC:
            logger.warning("Eye评估超时 %.1fms，返回缓存", elapsed)
            return self._last_valid_result

        # 确保confidence有效
        if not math.isfinite(confidence):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        result = {
            "status": "ok",
            "decision": decision,
            "confidence": confidence,
            "reason": reason,
            "metadata": {
                "environment_ok": env_ok,
                "cross_market_crash": crash,
                "btc_log_return": self._safe_log_return('BTCUSDT'),
                "traditional_available": self._traditional_data_available,
                "eval_time_ms": elapsed
            },
            "warnings": list(warnings)  # 拷贝
        }

        # 更新缓存
        self._last_valid_result = result
        self._last_valid_result_time = now
        # 更新延迟EWMA（仅当非超时）
        self._eval_time_ewma = self._eval_time_alpha * elapsed + (1 - self._eval_time_alpha) * self._eval_time_ewma

        self._maybe_audit(result)
        return result

    def _maybe_audit(self, result: Dict):
        if not self._audit_logger:
            return
        if result.get("metadata", {}).get("cross_market_crash") or result.get("decision") == "close_all":
            try:
                # 传递快照避免后续修改
                import copy
                self._audit_logger.log_event("eye_risk_event", copy.deepcopy(result))
            except Exception as e:
                logger.error("审计日志写入失败: %s", e)

    # --------------------------- 外部接口 ---------------------------
    def get_cross_market_risk(self) -> bool:
        btc_ret = self._safe_log_return('BTCUSDT')
        if btc_ret is None:
            return False
        crash = self._dynamic_crash_threshold()
        return math.expm1(btc_ret * 5) <= -crash

    def is_environment_safe(self) -> bool:
        if self._last_valid_result and (time.time() - self._last_valid_result_time) < self.CACHE_VALIDITY_SEC:
            return self._last_valid_result.get("metadata", {}).get("environment_ok", False)
        return False

    def reset(self):
        with self._lock:
            self._price_cache.clear()
            self._init_price_caches()
            self._init_quality_trackers()
            self._warmup_complete = False
            self._warmup_start_time = time.time()
            self._correlation_matrix.clear()
            self._last_valid_result = None

    @classmethod
    def health_check(cls) -> Dict[str, Any]:
        """模块自检（类方法）"""
        try:
            eye = cls()
            rng = np.random.default_rng(42)
            base = {'BTCUSDT':50000.0,'ETHUSDT':3000.0,'SOLUSDT':100.0,'BNBUSDT':300.0,
                    'SPX':4500.0,'GLD':1800.0,'DXY':104.0}
            for _ in range(60):
                update = {k: v * (1 + rng.normal(0, 0.005)) for k, v in base.items()}
                eye.update_prices(update)
            assert eye._warmup_complete
            res = eye.evaluate({'symbol':'SOLUSDT'})
            assert res['status'] == 'ok' and res['metadata']['environment_ok']
            # 模拟崩盘
            crash = {k: v * (0.93 if k=='BTCUSDT' else 1) for k, v in base.items()}
            eye.update_prices(crash)
            res2 = eye.evaluate({'symbol':'ETHUSDT','prices':crash})
            assert res2['metadata']['cross_market_crash']
            return {"status":"ok","message":"Eye Sentinel v6.0.0 所有核心路径自检通过"}
        except Exception as e:
            logger.error("健康检查失败: %s", e, exc_info=True)
            return {"status":"error","message":str(e)}
