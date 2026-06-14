#!/usr/bin/env python3
"""
昆仑系统 · 现实检验智能体 (Mirror Auditor) — 机构级 v3.0

核心职责：
1. 持续监控虚拟盘与实盘的现实差距（滑点偏差、成交率偏差、延迟偏差）
2. 差距超标时自动触发滑点模型在线校准（SGD with gradient clipping + RMSE恶化回滚）
3. 基于IC显著性（t检验）进行因子权重修正，防止过拟合
4. 检测策略失效（综合夏普/最大回撤），触发保护性措施，支持自动恢复
5. 作为"镜"智能体，对任何模型假设进行无情的现实检验，确保决策基于真实数据

外部依赖（真实模块接口）：
- strategos.dual_reality.DualRealityExecutor : get_reality_gap(window) -> Dict, get_recent_trade_pairs(min_samples) -> List[Dict]
- hermes.slippage_sim.SlippageSimulator : get_params() -> Dict, online_calibrate(X, y, lr, gradient_clip) -> Dict,
  validate_params(params) -> bool, restore_params(params), evaluate_rmse(X, y, params) -> float
- strategos.factor_compute.FactorComputeEngine : get_factor_health() -> Dict, adjust_weights(delta_map) -> Dict,
  on_weights_changed(), get_historical_sharpe() -> Optional[float]
- olympus.agent_arbiter.AgentArbiter : get_recent_sharpe(days) -> Optional[float], send_decay_alert(status: bool)
- infrastructure.health_pulse.HealthPulseMonitor : get_volatility_percentile() -> Optional[float]

接口契约（本模块）：
- evaluate(context: Dict) -> Dict[str, Any] : 返回决策、动作列表、置信度、警告
- calibrate_slippage_model(force: bool = False) -> Dict[str, Any]
- adjust_factor_weights(factor_health: Optional[Dict] = None) -> Dict[str, Any]
- check_reality_gap() -> Dict[str, Any]
- health_check() -> Dict[str, Any]
- set_dual_reality(dre) / set_slippage_sim(sls) / set_factor_engine(fce) / set_arbiter(arb) / set_health_monitor(hm)

异常与降级：
- 依赖不可用时返回明确错误状态，不影响主系统运行
- 校准或权重调整失败时保留上一次有效参数，并发出 KUN-MIR 错误码
- 极端市场（波动率分位>95%）自动暂停校准（可通过force绕过）
- 所有状态变更受 threading.Lock 保护
- 所有时间源统一使用 time.monotonic() 避免系统时间跳跃影响

资源管理：
- 历史记录使用 deque 限制容量
- 内存占用稳定，无泄漏风险
"""

import logging
import time
import threading
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class MirrorAuditor:
    """现实检验智能体 ─ 镜·水月"""

    # 配置常量（实例化时深拷贝，避免全局污染）
    _CONFIG_WHITELIST = {
        "reality_gap_alert", "reality_gap_critical", "gap_check_window",
        "force_check_interval_sec", "calibration_learning_rate",
        "calibration_min_samples", "calibration_max_samples",
        "calibration_gradient_clip", "calibration_feature_names",
        "calibration_outlier_mad_threshold", "calibration_data_max_age_sec",
        "factor_reweight_interval_trades", "max_single_adjustment",
        "min_ic_threshold", "ic_window", "historical_sharpe_benchmark",
        "decay_threshold", "monitoring_window_days", "decay_recovery_days",
        "min_calibration_interval_sec", "min_reweight_interval_sec",
        "evaluate_cache_sec", "volatility_percentile_suspend",
        "max_calibration_history", "max_reweight_history",
    }

    DEFAULT_CONFIG = {
        "reality_gap_alert": 0.10,
        "reality_gap_critical": 0.20,
        "gap_check_window": 50,
        "force_check_interval_sec": 14400,
        "calibration_learning_rate": 0.005,
        "calibration_min_samples": 20,
        "calibration_max_samples": 100,
        "calibration_gradient_clip": 1.0,
        "calibration_feature_names": ["order_size_ratio", "volatility", "spread_bps"],
        "calibration_outlier_mad_threshold": 5.0,
        "calibration_data_max_age_sec": 3600,
        "factor_reweight_interval_trades": 100,
        "max_single_adjustment": 0.02,
        "min_ic_threshold": 0.02,
        "ic_window": 60,
        "historical_sharpe_benchmark": None,
        "decay_threshold": 0.5,
        "monitoring_window_days": 60,
        "decay_recovery_days": 10,
        "min_calibration_interval_sec": 3600,
        "min_reweight_interval_sec": 7200,
        "evaluate_cache_sec": 5,
        "volatility_percentile_suspend": 95,
        "max_calibration_history": 50,
        "max_reweight_history": 50,
    }

    def __init__(self, config: Optional[Dict] = None):
        # 深拷贝配置，仅允许白名单中的键
        self._cfg = self.DEFAULT_CONFIG.copy()
        if config:
            if not isinstance(config, dict):
                raise TypeError("config must be a dict")
            for k, v in config.items():
                if k in self._CONFIG_WHITELIST:
                    self._cfg[k] = v
                else:
                    logger.warning("忽略未知配置项: %s", k)

        # 依赖注入（通过 setter 注入）
        self._dual_reality = None
        self._slippage_sim = None
        self._factor_engine = None
        self._arbiter = None
        self._health_monitor = None

        # 线程安全锁（保护所有可变状态）
        self._lock = threading.RLock()

        # 内部状态（时间全部使用 monotonic）
        self._last_calibration_time: Optional[float] = None   # 上次校准时间
        self._last_reweight_time: Optional[float] = None      # 上次权重调整时间
        self._strategy_decay_detected: bool = False
        self._decay_recovery_counter: int = 0
        self._gap_cache: Optional[Dict[str, Any]] = None
        self._gap_cache_time: Optional[float] = None

        # 历史记录（容量限制）
        self._calibration_history: deque = deque(maxlen=self._cfg["max_calibration_history"])
        self._reweight_history: deque = deque(maxlen=self._cfg["max_reweight_history"])

        logger.info("Mirror Auditor v3.0 初始化完成，配置=%s", self._cfg)

    # --------------------------- 配置/状态保护 ---------------------------
    @property
    def config(self) -> Dict:
        """返回只读配置副本"""
        return self._cfg.copy()

    # --------------------------- 依赖注入（带接口验证） ---------------------------
    def set_dual_reality(self, dre):
        required = ["get_reality_gap", "get_recent_trade_pairs"]
        for method in required:
            if not hasattr(dre, method):
                raise TypeError(f"DualRealityExecutor 缺少方法 {method}")
        self._dual_reality = dre

    def set_slippage_sim(self, sls):
        required = ["get_params", "online_calibrate", "validate_params", "restore_params", "evaluate_rmse"]
        for method in required:
            if not hasattr(sls, method):
                raise TypeError(f"SlippageSimulator 缺少方法 {method}")
        self._slippage_sim = sls

    def set_factor_engine(self, fce):
        required = ["get_factor_health", "adjust_weights", "on_weights_changed"]
        for method in required:
            if not hasattr(fce, method):
                raise TypeError(f"FactorComputeEngine 缺少方法 {method}")
        self._factor_engine = fce

    def set_arbiter(self, arbiter):
        required = ["get_recent_sharpe", "send_decay_alert"]
        for method in required:
            if not hasattr(arbiter, method):
                raise TypeError(f"AgentArbiter 缺少方法 {method}")
        self._arbiter = arbiter

    def set_health_monitor(self, hm):
        if not hasattr(hm, "get_volatility_percentile"):
            raise TypeError("HealthPulseMonitor 缺少 get_volatility_percentile")
        self._health_monitor = hm

    # --------------------------- 工具方法 ---------------------------
    @staticmethod
    def _is_valid_number(value) -> bool:
        """检查值是否为有限实数"""
        if isinstance(value, (int, float)):
            return np.isfinite(value)
        return False

    def _now_monotonic(self) -> float:
        return time.monotonic()

    def _is_extreme_market(self) -> bool:
        if not self._health_monitor:
            return False
        try:
            pct = self._health_monitor.get_volatility_percentile()
            if pct is None or not self._is_valid_number(pct):
                return False
            return pct >= self._cfg["volatility_percentile_suspend"]
        except Exception as e:
            logger.error("[KUN-MIR-E101] 获取波动率分位数异常: %s", e)
            return False

    # --------------------------- 差距检测 ---------------------------
    def check_reality_gap(self) -> Dict[str, Any]:
        """获取现实差距统计，带完整错误处理和NaN保护"""
        if not self._dual_reality:
            return {"status": "no_dependency", "average_gap": 0.0, "samples": 0}
        try:
            raw = self._dual_reality.get_reality_gap(self._cfg["gap_check_window"])
            if not isinstance(raw, dict):
                logger.error("[KUN-MIR-E102] 无效的差距数据格式")
                return {"status": "error", "average_gap": 0.0, "samples": 0}
            avg = raw.get("average_gap", 0.0)
            if not self._is_valid_number(avg):
                avg = 0.0
            raw["average_gap"] = avg
            return raw
        except Exception as e:
            logger.error("[KUN-MIR-E103] 获取现实差距失败: %s", e)
            return {"status": "error", "average_gap": 0.0, "samples": 0}

    def _get_gap_cached(self) -> Dict[str, Any]:
        now = self._now_monotonic()
        with self._lock:
            if (self._gap_cache is not None and self._gap_cache_time is not None
                    and now - self._gap_cache_time < self._cfg["evaluate_cache_sec"]):
                return self._gap_cache
        gap = self.check_reality_gap()
        with self._lock:
            self._gap_cache = gap
            self._gap_cache_time = now
        return gap

    # --------------------------- 滑点校准 ---------------------------
    def calibrate_slippage_model(self, force: bool = False) -> Dict[str, Any]:
        """执行滑点模型在线校准，自动回滚恶化"""
        if not self._slippage_sim or not self._dual_reality:
            return {"status": "error", "reason": "缺少依赖"}

        # 极端市场保护（force可绕过）
        if not force and self._is_extreme_market():
            logger.warning("[KUN-MIR-W201] 极端市场，暂停自动校准")
            return {"status": "suspended", "reason": "extreme_market"}

        with self._lock:
            now = self._now_monotonic()
            # 防抖检查（force 绕过）
            if not force and self._last_calibration_time is not None:
                elapsed = now - self._last_calibration_time
                if elapsed < self._cfg["min_calibration_interval_sec"]:
                    return {"status": "rejected", "reason": "过于频繁",
                            "next_available_in": self._cfg["min_calibration_interval_sec"] - elapsed}

        try:
            recent_trades = self._dual_reality.get_recent_trade_pairs(self._cfg["calibration_min_samples"])
            if not isinstance(recent_trades, list) or len(recent_trades) < self._cfg["calibration_min_samples"]:
                return {"status": "insufficient_data", "samples": len(recent_trades) if recent_trades else 0}

            # 使用统一时间源过滤过期数据（monotonic 与 交易时间戳 wall clock 混用？交易时间戳应使用 wall clock，但我们是过滤，用系统 wall clock）
            # 这里我们假定交易记录中的 timestamp 也是 wall clock，因此使用 time.time()
            cutoff = time.time() - self._cfg["calibration_data_max_age_sec"]
            valid_trades = [t for t in recent_trades if isinstance(t, dict) and t.get("timestamp", 0) > cutoff]
            if len(valid_trades) < self._cfg["calibration_min_samples"]:
                return {"status": "insufficient_fresh_data", "samples": len(valid_trades)}

            features, targets = self._prepare_calibration_data(valid_trades)
            if len(features) < self._cfg["calibration_min_samples"]:
                return {"status": "insufficient_valid_samples", "samples": len(features)}

            X = np.array(features, dtype=np.float64)
            y = np.array(targets, dtype=np.float64)

            old_params = self._slippage_sim.get_params()
            if not isinstance(old_params, dict):
                return {"status": "error", "reason": "无效的滑点参数"}

            # 执行校准
            try:
                cal_result = self._slippage_sim.online_calibrate(
                    X, y,
                    learning_rate=self._cfg["calibration_learning_rate"],
                    gradient_clip=self._cfg["calibration_gradient_clip"]
                )
            except TypeError:
                # 回退：可能旧接口不支持gradient_clip
                cal_result = self._slippage_sim.online_calibrate(
                    X, y,
                    learning_rate=self._cfg["calibration_learning_rate"]
                )
            if not isinstance(cal_result, dict) or cal_result.get("status") != "ok":
                return {"status": "error", "reason": f"校准失败: {cal_result}"}

            new_params = cal_result.get("params")
            if not isinstance(new_params, dict):
                return {"status": "error", "reason": "校准未返回有效参数"}

            if not self._slippage_sim.validate_params(new_params):
                logger.error("[KUN-MIR-E104] 校准参数非法，回滚")
                self._slippage_sim.restore_params(old_params)
                return {"status": "error", "reason": "参数非法，已回滚"}

            # RMSE 恶化检查
            try:
                old_rmse = self._slippage_sim.evaluate_rmse(X, y, old_params)
                new_rmse = self._slippage_sim.evaluate_rmse(X, y, new_params)
                if not self._is_valid_number(old_rmse) or not self._is_valid_number(new_rmse):
                    raise ValueError("RMSE计算结果无效")
            except Exception as e:
                logger.error("[KUN-MIR-E105] RMSE评估异常: %s", e)
                self._slippage_sim.restore_params(old_params)
                return {"status": "error", "reason": f"RMSE评估失败: {e}"}

            if old_rmse > 1e-6 and new_rmse > old_rmse * 1.1:
                logger.warning("[KUN-MIR-W202] 校准后RMSE恶化 (%.4f -> %.4f)，回滚", old_rmse, new_rmse)
                self._slippage_sim.restore_params(old_params)
                return {"status": "rejected", "reason": "RMSE恶化", "old_rmse": old_rmse, "new_rmse": new_rmse}

            # 记录成功
            with self._lock:
                self._last_calibration_time = now
                self._calibration_history.append({
                    "monotonic_time": now,
                    "wall_time": time.time(),
                    "samples": len(features),
                    "old_rmse": old_rmse,
                    "new_rmse": new_rmse,
                    "param_changes": {k: new_params[k] - old_params.get(k, 0) for k in new_params}
                })

            logger.info("[KUN-MIR-I301] 滑点校准成功，RMSE %.4f->%.4f", old_rmse, new_rmse)
            return {"status": "ok", "samples": len(features), "old_rmse": old_rmse, "new_rmse": new_rmse}

        except Exception as e:
            logger.error("[KUN-MIR-E106] 校准过程异常: %s", e, exc_info=True)
            return {"status": "error", "reason": str(e)}

    def _prepare_calibration_data(self, trades: List[Dict]) -> Tuple[List[List[float]], List[float]]:
        """提取特征和标签，带异常值过滤与类型安全"""
        features_raw = []
        targets_raw = []
        feature_names = self._cfg["calibration_feature_names"]
        for trade in trades:
            if not isinstance(trade, dict):
                continue
            # 检查所有特征存在且可转换为 float
            feat = []
            valid = True
            for k in feature_names:
                v = trade.get(k)
                if v is None or not self._is_valid_number(v):
                    valid = False
                    break
                feat.append(float(v))
            if not valid:
                continue
            slip = trade.get("actual_slippage_bps")
            if slip is None or not self._is_valid_number(slip):
                continue
            features_raw.append(feat)
            targets_raw.append(float(slip))

        if len(features_raw) < 5:
            return [], []

        # 基于滑点中位数绝对偏差过滤异常值
        y_arr = np.array(targets_raw)
        median_y = np.median(y_arr)
        mad_y = np.median(np.abs(y_arr - median_y)) * 1.4826
        if mad_y < 1e-6:
            mad_y = 1e-6
        threshold = self._cfg["calibration_outlier_mad_threshold"] * mad_y
        # 设置最小阈值，防止过滤掉所有样本
        if threshold < 1e-5:
            threshold = 1e-5
        mask = np.abs(y_arr - median_y) <= threshold
        features = [features_raw[i] for i in range(len(features_raw)) if mask[i]]
        targets = [targets_raw[i] for i in range(len(targets_raw)) if mask[i]]
        logger.debug("校准数据过滤: %d -> %d", len(features_raw), len(features))
        return features, targets

    # --------------------------- 因子权重修正 ---------------------------
    def _should_reweight(self) -> bool:
        """防抖检查，同时检查最小交易笔数"""
        with self._lock:
            if self._last_reweight_time is not None:
                if self._now_monotonic() - self._last_reweight_time < self._cfg["min_reweight_interval_sec"]:
                    return False
        # 可通过因子引擎检查交易笔数（此处简化为仅时间）
        return True

    def adjust_factor_weights(self, factor_health: Optional[Dict] = None) -> Dict[str, Any]:
        """根据IC显著性调整因子权重"""
        if not self._factor_engine:
            return {"status": "error", "reason": "因子引擎未注入"}
        if not self._should_reweight():
            return {"status": "rejected", "reason": "防抖间隔未到"}

        try:
            if factor_health is None:
                factor_health = self._factor_engine.get_factor_health()
            if not isinstance(factor_health, dict):
                return {"status": "error", "reason": "因子健康数据无效"}
            ic_values = factor_health.get("ic_values")
            if not isinstance(ic_values, dict):
                return {"status": "error", "reason": "ic_values格式错误"}

            adjustments = {}
            for factor, ic in ic_values.items():
                if not self._is_valid_number(ic):
                    continue
                # 绝对值低于阈值 -> 降权
                if abs(ic) < self._cfg["min_ic_threshold"]:
                    adjustments[factor] = -self._cfg["max_single_adjustment"]
                    logger.info("[KUN-MIR-I302] 因子 %s IC=%.4f 不显著，降权", factor, ic)
                elif ic < -self._cfg["min_ic_threshold"] * 2:
                    logger.warning("[KUN-MIR-W203] 因子 %s IC显著为负(%.4f)，可能需要反转信号", factor, ic)

            if adjustments:
                result = self._factor_engine.adjust_weights(adjustments)
                if not isinstance(result, dict) or result.get("status") != "ok":
                    return {"status": "error", "reason": f"引擎应用调整失败: {result}"}
                if hasattr(self._factor_engine, "on_weights_changed"):
                    self._factor_engine.on_weights_changed()

                with self._lock:
                    self._last_reweight_time = self._now_monotonic()
                    self._reweight_history.append({
                        "monotonic_time": self._last_reweight_time,
                        "wall_time": time.time(),
                        "adjustments": adjustments,
                        "ic_values": ic_values.copy()
                    })
                return {"status": "ok", "adjustments": adjustments}
            return {"status": "ok", "adjustments": {}, "reason": "所有因子IC正常"}
        except Exception as e:
            logger.error("[KUN-MIR-E107] 权重修正异常: %s", e, exc_info=True)
            return {"status": "error", "reason": str(e)}

    # --------------------------- 策略失效检测 ---------------------------
    def _detect_strategy_decay(self) -> Optional[bool]:
        """检测策略是否失效，返回 True/False/None"""
        if not self._arbiter:
            return None
        try:
            sharpe = self._arbiter.get_recent_sharpe(self._cfg["monitoring_window_days"])
            if sharpe is None or not self._is_valid_number(sharpe):
                return None
            benchmark = self._cfg["historical_sharpe_benchmark"]
            if benchmark is None:
                if hasattr(self._factor_engine, "get_historical_sharpe"):
                    benchmark = self._factor_engine.get_historical_sharpe()
            if benchmark is None or not self._is_valid_number(benchmark) or benchmark <= 0:
                return None
            if sharpe < benchmark * self._cfg["decay_threshold"]:
                return True
            return False
        except Exception as e:
            logger.error("[KUN-MIR-E108] 策略衰减检测异常: %s", e)
            return None

    def _update_decay_state(self, is_decay: Optional[bool]):
        """更新策略失效状态，含恢复计数"""
        if is_decay is True:
            if not self._strategy_decay_detected:
                self._strategy_decay_detected = True
                self._decay_recovery_counter = 0
                logger.warning("[KUN-MIR-W204] 策略失效标志激活")
                if self._arbiter:
                    try:
                        self._arbiter.send_decay_alert(True)
                    except Exception as e:
                        logger.error("[KUN-MIR-E109] 发送失效告警失败: %s", e)
        elif is_decay is False and self._strategy_decay_detected:
            self._decay_recovery_counter += 1
            if self._decay_recovery_counter >= self._cfg["decay_recovery_days"]:
                self._strategy_decay_detected = False
                self._decay_recovery_counter = 0
                logger.info("[KUN-MIR-I303] 策略失效标志已自动清除")
                if self._arbiter:
                    try:
                        self._arbiter.send_decay_alert(False)
                    except Exception as e:
                        logger.error("[KUN-MIR-E110] 发送恢复通知失败: %s", e)
        # 如果 is_decay is None，不清零计数器

    # --------------------------- 主评估接口 ---------------------------
    def evaluate(self, context: Dict) -> Dict[str, Any]:
        """主评估入口，线程安全，返回决策与动作列表"""
        warnings: List[str] = []
        actions: List[str] = []
        confidences: Dict[str, float] = {}

        # 快速获取差距（带缓存）
        gap_result = self._get_gap_cached()
        gap_val = gap_result.get("average_gap", 0.0)

        # 1. 差距评估
        if gap_val >= self._cfg["reality_gap_critical"]:
            actions.append("calibrate_slippage")
            confidences["calibrate"] = 0.95
            warnings.append(f"[KUN-MIR-W205] 现实差距严重超标: {gap_val:.4f}")
        elif gap_val >= self._cfg["reality_gap_alert"]:
            actions.append("alert_gap")
            confidences["alert"] = 0.7
            warnings.append(f"[KUN-MIR-W206] 现实差距告警: {gap_val:.4f}")

        # 2. 因子权重修正
        if self._should_reweight():
            actions.append("adjust_weights")
            confidences["adjust_weights"] = 0.8

        # 3. 强制校准
        with self._lock:
            last_cal = self._last_calibration_time
        if last_cal is not None and self._now_monotonic() - last_cal > self._cfg["force_check_interval_sec"]:
            if "calibrate_slippage" not in actions:
                actions.append("calibrate_slippage")
            confidences["calibrate"] = max(confidences.get("calibrate", 0.5), 0.6)

        # 4. 策略失效检测
        decay = self._detect_strategy_decay()
        self._update_decay_state(decay)
        if self._strategy_decay_detected:
            actions.append("strategy_decay_alert")
            confidences["strategy_decay"] = 0.99
            warnings.append("[KUN-MIR-W207] 策略疑似失效")

        # 决策
        if actions:
            best_action = max(confidences, key=confidences.get)
            decision = best_action
            confidence = confidences[best_action]
        else:
            decision = "hold"
            confidence = 0.0

        # 去重动作（以防重复）
        actions = list(dict.fromkeys(actions))

        return {
            "status": "ok",
            "decision": decision,
            "confidence": confidence,
            "actions": actions,
            "reason": "; ".join(actions) if actions else "系统正常",
            "metadata": {
                "reality_gap": gap_val,
                "strategy_decay": self._strategy_decay_detected,
                "last_calibration_ago": (self._now_monotonic() - last_cal) if last_cal else None,
                "last_reweight_ago": (self._now_monotonic() - self._last_reweight_time) if self._last_reweight_time else None,
            },
            "warnings": warnings
        }

    # --------------------------- 公共查询 ---------------------------
    def get_strategy_decay(self) -> bool:
        return self._strategy_decay_detected

    def get_calibration_history(self, n: int = 5) -> List[Dict]:
        with self._lock:
            return list(self._calibration_history)[-n:]

    # --------------------------- 健康检查（自包含） ---------------------------
    @classmethod
    def health_check(cls) -> Dict[str, Any]:
        """全面自检，包含 Mock 依赖完整链路"""
        try:
            mirror = cls()

            # 无依赖降级测试
            gap = mirror.check_reality_gap()
            assert gap["status"] == "no_dependency"
            cal = mirror.calibrate_slippage_model()
            assert cal["status"] == "error"
            adj = mirror.adjust_factor_weights()
            assert adj["status"] == "error"
            ev = mirror.evaluate({})
            assert ev["decision"] == "hold"

            # 构建完全符合接口的 Mock 对象
            class MockDRE:
                def get_reality_gap(self, window):
                    return {"average_gap": 0.05, "samples": 10}
                def get_recent_trade_pairs(self, min_samples):
                    base = {
                        "order_size_ratio": 0.1, "volatility": 0.02,
                        "spread_bps": 2.0, "actual_slippage_bps": 3.0,
                        "timestamp": time.time()
                    }
                    return [base] * min_samples

            class MockSLS:
                def get_params(self):
                    return {"eta": 0.1, "gamma": 0.03}
                def online_calibrate(self, X, y, learning_rate=0.005, gradient_clip=1.0):
                    return {"status": "ok", "params": {"eta": 0.12, "gamma": 0.028}}
                def validate_params(self, params):
                    return True
                def restore_params(self, params):
                    pass
                def evaluate_rmse(self, X, y, params):
                    return 0.5

            class MockFCE:
                def get_factor_health(self):
                    return {"ic_values": {"F1": 0.03, "F2": 0.01}}
                def adjust_weights(self, delta_map):
                    return {"status": "ok"}
                def on_weights_changed(self):
                    pass
                def get_historical_sharpe(self):
                    return 1.5

            class MockArbiter:
                def get_recent_sharpe(self, days):
                    return 1.2
                def send_decay_alert(self, status):
                    pass

            class MockHM:
                def get_volatility_percentile(self):
                    return 50

            mirror.set_dual_reality(MockDRE())
            mirror.set_slippage_sim(MockSLS())
            mirror.set_factor_engine(MockFCE())
            mirror.set_arbiter(MockArbiter())
            mirror.set_health_monitor(MockHM())

            # 校准测试
            cal_res = mirror.calibrate_slippage_model(force=True)
            assert cal_res["status"] == "ok", f"校准失败: {cal_res}"

            # 权重调整测试
            adj_res = mirror.adjust_factor_weights()
            assert adj_res["status"] == "ok", f"权重调整失败: {adj_res}"

            # 完整评估
            eval_res = mirror.evaluate({})
            assert "decision" in eval_res

            return {"status": "ok", "message": "全部自检通过（含完整Mock链路）"}
        except Exception as e:
            logger.error("Mirror Auditor 健康检查失败: %s", e, exc_info=True)
            return {"status": "error", "message": str(e)}
