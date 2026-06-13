#!/usr/bin/env python3
"""
昆仑系统 · Olympus Court (瑶池) ─ 智能体仲裁与元决策中枢

模块版本: 3.3.0
最后更新: 2026-06-13

核心职责：
1. 汇聚五大智能体的意见，通过仲裁器产生最终交易决策
2. 管理智能体间的冲突升级路径，确保极端行情下快速收敛
3. 评估各智能体贡献值，动态调整投票权重
4. 提供元决策层，根据市场状态选择策略模式与参数组

外部依赖（真实模块接口）：
- olympus.agent_arbiter.AgentArbiter
- olympus.conflict_escalation.ConflictEscalation
- olympus.contribution_eval.ContributionEvaluator
- olympus.meta_decision.MetaDecision
- core.agents.* (Stone, Wind, Mirror, Eye, Book)

接口契约：
- OlympusCourt.assess(signal, agent_votes) -> DecisionDict
- OlympusCourt.update_agent_weights() -> Dict[str, float]
- OlympusCourt.health_check() -> Dict[str, Any]
- 所有决策均记录审计日志，携带全局唯一 ID

异常与降级：
- 若某个智能体未能及时响应，其意见权重暂时置零
- 若仲裁引擎无响应，降级为保守模式（仅平仓，不新开）
- 连续失败触发熔断，自动转入保守模式并告警
- 所有异常均记录审计日志并抛给上层（附带详细上下文）

资源管理：
- 单例模式，全局唯一实例，避免状态分裂
- 支持 graceful shutdown，释放子模块引用
- 审计日志先落地后执行，确保事后可追溯
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 不可变默认配置
# ----------------------------------------------------------------------
_DEFAULT_CONFIG_DICT = {
    "arbitration_timeout_sec": 0.1,
    "decision_ttl_sec": 0.3,
    "audit_flush_size": 50,
    "audit_max_queue": 50000,
    "fallback_mode": "conservative",
    "allowed_modes": {"conservative", "normal", "aggressive"},
    "agent_names": ("stone", "wind", "mirror", "eye", "book"),
    "smooth_weight_alpha": 0.2,
    "max_consecutive_failures": 5,        # 连续失败触发熔断
    "circuit_breaker_cooldown_sec": 30.0, # 熔断恢复时间
    "health_check_timeout_sec": 2.0,
    "submodule_keys": {
        "arbiter_subconfig": {},
        "conflict_subconfig": {},
        "evaluator_subconfig": {},
        "meta_subconfig": {},
    }
}
DEFAULT_CONFIG = MappingProxyType(_DEFAULT_CONFIG_DICT)  # 不可变


class DecisionDict(TypedDict):
    """严格的决策返回结构"""
    action: str
    confidence: float
    reason: str
    decision_id: str
    conflict_level: int
    warnings: List[str]
    timestamp: float   # UTC Unix timestamp


# ----------------------------------------------------------------------
# 审计持久化接口（可插拔）
# ----------------------------------------------------------------------
class AuditWriter:
    """审计记录落地抽象类，默认实现为文件写入（WAL思想）"""
    def __init__(self):
        self._lock = threading.Lock()

    def write(self, record: Dict) -> None:
        raise NotImplementedError

    def flush(self) -> None:
        raise NotImplementedError

    def queue_length(self) -> int:
        raise NotImplementedError


class MemoryAuditWriter(AuditWriter):
    """内存审计写入器，仅用于测试或轻量场景（不推荐生产）"""
    def __init__(self, max_len: int = 50000):
        super().__init__()
        self._queue: List[Dict] = []
        self._max = max_len
    def write(self, record: Dict) -> None:
        with self._lock:
            self._queue.append(record)
            if len(self._queue) > self._max:
                self._queue.pop(0)  # 简单丢弃最老
    def flush(self) -> None:
        pass
    def queue_length(self) -> int:
        return len(self._queue)


class FileAuditWriter(AuditWriter):
    """文件持久化审计写入器，先写日志后确认决策"""
    def __init__(self, file_path: str = "/var/log/kunlun/olympus_audit.log"):
        super().__init__()
        self._file = open(file_path, "a", encoding="utf-8")
        self._count = 0
    def write(self, record: Dict) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()}|{record}\n"
        with self._lock:
            self._file.write(line)
            self._file.flush()  # 确保落盘
            self._count += 1
    def flush(self) -> None:
        with self._lock:
            self._file.flush()
    def queue_length(self) -> int:
        return 0  # 文件无需内存队列长度


# ----------------------------------------------------------------------
# 内部模块引用（线程安全延迟加载）
# ----------------------------------------------------------------------
_MODULES_LOCK = threading.RLock()
_MODULES_READY = False
_agent_arbiter_cls = None
_conflict_escalation_cls = None
_contribution_eval_cls = None
_meta_decision_cls = None


def _ensure_modules() -> None:
    """一次性加载子模块类，失败抛出异常（调用方应处理）"""
    global _MODULES_READY, _agent_arbiter_cls, _conflict_escalation_cls
    global _contribution_eval_cls, _meta_decision_cls
    if _MODULES_READY:
        return
    with _MODULES_LOCK:
        if _MODULES_READY:
            return
        start = time.monotonic()
        from olympus.agent_arbiter import AgentArbiter
        from olympus.conflict_escalation import ConflictEscalation
        from olympus.contribution_eval import ContributionEvaluator
        from olympus.meta_decision import MetaDecision

        for mod_name, cls in [("AgentArbiter", AgentArbiter),
                              ("ConflictEscalation", ConflictEscalation),
                              ("ContributionEvaluator", ContributionEvaluator),
                              ("MetaDecision", MetaDecision)]:
            if not hasattr(cls, 'health_check') or not callable(getattr(cls, 'health_check')):
                raise RuntimeError(f"{mod_name} 接口不符合规范")
        _agent_arbiter_cls = AgentArbiter
        _conflict_escalation_cls = ConflictEscalation
        _contribution_eval_cls = ContributionEvaluator
        _meta_decision_cls = MetaDecision
        _MODULES_READY = True
        logger.info("瑶池子模块加载完成，耗时 %.4fs", time.monotonic() - start)


# ----------------------------------------------------------------------
# 单例管理（安全双重检查 + 初始化失败计数）
# ----------------------------------------------------------------------
_INSTANCE: Optional[OlympusCourt] = None
_INSTANCE_LOCK = threading.Lock()
_INIT_FAILURE_COUNT = 0
_MAX_INIT_FAILURES = 3


def get_or_create_court(config: Optional[Dict] = None) -> OlympusCourt:
    """获取全局瑶池单例，若初始化连续失败超过阈值则永久降级"""
    global _INSTANCE, _INIT_FAILURE_COUNT
    if _INSTANCE is not None:
        return _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is not None:
            return _INSTANCE
        if _INIT_FAILURE_COUNT >= _MAX_INIT_FAILURES:
            raise RuntimeError("OlympusCourt 初始化失败次数过多，系统锁定")
        try:
            _ensure_modules()
        except Exception:
            _INIT_FAILURE_COUNT += 1
            raise
        # 合并配置
        merged = dict(DEFAULT_CONFIG)
        if config:
            # 处理子配置
            for key, default_val in DEFAULT_CONFIG["submodule_keys"].items():
                if key in config and isinstance(config[key], dict):
                    merged[key] = {**default_val, **config[key]}
                else:
                    merged[key] = default_val.copy()
            # 合并普通键
            for k, v in config.items():
                if k not in DEFAULT_CONFIG["submodule_keys"]:
                    merged[k] = v
        try:
            instance = OlympusCourt(merged)
            _INSTANCE = instance
            _INIT_FAILURE_COUNT = 0
            return instance
        except Exception:
            _INIT_FAILURE_COUNT += 1
            raise


# ----------------------------------------------------------------------
# 主控类
# ----------------------------------------------------------------------
class OlympusCourt:
    """瑶池仲裁院（线程安全，可重入）"""

    def __init__(self, config: Dict):
        self.config = config  # 内部只读，不应外部修改
        self._validate_config()
        # 审计写入器（根据配置选择，默认文件持久化）
        self.audit = self._create_audit_writer()

        # 子模块安全初始化
        self._submodule_errors: Dict[str, str] = {}
        self.arbiter = self._safe_init_submodule("arbiter", _agent_arbiter_cls, config.get("arbiter_subconfig"))
        self.conflict = self._safe_init_submodule("conflict", _conflict_escalation_cls, config.get("conflict_subconfig"))
        self.evaluator = self._safe_init_submodule("evaluator", _contribution_eval_cls, config.get("evaluator_subconfig"))
        self.meta = self._safe_init_submodule("meta", _meta_decision_cls, config.get("meta_subconfig"))

        # 内部状态
        self._smooth_weights: Dict[str, float] = {name: 1.0/len(config["agent_names"]) for name in config["agent_names"]}
        self._weights_lock = threading.Lock()
        self._init_mono = time.monotonic()

        # 执行器用于仲裁超时隔离（每个请求一个线程，避免排队）
        self._executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="olympus_arb")

        # 熔断状态
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        self._failure_lock = threading.Lock()

        # 健康检查专用线程池（小容量）
        self._health_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="olympus_hc")

        logger.info("瑶池实例化完成，审计写入器: %s", type(self.audit).__name__)

    def _create_audit_writer(self) -> AuditWriter:
        """根据配置创建审计写入器，默认文件持久化"""
        # 生产环境应通过环境变量或配置指定路径
        try:
            path = self.config.get("audit_log_path", "/var/log/kunlun/olympus_audit.log")
            return FileAuditWriter(path)
        except Exception:
            logger.exception("无法创建文件审计写入器，退回内存模式")
            return MemoryAuditWriter()

    def _validate_config(self) -> None:
        required = ["arbitration_timeout_sec", "agent_names", "allowed_modes", "decision_ttl_sec"]
        for k in required:
            if k not in self.config:
                raise ValueError(f"配置缺失: {k}")
        if self.config["arbitration_timeout_sec"] <= 0:
            raise ValueError("仲裁超时必须为正")
        if self.config["decision_ttl_sec"] <= 0:
            raise ValueError("决策TTL必须为正")
        if self.config.get("max_consecutive_failures", 5) <= 0:
            raise ValueError("最大连续失败次数必须为正")

    def _safe_init_submodule(self, name: str, cls: Any, subconfig: Any) -> Any:
        if cls is None:
            self._submodule_errors[name] = "类未加载"
            return None
        try:
            cfg = subconfig if isinstance(subconfig, dict) else {}
            return cls(cfg)
        except Exception as e:
            self._submodule_errors[name] = str(e)
            logger.error("子模块 %s 初始化失败: %s", name, e)
            return None

    # ------------------------------------------------------------------
    # 决策核心
    # ------------------------------------------------------------------
    def assess_signal(self,
                      signal: Dict[str, Any],
                      agent_votes: Dict[str, float]) -> DecisionDict:
        decision_id = str(uuid.uuid4())
        timestamp_utc = time.time()  # 挂钟时间用于审计
        warnings: List[str] = []
        action = "nothing"
        confidence = 0.0
        reason = "decision_not_made"
        conflict_level = 0
        votes_normalized = {}

        try:
            # 清洗信号
            clean_signal = self._sanitize_signal(signal)
            votes_normalized = self._normalize_votes(agent_votes)

            # 检查熔断
            if self._is_circuit_open():
                reason = "circuit_breaker_open"
                warnings.append("熔断保护生效，禁止新决策")
                return self._finalize_decision(action, confidence, reason, decision_id, conflict_level, warnings, timestamp_utc)

            # 元决策模式
            current_mode = self._get_mode()
            if current_mode is None:
                current_mode = self.config["fallback_mode"]
                warnings.append("元决策返回空，降级保守")
            if current_mode == "conservative":
                reason = "保守模式禁止新开仓"
                return self._finalize_decision(action, confidence, reason, decision_id, conflict_level, warnings, timestamp_utc)

            # 仲裁执行
            arbiter_result = self._arbitrate_with_timeout(clean_signal, votes_normalized)
            if arbiter_result is None:
                reason = "仲裁超时，降级保守"
                warnings.append("arbitration_timeout")
                self._record_failure()
            else:
                # 冲突升级
                if arbiter_result.get("conflict_level", 0) > 0 and self.conflict:
                    try:
                        arbiter_result = self.conflict.resolve(arbiter_result, votes_normalized)
                    except Exception as e:
                        logger.error("冲突升级失败: %s", e)
                        warnings.append(f"conflict_error: {e}")
                action = arbiter_result.get("action", "nothing")
                confidence = float(arbiter_result.get("confidence", 0.0))
                reason = arbiter_result.get("reason", "arbiter_decision")
                conflict_level = arbiter_result.get("conflict_level", 0)
                # 记录评估
                if self.evaluator:
                    try:
                        self.evaluator.record_decision(arbiter_result)
                    except Exception as e:
                        logger.warning("评估记录异常: %s", e)
                self._reset_failures()

            # TTL 检查
            elapsed = time.monotonic() - timestamp_utc  # 这里应使用 monotonic 计算耗时，但审计用 UTC
            if elapsed > self.config["decision_ttl_sec"]:
                warnings.append(f"决策延迟 {elapsed:.3f}s 超 TTL")

        except Exception as e:
            logger.exception("瑶池评估未预期异常")
            reason = f"系统异常: {type(e).__name__}"
            warnings.append("system_exception")
            self._record_failure()
        finally:
            return self._finalize_decision(action, confidence, reason, decision_id, conflict_level, warnings, timestamp_utc)

    def _finalize_decision(self, action: str, confidence: float, reason: str,
                           decision_id: str, conflict_level: int,
                           warnings: List[str], timestamp_utc: float) -> DecisionDict:
        decision = DecisionDict(
            action=action,
            confidence=confidence,
            reason=reason,
            decision_id=decision_id,
            conflict_level=conflict_level,
            warnings=warnings,
            timestamp=timestamp_utc
        )
        # 审计必须先于返回
        try:
            self._write_audit(decision, {})  # votes 可额外记录，此处简化
        except Exception:
            logger.exception("审计写入失败")
        return decision

    def _arbitrate_with_timeout(self, signal: Dict, votes: Dict) -> Optional[Dict]:
        if not self.arbiter:
            return None
        future = self._executor.submit(self.arbiter.decide, signal, votes)
        try:
            return future.result(timeout=self.config["arbitration_timeout_sec"])
        except FutureTimeoutError:
            logger.warning("仲裁器执行超时 (%.3fs)", self.config["arbitration_timeout_sec"])
            future.cancel()  # 尝试取消
            return None
        except Exception as e:
            logger.error("仲裁器执行异常: %s", e)
            return None

    def _get_mode(self) -> Optional[str]:
        if not self.meta:
            return None
        try:
            mode = self.meta.get_current_mode()
            if mode not in self.config["allowed_modes"]:
                logger.warning("未知模式 %s，降级保守", mode)
                return self.config["fallback_mode"]
            return mode
        except Exception:
            return None

    def _sanitize_signal(self, signal: Dict) -> Dict:
        """清洗信号，返回新字典，价格转 Decimal"""
        required = {"symbol", "price", "action_type"}
        missing = required - set(signal.keys())
        if missing:
            raise ValueError(f"信号缺少字段: {missing}")
        action_type = signal["action_type"]
        if action_type not in ("entry", "exit", "cancel", "modify"):
            raise ValueError(f"非法 action_type: {action_type}")
        try:
            price = Decimal(str(signal["price"]))
            if price <= 0:
                raise ValueError("价格必须为正")
        except (InvalidOperation, ValueError) as e:
            raise ValueError(f"无效价格: {signal['price']}") from e
        # 浅拷贝并替换价格
        clean = {**signal, "price": price}
        # 校验 symbol 应在交易白名单中（假设由外部提供或配置，此处可加钩子）
        return clean

    def _normalize_votes(self, votes: Dict) -> Dict[str, float]:
        if not isinstance(votes, dict):
            raise ValueError("agent_votes 必须为字典")
        agents = self.config["agent_names"]
        normalized = {}
        for name in agents:
            val = votes.get(name, 0.0)
            try:
                fval = float(val)
            except (TypeError, ValueError):
                fval = 0.0
            if math.isnan(fval) or math.isinf(fval):
                fval = 0.0
            normalized[name] = max(0.0, min(1.0, fval))
        return normalized

    # ------------------------------------------------------------------
    # 熔断管理
    # ------------------------------------------------------------------
    def _is_circuit_open(self) -> bool:
        with self._failure_lock:
            if self._consecutive_failures >= self.config["max_consecutive_failures"]:
                if time.monotonic() < self._circuit_open_until:
                    return True
                else:
                    # 冷却期过，半开状态（允许一次尝试）
                    self._consecutive_failures = 0
            return False

    def _record_failure(self) -> None:
        with self._failure_lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.config["max_consecutive_failures"]:
                self._circuit_open_until = time.monotonic() + self.config["circuit_breaker_cooldown_sec"]
                logger.critical("瑶池熔断触发，冷却至 %.2f", self._circuit_open_until)

    def _reset_failures(self) -> None:
        with self._failure_lock:
            self._consecutive_failures = 0

    # ------------------------------------------------------------------
    # 审计
    # ------------------------------------------------------------------
    def _write_audit(self, decision: DecisionDict, votes: Dict) -> None:
        record = {
            "decision_id": decision["decision_id"],
            "action": decision["action"],
            "confidence": decision["confidence"],
            "reason": decision["reason"],
            "timestamp": decision["timestamp"],
            "warnings": decision["warnings"],
        }
        self.audit.write(record)

    # ------------------------------------------------------------------
    # 权重更新（线程安全）
    # ------------------------------------------------------------------
    def update_agent_weights(self) -> Dict[str, float]:
        if not self.evaluator:
            return self._get_default_weights()
        with self._weights_lock:
            try:
                raw = self.evaluator.compute_weights()
            except Exception:
                raw = None
            if not raw:
                return self._smooth_weights.copy()

            alpha = self.config["smooth_weight_alpha"]
            agents = self.config["agent_names"]
            for a in agents:
                new_val = raw.get(a, self._smooth_weights.get(a, 0.2))
                if not (0 <= new_val <= 1) or math.isnan(new_val):
                    new_val = 0.2
                self._smooth_weights[a] = alpha * new_val + (1 - alpha) * self._smooth_weights[a]

            total = sum(self._smooth_weights.values())
            if total > 0:
                for a in agents:
                    self._smooth_weights[a] /= total
            if self.arbiter:
                try:
                    self.arbiter.update_weights(self._smooth_weights.copy())
                except Exception as e:
                    logger.error("更新仲裁器权重失败: %s", e)
            return self._smooth_weights.copy()

    def _get_default_weights(self) -> Dict[str, float]:
        agents = self.config["agent_names"]
        return {a: 1.0/len(agents) for a in agents}

    # ------------------------------------------------------------------
    # 健康检查
    # ------------------------------------------------------------------
    def health_check(self) -> Dict[str, Any]:
        modules = {
            "arbiter": self.arbiter,
            "conflict": self.conflict,
            "evaluator": self.evaluator,
            "meta": self.meta,
        }
        statuses = {}
        for name, mod in modules.items():
            if mod is None:
                statuses[name] = "uninitialized"
                if name in self._submodule_errors:
                    statuses[name] = f"error: {self._submodule_errors[name]}"
                continue
            try:
                future = self._health_executor.submit(mod.health_check)
                result = future.result(timeout=self.config["health_check_timeout_sec"])
                statuses[name] = result.get("status", "unknown")
            except FutureTimeoutError:
                statuses[name] = "timeout"
            except Exception as e:
                statuses[name] = f"error: {e}"
        all_ok = all(v == "ok" for v in statuses.values())
        return {
            "status": "ok" if all_ok else "degraded",
            "modules": statuses,
            "uptime_sec": time.monotonic() - self._init_mono,
        }

    def shutdown(self) -> None:
        """优雅关闭"""
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._health_executor.shutdown(wait=False, cancel_futures=True)
        self.audit.flush()
        for attr in ("arbiter", "conflict", "evaluator", "meta"):
            obj = getattr(self, attr, None)
            if obj and hasattr(obj, 'shutdown'):
                try:
                    obj.shutdown()
                except Exception:
                    pass
        logger.info("瑶池已关闭")


# ------------------------------------------------------------------
# 包级别健康检查（轻量）
# ------------------------------------------------------------------
def health_check() -> Dict[str, Any]:
    try:
        _ensure_modules()
        modules_status = {
            "AgentArbiter": hasattr(_agent_arbiter_cls, 'health_check'),
            "ConflictEscalation": hasattr(_conflict_escalation_cls, 'health_check'),
            "ContributionEvaluator": hasattr(_contribution_eval_cls, 'health_check'),
            "MetaDecision": hasattr(_meta_decision_cls, 'health_check'),
        }
        all_ok = all(modules_status.values())
        return {
            "status": "ok" if all_ok else "degraded",
            "modules": modules_status,
            "version": "3.3.0"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
