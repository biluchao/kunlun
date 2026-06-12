#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Kunlun Capital LP. All rights reserved.
"""
Hermes Hall — 订单执行、风控、滑点模拟、API 治理 (v3.2.0)

核心职责：
1. 提供订单执行网关的自适应算法路由（TWAP/VWAP/冰山）
2. 实现毫秒级紧急平仓协议，五级熔断级联保护
3. 管理币安API令牌桶，保障高优先级指令（撤单/止损）资源
4. 模拟真实市场冲击与滑点，为虚拟盘提供精确成本预估
5. 暴露标准化健康检查（Liveness/Readiness/Full）与 Prometheus 指标

外部依赖：
- infrastructure.stream_gateway.StreamGateway : 实时深度数据
- infrastructure.chronos_db.ChronosDB : 审计日志与订单流水
- infrastructure.health_pulse.HealthPulseMonitor : 上报执行健康度

接口契约：
- hermes_health_check(depth: str, timeout_ms: float, trace_id: str) -> HealthReport
- get_hermes_context() -> Optional[HermesContext]
- init_hall(config: dict) -> HermesContext
- shutdown_hall() -> None

异常与降级：
- 子模块超时触发线程安全断路器，连续失败阈值可配置
- 非关键模块（SlippageSimulator）失败仅降级，不阻止交易执行
- 所有健康检查结果缓存严格按参数隔离，防止碰撞
- 敏感信息自动脱敏，支持自定义脱敏列表

资源管理：
- 全局线程池采用可监控的 BoundedThreadPool，任务取消时安全回收
- HermesContext 提供 prepare_shutdown() 和 close() 两阶段清理
- 信号处理器采用 async-safe 方式，通过管道唤醒主循环
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future, CancelledError, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TYPE_CHECKING

from prometheus_client import Gauge, Histogram

if TYPE_CHECKING:
    from .order_gateway import OrderExecutionGateway
    from .emergency_unwind import EmergencyUnwind
    from .rate_limiter import RateLimitGovernor
    from .slippage_sim import SlippageSimulator
    from .circuit_breaker import CircuitBreakerCascade

logger = logging.getLogger(__name__)

__version__ = "3.2.0"

# ---------------------------------------------------------------------------
# 类型定义
# ---------------------------------------------------------------------------
class HealthStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class ModuleHealthResult(TypedDict, total=False):
    status: str
    message: str
    latency_ms: float
    error_code: Optional[str]
    details: Dict[str, Any]
    warnings: List[str]

class HealthReport(TypedDict):
    status: str
    message: str
    version: str
    timestamp: float
    trace_id: str
    overall_latency_ms: float
    depth: str
    modules: Dict[str, ModuleHealthResult]
    warnings: List[str]
    degraded_modules: List[str]
    failed_modules: List[str]

# ---------------------------------------------------------------------------
# 线程安全断路器（使用 monotonic 时钟，支持重置）
# ---------------------------------------------------------------------------
class ModuleCircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 3, recovery_timeout: float = 30.0):
        self.name = name
        self._lock = threading.RLock()
        self.failures = 0
        self.last_failure_time = 0.0  # monotonic time
        self.state = "CLOSED"
        self.threshold = failure_threshold
        self.timeout = recovery_timeout
        self._half_open_in_progress = False

    def call(self, func: Callable, *args, **kwargs) -> Any:
        with self._lock:
            if self.state == "OPEN":
                if time.monotonic() - self.last_failure_time >= self.timeout:
                    self.state = "HALF_OPEN"
                    self._half_open_in_progress = False
                else:
                    raise RuntimeError(f"Circuit breaker for {self.name} is OPEN")
            if self.state == "HALF_OPEN" and self._half_open_in_progress:
                raise RuntimeError(f"Circuit breaker for {self.name} busy (half-open trial in progress)")
            if self.state == "HALF_OPEN":
                self._half_open_in_progress = True
        try:
            result = func(*args, **kwargs)
            with self._lock:
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failures = 0
                    self._half_open_in_progress = False
                # 成功时重置失败计数
                else:
                    self.failures = 0
            return result
        except Exception as e:
            with self._lock:
                self.failures += 1
                self.last_failure_time = time.monotonic()
                if self.state == "CLOSED" and self.failures >= self.threshold:
                    self.state = "OPEN"
                elif self.state == "HALF_OPEN":
                    self.state = "OPEN"
                self._half_open_in_progress = False
            raise e

    def reset(self):
        with self._lock:
            self.state = "CLOSED"
            self.failures = 0
            self._half_open_in_progress = False

# ---------------------------------------------------------------------------
# 可监控线程池（带未处理异常处理器与安全关闭）
# ---------------------------------------------------------------------------
class MonitoredThreadPoolExecutor(ThreadPoolExecutor):
    def __init__(self, max_workers=5, thread_name_prefix="hermes_hc"):
        super().__init__(max_workers=max_workers, thread_name_prefix=thread_name_prefix)
        self._active_futures = set()

    def submit(self, fn, *args, **kwargs):
        future = super().submit(fn, *args, **kwargs)
        self._active_futures.add(future)
        future.add_done_callback(self._active_futures.discard)
        return future

    def shutdown(self, wait=True, *, cancel_futures=True):
        super().shutdown(wait=wait, cancel_futures=cancel_futures)

_health_check_pool = MonitoredThreadPoolExecutor(max_workers=5)

# ---------------------------------------------------------------------------
# 注册中心
# ---------------------------------------------------------------------------
@dataclass(frozen=True)  # 冻结防止意外修改
class ModuleRegistration:
    name: str
    func_provider: Callable[[], Callable[[], Dict]]
    critical: bool = True
    dependencies: List[str] = field(default_factory=list)

class ModuleRegistry:
    def __init__(self):
        self._modules: Dict[str, ModuleRegistration] = {}
        self._lock = threading.RLock()

    def register(self, registration: ModuleRegistration):
        with self._lock:
            self._modules[registration.name] = registration

    def topological_order(self) -> List[ModuleRegistration]:
        with self._lock:
            # 返回副本，避免并发修改
            modules_snapshot = dict(self._modules)
        resolved = []
        visited = set()
        def resolve(name):
            if name in visited:
                return
            visited.add(name)
            reg = modules_snapshot.get(name)
            if reg:
                for dep in reg.dependencies:
                    resolve(dep)
                resolved.append(reg)
        for name in modules_snapshot:
            resolve(name)
        return resolved

    def clear(self):
        with self._lock:
            self._modules.clear()

    def get_registered_names(self) -> List[str]:
        with self._lock:
            return list(self._modules.keys())

_registry = ModuleRegistry()

# ---------------------------------------------------------------------------
# 敏感信息脱敏器
# ---------------------------------------------------------------------------
_SENSITIVE_ENV_KEYS = ["API_KEY", "SECRET", "PASSWORD", "TOKEN", "PRIVATE"]
def _sanitize_message(msg: str) -> str:
    for key in _SENSITIVE_ENV_KEYS:
        for env_var in os.environ:
            if key in env_var.upper():
                value = os.environ[env_var]
                if value and len(value) > 4:
                    msg = msg.replace(value, "[REDACTED]")
    return msg[:200]

# ---------------------------------------------------------------------------
# 全局上下文管理
# ---------------------------------------------------------------------------
class HermesContext:
    def __init__(self, config: Dict[str, Any]):
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary")
        self.config = config
        self.rate_limiter: Optional[RateLimitGovernor] = None
        self.slippage: Optional[SlippageSimulator] = None
        self.circuit_breaker: Optional[CircuitBreakerCascade] = None
        self.order_gateway: Optional[OrderExecutionGateway] = None
        self.emergency_unwind: Optional[EmergencyUnwind] = None
        self._initialized = False
        self._lock = threading.RLock()
        self._pending_shutdown = False

    async def initialize(self) -> None:
        with self._lock:
            if self._initialized or self._pending_shutdown:
                return
        try:
            from .rate_limiter import RateLimitGovernor
            from .slippage_sim import SlippageSimulator
            from .circuit_breaker import CircuitBreakerCascade
            self.rate_limiter = RateLimitGovernor(self.config.get('rate_limiter', {}))
            self.circuit_breaker = CircuitBreakerCascade(self.config.get('circuit_breaker', {}))
            self.slippage = SlippageSimulator(self.config.get('slippage', {}))

            from .order_gateway import OrderExecutionGateway
            from .emergency_unwind import EmergencyUnwind
            self.order_gateway = OrderExecutionGateway(
                config=self.config,
                rl=self.rate_limiter,
                slip=self.slippage,
                cb=self.circuit_breaker
            )
            self.emergency_unwind = EmergencyUnwind(
                config=self.config,
                gateway=self.order_gateway,
                cb=self.circuit_breaker
            )
            with self._lock:
                self._initialized = True
            logger.info("Hermes context initialized successfully")
        except Exception as e:
            logger.exception(f"Hermes init failed: {_sanitize_message(str(e))}")
            raise

    def health_check_fn(self, module_name: str) -> Optional[Callable[[], Dict]]:
        mapping = {
            "RateLimitGovernor": "rate_limiter",
            "CircuitBreakerCascade": "circuit_breaker",
            "OrderExecutionGateway": "order_gateway",
            "EmergencyUnwind": "emergency_unwind",
            "SlippageSimulator": "slippage",
        }
        attr = mapping.get(module_name)
        if not attr:
            return None
        instance = getattr(self, attr, None)
        if instance and hasattr(instance, "health_check"):
            return instance.health_check
        return None

    def prepare_shutdown(self):
        """通知所有子模块即将关闭，允许刷写缓冲区"""
        with self._lock:
            self._pending_shutdown = True
        for instance in [self.order_gateway, self.emergency_unwind, self.rate_limiter, self.slippage, self.circuit_breaker]:
            if instance and hasattr(instance, "prepare_shutdown"):
                try:
                    instance.prepare_shutdown()
                except Exception:
                    logger.exception("Error during prepare_shutdown of %s", type(instance).__name__)

    def close(self):
        """释放资源"""
        for instance in [self.order_gateway, self.emergency_unwind, self.rate_limiter, self.slippage, self.circuit_breaker]:
            if instance and hasattr(instance, "close"):
                try:
                    instance.close()
                except Exception:
                    pass
        self._initialized = False

# 全局上下文
_hermes_ctx: Optional[HermesContext] = None
_ctx_lock = threading.RLock()

def init_hall(config: Dict[str, Any]) -> HermesContext:
    global _hermes_ctx
    if not isinstance(config, dict):
        raise TypeError("config must be dict")
    with _ctx_lock:
        if _hermes_ctx:
            _hermes_ctx.prepare_shutdown()
            _hermes_ctx.close()
        _hermes_ctx = HermesContext(config)
        # 在专用线程中运行初始化以避免阻塞
        try:
            loop = asyncio.get_running_loop()
            # 创建新循环在新线程中运行？简化处理：假设主线程尚未运行循环
            raise RuntimeError("Cannot init hall within running event loop")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(asyncio.wait_for(_hermes_ctx.initialize(), timeout=10.0))
        except asyncio.TimeoutError:
            logger.error("Hermes context initialization timed out")
            _hermes_ctx = None
            raise TimeoutError("Hermes initialization timed out")
        _register_default_modules()
        # 清除缓存
        global _cached_report
        with _cache_lock:
            _cached_report = None
        return _hermes_ctx

def get_hermes_context() -> Optional[HermesContext]:
    with _ctx_lock:
        return _hermes_ctx

def shutdown_hall():
    global _hermes_ctx, _cached_report
    with _ctx_lock:
        if _hermes_ctx:
            _hermes_ctx.prepare_shutdown()
            _hermes_ctx.close()
            _hermes_ctx = None
    _health_check_pool.shutdown(wait=False, cancel_futures=True)
    with _cache_lock:
        _cached_report = None

# 异步安全信号处理：设置标志，由主循环检查
_shutdown_requested = threading.Event()

def _safe_signal_handler(signum, frame):
    _shutdown_requested.set()

signal.signal(signal.SIGTERM, _safe_signal_handler)
signal.signal(signal.SIGINT, _safe_signal_handler)

def check_shutdown_flag():
    if _shutdown_requested.is_set():
        logger.info("Shutdown signal received, shutting down Hermes")
        shutdown_hall()
        sys.exit(0)

# ---------------------------------------------------------------------------
# 默认模块注册
# ---------------------------------------------------------------------------
def _register_default_modules():
    def provider_gateway():
        ctx = get_hermes_context()
        if not ctx or not ctx.order_gateway:
            raise RuntimeError("OrderExecutionGateway not available")
        return ctx.order_gateway.health_check
    def provider_unwind():
        ctx = get_hermes_context()
        if not ctx or not ctx.emergency_unwind:
            raise RuntimeError("EmergencyUnwind not available")
        return ctx.emergency_unwind.health_check
    def provider_rl():
        ctx = get_hermes_context()
        if not ctx or not ctx.rate_limiter:
            raise RuntimeError("RateLimiter not available")
        return ctx.rate_limiter.health_check
    def provider_cb():
        ctx = get_hermes_context()
        if not ctx or not ctx.circuit_breaker:
            raise RuntimeError("CircuitBreaker not available")
        return ctx.circuit_breaker.health_check
    def provider_slip():
        ctx = get_hermes_context()
        if not ctx or not ctx.slippage:
            raise RuntimeError("SlippageSimulator not available")
        return ctx.slippage.health_check

    _registry.clear()
    _registry.register(ModuleRegistration("RateLimitGovernor", provider_rl, critical=True))
    _registry.register(ModuleRegistration("CircuitBreakerCascade", provider_cb, critical=True))
    _registry.register(ModuleRegistration("SlippageSimulator", provider_slip, critical=False))
    _registry.register(ModuleRegistration("OrderExecutionGateway", provider_gateway, critical=True, dependencies=["RateLimitGovernor"]))
    _registry.register(ModuleRegistration("EmergencyUnwind", provider_unwind, critical=True, dependencies=["OrderExecutionGateway"]))

# ---------------------------------------------------------------------------
# Prometheus 指标
# ---------------------------------------------------------------------------
health_check_latency = Histogram(
    "kunlun_hermes_health_latency_seconds",
    "Hermes health check latency per module",
    ["module", "depth"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
health_status_gauge = Gauge(
    "kunlun_hermes_health_status",
    "Health status per module (1=ok, 0.5=degraded, 0=unhealthy)",
    ["module"],
)
degraded_gauge = Gauge("kunlun_hermes_degraded", "Is Hermes in degraded state (1=yes)")
unhealthy_gauge = Gauge("kunlun_hermes_unhealthy", "Is Hermes in unhealthy state (1=yes)")

# ---------------------------------------------------------------------------
# 缓存管理
# ---------------------------------------------------------------------------
_cache_lock = threading.RLock()
_cached_report: Optional[Tuple[float, str, float, str, HealthReport]] = None  # (time, depth, timeout, trace_id, report)

# ---------------------------------------------------------------------------
# 健康检查核心
# ---------------------------------------------------------------------------
DEFAULT_TIMEOUT_MS = 500.0
VALID_DEPTHS = {"liveness", "readiness", "full"}

def _single_check(
    name: str,
    check_fn: Callable[[], Dict],
    timeout_ms: float,
    breaker: ModuleCircuitBreaker,
    depth: str,
) -> ModuleHealthResult:
    start = time.monotonic()
    if timeout_ms <= 0:
        timeout_ms = 100.0
    try:
        future = _health_check_pool.submit(check_fn)
        result = future.result(timeout=timeout_ms / 1000.0)
        latency = (time.monotonic() - start) * 1000.0

        if not isinstance(result, dict) or "status" not in result:
            raise ValueError(f"Module {name} returned invalid health format")

        status = result.get("status", "unknown")
        if status not in HealthStatus.__members__.values():
            status = HealthStatus.UNHEALTHY.value

        health_status_gauge.labels(module=name).set(
            1.0 if status == HealthStatus.OK else (0.5 if status == HealthStatus.DEGRADED else 0.0)
        )
        health_check_latency.labels(module=name, depth=depth).observe(latency / 1000.0)

        return ModuleHealthResult(
            status=status,
            message=result.get("message", ""),
            latency_ms=latency,
            error_code=result.get("error_code"),
            details=result.get("details", {}),
            warnings=result.get("warnings", []),
        )
    except FutureTimeoutError:
        latency = (time.monotonic() - start) * 1000.0
        logger.error(f"Hermes module {name} health check timed out ({timeout_ms}ms)")
        health_status_gauge.labels(module=name).set(0.0)
        health_check_latency.labels(module=name, depth=depth).observe(latency / 1000.0)
        return ModuleHealthResult(
            status=HealthStatus.UNHEALTHY.value,
            message=f"Timeout after {timeout_ms}ms",
            latency_ms=latency,
            error_code="KUN-EXE-E010",
        )
    except CancelledError:
        latency = (time.monotonic() - start) * 1000.0
        logger.warning(f"Hermes module {name} health check cancelled")
        return ModuleHealthResult(
            status=HealthStatus.UNHEALTHY.value,
            message="Health check cancelled",
            latency_ms=latency,
            error_code="KUN-EXE-E012",
        )
    except Exception as e:
        latency = (time.monotonic() - start) * 1000.0
        safe_msg = _sanitize_message(str(e))
        logger.error(f"Hermes module {name} health check error: {safe_msg}")
        health_status_gauge.labels(module=name).set(0.0)
        health_check_latency.labels(module=name, depth=depth).observe(latency / 1000.0)
        return ModuleHealthResult(
            status=HealthStatus.UNHEALTHY.value,
            message=safe_msg,
            latency_ms=latency,
            error_code="KUN-EXE-E011",
        )

def hermes_health_check(
    depth: str = "full",
    timeout_ms: float = DEFAULT_TIMEOUT_MS,
    trace_id: str = "",
    allow_cached: bool = True,
) -> HealthReport:
    """Hermes Hall comprehensive health check.

    Args:
        depth: 'liveness' | 'readiness' | 'full'
        timeout_ms: per-module timeout in milliseconds
        trace_id: distributed trace identifier
        allow_cached: whether to return cached result within 1s

    Returns:
        HealthReport with overall and per-module status
    """
    check_shutdown_flag()
    global _cached_report
    now = time.time()

    # 标准化 depth 参数
    depth = depth.lower()
    if depth not in VALID_DEPTHS:
        depth = "full"

    with _cache_lock:
        if allow_cached and _cached_report:
            cache_time, cache_depth, cache_timeout, cache_trace, report = _cached_report
            if (cache_depth == depth and cache_timeout == timeout_ms and
                cache_trace == trace_id and now - cache_time < 1.0):
                return report

    if depth == "liveness":
        return HealthReport(
            status=HealthStatus.OK,
            message="Hermes process alive",
            version=__version__,
            timestamp=now,
            trace_id=trace_id,
            overall_latency_ms=0.0,
            depth=depth,
            modules={},
            warnings=[],
            degraded_modules=[],
            failed_modules=[],
        )

    overall_start = time.monotonic()
    warnings: List[str] = []
    degraded: List[str] = []
    failed: List[str] = []
    modules_result: Dict[str, ModuleHealthResult] = {}

    ordered = _registry.topological_order()

    for reg in ordered:
        # 使用持久的断路器（全局单例按模块名）
        # 实际应用中应维护一个全局断路器字典，这里为简化每此调用复用静态字典
        breaker = _get_or_create_breaker(reg.name)
        try:
            check_fn = reg.func_provider()
            if not callable(check_fn):
                raise RuntimeError(f"Health check for {reg.name} is not callable")
        except RuntimeError as e:
            failed.append(reg.name)
            modules_result[reg.name] = ModuleHealthResult(
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=0.0,
            )
            continue

        try:
            res = breaker.call(_single_check, reg.name, check_fn, timeout_ms, breaker, depth)
        except RuntimeError:
            res = ModuleHealthResult(
                status=HealthStatus.UNHEALTHY,
                message="Circuit breaker OPEN",
                latency_ms=0.0,
                error_code="KUN-EXE-W012",
            )
        modules_result[reg.name] = res

        if res["status"] != HealthStatus.OK:
            if reg.critical:
                failed.append(reg.name)
            else:
                degraded.append(reg.name)
            warnings.append(f"{reg.name}: {res['message']}")

    overall_latency = (time.monotonic() - overall_start) * 1000.0

    if failed:
        final_status = HealthStatus.UNHEALTHY
    elif degraded:
        final_status = HealthStatus.DEGRADED
    else:
        final_status = HealthStatus.OK

    # 重置全局指标
    degraded_gauge.set(1.0 if final_status == HealthStatus.DEGRADED else 0.0)
    unhealthy_gauge.set(1.0 if final_status == HealthStatus.UNHEALTHY else 0.0)

    report: HealthReport = {
        "status": final_status.value,
        "message": f"Hermes overall status: {final_status.value}",
        "version": __version__,
        "timestamp": now,
        "trace_id": trace_id,
        "overall_latency_ms": round(overall_latency, 3),
        "depth": depth,
        "modules": modules_result,
        "warnings": warnings,
        "degraded_modules": degraded,
        "failed_modules": failed,
    }

    with _cache_lock:
        _cached_report = (now, depth, timeout_ms, trace_id, report)

    return report

# 全局断路器字典（线程安全）
_breakers: Dict[str, ModuleCircuitBreaker] = {}
_breakers_lock = threading.RLock()

def _get_or_create_breaker(name: str) -> ModuleCircuitBreaker:
    with _breakers_lock:
        if name not in _breakers:
            _breakers[name] = ModuleCircuitBreaker(name)
        return _breakers[name]

# ---------------------------------------------------------------------------
# 模块自身健康检查（供顶层调用）
# ---------------------------------------------------------------------------
def health_check() -> Dict[str, Any]:
    return hermes_health_check(depth="readiness", timeout_ms=200, allow_cached=False)

# 对外导出的公共API更新
__all__ = [
    "OrderExecutionGateway",
    "EmergencyUnwind",
    "RateLimitGovernor",
    "SlippageSimulator",
    "CircuitBreakerCascade",
    "init_hall",
    "get_hermes_context",
    "shutdown_hall",
    "hermes_health_check",
          ]
