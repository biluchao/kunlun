# -*- coding: utf-8 -*-
"""
Kunlun · Strategos Hall (天玑殿) — Alpha Generation & Signal Processing Hub

Core Responsibilities:
1. Provide on-demand access to all strategos submodules (Factor Engine, Hunger
   Regulator, Footprint Detector, Dual Reality Executor, Virtual Brokerage).
2. Enforce a unified interface contract: every submodule must expose a
   `health_check() -> Dict[str, Any]` classmethod.
3. Offer thread-safe lazy loading with initialization ordering and failure
   isolation.
4. Expose a read-only package health probe suitable for Kubernetes liveness/
   readiness checks, with timeout control and degraded-state reporting.
5. Integrate with the central error registry (KUN-XXX-XXXX) and structured
   logging context.

External Dependencies (true module interfaces):
- core.infrastructure.error_registry.ErrorRegistry : error code lookup
- core.infrastructure.chronos_db.ChronosDB          : (optional) for deep checks

Interface Contract:
- get_module(name: str) -> Any
    Thread-safe accessor. Raises ImportError with KUN error code on failure.
- package_health_check(deep: bool = False, timeout: float = 5.0) -> Dict[str, Any]
    Returns dict with keys: status, version, modules, aggregate_score.
- package_health_check always returns a dict with at least "status" and "reason".
- All submodule health_check() must be side-effect free and read-only.

Exception & Degradation:
- If a submodule fails to import, a degraded status is recorded and the
  corresponding module entry is marked "degraded" with a KUN-FAC-E... error.
- The rest of the package continues to function.

Resource Management:
- No long-lived resources are held. Lazy loading is idempotent.
- Module references are cached in a thread-safe manner.
"""

from __future__ import annotations

import importlib
import logging
import threading
import time
from typing import Any, Callable, Dict, Final, Optional, List, Tuple
from dataclasses import dataclass, field

# ── Centralized Error Registry ──────────────────────────────────────────────
try:
    from core.infrastructure.error_registry import ErrorRegistry
except ImportError:
    ErrorRegistry = None  # type: ignore[assignment]

# ── Structured Logger ───────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Package Metadata ────────────────────────────────────────────────────────
__all__: Final[List[str]] = [
    "FactorComputeEngine",
    "SignalHungerRegulator",
    "FootprintDetector",
    "DualRealityExecutor",
    "VirtualBrokerage",
    "package_health_check",
    "STRATEGOS_VERSION",
]

STRATEGOS_VERSION: Final[str] = "3.0.0"  # Semantic version, matches Git tag v3.0.0

# ── Submodule Registry with Dependency Order ────────────────────────────────
@dataclass(frozen=True)
class _ModuleSpec:
    """Describes a lazily loaded submodule."""
    module_path: str          # relative to this package, e.g. '.factor_compute'
    class_name: str           # exported class name
    dependencies: Tuple[str, ...] = ()  # names of other ModuleSpec keys
    optional: bool = False    # if True, failure to load results in None instead of error

_MODULE_REGISTRY: Final[Dict[str, _ModuleSpec]] = {
    "FactorComputeEngine": _ModuleSpec(
        module_path=".factor_compute",
        class_name="FactorComputeEngine",
    ),
    "SignalHungerRegulator": _ModuleSpec(
        module_path=".signal_hunger",
        class_name="SignalHungerRegulator",
        dependencies=("FactorComputeEngine",),
    ),
    "FootprintDetector": _ModuleSpec(
        module_path=".footprint_detector",
        class_name="FootprintDetector",
    ),
    "DualRealityExecutor": _ModuleSpec(
        module_path=".dual_reality",
        class_name="DualRealityExecutor",
        dependencies=("VirtualBrokerage",),
    ),
    "VirtualBrokerage": _ModuleSpec(
        module_path=".virtual_brokerage",
        class_name="VirtualBrokerage",
    ),
}

# ── Thread-Safe Lazy Loader with Dependency Resolution ──────────────────────
_loader_lock = threading.RLock()
_resolved_cache: Dict[str, Optional[type]] = {}
_import_failures: Dict[str, str] = {}  # module_name -> error message

def _resolve_dependencies(spec: _ModuleSpec) -> None:
    """Ensure all dependencies are loaded before the module itself."""
    for dep_name in spec.dependencies:
        _get_module_unsafe(dep_name)  # Recursively load; lock already held externally

def _get_module_unsafe(name: str) -> Optional[type]:
    """Core import logic; must be called under _loader_lock."""
    if name in _resolved_cache:
        return _resolved_cache[name]
    spec = _MODULE_REGISTRY.get(name)
    if spec is None:
        _resolved_cache[name] = None
        return None
    # Load dependencies first
    _resolve_dependencies(spec)
    try:
        mod = importlib.import_module(spec.module_path, package=__package__)
        cls = getattr(mod, spec.class_name)
    except ImportError as exc:
        err_code = "KUN-FAC-E012" if not spec.optional else "KUN-FAC-W013"
        msg = f"[{err_code}] Failed to import {spec.class_name} from {spec.module_path}: {exc}"
        if not spec.optional:
            logger.critical(msg, exc_info=True)
            _import_failures[name] = msg
            raise ImportError(msg) from exc
        logger.warning(msg)
        cls = None
    except AttributeError as exc:
        err_code = "KUN-FAC-E014"
        msg = f"[{err_code}] Module {spec.module_path} missing class {spec.class_name}: {exc}"
        logger.critical(msg)
        _import_failures[name] = msg
        raise ImportError(msg) from exc
    _resolved_cache[name] = cls
    return cls

def get_module(name: str) -> Any:
    """Thread-safe accessor for a strategos submodule.

    Returns:
        The class object if successfully loaded.

    Raises:
        ImportError: if the module is mandatory and cannot be loaded.
        AttributeError: if the name is not a registered module.
    """
    with _loader_lock:
        if name not in _MODULE_REGISTRY:
            raise AttributeError(f"Strategos hall has no module '{name}'. Available: {list(_MODULE_REGISTRY.keys())}")
        return _get_module_unsafe(name)

# ── Module-Level Attribute Hook (thread-safe via `get_module`) ──────────────
def __getattr__(name: str) -> Any:
    """Enable lazy access to registered submodules as package attributes."""
    # Block dangerous recursive patterns
    if name.startswith("_"):
        raise AttributeError(name)
    return get_module(name)

# ── Health Check Engine ─────────────────────────────────────────────────────
_HEALTH_CACHE: Dict[str, Dict[str, Any]] = {}
_HEALTH_CACHE_TS: float = 0.0
_HEALTH_CACHE_TTL: float = 5.0  # seconds; prevent hammering

def package_health_check(deep: bool = False, timeout: float = 5.0) -> Dict[str, Any]:
    """Non-blocking, read-only health probe for the entire Strategos Hall.

    Args:
        deep: If True, attempts to load unloaded modules and run all checks.
        timeout: Maximum total time in seconds.

    Returns:
        Dict with keys:
        - status: "ok" | "degraded"
        - version: str
        - modules: dict of module_name -> {status, message, ...}
        - aggregate_score: int 0-100
        - reason: str
    """
    now = time.monotonic()
    # Return cached lightweight result if recent enough and not deep
    if not deep and (now - _HEALTH_CACHE_TS) < _HEALTH_CACHE_TTL:
        return _HEALTH_CACHE

    results: Dict[str, Dict[str, Any]] = {}
    all_ok = True
    deadline = now + timeout

    # Helper to check a module without loading heavy dependencies unnecessarily
    def probe_module(name: str, spec: _ModuleSpec) -> None:
        nonlocal all_ok
        # If already loaded, run health check
        cls = _resolved_cache.get(name)
        if cls is None and deep and not spec.optional:
            try:
                cls = get_module(name)
            except ImportError as exc:
                results[name] = {
                    "status": "error",
                    "message": str(exc),
                    "loaded": False,
                }
                all_ok = False
                return
        if cls is not None:
            try:
                if hasattr(cls, 'health_check'):
                    start = time.monotonic()
                    res = cls.health_check()
                    elapsed = time.monotonic() - start
                    results[name] = {
                        "status": res.get("status", "unknown"),
                        "message": res.get("message", ""),
                        "elapsed_ms": round(elapsed * 1000, 2),
                        "loaded": True,
                    }
                    if res.get("status") != "ok":
                        all_ok = False
                else:
                    results[name] = {"status": "error", "message": "health_check not implemented", "loaded": True}
                    all_ok = False
            except Exception as exc:
                logger.exception(f"[KUN-FAC-E015] Health check failed for {name}")
                results[name] = {"status": "error", "message": str(exc), "loaded": True}
                all_ok = False
        else:
            results[name] = {"status": "not_loaded", "message": "module not yet loaded", "loaded": False}
            if not spec.optional:
                all_ok = False
        if time.monotonic() > deadline:
            results.setdefault(name, {})["timeout"] = True
            all_ok = False
            raise TimeoutError  # caught below

    try:
        for name, spec in _MODULE_REGISTRY.items():
            if time.monotonic() > deadline:
                all_ok = False
                results["timeout"] = {"status": "timeout", "message": "health check timed out"}
                break
            probe_module(name, spec)
    except TimeoutError:
        pass

    aggregate_score = _calculate_health_score(results)
    status = "ok" if all_ok else "degraded"

    output = {
        "status": status,
        "version": STRATEGOS_VERSION,
        "modules": results,
        "aggregate_score": aggregate_score,
        "reason": "All modules operational" if all_ok else "Some modules degraded or not loaded",
    }

    if not deep:
        _HEALTH_CACHE = output
        _HEALTH_CACHE_TS = now

    return output

def _calculate_health_score(module_results: Dict[str, Dict[str, Any]]) -> int:
    """Calculate a weighted health score (0-100) based on module statuses."""
    weights = {
        "ok": 1.0,
        "degraded": 0.6,
        "not_loaded": 0.3,
        "error": 0.0,
        "timeout": 0.0,
    }
    total = 0
    count = len(module_results)
    if count == 0:
        return 0
    for name, info in module_results.items():
        st = info.get("status", "error")
        total += weights.get(st, 0.0)
    return max(0, min(100, int((total / count) * 100)))

# ── Shutdown / Reset Support ────────────────────────────────────────────────
def reset_package() -> None:
    """Clear all cached modules and health cache. Useful for testing and
    controlled restarts."""
    with _loader_lock:
        _resolved_cache.clear()
        _import_failures.clear()
    global _HEALTH_CACHE, _HEALTH_CACHE_TS
    _HEALTH_CACHE = {}
    _HEALTH_CACHE_TS = 0.0
    logger.info("[KUN-FAC-I016] Strategos hall reset complete")

# ── Startup Diagnostic (only if executed as main) ───────────────────────────
if __name__ == "__main__":
    # Quick self-diagnostic
    print(f"Strategos Hall v{STRATEGOS_VERSION} self-test:")
    print(package_health_check(deep=True, timeout=10))
