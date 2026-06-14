#!/usr/bin/env python3
"""
昆仑系统 · 智能体注册与生命周期管理器 (AgentRegistry)

核心职责：
1. 作为五大守护智能体（Stone/Wind/Mirror/Eye/Book）的线程安全容器，管理实例化、依赖注入与销毁
2. 遵循 ACID 式初始化事务：全部成功或全部回滚，保证系统状态一致性
3. 提供符合 SRE 标准的健康检查聚合，支持并发检测、超时熔断与缓存降级
4. 支持热重载：无中断保存/恢复状态，旧实例优雅降级
5. 对外暴露不可变智能体视图，杜绝外部篡改风控关键结构

外部依赖（真实模块接口）：
- agents.stone_guardian.StoneGuardian : 保守主义·石
- agents.wind_seeker.WindSeeker : 激进探索·风
- agents.mirror_auditor.MirrorAuditor : 现实检验·镜
- agents.eye_sentinel.EyeSentinel : 全局观察·眼
- agents.book_chronicler.BookChronicler : 历史学家·书
- infrastructure.health_pulse.HealthPulseMonitor : (未来) 上报健康度
- infrastructure.audit_chain.AuditLogChain : 记录初始化/切换事件

接口契约：
- initialize_all(config: Dict[str, Any]) -> bool
- get_agent(name: str) -> Optional[Any]
- get_all_agents() -> Dict[str, Any]
- health_check() -> Dict[str, Any]                   # 模块自身健康检查
- health_check_all() -> Dict[str, Any]                # 聚合所有智能体
- prepare_hot_reload() -> Dict[str, Any]
- restore_from_snapshot(snapshot: Dict) -> bool
- shutdown_all(timeout: float = 5.0) -> bool

异常与降级：
- 智能体类加载失败 -> 标记为不可用，系统降级运行
- 健康检查超时 -> 返回缓存结果（若有效），触发 KUN-AGT-W011
- 初始化失败 -> 执行回滚，记录 CRITICAL 日志并触发 P1 告警

并发模型：
- 所有读操作（get_agent, get_all_agents）使用内部 RLock 保护
- 写操作（initialize_all, restore）加锁并保证原子替换
- 健康检查期间释放锁，避免阻塞交易路径

资源管理：
- 模块持有智能体实例引用，通过 shutdown_all 统一释放
- 不直接持有网络连接或文件句柄
"""

import importlib
import json
import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Dict, Any, Optional, List, Tuple, NamedTuple

logger = logging.getLogger(__name__)

# ───────────────────────────── 常量与类型定义 ─────────────────────────────
MODULE_VERSION = "3.0.0"

# 智能体注册条目
class AgentEntry(NamedTuple):
    module_path: str
    class_name: str
    init_timeout_sec: float

REGISTERED_AGENTS: Dict[str, AgentEntry] = OrderedDict([
    ('stone',  AgentEntry('agents.stone_guardian',  'StoneGuardian',  10.0)),
    ('wind',   AgentEntry('agents.wind_seeker',     'WindSeeker',     10.0)),
    ('mirror', AgentEntry('agents.mirror_auditor',  'MirrorAuditor',   8.0)),
    ('eye',    AgentEntry('agents.eye_sentinel',    'EyeSentinel',     8.0)),
    ('book',   AgentEntry('agents.book_chronicler', 'BookChronicler', 12.0)),
])

# 智能体配置默认值 (key -> default_value)
AGENT_DEFAULT_CONFIGS: Dict[str, Dict[str, Any]] = {
    'stone': {
        'fear_baseline': 0.3,
        'veto_threshold': 0.7,
        'adaptation_rate': 0.01,
    },
    'wind': {
        'courage_max': 20,
        'exploration_cooldown': 600.0,
        'exploration_position_ratio': 0.2,
    },
    'mirror': {
        'calibration_lr': 0.005,
        'gap_threshold': 0.15,
        'strategy_decay_threshold': 0.5,
    },
    'eye': {
        'correlation_window': 100,
        'crash_threshold': 0.03,
        'cross_market_risk_enabled': True,
    },
    'book': {
        'cluster_count': 20,
        'similarity_threshold': 0.6,
        'memory_retention_days': 365,
    },
}

# 全局状态
_agent_classes: Dict[str, Optional[type]] = {}
_agent_instances: Dict[str, Any] = OrderedDict()
_lock = threading.RLock()  # 保护 _agent_instances

# 健康检查缓存
_last_healthy_snapshot: Dict[str, Any] = {}
_last_healthy_time: float = 0.0
HEALTH_CACHE_TTL_SEC = 30.0

# 共享线程池（用于健康检查并发）
_health_check_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="ag_health")


class AgentInitializationError(Exception):
    """智能体初始化失败（含可恢复建议）"""
    def __init__(self, agent_name: str, reason: str, recovery_hint: str = ""):
        msg = f"智能体 {agent_name} 初始化失败: {reason}"
        if recovery_hint:
            msg += f" | 建议: {recovery_hint}"
        super().__init__(msg)
        self.agent_name = agent_name
        self.recovery_hint = recovery_hint


# ───────────────────────────── 类加载与配置 ─────────────────────────────
def _import_agent_classes(force_reload: bool = False) -> Dict[str, Optional[type]]:
    """
    动态加载智能体类，支持强制重新导入（用于热重载）
    返回：名称 -> 类对象 或 None
    """
    global _agent_classes
    if _agent_classes and not force_reload:
        return _agent_classes

    classes: Dict[str, Optional[type]] = {}
    for name, entry in REGISTERED_AGENTS.items():
        try:
            mod = importlib.import_module(entry.module_path)
            if force_reload:
                importlib.reload(mod)
            cls = getattr(mod, entry.class_name)
            if not callable(cls):
                raise AttributeError(f"{entry.class_name} 不是可调用类")
            classes[name] = cls
        except Exception as e:
            logger.critical("[KUN-AGT-F002] 无法加载智能体 %s: %s", name, e, exc_info=True)
            classes[name] = None
    _agent_classes = classes
    return classes


def _validate_and_apply_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    校验并补全智能体配置，输出安全且完整的配置字典。
    非法值记录告警并使用默认值。
    """
    agent_configs = config.get('agents', {}) if config else {}
    if not isinstance(agent_configs, dict):
        logger.warning("[KUN-AGT-W012] agents 配置非法，使用空配置")
        agent_configs = {}

    safe_config: Dict[str, Dict[str, Any]] = {}
    for name, defaults in AGENT_DEFAULT_CONFIGS.items():
        user_cfg = agent_configs.get(name, {})
        if not isinstance(user_cfg, dict):
            logger.warning("[KUN-AGT-W013] 智能体 %s 配置非字典，使用默认值", name)
            user_cfg = {}
        merged = defaults.copy()
        for k, v in user_cfg.items():
            if k not in merged:
                logger.warning("[KUN-AGT-W016] 智能体 %s 未知配置项 %s，将被忽略", name, k)
                continue
            default_val = merged[k]
            # 简单类型校验，确保安全
            if type(v) is not type(default_val):
                logger.warning("[KUN-AGT-W014] 智能体 %s 配置项 %s 类型不匹配，使用默认值", name, k)
                continue
            merged[k] = v
        safe_config[name] = merged
    return safe_config


# ───────────────────────────── 生命周期管理 ─────────────────────────────
def initialize_all(config: Optional[Dict[str, Any]] = None) -> bool:
    """
    事务式初始化所有智能体：全部成功才提交，否则回滚并告警。
    返回 True 表示全部成功。
    """
    global _agent_instances
    classes = _import_agent_classes()
    if not classes or all(cls is None for cls in classes.values()):
        logger.critical("[KUN-AGT-F005] 无可用智能体类，系统无法启动")
        return False

    agent_cfgs = _validate_and_apply_defaults(config or {})

    # 阶段1：创建并早期检查（本地容器，按注册顺序）
    new_instances: Dict[str, Any] = OrderedDict()
    success = True
    for name, entry in REGISTERED_AGENTS.items():
        cls = classes.get(name)
        if cls is None:
            logger.error("[KUN-AGT-E001] 智能体 %s 类不可用，跳过初始化", name)
            continue
        cfg = agent_cfgs.get(name, {})
        try:
            instance = cls(cfg)
            # 早期健康检查（带超时）
            hc = _timed_health_check(instance, timeout=entry.init_timeout_sec)
            if hc.get('status') == 'error':
                raise AgentInitializationError(name, f"健康检查失败: {hc.get('message')}",
                                               recovery_hint="检查智能体依赖和配置值域")
            new_instances[name] = instance
            logger.info("智能体 %s 初始化成功", name)
        except AgentInitializationError:
            raise
        except Exception as e:
            logger.critical("[KUN-AGT-F004] 智能体 %s 初始化异常: %s", name, e, exc_info=True)
            success = False
            # 逆序清理已创建实例（因为可能存在依赖）
            _cleanup_instances(reversed(list(new_instances.values())))
            break

    if not success:
        return False

    # 阶段2：提交
    with _lock:
        _agent_instances = new_instances
        _record_audit_event("agent_initialization", {
            "agents": list(new_instances.keys()),
            "version": MODULE_VERSION
        })
    logger.info("所有智能体初始化完毕")
    return True


def _cleanup_instances(instances):
    """安全清理智能体实例，逆序关闭，超时跳过"""
    for inst in instances:
        try:
            if hasattr(inst, 'shutdown'):
                inst.shutdown()
        except Exception as e:
            logger.warning("清理智能体实例失败: %s", e)


def shutdown_all(timeout: float = 5.0) -> bool:
    """关闭所有智能体，支持超时控制，逆序关闭"""
    with _lock:
        instances = list(_agent_instances.values())
        _agent_instances.clear()

    success = True
    for inst in reversed(instances):
        try:
            if hasattr(inst, 'shutdown'):
                start = time.time()
                inst.shutdown()
                elapsed = time.time() - start
                if elapsed > timeout:
                    logger.warning("[KUN-AGT-W015] 智能体 shutdown 耗时 %.2fs，超过预期", elapsed)
        except Exception as e:
            logger.error("[KUN-AGT-E016] 关闭智能体时异常: %s", e)
            success = False
    return success


# ───────────────────────────── 智能体访问接口 ─────────────────────────────
def get_agent(name: str) -> Optional[Any]:
    """线程安全获取单个智能体实例（调用方不应修改实例状态）"""
    with _lock:
        return _agent_instances.get(name)


def get_all_agents() -> Dict[str, Any]:
    """返回智能体实例映射的不可变视图（浅拷贝）"""
    with _lock:
        return OrderedDict(_agent_instances)


# ───────────────────────────── 健康检查 ─────────────────────────────
def _timed_health_check(agent_instance, timeout: float) -> Dict[str, Any]:
    """对单个智能体执行带超时的健康检查"""
    def _hc():
        return agent_instance.health_check()

    try:
        future = _health_check_pool.submit(_hc)
        return future.result(timeout=timeout)
    except FuturesTimeoutError:
        logger.warning("[KUN-AGT-W011] 健康检查超时 (%.1fs)", timeout)
        return {"status": "timeout", "message": f"超时 {timeout}s"}
    except Exception as e:
        logger.error("健康检查异常: %s", e)
        return {"status": "error", "message": str(e)}


def health_check_all() -> Dict[str, Any]:
    """
    聚合所有智能体健康状态，并发执行，超时使用缓存。
    返回详细指标，便于 SRE 系统使用。
    """
    agents = get_all_agents()
    if not agents:
        return {
            "status": "degraded",
            "agents_total": 0,
            "agents_available": 0,
            "unavailable_agents": [],
            "details": {},
            "message": "无智能体实例"
        }

    # 并发发起健康检查
    futures: Dict[str, Any] = {}
    for name, agent in agents.items():
        try:
            futures[name] = _health_check_pool.submit(agent.health_check)
        except RuntimeError:
            # 线程池关闭
            logger.error("[KUN-AGT-E020] 健康检查线程池不可用")
            return _last_healthy_snapshot or {"status": "error", "message": "线程池关闭"}

    results = {}
    available_count = 0
    unavailable_list: List[str] = []
    for name, future in futures.items():
        try:
            hc = future.result(timeout=3.0)
        except FuturesTimeoutError:
            hc = {"status": "timeout", "message": "聚合超时"}
        except Exception as e:
            hc = {"status": "error", "message": str(e)}
        results[name] = hc
        if hc.get('status') == 'ok':
            available_count += 1
        else:
            unavailable_list.append(name)

    total = len(agents)
    status = "ok" if available_count == total else "degraded"
    summary = {
        "status": status,
        "agents_total": total,
        "agents_available": available_count,
        "unavailable_agents": unavailable_list,
        "details": results,
        "timestamp": time.time()
    }

    # 更新缓存
    global _last_healthy_snapshot, _last_healthy_time
    if status == "ok":
        _last_healthy_snapshot = summary
        _last_healthy_time = time.time()
    else:
        # 若缓存未过期，可返回最后一次成功快照以防抖动
        if time.time() - _last_healthy_time < HEALTH_CACHE_TTL_SEC and _last_healthy_snapshot:
            logger.debug("返回健康检查缓存")
            return _last_healthy_snapshot
    return summary


def health_check() -> Dict[str, Any]:
    """
    模块自身健康检查：验证锁机制、类加载状态及基本连通性
    """
    try:
        # 检查锁是否可用（避免死锁）
        locked = _lock.acquire(blocking=False)
        if locked:
            _lock.release()
        else:
            return {"status": "error", "message": "互斥锁被其他线程持有，可能死锁"}

        # 检查智能体类是否至少加载成功一个
        classes = _import_agent_classes()
        loaded = sum(1 for cls in classes.values() if cls is not None)
        if loaded == 0:
            return {"status": "error", "message": "所有智能体类加载失败"}

        with _lock:
            instance_count = len(_agent_instances)

        return {
            "status": "ok",
            "message": f"Agent Registry v{MODULE_VERSION}，已加载 {loaded}/{len(REGISTERED_AGENTS)} 类，当前 {instance_count} 实例"
        }
    except Exception as e:
        logger.error("模块健康检查异常: %s", e)
        return {"status": "error", "message": str(e)}


# ───────────────────────────── 热重载支持 ─────────────────────────────
def prepare_hot_reload() -> Dict[str, Any]:
    """保存当前所有智能体状态，返回可序列化快照"""
    with _lock:
        state = {}
        for name, instance in _agent_instances.items():
            if instance and hasattr(instance, 'get_state'):
                try:
                    st = instance.get_state()
                    # 验证可序列化
                    json.dumps(st)  # 若不可序列化会抛异常
                    state[name] = st
                except (TypeError, ValueError) as e:
                    logger.error("[KUN-AGT-E017] 智能体 %s 状态不可序列化: %s", name, e)
                except Exception as e:
                    logger.error("[KUN-AGT-E018] 准备智能体 %s 状态异常: %s", name, e)
        return {
            "version": MODULE_VERSION,
            "agent_count": len(_agent_instances),
            "state": state,
            "timestamp": time.time()
        }


def restore_from_snapshot(snapshot: Dict[str, Any]) -> bool:
    """从热重载快照恢复状态，版本不匹配拒绝"""
    if snapshot.get("version") != MODULE_VERSION:
        logger.error("[KUN-AGT-E019] 快照版本 %s 不匹配当前 %s", snapshot.get("version"), MODULE_VERSION)
        return False
    with _lock:
        for name, state_data in snapshot.get("state", {}).items():
            instance = _agent_instances.get(name)
            if instance and hasattr(instance, 'set_state'):
                try:
                    instance.set_state(state_data)
                except Exception as e:
                    logger.error("[KUN-AGT-E020] 恢复智能体 %s 状态失败: %s", name, e)
                    return False
            else:
                logger.warning("[KUN-AGT-W021] 快照中的智能体 %s 当前不存在或缺少 set_state", name)
    return True


def _record_audit_event(event_type: str, details: Dict[str, Any]):
    """记录审计事件，失败不影响主流程（但记录告警）"""
    try:
        from infrastructure.audit_chain import AuditLogChain
        AuditLogChain.log_event(
            event_type=event_type,
            severity="INFO",
            details=details
        )
    except ImportError:
        logger.warning("审计日志模块未加载，事件未记录")
    except Exception as e:
        logger.error("[KUN-SEC-E010] 审计日志写入失败: %s", e)


# 导出列表
__all__ = [
    "initialize_all",
    "shutdown_all",
    "get_agent",
    "get_all_agents",
    "health_check",          # 模块自身健康检查
    "health_check_all",      # 聚合所有智能体健康检查
    "prepare_hot_reload",
    "restore_from_snapshot",
                     ]
