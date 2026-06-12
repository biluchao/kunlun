"""
昆仑系统 · 核心包初始化 (Kunlun Core)

负责：
1. 统一配置日志格式
2. 声明核心包版本
3. 暴露常用基类与异常

外部依赖：无
接口契约：无公开方法
"""

import logging
import sys
from typing import Dict, Any

__version__ = "3.0.0"
__codename__ = "Qilin"

# 统一日志格式（异步日志在各自模块内配置）
LOGGING_FORMAT = (
    "[%(asctime)s] [%(levelname)-5s] [%(name)s] "
    "%(message)s [%(filename)s:%(lineno)d]"
)
logging.basicConfig(
    level=logging.INFO,
    format=LOGGING_FORMAT,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class KunlunError(Exception):
    """昆仑系统基础异常"""
    def __init__(self, error_code: str, message: str, detail: Dict[str, Any] = None):
        self.error_code = error_code
        self.message = message
        self.detail = detail or {}
        super().__init__(f"[{error_code}] {message}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "detail": self.detail,
        }


class HealthCheckFailed(KunlunError):
    """模块健康检查失败"""
    pass


class ConfigurationError(KunlunError):
    """配置错误"""
    pass


class DependencyMissing(KunlunError):
    """依赖缺失"""
    pass


# 核心包不自动加载子模块，由 ModuleAssembler 按需装配
__all__ = [
    "__version__",
    "KunlunError",
    "HealthCheckFailed",
    "ConfigurationError",
    "DependencyMissing",
]
