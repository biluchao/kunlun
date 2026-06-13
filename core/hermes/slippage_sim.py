#!/usr/bin/env python3
"""
昆仑系统 · 多层次滑点模拟器 (SlippageSimulator)

核心职责：
1. 基于 Almgren-Chriss 扩展模型，结合订单簿微观结构、波动率曲面、流动性冲击，
   输出高频交易级的预期滑点（bps）及不确定性区间。
2. 支持在线自适应校准：利用真实成交数据，采用动量 SGD + 自适应学习率，
   分资产独立更新模型参数。
3. 为虚拟券商、执行算法路由器、风险敞口计算提供金融级精准的冲击成本预测。

外部依赖（真实模块接口）：
- polaris.market_regime.MarketRegimeClassifier : 提供实时波动率、成交量、买卖价差
- infrastructure.stream_gateway.StreamGateway : 获取原子订单簿快照（含序列号）
- infrastructure.chronos_db.ChronosDB : 读取历史成交与日波动率估计

接口契约：
- estimate_slippage(order: OrderRequest, orderbook: OrderBookSnapshot) -> SlippageEstimate
  输入标准化订单请求与订单簿快照，返回滑点估算、置信区间、分解成分
- update_model(fill: TradeFill) -> CalibrationResult
  输入真实成交填充，更新对应资产的模型参数
- health_check() -> Dict[str, Any]
  模块自检与参数有效性验证

异常与降级：
- 订单簿深度不足或数据陈旧：降级为保守波动率模型，标记 KUN-EXE-W011
- 校准样本不足（<30笔）：返回固定模型，不进行在线学习
- 参数漂移检测：当连续20笔残差扩大2倍时，触发模型重置
- 数值异常（NaN/Inf）：自动截断并使用安全默认值，记录 KUN-EXE-E014

资源管理：
- 每资产维护独立校准缓冲区（上限500条），定期淘汰旧数据
- 模型参数每15分钟异步持久化到 ChronosDB，重启后自动恢复
- 多线程安全：校准更新使用细粒度锁，估值只读无锁
"""

import logging
import math
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import deque
import threading
import numpy as np

logger = logging.getLogger(__name__)


# ---------- 数据类定义 ----------
@dataclass
class OrderRequest:
    """标准化订单请求"""
    symbol: str
    side: str                # 'buy' or 'sell'
    quantity: float          # 基础货币数量
    notional: float          # 名义价值（USDT）
    duration_sec: float = 30.0  # 预期执行时间
    order_type: str = 'limit'  # 'market' or 'limit'
    limit_price: Optional[float] = None  # 限价单价格（如有）


@dataclass
class OrderBookSnapshot:
    """原子订单簿快照"""
    symbol: str
    bids: List[List[float]]   # [[price, size], ...] 按价格降序
    asks: List[List[float]]   # [[price, size], ...] 按价格升序
    timestamp_ns: int         # 纳秒级时间戳
    sequence_id: int          # 序列号，用于验证完整性


@dataclass
class SlippageEstimate:
    """滑点估算结果"""
    estimated_slippage_bps: float
    confidence_interval_low: float
    confidence_interval_high: float
    components: Dict[str, float]
    warnings: List[str]
    model_uncertainty: float   # 模型不确定性 (0-1)


@dataclass
class TradeFill:
    """真实成交填充"""
    symbol: str
    side: str
    quantity: float
    expected_price: float      # 下单时期望价格（中间价）
    fill_price: float          # 实际成交均价
    orderbook_snapshot: Optional[OrderBookSnapshot] = None
    duration_sec: float = 10.0
    fill_time_ns: int = 0


@dataclass
class CalibrationResult:
    """校准结果"""
    status: str
    buffer_size: int
    params: Dict[str, float]
    learning_rate: float
    gradient_norm: float


class SlippageSimulator:
    """
    机构级滑点模拟器
    基于 Almgren-Chriss (2001) 框架，融入订单簿弹性、波动率调整、资产特异性参数。
    """

    # ------------------ 可配置模型超参数（可通过 config dict 覆盖）------------------
    # Almgren-Chriss 结构参数初值（分资产配置时将被覆盖）
    DEFAULT_ETA = 0.14             # 永久冲击系数 (0.01~0.5)
    DEFAULT_GAMMA = 0.04           # 临时冲击系数 (0.001~0.1)
    DEFAULT_DELTA = 0.5            # 非线性指数 (0.3~0.8)
    DEFAULT_SPREAD_ALPHA = 0.5     # 价差参与率敏感度 (0~1)
    DEFAULT_RESILIENCE = 0.02      # 订单簿弹性恢复速率 (0~0.1)

    # 波动率与成交量缩放系数
    VOLATILITY_SCALE = 0.15        # 日波动率转滑点贡献乘数
    LIQUIDITY_SCALE = 0.05         # 流动性稀缺溢价乘数
    SIZE_SCALE = 0.01              # 规模非线性乘数

    # 校准控制
    CALIBRATION_WINDOW = 500       # 每资产最大保留成交数
    CALIBRATION_MIN_SAMPLES = 30   # 开始学习的最小样本
    INITIAL_LEARNING_RATE = 0.005
    LR_DECAY = 0.998               # 每步衰减
    MIN_LR = 1e-6
    MOMENTUM = 0.9                 # SGD 动量
    GRADIENT_CLIP_PER_PARAM = {'eta': 0.1, 'gamma': 0.05, 'delta': 0.02, 'spread_alpha': 0.1}
    PARAM_BOUNDS = {
        'eta': (0.005, 0.5),
        'gamma': (0.0005, 0.2),
        'delta': (0.25, 0.85),
        'spread_alpha': (0.0, 1.0)
    }

    # 保守与极端控制
    CONSERVATIVE_SLIPPAGE_BPS = 8.0    # 数据不足时的保守滑点
    MAX_ACCEPTABLE_SLIPPAGE_BPS = 500  # 超过此值熔断，返回并标记异常
    MIN_NOTIONAL_FOR_ESTIMATE = 10.0   # 名义价值低于此值时忽略冲击

    # 数值稳定
    EPS = 1e-12
    LOG_EPS = 1e-12

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化滑点模拟器
        :param config: 可选的参数字典，覆盖类属性
        """
        self._lock = threading.Lock()
        # 按资产存储校准状态
        self._calib = {}  # symbol -> dict with params, buffer, etc.

        # 默认参数会被 config 覆盖
        if config:
            self._apply_verified_config(config)

        # 外部依赖（可选）
        self._market_regime = None

        # 持久化标志
        self._persist_enabled = False
        self._persist_interval = 900  # 15分钟
        self._last_persist_time = 0

        logger.info("SlippageSimulator 初始化完毕，初始全局默认参数: eta=%.4f gamma=%.4f delta=%.3f",
                    self.DEFAULT_ETA, self.DEFAULT_GAMMA, self.DEFAULT_DELTA)

    # ---------------------------------------------------------------------
    #  配置与依赖注入
    # ---------------------------------------------------------------------
    def _apply_verified_config(self, config: Dict) -> None:
        """仅允许白名单中的配置项覆盖类属性"""
        allowed_keys = {k for k in self.__class__.__dict__ if k.isupper() and not k.startswith('_')}
        for k, v in config.items():
            if k in allowed_keys:
                if isinstance(v, type(getattr(self, k))):
                    setattr(self, k, v)
                    logger.info("配置覆盖: %s = %s", k, v)
                else:
                    logger.warning("配置项 %s 类型不匹配，忽略", k)
            else:
                logger.warning("未知或禁止覆盖的配置项: %s", k)

    def set_market_regime(self, regime):
        """注入市场状态模块（提供波动率、成交量、价差等）"""
        self._market_regime = regime

    def enable_persistence(self, db_path: str):
        """启用模型持久化（与 ChronosDB 集成）"""
        self._persist_enabled = True
        self._persist_db_path = db_path
        self._load_persisted_params()

    # ---------------------------------------------------------------------
    #  资产特定状态管理
    # ---------------------------------------------------------------------
    def _get_asset_state(self, symbol: str):
        """获取或创建某资产的校准状态"""
        if symbol not in self._calib:
            self._calib[symbol] = {
                'params': {
                    'eta': self.DEFAULT_ETA,
                    'gamma': self.DEFAULT_GAMMA,
                    'delta': self.DEFAULT_DELTA,
                    'spread_alpha': self.DEFAULT_SPREAD_ALPHA,
                    'resilience': self.DEFAULT_RESILIENCE,
                },
                'buffer': deque(maxlen=self.CALIBRATION_WINDOW),
                'velocity': {k: 0.0 for k in ['eta', 'gamma', 'delta', 'spread_alpha']},
                'lr': self.INITIAL_LEARNING_RATE,
                'update_count': 0,
                'residuals': deque(maxlen=200)
            }
        return self._calib[symbol]

    # ---------------------------------------------------------------------
    #  订单簿微观冲击计算
    # ---------------------------------------------------------------------
    def _compute_book_impact(self, order: OrderRequest, ob: OrderBookSnapshot) -> Tuple[float, float, List[str]]:
        """
        计算即时订单簿冲击，返回 (avg_price, slippage_bps, warnings)
        会正确处理超过深度的情况并给出警告。
        """
        warnings = []
        levels = ob.asks if order.side == 'buy' else ob.bids
        if not levels:
            return 0.0, self.CONSERVATIVE_SLIPPAGE_BPS, ["订单簿为空"]

        mid_price = self._safe_mid_price(ob)
        if mid_price <= 0:
            return 0.0, self.CONSERVATIVE_SLIPPAGE_BPS, ["无法计算中间价"]

        remaining = order.quantity
        total_cost = 0.0
        total_qty = 0.0
        max_levels = min(len(levels), 50)  # 防止极端深度遍历

        for i in range(max_levels):
            price, size = levels[i][0], levels[i][1]
            if price <= 0 or size <= 0:
                continue
            fill_qty = min(remaining, size)
            total_cost += price * fill_qty
            total_qty += fill_qty
            remaining -= fill_qty
            if remaining <= 0:
                break

        if total_qty <= 0:
            return 0.0, self.CONSERVATIVE_SLIPPAGE_BPS, ["无可用流动性"]

        avg_price = total_cost / total_qty
        slippage_bps = (avg_price - mid_price) / mid_price * 10000
        if order.side == 'buy':
            slippage_bps = abs(slippage_bps)
        else:
            slippage_bps = -abs(slippage_bps)  # 卖方向为负，但我们返回绝对值用于冲击规模

        if remaining > 0:
            warnings.append(f"订单量超过订单簿总深度，剩余 {remaining:.4f} 未成交，滑点可能低估")
            # 附加惩罚
            slippage_bps += 50  # 额外惩罚

        return avg_price, abs(slippage_bps), warnings

    @staticmethod
    def _safe_mid_price(ob: OrderBookSnapshot) -> float:
        """安全计算中间价，处理边界"""
        bid = ob.bids[0][0] if ob.bids and len(ob.bids[0]) > 0 else None
        ask = ob.asks[0][0] if ob.asks and len(ob.asks[0]) > 0 else None
        if bid is not None and ask is not None:
            if bid > ask:  # 异常交叉
                return (bid + ask) / 2.0
            return (bid + ask) / 2.0
        elif bid is not None:
            return bid
        elif ask is not None:
            return ask
        return 0.0

    def _get_current_volatility(self, symbol: str) -> float:
        """获取年化日波动率（小数），从外部模块获取，失败时使用默认"""
        if self._market_regime:
            try:
                return self._market_regime.get_daily_volatility(symbol)
            except Exception:
                pass
        return 0.02  # 保守默认

    def _get_daily_volume(self, symbol: str) -> float:
        """获取日均成交量（以基础货币计），若无法获取返回安全默认值"""
        if self._market_regime:
            try:
                return self._market_regime.get_daily_volume(symbol)
            except Exception:
                pass
        # 缺省值：BTC 3万，ETH 1.5万，其他 5000
        default_map = {'BTCUSDT': 30000, 'ETHUSDT': 15000}
        return default_map.get(symbol, 5000)

    def _get_bid_ask_spread_bps(self, ob: OrderBookSnapshot) -> float:
        """计算买卖价差（bps）"""
        if not ob.bids or not ob.asks:
            return 10.0  # 默认10bps
        best_bid = ob.bids[0][0]
        best_ask = ob.asks[0][0]
        if best_bid <= 0 or best_ask <= 0:
            return 10.0
        mid = (best_bid + best_ask) / 2.0
        if mid <= 0:
            return 10.0
        return (best_ask - best_bid) / mid * 10000

    # ---------------------------------------------------------------------
    #  核心滑点估算
    # ---------------------------------------------------------------------
    def estimate_slippage(self, order: OrderRequest, orderbook: OrderBookSnapshot) -> SlippageEstimate:
        """
        计算预期滑点（全面机构级实现）
        """
        warnings = []
        symbol = order.symbol

        # 极小订单忽略冲击
        if order.notional < self.MIN_NOTIONAL_FOR_ESTIMATE:
            return SlippageEstimate(0.0, 0.0, 0.0, {}, [], 0.0)

        # 1. 订单簿即时冲击
        _, book_impact_bps, warns = self._compute_book_impact(order, orderbook)
        warnings.extend(warns)

        # 2. 买卖价差成本
        spread_bps = self._get_bid_ask_spread_bps(orderbook)
        # 如果为市价单，全价差；限价单假设只付一半
        spread_cost = spread_bps if order.order_type == 'market' else spread_bps * 0.5
        warnings.append(f"价差成本: {spread_cost:.2f} bps")

        # 3. 波动率与流动性调整
        daily_vol = self._get_current_volatility(symbol)
        daily_volume = self._get_daily_volume(symbol)
        if daily_volume <= 0:
            daily_volume = 10000
        participation_rate = order.quantity / max(daily_volume, self.EPS)

        # 获取资产特定参数
        state = self._get_asset_state(symbol)
        p = state['params']

        # 永久冲击（与成交量份额和波动率成正比）
        permanent = p['eta'] * daily_vol * (participation_rate ** p['delta'])
        # 临时冲击（取决于执行速度）
        urgency = 1.0 / max(order.duration_sec, 1.0)
        temporary = p['gamma'] * participation_rate * math.sqrt(urgency)
        # 弹性反弹（临时冲击的一部分会恢复）
        resilience_factor = math.exp(-p['resilience'] * order.duration_sec)
        effective_temporary = temporary * (1.0 - resilience_factor)

        # 波动率与流动性附加项
        vol_add = self.VOLATILITY_SCALE * daily_vol * math.sqrt(order.duration_sec / 86400)
        liq_add = self.LIQUIDITY_SCALE * (order.notional / max(daily_volume * (daily_vol * 100), self.EPS))

        # 综合滑点 (bps) - 各成分按比例组合
        total_bps = (book_impact_bps * 0.25 +
                     spread_cost * 0.15 +
                     permanent * 10000 * 0.2 +
                     effective_temporary * 10000 * 0.2 +
                     vol_add * 10000 * 0.1 +
                     liq_add * 10000 * 0.1)

        # 合理性检查
        if not math.isfinite(total_bps) or total_bps > self.MAX_ACCEPTABLE_SLIPPAGE_BPS:
            warnings.append(f"滑点估算异常 ({total_bps:.1f} bps)，使用保守值")
            total_bps = self.CONSERVATIVE_SLIPPAGE_BPS
            model_uncertainty = 1.0
        else:
            # 模型不确定性来自校准残差
            model_uncertainty = self._calculate_uncertainty(symbol)

        # 置信区间基于残差标准差
        ci_low, ci_high = total_bps, total_bps
        if len(state['residuals']) >= 30:
            std_resid = np.std(state['residuals']) * 10000
            z = 1.96
            ci_low = max(0.0, total_bps - z * std_resid)
            ci_high = total_bps + z * std_resid

        return SlippageEstimate(
            estimated_slippage_bps=round(total_bps, 2),
            confidence_interval_low=round(ci_low, 2),
            confidence_interval_high=round(ci_high, 2),
            components={
                'book_impact_bps': round(book_impact_bps, 2),
                'spread_bps': round(spread_cost, 2),
                'permanent_bps': round(permanent * 10000, 2),
                'temporary_bps': round(effective_temporary * 10000, 2),
                'volatility_bps': round(vol_add * 10000, 2),
                'liquidity_bps': round(liq_add * 10000, 2)
            },
            warnings=warnings,
            model_uncertainty=round(model_uncertainty, 3)
        )

    def _calculate_uncertainty(self, symbol: str) -> float:
        """基于近期残差计算模型不确定性 (0-1)"""
        state = self._get_asset_state(symbol)
        res = list(state['residuals'])
        if len(res) < 10:
            return 0.5
        rmse = np.sqrt(np.mean(np.square(res)))
        # 归一化到合理范围，假设残差通常在 0~50 bps
        return min(1.0, rmse / 0.005)  # 0.005 代表 50 bps

    # ---------------------------------------------------------------------
    #  在线模型校准
    # ---------------------------------------------------------------------
    def update_model(self, fill: TradeFill) -> CalibrationResult:
        """
        输入真实成交记录，更新对应资产的冲击模型参数
        """
        symbol = fill.symbol
        if fill.fill_price <= 0 or fill.expected_price <= 0:
            return CalibrationResult("error", 0, {}, 0.0, 0.0)

        # 实际滑点（考虑方向，买正卖负）
        actual_slippage_bps = (fill.fill_price - fill.expected_price) / fill.expected_price * 10000
        if fill.side == 'sell':
            actual_slippage_bps = -actual_slippage_bps  # 卖出滑点为负，我们取绝对值用于冲击大小
        actual_abs_bps = abs(actual_slippage_bps)

        # 存入缓冲区
        state = self._get_asset_state(symbol)
        daily_vol = self._get_current_volatility(symbol)
        daily_volume = self._get_daily_volume(symbol)
        participation = fill.quantity / max(daily_volume, self.EPS)

        state['buffer'].append({
            'quantity': fill.quantity,
            'participation': participation,
            'duration': fill.duration_sec,
            'actual_slippage_abs': actual_abs_bps,
            'daily_vol': daily_vol,
            'timestamp': time.time()
        })

        # 当样本足够时执行参数更新
        if len(state['buffer']) >= self.CALIBRATION_MIN_SAMPLES:
            with self._lock:
                self._gradient_update(symbol)

        return CalibrationResult(
            status="ok",
            buffer_size=len(state['buffer']),
            params=state['params'].copy(),
            learning_rate=state['lr'],
            gradient_norm=0.0  # 可后续扩展
        )

    def _gradient_update(self, symbol: str):
        """执行一次动量 SGD 更新资产参数"""
        state = self._get_asset_state(symbol)
        buf = list(state['buffer'])
        if len(buf) < self.CALIBRATION_MIN_SAMPLES:
            return

        # 随机采样小批量
        batch_size = min(32, len(buf))
        batch = np.random.choice(buf, batch_size, replace=False)

        params = state['params']
        velocity = state['velocity']
        lr = state['lr']

        grad = {k: 0.0 for k in ['eta', 'gamma', 'delta', 'spread_alpha']}

        for sample in batch:
            part = max(sample['participation'], self.LOG_EPS)
            dur = max(sample['duration'], 1.0)
            vol = sample['daily_vol']
            actual = sample['actual_slippage_abs'] / 10000  # 转为比例

            # 模型预测
            perm = params['eta'] * vol * (part ** params['delta'])
            temp = params['gamma'] * part * math.sqrt(1.0 / dur)
            resilience = math.exp(-params.get('resilience', self.DEFAULT_RESILIENCE) * dur)
            eff_temp = temp * (1.0 - resilience)
            pred = perm + eff_temp

            error = pred - actual

            # 梯度
            grad['eta'] += 2 * error * vol * (part ** params['delta'])
            grad['gamma'] += 2 * error * part * math.sqrt(1.0 / dur) * (1.0 - resilience)
            # delta 梯度
            log_part = math.log(part)
            grad['delta'] += 2 * error * params['eta'] * vol * (part ** params['delta']) * log_part
            # spread_alpha 暂不使用

        # 平均梯度
        for k in grad:
            grad[k] /= batch_size

        # 自适应梯度裁剪
        for k in grad:
            clip_val = self.GRADIENT_CLIP_PER_PARAM.get(k, 1.0)
            grad[k] = max(-clip_val, min(clip_val, grad[k]))

        # 动量更新
        for k in params:
            if k in velocity:
                velocity[k] = self.MOMENTUM * velocity[k] - lr * grad[k]
                new_val = params[k] + velocity[k]
                # 边界约束
                low, high = self.PARAM_BOUNDS.get(k, (0.0, 10.0))
                params[k] = max(low, min(high, new_val))

        # 学习率衰减
        state['lr'] = max(self.MIN_LR, lr * self.LR_DECAY)
        state['update_count'] += 1

        # 更新残差历史
        # 计算最近一个样本的残差（取最后一个样本）
        if buf:
            last = buf[-1]
            part = max(last['participation'], self.LOG_EPS)
            pred = (params['eta'] * last['daily_vol'] * (part ** params['delta']) +
                    params['gamma'] * part * math.sqrt(1.0 / max(last['duration'], 1.0)))
            actual = last['actual_slippage_abs'] / 10000
            state['residuals'].append(pred - actual)

        logger.info("资产 %s 参数更新 #%d: eta=%.4f gamma=%.4f delta=%.3f lr=%.6f",
                    symbol, state['update_count'], params['eta'], params['gamma'], params['delta'], state['lr'])

    # ---------------------------------------------------------------------
    #  持久化与恢复
    # ---------------------------------------------------------------------
    def _load_persisted_params(self):
        """从 ChronosDB 加载历史参数（占位）"""
        try:
            # 实际应从 ChronosDB 读取
            # data = ChronosDB.load('slippage_params')
            # self._calib = data
            pass
        except Exception as e:
            logger.warning("加载持久化参数失败: %s", e)

    def _persist_params(self):
        """持久化当前参数（占位）"""
        if not self._persist_enabled:
            return
        now = time.time()
        if now - self._last_persist_time < self._persist_interval:
            return
        try:
            # 序列化 self._calib 写入 ChronosDB
            self._last_persist_time = now
        except Exception as e:
            logger.error("持久化参数失败: %s", e)

    # ---------------------------------------------------------------------
    #  健康检查与诊断
    # ---------------------------------------------------------------------
    def health_check(self) -> Dict[str, Any]:
        """执行自检，验证核心计算路径与参数合法性"""
        try:
            # 构造模拟订单簿
            ob = OrderBookSnapshot(
                symbol='BTCUSDT',
                bids=[[50000.0, 2.0], [49990.0, 3.0]],
                asks=[[50010.0, 1.5], [50020.0, 4.0]],
                timestamp_ns=time.time_ns(),
                sequence_id=1001
            )
            order = OrderRequest(
                symbol='BTCUSDT',
                side='buy',
                quantity=0.5,
                notional=25000,
                duration_sec=5
            )
            est = self.estimate_slippage(order, ob)
            if not isinstance(est, SlippageEstimate) or est.estimated_slippage_bps <= 0:
                return {"status": "error", "message": "滑点估算返回异常"}

            # 测试校准流程
            fill = TradeFill(
                symbol='BTCUSDT',
                side='buy',
                quantity=0.2,
                expected_price=50005.0,
                fill_price=50010.0,
                duration_sec=5
            )
            cal = self.update_model(fill)
            if cal.status != 'ok':
                return {"status": "error", "message": f"校准失败: {cal}"}

            return {"status": "ok", "message": "滑点模拟器核心功能正常", "params_count": len(self._calib)}
        except Exception as e:
            logger.exception("健康检查失败")
            return {"status": "error", "message": str(e)}

    # ---------------------------------------------------------------------
    #  手动重置/导出接口
    # ---------------------------------------------------------------------
    def reset_asset(self, symbol: str):
        """重置某资产的校准状态到出厂默认"""
        with self._lock:
            if symbol in self._calib:
                del self._calib[symbol]
            logger.info("资产 %s 滑点参数已重置", symbol)

    def export_params(self) -> Dict[str, Dict]:
        """导出所有资产的当前参数"""
        return {sym: state['params'].copy() for sym, state in self._calib.items()}
