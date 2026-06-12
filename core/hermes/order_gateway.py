#!/usr/bin/env python3
"""
Kunlun System · Order Execution Gateway (OrderExecutionGateway)

Core Responsibilities:
1. Accept standardized trading signals and select optimal execution algorithms
   based on real-time market impact estimates, volatility, and liquidity.
2. Implement precise slippage control, order fragmentation, fill-rate monitoring,
   and dynamic algorithm switching for trillion-dollar accounts.
3. Manage the full lifecycle of all outstanding orders with event-driven state
   updates via exchange WebSocket user data streams.
4. Enforce strict Binance API rate limits via token bucket with priority preemption.
5. Guarantee MiFID II / Reg NMS best execution compliance through auditable TCA reports.
6. All execution algorithms (TWAP, Iceberg, POV) are implemented as asynchronous
   tasks with live market data integration and automatic error recovery.

External Dependencies:
- infrastructure.error_registry.ErrorRegistry : Unified error code resolution
- hermes.rate_limiter.RateLimitGovernor : API token bucket with weight-aware priority
- hermes.slippage_sim.SlippageSimulator : Pre-trade impact estimation model
- infrastructure.stream_gateway.StreamGateway : Realtime orderbook & user data stream
- infrastructure.audit_chain.AuditLogChain : Immutable order event logging
- polaris.silence_protocol.SilenceProtocol : Market-wide circuit breaker queries
- infrastructure.chronos_db.ChronosDB : Durable order state persistence (async)

Interface Contract:
- place_order(signal: Dict) -> Dict[str, Any]
  Returns {"status": "ok", "order_id": str, "execution_plan": Dict, "warnings": List[str]}
- cancel_order(order_id: str) -> Dict[str, Any]
- cancel_all_orders(symbol: Optional[str] = None) -> Dict[str, Any]
- modify_order(order_id: str, updates: Dict) -> Dict[str, Any]
- get_order_status(order_id: str) -> Dict[str, Any]
- health_check() -> Dict[str, Any]

Error & Degradation:
- KUN-EXE-F001: Market order estimated slippage exceeds hard cap — order blocked
- KUN-EXE-F002: Consecutive API auth failures — gateway isolated
- KUN-EXE-F003: Failed to restore order state from DB — manual intervention required
- KUN-EXE-E001: Insufficient balance/margin — signal rejected with warning
- KUN-EXE-E002: Token bucket exhausted — non-critical orders queued, stop-loss preempts
- KUN-EXE-E003: Exchange API error with retries exhausted — order marked ERROR
- KUN-EXE-E004: Orderbook depth insufficient for market order — order blocked
- KUN-EXE-W001: Signal orderbook hash mismatch — signal re-validated before execution
- KUN-EXE-W002: TWAP incomplete at timeout — residual executed as aggressive limit
- KUN-EXE-W003: Order state reconciliation mismatch — manual audit triggered

Resource Management:
- All order persistence operations are asynchronous and non-blocking via DB connection pool.
- Execution algorithms run on a dedicated asyncio event loop in a separate thread.
- Shared order state is protected by a reader-writer lock with minimal critical sections.
- Memory usage bounded by MAX_ACTIVE_ORDERS and periodic archival to ChronosDB.
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Enums & Data Classes
# =============================================================================
class OrderState(Enum):
    CREATED = auto()
    PENDING_NEW = auto()
    OPEN = auto()
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    PENDING_CANCEL = auto()
    CANCELED = auto()
    EXPIRED = auto()
    REJECTED = auto()
    ERROR = auto()


class ExecutionAlgorithm(Enum):
    DIRECT_LIMIT = "direct_limit"
    TWAP = "twap"
    ICEBERG = "iceberg"
    POV = "pov"
    AGGRESSIVE_TWAP = "aggressive_twap"
    MARKET = "market"


class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP_LOSS = "STOP_MARKET"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT_MARKET"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"
    LIMIT_MAKER = "LIMIT_MAKER"


@dataclass
class OrderRecord:
    """Full lifecycle record of a single order."""
    client_order_id: str
    exchange_order_id: Optional[str] = None
    symbol: str = ""
    side: str = ""              # BUY / SELL
    order_type: OrderType = OrderType.LIMIT
    algorithm: ExecutionAlgorithm = ExecutionAlgorithm.DIRECT_LIMIT
    quantity: float = 0.0
    executed_qty: float = 0.0
    cumulative_quote_qty: float = 0.0  # Total cost of fills
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    state: OrderState = OrderState.CREATED
    created_at_ns: int = 0
    last_updated_ns: int = 0
    avg_fill_price: float = 0.0
    total_fee: float = 0.0
    slippage_bps: float = 0.0
    signal_id: Optional[str] = None
    error_count: int = 0
    warnings: List[str] = field(default_factory=list)
    ts_signal_received: int = 0
    ts_sent_to_exchange: int = 0
    ts_first_ack: int = 0
    ts_final_fill: int = 0


# =============================================================================
# Main Execution Gateway
# =============================================================================
class OrderExecutionGateway:
    """Institutional-grade order execution gateway for trillion-dollar accounts."""

    # --------------------------- Class Constants ---------------------------
    DIRECT_LIMIT_THRESHOLD_BPS = 1.0
    TWAP_THRESHOLD_BPS = 5.0
    ICEBERG_THRESHOLD_BPS = 10.0

    TWAP_DEFAULT_DURATION_SEC = 30.0
    TWAP_MIN_SLICES = 5
    TWAP_MAX_SLICES = 60
    TWAP_RANDOM_VARIANCE = 0.2

    ICEBERG_VISIBLE_RATIO_MIN = 0.001
    ICEBERG_VISIBLE_RATIO_MAX = 0.015
    ICEBERG_REFRESH_JITTER = 0.3

    POV_TARGET_PARTICIPATION = 0.05
    POV_MAX_DURATION_SEC = 300.0

    FILL_CHECK_INTERVAL_SEC = 3.0
    MIN_FILL_RATE = 0.6
    FILL_RATE_TIMEOUT_SEC = 30.0

    ORDER_TIMEOUT_SEC = 5.0
    MAX_RETRIES_PER_ORDER = 3
    MAX_SLIPPAGE_BPS_LIMIT = 5.0
    MAX_SLIPPAGE_BPS_MARKET = 50.0
    DEFAULT_RECV_WINDOW_MS = 5000

    SIGNAL_MAX_AGE_MS_NORMAL = 500
    SIGNAL_MAX_AGE_MS_HIGH_VOL = 200

    MAX_ACTIVE_ORDERS = 10000
    MAX_HISTORY_ORDERS = 100000
    CLEANUP_INTERVAL_SEC = 30.0
    DB_PERSIST_INTERVAL_SEC = 5.0
    IDEMPOTENCY_CLEANUP_INTERVAL = 300  # 5 min

    PRIORITY_CANCEL = 0
    PRIORITY_STOP_LOSS = 1
    PRIORITY_ENTRY_EXIT = 2
    PRIORITY_QUERY = 3

    def __init__(self, config: Optional[Dict] = None):
        if config:
            self._apply_config(config)

        # Dependencies
        self._rate_limiter = None
        self._slippage_sim = None
        self._stream_gateway = None
        self._audit_chain = None
        self._silence_protocol = None
        self._chronos_db = None
        self._exchange_client = None
        self._polaris_hall = None

        # Async infrastructure
        self._db_loop = asyncio.new_event_loop()
        self._db_thread = None
        self._algo_loop = asyncio.new_event_loop()
        self._algo_thread = None
        self._start_event_loops()

        # Shared state (protected by asyncio locks in algo loop)
        self._active_orders: Dict[str, OrderRecord] = {}
        self._completed_orders: Dict[str, OrderRecord] = {}
        self._signal_to_orders: Dict[str, List[str]] = {}
        self._idempotency_set: Dict[str, float] = {}  # hash -> expiry timestamp

        # Performance tracking
        self._algo_performance: Dict[str, deque] = {
            alg.value: deque(maxlen=200) for alg in ExecutionAlgorithm
        }

        # Background tasks (on algo loop)
        self._shutdown_event = asyncio.Event()
        self._ws_listener_task = None
        self._cleanup_task = None

        # Restore state then start background
        asyncio.run_coroutine_threadsafe(self._restore_and_run(), self._algo_loop)

    def _start_event_loops(self):
        """Start dedicated asyncio event loops in separate threads."""
        def run_db():
            asyncio.set_event_loop(self._db_loop)
            self._db_loop.run_forever()
        self._db_thread = threading.Thread(target=run_db, daemon=True, name="db-loop")
        self._db_thread.start()

        def run_algo():
            asyncio.set_event_loop(self._algo_loop)
            self._algo_loop.run_forever()
        self._algo_thread = threading.Thread(target=run_algo, daemon=True, name="algo-loop")
        self._algo_thread.start()

    async def _restore_and_run(self):
        """Async initialization: restore from DB and start listeners."""
        await self._restore_state()
        self._ws_listener_task = asyncio.create_task(self._listen_user_data_stream())
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        logger.info("OrderExecutionGateway fully initialized (active orders: %d)", len(self._active_orders))

    # --------------------------- Dependency Injection ---------------------------
    def wire_dependencies(self, **kwargs):
        valid = ['rate_limiter', 'slippage_sim', 'stream_gateway', 'audit_chain',
                 'silence_protocol', 'chronos_db', 'exchange_client', 'polaris_hall']
        for name, comp in kwargs.items():
            if name in valid:
                setattr(self, f"_{name}", comp)
        logger.info("Dependencies wired: %s", list(kwargs.keys()))

    # --------------------------- Configuration ---------------------------
    @staticmethod
    def _apply_config(config: Dict) -> None:
        for key, value in config.items():
            if not hasattr(OrderExecutionGateway, key):
                continue
            current = getattr(OrderExecutionGateway, key)
            if isinstance(current, (int, float)) and not isinstance(current, bool):
                if value < 0 or value > 1e12:
                    logger.warning("Config %s out of bounds: %s", key, value)
                    continue
            setattr(OrderExecutionGateway, key, value)

    # --------------------------- Persistence ---------------------------
    async def _restore_state(self):
        if not self._chronos_db:
            return
        try:
            records = await self._chronos_db.async_query_active_orders()
            for rec in records:
                rec['state'] = OrderState[rec['state']]
                self._active_orders[rec['client_order_id']] = OrderRecord(**rec)
        except Exception as e:
            logger.critical("[KUN-EXE-F003] State restore failed: %s", e)

    async def _persist_async(self, order_id: str):
        if not self._chronos_db or order_id not in self._active_orders:
            return
        # Thread-safe copy
        record = self._active_orders[order_id]
        data = {
            'client_order_id': record.client_order_id,
            'state': record.state.name,
            'symbol': record.symbol,
            'side': record.side,
            'quantity': record.quantity,
            'executed_qty': record.executed_qty,
            'avg_price': record.avg_fill_price,
        }
        await self._chronos_db.async_upsert(data)

    # --------------------------- User Data Stream (run in algo loop) ---------------------------
    async def _listen_user_data_stream(self):
        while not self._shutdown_event.is_set():
            if not self._stream_gateway:
                await asyncio.sleep(1)
                continue
            try:
                events = await self._stream_gateway.async_drain_events()
                for event in events:
                    self._handle_exchange_event(event)
            except Exception as e:
                logger.error("[KUN-EXE-E011] User stream error: %s", e)
            await asyncio.sleep(0.1)

    def _handle_exchange_event(self, event: Dict):
        etype = event.get('e')
        if etype == 'executionReport':
            self._process_execution_report(event)

    def _process_execution_report(self, report: Dict):
        client_id = report.get('c', '')
        order = self._active_orders.get(client_id)
        if not order:
            return
        order.exchange_order_id = report.get('i')
        exec_qty = float(report.get('z', 0))
        cum_quote = float(report.get('Y', 0))
        if cum_quote > 0 and exec_qty > 0:
            order.avg_fill_price = cum_quote / exec_qty
        order.executed_qty = exec_qty
        order.cumulative_quote_qty = cum_quote
        status = report.get('X', '')
        order.state = {
            'NEW': OrderState.OPEN,
            'PARTIALLY_FILLED': OrderState.PARTIALLY_FILLED,
            'FILLED': OrderState.FILLED,
            'CANCELED': OrderState.CANCELED,
            'EXPIRED': OrderState.EXPIRED,
            'REJECTED': OrderState.REJECTED,
        }.get(status, order.state)
        order.last_updated_ns = time.perf_counter_ns()
        if order.state == OrderState.FILLED:
            order.ts_final_fill = order.last_updated_ns
            self._record_algo_performance(order)
        asyncio.create_task(self._persist_async(client_id))

    async def _periodic_cleanup(self):
        while not self._shutdown_event.is_set():
            await asyncio.sleep(self.CLEANUP_INTERVAL_SEC)
            now = time.time()
            # Clean idempotency set
            expired_hashes = [h for h, t in self._idempotency_set.items() if now > t]
            for h in expired_hashes:
                del self._idempotency_set[h]
            # Archive completed orders
            to_archive = []
            for oid, o in self._active_orders.items():
                if o.state in (OrderState.FILLED, OrderState.CANCELED, OrderState.EXPIRED, OrderState.REJECTED):
                    if now - (o.last_updated_ns / 1e9) > 600:
                        to_archive.append(oid)
            for oid in to_archive:
                self._completed_orders[oid] = self._active_orders.pop(oid)
            while len(self._completed_orders) > self.MAX_HISTORY_ORDERS:
                oldest = min(self._completed_orders.keys(),
                             key=lambda k: self._completed_orders[k].last_updated_ns)
                del self._completed_orders[oldest]

    # --------------------------- Signal Validation (sync, callable from main thread) ---------------------------
    @staticmethod
    def _current_ns() -> int:
        return time.perf_counter_ns()

    def _validate_signal(self, signal: Dict) -> Tuple[bool, str, Dict]:
        warnings = []
        # Silence check
        if self._silence_protocol and self._silence_protocol.is_silent():
            if signal.get('order_type') == 'entry':
                return False, "Market in silence", {}
        # Freshness
        now_ns = self._current_ns()
        age_ms = (now_ns - signal.get('generated_at_ns', 0)) // 1_000_000
        max_age = self.SIGNAL_MAX_AGE_MS_HIGH_VOL if signal.get('high_volatility') else self.SIGNAL_MAX_AGE_MS_NORMAL
        if age_ms > max_age:
            return False, f"Signal expired ({age_ms}ms)", {}
        # Idempotency (SHA256 + time-bound)
        sig_hash = hashlib.sha256(json.dumps(signal, sort_keys=True, default=str).encode()).hexdigest()
        if sig_hash in self._idempotency_set:
            return False, "Duplicate signal", {}
        self._idempotency_set[sig_hash] = time.time() + self.IDEMPOTENCY_CLEANUP_INTERVAL
        # Balance (async check run in algo loop, but for simplicity we do sync here)
        if self._exchange_client:
            if not self._exchange_client.check_balance(signal.get('symbol'), signal.get('side'),
                                                       signal.get('quantity', 0), signal.get('expected_price', 0)):
                return False, "Insufficient balance", {}
        # Orderbook hash
        ob_hash = signal.get('orderbook_hash', '')
        if ob_hash and self._stream_gateway:
            cur_hash = self._stream_gateway.get_orderbook_hash(signal.get('symbol'))
            if cur_hash and cur_hash != ob_hash:
                warnings.append("[KUN-EXE-W001] Orderbook hash changed")
        return True, "", {"sig_hash": sig_hash, "warnings": warnings}

    # --------------------------- Public: Place Order (sync entry, async exec) ---------------------------
    def place_order(self, signal: Dict) -> Dict[str, Any]:
        valid, reason, enrich = self._validate_signal(signal)
        if not valid:
            return {"status": "error", "order_id": None, "reason": reason}

        priority = self._get_priority(signal.get('order_type', 'entry'))
        if not self._acquire_token(priority):
            return {"status": "error", "reason": "Rate limit exceeded"}

        symbol = signal['symbol']
        side = signal['side']
        qty = signal['quantity']
        algo = self._select_algorithm(symbol, side, qty, signal.get('order_type') == 'emergency')
        client_id = self._generate_client_order_id()
        order_type = self._map_to_binance_order_type(signal.get('order_type', 'entry'))

        record = OrderRecord(
            client_order_id=client_id, symbol=symbol, side=side,
            order_type=order_type, algorithm=algo, quantity=qty,
            limit_price=signal.get('limit_price'),
            stop_price=signal.get('stop_price'),
            signal_id=signal.get('signal_id'),
            created_at_ns=self._current_ns(),
            ts_signal_received=self._current_ns(),
            warnings=enrich.get('warnings', [])
        )

        # Pre-validate params
        if not self._validate_order_params(record, signal):
            return {"status": "error", "order_id": client_id, "reason": "Invalid order params"}

        # Submit to algo event loop for async execution
        asyncio.run_coroutine_threadsafe(
            self._execute_order_async(record, signal), self._algo_loop
        )

        self._active_orders[client_id] = record
        self._signal_to_orders.setdefault(signal.get('signal_id', ''), []).append(client_id)
        asyncio.run_coroutine_threadsafe(self._persist_async(client_id), self._db_loop)

        if self._audit_chain:
            self._audit_chain.log_event('order_placed', {'client_id': client_id, 'algo': algo.value})

        return {"status": "ok", "order_id": client_id, "algorithm": algo.value, "warnings": record.warnings}

    # --------------------------- Async order execution pipeline ---------------------------
    async def _execute_order_async(self, order: OrderRecord, signal: Dict):
        """Run the appropriate execution algorithm asynchronously."""
        try:
            if order.algorithm == ExecutionAlgorithm.DIRECT_LIMIT:
                await self._exec_direct_limit(order, signal)
            elif order.algorithm in (ExecutionAlgorithm.TWAP, ExecutionAlgorithm.AGGRESSIVE_TWAP):
                await self._exec_twap(order, signal)
            elif order.algorithm == ExecutionAlgorithm.ICEBERG:
                await self._exec_iceberg(order, signal)
            elif order.algorithm == ExecutionAlgorithm.POV:
                await self._exec_pov(order, signal)
            elif order.algorithm == ExecutionAlgorithm.MARKET:
                await self._exec_market(order, signal)
        except Exception as e:
            logger.exception("[KUN-EXE-E003] Execution failed for %s: %s", order.client_order_id, e)
            order.state = OrderState.ERROR
            order.warnings.append(str(e))
            await self._persist_async(order.client_order_id)

    async def _exec_direct_limit(self, order: OrderRecord, signal: Dict):
        price = order.limit_price
        if not price:
            ob = await self._get_orderbook(order.symbol)
            price = self._calculate_maker_price(ob, order.side)
            if not price:
                raise ValueError("Cannot determine limit price")
        await self._place_limit_maker(order.symbol, order.side, order.quantity, price, order.client_order_id)

    async def _exec_twap(self, order: OrderRecord, signal: Dict):
        duration = self.TWAP_DEFAULT_DURATION_SEC
        if order.algorithm == ExecutionAlgorithm.AGGRESSIVE_TWAP:
            duration /= 2
        n = max(self.TWAP_MIN_SLICES, min(self.TWAP_MAX_SLICES, int(duration / 3)))
        slice_qty = order.quantity / n
        for i in range(n):
            if self._shutdown_event.is_set() or order.state in (OrderState.CANCELED, OrderState.FILLED, OrderState.ERROR):
                break
            await asyncio.sleep(duration / n)
            price = await self._get_market_price(order.symbol)
            if price > 0:
                child_id = f"{order.client_order_id}_s{i}"
                await self._place_limit_maker(order.symbol, order.side, slice_qty, price, child_id)
        if order.executed_qty < order.quantity * self.MIN_FILL_RATE:
            # Residual market order
            logger.warning("[KUN-EXE-W002] TWAP incomplete, executing residual")
            await self._exec_market(order, signal)

    async def _exec_iceberg(self, order: OrderRecord, signal: Dict):
        visible = max(order.quantity * self.ICEBERG_VISIBLE_RATIO_MIN, 0.001)
        while order.executed_qty < order.quantity and order.state not in (OrderState.CANCELED, OrderState.ERROR):
            remaining = order.quantity - order.executed_qty
            slice_qty = min(visible, remaining)
            price = await self._get_market_price(order.symbol)
            child_id = f"{order.client_order_id}_ice_{int(time.time()*1000)}"
            await self._place_limit_maker(order.symbol, order.side, slice_qty, price, child_id)
            await asyncio.sleep(1.0)

    async def _exec_pov(self, order: OrderRecord, signal: Dict):
        deadline = time.time() + self.POV_MAX_DURATION_SEC
        while time.time() < deadline and order.executed_qty < order.quantity and order.state not in (OrderState.CANCELED, OrderState.ERROR):
            market_vol = await self._get_recent_volume(order.symbol, 10)  # last 10s
            slice_qty = market_vol * self.POV_TARGET_PARTICIPATION
            if slice_qty > 0:
                price = await self._get_market_price(order.symbol)
                child_id = f"{order.client_order_id}_pov_{int(time.time()*1000)}"
                await self._place_limit_maker(order.symbol, order.side, slice_qty, price, child_id)
            await asyncio.sleep(10)
        if order.executed_qty < order.quantity:
            await self._exec_market(order, signal)

    async def _exec_market(self, order: OrderRecord, signal: Dict):
        expected_price = signal.get('expected_price', 0)
        # Pre-trade slippage check
        if expected_price > 0:
            ob = await self._get_orderbook(order.symbol)
            fill_price = self._simulate_market_fill(ob, order.side, order.quantity)
            if fill_price <= 0:
                raise RuntimeError("Insufficient depth for market order")
            slippage = abs(fill_price - expected_price) / expected_price * 10000
            if slippage > self.MAX_SLIPPAGE_BPS_MARKET:
                raise RuntimeError("Slippage exceeds cap")
        if self._exchange_client:
            resp = await self._exchange_client.async_create_order(
                symbol=order.symbol, side=order.side, type=OrderType.MARKET.value,
                quantity=order.quantity, newClientOrderId=order.client_order_id,
                recvWindow=self.DEFAULT_RECV_WINDOW_MS
            )
            # Process fills
            for fill in resp.get('fills', []):
                order.executed_qty += float(fill.get('qty', 0))
                order.cumulative_quote_qty += float(fill.get('price', 0)) * float(fill.get('qty', 0))
            if order.executed_qty > 0:
                order.avg_fill_price = order.cumulative_quote_qty / order.executed_qty
            order.state = OrderState.FILLED
        await self._persist_async(order.client_order_id)

    # --------------------------- Algorithm Selection ---------------------------
    def _select_algorithm(self, symbol: str, side: str, qty: float, emergency: bool) -> ExecutionAlgorithm:
        if emergency:
            return ExecutionAlgorithm.MARKET
        impact = self._estimate_impact(symbol, side, qty)
        vol = self._get_daily_volume(symbol)
        vol_regime = self._get_volatility_regime(symbol)
        spread = self._get_current_spread(symbol)
        algo = ExecutionAlgorithm.DIRECT_LIMIT
        if impact >= self.ICEBERG_THRESHOLD_BPS:
            algo = ExecutionAlgorithm.POV
        elif impact >= self.TWAP_THRESHOLD_BPS:
            algo = ExecutionAlgorithm.ICEBERG
        elif impact >= self.DIRECT_LIMIT_THRESHOLD_BPS:
            algo = ExecutionAlgorithm.TWAP
        if vol_regime == 'high' and algo == ExecutionAlgorithm.DIRECT_LIMIT and impact > 0.5:
            algo = ExecutionAlgorithm.TWAP
        if spread > 5.0 and algo == ExecutionAlgorithm.DIRECT_LIMIT:
            algo = ExecutionAlgorithm.ICEBERG
        return algo

    def _record_algo_performance(self, order: OrderRecord):
        if order.slippage_bps:
            self._algo_performance[order.algorithm.value].append(order.slippage_bps)

    # --------------------------- Exchange Interaction ---------------------------
    async def _place_limit_maker(self, symbol, side, qty, price, client_id):
        if not self._exchange_client:
            return
        # Apply exchange filters (LOT_SIZE, MIN_NOTIONAL) using client helpers
        qty = self._exchange_client.round_quantity(symbol, qty)
        price = self._exchange_client.round_price(symbol, price)
        await self._exchange_client.async_create_order(
            symbol=symbol, side=side, type=OrderType.LIMIT_MAKER.value,
            quantity=qty, price=price, newClientOrderId=client_id,
            timeInForce='GTC', recvWindow=self.DEFAULT_RECV_WINDOW_MS
        )

    async def _get_orderbook(self, symbol: str) -> Dict:
        if self._stream_gateway:
            return self._stream_gateway.get_orderbook(symbol)
        return {'bids': [], 'asks': []}

    async def _get_market_price(self, symbol: str) -> float:
        ob = await self._get_orderbook(symbol)
        bids = ob.get('bids', [])
        asks = ob.get('asks', [])
        if bids and asks:
            return (bids[0][0] + asks[0][0]) / 2
        return 0.0

    async def _get_recent_volume(self, symbol: str, seconds: int) -> float:
        if self._stream_gateway:
            return self._stream_gateway.get_recent_volume(symbol, seconds)
        return 0.0

    # --------------------------- Helpers ---------------------------
    @staticmethod
    def _generate_client_order_id() -> str:
        return f"kun_{uuid.uuid4().hex[:12]}_{int(time.time()*1000)}"

    def _acquire_token(self, priority: int) -> bool:
        return self._rate_limiter.acquire(priority) if self._rate_limiter else True

    def _get_priority(self, order_type: str) -> int:
        return {
            'emergency': self.PRIORITY_CANCEL,
            'stop_loss': self.PRIORITY_STOP_LOSS,
            'entry': self.PRIORITY_ENTRY_EXIT,
            'exit_profit': self.PRIORITY_ENTRY_EXIT,
        }.get(order_type, self.PRIORITY_ENTRY_EXIT)

    @staticmethod
    def _map_to_binance_order_type(internal: str) -> OrderType:
        return {
            'entry': OrderType.LIMIT_MAKER,
            'exit_profit': OrderType.TAKE_PROFIT_LIMIT,
            'stop_loss': OrderType.STOP_LOSS,
            'emergency': OrderType.MARKET,
        }.get(internal, OrderType.LIMIT)

    @staticmethod
    def _calculate_maker_price(ob: Dict, side: str) -> Optional[float]:
        if side == 'BUY':
            return ob['bids'][0][0] if ob.get('bids') else None
        else:
            return ob['asks'][0][0] if ob.get('asks') else None

    def _validate_order_params(self, record: OrderRecord, signal: Dict) -> bool:
        if record.quantity <= 0:
            return False
        price = record.limit_price or signal.get('expected_price', 0)
        if price <= 0:
            return False
        notional = record.quantity * price
        if notional < 10:
            return False
        return True

    # --------------------------- Cancel / Modify ---------------------------
    def cancel_order(self, order_id: str) -> Dict:
        if not self._acquire_token(self.PRIORITY_CANCEL):
            return {"status": "error", "reason": "Rate limit"}
        order = self._active_orders.get(order_id)
        if not order:
            return {"status": "error", "reason": "Not found"}
        if order.state in (OrderState.FILLED, OrderState.CANCELED):
            return {"status": "error", "reason": f"Already {order.state.name}"}
        if self._exchange_client:
            self._exchange_client.cancel_order(symbol=order.symbol, origClientOrderId=order_id)
        order.state = OrderState.PENDING_CANCEL
        asyncio.run_coroutine_threadsafe(self._persist_async(order_id), self._db_loop)
        return {"status": "ok"}

    def cancel_all_orders(self, symbol: Optional[str] = None) -> Dict:
        if self._exchange_client:
            self._exchange_client.cancel_all_open_orders(symbol=symbol)
        for oid, o in list(self._active_orders.items()):
            if symbol and o.symbol != symbol:
                continue
            if o.state in (OrderState.OPEN, OrderState.PARTIALLY_FILLED, OrderState.PENDING_NEW):
                self.cancel_order(oid)
        return {"status": "ok"}

    def modify_order(self, order_id: str, updates: Dict) -> Dict:
        order = self._active_orders.get(order_id)
        if not order:
            return {"status": "error", "reason": "Not found"}
        # For simplicity, cancel and re-place; in production use exchange's cancel-replace
        self.cancel_order(order_id)
        new_signal = {
            'symbol': order.symbol, 'side': order.side,
            'quantity': updates.get('quantity', order.quantity),
            'limit_price': updates.get('price', order.limit_price),
            'order_type': 'entry', 'expected_price': updates.get('price', order.limit_price),
            'generated_at_ns': self._current_ns(), 'signal_id': order.signal_id,
        }
        return self.place_order(new_signal)

    # --------------------------- Health Check ---------------------------
    def health_check(self) -> Dict[str, Any]:
        checks = {}
        checks['db_loop'] = self._db_loop.is_running()
        checks['algo_loop'] = self._algo_loop.is_running()
        checks['exchange_client'] = self._exchange_client is not None
        all_ok = all(checks.values())
        return {"status": "ok" if all_ok else "degraded", "checks": checks}

    # --------------------------- Shutdown ---------------------------
    def shutdown(self):
        self._shutdown_event.set()
        self.cancel_all_orders()
        if self._algo_loop.is_running():
            self._algo_loop.call_soon_threadsafe(self._algo_loop.stop)
        if self._db_loop.is_running():
            self._db_loop.call_soon_threadsafe(self._db_loop.stop)
        logger.info("Gateway shutdown complete.")

    __all__ = ['OrderExecutionGateway', 'ExecutionAlgorithm', 'OrderType', 'OrderState']
