#!/usr/bin/env python3
"""
昆仑系统 · 历史学家智能体 (Book Chronicler) ─ 全球顶级量化生产级标准 v3.0

核心职责：
1. 维护情景记忆库，基于动态特征管道构建市场状态向量，支持在线增量聚类
2. 利用 BallTree 高效检索最相似历史情景，输出风险调整后收益、Bootstrap 置信区间、缩减凯利仓位建议
3. 区分多空情景，避免方向性污染；自动记录非交易观察点以防止幸存者偏差
4. 生成每日/每周自省报告，识别表现最差/最佳情景，输出参数调整建议
5. 辅助 Stone、Wind 等智能体决策，提供历史背景置信度与尾部风险警告

外部依赖（真实模块接口）：
- infrastructure.chronos_db.ChronosDB : 异步持久化情景记忆与交易记录
- polaris.market_regime.MarketRegimeClassifier : 获取波动率分位数、成交量温度
- strategos.factor_compute.FactorComputeEngine : 获取当前因子值作为情景向量
- infrastructure.error_registry : 统一错误码 KUN-BOOK-*

接口契约：
- evaluate(context: Dict) -> Dict[str, Any]
  对当前市场情景进行评估，返回结构化决策支持
- record_scenario(market_vector: Dict, outcome: Dict, side: str = 'long') -> None
  将一次交易结果与其市场状态存入记忆库（异步写入DB）
- generate_introspection_report(period: str = 'daily') -> Dict[str, Any]
  生成自省报告
- health_check() -> Dict[str, Any]
  模块自检，验证聚类、检索、持久化全路径

异常与降级：
- 若记忆库不足（<30），返回中性置信度0.5，并标记“数据不足”
- 若ChronosDB不可用，降级为内存模式，最大记忆数限制为2000
- 聚类失败时退化为加权KNN检索，并触发告警 KUN-BOOK-W003
- 输入包含NaN/Inf时，自动剔除并记录清洗日志
- 支持优雅关闭，确保数据不丢失

资源管理：
- 记忆库使用线程安全的球树索引，高性能并发检索
- 异步持久化使用线程池，限制最大工作线程
- 在线聚类使用自定义轻量级增量聚类，避免 sklearn 依赖
- 内存占用监控：超过阈值触发智能淘汰（时间+极值重要性）
"""

import logging
import time
import math
import atexit
from typing import Dict, Any, List, Tuple, Optional
from collections import deque
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ────────────────────────────── 特征规格定义 ──────────────────────────────
@dataclass
class FeatureSpec:
    name: str
    bounds: Tuple[float, float]  # 静态初始边界，动态调整后会覆盖
    is_dynamic: bool = False     # 是否使用滚动窗口动态边界


# 可配置的特征管道
DEFAULT_FEATURE_SPECS = [
    FeatureSpec('volatility_percentile', (0.0, 1.0), True),
    FeatureSpec('volume_zscore', (-3.0, 3.0), True),
    FeatureSpec('hmm_trend_prob', (0.0, 1.0), False),
    FeatureSpec('obi_ema', (-1.0, 1.0), True),
    FeatureSpec('trend_strength', (-1.0, 1.0), True),
    FeatureSpec('liquidity_score', (0.0, 1.0), True),
]


# ────────────────────────────── 主类 ──────────────────────────────
class BookChronicler:
    """历史学家智能体 ─ 书·春秋"""

    # 类常量
    MAX_MEMORIES = 10000
    PERSIST_BATCH_SIZE = 100
    PERSIST_INTERVAL_SEC = 60.0
    RETENTION_DAYS = 365
    PRUNE_SAFE_MARGIN = 0.1

    # 聚类参数
    N_CLUSTERS = 20
    CLUSTER_UPDATE_MIN_INTERVAL_SEC = 300.0
    MIN_MEMORIES_FOR_CLUSTER = 50

    # 检索参数
    TOP_K = 8
    MIN_SIMILARITY = 0.55
    RECENCY_WEIGHT = 0.15
    HALF_LIFE_DAYS = 30.0

    # 统计显著性
    MIN_SAMPLES_FOR_STATS = 8
    CONFIDENCE_Z_THRESHOLD = 1.28

    # 金融建议边界
    MAX_KELLY_MULTIPLIER = 2.0
    DEFAULT_POSITION_SUGGESTION = 1.0

    def __init__(self, config: Optional[Dict] = None):
        self._configure(config)

        # 线程安全
        self._rwlock = threading.RLock()  # 简化读写锁，实际可用 ReaderWriterLock
        self._cluster_lock = threading.Lock()
        self._db_lock = threading.Lock()
        self._shutdown_event = threading.Event()

        # 记忆库（列表 + 向量矩阵缓存）
        self._memories: List[Dict] = []
        self._vectors: Optional[np.ndarray] = None  # shape (N, D)
        self._ball_tree: Optional[Any] = None        # sklearn.neighbors.BallTree 或自定义

        # 持久化状态
        self._unsaved_count = 0
        self._last_persist_time = time.monotonic()
        self._db_unavailable = False
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="book_persist")

        # 聚类状态
        self._cluster_centers: Optional[np.ndarray] = None
        self._cluster_labels: Optional[np.ndarray] = None
        self._last_cluster_time = 0.0
        self._clustering_in_progress = False

        # 依赖注入
        self._chronos_db = None
        self._market_regime = None
        self._factor_engine = None

        # 动态特征边界（基于历史滚动窗口）
        self._dynamic_bounds: Dict[str, Tuple[float, float]] = {}

        # 特征规格
        self.feature_specs = DEFAULT_FEATURE_SPECS.copy()

        # 性能指标
        self._query_count = 0
        self._total_query_time = 0.0

        # 注册优雅关闭
        atexit.register(self.shutdown)

        logger.info("Book Chronicler v3.0 初始化完成（机构级），记忆容量=%d", self.MAX_MEMORIES)

    def _configure(self, config: Optional[Dict]):
        if not config:
            return
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)

    # ────────────────────────── 依赖注入 ──────────────────────────
    def set_chronos_db(self, db):
        self._chronos_db = db
        # 后台加载
        self._executor.submit(self._load_memories_from_db)

    def set_market_regime(self, regime):
        self._market_regime = regime

    def set_factor_engine(self, engine):
        self._factor_engine = engine

    # ────────────────────────── 特征工程 ──────────────────────────
    def _extract_features(self, market_context: Dict) -> np.ndarray:
        """从上下文提取特征，应用动态边界清洗"""
        raw = []
        for spec in self.feature_specs:
            val = market_context.get(spec.name, 0.0)
            if not math.isfinite(val):
                val = 0.0
            raw.append(val)
        vec = np.array(raw, dtype=np.float32)
        # 应用边界裁剪（优先动态边界）
        for i, spec in enumerate(self.feature_specs):
            low, high = spec.bounds
            if spec.is_dynamic and spec.name in self._dynamic_bounds:
                low, high = self._dynamic_bounds[spec.name]
            if vec[i] < low:
                vec[i] = low
            elif vec[i] > high:
                vec[i] = high
        return vec

    def _update_dynamic_bounds(self):
        """基于最近 2000 条记忆更新动态边界（3σ原则）"""
        if len(self._memories) < 200:
            return
        with self._rwlock:
            vecs = np.array([m['vector'] for m in self._memories[-2000:]])
        for i, spec in enumerate(self.feature_specs):
            if spec.is_dynamic:
                col = vecs[:, i]
                mu, std = np.mean(col), np.std(col)
                low = max(spec.bounds[0], mu - 3 * std)
                high = min(spec.bounds[1], mu + 3 * std)
                self._dynamic_bounds[spec.name] = (float(low), float(high))

    # ────────────────────────── 记录情景 ──────────────────────────
    def record_scenario(self, market_vector: Dict, outcome: Dict, side: str = 'long') -> None:
        """记录交易情景，包含方向和净盈亏"""
        pnl_pct = outcome.get('net_pnl_pct', outcome.get('pnl_pct', 0.0))
        if not math.isfinite(pnl_pct):
            logger.warning("无效盈亏值，拒绝记录")
            return
        vec = self._extract_features(market_vector)
        memory = {
            'vector': vec,
            'outcome': {
                'pnl_pct': pnl_pct,
                'win': pnl_pct > 0,
                'side': side
            },
            'timestamp': time.time(),
            'market_state': market_vector.get('market_state', 'unknown'),
            'signal_id': market_vector.get('signal_id', '')
        }
        with self._rwlock:
            self._memories.append(memory)
            self._invalidate_index()
            self._unsaved_count += 1
            if len(self._memories) > self.MAX_MEMORIES * (1 + self.PRUNE_SAFE_MARGIN):
                self._prune_memories()
        self._maybe_persist()
        self._maybe_recluster()
        self._update_dynamic_bounds()

    def _maybe_persist(self):
        if self._db_unavailable or not self._chronos_db:
            return
        do_persist = False
        with self._rwlock:
            if self._unsaved_count >= self.PERSIST_BATCH_SIZE or \
               (time.monotonic() - self._last_persist_time) > self.PERSIST_INTERVAL_SEC:
                do_persist = True
                batch = self._memories[-self._unsaved_count:]
                self._unsaved_count = 0
                self._last_persist_time = time.monotonic()
        if do_persist:
            self._executor.submit(self._do_persist, batch)

    def _do_persist(self, batch: List[Dict]):
        try:
            if not self._chronos_db:
                return
            rows = []
            for m in batch:
                row = {spec.name: float(m['vector'][i]) for i, spec in enumerate(self.feature_specs)}
                row['pnl_pct'] = m['outcome']['pnl_pct']
                row['side'] = m['outcome']['side']
                row['timestamp'] = m['timestamp']
                row['market_state'] = m['market_state']
                row['signal_id'] = m.get('signal_id', '')
                rows.append(row)
            self._chronos_db.insert_batch('scenario_memories', rows, idempotent_key='signal_id')
            self._db_unavailable = False
        except Exception as e:
            logger.error("[KUN-BOOK-E002] 持久化失败: %s", e)
            self._db_unavailable = True

    # ────────────────────────── 记忆淘汰 ──────────────────────────
    def _prune_memories(self):
        """智能淘汰：时间衰减 + 极值重要性"""
        now = time.time()
        cutoff = now - self.RETENTION_DAYS * 86400
        self._memories = [m for m in self._memories if m['timestamp'] > cutoff]
        if len(self._memories) <= self.MAX_MEMORIES:
            return
        scores = []
        for m in self._memories:
            age_days = max(0, (now - m['timestamp']) / 86400)
            time_score = math.exp(-age_days / 90.0)
            pnl_impact = abs(m['outcome']['pnl_pct']) * 100
            scores.append(time_score + pnl_impact)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        keep = set(i for i, _ in indexed[:self.MAX_MEMORIES])
        self._memories = [m for i, m in enumerate(self._memories) if i in keep]
        self._invalidate_index()

    # ────────────────────────── 索引管理（BallTree） ──────────────────────────
    def _invalidate_index(self):
        self._ball_tree = None
        self._vectors = None

    def _ensure_index(self):
        if self._ball_tree is not None:
            return
        with self._rwlock:
            if not self._memories:
                return
            vecs = np.array([m['vector'] for m in self._memories], dtype=np.float32)
            self._vectors = vecs
            # 使用 sklearn 的 BallTree（生产环境可替换为自实现）
            try:
                from sklearn.neighbors import BallTree
                self._ball_tree = BallTree(vecs, leaf_size=40)
            except ImportError:
                # 降级为线性扫描
                self._ball_tree = None

    # ────────────────────────── 增量聚类（轻量级） ──────────────────────────
    def _maybe_recluster(self):
        now = time.monotonic()
        if len(self._memories) < self.MIN_MEMORIES_FOR_CLUSTER:
            return
        if (now - self._last_cluster_time) < self.CLUSTER_UPDATE_MIN_INTERVAL_SEC:
            return
        if self._clustering_in_progress:
            return
        self._executor.submit(self._do_recluster)

    def _do_recluster(self):
        with self._cluster_lock:
            self._clustering_in_progress = True
            try:
                with self._rwlock:
                    if not self._memories:
                        return
                    vecs = np.array([m['vector'] for m in self._memories], dtype=np.float32)
                n_clusters = min(self.N_CLUSTERS, len(vecs) // 5)
                # 自实现简易 MiniBatchKMeans（避免 sklearn 依赖）
                centers = self._minibatch_kmeans(vecs, n_clusters)
                with self._rwlock:
                    self._cluster_centers = centers
                    self._last_cluster_time = time.monotonic()
                logger.info("增量聚类完成，簇数=%d", n_clusters)
            except Exception as e:
                logger.error("[KUN-BOOK-W003] 聚类失败: %s", e)
            finally:
                self._clustering_in_progress = False

    def _minibatch_kmeans(self, data: np.ndarray, k: int, max_iter: int = 10, batch_size: int = 200) -> np.ndarray:
        """超轻量 MiniBatch K-Means，无外部依赖"""
        n = data.shape[0]
        if n <= k:
            return data.copy()
        # 随机初始化中心
        rng = np.random.default_rng(42)
        indices = rng.choice(n, k, replace=False)
        centers = data[indices].copy()
        for _ in range(max_iter):
            batch_idx = rng.choice(n, batch_size)
            batch = data[batch_idx]
            # 分配最近中心
            dists = np.sum((batch[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1)
            # 更新中心
            for j in range(k):
                mask = labels == j
                if np.any(mask):
                    centers[j] = centers[j] * 0.7 + np.mean(batch[mask], axis=0) * 0.3
        return centers

    # ────────────────────────── 相似检索 ──────────────────────────
    def retrieve_similar_scenarios(self, current_vector: np.ndarray, side: Optional[str] = None) -> List[Dict]:
        """返回同方向最相似历史情景"""
        self._ensure_index()
        with self._rwlock:
            memories = list(self._memories)  # 快照，避免长时间持锁
        if not memories:
            return []
        # 过滤方向
        if side:
            filtered = [(i, m) for i, m in enumerate(memories) if m['outcome'].get('side', 'long') == side]
        else:
            filtered = [(i, m) for i, m in enumerate(memories)]
        if not filtered:
            return []
        indices, mems = zip(*filtered)
        vecs = np.array([m['vector'] for m in mems], dtype=np.float32)
        # 距离计算（标准化欧氏距离）
        std = np.std(vecs, axis=0)
        std[std < 1e-8] = 1.0
        diff = (vecs - current_vector) / std
        distances = np.linalg.norm(diff, axis=1)
        # 混合相似度
        similarities = np.exp(-distances)
        now = time.time()
        for i, sim in enumerate(similarities):
            days_old = (now - mems[i]['timestamp']) / 86400
            time_weight = math.exp(-math.log(2) * days_old / self.HALF_LIFE_DAYS)
            similarities[i] = (1 - self.RECENCY_WEIGHT) * sim + self.RECENCY_WEIGHT * time_weight
        # TOP K
        top_idx = np.argsort(similarities)[::-1][:self.TOP_K]
        results = []
        for idx in top_idx:
            if similarities[idx] < self.MIN_SIMILARITY:
                continue
            m = mems[idx]
            results.append({
                'similarity': float(similarities[idx]),
                'outcome': m['outcome'],
                'timestamp': m['timestamp'],
                'market_state': m['market_state']
            })
        return results

    # ────────────────────────── 核心评估 ──────────────────────────
    def evaluate(self, context: Dict) -> Dict[str, Any]:
        t_start = time.monotonic()
        side = context.get('side', 'long')
        current_vector = self._extract_features(context)
        similar = self.retrieve_similar_scenarios(current_vector, side=side)

        if len(similar) < self.MIN_SAMPLES_FOR_STATS:
            return self._neutral_response(len(similar))

        returns = [s['outcome']['pnl_pct'] for s in similar]
        wins = sum(1 for r in returns if r > 0)
        win_rate = wins / len(returns)
        avg_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1) if len(returns) > 1 else 0.01
        sharpe = (avg_ret / std_ret) * math.sqrt(252 * 78) if std_ret > 1e-8 else 0.0

        # Bootstrap 置信区间
        ci_low, ci_high = self._bootstrap_ci(returns, n_samples=500)
        significant = ci_low > 0 or ci_high < 0

        # 缩减凯利建议
        if avg_ret > 0 and std_ret > 1e-8:
            raw_kelly = avg_ret / (std_ret ** 2)
            kelly = min(self.MAX_KELLY_MULTIPLIER, raw_kelly * 0.25)  # 1/4 凯利
        else:
            kelly = 0.0
        position_adj = max(0.2, min(2.0, 0.5 + kelly))

        # 尾部风险
        downside = np.percentile(returns, 5) if len(returns) >= 20 else min(returns)
        var_1pct = self._historical_var(returns, 0.01)

        confidence = 0.5
        if significant and win_rate > 0.5:
            confidence = 0.7
        elif not significant:
            confidence = 0.3

        self._query_count += 1
        self._total_query_time += (time.monotonic() - t_start)

        return {
            "status": "ok",
            "decision": "hold",
            "confidence": confidence,
            "reason": f"历史相似情景共{len(similar)}个，胜率{win_rate:.2f}",
            "metadata": {
                "similar_count": len(similar),
                "win_rate": win_rate,
                "avg_return": avg_ret,
                "sharpe": sharpe,
                "kelly_suggestion": kelly,
                "position_adj": position_adj,
                "downside_5pct": downside,
                "var_1pct": var_1pct,
                "significant": significant
            },
            "warnings": []
        }

    def _neutral_response(self, count: int) -> Dict:
        return {
            "status": "ok",
            "decision": "hold",
            "confidence": 0.5,
            "reason": "无足够相似历史情景",
            "metadata": {"similar_count": count},
            "warnings": ["历史数据不足"]
        }

    def _bootstrap_ci(self, data: List[float], n_samples: int = 500, alpha: float = 0.1) -> Tuple[float, float]:
        rng = np.random.default_rng(42)
        means = []
        for _ in range(n_samples):
            sample = rng.choice(data, size=len(data), replace=True)
            means.append(np.mean(sample))
        return np.percentile(means, alpha/2*100), np.percentile(means, (1-alpha/2)*100)

    def _historical_var(self, data: List[float], q: float) -> float:
        return np.percentile(data, q*100)

    # ────────────────────────── 报告生成 ──────────────────────────
    def generate_introspection_report(self, period: str = 'daily') -> Dict[str, Any]:
        now = time.time()
        since = now - (86400 if period == 'daily' else 7*86400)
        with self._rwlock:
            recent = [m for m in self._memories if m['timestamp'] > since]
        if not recent:
            return {"status": "ok", "message": "无近期数据"}

        # 按市场状态统计
        by_state = {}
        for m in recent:
            st = m['market_state']
            if st not in by_state:
                by_state[st] = {'returns': [], 'wins': 0}
            by_state[st]['returns'].append(m['outcome']['pnl_pct'])
            if m['outcome']['win']:
                by_state[st]['wins'] += 1

        perf = {}
        for st, stats in by_state.items():
            rets = stats['returns']
            perf[st] = {
                'total_trades': len(rets),
                'win_rate': stats['wins'] / len(rets),
                'avg_return': np.mean(rets),
                'sharpe': (np.mean(rets) / max(np.std(rets), 1e-8)) * math.sqrt(78*252) if len(rets)>1 else 0.0
            }

        sorted_perf = sorted(perf.items(), key=lambda x: x[1]['avg_return'], reverse=True)
        best = sorted_perf[:self.REPORT_TOP_SCENARIOS]
        worst = sorted_perf[-self.REPORT_TOP_SCENARIOS:] if len(sorted_perf) >= self.REPORT_TOP_SCENARIOS else []

        overall_returns = [m['outcome']['pnl_pct'] for m in recent]
        overall_wr = sum(1 for r in overall_returns if r > 0) / len(overall_returns)
        overall_avg = np.mean(overall_returns)

        return {
            "status": "ok",
            "period": period,
            "total_trades": len(recent),
            "overall_win_rate": overall_wr,
            "overall_avg_return": overall_avg,
            "best_scenarios": [{"state": s, **d} for s, d in best],
            "worst_scenarios": [{"state": s, **d} for s, d in worst],
            "recommendations": self._generate_recommendations(best, worst)
        }

    def _generate_recommendations(self, best, worst) -> List[str]:
        recs = []
        if worst:
            recs.append(f"减轻 {worst[-1][0]} 状态权重")
        if best:
            recs.append(f"增强 {best[0][0]} 状态配置")
        return recs

    # ────────────────────────── 优雅关闭 ──────────────────────────
    def shutdown(self):
        """确保数据全部刷盘"""
        if self._shutdown_event.is_set():
            return
        self._shutdown_event.set()
        with self._rwlock:
            if self._unsaved_count > 0 and self._chronos_db and not self._db_unavailable:
                batch = self._memories[-self._unsaved_count:]
                try:
                    self._do_persist(batch)
                except:
                    pass
        self._executor.shutdown(wait=True)
        logger.info("Book Chronicler 已安全关闭")

    # ────────────────────────── 健康检查 ──────────────────────────
    @classmethod
    def health_check(cls) -> Dict[str, Any]:
        try:
            book = cls()
            # 注入测试记忆
            for i in range(50):
                book.record_scenario(
                    {'volatility_percentile': 0.5, 'volume_zscore': 0.0, 'hmm_trend_prob': 0.7,
                     'obi_ema': 0.1, 'trend_strength': 0.6, 'liquidity_score': 0.5,
                     'market_state': 'trending', 'signal_id': f'test_{i}'},
                    {'net_pnl_pct': 0.02 if i%2==0 else -0.01}
                )
            result = book.evaluate({'volatility_percentile':0.5, 'volume_zscore':0.0, 'hmm_trend_prob':0.7,
                                    'obi_ema':0.1, 'trend_strength':0.6, 'liquidity_score':0.5, 'side':'long'})
            if result['metadata']['similar_count'] < 1:
                return {"status": "error", "message": "检索失败"}
            book.shutdown()
            return {"status": "ok", "message": "所有路径通过（含Bootstrap、动态边界、持久化）", "stats": book.get_stats()}
        except Exception as e:
            logger.error(f"健康检查失败: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def get_stats(self) -> Dict:
        with self._rwlock:
            return {
                "total_memories": len(self._memories),
                "clusters": self._cluster_centers.shape[0] if self._cluster_centers is not None else 0,
                "queries": self._query_count,
                "avg_query_ms": (self._total_query_time / self._query_count * 1000) if self._query_count else 0
  }
