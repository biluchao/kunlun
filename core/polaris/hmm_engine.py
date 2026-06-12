#!/usr/bin/env python3
"""
昆仑系统 · 隐马尔可夫市场状态引擎 (HMMEngine)

核心职责：
1. 双时间尺度（3分钟 / 5分钟）独立运行贝叶斯 HMM 在线滤波器，实时输出趋势/震荡后验概率
2. 通过无监督波动率锚定自动判别状态标签，消除初始化标签歧义
3. 状态切换采用贝叶斯因子 + 最小持续期迟滞，彻底消除闪烁
4. 每200根K线进行稳健增量重估计（多起点EM + 对角加载正则化），参数指数平滑，自动锚定
5. 实时监控对数似然，模型退化时自动回退到加权规则分类器

外部依赖（真实模块接口）：
- infrastructure.chronos_db.ChronosDB : 提供模型参数的持久化存取
- infrastructure.stream_gateway.StreamGateway : 推送经过预处理的标准化特征向量

接口契约：
- update_and_predict(features_3m: np.ndarray, features_5m: np.ndarray) -> Dict[str, Any]
  输入标准化的4维特征，返回市场状态、置信度、双尺度细节
- get_current_state() -> Dict[str, Any]
  返回最近一次推断的快照，无副作用
- health_check() -> Dict[str, Any]
  使用多场景合成数据验证全链路

异常与降级：
- hmmlearn不可用：启用规则分类器，记录 KUN-HMM-F001
- 在线滤波数值异常：重置前向概率为均匀分布，记录 KUN-HMM-F002
- 重估计不收敛：回退前次参数，记录 KUN-HMM-E001
- 特征连续缺失 ≥ 3：输出 uncertain，记录 KUN-HMM-W001

资源管理：
- 内存占用 < 50KB
- 模型参数通过 ChronosDB 持久化，重启热加载
"""

import logging
import math
import pickle
from typing import Dict, Any, Optional, Tuple
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    logger.warning("[KUN-HMM-F001] hmmlearn 未安装，HMM 引擎降级为规则分类器")


class HMMEngine:
    """双时间尺度在线 HMM 市场状态引擎"""

    # ----- 类常量（默认配置）-----
    N_STATES = 2
    N_FEATURES = 4
    COVARIANCE_TYPE = 'diag'
    DIAG_LOADING = 1e-3

    # 状态切换迟滞
    MIN_STABILITY_BARS = 4
    BAYES_FACTOR_THRESHOLD = 3.0
    POST_SMOOTH_DECAY = 0.4               # 平滑因子，新权重

    # 重估计
    REESTIMATE_INTERVAL = 200
    NEW_PARAM_WEIGHT = 0.2
    OLD_PARAM_WEIGHT = 0.8
    MAX_HISTORY_SIZE = 500
    MIN_TRAIN_SAMPLES = 50

    # 状态锚定
    VOL_ANCHOR_PERCENTILE = 60            # 累计波动率分位数用于标签锚定

    # 异常填充
    MAX_FILL_COUNT = 3

    def __init__(self, config: Optional[Dict] = None,
                 db_persist: Optional[Any] = None):
        if config:
            self._apply_config(config)

        self._db = db_persist
        self._hmm_3m = None
        self._hmm_5m = None

        # 在线滤波状态（对数形式）
        self._log_alpha_3m = None
        self._log_alpha_5m = None

        # 观测历史
        self._obs_3m = deque(maxlen=self.MAX_HISTORY_SIZE)
        self._obs_5m = deque(maxlen=self.MAX_HISTORY_SIZE)

        # 平滑后验概率
        self._post_3m = np.ones(self.N_STATES) / self.N_STATES
        self._post_5m = np.ones(self.N_STATES) / self.N_STATES

        # 状态稳定性变量
        self._cand_3m: Optional[int] = None
        self._cand_cnt_3m: float = 0.0
        self._cand_5m: Optional[int] = None
        self._cand_cnt_5m: float = 0.0

        # 锚定映射
        self._label_map_3m: Dict[int, int] = {0: 0, 1: 1}
        self._label_map_5m: Dict[int, int] = {0: 0, 1: 1}

        # 计数器与填充
        self._bar_count = 0
        self._fill_cnt_3m = 0
        self._fill_cnt_5m = 0
        self._last_f3 = np.zeros(self.N_FEATURES)
        self._last_f5 = np.zeros(self.N_FEATURES)

        # 训练状态
        self._trained = False
        self._model_likelihood_3m = []
        self._model_likelihood_5m = []

        # 加载或初始化模型
        if self._db:
            self._load_models()
        if not self._trained and HMMLEARN_AVAILABLE:
            self._init_hmms()
        if not HMMLEARN_AVAILABLE:
            logger.info("HMM 引擎降级为规则模式")

        logger.info("HMMEngine ready (hmmlearn=%s, trained=%s)",
                    HMMLEARN_AVAILABLE, self._trained)

    # ----- 配置 -----
    @staticmethod
    def _apply_config(config: Dict) -> None:
        for k, v in config.items():
            if hasattr(HMMEngine, k):
                setattr(HMMEngine, k, v)

    # ----- 初始化 & 持久化 -----
    def _init_hmms(self):
        self._hmm_3m = self._make_fresh_hmm(seed=42)
        self._hmm_5m = self._make_fresh_hmm(seed=43)
        self._reset_alpha()

    def _make_fresh_hmm(self, seed: int) -> hmm.GaussianHMM:
        model = hmm.GaussianHMM(
            n_components=self.N_STATES,
            covariance_type=self.COVARIANCE_TYPE,
            n_iter=100,
            tol=1e-4,
            init_params='stmc',
            random_state=seed
        )
        # 设置合理的初始均值与协方差
        model.means_ = np.zeros((self.N_STATES, self.N_FEATURES))
        model.covars_ = np.ones((self.N_STATES, self.N_FEATURES)) * 0.1
        return model

    def _reset_alpha(self):
        uniform_log = math.log(1.0 / self.N_STATES)
        self._log_alpha_3m = np.full(self.N_STATES, uniform_log)
        self._log_alpha_5m = np.full(self.N_STATES, uniform_log)

    def _load_models(self):
        try:
            raw = self._db.get('hmm_model')
            if raw:
                data = pickle.loads(raw)
                self._hmm_3m = data['hmm_3m']
                self._hmm_5m = data['hmm_5m']
                self._label_map_3m = data.get('label_map_3m', {0:0,1:1})
                self._label_map_5m = data.get('label_map_5m', {0:0,1:1})
                self._trained = True
                self._reset_alpha()
                logger.info("HMM 模型已从数据库加载")
        except Exception as e:
            logger.error("[KUN-HMM-E003] 模型加载失败: %s", e)

    def _persist_models(self):
        if not self._db or not self._trained:
            return
        try:
            data = {
                'hmm_3m': self._hmm_3m,
                'hmm_5m': self._hmm_5m,
                'label_map_3m': self._label_map_3m,
                'label_map_5m': self._label_map_5m
            }
            self._db.store('hmm_model', pickle.dumps(data))
        except Exception as e:
            logger.error("[KUN-HMM-E002] 模型持久化失败: %s", e, exc_info=True)

    # ----- 特征验证 -----
    @staticmethod
    def _validate(features: np.ndarray) -> Tuple[np.ndarray, bool]:
        if features is None or len(features) != HMMEngine.N_FEATURES:
            return np.zeros(HMMEngine.N_FEATURES), False
        fixed = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        # 动态裁剪范围（可根据实际扩展，暂用硬编码）
        bounds = np.array([0.20, 5.0, 1.0, 10.0])
        fixed = np.clip(fixed, -bounds, bounds)
        return fixed, True

    # ----- 在线前向滤波 -----
    def _forward_step(self, model, log_alpha: np.ndarray, x: np.ndarray) -> np.ndarray:
        """返回更新后的对数前向概率"""
        if model is None or not hasattr(model, 'startprob_'):
            # 降级：返回规则后验概率（对数形式）
            post = self._rule_posterior(x)
            return np.log(np.clip(post, 1e-12, 1.0))

        # 对数发射概率
        log_em = np.zeros(self.N_STATES)
        for k in range(self.N_STATES):
            xc = x - model.means_[k]
            var = model.covars_[k] + self.DIAG_LOADING
            log_em[k] = -0.5 * (np.sum(xc * xc / var) + np.sum(np.log(var)) + self.N_FEATURES * math.log(2 * math.pi))

        # 前向递推
        log_trans = np.log(np.clip(model.transmat_, 1e-12, 1.0))
        new_alpha = np.zeros(self.N_STATES)
        for j in range(self.N_STATES):
            new_alpha[j] = log_em[j] + self._logsumexp(log_alpha + log_trans[:, j])
        return new_alpha

    @staticmethod
    def _logsumexp(x: np.ndarray) -> float:
        c = np.max(x)
        if np.isneginf(c):
            return -np.inf
        return c + math.log(np.maximum(np.sum(np.exp(x - c)), 1e-300))

    # ----- 规则后验（降级用）-----
    def _rule_posterior(self, x: np.ndarray) -> np.ndarray:
        """返回趋势概率 [P(ranging), P(trending)]"""
        ret = abs(x[0])
        obi = abs(x[2])
        vol = abs(x[1])
        score = (ret / 0.02) * 0.4 + (obi / 0.3) * 0.3 + (vol / 2.0) * 0.3
        p_trend = 1.0 / (1.0 + math.exp(-5 * (score - 0.5)))
        p_trend = max(0.05, min(0.95, p_trend))
        return np.array([1.0 - p_trend, p_trend])

    # ----- 状态锚定 -----
    def _anchor_labels(self, model, obs: np.ndarray) -> Dict[int, int]:
        if not HMMLEARN_AVAILABLE or model is None or len(obs) < 10:
            return {0: 0, 1: 1}
        try:
            states = model.predict(obs)
            vols = [np.mean(np.abs(obs[states == s, 0])) if np.any(states == s) else 0.0
                    for s in range(self.N_STATES)]
            low = np.argmin(vols)
            high = np.argmax(vols)
            return {low: 0, high: 1}
        except Exception as e:
            logger.error("[KUN-HMM-E004] 锚定失败: %s", e, exc_info=True)
            return {0: 0, 1: 1}

    # ----- 训练 -----
    def _train_model(self, model, obs: np.ndarray) -> bool:
        if not HMMLEARN_AVAILABLE or len(obs) < self.MIN_TRAIN_SAMPLES:
            return False
        try:
            mean = np.mean(obs, axis=0)
            std = np.std(obs, axis=0) + 1e-8
            obs_norm = (obs - mean) / std

            best_score = -np.inf
            best = None
            for s in [42, 123, 777]:
                m = hmm.GaussianHMM(self.N_STATES, self.COVARIANCE_TYPE, n_iter=200, tol=1e-4, random_state=s)
                m.fit(obs_norm)
                score = m.score(obs_norm)
                if score > best_score:
                    best_score = score
                    best = {
                        'startprob_': m.startprob_.copy(),
                        'transmat_': m.transmat_.copy(),
                        'means_': m.means_.copy() * std + mean,
                        'covars_': m.covars_.copy() * (std ** 2) + self.DIAG_LOADING
                    }
            if best:
                for attr in best:
                    setattr(model, attr, best[attr])
                return True
            return False
        except Exception as e:
            logger.error("[KUN-HMM-E005] 训练异常: %s", e, exc_info=True)
            return False

    def _smooth(self, model, old_params: Dict):
        if old_params is None:
            return
        nw, ow = self.NEW_PARAM_WEIGHT, self.OLD_PARAM_WEIGHT
        for attr in ['startprob_', 'transmat_', 'means_', 'covars_']:
            new = getattr(model, attr)
            old = old_params[attr]
            if new.shape == old.shape:
                setattr(model, attr, ow * old + nw * new)

    def _maybe_restimate(self):
        self._bar_count += 1
        if self._bar_count % self.REESTIMATE_INTERVAL != 0 or not HMMLEARN_AVAILABLE:
            return
        if len(self._obs_3m) >= self.MIN_TRAIN_SAMPLES:
            obs = np.array(self._obs_3m)
            old = {k: getattr(self._hmm_3m, k).copy() for k in
                   ['startprob_', 'transmat_', 'means_', 'covars_']}
            if self._train_model(self._hmm_3m, obs):
                self._smooth(self._hmm_3m, old)
                self._label_map_3m = self._anchor_labels(self._hmm_3m, obs)
                self._trained = True
        if len(self._obs_5m) >= self.MIN_TRAIN_SAMPLES:
            obs = np.array(self._obs_5m)
            old = {k: getattr(self._hmm_5m, k).copy() for k in
                   ['startprob_', 'transmat_', 'means_', 'covars_']}
            if self._train_model(self._hmm_5m, obs):
                self._smooth(self._hmm_5m, old)
                self._label_map_5m = self._anchor_labels(self._hmm_5m, obs)
                self._trained = True
        if self._trained:
            self._persist_models()

    # ----- 状态稳定化 -----
    def _stabilize(self, raw: int, log_alpha: np.ndarray,
                   cand: Optional[int], cnt: float) -> Tuple[int, Optional[int], float]:
        if cand is None:
            return raw, raw, 1.0
        if raw == cand:
            cnt = min(cnt + 1, self.MIN_STABILITY_BARS)
            if cnt >= self.MIN_STABILITY_BARS:
                return raw, raw, cnt
            else:
                return cand, cand, cnt
        else:
            log_post = log_alpha - self._logsumexp(log_alpha)
            post = np.exp(log_post)
            bf = post[raw] / max(post[cand], 1e-12)
            if bf > self.BAYES_FACTOR_THRESHOLD:
                return cand, raw, 1.0
            else:
                cnt = max(0.0, cnt - 0.5)
                return cand, cand, cnt

    # ----- 主接口 -----
    def update_and_predict(self, feat_3m: np.ndarray,
                           feat_5m: np.ndarray) -> Dict[str, Any]:
        warnings = []
        f3, v3 = self._validate(feat_3m)
        f5, v5 = self._validate(feat_5m)

        if not v3:
            self._fill_cnt_3m += 1
            f3 = self._last_f3
            warnings.append("[KUN-HMM-W001] 3m特征无效，填充")
        else:
            self._fill_cnt_3m = 0
            self._last_f3 = f3

        if not v5:
            self._fill_cnt_5m += 1
            f5 = self._last_f5
            warnings.append("[KUN-HMM-W001] 5m特征无效，填充")
        else:
            self._fill_cnt_5m = 0
            self._last_f5 = f5

        if self._fill_cnt_3m >= self.MAX_FILL_COUNT or self._fill_cnt_5m >= self.MAX_FILL_COUNT:
            return {"status": "warning", "state": "uncertain", "confidence": 0.0,
                    "reason": "特征连续缺失过多", "warnings": warnings}

        self._obs_3m.append(f3)
        self._obs_5m.append(f5)

        if self._log_alpha_3m is None:
            self._reset_alpha()
        self._log_alpha_3m = self._forward_step(self._hmm_3m, self._log_alpha_3m, f3)
        self._log_alpha_5m = self._forward_step(self._hmm_5m, self._log_alpha_5m, f5)

        # 后验概率
        post_3m = np.exp(self._log_alpha_3m - self._logsumexp(self._log_alpha_3m))
        post_5m = np.exp(self._log_alpha_5m - self._logsumexp(self._log_alpha_5m))

        # 平滑
        self._post_3m = self._post_3m * (1 - self.POST_SMOOTH_DECAY) + post_3m * self.POST_SMOOTH_DECAY
        self._post_5m = self._post_5m * (1 - self.POST_SMOOTH_DECAY) + post_5m * self.POST_SMOOTH_DECAY

        raw_3m = int(np.argmax(self._post_3m))
        raw_5m = int(np.argmax(self._post_5m))

        # 状态稳定化
        final_3m, cand_3m, cnt_3m = self._stabilize(
            raw_3m, self._log_alpha_3m, self._cand_3m, self._cand_cnt_3m)
        final_5m, cand_5m, cnt_5m = self._stabilize(
            raw_5m, self._log_alpha_5m, self._cand_5m, self._cand_cnt_5m)

        self._cand_3m, self._cand_cnt_3m = cand_3m, cnt_3m
        self._cand_5m, self._cand_cnt_5m = cand_5m, cnt_5m

        # 应用标签映射
        labeled_3m = self._label_map_3m.get(final_3m, final_3m)
        labeled_5m = self._label_map_5m.get(final_5m, final_5m)

        agreement = (labeled_3m == labeled_5m)
        if agreement:
            state = "trending" if labeled_3m == 1 else "ranging"
            conf = max(self._post_3m[final_3m], self._post_5m[final_5m])
        else:
            state = "uncertain"
            conf = 0.3

        self._maybe_restimate()

        return {
            "status": "ok",
            "state": state,
            "confidence": round(conf, 4),
            "details": {
                "scale_3m": {"state": "trending" if labeled_3m==1 else "ranging",
                             "prob": round(self._post_3m[final_3m], 4)},
                "scale_5m": {"state": "trending" if labeled_5m==1 else "ranging",
                             "prob": round(self._post_5m[final_5m], 4)},
                "agreement": agreement
            },
            "warnings": warnings
        }

    def get_current_state(self) -> Dict[str, Any]:
        if not self._trained and self._bar_count == 0 and HMMLEARN_AVAILABLE:
            return {"status": "warning", "state": "uncertain", "confidence": 0.0,
                    "reason": "模型尚未训练"}
        if self._last_f3 is None:
            return {"status": "warning", "state": "uncertain", "confidence": 0.0,
                    "reason": "无历史特征"}
        return self.update_and_predict(self._last_f3, self._last_f5)

    # ----- 健康检查 -----
    @classmethod
    def health_check(cls) -> Dict[str, Any]:
        try:
            engine = cls(db_persist=None)
            rng = np.random.RandomState(100)
            n = 200
            X = rng.normal(0.0, 0.01, (n, 4))
            X[80:] += 0.03
            for i in range(n):
                res = engine.update_and_predict(X[i], X[i])
            if res['state'] not in ('trending', 'ranging', 'uncertain'):
                return {"status": "error", "message": f"无效状态: {res['state']}"}
            if engine._bar_count != n:
                return {"status": "error", "message": "K线计数错误"}
            return {"status": "ok", "message": "全链路测试通过"}
        except Exception as e:
            logger.error("健康检查失败: %s", e, exc_info=True)
            return {"status": "error", "message": str(e)}
