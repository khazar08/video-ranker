"""Stage-2 rankers.

Two rankers with a shared (fit / predict) interface, both consuming the dense
feature matrix from `features.FeatureStore` plus a per-user `group` array (query
group sizes, LightGBM-style):

    * LambdaMARTRanker -- LightGBM `objective="lambdarank"`, the gradient-boosted
      learning-to-rank baseline. Optimises NDCG directly.
    * NeuralRanker     -- a small MLP scorer trained with a configurable loss:
        listwise:  "listnet"  (top-1 cross-entropy) / "listmle" (Plackett-Luce)
        pairwise:  "ranknet"  (logistic pairs)       / "bpr"     (BPR pairs)
      This supports the listwise-vs-pairwise ablation in the results table.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def _group_bounds(group: np.ndarray) -> np.ndarray:
    """Cumulative offsets [0, g0, g0+g1, ...] delimiting each query's rows."""
    return np.concatenate([[0], np.cumsum(np.asarray(group))]).astype(np.int64)


class Ranker:
    name = "base"

    def fit(self, X: np.ndarray, y: np.ndarray, group: np.ndarray) -> "Ranker":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# --- LightGBM LambdaMART ----------------------------------------------------

class LambdaMARTRanker(Ranker):
    """Gradient-boosted LambdaMART via LightGBM (`objective="lambdarank"`)."""

    name = "lambdamart"

    def __init__(self, num_leaves: int = 31, learning_rate: float = 0.05,
                 n_estimators: int = 300, min_child_samples: int = 20,
                 subsample: float = 0.8, colsample_bytree: float = 0.8,
                 reg_lambda: float = 1.0, eval_at: tuple = (10,),
                 seed: int = 42, feature_names: Optional[List[str]] = None,
                 **_ignore):
        self.params = dict(
            objective="lambdarank", metric="ndcg", num_leaves=num_leaves,
            learning_rate=learning_rate, min_child_samples=min_child_samples,
            subsample=subsample, subsample_freq=1,
            colsample_bytree=colsample_bytree, reg_lambda=reg_lambda,
            ndcg_eval_at=list(eval_at), seed=seed, verbose=-1,
            force_row_wise=True,
        )
        self.n_estimators = n_estimators
        self.feature_names = feature_names
        self.model = None

    def fit(self, X, y, group,
            valid: Optional[tuple] = None) -> "LambdaMARTRanker":
        import lightgbm as lgb

        dtrain = lgb.Dataset(X, label=y, group=group,
                             feature_name=self.feature_names or "auto")
        valid_sets, valid_names, callbacks = [dtrain], ["train"], []
        if valid is not None:
            Xv, yv, gv = valid
            dvalid = lgb.Dataset(Xv, label=yv, group=gv, reference=dtrain)
            valid_sets.append(dvalid)
            valid_names.append("valid")
            callbacks.append(lgb.early_stopping(50, verbose=False))
        self.model = lgb.train(
            self.params, dtrain, num_boost_round=self.n_estimators,
            valid_sets=valid_sets, valid_names=valid_names, callbacks=callbacks,
        )
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def feature_importance(self) -> dict:
        if self.model is None:
            return {}
        names = self.model.feature_name()
        gains = self.model.feature_importance(importance_type="gain")
        return dict(sorted(zip(names, gains.tolist()), key=lambda kv: -kv[1]))


# --- Neural ranker ----------------------------------------------------------

class NeuralRanker(Ranker):
    """MLP scorer trained with a listwise or pairwise learning-to-rank loss."""

    name = "neural"

    def __init__(self, loss: str = "listnet", hidden: tuple = (128, 64),
                 dropout: float = 0.1, epochs: int = 30, lr: float = 1e-3,
                 weight_decay: float = 1e-5, queries_per_batch: int = 64,
                 pairs_per_query: int = 20, seed: int = 42, verbose: bool = False,
                 device: str = "cpu",
                 feature_names: Optional[List[str]] = None, **_ignore):
        self.loss = loss.lower()
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.queries_per_batch = queries_per_batch
        self.pairs_per_query = pairs_per_query
        self.seed = seed
        self.verbose = verbose
        self.device = device
        self.feature_names = feature_names
        self.model = None
        if self.loss not in ("listnet", "listmle", "ranknet", "bpr"):
            raise ValueError(f"unknown loss '{loss}'")

    # -- model ----------------------------------------------------------------
    def _build(self, in_dim: int):
        import torch.nn as nn
        layers, d = [], in_dim
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(self.dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        return nn.Sequential(*layers)

    # -- losses (operate on one query's scores/labels) ------------------------
    @staticmethod
    def _listnet_loss(scores, labels):
        import torch
        # Top-1 ListNet: cross-entropy between softmax(labels) and softmax(scores).
        p_true = torch.softmax(labels, dim=0)
        log_p_pred = torch.log_softmax(scores, dim=0)
        return -(p_true * log_p_pred).sum()

    @staticmethod
    def _listmle_loss(scores, labels):
        import torch
        # Plackett-Luce likelihood of the label-sorted permutation.
        order = torch.argsort(labels, descending=True)
        s = scores[order]
        # log-cumsum-exp from the tail gives the normaliser at each position.
        rev_logcumsum = torch.logcumsumexp(s.flip(0), dim=0).flip(0)
        return (rev_logcumsum - s).sum()

    def _pairwise_loss(self, scores, labels, rng):
        """Pairwise loss over (higher-label, lower-label) pairs within a query.

        The per-pair loss is logistic in both cases; RankNet and BPR differ in
        *which* pairs they use:
          - ranknet: ALL pos x neg pairs (full, deterministic gradient), the
            canonical RankNet formulation.
          - bpr:     a random sample of `pairs_per_query` pairs (stochastic),
            the Bayesian Personalized Ranking style.
        With binary relevance these converge but differ in gradient variance.
        """
        import torch
        labels_np = labels.detach().cpu().numpy()
        thr = labels_np.min()
        pos_idx = np.flatnonzero(labels_np > thr)
        neg_idx = np.flatnonzero(labels_np <= thr)
        if pos_idx.size == 0 or neg_idx.size == 0:
            return None

        if self.loss == "ranknet":
            pi = torch.as_tensor(pos_idx, device=scores.device)
            ni = torch.as_tensor(neg_idx, device=scores.device)
            # [P, N] matrix of score differences over every pos-neg pair.
            diff = scores[pi].unsqueeze(1) - scores[ni].unsqueeze(0)
            return torch.nn.functional.softplus(-diff).mean()

        # bpr: sample pairs
        m = min(self.pairs_per_query, pos_idx.size * neg_idx.size)
        pi = pos_idx[rng.integers(0, pos_idx.size, size=m)]
        ni = neg_idx[rng.integers(0, neg_idx.size, size=m)]
        diff = scores[pi] - scores[ni]
        return -torch.nn.functional.logsigmoid(diff).mean()

    # -- fit / predict --------------------------------------------------------
    def fit(self, X, y, group) -> "NeuralRanker":
        import torch

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # Standardize features (stored for predict); NN needs comparable scales.
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-6
        Xs = (X - self.mu) / self.sd

        self.device = torch.device(self.device) if isinstance(self.device, str) \
            else self.device
        Xt = torch.from_numpy(Xs).to(self.device)
        yt = torch.from_numpy(y).to(self.device)
        bounds = _group_bounds(group)
        n_queries = len(bounds) - 1

        self.model = self._build(X.shape[1]).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        listwise = self.loss in ("listnet", "listmle")

        self.model.train()
        for epoch in range(self.epochs):
            q_order = rng.permutation(n_queries)
            total, steps = 0.0, 0
            for bstart in range(0, n_queries, self.queries_per_batch):
                batch_q = q_order[bstart:bstart + self.queries_per_batch]
                # Gather all rows for this mini-batch's queries and run ONE
                # forward pass; then split scores back per query for the loss.
                spans, row_idx = [], []
                for q in batch_q:
                    lo, hi = int(bounds[q]), int(bounds[q + 1])
                    if hi - lo < 2:
                        continue
                    spans.append((len(row_idx), len(row_idx) + (hi - lo)))
                    row_idx.extend(range(lo, hi))
                if not row_idx:
                    continue
                idx_t = torch.as_tensor(row_idx, dtype=torch.long, device=self.device)
                all_scores = self.model(Xt[idx_t]).squeeze(1)
                all_labels = yt[idx_t]

                opt.zero_grad()
                losses = []
                for (a, b) in spans:
                    s, lab = all_scores[a:b], all_labels[a:b]
                    if lab.max() == lab.min():
                        continue  # no ranking signal in this query
                    if listwise:
                        loss = (self._listnet_loss(s, lab) if self.loss == "listnet"
                                else self._listmle_loss(s, lab))
                    else:
                        loss = self._pairwise_loss(s, lab, rng)
                        if loss is None:
                            continue
                    losses.append(loss)
                if not losses:
                    continue
                batch_loss = torch.stack(losses).mean()
                batch_loss.backward()
                opt.step()
                total += float(batch_loss.item())
                steps += 1
            if self.verbose:
                avg = total / max(steps, 1)
                print(f"  [neural:{self.loss}] epoch {epoch + 1}/{self.epochs} "
                      f"loss={avg:.4f}")
        return self

    def to_cpu(self) -> "NeuralRanker":
        """Move the scorer to CPU so the ranker pickles/serves portably."""
        import torch
        self.device = torch.device("cpu")
        self.model.to(self.device)
        return self

    def predict(self, X) -> np.ndarray:
        import torch
        X = np.asarray(X, dtype=np.float32)
        Xs = (X - self.mu) / self.sd
        self.model.eval()
        out = np.empty(len(X), dtype=np.float32)
        with torch.no_grad():
            Xt = torch.from_numpy(Xs).to(self.device)
            for start in range(0, len(X), 16384):
                chunk = Xt[start:start + 16384]
                out[start:start + chunk.shape[0]] = \
                    self.model(chunk).squeeze(1).cpu().numpy()
        return out


def build_ranker(kind: str, params: dict, feature_names: Optional[List[str]] = None
                 ) -> Ranker:
    """Factory: 'lambdamart' or 'neural'."""
    kind = kind.lower()
    if kind in ("lambdamart", "lgbm", "lightgbm"):
        return LambdaMARTRanker(feature_names=feature_names, **params)
    if kind in ("neural", "mlp", "neural_ranker"):
        return NeuralRanker(feature_names=feature_names, **params)
    raise ValueError(f"Unknown ranker '{kind}'")
