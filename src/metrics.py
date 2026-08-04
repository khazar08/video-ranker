"""Ranking metrics, grouped by user (query).

All metrics operate on a single query's *relevance vector in ranked order*:
`rels[i]` is the (binary or graded) relevance of the item the model placed at
rank `i` (0-indexed), best first. Higher is better.

Public helpers:
    dcg_at_k / ndcg_at_k / recall_at_k / average_precision_at_k /
    reciprocal_rank        -- single-query scalars
    ranked_relevance       -- turn (scores, labels) into a ranked rel vector
    compute_grouped_metrics-- mean over users for a batch of queries

Definitions follow the standard TREC / recsys conventions so the numbers are
comparable to published baselines. See tests/test_metrics.py for a worked
hand-computed example that pins down NDCG.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np

# Metric families we report, at these cutoffs.
DEFAULT_KS: tuple[int, ...] = (5, 10, 20)


def ranked_relevance(scores: Sequence[float], labels: Sequence[float]) -> np.ndarray:
    """Sort `labels` by descending `scores` -> relevance vector in ranked order.

    Ties are broken deterministically (stable sort on the negated score) so the
    metric is reproducible regardless of the input ordering.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same shape")
    order = np.argsort(-scores, kind="stable")
    return labels[order]


def dcg_at_k(rels: Sequence[float], k: int) -> float:
    """Discounted cumulative gain over the top-k, gain = (2**rel - 1)."""
    rels = np.asarray(rels, dtype=np.float64)[:k]
    if rels.size == 0:
        return 0.0
    gains = np.power(2.0, rels) - 1.0
    discounts = np.log2(np.arange(2, rels.size + 2))  # log2(rank+1), rank>=1
    return float(np.sum(gains / discounts))


def ndcg_at_k(rels: Sequence[float], k: int) -> float:
    """Normalized DCG@k = DCG@k / ideal DCG@k. Returns 0 if no relevance."""
    rels = np.asarray(rels, dtype=np.float64)
    ideal = np.sort(rels)[::-1]
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(rels, k) / idcg


def recall_at_k(rels: Sequence[float], k: int, n_relevant: int | None = None) -> float:
    """Fraction of a user's relevant items that appear in the top-k.

    `n_relevant` is the total number of relevant items for the user (its
    denominator). If omitted, it is taken as the count of positives present in
    `rels` (assumes `rels` contains the full candidate list for the user).
    """
    rels = np.asarray(rels, dtype=np.float64)
    binary = (rels > 0).astype(np.float64)
    denom = n_relevant if n_relevant is not None else int(binary.sum())
    if denom <= 0:
        return 0.0
    hits = float(binary[:k].sum())
    return hits / denom


def average_precision_at_k(rels: Sequence[float], k: int,
                           n_relevant: int | None = None) -> float:
    """Average precision over the top-k (binary relevance).

    AP@k = (1 / min(R, k)) * sum_{i<=k} Precision@i * rel_i, where R is the total
    number of relevant items for the user.
    """
    rels = np.asarray(rels, dtype=np.float64)
    binary = (rels[:k] > 0).astype(np.float64)
    R = n_relevant if n_relevant is not None else int((rels > 0).sum())
    if R <= 0:
        return 0.0
    ranks = np.arange(1, binary.size + 1)
    precision_at_i = np.cumsum(binary) / ranks
    ap = float(np.sum(precision_at_i * binary)) / min(R, k)
    return ap


def reciprocal_rank(rels: Sequence[float], k: int | None = None) -> float:
    """Reciprocal rank of the first relevant item (0 if none within k)."""
    rels = np.asarray(rels, dtype=np.float64)
    if k is not None:
        rels = rels[:k]
    nz = np.flatnonzero(rels > 0)
    if nz.size == 0:
        return 0.0
    return 1.0 / (nz[0] + 1)


# --- Aggregation over users -------------------------------------------------

def compute_grouped_metrics(
    ranked_rels: Iterable[np.ndarray],
    n_relevant: Sequence[int] | None = None,
    ks: Sequence[int] = DEFAULT_KS,
) -> Dict[str, float]:
    """Mean ranking metrics over a collection of per-user ranked rel vectors.

    Args:
        ranked_rels: iterable, one relevance-in-ranked-order vector per user.
        n_relevant: optional per-user total relevant counts (for Recall/MAP
            denominators). If None, inferred from each vector.
        ks: cutoffs to report.

    Returns:
        Flat dict like {"ndcg@10": .., "mrr@10": .., "map@10": .., "recall@10": ..}
        plus a single "mrr" (untruncated) for reference.
    """
    ranked_rels = list(ranked_rels)
    n_users = len(ranked_rels)
    out: Dict[str, float] = {}
    if n_users == 0:
        for k in ks:
            for name in ("ndcg", "mrr", "map", "recall"):
                out[f"{name}@{k}"] = 0.0
        out["mrr"] = 0.0
        out["n_users"] = 0
        return out

    if n_relevant is None:
        n_relevant = [int((r > 0).sum()) for r in ranked_rels]

    for k in ks:
        ndcgs, mrrs, aps, recalls = [], [], [], []
        for rels, nrel in zip(ranked_rels, n_relevant):
            ndcgs.append(ndcg_at_k(rels, k))
            mrrs.append(reciprocal_rank(rels, k))
            aps.append(average_precision_at_k(rels, k, nrel))
            recalls.append(recall_at_k(rels, k, nrel))
        out[f"ndcg@{k}"] = float(np.mean(ndcgs))
        out[f"mrr@{k}"] = float(np.mean(mrrs))
        out[f"map@{k}"] = float(np.mean(aps))
        out[f"recall@{k}"] = float(np.mean(recalls))

    out["mrr"] = float(np.mean([reciprocal_rank(r) for r in ranked_rels]))
    out["n_users"] = n_users
    return out


def metrics_from_scores(
    user_ids: Sequence,
    scores: Sequence[float],
    labels: Sequence[float],
    n_relevant: Dict | None = None,
    ks: Sequence[int] = DEFAULT_KS,
) -> Dict[str, float]:
    """Convenience: compute grouped metrics from flat (user, score, label) rows.

    Rows are grouped by `user_ids`; within each group items are ranked by score.
    `n_relevant` optionally maps user_id -> total relevant count (use when the
    scored candidate set does not contain *all* of a user's held-out positives,
    e.g. after stage-1 retrieval, so Recall/MAP denominators stay honest).
    """
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    order = np.argsort(user_ids, kind="stable")
    user_ids, scores, labels = user_ids[order], scores[order], labels[order]
    # Group boundaries where the (sorted) user id changes. Works for any dtype
    # (ints or strings), unlike np.diff which requires a numeric subtract.
    changes = np.flatnonzero(user_ids[1:] != user_ids[:-1]) + 1
    groups = np.split(np.arange(user_ids.size), changes)

    ranked_rels: List[np.ndarray] = []
    nrel_list: List[int] = []
    for idx in groups:
        uid = user_ids[idx[0]]
        rels = ranked_relevance(scores[idx], labels[idx])
        ranked_rels.append(rels)
        if n_relevant is not None and uid in n_relevant:
            nrel_list.append(int(n_relevant[uid]))
        else:
            nrel_list.append(int((labels[idx] > 0).sum()))

    return compute_grouped_metrics(ranked_rels, nrel_list, ks)
