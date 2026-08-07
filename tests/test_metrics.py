import math
import numpy as np
import pytest
from src import metrics as M


# --- NDCG, worked by hand ---------------------------------------------------

def test_ndcg_binary_hand_computed():
    # Ranked relevance (best first): item1 relevant, item2 not, item3 relevant.
    rels = [1, 0, 1]
    # DCG@3 = (2^1-1)/log2(2) + (2^0-1)/log2(3) + (2^1-1)/log2(4)
    #       = 1/1 + 0 + 1/2 = 1.5
    dcg = 1.0 / math.log2(2) + 0.0 + 1.0 / math.log2(4)
    assert M.dcg_at_k(rels, 3) == pytest.approx(dcg)
    assert dcg == pytest.approx(1.5)

    # Ideal order [1,1,0]: IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309297...
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert M.ndcg_at_k(rels, 3) == pytest.approx(dcg / idcg)
    assert M.ndcg_at_k(rels, 3) == pytest.approx(0.9197207891481876)


def test_ndcg_perfect_and_empty():
    assert M.ndcg_at_k([1, 1, 0, 0], 4) == pytest.approx(1.0)
    assert M.ndcg_at_k([0, 0, 0], 3) == 0.0          # no relevance -> 0
    assert M.ndcg_at_k([], 5) == 0.0


def test_ndcg_graded_hand_computed():
    # Graded relevance, gain = 2^rel - 1.
    rels = [3, 2, 3, 0, 1, 2]
    gains = [2 ** r - 1 for r in rels]
    disc = [math.log2(i + 1) for i in range(1, len(rels) + 1)]
    dcg = sum(g / d for g, d in zip(gains, disc))
    ideal = sorted(rels, reverse=True)
    igains = [2 ** r - 1 for r in ideal]
    idcg = sum(g / d for g, d in zip(igains, disc))
    assert M.ndcg_at_k(rels, 6) == pytest.approx(dcg / idcg)


# --- Recall -----------------------------------------------------------------

def test_recall_at_k():
    rels = [1, 0, 0, 1, 0]           # 2 relevant, both within top-4
    assert M.recall_at_k(rels, 4) == pytest.approx(2 / 2)
    assert M.recall_at_k(rels, 1) == pytest.approx(1 / 2)
    # Honest denominator when candidate list is truncated (5 total relevant).
    assert M.recall_at_k(rels, 5, n_relevant=5) == pytest.approx(2 / 5)
    assert M.recall_at_k([0, 0], 2) == 0.0


# --- MRR --------------------------------------------------------------------

def test_reciprocal_rank():
    assert M.reciprocal_rank([0, 0, 1, 0]) == pytest.approx(1 / 3)
    assert M.reciprocal_rank([1, 0, 0]) == pytest.approx(1.0)
    assert M.reciprocal_rank([0, 0, 1], k=2) == 0.0   # relevant beyond cutoff
    assert M.reciprocal_rank([0, 0, 0]) == 0.0


# --- MAP --------------------------------------------------------------------

def test_average_precision_hand_computed():
    rels = [1, 0, 1]                 # R = 2
    # P@1=1, P@2=0.5, P@3=2/3; AP = (1*1 + 0*0.5 + 1*2/3)/min(2,3)
    ap = (1.0 + (2 / 3)) / 2
    assert M.average_precision_at_k(rels, 3) == pytest.approx(ap)
    assert M.average_precision_at_k(rels, 3) == pytest.approx(0.8333333333)
    assert M.average_precision_at_k([0, 0], 2) == 0.0


# --- ranked_relevance + grouping -------------------------------------------

def test_ranked_relevance_orders_by_score():
    scores = [0.1, 0.9, 0.4]
    labels = [0, 1, 1]
    # Highest score (0.9 -> label 1) first, then 0.4 -> 1, then 0.1 -> 0.
    np.testing.assert_array_equal(
        M.ranked_relevance(scores, labels), np.array([1.0, 1.0, 0.0])
    )


def test_ranked_relevance_tie_break_is_stable():
    scores = [0.5, 0.5, 0.5]
    labels = [0, 1, 0]
    # Stable sort preserves input order on ties -> deterministic.
    np.testing.assert_array_equal(
        M.ranked_relevance(scores, labels), np.array([0.0, 1.0, 0.0])
    )


def test_metrics_from_scores_two_users():
    # user A: perfect ranking; user B: relevant item last.
    user_ids = ["A", "A", "B", "B"]
    scores = [0.9, 0.1, 0.1, 0.9]
    labels = [1, 0, 1, 0]
    out = M.metrics_from_scores(user_ids, scores, labels, ks=[2])
    # A: ndcg=1, mrr=1. B: relevant at rank 2 -> ndcg=1/log2(3)/1=0.6309, mrr=0.5
    assert out["ndcg@2"] == pytest.approx((1.0 + 1.0 / math.log2(3)) / 2)
    assert out["mrr@2"] == pytest.approx((1.0 + 0.5) / 2)
    assert out["n_users"] == 2


def test_compute_grouped_metrics_empty():
    out = M.compute_grouped_metrics([], ks=[5, 10])
    assert out["ndcg@5"] == 0.0 and out["n_users"] == 0
