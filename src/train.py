from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from . import metrics as M
from .config import REPO_ROOT, Config, load_config, resolve_paths, seed_everything
from .data_prep import (Dataset, prepare_dataset, sample_eval_candidates,
                        sample_train_candidates)
from .features import FeatureStore
from .ranking import build_ranker
from .retrieval import build_retriever



def _dataset_cache_path(cfg: Config) -> Path:
    d = cfg.get("data", {})
    key = json.dumps({k: d.get(k) for k in sorted(d)}, sort_keys=True)
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    folder = "ml-latest-small" if d.get("small") else "ml-25m"
    return REPO_ROOT / "data" / "processed" / f"{folder}_{h}.pkl"


def load_dataset(cfg: Config) -> Dataset:
    d = cfg.data
    dataset_dir = REPO_ROOT / "data" / ("ml-latest-small" if d.get("small") else "ml-25m")
    cache = _dataset_cache_path(cfg)
    return prepare_dataset(
        dataset_dir=dataset_dir,
        positive_threshold=d.get("positive_threshold", 4.0),
        min_interactions=d.get("min_interactions", 5),
        n_test=d.get("n_test", 2),
        n_valid=d.get("n_valid", 2),
        max_users=d.get("max_users"),
        confidence=d.get("confidence", "linear"),
        alpha=d.get("alpha", 2.0),
        seed=cfg.get("seed", 42),
        cache_path=cache,
    )



def evaluate_retrieval(retriever, ds: Dataset, ks=(100, 200)) -> Dict[str, float]:
    """Recall@k of the retriever's top-max(ks) list against test positives."""
    users = np.array(sorted(ds.positives_by_user("test").keys()))
    test = ds.positives_by_user("test")
    ids, _ = retriever.recommend(users, max(ks))
    out = {f"recall@{k}": [] for k in ks}
    for i, u in enumerate(users):
        pos = set(test[u].tolist())
        if not pos:
            continue
        row = ids[i]
        for k in ks:
            hits = len(pos & set(row[:k].tolist()))
            out[f"recall@{k}"].append(hits / len(pos))
    return {k: float(np.mean(v)) if v else 0.0 for k, v in out.items()}


# --- candidate assembly -----------------------------------------------------

def _attach(candidates: pd.DataFrame, fs: FeatureStore, retriever) -> tuple:
    """Sort by user, build features + retrieval-score column, return (X, df, group)."""
    cand = candidates.sort_values("user_idx", kind="stable").reset_index(drop=True)
    ret_score = retriever.score(cand["user_idx"].to_numpy(), cand["item_idx"].to_numpy())
    X, names = fs.transform(cand, extra={"retrieval_score": ret_score})
    group = cand.groupby("user_idx", sort=True).size().to_numpy()
    return X, names, cand, group


def _metric_row(cand: pd.DataFrame, scores: np.ndarray, ds: Dataset,
                ks) -> Dict[str, float]:
    test = ds.positives_by_user("test")
    n_rel = {u: len(v) for u, v in test.items()}
    return M.metrics_from_scores(
        cand["user_idx"].to_numpy(), scores, cand["label"].to_numpy(),
        n_relevant=n_rel, ks=ks)


# --- main experiment --------------------------------------------------------

def run_experiment(cfg: Config) -> List[dict]:
    seed_everything(cfg.get("seed", 42))
    paths = resolve_paths(cfg)
    ks = tuple(cfg.get("eval", {}).get("ks", [5, 10, 20]))
    exp_name = cfg["_config_name"]
    print(f"\n=== Experiment: {exp_name} ===")

    ds = load_dataset(cfg)
    print(f"users={ds.n_users} items={ds.n_items} "
          f"train/valid/test="
          f"{len(ds.train)}/{len(ds.valid)}/{len(ds.test)}")

    # -- stage 1: retrieval --
    retr = build_retriever(cfg.retrieval.model, dict(cfg.retrieval.get("params", {})))
    print(f"Fitting retriever: {retr.name}")
    retr.fit(ds)
    recall = evaluate_retrieval(retr, ds, ks=tuple(cfg.get("eval", {}).get(
        "recall_ks", [100, 200])))
    print(f"  retrieval {recall}  ({getattr(retr, 'fit_seconds', 0):.1f}s)")

    # -- features (leakage-free: train history for ranker, train+valid for eval) --
    fs_train = FeatureStore(ds, ds.train)
    fs_eval = FeatureStore(ds, pd.concat([ds.train, ds.valid], ignore_index=True))

    # -- ranker-training candidates --
    d = cfg.data
    train_cand = sample_train_candidates(
        ds, k_neg=cfg.get("ranker_train", {}).get("k_neg", 4),
        hard_fraction=cfg.get("ranker_train", {}).get("hard_fraction", 0.5),
        seed=cfg.get("seed", 42))
    Xtr, feat_names, train_cand, gtr = _attach(train_cand, fs_train, retr)

    # -- eval candidate pools --
    n_neg = cfg.get("eval", {}).get("n_negatives", 100)
    eval_random = sample_eval_candidates(ds, n_neg=n_neg, mode="random",
                                         seed=cfg.get("seed", 42) + 1)
    eval_hard = sample_eval_candidates(ds, n_neg=n_neg, mode="hard",
                                       seed=cfg.get("seed", 42) + 2)
    Xr, _, cand_r, _ = _attach(eval_random, fs_eval, retr)
    Xh, _, cand_h, _ = _attach(eval_hard, fs_eval, retr)

    # cold vs active users by train interaction count.
    train_counts = ds.train.groupby("user_idx").size()
    cold_thr = cfg.get("eval", {}).get("cold_threshold", 20)
    cold_users = set(train_counts[train_counts <= cold_thr].index)

    # -- rankers (possibly several losses for the neural ablation) --
    rk_kind = cfg.ranking.model
    rk_params = dict(cfg.ranking.get("params", {}))
    losses = rk_params.pop("loss", None)
    loss_variants = losses if isinstance(losses, list) else [losses]

    results: List[dict] = []
    best_bundle = None
    for loss in loss_variants:
        params = dict(rk_params)
        tag = exp_name
        if loss is not None:
            params["loss"] = loss
            tag = f"{exp_name}:{loss}"
        print(f"Training ranker: {rk_kind}"
              + (f" (loss={loss})" if loss else ""))
        ranker = build_ranker(rk_kind, params, feature_names=feat_names)
        t0 = time.time()
        ranker.fit(Xtr, train_cand["label"].to_numpy(), gtr)
        fit_s = time.time() - t0

        # score eval pools
        sr = ranker.predict(Xr)
        sh = ranker.predict(Xh)

        m_random = _metric_row(cand_r, sr, ds, ks)
        m_hard = _metric_row(cand_h, sh, ds, ks)

        cold_mask = cand_r["user_idx"].isin(cold_users).to_numpy()
        m_cold = _metric_row(cand_r[cold_mask], sr[cold_mask], ds, ks)
        m_active = _metric_row(cand_r[~cold_mask], sr[~cold_mask], ds, ks)

        row = {
            "experiment": tag,
            "retriever": retr.name,
            "ranker": ranker.name,
            "loss": loss,
            "loss_family": ("listwise" if loss in ("listnet", "listmle")
                            else "pairwise" if loss in ("ranknet", "bpr")
                            else None),
            "n_users": ds.n_users,
            "n_items": ds.n_items,
            "retrieval": recall,
            "metrics_random": m_random,
            "metrics_hard": m_hard,
            "metrics_cold": m_cold,
            "metrics_active": m_active,
            "ranker_fit_seconds": round(fit_s, 2),
            "retriever_fit_seconds": round(getattr(retr, "fit_seconds", 0.0), 2),
            "config": dict(cfg),
        }
        if hasattr(ranker, "feature_importance"):
            row["feature_importance"] = ranker.feature_importance()
        results.append(row)

        headline = {k: round(m_random[k], 4) for k in
                    (f"ndcg@{ks[1]}", f"recall@{ks[1]}", f"map@{ks[1]}",
                     f"mrr@{ks[1]}")}
        print(f"  {tag}: {headline}")

        # keep first (or configured) ranker for the serving bundle
        if best_bundle is None:
            best_bundle = (ranker, fs_eval)

    # -- persist results --
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    for row in results:
        safe = row["experiment"].replace(":", "__")
        with open(paths.results_dir / f"{safe}.json", "w") as f:
            json.dump(row, f, indent=2, default=_json_default)
    print(f"Wrote {len(results)} result(s) to {paths.results_dir}")

    # -- optional serving bundle --
    if cfg.get("output", {}).get("save_serving", False) and best_bundle is not None:
        _save_serving_bundle(cfg, ds, retr, best_bundle[0], best_bundle[1], paths)

    return results


def _save_serving_bundle(cfg, ds, retriever, ranker, fs_eval, paths) -> None:
    import pickle
    if hasattr(retriever, "to_cpu"):
        retriever.to_cpu()
    if hasattr(ranker, "to_cpu"):
        ranker.to_cpu()
    bundle = {
        "dataset": ds,
        "retriever": retriever,
        "ranker": ranker,
        "feature_store": fs_eval,
        "config_name": cfg["_config_name"],
    }
    out = REPO_ROOT / "artifacts" / "serving_bundle.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved serving bundle -> {out}")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_experiment(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
