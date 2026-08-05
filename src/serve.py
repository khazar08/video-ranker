from __future__ import annotations
import pickle
from functools import lru_cache
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from .config import REPO_ROOT

BUNDLE_PATH = REPO_ROOT / "artifacts" / "serving_bundle.pkl"

app = FastAPI(title="Two-Stage Video Ranker",
              description="Retrieval -> ranking recommender over MovieLens.",
              version="1.0")


class RankRequest(BaseModel):
    user_id: int
    k: int = 10
    candidates: int = 200


class RankedItem(BaseModel):
    rank: int
    movie_id: int
    title: str
    score: float
    retrieval_score: float


class RankResponse(BaseModel):
    user_id: int
    config: str
    n_candidates: int
    items: List[RankedItem]


@lru_cache(maxsize=1)
def _bundle() -> dict:
    if not BUNDLE_PATH.exists():
        raise HTTPException(
            503, f"Serving bundle not found at {BUNDLE_PATH}. Train a config "
                 f"with output.save_serving: true first.")
    with open(BUNDLE_PATH, "rb") as f:
        return pickle.load(f)


def _rank_user(user_id: int, k: int, candidates: int) -> RankResponse:
    b = _bundle()
    ds, retriever, ranker, fs = (b["dataset"], b["retriever"],
                                 b["ranker"], b["feature_store"])

    if user_id not in ds.user_map:
        raise HTTPException(404, f"Unknown user_id {user_id}.")
    u = ds.user_map[user_id]

    # Stage 1: retrieve candidate items (train-seen already filtered).
    ids, ret_scores = retriever.recommend(np.array([u]), candidates)
    item_idx = ids[0]
    ret_score = ret_scores[0].astype(np.float32)

    cand = pd.DataFrame({"user_idx": np.full(len(item_idx), u), "item_idx": item_idx})

    # Stage 2: features (with the stage-1 score) + ranker score.
    X, _ = fs.transform(cand, extra={"retrieval_score": ret_score})
    scores = np.asarray(ranker.predict(X), dtype=np.float32)

    order = np.argsort(-scores)[:k]
    items = []
    for rank, j in enumerate(order, start=1):
        it = int(item_idx[j])
        items.append(RankedItem(
            rank=rank,
            movie_id=int(ds.item_id_inv[it]),
            title=str(ds.item_title[it]) or f"movie {ds.item_id_inv[it]}",
            score=float(scores[j]),
            retrieval_score=float(ret_score[j]),
        ))
    return RankResponse(user_id=user_id, config=b.get("config_name", "?"),
                        n_candidates=int(len(item_idx)), items=items)


@app.get("/")
def root() -> dict:
    ready = BUNDLE_PATH.exists()
    return {"service": "two-stage-video-ranker", "bundle_loaded": ready,
            "usage": "GET /rank?user_id=1&k=10"}

@app.get("/health")
def health() -> dict:
    b = _bundle()
    return {"status": "ok", "config": b.get("config_name"),
            "n_users": b["dataset"].n_users, "n_items": b["dataset"].n_items}


@app.get("/rank", response_model=RankResponse)
def rank_get(user_id: int = Query(..., ge=1),
             k: int = Query(10, ge=1, le=100),
             candidates: int = Query(200, ge=10, le=1000)) -> RankResponse:
    return _rank_user(user_id, k, candidates)


@app.post("/rank", response_model=RankResponse)
def rank_post(req: RankRequest) -> RankResponse:
    return _rank_user(req.user_id, req.k, req.candidates)
