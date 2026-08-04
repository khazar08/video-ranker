from __future__ import annotations
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class Dataset:
    """Everything downstream stages need, in index space (0..n_users/items-1)."""

    ratings: pd.DataFrame            # user_idx, item_idx, rating, ts, split
    n_users: int
    n_items: int
    user_map: Dict                  # original userId -> user_idx
    item_map: Dict                  # original movieId -> item_idx
    item_id_inv: np.ndarray         # item_idx -> original movieId
    item_genres: sparse.csr_matrix  # [n_items, n_genres] multi-hot
    genre_vocab: List[str]
    item_year: np.ndarray           # [n_items] release year (0 if unknown)
    item_title: np.ndarray          # [n_items] title strings
    item_pop: np.ndarray            # [n_items] train interaction counts
    seen: Dict[int, set]            # user_idx -> set of all interacted item_idx
    train_seen: Dict[int, np.ndarray]  # user_idx -> train-only item_idx (for masking)
    train_ui: sparse.csr_matrix     # [n_users, n_items] train confidence matrix
    config: dict = field(default_factory=dict)

    def split(self, name: str) -> pd.DataFrame:
        return self.ratings[self.ratings["split"] == name]

    @property
    def train(self) -> pd.DataFrame:
        return self.split("train")

    @property
    def valid(self) -> pd.DataFrame:
        return self.split("valid")

    @property
    def test(self) -> pd.DataFrame:
        return self.split("test")

    def positives_by_user(self, split: str) -> Dict[int, np.ndarray]:
        """Map user_idx -> array of item_idx for a given split."""
        df = self.split(split)
        return {u: g["item_idx"].to_numpy()
                for u, g in df.groupby("user_idx", sort=False)}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path) -> "Dataset":
        with open(path, "rb") as f:
            return pickle.load(f)


def _parse_movies(movies: pd.DataFrame, item_map: Dict) -> tuple:
    """Build multi-hot genre matrix, release-year and title arrays (index space)."""
    n_items = len(item_map)
    movies = movies[movies["movieId"].isin(item_map)].copy()
    movies["item_idx"] = movies["movieId"].map(item_map)

    # Genres are '|'-separated; "(no genres listed)" -> empty.
    genre_lists = movies["genres"].apply(
        lambda s: [] if s == "(no genres listed)" else s.split("|"))
    vocab = sorted({g for gs in genre_lists for g in gs})
    gidx = {g: j for j, g in enumerate(vocab)}

    rows, cols = [], []
    for it, gs in zip(movies["item_idx"], genre_lists):
        for g in gs:
            rows.append(it)
            cols.append(gidx[g])
    item_genres = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_items, len(vocab)))

    # Release year from a trailing "(YYYY)" in the title.
    year = np.zeros(n_items, dtype=np.int32)
    title = np.empty(n_items, dtype=object)
    title[:] = ""
    yr = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0]
    for it, t, y in zip(movies["item_idx"], movies["title"], yr):
        title[it] = t
        if isinstance(y, str) and y.isdigit():
            year[it] = int(y)
    return item_genres, vocab, year, title


def prepare_dataset(
    dataset_dir: Path,
    positive_threshold: float = 4.0,
    min_interactions: int = 5,
    n_test: int = 2,
    n_valid: int = 2,
    max_users: Optional[int] = None,
    confidence: str = "linear",   # "linear" -> 1 + alpha*(rating-thr), or "binary"
    alpha: float = 2.0,
    seed: int = 42,
    cache_path: Optional[Path] = None,
) -> Dataset:
    """Load ratings/movies and produce a `Dataset` with the temporal split."""
    if cache_path is not None and Path(cache_path).exists():
        return Dataset.load(Path(cache_path))

    rng = np.random.default_rng(seed)
    ratings = pd.read_csv(dataset_dir / "ratings.csv")
    movies = pd.read_csv(dataset_dir / "movies.csv")

    # Keep positives only (implicit-feedback style engagement signal).
    pos = ratings[ratings["rating"] >= positive_threshold].copy()

    # Drop users without enough history to form train + valid + test.
    need = min_interactions
    counts = pos.groupby("userId").size()
    keep_users = counts[counts >= max(need, n_test + n_valid + 1)].index
    pos = pos[pos["userId"].isin(keep_users)]

    # Optional user subsample for fast iteration / tractable 25M runs.
    if max_users is not None and pos["userId"].nunique() > max_users:
        chosen = rng.choice(pos["userId"].unique(), size=max_users, replace=False)
        pos = pos[pos["userId"].isin(chosen)]

    # Contiguous index maps.
    uids = np.sort(pos["userId"].unique())
    iids = np.sort(pos["movieId"].unique())
    user_map = {u: i for i, u in enumerate(uids)}
    item_map = {m: i for i, m in enumerate(iids)}
    item_id_inv = iids.copy()

    pos = pos.assign(
        user_idx=pos["userId"].map(user_map),
        item_idx=pos["movieId"].map(item_map),
        ts=pos["timestamp"].astype(np.int64),
    )[["user_idx", "item_idx", "rating", "ts"]]

    # Temporal per-user split. Sort by (ts, item_idx) for a deterministic tie
    # order, then label the tail as valid/test.
    pos = pos.sort_values(["user_idx", "ts", "item_idx"]).reset_index(drop=True)
    rank_from_end = pos.groupby("user_idx").cumcount(ascending=False)
    split = np.full(len(pos), "train", dtype=object)
    split[rank_from_end.to_numpy() < n_test] = "test"
    split[(rank_from_end.to_numpy() >= n_test)
          & (rank_from_end.to_numpy() < n_test + n_valid)] = "valid"
    pos["split"] = split

    n_users, n_items = len(user_map), len(item_map)

    # Genre / year / title metadata.
    item_genres, genre_vocab, item_year, item_title = _parse_movies(movies, item_map)

    # Seen sets (all splits) for negative-sampling exclusion.
    seen: Dict[int, set] = {u: set(g["item_idx"].tolist())
                            for u, g in pos.groupby("user_idx", sort=False)}

    # Train popularity + confidence matrix for ALS.
    train = pos[pos["split"] == "train"]
    # Train-only seen items: what retrieval is allowed to filter out. Valid/test
    # positives are deliberately NOT here, so they remain retrievable.
    train_seen: Dict[int, np.ndarray] = {
        u: g["item_idx"].to_numpy().copy()
        for u, g in train.groupby("user_idx", sort=False)}
    item_pop = np.bincount(train["item_idx"].to_numpy(), minlength=n_items).astype(np.float64)
    if confidence == "binary":
        conf = np.ones(len(train), dtype=np.float32)
    else:  # linear confidence weighting on top of the positive threshold
        conf = (1.0 + alpha * (train["rating"].to_numpy() - positive_threshold)).astype(np.float32)
    train_ui = sparse.csr_matrix(
        (conf, (train["user_idx"].to_numpy(), train["item_idx"].to_numpy())),
        shape=(n_users, n_items))

    ds = Dataset(
        ratings=pos, n_users=n_users, n_items=n_items,
        user_map=user_map, item_map=item_map, item_id_inv=item_id_inv,
        item_genres=item_genres, genre_vocab=genre_vocab,
        item_year=item_year, item_title=item_title, item_pop=item_pop,
        seen=seen, train_seen=train_seen, train_ui=train_ui,
        config=dict(positive_threshold=positive_threshold,
                    min_interactions=min_interactions, n_test=n_test,
                    n_valid=n_valid, max_users=max_users, seed=seed),
    )
    if cache_path is not None:
        ds.save(Path(cache_path))
    return ds


# --- Negative sampling / candidate generation -------------------------------

def _sample_negatives_for_user(
    n_needed: int, seen: set, n_items: int, rng: np.random.Generator,
    pop_probs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Sample `n_needed` distinct item_idx not in `seen`.

    If `pop_probs` is given, sample popularity-weighted (hard negatives),
    otherwise uniform. Uses oversample-and-reject for efficiency.
    """
    if n_needed <= 0:
        return np.empty(0, dtype=np.int64)
    picked: set = set()
    out: List[int] = []
    attempts = 0
    while len(out) < n_needed and attempts < 50:
        batch = max(n_needed * 2, 16)
        if pop_probs is not None:
            cand = rng.choice(n_items, size=batch, p=pop_probs)
        else:
            cand = rng.integers(0, n_items, size=batch)
        for c in cand:
            c = int(c)
            if c not in seen and c not in picked:
                picked.add(c)
                out.append(c)
                if len(out) == n_needed:
                    break
        attempts += 1
    return np.asarray(out, dtype=np.int64)


def sample_train_candidates(
    ds: Dataset, k_neg: int = 4, hard_fraction: float = 0.5, seed: int = 0,
) -> pd.DataFrame:
    """Ranker-training candidates: valid positives (label 1) + sampled negatives.

    For each valid positive we draw `k_neg` negatives; a `hard_fraction` of them
    are popularity-weighted, the rest uniform. Returns columns
    [user_idx, item_idx, label].
    """
    rng = np.random.default_rng(seed)
    pop_probs = _pop_probs(ds)
    rows_u, rows_i, rows_y = [], [], []

    valid_by_user = ds.positives_by_user("valid")
    for u, items in valid_by_user.items():
        n_pos = len(items)
        rows_u.extend([u] * n_pos)
        rows_i.extend(items.tolist())
        rows_y.extend([1] * n_pos)

        n_neg = n_pos * k_neg
        n_hard = int(round(n_neg * hard_fraction))
        hard = _sample_negatives_for_user(n_hard, ds.seen[u], ds.n_items, rng, pop_probs)
        easy = _sample_negatives_for_user(n_neg - n_hard, ds.seen[u], ds.n_items, rng)
        negs = np.concatenate([hard, easy])
        rows_u.extend([u] * len(negs))
        rows_i.extend(negs.tolist())
        rows_y.extend([0] * len(negs))

    return pd.DataFrame({"user_idx": rows_u, "item_idx": rows_i, "label": rows_y})


def sample_eval_candidates(
    ds: Dataset, n_neg: int = 100, mode: str = "random", seed: int = 1,
) -> pd.DataFrame:
    """Eval candidate pool: all test positives (label 1) + `n_neg` negatives.

    `mode="random"` -> uniform negatives (easy slice); `mode="hard"` ->
    popularity-weighted negatives. The pool contains every test positive so the
    Recall/MAP denominator (n_relevant = #test positives) is exact.
    Returns [user_idx, item_idx, label].
    """
    rng = np.random.default_rng(seed)
    pop_probs = _pop_probs(ds) if mode == "hard" else None
    rows_u, rows_i, rows_y = [], [], []

    test_by_user = ds.positives_by_user("test")
    for u, items in test_by_user.items():
        rows_u.extend([u] * len(items))
        rows_i.extend(items.tolist())
        rows_y.extend([1] * len(items))

        negs = _sample_negatives_for_user(n_neg, ds.seen[u], ds.n_items, rng, pop_probs)
        rows_u.extend([u] * len(negs))
        rows_i.extend(negs.tolist())
        rows_y.extend([0] * len(negs))

    return pd.DataFrame({"user_idx": rows_u, "item_idx": rows_i, "label": rows_y})


def _pop_probs(ds: Dataset) -> np.ndarray:
    """Popularity distribution over items (smoothed), for hard negatives."""
    pop = ds.item_pop + 1.0        # Laplace smoothing so unseen items are reachable
    return pop / pop.sum()
