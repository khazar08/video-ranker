from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from .data_prep import Dataset

class FeatureStore:

    def __init__(self, ds: Dataset, history: pd.DataFrame):
        self.ds = ds
        G = ds.item_genres.shape[1]
        self.n_genres = G
        self.genre_names = list(ds.genre_vocab)

        n_users, n_items = ds.n_users, ds.n_items
        item_genre_dense = ds.item_genres.toarray().astype(np.float32)  # [n_items, G]
        self._item_genre = item_genre_dense

        global_mean = float(history["rating"].mean()) if len(history) else 4.0
        self.global_mean = global_mean

        # --- user features ---
        self.u_count = np.zeros(n_users, dtype=np.float32)
        self.u_mean = np.full(n_users, global_mean, dtype=np.float32)
        self.u_recency = np.zeros(n_users, dtype=np.float32)
        self.u_span = np.zeros(n_users, dtype=np.float32)
        self.u_genre_aff = np.zeros((n_users, G), dtype=np.float32)

        ts = history["ts"].to_numpy()
        ts_min, ts_max = (ts.min(), ts.max()) if len(ts) else (0, 1)
        ts_range = max(ts_max - ts_min, 1)

        for u, g in history.groupby("user_idx", sort=False):
            items = g["item_idx"].to_numpy()
            self.u_count[u] = len(items)
            self.u_mean[u] = g["rating"].mean()
            last, first = g["ts"].max(), g["ts"].min()
            self.u_recency[u] = (last - ts_min) / ts_range
            self.u_span[u] = (last - first) / ts_range
            aff = item_genre_dense[items].sum(axis=0)
            s = aff.sum()
            if s > 0:
                aff = aff / s
            self.u_genre_aff[u] = aff
        # log1p the raw count so its scale is comparable to the other features.
        self.u_count_log = np.log1p(self.u_count)

        # --- item features (from the same history) ---
        self.i_count = np.zeros(n_items, dtype=np.float32)
        self.i_mean = np.full(n_items, global_mean, dtype=np.float32)
        for it, g in history.groupby("item_idx", sort=False):
            self.i_count[it] = len(g)
            self.i_mean[it] = g["rating"].mean()
        self.i_count_log = np.log1p(self.i_count)

        year = ds.item_year.astype(np.float32)
        self.i_year_known = (year > 0).astype(np.float32)
        yr_valid = year[year > 0]
        y_min, y_max = (yr_valid.min(), yr_valid.max()) if yr_valid.size else (1900, 2000)
        self.i_year_norm = np.where(year > 0, (year - y_min) / max(y_max - y_min, 1), 0.0
                                    ).astype(np.float32)

        self._names = self._build_names()

    # --- feature names, in the exact order transform() emits them -----------
    def _build_names(self) -> List[str]:
        names = [
            "u_count_log", "u_mean", "u_recency", "u_span",
            "i_count_log", "i_mean", "i_year_norm", "i_year_known",
            "cross_genre_match",
        ]
        names += [f"u_aff_{g}" for g in self.genre_names]
        names += [f"i_genre_{g}" for g in self.genre_names]
        return names

    @property
    def feature_names(self) -> List[str]:
        return list(self._names)

    def transform(
        self,
        candidates: pd.DataFrame,
        extra: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, List[str]]:

        u = candidates["user_idx"].to_numpy()
        it = candidates["item_idx"].to_numpy()

        u_aff = self.u_genre_aff[u]                 # [N, G]
        i_gen = self._item_genre[it]                # [N, G]
        genre_match = np.sum(u_aff * i_gen, axis=1)  # cross feature

        base = np.column_stack([
            self.u_count_log[u], self.u_mean[u], self.u_recency[u], self.u_span[u],
            self.i_count_log[it], self.i_mean[it], self.i_year_norm[it],
            self.i_year_known[it], genre_match,
        ]).astype(np.float32)

        X = np.hstack([base, u_aff, i_gen]).astype(np.float32)
        names = list(self._names)

        if extra:
            cols, extra_names = [], []
            for name, arr in extra.items():
                arr = np.asarray(arr, dtype=np.float32).reshape(-1, 1)
                if arr.shape[0] != X.shape[0]:
                    raise ValueError(f"extra['{name}'] length mismatch")
                cols.append(arr)
                extra_names.append(name)
            X = np.hstack([X] + cols).astype(np.float32)
            names = names + extra_names

        return X, names
