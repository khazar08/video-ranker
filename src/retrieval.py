from __future__ import annotations
import time
from typing import Optional, Tuple
import numpy as np
from .data_prep import Dataset


class Retriever:

    name: str = "base"

    def fit(self, ds: Dataset) -> "Retriever":
        raise NotImplementedError

    def recommend(self, users: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def score(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# --- ALS --------------------------------------------------------------------

class ALSRetriever(Retriever):
    """Alternating Least Squares matrix factorization via `implicit`."""

    name = "als"

    def __init__(self, factors: int = 64, regularization: float = 0.05,
                 iterations: int = 20, seed: int = 42, **_ignore):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.seed = seed
        self.model = None
        self._train_ui = None

    def fit(self, ds: Dataset) -> "ALSRetriever":
        from implicit.als import AlternatingLeastSquares

        self._ds = ds
        self._train_ui = ds.train_ui.tocsr()
        self.model = AlternatingLeastSquares(
            factors=self.factors, regularization=self.regularization,
            iterations=self.iterations, random_state=self.seed,
            use_gpu=False,
        )
        t0 = time.time()
        self.model.fit(self._train_ui, show_progress=False)
        self.fit_seconds = time.time() - t0
        return self

    def recommend(self, users: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray]:
        users = np.asarray(users)
        ids, scores = self.model.recommend(
            users, self._train_ui[users], N=N,
            filter_already_liked_items=True,
        )
        return np.asarray(ids), np.asarray(scores)

    def score(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        uf = np.asarray(self.model.user_factors)
        vf = np.asarray(self.model.item_factors)
        u = np.asarray(user_idx)
        it = np.asarray(item_idx)
        return np.sum(uf[u] * vf[it], axis=1).astype(np.float32)


# --- Two-tower --------------------------------------------------------------

import torch  # noqa: E402  (top-level so the model class below is picklable)
import torch.nn as nn  # noqa: E402


class TwoTowerNet(nn.Module):
    """User tower (id embedding) + item tower (id embedding + genre content).

    Defined at module level (not nested in the retriever) so a fitted retriever
    pickles cleanly for the serving bundle.
    """

    def __init__(self, n_users: int, n_items: int, n_genres: int,
                 dim: int, hidden: int):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.05)
        nn.init.normal_(self.item_emb.weight, std=0.05)
        self.user_mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.item_mlp = nn.Sequential(
            nn.Linear(dim + n_genres, hidden), nn.ReLU(), nn.Linear(hidden, dim))

    def user_vec(self, u):
        return self.user_mlp(self.user_emb(u))

    def item_vec(self, it, genres):
        return self.item_mlp(torch.cat([self.item_emb(it), genres], dim=1))


class TwoTowerRetriever(Retriever):
    """Two-tower retriever: user/item towers, dot-product, in-batch softmax.

    The item tower consumes an id embedding concatenated with the item's genre
    multi-hot (light content signal); the user tower is an id embedding. Both
    pass through a small MLP to the shared embedding dimension. Training uses
    in-batch negatives with a sampled-softmax (cross-entropy) loss and an
    optional log-Q popularity correction.
    """

    name = "two_tower"

    def __init__(self, dim: int = 64, hidden: int = 128, epochs: int = 8,
                 batch_size: int = 1024, lr: float = 1e-2, temperature: float = 0.07,
                 logq_correction: bool = True, weight_decay: float = 1e-5,
                 seed: int = 42, device: str = "cpu", verbose: bool = False,
                 **_ignore):
        self.dim = dim
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.temperature = temperature
        self.logq_correction = logq_correction
        self.weight_decay = weight_decay
        self.seed = seed
        # CPU by default so runs are bit-reproducible (MPS/CUDA reductions are
        # nondeterministic); set device: mps|cuda in the config to trade
        # reproducibility for speed on large data.
        self.device = device
        self.verbose = verbose
        self.model = None

    def _build_model(self, ds: Dataset) -> TwoTowerNet:
        return TwoTowerNet(ds.n_users, ds.n_items, ds.item_genres.shape[1],
                           self.dim, self.hidden)

    def fit(self, ds: Dataset) -> "TwoTowerRetriever":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._ds = ds
        self.device = (torch.device(self.device) if isinstance(self.device, str)
                       else self.device)
        self.model = self._build_model(ds).to(self.device)

        genres = torch.from_numpy(ds.item_genres.toarray().astype("float32")).to(self.device)
        self._item_genres_t = genres

        # log-Q correction term: probability of sampling each item (~popularity).
        pop = ds.item_pop + 1.0
        logq = torch.from_numpy(np.log(pop / pop.sum()).astype("float32")).to(self.device)

        train = ds.train
        users = torch.from_numpy(train["user_idx"].to_numpy().copy()).long()
        items = torch.from_numpy(train["item_idx"].to_numpy().copy()).long()
        n = users.shape[0]

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        g = torch.Generator().manual_seed(self.seed)

        t0 = time.time()
        self.model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            total = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                if idx.numel() < 2:
                    continue
                u = users[idx].to(self.device)
                it = items[idx].to(self.device)
                uv = self.model.user_vec(u)
                iv = self.model.item_vec(it, genres[it])
                # in-batch logits: user i vs every positive item in the batch.
                logits = (uv @ iv.t()) / self.temperature
                if self.logq_correction:
                    logits = logits - logq[it].unsqueeze(0)
                target = torch.arange(idx.numel(), device=self.device)
                loss = torch.nn.functional.cross_entropy(logits, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * idx.numel()
            if self.verbose:
                print(f"  [two_tower] epoch {epoch + 1}/{self.epochs} "
                      f"loss={total / n:.4f}")
        self.fit_seconds = time.time() - t0

        # Precompute item vectors for fast retrieval / scoring.
        self._refresh_item_vecs()
        return self

    def _refresh_item_vecs(self):
        self.model.eval()
        with torch.no_grad():
            all_items = torch.arange(self._ds.n_items, device=self.device)
            self._item_vecs = self.model.item_vec(
                all_items, self._item_genres_t)  # [n_items, dim]

    def recommend(self, users: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray]:
        users = np.asarray(users)
        self.model.eval()
        out_ids = np.empty((len(users), N), dtype=np.int64)
        out_scores = np.empty((len(users), N), dtype=np.float32)
        iv = self._item_vecs  # [n_items, dim]
        with torch.no_grad():
            for start in range(0, len(users), 512):
                batch = users[start:start + 512]
                uv = self.model.user_vec(
                    torch.from_numpy(batch).long().to(self.device))  # [b, dim]
                scores = uv @ iv.t()                                 # [b, n_items]
                # mask each user's *train-seen* items so we retrieve novel items
                # (valid/test positives stay retrievable -- no leakage of labels).
                for r, u in enumerate(batch):
                    seen = self._ds.train_seen.get(int(u))
                    if seen is not None and len(seen):
                        scores[r, seen] = -1e9
                top_scores, top_idx = torch.topk(scores, N, dim=1)
                out_ids[start:start + len(batch)] = top_idx.cpu().numpy()
                out_scores[start:start + len(batch)] = top_scores.cpu().numpy()
        return out_ids, out_scores

    def score(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        self.model.eval()
        u = np.asarray(user_idx)
        it = np.asarray(item_idx)
        out = np.empty(len(u), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(u), 8192):
                ub = torch.from_numpy(u[start:start + 8192]).long().to(self.device)
                ib = it[start:start + 8192]
                uv = self.model.user_vec(ub)
                iv = self._item_vecs[torch.from_numpy(ib).long().to(self.device)]
                out[start:start + len(ib)] = torch.sum(uv * iv, dim=1).cpu().numpy()
        return out

    def to_cpu(self) -> "TwoTowerRetriever":
        """Move tensors to CPU so the retriever pickles/serves portably."""
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self._item_genres_t = self._item_genres_t.to(self.device)
        self._refresh_item_vecs()
        return self


def build_retriever(kind: str, params: dict) -> Retriever:
    """Factory: 'als' or 'two_tower'."""
    kind = kind.lower()
    if kind == "als":
        return ALSRetriever(**params)
    if kind in ("two_tower", "twotower", "tt"):
        return TwoTowerRetriever(**params)
    raise ValueError(f"Unknown retriever '{kind}'")
