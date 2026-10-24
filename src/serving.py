"""Lightweight, Spark-free serving layer built from exported factors (numpy).

Handles three situations a production recommender must:
  1. known user          -> dot product of learned user and item factors
  2. brand-new user      -> "fold-in": average the factors of the few movies they said they like
  3. user with no signal -> popularity fallback
"""
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


class Recommender:
    def __init__(self, user_ids, user_factors, item_ids, item_factors, counts: Dict[int, float],
                 titles: Optional[Dict[int, str]] = None, lam: float = 0.0):
        self.uidx = {int(u): i for i, u in enumerate(user_ids)}
        self.U = np.asarray(user_factors, dtype=float)
        self.item_ids = np.asarray(item_ids, dtype=int)
        self.iidx = {int(m): i for i, m in enumerate(self.item_ids)}
        self.V = np.asarray(item_factors, dtype=float)
        self.counts = counts
        self.lam = lam   # popularity blend weight, same formula as training-time recommend_hybrid
        self.titles = titles or {}

    def _blend(self, raw: np.ndarray) -> np.ndarray:
        """predicted rating + lam * log(1 + #ratings)"""
        pop = np.log1p(np.array([self.counts.get(int(m), 0.0) for m in self.item_ids]))
        return raw + self.lam * pop

    def _top(self, scores: np.ndarray, exclude: Iterable[int], k: int) -> List[dict]:
        scores = scores.copy()
        for m in exclude:
            if m in self.iidx:
                scores[self.iidx[m]] = -np.inf
        top = np.argsort(-scores)[:k]
        return [{"movie_id": int(self.item_ids[i]), "title": self.titles.get(int(self.item_ids[i]), ""),
                 "score": float(scores[i])} for i in top if np.isfinite(scores[i])]

    def popular(self, k: int = 10, exclude: Iterable[int] = ()) -> List[dict]:
        s = np.array([self.counts.get(int(m), 0.0) for m in self.item_ids])   # most-rated
        return self._top(s, exclude, k)

    def for_user(self, user_id: int, seen: Iterable[int] = (), k: int = 10) -> List[dict]:
        if user_id not in self.uidx:
            return self.popular(k, seen)               # unknown user -> popularity fallback
        return self._top(self._blend(self.V @ self.U[self.uidx[user_id]]), seen, k)

    def for_new_user(self, liked_movie_ids: Sequence[int], k: int = 10) -> List[dict]:
        idx = [self.iidx[m] for m in liked_movie_ids if m in self.iidx]
        if not idx:
            return self.popular(k)
        profile = self.V[idx].mean(axis=0)             # fold-in approximation of a user vector
        return self._top(self._blend(self.V @ profile), liked_movie_ids, k)

    def similar_items(self, movie_id: int, k: int = 10) -> List[dict]:
        if movie_id not in self.iidx:
            return []
        v = self.V[self.iidx[movie_id]]
        sims = (self.V @ v) / (np.linalg.norm(self.V, axis=1) * np.linalg.norm(v) + 1e-12)
        return self._top(sims, [movie_id], k)
