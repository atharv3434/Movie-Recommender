"""Synthetic movie ratings with *known* structure.

Each genre has a latent vector; movies sit near their genre's vector; each user likes 1-2 genres.
A rating depends on user-movie affinity + user bias + movie bias + noise. Users mostly rate popular
movies (like real platforms), so a plain "most popular" recommender is a genuinely strong baseline.
We save each user's true favourite genres so we can validate recommendations against ground truth.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402

rng = np.random.default_rng(C.SEED)
N_USERS, N_ITEMS, DIM = 3000, 600, 4
GENRES = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Thriller", "Documentary", "Animation"]

centroids = rng.normal(0, 1, (len(GENRES), DIM))
item_genre = rng.integers(0, len(GENRES), N_ITEMS)
Q = centroids[item_genre] + 0.35 * rng.normal(size=(N_ITEMS, DIM))

fav = [rng.choice(len(GENRES), size=rng.choice([1, 2]), replace=False) for _ in range(N_USERS)]
P = np.stack([centroids[f].mean(axis=0) for f in fav]) + 0.3 * rng.normal(size=(N_USERS, DIM))

affinity = P @ Q.T
affinity = (affinity - affinity.mean()) / affinity.std()
user_bias, item_bias = rng.normal(0, 0.3, N_USERS), rng.normal(0, 0.3, N_ITEMS)

pop = 1.0 / np.arange(1, N_ITEMS + 1) ** 0.8            # Zipf-like popularity
pop = pop[rng.permutation(N_ITEMS)]
pop /= pop.sum()
n_rated = np.clip((rng.pareto(1.6, N_USERS) + 1) * 12, 10, 300).astype(int)

rows = []
for u in range(N_USERS):
    items = rng.choice(N_ITEMS, size=n_rated[u], replace=False, p=pop)
    r = 3.3 + 1.0 * affinity[u, items] + user_bias[u] + item_bias[items] + rng.normal(0, 0.5, len(items))
    r = np.clip(np.round(r), 1, 5)
    rows += [(u + 1, i + 1, int(x)) for i, x in zip(items, r)]

ratings = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating"]).sample(frac=1, random_state=C.SEED)
C.DATA.mkdir(exist_ok=True)
ratings.to_csv(C.RATINGS_CSV, index=False)
pd.DataFrame({"movie_id": np.arange(1, N_ITEMS + 1),
              "title": [f"{GENRES[g]} Movie {i + 1}" for i, g in enumerate(item_genre)],
              "genre": [GENRES[g] for g in item_genre]}).to_csv(C.ITEMS_CSV, index=False)
pd.DataFrame({"user_id": np.arange(1, N_USERS + 1),
              "fav_genres": ["|".join(GENRES[g] for g in f) for f in fav]}).to_csv(C.USERS_CSV, index=False)

print(f"users={N_USERS:,} movies={N_ITEMS} ratings={len(ratings):,} "
      f"sparsity={1 - len(ratings) / (N_USERS * N_ITEMS):.1%} mean rating={ratings.rating.mean():.2f}")
