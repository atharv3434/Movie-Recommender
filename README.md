# Movie Recommender (PySpark ALS) - with an honest evaluation

Collaborative-filtering recommender built on Spark MLlib **ALS**, compared against non-personalised
baselines, tuned with cross-validation, and extended with a popularity blend after the offline
evaluation exposed an **exposure-bias** problem. Includes a Spark-free serving layer with
cold-start handling.

## Quick start
```bash
pip install -r requirements.txt     # needs Java 17+
python scripts/generate_data.py     # 3,000 users x 600 movies, 84k ratings (known true tastes)
python src/train.py                 # ~7 min locally: CV tuning, lambda selection, evaluation, export
python scripts/report.py            # charts in reports/
pytest -q                           # 9 tests
```

## Results (20% of each user's ratings held out; 2,355 users evaluated)
Rating prediction (RMSE, lower is better): global mean **1.079** | per-movie mean **0.972** | ALS **0.653**.

| System (top-10) | Precision | Recall | NDCG | Recs in user's TRUE favourite genres |
|---|---|---|---|---|
| Top-rated (shrunk mean rating) | 0.0145 | 0.0471 | 0.0336 | 18.5% |
| Most-rated | 0.0589 | 0.2268 | 0.1830 | 18.7% |
| ALS only (lambda=0) | 0.0093 | 0.0348 | 0.0195 | **45.3%** |
| **ALS + popularity blend (lambda=1.5)** | **0.0686** | **0.2651** | **0.2138** | 30.7% |

The last column uses ground truth from the simulator that no model ever sees.

## The key finding: offline hit-rate rewards exposure, not taste
Pure ALS predicts ratings far better than any baseline and matches users' real tastes best (45%),
yet it **loses** to "most-rated" on hit-rate. Reason: users can only rate what they chose to
watch, mostly popular titles, so held-out "liked" items are dominated by popular ones, while ALS
ranks the entire catalog, including obscure movies nobody would have rated.

Fix: score = predicted rating + lambda * log(1 + #ratings). One knob, lambda, trades taste against
popularity (see `reports/tradeoff.png`). lambda was chosen on a **validation split carved from the
training data**; the test set was used once, for the final table.

## Method notes
- **Per-user split**: ~20% of every user's ratings are held out, keeping at least one training
  rating per user (a global random split can strand users with no history). Tested for no leakage.
- **Baselines first**: my first "popularity" baseline (top by mean rating) was a straw man;
  "most-rated" is the honest one. Both are reported.
- **Exact top-k over unseen items** via users x movies cross join (fine at this scale; at real
  scale use `recommendForAllUsers` with over-fetch, or approximate nearest neighbours).
- **Serving** (`src/serving.py`, numpy only): known user (factor dot product + blend), brand-new user
  (average factors of a few liked movies, "fold-in"), unknown user (most-rated fallback),
  and item-item similarity. In the demo a new user who likes 5 Sci-Fi movies gets 50% Sci-Fi in the top 10.

## Limitations (please read)
- **Synthetic data** with clean latent structure; ALS is nearly the true generating model, so
  absolute numbers flatter it. Real data is noisier. The *methodology* and the exposure-bias
  effect are what carry over.
- **lambda sits at the edge of its search grid (1.5)** and validation NDCG was still rising. NDCG
  alone also rewards pushing toward pure popularity. A better rule is to pick the lambda with the best
  taste-match among those beating the most-rated baseline. Extend `LAMBDAS` in `src/train.py` to explore.
- Offline metrics can't measure true preference when ratings are missing-not-at-random. Confirm
  with an online A/B test.
- Single run, no confidence intervals; differences of ~0.005 precision should not be over-read.
- ALS here has no user/item bias terms and no implicit-feedback variant (`implicitPrefs=True`),
  both natural next steps, as are time-aware splits.

## Layout
```
scripts/generate_data.py   simulator             src/recsys.py    split, ALS tuning, baselines, metrics, hybrid
scripts/report.py          charts                src/serving.py   numpy serving + cold start
src/train.py               pipeline              tests/           pytest (split, baselines, metrics, serving)
```
