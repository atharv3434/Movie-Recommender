"""End-to-end: split -> baselines -> tuned ALS -> ranking evaluation -> export -> serving demo."""


import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyspark.ml.recommendation import ALS  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from src import config as C  # noqa: E402
from src import recsys as R  # noqa: E402
from src.serving import Recommender  # noqa: E402
from src.spark_session import get_spark  # noqa: E402


LAMBDAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5]


def main() -> None:
    t0 = time.time()
    spark = get_spark()
    ratings = R.load_ratings(spark)
    items = spark.read.csv(C.ITEMS_CSV, header=True).withColumn("movie_id", F.col("movie_id").cast("int"))
    truth = spark.read.csv(C.USERS_CSV, header=True).withColumn("user_id", F.col("user_id").cast("int"))

    train, test = R.split_per_user(ratings)
    train, test = train.cache(), test.cache()
    print(f"train={train.count():,}  test={test.count():,}  users={ratings.select('user_id').distinct().count():,}")

    # ---------------- baselines ----------------
    mu = train.agg(F.avg("rating")).first()[0]
    pop = R.popularity_table(train).cache()
    rmse_global = R.rmse(test.withColumn("prediction", F.lit(mu)))
    rmse_item = R.rmse(test.join(pop, "movie_id", "left")
                       .withColumn("prediction", F.coalesce("pop_score", F.lit(mu))))

    # ---------------- ALS with cross-validation (rank, regParam) ----------------
    print("tuning ALS (rank x regParam, 3-fold CV) ...")
    cv = R.tune_als(train)
    grid = sorted(zip(cv.avgMetrics, [{p.name: v for p, v in pm.items()} for pm in cv.getEstimatorParamMaps()]),
                  key=lambda t: t[0])
    for score, params in grid:
        print(f"   cv rmse={score:.4f}  {params}")
    best = cv.bestModel
    best_params = grid[0][1]   # grid is sorted by CV RMSE ascending (ALSModel does not expose regParam)
    rmse_als = R.rmse(best.transform(test).filter(F.col("prediction").isNotNull()))
    print(f"best params: {best_params}")

    # ---------------- choose the popularity blend (lambda) on a VALIDATION split ----------------
    # The test set must not influence any choice, so lambda is picked from a hold-out carved out of train.
    movies = items.select("movie_id")
    fit_part, val_part = R.split_per_user(train, seed=C.SEED + 1)
    fit_part, val_part = fit_part.cache(), val_part.cache()
    pop_fit = R.popularity_table(fit_part).cache()
    val_model = ALS(userCol="user_id", itemCol="movie_id", ratingCol="rating", coldStartStrategy="drop",
                    maxIter=12, seed=C.SEED, rank=int(best_params["rank"]),
                    regParam=float(best_params["regParam"])).fit(fit_part)
    val_users = val_part.select("user_id").distinct()
    print("\nselecting lambda on validation split (objective: NDCG@10)")
    val_scores = {}
    for lam in LAMBDAS:
        recs = R.recommend_hybrid(val_model, val_users, movies, pop_fit, fit_part, C.TOP_K, lam)
        val_scores[lam] = R.ranking_metrics(recs, val_part)["ndcgAtK"]
        print(f"   lambda={lam:<4} val NDCG@10={val_scores[lam]:.4f}")
    lam_star = max(val_scores, key=val_scores.get)
    print(f"chosen lambda = {lam_star}")

    # ---------------- final evaluation on the untouched TEST set ----------------
    users = test.select("user_id").distinct()
    systems = {
        "top_rated": R.recommend_popular(pop, users, train, C.TOP_K, "pop_score"),
        "most_rated": R.recommend_popular(pop, users, train, C.TOP_K, "n_ratings"),
        "als_pure": R.recommend_hybrid(best, users, movies, pop, train, C.TOP_K, 0.0),
        "als_hybrid": R.recommend_hybrid(best, users, movies, pop, train, C.TOP_K, lam_star),
    }
    results = {}
    for name, recs in systems.items():
        recs = recs.cache()
        results[name] = {**R.ranking_metrics(recs, test), "genre_alignment": R.genre_alignment(recs, items, truth)}
    sweep = {}
    for lam in LAMBDAS:                               # informational trade-off curve (not used for selection)
        recs = R.recommend_hybrid(best, users, movies, pop, train, C.TOP_K, lam).cache()
        r = R.ranking_metrics(recs, test)
        sweep[str(lam)] = {"ndcgAtK": r["ndcgAtK"], "genre_alignment": R.genre_alignment(recs, items, truth)}

    metrics = {"rmse": {"global_mean": rmse_global, "item_mean_shrunk": rmse_item, "als": rmse_als},
               "test_ranking_at_10": results, "lambda_sweep_test": sweep,
               "validation_ndcg_by_lambda": {str(k): v for k, v in val_scores.items()},
               "chosen_lambda": lam_star, "best_params": best_params}

    print("\n=========== RESULTS (held-out 20% of each user's ratings) ===========")
    print(f"RMSE   global mean {rmse_global:.3f} | item mean {rmse_item:.3f} | ALS {rmse_als:.3f}")
    print(f"{'system @10':<22}{'prec':>8}{'recall':>8}{'NDCG':>8}{'fav-genre':>11}")
    for name, r in results.items():
        print(f"{name:<22}{r['precisionAtK']:>8.4f}{r['recallAtK']:>8.4f}{r['ndcgAtK']:>8.4f}{r['genre_alignment']:>11.1%}")
    print(f"users evaluated: {results['als_hybrid']['users_evaluated']:,}")

    # ---------------- export ----------------
    C.OUT.mkdir(parents=True, exist_ok=True)
    best.write().overwrite().save(C.MODEL_DIR)
    pop_all = R.popularity_table(ratings)
    full_recs = (R.recommend_hybrid(best, ratings.select("user_id").distinct(), movies, pop_all, ratings,
                                    C.TOP_K, lam_star)
                 .join(items.select("movie_id", "title", "genre"), "movie_id"))
    full_recs.write.mode("overwrite").parquet(C.RECS)
    uf = sorted((r["id"], r["features"]) for r in best.userFactors.collect())
    vf = sorted((r["id"], r["features"]) for r in best.itemFactors.collect())
    pop_dict = {r["movie_id"]: float(r["n_ratings"]) for r in pop_all.collect()}
    np.savez(C.FACTORS_NPZ, user_ids=[u for u, _ in uf], U=[f for _, f in uf],
             item_ids=[m for m, _ in vf], V=[f for _, f in vf],
             pop_ids=list(pop_dict), pop_scores=list(pop_dict.values()))
    with open(C.METRICS_JSON, "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    spark.stop()

    # ---------------- serving demo (no Spark needed) ----------------
    it = pd.read_csv(C.ITEMS_CSV)
    z = np.load(C.FACTORS_NPZ)
    rec = Recommender(z["user_ids"], z["U"], z["item_ids"], z["V"],
                      dict(zip(z["pop_ids"].tolist(), z["pop_scores"].tolist())),
                      dict(zip(it.movie_id, it.title)), lam=lam_star)
    genre_of = dict(zip(it.movie_id, it.genre))
    print("\n=========== SERVING DEMO ===========")
    scifi = it[it.genre == "Sci-Fi"].movie_id.tolist()[:5]
    new = rec.for_new_user(scifi, k=10)
    share = np.mean([genre_of[r["movie_id"]] == "Sci-Fi" for r in new])
    print(f"cold-start user who likes 5 Sci-Fi movies -> {share:.0%} of top-10 are Sci-Fi")
    unknown = rec.for_user(999999, k=3)
    print("unknown user -> popularity fallback:", [r["title"] for r in unknown])
    sim = rec.similar_items(scifi[0], k=5)
    print(f"movies similar to '{rec.titles[scifi[0]]}':", [r["title"] for r in sim])
    print(f"\nDONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
