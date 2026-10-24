"""Training, baselines and evaluation (all DataFrame-in / DataFrame-out)."""

from typing import Tuple

from pyspark.ml.evaluation import RankingEvaluator, RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel, ParamGridBuilder
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src import config as C


# ------------------------------------------------------------------ data
def load_ratings(spark) -> DataFrame:
    return (spark.read.csv(C.RATINGS_CSV, header=True)
            .select(F.col("user_id").cast("int"), F.col("movie_id").cast("int"),
                    F.col("rating").cast("float")))


def split_per_user(ratings: DataFrame, test_frac: float = 0.2, seed: int = C.SEED) -> Tuple[DataFrame, DataFrame]:
    """Hold out ~test_frac of *each user's* ratings, always keeping >=1 training rating per user.

    A global random split can leave users with no training data; splitting per user guarantees
    that every test user has history to learn from (so we evaluate ranking, not cold start).
    """
    w = Window.partitionBy("user_id").orderBy(F.rand(seed))
    r = (ratings.withColumn("rn", F.row_number().over(w))
         .withColumn("n", F.count("*").over(Window.partitionBy("user_id"))))
    is_test = (F.col("rn") <= F.greatest(F.lit(1), F.floor(F.col("n") * test_frac))) & (F.col("n") >= 5)
    return r.filter(~is_test).drop("rn", "n"), r.filter(is_test).drop("rn", "n")


# ------------------------------------------------------------------ models
def tune_als(train: DataFrame, seed: int = C.SEED) -> CrossValidatorModel:
    als = ALS(userCol="user_id", itemCol="movie_id", ratingCol="rating",
              coldStartStrategy="drop", maxIter=12, seed=seed)
    grid = (ParamGridBuilder()
            .addGrid(als.rank, [4, 8])
            .addGrid(als.regParam, [0.05, 0.1, 0.2])
            .build())
    cv = CrossValidator(estimator=als, estimatorParamMaps=grid, numFolds=3, parallelism=2, seed=seed,
                        evaluator=RegressionEvaluator(metricName="rmse", labelCol="rating",
                                                      predictionCol="prediction"))
    return cv.fit(train)


def rmse(pred: DataFrame) -> float:
    return RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction").evaluate(pred)


def popularity_table(train: DataFrame, shrink: int = 20) -> DataFrame:
    """Bayesian-shrunk mean rating: movies with few ratings are pulled toward the global mean."""
    mu = train.agg(F.avg("rating")).first()[0]
    return (train.groupBy("movie_id")
            .agg(F.sum("rating").alias("s"), F.count("*").alias("n"))
            .withColumn("pop_score", (F.col("s") + shrink * F.lit(mu)) / (F.col("n") + shrink))
            .select("movie_id", "pop_score", F.col("n").alias("n_ratings")))


# ------------------------------------------------------------------ recommendation
def _top_k_unseen(scored_pairs: DataFrame, k: int, score_col: str) -> DataFrame:
    w = Window.partitionBy("user_id").orderBy(F.desc(score_col), "movie_id")
    return (scored_pairs.withColumn("rank", F.row_number().over(w)).filter(F.col("rank") <= k)
            .select("user_id", "movie_id", F.col(score_col).alias("score"), "rank"))


def recommend_als(model: ALSModel, users: DataFrame, items: DataFrame, seen: DataFrame, k: int) -> DataFrame:
    """Exact top-k over all UNSEEN items for each user (users x items, minus already-rated)."""
    pairs = users.crossJoin(items).join(seen.select("user_id", "movie_id"), ["user_id", "movie_id"], "left_anti")
    scored = model.transform(pairs).filter(F.col("prediction").isNotNull())
    return _top_k_unseen(scored, k, "prediction")


def recommend_popular(pop: DataFrame, users: DataFrame, seen: DataFrame, k: int,
                      score_col: str = "pop_score") -> DataFrame:
    """Non-personalised baseline. score_col: 'pop_score' (shrunk mean rating) or 'n_ratings' (most rated)."""
    pairs = (users.crossJoin(pop.select("movie_id", score_col))
             .join(seen.select("user_id", "movie_id"), ["user_id", "movie_id"], "left_anti"))
    return _top_k_unseen(pairs, k, score_col)


def recommend_hybrid(model: ALSModel, users: DataFrame, items: DataFrame, pop: DataFrame,
                     seen: DataFrame, k: int, lam: float) -> DataFrame:
    """score = ALS predicted rating + lam * log(1 + #ratings).

    lam=0 is pure personalised ALS; large lam approaches 'most rated'. It is a single, tunable knob
    trading taste-match against the exposure effect that dominates offline hit-rate.
    """
    pairs = users.crossJoin(items).join(seen.select("user_id", "movie_id"), ["user_id", "movie_id"], "left_anti")
    scored = (model.transform(pairs).filter(F.col("prediction").isNotNull())
              .join(pop.select("movie_id", "n_ratings"), "movie_id", "left")
              .withColumn("hybrid", F.col("prediction") + F.lit(lam) * F.log1p(F.coalesce("n_ratings", F.lit(0)))))
    return _top_k_unseen(scored, k, "hybrid")


def top_n_most_rated(pop: DataFrame, n: int) -> DataFrame:
    """Candidate set for two-stage recommendation."""
    return pop.orderBy(F.desc("n_ratings")).limit(n).select("movie_id")


# ------------------------------------------------------------------ ranking evaluation
def ranking_metrics(recs: DataFrame, test: DataFrame, k: int = C.TOP_K) -> dict:
    """precision@k / recall@k / NDCG@k, where 'relevant' = held-out rating >= RELEVANT_RATING.

    Evaluated over users that have at least one relevant held-out item.
    """
    labels = (test.filter(F.col("rating") >= C.RELEVANT_RATING).groupBy("user_id")
              .agg(F.collect_list(F.col("movie_id").cast("double")).alias("label")))
    ordered = (recs.groupBy("user_id")
               .agg(F.sort_array(F.collect_list(F.struct("rank", "movie_id"))).alias("s"))
               .select("user_id", F.transform("s", lambda x: x["movie_id"].cast("double")).alias("prediction")))
    joined = labels.join(ordered, "user_id")
    out = {}
    for name in ["precisionAtK", "recallAtK", "ndcgAtK"]:
        out[name] = RankingEvaluator(metricName=name, k=k).evaluate(joined)
    out["users_evaluated"] = joined.count()
    return out


def genre_alignment(recs: DataFrame, items: DataFrame, users_truth: DataFrame) -> float:
    """Fraction of recommended movies that belong to the user's TRUE favourite genres.
    Ground truth comes from the simulator and is never shown to any model."""
    j = (recs.join(items.select("movie_id", "genre"), "movie_id").join(users_truth, "user_id")
         .withColumn("hit", F.expr("array_contains(split(fav_genres, '[|]'), genre)").cast("int")))
    return j.agg(F.avg("hit")).first()[0]
