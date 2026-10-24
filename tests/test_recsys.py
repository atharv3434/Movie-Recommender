import numpy as np
import pytest
from pyspark.sql import SparkSession

from src import recsys as R
from src.serving import Recommender


@pytest.fixture(scope="session")
def spark():
    s = (SparkSession.builder.master("local[1]").appName("tests")
         .config("spark.sql.shuffle.partitions", "2").config("spark.ui.enabled", "false").getOrCreate())
    yield s
    s.stop()


def _ratings(spark):
    rows = [(u, m, float(1 + (u + m) % 5)) for u in range(1, 21) for m in range(1, 16)]   # 20 users x 15 movies
    return spark.createDataFrame(rows, ["user_id", "movie_id", "rating"])


def test_split_has_no_leakage_and_keeps_every_user_in_train(spark):
    r = _ratings(spark)
    train, test = R.split_per_user(r, 0.2)
    keys = lambda d: {(x.user_id, x.movie_id) for x in d.collect()}
    assert keys(train).isdisjoint(keys(test))
    assert len(keys(train)) + len(keys(test)) == 20 * 15
    assert train.select("user_id").distinct().count() == 20          # nobody lost all history
    assert test.count() == 20 * 3                                    # floor(15 * 0.2) = 3 per user


def test_popularity_shrinkage_pulls_small_samples_to_global_mean(spark):
    # movie 1: one 5-star rating; movie 2: 100 ratings averaging 4.0; global mean ~ 4.0
    rows = [(1, 1, 5.0)] + [(u, 2, 4.0) for u in range(2, 102)]
    pop = {r.movie_id: r for r in R.popularity_table(
        spark.createDataFrame(rows, ["user_id", "movie_id", "rating"]), shrink=20).collect()}
    mu = (5.0 + 100 * 4.0) / 101                                       # global mean of the training ratings
    assert pop[1].pop_score == pytest.approx((5.0 + 20 * mu) / 21)     # one 5-star rating is shrunk hard...
    assert pop[1].pop_score < 4.1                                      # ...from 5.0 to ~4.06
    assert pop[2].pop_score == pytest.approx((400.0 + 20 * mu) / 120)  # many ratings barely move


def test_popular_baseline_excludes_seen_items(spark):
    pop = spark.createDataFrame([(1, 5.0, 100), (2, 4.0, 50), (3, 3.0, 10)], ["movie_id", "pop_score", "n_ratings"])
    users = spark.createDataFrame([(1,)], ["user_id"])
    seen = spark.createDataFrame([(1, 1)], ["user_id", "movie_id"])        # user 1 already saw movie 1
    recs = R.recommend_popular(pop, users, seen, 2, "n_ratings").orderBy("rank").collect()
    assert [r.movie_id for r in recs] == [2, 3]


def test_ranking_metrics_perfect_and_zero(spark):
    test = spark.createDataFrame([(1, 10, 5.0), (1, 11, 4.0)], ["user_id", "movie_id", "rating"])
    perfect = spark.createDataFrame([(1, 10, 1.0, 1), (1, 11, 0.9, 2)], ["user_id", "movie_id", "score", "rank"])
    wrong = spark.createDataFrame([(1, 99, 1.0, 1), (1, 98, 0.9, 2)], ["user_id", "movie_id", "score", "rank"])
    assert R.ranking_metrics(perfect, test, k=2)["ndcgAtK"] == pytest.approx(1.0)
    assert R.ranking_metrics(wrong, test, k=2)["ndcgAtK"] == pytest.approx(0.0)


# ------------------------------------------------------------ serving
def _rec(lam=0.0, counts=None):
    # 2-d factors: items 1,2 point "right", 3,4 point "up"; user 7 likes "right"
    V = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
    return Recommender([7], [[1.0, 0.0]], [1, 2, 3, 4], V,
                       counts or {1: 10, 2: 10, 3: 1000, 4: 5}, {1: "A", 2: "B", 3: "C", 4: "D"}, lam=lam)


def test_known_user_gets_taste_match_and_seen_items_are_excluded():
    out = _rec().for_user(7, seen=[1], k=2)
    assert [r["movie_id"] for r in out] == [2, 4]


def test_unknown_user_falls_back_to_most_rated():
    assert _rec().for_user(12345, k=1)[0]["movie_id"] == 3


def test_new_user_fold_in_uses_liked_items():
    out = _rec().for_new_user([3], k=1)
    assert out[0]["movie_id"] == 4                    # 3 is excluded, 4 is its nearest neighbour


def test_popularity_blend_can_override_taste():
    assert _rec(lam=0.0).for_user(7, k=1)[0]["movie_id"] == 1
    assert _rec(lam=1.0).for_user(7, k=1)[0]["movie_id"] == 3    # log(1001)=6.9 swamps the taste signal


def test_similar_items_excludes_itself():
    out = _rec().similar_items(1, k=3)
    assert 1 not in [r["movie_id"] for r in out] and out[0]["movie_id"] == 2
