from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RATINGS_CSV = str(DATA / "ratings.csv")
ITEMS_CSV = str(DATA / "items.csv")
USERS_CSV = str(DATA / "users.csv")       # ground truth taste (only used for validation, never for training)
OUT = DATA / "output"
MODEL_DIR = str(OUT / "als_model")
RECS = str(OUT / "recommendations")
METRICS_JSON = str(OUT / "metrics.json")
FACTORS_NPZ = str(OUT / "factors.npz")

SEED = 42
TOP_K = 10
RELEVANT_RATING = 4.0       # a held-out rating >= this counts as "the user liked it"
