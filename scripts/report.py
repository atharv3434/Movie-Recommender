"""Charts from the saved metrics.json."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402

m = json.load(open(C.METRICS_JSON))
res = m["test_ranking_at_10"]
names = {"top_rated": "Top-rated\n(no personal.)", "most_rated": "Most-rated\n(no personal.)",
         "als_pure": "ALS only\n(λ=0)", "als_hybrid": f"ALS + pop.\n(λ={m['chosen_lambda']})"}
colors = ["#b0b7c3", "#8895a7", "#4c8bf5", "#1a56c9"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for ax, (key, title) in zip(axes, [("ndcgAtK", "NDCG@10 (hit-rate, offline)"), ("recallAtK", "Recall@10"),
                                   ("genre_alignment", "Recs in user's true favourite genres")]):
    vals = [res[k][key] for k in names]
    bars = ax.bar(list(names.values()), vals, color=colors)
    ax.set_title(title, fontsize=11); ax.tick_params(axis="x", labelsize=8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}" if key != "genre_alignment" else f"{v:.0%}",
                ha="center", va="bottom", fontsize=9)
plt.tight_layout(); plt.savefig(C.ROOT / "reports" / "comparison.png", dpi=130); plt.close()

sw = m["lambda_sweep_test"]
lams = sorted(sw, key=float)
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.plot([float(l) for l in lams], [sw[l]["ndcgAtK"] for l in lams], "o-", color="#1a56c9", label="NDCG@10")
ax.set_xlabel("lambda (weight on popularity)"); ax.set_ylabel("NDCG@10", color="#1a56c9")
ax2 = ax.twinx()
ax2.plot([float(l) for l in lams], [sw[l]["genre_alignment"] for l in lams], "s--", color="#d9534f", label="taste match")
ax2.set_ylabel("share of recs in true favourite genres", color="#d9534f")
ax.axvline(m["chosen_lambda"], color="grey", ls=":"); ax.set_title("Popularity vs. personalisation trade-off")
plt.tight_layout(); plt.savefig(C.ROOT / "reports" / "tradeoff.png", dpi=130)
print("charts saved to reports/")
