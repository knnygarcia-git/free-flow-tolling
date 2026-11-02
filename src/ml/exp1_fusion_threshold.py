"""
Experiment 1: fuse the reads with a model and pick the threshold by cost.

Question: can we bill more passes automatically, with fewer mistakes, than the fixed rule
"tag read OR OCR confidence >= 90%"?

Method
  1. Billing candidate: the tag's plate (if read) or the best OCR read.
  2. Target: is the candidate the true plate? (in real life: a label coming from human review)
  3. Model: gradient boosting with isotonic calibration -> P(candidate is correct).
  4. Threshold: bill automatically if P >= t; otherwise send to human review.
     t is chosen to minimize the expected cost:
        cost = reviews x REVIEW_COST + wrong charges x ERROR_COST
  5. Time-based validation: train on days 1-2, test on day 3.

It also computes an evidence-storage policy driven by the same probability.

Outputs: figures 06, 07, 08 and 12; reports/tables/exp1_*
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, roc_auc_score

from common import (AMBER, FEATURES, FIG, GRAY, GREEN, INK_2, INK_3, LIGHT_GRAY, NAMES, TAB,
                    TEST_DAY, TRAIN_DAYS, build_features, grid_x, load, num, save, title)

REVIEW_COST = 0.60       # R$ per image checked by an operator (assumption)
ERROR_COST = 30.00       # R$ per charge on the wrong plate: dispute, refund, customer service (assumption)
KB = 1e3
FULL_EVIDENCE = 965.3 * KB    # full package (see src/data_volume.py)
SLIM_EVIDENCE = 35.0 * KB     # plate crops + metadata + laser profile only


def summarize(name, accept, correct):
    n = len(correct)
    rev = (~accept).sum()
    errors = (accept & (correct == 0)).sum()
    return {
        "strategy": name,
        "auto_billed_pct": 100 * accept.mean(),
        "review_pct": 100 * rev / n,
        "wrong_charges": int(errors),
        "errors_per_10k": 1e4 * errors / n,
        "auto_accuracy_pct": 100 * (correct[accept] == 1).mean(),
        "cost_R$": rev * REVIEW_COST + errors * ERROR_COST,
        "cost_per_10k_R$": 1e4 * (rev * REVIEW_COST + errors * ERROR_COST) / n,
    }


def main():
    print("Building features...")
    df = build_features(load())
    tr, te = df[df.day.isin(TRAIN_DAYS)], df[df.day == TEST_DAY]
    Xtr, ytr, Xte, yte = tr[FEATURES], tr["correct"], te[FEATURES], te["correct"].to_numpy()
    print(f"train {len(tr):,} | test {len(te):,} | wrong candidates in test: {(yte == 0).mean():.2%}")

    base = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=0)
    model = CalibratedClassifierCV(base, method="isotonic", cv=3).fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    print(f"AUC {roc_auc_score(yte, p):.4f} | Brier {brier_score_loss(yte, p):.5f}")

    # --- current rule (baseline)
    rule = (te["tag_read"] | (te["conf_best"] >= 0.90)).to_numpy()
    # --- cost curve
    ts = np.unique(np.concatenate([np.linspace(0.5, 0.999, 400), [0.9995, 0.9999]]))
    costs = np.array([summarize("", p >= t, yte)["cost_R$"] for t in ts])
    t_best = ts[np.argmin(costs)]
    t_theory = 1 - REVIEW_COST / ERROR_COST
    # the same curve for the rule (moving only the OCR confidence threshold)
    ts_conf = np.linspace(0.5, 0.999, 300)
    costs_rule = np.array([summarize("", (te["tag_read"] | (te["conf_best"] >= t)).to_numpy(),
                                     yte)["cost_R$"] for t in ts_conf])

    table = pd.DataFrame([
        summarize("Current rule (tag or OCR >= 90%)", rule, yte),
        summarize(f"Rule with the best OCR threshold ({ts_conf[np.argmin(costs_rule)]:.3f})",
                  (te["tag_read"] | (te["conf_best"] >= ts_conf[np.argmin(costs_rule)])).to_numpy(), yte),
        summarize(f"Model, best threshold ({t_best:.3f})", p >= t_best, yte),
    ])
    table.to_csv(TAB / "exp1_comparison.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n" + table.round(2).to_string(index=False))

    # cases the rule let through and the model caught
    caught = te[rule & (p < t_best) & (yte == 0)]
    kinds = pd.Series(np.where(caught["_tag_other_vehicle"], "tag on another vehicle",
                               "wrong OCR with high confidence")).value_counts()
    print("\nWrong charges avoided by the model:\n" + kinds.to_string())

    # ---------------------------------------------------------------- figures
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    y_mod, y_rule = costs / len(te) * 1e4, costs_rule / len(te) * 1e4
    ax.plot(ts_conf, y_rule, color=GRAY, lw=2)
    ax.plot(ts, y_mod, color=GREEN, lw=2)
    ax.fill_between(ts, y_mod, color=GREEN, alpha=0.08, lw=0)
    b = summarize("", rule, yte)["cost_per_10k_R$"]
    m = costs.min() / len(te) * 1e4
    ax.scatter([0.90], [b], s=40, color=GRAY, zorder=4, edgecolor="white", linewidth=2)
    ax.annotate(f"common rule (90%)\nR$ {num(b)}", (0.90, b), xytext=(-12, -34),
                textcoords="offset points", ha="right", fontsize=9, color=INK_2)
    ax.scatter([t_best], [m], s=40, color=GREEN, zorder=4, edgecolor="white", linewidth=2)
    ax.annotate(f"model at its best cut-off\nR$ {num(m)}", (t_best, m), xytext=(-10, 14),
                textcoords="offset points", ha="right", fontsize=9, color=INK_2)
    ax.annotate("rule: cut by the camera's confidence", (0.6, y_rule[np.searchsorted(ts_conf, 0.6)]),
                xytext=(0, 8), textcoords="offset points", fontsize=9, color=INK_3)
    ax.annotate("model: cut by the real chance of being right", (0.6, y_mod[np.searchsorted(ts, 0.6)]),
                xytext=(0, 8), textcoords="offset points", fontsize=9, color=INK_3)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0, max(y_rule.max(), y_mod.max()) * 1.1)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ["50%", "60%", "70%", "80%", "90%", "100%"])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"R$ {num(v)}"))
    ax.set_xlabel("cut-off: bill automatically above it, call a person below it")
    ax.spines["bottom"].set_color(LIGHT_GRAY)
    fig.tight_layout()
    title(fig, "What each cut-off costs",
          "Human checks (R$ 0.60) + wrong charges (R$ 30), per 10,000 passes. Simulated data.")
    save(fig, FIG / "06_cost_by_threshold.png")

    # calibration: raw camera confidence vs. the model's probability (passes without a tag)
    ocr = (te["source"] == "ocr").to_numpy()
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.grid(True, axis="both")
    ax.plot([0, 1], [0, 1], color=LIGHT_GRAY, lw=1.5)
    ax.annotate("perfect forecast", (0.42, 0.42), xytext=(6, -14), textcoords="offset points",
                fontsize=8.5, color=INK_3, rotation=45)
    edges = np.array([0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0001])
    for name, score, color in [("camera confidence", te["conf_best"].to_numpy()[ocr], AMBER),
                               ("model probability", p[ocr], GREEN)]:
        target = yte[ocr]
        group = np.digitize(score, edges) - 1
        mp, fr = [], []
        for g in range(len(edges) - 1):
            m_ = group == g
            if m_.sum() >= 40:                    # only bins with enough passes
                mp.append(score[m_].mean())
                fr.append(target[m_].mean())
        ax.plot(mp, fr, color=color, lw=2)
        ax.scatter(mp, fr, s=34, color=color, zorder=4, edgecolor="white", linewidth=1.5, label=name)
    ax.set_xlim(0.2, 1.01)
    ax.set_ylim(0.2, 1.01)
    pct = [0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_xticks(pct, [f"{int(v*100)}%" for v in pct])
    ax.set_yticks(pct, [f"{int(v*100)}%" for v in pct])
    ax.set_xlabel("what was promised")
    ax.set_ylabel("what happened")
    ax.legend(loc="upper left")
    fig.tight_layout()
    title(fig, "The camera promises more than it delivers",
          "Passes without a tag. The closer to the diagonal, the more honest the forecast.")
    save(fig, FIG / "07_calibration.png")

    sample = te.sample(min(15000, len(te)), random_state=1)
    imp = permutation_importance(model, sample[FEATURES], sample["correct"], scoring="roc_auc",
                                 n_repeats=4, random_state=0)
    s = pd.Series(imp.importances_mean, index=[NAMES[f] for f in FEATURES]).sort_values().tail(8)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    pos = np.arange(len(s))
    colors = [LIGHT_GRAY] * (len(s) - 1) + [GREEN]
    ax.barh(pos, s.values, color=colors, height=0.6)
    ax.set_yticks(pos, s.index)
    ax.tick_params(axis="y", labelsize=9.5, labelcolor=INK_2)
    grid_x(ax)
    ax.set_xticks([])
    fig.tight_layout()
    title(fig, "The clues that help the model most",
          "How much the forecast gets worse when a clue is shuffled. Longer bar, more useful clue.")
    save(fig, FIG / "08_feature_importance.png")

    # ---------------------------------------------------------------- storage
    # Policy: keep the full package only where the evidence may be needed
    #   (billing by plate, review, or a tag whose read the model does not confirm).
    #   Tag confirmed with P >= 0.999: keep the slim version.
    full = (~te["tag_read"].to_numpy()) | (p < 0.999)
    gb_now = len(te) * FULL_EVIDENCE / 1e9
    gb_new = (full.sum() * FULL_EVIDENCE + (~full).sum() * SLIM_EVIDENCE) / 1e9
    storage = {"passes_test_day": int(len(te)), "gb_day_keep_everything": round(gb_now, 2),
               "gb_day_risk_based_policy": round(gb_new, 2),
               "reduction_pct": round(100 * (1 - gb_new / gb_now), 1),
               "full_packages_pct": round(100 * full.mean(), 1)}
    fig, ax = plt.subplots(figsize=(8, 2.4))
    ax.barh([1, 0], [gb_now, gb_new], color=[LIGHT_GRAY, GREEN], height=0.45)
    ax.set_yticks([1, 0], ["Keep every photo", "Keep according to risk"])
    for yv, v in ((1, gb_now), (0, gb_new)):
        ax.text(v + gb_now * 0.015, yv, f"{num(v, 1)} GB per day", va="center", fontsize=9.5,
                color=INK_2, fontweight="medium")
    grid_x(ax)
    ax.set_xticks([])
    ax.set_xlim(0, gb_now * 1.3)
    ax.tick_params(axis="y", labelsize=10, labelcolor=INK_2)
    fig.tight_layout()
    title(fig, f"{num(storage['reduction_pct'])}% less storage by keeping full photos only when needed",
          "4 simulated gantries. The slim version is kept for tags confirmed by the plate.")
    save(fig, FIG / "12_storage_policy.png")
    print("\nStorage:", storage)

    res = {"auc": roc_auc_score(yte, p), "brier": brier_score_loss(yte, p),
           "best_threshold": float(t_best), "theoretical_threshold": t_theory,
           "review_cost": REVIEW_COST, "error_cost": ERROR_COST,
           "comparison": table.round(3).to_dict(orient="records"),
           "errors_avoided": kinds.to_dict(), "storage": storage}
    (TAB / "exp1_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
