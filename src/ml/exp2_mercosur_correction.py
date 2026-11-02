"""
Experiment 2: fix the plate read using the Mercosur format and every frame.

The camera photographs each vehicle several times (front and rear). The OCR returns one read per
frame, with a confidence. Four ways to arrive at the final plate:

  A. Best frame               -> the read with the highest confidence (what many systems do)
  B. A + format rule          -> Mercosur is LLLNLNN: a digit in a letter position becomes the
                                 look-alike letter (0->O, 8->B, 1->I ...) and vice versa
  C. Per-character vote       -> for each position, the character with the highest summed
                                 confidence across all frames wins; then the format rule
  D. Probabilistic model      -> learns the confusion matrix P(read | true) on training data and
                                 picks, per position, the most likely true character given ALL
                                 frames, restricted to the position type (letter or digit)

Only passes without a tag (the ones that depend on OCR). Train on days 1-2, test on day 3.
Outputs: figures 09 and 10; reports/tables/exp2_results.json
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (BLUE, FIG, GREEN, INK_2, INK_3, LIGHT_GRAY, TAB, TEST_DAY, TRAIN_DAYS, grid_x,
                    load, num, parse, save, title)

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
ALPHA = LETTERS + DIGITS
IDX = {c: i for i, c in enumerate(ALPHA)}
LETTER_POS = [True, True, True, False, True, False, False]
DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "3": "B", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B", "9": "P"}
LETTER_TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "T": "7", "Z": "2", "B": "8",
                   "S": "5", "G": "6", "A": "4", "E": "3"}


def fix_format(plate: str) -> str:
    out = []
    for c, is_letter in zip(plate, LETTER_POS):
        if is_letter and c.isdigit():
            c = DIGIT_TO_LETTER.get(c, c)
        elif not is_letter and c.isalpha():
            c = LETTER_TO_DIGIT.get(c, c)
        out.append(c)
    return "".join(out)


def frames(row) -> list[tuple[str, float]]:
    return parse(row.ocr_front) + parse(row.ocr_rear)


def best_frame(qs):
    return max(qs, key=lambda x: x[1])[0]


def vote(qs):
    plate = []
    for j in range(7):
        score = {}
        for p, c in qs:
            score[p[j]] = score.get(p[j], 0) + c
        # respect the position type before voting
        valid = {k: v for k, v in score.items() if k.isalpha() == LETTER_POS[j]}
        choice = max(valid or score, key=(valid or score).get)
        plate.append(choice)
    return fix_format("".join(plate))


def learn_confusion(df_train: pd.DataFrame, smooth: float = 0.5) -> np.ndarray:
    """log P(read | true), estimated by counting with Laplace smoothing."""
    C = np.full((36, 36), smooth)
    for row in df_train.itertuples():
        true = row.true_plate
        for p, _ in frames(row):
            for a, b in zip(true, p):
                C[IDX[a], IDX[b]] += 1
    return np.log(C / C.sum(axis=1, keepdims=True))


def decode(qs, logP: np.ndarray, conf_weight: float = 2.0) -> str:
    plate = []
    for j in range(7):
        cands = LETTERS if LETTER_POS[j] else DIGITS
        idx_c = [IDX[c] for c in cands]
        score = np.zeros(len(cands))
        for p, conf in qs:
            w = 1.0 + conf_weight * (conf - 0.5)        # more confident frames weigh more
            score += w * logP[idx_c, IDX[p[j]]]
        plate.append(cands[int(np.argmax(score))])
    return "".join(plate)


def main():
    df = load()
    df = df[~df["tag_read"]].copy()                     # only the ones that depend on OCR
    tr, te = df[df.day.isin(TRAIN_DAYS)], df[df.day == TEST_DAY]
    print(f"passes without a tag: train {len(tr):,} | test {len(te):,}")

    logP = learn_confusion(tr)
    res = {"A. best frame": [], "B. + Mercosur rule": [], "C. vote + rule": [],
           "D. confusion model": []}
    for row in te.itertuples():
        qs = frames(row)
        a = best_frame(qs)
        res["A. best frame"].append(a)
        res["B. + Mercosur rule"].append(fix_format(a))
        res["C. vote + rule"].append(vote(qs))
        res["D. confusion model"].append(decode(qs, logP))

    true = te["true_plate"].to_numpy()
    rows = []
    for name, pred in res.items():
        pred = np.array(pred)
        hit = (pred == true)
        char = np.mean([sum(x == y for x, y in zip(p, r)) / 7 for p, r in zip(pred, true)])
        rows.append({"method": name, "plate_accuracy_pct": 100 * hit.mean(),
                     "errors_per_10k": 1e4 * (1 - hit.mean()), "char_accuracy_pct": 100 * char,
                     "accuracy_night_rain_pct": 100 * hit[(te.night | te.rain).to_numpy() == 1].mean(),
                     "accuracy_dirty_plate_pct": 100 * hit[(te.plate_state == "dirty").to_numpy()].mean()})
    tab = pd.DataFrame(rows)
    base_err = tab.loc[0, "errors_per_10k"]
    tab["error_reduction_pct"] = 100 * (1 - tab["errors_per_10k"] / base_err)
    pd.set_option("display.width", 220)
    print(tab.round(2).to_string(index=False))

    # where are the errors left over by the best method?
    best = tab.loc[tab["errors_per_10k"].idxmin(), "method"]
    left = np.array(res[best]) != true
    state = te["plate_state"].to_numpy()
    remaining = pd.Series(state[left]).value_counts(normalize=True).mul(100).round(1)
    print(f"\nErrors left in '{best}', by plate condition (%):\n" + remaining.to_string())

    # ---------------------------------------------------------------- figures
    names = {"A. best frame": "Best photo only",
             "B. + Mercosur rule": "Best photo + format rule",
             "C. vote + rule": "Vote across photos + rule",
             "D. confusion model": "Model of confused letters"}
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    pos = np.arange(len(tab))
    winner = int(tab["errors_per_10k"].idxmin())
    colors = [GREEN if i == winner else LIGHT_GRAY for i in range(len(tab))]
    ax.barh(pos, tab["errors_per_10k"], color=colors, height=0.55)
    ax.set_yticks(pos, [names[m] for m in tab["method"]])
    for i, (e, r) in enumerate(zip(tab["errors_per_10k"], tab["error_reduction_pct"])):
        txt = num(e) + ("" if i == 0 else f"   −{num(r)}%")
        ax.text(e + base_err * 0.015, i, txt, va="center", fontsize=9.5, color=INK_2,
                fontweight="medium" if i == winner else "regular")
    ax.invert_yaxis()
    grid_x(ax)
    ax.set_xlim(0, base_err * 1.3)
    ax.tick_params(axis="y", labelsize=9.5, labelcolor=INK_2)
    ax.set_xlabel("misread plates per 10,000 passes without a tag")
    fig.tight_layout()
    title(fig, "Voting across photos fixes 30% of the errors",
          "Passes without a tag on the test day (simulated data). What is left is dirty or bent plates.")
    save(fig, FIG / "09_mercosur_correction.png")

    # most frequent swaps in training
    count = []
    for row in tr.itertuples():
        for p, _ in frames(row):
            for a, b in zip(row.true_plate, p):
                if a != b:
                    count.append(f"{a} read as {b}")
    top = pd.Series(count).value_counts().head(10)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    pos = np.arange(len(top))
    ax.barh(pos, top.values, color=BLUE, height=0.6)
    ax.set_yticks(pos, top.index)
    for i, v in enumerate(top.values):
        ax.text(v + top.max() * 0.015, i, num(v), va="center", fontsize=9, color=INK_3)
    grid_x(ax)
    ax.set_xticks([])
    ax.set_xlim(0, top.max() * 1.15)
    ax.tick_params(axis="y", labelsize=9.5, labelcolor=INK_2)
    fig.tight_layout()
    title(fig, "The characters the camera confuses most",
          "Times each swap showed up in the training photos (simulated data).")
    save(fig, FIG / "10_ocr_confusions.png")

    (TAB / "exp2_results.json").write_text(json.dumps(
        {"test_passes": int(len(te)), "methods": tab.round(3).to_dict(orient="records"),
         "top_confusions": top[::-1].to_dict(), "best_method": best,
         "remaining_errors_by_plate_state_pct": remaining.to_dict()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
