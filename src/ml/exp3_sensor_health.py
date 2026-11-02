"""
Experiment 3: catch a degraded sensor before it turns into complaints.

Two failures happen on the last simulated day:
  * dirty lens on the front camera (BR-376 km 196 N, lane 2), getting worse from 1 pm on
  * burnt-out infrared illuminator (BR-369 km 126 E, lane 1) at 8:30 pm, at night

Indicator: share of passes in which the front camera reads the plate with confidence >= 0.90.
The catch is that this indicator also drops with rain and at night, so a fixed-threshold alarm
goes off every time it rains. Therefore:

  1. A model (gradient boosting) learns on days 1-2 the EXPECTED rate of confident reads given
     the conditions: hour, night, rain, speed, vehicle kind, gantry and lane.
  2. Every 10-minute window compares observed vs. expected (z-score).
  3. CUSUM accumulates persistent negative deviations and fires when it crosses a limit.

Compared with the naive alarm "rate < 85% in the window".
Outputs: figure 11; reports/tables/exp3_results.json
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.ensemble import HistGradientBoostingClassifier

from common import (ALERT, AMBER, BLUE, FIG, GRAY, INK_2, INK_3, LIGHT_GRAY, TAB, TEST_DAY,
                    TRAIN_DAYS, load, reads, save, title)

WINDOW_MIN = 10
NAIVE_THRESHOLD = 0.85
CUSUM_K, CUSUM_H = 0.5, 8.0
FAILURES = {("BR-376 km 196 N", 2): ("dirty lens (gradual)", 13.0),
            ("BR-369 km 126 E", 1): ("burnt-out IR illuminator", 20.5)}


def prepare():
    df = load()
    df = df[df["kind"] != "motorcycle"].copy()          # motorcycles have no front plate
    r = reads(df)
    df["ok"] = (r["conf_front"] >= 0.90).astype(int)
    df["window"] = (df["hour"] * 60 // WINDOW_MIN).astype(int)
    for c in ("gantry", "kind"):
        df[c + "_code"] = df[c].astype("category").cat.codes
    return df


FEATS = ["hour", "night", "rain", "speed_kmh", "kind_code", "gantry_code", "lane"]


def cusum(z):
    s, out = 0.0, []
    for v in z:
        s = max(0.0, s - v - CUSUM_K)
        out.append(s)
    return np.array(out)


def main():
    df = prepare()
    tr = df[df.day.isin(TRAIN_DAYS)]
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0,
                                           categorical_features=[4, 5])
    model.fit(tr[FEATS], tr["ok"])
    df["expected"] = model.predict_proba(df[FEATS])[:, 1]

    g = (df.groupby(["day", "gantry", "lane", "window"])
           .agg(n=("ok", "size"), obs=("ok", "sum"), exp=("expected", "sum"),
                var=("expected", lambda p: (p * (1 - p)).sum()), rain=("rain", "max"),
                night=("night", "max"))
           .reset_index())
    g = g[g["n"] >= 5]
    g["rate"] = g["obs"] / g["n"]
    g["rate_exp"] = g["exp"] / g["n"]
    g["z"] = (g["obs"] - g["exp"]) / np.sqrt(g["var"].clip(lower=1e-6))
    g["hour"] = g["window"] * WINDOW_MIN / 60

    rows, series = [], {}
    for (day, gantry, lane), s in g.groupby(["day", "gantry", "lane"]):
        s = s.sort_values("hour").copy()
        s["cusum"] = cusum(s["z"].to_numpy())
        s["alarm"] = s["cusum"] > CUSUM_H
        s["naive_alarm"] = s["rate"] < NAIVE_THRESHOLD
        series[(day, gantry, lane)] = s
        failure = FAILURES.get((gantry, lane)) if day == TEST_DAY else None
        start = failure[1] if failure else None

        def alarm_summary(col):
            al = s[s[col]]
            if failure:
                before = al[al["hour"] < start]
                after = al[al["hour"] >= start]
                delay = (after["hour"].iloc[0] - start) * 60 if len(after) else None
                return len(before), delay
            return len(al), None
        fa_m, delay_m = alarm_summary("alarm")
        fa_n, delay_n = alarm_summary("naive_alarm")
        rows.append({"day": day + 1, "gantry": gantry, "lane": lane,
                     "failure": failure[0] if failure else "",
                     "false_alarm_windows_model": fa_m, "detection_delay_model_min": delay_m,
                     "false_alarm_windows_naive": fa_n, "detection_delay_naive_min": delay_n})
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(tab.to_string(index=False))
    summary = {
        "false_alarms_model_total": int(tab["false_alarm_windows_model"].sum()),
        "false_alarms_naive_total": int(tab["false_alarm_windows_naive"].sum()),
        "detections": tab[tab.failure != ""][["gantry", "lane", "failure", "detection_delay_model_min",
                                              "detection_delay_naive_min"]].to_dict(orient="records"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # ---------------------------------------------------------------- figure
    failure_text = {"dirty lens (gradual)": "the lens starts getting dirty",
                    "burnt-out IR illuminator": "the infrared flash burns out"}
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True)
    for ax, ((gantry, lane), (name, start)) in zip(axes, FAILURES.items()):
        s = series[(TEST_DAY, gantry, lane)]
        for h0, h1 in _intervals(s, "night"):
            ax.axvspan(h0, h1, color="#f1f4f2", lw=0)
        for h0, h1 in _intervals(s, "rain"):
            ax.axvspan(h0, h1, color=BLUE, alpha=0.08, lw=0)
        ax.plot(s["hour"], s["rate_exp"], color=GRAY, lw=1.6)
        ax.plot(s["hour"], s["rate"], color=BLUE, lw=1.6)
        naive = s[s["naive_alarm"]]
        ax.scatter(naive["hour"], np.full(len(naive), 0.30), marker="|", color=AMBER, s=70, lw=1.6)
        ax.axvline(start, color=INK_3, lw=1)
        ax.annotate(failure_text[name], (start, 0.55), xytext=(-6, 0), textcoords="offset points",
                    ha="right", fontsize=8.5, color=INK_2)
        al = s[s["alarm"]]
        if len(al):
            h_al = al["hour"].iloc[0]
            ax.axvline(h_al, color=ALERT, lw=2)
            delay = (h_al - start) * 60
            when = "within the first 10 min" if delay < WINDOW_MIN else f"{delay:.0f} min later"
            ax.annotate(f"the model warns\n{when}", (h_al, 0.55), xytext=(6, 0),
                        textcoords="offset points", fontsize=8.5, color=ALERT, va="center")
        ax.set_ylim(0.22, 1.04)
        ax.set_yticks([0.4, 0.6, 0.8, 1.0], ["40%", "60%", "80%", "100%"])
        ax.text(0, 1.06, f"{gantry} · lane {lane}", transform=ax.transAxes, fontsize=9.5,
                fontweight="medium", color=INK_2)
    axes[1].set_xlim(0, 24)
    axes[1].set_xticks(range(0, 25, 3), [f"{h}:00" for h in range(0, 25, 3)])
    axes[1].spines["bottom"].set_color(LIGHT_GRAY)
    legend = [Line2D([], [], color=BLUE, lw=2, label="what the camera read"),
              Line2D([], [], color=GRAY, lw=2, label="what it should read"),
              Line2D([], [], color=AMBER, marker="|", lw=0, markersize=10, markeredgewidth=1.6,
                     label="simple alarm (below 85%)"),
              Patch(color=BLUE, alpha=0.15, label="rain"),
              Patch(color="#e6ebe8", label="night")]
    fig.tight_layout(h_pad=2.2)
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5,
               fontsize=8.5, handlelength=1.6, columnspacing=1.4)
    title(fig, "The alarm that doesn't panic when it rains",
          "Plates read with confidence, every 10 minutes, on the test day (simulated data).")
    save(fig, FIG / "11_sensor_health.png")

    tab.to_csv(TAB / "exp3_alarms.csv", index=False)
    (TAB / "exp3_results.json").write_text(json.dumps(
        {"window_min": WINDOW_MIN, "cusum_k": CUSUM_K, "cusum_h": CUSUM_H,
         "naive_threshold": NAIVE_THRESHOLD, **summary}, indent=2, ensure_ascii=False, default=float))


def _intervals(s, col):
    out, start = [], None
    for h, v in zip(s["hour"], s[col]):
        if v and start is None:
            start = h
        if not v and start is not None:
            out.append((start, h))
            start = None
    if start is not None:
        out.append((start, 24))
    return out


if __name__ == "__main__":
    main()
