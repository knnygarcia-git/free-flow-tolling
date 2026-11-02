"""
Bulk generator of synthetic gantry passes for the machine learning experiments.

Simulates several days of operation of 4 gantries (2 sites x 2 directions), in the data format a
Free Flow back office receives: several OCR reads per camera (front and rear) with a confidence,
the tag read (if any), the vehicle class measured by the laser, and a vehicle-registry lookup.

The generator knows the true plate of every vehicle, which makes it possible to measure accuracy.
In real life that label comes from human review and from disputes.

Simulated effects (all configurable in CONFIG):
  * per-character OCR errors, with realistic visual confusions (O/0/D/Q, B/8, I/1/L ...)
  * more errors at night, in rain, at high speed, and with dirty or damaged plates
  * OCR confidence that is informative but OVERCONFIDENT (like many real models)
  * tags installed on another vehicle (fraud) and registered class different from the measured one
  * misread plates that do not exist in the vehicle registry
  * sensor degradation on the last day: dirty lens (gradual) and burnt-out IR illuminator (sudden)

Output: data/synthetic/passes.parquet (or .csv.gz if pyarrow is missing)
"""
from __future__ import annotations

import string
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "synthetic"

LETTERS = string.ascii_uppercase
DIGITS = string.digits
ALPHABET = LETTERS + DIGITS                       # 36 symbols
IDX = {c: i for i, c in enumerate(ALPHABET)}
LETTER_POS = np.array([True, True, True, False, True, False, False])   # Mercosur LLLNLNN

# Groups of characters the OCR confuses (they look alike in a low-resolution IR image)
GROUPS = ["O0DQ", "B83", "I1LT", "S5", "Z2", "G6C", "A4", "EF", "MNH", "UV", "KX", "PR", "79"]

CONFIG = dict(
    days=3,
    seed=2026,
    gantries={"BR-376 km 196 N": 10_500, "BR-376 km 196 S": 10_000,
              "BR-369 km 126 E": 7_500, "BR-369 km 126 W": 7_000},   # vehicles/day
    p_tag=0.62,
    p_tag_other_vehicle=0.004,
    p_plate_dirty=0.08,
    p_plate_damaged=0.012,
    p_plate_covered=0.003,
    err_base=0.012,          # error per character, per frame, in ideal conditions
    # persistent error: a dirty/damaged plate is misread THE SAME WAY in several frames
    p_persistent={"ok": 0.002, "dirty": 0.05, "damaged": 0.14, "covered": 0.45},
    p_persistent_shows=0.75,
    night_factor=1.8,
    rain_factor=2.3,
    frames=(3, 6),           # reads per camera
    p_laser_error=0.004,
    p_plate_not_found=0.35,  # misread plate that does not exist in the registry
    # rain: (day, start_hour, end_hour)
    rain=[(0, 17.0, 19.5), (1, 6.0, 8.0), (2, 15.5, 18.5)],
    # degradations on the last day
    dirty_lens=dict(gantry="BR-376 km 196 N", lane=2, day=2, start=13.0, ramp_h=4.0, max_factor=7.0),
    burnt_ir=dict(gantry="BR-369 km 126 E", lane=1, day=2, start=20.5, factor=9.0),
)

KINDS = {  # kind: prob, ANTT class, mean speed km/h, dirt factor
    "car": (0.62, 1, 98, 1.0), "suv": (0.16, 1, 96, 1.0), "truck_2axle": (0.09, 2, 80, 1.6),
    "truck_5axle": (0.08, 7, 74, 2.0), "bus": (0.03, 4, 86, 1.2), "motorcycle": (0.02, 9, 95, 1.3),
}
PROFILE = np.array([1.4, 1.0, 0.9, 0.9, 1.2, 2.2, 4.0, 6.2, 6.6, 5.6, 5.3, 5.4,
                    5.5, 5.4, 5.5, 5.8, 6.4, 7.3, 7.6, 5.8, 4.3, 3.5, 2.7, 2.1])
PROFILE = PROFILE / PROFILE.sum()


def _confusion_table(rng) -> np.ndarray:
    """P(read | true) when there is an error: 85% inside the visual group, 15% anything else."""
    M = np.full((36, 36), 0.15 / 35)
    for g in GROUPS:
        for a in g:
            others = [b for b in g if b != a]
            for b in others:
                M[IDX[a], IDX[b]] += 0.85 / len(others)
    for c in ALPHABET:
        if not any(c in g for g in GROUPS):
            M[IDX[c]] = 1.0 / 35
        M[IDX[c], IDX[c]] = 0
        M[IDX[c]] /= M[IDX[c]].sum()
    return M


def _plates(rng, n) -> np.ndarray:
    """Mercosur plates as an (n, 7) matrix of alphabet indices."""
    p = np.empty((n, 7), np.int8)
    for j in range(7):
        if LETTER_POS[j]:
            p[:, j] = rng.integers(0, 26, n)
        else:
            p[:, j] = rng.integers(26, 36, n)
    return p


def to_text(m: np.ndarray) -> np.ndarray:
    lut = np.array(list(ALPHABET))
    return np.array(["".join(r) for r in lut[m]])


def generate(cfg: dict = CONFIG) -> pd.DataFrame:
    rng = np.random.default_rng(cfg["seed"])
    M = _confusion_table(rng)
    Mcum = M.cumsum(axis=1)

    blocks = []
    for name, aadt in cfg["gantries"].items():
        for day in range(cfg["days"]):
            n = rng.poisson(aadt)
            hour = rng.choice(24, n, p=PROFILE) + rng.random(n)
            blocks.append(pd.DataFrame({"gantry": name, "day": day, "hour": hour}))
    df = pd.concat(blocks, ignore_index=True)
    df = df.sort_values(["day", "hour"]).reset_index(drop=True)
    n = len(df)

    kinds = list(KINDS)
    probs = np.array([KINDS[k][0] for k in kinds])
    df["kind"] = rng.choice(kinds, n, p=probs / probs.sum())
    df["true_class"] = df["kind"].map({k: v[1] for k, v in KINDS.items()})
    heavy = df["kind"].isin(["truck_2axle", "truck_5axle", "bus"]).to_numpy()
    df["lane"] = np.where(heavy & (rng.random(n) < 0.8), 1, rng.integers(1, 3, n))
    vmean = df["kind"].map({k: v[2] for k, v in KINDS.items()}).to_numpy()
    df["speed_kmh"] = np.clip(rng.normal(vmean + np.where(df["lane"] == 2, 8, 0), 9), 30, 160)
    df["night"] = ((df["hour"] < 6) | (df["hour"] >= 18.5)).astype(int)
    rain = np.zeros(n, bool)
    for d, h0, h1 in cfg["rain"]:
        rain |= (df["day"].to_numpy() == d) & (df["hour"].between(h0, h1).to_numpy())
    df["rain"] = rain.astype(int)

    plate = _plates(rng, n)
    df["true_plate"] = to_text(plate)
    u = rng.random(n)
    dirt = df["kind"].map({k: v[3] for k, v in KINDS.items()}).to_numpy()
    state = np.where(u < cfg["p_plate_covered"], "covered",
            np.where(u < cfg["p_plate_covered"] + cfg["p_plate_damaged"], "damaged",
            np.where(rng.random(n) < cfg["p_plate_dirty"] * dirt, "dirty", "ok")))
    df["plate_state"] = state

    # --- error factor per event and per camera
    f_event = (cfg["err_base"]
               * np.where(df["night"], cfg["night_factor"], 1.0)
               * np.where(df["rain"], cfg["rain_factor"], 1.0)
               * (df["speed_kmh"].to_numpy() / 95) ** 2
               * pd.Series(state).map({"ok": 1, "dirty": 5, "damaged": 14, "covered": 60}).to_numpy())
    f_front = np.ones(n)
    f_rear = np.ones(n)
    dl, ir = cfg["dirty_lens"], cfg["burnt_ir"]
    m = ((df["gantry"] == dl["gantry"]) & (df["lane"] == dl["lane"]) & (df["day"] == dl["day"])
         & (df["hour"] >= dl["start"])).to_numpy()
    ramp = np.clip((df["hour"].to_numpy() - dl["start"]) / dl["ramp_h"], 0, 1)
    f_front = np.where(m, 1 + (dl["max_factor"] - 1) * ramp, f_front)
    m2 = ((df["gantry"] == ir["gantry"]) & (df["lane"] == ir["lane"]) & (df["day"] == ir["day"])
          & (df["hour"] >= ir["start"])).to_numpy()
    f_front = np.where(m2, ir["factor"], f_front)
    f_rear = np.where(m2, ir["factor"], f_rear)         # the illuminator serves both cameras
    df["_degraded"] = (m & (ramp > 0.15)) | m2
    moto = (df["kind"] == "motorcycle").to_numpy()      # motorcycles only have a rear plate

    # --- OCR reads per frame
    F = cfg["frames"][1]
    nq_front = np.where(moto, 0, rng.integers(cfg["frames"][0], F + 1, n))
    nq_rear = rng.integers(cfg["frames"][0], F + 1, n)
    reads = {}
    # persistent error: one swapped character that repeats in most frames
    p_pers = pd.Series(state).map(cfg["p_persistent"]).to_numpy()
    pers = rng.random((n, 7)) < p_pers[:, None]
    sub_pers = np.minimum((Mcum[plate.astype(int)] < rng.random((n, 7, 1))).sum(-1), 35)
    for cam, fcam, nq in (("front", f_front, nq_front), ("rear", f_rear, nq_rear)):
        quality = rng.lognormal(0, 0.35, (n, F))                 # frame-to-frame variation
        p_err = np.clip((f_event * fcam)[:, None] * quality, 0, 0.75)
        error = rng.random((n, F, 7)) < p_err[..., None]
        sub = (Mcum[plate[:, None, :].astype(int)] < rng.random((n, F, 7, 1))).sum(-1)
        sub = np.minimum(sub, 35)
        read = np.where(error, sub, plate[:, None, :])
        shows = pers[:, None, :] & (rng.random((n, F, 7)) < cfg["p_persistent_shows"])
        read = np.where(shows, sub_pers[:, None, :], read)
        n_err = (read != plate[:, None, :]).sum(-1)
        n_pers = shows.sum(-1)
        # confidence: informative, but overconfident and noisy. A persistent error barely lowers
        # the confidence: the image is sharp, it is the character itself that is worn or smudged.
        conf = (0.99 - 0.09 * (n_err - n_pers) - 0.025 * n_pers - 0.9 * p_err
                + rng.normal(0, 0.045, (n, F)))
        conf = np.clip(conf, 0.05, 0.999)
        valid = np.arange(F)[None, :] < nq[:, None]
        reads[cam] = (read, conf, valid)

    # serialize: "PLATE:conf|PLATE:conf|..."
    lut = np.array(list(ALPHABET))
    for cam, (read, conf, valid) in reads.items():
        txt = lut[read]
        col = []
        for i in range(n):
            col.append("|".join(f"{''.join(txt[i, k])}:{conf[i, k]:.3f}"
                                for k in range(F) if valid[i, k]))
        df[f"ocr_{cam}"] = col

    # --- RFID tag
    has_tag = rng.random(n) < cfg["p_tag"]
    df["has_tag"] = has_tag
    df["tag_read"] = has_tag & (rng.random(n) < 0.995)
    df["rssi_dbm"] = np.where(df["tag_read"], rng.normal(-58, 5, n), np.nan)
    other = df["tag_read"].to_numpy() & (rng.random(n) < cfg["p_tag_other_vehicle"])
    tag_plate = np.where(other, to_text(_plates(rng, n)), df["true_plate"])
    df["tag_plate"] = np.where(df["tag_read"], tag_plate, "")
    df["_tag_other_vehicle"] = other
    cat_tag_other = rng.choice([1, 1, 1, 2, 7], n)
    df["tag_class"] = np.where(df["tag_read"], np.where(other, cat_tag_other, df["true_class"]), 0)

    # --- laser
    laser_error = rng.random(n) < cfg["p_laser_error"]
    df["laser_class"] = np.where(laser_error, rng.choice([1, 2, 4, 7, 9], n), df["true_class"])

    df["ts"] = (pd.Timestamp("2026-06-01") + pd.to_timedelta(df["day"], unit="D")
                + pd.to_timedelta(df["hour"], unit="h"))
    df.insert(0, "pass_id", np.arange(n))
    return df


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    df = generate()
    try:
        df.to_parquet(OUT / "passes.parquet", index=False)
        dest = OUT / "passes.parquet"
    except ImportError:
        dest = OUT / "passes.csv.gz"
        df.to_csv(dest, index=False)
    print(f"{len(df):,} passes over {CONFIG['days']} days -> {dest}")
    print(df[["gantry", "kind", "has_tag", "plate_state", "night", "rain"]]
          .describe(include="all").T[["count", "unique", "top", "freq"]])
