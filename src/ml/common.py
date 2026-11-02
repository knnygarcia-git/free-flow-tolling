"""Shared helpers for the ML experiments: loading, OCR parsing and model features."""
from __future__ import annotations

import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "synthetic"
FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

MERCOSUR = re.compile(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$")
TRAIN_DAYS = [0, 1]          # time-based validation: train on days 1-2, test on day 3
TEST_DAY = 2

sys.path.insert(0, str(ROOT / "src"))
from style import (ALERT, AMBER, BLUE, GRAY, GREEN, INK, INK_2, INK_3, LIGHT_GRAY,  # noqa: E402,F401
                   grid_x, num, save, title)


def load() -> pd.DataFrame:
    p = DATA / "passes.parquet"
    if not p.exists():
        p = DATA / "passes.csv.gz"
    if not p.exists():
        raise SystemExit("Run this first: python src/ml/generate_events.py")
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    for c in ("ocr_front", "ocr_rear", "tag_plate"):
        df[c] = df[c].fillna("")
    return df


def parse(s: str) -> list[tuple[str, float]]:
    if not s:
        return []
    out = []
    for item in s.split("|"):
        plate, conf = item.split(":")
        out.append((plate, float(conf)))
    return out


def hamming(a: str, b: str) -> int:
    if not a or not b:
        return 7
    return sum(x != y for x, y in zip(a, b))


def registry(plate: str, true_plate: str, true_class: int, p_not_found: float = 0.35):
    """Simulates a lookup in the national vehicle registry.
    The true plate exists with its true class. A misread plate may not exist, or may exist and
    belong to another vehicle, of any class. Deterministic per plate."""
    if plate == true_plate:
        return True, true_class
    h = zlib.crc32(plate.encode())
    if (h % 1000) / 1000 < p_not_found:
        return False, 0
    return True, [1, 1, 1, 1, 2, 4, 7, 9][(h // 1000) % 8]


def reads(df: pd.DataFrame) -> pd.DataFrame:
    """Summarizes the OCR reads of each pass (best frame, agreement, etc.)."""
    rows = []
    for fr, rr in zip(df["ocr_front"], df["ocr_rear"]):
        f, r = parse(fr), parse(rr)
        allr = f + r
        best = max(allr, key=lambda x: x[1]) if allr else ("", 0.0)
        bf = max(f, key=lambda x: x[1]) if f else ("", 0.0)
        br_ = max(r, key=lambda x: x[1]) if r else ("", 0.0)
        agree = np.mean([p == best[0] for p, _ in allr]) if allr else 0.0
        rows.append((best[0], best[1], bf[1], br_[1], int(bf[0] == br_[0] and bf[0] != ""),
                     agree, len(allr), len(f) > 0))
    return pd.DataFrame(rows, index=df.index, columns=[
        "ocr_best", "conf_best", "conf_front", "conf_rear", "front_rear_match",
        "frac_frames_agree", "n_frames", "has_front"])


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Billing candidate (tag if read, otherwise best OCR) + model features."""
    r = reads(df)
    out = pd.concat([df, r], axis=1)
    out["candidate"] = np.where(out["tag_read"], out["tag_plate"], out["ocr_best"])
    out["source"] = np.where(out["tag_read"], "tag", "ocr")
    out["correct"] = (out["candidate"] == out["true_plate"]).astype(int)

    out["tag_ocr_distance"] = [hamming(a, b) if t else -1
                               for a, b, t in zip(out["tag_plate"], out["ocr_best"], out["tag_read"])]
    reg = [registry(c, tp, tc) for c, tp, tc in
           zip(out["candidate"], out["true_plate"], out["true_class"])]
    out["plate_exists"] = [int(e) for e, _ in reg]
    reg_class = np.array([c for _, c in reg])
    out["registry_class_eq_laser"] = (reg_class == out["laser_class"].to_numpy()).astype(int)
    out["tag_class_eq_laser"] = np.where(out["tag_read"],
                                         (out["tag_class"] == out["laser_class"]).astype(int), -1)
    out["valid_format"] = out["candidate"].str.match(MERCOSUR).astype(int)
    out["rssi_dbm"] = out["rssi_dbm"].fillna(-100)
    out["motorcycle"] = (out["laser_class"] == 9).astype(int)
    out["tag_read_i"] = out["tag_read"].astype(int)
    return out


FEATURES = ["tag_read_i", "rssi_dbm", "conf_best", "conf_front", "conf_rear",
            "front_rear_match", "frac_frames_agree", "n_frames", "tag_ocr_distance",
            "plate_exists", "registry_class_eq_laser", "tag_class_eq_laser", "valid_format",
            "speed_kmh", "night", "rain", "lane", "motorcycle"]

NAMES = {
    "tag_read_i": "tag read", "rssi_dbm": "tag signal strength", "conf_best": "OCR confidence",
    "conf_front": "front camera confidence", "conf_rear": "rear camera confidence",
    "front_rear_match": "front = rear", "frac_frames_agree": "frames that agree",
    "n_frames": "number of frames", "tag_ocr_distance": "tag vs. OCR difference",
    "plate_exists": "plate exists in registry", "registry_class_eq_laser": "registry class = laser",
    "tag_class_eq_laser": "tag class = laser", "valid_format": "valid Mercosur format",
    "speed_kmh": "speed", "night": "night", "rain": "rain", "lane": "lane", "motorcycle": "motorcycle",
}
