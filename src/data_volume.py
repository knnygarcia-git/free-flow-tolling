"""
How much data does a Free Flow gantry produce per minute?

A parametric model in four layers, from the sensor to the back office:

  A. Raw sensor data  -> what the cameras see (processed on the gantry and thrown away)
  B. Audit video      -> continuous compressed recording (H.265), if the concessionaire keeps it
  C. Evidence per pass -> plate photos, plate crops, context photo, laser profile, metadata
  D. Transaction      -> the record sent for billing (plate/tag, vehicle class, time, gantry)

Traffic comes from one of two sources:
  1. Scenarios (default): AADT = average daily traffic per site, assumptions stated below.
  2. Real ANTT data: if the CSV "Volume de tráfego nas praças de pedágio" is placed in
     data/raw/antt/*.csv, the AADT of each EPR Paraná gantry is computed from it.
     Download: https://dados.antt.gov.br/dataset/volume-trafego-praca-pedagio

Outputs:
  reports/tables/data_volume_scenarios.csv
  reports/figures/04_data_per_minute_24h.png
  reports/figures/05_data_layers.png
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import AMBER, BLUE, GREEN, INK_2, LIGHT_GRAY, grid_x, num, save, title

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tables"
ANTT_DIR = ROOT / "data" / "raw" / "antt"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

KB, MB, GB, TB = 1e3, 1e6, 1e9, 1e12


# --------------------------------------------------------------------------------------
# Parameters (all editable). Typical market values for ANPR + RFID systems.
# --------------------------------------------------------------------------------------
@dataclass
class Gantry:
    """One gantry covering ONE direction of a dual carriageway."""
    lanes: int = 2                      # traffic lanes covered (shoulder not counted)
    anpr_cams_per_lane: int = 2         # front + rear (reads front and rear plates)
    context_cams: int = 1               # wide color camera on the gantry
    # ANPR camera: monochrome global-shutter sensor with infrared illumination
    anpr_mp: float = 5.0                # megapixels
    anpr_fps: int = 30
    anpr_bytes_px: float = 1.0          # 8-bit mono
    # Context camera: 4K color
    ctx_mp: float = 8.3
    ctx_fps: int = 25
    ctx_bytes_px: float = 1.5           # YUV 4:2:0
    # Audit video (H.265) per camera
    video_mbps: float = 4.0


@dataclass
class EvidencePackage:
    """Average size (bytes) of what is stored for EACH vehicle that passes."""
    items: dict = field(default_factory=lambda: {
        "2 ANPR photos (JPEG, front + rear)": 2 * 250 * KB,
        "2 plate crops": 2 * 15 * KB,
        "1 color context photo": 400 * KB,
        "3D laser profile (class / axles)": 30 * KB,
        "JSON metadata (OCR, confidence, speed, lane)": 3 * KB,
        "RFID tag read (TID/EPC, RSSI)": 0.3 * KB,
    })

    @property
    def total(self) -> float:
        return sum(self.items.values())


TRANSACTION_BYTES = 2 * KB   # record sent to the billing back office

# AADT scenarios per SITE (both directions added up). Assumptions, not measurements:
#   low     ~ regional road (PR-182 / PR-280)
#   medium  ~ BR-369 between Londrina and Cornélio Procópio
#   high    ~ BR-376 on the Maringá-Apucarana corridor, close to dual-carriageway capacity
AADT_SCENARIOS = {"low": 8_000, "medium": 20_000, "high": 40_000}

# Typical hourly profile of an intercity highway (share of AADT in each hour, sums to 1).
# Morning peak ~7-8am and evening peak ~5-6pm, trucks overnight.
_PROFILE = np.array([1.4, 1.0, 0.9, 0.9, 1.2, 2.2, 4.0, 6.2, 6.6, 5.6, 5.3, 5.4,
                     5.5, 5.4, 5.5, 5.8, 6.4, 7.3, 7.6, 5.8, 4.3, 3.5, 2.7, 2.1])
HOURLY_PROFILE = _PROFILE / _PROFILE.sum()


# --------------------------------------------------------------------------------------
def fixed_rates(g: Gantry) -> dict:
    """Layers that do not depend on traffic (bytes per second, per gantry)."""
    n_anpr = g.lanes * g.anpr_cams_per_lane
    raw = (n_anpr * g.anpr_mp * 1e6 * g.anpr_bytes_px * g.anpr_fps
           + g.context_cams * g.ctx_mp * 1e6 * g.ctx_bytes_px * g.ctx_fps)
    video = (n_anpr + g.context_cams) * g.video_mbps * 1e6 / 8
    return {"A_raw_sensor_Bps": raw, "B_audit_video_Bps": video,
            "n_cameras": n_anpr + g.context_cams}


def data_per_minute(veh_min: float, ev: EvidencePackage) -> dict:
    return {"C_evidence_B_min": veh_min * ev.total,
            "D_transaction_B_min": veh_min * TRANSACTION_BYTES}


def aadt_from_antt() -> pd.DataFrame | None:
    """Reads the official ANTT CSV (if present) and returns the AADT per EPR toll site."""
    files = sorted(ANTT_DIR.glob("*.csv"))
    if not files:
        return None
    frames = []
    for f in files:
        df = pd.read_csv(f, sep=";", encoding="latin-1", dtype=str)
        df.columns = [c.strip().lower() for c in df.columns]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["volume"] = pd.to_numeric(
        df["volume_total"].str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
        errors="coerce")
    df["date"] = pd.to_datetime(df["mes_ano"], dayfirst=True, errors="coerce")
    epr = df[df["concessionaria"].str.contains("EPR", case=False, na=False)]
    if epr.empty:
        print("ANTT CSV found, but it has no EPR rows. Using the scenarios.")
        return None
    daily = epr.groupby(["concessionaria", "praca", "date"]).volume.sum().reset_index()
    return (daily.groupby(["concessionaria", "praca"]).volume.mean()
                 .rename("aadt").reset_index().sort_values("aadt", ascending=False))


def scenario_table(g: Gantry, ev: EvidencePackage, n_sites: int = 4) -> pd.DataFrame:
    fixed = fixed_rates(g)
    rows = []
    for name, aadt in AADT_SCENARIOS.items():
        peak_h = aadt * HOURLY_PROFILE.max()            # vehicles in the peak hour (site)
        veh_min_peak = peak_h / 60
        veh_min_avg = aadt / 1440
        d_peak = data_per_minute(veh_min_peak, ev)
        d_day = data_per_minute(aadt / 1440, ev)
        gantries_per_site = 2                            # one gantry per direction
        rows.append({
            "scenario": name,
            "aadt_site": aadt,
            "veh_peak_hour": round(peak_h),
            "veh_min_peak": round(veh_min_peak, 1),
            "veh_sec_peak": round(veh_min_peak / 60, 2),
            "veh_min_avg": round(veh_min_avg, 1),
            "C_evidence_MB_min_peak": d_peak["C_evidence_B_min"] / MB,
            "D_transaction_KB_min_peak": d_peak["D_transaction_B_min"] / KB,
            "C_evidence_GB_day_site": d_day["C_evidence_B_min"] * 1440 / GB,
            "B_video_GB_day_site": fixed["B_audit_video_Bps"] * 86400 * gantries_per_site / GB,
            "A_raw_GB_min_site": fixed["A_raw_sensor_Bps"] * 60 * gantries_per_site / GB,
            "C_evidence_TB_year_EPR_lot4":
                d_day["C_evidence_B_min"] * 1440 * 365 * n_sites / TB,
        })
    return pd.DataFrame(rows)


def fig_24h(ev: EvidencePackage) -> None:
    minutes = np.arange(1440)
    frac_min = np.repeat(HOURLY_PROFILE, 60) / 60
    frac_min = np.convolve(np.r_[frac_min[-60:], frac_min, frac_min[:60]], np.ones(61) / 61,
                           mode="same")[60:-60]
    frac_min /= frac_min.sum()
    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    colors = {"low": BLUE, "medium": GREEN, "high": AMBER}
    labels = {"low": "8k vehicles/day", "medium": "20k", "high": "40k"}
    for name, aadt in AADT_SCENARIOS.items():
        mb_min = aadt * frac_min * ev.total / MB
        ax.plot(minutes / 60, mb_min, color=colors[name], lw=2)
        if name == "high":
            ax.fill_between(minutes / 60, mb_min, color=colors[name], alpha=0.08, lw=0)
        ax.annotate(labels[name], (24, mb_min[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK_2)
    peak = AADT_SCENARIOS["high"] * frac_min.max() * ev.total / MB
    h_peak = minutes[np.argmax(frac_min)] / 60
    ax.scatter([h_peak], [peak], s=36, color=AMBER, zorder=4, edgecolor="white", linewidth=2)
    ax.annotate(f"peak: {num(peak)} MB per minute", (h_peak, peak), xytext=(-8, 6),
                textcoords="offset points", ha="right", fontsize=9, color=INK_2)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, peak * 1.18)
    ax.set_xticks(range(0, 25, 3), [f"{h}:00" for h in range(0, 25, 3)])
    ax.set_ylabel("MB per minute")
    ax.spines["bottom"].set_color(LIGHT_GRAY)
    fig.tight_layout()
    title(fig, "Data produced over the day at one site on the highway",
          "Photos and records for both directions, at three traffic levels (assumptions).")
    save(fig, FIG / "04_data_per_minute_24h.png")


def fig_layers(g: Gantry, ev: EvidencePackage) -> None:
    """Log scale: from raw sensor data to the billing record, at the medium-scenario peak."""
    fixed = fixed_rates(g)
    veh_min = AADT_SCENARIOS["medium"] * HOURLY_PROFILE.max() / 60 / 2   # one direction
    layers = {
        "What the cameras see": fixed["A_raw_sensor_Bps"] * 60,
        "Recorded video (if the company keeps it)": fixed["B_audit_video_Bps"] * 60,
        "Photos and evidence for each car": veh_min * ev.total,
        "Records sent for billing": veh_min * TRANSACTION_BYTES,
    }
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    names, vals = list(layers), [v / MB for v in layers.values()]
    colors = [LIGHT_GRAY, LIGHT_GRAY, GREEN, AMBER]
    pos = np.arange(len(names))
    ax.barh(pos, vals, color=colors, height=0.55)
    ax.set_yticks(pos, names)
    ax.set_xscale("log")
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        txt = (f"{num(v / 1000, 1)} GB" if v >= 1000 else
               (f"{num(v, 1)} MB" if v >= 1 else f"{num(v * 1000)} KB"))
        ax.text(v * 1.25, i, txt, va="center", fontsize=9.5, color=INK_2, fontweight="medium")
    grid_x(ax)
    ax.set_xlim(0.01, max(vals) * 40)
    ax.set_xticks([0.01, 1, 100, 1e4], ["10 KB", "1 MB", "100 MB", "10 GB"])
    ax.tick_params(axis="y", labelsize=9.5, labelcolor=INK_2)
    fig.tight_layout()
    title(fig, "One peak minute at one gantry",
          f"About {veh_min:.0f} cars per minute in one direction. Log scale: each grid line is 100 times the previous one.")
    save(fig, FIG / "05_data_layers.png")


if __name__ == "__main__":
    g, ev = Gantry(), EvidencePackage()
    fixed = fixed_rates(g)

    print("=" * 70)
    print("EVIDENCE PACKAGE PER PASS")
    for k, v in ev.items.items():
        print(f"  {k:<52s} {v/KB:8.1f} KB")
    print(f"  {'TOTAL':<52s} {ev.total/KB:8.1f} KB")
    print(f"\nPer gantry (1 direction, {g.lanes} lanes, {fixed['n_cameras']} cameras):")
    print(f"  A. raw sensor data:   {fixed['A_raw_sensor_Bps']/GB:6.2f} GB/s "
          f"(= {fixed['A_raw_sensor_Bps']*60/GB:,.0f} GB/min, processed on the gantry)")
    print(f"  B. audit video:       {fixed['B_audit_video_Bps']*60/MB:6.1f} MB/min")

    antt = aadt_from_antt()
    if antt is not None:
        print("\nReal AADT (ANTT) per EPR toll site:")
        print(antt.to_string(index=False))

    tab = scenario_table(g, ev)
    tab.to_csv(TAB / "data_volume_scenarios.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nSCENARIOS (per site = 2 gantries, one per direction)")
    print(tab.round(2).T.to_string())

    fig_24h(ev)
    fig_layers(g, ev)
    print(f"\nFigures in {FIG}")
