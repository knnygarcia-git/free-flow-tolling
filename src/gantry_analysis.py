"""
Exploratory analysis of the registry of Free Flow (barrier-free toll) gantries in Brazil, with a
focus on Paraná (EPR).

Source: the open "SEM-PARAR-FREE-FLOW" dataset (CC BY 4.0), which compiles the registry of
ANTT (the federal land-transport regulator) and the toll concessionaires. The files are in
data/raw/sem_parar/ and keep their original Portuguese column names.

Outputs:
  reports/figures/01_gantries_by_state.png
  reports/figures/02_gantries_timeline.png
  reports/figures/03_parana_gantries.png
  reports/tables/parana_gantries.csv
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from style import BLUE, GREEN, INK_2, INK_3, LIGHT_GRAY, grid_x, num, save, title

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "sem_parar"
FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# Original (Portuguese) column names -> English names used in the outputs
COLUMNS = {"concessionaria": "concessionaire", "rodovia": "highway_name", "sigla": "highway",
           "municipio": "municipality", "km": "km", "sentido": "direction",
           "porticos_no_registro": "gantries", "data_ativacao": "start_date", "status": "status",
           "uf": "state"}


def load() -> pd.DataFrame:
    df = pd.read_csv(RAW / "porticos-free-flow.csv").rename(columns=COLUMNS)
    df["km"] = pd.to_numeric(df["km"].astype(str).str.replace(",", "."), errors="coerce")
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["gantries"] = df["gantries"].fillna(1).astype(int)
    df["epr"] = df["concessionaire"].str.contains("EPR", na=False)
    return df


def summary(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("FREE FLOW GANTRY REGISTRY - BRAZIL")
    print("=" * 70)
    print(f"Records (sites):             {len(df)}")
    print(f"Physical gantries (sum):     {df.gantries.sum()}")
    print(f"States:                      {df.state.nunique()}")
    print(f"Concessionaires:             {df.concessionaire.nunique()}")
    print("\nStatus:\n" + df.status.value_counts().to_string())
    pr = df[df.state == "PR"]
    print(f"\nParaná: {len(pr)} sites / {pr.gantries.sum()} gantries")
    print(pr.groupby("concessionaire").gantries.sum().to_string())


def fig_by_state(df: pd.DataFrame) -> None:
    s = df.groupby("state").gantries.sum().sort_values()
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    colors = [GREEN if st == "PR" else LIGHT_GRAY for st in s.index]
    ax.barh(s.index, s.values, color=colors, height=0.6)
    for i, (st, v) in enumerate(s.items()):
        ax.text(v + 0.6, i, str(v), va="center", fontsize=12,
                color=INK_2 if st != "PR" else GREEN, fontweight="medium" if st == "PR" else "regular")
    grid_x(ax)
    ax.set_xlim(0, s.max() * 1.12)
    ax.tick_params(axis="y", labelsize=12.5)
    fig.tight_layout()
    title(fig, "Barrier-free toll gantries by state",
          f"{s.sum()} gantries in Brazil. Paraná, in green, has {s['PR']}.")
    save(fig, FIG / "01_gantries_by_state.png")


def fig_timeline(df: pd.DataFrame) -> None:
    d = (df.dropna(subset=["start_date"])
           .groupby("start_date").gantries.sum()
           .sort_index().cumsum())
    pr = (df[(df.state == "PR")].dropna(subset=["start_date"])
            .groupby("start_date").gantries.sum().sort_index().cumsum())
    end = d.index.max()
    d.loc[end + pd.Timedelta(days=20)] = d.iloc[-1]
    pr.loc[end + pd.Timedelta(days=20)] = pr.iloc[-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.step(d.index, d.values, where="post", color=BLUE, lw=2)
    ax.step(pr.index, pr.values, where="post", color=GREEN, lw=2)
    ax.fill_between(d.index, d.values, step="post", color=BLUE, alpha=0.07, lw=0)
    for series, name, color in ((d, "Brazil", BLUE), (pr, "Paraná", GREEN)):
        ax.scatter([series.index[-1]], [series.iloc[-1]], s=36, color=color, zorder=4,
                   edgecolor="white", linewidth=2)
        ax.annotate(f"{name}: {int(series.iloc[-1])}", (series.index[-1], series.iloc[-1]),
                    xytext=(8, 0), textcoords="offset points", va="center", fontsize=12,
                    color=INK_2)
    ax.set_ylim(0, d.max() * 1.15)
    ax.set_xlim(d.index.min(), end + pd.Timedelta(days=120))
    ax.spines["bottom"].set_color(LIGHT_GRAY)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.tight_layout()
    title(fig, "How many gantries were charging, month by month",
          "Running total since the first one, on BR-101 (Rio de Janeiro), in March 2023.")
    save(fig, FIG / "02_gantries_timeline.png")


def fig_parana(df: pd.DataFrame) -> pd.DataFrame:
    pr = df[df.state == "PR"].copy()
    pr["site"] = (pr["highway"] + " · " + pr["municipality"].str.replace("Presidente", "Pres.")
                  + " · km " + pr["km"].map(lambda k: num(k, 1)))
    pr = pr.sort_values(["epr", "concessionaire", "highway", "km"], ascending=[False, True, True, True])
    fig, ax = plt.subplots(figsize=(9, 6.0))
    colors = [GREEN if e else LIGHT_GRAY for e in pr.epr]
    pos = range(len(pr))
    ax.barh(pos, pr["gantries"], color=colors, height=0.55)
    ax.set_yticks(list(pos), pr["site"])
    for i, (n, company) in enumerate(zip(pr["gantries"], pr["concessionaire"])):
        company = company.split(" (")[0]
        ax.text(n + 0.05, i, f"{n}  ·  {company}", va="center", fontsize=11.5, color=INK_3)
    ax.invert_yaxis()
    grid_x(ax)
    ax.set_xticks([0, 1, 2])
    ax.set_xlim(0, 3.4)
    ax.set_xlabel("gantries at the site")
    fig.tight_layout()
    title(fig, "The gantries in Paraná",
          "EPR's in green. EPR Paraná has two at each site, one per direction.")
    save(fig, FIG / "03_parana_gantries.png")
    cols = ["concessionaire", "highway_name", "highway", "municipality", "km", "direction",
            "gantries", "start_date", "status"]
    pr[cols].to_csv(TAB / "parana_gantries.csv", index=False)
    return pr[cols]


if __name__ == "__main__":
    df = load()
    summary(df)
    fig_by_state(df)
    fig_timeline(df)
    table = fig_parana(df)
    print("\nGantries in Paraná:\n" + table.to_string(index=False))
    print(f"\nFigures in {FIG}")
