"""
Análise exploratória do cadastro de pórticos Free Flow no Brasil, com foco no Paraná (EPR).

Fonte: base aberta "SEM-PARAR-FREE-FLOW" (CC BY 4.0), que consolida o cadastro da ANTT
e das concessionárias. Os arquivos ficam em data/raw/sem_parar/.

Saídas:
  reports/figures/01_porticos_por_uf.png
  reports/figures/02_linha_do_tempo_ativacoes.png
  reports/figures/03_porticos_parana.png
  reports/tabelas/porticos_parana.csv
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "sem_parar"
FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tabelas"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

from estilo import (AZUL, CINZA, CINZA_CLARO, TINTA_2, TINTA_3, VERDE, br, grade_x, salvar,
                    titulo)


def carregar() -> pd.DataFrame:
    df = pd.read_csv(RAW / "porticos-free-flow.csv")
    df["km"] = pd.to_numeric(df["km"].astype(str).str.replace(",", "."), errors="coerce")
    df["data_ativacao"] = pd.to_datetime(df["data_ativacao"], errors="coerce")
    df["porticos_no_registro"] = df["porticos_no_registro"].fillna(1).astype(int)
    df["epr"] = df["concessionaria"].str.contains("EPR", na=False)
    return df


def resumo(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("CADASTRO DE PÓRTICOS FREE FLOW - BRASIL")
    print("=" * 70)
    print(f"Registros (locais):          {len(df)}")
    print(f"Pórticos físicos (soma):     {df.porticos_no_registro.sum()}")
    print(f"UFs:                         {df.uf.nunique()}")
    print(f"Concessionárias:             {df.concessionaria.nunique()}")
    print("\nStatus:\n" + df.status.value_counts().to_string())
    pr = df[df.uf == "PR"]
    print(f"\nParaná: {len(pr)} locais / {pr.porticos_no_registro.sum()} pórticos")
    print(pr.groupby("concessionaria").porticos_no_registro.sum().to_string())


def fig_por_uf(df: pd.DataFrame) -> None:
    s = df.groupby("uf").porticos_no_registro.sum().sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    cores = [VERDE if uf == "PR" else CINZA_CLARO for uf in s.index]
    ax.barh(s.index, s.values, color=cores, height=0.6)
    for i, (uf, v) in enumerate(s.items()):
        ax.text(v + 0.6, i, str(v), va="center", fontsize=9,
                color=TINTA_2 if uf != "PR" else VERDE, fontweight="medium" if uf == "PR" else "regular")
    grade_x(ax)
    ax.set_xlim(0, s.max() * 1.12)
    ax.tick_params(axis="y", labelsize=10)
    fig.tight_layout()
    titulo(fig, "Pórticos sem cancela por estado",
           f"{s.sum()} pórticos no Brasil. O Paraná, em verde, tem {s['PR']}.")
    salvar(fig, FIG / "01_porticos_por_uf.png")


def fig_linha_do_tempo(df: pd.DataFrame) -> None:
    d = (df.dropna(subset=["data_ativacao"])
           .groupby("data_ativacao").porticos_no_registro.sum()
           .sort_index().cumsum())
    pr = (df[(df.uf == "PR")].dropna(subset=["data_ativacao"])
            .groupby("data_ativacao").porticos_no_registro.sum().sort_index().cumsum())
    fim = d.index.max()
    d.loc[fim + pd.Timedelta(days=20)] = d.iloc[-1]
    pr.loc[fim + pd.Timedelta(days=20)] = pr.iloc[-1]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.step(d.index, d.values, where="post", color=AZUL, lw=2)
    ax.step(pr.index, pr.values, where="post", color=VERDE, lw=2)
    ax.fill_between(d.index, d.values, step="post", color=AZUL, alpha=0.07, lw=0)
    for serie, nome, cor in ((d, "Brasil", AZUL), (pr, "Paraná", VERDE)):
        ax.scatter([serie.index[-1]], [serie.iloc[-1]], s=36, color=cor, zorder=4,
                   edgecolor="white", linewidth=2)
        ax.annotate(f"{nome}: {int(serie.iloc[-1])}", (serie.index[-1], serie.iloc[-1]),
                    xytext=(8, 0), textcoords="offset points", va="center", fontsize=9,
                    color=TINTA_2)
    ax.set_ylim(0, d.max() * 1.15)
    ax.set_xlim(d.index.min(), fim + pd.Timedelta(days=120))
    ax.spines["bottom"].set_color(CINZA_CLARO)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    meses = ["", "jan", "", "", "", "", "", "jul", "", "", "", "", ""]
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f"{meses[mdates.num2date(x).month]} {mdates.num2date(x).year}"))
    fig.tight_layout()
    titulo(fig, "Quantos pórticos já cobravam, mês a mês",
           "Total acumulado desde o primeiro, na BR-101 (RJ), em março de 2023.")
    salvar(fig, FIG / "02_linha_do_tempo_ativacoes.png")


def fig_parana(df: pd.DataFrame) -> pd.DataFrame:
    pr = df[df.uf == "PR"].copy()
    pr["local"] = (pr["sigla"] + " · " + pr["municipio"].str.replace("Presidente", "Pres.")
                   + " · km " + pr["km"].map(lambda k: br(k, 1)))
    pr = pr.sort_values(["epr", "concessionaria", "sigla", "km"], ascending=[False, True, True, True])
    fig, ax = plt.subplots(figsize=(8, 4.4))
    cores = [VERDE if e else CINZA_CLARO for e in pr.epr]
    pos = range(len(pr))
    ax.barh(pos, pr["porticos_no_registro"], color=cores, height=0.55)
    ax.set_yticks(list(pos), pr["local"])
    for i, (n, emp) in enumerate(zip(pr["porticos_no_registro"], pr["concessionaria"])):
        emp = emp.split(" (")[0]
        ax.text(n + 0.05, i, f"{n}  ·  {emp}", va="center", fontsize=8.5, color=TINTA_3)
    ax.invert_yaxis()
    grade_x(ax)
    ax.set_xticks([0, 1, 2])
    ax.set_xlim(0, 3.4)
    ax.set_xlabel("pórticos no local")
    fig.tight_layout()
    titulo(fig, "Os pórticos do Paraná",
           "Em verde, os da EPR. A EPR Paraná tem dois em cada ponto, um por sentido.")
    salvar(fig, FIG / "03_porticos_parana.png")
    cols = ["concessionaria", "rodovia", "sigla", "municipio", "km", "sentido",
            "porticos_no_registro", "data_ativacao", "status"]
    pr[cols].to_csv(TAB / "porticos_parana.csv", index=False)
    return pr[cols]


if __name__ == "__main__":
    df = carregar()
    resumo(df)
    fig_por_uf(df)
    fig_linha_do_tempo(df)
    tabela = fig_parana(df)
    print("\nPórticos do Paraná:\n" + tabela.to_string(index=False))
    print(f"\nFiguras em {FIG}")
