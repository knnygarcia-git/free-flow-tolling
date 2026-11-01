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

# Paleta simples e consistente
AZUL, CINZA, LARANJA = "#2a6fdb", "#9aa3ad", "#e8833a"
plt.rcParams.update({
    "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 12, "font.size": 10,
})


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
    fig, ax = plt.subplots(figsize=(7, 4))
    cores = [LARANJA if uf == "PR" else AZUL for uf in s.index]
    ax.barh(s.index, s.values, color=cores)
    for i, v in enumerate(s.values):
        ax.text(v + 0.4, i, str(v), va="center")
    ax.set_title("Pórticos Free Flow cadastrados por UF")
    ax.set_xlabel("pórticos físicos")
    fig.tight_layout()
    fig.savefig(FIG / "01_porticos_por_uf.png")
    plt.close(fig)


def fig_linha_do_tempo(df: pd.DataFrame) -> None:
    d = (df.dropna(subset=["data_ativacao"])
           .groupby("data_ativacao").porticos_no_registro.sum()
           .sort_index().cumsum())
    pr = (df[(df.uf == "PR")].dropna(subset=["data_ativacao"])
            .groupby("data_ativacao").porticos_no_registro.sum().sort_index().cumsum())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.step(d.index, d.values, where="post", color=AZUL, lw=2, label="Brasil")
    ax.step(pr.index, pr.values, where="post", color=LARANJA, lw=2, label="Paraná")
    ax.set_title("Crescimento acumulado de pórticos Free Flow ativados")
    ax.set_ylabel("pórticos acumulados")
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "02_linha_do_tempo_ativacoes.png")
    plt.close(fig)


def fig_parana(df: pd.DataFrame) -> pd.DataFrame:
    pr = df[df.uf == "PR"].copy()
    pr["local"] = pr["sigla"] + " km " + pr["km"].map(lambda k: f"{k:g}") + " · " + pr["municipio"]
    pr = pr.sort_values(["concessionaria", "sigla", "km"])
    fig, ax = plt.subplots(figsize=(8, 4))
    cores = [LARANJA if e else CINZA for e in pr.epr]
    ax.barh(pr["local"], pr["porticos_no_registro"], color=cores)
    ax.invert_yaxis()
    ax.set_title("Paraná: pórticos por local (laranja = grupo EPR)")
    ax.set_xlabel("pórticos no local")
    ax.set_xticks([0, 1, 2])
    fig.tight_layout()
    fig.savefig(FIG / "03_porticos_parana.png")
    plt.close(fig)
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
