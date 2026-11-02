"""
Experimento 3: detectar sensor degradado antes que ele vire reclamação.

No último dia simulado acontecem duas falhas:
  * lente suja na câmera frontal (BR-376 km 196 N, faixa 2), piorando aos poucos a partir das 13h
  * iluminador infravermelho queimado (BR-369 km 126 L, faixa 1) às 20h30, à noite

Indicador: fração de passagens em que a câmera frontal lê a placa com confiança >= 0,90.
O problema é que esse indicador também cai com chuva e à noite. Um alarme por limiar fixo
dispara toda vez que chove. Por isso:

  1. Um modelo (gradient boosting) aprende nos dias 1-2 a taxa ESPERADA de leitura confiável
     dadas as condições: hora, noite, chuva, velocidade, tipo de veículo, pórtico e faixa.
  2. A cada janela de 10 minutos, compara observado x esperado (escore z).
  3. CUSUM acumula desvios negativos persistentes e dispara quando passa de um limite.

Compara com o alarme ingênuo "taxa < 85% na janela".
Saídas: figura 11; reports/tabelas/exp3_resultados.json
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from comum import (AZUL, CINZA, DIA_TESTE, DIAS_TREINO, FIG, LARANJA, TAB, VERMELHO, carregar,
                   leituras)
from comum import ALERTA, AMBAR, CINZA_CLARO, TINTA_2, TINTA_3, br, grade_x, salvar, titulo

JANELA_MIN = 10
LIMIAR_INGENUO = 0.85
CUSUM_K, CUSUM_H = 0.5, 8.0
FALHAS = {("BR-376 km 196 N", 2): ("lente suja (gradual)", 13.0),
          ("BR-369 km 126 L", 1): ("iluminador IR queimado", 20.5)}


def preparar():
    df = carregar()
    df = df[df["tipo"] != "moto"].copy()                  # moto não tem placa dianteira
    r = leituras(df)
    df["ok"] = (r["conf_frente"] >= 0.90).astype(int)
    df["janela"] = (df["hora"] * 60 // JANELA_MIN).astype(int)
    for c in ("portico", "tipo"):
        df[c + "_cod"] = df[c].astype("category").cat.codes
    return df


FEATS = ["hora", "noite", "chuva", "velocidade_kmh", "tipo_cod", "portico_cod", "faixa"]


def cusum(z):
    s, out = 0.0, []
    for v in z:
        s = max(0.0, s - v - CUSUM_K)
        out.append(s)
    return np.array(out)


def main():
    df = preparar()
    tr = df[df.dia.isin(DIAS_TREINO)]
    modelo = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0,
                                            categorical_features=[4, 5])
    modelo.fit(tr[FEATS], tr["ok"])
    df["esperado"] = modelo.predict_proba(df[FEATS])[:, 1]

    g = (df.groupby(["dia", "portico", "faixa", "janela"])
           .agg(n=("ok", "size"), obs=("ok", "sum"), esp=("esperado", "sum"),
                var=("esperado", lambda p: (p * (1 - p)).sum()), chuva=("chuva", "max"),
                noite=("noite", "max"))
           .reset_index())
    g = g[g["n"] >= 5]
    g["taxa"] = g["obs"] / g["n"]
    g["taxa_esp"] = g["esp"] / g["n"]
    g["z"] = (g["obs"] - g["esp"]) / np.sqrt(g["var"].clip(lower=1e-6))
    g["hora"] = g["janela"] * JANELA_MIN / 60

    linhas, series = [], {}
    for (dia, portico, faixa), s in g.groupby(["dia", "portico", "faixa"]):
        s = s.sort_values("hora").copy()
        s["cusum"] = cusum(s["z"].to_numpy())
        s["alarme"] = s["cusum"] > CUSUM_H
        s["alarme_ingenuo"] = s["taxa"] < LIMIAR_INGENUO
        series[(dia, portico, faixa)] = s
        falha = FALHAS.get((portico, faixa)) if dia == DIA_TESTE else None
        ini = falha[1] if falha else None
        def resumo_alarmes(col):
            al = s[s[col]]
            if falha:
                antes = al[al["hora"] < ini]
                depois = al[al["hora"] >= ini]
                atraso = (depois["hora"].iloc[0] - ini) * 60 if len(depois) else None
                return len(antes), atraso
            return len(al), None
        fa_m, atraso_m = resumo_alarmes("alarme")
        fa_i, atraso_i = resumo_alarmes("alarme_ingenuo")
        linhas.append({"dia": dia + 1, "portico": portico, "faixa": faixa,
                       "falha": falha[0] if falha else "",
                       "janelas_alarme_falso_modelo": fa_m, "atraso_deteccao_modelo_min": atraso_m,
                       "janelas_alarme_falso_ingenuo": fa_i, "atraso_deteccao_ingenuo_min": atraso_i})
    tab = pd.DataFrame(linhas)
    pd.set_option("display.width", 220)
    print(tab.to_string(index=False))
    resumo = {
        "falsos_alarmes_modelo_total": int(tab["janelas_alarme_falso_modelo"].sum()),
        "falsos_alarmes_ingenuo_total": int(tab["janelas_alarme_falso_ingenuo"].sum()),
        "deteccoes": tab[tab.falha != ""][["portico", "faixa", "falha", "atraso_deteccao_modelo_min",
                                            "atraso_deteccao_ingenuo_min"]].to_dict(orient="records"),
    }
    print(json.dumps(resumo, indent=2, ensure_ascii=False))

    # ---------------------------------------------------------------- figura
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    nomes_falha = {"lente suja (gradual)": "a lente começa a sujar",
                   "iluminador IR queimado": "o flash infravermelho queima"}
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True)
    for ax, ((portico, faixa), (nome, ini)) in zip(axes, FALHAS.items()):
        s = series[(DIA_TESTE, portico, faixa)]
        for h0, h1 in _intervalos(s, "noite"):
            ax.axvspan(h0, h1, color="#f1f4f2", lw=0)
        for h0, h1 in _intervalos(s, "chuva"):
            ax.axvspan(h0, h1, color=AZUL, alpha=0.08, lw=0)
        ax.plot(s["hora"], s["taxa_esp"], color=CINZA, lw=1.6)
        ax.plot(s["hora"], s["taxa"], color=AZUL, lw=1.6)
        ing = s[s["alarme_ingenuo"]]
        ax.scatter(ing["hora"], np.full(len(ing), 0.30), marker="|", color=AMBAR, s=70, lw=1.6)
        ax.axvline(ini, color=TINTA_3, lw=1)
        ax.annotate(nomes_falha[nome], (ini, 0.55), xytext=(-6, 0), textcoords="offset points",
                    ha="right", fontsize=8.5, color=TINTA_2)
        al = s[s["alarme"]]
        if len(al):
            h_al = al["hora"].iloc[0]
            ax.axvline(h_al, color=ALERTA, lw=2)
            atraso = (h_al - ini) * 60
            quando = "nos primeiros 10 min" if atraso < JANELA_MIN else f"{atraso:.0f} min depois"
            ax.annotate(f"o modelo avisa\n{quando}", (h_al, 0.55), xytext=(6, 0),
                        textcoords="offset points", fontsize=8.5, color=ALERTA, va="center")
        ax.set_ylim(0.22, 1.04)
        ax.set_yticks([0.4, 0.6, 0.8, 1.0], ["40%", "60%", "80%", "100%"])
        ax.text(0, 1.06, f"{portico} · faixa {faixa}", transform=ax.transAxes, fontsize=9.5,
                fontweight="medium", color=TINTA_2)
    axes[1].set_xlim(0, 24)
    axes[1].set_xticks(range(0, 25, 3), [f"{h}h" for h in range(0, 25, 3)])
    axes[1].spines["bottom"].set_color(CINZA_CLARO)
    legenda = [Line2D([], [], color=AZUL, lw=2, label="o que a câmera leu"),
               Line2D([], [], color=CINZA, lw=2, label="o que ela deveria ler"),
               Line2D([], [], color=AMBAR, marker="|", lw=0, markersize=10, markeredgewidth=1.6,
                      label="alarme simples (abaixo de 85%)"),
               Patch(color=AZUL, alpha=0.15, label="chuva"),
               Patch(color="#e6ebe8", label="noite")]
    fig.tight_layout(h_pad=2.2)
    fig.legend(handles=legenda, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5,
               fontsize=8.5, handlelength=1.6, columnspacing=1.4)
    titulo(fig, "O alarme que não se assusta com chuva",
           "Placas lidas com segurança a cada 10 minutos, no dia de teste (dados simulados).")
    salvar(fig, FIG / "11_saude_sensores.png")

    tab.to_csv(TAB / "exp3_alarmes.csv", index=False)
    (TAB / "exp3_resultados.json").write_text(json.dumps(
        {"janela_min": JANELA_MIN, "cusum_k": CUSUM_K, "cusum_h": CUSUM_H,
         "limiar_ingenuo": LIMIAR_INGENUO, **resumo}, indent=2, ensure_ascii=False, default=float))


def _intervalos(s, col):
    out, ini = [], None
    for h, v in zip(s["hora"], s[col]):
        if v and ini is None:
            ini = h
        if not v and ini is not None:
            out.append((ini, h))
            ini = None
    if ini is not None:
        out.append((ini, 24))
    return out


if __name__ == "__main__":
    main()
