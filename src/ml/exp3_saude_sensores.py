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
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    for ax, ((portico, faixa), (nome, ini)) in zip(axes, FALHAS.items()):
        s = series[(DIA_TESTE, portico, faixa)]
        for h0, h1 in _intervalos(s, "chuva"):
            ax.axvspan(h0, h1, color=AZUL, alpha=0.10, lw=0)
        for h0, h1 in _intervalos(s, "noite"):
            ax.axvspan(h0, h1, color=CINZA, alpha=0.12, lw=0)
        ax.plot(s["hora"], s["taxa_esp"], color=CINZA, lw=1.5, label="esperado pelo modelo")
        ax.plot(s["hora"], s["taxa"], color=AZUL, lw=1.3, label="observado")
        ax.axhline(LIMIAR_INGENUO, color=LARANJA, ls=":", lw=1)
        ing = s[s["alarme_ingenuo"]]
        ax.scatter(ing["hora"], np.full(len(ing), 0.31), marker="|", color=LARANJA, s=60,
                   label="alarme ingênuo (taxa < 85%)")
        al = s[s["alarme"]]
        if len(al):
            h_al = al["hora"].iloc[0]
            ax.axvline(h_al, color=VERMELHO, lw=2)
            atraso = (h_al - ini) * 60
            quando = ("na 1ª janela de 10 min" if atraso < JANELA_MIN else f"{atraso:.0f} min após a falha")
            ax.text(h_al + 0.15, 0.42, f"alarme do modelo\n{quando}",
                    color=VERMELHO, fontsize=8)
        ax.axvline(ini, color="#333", ls="--", lw=1)
        ax.text(ini - 0.15, 0.42, f"início: {nome}", ha="right", fontsize=8)
        ax.set_ylim(0.25, 1.02)
        ax.set_title(f"{portico} · faixa {faixa}", fontsize=10, loc="left")
        ax.set_ylabel("leituras confiáveis")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.95))
    axes[1].set_xlabel("hora do dia (dia de teste) · faixa azul = chuva · cinza = noite")
    axes[1].set_xlim(0, 24)
    axes[1].set_xticks(range(0, 25, 3))
    fig.suptitle("Saúde dos sensores: observado × esperado, com CUSUM", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "11_saude_sensores.png")
    plt.close(fig)

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
