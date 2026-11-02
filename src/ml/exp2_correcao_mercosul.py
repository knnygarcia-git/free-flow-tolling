"""
Experimento 2: corrigir a leitura da placa usando o formato Mercosul e todos os quadros.

A câmera fotografa cada veículo várias vezes (frente e traseira). O OCR devolve uma leitura
por quadro, com uma confiança. Quatro formas de chegar à placa final:

  A. Melhor quadro            -> a leitura com maior confiança (o que muitos sistemas fazem)
  B. A + regra de formato     -> Mercosul é LLLNLNN: dígito em posição de letra vira a letra
                                 parecida (0->O, 8->B, 1->I ...) e vice-versa
  C. Votação por caractere    -> para cada posição, vence o caractere com maior soma de
                                 confiança entre todos os quadros; depois aplica o formato
  D. Modelo probabilístico    -> aprende no treino a matriz de confusão P(lido | real) e escolhe,
                                 por posição, o caractere real mais provável dado TODOS os
                                 quadros, restrito ao tipo da posição (letra ou dígito)

Só passagens sem tag (as que dependem do OCR). Treina nos dias 1-2, testa no dia 3.
Saídas: figuras 09 e 10; reports/tabelas/exp2_resultados.json
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comum import (AZUL, CINZA, DIA_TESTE, DIAS_TREINO, FIG, LARANJA, TAB, VERDE, carregar,
                   parse)

LETRAS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITOS = "0123456789"
ALFA = LETRAS + DIGITOS
IDX = {c: i for i, c in enumerate(ALFA)}
POS_LETRA = [True, True, True, False, True, False, False]
DIG_PARA_LETRA = {"0": "O", "1": "I", "2": "Z", "3": "B", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B", "9": "P"}
LETRA_PARA_DIG = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "T": "7", "Z": "2", "B": "8",
                  "S": "5", "G": "6", "A": "4", "E": "3"}


def corrigir_formato(placa: str) -> str:
    out = []
    for c, letra in zip(placa, POS_LETRA):
        if letra and c.isdigit():
            c = DIG_PARA_LETRA.get(c, c)
        elif not letra and c.isalpha():
            c = LETRA_PARA_DIG.get(c, c)
        out.append(c)
    return "".join(out)


def quadros(row) -> list[tuple[str, float]]:
    return parse(row.ocr_frente) + parse(row.ocr_tras)


def melhor_quadro(qs):
    return max(qs, key=lambda x: x[1])[0]


def votacao(qs):
    placa = []
    for j in range(7):
        score = {}
        for p, c in qs:
            score[p[j]] = score.get(p[j], 0) + c
        # respeita o tipo da posição antes de votar
        validos = {k: v for k, v in score.items() if k.isalpha() == POS_LETRA[j]}
        escolha = max(validos or score, key=(validos or score).get)
        placa.append(escolha)
    return corrigir_formato("".join(placa))


def aprender_confusao(df_treino: pd.DataFrame, suav: float = 0.5) -> np.ndarray:
    """log P(lido | real), estimada por contagem com suavização de Laplace."""
    C = np.full((36, 36), suav)
    for row in df_treino.itertuples():
        real = row.placa_real
        for p, _ in quadros(row):
            for a, b in zip(real, p):
                C[IDX[a], IDX[b]] += 1
    return np.log(C / C.sum(axis=1, keepdims=True))


def decodificar(qs, logP: np.ndarray, peso_conf: float = 2.0) -> str:
    placa = []
    for j in range(7):
        cands = LETRAS if POS_LETRA[j] else DIGITOS
        idx_c = [IDX[c] for c in cands]
        score = np.zeros(len(cands))
        for p, conf in qs:
            w = 1.0 + peso_conf * (conf - 0.5)          # quadros mais confiáveis pesam mais
            score += w * logP[idx_c, IDX[p[j]]]
        placa.append(cands[int(np.argmax(score))])
    return "".join(placa)


def main():
    df = carregar()
    df = df[~df["tag_lida"]].copy()                       # só quem depende do OCR
    tr, te = df[df.dia.isin(DIAS_TREINO)], df[df.dia == DIA_TESTE]
    print(f"passagens sem tag: treino {len(tr):,} | teste {len(te):,}")

    logP = aprender_confusao(tr)
    res = {"A. melhor quadro": [], "B. + regra Mercosul": [], "C. votação + regra": [],
           "D. modelo de confusão": []}
    for row in te.itertuples():
        qs = quadros(row)
        a = melhor_quadro(qs)
        res["A. melhor quadro"].append(a)
        res["B. + regra Mercosul"].append(corrigir_formato(a))
        res["C. votação + regra"].append(votacao(qs))
        res["D. modelo de confusão"].append(decodificar(qs, logP))

    real = te["placa_real"].to_numpy()
    linhas = []
    for nome, pred in res.items():
        pred = np.array(pred)
        acerto = (pred == real)
        car = np.mean([sum(x == y for x, y in zip(p, r)) / 7 for p, r in zip(pred, real)])
        linhas.append({"metodo": nome, "acuracia_placa_pct": 100 * acerto.mean(),
                       "erros_por_10k": 1e4 * (1 - acerto.mean()), "acuracia_caractere_pct": 100 * car,
                       "acuracia_noite_chuva_pct": 100 * acerto[(te.noite | te.chuva).to_numpy() == 1].mean(),
                       "acuracia_placa_suja_pct": 100 * acerto[(te.estado_placa == "suja").to_numpy()].mean()})
    tab = pd.DataFrame(linhas)
    base_err = tab.loc[0, "erros_por_10k"]
    tab["reducao_erros_pct"] = 100 * (1 - tab["erros_por_10k"] / base_err)
    pd.set_option("display.width", 220)
    print(tab.round(2).to_string(index=False))

    # onde estão os erros que sobram no melhor método?
    melhor = tab.loc[tab["erros_por_10k"].idxmin(), "metodo"]
    sobra = np.array(res[melhor]) != real
    estado = te["estado_placa"].to_numpy()
    restantes = pd.Series(estado[sobra]).value_counts(normalize=True).mul(100).round(1)
    print(f"\nErros que sobram em '{melhor}', por estado da placa (%):\n" + restantes.to_string())

    # ---------------------------------------------------------------- figuras
    fig, ax = plt.subplots(figsize=(8, 3.8))
    cores = [CINZA, LARANJA, AZUL, VERDE]
    ax.barh(tab["metodo"], tab["erros_por_10k"], color=cores)
    for i, (e, r) in enumerate(zip(tab["erros_por_10k"], tab["reducao_erros_pct"])):
        txt = f"{e:,.0f}" + ("" if i == 0 else f"  (−{r:.0f}%)")
        ax.text(e + base_err * 0.01, i, txt, va="center")
    ax.invert_yaxis()
    ax.set_xlim(0, base_err * 1.25)
    ax.set_xlabel("placas lidas erradas a cada 10 mil passagens sem tag")
    ax.set_title("Corrigindo a leitura com o formato Mercosul e todos os quadros")
    fig.tight_layout()
    fig.savefig(FIG / "09_correcao_mercosul.png")
    plt.close(fig)

    # confusões mais frequentes aprendidas (fora da diagonal)
    P = np.exp(logP)
    np.fill_diagonal(P, 0)
    cont = []
    for row in tr.itertuples():
        for p, _ in quadros(row):
            for a, b in zip(row.placa_real, p):
                if a != b:
                    cont.append(f"{a}→{b}")
    top = pd.Series(cont).value_counts().head(12)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(top.index, top.values, color=AZUL)
    ax.set_xlabel("ocorrências no treino (por quadro)")
    ax.set_title("Trocas de caractere mais comuns (real → lido)")
    fig.tight_layout()
    fig.savefig(FIG / "10_confusoes_ocr.png")
    plt.close(fig)

    (TAB / "exp2_resultados.json").write_text(json.dumps(
        {"passagens_teste": int(len(te)), "metodos": tab.round(3).to_dict(orient="records"),
         "top_confusoes": top[::-1].to_dict(), "melhor_metodo": melhor,
         "erros_restantes_por_estado_pct": restantes.to_dict()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
