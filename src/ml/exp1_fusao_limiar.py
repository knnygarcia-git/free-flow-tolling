"""
Experimento 1: fusão das leituras com modelo e limiar escolhido por custo.

Pergunta: dá para cobrar automaticamente mais passagens, errando menos, do que a regra fixa
"tag lida OU OCR com confiança >= 90%"?

Método
  1. Candidato a cobrança: placa da tag (se lida) ou a melhor leitura de OCR.
  2. Alvo: o candidato é a placa verdadeira? (na vida real: rótulo vindo da revisão humana)
  3. Modelo: gradient boosting com calibração isotônica -> P(candidato correto).
  4. Limiar: aceita automaticamente se P >= t; senão manda para revisão humana.
     t é escolhido para minimizar o custo esperado:
        custo = revisões x CUSTO_REVISAO + cobranças erradas x CUSTO_ERRO
  5. Validação temporal: treina nos dias 1-2 e testa no dia 3.

Também calcula uma política de armazenamento de evidências guiada pela mesma probabilidade.

Saídas: figuras 06, 07, 08 e 12; reports/tabelas/exp1_*.csv
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, roc_auc_score

from comum import (AZUL, CINZA, DIA_TESTE, DIAS_TREINO, FEATURES, FIG, LARANJA, NOMES, TAB,
                   VERDE, VERMELHO, carregar, montar_features)

CUSTO_REVISAO = 0.60     # R$ por imagem revisada por um operador (hipótese)
CUSTO_ERRO = 30.00       # R$ por cobrança na placa errada: contestação, estorno, atendimento (hipótese)
KB = 1e3
EVID_COMPLETA = 965.3 * KB    # pacote completo (ver src/volume_dados.py)
EVID_REDUZIDA = 35.0 * KB     # só recortes da placa + metadados + perfil do laser


def resumo(nome, aceita, correto):
    n = len(correto)
    rev = (~aceita).sum()
    erros = (aceita & (correto == 0)).sum()
    return {
        "estrategia": nome,
        "cobranca_automatica_pct": 100 * aceita.mean(),
        "revisao_pct": 100 * rev / n,
        "cobrancas_erradas": int(erros),
        "erros_por_10k": 1e4 * erros / n,
        "acuracia_automatica_pct": 100 * (correto[aceita] == 1).mean(),
        "custo_R$": rev * CUSTO_REVISAO + erros * CUSTO_ERRO,
        "custo_por_10k_R$": 1e4 * (rev * CUSTO_REVISAO + erros * CUSTO_ERRO) / n,
    }


def main():
    print("Montando variáveis...")
    df = montar_features(carregar())
    tr, te = df[df.dia.isin(DIAS_TREINO)], df[df.dia == DIA_TESTE]
    Xtr, ytr, Xte, yte = tr[FEATURES], tr["correto"], te[FEATURES], te["correto"].to_numpy()
    print(f"treino {len(tr):,} | teste {len(te):,} | candidatos errados no teste: {(yte == 0).mean():.2%}")

    base = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=0)
    modelo = CalibratedClassifierCV(base, method="isotonic", cv=3).fit(Xtr, ytr)
    p = modelo.predict_proba(Xte)[:, 1]
    print(f"AUC {roc_auc_score(yte, p):.4f} | Brier {brier_score_loss(yte, p):.5f}")

    # --- regra atual (baseline)
    regra = (te["tag_lida"] | (te["conf_melhor"] >= 0.90)).to_numpy()
    # --- curva de custo
    ts = np.unique(np.concatenate([np.linspace(0.5, 0.999, 400), [0.9995, 0.9999]]))
    custos = np.array([resumo("", p >= t, yte)["custo_R$"] for t in ts])
    t_otimo = ts[np.argmin(custos)]
    t_teorico = 1 - CUSTO_REVISAO / CUSTO_ERRO
    # curva equivalente para a regra (variando só o limiar de confiança do OCR)
    ts_conf = np.linspace(0.5, 0.999, 300)
    custos_regra = np.array([resumo("", (te["tag_lida"] | (te["conf_melhor"] >= t)).to_numpy(),
                                    yte)["custo_R$"] for t in ts_conf])

    tabela = pd.DataFrame([
        resumo("Regra atual (tag ou OCR >= 90%)", regra, yte),
        resumo(f"Regra com melhor limiar de OCR ({ts_conf[np.argmin(custos_regra)]:.3f})",
               (te["tag_lida"] | (te["conf_melhor"] >= ts_conf[np.argmin(custos_regra)])).to_numpy(), yte),
        resumo(f"Modelo, limiar ótimo ({t_otimo:.3f})", p >= t_otimo, yte),
    ])
    tabela.to_csv(TAB / "exp1_comparacao.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n" + tabela.round(2).to_string(index=False))

    # casos que a regra deixava passar e o modelo pegou
    pegos = te[regra & (p < t_otimo) & (yte == 0)]
    tipos = pd.Series(np.where(pegos["_tag_outro_veiculo"], "tag em outro veículo",
                               "OCR errado com confiança alta")).value_counts()
    print("\nCobranças erradas evitadas pelo modelo:\n" + tipos.to_string())

    # ---------------------------------------------------------------- figuras
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(ts, custos / len(te) * 1e4, color=AZUL, lw=2, label="modelo (limiar na probabilidade)")
    ax.plot(ts_conf, custos_regra / len(te) * 1e4, color=CINZA, lw=2,
            label="regra (limiar na confiança do OCR)")
    b = resumo("", regra, yte)["custo_por_10k_R$"]
    ax.scatter([0.90], [b], color=LARANJA, zorder=5, s=50, label=f"regra atual: R$ {b:,.0f}".replace(",", "."))
    m = custos.min() / len(te) * 1e4
    ax.scatter([t_otimo], [m], color=VERDE, zorder=5, s=50, label=f"modelo no ótimo: R$ {m:,.0f}".replace(",", "."))
    ax.axvline(t_teorico, color=CINZA, ls=":", lw=1)
    ax.text(t_teorico - 0.004, ax.get_ylim()[1] * 0.92, f"1 − custo revisão/custo erro = {t_teorico:.2f}",
            ha="right", fontsize=8, color="#555")
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("limiar de aceitação automática")
    ax.set_ylabel("custo por 10 mil passagens (R$)")
    ax.set_title("Custo total (revisão + cobranças erradas) por limiar")
    ax.legend(frameon=False, fontsize=8, loc="center left")
    fig.tight_layout()
    fig.savefig(FIG / "06_custo_por_limiar.png")
    plt.close(fig)

    # calibração: confiança crua do OCR x probabilidade do modelo (só candidatos via OCR)
    ocr = (te["fonte"] == "ocr").to_numpy()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color=CINZA, ls="--", lw=1)
    for nome, score, cor in [("confiança crua do OCR", te["conf_melhor"].to_numpy()[ocr], LARANJA),
                             ("probabilidade do modelo", p[ocr], AZUL)]:
        fr, mp = calibration_curve(yte[ocr], score, n_bins=10, strategy="uniform")
        ax.plot(mp, fr, "o-", color=cor, lw=2, ms=4, label=nome)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("probabilidade prevista de acerto")
    ax.set_ylabel("acerto observado")
    ax.set_title("Calibração (passagens sem tag)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "07_calibracao.png")
    plt.close(fig)

    amostra = te.sample(min(15000, len(te)), random_state=1)
    imp = permutation_importance(modelo, amostra[FEATURES], amostra["correto"], scoring="roc_auc",
                                 n_repeats=4, random_state=0)
    s = pd.Series(imp.importances_mean, index=[NOMES[f] for f in FEATURES]).sort_values().tail(10)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(s.index, s.values, color=AZUL)
    ax.set_xlabel("queda de AUC ao embaralhar a variável")
    ax.set_title("O que mais pesa para saber se a leitura está certa")
    fig.tight_layout()
    fig.savefig(FIG / "08_importancia_variaveis.png")
    plt.close(fig)

    # ---------------------------------------------------------------- armazenamento
    # Política: pacote completo só onde a prova pode ser necessária
    #   (cobrança pela placa, revisão, ou tag cuja leitura o modelo não confirma).
    #   Tag confirmada com P >= 0,999: guarda versão reduzida.
    completa = (~te["tag_lida"].to_numpy()) | (p < 0.999)
    gb_atual = len(te) * EVID_COMPLETA / 1e9
    gb_nova = (completa.sum() * EVID_COMPLETA + (~completa).sum() * EVID_REDUZIDA) / 1e9
    arm = {"passagens_dia_teste": int(len(te)), "gb_dia_guardando_tudo": round(gb_atual, 2),
           "gb_dia_politica_por_risco": round(gb_nova, 2),
           "reducao_pct": round(100 * (1 - gb_nova / gb_atual), 1),
           "pacotes_completos_pct": round(100 * completa.mean(), 1)}
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    ax.barh(["política por risco", "guardar tudo"], [gb_nova, gb_atual], color=[VERDE, CINZA])
    for i, v in enumerate([gb_nova, gb_atual]):
        ax.text(v + gb_atual * 0.01, i, f"{v:.1f} GB/dia", va="center")
    ax.set_xlim(0, gb_atual * 1.25)
    ax.set_xlabel("evidências guardadas por dia (4 pórticos)")
    ax.set_title(f"Armazenamento: −{arm['reducao_pct']:.0f}% guardando o pacote completo só quando precisa")
    fig.tight_layout()
    fig.savefig(FIG / "12_politica_armazenamento.png")
    plt.close(fig)
    print("\nArmazenamento:", arm)

    res = {"auc": roc_auc_score(yte, p), "brier": brier_score_loss(yte, p),
           "limiar_otimo": float(t_otimo), "limiar_teorico": t_teorico,
           "custo_revisao": CUSTO_REVISAO, "custo_erro": CUSTO_ERRO,
           "comparacao": tabela.round(3).to_dict(orient="records"),
           "erros_evitados": tipos.to_dict(), "armazenamento": arm}
    (TAB / "exp1_resultados.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
