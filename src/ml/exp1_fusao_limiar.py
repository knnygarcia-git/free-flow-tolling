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
from comum import ALERTA, AMBAR, CINZA_CLARO, TINTA_2, TINTA_3, br, grade_x, salvar, titulo

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
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    y_mod, y_reg = custos / len(te) * 1e4, custos_regra / len(te) * 1e4
    ax.plot(ts_conf, y_reg, color=CINZA, lw=2)
    ax.plot(ts, y_mod, color=VERDE, lw=2)
    ax.fill_between(ts, y_mod, color=VERDE, alpha=0.08, lw=0)
    b = resumo("", regra, yte)["custo_por_10k_R$"]
    m = custos.min() / len(te) * 1e4
    ax.scatter([0.90], [b], s=40, color=CINZA, zorder=4, edgecolor="white", linewidth=2)
    ax.annotate(f"regra comum (90%)\nR$ {br(b)}", (0.90, b), xytext=(-12, -34),
                textcoords="offset points", ha="right", fontsize=9, color=TINTA_2)
    ax.scatter([t_otimo], [m], s=40, color=VERDE, zorder=4, edgecolor="white", linewidth=2)
    ax.annotate(f"modelo no melhor corte\nR$ {br(m)}", (t_otimo, m), xytext=(-10, 14),
                textcoords="offset points", ha="right", fontsize=9, color=TINTA_2)
    ax.annotate("regra: cortar pela certeza da câmera", (0.6, y_reg[np.searchsorted(ts_conf, 0.6)]),
                xytext=(0, 8), textcoords="offset points", fontsize=9, color=TINTA_3)
    ax.annotate("modelo: cortar pela chance real de acerto", (0.6, y_mod[np.searchsorted(ts, 0.6)]),
                xytext=(0, 8), textcoords="offset points", fontsize=9, color=TINTA_3)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0, max(y_reg.max(), y_mod.max()) * 1.1)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ["50%", "60%", "70%", "80%", "90%", "100%"])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"R$ {br(v)}"))
    ax.set_xlabel("ponto de corte: cobra sozinho acima dele, chama uma pessoa abaixo")
    ax.spines["bottom"].set_color(CINZA_CLARO)
    fig.tight_layout()
    titulo(fig, "Quanto custa cada ponto de corte",
           "Conferências (R$ 0,60) + cobranças erradas (R$ 30), a cada 10 mil passagens. Dados simulados.")
    salvar(fig, FIG / "06_custo_por_limiar.png")

    # calibração: certeza crua da câmera x chance calculada pelo modelo (só passagens sem tag)
    ocr = (te["fonte"] == "ocr").to_numpy()
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.grid(True, axis="both")
    ax.plot([0, 1], [0, 1], color=CINZA_CLARO, lw=1.5)
    ax.annotate("previsão perfeita", (0.42, 0.42), xytext=(6, -14), textcoords="offset points",
                fontsize=8.5, color=TINTA_3, rotation=45)
    bordas = np.array([0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0001])
    for nome, score, cor in [("certeza da câmera", te["conf_melhor"].to_numpy()[ocr], AMBAR),
                             ("chance calculada pelo modelo", p[ocr], VERDE)]:
        alvo = yte[ocr]
        grupo = np.digitize(score, bordas) - 1
        mp, fr = [], []
        for g in range(len(bordas) - 1):
            m_ = grupo == g
            if m_.sum() >= 40:                    # só faixas com passagens suficientes
                mp.append(score[m_].mean())
                fr.append(alvo[m_].mean())
        ax.plot(mp, fr, color=cor, lw=2)
        ax.scatter(mp, fr, s=34, color=cor, zorder=4, edgecolor="white", linewidth=1.5, label=nome)
    ax.set_xlim(0.2, 1.01)
    ax.set_ylim(0.2, 1.01)
    pct = [0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_xticks(pct, [f"{int(v*100)}%" for v in pct])
    ax.set_yticks(pct, [f"{int(v*100)}%" for v in pct])
    ax.set_xlabel("o que foi prometido")
    ax.set_ylabel("o que aconteceu")
    ax.legend(loc="upper left")
    fig.tight_layout()
    titulo(fig, "A câmera promete mais do que entrega",
           "Passagens sem tag. Quanto mais perto da diagonal, mais honesta a previsão.")
    salvar(fig, FIG / "07_calibracao.png")

    amostra = te.sample(min(15000, len(te)), random_state=1)
    imp = permutation_importance(modelo, amostra[FEATURES], amostra["correto"], scoring="roc_auc",
                                 n_repeats=4, random_state=0)
    s = pd.Series(imp.importances_mean, index=[NOMES[f] for f in FEATURES]).sort_values().tail(8)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    pos = np.arange(len(s))
    cores = [CINZA_CLARO] * (len(s) - 1) + [VERDE]
    ax.barh(pos, s.values, color=cores, height=0.6)
    ax.set_yticks(pos, s.index)
    ax.tick_params(axis="y", labelsize=9.5, labelcolor=TINTA_2)
    grade_x(ax)
    ax.set_xticks([])
    fig.tight_layout()
    titulo(fig, "As pistas que mais ajudam o modelo",
           "Quanto a previsão piora quando a pista é embaralhada. Barra maior, pista mais útil.")
    salvar(fig, FIG / "08_importancia_variaveis.png")

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
    fig, ax = plt.subplots(figsize=(8, 2.4))
    ax.barh([1, 0], [gb_atual, gb_nova], color=[CINZA_CLARO, VERDE], height=0.45)
    ax.set_yticks([1, 0], ["Guardar todas as fotos", "Guardar conforme o risco"])
    for yv, v in ((1, gb_atual), (0, gb_nova)):
        ax.text(v + gb_atual * 0.015, yv, f"{br(v, 1)} GB por dia", va="center", fontsize=9.5,
                color=TINTA_2, fontweight="medium")
    grade_x(ax)
    ax.set_xticks([])
    ax.set_xlim(0, gb_atual * 1.3)
    ax.tick_params(axis="y", labelsize=10, labelcolor=TINTA_2)
    fig.tight_layout()
    titulo(fig, f"{br(arm['reducao_pct'])}% menos espaço guardando as fotos completas só quando precisa",
           "4 pórticos simulados. A versão enxuta fica para tag confirmada pela placa.")
    salvar(fig, FIG / "12_politica_armazenamento.png")
    print("\nArmazenamento:", arm)

    res = {"auc": roc_auc_score(yte, p), "brier": brier_score_loss(yte, p),
           "limiar_otimo": float(t_otimo), "limiar_teorico": t_teorico,
           "custo_revisao": CUSTO_REVISAO, "custo_erro": CUSTO_ERRO,
           "comparacao": tabela.round(3).to_dict(orient="records"),
           "erros_evitados": tipos.to_dict(), "armazenamento": arm}
    (TAB / "exp1_resultados.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
