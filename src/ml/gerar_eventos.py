"""
Gerador de passagens sintéticas em massa para os experimentos de machine learning.

Simula vários dias de operação de 4 pórticos (2 locais x 2 sentidos), no formato de dados que
um backoffice de Free Flow recebe: várias leituras de OCR por câmera (frente e traseira), com
confiança, leitura da tag (se houver), categoria medida pelo laser e consulta ao cadastro.

O gerador conhece a placa verdadeira de cada veículo, o que permite medir a acurácia. Na vida
real esse rótulo vem da revisão humana e das contestações.

Fenômenos simulados (todos parametrizáveis em CONFIG):
  * erro de OCR por caractere, com confusões visuais realistas (O/0/D/Q, B/8, I/1/L ...)
  * mais erro à noite, com chuva, em alta velocidade, com placa suja ou danificada
  * confiança do OCR informativa mas SUPERCONFIANTE (como a de muitos modelos reais)
  * tag em outro veículo (fraude) e categoria cadastrada diferente da medida
  * placas lidas erradas que não existem no cadastro (consulta ao "DETRAN")
  * degradação de sensores no último dia: lente suja (gradual) e iluminador IR queimado (abrupto)

Saída: data/sintetico/passagens.parquet (ou .csv.gz se não houver pyarrow)
"""
from __future__ import annotations

import string
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "sintetico"

LETRAS = string.ascii_uppercase
DIGITOS = string.digits
ALFABETO = LETRAS + DIGITOS                       # 36 símbolos
IDX = {c: i for i, c in enumerate(ALFABETO)}
POS_LETRA = np.array([True, True, True, False, True, False, False])   # Mercosul LLLNLNN

# Grupos de caracteres que o OCR confunde (parecem iguais em imagem IR de baixa resolução)
GRUPOS = ["O0DQ", "B83", "I1LT", "S5", "Z2", "G6C", "A4", "EF", "MNH", "UV", "KX", "PR", "79"]

CONFIG = dict(
    dias=3,
    seed=2026,
    porticos={"BR-376 km 196 N": 10_500, "BR-376 km 196 S": 10_000,
              "BR-369 km 126 L": 7_500, "BR-369 km 126 O": 7_000},   # veículos/dia
    p_tag=0.62,
    p_tag_outro_veiculo=0.004,
    p_placa_suja=0.08,
    p_placa_danificada=0.012,
    p_placa_tampada=0.003,
    err_base=0.012,          # erro por caractere, por quadro, em condição ideal
    # erro persistente: placa suja/danificada é lida errada DO MESMO JEITO em vários quadros
    p_persistente={"ok": 0.002, "suja": 0.05, "danificada": 0.14, "tampada": 0.45},
    p_persistente_aparece=0.75,
    fator_noite=1.8,
    fator_chuva=2.3,
    quadros=(3, 6),          # leituras por câmera
    p_laser_erro=0.004,
    p_placa_inexistente=0.35,  # placa lida errada que não existe no cadastro
    # chuva: (dia, hora_ini, hora_fim)
    chuvas=[(0, 17.0, 19.5), (1, 6.0, 8.0), (2, 15.5, 18.5)],
    # degradações no último dia
    lente_suja=dict(portico="BR-376 km 196 N", faixa=2, dia=2, ini=13.0, rampa_h=4.0, fator_max=7.0),
    ir_queimado=dict(portico="BR-369 km 126 L", faixa=1, dia=2, ini=20.5, fator=9.0),
)

TIPOS = {  # tipo: prob, categoria ANTT, velocidade média km/h, fator de sujeira
    "carro": (0.62, 1, 98, 1.0), "suv": (0.16, 1, 96, 1.0), "caminhao2": (0.09, 2, 80, 1.6),
    "caminhao5": (0.08, 7, 74, 2.0), "onibus": (0.03, 4, 86, 1.2), "moto": (0.02, 9, 95, 1.3),
}
PERFIL = np.array([1.4, 1.0, 0.9, 0.9, 1.2, 2.2, 4.0, 6.2, 6.6, 5.6, 5.3, 5.4,
                   5.5, 5.4, 5.5, 5.8, 6.4, 7.3, 7.6, 5.8, 4.3, 3.5, 2.7, 2.1])
PERFIL = PERFIL / PERFIL.sum()


def _tabela_confusao(rng) -> np.ndarray:
    """P(lido | verdadeiro) quando há erro: 85% dentro do grupo visual, 15% qualquer outro."""
    M = np.full((36, 36), 0.15 / 35)
    for g in GRUPOS:
        for a in g:
            outros = [b for b in g if b != a]
            for b in outros:
                M[IDX[a], IDX[b]] += 0.85 / len(outros)
    for c in ALFABETO:
        if not any(c in g for g in GRUPOS):
            M[IDX[c]] = 1.0 / 35
        M[IDX[c], IDX[c]] = 0
        M[IDX[c]] /= M[IDX[c]].sum()
    return M


def _placas(rng, n) -> np.ndarray:
    """Placas Mercosul como matriz (n, 7) de índices do alfabeto."""
    p = np.empty((n, 7), np.int8)
    for j in range(7):
        if POS_LETRA[j]:
            p[:, j] = rng.integers(0, 26, n)
        else:
            p[:, j] = rng.integers(26, 36, n)
    return p


def para_texto(m: np.ndarray) -> np.ndarray:
    lut = np.array(list(ALFABETO))
    return np.array(["".join(r) for r in lut[m]])


def gerar(cfg: dict = CONFIG) -> pd.DataFrame:
    rng = np.random.default_rng(cfg["seed"])
    M = _tabela_confusao(rng)
    Mcum = M.cumsum(axis=1)

    blocos = []
    for nome, vdm in cfg["porticos"].items():
        for dia in range(cfg["dias"]):
            n = rng.poisson(vdm)
            hora = rng.choice(24, n, p=PERFIL) + rng.random(n)
            blocos.append(pd.DataFrame({"portico": nome, "dia": dia, "hora": hora}))
    df = pd.concat(blocos, ignore_index=True)
    df = df.sort_values(["dia", "hora"]).reset_index(drop=True)
    n = len(df)

    tipos = list(TIPOS)
    probs = np.array([TIPOS[t][0] for t in tipos])
    df["tipo"] = rng.choice(tipos, n, p=probs / probs.sum())
    df["categoria_real"] = df["tipo"].map({t: v[1] for t, v in TIPOS.items()})
    pesado = df["tipo"].isin(["caminhao2", "caminhao5", "onibus"]).to_numpy()
    df["faixa"] = np.where(pesado & (rng.random(n) < 0.8), 1, rng.integers(1, 3, n))
    vmed = df["tipo"].map({t: v[2] for t, v in TIPOS.items()}).to_numpy()
    df["velocidade_kmh"] = np.clip(rng.normal(vmed + np.where(df["faixa"] == 2, 8, 0), 9), 30, 160)
    df["noite"] = ((df["hora"] < 6) | (df["hora"] >= 18.5)).astype(int)
    chuva = np.zeros(n, bool)
    for d, h0, h1 in cfg["chuvas"]:
        chuva |= (df["dia"].to_numpy() == d) & (df["hora"].between(h0, h1).to_numpy())
    df["chuva"] = chuva.astype(int)

    placa = _placas(rng, n)
    df["placa_real"] = para_texto(placa)
    u = rng.random(n)
    sujeira = df["tipo"].map({t: v[3] for t, v in TIPOS.items()}).to_numpy()
    estado = np.where(u < cfg["p_placa_tampada"], "tampada",
             np.where(u < cfg["p_placa_tampada"] + cfg["p_placa_danificada"], "danificada",
             np.where(rng.random(n) < cfg["p_placa_suja"] * sujeira, "suja", "ok")))
    df["estado_placa"] = estado

    # --- fator de erro por evento e por câmera
    f_evento = (cfg["err_base"]
                * np.where(df["noite"], cfg["fator_noite"], 1.0)
                * np.where(df["chuva"], cfg["fator_chuva"], 1.0)
                * (df["velocidade_kmh"].to_numpy() / 95) ** 2
                * pd.Series(estado).map({"ok": 1, "suja": 5, "danificada": 14, "tampada": 60}).to_numpy())
    f_frente = np.ones(n)
    f_tras = np.ones(n)
    ls, ir = cfg["lente_suja"], cfg["ir_queimado"]
    m = ((df["portico"] == ls["portico"]) & (df["faixa"] == ls["faixa"]) & (df["dia"] == ls["dia"])
         & (df["hora"] >= ls["ini"])).to_numpy()
    rampa = np.clip((df["hora"].to_numpy() - ls["ini"]) / ls["rampa_h"], 0, 1)
    f_frente = np.where(m, 1 + (ls["fator_max"] - 1) * rampa, f_frente)
    m2 = ((df["portico"] == ir["portico"]) & (df["faixa"] == ir["faixa"]) & (df["dia"] == ir["dia"])
          & (df["hora"] >= ir["ini"])).to_numpy()
    f_frente = np.where(m2, ir["fator"], f_frente)
    f_tras = np.where(m2, ir["fator"], f_tras)          # o iluminador serve às duas câmeras
    df["_degradado"] = (m & (rampa > 0.15)) | m2
    moto = (df["tipo"] == "moto").to_numpy()             # moto só tem placa traseira

    # --- leituras de OCR por quadro
    F = cfg["quadros"][1]
    nq_frente = np.where(moto, 0, rng.integers(cfg["quadros"][0], F + 1, n))
    nq_tras = rng.integers(cfg["quadros"][0], F + 1, n)
    leituras = {}
    # erro persistente: um caractere trocado que se repete na maioria dos quadros
    p_pers = pd.Series(estado).map(cfg["p_persistente"]).to_numpy()
    pers = rng.random((n, 7)) < p_pers[:, None]
    sub_pers = np.minimum((Mcum[placa.astype(int)] < rng.random((n, 7, 1))).sum(-1), 35)
    for cam, fcam, nq in (("frente", f_frente, nq_frente), ("tras", f_tras, nq_tras)):
        qualidade = rng.lognormal(0, 0.35, (n, F))               # variação entre quadros
        p_err = np.clip((f_evento * fcam)[:, None] * qualidade, 0, 0.75)
        erro = rng.random((n, F, 7)) < p_err[..., None]
        sub = (Mcum[placa[:, None, :].astype(int)] < rng.random((n, F, 7, 1))).sum(-1)
        sub = np.minimum(sub, 35)
        lido = np.where(erro, sub, placa[:, None, :])
        aparece = pers[:, None, :] & (rng.random((n, F, 7)) < cfg["p_persistente_aparece"])
        lido = np.where(aparece, sub_pers[:, None, :], lido)
        n_err = (lido != placa[:, None, :]).sum(-1)
        n_pers = aparece.sum(-1)
        # confiança: informativa, mas superconfiante e ruidosa. O erro persistente quase não
        # derruba a confiança: a imagem é nítida, o caractere é que está apagado/borrado.
        conf = (0.99 - 0.09 * (n_err - n_pers) - 0.025 * n_pers - 0.9 * p_err
                + rng.normal(0, 0.045, (n, F)))
        conf = np.clip(conf, 0.05, 0.999)
        valido = np.arange(F)[None, :] < nq[:, None]
        leituras[cam] = (lido, conf, valido)

    # serializa: "PLACA:conf|PLACA:conf|..."
    lut = np.array(list(ALFABETO))
    for cam, (lido, conf, valido) in leituras.items():
        txt = lut[lido]
        col = []
        for i in range(n):
            col.append("|".join(f"{''.join(txt[i, k])}:{conf[i, k]:.3f}"
                                for k in range(F) if valido[i, k]))
        df[f"ocr_{cam}"] = col

    # --- tag RFID
    tem_tag = rng.random(n) < cfg["p_tag"]
    df["tem_tag"] = tem_tag
    df["tag_lida"] = tem_tag & (rng.random(n) < 0.995)
    df["rssi_dbm"] = np.where(df["tag_lida"], rng.normal(-58, 5, n), np.nan)
    outro = df["tag_lida"].to_numpy() & (rng.random(n) < cfg["p_tag_outro_veiculo"])
    placa_tag = np.where(outro, para_texto(_placas(rng, n)), df["placa_real"])
    df["placa_tag"] = np.where(df["tag_lida"], placa_tag, "")
    df["_tag_outro_veiculo"] = outro
    cat_tag_outro = rng.choice([1, 1, 1, 2, 7], n)
    df["categoria_tag"] = np.where(df["tag_lida"], np.where(outro, cat_tag_outro, df["categoria_real"]), 0)

    # --- laser
    erro_laser = rng.random(n) < cfg["p_laser_erro"]
    df["categoria_laser"] = np.where(erro_laser, rng.choice([1, 2, 4, 7, 9], n), df["categoria_real"])

    df["ts"] = (pd.Timestamp("2026-06-01") + pd.to_timedelta(df["dia"], unit="D")
                + pd.to_timedelta(df["hora"], unit="h"))
    df.insert(0, "id_passagem", np.arange(n))
    return df


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    df = gerar()
    try:
        df.to_parquet(OUT / "passagens.parquet", index=False)
        destino = OUT / "passagens.parquet"
    except ImportError:
        destino = OUT / "passagens.csv.gz"
        df.to_csv(destino, index=False)
    print(f"{len(df):,} passagens em {CONFIG['dias']} dias -> {destino}")
    print(df[["portico", "tipo", "tem_tag", "estado_placa", "noite", "chuva"]]
          .describe(include="all").T[["count", "unique", "top", "freq"]])
