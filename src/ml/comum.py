"""Funções compartilhadas pelos experimentos de ML: carga, parsing do OCR e variáveis (features)."""
from __future__ import annotations

import re
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DADOS = ROOT / "data" / "sintetico"
FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tabelas"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

MERCOSUL = re.compile(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$")
DIAS_TREINO = [0, 1]          # validação temporal: treina nos dias 1-2, testa no dia 3
DIA_TESTE = 2

import sys
sys.path.insert(0, str(ROOT / "src"))
from estilo import (ALERTA, AMBAR, AZUL, CINZA, CINZA_CLARO, TINTA, TINTA_2, TINTA_3,  # noqa: E402
                    VERDE, br, grade_x, salvar, titulo)

LARANJA, VERMELHO = AMBAR, ALERTA


def carregar() -> pd.DataFrame:
    p = DADOS / "passagens.parquet"
    if not p.exists():
        p = DADOS / "passagens.csv.gz"
    if not p.exists():
        raise SystemExit("Rode antes: python src/ml/gerar_eventos.py")
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    for c in ("ocr_frente", "ocr_tras", "placa_tag"):
        df[c] = df[c].fillna("")
    return df


def parse(s: str) -> list[tuple[str, float]]:
    if not s:
        return []
    out = []
    for item in s.split("|"):
        placa, conf = item.split(":")
        out.append((placa, float(conf)))
    return out


def hamming(a: str, b: str) -> int:
    if not a or not b:
        return 7
    return sum(x != y for x, y in zip(a, b))


def cadastro(placa: str, placa_real: str, categoria_real: int, p_inexistente: float = 0.35):
    """Simula a consulta ao cadastro nacional de veículos.
    A placa verdadeira existe com a categoria real. Uma placa lida errada pode não existir,
    ou existir e pertencer a outro veículo, com categoria qualquer. Determinístico por placa."""
    if placa == placa_real:
        return True, categoria_real
    h = zlib.crc32(placa.encode())
    if (h % 1000) / 1000 < p_inexistente:
        return False, 0
    return True, [1, 1, 1, 1, 2, 4, 7, 9][(h // 1000) % 8]


def leituras(df: pd.DataFrame) -> pd.DataFrame:
    """Resume as leituras de OCR de cada passagem (melhor quadro, concordâncias etc.)."""
    linhas = []
    for fr, tr in zip(df["ocr_frente"], df["ocr_tras"]):
        f, t = parse(fr), parse(tr)
        todos = f + t
        melhor = max(todos, key=lambda x: x[1]) if todos else ("", 0.0)
        bf = max(f, key=lambda x: x[1]) if f else ("", 0.0)
        bt = max(t, key=lambda x: x[1]) if t else ("", 0.0)
        concord = np.mean([p == melhor[0] for p, _ in todos]) if todos else 0.0
        linhas.append((melhor[0], melhor[1], bf[1], bt[1], int(bf[0] == bt[0] and bf[0] != ""),
                       concord, len(todos), len(f) > 0))
    return pd.DataFrame(linhas, index=df.index, columns=[
        "ocr_melhor", "conf_melhor", "conf_frente", "conf_tras", "frente_tras_iguais",
        "frac_quadros_concordam", "n_quadros", "tem_frente"])


def montar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Candidato a cobrança (tag se lida, senão melhor OCR) + variáveis para o modelo."""
    r = leituras(df)
    out = pd.concat([df, r], axis=1)
    out["candidato"] = np.where(out["tag_lida"], out["placa_tag"], out["ocr_melhor"])
    out["fonte"] = np.where(out["tag_lida"], "tag", "ocr")
    out["correto"] = (out["candidato"] == out["placa_real"]).astype(int)

    out["dist_tag_ocr"] = [hamming(a, b) if t else -1
                           for a, b, t in zip(out["placa_tag"], out["ocr_melhor"], out["tag_lida"])]
    cad = [cadastro(c, pr, cr) for c, pr, cr in
           zip(out["candidato"], out["placa_real"], out["categoria_real"])]
    out["placa_existe"] = [int(e) for e, _ in cad]
    cat_cad = np.array([c for _, c in cad])
    out["cat_cadastro_igual_laser"] = (cat_cad == out["categoria_laser"].to_numpy()).astype(int)
    out["cat_tag_igual_laser"] = np.where(out["tag_lida"],
                                          (out["categoria_tag"] == out["categoria_laser"]).astype(int), -1)
    out["formato_valido"] = out["candidato"].str.match(MERCOSUL).astype(int)
    out["rssi_dbm"] = out["rssi_dbm"].fillna(-100)
    out["moto"] = (out["categoria_laser"] == 9).astype(int)
    out["tag_lida_i"] = out["tag_lida"].astype(int)
    return out


FEATURES = ["tag_lida_i", "rssi_dbm", "conf_melhor", "conf_frente", "conf_tras",
            "frente_tras_iguais", "frac_quadros_concordam", "n_quadros", "dist_tag_ocr",
            "placa_existe", "cat_cadastro_igual_laser", "cat_tag_igual_laser", "formato_valido",
            "velocidade_kmh", "noite", "chuva", "faixa", "moto"]

NOMES = {
    "tag_lida_i": "tag lida", "rssi_dbm": "força do sinal da tag", "conf_melhor": "confiança do OCR",
    "conf_frente": "confiança câmera frontal", "conf_tras": "confiança câmera traseira",
    "frente_tras_iguais": "frente = traseira", "frac_quadros_concordam": "quadros que concordam",
    "n_quadros": "nº de quadros", "dist_tag_ocr": "distância tag × OCR",
    "placa_existe": "placa existe no cadastro", "cat_cadastro_igual_laser": "categoria cadastro = laser",
    "cat_tag_igual_laser": "categoria tag = laser", "formato_valido": "formato Mercosul válido",
    "velocidade_kmh": "velocidade", "noite": "noite", "chuva": "chuva", "faixa": "faixa", "moto": "moto",
}
