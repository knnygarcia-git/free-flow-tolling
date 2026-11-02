"""
Simulador visual de um pórtico Free Flow: detecção, rastreamento, leitura de placa (OCR),
leitura de tag RFID, classificação e geração de dados, num painel no estilo de dashboard
de visão computacional.

A CENA É SINTÉTICA: os veículos, as placas e as leituras são gerados por um modelo estocástico.
O que é real é a lógica do pipeline, igual à de um pórtico:

    câmera -> detecção -> rastreamento (ID) -> linha de disparo -> OCR da placa
           -> antena RFID lê a tag -> fusão tag x placa -> categoria -> evento de cobrança

Cada passagem vira um evento com o tamanho em bytes do pacote de evidência (o mesmo modelo de
src/volume_dados.py). Os eventos vão para reports/tabelas/eventos_simulados.csv.

Uso:
    python src/simulador_portico.py              # 30 s de cena, gera MP4 + GIF + PNG
    python src/simulador_portico.py --segundos 60 --seed 7
"""
from __future__ import annotations

import argparse
import random
import string
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from volume_dados import TRANSACAO_BYTES, PacoteEvidencia

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tabelas"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- layout
W, H = 1280, 820
SCENE_W, SCENE_H = 900, 540
PANEL_X = SCENE_W + 12
FPS = 15

# ---------------------------------------------------------------- câmera (pinhole simples)
HORIZON_Y = 95           # linha do horizonte na imagem
CAM_H = 6.0              # altura da câmera no pórtico (m)
FOCAL = 900.0            # distância focal em pixels (lente tele, típica de ANPR)
CX = SCENE_W * 0.47      # centro óptico horizontal
LANE_W = 3.6             # largura da faixa (m)
LANES_X = [-LANE_W / 2, LANE_W / 2]    # centro de cada faixa (m)
SHOULDER_X = LANE_W + 1.5
Z_MAX, Z_MIN = 120.0, 11.0
# Posições na pista (distância z até a câmera que filma a cena, em metros)
Z_GANTRY = 24.0          # o pórtico: estrutura sobre a pista; o laser varre exatamente aqui
Z_TRIGGER = 34.0         # linha de disparo: a câmera ANPR fotografa a placa 10 m antes do pórtico
Z_RFID = (24.0, 40.0)    # zona de leitura da antena RFID: 0 a 16 m antes do pórtico
BEAM_H = 6.6             # altura da viga do pórtico (m)
POSTE_X = (-LANE_W - 1.4, SHOULDER_X + 0.8)
# cores dos sensores (BGR) - as mesmas na cena, na vista lateral e nas legendas
COR_ANPR = (60, 60, 235)
COR_RFID = (230, 160, 40)
COR_LASER = (40, 210, 250)
COR_CTX = (200, 120, 200)

# ---------------------------------------------------------------- cores (BGR)
AMARELO = (40, 200, 240)
BRANCO = (255, 255, 255)
VERDE = (90, 200, 90)
VERMELHO = (70, 70, 230)
LARANJA_BGR = (58, 131, 232)
AZUL_BGR = (219, 111, 42)

CATEGORIAS = {
    # nome: (largura m, altura m, prob, cat ANTT, eixos, faixa de velocidade km/h)
    "carro":      (1.80, 1.45, 0.62, 1, 2, (85, 115)),
    "suv":        (1.95, 1.75, 0.16, 1, 2, (85, 112)),
    "caminhao2":  (2.50, 3.10, 0.09, 2, 2, (70, 88)),
    "caminhao5":  (2.60, 3.90, 0.08, 7, 5, (65, 82)),
    "onibus":     (2.55, 3.40, 0.03, 4, 3, (75, 95)),
    "moto":       (0.80, 1.40, 0.02, 9, 2, (85, 115)),
}
CORES_VEIC = [(200, 200, 205), (40, 40, 45), (180, 60, 50), (60, 60, 160), (235, 235, 235),
              (120, 125, 130), (30, 90, 160), (60, 130, 70)]

P_TAG = 0.62             # fração de veículos com tag (hipótese)
P_ERRO_OCR = 0.06        # chance de um caractere ser lido errado
CONFUSOES = {"O": "0", "0": "O", "B": "8", "8": "B", "I": "1", "1": "I", "S": "5", "5": "S",
             "Z": "2", "2": "Z", "D": "0", "G": "6"}


def fonte(tam: int, negrito: bool = False):
    nomes = (["DejaVuSans-Bold.ttf", "arialbd.ttf", "LiberationSans-Bold.ttf"] if negrito
             else ["DejaVuSans.ttf", "arial.ttf", "LiberationSans-Regular.ttf"])
    pastas = ["", "/usr/share/fonts/truetype/dejavu/", "/usr/share/fonts/truetype/liberation/",
              "C:/Windows/Fonts/", "/Library/Fonts/", "/System/Library/Fonts/Supplemental/"]
    for n in nomes:
        for p in pastas:
            try:
                return ImageFont.truetype(p + n, tam)
            except OSError:
                continue
    return ImageFont.load_default(tam)


F_SM, F_MD, F_LG, F_XL = fonte(12), fonte(14), fonte(18, True), fonte(30, True)
F_BOX, F_MONO = fonte(13, True), fonte(12)


def projetar(x: float, z: float, h: float = 0.0) -> tuple[float, float]:
    """Mundo (x lateral, z distância, h altura) -> pixel."""
    return CX + FOCAL * x / z, HORIZON_Y + FOCAL * (CAM_H - h) / z


def ip(pt) -> tuple[int, int]:
    return int(round(pt[0])), int(round(pt[1]))


def placa_mercosul(rng: random.Random) -> str:
    L = string.ascii_uppercase
    return (rng.choice(L) + rng.choice(L) + rng.choice(L) + rng.choice(string.digits)
            + rng.choice(L) + rng.choice(string.digits) + rng.choice(string.digits))


@dataclass
class Veiculo:
    vid: int
    tipo: str
    faixa: int
    z: float
    v: float                       # m/s
    placa: str
    tem_tag: bool
    cor: tuple
    trilha: deque = field(default_factory=lambda: deque(maxlen=9))
    ocr: str | None = None
    conf: float | None = None
    tag_lida: bool = False
    tag_id: str | None = None
    decisao: str | None = None      # TAG / OCR / REVISAO
    evento_emitido: bool = False

    @property
    def dims(self):
        return CATEGORIAS[self.tipo][:2]

    @property
    def cat(self):
        return CATEGORIAS[self.tipo][3]

    @property
    def eixos(self):
        return CATEGORIAS[self.tipo][4]

    def caixa(self):
        w, h = self.dims
        x = LANES_X[self.faixa]
        xl, yb = projetar(x - w / 2, self.z)
        xr, _ = projetar(x + w / 2, self.z)
        esc = FOCAL / self.z
        return int(xl), int(yb - h * esc), int(xr), int(yb)


class Simulador:
    def __init__(self, segundos: int, seed: int):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.n_frames = segundos * FPS
        self.dt = 1 / FPS
        self.veiculos: list[Veiculo] = []
        self.prox_id = 101
        self.eventos: list[dict] = []
        self.ev = PacoteEvidencia()
        self.bytes_total = 0.0
        self.serie_fluxo: list[float] = []      # veículos/min (janela móvel de 10 s)
        self.serie_mb: list[float] = []
        self.calor = np.zeros((SCENE_H // 6, SCENE_W // 6), np.float32)
        self.intensidades: list[str] = []
        # Perfil de demanda ao longo da cena (veículos/s nas duas faixas), em fases
        t = np.linspace(0, 1, self.n_frames)
        self.taxa = 0.3 + 1.3 * np.clip(np.sin(np.pi * t * 1.1) ** 2, 0, 1)
        self.fundo = None
        self.ultimo_recorte = None
        self.ultima_leitura = None
        self.estado = "Baixo"
        self.portico = None
        self.atividade = {}          # (sensor, faixa) -> instante da última leitura
        self.t = 0.0
        self.contagem_sensor = {"anpr": 0, "rfid": 0, "laser": 0}

    # ------------------------------------------------------------ dinâmica
    def gerar_chegadas(self, f: int):
        if self.rng.random() > self.taxa[f] * self.dt:
            return
        faixa = self.rng.choice([0, 1])
        ultimos = [v for v in self.veiculos if v.faixa == faixa]
        if ultimos and max(v.z for v in ultimos) > Z_MAX - 20:
            return
        tipos, pesos = zip(*[(k, c[2]) for k, c in CATEGORIAS.items()])
        tipo = self.rng.choices(tipos, pesos)[0]
        if faixa == 1 and tipo in ("caminhao2", "caminhao5", "onibus") and self.rng.random() < .7:
            faixa = 0   # pesados preferem a faixa da direita do motorista (esquerda da imagem)
        vmin, vmax = CATEGORIAS[tipo][5]
        self.veiculos.append(Veiculo(
            vid=self.prox_id, tipo=tipo, faixa=faixa, z=Z_MAX,
            v=self.rng.uniform(vmin, vmax) / 3.6, placa=placa_mercosul(self.rng),
            tem_tag=self.rng.random() < P_TAG, cor=self.rng.choice(CORES_VEIC)))
        self.prox_id += 1

    def ler_ocr(self, v: Veiculo):
        """Simula o OCR: confiança cai com placa pequena/suja; às vezes troca caractere."""
        lida = list(v.placa)
        erros = 0
        for i, c in enumerate(lida):
            if c in CONFUSOES and self.rng.random() < P_ERRO_OCR:
                lida[i] = CONFUSOES[c]
                erros += 1
        base = self.rng.uniform(0.90, 0.995) if v.tipo != "moto" else self.rng.uniform(.6, .9)
        v.ocr = "".join(lida)
        v.conf = max(0.35, base - 0.18 * erros)

    def fundir(self, v: Veiculo):
        """Regra de decisão: tag válida manda; sem tag, OCR com confiança alta; senão, revisão."""
        if v.tag_lida:
            v.decisao = "TAG"
        elif v.conf is not None and v.conf >= 0.90:
            v.decisao = "OCR"
        else:
            v.decisao = "REVISAO"

    def emitir_evento(self, v: Veiculo, t: float):
        tamanho = self.ev.total + TRANSACAO_BYTES
        self.bytes_total += tamanho
        self.eventos.append({
            "t_s": round(t, 2), "track_id": v.vid, "faixa": v.faixa + 1, "tipo": v.tipo,
            "categoria_antt": v.cat, "eixos": v.eixos, "velocidade_kmh": round(v.v * 3.6, 1),
            "placa_real": v.placa, "placa_ocr": v.ocr, "conf_ocr": round(v.conf, 3),
            "ocr_correto": v.ocr == v.placa, "tem_tag": v.tem_tag, "tag_lida": v.tag_lida,
            "tag_id": v.tag_id, "decisao": v.decisao, "bytes_evento": int(tamanho),
        })
        v.evento_emitido = True

    def passo(self, f: int):
        t = f * self.dt
        self.t = t
        self.gerar_chegadas(f)
        for faixa in (0, 1):
            fila = sorted([v for v in self.veiculos if v.faixa == faixa], key=lambda v: v.z)
            for i, v in enumerate(fila):
                vel = v.v
                if i > 0:   # não atravessa o veículo da frente
                    gap = v.z - fila[i - 1].z
                    if gap < 18:
                        vel = min(vel, fila[i - 1].v)
                v.z -= vel * self.dt
                x1, y1, x2, y2 = v.caixa()
                v.trilha.append(((x1 + x2) // 2, y2))
                if Z_RFID[0] <= v.z <= Z_RFID[1] and v.tem_tag and not v.tag_lida:
                    if self.rng.random() < 0.35:        # várias tentativas de leitura na zona
                        v.tag_lida = True
                        self.atividade[("rfid", v.faixa)] = t
                        self.contagem_sensor["rfid"] += 1
                        v.tag_id = "E2801160" + "".join(self.rng.choices("0123456789ABCDEF", k=8))
                if v.ocr is None and v.z <= Z_TRIGGER:
                    self.ler_ocr(v)
                    self.ultima_leitura = v
                    self.atividade[("anpr", v.faixa)] = t
                    self.contagem_sensor["anpr"] += 1
                if v.ocr is not None and not v.evento_emitido and v.z <= Z_GANTRY:
                    # passou sob o pórtico: o laser mede o perfil e fecha a passagem
                    self.atividade[("laser", v.faixa)] = t
                    self.contagem_sensor["laser"] += 1
                    self.fundir(v)
                    self.emitir_evento(v, t)
                a, b = max(0, x1 // 6), max(0, y1 // 6)
                self.calor[b:max(b, y2 // 6), a:max(a, x2 // 6)] += v.v * 0.08
        self.veiculos = [v for v in self.veiculos if v.z > Z_MIN]
        self.calor *= 0.93

        janela = [e for e in self.eventos if e["t_s"] > t - 10]
        fluxo = len(janela) * 6.0
        self.serie_fluxo.append(fluxo)
        self.serie_mb.append(sum(e["bytes_evento"] for e in janela) * 6 / 1e6)
        # histerese: só troca de estado se passar do limiar com folga
        limites = {"Baixo": (None, 25), "Moderado": (15, 42), "Alto": (34, None)}
        lo, hi = limites[self.estado]
        if hi is not None and fluxo > hi:
            self.estado = "Moderado" if self.estado == "Baixo" else "Alto"
        elif lo is not None and fluxo < lo:
            self.estado = "Moderado" if self.estado == "Alto" else "Baixo"
        self.intensidades.append(self.estado)

    # ------------------------------------------------------------ desenho
    def desenhar_estrada(self, img):
        if self.fundo is None:
            self.fundo = self._fundo_estatico()
        img[:] = self.fundo
        return img

    def _fundo_estatico(self):
        """Pórtico e câmera são fixos: o fundo é desenhado uma vez só."""
        img = np.zeros((SCENE_H, SCENE_W, 3), np.uint8)
        img[:HORIZON_Y] = (205, 175, 140)                     # céu
        img[HORIZON_Y:] = (70, 120, 80)                       # vegetação
        ruido = (self.np_rng.random((SCENE_H - HORIZON_Y, SCENE_W)) * 14).astype(np.int16)
        ruido = cv2.GaussianBlur(ruido.astype(np.float32), (5, 5), 0).astype(np.int16)
        img[HORIZON_Y:, :, 1] = np.clip(img[HORIZON_Y:, :, 1].astype(np.int16) + ruido - 7, 0, 255)
        # asfalto (inclui acostamento à direita)
        xl0, _ = projetar(-LANE_W - 0.4, Z_MAX)
        xr0, _ = projetar(SHOULDER_X, Z_MAX)
        xl1, yb = projetar(-LANE_W - 0.4, Z_MIN)
        xr1, _ = projetar(SHOULDER_X, Z_MIN)
        pts = np.array([[xl0, HORIZON_Y + 1], [xr0, HORIZON_Y + 1], [xr1, yb], [xl1, yb]], np.int32)
        cv2.fillPoly(img, [pts], (78, 78, 82))
        # canteiro central (esquerda)
        med = np.array([[0, yb], [xl1, yb], [xl0, HORIZON_Y + 1], [0, HORIZON_Y + 30]], np.int32)
        cv2.fillPoly(img, [med], (60, 105, 70))
        # linhas de bordo contínuas
        for x in (-LANE_W, LANE_W):
            p0, p1 = projetar(x, Z_MAX), projetar(x, Z_MIN)
            cv2.line(img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (235, 235, 235), 2,
                     cv2.LINE_AA)
        # linha tracejada central
        z = Z_MIN
        while z < Z_MAX:
            a, b = projetar(0, z), projetar(0, min(z + 4, Z_MAX))
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (235, 235, 235),
                     max(1, int(14 / z * 3)), cv2.LINE_AA)
            z += 12
        # --- áreas cobertas por cada sensor, desenhadas no asfalto e como "feixes" saindo do pórtico
        sobre = img.copy()
        for lx in LANES_X:
            # antena RFID: lóbulo que cobre de 0 a 16 m antes do pórtico
            q = [projetar(lx - 1.7, Z_RFID[1]), projetar(lx + 1.7, Z_RFID[1]),
                 projetar(lx + 1.7, Z_RFID[0]), projetar(lx - 1.7, Z_RFID[0])]
            cv2.fillPoly(sobre, [np.array(q, np.int32)], COR_RFID)
        cv2.addWeighted(sobre, 0.25, img, 0.75, 0, img)
        feixes = img.copy()
        for lx in LANES_X:
            ant = projetar(lx, Z_GANTRY, BEAM_H - 0.5)
            q = [ant, projetar(lx - 1.7, Z_RFID[1]), projetar(lx + 1.7, Z_RFID[1])]
            cv2.fillPoly(feixes, [np.array(q, np.int32)], COR_RFID)
            cam = projetar(lx - 0.7, Z_GANTRY, BEAM_H - 0.4)
            q = [cam, projetar(lx - 1.6, Z_TRIGGER + 1.2), projetar(lx + 1.6, Z_TRIGGER + 1.2),
                 projetar(lx + 1.6, Z_TRIGGER - 1.2), projetar(lx - 1.6, Z_TRIGGER - 1.2)]
            cv2.fillPoly(feixes, [cv2.convexHull(np.array(q, np.int32))], COR_ANPR)
        cv2.addWeighted(feixes, 0.16, img, 0.84, 0, img)
        # linha de disparo da câmera ANPR (10 m antes do pórtico)
        a, b = projetar(-LANE_W, Z_TRIGGER), projetar(LANE_W, Z_TRIGGER)
        cv2.line(img, ip(a), ip(b), COR_ANPR, 3, cv2.LINE_AA)
        # linha do laser, exatamente sob o pórtico
        a, b = projetar(-LANE_W, Z_GANTRY), projetar(LANE_W, Z_GANTRY)
        cv2.line(img, ip(a), ip(b), COR_LASER, 3, cv2.LINE_AA)
        return img

    def _camada_portico(self):
        """Estrutura do pórtico com os sensores, como camada (imagem + máscara) que fica na
        frente dos veículos que ainda não passaram e atrás dos que já passaram."""
        img = np.zeros((SCENE_H, SCENE_W, 3), np.uint8)
        z = Z_GANTRY
        aco, aco_esc = (150, 150, 155), (95, 95, 100)
        # postes
        for x in POSTE_X:
            b0, t0 = projetar(x - 0.25, z, 0), projetar(x + 0.25, z, BEAM_H + 0.6)
            cv2.rectangle(img, ip(t0), ip(b0), aco, -1)
            cv2.rectangle(img, ip(t0), ip(b0), aco_esc, 1)
            sapata = (projetar(x - 0.6, z, 0), projetar(x + 0.6, z, 0.4))
            cv2.rectangle(img, ip((sapata[0][0], sapata[1][1])), ip((sapata[1][0], sapata[0][1])),
                          (120, 120, 125), -1)
        # viga treliçada
        xa, xb = POSTE_X[0] - 0.25, POSTE_X[1] + 0.25
        topo, base = BEAM_H + 0.6, BEAM_H
        cv2.line(img, ip(projetar(xa, z, topo)), ip(projetar(xb, z, topo)), aco, 3)
        cv2.line(img, ip(projetar(xa, z, base)), ip(projetar(xb, z, base)), aco, 3)
        xs = np.arange(xa, xb, 0.6)
        for i, x in enumerate(xs[:-1]):
            h1, h2 = (base, topo) if i % 2 == 0 else (topo, base)
            cv2.line(img, ip(projetar(x, z, h1)), ip(projetar(xs[i + 1], z, h2)), aco_esc, 1,
                     cv2.LINE_AA)
        # sensores pendurados na viga, um conjunto por faixa
        for lx in LANES_X:
            # câmera ANPR (corpo + lente) e iluminador infravermelho
            c0, c1 = projetar(lx - 1.0, z, base), projetar(lx - 0.4, z, base - 0.35)
            cv2.rectangle(img, ip(c0), ip(c1), (40, 40, 45), -1)
            cv2.rectangle(img, ip(c0), ip(c1), COR_ANPR, 2)
            cv2.circle(img, ip(projetar(lx - 0.7, z, base - 0.18)), 3, (200, 200, 210), -1)
            # antena RFID (painel plano)
            a0, a1 = projetar(lx - 0.25, z, base), projetar(lx + 0.25, z, base - 0.55)
            cv2.rectangle(img, ip(a0), ip(a1), (235, 235, 235), -1)
            cv2.rectangle(img, ip(a0), ip(a1), COR_RFID, 2)
            # scanner laser
            l0, l1 = projetar(lx + 0.45, z, base), projetar(lx + 0.85, z, base - 0.25)
            cv2.rectangle(img, ip(l0), ip(l1), (30, 30, 30), -1)
            cv2.rectangle(img, ip(l0), ip(l1), COR_LASER, 2)
        # câmera de contexto no topo do poste do acostamento
        k0, k1 = projetar(POSTE_X[1] - 0.9, z, BEAM_H + 0.9), projetar(POSTE_X[1] - 0.2, z, BEAM_H + 0.5)
        cv2.rectangle(img, ip(k0), ip(k1), (40, 40, 45), -1)
        cv2.rectangle(img, ip(k0), ip(k1), COR_CTX, 2)
        mascara = img.any(axis=2)
        return img, mascara

    def posicoes_sensores(self) -> dict:
        """Pixel de cada sensor (para rótulos e luzes de atividade)."""
        z, base = Z_GANTRY, BEAM_H
        pos = {}
        for i, lx in enumerate(LANES_X):
            pos[("anpr", i)] = ip(projetar(lx - 0.7, z, base - 0.18))
            pos[("rfid", i)] = ip(projetar(lx, z, base - 0.27))
            pos[("laser", i)] = ip(projetar(lx + 0.65, z, base - 0.12))
        pos[("ctx", 0)] = ip(projetar(POSTE_X[1] - 0.55, z, BEAM_H + 0.7))
        return pos

    def desenhar_atividade(self, img):
        """Acende o sensor que acabou de ler: flash IR, antena e laser."""
        pos = self.posicoes_sensores()
        for (sensor, faixa), t0 in self.atividade.items():
            dt = self.t - t0
            if dt < 0 or dt > 0.45:
                continue
            forca = 1 - dt / 0.45
            x, y = pos[(sensor, faixa)]
            cor = {"anpr": (255, 255, 255), "rfid": (80, 230, 80), "laser": COR_LASER}[sensor]
            sobre = img.copy()
            cv2.circle(sobre, (x, y), int(8 + 14 * forca), cor, -1, cv2.LINE_AA)
            cv2.addWeighted(sobre, 0.6 * forca, img, 1 - 0.6 * forca, 0, img)
            if sensor == "laser":
                lx = LANES_X[faixa]
                a, b = projetar(lx - 1.8, Z_GANTRY), projetar(lx + 1.8, Z_GANTRY)
                leque = img.copy()
                cv2.fillPoly(leque, [np.array([(x, y), ip(a), ip(b)], np.int32)], COR_LASER)
                cv2.addWeighted(leque, 0.35 * forca, img, 1 - 0.35 * forca, 0, img)

    def desenhar_veiculo(self, img, v: Veiculo):
        x1, y1, x2, y2 = v.caixa()
        w, h = x2 - x1, y2 - y1
        if w < 3:
            return
        sombra = np.array([[x1 - w * .05, y2], [x2 + w * .05, y2],
                           [x2, y2 - h * .06], [x1, y2 - h * .06]], np.int32)
        cv2.fillPoly(img, [sombra], (40, 40, 40))
        pesado = v.tipo in ("caminhao2", "caminhao5", "onibus")
        cv2.rectangle(img, (x1, y1), (x2, y2), v.cor, -1)
        # para-brisa
        top = y1 + int(h * (0.08 if pesado else 0.12))
        bot = y1 + int(h * (0.42 if pesado else 0.48))
        cv2.rectangle(img, (x1 + int(w * .1), top), (x2 - int(w * .1), bot), (70, 55, 45), -1)
        # faróis
        fy = y1 + int(h * .72)
        r = max(1, int(w * .06))
        cv2.circle(img, (x1 + int(w * .15), fy), r, (210, 240, 255), -1)
        cv2.circle(img, (x2 - int(w * .15), fy), r, (210, 240, 255), -1)
        # placa Mercosul (branca com faixa azul)
        pw, ph = max(4, int(w * (.38 if v.tipo != "moto" else .7))), max(2, int(w * .12))
        px, py = (x1 + x2) // 2 - pw // 2, y1 + int(h * .80)
        cv2.rectangle(img, (px, py), (px + pw, py + ph), (245, 245, 245), -1)
        cv2.rectangle(img, (px, py), (px + pw, py + max(1, ph // 4)), (150, 60, 20), -1)
        if pw > 40:
            cv2.putText(img, v.placa, (px + 2, py + ph - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        pw / 150, (20, 20, 20), 1, cv2.LINE_AA)

    def desenhar_anotacoes(self, img, draw_ops):
        for v in sorted(self.veiculos, key=lambda v: -v.z):
            x1, y1, x2, y2 = v.caixa()
            if x2 - x1 < 8:
                continue
            cor = AMARELO
            if v.decisao == "TAG" or (v.decisao is None and v.tag_lida):
                cor = VERDE
            elif v.decisao == "REVISAO":
                cor = VERMELHO
            cv2.rectangle(img, (x1, y1), (x2, y2), cor, 2)
            pts = list(v.trilha)
            for i in range(1, len(pts)):
                cv2.line(img, pts[i - 1], pts[i], AMARELO, 1, cv2.LINE_AA)
            rotulo = f"#{v.vid}"
            if v.ocr:
                rotulo += f" {v.ocr} {v.conf*100:.0f}%"
            draw_ops.append(("box", (x1, y1), rotulo, cor))
            if v.z < 36:
                sub = f"Cat {v.cat} · {v.eixos} eixos · {v.v*3.6:.0f} km/h"
                sub += "  TAG ✓" if v.tag_lida else ""
                draw_ops.append(("sub", (x1, y2 + 2), sub, cor))

    def recorte_placa(self, v: Veiculo) -> np.ndarray:
        """Recorte da placa como a câmera ANPR infravermelha entrega ao OCR: monocromático,
        com ruído e desfoque proporcionais à distância/velocidade."""
        pw, ph = 170, 48
        img = Image.new("L", (pw, ph), 235)
        d = ImageDraw.Draw(img)
        d.rectangle((0, 0, pw - 1, 11), fill=70)                  # faixa azul (escura no IR)
        d.text((pw // 2 - 18, 0), "BRASIL", font=fonte(9, True), fill=230)
        d.text((10, 13), v.placa, font=fonte(28, True), fill=25)
        d.rectangle((0, 0, pw - 1, ph - 1), outline=40, width=2)
        a = np.array(img, np.float32)
        borrao = 1 + int((v.v * 3.6) / 60)
        a = cv2.GaussianBlur(a, (borrao * 2 + 1, 1), 0)           # arrasto de movimento
        a += self.np_rng.normal(0, 10 + 25 * (1 - (v.conf or 1)), a.shape)
        a = np.clip(a, 0, 255).astype(np.uint8)
        return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)

    def rotulos_sensores(self, d: ImageDraw.ImageDraw):
        """Etiquetas numeradas nos sensores do pórtico + legenda."""
        pos = self.posicoes_sensores()
        marcas = [("anpr", "1", COR_ANPR), ("rfid", "2", COR_RFID), ("laser", "3", COR_LASER)]
        for faixa in (0, 1):
            for k, num, cor in marcas:
                x, y = pos[(k, faixa)]
                self._etiqueta(d, x, y - 26 - (int(num) - 1) % 2 * 0, num, cor)
        x, y = pos[("ctx", 0)]
        self._etiqueta(d, x, y - 24, "4", COR_CTX)
        # legenda
        lx, ly = 12, 8
        d.rounded_rectangle((lx - 4, ly - 4, lx + 600, ly + 40), 6, fill=(20, 24, 28))
        legenda = [("1", "Câmera ANPR + flash IR", COR_ANPR), ("2", "Antena RFID", COR_RFID),
                   ("3", "Laser 3D (eixos)", COR_LASER), ("4", "Câmera de contexto", COR_CTX)]
        cx = lx
        for num, nome, cor in legenda:
            self._etiqueta(d, cx + 8, ly + 1, num, cor, centro=False)
            d.text((cx + 22, ly + 1), nome, font=F_SM, fill=(235, 235, 235))
            cx += 22 + int(d.textlength(nome, font=F_SM)) + 16
        d.text((lx, ly + 20), "Caixa:  verde = pela tag   amarela = pela placa   vermelha = revisão",
               font=F_SM, fill=(200, 200, 200))
        # rótulos das linhas no asfalto
        for z, txt, cor in [(Z_TRIGGER, "disparo da câmera (10 m antes)", COR_ANPR),
                            (Z_GANTRY, "laser sob o pórtico", COR_LASER)]:
            x, y = projetar(LANE_W + 0.2, z)
            rgb = (cor[2], cor[1], cor[0])
            bb = d.textbbox((x + 6, y - 9), txt, font=F_SM)
            d.rectangle((bb[0] - 3, bb[1] - 2, bb[2] + 3, bb[3] + 2), fill=(20, 24, 28))
            d.text((x + 6, y - 9), txt, font=F_SM, fill=rgb)
        x, y = projetar(-LANE_W - 0.3, (Z_RFID[0] + Z_RFID[1]) / 2)
        txt = "zona RFID (0-16 m)"
        bb = d.textbbox((0, 0), txt, font=F_SM)
        d.rectangle((x - bb[2] - 10, y - 10, x - 2, y + 8), fill=(20, 24, 28))
        d.text((x - bb[2] - 6, y - 9), txt, font=F_SM, fill=(COR_RFID[2], COR_RFID[1], COR_RFID[0]))

    def textos_vista_lateral(self, d):
        x0, y0, w, h = 12, SCENE_H + 8, SCENE_W - 24, 132
        chao = y0 + h - 26
        d_min, d_max = -60.0, 15.0
        px = lambda dist: int(x0 + 30 + (dist - d_min) / (d_max - d_min) * (w - 60))
        d.text((x0 + 8, y0 + 4), "Vista lateral (esquema) · sentido do tráfego →", font=F_BOX,
               fill=(40, 40, 40))
        for dist in range(-60, 16, 10):
            rot = "pórtico" if dist == 0 else f"{dist:+d} m"
            bb = d.textbbox((0, 0), rot, font=F_SM)
            d.text((px(dist) - bb[2] // 2, chao + 6), rot, font=F_SM, fill=(90, 90, 90))
        notas = [(-16, -1, "2 antena RFID: 0-16 m antes", COR_RFID),
                 (-10, -2, "1 câmera ANPR: placa a 10 m", COR_ANPR),
                 (0, -3, "3 laser: eixos, sob o pórtico", COR_LASER)]
        cx = x0 + 8
        for dist, _, txt, cor in notas:
            d.text((cx, y0 + 24), "■ " + txt, font=F_SM,
                   fill=(int(cor[2] * .8), int(cor[1] * .8), int(cor[0] * .8)))
            cx += int(d.textlength("■ " + txt, font=F_SM)) + 22

    @staticmethod
    def _etiqueta(d, x, y, num, cor, centro=True):
        r = 8
        cx, cy = (x, y) if centro else (x, y + 7)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(cor[2], cor[1], cor[0]),
                  outline=(255, 255, 255), width=1)
        d.text((cx - 4, cy - 8), num, font=F_BOX, fill=(15, 15, 15))

    def desenhar_vista_lateral(self, canvas):
        """Esquema lateral: onde cada sensor está e que trecho da pista ele cobre."""
        x0, y0, w, h = 12, SCENE_H + 8, SCENE_W - 24, 132
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (236, 238, 240), -1)
        chao = y0 + h - 26
        # escala: de 60 m antes do pórtico até 15 m depois
        d_min, d_max = -60.0, 15.0
        def px(dist):   # dist < 0: antes do pórtico
            return int(x0 + 30 + (dist - d_min) / (d_max - d_min) * (w - 60))
        def py(alt):
            return int(chao - alt * 11)
        cv2.line(canvas, (x0 + 10, chao), (x0 + w - 10, chao), (90, 90, 95), 2)
        for dist in range(-60, 16, 10):
            cv2.line(canvas, (px(dist), chao), (px(dist), chao + 4), (120, 120, 120), 1)
        # zonas no chão
        rf0, rf1 = Z_GANTRY - Z_RFID[1], Z_GANTRY - Z_RFID[0]      # -16 .. 0
        cv2.rectangle(canvas, (px(rf0), chao - 3), (px(rf1), chao + 3), COR_RFID, -1)
        trig = Z_GANTRY - Z_TRIGGER                                  # -10
        gx = px(0)
        topo = py(BEAM_H)
        sobre = canvas.copy()
        cv2.fillPoly(sobre, [np.array([(gx, topo + 8), (px(rf0), chao), (px(rf1), chao)])],
                     COR_RFID)
        cv2.fillPoly(sobre, [np.array([(gx - 4, topo + 6), (px(trig - 1.5), chao),
                                       (px(trig + 1.5), chao)])], COR_ANPR)
        cv2.addWeighted(sobre, 0.25, canvas, 0.75, 0, canvas)
        cv2.line(canvas, (gx + 3, topo + 6), (gx + 3, chao), COR_LASER, 2)
        # pórtico
        cv2.rectangle(canvas, (gx - 3, topo - 6), (gx + 3, chao), (130, 130, 135), -1)
        cv2.rectangle(canvas, (gx - 10, topo - 8), (gx + 10, topo), (130, 130, 135), -1)
        cv2.circle(canvas, (gx - 6, topo + 5), 4, COR_ANPR, -1)
        cv2.circle(canvas, (gx, topo + 8), 4, COR_RFID, -1)
        cv2.circle(canvas, (gx + 6, topo + 5), 4, COR_LASER, -1)
        # veículos (vista lateral, faixa 0 em cima da faixa 1 levemente deslocada)
        for v in self.veiculos:
            dist = Z_GANTRY - v.z
            if not (d_min <= dist <= d_max):
                continue
            comp = {"carro": 4.5, "suv": 4.8, "caminhao2": 8, "caminhao5": 18, "onibus": 13,
                    "moto": 2}[v.tipo]
            alt = v.dims[1]
            xf = px(dist)
            xt = px(dist - comp)
            cor = VERDE if v.decisao == "TAG" or v.tag_lida else (
                VERMELHO if v.decisao == "REVISAO" else AMARELO)
            off = 0 if v.faixa == 0 else -3
            cv2.rectangle(canvas, (xt, py(alt) + off), (xf, chao - 2 + off), v.cor, -1)
            cv2.rectangle(canvas, (xt, py(alt) + off), (xf, chao - 2 + off), cor, 2)

    def render(self, f: int) -> np.ndarray:
        canvas = np.full((H, W, 3), 248, np.uint8)
        cena = np.zeros((SCENE_H, SCENE_W, 3), np.uint8)
        self.desenhar_estrada(cena)
        if self.portico is None:
            self.portico = self._camada_portico()
        # ordem de pintura: veículos antes do pórtico -> pórtico -> veículos que já passaram
        ordem = sorted(self.veiculos, key=lambda v: -v.z)
        for v in ordem:
            if v.z > Z_GANTRY:
                self.desenhar_veiculo(cena, v)
        cena[self.portico[1]] = self.portico[0][self.portico[1]]
        for v in ordem:
            if v.z <= Z_GANTRY:
                self.desenhar_veiculo(cena, v)
        self.desenhar_atividade(cena)
        limpo = cena.copy()
        ops: list = []
        self.desenhar_anotacoes(cena, ops)
        canvas[:SCENE_H, :SCENE_W] = cena

        # --- painel lateral: mapa de calor de movimento sobre a cena em cinza
        pw = W - PANEL_X - 12
        calor = cv2.GaussianBlur(cv2.resize(self.calor, (pw, 150)), (0, 0), 4)
        calor = np.clip(calor / 12.0 * 255, 0, 255).astype(np.uint8)
        mapa = cv2.applyColorMap(calor, cv2.COLORMAP_JET)
        fundo = cv2.cvtColor(cv2.cvtColor(cv2.resize(limpo, (pw, 150)), cv2.COLOR_BGR2GRAY),
                             cv2.COLOR_GRAY2BGR)
        fundo = cv2.addWeighted(fundo, 0.35, np.full_like(fundo, (90, 30, 30)), 0.65, 0)
        alfa = (calor.astype(np.float32) / 255)[..., None] ** 0.6
        canvas[0:150, PANEL_X:W - 12] = (mapa * alfa + fundo * (1 - alfa)).astype(np.uint8)
        # barra de escala
        grad = np.tile(np.linspace(0, 255, W - PANEL_X - 12).astype(np.uint8), (6, 1))
        canvas[156:162, PANEL_X:W - 12] = cv2.applyColorMap(grad, cv2.COLORMAP_JET)

        # --- recorte da última placa lida
        if self.ultima_leitura is not None and (
                self.ultimo_recorte is None or self.ultimo_recorte[1] is not self.ultima_leitura):
            self.ultimo_recorte = (self.recorte_placa(self.ultima_leitura), self.ultima_leitura)
        ultimo = None
        if self.ultimo_recorte is not None:
            rc, ultimo = self.ultimo_recorte
            canvas[400:448, PANEL_X:PANEL_X + 170] = rc

        # --- gráfico de fluxo
        gx0, gy0, gw, gh = PANEL_X + 34, 520, W - PANEL_X - 56, 150
        cv2.rectangle(canvas, (gx0, gy0), (gx0 + gw, gy0 + gh), (238, 238, 238), -1)
        for k in range(1, 4):
            yy = gy0 + gh - int(gh * k / 4)
            cv2.line(canvas, (gx0, yy), (gx0 + gw, yy), (225, 225, 225), 1)
        serie = self.serie_fluxo
        if len(serie) > 1:
            xs = np.linspace(gx0, gx0 + gw * len(serie) / self.n_frames, len(serie))
            ys = gy0 + gh - np.clip(np.array(serie) / 80, 0, 1) * gh
            pts = np.stack([xs, ys], 1).astype(np.int32)
            cv2.polylines(canvas, [pts], False, (110, 80, 40), 2, cv2.LINE_AA)
            cv2.circle(canvas, tuple(pts[-1]), 4, (110, 80, 40), -1)

        # --- linha do tempo de intensidade (rodapé)
        tx0, ty0, tw, th = 12, 778, SCENE_W - 24, 26
        cores_int = {"Baixo": (225, 225, 215), "Moderado": (210, 220, 200), "Alto": (180, 200, 240)}
        seg_ini = 0
        blocos = []
        for i in range(1, len(self.intensidades) + 1):
            if i == len(self.intensidades) or self.intensidades[i] != self.intensidades[seg_ini]:
                a = tx0 + int(tw * seg_ini / self.n_frames)
                b = tx0 + int(tw * i / self.n_frames)
                cv2.rectangle(canvas, (a, ty0), (b, ty0 + th), cores_int[self.intensidades[seg_ini]], -1)
                cv2.line(canvas, (b, ty0), (b, ty0 + th), BRANCO, 2)
                blocos.append((a, b, self.intensidades[seg_ini]))
                seg_ini = i
        cv2.rectangle(canvas, (tx0 + int(tw * len(self.intensidades) / self.n_frames), ty0),
                      (tx0 + tw, ty0 + th), (236, 236, 236), -1)
        xm = tx0 + int(tw * f / self.n_frames)
        cv2.line(canvas, (xm, ty0 - 4), (xm, ty0 + th + 4), (60, 50, 40), 2)

        self.desenhar_vista_lateral(canvas)

        # ---------------------------------------------------- textos (PIL, com acentos)
        pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for tipo, (x, y), txt, cor in ops:
            fnt = F_BOX if tipo == "box" else F_SM
            rgb = (cor[2], cor[1], cor[0])
            if tipo == "box":
                y = max(0, y - 17)
            bb = d.textbbox((x, y), txt, font=fnt)
            d.rectangle((bb[0] - 2, bb[1] - 1, bb[2] + 2, bb[3] + 1), fill=(20, 20, 20))
            d.text((x, y), txt, font=fnt, fill=rgb)
        t = f / FPS
        relogio = f"00:{int(t):02d}"
        bb = d.textbbox((0, 0), relogio, font=F_XL)
        d.rounded_rectangle((W - 12 - (bb[2] + 24), 14, W - 22, 60), 8, fill=(25, 25, 30))
        d.text((W - 22 - bb[2] - 12, 18), relogio, font=F_XL, fill=(255, 255, 255))
        d.text((PANEL_X, 166), "0 km/h", font=F_SM, fill=(90, 90, 90))
        d.text((W - 70, 166), "120 km/h", font=F_SM, fill=(90, 90, 90))
        d.text((14, SCENE_H - 22), "CENA SINTÉTICA · pórtico BR (2 faixas + acostamento)",
               font=F_SM, fill=(255, 255, 255))
        self.rotulos_sensores(d)
        self.textos_vista_lateral(d)

        intens = self.intensidades[-1]
        d.text((PANEL_X, 190), intens, font=fonte(26, True), fill=(30, 40, 60))
        d.text((PANEL_X, 224), "fluxo nos últimos 10 s", font=F_SM, fill=(110, 110, 110))
        d.line((PANEL_X, 244, W - 12, 244), fill=(225, 225, 225))
        col2 = PANEL_X + 180
        d.text((PANEL_X, 252), "Veículos no quadro", font=F_SM, fill=(90, 90, 90))
        d.text((col2, 252), "Fluxo (veíc/min)", font=F_SM, fill=(90, 90, 90))
        visiveis = sum(1 for v in self.veiculos if v.z < Z_MAX - 5)
        d.text((PANEL_X, 270), f"{visiveis}", font=fonte(26, True), fill=(20, 20, 20))
        d.text((col2, 270), f"{self.serie_fluxo[-1]:.0f}", font=fonte(26, True), fill=(20, 20, 20))
        d.text((PANEL_X, 310), "Dados gerados", font=F_SM, fill=(90, 90, 90))
        d.text((col2, 310), "Taxa (MB/min)", font=F_SM, fill=(90, 90, 90))
        d.text((PANEL_X, 326), f"{self.bytes_total/1e6:,.1f} MB", font=fonte(22, True),
               fill=(20, 20, 20))
        d.text((col2, 326), f"{self.serie_mb[-1]:.1f}", font=fonte(22, True), fill=(20, 20, 20))
        n = len(self.eventos)
        if n:
            tag = sum(e["decisao"] == "TAG" for e in self.eventos) / n
            ocr = sum(e["decisao"] == "OCR" for e in self.eventos) / n
            rev = 1 - tag - ocr
            d.text((PANEL_X, 360), f"Passagens: {n}   TAG {tag:.0%} · OCR {ocr:.0%} · revisão {rev:.0%}",
                   font=F_SM, fill=(60, 60, 60))
        d.text((PANEL_X, 382), "Último recorte de placa (IR)", font=F_SM, fill=(90, 90, 90))
        if ultimo is not None:
            d.text((PANEL_X + 180, 402), f"{ultimo.ocr}", font=fonte(18, True), fill=(20, 20, 20))
            d.text((PANEL_X + 180, 426), f"conf. {ultimo.conf*100:.0f}%", font=F_SM,
                   fill=(90, 90, 90))
        d.text((PANEL_X, 462), "Último evento enviado", font=F_SM, fill=(90, 90, 90))
        if self.eventos:
            e = self.eventos[-1]
            d.text((PANEL_X, 478), f"#{e['track_id']} {e['placa_ocr']} Cat{e['categoria_antt']} "
                   f"{e['decisao']} {e['bytes_evento']/1e3:.0f} KB", font=F_MONO, fill=(40, 40, 40))
        # eixos do gráfico
        for k, val in enumerate([0, 20, 40, 60, 80]):
            d.text((PANEL_X, gy0 + gh - int(gh * k / 4) - 7), f"{val}", font=F_SM,
                   fill=(110, 110, 110))
        d.text((PANEL_X, gy0 - 18), "veículos/min", font=F_SM, fill=(90, 90, 90))
        d.text((gx0, gy0 + gh + 4), "0 s", font=F_SM, fill=(110, 110, 110))
        d.text((gx0 + gw - 26, gy0 + gh + 4), f"{self.n_frames // FPS} s", font=F_SM,
               fill=(110, 110, 110))
        # estado dos sensores
        d.text((PANEL_X, 712), "Leituras por sensor nesta cena", font=F_SM, fill=(90, 90, 90))
        itens = [("Câmera ANPR", "anpr", COR_ANPR), ("Antena RFID", "rfid", COR_RFID),
                 ("Laser 3D", "laser", COR_LASER)]
        for i, (nome, k, cor) in enumerate(itens):
            y = 732 + i * 24
            ativo = any(s == k and 0 <= self.t - t0 < 0.45 for (s, _), t0 in self.atividade.items())
            rgb = (cor[2], cor[1], cor[0])
            d.rectangle((PANEL_X, y + 2, PANEL_X + 12, y + 14), fill=rgb if ativo else None,
                        outline=rgb, width=2)
            d.text((PANEL_X + 20, y), nome, font=F_MD, fill=(40, 40, 40))
            d.text((PANEL_X + 160, y), f"{self.contagem_sensor[k]:>3d} leituras", font=F_MONO,
                   fill=(60, 60, 60))
            if ativo:
                d.text((PANEL_X + 270, y), "lendo", font=F_BOX, fill=(30, 150, 60))
        # rodapé
        d.text((12, 690), "Pipeline:  câmera → detecção → rastreamento → disparo → OCR da placa "
               "→ leitura RFID → fusão → categoria → evento", font=F_MD, fill=(40, 40, 40))
        d.text((12, 714), "Cada passagem salva ~965 KB de evidência (2 fotos ANPR, recortes, "
               "foto de contexto, perfil do laser, JSON) e envia ~2 KB ao backoffice.",
               font=F_SM, fill=(90, 90, 90))
        d.text((12, 750), "Intensidade de tráfego estimada", font=F_LG, fill=(30, 30, 30))
        d.text((SCENE_W - 110, 752), f"{t:.1f} / {self.n_frames / FPS:.0f} s", font=F_SM,
               fill=(90, 90, 90))
        for a, b, nome in blocos:
            if b - a > 70:
                d.text((a + 8, ty0 + 5), nome, font=F_SM, fill=(40, 40, 40))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segundos", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sem-gif", action="store_true")
    args = ap.parse_args()

    sim = Simulador(args.segundos, args.seed)
    # aquece a cena por 6 s para já começar com tráfego na pista
    aquec = 6 * FPS
    sim_aq = sim
    for f in range(aquec):
        sim_aq.passo(0)
    sim.eventos.clear()
    sim.bytes_total = 0.0
    sim.serie_fluxo.clear()
    sim.serie_mb.clear()
    sim.intensidades.clear()
    sim.contagem_sensor = {k: 0 for k in sim.contagem_sensor}
    for v in sim.veiculos:
        v.evento_emitido = v.evento_emitido or v.z < Z_GANTRY

    frames, densidade = [], []
    for f in range(sim.n_frames):
        sim.passo(f)
        frames.append(sim.render(f))
        densidade.append(sum(1 for v in sim.veiculos if v.z < 45) + 0.1 * len(sim.veiculos))

    import imageio.v2 as imageio
    mp4 = FIG / "simulacao_portico.mp4"
    with imageio.get_writer(mp4, fps=FPS, codec="libx264", quality=8,
                            macro_block_size=8) as w:
        for fr in frames:
            w.append_data(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    # quadro de destaque: o mais "cheio" de veículos próximos na segunda metade da cena
    meio = len(frames) // 3
    destaque = meio + int(np.argmax(densidade[meio:]))
    cv2.imwrite(str(FIG / "simulacao_portico_frame.png"), frames[destaque])
    if not args.sem_gif:
        gif_frames = [Image.fromarray(cv2.cvtColor(cv2.resize(fr, (800, round(800 * H / W)),
                      interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
                      for fr in frames[::2]]
        gif_frames[0].save(FIG / "simulacao_portico.gif", save_all=True,
                           append_images=gif_frames[1:], duration=int(2000 / FPS), loop=0,
                           optimize=True)

    ev = pd.DataFrame(sim.eventos)
    ev.to_csv(TAB / "eventos_simulados.csv", index=False)
    print(f"Passagens simuladas: {len(ev)} em {args.segundos} s")
    if len(ev):
        print(ev.decisao.value_counts(normalize=True).round(3).to_string())
        print(f"OCR correto: {ev.ocr_correto.mean():.1%} | "
              f"erros de OCR capturados pela tag: "
              f"{((~ev.ocr_correto) & (ev.decisao == 'TAG')).sum()}")
        print(f"Dados gerados: {ev.bytes_evento.sum()/1e6:.1f} MB "
              f"({ev.bytes_evento.sum()/1e6/args.segundos*60:.1f} MB/min)")
    print(f"Vídeo: {mp4}")


if __name__ == "__main__":
    main()
