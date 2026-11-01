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
W, H = 1280, 720
SCENE_W, SCENE_H = 900, 540
PANEL_X = SCENE_W + 12
FPS = 15

# ---------------------------------------------------------------- câmera (pinhole simples)
HORIZON_Y = 60           # linha do horizonte na imagem
CAM_H = 6.0              # altura da câmera no pórtico (m)
FOCAL = 900.0            # distância focal em pixels (lente tele, típica de ANPR)
CX = SCENE_W * 0.47      # centro óptico horizontal
LANE_W = 3.6             # largura da faixa (m)
LANES_X = [-LANE_W / 2, LANE_W / 2]    # centro de cada faixa (m)
SHOULDER_X = LANE_W + 1.5
Z_MAX, Z_MIN = 120.0, 8.5
Z_TRIGGER = 19.0         # linha de disparo da câmera ANPR
Z_RFID = (13.0, 32.0)    # zona de leitura da antena RFID

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


def projetar(x: float, z: float) -> tuple[float, float]:
    """Mundo (x lateral, z distância, chão y=0) -> pixel."""
    return CX + FOCAL * x / z, HORIZON_Y + FOCAL * CAM_H / z


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
                        v.tag_id = "E2801160" + "".join(self.rng.choices("0123456789ABCDEF", k=8))
                if v.ocr is None and v.z <= Z_TRIGGER:
                    self.ler_ocr(v)
                    self.ultima_leitura = v
                if v.ocr is not None and not v.evento_emitido and v.z <= Z_RFID[0]:
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
        # zona RFID (faixa translúcida) e linha de disparo ANPR
        sobre = img.copy()
        q = [projetar(-LANE_W, Z_RFID[1]), projetar(LANE_W, Z_RFID[1]),
             projetar(LANE_W, Z_RFID[0]), projetar(-LANE_W, Z_RFID[0])]
        cv2.fillPoly(sobre, [np.array(q, np.int32)], (219, 160, 60))
        cv2.addWeighted(sobre, 0.22, img, 0.78, 0, img)
        a, b = projetar(-LANE_W, Z_TRIGGER), projetar(LANE_W, Z_TRIGGER)
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (60, 60, 240), 2, cv2.LINE_AA)
        return img

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

    def render(self, f: int) -> np.ndarray:
        canvas = np.full((H, W, 3), 248, np.uint8)
        cena = np.zeros((SCENE_H, SCENE_W, 3), np.uint8)
        self.desenhar_estrada(cena)
        for v in sorted(self.veiculos, key=lambda v: -v.z):
            self.desenhar_veiculo(cena, v)
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
        tx0, ty0, tw, th = 12, 640, SCENE_W - 24, 26
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
        # legenda das zonas
        d.text((14, 8), "■ zona da antena RFID", font=F_SM, fill=(60, 160, 219))
        d.text((170, 8), "━ linha de disparo ANPR", font=F_SM, fill=(240, 60, 60))
        d.text((340, 8), "□ verde: tag   □ amarelo: OCR   □ vermelho: revisão", font=F_SM,
               fill=(255, 255, 255))

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
        # rodapé
        d.text((12, 552), "Pipeline:  câmera → detecção → rastreamento → disparo → OCR da placa "
               "→ leitura RFID → fusão → categoria → evento", font=F_MD, fill=(40, 40, 40))
        d.text((12, 578), "Cada passagem salva ~965 KB de evidência (2 fotos ANPR, recortes, "
               "foto de contexto, perfil do laser, JSON) e envia ~2 KB ao backoffice.",
               font=F_SM, fill=(90, 90, 90))
        d.text((12, 616), "Intensidade de tráfego estimada", font=F_LG, fill=(30, 30, 30))
        d.text((SCENE_W - 110, 618), f"{t:.1f} / {self.n_frames / FPS:.0f} s", font=F_SM,
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
    for v in sim.veiculos:
        v.evento_emitido = v.evento_emitido or v.z < Z_RFID[0]

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
        gif_frames = [Image.fromarray(cv2.cvtColor(cv2.resize(fr, (800, 450),
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
