"""
Estilo visual único para todos os gráficos do projeto.

Fundo branco, fonte Poppins (em assets/fonts, licença OFL), grade discreta, título alinhado à
esquerda com uma linha de explicação embaixo. A paleta foi conferida com um validador de
daltonismo e contraste:

  VERDE  destaque, o resultado que importa (modelo, Paraná, "dado real")
  AZUL   segunda série
  AMBAR  terceira série, e tudo que é suposição ou simulação
  CINZA  contexto e linha de base (nunca usado sozinho para identificar uma série)
  ALERTA só para alarme e falha
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FONTES = ROOT / "assets" / "fonts"

VERDE, AZUL, AMBAR = "#0b8457", "#2563eb", "#d97706"
CINZA, CINZA_CLARO = "#a3aca8", "#e3e8e5"
ALERTA = "#c2410c"
TINTA, TINTA_2, TINTA_3 = "#101814", "#47524c", "#7c8781"
GRADE = "#eef1ef"
FUNDO = "#ffffff"


def aplicar() -> str:
    familia = "DejaVu Sans"
    for ttf in sorted(FONTES.glob("Poppins-*.ttf")):
        font_manager.fontManager.addfont(str(ttf))
        familia = "Poppins"
    mpl.rcParams.update({
        "font.family": familia,
        "font.size": 10,
        "font.weight": "regular",
        "text.color": TINTA,
        "figure.facecolor": FUNDO,
        "axes.facecolor": FUNDO,
        "axes.edgecolor": CINZA_CLARO,
        "axes.linewidth": 1,
        "axes.labelcolor": TINTA_3,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRADE,
        "grid.linewidth": 1,
        "grid.linestyle": "-",
        "xtick.color": TINTA_3,
        "ytick.color": TINTA_3,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": TINTA_2,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.facecolor": FUNDO,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.25,
        "axes.formatter.use_locale": False,
        "text.parse_math": False,          # "R$" não pode virar fórmula
    })
    return familia


def titulo(fig, texto: str, sub: str | None = None):
    """Título à esquerda, acima da área do gráfico, com uma linha de explicação embaixo.
    Fica fora da figura e entra na imagem pelo bbox 'tight' do savefig."""
    from matplotlib.transforms import offset_copy
    fig.canvas.draw()
    x0 = min(ax.get_tightbbox(fig.canvas.get_renderer()).x0 for ax in fig.axes) / fig.bbox.width
    y_sub = offset_copy(fig.transFigure, fig=fig, y=8, units="points")
    y_tit = offset_copy(fig.transFigure, fig=fig, y=26 if sub else 10, units="points")
    fig.text(x0, 1.0, texto, ha="left", va="bottom", fontsize=14, fontweight="semibold",
             color=TINTA, transform=y_tit)
    if sub:
        fig.text(x0, 1.0, sub, ha="left", va="bottom", fontsize=9.5, color=TINTA_3,
                 transform=y_sub)


def grade_x(ax):
    """Para barras horizontais: grade vertical em vez de horizontal."""
    ax.grid(False)
    ax.grid(True, axis="x")
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(CINZA_CLARO)


def br(n: float, casas: int = 0) -> str:
    """Número no formato brasileiro: 12.345,6"""
    s = f"{n:,.{casas}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def salvar(fig, caminho):
    fig.savefig(caminho)
    plt.close(fig)


aplicar()
