"""
Quanto dado um pórtico Free Flow gera por minuto?

Modelo paramétrico em quatro camadas, do sensor ao backoffice:

  A. Bruto no sensor   -> o que as câmeras enxergam (processado na borda e descartado)
  B. Vídeo de auditoria -> gravação contínua comprimida (H.265), quando a concessionária guarda
  C. Evidência por passagem -> fotos da placa, recortes, foto de contexto, perfil do laser, metadados
  D. Transação -> o registro que vai para cobrança (placa/tag, categoria, horário, pórtico)

O tráfego vem de duas fontes possíveis:
  1. Cenários (padrão): VDM = volume diário médio por local, hipóteses declaradas abaixo.
  2. Dados reais da ANTT: se o CSV "Volume de tráfego nas praças de pedágio" estiver em
     data/raw/antt/*.csv, o VDM de cada pórtico da EPR Paraná é calculado a partir dele.
     Download: https://dados.antt.gov.br/dataset/volume-trafego-praca-pedagio

Saídas:
  reports/tabelas/volume_dados_cenarios.csv
  reports/figures/04_dados_por_minuto_24h.png
  reports/figures/05_camadas_de_dados.png
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tabelas"
ANTT_DIR = ROOT / "data" / "raw" / "antt"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

KB, MB, GB, TB = 1e3, 1e6, 1e9, 1e12


# --------------------------------------------------------------------------------------
# Parâmetros (todos editáveis). Valores típicos de mercado para sistemas ANPR + RFID.
# --------------------------------------------------------------------------------------
@dataclass
class Portico:
    """Um pórtico cobrindo UM sentido de uma rodovia duplicada."""
    faixas: int = 2                     # faixas de rolamento cobertas (acostamento à parte)
    cam_anpr_por_faixa: int = 2         # frontal + traseira (lê placa dianteira e traseira)
    cam_contexto: int = 1               # câmera colorida panorâmica do pórtico
    # Câmera ANPR: sensor monocromático global shutter com iluminação infravermelha
    anpr_mp: float = 5.0                # megapixels
    anpr_fps: int = 30
    anpr_bytes_px: float = 1.0          # 8 bits mono
    # Câmera de contexto: colorida 4K
    ctx_mp: float = 8.3
    ctx_fps: int = 25
    ctx_bytes_px: float = 1.5           # YUV 4:2:0
    # Vídeo de auditoria (H.265) por câmera
    video_mbps: float = 4.0


@dataclass
class PacoteEvidencia:
    """Tamanho médio (bytes) do que é salvo para CADA veículo que passa."""
    itens: dict = field(default_factory=lambda: {
        "2 fotos ANPR (JPEG, frente+trás)": 2 * 250 * KB,
        "2 recortes da placa": 2 * 15 * KB,
        "1 foto de contexto colorida": 400 * KB,
        "perfil 3D do laser (classificação/eixos)": 30 * KB,
        "metadados JSON (OCR, confiança, velocidade, faixa)": 3 * KB,
        "leitura da tag RFID (TID/EPC, RSSI)": 0.3 * KB,
    })

    @property
    def total(self) -> float:
        return sum(self.itens.values())


TRANSACAO_BYTES = 2 * KB   # registro enviado ao backoffice de cobrança

# Cenários de VDM por LOCAL (os dois sentidos somados). Hipóteses, não medições:
#   baixo  ~ rodovia regional (PR-182 / PR-280)
#   médio  ~ BR-369 entre Londrina e Cornélio Procópio
#   alto   ~ BR-376 no eixo Maringá-Apucarana, perto do limite de uma pista dupla
CENARIOS_VDM = {"baixo": 8_000, "médio": 20_000, "alto": 40_000}

# Perfil horário típico de rodovia intermunicipal (fração do VDM em cada hora, soma = 1).
# Pico da manhã ~7-8h e da tarde ~17-18h, madrugada com caminhões.
_PERFIL = np.array([1.4, 1.0, 0.9, 0.9, 1.2, 2.2, 4.0, 6.2, 6.6, 5.6, 5.3, 5.4,
                    5.5, 5.4, 5.5, 5.8, 6.4, 7.3, 7.6, 5.8, 4.3, 3.5, 2.7, 2.1])
PERFIL_HORARIO = _PERFIL / _PERFIL.sum()


# --------------------------------------------------------------------------------------
def taxas_fixas(p: Portico) -> dict:
    """Camadas que não dependem do tráfego (bytes por segundo, por pórtico)."""
    n_anpr = p.faixas * p.cam_anpr_por_faixa
    bruto = (n_anpr * p.anpr_mp * 1e6 * p.anpr_bytes_px * p.anpr_fps
             + p.cam_contexto * p.ctx_mp * 1e6 * p.ctx_bytes_px * p.ctx_fps)
    video = (n_anpr + p.cam_contexto) * p.video_mbps * 1e6 / 8
    return {"A_bruto_sensor_Bps": bruto, "B_video_auditoria_Bps": video,
            "n_cameras": n_anpr + p.cam_contexto}


def dados_por_minuto(veic_min: float, ev: PacoteEvidencia) -> dict:
    return {"C_evidencia_B_min": veic_min * ev.total,
            "D_transacao_B_min": veic_min * TRANSACAO_BYTES}


def vdm_da_antt() -> pd.DataFrame | None:
    """Lê o CSV oficial da ANTT (se existir) e devolve o VDM por praça da EPR Paraná."""
    arquivos = sorted(ANTT_DIR.glob("*.csv"))
    if not arquivos:
        return None
    frames = []
    for arq in arquivos:
        df = pd.read_csv(arq, sep=";", encoding="latin-1", dtype=str)
        df.columns = [c.strip().lower() for c in df.columns]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["volume"] = pd.to_numeric(
        df["volume_total"].str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
        errors="coerce")
    df["data"] = pd.to_datetime(df["mes_ano"], dayfirst=True, errors="coerce")
    epr = df[df["concessionaria"].str.contains("EPR", case=False, na=False)]
    if epr.empty:
        print("CSV da ANTT encontrado, mas sem linhas da EPR. Usando cenários.")
        return None
    diario = epr.groupby(["concessionaria", "praca", "data"]).volume.sum().reset_index()
    return (diario.groupby(["concessionaria", "praca"]).volume.mean()
                  .rename("vdm").reset_index().sort_values("vdm", ascending=False))


def tabela_cenarios(p: Portico, ev: PacoteEvidencia, n_locais: int = 4) -> pd.DataFrame:
    fixas = taxas_fixas(p)
    linhas = []
    for nome, vdm in CENARIOS_VDM.items():
        pico_h = vdm * PERFIL_HORARIO.max()             # veículos na hora de pico (local)
        veic_min_pico = pico_h / 60
        veic_min_medio = vdm / 1440
        d_pico = dados_por_minuto(veic_min_pico, ev)
        d_dia = dados_por_minuto(vdm / 1440, ev)
        portic_por_local = 2                             # um pórtico por sentido
        linhas.append({
            "cenario": nome,
            "vdm_local": vdm,
            "veic_hora_pico": round(pico_h),
            "veic_min_pico": round(veic_min_pico, 1),
            "veic_seg_pico": round(veic_min_pico / 60, 2),
            "veic_min_medio": round(veic_min_medio, 1),
            "C_evidencia_MB_min_pico": d_pico["C_evidencia_B_min"] / MB,
            "D_transacao_KB_min_pico": d_pico["D_transacao_B_min"] / KB,
            "C_evidencia_GB_dia_local": d_dia["C_evidencia_B_min"] * 1440 / GB,
            "B_video_GB_dia_local": fixas["B_video_auditoria_Bps"] * 86400 * portic_por_local / GB,
            "A_bruto_GB_min_local": fixas["A_bruto_sensor_Bps"] * 60 * portic_por_local / GB,
            "C_evidencia_TB_ano_rede_EPR_Lote4":
                d_dia["C_evidencia_B_min"] * 1440 * 365 * n_locais / TB,
        })
    return pd.DataFrame(linhas)


def fig_24h(ev: PacoteEvidencia) -> None:
    minutos = np.arange(1440)
    frac_min = np.repeat(PERFIL_HORARIO, 60) / 60
    # Suaviza a transição entre horas
    frac_min = np.convolve(frac_min, np.ones(31) / 31, mode="same")
    frac_min /= frac_min.sum()
    fig, ax = plt.subplots(figsize=(9, 4))
    cores = {"baixo": "#9aa3ad", "médio": "#2a6fdb", "alto": "#e8833a"}
    for nome, vdm in CENARIOS_VDM.items():
        mb_min = vdm * frac_min * ev.total / MB
        ax.plot(minutos / 60, mb_min, color=cores[nome], lw=2,
                label=f"VDM {vdm:,}".replace(",", ".") + f" ({nome})")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("hora do dia")
    ax.set_ylabel("MB por minuto (evidências)")
    ax.set_title("Dados de evidência gerados por minuto em UM local (2 sentidos)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "04_dados_por_minuto_24h.png")
    plt.close(fig)


def fig_camadas(p: Portico, ev: PacoteEvidencia) -> None:
    """Escala logarítmica: do bruto do sensor ao registro de cobrança, no pico do cenário médio."""
    fixas = taxas_fixas(p)
    veic_min = CENARIOS_VDM["médio"] * PERFIL_HORARIO.max() / 60 / 2   # um sentido
    camadas = {
        "A. Bruto no sensor\n(todas as câmeras)": fixas["A_bruto_sensor_Bps"] * 60,
        "B. Vídeo de auditoria\n(H.265)": fixas["B_video_auditoria_Bps"] * 60,
        "C. Evidências\n(fotos + laser + JSON)": veic_min * ev.total,
        "D. Transações\n(backoffice)": veic_min * TRANSACAO_BYTES,
    }
    fig, ax = plt.subplots(figsize=(8, 4))
    nomes, vals = list(camadas), [v / MB for v in camadas.values()]
    barras = ax.barh(nomes, vals, color=["#c9d3dd", "#9aa3ad", "#2a6fdb", "#e8833a"])
    ax.set_xscale("log")
    ax.invert_yaxis()
    for b, v in zip(barras, vals):
        txt = f"{v/1000:,.1f} GB/min" if v >= 1000 else (f"{v:,.1f} MB/min" if v >= 1
                                                          else f"{v*1000:,.0f} KB/min")
        ax.text(v * 1.15, b.get_y() + b.get_height() / 2, txt, va="center")
    ax.set_xlim(0.01, max(vals) * 30)
    ax.set_xlabel("MB por minuto (escala log)")
    ax.set_title(f"Um pórtico, um sentido, hora de pico (~{veic_min:.0f} veículos/min)")
    fig.tight_layout()
    fig.savefig(FIG / "05_camadas_de_dados.png")
    plt.close(fig)


if __name__ == "__main__":
    p, ev = Portico(), PacoteEvidencia()
    fixas = taxas_fixas(p)

    print("=" * 70)
    print("PACOTE DE EVIDÊNCIA POR PASSAGEM")
    for k, v in ev.itens.items():
        print(f"  {k:<52s} {v/KB:8.1f} KB")
    print(f"  {'TOTAL':<52s} {ev.total/KB:8.1f} KB")
    print(f"\nPor pórtico (1 sentido, {p.faixas} faixas, {fixas['n_cameras']} câmeras):")
    print(f"  A. bruto no sensor:   {fixas['A_bruto_sensor_Bps']/GB:6.2f} GB/s "
          f"(= {fixas['A_bruto_sensor_Bps']*60/GB:,.0f} GB/min, processado na borda)")
    print(f"  B. vídeo auditoria:   {fixas['B_video_auditoria_Bps']*60/MB:6.1f} MB/min")

    antt = vdm_da_antt()
    if antt is not None:
        print("\nVDM real (ANTT) por praça da EPR:")
        print(antt.to_string(index=False))

    tab = tabela_cenarios(p, ev)
    tab.to_csv(TAB / "volume_dados_cenarios.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nCENÁRIOS (por local = 2 pórticos, um por sentido)")
    print(tab.round(2).T.to_string())

    fig_24h(ev)
    fig_camadas(p, ev)
    print(f"\nFiguras em {FIG}")
