import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import time
import warnings

warnings.filterwarnings("ignore")

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Respira Melhor - UBS Vila Curuça", layout="wide")

# ================================
# CSS
# ================================
st.markdown("""
<style>
html { font-size: 16px; -webkit-text-size-adjust: 100%; }
body { overflow-x: hidden; }

.block-container {
    padding-top: 1.2rem !important;
    padding-left: clamp(0.8rem, 2vw, 2rem) !important;
    padding-right: clamp(0.8rem, 2vw, 2rem) !important;
    padding-bottom: 0.4rem !important;
    max-width: 100% !important;
}

h1 { margin-bottom: 0.25rem !important; }
h2 { font-size: clamp(0.9rem, 1.5vw, 1.3rem) !important; margin-bottom: 0.15rem !important; }
h3 { font-size: clamp(0.82rem, 1.2vw, 1.05rem) !important; }
.stPlotlyChart { width: 100% !important; }

/* CARDS */
.pol-grid {
    display: flex;
    flex-wrap: wrap;
    gap: clamp(5px, 0.7vw, 9px);
    margin: 8px 0 10px 0;
}
.pol-card {
    flex: 1 1 calc(16.66% - 9px);
    min-width: 105px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 9px;
    padding: clamp(8px, 0.9vw, 13px) clamp(10px, 1vw, 14px);
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    box-sizing: border-box;
}
.pol-card .pol-label { font-size: clamp(0.62rem, 0.8vw, 0.76rem); color: #666; margin-bottom: 2px; white-space: nowrap; }
.pol-card .pol-iqa   { font-size: clamp(1rem, 1.7vw, 1.55rem); font-weight: 700; line-height: 1.1; }
.pol-card .pol-conc  { font-size: clamp(0.58rem, 0.75vw, 0.7rem); color: #888; margin-top: 3px; }
@media (max-width: 900px) { .pol-card { flex: 1 1 calc(33.33% - 7px); } }
@media (max-width: 540px) { .pol-card { flex: 1 1 calc(50% - 5px); } }

/* BANNER IQA */
.iqa-banner {
    border-radius: 9px;
    margin: 0 0 10px 0;
    padding: clamp(10px, 1vw, 14px) clamp(12px, 1.4vw, 18px);
    display: flex; align-items: center;
    gap: clamp(8px, 1vw, 14px); flex-wrap: wrap;
}
.iqa-valor { font-size: clamp(0.95rem, 1.9vw, 1.7rem); font-weight: 700; line-height: 1.1; white-space: nowrap; }
.iqa-sep   { width: 1px; height: 30px; background: rgba(0,0,0,0.15); flex-shrink: 0; }
.iqa-rec   { font-size: clamp(0.66rem, 0.95vw, 0.84rem); color: #444; line-height: 1.3; }

/* NAVEGAÇÃO DO CARROSSEL */
.carousel-dots { display: flex; gap: 6px; align-items: center; }
.carousel-dot  { width: 8px; height: 8px; border-radius: 50%; background: rgba(0,0,0,0.15); display: inline-block; }
.carousel-dot.active { background: #1976D2; }
.carousel-label { font-size: 0.75rem; color: #777; }

/* BARRA DE PROGRESSO */
.prog-wrap { flex: 1; height: 3px; background: rgba(0,0,0,0.08); border-radius: 2px; overflow: hidden; }
.prog-fill { height: 3px; border-radius: 2px; background: #1976D2; transition: width 1.8s linear; }

@media (min-width: 1800px) {
    html { font-size: 18px; }
    .pol-card .pol-iqa   { font-size: 1.8rem !important; }
    .pol-card .pol-label { font-size: 0.8rem !important; }
    .iqa-valor { font-size: 2rem !important; }
}
@media (min-width: 2400px) {
    .block-container { max-width: 2400px !important; margin: 0 auto !important; }
}
</style>
""", unsafe_allow_html=True)

# ================================
# CONSTANTES
# ================================
BASE_URL     = "https://arcgis.cetesb.sp.gov.br/server/rest/services/QUALAR_views/Qualidade_Ar_QUALAR/MapServer"
ESTACAO_ALVO = "Itaim Paulista"

LAYERS = {0:"CO", 1:"MP10", 2:"MP2.5", 3:"NO2", 4:"O3", 5:"SO2"}

# ── LIMITES IQAr — CONAMA 506/2024 vigente desde 01/01/2026 ──────────────
# Fonte: Guia Técnico MMA/CETESB jan/2025
# MP2.5: limite "Boa" caiu de 45 → 15 µg/m³ (muito mais restritivo)
LIMITES_IQA = {
    "MP2.5": [(0,0,15,40),(15,40,50,80),(50,80,75,120),(75,120,125,200),(125,200,300,400)],
    "MP10":  [(0,0,45,40),(45,40,100,80),(100,80,150,120),(150,120,250,200),(250,200,600,400)],
    "O3":    [(0,0,100,40),(100,40,130,80),(130,80,160,120),(160,120,200,200),(200,200,800,400)],
    "NO2":   [(0,0,200,40),(200,40,240,80),(240,80,320,120),(320,120,1130,200),(1130,200,3750,400)],
    "CO":    [(0,0,9,40),(9,40,11,80),(11,80,13,120),(13,120,15,200),(15,200,50,400)],
    "SO2":   [(0,0,40,40),(40,40,50,80),(50,80,125,120),(125,120,800,200),(800,200,2620,400)],
}
CLASSES_IQA = [
    (0,40,"Boa","#00C853","🟢"),(41,80,"Moderada","#FFD600","🟡"),
    (81,120,"Ruim","#FF6D00","🟠"),(121,200,"Muito Ruim","#D50000","🔴"),
    (201,999,"Péssima","#6A1B9A","⚫"),
]
# Textos oficiais do Guia Técnico MMA/CETESB 2025 (CONAMA 506/2024)
RECOMENDACOES = {
    "Boa":        ("Nenhum efeito esperado à saúde. Aproveite o ar livre!", "✅"),
    "Moderada":   ("Crianças, idosos e pessoas com doenças respiratórias ou cardíacas podem sentir tosse seca e cansaço. A população em geral não é afetada.", "⚠️"),
    "Ruim":       ("Toda a população pode sentir tosse seca, cansaço e ardor nos olhos, nariz e garganta. Grupos sensíveis podem ter efeitos mais sérios — evite atividades ao ar livre.", "🚨"),
    "Muito Ruim": ("Toda a população pode ter agravamento dos sintomas: tosse, cansaço, ardor, falta de ar e respiração ofegante. Grupos sensíveis: não saia de casa.", "🔴"),
    "Péssima":    ("ALERTA CRÍTICO: Toda a população corre risco de doenças respiratórias e cardiovasculares graves. Permaneça em ambientes fechados e procure atendimento médico.", "☣️"),
}
FEATURES_MODELO = [
    "MP25","MP10","O3","NO2","hora_sin","hora_cos","dia_semana",
    "lag_1h","lag_2h","lag_3h","lag_6h","lag_12h","lag_24h",
    "mp10_lag_1h","mp10_lag_3h","mp10_lag_24h",
    "o3_lag_1h","o3_lag_3h","o3_lag_24h",
    "no2_lag_1h","no2_lag_3h","no2_lag_24h",
    "media_3h","media_6h","diff_1h",
]
RENAME_PARA_MODELO = {"MP2.5":"MP25"}
POLUENTES    = ["MP2.5","MP10","O3","NO2","CO","SO2"]

# Nomes amigáveis para leigos — exibidos nos cards e no banner
NOMES_POL = {
    "MP2.5": "Part. finas (MP2.5)",
    "MP10":  "Part. inaláveis (MP10)",
    "O3":    "Ozônio (O3)",
    "NO2":   "Dióxido de nitrogênio",
    "CO":    "Monóxido de carbono",
    "SO2":   "Dióxido de enxofre",
}
CHART_HEIGHT = 420
PLOTLY_CONFIG = {"responsive":True,"displayModeBar":False}

# FIX: intervalo padrão 20s — razoável para TV e monitor.
# O usuário pode ajustar no controle numérico da barra de navegação.
INTERVALO_PADRAO = 20

SLIDES = [
    ("📊 Qualidade do ar agora + histórico", "situacao"),
    ("🔮 Previsão de partículas finas",       "previsao"),
    ("📉 Índice de qualidade previsto",        "subindice"),
    ("ℹ️ Entenda o índice de qualidade",       "info"),
]
N_SLIDES = len(SLIDES)

# ================================
# FUNÇÕES
# ================================
def calcular_iqa_poluente(valor, poluente):
    if valor is None or pd.isna(valor) or valor < 0:
        return None
    for cl,il,ch,ih in LIMITES_IQA.get(poluente,[]):
        if cl <= valor <= ch:
            return float(il) if ch==cl else ((ih-il)/(ch-cl))*(valor-cl)+il
    return 400.0 if valor > 0 else 0.0

def classificar_iqa(iqa):
    for lo,hi,nome,cor,emoji in CLASSES_IQA:
        if lo <= iqa <= hi: return nome,cor,emoji
    return "Péssima","#6A1B9A","⚫"

@st.cache_data(ttl=600)
def coletar_dados():
    dados,erros=[],[]
    for layer,pol in LAYERS.items():
        try:
            r=requests.get(f"{BASE_URL}/{layer}/query",
                           params={"where":"1=1","outFields":"*","f":"json"},timeout=8)
            r.raise_for_status()
            for f in r.json().get("features",[]):
                if ESTACAO_ALVO.lower() not in f["attributes"].get("STATNM","").lower(): continue
                for i in range(1,49):
                    v,t=f["attributes"].get(f"M{i}"),f["attributes"].get(f"TM{i}")
                    if v is not None and t is not None:
                        dados.append({"poluente":pol,"valor":float(v),
                                      "datahora":pd.to_datetime(t,unit="ms")})
        except Exception as e: erros.append(f"{pol}: {e}")
    df=pd.DataFrame(dados).sort_values("datahora").reset_index(drop=True) if dados else pd.DataFrame()
    return df,erros

def preparar_pivot(df):
    p=df.pivot_table(index="datahora",columns="poluente",values="valor",aggfunc="mean")
    return p.resample("1h").mean().ffill().bfill()

@st.cache_resource
def carregar_modelo():
    base=Path(__file__).parent
    mp,fp=base/"modelo_xgboost.pkl",base/"features.pkl"
    if not mp.exists(): return None,None
    try:
        m=joblib.load(mp)
        f=joblib.load(fp) if fp.exists() else FEATURES_MODELO
        return m,f
    except Exception as e:
        st.warning(f"Erro ao carregar modelo: {e}"); return None,None

def preparar_features_forecast(hm,hmp10,ho3,hno2,ts,features):
    def sg(s,lag): return float(s.iloc[-lag]) if len(s)>=lag else (float(s.iloc[-1]) if len(s)>0 else 0.0)
    ang=2*np.pi*ts.hour/24; v=hm.values; u=float(v[-1]) if len(v)>0 else 0.0
    row={"MP25":u,"MP10":sg(hmp10,24),"O3":sg(ho3,24),"NO2":sg(hno2,24),
         "hora_sin":np.sin(ang),"hora_cos":np.cos(ang),"dia_semana":ts.dayofweek,
         "lag_1h":sg(hm,1),"lag_2h":sg(hm,2),"lag_3h":sg(hm,3),
         "lag_6h":sg(hm,6),"lag_12h":sg(hm,12),"lag_24h":sg(hm,24),
         "mp10_lag_1h":sg(hmp10,1),"mp10_lag_3h":sg(hmp10,3),"mp10_lag_24h":sg(hmp10,24),
         "o3_lag_1h":sg(ho3,1),"o3_lag_3h":sg(ho3,3),"o3_lag_24h":sg(ho3,24),
         "no2_lag_1h":sg(hno2,1),"no2_lag_3h":sg(hno2,3),"no2_lag_24h":sg(hno2,24),
         "media_3h":float(np.mean(v[-3:])) if len(v)>=3 else u,
         "media_6h":float(np.mean(v[-6:])) if len(v)>=6 else u,
         "diff_1h":float(v[-1]-v[-2]) if len(v)>=2 else 0.0}
    return pd.DataFrame([row]).reindex(columns=features,fill_value=0.0).values

def gerar_forecast(pivot,model,features,horas=6):
    h=pivot.rename(columns=RENAME_PARA_MODELO).copy().ffill().bfill()
    for p in ["MP25","MP10","O3","NO2"]:
        if p not in h.columns: h[p]=0.0
    mp25,mp10,o3,no2=h["MP25"].copy(),h["MP10"].copy(),h["O3"].copy(),h["NO2"].copy()
    preds,ts=[],[]
    for _ in range(horas):
        prx=mp25.index[-1]+timedelta(hours=1)
        X=preparar_features_forecast(mp25,mp10,o3,no2,prx,features)
        pred=float(max(model.predict(X)[0],0.0))
        _nxt = lambda s: float(s.iloc[-24]) if len(s)>=24 else float(s.iloc[-1])
        mp25=pd.concat([mp25,pd.Series([pred],    index=[prx])])
        mp10=pd.concat([mp10,pd.Series([_nxt(mp10)],index=[prx])])
        o3  =pd.concat([o3,  pd.Series([_nxt(o3)],  index=[prx])])
        no2 =pd.concat([no2, pd.Series([_nxt(no2)],  index=[prx])])
        preds.append(pred); ts.append(prx)
    return pd.Series(preds,index=ts,name="MP2.5")

# ================================
# RENDERS
# ================================
def render_header(coleta_ts):
    st.title("🌎 Respira Melhor — UBS Vila Curuçá")
    st.caption(f"Dados em tempo real · CETESB Itaim Paulista · Coletado em: {coleta_ts.strftime('%d/%m/%Y %H:%M')} · IQAr calculado conforme CONAMA 506/2024 (vigente desde 01/01/2026)")

def render_nav(intervalo):
    """
    Barra de navegação: ← · dots + label + barra de progresso · → · [intervalo(s)]
    FIX: sleep(2)+rerun incremental substitui o sleep(60) que travava o servidor.
    """
    slide   = st.session_state.slide
    auto    = st.session_state.auto
    elapsed = max(0.0, (datetime.now() - st.session_state.last_change).total_seconds())
    pct     = min(elapsed / intervalo * 100, 100) if auto else 0
    label, _ = SLIDES[slide]

    dots = "".join(
        f'<span class="carousel-dot{" active" if i==slide else ""}"></span>'
        for i in range(N_SLIDES)
    )

    c_prev, c_mid, c_next, c_pause, c_seg = st.columns([1, 10, 1, 1, 2])

    if c_prev.button("←", key="btn_prev", width="stretch"):
        st.session_state.slide = (slide - 1) % N_SLIDES
        st.session_state.last_change = datetime.now()
        st.rerun()

    # Dots + label + barra de progresso numa linha só
    c_mid.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;padding:6px 0;">'
        f'<div class="carousel-dots">{dots}</div>'
        f'<span class="carousel-label">{label} &nbsp;({slide+1}/{N_SLIDES})</span>'
        f'<div class="prog-wrap">'
        f'<div class="prog-fill" style="width:{pct:.1f}%"></div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if c_next.button("→", key="btn_next", width="stretch"):
        st.session_state.slide = (slide + 1) % N_SLIDES
        st.session_state.last_change = datetime.now()
        st.rerun()

    # Botão pause/play
    lbl = "⏸" if auto else "▶"
    if c_pause.button(lbl, key="btn_pause", width="stretch"):
        st.session_state.auto = not auto
        st.session_state.last_change = datetime.now()
        st.rerun()

    # Intervalo configurável
    novo = c_seg.number_input("s", min_value=5, max_value=120,
                              value=intervalo, step=5,
                              label_visibility="collapsed",
                              key="input_intervalo")
    if novo != intervalo:
        st.session_state.intervalo = novo

    return elapsed

def render_situacao(pivot):
    cards,iqas=[],{}
    for pol in POLUENTES:
        if pol in pivot.columns:
            serie=pivot[pol].dropna()
            valor=serie.iloc[-1] if not serie.empty else None
            if valor is not None:
                iqa=calcular_iqa_poluente(valor,pol)
                nome,cor,emoji=classificar_iqa(iqa); iqas[pol]=iqa
                un="ppm" if pol=="CO" else "µg/m³"
                cards.append(f'<div class="pol-card">'
                              f'<div class="pol-label">{emoji} {NOMES_POL.get(pol, pol)}</div>'
                              f'<div class="pol-iqa" style="color:{cor}">{iqa:.0f}</div>'
                              f'<div class="pol-conc">{valor:.1f} {un} · {nome}</div></div>')
            else:
                cards.append(f'<div class="pol-card"><div class="pol-label">⚪ {pol}</div>'
                             f'<div class="pol-iqa" style="color:#aaa">—</div>'
                             f'<div class="pol-conc">Sem dado recente</div></div>')
        else:
            cards.append(f'<div class="pol-card"><div class="pol-label">⚪ {pol}</div>'
                         f'<div class="pol-iqa" style="color:#aaa">—</div>'
                         f'<div class="pol-conc">Não monitorado</div></div>')

    st.markdown(f'<div class="pol-grid">{"".join(cards)}</div>',unsafe_allow_html=True)

    if iqas:
        ig=max(iqas.values()); pc=max(iqas,key=iqas.get)
        nc,cc,ec=classificar_iqa(ig); rt,re=RECOMENDACOES[nc]
        st.markdown(
            f'<div class="iqa-banner" style="background:{cc}18;border-left:5px solid {cc};">'
            f'<div class="iqa-valor" style="color:{cc}">{ec}&nbsp;Índice de Qualidade do Ar: {ig:.0f} — {nc}</div>'
            f'<div class="iqa-sep"></div>'
            f'<div class="iqa-rec">Principal poluente: <b>{NOMES_POL.get(pc, pc)}</b><br>{re} {rt}</div>'
            f'</div>',unsafe_allow_html=True)
        if nc == "Moderada":
            st.warning(f"⚠️ Atenção: Qualidade do ar {nc.upper()} em Itaim Paulista. {rt}")
        elif nc in ["Ruim","Muito Ruim","Péssima"]:
            st.error(f"🚨 ALERTA: Qualidade do ar {nc.upper()} em Itaim Paulista. {rt}")

def render_historico(pivot):
    # width="stretch" — API atual do Streamlit
    cols=[p for p in POLUENTES if p in pivot.columns and not pivot[p].dropna().empty]
    if cols:
        dl=pivot[cols].tail(48).reset_index().melt(id_vars="datahora",
                                                    var_name="Poluente",value_name="Concentração")
        fig=px.line(dl,x="datahora",y="Concentração",color="Poluente",
                    markers=False,title="Evolução dos poluentes nas últimas 48 horas")
        fig.update_layout(autosize=True,height=CHART_HEIGHT,
                          margin=dict(l=40,r=40,t=50,b=40),hovermode="x unified",
                          xaxis_title="Data/Hora",yaxis_title="Concentração",legend_title="Poluente")
        st.plotly_chart(fig,width="stretch",config=PLOTLY_CONFIG)
    else:
        st.warning("Sem dados históricos suficientes para exibir.")

def render_previsao(pivot, serie_forecast):
    if serie_forecast is None:
        st.warning("⚠️ Arquivo `modelo_xgboost.pkl` não encontrado."); return
    fig=go.Figure()
    if "MP2.5" in pivot.columns:
        h=pivot["MP2.5"].tail(24)
        fig.add_trace(go.Scatter(x=h.index,y=h.values,mode="lines",name="Medido (últimas 24h)",
                                 line=dict(color="#1976D2",width=2)))
    fig.add_trace(go.Scatter(x=serie_forecast.index,y=serie_forecast.values,
                             mode="lines+markers",name="Previsão (próximas 6h)",
                             line=dict(color="#FF6D00",width=2,dash="dash"),marker=dict(size=5)))
    fig.update_layout(autosize=True,height=CHART_HEIGHT,
                      margin=dict(l=40,r=40,t=50,b=40),hovermode="x unified",
                      title="Partículas finas (MP2.5) — medido hoje + previsão para as próximas 6h",
                      xaxis_title="Data/Hora",yaxis_title="Concentração (µg/m³)")
    # width="stretch" — API atual do Streamlit
    st.plotly_chart(fig,width="stretch",config=PLOTLY_CONFIG)
    st.caption("Previsão gerada por inteligência artificial · dados históricos da CETESB Itaim Paulista")

def render_subindice(pivot, serie_forecast, csv_bytes):
    if serie_forecast is None:
        st.warning("Sem previsão disponível."); return

    subindice=serie_forecast.rolling(3,min_periods=1).mean().apply(
        lambda v: calcular_iqa_poluente(v,"MP2.5"))

    fig=go.Figure()
    fig.add_trace(go.Scatter(x=subindice.index,y=subindice.values,
                             mode="lines+markers",name="Índice previsto",
                             line=dict(color="#FF6D00",width=3),
                             fill="tozeroy",fillcolor="rgba(255,109,0,0.1)"))
    for lo,hi,nome,cor,_ in CLASSES_IQA:
        fig.add_hrect(y0=lo,y1=min(hi,300),fillcolor=cor,opacity=0.07,line_width=0,
                      annotation_text=nome,annotation_position="right")
    y_max=max(60,float(subindice.max())+20)
    fig.update_layout(autosize=True,height=CHART_HEIGHT,
                      margin=dict(l=40,r=40,t=50,b=40),
                      title="Índice de Qualidade do Ar previsto para as próximas 6 horas",
                      xaxis_title="Data/Hora",yaxis_title="Índice de Qualidade do Ar",
                      yaxis=dict(range=[0,y_max]))
    # width="stretch" — API atual do Streamlit
    st.plotly_chart(fig,width="stretch",config=PLOTLY_CONFIG)

    im=subindice.max(); tm=subindice.idxmax()
    nf,_,ef=classificar_iqa(im); rf,_=RECOMENDACOES[nf]
    st.info(f"{ef} Pior momento previsto: **IQA {im:.0f} ({nf})** às {tm.strftime('%d/%m %H:%M')} · {rf}")

    with st.expander("📋 Ver dados brutos"):
        st.dataframe(pivot.tail(48).style.format("{:.2f}"),width="stretch")
        # FIX: key estável baseado no timestamp de coleta evita que o rerun
        # do carrossel invalide o arquivo temporário do download_button.
        st.download_button(
            "📥 Baixar CSV",
            csv_bytes,
            "itaim_paulista_48h.csv",
            "text/csv",
            key=f"dl_{st.session_state.get('csv_ts','0')}",
        )


def render_info():
    """
    Tela educativa — renderizada via st.iframe() para evitar
    sanitização do st.markdown que remove tags table/div complexas.
    """
    html_completo = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 13px; line-height: 1.45; color: #333;
    background: transparent; padding: 4px 2px 8px 2px;
  }
  .hdr {
    background: #E3F2FD; border-left: 4px solid #1976D2;
    border-radius: 8px; padding: 8px 14px; margin-bottom: 9px;
  }
  .hdr-t { font-size: 16px; font-weight: 700; color: #1565C0; margin-bottom: 3px; }
  .hdr-b { font-size: 11.5px; color: #333; line-height: 1.45; }
  .cols   { display: flex; gap: 14px; align-items: flex-start; }
  .left   { flex: 1.25; min-width: 0; }
  .right  { flex: 1;    min-width: 0; }
  .lbl {
    font-size: 10px; font-weight: 600; color: #666;
    text-transform: uppercase; letter-spacing: .05em; margin-bottom: 5px;
  }
  table   { width: 100%; border-collapse: collapse; font-size: 11px; }
  th {
    padding: 4px 6px; background: #F5F5F5;
    border-bottom: 2px solid #ddd; font-weight: 600; text-align: center;
  }
  th:first-child { text-align: left; }
  td { padding: 4px 6px; text-align: center; border-bottom: 1px solid #eee; }
  td:first-child { text-align: left; font-weight: 700; }
  .sc {
    display: flex; align-items: flex-start; gap: 7px;
    border-radius: 7px; padding: 6px 9px; margin-bottom: 5px;
  }
  .sc-ico { font-size: 14px; flex-shrink: 0; line-height: 1.3; }
  .sc-t   { font-weight: 700; font-size: 11.5px; margin-bottom: 2px; }
  .sc-b   { font-size: 10.5px; color: #333; line-height: 1.38; }
  .disc {
    margin-top: 9px; padding: 8px 12px;
    background: #E3F2FD; border-left: 3px solid #1565C0;
    border-radius: 7px; font-size: 10.5px; color: #333; line-height: 1.5;
  }
  .disc b { color: #1565C0; }
  .disc a { color: #1565C0; }
  .src {
    margin-top: 8px; padding: 4px 10px; background: #F5F5F5;
    border-radius: 6px; font-size: 9.5px; color: #aaa; line-height: 1.4;
  }
  @media (max-width: 650px) { .cols { flex-direction: column; } }
</style>
</head>
<body>

<div class="hdr">
  <div class="hdr-t">&#x2139;&#xFE0F; O que &eacute; o &Iacute;ndice de Qualidade do Ar (IQAr)?</div>
  <div class="hdr-b">
    Pense no IQAr como o <b>term&ocirc;metro do ar</b>: vai de <b>0 a 400</b>, quanto menor melhor.
    Sensores medem os poluentes a cada hora e uma f&oacute;rmula converte tudo numa nota &uacute;nica &mdash;
    sempre a do poluente mais preocupante. Tabela vigente desde <b>01/01/2026</b> (CONAMA 506/2024).
  </div>
</div>

<div class="cols">

  <!-- COLUNA ESQUERDA -->
  <div class="left">
    <div class="lbl">Tabela oficial &middot; CONAMA 506/2024</div>
    <table>
      <thead>
        <tr>
          <th>Qualidade</th><th>IQAr</th>
          <th>MP2.5<br>(&micro;g/m&sup3;)</th>
          <th>MP10<br>(&micro;g/m&sup3;)</th>
          <th>O3<br>(&micro;g/m&sup3;)</th>
          <th>NO2<br>(&micro;g/m&sup3;)</th>
          <th>CO<br>(ppm)</th>
          <th>SO2<br>(&micro;g/m&sup3;)</th>
        </tr>
      </thead>
      <tbody>
        <tr style="background:#E8F5E9;">
          <td style="color:#1B5E20;">&#127866; N1 Boa</td>
          <td>0&ndash;40</td><td>0&ndash;15</td><td>0&ndash;45</td>
          <td>0&ndash;100</td><td>0&ndash;200</td><td>0&ndash;9</td><td>0&ndash;40</td>
        </tr>
        <tr style="background:#FFFDE7;">
          <td style="color:#F57F17;">&#127864; N2 Moderada</td>
          <td>41&ndash;80</td><td>15&ndash;50</td><td>45&ndash;100</td>
          <td>100&ndash;130</td><td>200&ndash;240</td><td>9&ndash;11</td><td>40&ndash;50</td>
        </tr>
        <tr style="background:#FFF3E0;">
          <td style="color:#E65100;">&#128993; N3 Ruim</td>
          <td>81&ndash;120</td><td>50&ndash;75</td><td>100&ndash;150</td>
          <td>130&ndash;160</td><td>240&ndash;320</td><td>11&ndash;13</td><td>50&ndash;125</td>
        </tr>
        <tr style="background:#FFEBEE;">
          <td style="color:#B71C1C;">&#128308; N4 Muito Ruim</td>
          <td>121&ndash;200</td><td>75&ndash;125</td><td>150&ndash;250</td>
          <td>160&ndash;200</td><td>320&ndash;1130</td><td>13&ndash;15</td><td>125&ndash;800</td>
        </tr>
        <tr style="background:#F3E5F5;">
          <td style="color:#4A148C;">&#11035; N5 P&eacute;ssima</td>
          <td>201&ndash;400</td><td>125&ndash;300</td><td>250&ndash;600</td>
          <td>200&ndash;800</td><td>1130&ndash;3750</td><td>15&ndash;50</td><td>800&ndash;2620</td>
        </tr>
      </tbody>
    </table>

    <div class="disc">
      <b>&#128203; Este painel &eacute; informativo.</b>
      Recomenda&ccedil;&otilde;es baseadas no Guia MMA/CETESB jan/2025 e diretrizes OMS 2021.
      <b>N&atilde;o substitui orienta&ccedil;&atilde;o m&eacute;dica.</b><br>
      Em situa&ccedil;&atilde;o <b>Muito Ruim ou P&eacute;ssima</b>, siga a
      <a href="https://cetesb.sp.gov.br/ar/" target="_blank">CETESB</a>
      e a Vigil&acirc;ncia Sanit&aacute;ria local.<br>
      Sintomas graves: procure a UBS ou ligue <b>192 (SAMU)</b>.<br>
      Epis&oacute;dio Cr&iacute;tico: responsabilidade da CETESB (CONAMA 491/2018, Art. 10&ndash;11).
    </div>
  </div>

  <!-- COLUNA DIREITA -->
  <div class="right">
    <div class="lbl">O que voc&ecirc; pode sentir &mdash; por faixa</div>

    <div class="sc" style="background:#E8F5E9;">
      <span class="sc-ico">&#129001;</span>
      <div>
        <div class="sc-t" style="color:#1B5E20;">Boa &mdash; IQAr 0 a 40</div>
        <div class="sc-b">Nenhum efeito esperado. Aproveite o ar livre!</div>
      </div>
    </div>

    <div class="sc" style="background:#FFFDE7;">
      <span class="sc-ico">&#129000;</span>
      <div>
        <div class="sc-t" style="color:#F57F17;">Moderada &mdash; IQAr 41 a 80</div>
        <div class="sc-b"><b>Crian&ccedil;as, idosos e pessoas com asma ou doen&ccedil;as card&iacute;acas</b>
        podem sentir tosse seca e cansa&ccedil;o. A popula&ccedil;&atilde;o em geral n&atilde;o &eacute; afetada.</div>
      </div>
    </div>

    <div class="sc" style="background:#FFF3E0;">
      <span class="sc-ico">&#128992;</span>
      <div>
        <div class="sc-t" style="color:#E65100;">Ruim &mdash; IQAr 81 a 120</div>
        <div class="sc-b"><b>Toda a popula&ccedil;&atilde;o</b> pode sentir tosse, cansa&ccedil;o
        e ardor nos olhos, nariz e garganta. Evite atividades f&iacute;sicas ao ar livre.</div>
      </div>
    </div>

    <div class="sc" style="background:#FFEBEE;">
      <span class="sc-ico">&#128308;</span>
      <div>
        <div class="sc-t" style="color:#B71C1C;">Muito Ruim &mdash; IQAr 121 a 200</div>
        <div class="sc-b">Tosse intensa, falta de ar, respira&ccedil;&atilde;o ofegante.
        <b>Grupos sens&iacute;veis: n&atilde;o saia de casa.</b>
        Evite esfor&ccedil;os ao ar livre.</div>
      </div>
    </div>

    <div class="sc" style="background:#F3E5F5;">
      <span class="sc-ico">&#11035;</span>
      <div>
        <div class="sc-t" style="color:#4A148C;">P&eacute;ssima &mdash; IQAr acima de 200</div>
        <div class="sc-b"><b>ALERTA:</b> Risco grave de doen&ccedil;as respirat&oacute;rias
        e cardiovasculares. Permane&ccedil;a em ambientes fechados.
        Procure atendimento m&eacute;dico imediatamente.</div>
      </div>
    </div>

  </div>
</div>

<div class="src">
  &#128225; CETESB &mdash; Esta&ccedil;&atilde;o Itaim Paulista &middot;
  Atualiza&ccedil;&atilde;o a cada 10 min &middot;
  IQAr: CONAMA 506/2024 &middot; Guia MMA/CETESB jan/2025 &middot;
  Epis&oacute;dios cr&iacute;ticos: CONAMA 491/2018 Anexo III
</div>

</body>
</html>"""
    st.iframe(html_completo, height=520)


# ================================
# SESSION STATE
# ================================
if "slide"       not in st.session_state: st.session_state.slide       = 0
if "auto"        not in st.session_state: st.session_state.auto        = True
if "intervalo"   not in st.session_state: st.session_state.intervalo   = INTERVALO_PADRAO
if "last_change" not in st.session_state: st.session_state.last_change = datetime.now()

# ================================
# DADOS  (cache TTL=600 — reruns não recoletam)
# ================================
coleta_ts=datetime.now()
with st.spinner("Carregando dados da CETESB Itaim Paulista..."):
    df,erros_coleta=coletar_dados()

if df.empty:
    st.error("Não foi possível carregar dados da CETESB. Tente novamente em alguns minutos.")
    st.stop()

pivot     = preparar_pivot(df)
# FIX: salvar csv no session_state para persistir entre reruns do carrossel.
# O st.download_button cria um arquivo temporário que é invalidado pelo st.rerun().
# Armazenando no session_state, o mesmo bytes object é reutilizado.
_csv_novo = pivot.tail(48).to_csv().encode("utf-8")
if "csv_bytes" not in st.session_state or st.session_state.get("csv_ts") != coleta_ts.strftime("%Y%m%d%H"):
    st.session_state.csv_bytes = _csv_novo
    st.session_state.csv_ts    = coleta_ts.strftime("%Y%m%d%H")
csv_bytes = st.session_state.csv_bytes
model_xgb,features_xgb=carregar_modelo()

serie_forecast_cache=None
if model_xgb is not None:
    try:
        feat=features_xgb if features_xgb is not None else FEATURES_MODELO
        serie_forecast_cache=gerar_forecast(pivot,model_xgb,feat,horas=6)
    except Exception as e:
        st.warning(f"Erro ao gerar previsão: {e}")

# ================================
# LAYOUT
# ================================
render_header(coleta_ts)
elapsed = render_nav(st.session_state.intervalo)

slide_key = SLIDES[st.session_state.slide][1]

# st.empty() garante que só a área do gráfico é substituída no rerun —
# o cabeçalho e a barra de navegação não piscam.
slide_container = st.empty()
with slide_container.container():
    if slide_key == "situacao":
        render_situacao(pivot)
        render_historico(pivot)
    elif slide_key == "previsao":
        render_previsao(pivot, serie_forecast_cache)
    elif slide_key == "subindice":
        render_subindice(pivot, serie_forecast_cache, csv_bytes)
    elif slide_key == "info":
        render_info()

# ================================
# AUTO-AVANÇO
# FIX CRÍTICO: substituído sleep(60-elapsed) por sleep(2)+rerun incremental.
# O sleep longo bloqueava o processo do Streamlit para TODOS os usuários.
# Com sleep(2), o processo fica livre 58s de cada 60s, e só acorda para:
#   1. atualizar a barra de progresso (visual)
#   2. verificar se chegou a hora de trocar o slide
# Os dados ficam em cache — nenhuma requisição extra acontece.
# ================================
if st.session_state.auto:
    if elapsed >= st.session_state.intervalo:
        st.session_state.slide = (st.session_state.slide + 1) % N_SLIDES
        st.session_state.last_change = datetime.now()
    time.sleep(2)
    st.rerun()

# ── Erros de coleta (discreto) ──────────────────────────────────────────────
if erros_coleta:
    with st.expander(f"⚠️ {len(erros_coleta)} poluente(s) com falha na coleta — clique para ver"):
        for e in erros_coleta: st.warning(e)

