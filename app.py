import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Qualidade do Ar - UBS Vila Curuça", layout="wide")

# ================================
# CSS RESPONSIVO
# ================================
st.markdown("""
<style>
@viewport { width: device-width; }

.block-container {
    padding-top: 1rem !important;
    padding-left: 2vw !important;
    padding-right: 2vw !important;
    max-width: 100% !important;
}

h1 { font-size: clamp(1.4rem, 3vw, 2.4rem) !important; }
h2 { font-size: clamp(1.1rem, 2.2vw, 1.8rem) !important; }
h3 { font-size: clamp(0.95rem, 1.8vw, 1.4rem) !important; }

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    padding: clamp(6px, 1.2vw, 16px) !important;
}
[data-testid="metric-container"] label {
    font-size: clamp(0.7rem, 1.2vw, 1rem) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: clamp(1rem, 2.5vw, 2rem) !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: clamp(0.65rem, 1vw, 0.9rem) !important;
}

.iqa-banner h2 {
    font-size: clamp(1.3rem, 3.5vw, 2.2rem) !important;
    margin: 0.2rem 0 !important;
}
.iqa-banner p {
    font-size: clamp(0.75rem, 1.4vw, 1rem) !important;
    margin-top: 0.4rem !important;
}

@media (max-width: 640px) {
    [data-testid="column"] {
        min-width: 45% !important;
        flex: 1 1 45% !important;
    }
    .block-container {
        padding-left: 4vw !important;
        padding-right: 4vw !important;
    }
}

@media (min-width: 1800px) {
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 2.4rem !important;
    }
    [data-testid="metric-container"] label {
        font-size: 1.1rem !important;
    }
    .iqa-banner h2 { font-size: 3rem !important; }
    .iqa-banner p  { font-size: 1.2rem !important; }
}
</style>
""", unsafe_allow_html=True)

BASE_URL = "https://arcgis.cetesb.sp.gov.br/server/rest/services/QUALAR_views/Qualidade_Ar_QUALAR/MapServer"

ESTACAO_ALVO = "Itaim Paulista"

LAYERS = {
    0: "CO",
    1: "MP10",
    2: "MP2.5",
    3: "NO2",
    4: "O3",
    5: "SO2"
}

# Limites para IQA — Padrão oficial CETESB/Brasil
# Formato: (c_low, i_low, c_high, i_high)
# Faixas: Boa 0-40 | Moderada 41-80 | Ruim 81-120 | Muito Ruim 121-200 | Péssima 201-400
# Fonte: https://cetesb.sp.gov.br/ar/padroes-e-indices/
LIMITES_IQA = {
    # MP2.5 (µg/m³) — média 24h — escala 1:1 CETESB
    "MP2.5": [
        (0,   0,   40,  40),   # Boa
        (40,  40,  80,  80),   # Moderada
        (80,  80,  120, 120),  # Ruim
        (120, 120, 250, 200),  # Muito Ruim / Péssima
    ],
    # MP10 (µg/m³) — média 24h
    "MP10": [
        (0,   0,   50,  40),
        (50,  40,  100, 80),
        (100, 80,  150, 120),
        (150, 120, 250, 200),
        (250, 200, 420, 400),
    ],
    # O3 (µg/m³) — média 8h
    "O3": [
        (0,   0,   100, 40),
        (100, 40,  160, 80),
        (160, 80,  200, 120),
        (200, 120, 800, 200),
    ],
    # NO2 (µg/m³) — média 1h
    "NO2": [
        (0,   0,   200, 40),
        (200, 40,  240, 80),
        (240, 80,  320, 120),
        (320, 120, 1130,200),
    ],
    # CO (µg/m³) — média 8h  (9000 µg/m³ = 8 ppm)
    "CO": [
        (0,     0,   9000,  40),
        (9000,  40,  11000, 80),
        (11000, 80,  13000, 120),
        (13000, 120, 15000, 200),
    ],
    # SO2 (µg/m³) — média 24h
    "SO2": [
        (0,   0,   20,  40),
        (20,  40,  40,  80),
        (40,  80,  365, 120),
        (365, 120, 800, 200),
    ],
}

CLASSES_IQA = [
    (0,   40,  "Boa",        "#00C853", "🟢"),
    (41,  80,  "Moderada",   "#FFD600", "🟡"),
    (81,  120, "Ruim",       "#FF6D00", "🟠"),
    (121, 200, "Muito Ruim", "#D50000", "🔴"),
    (201, 999, "Péssima",    "#6A1B9A", "⚫"),
]

RECOMENDACOES = {
    "Boa":        ("Qualidade do ar satisfatória. Sem restrições.", "✅"),
    "Moderada":   ("Grupos sensíveis (crianças, idosos, asmáticos) devem evitar esforço físico prolongado ao ar livre.", "⚠️"),
    "Ruim":       ("Todos devem reduzir atividades ao ar livre. Grupos sensíveis devem evitar exposição.", "🚨"),
    "Muito Ruim": ("Evite atividades ao ar livre. Mantenha janelas fechadas. Grupos sensíveis não devem sair.", "🔴"),
    "Péssima":    ("ALERTA CRÍTICO: Permaneça em ambientes fechados. Procure atendimento médico se sentir sintomas.", "☣️"),
}

# ================================
# FUNÇÕES
# ================================

def calcular_iqa_poluente(valor, poluente):
    if valor is None or np.isnan(valor):
        return None
    tabela = LIMITES_IQA.get(poluente, [])
    for (c_low, i_low, c_high, i_high) in tabela:
        if c_low <= valor <= c_high:
            if c_high == c_low:
                return i_low
            return ((i_high - i_low) / (c_high - c_low)) * (valor - c_low) + i_low
    return 400 if valor > 0 else 0

def classificar_iqa(iqa):
    for (lo, hi, nome, cor, emoji) in CLASSES_IQA:
        if lo <= iqa <= hi:
            return nome, cor, emoji
    return "Péssima", "#6A1B9A", "⚫"

@st.cache_data(ttl=600)
def coletar_dados():
    dados = []
    for layer, poluente in LAYERS.items():
        url = f"{BASE_URL}/{layer}/query"
        params = {"where": "1=1", "outFields": "*", "f": "json"}
        try:
            r = requests.get(url, params=params, timeout=15)
            data = r.json()
            for f in data.get("features", []):
                estacao = f["attributes"].get("STATNM", "")
                if ESTACAO_ALVO.lower() not in estacao.lower():
                    continue
                for i in range(1, 49):
                    valor = f["attributes"].get(f"M{i}")
                    tempo = f["attributes"].get(f"TM{i}")
                    if valor is not None and tempo is not None:
                        dados.append({
                            "poluente": poluente,
                            "valor": float(valor),
                            "datahora": pd.to_datetime(tempo, unit="ms")
                        })
        except Exception:
            continue

    if not dados:
        return pd.DataFrame()

    df = pd.DataFrame(dados)
    df = df.sort_values("datahora").reset_index(drop=True)
    return df

def preparar_pivot(df):
    pivot = df.pivot_table(index="datahora", columns="poluente", values="valor", aggfunc="mean")
    pivot = pivot.resample("1h").mean()
    pivot = pivot.ffill().bfill()
    return pivot

# Features esperadas pelo modelo XGBoost (mesma ordem do treino)
FEATURES_MODELO = ['MP25', 'MP10', 'O3', 'NO2', 'hora', 'dia_semana',
                   'lag_1h', 'lag_2h', 'lag_3h', 'lag_6h', 'lag_12h', 'lag_24h',
                   'media_3h', 'media_6h', 'diff_1h']

# Mapeamento nome API → nome modelo
RENAME_PARA_MODELO = {"MP2.5": "MP25"}
RENAME_PARA_PIVOT  = {"MP25": "MP2.5"}

@st.cache_resource
def carregar_modelo():
    """Carrega modelo XGBoost e features do disco se disponíveis."""
    caminhos = [
        "modelo_xgboost.pkl",
        os.path.join(os.path.dirname(__file__), "modelo_xgboost.pkl"),
    ]
    for path in caminhos:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                features_path = path.replace("modelo_xgboost.pkl", "features.pkl")
features = joblib.load(features_path)
                return model, features
            except Exception as e:
                st.warning(f"Erro ao carregar modelo: {e}")
    return None, None

def preparar_features_forecast(historico_mp25, historico_mp10, historico_o3, historico_no2, prox_ts):
    """Monta o vetor de features para uma hora futura, igual ao pipeline de treino."""
    def safe_get(serie, lag):
        return float(serie.iloc[-lag]) if len(serie) >= lag else 0.0

    mp25_vals = historico_mp25.values
    row = {
        "MP25":       safe_get(historico_mp25, 1),
        "MP10":       safe_get(historico_mp10, 1),
        "O3":         safe_get(historico_o3,   1),
        "NO2":        safe_get(historico_no2,  1),
        "hora":       prox_ts.hour,
        "dia_semana": prox_ts.dayofweek,
        "lag_1h":     safe_get(historico_mp25, 1),
        "lag_2h":     safe_get(historico_mp25, 2),
        "lag_3h":     safe_get(historico_mp25, 3),
        "lag_6h":     safe_get(historico_mp25, 6),
        "lag_12h":    safe_get(historico_mp25, 12),
        "lag_24h":    safe_get(historico_mp25, 24),
        "media_3h":   float(np.mean(mp25_vals[-3:])) if len(mp25_vals) >= 3 else mp25_vals[-1],
        "media_6h":   float(np.mean(mp25_vals[-6:])) if len(mp25_vals) >= 6 else mp25_vals[-1],
        "diff_1h":    float(mp25_vals[-1] - mp25_vals[-2]) if len(mp25_vals) >= 2 else 0.0,
    }
    return np.array([[row[f] for f in FEATURES_MODELO]])

def gerar_forecast(pivot, model, horas=24):
    """Gera previsão de MP2.5 para as próximas `horas` horas usando o modelo XGBoost."""
    # Trabalha com séries históricas — renomeia MP2.5 → MP25
    hist = pivot.rename(columns=RENAME_PARA_MODELO).copy().ffill().bfill()

    poluentes_necessarios = ["MP25", "MP10", "O3", "NO2"]
    for p in poluentes_necessarios:
        if p not in hist.columns:
            hist[p] = 0.0

    mp25 = hist["MP25"].copy()
    mp10 = hist["MP10"].copy()
    o3   = hist["O3"].copy()
    no2  = hist["NO2"].copy()

    preds = []
    timestamps = []

    for i in range(horas):
        prox_ts = mp25.index[-1] + timedelta(hours=1)
        X = preparar_features_forecast(mp25, mp10, o3, no2, prox_ts)
        pred = float(max(model.predict(X)[0], 0))

        # appenda previsão ao histórico para alimentar próximos lags
        mp25 = pd.concat([mp25, pd.Series([pred], index=[prox_ts])])

        preds.append(pred)
        timestamps.append(prox_ts)

    return pd.Series(preds, index=timestamps, name="MP2.5")

# ================================
# CARREGAR DADOS
# ================================

st.title("🌎 Qualidade do Ar — UBS Vila Curuça")
st.caption(f"Dados em tempo real · CETESB Itaim Paulista · Atualizado: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

with st.spinner("Carregando dados da CETESB..."):
    df = coletar_dados()

if df.empty:
    st.error("Não foi possível carregar dados da CETESB. Tente novamente em alguns minutos.")
    st.stop()

pivot = preparar_pivot(df)

# ================================
# SITUAÇÃO ATUAL — MÉTRICAS
# ================================

st.subheader("📊 Situação Atual")

poluentes_display = ["MP2.5", "MP10", "O3", "NO2", "CO", "SO2"]
cols_ui = st.columns(6)

iqas_atuais = {}

for i, pol in enumerate(poluentes_display):
    if pol in pivot.columns:
        valor = pivot[pol].dropna().iloc[-1] if not pivot[pol].dropna().empty else None
        iqa = calcular_iqa_poluente(valor, pol) if valor is not None else None
        if iqa is not None:
            nome, cor, emoji = classificar_iqa(iqa)
            iqas_atuais[pol] = iqa
            cols_ui[i].metric(
                label=f"{emoji} {pol}",
                value=f"IQA {iqa:.0f}",
                delta=f"{valor:.1f} µg/m³" if pol not in ["CO"] else f"{valor:.0f} µg/m³",
                delta_color="off"
            )
        else:
            cols_ui[i].metric(label=f"⚪ {pol}", value="Sem dado")
    else:
        cols_ui[i].metric(label=f"⚪ {pol}", value="Offline")

# ================================
# IQA GERAL
# ================================

if iqas_atuais:
    iqa_geral = max(iqas_atuais.values())
    pol_critico = max(iqas_atuais, key=iqas_atuais.get)
    nome_classe, cor_classe, emoji_classe = classificar_iqa(iqa_geral)
    rec_texto, rec_emoji = RECOMENDACOES[nome_classe]

    st.markdown(f"""
    <div class="iqa-banner" style="
        background: {cor_classe}22;
        border-left: 6px solid {cor_classe};
        border-radius: 8px;
        padding: 18px 24px;
        margin: 16px 0;
    ">
        <div style="font-size: 2rem; font-weight: bold; color: {cor_classe}">
            {emoji_classe} IQA Geral: {iqa_geral:.0f} — {nome_classe}
        </div>
        <div style="font-size: 1rem; margin-top: 6px; color: #444;">
            Poluente crítico: <b>{pol_critico}</b> &nbsp;|&nbsp; {rec_emoji} {rec_texto}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if nome_classe in ["Ruim", "Muito Ruim", "Péssima"]:
        st.error(f"⚠️ ALERTA: Qualidade do ar {nome_classe.upper()} em Itaim Paulista. {rec_texto}")

# ================================
# HISTÓRICO
# ================================

st.subheader("📈 Histórico — Últimas 48h")

horas_hist = 48
cols_disponiveis = [p for p in poluentes_display if p in pivot.columns]

if cols_disponiveis:
    df_hist = pivot[cols_disponiveis].tail(horas_hist).reset_index()
    df_hist_long = df_hist.melt(id_vars="datahora", var_name="Poluente", value_name="Concentração")

    fig_hist = px.line(
        df_hist_long,
        x="datahora",
        y="Concentração",
        color="Poluente",
        markers=False,
        title="Concentração dos Poluentes (µg/m³)",
    )
    fig_hist.update_layout(
        autosize=True,
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode="x unified",
        xaxis_title="Data/Hora",
        yaxis_title="Concentração (µg/m³)",
        legend_title="Poluente",
        height=None,
    )
    st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.warning("Sem dados históricos suficientes para exibir.")

# ================================
# FORECAST 24H
# ================================

st.subheader("🔮 Previsão MP2.5 — Próximas 24h")
st.caption("Modelo XGBoost treinado com dados históricos da CETESB · Itaim Paulista")

model_xgb, features_xgb = carregar_modelo()

if model_xgb is None:
    st.warning("⚠️ Arquivo `modelo_xgboost.pkl` não encontrado na pasta do app. Coloque o arquivo junto com o `app.py` e reinicie.")
else:
    with st.spinner("Gerando previsão com XGBoost..."):
        serie_forecast = gerar_forecast(pivot, model_xgb, horas=24)

    # IQA previsto
    iqa_forecast = serie_forecast.apply(lambda v: calcular_iqa_poluente(v, "MP2.5"))

    # Gráfico concentração prevista MP2.5
    fig_fc = go.Figure()
    # histórico recente (últimas 24h) para contexto
    if "MP2.5" in pivot.columns:
        hist_recente = pivot["MP2.5"].tail(24)
        fig_fc.add_trace(go.Scatter(
            x=hist_recente.index, y=hist_recente.values,
            mode="lines", name="MP2.5 histórico",
            line=dict(color="#1976D2", width=2)
        ))
    fig_fc.add_trace(go.Scatter(
        x=serie_forecast.index, y=serie_forecast.values,
        mode="lines+markers", name="MP2.5 previsto",
        line=dict(color="#FF6D00", width=2, dash="dash"),
        marker=dict(size=5)
    ))
    fig_fc.update_layout(
        autosize=True,
        margin=dict(l=40, r=40, t=50, b=40),
        title="Concentração MP2.5 — Histórico + Previsão 24h",
        hovermode="x unified",
        xaxis_title="Data/Hora",
        yaxis_title="MP2.5 (µg/m³)",
        height=None,
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # Gráfico IQA previsto
    fig_iqa = go.Figure()
    fig_iqa.add_trace(go.Scatter(
        x=iqa_forecast.index,
        y=iqa_forecast.values,
        mode="lines+markers",
        name="IQA MP2.5 Previsto",
        line=dict(color="#FF6D00", width=3),
        fill="tozeroy",
        fillcolor="rgba(255,109,0,0.1)"
    ))
    for (lo, hi, nome, cor, _) in CLASSES_IQA:
        fig_iqa.add_hrect(y0=lo, y1=min(hi, 300), fillcolor=cor, opacity=0.07,
                          line_width=0, annotation_text=nome, annotation_position="right")
    fig_iqa.update_layout(
        autosize=True,
        margin=dict(l=40, r=40, t=50, b=40),
        title="IQA MP2.5 Previsto — Próximas 24h",
        xaxis_title="Data/Hora",
        yaxis_title="IQA",
        height=None,
        yaxis=dict(range=[0, 250])
    )
    st.plotly_chart(fig_iqa, use_container_width=True)

    # Alerta futuro
    iqa_max_futuro = iqa_forecast.max()
    ts_max = iqa_forecast.idxmax()
    nome_fut, cor_fut, emoji_fut = classificar_iqa(iqa_max_futuro)
    rec_fut, _ = RECOMENDACOES[nome_fut]
    st.info(f"{emoji_fut} Pior momento previsto: **IQA {iqa_max_futuro:.0f} ({nome_fut})** às {ts_max.strftime('%d/%m %H:%M')} · {rec_fut}")

# ================================
# TABELA RESUMO
# ================================

with st.expander("📋 Ver dados brutos"):
    st.dataframe(pivot.tail(48).style.format("{:.2f}"), use_container_width=True)

    csv = pivot.tail(48).to_csv().encode("utf-8")
    st.download_button("📥 Baixar CSV", csv, "itaim_paulista_48h.csv", "text/csv")


