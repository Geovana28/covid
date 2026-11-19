"""
app.py — Plataforma de Inteligência & Vigilância Epidemiológica da COVID-19
Navegação dual: Brasil (Nacional/Estados) & Global (Panorama Mundial).
Desenvolvido com Streamlit, Plotly e Modelagem Matemática SIR / Verhulst EDO.
"""

import importlib
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(__file__))

import data_loader
importlib.reload(data_loader)
from data_loader import (
    load_epidemiological_waves_timeseries,
    load_global_comparison_timeseries,
    load_global_countries_data,
    load_uf_summary_data,
)

import sir_model
importlib.reload(sir_model)
from sir_model import simular_sir

import logistic_model
importlib.reload(logistic_model)
from logistic_model import ajustar_modelo_logistico

ACCENT = "#14b8a6"
ACCENT_ALERT = "#f43f5e"
SERIES_PALETTE = ["#14b8a6", "#f4f4f6", "#a1a1aa", "#f59e0b", "#38bdf8", "#f43f5e"]

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="COVID-19 Intelligence — Brazil & Global Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- MONOCHROME STRUCTURAL ENGINEERING DESIGN SYSTEM ---
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    .main .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }}
    .stApp {{
        background-color: #0b0b0d !important;
        color: #f4f4f6 !important;
    }}
    header[data-testid="stHeader"] {{
        background-color: #0b0b0d !important;
        border-bottom: 1px solid #18181c;
    }}
    section[data-testid="stSidebar"] {{
        background-color: #121215 !important;
        border-right: 1px solid #242429;
    }}
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li {{
        color: #a1a1aa !important;
        font-size: 0.82rem;
    }}
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {{
        color: #f4f4f6 !important;
        font-size: 0.8rem !important;
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.4rem !important;
    }}
    h1, h2, h3, h4 {{
        color: #f4f4f6 !important;
        letter-spacing: -0.02em;
    }}
    /* Inputs, Selectboxes, Radio */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input {{
        background-color: #0b0b0d !important;
        border-color: #323238 !important;
        color: #f4f4f6 !important;
        border-radius: 4px !important;
    }}
    [data-testid="stRadio"] label p,
    [data-testid="stWidgetLabel"] p {{
        color: #d4d4d8 !important;
        font-size: 0.84rem !important;
    }}
    /* Architectural Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        border-bottom: 1px solid #242429;
        background: transparent !important;
        padding: 0;
    }}
    button[data-baseweb="tab"] {{
        color: #71717a !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        padding: 10px 14px !important;
        background: transparent !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {ACCENT} !important;
    }}
    div[data-baseweb="tab-highlight"] {{
        background-color: {ACCENT} !important;
    }}
    /* Buttons */
    .stButton > button {{
        background-color: #121215 !important;
        color: #e4e4e7 !important;
        border: 1px solid #323238 !important;
        border-radius: 4px !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }}
    .stButton > button:hover {{
        background-color: #18181c !important;
        border-color: {ACCENT} !important;
        color: #f4f4f6 !important;
    }}
    /* Header & KPI Strip */
    .page-header {{
        border-bottom: 1px solid #242429;
        padding-bottom: 14px;
        margin-bottom: 20px;
    }}
    .page-eyebrow {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #71717a;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }}
    .page-eyebrow span {{
        color: {ACCENT};
    }}
    .page-title {{
        margin: 4px 0 6px 0;
        font-size: 1.75rem;
        font-weight: 600;
        color: #f4f4f6;
        letter-spacing: -0.02em;
    }}
    .page-subtitle {{
        color: #a1a1aa;
        font-size: 0.9rem;
        margin: 0;
    }}
    .kpi-strip {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        background: #121215;
        border: 1px solid #242429;
        border-radius: 6px;
        margin-bottom: 20px;
        overflow: hidden;
    }}
    @media (max-width: 900px) {{
        .kpi-strip {{ grid-template-columns: repeat(2, 1fr); }}
    }}
    .kpi-cell {{
        padding: 15px 18px;
        border-right: 1px solid #242429;
    }}
    .kpi-cell:last-child {{ border-right: none; }}
    .kpi-lbl {{
        font-size: 0.69rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #71717a;
        font-weight: 500;
    }}
    .kpi-num {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.38rem;
        font-weight: 600;
        color: #f4f4f6;
        margin: 5px 0 3px 0;
        white-space: nowrap;
    }}
    .kpi-num.accent {{ color: {ACCENT}; }}
    .kpi-num.alert {{ color: {ACCENT_ALERT}; }}
    .kpi-meta {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #a1a1aa;
    }}
    .section-note {{
        color: #a1a1aa;
        font-size: 0.88rem;
        line-height: 1.55;
        margin-bottom: 14px;
    }}
    .ref-panel {{
        background: #121215;
        border: 1px solid #242429;
        border-radius: 6px;
        padding: 16px 20px;
        margin-top: 18px;
    }}
    .ref-panel h4 {{
        margin: 0 0 8px 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {ACCENT} !important;
    }}
    .ref-panel p {{
        font-size: 0.86rem;
        color: #a1a1aa;
        margin-bottom: 8px;
        line-height: 1.55;
    }}
    .ref-panel code {{
        background: #0b0b0d;
        border: 1px solid #242429;
        color: #f4f4f6;
        padding: 2px 6px;
        border-radius: 3px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
    }}
    div[data-testid="stDataFrame"] {{
        border: 1px solid #242429;
        border-radius: 6px;
        overflow: hidden;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def render_kpi_strip(items):
    """Render a unified monochrome KPI strip with zero truncation and zero pill arrows."""
    cells = []
    for label, value, meta, tone in items:
        cls = f"kpi-num {tone}".strip()
        cells.append(
            f'<div class="kpi-cell">'
            f'<div class="kpi-lbl">{label}</div>'
            f'<div class="{cls}">{value}</div>'
            f'<div class="kpi-meta">{meta}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="kpi-strip">{"".join(cells)}</div>', unsafe_allow_html=True)


def apply_dark_layout(fig, height=460, title=None, xaxis_title=None, yaxis_title=None):
    """Apply consistent monochrome dark layout to all Plotly charts."""
    layout_kwargs = dict(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#121215",
        font=dict(family="Inter, -apple-system, sans-serif", size=12, color="#a1a1aa"),
        margin=dict(l=16, r=16, t=44 if title else 24, b=24),
        hoverlabel=dict(
            bgcolor="#121215",
            bordercolor="#323238",
            font=dict(color="#f4f4f6", family="JetBrains Mono, monospace", size=11),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color="#a1a1aa"),
        ),
        xaxis=dict(gridcolor="#242429", zerolinecolor="#323238", linecolor="#242429"),
        yaxis=dict(gridcolor="#242429", zerolinecolor="#323238", linecolor="#242429"),
    )
    if title:
        layout_kwargs["title"] = dict(text=title, font=dict(size=14, color="#f4f4f6"), x=0, xanchor="left")
    if xaxis_title:
        layout_kwargs["xaxis_title"] = xaxis_title
    if yaxis_title:
        layout_kwargs["yaxis_title"] = yaxis_title
    fig.update_layout(**layout_kwargs)
    return fig


# --- CARREGAMENTO DE DADOS COM CACHE ---
@st.cache_data
def get_all_data():
    df_uf = load_uf_summary_data()
    df_series = load_epidemiological_waves_timeseries()
    df_global = load_global_countries_data()
    df_comp = load_global_comparison_timeseries()
    return df_uf, df_series, df_global, df_comp


df_uf, df_series, df_global, df_comp = get_all_data()

# --- SIDEBAR: SELETOR DE ESCOPO ---
with st.sidebar:
    st.markdown("### 01 / COVID-19 Intelligence")
    st.caption("Vigilância epidemiológica, inferência vacinal e modelagem diferencial (SIR + Verhulst EDO).")

    st.markdown("---")
    st.markdown("#### 02 / Escopo Geográfico")
    modo_escopo = st.radio(
        "Selecione a visualização:",
        options=["Brasil (Nacional & Estados)", "Global (Panorama Mundial)"],
        index=0,
    )

    if "Brasil" in modo_escopo:
        st.markdown("---")
        st.markdown("#### 03 / Filtro Regional")
        regiao_sel = st.selectbox(
            "Região Brasileira:",
            options=["Brasil (Todas)", "Sudeste", "Nordeste", "Sul", "Norte", "Centro-Oeste"],
            index=0,
        )
        if regiao_sel != "Brasil (Todas)":
            df_uf_filtered = df_uf[df_uf["regiao"] == regiao_sel]
        else:
            df_uf_filtered = df_uf.copy()
    else:
        st.markdown("---")
        st.markdown("#### 03 / Filtro Continental")
        continente_sel = st.selectbox(
            "Continente:",
            options=["Todos os Continentes", "América do Sul", "América do Norte", "Europa", "Ásia", "África", "Oceania"],
            index=0,
        )
        if continente_sel != "Todos os Continentes":
            df_global_filtered = df_global[df_global["continente"].str.contains(continente_sel)]
        else:
            df_global_filtered = df_global.copy()

    st.markdown("---")
    st.markdown("#### 04 / Metodologia")
    st.markdown(
        """
        - **Fontes:** OMS, Ministério da Saúde e Our World in Data.
        - **Modelos:** EDO Compartimental SIR e Logística de Verhulst (UFMG).
        - **Métrica:** Descolamento Casos × Óbitos pós-vacinação.
        """
    )
    st.markdown("---")
    st.caption("Geovana Moreira • Python, Streamlit, Plotly & SciPy")

# ==============================================================================
# VISÃO 1: BRASIL (ESTADOS E NACIONAL)
# ==============================================================================
if "Brasil" in modo_escopo:
    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-eyebrow"><span>BR-27</span> // VIGILÂNCIA EPIDEMIOLÓGICA & MODELAGEM DIFERENCIAL (SIR + VERHULST EDO)</div>
            <h1 class="page-title">Brasil — Análise Epidemiológica da COVID-19</h1>
            <p class="page-subtitle">Exploração das 27 UFs, séries temporais das variantes (Ancestral, Gama, Delta, Ômicron), efeito causal da vacinação e simulação matemática.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tot_casos = df_uf["casos_acumulados"].sum()
    tot_obitos = df_uf["obitos_acumulados"].sum()
    letalidade_media = (tot_obitos / tot_casos) * 100.0
    cobertura_media = df_uf["vacina_2doses_pct"].mean()

    render_kpi_strip(
        [
            ("Casos Notificados", f"{tot_casos:,.0f}".replace(",", "."), "Brasil acumulado (27 UFs)", ""),
            ("Óbitos Confirmados", f"{tot_obitos:,.0f}".replace(",", "."), "Série histórica consolidada", "alert"),
            ("Taxa de Letalidade", f"{letalidade_media:.2f}%", "Queda de ~90% pós-vacina", ""),
            ("Cobertura Vacinal (2 Doses)", f"{cobertura_media:.1f}%", "Média nacional (meta OMS)", "accent"),
        ]
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "01 / Ondas & Variantes",
            "02 / Distribuição UFs",
            "03 / Efeito Vacinal",
            "04 / Simulador SIR",
            "05 / Projeção Logística EDO",
        ]
    )

    with tab1:
        st.subheader("Dinâmica Temporal das Ondas e Descolamento de Curvas")
        st.markdown(
            '<p class="section-note">Observe o <b>descolamento histórico</b>: na 1ª e 2ª ondas (Ancestral e Gama), o pico de casos acompanhava diretamente o pico de óbitos. A partir de 2022 (<b>Ômicron</b>), os casos explodiram enquanto os óbitos permaneceram contidos pela cobertura vacinal.</p>',
            unsafe_allow_html=True,
        )

        col_t1, col_t2 = st.columns([3, 1])
        with col_t2:
            metrica_tempo = st.radio("Agregação temporal:", ["Média Móvel (7 dias)", "Totais Semanais"], horizontal=False)

        y_casos = "casos_diarios_mm7" if metrica_tempo == "Média Móvel (7 dias)" else "casos_semanais"
        y_obitos = "obitos_diarios_mm7" if metrica_tempo == "Média Móvel (7 dias)" else "obitos_semanais"

        fig_waves = go.Figure()
        fig_waves.add_trace(
            go.Scatter(
                x=df_series["data"],
                y=df_series[y_casos],
                name="Casos Notificados",
                line=dict(color=ACCENT, width=2.2),
                yaxis="y1",
            )
        )
        fig_waves.add_trace(
            go.Scatter(
                x=df_series["data"],
                y=df_series[y_obitos],
                name="Óbitos Confirmados",
                line=dict(color=ACCENT_ALERT, width=2.2),
                yaxis="y2",
            )
        )

        vrect_font = dict(color="#a1a1aa", size=10, family="JetBrains Mono, monospace")
        fig_waves.add_vrect(x0="2020-03-01", x1="2021-01-01", fillcolor="#ffffff", opacity=0.025, line_width=0, annotation_text="Ancestral", annotation_position="top left", annotation_font=vrect_font)
        fig_waves.add_vrect(x0="2021-01-01", x1="2021-07-01", fillcolor="#f43f5e", opacity=0.06, line_width=0, annotation_text="Gama (P.1)", annotation_position="top left", annotation_font=vrect_font)
        fig_waves.add_vrect(x0="2021-07-01", x1="2021-12-15", fillcolor="#f59e0b", opacity=0.05, line_width=0, annotation_text="Delta", annotation_position="top left", annotation_font=vrect_font)
        fig_waves.add_vrect(x0="2021-12-15", x1="2023-03-01", fillcolor="#14b8a6", opacity=0.05, line_width=0, annotation_text="Ômicron", annotation_position="top left", annotation_font=vrect_font)

        apply_dark_layout(fig_waves, height=470)
        fig_waves.update_layout(
            yaxis=dict(title=dict(text="Casos", font=dict(color=ACCENT)), tickfont=dict(color=ACCENT), gridcolor="#242429"),
            yaxis2=dict(title=dict(text="Óbitos", font=dict(color=ACCENT_ALERT)), tickfont=dict(color=ACCENT_ALERT), overlaying="y", side="right", showgrid=False),
        )
        st.plotly_chart(fig_waves, use_container_width=True)

    with tab2:
        st.subheader(f"Dispersão Espacial dos Indicadores — {regiao_sel}")
        col_m1, col_m2 = st.columns([2, 1])
        with col_m1:
            indicador_mapa = st.selectbox(
                "Métrica para dimensionar os pontos:",
                ["Mortalidade por 100 mil hab", "Taxa de Letalidade (%)", "Total de Óbitos Acumulados"],
            )
        with col_m2:
            map_styles = {
                "Dark (OpenFreeMap)": "https://tiles.openfreemap.org/styles/dark",
                "Light (OpenFreeMap)": "https://tiles.openfreemap.org/styles/positron",
                "OpenStreetMap": "open-street-map",
            }
            map_style_label = st.selectbox("Estilo do mapa:", list(map_styles.keys()))
            map_style = map_styles[map_style_label]

        m_col = "mortalidade_100k" if "Mortalidade" in indicador_mapa else ("letalidade_pct" if "Letalidade" in indicador_mapa else "obitos_acumulados")
        scatter_fn = getattr(px, "scatter_map", None) or getattr(px, "scatter_mapbox")
        map_kwargs = {
            "lat": "lat",
            "lon": "lon",
            "size": m_col,
            "color": "status_risco",
            "color_discrete_map": {
                "Crítico (Histórico Alto)": "#f43f5e",
                "Alerta (Moderado)": "#f59e0b",
                "Controlado": "#14b8a6",
            },
            "hover_name": "estado",
            "custom_data": ["uf", "incidencia_100k", "mortalidade_100k", "letalidade_pct", "vacina_2doses_pct"],
            "zoom": 3.6,
            "center": {"lat": -14.2350, "lon": -51.9253},
            "size_max": 34,
            "template": "plotly_dark",
        }
        if hasattr(px, "scatter_map"):
            map_kwargs["map_style"] = map_style
        else:
            map_kwargs["mapbox_style"] = "open-street-map"

        fig_uf_map = scatter_fn(df_uf_filtered, **map_kwargs)
        fig_uf_map.update_traces(
            hovertemplate="<b>%{hovertext} (%{customdata[0]})</b><br>Incidência: <b>%{customdata[1]:,.0f}</b>/100k<br>Mortalidade: <b>%{customdata[2]:.1f}</b>/100k<br>Letalidade: <b>%{customdata[3]:.2f}%</b><br>Vacinação: <b>%{customdata[4]:.1f}%</b><extra></extra>"
        )
        fig_uf_map.update_layout(
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#121215",
            margin=dict(l=0, r=0, t=8, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(color="#a1a1aa")),
        )
        st.plotly_chart(fig_uf_map, use_container_width=True)

        st.dataframe(
            df_uf_filtered[
                ["uf", "estado", "regiao", "populacao", "casos_acumulados", "obitos_acumulados", "mortalidade_100k", "letalidade_pct", "vacina_2doses_pct"]
            ].rename(
                columns={
                    "uf": "UF",
                    "estado": "Estado",
                    "regiao": "Região",
                    "populacao": "População",
                    "casos_acumulados": "Casos",
                    "obitos_acumulados": "Óbitos",
                    "mortalidade_100k": "Mortalidade/100k",
                    "letalidade_pct": "Letalidade (%)",
                    "vacina_2doses_pct": "Vacinação (%)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with tab3:
        st.subheader("Correlação Estatística — Cobertura Vacinal vs. Desaceleração de Óbitos")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            fig_scatter_vax = px.scatter(
                df_uf,
                x="vacina_2doses_pct",
                y="letalidade_pct",
                color="regiao",
                size="populacao",
                hover_name="estado",
                trendline="ols",
                color_discrete_sequence=SERIES_PALETTE,
                labels={"vacina_2doses_pct": "Cobertura Vacinal Completa (%)", "letalidade_pct": "Taxa de Letalidade (%)", "regiao": "Região"},
            )
            apply_dark_layout(fig_scatter_vax, height=430, title="Cobertura Vacinal (%) × Taxa de Letalidade (%)")
            st.plotly_chart(fig_scatter_vax, use_container_width=True)
        with col_v2:
            fig_vax_time = px.line(
                df_series,
                x="data",
                y=["cobertura_vacinal_pct", "taxa_letalidade_semanal"],
                color_discrete_sequence=[ACCENT, ACCENT_ALERT],
                labels={"data": "Data", "value": "Percentual (%)", "variable": "Indicador"},
            )
            apply_dark_layout(fig_vax_time, height=430, title="Evolução Temporal — Cobertura Vacinal vs. Letalidade")
            st.plotly_chart(fig_vax_time, use_container_width=True)

    with tab4:
        st.subheader("Simulador Epidemiológico de Intervenções — Modelo SIR")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            r0_input = st.slider("Taxa de Reprodução Básica (R0):", 1.1, 5.0, 2.6, 0.1)
            duracao_infeccao = st.slider("Tempo de Infecção (Dias):", 5, 21, 10)
        with col_s2:
            intervencao_input = st.slider("Aderência a Máscaras / Distanciamento (%):", 0, 80, 25, 5)
            cobertura_sim_vax = st.slider("População Vacinada no Início (%):", 0, 90, 40, 5)
        with col_s3:
            pop_sim = st.number_input("Tamanho da População Alvo:", value=5000000, step=500000)
            capacidade_leitos = st.number_input("Capacidade de Leitos UTI:", value=3000, step=500)

        resultado_sir = simular_sir(
            populacao_total=int(pop_sim),
            infectados_iniciais=150,
            dias_simulacao=180,
            r0_base=r0_input,
            tempo_recuperacao_dias=float(duracao_infeccao),
            reducao_contato_pct=float(intervencao_input),
            cobertura_vacinal_pct=float(cobertura_sim_vax),
        )

        pico_uti = int(np.max(resultado_sir["Necessidade_UTI"]))
        uti_status = "Sobrecarga acima do limite" if pico_uti > capacidade_leitos else "Dentro da capacidade UTI"
        uti_tone = "alert" if pico_uti > capacidade_leitos else "accent"

        render_kpi_strip(
            [
                ("R Efetivo (Rt)", f"{resultado_sir['r_efetivo']:.2f}", "Limiar epidêmico = 1.00", "alert" if resultado_sir["r_efetivo"] > 1.0 else "accent"),
                ("Pico de Infectados", f"{resultado_sir['pico_infectados']:,}".replace(",", "."), "Indivíduos simultâneos", ""),
                ("Pico Demanda UTI", f"{pico_uti:,}".replace(",", "."), uti_status, uti_tone),
                ("População Atingida", f"{resultado_sir['pct_populacao_atingida']:.1f}%", "Taxa de ataque acumulada", ""),
            ]
        )

        fig_sir = go.Figure()
        fig_sir.add_trace(go.Scatter(x=resultado_sir["dias"], y=resultado_sir["Suscetíveis"], name="Suscetíveis (S)", line=dict(color="#a1a1aa", width=2)))
        fig_sir.add_trace(go.Scatter(x=resultado_sir["dias"], y=resultado_sir["Infectados"], name="Infectados Ativos (I)", line=dict(color=ACCENT_ALERT, width=2.8)))
        fig_sir.add_trace(go.Scatter(x=resultado_sir["dias"], y=resultado_sir["Recuperados_Imunes"], name="Recuperados (R)", line=dict(color=ACCENT, width=2.2)))
        fig_sir.add_trace(go.Scatter(x=resultado_sir["dias"], y=resultado_sir["Necessidade_UTI"], name="Demanda UTI", line=dict(color="#f59e0b", width=2.2, dash="dot")))
        fig_sir.add_hline(
            y=capacidade_leitos,
            line_dash="dash",
            line_color="#f43f5e",
            annotation_text=f"Capacidade UTI ({capacidade_leitos:,})",
            annotation_font=dict(color="#f43f5e", size=11, family="JetBrains Mono, monospace"),
        )
        apply_dark_layout(fig_sir, height=450, title="Curva Epidêmica Dinâmica — Modelo Compartimental SIR", xaxis_title="Dias", yaxis_title="Indivíduos")
        st.plotly_chart(fig_sir, use_container_width=True)

    with tab5:
        st.subheader("Modelo de Crescimento Logístico (EDO) & Mínimos Quadrados")
        st.markdown(
            '<p class="section-note">Implementação analítica da <b>Equação Diferencial Logística de Verhulst</b>, adaptando a metodologia do Prof. Dr. Reginaldo J. Santos (Departamento de Matemática — UFMG).</p>',
            unsafe_allow_html=True,
        )

        col_lg1, col_lg2 = st.columns([2, 1])
        with col_lg1:
            recorte_onda = st.selectbox(
                "Recorte epidemiológico para calibração:",
                options=[
                    "Onda 1 (Ancestral: Mar/2020 a Nov/2020)",
                    "Onda 2 (Gama/P.1: Dez/2020 a Jul/2021)",
                    "Onda 4 (Ômicron: Dez/2021 a Mai/2022)",
                    "Série Histórica Completa (2020 a 2023)",
                ],
                index=0,
            )
        with col_lg2:
            dias_proj = st.slider("Horizonte de Projeção Futura (semanas):", min_value=4, max_value=24, value=12, step=2)

        if "Onda 1" in recorte_onda:
            df_sub = df_series[(df_series["data"] >= "2020-03-01") & (df_series["data"] <= "2020-11-30")].copy()
        elif "Onda 2" in recorte_onda:
            df_sub = df_series[(df_series["data"] >= "2020-12-01") & (df_series["data"] <= "2021-07-31")].copy()
        elif "Onda 4" in recorte_onda:
            df_sub = df_series[(df_series["data"] >= "2021-12-01") & (df_series["data"] <= "2022-05-31")].copy()
        else:
            df_sub = df_series.copy()

        df_sub["casos_acumulados_recorte"] = df_sub["casos_semanais"].cumsum()

        res_logistico = ajustar_modelo_logistico(
            datas=df_sub["data"],
            casos_acumulados=df_sub["casos_acumulados_recorte"],
            dias_projecao=dias_proj,
        )

        dia_inflexao = res_logistico["t_inflexao_dia"]
        data_inflexao_str = res_logistico["datas"][min(dia_inflexao, len(res_logistico["datas"]) - 1)].strftime("%d/%m/%Y")

        render_kpi_strip(
            [
                ("Teto Projetado (yM)", f"{res_logistico['yM_capacidade_maxima']:,.0f}".replace(",", "."), "Capacidade máxima assintótica", ""),
                ("Acurácia (R² Ajuste)", f"{res_logistico['r2_ajuste'] * 100:.2f}%", "Mínimos quadrados não-linear", "accent"),
                ("Taxa de Propagação (r)", f"{res_logistico['taxa_r']:.4f}", f"K = {res_logistico['constante_K_edo']:.2e}", ""),
                ("Ponto de Inflexão", f"Semana {dia_inflexao}", data_inflexao_str, ""),
            ]
        )

        fig_log = go.Figure()
        fig_log.add_trace(
            go.Scatter(
                x=df_sub["data"],
                y=df_sub["casos_acumulados_recorte"],
                mode="markers",
                name="Casos Observados",
                marker=dict(color="#f4f4f6", size=6, symbol="circle", line=dict(width=1, color="#323238")),
                hovertemplate="<b>Real</b><br>Data: %{x|%d/%m/%Y}<br>Casos Acumulados: %{y:,.0f}<extra></extra>",
            )
        )

        n_hist = res_logistico["n_historico"]
        datas_hist = res_logistico["datas"][:n_hist]
        curva_hist = res_logistico["curva_ajustada_e_projetada"][:n_hist]
        fig_log.add_trace(
            go.Scatter(
                x=datas_hist,
                y=curva_hist,
                mode="lines",
                name="Ajuste Logístico EDO (Treino)",
                line=dict(color=ACCENT, width=2.6),
                hovertemplate="<b>Ajuste EDO</b><br>Data: %{x|%d/%m/%Y}<br>Casos Ajustados: %{y:,.0f}<extra></extra>",
            )
        )

        datas_fut = res_logistico["datas"][n_hist - 1 :]
        curva_fut = res_logistico["curva_ajustada_e_projetada"][n_hist - 1 :]
        fig_log.add_trace(
            go.Scatter(
                x=datas_fut,
                y=curva_fut,
                mode="lines",
                name=f"Projeção Futura (+{dias_proj} semanas)",
                line=dict(color="#f59e0b", width=2.4, dash="dash"),
                hovertemplate="<b>Projeção</b><br>Data: %{x|%d/%m/%Y}<br>Projeção Teórica: %{y:,.0f}<extra></extra>",
            )
        )

        fig_log.add_hline(
            y=res_logistico["yM_capacidade_maxima"],
            line_dash="dot",
            line_color="#a1a1aa",
            annotation_text=f"Platô Teórico y_M ({res_logistico['yM_capacidade_maxima']:,.0f})",
            annotation_position="bottom right",
            annotation_font=dict(color="#a1a1aa", size=11, family="JetBrains Mono, monospace"),
        )

        apply_dark_layout(
            fig_log,
            height=470,
            title=f"Ajuste Logístico & Projeção Assintótica — {recorte_onda}",
            xaxis_title="Linha do Tempo",
            yaxis_title="Casos Acumulados",
        )
        st.plotly_chart(fig_log, use_container_width=True)

        st.markdown(
            """
            <div class="ref-panel">
                <h4>Fundamentação Matemática & Referência Acadêmica</h4>
                <p>
                    O crescimento populacional e epidemiológico com limitação de recursos segue a <b>Equação Diferencial de Verhulst</b>:<br>
                    <code>dy/dt = K · y · (y_M - y)</code> com solução analítica <code>y(t) = y_M / [1 + ((y_M - y_0)/y_0) · e^(-y_M · K · t)]</code>.
                </p>
                <p style="margin-bottom:0; color:#71717a;">
                    <b>Referência (ABNT):</b> SANTOS, Reginaldo J. <i>Um Modelo para o Surto de Coronavírus no Brasil</i>. Departamento de Matemática, ICEx, Universidade Federal de Minas Gerais (UFMG), Belo Horizonte, 2020.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==============================================================================
# VISÃO 2: GLOBAL (MUNDO / PAÍSES)
# ==============================================================================
else:
    st.markdown(
        """
        <div class="page-header">
            <div class="page-eyebrow"><span>GLOBAL-OWID</span> // VIGILÂNCIA COMPARATIVA INTERNACIONAL & IMUNIZAÇÃO</div>
            <h1 class="page-title">Panorama Global da Pandemia de COVID-19</h1>
            <p class="page-subtitle">Comparativo internacional de mortalidade, mapa-múndi de incidência, sincronia de ondas e assimetria de acesso a vacinas.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    g_casos = df_global["casos_totais"].sum()
    g_obitos = df_global["obitos_totais"].sum()
    g_letalidade = (g_obitos / g_casos) * 100.0
    g_vacina = df_global["vacina_pct"].mean()

    render_kpi_strip(
        [
            ("Casos (Amostra Global)", f"{g_casos:,.0f}".replace(",", "."), "Países monitorados", ""),
            ("Óbitos Confirmados", f"{g_obitos:,.0f}".replace(",", "."), "Consolidado internacional", "alert"),
            ("Letalidade Média", f"{g_letalidade:.2f}%", "Referência mundial", ""),
            ("Vacinação Completa", f"{g_vacina:.1f}%", "Forte assimetria entre blocos", "accent"),
        ]
    )

    gtab1, gtab2, gtab3 = st.tabs(
        [
            "01 / Mapa-Múndi",
            "02 / Comparador Global",
            "03 / Desigualdade Vacinal",
        ]
    )

    with gtab1:
        st.subheader("Mapeamento Global dos Impactos da COVID-19")
        col_gm1, col_gm2 = st.columns([2, 1])
        with col_gm1:
            metrica_global = st.selectbox(
                "Métrica para o mapa-múndi:",
                ["Óbitos por 1 Milhão de hab", "Casos por 100 mil hab", "Taxa de Letalidade (%)", "Total de Óbitos"],
            )
        with col_gm2:
            tipo_mapa = st.selectbox("Projeção do mapa:", ["natural earth", "orthographic", "equirectangular"])

        map_var = (
            "mortalidade_1m"
            if "1 Milhão" in metrica_global
            else ("incidencia_100k" if "100 mil" in metrica_global else ("letalidade_pct" if "Letalidade" in metrica_global else "obitos_totais"))
        )

        fig_world = px.scatter_geo(
            df_global_filtered,
            locations="codigo",
            color="continente",
            hover_name="pais",
            size=map_var,
            projection=tipo_mapa,
            color_discrete_sequence=SERIES_PALETTE,
            custom_data=["populacao", "casos_totais", "obitos_totais", "mortalidade_1m", "vacina_pct"],
        )
        fig_world.update_traces(
            hovertemplate="<b>%{hovertext}</b><br>População: %{customdata[0]:,.0f}<br>Casos: %{customdata[1]:,.0f}<br>Óbitos: %{customdata[2]:,.0f}<br>Mortalidade/1M: %{customdata[3]:.1f}<br>Vacinação: %{customdata[4]:.1f}%<extra></extra>"
        )
        apply_dark_layout(fig_world, height=520, title=f"Distribuição Geográfica — {metrica_global}")
        fig_world.update_geos(
            bgcolor="rgba(0,0,0,0)",
            landcolor="#18181c",
            lakecolor="#0b0b0d",
            oceancolor="#0b0b0d",
            showocean=True,
            subunitcolor="#242429",
            countrycolor="#323238",
            framecolor="#242429",
        )
        st.plotly_chart(fig_world, use_container_width=True)

        st.dataframe(
            df_global_filtered[
                ["pais", "codigo", "continente", "populacao", "casos_totais", "obitos_totais", "mortalidade_1m", "letalidade_pct", "vacina_pct"]
            ].rename(
                columns={
                    "pais": "País",
                    "codigo": "Código",
                    "continente": "Continente",
                    "populacao": "População",
                    "casos_totais": "Casos Totais",
                    "obitos_totais": "Óbitos Totais",
                    "mortalidade_1m": "Óbitos/1M hab",
                    "letalidade_pct": "Letalidade (%)",
                    "vacina_pct": "Vacinação (%)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with gtab2:
        st.subheader("Comparativo de Curvas de Contágio Entre Países (2020–2023)")
        st.markdown(
            '<p class="section-note">Compare a dinâmica temporal de novos casos por milhão de habitantes para observar o impacto de estratégias nacionais, variantes locais e ritmo de imunização.</p>',
            unsafe_allow_html=True,
        )

        paises_disponiveis = [col for col in df_comp.columns if col != "data"]
        paises_sel = st.multiselect(
            "Selecione os países para comparar:",
            options=paises_disponiveis,
            default=["Brasil", "Estados Unidos", "Reino Unido", "Japão"],
        )

        if paises_sel:
            fig_comp = go.Figure()
            for i, p in enumerate(paises_sel):
                fig_comp.add_trace(
                    go.Scatter(
                        x=df_comp["data"],
                        y=df_comp[p],
                        name=p,
                        line=dict(color=SERIES_PALETTE[i % len(SERIES_PALETTE)], width=2.3),
                        hovertemplate=f"<b>{p}</b><br>Data: %{{x|%d/%m/%Y}}<br>Novos Casos/1M: %{{y:.1f}}<extra></extra>",
                    )
                )

            apply_dark_layout(
                fig_comp,
                height=470,
                xaxis_title="Período Epidemiológico",
                yaxis_title="Novos Casos Bi-Semanais / 1M hab",
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.warning("Selecione pelo menos um país para visualizar a comparação.")

    with gtab3:
        st.subheader("Disparidade Global de Acesso a Vacinas")
        st.markdown(
            '<p class="section-note">Enquanto economias centrais e o Brasil (PNI/SUS) superaram 80% de cobertura vacinal completa, países de menor renda enfrentaram gargalos severos de distribuição.</p>',
            unsafe_allow_html=True,
        )

        df_vax_sorted = df_global.sort_values(by="vacina_pct", ascending=True)
        fig_bar_vax = px.bar(
            df_vax_sorted,
            x="vacina_pct",
            y="pais",
            orientation="h",
            text="vacina_pct",
            labels={"vacina_pct": "Cobertura Vacinal (%)", "pais": "País"},
        )
        fig_bar_vax.update_traces(
            marker_color=[ACCENT if p == "Brasil" else "#52525b" for p in df_vax_sorted["pais"]],
            texttemplate="%{text:.1f}%",
            textposition="outside",
            textfont=dict(color="#a1a1aa", family="JetBrains Mono, monospace", size=11),
        )
        apply_dark_layout(fig_bar_vax, height=520, title="Taxa de Cobertura Vacinal Completa por País (%) — Brasil em destaque")
        fig_bar_vax.update_layout(xaxis=dict(range=[0, 108], gridcolor="#242429"))
        st.plotly_chart(fig_bar_vax, use_container_width=True)

        st.markdown(
            """
            <div class="ref-panel">
                <h4>Nota Epidemiológica</h4>
                <p style="margin-bottom:0;">
                    Países com infraestrutura histórica de vacinação pública universal — como o Programa Nacional de Imunizações (PNI/SUS) no Brasil — superaram economias de maior PIB per capita na adesão vacinal adulta.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
