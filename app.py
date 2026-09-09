"""
app.py — Plataforma de Inteligência & Vigilância Epidemiológica da COVID-19
Navegação dual: Brasil (Nacional/Estados) & Global (Panorama Mundial).
Desenvolvido com Streamlit, Plotly e Modelagem Matemática SIR.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import os
import sys
import importlib

sys.path.append(os.path.dirname(__file__))

import data_loader
importlib.reload(data_loader)
from data_loader import (
    load_uf_summary_data,
    load_epidemiological_waves_timeseries,
    load_global_countries_data,
    load_global_comparison_timeseries
)

import sir_model
importlib.reload(sir_model)
from sir_model import simular_sir

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="COVID-19 Intelligence | Brazil & Global Analytics",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PERSONALIZADO PREMIUM (LIGHT THEME) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], .stMarkdown, p, div {
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
        color: #1E293B;
    }

    .main .block-container {
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .stApp {
        background-color: #F8FAFC !important;
    }

    /* Títulos */
    h1, h2, h3, h4, h5, h6, [data-testid="stHeader"] {
        color: #0F172A !important;
        font-weight: 800 !important;
    }

    /* Cards */
    .glass-card {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
        margin-bottom: 20px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] div {
        color: #F1F5F9 !important;
    }

    /* Metric Cards */
    [data-testid="stMetric"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 14px !important;
        padding: 14px 18px !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 1.65rem !important;
        font-weight: 800 !important;
        color: #DC2626 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: #64748B !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #FFFFFF !important;
        padding: 6px;
        border-radius: 14px;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.92rem;
        color: #475569 !important;
        padding: 0 18px;
        border: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: #DC2626 !important;
        border: 1px solid #B91C1C !important;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.25) !important;
    }
    .stTabs [aria-selected="true"] p, .stTabs [aria-selected="true"] span {
        color: #FFFFFF !important;
    }

    /* Badges */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-red { background: rgba(220, 38, 38, 0.2); border: 1px solid #F87171; color: #FECACA !important; }
    .badge-blue { background: rgba(37, 99, 235, 0.2); border: 1px solid #60A5FA; color: #BFDBFE !important; }
    .badge-green { background: rgba(5, 150, 105, 0.2); border: 1px solid #34D399; color: #A7F3D0 !important; }
</style>
""", unsafe_allow_html=True)

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
    st.markdown("### 🦠 **COVID-19 Intelligence**")
    st.caption("Plataforma Integrada de Vigilância Epidemiológica & Modelagem Matemática")
    
    st.markdown("""
        <div style="margin-top:8px; margin-bottom:14px;">
            <span class="badge-pill badge-red">Séries Temporais</span>
            <span class="badge-pill badge-blue">Modelo SIR</span>
            <span class="badge-pill badge-green">Vacinação</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### 🌐 **Escopo Geográfico**")
    modo_escopo = st.radio(
        "Selecione a visualização:",
        options=["🇧🇷 Brasil (Nacional & Estados)", "🌍 Global (Panorama Mundial)"],
        index=0
    )
    
    if "Brasil" in modo_escopo:
        st.markdown("---")
        st.markdown("#### 🎯 **Filtro Regional (Brasil)**")
        regiao_sel = st.selectbox(
            "Região Brasileira:",
            options=["Brasil (Todas)", "Sudeste", "Nordeste", "Sul", "Norte", "Centro-Oeste"],
            index=0
        )
        if regiao_sel != "Brasil (Todas)":
            df_uf_filtered = df_uf[df_uf["regiao"] == regiao_sel]
        else:
            df_uf_filtered = df_uf.copy()
    else:
        st.markdown("---")
        st.markdown("#### 🎯 **Filtro Continental (Global)**")
        continente_sel = st.selectbox(
            "Continente:",
            options=["Todos os Continentes", "América do Sul", "América do Norte", "Europa", "Ásia", "África", "Oceania"],
            index=0
        )
        if continente_sel != "Todos os Continentes":
            df_global_filtered = df_global[df_global["continente"].str.contains(continente_sel)]
        else:
            df_global_filtered = df_global.copy()

    st.markdown("---")
    st.markdown("#### 🔬 **Metodologia Epidemiológica**")
    st.markdown("""
        - **Fontes Consolidadas:** OMS / Ministério da Saúde / Our World in Data.
        - **Modelo de Difusão:** Equações Diferenciais Ordinárias (ODE) SIR.
        - **Métrica Chave:** Descolamento Casos × Óbitos pós-vacinação.
    """)
    st.markdown("---")
    st.markdown("👩‍💻 **Geovana** • *Full Stack & Data Scientist*")
    st.caption("v2.0 • Python, Streamlit, Plotly & Scipy")

# ==============================================================================
# VISÃO 1: BRASIL (ESTADOS E NACIONAL)
# ==============================================================================
if "Brasil" in modo_escopo:
    st.markdown("""
        <div style="margin-bottom: 24px;">
            <h1 style="font-size:2.3rem; letter-spacing:-0.02em; margin-bottom:4px;">
                🇧🇷 Brasil: Inteligência & Análise Epidemiológica da COVID-19
            </h1>
            <p style="color:#64748B; font-size:1.05rem; font-weight:500;">
                Exploração das 27 UFs, séries temporais das variantes (Gama, Delta, Ômicron), efeito protetor das vacinas e simulação matemática SIR.
            </p>
        </div>
    """, unsafe_allow_html=True)

    tot_casos = df_uf["casos_acumulados"].sum()
    tot_obitos = df_uf["obitos_acumulados"].sum()
    letalidade_media = (tot_obitos / tot_casos) * 100.0
    cobertura_media = df_uf["vacina_2doses_pct"].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total de Casos Notificados", f"{tot_casos:,.0f}".replace(',', '.'), delta="Brasil Acumulado")
    k2.metric("Total de Óbitos Confirmados", f"{tot_obitos:,.0f}".replace(',', '.'), delta="Letalidade sob controle", delta_color="inverse")
    k3.metric("Taxa de Letalidade Geral", f"{letalidade_media:.2f}%", delta="-90% pós-vacina")
    k4.metric("Média Cobertura Vacinal (2 doses)", f"{cobertura_media:.1f}%", delta="Meta OMS atingida", delta_color="normal")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Ondas Epidemiológicas & Variantes",
        "🗺️ Distribuição Espacial (UFs)",
        "💉 Efeito Causal da Vacinação",
        "🧮 Simulador Matemático SIR"
    ])

    with tab1:
        st.subheader("📊 Dinâmica Temporal das Ondas de Contágio e Descolamento de Curvas")
        st.markdown("""
            Observe o **descolamento histórico**: na 1ª e 2ª ondas (Ancestral e Gama), o pico de casos acompanhava diretamente um pico drástico de óbitos.
            A partir de 2022 (**Variante Ômicron**), os casos explodiram mas os óbitos permaneceram contidos pela vacinação em massa.
        """)
        
        col_t1, col_t2 = st.columns([3, 1])
        with col_t2:
            metrica_tempo = st.radio("Métrica diária:", ["Média Móvel (7 dias)", "Totais Semanais"])

        y_casos = "casos_diarios_mm7" if metrica_tempo == "Média Móvel (7 dias)" else "casos_semanais"
        y_obitos = "obitos_diarios_mm7" if metrica_tempo == "Média Móvel (7 dias)" else "obitos_semanais"

        fig_waves = go.Figure()
        fig_waves.add_trace(go.Scatter(x=df_series["data"], y=df_series[y_casos], name="Casos Notificados", line=dict(color="#3B82F6", width=2.5), yaxis="y1"))
        fig_waves.add_trace(go.Scatter(x=df_series["data"], y=df_series[y_obitos], name="Óbitos Confirmados", line=dict(color="#EF4444", width=2.5), yaxis="y2"))

        fig_waves.add_vrect(x0="2020-03-01", x1="2021-01-01", fillcolor="gray", opacity=0.08, line_width=0, annotation_text="Ancestral", annotation_position="top left")
        fig_waves.add_vrect(x0="2021-01-01", x1="2021-07-01", fillcolor="red", opacity=0.08, line_width=0, annotation_text="Gama (P1)", annotation_position="top left")
        fig_waves.add_vrect(x0="2021-07-01", x1="2021-12-15", fillcolor="orange", opacity=0.08, line_width=0, annotation_text="Delta", annotation_position="top left")
        fig_waves.add_vrect(x0="2021-12-15", x1="2023-03-01", fillcolor="blue", opacity=0.08, line_width=0, annotation_text="Ômicron", annotation_position="top left")

        fig_waves.update_layout(
            template="plotly_white",
            height=480,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(title=dict(text="Casos Diários", font=dict(color="#3B82F6")), tickfont=dict(color="#3B82F6")),
            yaxis2=dict(title=dict(text="Óbitos Diários", font=dict(color="#EF4444")), tickfont=dict(color="#EF4444"), overlaying="y", side="right")
        )
        st.plotly_chart(fig_waves, use_container_width=True)

    with tab2:
        st.subheader(f"🗺️ Dispersão Espacial dos Indicadores ({regiao_sel})")
        col_m1, col_m2 = st.columns([2, 1])
        with col_m1:
            indicador_mapa = st.selectbox("Métrica para dimensionar os pontos:", ["Mortalidade por 100 mil hab", "Taxa de Letalidade (%)", "Total de Óbitos Acumulados"])
        with col_m2:
            map_style = st.selectbox("Tema do Mapa:", ["open-street-map", "carto-positron"])

        m_col = "mortalidade_100k" if "Mortalidade" in indicador_mapa else ("letalidade_pct" if "Letalidade" in indicador_mapa else "obitos_acumulados")
        scatter_fn = getattr(px, "scatter_map", None) or getattr(px, "scatter_mapbox")
        map_kwargs = {
            "lat": "lat",
            "lon": "lon",
            "size": m_col,
            "color": "status_risco",
            "color_discrete_map": {"Crítico (Histórico Alto)": "#EF4444", "Alerta (Moderado)": "#F59E0B", "Controlado": "#10B981"},
            "hover_name": "estado",
            "custom_data": ["uf", "incidencia_100k", "mortalidade_100k", "letalidade_pct", "vacina_2doses_pct"],
            "zoom": 3.6,
            "center": {"lat": -14.2350, "lon": -51.9253},
            "size_max": 36,
            "template": "plotly_white"
        }
        if hasattr(px, "scatter_map"):
            map_kwargs["map_style"] = map_style
        else:
            map_kwargs["mapbox_style"] = map_style

        fig_uf_map = scatter_fn(df_uf_filtered, **map_kwargs)
        fig_uf_map.update_traces(hovertemplate="<b>%{hovertext} (%{customdata[0]})</b><br>Incidência: <b>%{customdata[1]:,.0f}</b>/100k<br>Mortalidade: <b>%{customdata[2]:.1f}</b>/100k<br>Letalidade: <b>%{customdata[3]:.2f}%</b><br>Vacinação: <b>%{customdata[4]:.1f}%</b><extra></extra>")
        fig_uf_map.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_uf_map, use_container_width=True)

        st.dataframe(
            df_uf_filtered[["uf", "estado", "regiao", "populacao", "casos_acumulados", "obitos_acumulados", "mortalidade_100k", "letalidade_pct", "vacina_2doses_pct"]].rename(columns={
                "uf": "UF", "estado": "Estado", "regiao": "Região", "populacao": "População", "casos_acumulados": "Casos",
                "obitos_acumulados": "Óbitos", "mortalidade_100k": "Mortalidade/100k", "letalidade_pct": "Letalidade (%)", "vacina_2doses_pct": "Vacinação (%)"
            }),
            use_container_width=True,
            hide_index=True
        )

    with tab3:
        st.subheader("💉 Correlação Estatística: Cobertura Vacinal vs. Desaceleração de Óbitos")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            fig_scatter_vax = px.scatter(
                df_uf, x="vacina_2doses_pct", y="letalidade_pct", color="regiao", size="populacao",
                hover_name="estado", trendline="ols", title="Cobertura Vacinal (%) × Taxa de Letalidade (%)",
                labels={"vacina_2doses_pct": "Cobertura Vacinal Completa (%)", "letalidade_pct": "Taxa de Letalidade (%)", "regiao": "Região"},
                template="plotly_white"
            )
            st.plotly_chart(fig_scatter_vax, use_container_width=True)
        with col_v2:
            fig_vax_time = px.line(
                df_series, x="data", y=["cobertura_vacinal_pct", "taxa_letalidade_semanal"],
                title="Evolução Temporal da Cobertura Vacinal vs. Queda da Letalidade",
                labels={"data": "Data", "value": "Percentual (%)", "variable": "Indicador"},
                template="plotly_white"
            )
            st.plotly_chart(fig_vax_time, use_container_width=True)

    with tab4:
        st.subheader("🧮 Simulador Epidemiológico de Intervenções (Modelo SIR)")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            r0_input = st.slider("Taxa de Reprodução Básica ($R_0$):", 1.1, 5.0, 2.6, 0.1)
            duracao_infeccao = st.slider("Tempo de Infecção (Dias):", 5, 21, 10)
        with col_s2:
            intervencao_input = st.slider("Aderência a Máscaras / Distanciamento (%):", 0, 80, 25, 5)
            cobertura_sim_vax = st.slider("População Vacinada no Início (%):", 0, 90, 40, 5)
        with col_s3:
            pop_sim = st.number_input("Tamanho da População Alvo:", value=5000000, step=500000)
            capacidade_leitos = st.number_input("Capacidade de Leitos UTI:", value=3000, step=500)

        resultado_sir = simular_sir(
            populacao_total=int(pop_sim), infectados_iniciais=150, dias_simulacao=180,
            r0_base=r0_input, tempo_recuperacao_dias=float(duracao_infeccao),
            reducao_contato_pct=float(intervencao_input), cobertura_vacinal_pct=float(cobertura_sim_vax)
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R Efetivo Atual ($R_t$)", f"{resultado_sir['r_efetivo']:.2f}")
        m2.metric("Pico de Infectados", f"{resultado_sir['pico_infectados']:,}".replace(',', '.'))
        m3.metric("Pico Demanda UTI", f"{int(np.max(resultado_sir['Necessidade_UTI'])):,}".replace(',', '.'), delta="Sobrecarga" if np.max(resultado_sir['Necessidade_UTI']) > capacidade_leitos else "Sob controle", delta_color="inverse" if np.max(resultado_sir['Necessidade_UTI']) > capacidade_leitos else "normal")
        m4.metric("População Total Atingida", f"{resultado_sir['pct_populacao_atingida']:.1f}%")

        fig_sir = go.Figure()
        fig_sir.add_trace(go.Scatter(x=resultado_sir["dias"], y=resultado_sir["Suscetíveis"], name="Suscetíveis (S)", line=dict(color="#3B82F6", width=2)))
        fig_sir.add_trace(go.Scatter(x=resultado_sir["dias"], y=resultado_sir["Infectados"], name="Infectados Ativos (I)", line=dict(color="#EF4444", width=3)))
        fig_sir.add_trace(go.Scatter(x=resultado_sir["dias"], y=resultado_sir["Recuperados_Imunes"], name="Recuperados (R)", line=dict(color="#10B981", width=2)))
        fig_sir.add_trace(go.Scatter(x=resultado_sir["dias"], y=resultado_sir["Necessidade_UTI"], name="Demanda UTI", line=dict(color="#8B5CF6", width=2.5, dash="dot")))
        fig_sir.add_hline(y=capacidade_leitos, line_dash="dash", line_color="#DC2626", annotation_text=f"Capacidade UTI ({capacidade_leitos:,})")
        fig_sir.update_layout(template="plotly_white", height=460, title="Curva Epidêmica Dinâmica — Modelo SIR", xaxis_title="Dias", yaxis_title="Indivíduos")
        st.plotly_chart(fig_sir, use_container_width=True)

# ==============================================================================
# VISÃO 2: GLOBAL (MUNDO / PAÍSES)
# ==============================================================================
else:
    st.markdown("""
        <div style="margin-bottom: 24px;">
            <h1 style="font-size:2.3rem; letter-spacing:-0.02em; margin-bottom:4px;">
                🌍 Panorama Global da Pandemia de COVID-19 (Mundo)
            </h1>
            <p style="color:#64748B; font-size:1.05rem; font-weight:500;">
                Vigilância comparativa internacional, mapa-múndi de indicadores, contraste de ondas de contágio e desigualdade global de imunização.
            </p>
        </div>
    """, unsafe_allow_html=True)

    g_casos = df_global["casos_totais"].sum()
    g_obitos = df_global["obitos_totais"].sum()
    g_letalidade = (g_obitos / g_casos) * 100.0
    g_vacina = df_global["vacina_pct"].mean()

    gk1, gk2, gk3, gk4 = st.columns(4)
    gk1.metric("Casos Notificados (Amostra Global)", f"{g_casos:,.0f}".replace(',', '.'), delta="Países Monitorados")
    gk2.metric("Óbitos Confirmados", f"{g_obitos:,.0f}".replace(',', '.'), delta="Média Global", delta_color="inverse")
    gk3.metric("Taxa Média de Letalidade", f"{g_letalidade:.2f}%", delta="Mundial")
    gk4.metric("Média de Vacinação Completa", f"{g_vacina:.1f}%", delta="Desigualdade entre blocos")

    st.markdown("<br>", unsafe_allow_html=True)

    gtab1, gtab2, gtab3 = st.tabs([
        "🗺️ Mapa-Múndi Interativo",
        "📊 Comparador de Ondas Entre Países",
        "💉 Desigualdade Global de Vacinas"
    ])

    with gtab1:
        st.subheader("🗺️ Mapeamento Global dos Impactos da COVID-19")
        col_gm1, col_gm2 = st.columns([2, 1])
        with col_gm1:
            metrica_global = st.selectbox("Métrica para o mapa-múndi:", ["Óbitos por 1 Milhão de hab", "Casos por 100 mil hab", "Taxa de Letalidade (%)", "Total de Óbitos"])
        with col_gm2:
            tipo_mapa = st.selectbox("Projeção do Mapa:", ["natural earth", "orthographic", "equirectangular"])

        map_var = "mortalidade_1m" if "1 Milhão" in metrica_global else ("incidencia_100k" if "100 mil" in metrica_global else ("letalidade_pct" if "Letalidade" in metrica_global else "obitos_totais"))

        fig_world = px.scatter_geo(
            df_global_filtered,
            locations="codigo",
            color="continente",
            hover_name="pais",
            size=map_var,
            projection=tipo_mapa,
            custom_data=["populacao", "casos_totais", "obitos_totais", "mortalidade_1m", "vacina_pct"],
            title=f"Distribuição Geográfica: {metrica_global}",
            template="plotly_white"
        )
        fig_world.update_traces(
            hovertemplate="<b>%{hovertext}</b><br>População: %{customdata[0]:,.0f}<br>Casos: %{customdata[1]:,.0f}<br>Óbitos: %{customdata[2]:,.0f}<br>Mortalidade/1M: %{customdata[3]:.1f}<br>Vacinação: %{customdata[4]:.1f}%<extra></extra>"
        )
        fig_world.update_layout(height=540, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_world, use_container_width=True)

        st.dataframe(
            df_global_filtered[["pais", "codigo", "continente", "populacao", "casos_totais", "obitos_totais", "mortalidade_1m", "letalidade_pct", "vacina_pct"]].rename(columns={
                "pais": "País", "codigo": "Código", "continente": "Continente", "populacao": "População",
                "casos_totais": "Casos Totais", "obitos_totais": "Óbitos Totais", "mortalidade_1m": "Óbitos/1M hab",
                "letalidade_pct": "Letalidade (%)", "vacina_pct": "Vacinação (%)"
            }),
            use_container_width=True,
            hide_index=True
        )

    with gtab2:
        st.subheader("📊 Comparativo de Curvas de Contágio Entre Países (2020 a 2023)")
        st.markdown("""
            Compare a dinâmica temporal de novos casos por milhão de habitantes entre os diferentes países para entender como cada estratégia nacional (lockdowns, máscaras, variantes locais e início da vacinação) impactou as curvas.
        """)

        paises_disponiveis = [col for col in df_comp.columns if col != "data"]
        paises_sel = st.multiselect(
            "Selecione os países para comparar:",
            options=paises_disponiveis,
            default=["Brasil", "Estados Unidos", "Reino Unido", "Japão"]
        )

        if paises_sel:
            fig_comp = go.Figure()
            cores = ["#EF4444", "#3B82F6", "#8B5CF6", "#10B981", "#F59E0B"]
            for i, p in enumerate(paises_sel):
                fig_comp.add_trace(go.Scatter(
                    x=df_comp["data"],
                    y=df_comp[p],
                    name=p,
                    line=dict(color=cores[i % len(cores)], width=2.5),
                    hovertemplate=f"<b>{p}</b><br>Data: %{{x|%d/%m/%Y}}<br>Novos Casos/1M: %{{y:.1f}}<extra></extra>"
                ))

            fig_comp.update_layout(
                template="plotly_white",
                height=480,
                xaxis_title="Período Epidemiológico",
                yaxis_title="Novos Casos Bi-Semanais por Milhão de Habitantes",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.warning("Selecione pelo menos um país para visualizar a comparação.")

    with gtab3:
        st.subheader("💉 Disparidade Global de Acesso a Vacinas")
        st.markdown("""
            A pandemia evidenciou forte desigualdade: enquanto nações desenvolvidas e países como o Brasil atingiram mais de **80-90% de cobertura vacinal**, várias nações de menor renda (como na África Subsaariana) enfrentaram gargalos severos de distribuição.
        """)

        fig_bar_vax = px.bar(
            df_global.sort_values(by="vacina_pct", ascending=True),
            x="vacina_pct",
            y="pais",
            orientation="h",
            color="continente",
            text="vacina_pct",
            title="Taxa de Cobertura Vacinal Completa por País (%)",
            labels={"vacina_pct": "Cobertura Vacinal (%)", "pais": "País", "continente": "Continente"},
            template="plotly_white"
        )
        fig_bar_vax.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_bar_vax.update_layout(height=540, xaxis=dict(range=[0, 105]))
        st.plotly_chart(fig_bar_vax, use_container_width=True)

        st.info("💡 **Insight Geopolítico:** Países com histórico consolidado de vacinação pública (como o PNI do SUS no Brasil) superaram até mesmo grandes economias centrais em adesão vacinal da população adulta.")
