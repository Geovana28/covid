"""
data_loader.py — Gerador e processador de dados epidemiológicos da COVID-19 no Brasil.
Contém agregações nacionais, estaduais (27 UFs com coordenadas geográficas)
e séries temporais das ondas (Ancestral, Gama, Delta e Ômicron).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Informações demográficas e espaciais das 27 UFs do Brasil
UF_METADATA = [
    {"uf": "SP", "nome": "São Paulo", "regiao": "Sudeste", "lat": -23.5505, "lon": -46.6333, "pop": 44411238},
    {"uf": "RJ", "nome": "Rio de Janeiro", "regiao": "Sudeste", "lat": -22.9068, "lon": -43.1729, "pop": 16054524},
    {"uf": "MG", "nome": "Minas Gerais", "regiao": "Sudeste", "lat": -19.9167, "lon": -43.9345, "pop": 20538718},
    {"uf": "BA", "nome": "Bahia", "regiao": "Nordeste", "lat": -12.9714, "lon": -38.5014, "pop": 14136417},
    {"uf": "PR", "nome": "Paraná", "regiao": "Sul", "lat": -25.4284, "lon": -49.2733, "pop": 11443208},
    {"uf": "RS", "nome": "Rio Grande do Sul", "regiao": "Sul", "lat": -30.0346, "lon": -51.2177, "pop": 10880506},
    {"uf": "PE", "nome": "Pernambuco", "regiao": "Nordeste", "lat": -8.0476, "lon": -34.8770, "pop": 9058155},
    {"uf": "CE", "nome": "Ceará", "regiao": "Nordeste", "lat": -3.7172, "lon": -38.5433, "pop": 8791688},
    {"uf": "PA", "nome": "Pará", "regiao": "Norte", "lat": -1.4558, "lon": -48.4902, "pop": 8116132},
    {"uf": "SC", "nome": "Santa Catarina", "regiao": "Sul", "lat": -27.5954, "lon": -48.5480, "pop": 7609601},
    {"uf": "GO", "nome": "Goiás", "regiao": "Centro-Oeste", "lat": -16.6869, "lon": -49.2648, "pop": 7055228},
    {"uf": "MA", "nome": "Maranhão", "regiao": "Nordeste", "lat": -2.5307, "lon": -44.3068, "pop": 6775152},
    {"uf": "PB", "nome": "Paraíba", "regiao": "Nordeste", "lat": -7.1150, "lon": -34.8641, "pop": 3974495},
    {"uf": "AM", "nome": "Amazonas", "regiao": "Norte", "lat": -3.1190, "lon": -60.0217, "pop": 3941175},
    {"uf": "ES", "nome": "Espírito Santo", "regiao": "Sudeste", "lat": -20.3155, "lon": -40.3128, "pop": 3833486},
    {"uf": "MT", "nome": "Mato Grosso", "regiao": "Centro-Oeste", "lat": -15.6014, "lon": -56.0979, "pop": 3658813},
    {"uf": "RN", "nome": "Rio Grande do Norte", "regiao": "Nordeste", "lat": -5.7945, "lon": -35.2110, "pop": 3302406},
    {"uf": "PI", "nome": "Piauí", "regiao": "Nordeste", "lat": -5.0920, "lon": -42.8038, "pop": 3269200},
    {"uf": "AL", "nome": "Alagoas", "regiao": "Nordeste", "lat": -9.6658, "lon": -35.7351, "pop": 3127511},
    {"uf": "DF", "nome": "Distrito Federal", "regiao": "Centro-Oeste", "lat": -15.7975, "lon": -47.8919, "pop": 2817068},
    {"uf": "MS", "nome": "Mato Grosso do Sul", "regiao": "Centro-Oeste", "lat": -20.4697, "lon": -54.6201, "pop": 2756700},
    {"uf": "SE", "nome": "Sergipe", "regiao": "Nordeste", "lat": -10.9472, "lon": -37.0731, "pop": 2209558},
    {"uf": "RO", "nome": "Rondônia", "regiao": "Norte", "lat": -8.7619, "lon": -63.9039, "pop": 1581016},
    {"uf": "TO", "nome": "Tocantins", "regiao": "Norte", "lat": -10.1844, "lon": -48.3336, "pop": 1511459},
    {"uf": "AC", "nome": "Acre", "regiao": "Norte", "lat": -9.9754, "lon": -67.8249, "pop": 830026},
    {"uf": "AP", "nome": "Amapá", "regiao": "Norte", "lat": 0.0356, "lon": -51.0705, "pop": 733508},
    {"uf": "RR", "nome": "Roraima", "regiao": "Norte", "lat": 2.8235, "lon": -60.6758, "pop": 636303},
]

def load_uf_summary_data() -> pd.DataFrame:
    """Gera tabela consolidada de indicadores por UF."""
    np.random.seed(42)
    rows = []
    
    for item in UF_METADATA:
        pop = item["pop"]
        # Taxa de incidência real acumulada gira em torno de 15.000 a 26.000 por 100k
        taxa_incidencia = np.random.uniform(16000, 24000)
        casos = int((taxa_incidencia / 100000.0) * pop)
        
        # Letalidade média 1.5% a 2.4%
        letalidade_pct = np.random.uniform(1.4, 2.3)
        obitos = int(casos * (letalidade_pct / 100.0))
        mortalidade_100k = round((obitos / pop) * 100000.0, 1)
        
        # Cobertura Vacinal (% esquema completo 2 doses)
        vacina_2doses_pct = round(np.random.uniform(76.0, 89.5), 1)
        vacina_reforco_pct = round(np.random.uniform(52.0, 72.0), 1)
        
        # Status de Risco Epidemiológico
        if mortalidade_100k > 360:
            status = "Crítico (Histórico Alto)"
        elif mortalidade_100k > 310:
            status = "Alerta (Moderado)"
        else:
            status = "Controlado"
            
        rows.append({
            "uf": item["uf"],
            "estado": item["nome"],
            "regiao": item["regiao"],
            "populacao": pop,
            "lat": item["lat"],
            "lon": item["lon"],
            "casos_acumulados": casos,
            "obitos_acumulados": obitos,
            "incidencia_100k": round(taxa_incidencia, 1),
            "mortalidade_100k": mortalidade_100k,
            "letalidade_pct": round(letalidade_pct, 2),
            "vacina_2doses_pct": vacina_2doses_pct,
            "vacina_reforco_pct": vacina_reforco_pct,
            "status_risco": status
        })
        
    df = pd.DataFrame(rows)
    return df.sort_values(by="obitos_acumulados", ascending=False).reset_index(drop=True)

def load_epidemiological_waves_timeseries() -> pd.DataFrame:
    """Gera série temporal detalhada das 4 ondas da COVID-19 no Brasil."""
    np.random.seed(101)
    
    datas = pd.date_range(start="2020-03-01", end="2023-03-01", freq="W-MON")
    n = len(datas)
    
    # Simulação das 4 principais ondas com curvas gaussianas sobrepostas
    # 1. Onda 1 (Ancestral): pico ~Jul/2020 (semana ~18)
    # 2. Onda 2 (Gama/P.1): pico ~Mar-Abr/2021 (semana ~56)
    # 3. Onda 3 (Delta): ~Ago-Set/2021 (semana ~78, casos moderados, óbitos contidos por vacina)
    # 4. Onda 4 (Ômicron): pico gigante em Jan/2022 (semana ~98, casos astronômicos, letalidade muito baixa)
    
    x = np.arange(n)
    onda1 = 300000 * np.exp(-((x - 18) ** 2) / (2 * (8 ** 2)))
    onda2 = 520000 * np.exp(-((x - 56) ** 2) / (2 * (9 ** 2)))
    onda3 = 240000 * np.exp(-((x - 76) ** 2) / (2 * (7 ** 2)))
    onda4 = 1150000 * np.exp(-((x - 98) ** 2) / (2 * (6 ** 2)))
    
    base_noise = np.random.normal(15000, 4000, n)
    casos_semanais = np.maximum(onda1 + onda2 + onda3 + onda4 + base_noise, 5000)
    
    # Vacinação iniciando na semana ~46 (Jan/2021) com curva logística
    vax_progress = 1 / (1 + np.exp(-(x - 65) / 10))  # 0 a 1
    cobertura_vacinal = np.clip(vax_progress * 86.0, 0, 86.0)
    
    # Óbitos: fortemente impactados pela vacina após semana 60
    # Letalidade de ~2.8% antes da vacina despenca para ~0.18% com a Ômicron/vacinação
    fator_letalidade = 0.028 * (1 - 0.90 * (cobertura_vacinal / 100.0))
    # Variante Gama foi mais letal
    fator_gama = 1.35 * np.exp(-((x - 56) ** 2) / (2 * (10 ** 2)))
    
    obitos_semanais = casos_semanais * (fator_letalidade + (fator_gama * 0.008)) + np.random.normal(500, 150, n)
    obitos_semanais = np.maximum(obitos_semanais, 120)
    
    df = pd.DataFrame({
        "data": datas,
        "casos_semanais": casos_semanais.astype(int),
        "obitos_semanais": obitos_semanais.astype(int),
        "cobertura_vacinal_pct": np.round(cobertura_vacinal, 1),
    })
    
    # Média móvel e taxas
    df["casos_diarios_mm7"] = np.round(df["casos_semanais"] / 7.0, 1)
    df["obitos_diarios_mm7"] = np.round(df["obitos_semanais"] / 7.0, 1)
    df["taxa_letalidade_semanal"] = np.round((df["obitos_semanais"] / df["casos_semanais"]) * 100.0, 2)
    
    # Identificação da Variante Dominante
    def classificar_variante(d):
        if d < pd.Timestamp("2021-01-01"):
            return "Variante Ancestral (B.1.1.28)"
        elif d < pd.Timestamp("2021-07-01"):
            return "Variante Gama (P.1)"
        elif d < pd.Timestamp("2021-12-15"):
            return "Variante Delta (B.1.617.2)"
        else:
            return "Variante Ômicron (B.1.1.529)"
            
    df["variante_predominante"] = df["data"].apply(classificar_variante)
    return df

# Dados Globais por País (Principais nações e continentes)
GLOBAL_COUNTRIES = [
    {"pais": "Estados Unidos", "codigo": "USA", "continente": "América do Norte", "pop": 331893745, "casos": 103436829, "obitos": 1127152, "vacina_pct": 81.2, "lat": 37.0902, "lon": -95.7129},
    {"pais": "Índia", "codigo": "IND", "continente": "Ásia", "pop": 1408044253, "casos": 44998525, "obitos": 531930, "vacina_pct": 74.5, "lat": 20.5937, "lon": 78.9629},
    {"pais": "França", "codigo": "FRA", "continente": "Europa", "pop": 67750000, "casos": 40138560, "obitos": 167642, "vacina_pct": 83.1, "lat": 46.2276, "lon": 2.2137},
    {"pais": "Alemanha", "codigo": "DEU", "continente": "Europa", "pop": 83200000, "casos": 38437729, "obitos": 174979, "vacina_pct": 78.0, "lat": 51.1657, "lon": 10.4515},
    {"pais": "Brasil", "codigo": "BRA", "continente": "América do Sul", "pop": 214300000, "casos": 37717062, "obitos": 704659, "vacina_pct": 86.4, "lat": -14.2350, "lon": -51.9253},
    {"pais": "Japão", "codigo": "JPN", "continente": "Ásia", "pop": 125700000, "casos": 33803572, "obitos": 74694, "vacina_pct": 84.3, "lat": 36.2048, "lon": 138.2529},
    {"pais": "Coreia do Sul", "codigo": "KOR", "continente": "Ásia", "pop": 51740000, "casos": 34571873, "obitos": 35934, "vacina_pct": 87.2, "lat": 35.9078, "lon": 127.7669},
    {"pais": "Itália", "codigo": "ITA", "continente": "Europa", "pop": 59110000, "casos": 26723249, "obitos": 196452, "vacina_pct": 85.0, "lat": 41.8719, "lon": 12.5674},
    {"pais": "Reino Unido", "codigo": "GBR", "continente": "Europa", "pop": 67330000, "casos": 24967389, "obitos": 232112, "vacina_pct": 80.5, "lat": 55.3781, "lon": -3.4360},
    {"pais": "Rússia", "codigo": "RUS", "continente": "Europa/Ásia", "pop": 143400000, "casos": 22900755, "obitos": 399536, "vacina_pct": 55.4, "lat": 61.5240, "lon": 105.3188},
    {"pais": "Argentina", "codigo": "ARG", "continente": "América do Sul", "pop": 45810000, "casos": 10044957, "obitos": 130472, "vacina_pct": 84.1, "lat": -38.4161, "lon": -63.6167},
    {"pais": "África do Sul", "codigo": "ZAF", "continente": "África", "pop": 59390000, "casos": 4072533, "obitos": 102595, "vacina_pct": 36.8, "lat": -30.5595, "lon": 22.9375},
    {"pais": "Austrália", "codigo": "AUS", "continente": "Oceania", "pop": 25690000, "casos": 11853144, "obitos": 24414, "vacina_pct": 86.9, "lat": -25.2744, "lon": 133.7751},
    {"pais": "Canadá", "codigo": "CAN", "continente": "América do Norte", "pop": 38250000, "casos": 4946090, "obitos": 59034, "vacina_pct": 83.7, "lat": 56.1304, "lon": -106.3468},
    {"pais": "México", "codigo": "MEX", "continente": "América do Norte", "pop": 126700000, "casos": 7702809, "obitos": 334958, "vacina_pct": 64.9, "lat": 23.6345, "lon": -102.5528},
    {"pais": "Chile", "codigo": "CHL", "continente": "América do Sul", "pop": 19490000, "casos": 5384852, "obitos": 64497, "vacina_pct": 92.4, "lat": -35.6751, "lon": -71.5430},
    {"pais": "Nigéria", "codigo": "NGA", "continente": "África", "pop": 213400000, "casos": 267184, "obitos": 3155, "vacina_pct": 30.2, "lat": 9.0820, "lon": 8.6753},
]

def load_global_countries_data() -> pd.DataFrame:
    """Retorna DataFrame consolidado de países no mundo."""
    rows = []
    for c in GLOBAL_COUNTRIES:
        mortalidade_1m = round((c["obitos"] / c["pop"]) * 1000000.0, 1)
        incidencia_100k = round((c["casos"] / c["pop"]) * 100000.0, 1)
        letalidade_pct = round((c["obitos"] / c["casos"]) * 100.0, 2)
        
        rows.append({
            "pais": c["pais"],
            "codigo": c["codigo"],
            "continente": c["continente"],
            "populacao": c["pop"],
            "casos_totais": c["casos"],
            "obitos_totais": c["obitos"],
            "mortalidade_1m": mortalidade_1m,
            "incidencia_100k": incidencia_100k,
            "letalidade_pct": letalidade_pct,
            "vacina_pct": c["vacina_pct"],
            "lat": c["lat"],
            "lon": c["lon"]
        })
    return pd.DataFrame(rows).sort_values(by="obitos_totais", ascending=False).reset_index(drop=True)

def load_global_comparison_timeseries() -> pd.DataFrame:
    """Gera série temporal para comparação multi-país de novos casos por milhão de habitantes."""
    np.random.seed(42)
    datas = pd.date_range(start="2020-03-01", end="2023-01-01", freq="2W-MON")
    n = len(datas)
    x = np.arange(n)
    
    # Perfis de onda por país
    # Brasil: ondas fortes em 2020, pico severo Gama em 2021, pico de Ômicron em 2022
    bra_cases = 180 * np.exp(-((x - 8) ** 2) / 30) + 360 * np.exp(-((x - 28) ** 2) / 40) + 750 * np.exp(-((x - 49) ** 2) / 25) + np.random.normal(20, 5, n)
    
    # EUA: pico inverno 2020, pico Delta outono 2021, pico Ômicron monstruoso
    usa_cases = 240 * np.exp(-((x - 19) ** 2) / 35) + 380 * np.exp(-((x - 39) ** 2) / 30) + 1400 * np.exp(-((x - 49) ** 2) / 20) + np.random.normal(30, 8, n)
    
    # Reino Unido: pico inicial, inverno 2020 e Ômicron precoce
    gbr_cases = 150 * np.exp(-((x - 6) ** 2) / 25) + 420 * np.exp(-((x - 20) ** 2) / 30) + 980 * np.exp(-((x - 47) ** 2) / 25) + np.random.normal(25, 6, n)
    
    # Japão: conteve quase tudo até a variante Ômicron em 2022
    jpn_cases = 30 * np.exp(-((x - 12) ** 2) / 40) + 60 * np.exp(-((x - 38) ** 2) / 30) + 1100 * np.exp(-((x - 52) ** 2) / 25) + np.random.normal(10, 3, n)
    
    # Índia: Onda Delta massiva no início de 2021
    ind_cases = 60 * np.exp(-((x - 14) ** 2) / 35) + 290 * np.exp(-((x - 31) ** 2) / 18) + 240 * np.exp(-((x - 49) ** 2) / 25) + np.random.normal(15, 4, n)

    df_comp = pd.DataFrame({
        "data": datas,
        "Brasil": np.maximum(np.round(bra_cases, 1), 5),
        "Estados Unidos": np.maximum(np.round(usa_cases, 1), 5),
        "Reino Unido": np.maximum(np.round(gbr_cases, 1), 5),
        "Japão": np.maximum(np.round(jpn_cases, 1), 2),
        "Índia": np.maximum(np.round(ind_cases, 1), 3)
    })
    return df_comp

