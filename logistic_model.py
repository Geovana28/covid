"""
logistic_model.py — Motor Matemático de Crescimento Logístico & Ajuste por Mínimos Quadrados.
Baseado na modelagem de Equações Diferenciais Ordinárias (EDO) e Notas Técnicas
do Prof. Dr. Reginaldo J. Santos (Departamento de Matemática - UFMG).

A Equação Diferencial Logística de Verhulst:
    dy/dt = K * y * (y_M - y)

Cuja solução analítica com condição inicial y(t_0) = y_0 é:
    y(t) = y_M / (1 + ((y_M - y_0) / y_0) * exp(-y_M * K * (t - t_0)))
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_absolute_error

def funcao_logistica(t, y0, yM, r):
    """
    Formulação analítica da curva logística:
    t: vetor de tempo (em dias contínuos a partir de 0)
    y0: valor inicial da população/casos no tempo t=0
    yM: capacidade de suporte assintótica (teto máximo projetado)
    r: taxa de crescimento combinada (r = y_M * K)
    """
    # Proteção numérica contra overflow no expoente
    expoente = np.clip(-r * t, -700, 700)
    razao = (yM - y0) / max(y0, 1e-5)
    return yM / (1.0 + razao * np.exp(expoente))

def ajustar_modelo_logistico(datas, casos_acumulados, dias_projecao=60):
    """
    Ajusta a curva logística aos dados observados utilizando o método
    de Mínimos Quadrados não-lineares (via SciPy curve_fit).
    
    Retorna:
        dict com métricas de calibração, parâmetros estimados (yM, K, r)
        e série temporal estendida (histórico + projeção futura).
    """
    y_real = np.array(casos_acumulados, dtype=float)
    n_pontos = len(y_real)
    if n_pontos < 5:
        raise ValueError("São necessários pelo menos 5 pontos para calibrar o modelo logístico.")

    t_real = np.arange(n_pontos, dtype=float)
    y0_inicial = max(y_real[0], 1.0)
    yM_chute = max(y_real[-1] * 1.3, y0_inicial * 2.0)
    r_chute = 0.05

    # Limites e chute inicial para Mínimos Quadrados
    p0 = [y0_inicial, yM_chute, r_chute]
    bounds = (
        [0.0, y_real[-1], 1e-5],              # Limites inferiores
        [y_real[-1] * 1.5, y_real[-1] * 15.0, 2.0]  # Limites superiores
    )

    try:
        popt, pcov = curve_fit(
            funcao_logistica,
            t_real,
            y_real,
            p0=p0,
            bounds=bounds,
            maxfev=10000
        )
        y0_est, yM_est, r_est = popt
    except Exception:
        # Fallback heurístico caso não convirja
        y0_est = y0_inicial
        yM_est = y_real[-1] * 1.25
        r_est = 0.03

    # K intrínseco da EDO: dy/dt = K * y * (y_M - y) => r = K * y_M => K = r / y_M
    K_est = r_est / max(yM_est, 1.0)

    # Valores ajustados no intervalo histórico
    y_pred_hist = funcao_logistica(t_real, y0_est, yM_est, r_est)
    
    r2 = r2_score(y_real, y_pred_hist)
    mae = mean_absolute_error(y_real, y_pred_hist)
    erro_pct_medio = np.mean(np.abs((y_real - y_pred_hist) / np.maximum(y_real, 1.0))) * 100.0

    # Vetor de tempo estendido com projeção futura
    t_total = np.arange(n_pontos + dias_projecao, dtype=float)
    y_projecao_total = funcao_logistica(t_total, y0_est, yM_est, r_est)

    # Datas correspondentes
    datas_pd = pd.to_datetime(pd.Series(datas))
    delta_dias = (datas_pd.iloc[-1] - datas_pd.iloc[-2]).days if len(datas_pd) > 1 else 7
    if delta_dias <= 0:
        delta_dias = 1
        
    datas_futuras = [
        datas_pd.iloc[-1] + pd.Timedelta(days=int((i + 1) * delta_dias))
        for i in range(dias_projecao)
    ]
    datas_completas = list(datas_pd) + datas_futuras


    # Ponto de inflexão (máxima taxa diária dy/dt, ocorre quando y = yM / 2)
    t_inflexao = (1.0 / r_est) * np.log(max((yM_est - y0_est) / y0_est, 1e-5)) if y0_est > 0 else 0
    t_inflexao_dia = max(0, int(round(t_inflexao)))

    return {
        "y0_estimado": float(y0_est),
        "yM_capacidade_maxima": float(yM_est),
        "taxa_r": float(r_est),
        "constante_K_edo": float(K_est),
        "r2_ajuste": float(max(0.0, r2)),
        "mae": float(mae),
        "erro_pct_medio": float(erro_pct_medio),
        "t_inflexao_dia": t_inflexao_dia,
        "datas": datas_completas,
        "curva_ajustada_e_projetada": y_projecao_total,
        "n_historico": n_pontos,
        "n_futuro": dias_projecao
    }
