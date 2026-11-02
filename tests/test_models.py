"""
Unit tests for COVID-19 Mathematical Models (SIR ODE & Verhulst Logistic Regression).
"""
import numpy as np
import pytest
from sir_model import simular_sir, deriv_sir
from logistic_model import funcao_logistica, ajustar_modelo_logistico


def test_sir_population_conservation():
    """Total population S(t) + I(t) + R(t) must be conserved across all simulation steps."""
    N = 1_000_000
    res = simular_sir(
        populacao_total=N,
        infectados_iniciais=500,
        dias_simulacao=120,
        r0_base=2.5,
        tempo_recuperacao_dias=10.0,
        reducao_contato_pct=20.0,
        cobertura_vacinal_pct=30.0,
        eficacia_vacina_pct=80.0,
    )
    total_t = res["Suscetíveis"] + res["Infectados"] + res["Recuperados_Imunes"]
    # Rounding to int in each compartment can introduce at most +/- 2 difference
    assert np.all(np.abs(total_t - N) <= 3)


def test_sir_deriv_zero_sum():
    """The derivatives dS/dt + dI/dt + dR/dt must sum strictly to 0.0 (closed population)."""
    dS, dI, dR = deriv_sir((900_000, 10_000, 90_000), t=0, N=1_000_000, beta=0.3, gamma=0.1)
    assert pytest.approx(dS + dI + dR, abs=1e-9) == 0.0


def test_sir_subcritical_reproduction_number():
    """When effective R_t < 1 (e.g., strong intervention or low R0), infections should decay monotonically."""
    res = simular_sir(
        populacao_total=1_000_000,
        infectados_iniciais=1_000,
        dias_simulacao=60,
        r0_base=0.7,
        tempo_recuperacao_dias=10.0,
    )
    assert res["r_efetivo"] < 1.0
    assert res["dia_pico"] == 0
    assert res["Infectados"][-1] < res["Infectados"][0]


def test_logistic_verhulst_fit_synthetic_wave():
    """Non-linear least squares should accurately recover the carrying capacity yM on synthetic logistic data."""
    t = np.arange(30, dtype=float)
    y_true = funcao_logistica(t, y0=100.0, yM=50_000.0, r=0.22)
    datas = [f"2021-01-{i+1:02d}" for i in range(30)]
    fit = ajustar_modelo_logistico(datas, y_true, dias_projecao=15)
    assert fit["r2_ajuste"] > 0.99
    assert pytest.approx(fit["yM_capacidade_maxima"], rel=0.05) == 50_000.0

