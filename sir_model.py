"""
sir_model.py — Motor de Simulação Matemática Epidemiológica.
Implementa o clássico modelo SIR (Susceptible - Infectious - Recovered)
resolvido numericamente através de equações diferenciais ordinárias (scipy.integrate.odeint).
Permite modelar o impacto de NPIs (Intervenções Não-Farmacológicas como Máscaras/Lockdown)
e da taxa de Cobertura Vacinal.
"""

import numpy as np
from scipy.integrate import odeint

def deriv_sir(y, t, N, beta, gamma):
    """
    Equações diferenciais do modelo SIR:
    dS/dt = -beta * S * I / N
    dI/dt =  beta * S * I / N - gamma * I
    dR/dt =  gamma * I
    """
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt =  beta * S * I / N - gamma * I
    dRdt =  gamma * I
    return dSdt, dIdt, dRdt

def simular_sir(
    populacao_total: int = 10000000,
    infectados_iniciais: int = 100,
    dias_simulacao: int = 180,
    r0_base: float = 2.8,
    tempo_recuperacao_dias: float = 10.0,
    reducao_contato_pct: float = 0.0,
    cobertura_vacinal_pct: float = 0.0,
    eficacia_vacina_pct: float = 85.0
):
    """
    Executa a simulação epidemiológica com parâmetros configuráveis.
    
    Retorna:
        dict com arrays temporais de Suscetíveis, Infectados, Recuperados,
        pico de infectados e estimativa de sobrecarga hospitalar (leitos UTI).
    """
    N = populacao_total
    
    # Efeito da vacinação remove indivíduos suscetíveis inicialmente
    fracao_imune_vacina = (cobertura_vacinal_pct / 100.0) * (eficacia_vacina_pct / 100.0)
    S0 = N * (1.0 - fracao_imune_vacina) - infectados_iniciais
    I0 = infectados_iniciais
    R0_pop = N * fracao_imune_vacina
    
    # Parâmetros epidemiológicos
    gamma = 1.0 / tempo_recuperacao_dias
    # Beta efetivo ajustado por intervenções (máscaras, distanciamento)
    fator_intervencao = (1.0 - (reducao_contato_pct / 100.0))
    beta_efetivo = (r0_base * gamma) * fator_intervencao
    
    r_efetivo = (beta_efetivo / gamma) * (S0 / N)
    
    t = np.linspace(0, dias_simulacao, dias_simulacao)
    y0 = S0, I0, R0_pop
    
    ret = odeint(deriv_sir, y0, t, args=(N, beta_efetivo, gamma))
    S, I, R = ret.T
    
    # Estimativa de internações em UTI (~2.5% dos infectados ativos)
    necessidade_uti = I * 0.025
    pico_infectados = int(np.max(I))
    dia_pico = int(np.argmax(I))
    total_infectados_final = int(N - S[-1])
    
    return {
        "dias": t,
        "Suscetíveis": np.round(S).astype(int),
        "Infectados": np.round(I).astype(int),
        "Recuperados_Imunes": np.round(R).astype(int),
        "Necessidade_UTI": np.round(necessidade_uti).astype(int),
        "r_efetivo": round(r_efetivo, 2),
        "pico_infectados": pico_infectados,
        "dia_pico": dia_pico,
        "total_infectados_final": total_infectados_final,
        "pct_populacao_atingida": round((total_infectados_final / N) * 100.0, 1)
    }
