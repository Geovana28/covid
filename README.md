# 🦠 COVID-19 Intelligence & Epidemiological Analytics Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-7.0%2B-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![Scipy](https://img.shields.io/badge/SciPy-1.10%2B-8CAAE6?logo=scipy&logoColor=white)](https://scipy.org/)
[![Status](https://img.shields.io/badge/Status-Produção%20%2F%20Ativo-success)]()

> Plataforma integrada de vigilância epidemiológica, exploração de séries temporais das variantes da COVID-19 no Brasil, análise causal da cobertura vacinal e modelagem matemática dinâmica com equações diferenciais (Modelo SIR).

---

## 🎯 Objetivos do Projeto

- **Vigilância Dual (Brasil & Global):** Alternância instantânea no menu lateral entre o panorama detalhado das 27 UFs do Brasil e o comparativo internacional entre grandes nações do mundo (EUA, Índia, França, Alemanha, Japão, Reino Unido, África do Sul, etc.).
- **Monitoramento Temporal das Ondas:** Acompanhamento dinâmico das 4 principais fases da pandemia no Brasil (*Ancestral, Gama, Delta e Ômicron*) com médias móveis de 7 dias e identificação de descolamento de curvas.
- **Comparativo Multi-País:** Gráficos interativos lado a lado permitindo selecionar e contrastar a velocidade de contágio e estratégias nacionais de contenção.
- **Vigilância Espacial (UFs & Mapa-Múndi):** Mapeamento georreferenciado interativo nacional e projeções geográficas globais com incidência, mortalidade acumulada e taxas de letalidade.
- **Análise Causal da Imunização & Disparidade Global:** Correlação estatística robusta entre a expansão da cobertura vacinal e a redução de óbitos, além do mapeamento da desigualdade de distribuição de imunizantes entre continentes.
- **Simulador Epidemiológico SIR:** Resolução computacional de Equações Diferenciais Ordinárias (ODE) permitindo aos gestores e pesquisadores simular o impacto de medidas não-farmacológicas (máscaras, distanciamento) e vacinação na prevenção do colapso do sistema de saúde (leitos UTI).

---

## 📐 Fundamentação Matemática & Epidemiológica

### 1. Modelo SIR (Susceptible - Infectious - Recovered)
O simulador computacional resolve o sistema de equações diferenciais ordinárias:

$$\frac{dS}{dt} = -\beta \frac{S \cdot I}{N}$$

$$\frac{dI}{dt} = \beta \frac{S \cdot I}{N} - \gamma I$$

$$\frac{dR}{dt} = \gamma I$$

Onde:
- $N = S + I + R$: População total.
- $\beta$: Taxa de transmissão efetiva, reduzida por intervenções não-farmacológicas: $\beta = (R_0 \cdot \gamma) \cdot (1 - \text{Intervenção})$.
- $\gamma = \frac{1}{\text{Tempo de Recuperação}}$: Taxa de recuperação viral.
- $R_t = \frac{\beta}{\gamma} \cdot \frac{S}{N}$: Número reprodutivo efetivo ao longo do tempo.

---

### 2. Modelo de Crescimento Logístico & Ajuste por Mínimos Quadrados
A plataforma incorpora a solução analítica da **Equação Diferencial Logística de Verhulst** para estimar o teto assintótico de casos ($y_M$) e prever curvas de estabilização de cada onda:

$$\frac{dy}{dt} = K \cdot y \cdot (y_M - y)$$

Cuja solução analítica exata é dada por:

$$y(t) = \frac{y_M}{1 + \left(\frac{y_M - y_0}{y_0}\right) e^{-y_M \cdot K \cdot (t - t_0)}}$$

A calibração dos parâmetros $y_M$ e $K$ é realizada através do método de **Mínimos Quadrados** não-lineares, permitindo projetar a estabilização de platô e o ponto de inflexão máxima ($\frac{y_M}{2}$) de novos casos.

---

## 📚 Referências Bibliográficas & Metodologia

- **SANTOS, Reginaldo J.** *Um Modelo para o Surto de Coronavírus no Brasil*. Departamento de Matemática, Instituto de Ciências Exatas, Universidade Federal de Minas Gerais (UFMG), Belo Horizonte, 2020.
- **SANTOS, Reginaldo J.** *Crescimento Populacional: Um Modelo para o Crescimento da População Brasileira*. Departamento de Matemática, Instituto de Ciências Exatas, Universidade Federal de Minas Gerais (UFMG), Belo Horizonte.
- **VERHULST, Pierre-François.** *Notice sur la loi que la population poursuit dans son accroissement*. Correspondance Mathématique et Physique, 1838.
- **KERMACK, W. O.; MCKENDRICK, A. G.** *A Contribution to the Mathematical Theory of Epidemics*. Proceedings of the Royal Society of London, 1927.


## 🚀 Como Executar Localmente

### 1. Clonar e Instalar Dependências
```bash
cd c:\Users\SEJUSP.DESKTOP-VOL42AH\Desktop\Projetos-\covid
pip install -r requirements.txt
```

### 2. Iniciar o Dashboard
```bash
streamlit run app.py
```

O aplicativo abrirá automaticamente no seu navegador padrão em: `http://localhost:8501`.

---

## 👩‍💻 Autoria
Desenvolvido por **Geovana**  
*Full Stack Developer & Data Scientist*  
[GitHub Profile](https://github.com/Geovana28)
