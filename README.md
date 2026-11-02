# 🦠 COVID-19 Intelligence & Epidemiological Analytics Platform

[![Live Interactive Simulator](https://img.shields.io/badge/Live_Showcase-GitHub_Pages-14b8a6?style=for-the-badge&logo=googlechrome&logoColor=white)](https://geovana28.github.io/covid/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Model: SIR ODE](https://img.shields.io/badge/Dynamical_System-Runge--Kutta_4-blue?style=for-the-badge)](https://github.com/Geovana28/covid)
[![Status](https://img.shields.io/badge/Status-Produção%20%2F%20Ativo-success?style=for-the-badge)]()

> 💡 **Experimente o Simulador Interativo:** Acesse o [**Interactive Showcase (GitHub Pages)**](https://geovana28.github.io/covid/) para simular a resolução numérica RK4 das equações SIR, impacto de intervenções não-farmacológicas (NPI) e saturação de UTI direto pelo navegador.

> Plataforma integrada de vigilância epidemiológica, exploração de séries temporais das ondas da COVID-19 no Brasil, análise temporal da cobertura vacinal e modelagem matemática dinâmica com equações diferenciais (Modelo SIR e Crescimento Logístico de Verhulst).

---

## 🎯 Objetivos do Projeto

- **Vigilância Dual (Brasil & Global):** Alternância instantânea no menu lateral entre o panorama detalhado das 27 UFs do Brasil e o comparativo internacional entre grandes nações do mundo (EUA, Índia, França, Alemanha, Japão, Reino Unido, África do Sul, etc.).
- **Monitoramento Temporal das Ondas:** Acompanhamento dinâmico das 4 principais fases da pandemia no Brasil (*Ancestral, Gama, Delta e Ômicron*) com médias móveis de 7 dias e identificação de descolamento entre curvas de casos e óbitos.
- **Comparativo Multi-País:** Gráficos interativos lado a lado permitindo selecionar e contrastar a velocidade de contágio e estratégias nacionais de contenção.
- **Vigilância Espacial (UFs & Mapa-Múndi):** Mapeamento georreferenciado interativo nacional e projeções geográficas globais com incidência, mortalidade acumulada e taxas de letalidade.
- **Análise Temporal da Imunização & Disparidade Global:** Avaliação da correlação temporal entre a expansão da cobertura vacinal e a redução da letalidade observada, além do mapeamento da desigualdade de distribuição de imunizantes entre continentes.
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


## ⚠️ Limitações Metodológicas e Hipóteses do Modelo

1. **Mistura Homogênea (*Homogeneous Mixing*)**: O modelo SIR assume população fechada ($S(t) + I(t) + R(t) = N$) e contato homogêneo entre indivíduos, não capturando heterogeneidades etárias ou redes de contato espacialmente estruturadas.
2. **Parâmetros Constantes por Onda**: As taxas de transmissão ($\beta$) e recuperação ($\gamma$) são tratadas como constantes dentro da janela simulada, e o modelo básico SIR não incorpora perda de imunidade (*waning immunity*) ou reinfecção por escape antigênico.
3. **Correlação Temporal vs. Inferência Causal**: O descolamento observado entre as curvas de casos e óbitos concomitantemente ao avanço vacinal constitui evidência ecológica e temporal consistente com a literatura clínica, mas não substitui desenhos formais de inferência causal ao nível individual (como ensaios clínicos randomizados ou desenhos de coorte com controle de confundidores).

---

## 🚀 Como Executar Localmente

### 1. Clonar e Instalar Dependências
```bash
git clone https://github.com/Geovana28/covid.git
cd covid
pip install -r requirements.txt
```

### 2. Iniciar o Dashboard Streamlit ou Versão Web
```bash
streamlit run app.py
# Ou abra index.html diretamente no navegador para o simulador standalone
```

### 3. Executar Testes Automatizados
```bash
pytest tests/ -v
```

---

## 📄 Licença e Autoria
Distribuído sob a licença MIT. Veja `LICENSE` para mais detalhes.  
Desenvolvido por **Geovana** — [GitHub Profile](https://github.com/Geovana28)
