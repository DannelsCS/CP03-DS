# 📉 Dashboard de Churn de Clientes — Telecom

Checkpoint 3: uma aplicação analítica interativa em **Streamlit** que junta a análise exploratória (Parte 1) e o modelo preditivo de cancelamento de clientes (Parte 2).

🔗 **Aplicação online:** `https://cp03-ds-ezv5gdyihthntpeh6ilvhe.streamlit.app/`
🔗 **Repositório:** `https://github.com/DannelsCS/CP03-DS/tree/main`

## Fonte dos dados

[Telco Customer Churn — IBM](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv): 7.043 clientes de uma operadora de telecom e 21 variáveis (perfil, serviços contratados, contrato, cobrança). O alvo é `Churn`, que indica se o cliente cancelou no último mês. Uma cópia está em `data/telco_churn.csv`.

## O que a dashboard tem

| Aba | Conteúdo |
|---|---|
| 📊 **Visão geral** | KPIs: total de clientes, taxa de churn (comparada à base), receita mensal, receita perdida, % da receita perdida, permanência média e ticket médio. Também tem gráficos de churn por contrato e por faixa de permanência. |
| 🔬 **Exploração** | Gráficos interativos (Plotly) com seleção de variável: churn por categoria, histogramas, dispersão e matriz de correlação. Gráficos estáticos (Matplotlib): boxplot e mapa de calor internet × contrato. |
| 🤖 **Modelo preditivo** | Comparação de 3 modelos, métricas, curva ROC, matriz de confusão e importância das variáveis. |
| 🧮 **Simulador** | Formulário com o perfil de um cliente. Devolve a probabilidade de cancelamento e sugere ações de retenção. |
| 🗂️ **Fonte dos dados** | Origem, tratamento aplicado, dicionário de variáveis, tabela filtrada e download em CSV. |

**Filtros** (barra lateral): tipo de contrato, tipo de internet, faixa de permanência, forma de pagamento, gênero, idoso e faixa de valor mensal.

## Estrutura do projeto

```
├── app.py                          # Dashboard (Streamlit)
├── data/telco_churn.csv            # Dataset original
├── models/
│   ├── modelo_churn.joblib         # Pipeline treinado (pré-processamento + modelo)
│   └── metricas.json               # Métricas dos modelos no conjunto de teste
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb   # Parte 1
│   └── 02_modelo_preditivo.ipynb       # Parte 2
├── src/
│   ├── dados.py                    # Carregamento e tratamento (usado em todo o projeto)
│   └── treinar_modelo.py           # Treino e comparação dos modelos
├── .streamlit/config.toml          # Tema
└── requirements.txt
```

## Como executar localmente

Pré-requisito: **Python 3.11 ou superior**.

```bash
# 1. Clonar o repositório
git clone COLE_AQUI_A_URL_DO_GITHUB
cd NOME_DO_REPOSITORIO

# 2. Criar e ativar um ambiente virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Instalar as dependências
pip install -r requirements.txt

# 4. Rodar a dashboard
streamlit run app.py
```

A aplicação abre em `http://localhost:8501`.

### Opcional: retreinar o modelo ou abrir os notebooks

```bash
python -m src.treinar_modelo          # recria models/modelo_churn.joblib e metricas.json

pip install -r requirements-dev.txt   # instala o Jupyter
jupyter notebook notebooks/
```

## Modelo preditivo (resumo)

- Divisão: 80% treino / 20% teste, estratificada.
- Pré-processamento num `Pipeline`: `StandardScaler` nas variáveis numéricas e `OneHotEncoder` nas categóricas.
- Modelos comparados: Regressão Logística, Random Forest e Gradient Boosting. Os dois primeiros usam `class_weight="balanced"`, porque a base tem só 26,5% de churn.
- **Modelo escolhido: Random Forest**, pelo maior F1. As AUCs empataram em ~0,84. O critério prioriza encontrar quem vai cancelar (recall ≈ 0,79) sem gerar alarmes falsos demais.

| Modelo | Acurácia | Precisão | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Regressão Logística | 0,738 | 0,504 | 0,783 | 0,614 | 0,842 |
| **Random Forest** | 0,750 | 0,519 | **0,794** | **0,628** | 0,843 |
| Gradient Boosting | 0,805 | 0,669 | 0,524 | 0,588 | 0,844 |

## Principais conclusões

1. Contrato **mensal** tem ~43% de churn. No anual cai para ~11% e no bienal para ~3%.
2. Os **primeiros 12 meses** são críticos: ~47% de churn nessa faixa, contra ~10% depois de 4 anos.
3. **Fibra óptica** (~42%) e **boleto eletrônico** (~45%) concentram cancelamentos.
4. Clientes **sem suporte técnico ou sem segurança online** cancelam cerca de 2,5× mais.

## Hospedagem (Streamlit Community Cloud)

1. Suba este projeto para um repositório **público** no GitHub.
2. Acesse [share.streamlit.io](https://share.streamlit.io) e entre com a conta do GitHub.
3. Clique em **Create app**, depois em **Deploy a public app from GitHub**.
4. Escolha o repositório, o branch `main` e o arquivo `app.py`. Em *Advanced settings*, selecione Python 3.12 ou superior.
5. Clique em **Deploy** e copie a URL gerada para o topo deste README.

## Tecnologias

Python · Streamlit · Pandas · Plotly · Matplotlib · scikit-learn
