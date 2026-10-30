"""Carregamento e tratamento do dataset Telco Customer Churn (IBM).

Usado pelos notebooks (Partes 1 e 2), pelo script de treino e pela dashboard,
para que todos trabalhem exatamente com os mesmos dados tratados.
"""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_DADOS = RAIZ / "data" / "telco_churn.csv"
CAMINHO_MODELO = RAIZ / "models" / "modelo_churn.joblib"
CAMINHO_METRICAS = RAIZ / "models" / "metricas.json"

FONTE_NOME = "Telco Customer Churn — IBM"
FONTE_URL = (
    "https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/"
    "data/Telco-Customer-Churn.csv"
)

# Nomes das colunas em português
COLUNAS = {
    "customerID": "id_cliente",
    "gender": "genero",
    "SeniorCitizen": "idoso",
    "Partner": "tem_parceiro",
    "Dependents": "tem_dependentes",
    "tenure": "meses_contrato",
    "PhoneService": "servico_telefone",
    "MultipleLines": "multiplas_linhas",
    "InternetService": "tipo_internet",
    "OnlineSecurity": "seguranca_online",
    "OnlineBackup": "backup_online",
    "DeviceProtection": "protecao_aparelho",
    "TechSupport": "suporte_tecnico",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_filmes",
    "Contract": "tipo_contrato",
    "PaperlessBilling": "fatura_digital",
    "PaymentMethod": "forma_pagamento",
    "MonthlyCharges": "valor_mensal",
    "TotalCharges": "valor_total",
    "Churn": "churn",
}

# Valores das categorias em português
VALORES = {
    "Yes": "Sim",
    "No": "Não",
    "Female": "Feminino",
    "Male": "Masculino",
    "No phone service": "Sem telefone",
    "No internet service": "Sem internet",
    "Fiber optic": "Fibra óptica",
    "Month-to-month": "Mensal",
    "One year": "Anual",
    "Two year": "Bienal",
    "Electronic check": "Boleto eletrônico",
    "Mailed check": "Boleto pelo correio",
    "Bank transfer (automatic)": "Débito automático",
    "Credit card (automatic)": "Cartão de crédito (automático)",
}

# Rótulos amigáveis para exibir na dashboard
ROTULOS = {
    "genero": "Gênero",
    "idoso": "Idoso (65+)",
    "tem_parceiro": "Tem parceiro(a)",
    "tem_dependentes": "Tem dependentes",
    "meses_contrato": "Meses como cliente",
    "servico_telefone": "Serviço de telefone",
    "multiplas_linhas": "Múltiplas linhas",
    "tipo_internet": "Tipo de internet",
    "seguranca_online": "Segurança online",
    "backup_online": "Backup online",
    "protecao_aparelho": "Proteção de aparelho",
    "suporte_tecnico": "Suporte técnico",
    "streaming_tv": "Streaming de TV",
    "streaming_filmes": "Streaming de filmes",
    "tipo_contrato": "Tipo de contrato",
    "fatura_digital": "Fatura digital",
    "forma_pagamento": "Forma de pagamento",
    "valor_mensal": "Valor mensal (US$)",
    "valor_total": "Valor total pago (US$)",
    "faixa_permanencia": "Faixa de permanência",
    "churn": "Cancelou (churn)",
}

NUMERICAS = ["meses_contrato", "valor_mensal", "valor_total"]
CATEGORICAS = [
    "genero", "idoso", "tem_parceiro", "tem_dependentes", "servico_telefone",
    "multiplas_linhas", "tipo_internet", "seguranca_online", "backup_online",
    "protecao_aparelho", "suporte_tecnico", "streaming_tv", "streaming_filmes",
    "tipo_contrato", "fatura_digital", "forma_pagamento",
]
ALVO = "churn"

FAIXAS_PERMANENCIA = ["0–12 meses", "13–24 meses", "25–48 meses", "49–72 meses"]


def carregar_dados(caminho: Path = CAMINHO_DADOS) -> pd.DataFrame:
    """Lê o CSV original e devolve o DataFrame tratado e traduzido."""
    df = pd.read_csv(caminho)
    df = df.rename(columns=COLUNAS)

    # valor_total vem como texto e tem 11 linhas em branco (clientes com 0 meses)
    df["valor_total"] = pd.to_numeric(df["valor_total"].astype(str).str.strip(),
                                      errors="coerce")
    df["valor_total"] = df["valor_total"].fillna(0.0)

    df["idoso"] = df["idoso"].map({0: "Não", 1: "Sim"})

    texto = [c for c in df.columns if c not in NUMERICAS + ["id_cliente"]]
    for c in texto:
        df[c] = df[c].astype(str).replace(VALORES)

    df["faixa_permanencia"] = pd.cut(
        df["meses_contrato"], bins=[-1, 12, 24, 48, 72], labels=FAIXAS_PERMANENCIA
    ).astype(str)
    return df


def separar_x_y(df: pd.DataFrame):
    """Features e alvo (1 = cancelou) para o modelo."""
    X = df[NUMERICAS + CATEGORICAS].copy()
    y = (df[ALVO] == "Sim").astype(int)
    return X, y
