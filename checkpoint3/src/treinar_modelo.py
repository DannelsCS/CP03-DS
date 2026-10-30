"""Parte 2 — Treina e compara modelos de previsão de churn.

Executar a partir da raiz do projeto:
    python -m src.treinar_modelo

Gera:
    models/modelo_churn.joblib  -> pipeline completo (pré-processamento + modelo)
    models/metricas.json        -> métricas de todos os modelos no conjunto de teste
"""
import json

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.dados import (CAMINHO_METRICAS, CAMINHO_MODELO, CATEGORICAS,
                       NUMERICAS, carregar_dados, separar_x_y)

SEMENTE = 42


def criar_preprocessador() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
    ])


def candidatos() -> dict:
    return {
        "Regressão Logística": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=SEMENTE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=SEMENTE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=SEMENTE),
    }


def avaliar(pipe, X_te, y_te) -> dict:
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_te, proba)
    # reduz a curva ROC para ~50 pontos (só para o gráfico)
    idx = np.linspace(0, len(fpr) - 1, 50).astype(int)
    return {
        "acuracia": accuracy_score(y_te, pred),
        "precisao": precision_score(y_te, pred),
        "recall": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "roc_auc": roc_auc_score(y_te, proba),
        "matriz_confusao": confusion_matrix(y_te, pred).tolist(),
        "roc": {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist()},
    }


def importancias(pipe) -> list:
    """Importância de cada feature já agregada de volta à coluna original."""
    modelo = pipe.named_steps["modelo"]
    nomes = pipe.named_steps["prep"].get_feature_names_out()
    if hasattr(modelo, "feature_importances_"):
        valores = modelo.feature_importances_
    else:
        valores = np.abs(modelo.coef_[0])
    agregado = {}
    for nome, v in zip(nomes, valores):
        bloco, resto = nome.split("__", 1)
        coluna = resto if bloco == "num" else next(
            c for c in CATEGORICAS if resto == c or resto.startswith(c + "_"))
        agregado[coluna] = agregado.get(coluna, 0) + float(v)
    total = sum(agregado.values())
    return sorted(([k, v / total] for k, v in agregado.items()),
                  key=lambda kv: kv[1], reverse=True)


def treinar():
    df = carregar_dados()
    X, y = separar_x_y(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEMENTE)

    resultados, pipes = {}, {}
    for nome, modelo in candidatos().items():
        pipe = Pipeline([("prep", criar_preprocessador()), ("modelo", modelo)])
        pipe.fit(X_tr, y_tr)
        resultados[nome] = avaliar(pipe, X_te, y_te)
        pipes[nome] = pipe
        r = resultados[nome]
        print(f"{nome:22s} AUC={r['roc_auc']:.3f}  recall={r['recall']:.3f}  "
              f"F1={r['f1']:.3f}  acurácia={r['acuracia']:.3f}")

    # Critério: maior F1. As AUCs ficam praticamente empatadas (~0,84), e em
    # churn perder um cliente que ia cancelar (falso negativo) custa caro,
    # então priorizamos o equilíbrio entre recall e precisão.
    melhor = max(resultados, key=lambda n: resultados[n]["f1"])
    print(f"\nModelo escolhido: {melhor}")

    CAMINHO_MODELO.parent.mkdir(exist_ok=True)
    joblib.dump(pipes[melhor], CAMINHO_MODELO)
    with open(CAMINHO_METRICAS, "w", encoding="utf-8") as f:
        json.dump({
            "modelo_escolhido": melhor,
            "criterio": "maior F1 no conjunto de teste (20%, estratificado)",
            "n_treino": len(X_tr),
            "n_teste": len(X_te),
            "resultados": resultados,
            "importancias": importancias(pipes[melhor]),
        }, f, ensure_ascii=False, indent=2)
    return pipes[melhor]


if __name__ == "__main__":
    treinar()
