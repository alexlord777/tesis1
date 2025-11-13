import numpy as np
from microtc.textmodel import TextModel
from sklearn.model_selection import cross_val_predict
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score
)

def valuation(config, X, y, cv=5):
    try:
        model = TextModel(**config)
        model.fit(X)
        X_trans = model.transform(X)

        # Si no hay características, penalizar
        if X_trans.shape[1] == 0:
            raise ValueError("El modelo no generó características válidas.")

        clf = LinearSVC(random_state=42, dual=False)

        # Validación cruzada por predicción
        y_pred = cross_val_predict(clf, X_trans, y, cv=cv)

        # Calcular métricas
        precision = precision_score(y, y_pred, average="macro", zero_division=0)
        recall = recall_score(y, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y, y_pred, average="macro", zero_division=0)
        accuracy = accuracy_score(y, y_pred)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy
        }

    except Exception as e:
        print(f"Error en valuation(): {e}")
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": -np.inf,
            "accuracy": 0.0
        }


def valuationI(X, y, cv=5):
    try:
        model = TextModel()
        model.fit(X)
        X_trans = model.transform(X)

        if X_trans.shape[1] == 0:
            raise ValueError("El modelo no generó características válidas.")

        clf = LinearSVC(random_state=42, dual=False)
        y_pred = cross_val_predict(clf, X_trans, y, cv=cv)

        f1 = f1_score(y, y_pred, average="macro", zero_division=0)

        return f1  # Solo devuelves F1 base

    except Exception as e:
        print(f"Error en valuationI(): {e}")
        return -np.inf
