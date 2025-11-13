from sklearn.metrics import roc_curve, auc
from microtc.textmodel import TextModel
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelBinarizer
import pandas as pd

def generar_curva_roc(config, X, y, nombre_archivo="curva_roc_datos.csv"):
    model = TextModel(**config)
    model.fit(X)
    X_trans = model.transform(X)

    clf = LinearSVC(random_state=42, dual=False)
    clf.fit(X_trans, y)

    if len(np.unique(y)) == 2:  # problema binario
        # Convertir etiquetas si no son numéricas
        lb = LabelBinarizer()
        y_bin = lb.fit_transform(y).ravel()

        # Obtener scores del clasificador
        y_scores = clf.decision_function(X_trans)

        # Calcular curva ROC
        fpr, tpr, _ = roc_curve(y_bin, y_scores)
        roc_auc = auc(fpr, tpr)

        # Crear DataFrame con los datos
        df = pd.DataFrame({
            "FalsePositiveRate": fpr,
            "TruePositiveRate": tpr
        })

        # Agregar el valor del AUC al final del archivo
        df.loc[len(df)] = [None, None]
        df.loc[len(df)] = ["AUC", roc_auc]

        # Guardar CSV
        df.to_csv(nombre_archivo, index=False, encoding="utf-8")
        print(f"Datos de la curva ROC guardados en {nombre_archivo} (AUC = {roc_auc:.3f})")

    else:
        print("Curva ROC no generada: problema multiclase detectado.")
