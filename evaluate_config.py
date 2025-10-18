import numpy as np
from microtc.textmodel import TextModel
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
import warnings

def valuation(config,X, y,cv=5):
  
    try:
       
        # Crear modelo de texto
        model = TextModel(**config)
        model.fit(X)
        X_trans = model.transform(X)
        # Clasificador base
        clf = LinearSVC(random_state=42, dual=False)
        # Evaluación cruzada
        f1_scores = cross_val_score(clf, X_trans, y, cv=cv, scoring="f1_macro")
        score = np.mean(f1_scores)
        return score

    except Exception as e:
        return -np.inf


def valuationI(X, y, cv=5, ):
    try:
       
        # Crear modelo de texto
        model = TextModel() 
        model.fit(X)
        X_trans = model.transform(X)
        # Clasificador base
        clf = LinearSVC(random_state=42, dual=False)
        # Evaluación cruzada
        f1_scores = cross_val_score(clf, X_trans, y, cv=cv, scoring="f1_macro")
        score = np.mean(f1_scores)
        return score

    except Exception as e:
        return -np.inf