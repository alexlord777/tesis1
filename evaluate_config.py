import numpy as np
from microtc.textmodel import TextModel
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.svm import LinearSVC

def valuation(config,x,y,cv=5):
    try:
        model=TextModel(**config)
        print("pasa1")
        model.fit(x)
        print("pasa2")
        x_trans=model.transform(x)
        
        clf=LinearSVC(random_state=42,dual="auto")
        print("pasa3")
        f1_scores=cross_val_score(clf,x_trans,y,cv=cv,scoring='f1_macro')
        return np.mean(f1_scores)
    except Exception as e:
        print(f"Error al evaluar la configuración: {config} ")
        return -1

def valuationI(x,y,cv=5):
    try:
        model=TextModel()
        print("pasa1")
        model.fit(x)
        print("pasa2")
        x_trans=model.transform(x)
        clf=LinearSVC(random_state=42)
        f1_scores=cross_val_score(clf,x_trans,y,cv=cv,scoring='f1_macro')
        return np.mean(f1_scores)
    except Exception as e:
       # print(f"Error al evaluar la configuración: {config} ")
        return -1
