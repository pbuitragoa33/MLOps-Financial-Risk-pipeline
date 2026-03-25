# Modelo Heurístico 


# Librerías

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import (
    StratifiedKFold, ShuffleSplit, cross_val_score,
    learning_curve, train_test_split
    )
from sklearn.metrics import (accuracy_score, precision_score,
    recall_score, f1_score, classification_report, 
    ConfusionMatrixDisplay
    )
from cargar_datos import cargar_datos_scoring

load_dotenv() 

pth_outputs = Path(os.getenv("OUTPUTS"))


# ----------------------------------------------------- Clase del Modelo Heurístico -----------------------------------------------------


class ModeloHeuristico(BaseEstimator, ClassifierMixin):

    """
    Reglas de Clasificación Modelo Heurístico:

    - puntaje_datacredito < 800 : Aumenta Riesgo 
    - huella_consulta > 10 : Aumenta Riesgo
    - edad_cliente < 30 : Aunmenta Riesgo
    - saldo_total > 800000,0 : Mitiga riesgo

    """

    # Constructor de la clase que hereda de BaseEstimator y ClassifierMixin

    def __init__(self, puntaje_datacredito_THRESHOLD: int = 800, huella_consulta_THRESHOLD: int = 10, 
                 edad_cliente_THRESHOLD: int = 30, saldo_total_THRESHOLD: float = 800000.0,
                 target_positivo: str = 1, target_negativo: str = 0):
        
        self.puntaje_datacredito_THRESHOLD = puntaje_datacredito_THRESHOLD
        self.huella_consulta_THRESHOLD = huella_consulta_THRESHOLD
        self.edad_cliente_THRESHOLD = edad_cliente_THRESHOLD
        self.saldo_total_THRESHOLD = saldo_total_THRESHOLD
        self.target_positivo = target_positivo
        self.target_negativo = target_negativo


    def fit(self, X, y = None):

        """
        Método de entrenamiento del modelo. En este caso, no se realiza ningún ajuste ya que el modelo es heurístico.
        """

        if y is not None:
                
            self.classes_ = np.unique(y)
            self.y_dtype = pd.Series(y).dtype

        else:

            self.y_dtype = np.int64

        return self
        

    def predict(self, X):

        """
        Métodos de predicción, aplicando las reglas de clasificación.
        Ahora será clase positiva si cumple 3 o más de las 4 validaciones.

        """

        predicciones = []

        for _, row in X.iterrows():

            validaciones = 0

            if row['puntaje_datacredito'] >= self.puntaje_datacredito_THRESHOLD:

                validaciones += 1

            if row['huella_consulta'] <= self.huella_consulta_THRESHOLD:

                validaciones += 1

            if row['edad_cliente'] >= self.edad_cliente_THRESHOLD:

                validaciones += 1

            if row['saldo_total'] >= self.saldo_total_THRESHOLD:

                validaciones += 1

            if validaciones >= 3:

                predicciones.append(self.target_positivo)

            else:

                predicciones.append(self.target_negativo)

        return np.array(predicciones) 



# -----------------------------------------------------  Métricas y Cross Validation -----------------------------------------------------


model = ModeloHeuristico()

# Métricas de Evaluación

scoring_metrics = ["accuracy", "f1", "precision", "recall"]


# Función para calcular métricas de evaluación

def metricas_CV_modelo(model, X_train : pd.DataFrame, y_train : pd.Series, scoring_metrics = None, cv = 10, random_state = 42):

    if scoring_metrics is None:

        scoring_metrics = ["accuracy", "f1", "precision", "recall"]

    stratifies_KFold = StratifiedKFold(n_splits = cv, shuffle = True, random_state = random_state)
    
    resultados_cv = {}

    for metric in scoring_metrics:

        cv_scores = cross_val_score(model, X_train, y_train, cv = stratifies_KFold, scoring = metric, n_jobs = -1)

        resultados_cv[metric] = cv_scores

        print(f"{metric.capitalize()}: mean = {cv_scores.mean():.4f}, std = {cv_scores.std():.4f}")

    return pd.DataFrame(resultados_cv)


# Función de curvas de aprendizaje

def curvas_aprendizaje_model(estimator, X, y, scoring = "accuracy"):

    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(
        estimator = estimator,
        X = X,
        y = y,
        train_sizes = np.linspace(0.1, 1.0, 5),
        cv = ShuffleSplit(n_splits = 30, test_size = 0.2, random_state = 42), 
        n_jobs = -1,
        scoring = scoring,
        return_times = True
    )
    
    train_scores_mean = np.mean(train_scores, axis = 1)
    train_scores_std = np.std(train_scores, axis = 1)
    test_scores_mean = np.mean(test_scores, axis = 1)
    test_scores_std = np.std(test_scores, axis = 1)
    fit_times_mean = np.mean(fit_times, axis = 1)
    fit_times_std = np.std(fit_times, axis = 1)
    score_times_mean = np.mean(score_times, axis = 1)
    score_times_std = np.std(score_times, axis = 1)

    # Gráfica de Curvas de Aprendizaje

    fig, ax = plt.subplots(figsize = (12, 8))

    ax.plot(train_sizes, train_scores_mean, label = "Training Score", color = "blue")
    ax.plot(train_sizes, test_scores_mean, label = "Cross-Validation Score", color = "orange")
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha = 0.3, color = "blue")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha = 0.3, color = "orange")
    ax.set_title("Learning Curves", fontsize = 16)
    ax.set_xlabel("Training Set Size", fontsize = 14)
    ax.set_ylabel(scoring.capitalize(), fontsize = 14)
    ax.legend(loc = "best")
    plt.tight_layout()
    plt.savefig(pth_outputs / "heuristic_learning_curves_2.png", dpi = 300, bbox_inches = "tight")

    # Gráfica de Tiempos de Entrenamiento y Evaluación

    fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True,figsize = (12, 8))

    ax[0].plot(train_sizes, fit_times_mean, label = "Fit Time", color = "green")
    ax[0].fill_between(train_sizes, fit_times_mean - fit_times_std, fit_times_mean + fit_times_std, alpha = 0.3, color = "green")
    ax[0].set_title("Fit Time", fontsize = 16)
    ax[0].set_ylabel("Time (seconds)", fontsize = 14)   

    ax[1].plot(train_sizes, score_times_mean, label = "Score Time", color = "red")
    ax[1].fill_between(train_sizes, score_times_mean - score_times_std, score_times_mean + score_times_std, alpha = 0.3, color = "red")
    ax[1].set_title("Score Time", fontsize = 16)
    ax[1].set_xlabel("Training Set Size", fontsize = 14)
    ax[1].set_ylabel("Time (seconds)", fontsize = 14)

    plt.tight_layout()
    plt.savefig(pth_outputs / "heuristic_score_times_2.png", dpi = 300, bbox_inches = "tight")




# -----------------------------------------------------  Entrenamiento -----------------------------------------------------


def entrenar_modelo_heuristico(path : str, target_column : str = "Pago_atiempo", test_size : float = 0.2, random_state : int = 42):

    # Cargar el dataset con el modulo de carga de datos desde BigQuery

    df = cargar_datos_scoring()

    # Separar características y variable objetivo

    X = df.drop(columns = [target_column])
    y = df[target_column]

    # Dividir el dataset en conjuntos de entrenamiento y prueba

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state, stratify = y)

    # Modelo Heurístico (Primero instanciar y luego fit)

    modelo_heuristico = ModeloHeuristico()
    modelo_heuristico.fit(X_train, y_train)

    # Predecir

    y_pred = modelo_heuristico.predict(X_test)

    # Imprimir el reporte de clasificación

    print("Reporte de Clasificación del Modelo Heurístico:")
    print(classification_report(y_test, y_pred))

    # Métricas de clasificación

    accuracy_heuristico = accuracy_score(y_test, y_pred)
    precision_heuristico = precision_score(y_test, y_pred)
    f1_heuristico = f1_score(y_test, y_pred)

    print(f"Accuracy en Test:{round(accuracy_heuristico, 3)}")
    print(f"Precision en Test:{round(precision_heuristico, 3)}")
    print(f"F1 Score en Test:{round(f1_heuristico, 3)}")



    # Matriz de Confusión

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred,
                                             display_labels = modelo_heuristico.classes_, cmap = "Blues")
    plt.title("Matriz de Confusión del Modelo Heurístico", fontsize = 16)
    plt.tight_layout()
    plt.savefig(pth_outputs / "heuristic_confusion_matrix_2.png", dpi = 300, bbox_inches = "tight")


    # Cross Validation

    print("Resultados de Cross Validation del Modelo Heurístico:")

    resultados_cv = metricas_CV_modelo(modelo_heuristico, X_train, y_train, scoring_metrics = scoring_metrics, cv = 10, random_state = random_state)
    print(resultados_cv)

    # Curvas de Aprendizaje

    curvas_aprendizaje_model(modelo_heuristico, X_train, y_train, scoring = "accuracy")

    return {
        "df": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "modelo_heuristico": modelo_heuristico,
        "metricas_cv": resultados_cv
    }



# Main

if __name__ == "__main__":

    load_dotenv() 

    pth = Path(os.getenv("DATA_FILE"))

    resultados = entrenar_modelo_heuristico(pth / "BD_creditos.xlsx", target_column = "Pago_atiempo", test_size = 0.3, random_state = 42)