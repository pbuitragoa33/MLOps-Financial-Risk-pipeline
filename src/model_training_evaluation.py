# Entrenamiento, Optimización de Hiperparámetros, Evaluación y Reporte de Métricas


# Librerías 


import __main__
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import warnings
import pickle
from tqdm import tqdm

from sklearn.base import clone
from sklearn.pipeline import Pipeline as SklearnPipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImblearnPipeline

from sklearn.metrics import (accuracy_score, precision_score, recall_score, roc_auc_score,
                             average_precision_score, f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_curve, brier_score_loss, log_loss)

from sklearn.model_selection import (cross_val_score, StratifiedKFold, learning_curve, train_test_split)
from sklearn.calibration import calibration_curve

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


# Importar Clases de Preprocesamiento

from ft_engineering import AgruparCategorias, ToDF, Outliers, ColumnasNulos, Imputacion, NuevasVariables, ToCategory, ColumnasIrrelevantes, EliminarCategorias


__main__.AgruparCategorias = AgruparCategorias
__main__.ToDF = ToDF
__main__.Outliers = Outliers
__main__.ColumnasNulos = ColumnasNulos
__main__.Imputacion = Imputacion
__main__.NuevasVariables = NuevasVariables
__main__.ToCategory = ToCategory
__main__.ColumnasIrrelevantes = ColumnasIrrelevantes
__main__.EliminarCategorias = EliminarCategorias



# -------------------------------------------------------------------------------------------------
# 1. CONFIGURACIÓN GLOBAL Y CARGA DE ARTEFACTOS
# -------------------------------------------------------------------------------------------------

load_dotenv()

ruta_artifacts = Path(os.getenv("ARTIFACTS"))


# Cargar el pipeline de preprocesamiento

with open(ruta_artifacts / "pipeline_ml.pkl", "rb") as f:

    pipeline_ml = pickle.load(f)



# Aplanar el pipeline recursivamente sin importar el tipo exacto de Pipeline.


def _es_pipeline(estimador):

    return hasattr(estimador, 'steps') and hasattr(estimador, 'fit') and isinstance(getattr(estimador, 'steps', None), list)

def obtener_pasos_planos(pipeline):

    pasos = []

    for nombre, estimador in pipeline.steps:

        if _es_pipeline(estimador):

            pasos.extend(obtener_pasos_planos(estimador))

        else:

            pasos.append((nombre, estimador))

    return pasos


# Invocar la función

pasos_preprocesamiento = obtener_pasos_planos(pipeline_ml)

SMOTE_RATIO = 0.33
TEST_SIZE = 0.2
cv_strategy = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)


# ---------------------------------------- FUNCIONES DE TUNING SIN SMOTE ----------------------------------------------------

# Regresión Logística

def logistic_regression_tuning_no_smote(trial, X_train, y_train, cv = cv_strategy, metric = "f1_macro"):

    solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear", "saga"])
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    C = trial.suggest_float("C", 1e-4, 10, log = True)
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
    
    l1_ratio  =  trial.suggest_float("l1_ratio", 0.0, 1.0) if penalty  ==  "elasticnet" else None

    if solver == "lbfgs" and penalty != "l2": return 0.0
    if solver == "liblinear" and penalty == "elasticnet": return 0.0
    if solver != "saga" and penalty == "elasticnet": return 0.0

    model = LogisticRegression(solver = solver, penalty = penalty, C = C, class_weight = class_weight, l1_ratio = l1_ratio)

    full_pipeline = SklearnPipeline(steps = [('preprocessor', pipeline_ml), ('model', model)])

    return cross_val_score(full_pipeline, X_train, y_train, cv = cv, scoring = metric).mean()


# Árbol de Decisión

def decision_tree_tuning_no_smote(trial, X_train, y_train, cv = cv_strategy, metric = "f1_macro"):
     
    model = DecisionTreeClassifier(
        max_depth = trial.suggest_int("max_depth", 2, 7),
        min_samples_split = trial.suggest_int("min_samples_split", 10, 50),
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 20),
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"]),
        random_state = 42,
        class_weight = trial.suggest_categorical("class_weight", ["balanced", None])
    )
     
    full_pipeline = SklearnPipeline(steps = [('preprocessor', pipeline_ml), ('model', model)])
     
    return cross_val_score(full_pipeline, X_train, y_train, cv = cv, scoring = metric).mean()


# Random Forest

def random_forest_tuning_no_smote(trial, X_train, y_train, cv = cv_strategy, metric = "f1_macro"):

    model = RandomForestClassifier(
        n_estimators = trial.suggest_int("n_estimators", 50, 150),
        max_depth = trial.suggest_int("max_depth", 3, 15),
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 30),
        min_samples_split = trial.suggest_int("min_samples_split", 10, 50),
        random_state = 42, n_jobs = 1,
        class_weight = trial.suggest_categorical("class_weight", ["balanced", None])
    )

    full_pipeline = SklearnPipeline(steps = [('preprocessor', pipeline_ml), ('model', model)])

    return cross_val_score(full_pipeline, X_train, y_train, cv = cv, scoring = metric).mean()


# LightGBM

def lightGBM_tuning_no_smote(trial, X_train, y_train, cv = cv_strategy, metric = "f1_macro"):

    model = LGBMClassifier(
        n_estimators = trial.suggest_int("n_estimators", 50, 150),
        max_depth = trial.suggest_int("max_depth", 2, 7),
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-1, log = True),
        num_leaves = trial.suggest_int("num_leaves", 10, 30),
        subsample = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        random_state = 42, n_jobs = 1, verbosity = -1
    )

    full_pipeline = SklearnPipeline(steps = [('preprocessor', pipeline_ml), ('model', model)])

    return cross_val_score(full_pipeline, X_train, y_train, cv = cv, scoring = metric).mean()


# SVM (Máquina de Soporte Vectorial)

def svm_tuning_no_smote(trial, X_train, y_train, cv = cv_strategy, metric = "f1_macro"):

    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'sigmoid'])
    gamma = trial.suggest_float('gamma', 1e-3, 1e-1, log = True)

    model = SVC(kernel = kernel, gamma = gamma)

    full_pipeline = SklearnPipeline(steps = [('preprocessor', pipeline_ml), ('model', model)])

    return cross_val_score(full_pipeline, X_train, y_train, cv = cv, scoring = metric).mean()


# XGBoost

def xgboost_tuning_no_smote(trial, X_train, y_train, cv = cv_strategy, metric = "f1_macro"):

    model  =  XGBClassifier(
        n_estimators = trial.suggest_int("n_estimators", 50, 150),
        max_depth = trial.suggest_int("max_depth", 2, 7),
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-1, log = True),
        subsample = trial.suggest_float("subsample", 0.5, 1.0),
        eval_metric = 'logloss', random_state = 42, n_jobs = 1
    )

    full_pipeline = SklearnPipeline(steps = [('preprocessor', pipeline_ml), ('model', model)])

    return cross_val_score(full_pipeline, X_train, y_train, cv = cv, scoring = metric).mean()



# -------------------------------------------------------------------------------------------------
# 2. FUNCIONES DE EVALUACIÓN PROFUNDA Y VISUALIZACIÓN
# -------------------------------------------------------------------------------------------------


# Función de validación cruzada con métricas específicas para cada clase evaluando el pipeline completo

def crossval_detailed_metrics(pipeline_estimator, X_train, y_train, cv = 5):

    skf  =  StratifiedKFold(n_splits = cv, shuffle = True, random_state = 42)

    metricas_por_fold  =  []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):

        if isinstance(X_train, pd.DataFrame):

            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        else:

            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model_fold = clone(pipeline_estimator)

        model_fold.fit(X_tr, y_tr)

        y_pred = model_fold.predict(X_val)

        fold_metricas  =  {
            'fold': fold + 1,
            'f1_macro': f1_score(y_val, y_pred, average = "macro"),
            'precision_0': precision_score(y_val, y_pred, pos_label = 0, zero_division = 0),
            'recall_0': recall_score(y_val, y_pred, pos_label = 0, zero_division = 0),
            'f1_0': f1_score(y_val, y_pred, pos_label = 0, zero_division = 0),
            'precision_1': precision_score(y_val, y_pred, pos_label = 1, zero_division = 0),
            'recall_1': recall_score(y_val, y_pred, pos_label = 1, zero_division = 0),
            'f1_1': f1_score(y_val, y_pred, pos_label = 1, zero_division = 0)
        }

        metricas_por_fold.append(fold_metricas)

    cv_results  =  pd.DataFrame(metricas_por_fold)

    print(f"\nRESUMEN VALIDACIÓN CRUZADA (Train)")
    print("-" * 60)

    for metrica in ['f1_macro', 'recall_0', 'precision_0', 'f1_0', 'recall_1', 'precision_1', 'f1_1']:

        valor  =  cv_results[metrica]

        print(f"{metrica:15}: {valor.mean():.4f} ± {valor.std():.4f}")

    return cv_results



# Función de visualización --> Curvas de Aprendizaje, Escalabilidad y Gap de Rendimiento

def plot_learning_curves(estimator, X, y, scoring = "f1_macro", model_name = "Model", save_path = None):

    print(f"\nGenerando curvas de aprendizaje para {model_name}...")

    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(
        estimator = estimator, X = X, y = y,
        train_sizes = np.linspace(0.1, 1.0, 5),
        cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42),
        n_jobs = 1, scoring = scoring, return_times = True
    )

    train_mean, train_std = train_scores.mean(axis = 1), train_scores.std(axis = 1)
    test_mean, test_std = test_scores.mean(axis = 1), test_scores.std(axis = 1)
    fit_times_mean, fit_times_std = fit_times.mean(axis = 1), fit_times.std(axis = 1)
    score_times_mean, score_times_std = score_times.mean(axis = 1), score_times.std(axis = 1)

    fig, axes  =  plt.subplots(2, 2, figsize = (15, 10))
    fig.suptitle(f'Análisis de Curvas de Aprendizaje - {model_name}', fontsize = 16)


    # 1. Curva principal

    axes[0, 0].plot(train_sizes, train_mean, "o-", label = "Entrenamiento", color = 'blue')
    axes[0, 0].plot(train_sizes, test_mean,  "o-", label = "Validación (CV)", color = 'red')
    axes[0, 0].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha = 0.1, color = 'blue')
    axes[0, 0].fill_between(train_sizes, test_mean  - test_std,  test_mean  + test_std,  alpha = 0.1, color = 'red')
    axes[0, 0].set_title(f"Curva de aprendizaje ({scoring})")
    axes[0, 0].set_xlabel("Ejemplos de entrenamiento")
    axes[0, 0].set_ylabel(scoring.upper())
    axes[0, 0].legend(loc = "best")
    axes[0, 0].grid(True, alpha = 0.3)


    # 2. Tiempo de entrenamiento

    axes[0, 1].plot(train_sizes, fit_times_mean, "o-", color = 'springgreen')
    axes[0, 1].fill_between(train_sizes, fit_times_mean - fit_times_std, fit_times_mean + fit_times_std, alpha = 0.1, color = 'green')
    axes[0, 1].set_title("Escalabilidad (Tiempo de Entrenamiento)")
    axes[0, 1].set_xlabel("Ejemplos de entrenamiento")
    axes[0, 1].set_ylabel("Tiempo (seg)")
    axes[0, 1].grid(True, alpha = 0.3)


    # 3. Tiempo de predicción

    axes[1, 0].plot(train_sizes, score_times_mean, "o-", color = 'orangered')
    axes[1, 0].fill_between(train_sizes, score_times_mean - score_times_std, score_times_mean + score_times_std, alpha = 0.1, color = 'orange')
    axes[1, 0].set_title("Eficiencia (Tiempo de Predicción)")
    axes[1, 0].set_xlabel("Ejemplos de entrenamiento")
    axes[1, 0].set_ylabel("Tiempo (seg)")
    axes[1, 0].grid(True, alpha = 0.3)


    # 4. Gap (Overfitting)

    gap = train_mean - test_mean
    axes[1, 1].plot(train_sizes, gap, "o-", color = 'darkviolet')
    axes[1, 1].axhline(y = 0, color = 'black', linestyle = '--', alpha = 0.5)
    axes[1, 1].set_title("Gap Entrenamiento-Validación (Overfitting)")
    axes[1, 1].set_xlabel("Ejemplos de entrenamiento")
    axes[1, 1].set_ylabel(f"Diferencia {scoring}")
    axes[1, 1].grid(True, alpha = 0.3)

    plt.tight_layout()
    
    if save_path:

        plt.savefig(save_path)

    plt.close()



# Función de visualización de gráficos avanzados --> ROC, PRC, Calibración, Matriz de Confusión Normalizada

def plot_advanced_evaluation(pipeline_model, X_test, y_test, model_name = "Model", save_path = None):

    print(f"\nGenerando dashboard para {model_name}...")

    y_proba = pipeline_model.predict_proba(X_test)[:, 1]
    y_pred = pipeline_model.predict(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc_val = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    cm = confusion_matrix(y_test, y_pred, normalize = "true")
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins = 10)

    print("\nMÉTRICAS PROBABILÍSTICAS (TEST)")
    print("-" * 40)
    print(f"ROC-AUC        : {roc_auc_val:.4f}")
    print(f"PR-AUC         : {pr_auc:.4f}")
    print(f"Brier Score    : {brier_score_loss(y_test, y_proba):.4f}")
    print(f"Log Loss       : {log_loss(y_test, y_proba):.4f}")

    fig, axes = plt.subplots(2, 2, figsize = (14, 12))

    fig.suptitle(f'Dashboard Avanzado de Evaluación - {model_name}', fontsize = 16, y = 0.98)


    # 1. ROC Curve

    axes[0, 0].plot(fpr, tpr, label = f"ROC curve (AUC  =  {roc_auc_val:.4f})", color = 'darkred', lw = 2)
    axes[0, 0].plot([0, 1], [0, 1], linestyle = "--", color = 'navy')
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].set_title("Curva ROC")
    axes[0, 0].legend(loc = "lower right")
    axes[0, 0].grid(alpha = 0.3)


    # 2. Precision-Recall Curve

    axes[0, 1].plot(recall, precision, label = f"PR curve (AUC  =  {pr_auc:.4f})", color = 'royalblue', lw = 2)
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].set_title("Curva Precision-Recall")
    axes[0, 1].legend(loc = "lower left")
    axes[0, 1].grid(alpha = 0.3)


    # 3. Curva de Calibración

    axes[1, 0].plot(prob_pred, prob_true, "s-", label = f"{model_name}", color = 'limegreen')
    axes[1, 0].plot([0, 1], [0, 1], "k:", label = "Perfectamente calibrado")
    axes[1, 0].set_xlabel("Probabilidad Media Predicha")
    axes[1, 0].set_ylabel("Fracción de Positivos Reales")
    axes[1, 0].set_title("Curva de Calibración (Reliability)")
    axes[1, 0].legend(loc = "lower right")
    axes[1, 0].grid(alpha = 0.3)


    # 4. Matriz de Confusión Normalizada

    sns.heatmap(cm, annot = True, fmt = ".2f", cmap = "viridis",
                xticklabels = ["No Pago", "Pago"], yticklabels = ["No Pago", "Pago"], ax = axes[1, 1])
    
    axes[1, 1].set_title("Matriz de Confusión (Normalizada por Fila)")
    axes[1, 1].set_ylabel("Clase Real")
    axes[1, 1].set_xlabel("Predicción")

    plt.tight_layout(rect = [0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path)
    plt.close()



# Funcipon de Entrenamiento Model Final y análsiis de Estabilidad

def evaluate_and_analyze(pipeline_model, X_train, y_train, X_test, y_test, model_name = "Model"):

    print(f"\n{'=' * 60}")
    print(f"EVALUACIÓN Y ANÁLISIS: {model_name.upper()}")
    print(f"{'=' * 60}")

    cv_results = crossval_detailed_metrics(pipeline_model, X_train, y_train, cv = 5)
    pipeline_model.fit(X_train, y_train)

    y_test_pred = pipeline_model.predict(X_test)
    test_f1_macro = f1_score(y_test, y_test_pred, average = "macro")
    
    print(f"\n ===CLASSIFICATION REPORT SOBRE TEST===")
    print(classification_report(y_test, y_test_pred, target_names = ['No Pago (0)', 'Pago (1)']))

    cv_f1_macro_mean = cv_results['f1_macro'].mean()
    f1_diff = abs(test_f1_macro - cv_f1_macro_mean)

    print(f"ANÁLISIS DE ESTABILIDAD (F1-Macro):")
    print(f"• Rendimiento esperado (CV):   {cv_f1_macro_mean:.4f}")
    print(f"• Rendimiento real (Test):     {test_f1_macro:.4f}")
    print(f"• Diferencia (Over/Under-fit): {f1_diff:.4f}")

    if f1_diff < 0.05:

        print("-> Modelo MUY ESTABLE y confiable.")

    elif f1_diff < 0.10:

        print("-> Diagnóstico: Modelo MODERADAMENTE estable.")

    else:

        print("-> Diagnóstico: Modelo INESTABLE y no confiable")

    return cv_results, test_f1_macro



# -------------------------------------------------------------------------------------------------
# 3. FUNCIONES DE AJUSTE DE HIPERPARÁMETROS (CON SMOTE INCLUIDO)
# -------------------------------------------------------------------------------------------------


# Regresión Logística

def logistic_regression_tuning_smote(trial, X_train, y_train, cv = cv_strategy, metric = "f1_macro"):

    solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear", "saga"])
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    C = trial.suggest_float("C", 1e-4, 10, log = True)
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0) if penalty == "elasticnet" else None

    if solver == "lbfgs" and penalty != "l2": return 0.0
    if solver == "liblinear" and penalty == "elasticnet": return 0.0
    if solver != "saga" and penalty == "elasticnet": return 0.0

    model = LogisticRegression(solver = solver, penalty = penalty, C = C, class_weight = class_weight, l1_ratio = l1_ratio)
    
    full_pipeline = ImblearnPipeline(steps = pasos_preprocesamiento + [
        ('smote', SMOTE(sampling_strategy = 0.33, random_state = 42)),   
        ('model', model)                                              
    ])
    
    return cross_val_score(full_pipeline, X_train, y_train, cv = cv, scoring = metric).mean()


# Árbol de Decisión

def decision_tree_tuning_smote(trial, X_train, y_train, cv = cv_strategy, metric = "f1_macro"):

    model = DecisionTreeClassifier(
        max_depth = trial.suggest_int("max_depth", 2, 7),
        min_samples_split = trial.suggest_int("min_samples_split", 10, 50),
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 20),
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"]),
        random_state = 42, class_weight = trial.suggest_categorical("class_weight", ["balanced", None])
    )

    full_pipeline = ImblearnPipeline(steps = pasos_preprocesamiento + [
        ('smote', SMOTE(sampling_strategy = 0.33, random_state = 42)),   
        ('model', model)                                              
    ])

    return cross_val_score(full_pipeline, X_train, y_train, cv = cv, scoring = metric).mean()


# Random Forest

def random_forest_tuning_smote(trial, X_train, y_train, cv = cv_strategy, metric = "f1_macro"):
    
    model = RandomForestClassifier(
        n_estimators = trial.suggest_int("n_estimators", 50, 150),
        max_depth = trial.suggest_int("max_depth", 3, 15),
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 30),
        min_samples_split = trial.suggest_int("min_samples_split", 10, 50),
        random_state = 42, n_jobs = -1, class_weight = trial.suggest_categorical("class_weight", ["balanced", None])
    )

    full_pipeline = ImblearnPipeline(steps = pasos_preprocesamiento + [
        ('smote', SMOTE(sampling_strategy = 0.33, random_state = 42)),   
        ('model', model)                                              
    ])

    return cross_val_score(full_pipeline, X_train, y_train, cv = cv, scoring = metric).mean()


# LightGBM

def lightGBM_tuning_smote(trial, X_train, y_train, cv = cv_strategy, metric = "f1_macro"):

    model = LGBMClassifier(
        n_estimators = trial.suggest_int("n_estimators", 50, 150),
        max_depth = trial.suggest_int("max_depth", 2, 7),
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-1, log = True),
        num_leaves = trial.suggest_int("num_leaves", 10, 30),
        subsample = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        random_state = 42, n_jobs = 1, verbosity = -1
    )

    full_pipeline = ImblearnPipeline(steps = pasos_preprocesamiento + [
        ('smote', SMOTE(sampling_strategy = 0.33, random_state = 42)),   
        ('model', model)                                              
    ])   

    return cross_val_score(full_pipeline, X_train, y_train, cv = cv, scoring = metric).mean()


# SVM (Máquina de Soporte Vectorial)

def svm_tuning_smote(trial, X_train, y_train, cv = cv_strategy, metric = "f1_macro"):

    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'sigmoid'])
    gamma = trial.suggest_float('gamma', 1e-3, 1e-1, log = True)
    model = SVC(kernel = kernel, gamma = gamma, probability = True)

    full_pipeline = ImblearnPipeline(steps = pasos_preprocesamiento + [
        ('smote', SMOTE(sampling_strategy = 0.33, random_state = 42)),   
        ('model', model)                                              
    ])

    return cross_val_score(full_pipeline, X_train, y_train, cv = cv, scoring = metric).mean()


# XGBoost

def xgboost_tuning_smote(trial, X_train, y_train, cv = cv_strategy, metric = "f1_macro"):

    model = XGBClassifier(
        n_estimators = trial.suggest_int("n_estimators", 50, 150),
        max_depth = trial.suggest_int("max_depth", 2, 7),
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-1, log = True),
        subsample = trial.suggest_float("subsample", 0.5, 1.0),
        eval_metric = 'logloss', random_state = 42, n_jobs = 1
    )

    full_pipeline = ImblearnPipeline(steps = pasos_preprocesamiento + [
        ('smote', SMOTE(sampling_strategy = 0.33, random_state = 42)),   
        ('model', model)                                              
    ])

    return cross_val_score(full_pipeline, X_train, y_train, cv = cv, scoring = metric).mean()



# -------------------------------------------------------------------------------------------------
# FUNCIONES AUXILIARES (UMBRALES)
# -------------------------------------------------------------------------------------------------

def optimizar_y_graficar_umbral(modelo_pipeline, X_test, y_test, nombre_modelo, ax):

    proba_clase_0 = modelo_pipeline.predict_proba(X_test)[:, 0]
    umbrales = np.linspace(0.05, 0.95, 91)

    recalls, precisions, f1_macros = [], [], []
    
    for t in umbrales:

        y_pred_simulada = np.where(proba_clase_0 >= t, 0, 1)

        recalls.append(recall_score(y_test, y_pred_simulada, pos_label = 0, zero_division = 0))
        precisions.append(precision_score(y_test, y_pred_simulada, pos_label = 0, zero_division = 0))
        f1_macros.append(f1_score(y_test, y_pred_simulada, average = 'macro'))
        
    mejor_idx = np.argmax(f1_macros)
    mejor_umbral = umbrales[mejor_idx]
    
    ax.plot(umbrales, recalls, label = 'Recall C0', color = 'deeppink', alpha = 0.8)
    ax.plot(umbrales, precisions, label = 'Precision C0', color = 'forestgreen', alpha = 0.8)
    ax.plot(umbrales, f1_macros, label = 'F1-Macro', color = 'cornflowerblue', linestyle = '--')
    ax.axvline(x = mejor_umbral, color = 'black', linestyle = ':', alpha = 0.7)
    ax.set_title(f'{nombre_modelo}\n(Best Threshold: {mejor_umbral:.2f})', fontsize = 10)
    ax.set_xlabel('Umbral')
    ax.set_ylabel('Score')
    ax.legend(fontsize = 'small', loc = 'lower left')
    ax.grid(alpha = 0.2)



# -------------------------------------------------------------------------------------------------
# BLOQUE DE EJECUCIÓN PRINCIPAL
# -------------------------------------------------------------------------------------------------

def main():
    

    # --- 1. CONFIGURACIÓN DE RUTAS GENÉRICAS ---
    
    load_dotenv()

    pth = Path(os.getenv("DATA_FOLDER"))
    ruta_guardado = Path(os.getenv("ARTIFACTS"))
    ruta_reporte = Path(os.getenv("REPORT"))

    os.makedirs(ruta_guardado, exist_ok = True)
    os.makedirs(ruta_reporte, exist_ok = True)
    

    # --- 2. CARGAR DATOS ---

    print("Cargando datos y pipeline...")

    X = pd.read_csv(pth / "X_base.csv")
    y = pd.read_csv(pth / "y_base.csv")["Pago_atiempo"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SIZE, random_state = 42, stratify = y)

    # Resultados Modelos

    resultados_modelos = {}


    # --- 3. HIPERPARAMETRIZACIÓN SIN SMOTE ---

    modelos_no_smote = {
        'Logistic Regression': (logistic_regression_tuning_no_smote, LogisticRegression),
        'Decision Tree': (decision_tree_tuning_no_smote, DecisionTreeClassifier),
        'SVM': (svm_tuning_no_smote, SVC),
        'Random Forest': (random_forest_tuning_no_smote, RandomForestClassifier),
        'XGBoost': (xgboost_tuning_no_smote, XGBClassifier),
        'LightGBM': (lightGBM_tuning_no_smote, LGBMClassifier),
    }

    for name, (objective, constructor) in modelos_no_smote.items():

        print(f"\n{'-' * 80}")
        print(f"Optimizating WITHOUT SMOTE {name} ...")
        print(f"{'-' * 80}")

        study = optuna.create_study(direction = 'maximize')
        study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials = 10)

        print(f"Mejor F1 de los trials SIN SMOTE: {round(study.best_value, 3)}")
        print(f"Hiperparámetros SIN SMOTE: {study.best_params}")

        best_params = study.best_params.copy()
        final_model_base = constructor(**best_params)
        
        final_pipeline_no_smote = SklearnPipeline(steps=[('preprocessor', pipeline_ml), ('model', final_model_base)])
        
        final_pipeline_no_smote.fit(X_train, y_train)

        y_test_pred = final_pipeline_no_smote.predict(X_test)
        test_f1_macro = f1_score(y_test, y_test_pred, average="macro")

        print(f"Test F1 macro SIN SMOTE: {round(test_f1_macro, 3)}")


    # --- 4. HIPERPARAMETRIZACIÓN CON SMOTE, EVALUACIÓN Y GRÁFICOS ---

    modelos_smote = {
        'Logistic Regression': (logistic_regression_tuning_smote, LogisticRegression),
        'Decision Tree': (decision_tree_tuning_smote, DecisionTreeClassifier),
        'Random Forest': (random_forest_tuning_smote, RandomForestClassifier),
        'XGBoost': (xgboost_tuning_smote, XGBClassifier),
        'LightGBM': (lightGBM_tuning_smote, LGBMClassifier),
        'SVM': (svm_tuning_smote, SVC),
    }

    for name, (objective, constructor) in modelos_smote.items():
        print(f"\n\n{'*' * 80}")
        print(f"INICIANDO OPTIMIZACIÓN Y EVALUACIÓN PARA: {name}")
        print(f"{'*' * 80}")

        study = optuna.create_study(direction = 'maximize')
        study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials = 20)  

        best_params = study.best_params.copy()

        if name == 'SVM':

            best_params['probability'] = True  

        best_base_model = constructor(**best_params)

        final_pipeline = ImblearnPipeline(steps = pasos_preprocesamiento + [
        ('smote', SMOTE(sampling_strategy = 0.33, random_state = 42)),   
        ('model', best_base_model)                                              
])

        cv_detailed, test_f1 = evaluate_and_analyze(
            pipeline_model = final_pipeline,
            X_train = X_train, y_train = y_train,
            X_test = X_test, y_test = y_test,
            model_name = name
        )

        ruta_learning = os.path.join(ruta_reporte, f"CURVA_APRENDIZAJE_{name.replace(' ', '_')}.png")

        plot_learning_curves(
            estimator = final_pipeline,
            X = X_train, y = y_train,
            scoring = "f1_macro",
            model_name = name,
            save_path = ruta_learning
        )

        ruta_eval = os.path.join(ruta_reporte, f"DASHBOARD_METRICAS_{name.replace(' ', '_')}.png")

        plot_advanced_evaluation(
            pipeline_model = final_pipeline, 
            X_test = X_test, 
            y_test = y_test, 
            model_name = name,
            save_path = ruta_eval
        )
        
        resultados_modelos[name] = {
            'model': final_pipeline,
            'parameters': study.best_params,
            'test_f1_macro': test_f1,
            'study': study
        }


    # --- 5. GUARDAR MODELOS FINALISTAS ---

    modelos_finalistas = ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM']
    
    for nombre in modelos_finalistas:

        if nombre in resultados_modelos:

            modelo_pipeline = resultados_modelos[nombre]['model']
            nombre_archivo = f"{nombre.replace(' ', '_')}_final.pkl"
            ruta_completa = os.path.join(ruta_guardado, nombre_archivo)
            
            with open(ruta_completa, 'wb') as archivo:

                pickle.dump(modelo_pipeline, archivo)

            print(f"Modelo {nombre} guardado exitosamente.")

        else:

            print(f"Error")


    # --- 6. OPTIMIZACIÓN DE UMBRALES Y GRÁFICO FINAL ---

    print("\nGenerando gráfico de optimización de umbrales...")

    fig, axes = plt.subplots(2, 2, figsize = (15, 12))
    axes = axes.flatten()  

    for i, nombre in enumerate(modelos_finalistas):

        if nombre in resultados_modelos:

            modelo_actual = resultados_modelos[nombre]['model']

            optimizar_y_graficar_umbral(
                modelo_pipeline = modelo_actual,
                X_test = X_test,
                y_test = y_test,
                nombre_modelo = nombre,
                ax = axes[i]
            )

    plt.tight_layout()

    ruta_umbrales = os.path.join(ruta_reporte, "UMBRALES_OPTIMOS_CLASIFICADORES.png")
    
    plt.savefig(ruta_umbrales)
    
    plt.close()
    

# Ejecucuón del Main

if __name__ == "__main__":

    main()