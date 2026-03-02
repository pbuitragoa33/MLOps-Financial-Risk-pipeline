# Model Training and Evaluation


# Librerias 

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import warnings
warnings.filterwarnings("ignore")
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.metrics import (accuracy_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, f1_score, precision_recall_curve, roc_curve, confusion_matrix, classification_report)

from sklearn.model_selection import (cross_val_score, StratifiedKFold, ShuffleSplit, learning_curve, train_test_split)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------------------------------------------------------------------------------------------

# Función de partición de datos

def data_division(data: pd.DataFrame, test_size: float, target_column: str = "Pago_atiempo"):

    # Definir X, y

    X = data.drop(columns = [target_column])
    y = data[target_column]

    # Definir los subsets de train y test

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42, stratify = y)

    return X_train, X_test, y_train, y_test


# Resultados en un diccionario

resultados_modelos = {}

# Definir StratifiedKFold

cv_strategy = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)




# --------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------- AJUSTE DE HIPERPARÁMETROS  ------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------- AJUSTE DE HIPERPARÁMETROS SIN SMOTE ------------------------------------------------------------------------------------


# ---------------------------------------- Tuning Regresión Logística  ------------------------------------------------------------------------------------

def logistic_regression_tuning(trial, cv = cv_strategy, metric: str = "f1_macro"):

    solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear", "saga"])
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    C = trial.suggest_float("C", 1e-4, 10, log = True)
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

    if penalty == "elasticnet":

        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)

    else:

        l1_ratio = None

    if solver == "lbfgs" and penalty != "l2":

        return 0.0
    
    if solver == "liblinear" and penalty == "elasticnet":

        return 0.0
    
    if solver != "saga" and penalty == "elasticnet":

        return 0.0

    model = LogisticRegression(solver = solver, penalty = penalty, C = C, class_weight = class_weight,l1_ratio = l1_ratio)
    
    return cross_val_score(model, X_train, y_train, cv = cv, scoring = metric).mean()


# ---------------------------------------- Tuning Árbol de Decisión ------------------------------------------------------------------------------------

def decision_tree_tuning(trial, cv = cv_strategy, metric: str = "f1_macro"):

     model = DecisionTreeClassifier(
        max_depth = trial.suggest_int("max_depth", 2, 7),
        min_samples_split = trial.suggest_int("min_samples_split", 10, 50),
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 20),
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"]),
        random_state = 42,
        class_weight = trial.suggest_categorical("class_weight", ["balanced", None])
    )

     return cross_val_score(model, X_train, y_train, cv = cv, scoring = metric).mean()


# ---------------------------------------- Tuning Random Forest ------------------------------------------------------------------------------------

def random_forest_tuning(trial, cv = cv_strategy, metric: str = "f1_macro"):

    model = RandomForestClassifier(
        n_estimators = trial.suggest_int("n_estimators", 50, 150),
        max_depth = trial.suggest_int("max_depth", 3, 15),
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 30),
        min_samples_split = trial.suggest_int("min_samples_split", 10, 50),
        random_state = 42, n_jobs = -1,
        class_weight = trial.suggest_categorical("class_weight", ["balanced", None])
    )
    return cross_val_score(model, X_train, y_train, cv = cv, scoring = metric).mean()


# ---------------------------------------- Tuning Light GBM ------------------------------------------------------------------------------------

def lightGBM_tuning(trial, cv = cv_strategy, metric: str = "f1_macro"):
    
    model = LGBMClassifier(
        n_estimators = trial.suggest_int("n_estimators", 50, 150),
        max_depth = trial.suggest_int("max_depth", 2, 7),
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-1, log = True),
        num_leaves = trial.suggest_int("num_leaves", 10, 30),
        subsample = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        random_state = 42, n_jobs=-1, verbosity = -1
    )

    return cross_val_score(model, X_train, y_train, cv = cv, scoring = metric).mean()


# ---------------------------------------- Tuning SVM ------------------------------------------------------------------------------------

def svm_tuning(trial, cv = cv_strategy, metric: str = "f1_macro"):

    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'sigmoid'])
    gamma = trial.suggest_float('gamma', 1e-3, 1e-1, log = True)
    model = SVC(kernel = kernel, gamma = gamma)

    return cross_val_score(model, X_train, y_train, cv = cv, scoring = metric).mean()


# ---------------------------------------- Tuning XGBoost  ------------------------------------------------------------------------------------

def xgboost_tuning(trial, cv = cv_strategy, metric: str = "f1_macro"):
    
    model = XGBClassifier(
        n_estimators = trial.suggest_int("n_estimators", 50, 150),
        max_depth = trial.suggest_int("max_depth", 2, 7),
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-1, log = True),
        subsample = trial.suggest_float("subsample", 0.5, 1.0),
        eval_metric = 'logloss',random_state = 42,n_jobs = -1
    )

    return cross_val_score(model, X_train, y_train, cv = cv, scoring = metric).mean()




# ---------------------------------------- HIPERPARAMETRIZACIÓN COMPLETA  ------------------------------------------------------------------------------------

modelos_clasificadores = {
    'Logistic Regression': (logistic_regression_tuning, LogisticRegression),
    'Decision Tree': (decision_tree_tuning, DecisionTreeClassifier),
    'SVM': (svm_tuning, SVC),
    'Random Forest': (random_forest_tuning, RandomForestClassifier),
    'XGBoost': (xgboost_tuning, XGBClassifier),
    'LightGBM': (lightGBM_tuning, LGBMClassifier),

}


for name, (objective, constructor) in modelos_clasificadores.items():

    print("------------------------------------------------------------------------------------------------")
    print("Optimizating WITHOUT SMOTE", name, " ...")
    print("------------------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------------------")

    study = optuna.create_study(direction = 'maximize')
    study.optimize(objective, n_trials = 10)

    print(f"Mejor F1 de los trials SIN SMOTE: {round(study.best_value, 3)}")
    print(f"Hiperparámetros SIN SMOTE: {study.best_params}")

    # Train the model with the best hyperparameter fit
  
    final_model = constructor(**study.best_params)


    # Fit the model

    final_model.fit(X_train, y_train)

    # F1 Macro

    y_train_pred = final_model.predict(X_train)
    train_f1_macro = f1_score(y_train, y_train_pred, average = "macro")

    print(f"Train F1 macro SIN SMOTE: {round(train_f1_macro, 3)}")

    # Ver comportamiento en el test

    y_test_pred = final_model.predict(X_test)
    test_f1_macro = f1_score(y_test, y_test_pred, average = "macro")

    print(f"Test F1 macro SIN SMOTE: {round(test_f1_macro, 3)}")
    
    print("Classification Report sobre Test SIN SMOTE")
    print(classification_report(y_test, final_model.predict(X_test)))

    # Guardar resultados

    resultados_modelos[name] = {
        'model': final_model,
        'parameters': study.best_params,
        'train_f1_model': train_f1_macro,
        'test_f1_model': test_f1_macro,
        'study': study
    }



# ---------------------------------------- AJUSTE DE HIPERPARÁMETROS CON SMOTE  ------------------------------------------------------------------------------------


# ---------------------------------------- Tuning Regresión Logística  ------------------------------------------------------------------------------------

def logistic_regression_tuning(trial, cv = cv_strategy, metric: str = "f1_macro"):

    solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear", "saga"])
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    C = trial.suggest_float("C", 1e-4, 10, log = True)
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

    if penalty == "elasticnet":

        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)

    else:

        l1_ratio = None

    if solver == "lbfgs" and penalty != "l2":

        return 0.0

    if solver == "liblinear" and penalty == "elasticnet":

        return 0.0

    if solver != "saga" and penalty == "elasticnet":

        return 0.0

    model = LogisticRegression(solver = solver, penalty = penalty, C = C, class_weight = class_weight,l1_ratio = l1_ratio)

    smote = SMOTE(sampling_strategy = 0.33, random_state = 42)
    pipeline = Pipeline(steps = [('smote', smote), ('model', model)])

    return cross_val_score(pipeline, X_train, y_train, cv = cv, scoring = metric).mean()


# ---------------------------------------- Tuning Árbol de Decisión ------------------------------------------------------------------------------------

def decision_tree_tuning(trial, cv = cv_strategy, metric: str = "f1_macro"):

    model = DecisionTreeClassifier(
        max_depth = trial.suggest_int("max_depth", 2, 7),
        min_samples_split = trial.suggest_int("min_samples_split", 10, 50),
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 20),
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"]),
        random_state = 42,
        class_weight = trial.suggest_categorical("class_weight", ["balanced", None])
    )

    smote = SMOTE(sampling_strategy = 0.33, random_state = 42)
    pipeline = Pipeline(steps = [('smote', smote), ('model', model)])

    return cross_val_score(pipeline, X_train, y_train, cv = cv, scoring = metric).mean()


# ---------------------------------------- Tuning Random Forest ------------------------------------------------------------------------------------

def random_forest_tuning(trial, cv = cv_strategy, metric: str = "f1_macro"):

    model = RandomForestClassifier(
        n_estimators = trial.suggest_int("n_estimators", 50, 150),
        max_depth = trial.suggest_int("max_depth", 3, 15),
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 30),
        min_samples_split = trial.suggest_int("min_samples_split", 10, 50),
        random_state = 42, n_jobs = -1,
        class_weight = trial.suggest_categorical("class_weight", ["balanced", None])
    )

    smote = SMOTE(sampling_strategy = 0.33, random_state = 42)
    pipeline = Pipeline(steps = [('smote', smote), ('model', model)])

    return cross_val_score(pipeline, X_train, y_train, cv = cv, scoring = metric).mean()


# ---------------------------------------- Tuning Light GBM ------------------------------------------------------------------------------------

def lightGBM_tuning(trial, cv = cv_strategy, metric: str = "f1_macro"):

    model = LGBMClassifier(
        n_estimators = trial.suggest_int("n_estimators", 50, 150),
        max_depth = trial.suggest_int("max_depth", 2, 7),
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-1, log = True),
        num_leaves = trial.suggest_int("num_leaves", 10, 30),
        subsample = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        random_state = 42, n_jobs=-1, verbosity = -1
    )

    smote = SMOTE(sampling_strategy = 0.33, random_state = 42)
    pipeline = Pipeline(steps = [('smote', smote), ('model', model)])

    return cross_val_score(pipeline, X_train, y_train, cv = cv, scoring = metric).mean()


# ---------------------------------------- Tuning SVM ------------------------------------------------------------------------------------

def svm_tuning(trial, cv = cv_strategy, metric: str = "f1_macro"):

    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'sigmoid'])
    gamma = trial.suggest_float('gamma', 1e-3, 1e-1, log = True)
    model = SVC(kernel = kernel, gamma = gamma)

    smote = SMOTE(sampling_strategy = 0.33, random_state = 42)
    pipeline = Pipeline(steps = [('smote', smote), ('model', model)])

    return cross_val_score(pipeline, X_train, y_train, cv = cv, scoring = metric).mean()


# ---------------------------------------- Tuning XGBoost  ------------------------------------------------------------------------------------

def xgboost_tuning(trial, cv = cv_strategy, metric: str = "f1_macro"):

    model = XGBClassifier(
        n_estimators = trial.suggest_int("n_estimators", 50, 150),
        max_depth = trial.suggest_int("max_depth", 2, 7),
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-1, log = True),
        subsample = trial.suggest_float("subsample", 0.5, 1.0),
        eval_metric = 'logloss',random_state = 42,n_jobs = -1
    )

    smote = SMOTE(sampling_strategy = 0.33, random_state = 42)
    pipeline = Pipeline(steps = [('smote', smote), ('model', model)])

    return cross_val_score(pipeline, X_train, y_train, cv = cv, scoring = metric).mean()




# ---------------------------------------- HIPERPARAMETRIZACIÓN COMPLETA  ------------------------------------------------------------------------------------

modelos_clasificadores = {
    'Logistic Regression': (logistic_regression_tuning, LogisticRegression),
    'Decision Tree': (decision_tree_tuning, DecisionTreeClassifier),
    'SVM': (svm_tuning, SVC),
    'Random Forest': (random_forest_tuning, RandomForestClassifier),
    'XGBoost': (xgboost_tuning, XGBClassifier),
    'LightGBM': (lightGBM_tuning, LGBMClassifier),

}


for name, (objective, constructor) in modelos_clasificadores.items():

    print("------------------------------------------------------------------------------------------------")
    print("Optimizating WITH SMOTE", name, " ...")
    print("------------------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------------------")

    study = optuna.create_study(direction = 'maximize')
    study.optimize(objective, n_trials = 10)

    print(f"Mejor F1 de los trials CON SMOTE: {round(study.best_value, 3)}")
    print(f"Hiperparámetros CON SMOTE: {study.best_params}")

    # Train the model with the best hyperparameter fit

    final_model = constructor(**study.best_params)

    # Smote

    final_model_sm = Pipeline(steps=[
        ('smote', SMOTE(random_state=42)),
        ('model', final_model)
    ])


    # Fit the model

    final_model_sm.fit(X_train, y_train)

    # F1 Macro

    y_train_pred = final_model_sm.predict(X_train)
    train_f1_macro = f1_score(y_train, y_train_pred, average = "macro")

    print(f"Train F1 macro CON SMOTE: {round(train_f1_macro, 3)}")

    # Ver comportamiento en el set

    y_test_pred = final_model_sm.predict(X_test)
    test_f1_macro = f1_score(y_test, y_test_pred, average = "macro")

    print(f"Test F1 macro: {round(test_f1_macro, 3)}")

    print("Classification Report sobre Train CON SMOTE")
    print(classification_report(y_train, final_model_sm.predict(X_train)))
    print("------------------------------------------------------------------------------------------------")


    # Ver comportamiento en el test

    print("Classification Report sobre Test CON SMOTE")
    print(classification_report(y_test, final_model.predict(X_test)))

    # Guardar resultados

    resultados_modelos[name] = {
        'model': final_model,
        'parameters': study.best_params,
        'train_f1_model': train_f1_macro,
        'test_f1_model': test_f1_macro,
        'study': study
    }

