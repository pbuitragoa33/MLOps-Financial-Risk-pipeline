# Plantilla Genérica para Feature Engineering 


# Librerías

import pandas as pd
import numpy as np
import os
import pickle
from dotenv import load_dotenv
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from cargar_datos import cargar_datos_scoring


# ----------------------------------------------------- Clases de Transformación -----------------------------------------------------


# ---------------------------------------------------------- 1. Clase ToDF ----------------------------------------------------------


# La clase es un transformador personalizado de sklearn que se encarga de:

# 1) Escalar variables numéricas usando StandardScaler.
# 2) Aplica OneHotEncoder a las variables categóricas.
# 3) Devuelve un DataFrame con los nombres de las columnas resultantes y el mismo índice que el DataFrame original.

class ToDF(BaseEstimator, TransformerMixin):

    # Constructor de la clase --> recibe las listas de columnas numéricas y categóricas.

    def __init__(self, numeric_features, categorical_features):

        """
        Parámetros:

        - numeric_features: lista de nombres de columnas numéricas.
        - categorical_features: lista de nombres de columnas categóricas.

        """

        # Guardar los nombres de las columnas.

        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

        # Para almacenar el ColumnTransformer después de ajustarlo.

        self.ct_ = None


    def fit(self, X, y = None):

        """
        Método fit: ajusta el ColumnTransformer a los datos de entrenamiento
        """

        try:

            ohe = OneHotEncoder(handle_unknown = "ignore", sparse_output = False)

        except TypeError:

            ohe = OneHotEncoder(handle_unknown = "ignore", sparse = False)

        # ColumnTransformer que aplica StandardScaler a las columnas numéricas y OneHotEncoder a las columnas categóricas.    

        self.ct_ = ColumnTransformer(
            transformers = [
                ("num", StandardScaler(), self.numeric_features),
                ("cat", ohe, self.categorical_features),
            ]
        )

        # Se ajusta el ColumnTransformer a los datos.

        self.ct_.fit(X, y)

        return self
    

    def transform(self, X):

        """
        Método transform: transforma los datos y devuelve un DataFrame con las columnas resultantes
        """

        # Transformar los datos usando el ColumnTransformer ajustado.

        Xt = self.ct_.transform(X)

        try:

            # Obtiene los nombres de las columnas resultantes.

            feat_names = self.ct_.get_feature_names_out()

        except AttributeError:

            feat_names = []

            # Itera sobre los transformadores para obtener los nombres de las .

            for name, trans, cols in self.ct_.transformers_:

                if name == "remainder" and trans == "drop":

                    continue

                if hasattr(trans, "get_feature_names_out"):

                    feat_names.extend(trans.get_feature_names_out(cols))

                else:

                    feat_names.extend(cols)

        # Devuelve un DataFrame con los datos transformados, los nombres de las columnas y el mismo índice que el DataFrame original.

        return pd.DataFrame(Xt, columns = feat_names, index = X.index)



# ---------------------------------------------------------- 2. Clase ColumnasNulas ----------------------------------------------------------


# Es un trabsformador que elimina columnas, ya que recibe una lista dada y devuelve el DataFrame sin esas columnas. 

class ColumnasNulos(BaseEstimator, TransformerMixin):

    """
    Parámetros:

    - cols_to_drop: lista de nombres de columnas a eliminar.

    """

    # Constructor de la clase --> recibe la lista de columnas a eliminar.

    def __init__(self, cols_to_drop):

        self.cols_to_drop = cols_to_drop

    def fit(self, X, y = None):

        """
        Método fit: no hace nada, solo devuelve self.
        """

        return self

    def transform(self, X):

        """
        Método transform: elimina las columnas especificadas en cols_to_drop y devuelve el DataFrame resultante.
        """

        return X.drop(columns = self.cols_to_drop, errors = "ignore")



# ---------------------------------------------------------- 3. Clase Imputación ----------------------------------------------------------


# Permite la imputación de valores nulos usando las estrategias de media, mediana o KNN.


class Imputacion(BaseEstimator, TransformerMixin):

    # Constructor de la clase --> recibe la estrategia de imputación, las columnas a imputar y el número de vecinos para KNN.

    def __init__(self, strategy = "median", cols = None, n_neighbors = 5):

        """
        Parámetros:

        - strategy: "mean", "median", "knn", "most_frequent"
        - cols: columnas a imputar. Si es None:
                - para estrategias numéricas → solo columnas numéricas
                - para most_frequent → solo columnas categóricas
        - n_neighbors: número de vecinos (solo para knn)
        """

        self.strategy = strategy
        self.cols = cols
        self.n_neighbors = n_neighbors
        self.imputer_ = None
        self.cols_ = None  

    def fit(self, X, y = None):

        """
        Método fit: ajusta el imputador a los datos de entrenamiento.
        """

        X = X.copy()

        # Validar estrategia y seleccionar columnas a imputar 
        # Si es categórica --> most_frequent
        # Si es numérica --> mean, median o knn

        if self.cols is None:

            if self.strategy in ["mean", "median", "knn"]:

                self.cols_ = X.select_dtypes(include = [np.number]).columns.tolist()

            elif self.strategy == "most_frequent":

                self.cols_ = X.select_dtypes(exclude = [np.number]).columns.tolist()

            else:

                raise ValueError("strategy debe ser: 'mean', 'median', 'knn' o 'most_frequent'")

        else:

            self.cols_ = self.cols

        if len(self.cols_) == 0:

            raise ValueError("No hay columnas válidas para imputar con la estrategia seleccionada.")

        X_fit = X[self.cols_]

        # Validar que las columnas seleccionadas sean del tipo adecuado para la estrategia elegida.

        if self.strategy in ["mean", "median", "knn"]:

            if not all(np.issubdtype(dtype, np.number) for dtype in X_fit.dtypes):

                raise ValueError("Las estrategias 'mean', 'median' y 'knn' solo permiten columnas numéricas.")

        if self.strategy == "most_frequent":

            if all(np.issubdtype(dtype, np.number) for dtype in X_fit.dtypes):
                
                raise ValueError("'most_frequent' está pensada para columnas categóricas.")
            

        # Crear el imputador según la estrategia seleccionada.

        if self.strategy in ["mean", "median", "most_frequent"]:

            self.imputer_ = SimpleImputer(strategy = self.strategy)

        elif self.strategy == "knn":

            self.imputer_ = KNNImputer(n_neighbors = self.n_neighbors)

        # Ajuste

        self.imputer_.fit(X_fit)

        return self

    def transform(self, X):

        """ 
        Método transform: imputa los valores nulos en las columnas seleccionadas y devuelve el DataFrame resultante. 
        """

        X = X.copy()

        X_trans = X[self.cols_]

        imputed = self.imputer_.transform(X_trans)

        X_imputed_df = pd.DataFrame(imputed, columns = X_trans.columns, index = X_trans.index)

        X[self.cols_] = X_imputed_df

        # Retorna el DataFrame con las columnas imputadas. Las demás columnas permanecen sin cambios.

        return X
    


# ---------------------------------------------------------- 4. Clase Outliers ----------------------------------------------------------


# Esta clase se encarga de eliminar filas que contengan valores atípicos en las columnas (Es más hardcodeada)

class Outliers(BaseEstimator, TransformerMixin):

    def fit(self, X, y = None):

        """
        Método fit: no hace nada, solo devuelve self.
        """

        return self

    def transform(self, X):

        """
        Método transform: elimina filas que contengan valores atípicos en las columnas especificadas
        """

        X = X.copy()

        # Casos Particulares (edad, puntaje_datacrefido, ...)

        if "edad_cliente" in X.columns:

            X["edad_cliente"] = X["edad_cliente"].clip(upper = 100)

        if "puntaje_datacredito" in X.columns:

            X["puntaje_datacredito"] = X["puntaje_datacredito"].clip(lower = 150, upper = 950)

        return X
    


# ---------------------------------------------------------- 5. Clase NuevasVariables ----------------------------------------------------------


# Esta clase se encarga de crear nuevas variables a partir de las existentes a partir de otras existentes.


class NuevasVariables(BaseEstimator, TransformerMixin):

    """
    Método fit: no hace nada, solo devuelve self.
    """

    def fit(self, X, y = None):

        return self
    
    """
    Método transform:
    """

    def transform(self, X):

        X = X.copy()

        # Casos Particulares

        X['plazo_prestamo'] = pd.cut(X['plazo_meses'], bins = [0, 6, 18, 120], labels = ['corto_plazo', 'mediano_plazo', 'largo_plazo'], include_lowest = True)
        X['apalancamiento'] = ((X['capital_prestado'] + X['total_otros_prestamos']) / X['salario_cliente'].replace(0, np.nan))
        X['exposicion_por_punto'] = (X['capital_prestado'] / X['puntaje_datacredito']).replace(0, np.nan)
        X['intensidad_credito'] = (X['cant_creditosvigentes'] / X['edad_cliente'].replace(0, np.nan))

        return X



# ---------------------------------------------------------- 6. Clase ToCategory ----------------------------------------------------------


# Lo que hace esta clase es convertir las columnas especificadas a tipo 'category


class ToCategory(BaseEstimator, TransformerMixin):

    # Constructor de la clase --> recibe la lista de columnas a convertir a categoría.

    def __init__(self, cols):

        self.cols = cols

    def fit(self, X,  y = None):

        """
        Método fit: no hace nada, solo devuelve self.
        """

        return self

    def transform(self, X):

        """
        Método transform: convierte las columnas especificadas a tipo 'category' y devuelve el DataFrame resultante.
        """

        X = X.copy()

        for c in self.cols:

            if c in X.columns:

                X[c] = X[c].astype("category")

        return X
    


# ---------------------------------------------------------- 7. Clase ColumnasIrrelevantes ----------------------------------------------------------


# Lo que hace esta clase es eliminar columnas irrelevantes del DataFrame

class ColumnasIrrelevantes(BaseEstimator, TransformerMixin):

    # Constructor de la clase --> recibe la lista de columnas a eliminar.

    def __init__(self, cols_to_drop):
        
        self.cols_to_drop = cols_to_drop

    def fit(self, X, y = None):

        """
        Método fit: no hace nada, solo devuelve self.
        """

        return self

    def transform(self, X):

        """
        Método transform: elimina las columnas especificadas en cols_to_drop y devuelve el DataFrame resultante.
        """

        return X.drop(columns = self.cols_to_drop, errors = "ignore")



# ---------------------------------------------------------- 8. Clase EliminarCategorías ----------------------------------------------------------


# Esta clase elimina filas con categorías específicasen una columnas.

class EliminarCategorias(BaseEstimator, TransformerMixin):

    # Constructor de la clase --> recibe el nombre de la columna objetivo y la lista de categorías a eliminar.  

    def __init__(self, target_col, cats_to_drop):
        
        # Guardar el nombre de la columna objetivo y la lista de categorías a eliminar.

        self.target_col = target_col
        self.cats_to_drop = cats_to_drop

    def fit(self, X, y = None):

        """
        Método fit: no hace nada, solo devuelve self.
        """

        return self

    def transform(self, X):

        """
        Método transform: elimina filas que contengan las categorías especificadas en cats_to_drop en la columna target_col y devuelve el DataFrame resultante.
        """

        X = X.copy()

        if self.target_col in X.columns:

            X.loc[X[self.target_col].isin(self.cats_to_drop), self.target_col] = np.nan

        return X


# ---------------------------------------------------------- 9. Clase AgruparCategorias ----------------------------------------------------------


# Esta clase se encarga de reagrupar categorías de una variable categórica

class AgruparCategorias(BaseEstimator, TransformerMixin):

    def __init__(self, target_col, cats_to_group, new_value = "otros"):

        """
        target_col: columna donde se agruparán categorías
        cats_to_group: lista de valores a agrupar
        new_value: nombre de la nueva categoría
        """

        self.target_col = target_col
        self.cats_to_group = cats_to_group
        self.new_value = new_value

    def fit(self, X, y = None):

        """
        Método fit: no hace nada, solo devuelve self.
        """

        return self

    def transform(self, X):

        """
        Método transform: reagrupa las clases a otra categoría.
        """

        X = X.copy()

        if self.target_col not in X.columns:

            return X

        categorias_a_agrupar = [str(c) for c in self.cats_to_group]
        nuevo_valor_str = str(self.new_value)

        # Formateo de datos con regex

        X[self.target_col] = (
            X[self.target_col].astype(str).str.replace(r'\.0$', '', regex = True).str.strip()
        )

        # Agrupación usando las variables locales que ya son strings

        X[self.target_col] = X[self.target_col].apply(
            lambda x: nuevo_valor_str if x in categorias_a_agrupar else x
        )

        # Cambiar a tipo category
        
        X[self.target_col] = X[self.target_col].astype("category")

        return X
    



# ---------------------------------------------------------- Definición de Varibles ----------------------------------------------------------


# 1. Variables Irrelevantes

variables_irrelevantes = ['fecha_prestamo', 'creditos_sectorCooperativo', 'creditos_sectorReal', 
                        'creditos_sectorFinanciero', 'saldo_principal', 'saldo_mora_codeudor', 
                        'puntaje', 'cuota_pactada', 'exposicion_por_punto']


# 2. Variables con Muchos Nulos

variables_nulidad = ['tendencia_ingresos', 'promedio_ingresos_datacredito']


# 3. Variables que deben ser Categóricas 

variables_categoricas = ['tipo_credito', 'tipo_laboral']


# 4. Numéricas y Categóricas

numeric_features = ['capital_prestado', 'edad_cliente', 'salario_cliente', 'total_otros_prestamos', 'puntaje_datacredito',
                    'cant_creditosvigentes', 'huella_consulta', 'saldo_mora', 'saldo_total', 'apalancamiento', 
                    'intensidad_credito']

categorical_features = ['plazo_prestamo', 'tipo_credito', 'tipo_laboral']


# ---------------------------------------------------------- Pipeline Base Model ----------------------------------------------------------


pipeline_basemodel = Pipeline(steps = [
    ("agrupar_categorias", AgruparCategorias(target_col = "tipo_credito", cats_to_group = [6, 7, 10, 68], new_value = "otros")),
    ("outliers", Outliers()),
    ("nuevas_variables", NuevasVariables()),
    ("columnas_irrelevantes", ColumnasIrrelevantes(cols_to_drop = variables_irrelevantes)),
    ("to_category", ToCategory(cols = variables_categoricas)),
    ("eliminar_nulos", ColumnasNulos(cols_to_drop = variables_nulidad)),
    ("imputacion", Imputacion(strategy = 'knn'))
])



# ---------------------------------------------------------- Pipeline Machine Learning ----------------------------------------------------------


# Preprocesador --> ColumnTransformer

preprocessor = ColumnTransformer(
    transformers = [
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown = "ignore"), categorical_features)
    ]
)

# Pipeline Completo

pipeline_ml = Pipeline(steps = [
    ("basemodel", pipeline_basemodel),
    ("preprocessor", ToDF(numeric_features = numeric_features, categorical_features = categorical_features))
    ]
)



# ---------------------------------------------------------- Ejecución del Pipeline ----------------------------------------------------------


# Función main

def main():

    print("Feature Engineering en proceso...")

    # Varibale de entorno

    load_dotenv()

    # Ruta de los archivos

    pipeline_pickle = Path(os.getenv("ARTIFACTS"))
    pth = Path(os.getenv("DATA_FOLDER"))

    pipeline_pickle.mkdir(parents = True, exist_ok = True)

    # Leer la base de datos

    df = cargar_datos_scoring()

    # Eliminar manualmente outliers

    df = df[df["edad_cliente"] <= 100]
    df = df[df["puntaje_datacredito"].between(150, 950)]

    # Separación de X y y

    target_name = "Pago_atiempo"

    X = df.drop(columns = [target_name])
    y = df[target_name]

    # Guardar X y y crudos para luego hacer fit_transform

    X.to_csv(pth / "X_base.csv", index = False)
    y.to_csv(pth / "y_base.csv", index = False)

    # Guardar el pipeline para el entrenamiento y producción

    with open(pipeline_pickle / "pipeline_ml.pkl", "wb") as f:

        pickle.dump(pipeline_ml, f)


    print("Feature Engineering finalizado")



# Main

if __name__ == "__main__":

    main()