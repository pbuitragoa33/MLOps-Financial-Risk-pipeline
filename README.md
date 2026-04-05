# Financial Risk Pipeline

Pipeline de Ciencia de Datos en Producción para evaluacion de riesgo crediticio, desde la ingesta de datos hasta el despliegue y monitoreo en post-producción.

## 1. Resumen del Proyecto

Este repositorio implementa un flujo end-to-end para predecir probabilidad de impago de clientes de una entidad financiera.

Incluye:
- Ingesta de datos desde Google BigQuery.
- Análisis exploratorio de datos (EDA) completo.
- Feature engineering con transformadores personalizados de sklearn.
- Modelo heuristico basado en reglas de negocio y en el EDA.
- Entrenamiento y optimizacion de varios modelos.
- API de inferencia con FastAPI.
- Interfaz web con Streamlit.
- Registro de predicciónes y monitoreo de drift.

## 2. Objetivo

Construir un sistema reproducible y desplegable que:
- evalua riesgo de impago,
- expone predicciónes por API,
- ofrece interfaz para usuarios no tecnicos,
- y permite monitoreo de cambios de distribución post-despliegue.

## 3. Arquitectura Funcional

Flujo principal:

1. Carga de datos
- `src/cargar_datos.py` consulta `pro-cientificos-pba.Financiero.scoring_creditos` en BigQuery.

2. Preprocesamiento
- `src/ft_engineering.py` define clases de transformacion y genera `pipeline_ml.pkl`.
- tambien exporta `X_base.csv` y `y_base.csv` para entrenamiento.

3. Entrenamiento y evaluacion
- `src/model_training_evaluation.py` hace tuning (ajuste de hiperparámetos) con Optuna, compara modelos y guarda los finalistas.

4. Despliegue y API
- API single-model: `src/model_deploy.py`.
- UI single-model: `src/model_interface.py`.
- API multi-model: `src/multimodel_deploy.py`.
- UI multi-model: `src/multimodel_interface.py`.

Nota: importantes solo los de single-model para el curso.

5. Observabilidad
- Logs de predicción en `src/data/logs_produccion.csv`.
- Monitoreo de drift en `src/model_monitoring.ipynb`.

## 4. Estructura del Repositorio

```text
Financial_Risk_pipeline/
|- docker-compose.yml
|- Dockerfile
|- README.md
|- requirements.txt
|- set_up.bat
|- .env
|- .dockerignore
|- .gitignore
|- env_project/
|- outputs/
|  |- artefactos/
|  |- modelo_heuristico/
|  |- reporte_training_evaluation/
|- src/
	 |- cargar_datos.py
	 |- comprension_eda.ipynb
	 |- config.json
	 |- ft_engineering.py
	 |- heuristic_model.py
	 |- model_deploy.py
	 |- model_interface.py
	 |- model_monitoring.ipynb
	 |- model_training_evaluation.py
	 |- multimodel_deploy.py
	 |- multimodel_interface.py
	 |- data/
			|- logs_produccion.csv
```

## 5. Qué hace cada archivo?

### 5.1 raíz

- `README.md`
	- Documentación del proyecto.

- `requirements.txt`
	- Dependencias Python para entrenamiento, despliegue, análisis y notebooks.

- `Dockerfile`
	- Imagen base `python:3.13-slim`.
	- Instala dependencias del sistema y Python.
	- Expone puertos `8000` (API) y `8501` (frontend).

- `docker-compose.yml`
	- Orquesta dos servicios:
		- `api`: FastAPI con Uvicorn.
		- `frontend`: Streamlit.

- `set_up.bat`
	- Archivo de setup para Windows.

- `.env`
	- Configuración de rutas y proyecto GCP.

- `.dockerignore`
	- Excluye archivos/carpetas no necesarias para build.

- `.gitignore`
	- Excluye entorno virtual, cache, notebooks checkpoints, `.env`, logs y archivos del sistema.

- `env_project/`
	- Entorno virtual local del proyecto.

### 5.2 Carpeta src

- `src/cargar_datos.py`
	- Función `cargar_datos_scoring()`:
		- lee `PROJECT_GCP` desde `.env`.
		- crea cliente BigQuery.
		- ejecuta consulta SQL.
		- devuelve `DataFrame`.

- `src/ft_engineering.py`
	- Define transformadores personalizados:
		- `ToDF`
		- `ColumnasNulos`
		- `Imputacion`
		- `Outliers`
		- `NuevasVariables`
		- `ToCategory`
		- `ColumnasIrrelevantes`
		- `EliminarCategorias`
		- `AgruparCategorias`
	- Construye `pipeline_basemodel` y `pipeline_ml`.
	- Guarda artefactos de preprocesamiento:
		- `outputs/artefactos/pipeline_ml.pkl`
		- `src/data/X_base.csv`
		- `src/data/y_base.csv`

- `src/model_training_evaluation.py`
	- Entrenamiento principal y comparación de modelos ML.
	- Tuning con Optuna (con y sin SMOTE).
	- Modelos evaluados:
		- Logistic Regression
		- Decision Tree
		- Random Forest
		- XGBoost
		- LightGBM
		- SVM
	- Exporta:
		- modelos finalistas `.pkl` a `outputs/artefactos/`
		- reportes graficos por modelo en `outputs/reporte_training_evaluation/`
		- grafico de umbrales optimos.

- `src/heuristic_model.py`
	- Baseline por reglas de negocio (modelo heuristico).
	- Evalua métricas, genera confusion matrix y learning curves.
	- Guarda imagenes en `outputs/modelo_heuristico/`.

- `src/model_deploy.py`
	- API FastAPI para predicción con modelo unico.
	- Carga `Logistic_Regression_final.pkl` desde `ARTIFACTS`.
	- Endpoints:
		- `GET /`
		- `POST /predict`
	- Guarda logs de inferencia en `src/data/logs_produccion.csv`.

- `src/model_interface.py`
	- Frontend Streamlit para consumir la API de modelo unico.
	- Recolecta inputs de cliente, invoca `/predict` y muestra estado/probabilidades.

- `src/multimodel_deploy.py`
	- API FastAPI que carga 4 modelos en paralelo:
		- Logistic Regression
		- Random Forest
		- XGBoost
		- LightGBM
	- Retorna resultado individual por modelo.

- `src/multimodel_interface.py`
	- Frontend Streamlit para panel multi-modelo.
	- Calcula consenso por promedio de riesgo (`prob_impago_raw`).

- `src/comprension_eda.ipynb`
	- Notebook de EDA completo.
	- Incluye diccionario de datos, nulos, duplicados, estadistica descriptiva, sesgo, curtosis y visualizaciones.

- `src/model_monitoring.ipynb`
	- Notebook de monitoreo de drift en producción.
	- Compara referencia vs producción con:
		- KS-Test (numericas)
		- Chi-Square (categoricas)
		- PSI

- `src/data/logs_produccion.csv`
	- Registro histórico de inferencias:
		- variables de entrada
		- timestamp
		- predicción
		- probabilidad de impago

### 5.3 Carpeta outputs

- `outputs/artefactos/`
	- artefactos serializados:
		- `pipeline_ml.pkl`
		- `Logistic_Regression_final.pkl`
		- `Random_Forest_final.pkl`
		- `XGBoost_final.pkl`
		- `LightGBM_final.pkl`

- `outputs/modelo_heuristico/`
	- reportes visuales del baseline heuristico.

- `outputs/reporte_training_evaluation/`
	- subcarpetas por modelo con:
		- curvas de aprendizaje
		- dashboard de metricas
	- incluye `UMBRALES_OPTIMOS_CLASIFICADORES.png`.

## 6. Stack y Tecnologías usadas

### Lenguaje
- Python 3.13

### Ciencia de datos y ML
- pandas
- numpy
- scikit-learn
- imbalanced-learn (SMOTE)
- optuna
- xgboost
- lightgbm
- scipy
- joblib/pickle

### Backend
- fastapi
- uvicorn
- pydantic

### Frontend
- streamlit
- requests

### Visualización
- matplotlib
- seaborn

### Integración de datos
- google-cloud-bigquery
- python-dotenv

### MLOps y contenedores
- docker
- docker-compose

### Notebooks
- jupyter

## 7. Variables de entorno

Definidas en `.env`:

- `DATA_FOLDER`
	- ruta de datos locales (`X_base.csv`, `y_base.csv`, `logs_produccion.csv`).

- `OUTPUTS`
	- ruta raíz de salidas.

- `ARTIFACTS`
	- ruta de modelos y pipeline serializados.

- `REPORT`
	- ruta de reportes graficos de entrenamiento/evaluación.

- `PROJECT_GCP`
	- id de proyecto para cliente BigQuery.

## 8. Como ejecutar el proyecto?

### 8.1 Ejecución local

Requisitos:
- Python 3.13
- entorno virtual activado
- credenciales GCP para BigQuery

Pasos recomendados:

1. Activar entorno virtual

```powershell
& "env_project\Scripts\Activate.ps1"
```

2. Instalar dependencias (si aplica)

```powershell
pip install -r requirements.txt
```

3. Generar pipeline y datasets base

```powershell
cd src
python ft_engineering.py
```

4. Entrenar y evaluar modelos

```powershell
python model_training_evaluation.py
```

5. Levantar API single-model

```powershell
uvicorn model_deploy:app --host 0.0.0.0 --port 8000
```

6. Levantar frontend single-model (otra terminal)

```powershell
streamlit run model_interface.py
```

### 8.2 Modo multi-modelo

API:

```powershell
uvicorn multimodel_deploy:app --host 0.0.0.0 --port 8000
```

Frontend:

```powershell
streamlit run multimodel_interface.py
```

### 8.3 Ejecución con Docker

Desde raíz del repo:

```powershell
docker compose up --build
```

Servicios:
- API: http://localhost:8000
- Frontend: http://localhost:8501

## 9. Endpoints de API

### `GET /`
Endpoint de salud.

### `POST /predict`
Recibe las variables de entrada del cliente y retorna:
- clase predicha
- estado de riesgo
- probabilidad de impago
- probabilidad de pago

En multi-modelo, retorna un bloque por modelo experto.

## 10. Monitoreo de drift

El notebook `src/model_monitoring.ipynb` realiza comparacion de distribuciones entre:
- datos de referencia (`X_base.csv`)
- datos de produccion (`logs_produccion.csv`)

Metricas:
- KS-Test
- Chi-Square
- PSI

Reglas PSI usadas:
- `PSI < 0.1`: sin impacto.
- `0.1 <= PSI < 0.2`: precaución.
- `PSI >= 0.2`: peligro.

## 11. Estado actual y notas

- El repositorio incluye `env_project/` con paquetes instalados localmente.
- El flujo depende de BigQuery y credenciales GCP validas.