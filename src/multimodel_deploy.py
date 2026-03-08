# Despliegue del Modelo (uvicorn multimodel_deploy:app --reload)

# API del modelo - Backend --> con FastAPI


# Librerías y Clases del Pipeline

import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path

import __main__
from ft_engineering import (AgruparCategorias, ToDF, Outliers, ColumnasNulos, 
                            Imputacion, NuevasVariables, ToCategory, 
                            ColumnasIrrelevantes, EliminarCategorias)

__main__.AgruparCategorias = AgruparCategorias
__main__.ToDF = ToDF
__main__.Outliers = Outliers
__main__.ColumnasNulos = ColumnasNulos
__main__.Imputacion = Imputacion
__main__.NuevasVariables = NuevasVariables
__main__.ToCategory = ToCategory
__main__.ColumnasIrrelevantes = ColumnasIrrelevantes
__main__.EliminarCategorias = EliminarCategorias



# ---------------------------------
# Creación de instancia de la API
# ---------------------------------

app = FastAPI(
    title = "API de Riesgo Crediticio (Panel de Expertos)",
    description = "API que evalúa el riesgo usando 4 modelos de Machine Learning simultáneamente."
)

# ---------------------------------
# Carga MÚLTIPLE de Modelos
# ---------------------------------


load_dotenv()

objetos_pth = Path(os.getenv("ARTIFACTS"))

modelos_cargados = {}

lista_modelos = [
    "Logistic_Regression_final.pkl", 
    "Random_Forest_final.pkl", 
    "XGBoost_final.pkl", 
    "LightGBM_final.pkl"
]

print("Iniciando carga del Panel de Expertos...")

for nombre_archivo in lista_modelos:

    ruta_modelo = objetos_pth / nombre_archivo
    nombre_limpio = nombre_archivo.replace("_final.pkl", "").replace("_", " ")
    
    try: 

        if ruta_modelo.exists():

            modelos_cargados[nombre_limpio] = joblib.load(ruta_modelo)

            print(f"{nombre_limpio} cargado exitosamente.")

        else:

            print(f"Archivo no encontrado: {nombre_archivo}")

    except Exception as e:

        print(f"Error al cargar {nombre_archivo}: {e}")


# --------------------
# Clase de InputData 
# --------------------

class InputData(BaseModel):
    capital_prestado: float
    edad_cliente: int
    salario_cliente: float
    total_otros_prestamos: float
    puntaje_datacredito: float
    cant_creditosvigentes: int
    huella_consulta: int
    saldo_mora: float
    saldo_total: float
    plazo_meses: int
    tipo_credito: int
    tipo_laboral: str


# ---------------------------
# Manejo de Solicitudes HTTP 
# ---------------------------

@app.get("/")

def home():

    return {"mensaje": f"API operativa con {len(modelos_cargados)} modelos expertos listos."}


@app.post("/predict")

def predict(data: InputData):

    df = pd.DataFrame([data.model_dump()])
    resultados_expertos = {}

    try:

        for nombre, pipeline in modelos_cargados.items():

            prediccion = pipeline.predict(df)[0]
            probabilidades = pipeline.predict_proba(df)[0]
            
            prob_impago = probabilidades[0]
            prob_pago = probabilidades[1]

            estado = "ALERTA: El cliente NO PAGARÁ" if prediccion == 0 else "APROBADO: El cliente SÍ PAGARÁ"

            resultados_expertos[nombre] = {
                "prediccion_clase": int(prediccion),
                "estado": estado,
                "probabilidad_de_impago": f"{prob_impago * 100:.2f}%",
                "probabilidad_de_pago": f"{prob_pago * 100:.2f}%",
                "prob_impago_raw": float(prob_impago) 
            }

        return resultados_expertos

    except Exception as e:

        raise HTTPException(status_code=500, detail = f"Error procesando la predicción: {str(e)}")