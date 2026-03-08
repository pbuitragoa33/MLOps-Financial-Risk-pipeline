# Despliegue del Modelo (uvicorn model_deploy:app --reload)


# API del modelo - Backend --> con FastAPI


# Librerías

import numpy as np
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import os
import uvicorn


# ---------------------------------
# Creación de instancia de la API
# ---------------------------------

app = FastAPI(title = "API de Predicción de Pagos", 
              description = "API para predecir la probabilidad de impago de clientes " 
              "usando Regresión Logística.")


# ---------------------------------
# Carga del modelo y del pipeline
# ---------------------------------

load_dotenv()


objetos_pth = Path(os.getenv("ARTIFACTS"))

try: 

    pipeline_modelo_LR = joblib.load(objetos_pth / "Logistic_Regression_final.pkl")
    
    print("Pipeline y modelo cargado exitosamente.")

except Exception as e:

    print(f"Error al cargar objetos: {e}")


# --------------------
# Clase de InputData 
# --------------------


class InputData(BaseModel):

    """
    Pedir los datos para ingresar al modelo para poder hacer predicciones.
    """

    capital_prestado: float
    edad_cliente: int
    salario_cliente: int
    total_otros_prestamos: int
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

    return {"mensaje": "La API para la predicción está funcionando..."}


@app.post("/predict")

def predict(data: InputData):

    df = pd.DataFrame({
        "capital_prestado": [data.capital_prestado],
        "edad_cliente": [data.edad_cliente],
        "salario_cliente": [data.salario_cliente],
        "total_otros_prestamos": [data.total_otros_prestamos],
        "puntaje_datacredito": [data.puntaje_datacredito],
        "cant_creditosvigentes": [data.cant_creditosvigentes],
        "huella_consulta": [data.huella_consulta],
        "saldo_mora": [data.saldo_mora],
        "saldo_total": [data.saldo_total],
        "plazo_meses": [data.plazo_meses],
        "tipo_credito": [data.tipo_credito],
        "tipo_laboral": [data.tipo_laboral]
    })

    try:

        # Ejecución de Pipeline (incluye pipeline de procesamiento y el modelo para predecir)

        df = pd.DataFrame([data.model_dump()])

        # Predección

        prediccion = pipeline_modelo_LR.predict(df)[0]

        # Probabilidades para ambas clases

        probabilidades = pipeline_modelo_LR.predict_proba(df)[0]
        prob_impago = probabilidades[0] 
        prob_pago = probabilidades[1]

        # Lógica de negocio

        if prediccion == 0:

            estado = "ALERTA: El cliente NO PAGARÁ (Moroso)"

        else:

            estado = "APROBADO: El cliente SÍ PAGARÁ"

        return {
            "prediccion_clase": int(prediccion),
            "estado": estado,
            "probabilidad_de_impago": f"{prob_impago * 100:.2f}%",
            "probabilidad_de_pago": f"{prob_pago * 100:.2f}%"
        }

    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Error procesando la predicción: {str(e)}")