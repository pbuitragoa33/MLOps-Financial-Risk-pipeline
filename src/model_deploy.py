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

app = FastAPI(title = "API de Predicción de Pagos")


# ---------------------------------
# Carga del modelo y del pipeline
# ---------------------------------

load_dotenv()


objetos_pth = Path(os.getenv("OBJECTS"))

try: 

    pipeline = joblib.load(objetos_pth / "pipeline_ml.pkl")
    modelo_LR = joblib.load(objetos_pth / "Logistic_Regression_final.pkl")

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
    apalancamiento: float
    intensidad_credito: float
    plazo_prestamo: int
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
        "apalancamiento": [data.apalancamiento],
        "intensidad_credito": [data.intensidad_credito],
        "plazo_prestamo": [data.plazo_prestamo],
        "tipo_credito": [data.tipo_credito],
        "tipo_laboral": [data.tipo_laboral]
    })

    try:

        # Ejecución de Pipeline

        X_procesado = pipeline.transform(df)

        # Predección

        prediccion = modelo_LR.predict(X_procesado)

        resultado = int(prediccion[0])

        # Lógica de negocio

        estado = "El cliente SI PAGA" if resultado > 0 else  "El cliente NO PAGA"

        return {
            "predicción": resultado,
            "estado": estado
        }

    except Exception as e:

        raise HTTPException(status_code = 500, detail = str(e)) 