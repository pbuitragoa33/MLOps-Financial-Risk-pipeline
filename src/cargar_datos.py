# Módulo para cargar datos desde BigQuery con credenciales por defecto de la aplicación.


# Librerías necesarias

import pandas as pd
from google.cloud import bigquery
import os
from dotenv import load_dotenv


# Función para cargar datos 

def cargar_datos_scoring() -> pd.DataFrame:

    """
    Cargar los datos de la tabla 'scoring_creditos' desde BigQuery utilizndo
    las credenciales por defecto de la aplicación (gcloud auth application-default login).
    """

    load_dotenv()

    # 1. Inicializar el cliente (nombre del proyecto)

    client = bigquery.Client(project = os.getenv("PROJECT_GCP"))
    
    # 2. Definir la consulta

    query = """
        SELECT *
        FROM `pro-cientificos-pba.Financiero.scoring_creditos`
    """
    
    # 3. Ejecutar la consulta

    print("Consultando datos en BigQuery...")

    query_job = client.query(query)
    
    # Convertir a DataFrame

    df_creditos = query_job.to_dataframe()

    print(f"Datos cargados exitosamente. Forma del DataFrame: {df_creditos.shape}")
    
    return df_creditos


# Main para probar la función de carga de datos

if __name__ == "__main__":

    df = cargar_datos_scoring()

    print(df.head())