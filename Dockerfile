# Dockerfile

# Imagen oficial de Python

FROM python:3.13-slim

# Directorio de trabajo dentro del contenedor

WORKDIR /app

# Instalar las dependencias del sistema que puedan ser requeridas por algunos paquetes de ML

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar el requirements.txt al contenedor e instalar las dependencias de Python

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de los archivos del repositorio al contenedor

COPY . .

# Expone los puertos que utiliza la aplicación
# 8000 para FastAPI (Backend)
# 8501 para Streamlit (Frontend)

EXPOSE 8000 8501

# Añadir el directorio src al PYTHONPATH para que los imports funcionen correctamente

ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Comando por defecto para iniciar el Backend (FastAPI).

# docker run -p 8501:8501 financial_risk_pipeline streamlit run src/model_interface.py

CMD ["uvicorn", "src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]