# Interfaz del Modelo --> Parte Gráfica (streamlit run model_interface)

# Interfaz (UI) - Frontend --> con Streamlit


# Librerías 

import streamlit as st
import requests



# --------
# Título
# --------

st.title("Sistema de Evaluación de Créditos")


# ----------------------------
# Variables y Rangos Posibles
# ----------------------------


capital_prestado = st.number_input("Capital Prestado", min_value = 360000.0, max_value = 45000000.0)
edad_cliente = st.number_input("Edad Cliente", min_value = 18, max_value = 90)
salario_cliente = st.number_input("Salario Cliente", min_value = 0, max_value = 23000000000)
total_otros_prestamos = st.number_input("Capital Prestado otros préstamos (no este)", min_value = 0, max_value = 6800000000)
puntaje_datacredito = st.number_input("Puntaje Datacrédito", min_value = 150.0, max_value = 950.0)
cant_creditosvigentes = st.number_input("Número de créditos vigentes (activos)", min_value = 0, max_value = 65)
huella_consulta = st.number_input("Huella Consulta (veces consultado en centrales)", min_value = 0, max_value = 30)
saldo_mora = st.number_input("Saldo en Mora", min_value = 0.0, max_value = 15000.0)
saldo_total = st.number_input("Saldo Total (intereses y cobros)", min_value = 0.0, max_value = 550000.0)
apalancamiento = st.number_input("Apalancamiento ((capital_prestado + total_otros_prestamos) / salario_cliente)", min_value = 0.0, max_value = 1500.0)    
intensidad_credito = st.number_input("Intensidad Crediticia (cant_creditosvigentes / edad_cliente)", min_value = 0.0, max_value = 1500.0)
plazo_prestamo = st.number_input("Plazo en Meses del Préstamo", min_value = 2, max_value = 100)
tipo_credito = st.selectbox("Tipo de Crédito", options = [4, 6, 7, 9, 10, 68])
tipo_laboral = st.selectbox("Tipo laboral",options = ["Empleado", "Independiente"])



# -----------------------------------------------------------
# Evaluación y Despliegue del Modelo y llamada a la API"
# -----------------------------------------------------------


if st.button("Evaluar y Predecir"):

    data = {"capital_prestado": capital_prestado,
            "edad_cliente": edad_cliente,
            "salario_cliente": salario_cliente,
            "total_otros_prestamos": total_otros_prestamos,
            "puntaje_datacredito": puntaje_datacredito,
            "cant_creditosvigentes": cant_creditosvigentes,
            "huella_consulta": huella_consulta,
            "saldo_mora": saldo_mora,
            "saldo_total": saldo_total,
            "apalancamiento": apalancamiento,
            "intensidad_credito": intensidad_credito,
            "plazo_prestamo": plazo_prestamo,
            "tipo_credito": tipo_credito,
            "tipo_laboral": tipo_laboral
        }

    response = requests.post("http://127.0.0.1:8000/predict", json = data)
    
    if response.status_code == 200:

        res = response.json()

        st.success(f"Resultado: {res['estado']}")

    else:

        st.error("Error en la conexión con la API")