# Interfaz del Modelo --> Parte Gráfica (streamlit run model_interface.py)

# Interfaz (UI) - Frontend --> con Streamlit


# Librerías 

import streamlit as st
import os
import requests



# ----------------------------
# Configuración de la página
# ----------------------------

st.set_page_config(page_title = "Evaluador de Riesgo Crediticio", layout = "centered")

st.title("Sistema de Evaluación de Créditos")
st.markdown("Ingrese los datos del cliente para calcular la probabilidad de incumplimiento o default")
st.divider()


# ----------------------------
# Variables y Rangos Posibles
# ----------------------------


col1, col2 = st.columns(2)

with col1:

    st.subheader("Datos Personales y Laborales")
    edad_cliente = st.number_input("Edad Cliente", min_value = 18, max_value = 90, value = 35)
    salario_cliente = st.number_input("Salario Cliente ($)", min_value = 0.0, max_value = 23000000000.0, value = 2500000.0)
    tipo_laboral = st.selectbox("Tipo laboral", options = ["Empleado", "Independiente"])
    
    st.subheader("Historial Crediticio")
    puntaje_datacredito = st.number_input("Puntaje Datacrédito", min_value = 150.0, max_value = 950.0, value = 700.0)
    huella_consulta = st.number_input("Consultas a centrales (Últimos meses)", min_value = 0, max_value = 30, value = 2)
    cant_creditosvigentes = st.number_input("Número de créditos activos", min_value = 0, max_value = 65, value=1)
    
with col2:

    st.subheader("Detalles del Préstamo Solicitado")
    capital_prestado = st.number_input("Capital Solicitado ($)", min_value = 360000.0, max_value = 45000000.0, value = 5000000.0)
    plazo_meses = st.number_input("Plazo del Préstamo (Meses)", min_value = 2, max_value = 100, value = 12)
    tipo_credito = st.selectbox("Código / Tipo de Crédito", options = [4, 6, 7, 9, 10, 68])
    
    st.subheader("Obligaciones y Mora Actual")
    total_otros_prestamos = st.number_input("Deuda en otros préstamos ($)", min_value = 0.0, max_value = 6800000000.0, value = 0.0)
    saldo_total = st.number_input("Saldo Total de Mora en Deudas Actuales ($)", min_value = 0.0, max_value = 550000.0, value = 0.0)
    saldo_mora = st.number_input("Saldo en Mora Actual ($)", min_value = 0.0, max_value = 15000.0, value = 0.0)

st.divider()



# -----------------------------------------------------------
# Evaluación y Despliegue del Modelo y llamada a la API"
# -----------------------------------------------------------


_, col_btn, _ = st.columns([1, 1, 1])

with col_btn:

    evaluar = st.button("Evaluar y Predecir", use_container_width = True)

if evaluar:

    data = {
        "capital_prestado": capital_prestado,
        "edad_cliente": edad_cliente,
        "salario_cliente": salario_cliente,
        "total_otros_prestamos": total_otros_prestamos,
        "puntaje_datacredito": puntaje_datacredito,
        "cant_creditosvigentes": cant_creditosvigentes,
        "huella_consulta": huella_consulta,
        "saldo_mora": saldo_mora,
        "saldo_total": saldo_total,
        "plazo_meses": plazo_meses,
        "tipo_credito": tipo_credito,
        "tipo_laboral": tipo_laboral
    }

    # Llamada a la API por medio de un spinner visual
    
    with st.spinner('Analizando el perfil de riesgo...'):

        try:

            response = requests.post(os.getenv("API_URL", "http://127.0.0.1:8000/predict"), json = data)
            
            if response.status_code == 200:

                res = response.json()
                
                st.subheader("Resultados del Análisis")
                
                # Estado principal con colores (Rojo para No Paga, Verde para Paga)

                if res['prediccion_clase'] == 0: 

                    st.error(res['estado'])

                else:

                    st.success(res['estado'])
                
                # Mostrar las probabilidades extraídas de la API

                c1, c2 = st.columns(2)

                c1.metric(label = "🔴 Probabilidad de Impago (Riesgo)", value = res['probabilidad_de_impago'])
                c2.metric(label = "🟢 Probabilidad de Pago Seguro", value = res['probabilidad_de_pago'])

            else:

                st.error(f"Error en la API: {response.text}")
                
        except requests.exceptions.ConnectionError:

            st.error("No se pudo conectar a la API.")
