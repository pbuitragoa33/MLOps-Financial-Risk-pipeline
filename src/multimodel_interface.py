# Interfaz del Modelo --> Parte Gráfica (streamlit run multimodel_interface.py)
# Interfaz (UI) - Frontend --> con Streamlit


# Librerías

import streamlit as st
import os
import requests



# ----------------------------
# Configuración de la página
# ----------------------------

st.set_page_config(page_title = "Evaluador Multi-Modelo", layout = "wide")

st.title("Evaluación de Créditos por 4 modelos")
st.markdown("Ingrese los datos. El sistema consultará a 4 modelos simultáneamente")
st.divider()

# ----------------------------
# Variables y Rangos 
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
# Evaluación y Despliegue Multi-Modelo
# -----------------------------------------------------------

_, col_btn, _ = st.columns([1, 1, 1])

with col_btn:
    evaluar = st.button("Iniciar Análisis Multi-Modelo", use_container_width = True)

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

    with st.spinner('Los 4 modelos están evaluando el perfil...'):

        try:

            response = requests.post(os.getenv("API_URL", "http://127.0.0.1:8000/predict"), json = data)
            
            if response.status_code == 200:

                resultados = response.json()
                nombres_modelos = list(resultados.keys())
                
                # Consenso: Promedio del Riesgo

                riesgo_promedio = sum(mod['prob_impago_raw'] for mod in resultados.values()) / len(resultados)
                
                st.markdown(f"### Consenso del Sistema: Riesgo de Impago del **{riesgo_promedio * 100:.1f}%**")
                
                if riesgo_promedio > 0.50:

                    st.error("Veredicto Global: ALTO RIESGO - Se sugiere RECHAZAR")

                else:

                    st.success("Veredicto Global: RIESGO ACEPTABLE - Se sugiere APROBAR")
                
                st.divider()
                st.subheader("Desglose por Modelo Experto")
                
                # Crear cuadrícula 2x2 para mostrar los 4 modelos

                col_m1, col_m2 = st.columns(2)
                
                for i, nombre in enumerate(nombres_modelos):

                    columna_actual = col_m1 if i % 2 == 0 else col_m2
                    
                    with columna_actual:

                        with st.container(border=True):

                            st.markdown(f"#### {nombre}")
                            datos_mod = resultados[nombre]
                            
                            if datos_mod['prediccion_clase'] == 0:

                                st.error(datos_mod['estado'])

                            else:

                                st.success(datos_mod['estado'])
                            
                            c1, c2 = st.columns(2)

                            c1.metric(label = "🔴 Riesgo", value = datos_mod['probabilidad_de_impago'])
                            c2.metric(label = "🟢 Pago", value = datos_mod['probabilidad_de_pago'])

            else:

                st.error(f"Error en la API: {response.text}")
                
        except requests.exceptions.ConnectionError:

            st.error("No se pudo conectar a la API")
