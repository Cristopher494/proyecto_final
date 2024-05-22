import streamlit as st
from PIL import Image
import requests
import numpy as np
from keras.models import load_model
from streamlit_lottie import st_lottie
import time
from streamlit_extras.switch_page_button import switch_page
from utils import load_image, load_ben_color

# Cargar el modelo
modelo = load_model('./project/models/densenet121_model.h5')

# Función para cargar la animación Lottie
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# URL de la animación Lottie
lottie_coding = load_lottie_url("https://lottie.host/132a64cd-b421-46f4-99ec-8387c1ced1af/4L0M7FMGWD.json")

# Página de diagnóstico
def diagnosis_page():
    # Botón para volver a la página principal
    if st.button("Volver al inicio"):
        switch_page("Inicio")
    
    st.title("Diagnóstico Inteligente")

    # Sección de datos del usuario
    st.subheader("Por favor, ingrese sus datos:")
    
    gender = st.selectbox("Género", ["Femenino", "Masculino"])
    age = st.text_input("Edad", max_chars=2)
    diabetes_diagnosis = st.selectbox("Diagnóstico de Diabetes", ["Sí", "No"])
    
    st.write("---")

    st.subheader("¿Qué imagen tengo que cargar?")
    st.write("Para que el sistema devuelva una predicción acertada, es necesario cargar una imagen obtenida a partir de una ***Retinografía o Imagen de Fondo de Ojo***")

    st.subheader("¿Qué resultados obtendré?")
    st.write('<ul><li><b>No se detecta retinopatía diabética</b>: No se han podido identificar signos de rinopatía</li>', unsafe_allow_html=True)
    st.write('<ul><li><b>Se sugiere evaluación profesional</b>Detección moderada de signos de Rinopatía, se sugiere profundizar el diagnóstico con un profesional</li>', unsafe_allow_html=True)
    st.write('<ul><li><b>Retinopatía diabética detectada</b>: Evidente detección de signos de Rinopatía: se sugiere realizar consulta profesional cuanto antes.</li>', unsafe_allow_html=True)
    
    # Carga de imagen
    image_file = st.file_uploader("Carga una imagen", type=["jpg", "jpeg", "png"])
        
    if image_file is not None:
        # Mostrar la imagen cargada
        img = Image.open(image_file)
        
        # Botón para activar la predicción
        if st.button("Realizar Predicción"):
            # Crear un marcador de posición para la animación
            placeholder = st.empty()
           
            with st.spinner('Procesando la imagen...'):
                # Mostrar la animación de procesamiento
                with placeholder:
                    st_lottie(lottie_coding, height=100, key='processing')
                    time.sleep(8)  # Pausar por 2 segundos (ajustar según sea necesario)

                # Preprocesar la imagen
                img_array = np.array(img)
                processed_image = load_ben_color(img_array)
                processed_image = np.expand_dims(processed_image, axis=0)
                       
                # Realizar la predicción con el modelo
                prediction = modelo.predict(processed_image)
                probability = prediction[0][0]

                # Ocultar la animación después del procesamiento
                placeholder.empty()

                # Mostrar el resultado de la predicción
                st.write("Resultado de la predicción:")
                st.write(probability)

                if probability < 0.2:
                    st.info("No se detecta retinopatía diabética" )
                elif 0.2 <= probability < 0.4:
                    st.warning("Se sugiere evaluación profesional")
                else:
                    st.error("Retinopatía diabética detectada")   
        
        st.image(img, caption="Imagen cargada", use_column_width=True)

    st.write("---")

# Ejecutar la página de diagnóstico
if __name__ == "__main__":
    diagnosis_page()
