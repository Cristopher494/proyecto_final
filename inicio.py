import streamlit as st
from PIL import Image
import requests
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_lottie import st_lottie
from utils import load_ben_color
import time
from streamlit_extras.switch_page_button import switch_page

# Cargar el modelo
modelo = load_model('./project/models/densenet121_preprocess-retin.keras')

def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# URL de la animación Lottie
lottie_coding = load_lottie_url("https://lottie.host/132a64cd-b421-46f4-99ec-8387c1ced1af/4L0M7FMGWD.json")

# Página principal
def main_page():
    st.title("Detección de Retinopatía Diabética")
    st_lottie(lottie_coding, height=300, key='coding')
    st.write('<h4 style="font-size: 24px;">Desde el Centro de Oftalmología Geeks ponemos a disposición el Sistema de Diagnóstico Automático de Rinopatía</h4>', unsafe_allow_html=True)
    if st.button("Realizar diagnóstico inteligente"):
        switch_page("Diagnóstico Inteligente")
    
    st.write("---")

    st.subheader("¿Qué es la retinopatía diabética?")
    st.image("./pages/Animation/Untitled_Project_V2.gif", use_column_width=True)
    st.write("La retina es la capa del fondo del ojo que recibe los estímulos luminosos y percibe las imágenes que serán enviadas a nuestro cerebro")
    st.write("Al igual que en el resto de complicaciones crónicas de la diabetes mellitus, **la presencia de niveles elevados de glucosa en sangre durante muchos años, produce alteraciones en los vasos sanguíneos de la retina que originan daño en este importante tejido ocular.**")
    st.write("Dichas alteraciones vasculares dificultan el aporte de oxígeno a la retina, en la cual se producen microaneurismas, hemorragias, así como fuga de lípidos y proteínas desde el interior de los capilares dañados. Esto ocurre fundamentalmente cuando la diabetes no se controla correctamente y recibe el nombre de retinopatía diabética.")
    st.write("Estos cambios son progresivos y la falta de oxigenación de la retina estimula el crecimiento de más vasos que intentan suplir el déficit de aporte sanguíneo. La aparición de estos nuevos vasos (fase denominada retinopatía diabética proliferante) puede producir daños irreversibles en la retina")

# Sistema de navegación
def main():
    main_page()

if __name__ == "__main__":
    main()