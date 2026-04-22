"""
Detector de Fracturas Óseas
Sistema de análisis radiológico con red neuronal
"""

# =========================
# IMPORTACIONES
# =========================
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import json
import cv2
from datetime import datetime
import matplotlib.pyplot as plt

# =========================
# CONFIGURACIÓN GENERAL
# =========================
st.set_page_config(
    page_title="Detector de Fracturas",
    layout="wide"
)

# =========================
# RUTAS (IMPORTANTE PARA HUGGING FACE)
# =========================
MODEL_PATH = "modelo_fracturas.h5"
HISTORY_FILE = "historial.json"
HISTORY_DIR = "historial_imagenes"

IMG_SIZE = (128, 128)

# Crear carpeta si no existe
os.makedirs(HISTORY_DIR, exist_ok=True)

# =========================
# ESTADOS DE SESIÓN
# =========================
if 'quick_mode' not in st.session_state:
    st.session_state.quick_mode = False

# =========================
# CARGA DEL MODELO
# =========================
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False  # 🔥 evita error de deserialización
        )
        return model
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        return None

# =========================
# PREPROCESAMIENTO
# =========================
def preprocess_image(img):
    img = img.convert('L').resize(IMG_SIZE)
    return np.array(img) / 255.0

# =========================
# PREDICCIÓN
# =========================
def predict(model, img):
    img_array = preprocess_image(img)
    img_input = img_array.reshape(1, 128, 128, 1)

    prob = model.predict(img_input, verbose=0)[0][0]
    prob = 1 - prob

    if prob > 0.5:
        return "Fractura detectada", prob * 100
    else:
        return "Sin fractura", (1 - prob) * 100

# =========================
# INTERFAZ PRINCIPAL
# =========================
def main():
    st.title("Detector de Fracturas Óseas")

    # Cargar modelo
    model = load_model()

    if model is None:
        st.stop()

    # Subida de imágenes
    uploaded_files = st.file_uploader(
        "Sube radiografías",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            img = Image.open(file)

            pred, prob = predict(model, img)

            st.subheader(file.name)
            st.image(img, use_container_width=True)

            st.write(f"Resultado: {pred}")
            st.write(f"Probabilidad: {prob:.2f}%")

# =========================
# EJECUCIÓN
# =========================
if __name__ == "__main__":
    main()