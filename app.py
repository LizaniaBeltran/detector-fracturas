"""
Detector de Fracturas Oseas - Aplicacion Streamlit
"""

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import json
from datetime import datetime
import cv2
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Detector de Fracturas Oseas",
    layout="wide"
)

MODEL_PATH = "modelo_fracturas.h5"
HISTORY_FILE = "historial.json"
HISTORY_DIR = "historial_imagenes"
IMG_SIZE = (128, 128)

os.makedirs(HISTORY_DIR, exist_ok=True)

if 'patient_name' not in st.session_state:
    st.session_state.patient_name = ""
if 'patient_age' not in st.session_state:
    st.session_state.patient_age = ""
if 'patient_notes' not in st.session_state:
    st.session_state.patient_notes = ""

st.markdown("""
<style>
    .stApp { background-color: #f0f0f0; }
    h1 { color: #1e40af; }
    .sidebar { background-color: #e0e0e0; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        model_file = hf_hub_download(
            repo_id="lizzybel22/detector-fracturas",
            filename="modelo_fracturas.h5"
        )
        return tf.keras.models.load_model(model_file)
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None


def check_image_quality(img):
    img_gray = img.convert('L').resize((256, 256))
    img_np = np.array(img_gray)
    brightness = np.mean(img_np)
    if brightness < 30 or brightness > 225:
        return "Baja calidad - Brillo inadecuado", False
    contrast = np.std(img_np)
    if contrast < 20:
        return "Baja calidad - Poco contraste", False
    laplacian = cv2.Laplacian(img_np, cv2.CV_64F)
    sharpness = laplacian.var()
    if sharpness < 100:
        return "Imagen borrosa", False
    return "Calidad aceptable", True


def preprocess_image(img):
    img = img.convert('L').resize(IMG_SIZE)
    return np.array(img) / 255.0


def generate_attention_heatmap(model, img_array):
    img_input = img_array.reshape(1, 128, 128, 1)
    layer_outputs = [layer.output for layer in model.layers[0:4]]
    conv_model = tf.keras.Model(inputs=model.inputs[0], outputs=layer_outputs[-1])
    conv_output = conv_model.predict(img_input, verbose=0)
    if len(conv_output.shape) == 4:
        conv_output = conv_output[0]
    heatmap = np.mean(conv_output, axis=-1) if len(conv_output.shape) == 3 else conv_output
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = cv2.resize(heatmap, (256, 256))
    heatmap = np.uint8(heatmap * 255)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap


def create_marked_image(original_img, heatmap):
    orig_np = np.array(original_img.convert('L').resize((256, 256)))
    heatmap_np = np.array(heatmap)
    heatmap_gray = cv2.cvtColor(heatmap_np, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(heatmap_gray, 128, 255, cv2.THRESH_BINARY)
    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)
    original_rgb = np.stack([orig_np] * 3, axis=-1)
    overlay = cv2.addWeighted(original_rgb, 0.6, mask_colored, 0.4, 0)
    return Image.fromarray(overlay)


def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []


def save_history(history_data):
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history_data, f, indent=2)
    except:
        pass


def process_image(model, image_file):
    img = Image.open(image_file)
    quality_msg, good_quality = check_image_quality(img)
    if not good_quality:
        return img, img.copy(), quality_msg, 0.0, quality_msg, False
    img_array = preprocess_image(img)
    img_input = img_array.reshape(1, 128, 128, 1)
    prob = model.predict(img_input, verbose=0)[0][0]
    prob = 1 - prob
    if prob > 0.5:
        prediction = "Fractura detectada"
        probability = prob * 100
    else:
        prediction = "Sin fractura"
        probability = (1 - prob) * 100
    heatmap = generate_attention_heatmap(model, img_array)
    marked = create_marked_image(img, heatmap)
    return img, marked, prediction, probability, "", True


def main():
    with st.sidebar:
        st.title("Menu")
        menu = st.radio("", ["Analisis", "Comparar", "Historial", "Acerca de"])
        st.write("---")
        st.caption("Detector de Fracturas Oseas v1.0")

    st.title("Detector de Fracturas Oseas")
    st.info("Sistema de Analisis Radiologico")

    model = load_model()
    if model is None:
        st.error("No se pudo cargar el modelo")
        return

    if menu == "Analisis":
        st.subheader("Precision del modelo: ~89%")
        st.caption("Se recomienda verificar resultados con un profesional de la salud.")

        st.subheader("Datos del Paciente")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.patient_name = st.text_input("Nombre", st.session_state.patient_name)
        with col2:
            st.session_state.patient_age = st.text_input("Edad", st.session_state.patient_age)

        st.session_state.patient_notes = st.text_area("Notas", st.session_state.patient_notes)

        st.subheader("Cargar Radiografias")
        uploaded = st.file_uploader("Seleccione imagenes (JPG, PNG)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

        if uploaded:
            results = []
            progress_bar = st.progress(0)

            for i, upl in enumerate(uploaded):
                img, marked, pred, prob, exp, valid = process_image(model, upl)
                results.append({
                    'filename': upl.name,
                    'prediction': pred,
                    'probability': prob,
                    'original': img,
                    'marked': marked,
                    'valid': valid
                })
                progress_bar.progress((i+1)/len(uploaded))

            total = len(results)
            fracturadas = sum(1 for r in results if "fractura" in r['prediction'].lower() and "sin" not in r['prediction'].lower())
            normales = total - fracturadas

            st.subheader("Resumen")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total", total)
            c2.metric("Con Fractura", fracturadas)
            c3.metric("Normales", normales)

            st.subheader("Resultados")
            for r in results:
                has_f = "fractura" in r['prediction'].lower() and "sin" not in r['prediction'].lower()
                color = "red" if has_f else "green"
                risk = "Alto" if r['probability'] >= 70 else ("Moderado" if r['probability'] >= 40 else "Bajo")
                risk_color = "red" if r['probability'] >= 70 else ("orange" if r['probability'] >= 40 else "green")

                st.markdown(f"**{r['filename']}**")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.image(r['original'], caption="Original", use_container_width=True)
                with c2:
                    st.image(r['marked'], caption="Zona de Atencion", use_container_width=True)
                with c3:
                    st.markdown(f":{color}[**{r['prediction']}**]")
                    st.markdown(f":{risk_color}[**{r['probability']:.1f}%**]")
                    st.caption(f"Riesgo: {risk}")

                if has_f:
                    st.warning("Se recomienda evaluacion clinica")
                st.write("---")

            if st.button("Guardar Resultados"):
                hist = load_history()
                sid = str(datetime.now().timestamp())
                for r in results:
                    r['original'].save(os.path.join(HISTORY_DIR, f"{sid}_{r['filename']}"))
                    hist.append({
                        'id': sid,
                        'filename': r['filename'],
                        'prediction': r['prediction'],
                        'probability': float(r['probability']),
                        'timestamp': datetime.now().isoformat(),
                        'patient_name': st.session_state.patient_name,
                        'patient_age': st.session_state.patient_age,
                        'notes': st.session_state.patient_notes
                    })
                save_history(hist)
                st.success("Resultados guardados")

    elif menu == "Comparar":
        st.subheader("Comparar 2 Radiografias")
        c1, c2 = st.columns(2)
        with c1:
            img1 = st.file_uploader("Imagen 1", type=['jpg', 'jpeg', 'png'])
        with c2:
            img2 = st.file_uploader("Imagen 2", type=['jpg', 'jpeg', 'png'])

        if img1 and img2:
            p1 = process_image(model, img1)
            p2 = process_image(model, img2)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(p1[0], caption=p1[2], use_container_width=True)
            with c2:
                st.image(p2[0], caption=p2[2], use_container_width=True)
            with c3:
                diff = abs(p1[3] - p2[3])
                st.metric("Diferencia", f"{diff:.1f}%")

    elif menu == "Historial":
        st.subheader("Historial de Estudios")
        hist = load_history()
        total_studies = len(set([h['id'] for h in hist])) if hist else 0
        st.metric("Estudios realizados", total_studies)

        if not hist:
            st.info("No hay estudios guardados.")
        else:
            grup = {}
            for h in hist:
                if h['id'] not in grup:
                    grup[h['id']] = []
                grup[h['id']].append(h)

            for sid in sorted(grup.keys(), reverse=True)[:15]:
                items = grup[sid]
                total = len(items)
                fracs = sum(1 for i in items if "fractura" in i['prediction'].lower() and "sin" not in i['prediction'].lower())
                fecha = items[0].get('timestamp', '')[:10] if items else ""
                paciente = items[0].get('patient_name', '')
                edad = items[0].get('patient_age', '')

                label = f"Estudio {fecha} - {total} imagenes ({fracs} fracturas)"
                if paciente:
                    label += f" - {paciente}"
                if edad:
                    label += f" ({edad} anos)"

                with st.expander(label):
                    for item in items:
                        color = "red" if "fractura" in item['prediction'].lower() and "sin" not in item['prediction'].lower() else "green"
                        st.markdown(f":{color}[**{item['prediction']}**] **{item['filename']}** - {item['probability']:.1f}%")

    elif menu == "Acerca de":
        st.subheader("Detector de Fracturas Oseas")
        st.write("Sistema de soporte de diagnostico basado en redes neuronales convolucionales.")

        st.subheader("Caracteristicas")
        st.write("- Analisis automatico de radiografias")
        st.write("- Porcentaje de probabilidad de fractura")
        st.write("- Resaltado de zonas de atencion (heatmap)")
        st.write("- Datos del paciente")
        st.write("- Historial de estudios")

        st.warning("AVISO LEGAL: Esta herramienta es solo un apoyo diagnostico. Los resultados deben ser verificados por un profesional de la salud. No sustituye el diagnostico medico profesional.")


if __name__ == "__main__":
    main()