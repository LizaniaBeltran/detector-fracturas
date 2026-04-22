import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import json
from datetime import datetime
import cv2
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Detector de Fracturas", layout="wide")

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


@st.cache_resource
def load_model():
    try:
        model_file = hf_hub_download(repo_id="lizzybel22/detector-fracturas", filename="modelo_fracturas.h5")
        return tf.keras.models.load_model(model_file)
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def check_quality(img):
    img_g = img.convert('L').resize((256, 256))
    img_np = np.array(img_g)
    bright = np.mean(img_np)
    if bright < 30 or bright > 225:
        return "Brillo bajo", False
    contrast = np.std(img_np)
    if contrast < 20:
        return "Poco contraste", False
    lap = cv2.Laplacian(img_np, cv2.CV_64F)
    if lap.var() < 100:
        return "Borroso", False
    return "OK", True


def preprocess(img):
    return np.array(img.convert('L').resize(IMG_SIZE)) / 255.0


def get_heatmap(model, img_arr):
    inp = img_arr.reshape(1, 128, 128, 1)
    layer_out = [l.output for l in model.layers[0:4]]
    conv_m = tf.keras.Model(inputs=model.inputs[0], outputs=layer_out[-1])
    out = conv_m.predict(inp, verbose=0)
    if len(out.shape) == 4:
        out = out[0]
    hm = np.mean(out, axis=-1) if len(out.shape) == 3 else out
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    hm = cv2.resize(hm, (256, 256))
    hm = np.uint8(hm * 255)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    return hm


def make_overlay(orig, hm):
    orig_np = np.array(orig.convert('L').resize((256, 256)))
    hm_np = np.array(hm)
    hm_gray = cv2.cvtColor(hm_np, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(hm_gray, 128, 255, cv2.THRESH_BINARY)
    mask_c = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    mask_c = cv2.cvtColor(mask_c, cv2.COLOR_BGR2RGB)
    orig_rgb = np.stack([orig_np] * 3, axis=-1)
    return Image.fromarray(cv2.addWeighted(orig_rgb, 0.6, mask_c, 0.4, 0))


def load_hist():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except:
            return []
    return []


def save_hist(data):
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except:
        pass


def process_img(model, f):
    img = Image.open(f)
    msg, ok = check_quality(img)
    if not ok:
        return img, img.copy(), msg, 0.0, msg, False
    arr = preprocess(img)
    inp = arr.reshape(1, 128, 128, 1)
    prob = model.predict(inp, verbose=0)[0][0]
    prob = 1 - prob
    if prob > 0.5:
        pred = "Fractura"
        prob_p = prob * 100
    else:
        pred = "Sin fractura"
        prob_p = (1 - prob) * 100
    hm = get_heatmap(model, arr)
    marked = make_overlay(img, hm)
    return img, marked, pred, prob_p, "", True


def main():
    st.sidebar.title("Menu")
    menu = st.sidebar.radio("", ["Analisis", "Comparar", "Historial", "Acerca de"])

    st.title("Detector de Fracturas Oseas")

    model = load_model()
    if model is None:
        return

    if menu == "Analisis":
        st.info("Precision: ~89%")

        col1, col2 = st.columns(2)
        st.session_state.patient_name = col1.text_input("Nombre", st.session_state.patient_name)
        st.session_state.patient_age = col2.text_input("Edad", st.session_state.patient_age)
        st.session_state.patient_notes = st.text_area("Notas", st.session_state.patient_notes)

        uploaded = st.file_uploader("Cargar radiografias", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

        if uploaded:
            results = []
            bar = st.progress(0)
            for i, upl in enumerate(uploaded):
                img, marked, pred, prob, exp, valid = process_img(model, upl)
                results.append({'filename': upl.name, 'prediction': pred, 'probability': prob, 'original': img, 'marked': marked})
                bar.progress((i+1)/len(uploaded))

            total = len(results)
            frac = sum(1 for r in results if r['prediction'] == "Fractura")

            st.subheader("Resumen")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total", total)
            c2.metric("Fractura", frac)
            c3.metric("Normal", total - frac)

            st.subheader("Resultados")
            for r in results:
                color = "red" if r['prediction'] == "Fractura" else "green"
                st.write(f"**{r['filename']}**")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.image(r['original'], caption="Original")
                with c2:
                    st.image(r['marked'], caption="Zona")
                with c3:
                    st.markdown(f":{color}[**{r['prediction']}**]")
                    st.write(f"{r['probability']:.1f}%")

            if st.button("Guardar"):
                hist = load_hist()
                sid = str(datetime.now().timestamp())
                for r in results:
                    r['original'].save(os.path.join(HISTORY_DIR, f"{sid}_{r['filename']}"))
                    hist.append({'id': sid, 'filename': r['filename'], 'prediction': r['prediction'], 'probability': float(r['probability']), 'timestamp': datetime.now().isoformat(), 'patient_name': st.session_state.patient_name, 'patient_age': st.session_state.patient_age})
                save_hist(hist)
                st.success("Guardado")

    elif menu == "Comparar":
        c1, c2 = st.columns(2)
        img1 = c1.file_uploader("Img1", type=['jpg', 'jpeg', 'png'])
        img2 = c2.file_uploader("Img2", type=['jpg', 'jpeg', 'png'])
        if img1 and img2:
            p1 = process_img(model, img1)
            p2 = process_img(model, img2)
            c1, c2, c3 = st.columns(3)
            c1.image(p1[0])
            c2.image(p2[0])
            c3.metric("Diferencia", f"{abs(p1[3]-p2[3]):.1f}%")

    elif menu == "Historial":
        hist = load_hist()
        st.metric("Estudios", len(set(h['id'] for h in hist)) if hist else 0)
        if hist:
            for sid in sorted(set(h['id'] for h in hist), reverse=True)[:10]:
                items = [h for h in hist if h['id'] == sid]
                fecha = items[0]['timestamp'][:10]
                pac = items[0].get('patient_name', '')
                st.expander(f"{fecha} - {len(items)} img" + (f" - {pac}" if pac else "")).write(", ".join(f"{i['prediction']} ({i['probability']:.0f}%)" for i in items))

    elif menu == "Acerca de":
        st.write("Sistema de apoyo diagnostico basado en CNN.")
        st.warning("Solo apoyo. Verifique con profesional.")
        st.write("v1.0")


if __name__ == "__main__":
    main()