"""
Detector de Fracturas Oseas - Aplicacion Streamlit
Sistema de analisis radiologico para deteccion de fracturas
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

# Configuracion de la pagina de Streamlit
st.set_page_config(
    page_title="Detector de Fracturas Oseas",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Rutas de archivos y configuraciones
# Aqui definimos donde esta el modelo entrenado y donde se guardara el historial
MODEL_PATH = r"C:\Users\lizan\Downloads\Fracturas\modelo_fracturas.h5"
HISTORY_FILE = r"C:\Users\lizan\Downloads\Fracturas\historial.json"
HISTORY_DIR = r"C:\Users\lizan\Downloads\Fracturas\historial_imagenes"
IMG_SIZE = (128, 128)

# Crear directorio para guardar imagenes del historial si no existe
os.makedirs(HISTORY_DIR, exist_ok=True)

# Variables de sesion para mantener datos entre recargas
# Estas variables guardan el estado del paciente durante la sesion de la app
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'patient_name' not in st.session_state:
    st.session_state.patient_name = ""
if 'patient_age' not in st.session_state:
    st.session_state.patient_age = ""
if 'patient_notes' not in st.session_state:
    st.session_state.patient_notes = ""
if 'analisis_count' not in st.session_state:
    st.session_state.analisis_count = 0

# Estilos CSS para la interfaz
# Aqui definimos el aspecto visual de la aplicacion (colores, fuentes, tamaños)
st.markdown("""
<style>
    * { font-family: 'Segoe UI', Arial, sans-serif; }
    .stApp { background: #e5e5e5; }
    .main-header { background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); padding: 25px 30px; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.15); }
    .main-header h1 { color: #ffffff; font-size: 28px; font-weight: 700; margin: 0; }
    .main-header p { color: #bfdbfe; font-size: 14px; margin: 8px 0 0 0; }
    .menu-btn { display: block; width: 100%; padding: 14px 18px; margin: 6px 0; background: #eff6ff; border: 2px solid #bfdbfe; border-radius: 8px; color: #1e40af; font-size: 15px; font-weight: 600; text-align: left; cursor: pointer; }
    .menu-btn:hover { background: #3b82f6; color: #ffffff; }
    .card { background: #ffffff; border: 2px solid #e5e7eb; border-radius: 10px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    .card h3 { color: #1e40af; font-size: 18px; font-weight: 700; margin: 0 0 15px 0; }
    .card h4 { color: #1e40af; font-size: 15px; font-weight: 700; margin: 0 0 10px 0; }
    label { color: #000000 !important; }
    .metric-card { background: #ffffff; border: 2px solid #e5e7eb; border-radius: 10px; padding: 18px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    .metric-card h3 { color: #1e40af; font-size: 32px; font-weight: 700; margin: 0; }
    .metric-card small { color: #6b7280; font-size: 12px; text-transform: uppercase; }
    .metric-card.red h3 { color: #dc2626; }
    .metric-card.green h3 { color: #16a34a; }
    .badge { display: inline-block; padding: 8px 14px; border-radius: 6px; font-size: 13px; font-weight: 700; }
    .badge-green { background: #dcfce7; color: #166534; }
    .badge-red { background: #fee2e2; color: #991b1b; }
    .badge-yellow { background: #fef3c7; color: #92400e; }
    .img-label { color: #1e40af; font-size: 13px; font-weight: 700; text-transform: uppercase; text-align: center; margin-bottom: 8px; }
    .warning-box { background: #fef3c7; border: 2px solid #facc15; border-radius: 8px; padding: 15px 20px; margin: 15px 0; }
    .warning-box p { color: #92400e; font-size: 14px; font-weight: 600; margin: 0; }
    .legal-box { background: #fef2f2; border: 2px solid #fecaca; border-radius: 8px; padding: 20px; margin: 20px 0; }
    .legal-box h4 { color: #991b1b; font-size: 16px; font-weight: 700; margin: 0 0 10px 0; }
    .legal-box p { color: #7f1d1d; font-size: 13px; line-height: 1.6; }
    .features-list { background: #eff6ff; border-radius: 10px; padding: 20px 25px; margin: 15px 0; }
    .features-list h4 { color: #1e40af; font-size: 16px; font-weight: 700; margin: 0 0 15px 0; }
    .features-list ul { margin: 0; padding-left: 20px; }
    .features-list li { color: #1e3a8a; font-size: 14px; margin: 8px 0; }
    .sidebar-title { color: #ffffff; font-size: 20px; font-weight: 700; margin-bottom: 8px; }
    .sidebar-subtitle { color: #bfdbfe; font-size: 13px; }
    .confidence-high { color: #dc2626; }
    .confidence-medium { color: #d97706; }
    .confidence-low { color: #16a34a; }
    .accuracy-info { background: #dbeafe; border: 2px solid #93c5fd; border-radius: 8px; padding: 15px; margin: 10px 0; }
    .accuracy-info h4 { color: #1e40af; font-size: 14px; font-weight: 700; }
    .accuracy-info p { color: #1e3a8a; font-size: 13px; }
</style>
""", unsafe_allow_html=True)


# ============================================
# FUNCIONES DEL SISTEMA
# ============================================


@st.cache_resource
def load_model():
    """
    Carga el modelo de red neuronal desde Hugging Face
    Si no existe localmente, lo descarga automaticamente
    """
    model_file = hf_hub_download(
        repo_id="lizzybel22/detector-fracturas",
        filename="modelo_fracturas.h5"
    )
    return tf.keras.models.load_model(model_file)


def check_image_quality(img):
    """
    Verifica la calidad de la radiografia cargada
    Analiza tres parametros: brillo, contraste y nitidez
    Returns: (mensaje explicativo, True/False si la calidad es aceptable)
    """
    # Convertir a escala de grises y redimensionar a 256x256 para analisis
    img_gray = img.convert('L').resize((256, 256))
    img_np = np.array(img_gray)
    
    # 1. Verificar brillo (promedio de pixeles)
    # Si esta muy oscuro (<30) o muy brillante (>225), la imagen no sirve
    brightness = np.mean(img_np)
    if brightness < 30 or brightness > 225:
        return "Baja calidad - Brillo inadecuado", False
    
    # 2. Verificar contraste (desviacion estandar de los pixeles)
    # Si el contraste es muy bajo (<20), la imagen es muy plana
    contrast = np.std(img_np)
    if contrast < 20:
        return "Baja calidad - Poco contraste", False
    
    # 3. Verificar nitidez usando el operador Laplaciano
    # Calcula la varianza - si es muy baja, la imagen esta borrosa
    laplacian = cv2.Laplacian(img_np, cv2.CV_64F)
    sharpness = laplacian.var()
    if sharpness < 100:
        return "Imagen borrosa", False
    
    # Si paso todas las pruebas, la calidad es aceptable
    return "Calidad aceptable", True


def get_risk_level(prob):
    """
    Determina el nivel de riesgo basado en la probabilidad de fractura
    Returns: (texto del riesgo, clase CSS para el badge de color)
    """
    if prob < 40:
        return "Bajo riesgo", "badge-green"
    elif prob < 70:
        return "Riesgo moderado", "badge-yellow"
    else:
        return "Alto riesgo", "badge-red"


def get_confidence_class(prob):
    """
    Retorna la clase CSS para el color del porcentaje de confianza
    Alto = rojo, Medio = naranja, Bajo = verde
    """
    if prob >= 70:
        return "confidence-high"
    elif prob >= 40:
        return "confidence-medium"
    else:
        return "confidence-low"


def get_clinical_recommendation(prediction, prob):
    """
    Genera una recomendacion clinica basada en el resultado del analisis
    Incluye sugerencias de seguimiento segun el nivel de probabilidad
    """
    if prediction == "Sin fractura":
        if prob < 20:
            return "Resultado normal. Continuar con controles routine."
        elif prob < 50:
            return "Se sugiere revision en 7 dias si hay sintomas."
        else:
            return "Se recomienda evaluacion clinica para descartar fractura."
    else:
        if prob < 60:
            return "Control en 48-72 horas. Observar sintomas."
        elif prob < 80:
            return "Se recomienda evaluacion prioritaria en 24-48 horas."
        else:
            return "Se sugiere evaluacion urgente. Posible fractura significativa."


def preprocess_image(img):
    """
    Prepara la imagen para el modelo de red neuronal:
    - Convierte a escala de grises (blanco y negro)
    - Redimensiona al tamano esperado por el modelo (128x128 pixeles)
    - Normaliza valores de pixeles al rango 0-1 (divide entre 255)
    """
    img = img.convert('L').resize(IMG_SIZE)
    return np.array(img) / 255.0


def generate_attention_heatmap(model, img_array):
    """
    Genera un mapa de calor (heatmap) que muestra las zonas de la imagen
    que la red neuronal considera importantes para hacer la prediccion.
    Usa las capas convolucionales del modelo para detectar patrones.
    
    Returns: array numpy con el heatmap en color
    """
    # Prepara imagen para el modelo (agrega la dimension extra para batch)
    img_input = img_array.reshape(1, 128, 128, 1)
    
    # Obtiene  salidas de las primeras 4 capas convolucionales
    # Estas capas contienen los patrones que el modelo detecta
    layer_outputs = [layer.output for layer in model.layers[0:4]]
    
    # Crea un modelo temporal para obtener las activaciones de esas capas
    conv_model = tf.keras.Model(inputs=model.inputs[0], outputs=layer_outputs[-1])
    conv_output = conv_model.predict(img_input, verbose=0)
    
    # Procesa las activaciones para crear el heatmap
    # Reduce las dimensiones promediando los filtros
    if len(conv_output.shape) == 4:
        conv_output = conv_output[0]
    heatmap = np.mean(conv_output, axis=-1) if len(conv_output.shape) == 3 else conv_output
    
    # Normalizar valores al rango 0-1
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    # Redimensionar a 256x256 para mostrar sobre la imagen original
    heatmap = cv2.resize(heatmap, (256, 256))
    # Convertir a valores 0-255 (formato de imagen)
    heatmap = np.uint8(heatmap * 255)
    
    # Aplicar mapa de color "JET" que va de azul (frio) a rojo (caliente)
    # Azul = baja importancia, Rojo = alta importancia
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap


def create_marked_image(original_img, heatmap, threshold=0.5):
    """
    Superpone el heatmap sobre la imagen original para crear una visualizacion
    de las zonas que el modelo considera importantes.
    
    Returns: imagen PIL con el heatmap superpuesto
    """
    # Convertir imagen original a numpy array
    orig_np = np.array(original_img.convert('L').resize((256, 256)))
    heatmap_np = np.array(heatmap)
    
    # Convertir heatmap a escala de grises y aplicar umbral (threshold)
    # Esto crea una mascara binaria (blanco o negro)
    heatmap_gray = cv2.cvtColor(heatmap_np, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(heatmap_gray, int(255 * threshold), 255, cv2.THRESH_BINARY)
    
    # Colorear la mascara con el mapa de colores
    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)
    
    # Crear imagen RGB a partir de la imagen en escala de grises
    # Stack replica el canal 3 veces para tener RGB
    original_rgb = np.stack([orig_np] * 3, axis=-1)
    
    # Combinar imagen original (60%) con el mapa de calor (40%)
    # Esto crea el efecto de superposicion
    overlay = cv2.addWeighted(original_rgb, 0.6, mask_colored, 0.4, 0)
    return Image.fromarray(overlay)


def load_history():
    """
    Carga el historial de estudios desde el archivo JSON
    Returns: lista de analisis guardados (lista vacia si no existe)
    """
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []


def save_history(history_data):
    """
    Guarda el historial de estudios en archivo JSON
    Si hay error, simplemente no guarda (silencioso)
    """
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history_data, f, indent=2)
    except:
        pass


def process_image(model, image_file):
    """
    Funcion principal de procesamiento de una imagen:
    1. Verifica la calidad de la imagen
    2. Prepara la imagen para el modelo
    3. Obtiene la prediccion del modelo
    4. Genera el heatmap de atencion
    
    Returns: (imagen_original, imagen_marcada, prediccion, probabilidad, mensaje_error, es_valida)
    """
    img = Image.open(image_file)
    quality_msg, good_quality = check_image_quality(img)
    
    # Si la calidad no es aceptable, retornar con error
    if not good_quality:
        return img, img.copy(), quality_msg, 0.0, quality_msg, False
    
    # Preprocesar imagen (escala de grises, redimension, normalizacion)
    img_array = preprocess_image(img)
    img_input = img_array.reshape(1, 128, 128, 1)
    
    # Obtener prediccion del modelo
    # La red neuronal devuelve una probabilidad entre 0 y 1
    prob = model.predict(img_input, verbose=0)[0][0]
    
    # Invertir probabilidad porque el modelo esta entrenado al reves
    # (Esto es un fix para corregir el entrenamiento)
    prob = 1 - prob
    
    # Determinar prediccion y probabilidad segun el umbral de 0.5
    if prob > 0.5:
        prediction = "Fractura detectada"
        probability = prob * 100
    else:
        prediction = "Sin fractura"
        probability = (1 - prob) * 100
    
    # Generar heatmap de atencion (zonas importantes para la prediccion)
    heatmap = generate_attention_heatmap(model, img_array)
    marked = create_marked_image(img, heatmap)
    
    return img, marked, prediction, probability, "", True


def main():
    """
    Funcion principal de la aplicacion Streamlit
    Maneja la interfaz de usuario, navegacion y flujo de la app
    """
    
    # ============================================
    # BARRA LATERAL - MENU DE NAVEGACION
    # ============================================
    
    with st.sidebar:
        # Mostrar titulo y subtitulo del menu
        st.markdown("""<div class="sidebar-title">Menu Principal</div><div class="sidebar-subtitle">Seleccione una opcion</div>""", unsafe_allow_html=True)
        st.markdown("---")
        
        # Radio button para navegar entre las secciones
        menu = st.radio("Navegacion", ["Analisis", "Comparar", "Historial", "Acerca de"])
        st.markdown("---")
        
        # Informacion del sistema en la barra lateral
        st.markdown("""<div class="sidebar-subtitle">Informacion del Sistema</div>""", unsafe_allow_html=True)
        st.write("Version: 1.0")
        st.write("Detector de Fracturas Oseas")

    # Encabezado principal de la aplicacion (banner azul)
    st.markdown("""<div class="main-header"><h1>Detector de Fracturas Oseas</h1><p>Sistema de Analisis Radiologico - Soporte de Diagnostico</p></div>""", unsafe_allow_html=True)

    # Cargar el modelo de red neuronal al inicio
    # Si hay error, mostrar mensaje y salir
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return

    # ============================================
    # SECCION: ANALISIS DE RADIOGRAFIAS
    # ============================================
    
    if menu == "Analisis":
        # Mostrar informacion sobre la precision del modelo
        st.markdown("""<div class="accuracy-info"><h4>Precision del modelo</h4><p>El modelo entrenado tiene aproximadamente 89% de precision en datos de validacion. Se recomienda siempre verificar los resultados con un profesional de la salud.</p></div>""", unsafe_allow_html=True)
        
        # Formulario de datos del paciente (nombre, edad, notas)
        st.markdown("""<div class="card"><h3>Datos del Paciente</h3></div>""", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.patient_name = st.text_input("Nombre del paciente", st.session_state.patient_name)
        with col2:
            st.session_state.patient_age = st.text_input("Edad", st.session_state.patient_age)
        with col3:
            st.session_state.patient_notes = st.text_area("Notas / Observaciones", st.session_state.patient_notes)
        
        # Seccion para cargar imagenes
        st.markdown("""<div class="card"><h3>Cargar Radiografias</h3></div>""", unsafe_allow_html=True)
        uploaded = st.file_uploader("Seleccione imagenes (JPG, PNG)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
        
        # Si el usuario cargo imagenes, procesarlas
        if uploaded:
            results = []
            barra = st.progress(0)
            
            # Procesar cada imagen cargada una por una
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
                # Actualizar barra de progreso
                barra.progress((i+1)/len(uploaded))
            
            # Incrementar contador de analisis
            st.session_state.analisis_count += len(uploaded)
            
            # Calcular estadisticas del analisis (totales, fracturadas, normales)
            total = len(results)
            fracturadas = sum(1 for r in results if "fractura" in r['prediction'].lower() and "sin" not in r['prediction'].lower())
            normales = total - fracturadas
            
            # Mostrar resumen del analisis en tarjetas
            st.markdown("""<div class="card"><h3>Resumen del Analisis</h3></div>""", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""<div class="metric-card"><h3>{total}</h3><small>Total</small></div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="metric-card red"><h3>{fracturadas}</h3><small>Con Fractura</small></div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="metric-card green"><h3>{normales}</h3><small>Normales</small></div>""", unsafe_allow_html=True)
            
            # Mostrar resultados individuales de cada imagen
            st.markdown("""<div class="card"><h3>Resultados</h3></div>""", unsafe_allow_html=True)
            for r in results:
                # Determinar si tiene fractura para mostrar el color correcto
                has_f = "fractura" in r['prediction'].lower() and "sin" not in r['prediction'].lower()
                risk, rc = get_risk_level(r['probability'])
                conf_class = get_confidence_class(r['probability'])
                badge = "badge-red" if has_f else "badge-green"
                pc = "badge-red" if r['probability'] >= 70 else ("badge-yellow" if r['probability'] >= 40 else "badge-green")
                clinical = get_clinical_recommendation(r['prediction'], r['probability'])
                
                # Mostrar tarjeta con nombre del archivo
                st.markdown(f"""<div class="card"><h4>{r['filename']}</h4></div>""", unsafe_allow_html=True)
                
                # Mostrar 3 columnas: imagen original, heatmap, resultado
                c1, c2, c3 = st.columns([1,1,1])
                with c1:
                    st.markdown('<div class="img-label">Original</div>', unsafe_allow_html=True)
                    st.image(r['original'], use_container_width=True)
                with c2:
                    st.markdown('<div class="img-label">Zona de Atencion</div>', unsafe_allow_html=True)
                    st.image(r['marked'], use_container_width=True)
                with c3:
                    # Mostrar resultado con formato HTML
                    st.markdown(f"""<br><span class="{badge}">{r['prediction']}</span>
                    <br><br><span class="{conf_class}" style="font-size:24px;font-weight:700;">{r['probability']:.1f}%</span>
                    <p style="font-size:12px;color:#6b7280;">Probabilidad de fractura</p>
                    <span class="{pc}">{risk}</span>
                    <hr style="margin:10px 0;">
                    <p style="font-size:13px;color:#1e40af;font-weight:600;text-align:center;">Recomendacion clinica:</p>
                    <p style="font-size:12px;color:#374151;text-align:center;">{clinical}</p>""", unsafe_allow_html=True)
            
            # Boton para guardar resultados en el historial
            colb1, colb2 = st.columns(2)
            with colb1:
                if st.button("Guardar Resultados", use_container_width=True):
                    hist = load_history()
                    sid = str(datetime.now().timestamp())
                    # Guardar cada resultado
                    for r in results:
                        # Guardar imagen en el directorio de historial
                        r['original'].save(os.path.join(HISTORY_DIR, f"{sid}_{r['filename']}"))
                        # Guardar datos en JSON (historial)
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
            
            # Advertencia si se detectaron fracturas
            if fracturadas > 0:
                st.markdown("""<div class="warning-box"><p><strong>Recomendacion:</strong> Acuda a un profesional de la salud para una evaluacion mas precisa.</p></div>""", unsafe_allow_html=True)

    # ============================================
    # SECCION: COMPARAR 2 RADIOGRAFIAS
    # ============================================
    
    elif menu == "Comparar":
        st.markdown("""<div class="card"><h3>Comparar 2 Radiografias</h3></div>""", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            img1 = st.file_uploader("Imagen 1", type=['jpg', 'jpeg', 'png'])
        with c2:
            img2 = st.file_uploader("Imagen 2", type=['jpg', 'jpeg', 'png'])
        
        # Si ambas imagenes estan cargadas, procesarlas y compararlas
        if img1 and img2:
            p1 = process_image(model, img1)
            p2 = process_image(model, img2)
            
            # Mostrar las dos imagenes y la diferencia de probabilidad
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(p1[0], caption=p1[2], use_container_width=True)
            with c2:
                st.image(p2[0], caption=p2[2], use_container_width=True)
            with c3:
                diff = abs(p1[3] - p2[3])
                st.markdown(f"""<br><h4 style="color:#1e40af;">Diferencia</h4><h2 style="color:#1e40af;">{diff:.1f}%</h2>""", unsafe_allow_html=True)

    # ============================================
    # SECCION: HISTORIAL DE ESTUDIOS
    # ============================================
    
    elif menu == "Historial":
        st.markdown("""<div class="card"><h3>Historial de Estudios</h3></div>""", unsafe_allow_html=True)
        hist = load_history()
        # Contar estudios unicos (por ID)
        total_studies = len(set([h['id'] for h in hist])) if hist else 0
        
        st.metric("Estudiosrealizados", total_studies)
        
        if not hist:
            st.info("No hay estudios guardados.")
        else:
            # Agrupar resultados por estudio (usando el ID como clave)
            grup = {}
            for h in hist:
                if h['id'] not in grup:
                    grup[h['id']] = []
                grup[h['id']].append(h)
            
            # Mostrar estudios recientes (ultimos 15)
            for sid in sorted(grup.keys(), reverse=True)[:15]:
                items = grup[sid]
                total = len(items)
                fracs = sum(1 for i in items if "fractura" in i['prediction'].lower() and "sin" not in i['prediction'].lower())
                fecha = items[0].get('timestamp', '')[:10] if items else ""
                paciente = items[0].get('patient_name', '') if items else ""
                edad = items[0].get('patient_age', '') if items else ""
                # Formatear info del paciente
                paciente_info = f"{paciente}" + (f" ({edad} anos)" if edad else "") if paciente else (f"{edad} anos" if edad else "")
                
                # Mostrar cada estudio en un expander colapsable
                with st.expander(f"Estudio {fecha} - {total} imagenes ({fracs} fracturas)" + (f" - {paciente_info}" if paciente_info else "")):
                    for item in items:
                        badge = "badge-red" if "fractura" in item['prediction'].lower() and "sin" not in item['prediction'].lower() else "badge-green"
                        st.markdown(f"""<span class="{badge}">{item['prediction']}</span> <strong>{item['filename']}</strong> - {item['probability']:.1f}%""", unsafe_allow_html=True)

    # ============================================
    # SECCION: ACERCA DE / INFO DEL SISTEMA
    # ============================================
    
    elif menu == "Acerca de":
        # Descripcion del sistema
        st.markdown("""<div class="card"><h3>Detector de Fracturas Oseas</h3><p>Sistema de soporte de diagnostico basado en redes neuronales convolucionales para el analisis de radiografias y deteccion de posibles fracturas oseas.</p></div>""", unsafe_allow_html=True)
        
        # Lista de caracteristicas del sistema
        st.markdown("""
        <div class="features-list">
            <h4>Caracteristicas del Sistema</h4>
            <ul>
                <li>Analisis automatizado de multiples radiografias</li>
                <li>Nivel de confianza (Bajo/Moderado/Alto riesgo)</li>
                <li>Porcentaje de probabilidad de fractura</li>
                <li>Resaltado de zonas de atencion (heatmap)</li>
                <li>Deteccion de calidad de imagen</li>
                <li>Recomendacion clinica automatizada</li>
                <li>Comparacion de 2 radiografias</li>
                <li>Historial de estudios</li>
                <li>Datos del paciente</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Aviso legal - muy importante para aplicaciones medicas
        st.markdown("""
        <div class="legal-box">
            <h4>Aviso Legal Importante</h4>
            <p>Esta herramienta es solo un apoyo diagnostico. Los resultados deben ser siempre verificados por un profesional de la salud calificado. Esta aplicacion no sustituye el diagnostico medico profesional, el examen fisico ni las pruebas diagnosticas complementarias.</p>
            <p style="margin-top:10px;">El uso de este sistema es responsabilidad del usuario. Los autores y desarrolladores no se hacen responsables de las decisiones tomadas basandose en los resultados de esta aplicacion.</p>
            <p style="margin-top:10px;"><strong>Este NO es un diagnostico medico definitivo.</strong></p>
        </div>
        """, unsafe_allow_html=True)


# Punto de entrada de la aplicacion
# Esta linea se ejecuta cuando se corre el archivo directamente
if __name__ == "__main__":
    main()
