import streamlit as st
import requests
from PIL import Image
from io import BytesIO

st.title("Detecção de Objetos com YOLO via FastAPI")

st.markdown("""
Este aplicativo permite que você envie uma imagem para a API de detecção de objetos e visualize o resultado.
Ajuste os parâmetros conforme necessário.
""")

# Seção para upload de imagem
uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])

# Parâmetros para a detecção
conf_threshold = st.slider("Nível de Confiança Mínimo", min_value=0.1, max_value=1.0, value=0.3, step=0.05)
model_version = st.selectbox("Versão do Modelo YOLO", options=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])

if uploaded_file is not None:
    # Mostra a imagem original
    st.image(uploaded_file, caption="Imagem Original", use_column_width=True)
    
    if st.button("Detectar Objetos"):
        # Prepara a requisição para a API
        files = {"file": uploaded_file.getvalue()}  # Conteúdo binário do arquivo
        params = {
            "conf_threshold": conf_threshold,
            "model_version": model_version
        }
        
        # Supondo que sua API esteja rodando localmente na porta 8000
        try:
            response = requests.post(
                "http://127.0.0.1:8000/detect/",
                files=files,
                params=params
            )
        except requests.exceptions.RequestException as e:
            st.error(f"Erro na requisição: {e}")
            st.stop()
        
        if response.status_code == 200:
            # Exibe a imagem processada
            processed_image = Image.open(BytesIO(response.content))
            st.image(processed_image, caption="Imagem Processada", use_column_width=True)
        else:
            st.error(f"Erro na requisição: {response.status_code}")