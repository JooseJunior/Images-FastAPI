# API de Detecção de Objetos com FastAPI & YOLO

Este projeto demonstra como criar uma API que recebe imagens, faz a detecção de objetos utilizando um modelo YOLO (via biblioteca [ultralytics](https://github.com/ultralytics/ultralytics)) e retorna a imagem processada com bounding boxes destacadas.

---

## Tecnologias Utilizadas

- **Python** 3.9+ (ou versão compatível)
- **FastAPI** (para criação da API)
- **Uvicorn** (servidor ASGI para executar a aplicação FastAPI)
- **OpenCV** (tratamento e desenho das bounding boxes nas imagens)
- **Pillow** (manipulação de imagens)
- **Numpy** (manipulação de arrays)
- **Ultralytics** (modelos YOLO, como YOLOv8)
- **python-multipart** (para upload de arquivos na API)

---

## Pré-requisitos

- **Python** instalado (3.9 ou superior é recomendado).
- **pip** (gerenciador de pacotes do Python).
- **Git** para clonar o repositório. (Opcional)

---

## Instruções

1. **Clone ou baixe este repositório**:

   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git

2. **Ambiente Virtual**:
   ```bash
   python -m venv venv

3. **Ativando ambiente virtual**:
   ```bash
   venv\Scripts\activate

4. **Instalação de Dependências**:
   ```bash
   pip install -r requirements.txt

## Executando aplicação via Documentação

5. **Execução da aplicação**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload

6. **Teste via documentação Swagger**:
    * Abra http://127.0.0.1:8000/docs no navegador.
    * Localize o endpoint POST /detect/.
    * Clique em Try it out.
    * Clique em Choose file e selecione uma imagem (por exemplo, imagem.jpg ou imagem.png).
    * Clique em Execute.
    * Se tudo estiver correto, a resposta será a imagem processada.

## Executando aplicação via Streamlit

7. **Execução da aplicação**:
   ```bash
   streamlit run app_streamlit.py

8. **Teste via Streamlit**:
    * Abra http://localhost:8501/ no navegador.
    * Selecione a imagem
    * Caso deseje alterar o nivel de confiança e versão do modelo Yolo, pode alterar (opcional).
    * Clique em Detectar Objetos
    * Se tudo estiver correto, a resposta será a imagem processada.