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

## Instalação

1. **Clone ou baixe este repositório**:

   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git

2. **Ambiente Virtual**:
   ```bash
   python -m venv venv
   venv\Scripts\activate

3. **Instalação de Dependências**:
   ```bash
   pip install -r requirements.txt

4. **Execução da aplicação**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload

4. **Teste via documentação Swagger**:
    * Abra http://127.0.0.1:8000/docs no navegador.
    * Localize o endpoint POST /detect/.
    * Clique em Try it out.
    * Clique em Choose file e selecione uma imagem (por exemplo, imagem.jpg ou imagem.png).
    * Clique em Execute.
    * Se tudo estiver correto, a resposta será a imagem processada (em bytes). No Swagger, ela aparecerá em base64.