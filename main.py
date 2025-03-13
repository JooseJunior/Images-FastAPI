from fastapi import FastAPI, File, UploadFile
import uvicorn
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi.responses import StreamingResponse

app = FastAPI(
    title="API de Detecção de Objetos com YOLO",
    description="Recebe imagens, detecta objetos usando YOLO e retorna a imagem com as detecções.",
    version="1.0.0",
)

# Carregando um modelo pré-treinado (ex: YOLOv8n)
# Caso queira outra versão, troque o nome do arquivo .pt
model = YOLO("yolov8n.pt")  

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    """
    Recebe uma imagem, realiza a detecção de objetos e retorna a imagem com bounding boxes.
    """

    # 1. Ler o conteúdo do arquivo enviado
    image_bytes = await file.read()
    # 2. Converter para formato que o modelo YOLO entende
    np_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # 3. Executar detecção com YOLO
    # Podemos ajustar parâmetros como conf threshold (conf=0.5, por exemplo)
    results = model.predict(source=image, conf=0.3)  # conf é o nível de confiança

    # 4. Obter as detecções
    # results[0].boxes possui as bounding boxes detectadas
    # results[0].names -> rótulos
    # results[0].probs -> probabilidades (quando aplicável)
    boxes = results[0].boxes

    # 5. Desenhar as bounding boxes na imagem usando OpenCV
    for box in boxes:
        # Cada box é [x1, y1, x2, y2, conf, class]
        #   x1, y1 -> canto superior esquerdo
        #   x2, y2 -> canto inferior direito
        #   conf -> confiança
        #   class -> índice da classe

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()
        class_id = int(box.cls[0].cpu().numpy())

        # Nome da classe
        class_name = model.names[class_id]

        # Desenhar retângulo
        cv2.rectangle(image, 
                      (int(x1), int(y1)), 
                      (int(x2), int(y2)), 
                      (0, 255, 0), 2)

        # Escrever texto (nome da classe + confiança)
        label = f"{class_name} {conf:.2f}"
        cv2.putText(image, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 2)

    # 6. Converter a imagem processada para bytes e retornar na resposta
    # Precisamos converter de volta para JPEG ou PNG para enviar
    _, encoded_img = cv2.imencode(".jpg", image)
    # Encapsular em BytesIO para enviar via StreamingResponse
    return StreamingResponse(BytesIO(encoded_img.tobytes()), media_type="image/jpeg")


# Ponto de entrada para executar usando: python main.py
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
