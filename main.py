from fastapi import FastAPI, File, UploadFile, Query
import uvicorn
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
from fastapi.responses import StreamingResponse

app = FastAPI(
    title="API de Detecção de Objetos com YOLO",
    description="Recebe imagens, detecta objetos com YOLO e retorna a imagem com as detecções. "
                "Você pode ajustar o nível de confiança e escolher a versão do modelo.",
    version="1.1.0",
)

def load_model(model_version: str = "yolov8n.pt"):
    # Carrega o modelo YOLO desejado.
    return YOLO(model_version)

@app.post("/detect/")
async def detect_objects(
    file: UploadFile = File(...),
    conf_threshold: float = Query(0.3, description="Nível de confiança mínimo para a detecção"),
    model_version: str = Query("yolov8n.pt", description="Versão do modelo YOLO (ex: yolov8n.pt, yolov8s.pt)")
):
    """
    Recebe uma imagem, realiza a detecção de objetos usando o modelo YOLO especificado e retorna a imagem processada.
    """
    # Carrega o modelo especificado
    model = load_model(model_version)

    # 1. Ler o conteúdo do arquivo enviado
    image_bytes = await file.read()
    np_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # 2. Detectar objetos com YOLO
    results = model.predict(source=image, conf=conf_threshold)

    # 3. Processar os resultados: desenhar bounding boxes
    boxes = results[0].boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())
        cls_name = model.names[cls_id]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            image, f"{cls_name} {conf:.2f}", 
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    # 4. Converter a imagem processada para bytes e retornar
    _, encoded_img = cv2.imencode(".jpg", image)
    return StreamingResponse(BytesIO(encoded_img.tobytes()), media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
