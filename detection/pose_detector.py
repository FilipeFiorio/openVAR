from ultralytics import YOLO
import numpy as np

# Classe responsável por executar a inferência de pose YOLO para detectar jogadores e seus tornozelos (pés).
class PoseDetector:
    # Inicializa o modelo YOLOv11 especialista em estimativa de pose humana.
    def __init__(self, model_path="yolo11x-pose.pt"):
        self.model = YOLO(model_path)

    # Executa o modelo de pose na imagem e retorna os dados mapeados de cada jogador (id, bbox, pés e fallbacks).
    def detect_players(self, image):
        resultados = self.model(image, conf=0.02, iou=0.6, imgsz=1280, verbose=False)
        boxes = resultados[0].boxes
        kps_all = resultados[0].keypoints
        
        jogadores = []
        for i, box in enumerate(boxes):
            if int(box.cls[0]) != 0: continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            if (x2 - x1) < 12 or (y2 - y1) < 35: continue
            
            pe_esquerdo = None
            pe_direito = None
            
            if kps_all is not None and i < len(kps_all.data):
                kps = kps_all.data[i]
                if len(kps) > 16:
                    if kps[15][2] > 0.4:
                        pe_esquerdo = (int(kps[15][0]), int(kps[15][1]))
                    if kps[16][2] > 0.4:
                        pe_direito = (int(kps[16][0]), int(kps[16][1]))
            
            if pe_esquerdo is None:
                pe_esquerdo = (int(x1 + (x2 - x1) * 0.25), y2)
            if pe_direito is None:
                pe_direito = (int(x1 + (x2 - x1) * 0.75), y2)
                
            pe_x = (pe_esquerdo[0] + pe_direito[0]) // 2
            pe_y = max(pe_esquerdo[1], pe_direito[1])
 
            jogadores.append({
                "id": i,
                "bbox": (x1, y1, x2, y2),
                "pe_esquerdo": pe_esquerdo,
                "pe_direito": pe_direito,
                "pe_x": pe_x,
                "pe_y": pe_y
            })
            
        return jogadores
