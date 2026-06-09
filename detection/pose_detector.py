from ultralytics import YOLO  # Importa a biblioteca YOLO da Ultralytics para IA
import numpy as np  # Importa a biblioteca NumPy para processamento matemático

class PoseDetector:  # Declara a classe responsável pela detecção de poses
    def __init__(self, model_path="yolo11x-pose.pt"):  # Construtor do detector de pose
        self.model = YOLO(model_path)  # Carrega o modelo YOLO especialista em poses

    def detect_players(self, image):  # Método principal para detecção de jogadores e pés
        """
        Retorna uma lista de dicionários contendo os dados de cada jogador.
        Extrai os tornozelos (índices 15 e 16) para projetar os pés no gramado.
        """
        resultados = self.model(image, conf=0.02, iou=0.6, imgsz=1280, verbose=False)  # Executa inferência do YOLO
        boxes = resultados[0].boxes  # Coleta caixas delimitadoras (bboxes) encontradas
        kps_all = resultados[0].keypoints  # Coleta keypoints de pose humana estimados
        
        jogadores = []  # Inicializa lista contendo os dados dos jogadores
        for i, box in enumerate(boxes):  # Itera sobre cada caixa de detecção
            if int(box.cls[0]) != 0: continue  # Filtra para manter somente classe 0 (pessoas)
            
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Converte limites de bbox para inteiros
            if (x2 - x1) < 12 or (y2 - y1) < 35: continue  # Descarta detecções ruidosas muito pequenas
            
            pe_esquerdo = None  # Inicializa marcador do pé esquerdo
            pe_direito = None  # Inicializa marcador do pé direito
            
            if kps_all is not None and i < len(kps_all.data):  # Se há keypoints válidos na detecção
                kps = kps_all.data[i]  # Pega os keypoints do jogador atual
                if len(kps) > 16:  # Se possui o número mínimo de juntas necessárias
                    if kps[15][2] > 0.4:  # Confiança do tornozelo esquerdo aceitável
                        pe_esquerdo = (int(kps[15][0]), int(kps[15][1]))  # Salva ponto do pé esquerdo
                    if kps[16][2] > 0.4:  # Confiança do tornozelo direito aceitável
                        pe_direito = (int(kps[16][0]), int(kps[16][1]))  # Salva ponto do pé direito
            
            if pe_esquerdo is None:  # Caso o pé esquerdo não tenha sido detectado
                pe_esquerdo = (int(x1 + (x2 - x1) * 0.25), y2)  # Fallback na base esquerda da caixa
            if pe_direito is None:  # Caso o pé direito não tenha sido detectado
                pe_direito = (int(x1 + (x2 - x1) * 0.75), y2)  # Fallback na base direita da caixa
                
            pe_x = (pe_esquerdo[0] + pe_direito[0]) // 2  # Posição X padrão é o ponto médio dos pés
            pe_y = max(pe_esquerdo[1], pe_direito[1])  # Posição Y padrão é a menor no plano vertical
 
            jogadores.append({  # Insere dicionário com dados coletados do jogador
                "id": i,  # ID identificador único
                "bbox": (x1, y1, x2, y2),  # Limites de caixa no plano 2D
                "pe_esquerdo": pe_esquerdo,  # Ponto do pé esquerdo
                "pe_direito": pe_direito,  # Ponto do pé direito
                "pe_x": pe_x,  # X guia inicial
                "pe_y": pe_y  # Y guia inicial
            })  # Fim do append
            
        return jogadores  # Retorna lista estruturada de jogadores detectados
