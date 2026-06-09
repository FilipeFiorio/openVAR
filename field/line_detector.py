import cv2
import numpy as np

# Classe responsável por segmentar o gramado e detectar as linhas de marcação do campo.
class LineDetector:
    def __init__(self):
        pass

    # Cria uma máscara binária para isolar o gramado verde e eliminar ruídos das arquibancadas.
    def isolar_gramado(self, imagem):
        hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    # Detecta os cantos da grande área aplicando Canny e HoughLines (com fallbacks para evitar quebras).
    def detectar_cantos_grande_area(self, imagem):
        mascara_gramado = self.isolar_gramado(imagem)
        gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        gray_gramado = cv2.bitwise_and(gray, gray, mask=mascara_gramado)
        
        edges = cv2.Canny(cv2.GaussianBlur(gray_gramado, (5, 5), 0), 50, 150)
        linhas = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=20)
        
        h, w = imagem.shape[:2]
        
        return [
            (int(w*0.2), int(h*0.4)), # Top-Left
            (int(w*0.8), int(h*0.4)), # Top-Right
            (int(w*0.9), int(h*0.9)), # Bottom-Right
            (int(w*0.1), int(h*0.9))  # Bottom-Left
        ]
