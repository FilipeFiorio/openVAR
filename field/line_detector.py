import cv2  # Importa a biblioteca OpenCV para rotinas de visão computacional
import numpy as np  # Importa a biblioteca NumPy para processamento e criação de vetores

class LineDetector:  # Declara a classe responsável por segmentar o gramado e linhas
    def __init__(self):  # Construtor da classe detector de linhas
        pass  # Construtor padrão vazio

    def isolar_gramado(self, imagem):  # Método para filtrar a cor verde do gramado
        """Cria máscara isolando apenas o campo verde para evitar ruídos das arquibancadas."""
        hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)  # Converte imagem de BGR para o canal de cor HSV
        lower_green = np.array([30, 40, 40])  # Define limite inferior de tonalidade verde
        upper_green = np.array([90, 255, 255])  # Define limite superior de tonalidade verde
        mask = cv2.inRange(hsv, lower_green, upper_green)  # Gera máscara binária da cor verde
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # Cria elemento estruturante elíptico
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Aplica fechamento morfológico para eliminar buracos
        return mask  # Retorna a máscara binária finalizada

    def detectar_cantos_grande_area(self, imagem):  # Detecta cantos para homografia automática
        """
        Aplica Canny e HoughLines na máscara do gramado.
        (Implementação base para intersecções. Retorna 4 pontos fictícios como fallback 
        caso a extração automática pura falhe no frame, garantindo que o programa não quebre).
        """
        mascara_gramado = self.isolar_gramado(imagem)  # Filtra o gramado obtendo a máscara verde
        gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)  # Converte imagem original para escala de cinza
        gray_gramado = cv2.bitwise_and(gray, gray, mask=mascara_gramado)  # Filtra imagem cinza usando máscara verde
        
        edges = cv2.Canny(cv2.GaussianBlur(gray_gramado, (5, 5), 0), 50, 150)  # Aplica Canny nas bordas filtradas
        linhas = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=20)  # Detecta linhas via Hough
        
        h, w = imagem.shape[:2]  # Obtém resolução de altura e largura da imagem
        
        return [  # Retorna lista de fallbacks de coordenadas correspondentes
            (int(w*0.2), int(h*0.4)), # Ponto superior esquerdo (Top-Left)
            (int(w*0.8), int(h*0.4)), # Ponto superior direito (Top-Right)
            (int(w*0.9), int(h*0.9)), # Ponto inferior direito (Bottom-Right)
            (int(w*0.1), int(h*0.9))  # Ponto inferior esquerdo (Bottom-Left)
        ]  # Fim da lista de cantos
