import cv2
import numpy as np

class FieldHomography:
    def __init__(self):
        self.H = None
        self.H_inv = None
        # Dimensões da Grande Área oficial FIFA em centímetros (40.32m x 16.5m)
        # Pontos 2D aéreos (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
        self.pontos_aereos = np.array([
            [0, 0],       # Superior Esquerdo: Linha de fundo, lado distante
            [1650, 0],    # Superior Direito: Linha de 16.5m, lado distante
            [1650, 4032], # Inferior Direito: Linha de 16.5m, lado próximo
            [0, 4032]     # Inferior Esquerdo: Linha de fundo, lado próximo
        ], dtype=np.float32)

    def calcular_matriz(self, pontos_tv):
        """Gera a matriz de Homografia cruzando a TV com a planta real."""
        pts_tv = np.array(pontos_tv, dtype=np.float32)
        self.H, _ = cv2.findHomography(pts_tv, self.pontos_aereos, 0)
        if self.H is not None:
            self.H_inv = np.linalg.inv(self.H)
        return self.H

    def pixel_para_metros(self, x_tv, y_tv):
        """Converte uma coordenada da TV para o plano aéreo em centímetros."""
        if self.H is None: return (x_tv, y_tv)
        ponto = np.array([[[x_tv, y_tv]]], dtype=np.float32)
        ponto_projetado = cv2.perspectiveTransform(ponto, self.H)
        return (ponto_projetado[0][0][0], ponto_projetado[0][0][1])

    def metros_para_pixel(self, x_cm, y_cm):
        """Retroprojeta uma coordenada do plano aéreo de volta para a tela da TV."""
        if self.H_inv is None: return (int(x_cm), int(y_cm))
        ponto = np.array([[[x_cm, y_cm]]], dtype=np.float32)
        ponto_tv = cv2.perspectiveTransform(ponto, self.H_inv)
        return (int(ponto_tv[0][0][0]), int(ponto_tv[0][0][1]))
