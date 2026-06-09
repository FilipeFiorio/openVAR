import cv2
import numpy as np

# Classe responsável por gerenciar a projeção de perspectiva (homografia) entre a tela e o plano FIFA real.
class FieldHomography:
    # Inicializa as matrizes de homografia e define as coordenadas aéreas em centímetros da grande área.
    def __init__(self):
        self.H = None
        self.H_inv = None
        # Dimensões da Grande Área oficial FIFA em centímetros (40.32m x 16.5m)
        self.pontos_aereos = np.array([
            [0, 0],       # Superior Esquerdo: Linha de fundo, lado distante
            [1650, 0],    # Superior Direito: Linha de 16.5m, lado distante
            [1650, 4032], # Inferior Direito: Linha de 16.5m, lado próximo
            [0, 4032]     # Inferior Esquerdo: Linha de fundo, lado próximo
        ], dtype=np.float32)

    # Gera a matriz de homografia cruzando os 4 pontos da TV com a planta real do campo.
    def calcular_matriz(self, pontos_tv):
        pts_tv = np.array(pontos_tv, dtype=np.float32)
        self.H, _ = cv2.findHomography(pts_tv, self.pontos_aereos, 0)
        if self.H is not None:
            self.H_inv = np.linalg.inv(self.H)
        return self.H

    # Converte uma coordenada de pixel da tela da TV para metros (centímetros) reais no gramado.
    def pixel_para_metros(self, x_tv, y_tv):
        if self.H is None: return (x_tv, y_tv)
        ponto = np.array([[[x_tv, y_tv]]], dtype=np.float32)
        ponto_projetado = cv2.perspectiveTransform(ponto, self.H)
        return (ponto_projetado[0][0][0], ponto_projetado[0][0][1])

    # Retroprojeta uma coordenada real em centímetros de volta para a tela de pixels da TV.
    def metros_para_pixel(self, x_cm, y_cm):
        if self.H_inv is None: return (int(x_cm), int(y_cm))
        ponto = np.array([[[x_cm, y_cm]]], dtype=np.float32)
        ponto_tv = cv2.perspectiveTransform(ponto, self.H_inv)
        return (int(ponto_tv[0][0][0]), int(ponto_tv[0][0][1]))
