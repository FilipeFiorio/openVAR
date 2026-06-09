import cv2  # Importa a biblioteca OpenCV para funções de projeção de perspectiva
import numpy as np  # Importa a biblioteca NumPy para manipulação de matrizes

class FieldHomography:  # Declara a classe para manipulação espacial do plano do gramado
    def __init__(self):  # Construtor da classe de homografia
        self.H = None  # Inicializa matriz de homografia direta (TV -> Planta Real)
        self.H_inv = None  # Inicializa matriz de homografia inversa (Planta Real -> TV)
        # Dimensões da Grande Área oficial FIFA em centímetros (40.32m x 16.5m)
        # Pontos 2D aéreos (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
        self.pontos_aereos = np.array([  # Declara as coordenadas Fifa no plano 2D real
            [0, 0],       # Superior Esquerdo: Linha de fundo, lado distante
            [1650, 0],    # Superior Direito: Linha de 16.5m, lado distante
            [1650, 4032], # Inferior Direito: Linha de 16.5m, lado próximo
            [0, 4032]     # Inferior Esquerdo: Linha de fundo, lado próximo
        ], dtype=np.float32)  # Declara tipo de precisão flutuante float32

    def calcular_matriz(self, pontos_tv):  # Calcula homografia entre a TV e a planta real
        """Gera a matriz de Homografia cruzando a TV com a planta real."""
        pts_tv = np.array(pontos_tv, dtype=np.float32)  # Converte cliques em array de ponto flutuante
        self.H, _ = cv2.findHomography(pts_tv, self.pontos_aereos, 0)  # Computa a homografia direta via OpenCV
        if self.H is not None:  # Se a homografia foi calculada com sucesso
            self.H_inv = np.linalg.inv(self.H)  # Calcula e armazena a matriz inversa da homografia
        return self.H  # Retorna a matriz direta computada

    def pixel_para_metros(self, x_tv, y_tv):  # Projeta coordenadas da imagem da TV para o plano FIFA real
        """Converte uma coordenada da TV para o plano aéreo em centímetros."""
        if self.H is None: return (x_tv, y_tv)  # Retorna entrada se homografia estiver nula
        ponto = np.array([[[x_tv, y_tv]]], dtype=np.float32)  # Estrutura coordenada em formato 3D array
        ponto_projetado = cv2.perspectiveTransform(ponto, self.H)  # Projeta ponto via transformada perspectiva
        return (ponto_projetado[0][0][0], ponto_projetado[0][0][1])  # Retorna tupla com (x_cm, y_cm) calculada

    def metros_para_pixel(self, x_cm, y_cm):  # Projeta coordenadas da planta real de volta para a TV
        """Retroprojeta uma coordenada do plano aéreo de volta para a tela da TV."""
        if self.H_inv is None: return (int(x_cm), int(y_cm))  # Retorna inteiros se homografia inversa nula
        ponto = np.array([[[x_cm, y_cm]]], dtype=np.float32)  # Estrutura coordenada real em array
        ponto_tv = cv2.perspectiveTransform(ponto, self.H_inv)  # Aplica transformada perspectiva reversa
        return (int(ponto_tv[0][0][0]), int(ponto_tv[0][0][1]))  # Retorna tupla com coordenadas em pixel (X, Y)
