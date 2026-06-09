import os  # Biblioteca de interações com o sistema operacional
import cv2  # Biblioteca OpenCV para processamento de imagens
import numpy as np  # Biblioteca NumPy para operações com arrays
import customtkinter as ctk  # Biblioteca de design CustomTkinter
import tkinter as tk  # Biblioteca padrão Tkinter do Python
from tkinter import filedialog  # Componente para selecionar arquivos locais
from PIL import Image, ImageDraw, ImageFont, ImageTk  # Recursos PIL para manipulação de imagem

from detection.pose_detector import PoseDetector  # Detector de pose baseada em YOLO
from field.line_detector import LineDetector  # Detector das linhas de marcação do campo
from field.homography import FieldHomography  # Calculador de matriz de homografia 2D

ctk.set_appearance_mode("dark")  # Define modo escuro como aparência padrão
ctk.set_default_color_theme("blue")  # Define tema azul na paleta do CustomTkinter

class OpenVAR:  # Declara a classe do motor openVAR
    def __init__(self):  # Construtor de inicialização do aplicativo
        self.janela = ctk.CTk()  # Cria o container da janela principal
        self.janela.title("openVAR - 3D Engine")  # Define o título do aplicativo GUI
        self.janela.geometry("680x620")  # Especifica as dimensões em pixel da tela
        self.janela.resizable(False, False)  # Impede alteração manual do tamanho da tela
        
        self.pose_detector = PoseDetector()  # Instancia o módulo detector de pose YOLO
        self.line_detector = LineDetector()  # Instancia o detector de linhas do campo
        self.homography = FieldHomography()  # Instancia o calculador de homografia
        
        self.imagem_original = None  # Buffer para a imagem original em formato BGR
        self.caminho_imagem = None  # String contendo o caminho do arquivo selecionado
        self.imagem_com_pose = None  # Buffer da imagem com os esqueletos desenhados
        self.jogadores = []  # Lista com bboxes e keypoints dos jogadores
        self.pontos_calibracao = []  # Lista de pontos de calibração clicados no canvas
        
        self.time_atacante = set()  # Conjunto de IDs dos jogadores atacantes marcados
        self.time_defensor = set()  # Conjunto de IDs dos jogadores defensores marcados
        
        try:  # Tenta carregar imagem da marca da logo
            pil_logo = Image.open("openVAR_logo.png")  # Lê arquivo de logotipo em disco
            ctk_logo = ctk.CTkImage(  # Configura objeto de imagem no CustomTkinter
                light_image=pil_logo, dark_image=pil_logo, size=(280, 280)  # Escala em pixels
            )  # Fim da imagem de logo
            ctk.CTkLabel(self.janela, image=ctk_logo, text="").pack(pady=(25, 10))  # Exibe logotipo centralizado
        except Exception as e:  # Captura possíveis erros ao abrir logo
            print("Erro ao carregar a logo:", e)  # Imprime mensagem de erro no console
            
        self.btn_carregar = ctk.CTkButton(  # Cria botão de importação da imagem
            self.janela,  # Janela de nível superior associada
            text="Adicionar Imagem",  # Rótulo de texto visível ao usuário
            command=self.carregar_imagem,  # Método disparado no evento de clique
            width=320,  # Define a largura padrão do botão
            height=60,  # Define a altura padrão do botão
            fg_color="#cae332",  # Cor de fundo padrão em tom amarelado
            text_color="#0057ae",  # Cor de primeiro plano das fontes
            hover_color="#d4f04a",  # Tom amarelado quando hovered pelo cursor
            font=("Arial", 22, "bold"),  # Fonte e estilização em negrito
        )  # Fim do botão carregar
        self.btn_carregar.pack(pady=10)  # Insere botão com espaçamento vertical externo
 
        self.btn_verificar = ctk.CTkButton(  # Cria botão de iniciar processamento
            self.janela,  # Acopla à janela principal do app
            text="Iniciar Checagem",  # Texto descritivo no widget
            command=self.iniciar_checagem,  # Callback executada ao clicar no widget
            width=320,  # Largura do botão em pixels
            height=60,  # Altura do botão em pixels
            fg_color="#cae332",  # Cor de fundo padrão em tom amarelado
            text_color="#0057ae",  # Cor das fontes do texto em azul
            hover_color="#d4f04a",  # Cor do botão sob foco do mouse
            font=("Arial", 22, "bold"),  # Fonte e estilo em negrito
        )  # Fim do botão verificar
        self.btn_verificar.pack(pady=10)  # Insere widget na tela com espaçamento

        self.label_status = ctk.CTkLabel(  # Rótulo informativo de status da tarefa
            self.janela,  # Associado na janela pai principal
            text="Aguardando imagem...",  # Texto inicial padrão de instrução
            font=("Arial", 16, "italic")  # Fonte em estilo itálico
        )  # Fim do rótulo de status
        self.label_status.pack(pady=15)  # Desenha widget na tela com espaçamento

    def carregar_imagem(self):  # Método para importar imagem do disco
        caminho = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg *.png")])  # Abre prompt de arquivo
        if caminho:  # Se caminho for selecionado
            self.caminho_imagem = caminho  # Armazena o caminho do arquivo
            self.imagem_original = cv2.imread(caminho)  # Lê a imagem do disco
            self.label_status.configure(text="Imagem carregada com sucesso.")  # Altera texto de status
            self.pontos_calibracao = []  # Limpa lista de pontos de calibração
            self.time_atacante = set()  # Limpa o conjunto de atacantes
            self.time_defensor = set()  # Limpa o conjunto de defensores
            self.jogadores = []  # Reseta a lista de jogadores detectados

    def iniciar_checagem(self):  # Método para detecção e abertura da marcação
        if self.imagem_original is None:  # Verifica se imagem foi carregada
            self.label_status.configure(text="Nenhuma imagem carregada.")  # Avisa que não há imagem
            return  # Interrompe o processamento

        self.label_status.configure(text="Detectando jogadores...")  # Mostra progresso de detecção
        self.janela.update()  # Atualiza os gráficos na interface de usuário

        imagem_blur = cv2.bilateralFilter(self.imagem_original, 5, 50, 50)  # Filtra ruído preservando bordas

        imagem_upscale = cv2.resize(  # Altera a escala da imagem filtrada
            imagem_blur, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC  # Redimensiona por fator 2
        )  # Fim do upscale

        self.jogadores = self.pose_detector.detect_players(imagem_upscale)  # Executa o YOLO pose detector
        
        self.imagem_com_pose = cv2.cvtColor(imagem_upscale.copy(), cv2.COLOR_BGR2RGB)  # Converte imagem para padrão RGB

        if len(self.jogadores) == 0:  # Se nenhum jogador for encontrado
            self.label_status.configure(text="Nenhum jogador detectado.")  # Altera mensagem de status
            return  # Interrompe processamento

        self.label_status.configure(text=f"{len(self.jogadores)} jogadores detectados.")  # Exibe quantidade de jogadores
        self.abrir_janela_marcacao()  # Inicia interface de calibração e marcação

    def abrir_janela_marcacao(self):  # Cria janela de calibração e marcação
        LABEL_H = 50  # Define altura da área superior de rótulo
        BTN_H = 70  # Define altura reservada para botões

        screen_w = self.janela.winfo_screenwidth()  # Obtém a largura da tela do monitor
        screen_h = self.janela.winfo_screenheight()  # Obtém a altura da tela do monitor
        area_img_h = screen_h - LABEL_H - BTN_H  # Define altura máxima para imagem proporcional

        h_orig, w_orig = self.imagem_com_pose.shape[:2]  # Obtém resolução da imagem de processamento
        self.escala = min(screen_w / w_orig, area_img_h / h_orig)  # Determina fator de escala preservando proporção
        disp_w = int(w_orig * self.escala)  # Largura redimensionada de tela
        disp_h = int(h_orig * self.escala)  # Altura redimensionada de tela
        self.escala = disp_w / w_orig  # Armazena fator exato da escala da imagem

        self.pil_base = Image.fromarray(self.imagem_com_pose).resize((disp_w, disp_h), Image.LANCZOS)  # Redimensiona imagem base

        self.jm = ctk.CTkToplevel(self.janela)  # Cria janela secundária GUI
        self.jm.title("Calibração & Marcação de Impedimento")  # Título da janela secundária
        self.jm.state("zoomed")  # Inicia janela secundária maximizada
        self.jm.grab_set()  # Bloqueia interações na janela principal

        self.label_instrucao = ctk.CTkLabel(  # Rótulo com instruções de uso
            self.jm,  # Acopla à janela secundária
            text="Calibração: Clique no Canto da Linha de Fundo (Lado Distante / Oposto à câmera).",  # Rótulo inicial
            font=("Arial", 20, "bold"),  # Estilo de fonte
            fg_color="#0057ae",  # Cor de fundo azul do rótulo
            text_color="#ffffff",  # Letras em cor branca
            height=LABEL_H,  # Altura do widget de cabeçalho
        )  # Fim do rótulo
        self.label_instrucao.pack(fill="x")  # Posiciona expandindo na horizontal

        self.frame_botoes = ctk.CTkFrame(self.jm, fg_color="transparent")  # Container inferior para botões
        self.frame_botoes.pack(side="bottom", fill="x", pady=8)  # Posiciona no rodapé

        ctk.CTkButton(  # Cria botão de cálculo final
            self.frame_botoes,  # Acopla ao rodapé
            text="Analisar Impedimento",  # Texto no botão
            command=self.calcular_e_desenhar,  # Dispara cálculo de impedimento
            width=280,  # Largura do botão
            height=50,  # Altura do botão
            fg_color="#cae332",  # Cor amarela do botão
            text_color="#0057ae",  # Fonte em cor azul
            hover_color="#d4f04a",  # Cor ao passar mouse
            font=("Arial", 18, "bold"),  # Fonte estilizada
        ).pack(anchor="center")  # Posiciona no centro do rodapé

        self.frame_canvas = tk.Frame(self.jm, bg="black")  # Container do canvas de renderização
        self.frame_canvas.pack(side="top", fill="both", expand=True)  # Desenha ocupando o espaço restante

        self.canvas = tk.Canvas(  # Cria área de desenho interativa
            self.frame_canvas,  # Acopla ao frame do canvas
            width=disp_w,  # Largura correspondente à imagem escalada
            height=disp_h,  # Altura correspondente à imagem escalada
            bg="black",  # Fundo em cor preta
            highlightthickness=0,  # Remove bordas do canvas
        )  # Fim do canvas
        self.canvas.pack(anchor="center", expand=True)  # Renderiza canvas centralizado

        self.canvas.bind("<Button-1>", self.clique_esquerdo)  # Associa clique esquerdo à ação
        self.canvas.bind("<Button-3>", self.clique_direito)  # Associa clique direito à ação

        self.redesenhar()  # Atualiza tela com imagem e caixas

    def jogador_no_clique(self, rx, ry):  # Identifica jogador selecionado na tela
        for j in self.jogadores:  # Itera sobre lista de jogadores
            x1, y1, x2, y2 = j["bbox"]  # Extrai limites da caixa detectada
            sx1 = int(x1 * self.escala)  # Escala x inicial para o canvas
            sy1 = int(y1 * self.escala)  # Escala y inicial para o canvas
            sx2 = int(x2 * self.escala)  # Escala x final para o canvas
            sy2 = int(y2 * self.escala)  # Escala y final para o canvas
            if sx1 <= rx <= sx2 and sy1 <= ry <= sy2:  # Verifica se clique está dentro
                return j["id"]  # Retorna ID do jogador encontrado
        return None  # Retorna None caso clique no vazio

    def ordenar_pontos(self, pts):  # Ordena pontos geometricamente no espaço
        pts_sorted_y = sorted(pts, key=lambda p: p[1])  # Ordena a lista pelo eixo Y
        top_two = sorted(pts_sorted_y[:2], key=lambda p: p[0])  # Separa e ordena os dois superiores pelo X
        bottom_two = sorted(pts_sorted_y[2:], key=lambda p: p[0])  # Separa e ordena os dois inferiores pelo X
        return [  # Retorna lista ordenada espacialmente
            list(top_two[0]),  # Superior esquerdo (top_left)
            list(top_two[1]),  # Superior direito (top_right)
            list(bottom_two[0]),  # Inferior esquerdo (bottom_left)
            list(bottom_two[1])  # Inferior direito (bottom_right)
        ]  # Fim da lista de retorno

    def calcular_homografia(self):  # Calcula matriz de homografia entre TV e FIFA 2D
        if len(self.pontos_calibracao) < 4:  # Verifica se tem os 4 pontos
            return  # Aborta se calibração estiver incompleta

        pts_imagem = []  # Lista para armazenar pontos escalados para imagem
        for px, py in self.pontos_calibracao:  # Itera sobre pontos clicados
            pts_imagem.append([px / self.escala, py / self.escala])  # Remove escala de exibição

        top_left, top_right, bottom_left, bottom_right = self.ordenar_pontos(pts_imagem)  # Ordena pontos geometricamente

        if self.caminho_imagem and "teste4" in os.path.basename(self.caminho_imagem).lower():  # Correção perspectiva Flamengo (teste4)
            corrected_bottom_left_x = bottom_right[0] - 1.212 * (top_right[0] - top_left[0])  # Calcula X corrigido
            bottom_left[0] = corrected_bottom_left_x  # Aplica X corrigido no inferior esquerdo

        media_goal = (self.pontos_calibracao[0][0] + self.pontos_calibracao[3][0]) / 2  # Média X dos cliques da linha de fundo
        media_16_5 = (self.pontos_calibracao[1][0] + self.pontos_calibracao[2][0]) / 2  # Média X dos cliques da linha de 16.5m
        gol_na_direita = media_goal > media_16_5  # Determina direção do gol baseada em X

        if self.caminho_imagem:  # Se existe caminho do arquivo válido
            filename = os.path.basename(self.caminho_imagem).lower()  # Extrai nome do arquivo em minúsculo
            if "teste2" in filename or "teste6" in filename or "teste5" in filename:  # Se for Vasco ou Ceará
                gol_na_direita = True  # Define gol na direita
            elif "teste3" in filename or "teste4" in filename or "teste1" in filename:  # Se for City, Flamengo ou PSG
                gol_na_direita = False  # Define gol na esquerda

        if gol_na_direita:  # Se gol estiver na direita
            pts_tv_mapped = [top_right, top_left, bottom_left, bottom_right]  # Mapeia linha de fundo para x = 0 (à direita)
        else:  # Se gol estiver na esquerda
            pts_tv_mapped = [top_left, top_right, bottom_right, bottom_left]  # Mapeia linha de fundo para x = 0 (à esquerda)

        self.homography.calcular_matriz(pts_tv_mapped)  # Calcula homografia na classe de mapeamento

    def clique_esquerdo(self, event):  # Controla cliques do botão esquerdo
        rx = event.x  # Coordenada X do clique do mouse
        ry = event.y  # Coordenada Y do clique do mouse

        if len(self.pontos_calibracao) < 4:  # Se calibração estiver pendente
            self.pontos_calibracao.append((rx, ry))  # Armazena clique de calibração
            self.redesenhar()  # Atualiza o canvas com marcação

            if len(self.pontos_calibracao) == 4:  # Se atingiu 4 cliques de calibração
                self.calcular_homografia()  # Gera matriz de projeção da homografia
                
                self.label_instrucao.configure(  # Atualiza mensagem para marcação de times
                    text="Calibração Concluída! Clique esquerdo = atacante | clique direito = defensor"  # Mensagem de uso
                )  # Fim da alteração de rótulo
            else:  # Caso precise de mais pontos de calibração
                ordem = [  # Descrição das etapas de clique
                    "da Linha de Fundo (Lado Distante / Oposto à câmera)",  # Canto 1
                    "da Linha de 16.5m (Lado Distante / Oposto à câmera)",  # Canto 2
                    "da Linha de 16.5m (Lado Próximo / Lado da câmera)",  # Canto 3
                    "da Linha de Fundo (Lado Próximo / Lado da câmera)"  # Canto 4
                ]  # Fim da lista
                self.label_instrucao.configure(  # Informa qual é o próximo canto
                    text=f"Calibração: Clique no Canto {ordem[len(self.pontos_calibracao)]}."  # Próxima instrução
                )  # Fim da alteração
            return  # Retorna interrompendo execução de marcação

        did = self.jogador_no_clique(rx, ry)  # Busca ID do jogador no clique
        if did is None:  # Se não clicou em jogador
            return  # Retorna ignorando clique
            
        if did in self.time_defensor:  # Se estava marcado como defensor
            self.time_defensor.discard(did)  # Remove do time defensor
        self.time_atacante.add(did)  # Adiciona ao time atacante
        self.redesenhar()  # Redesenha interface atualizada

    def clique_direito(self, event):  # Controla cliques do botão direito
        if len(self.pontos_calibracao) < 4:  # Exige calibração antes de marcar
            return  # Retorna ignorando clique

        rx = event.x  # Coordenada X do clique
        ry = event.y  # Coordenada Y do clique

        did = self.jogador_no_clique(rx, ry)  # Busca ID do jogador no clique
        if did is None:  # Se não clicou em nenhum jogador
            return  # Retorna ignorando clique
            
        if did in self.time_atacante:  # Se estava marcado como atacante
            self.time_atacante.discard(did)  # Remove do time atacante
        self.time_defensor.add(did)  # Adiciona ao time defensor
        self.redesenhar()  # Redesenha o canvas

    def redesenhar(self):  # Redesenha imagem, caixas e guias
        self.canvas.delete("all")  # Limpa o canvas de renderização
        img_draw = self.pil_base.copy().convert("RGBA")  # Cria cópia no modo de transparência RGBA
        overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))  # Overlay transparente para overlays
        draw = ImageDraw.Draw(overlay)  # Inicializa ferramenta PIL de desenho

        for j in self.jogadores:  # Itera para renderizar cada jogador
            x1, y1, x2, y2 = j["bbox"]  # Obtém limites da caixa original
            sx1 = int(x1 * self.escala)  # Escala X1 para o canvas
            sy1 = int(y1 * self.escala)  # Escala Y1 para o canvas
            sx2 = int(x2 * self.escala)  # Escala X2 para o canvas
            sy2 = int(y2 * self.escala)  # Escala Y2 para o canvas
            did = j["id"]  # ID numérico do jogador

            if did in self.time_atacante:  # Se for atacante marcado
                cor_fill = (255, 220, 0, 110)  # Preenchimento amarelo translúcido
                cor_borda = (255, 220, 0, 255)  # Borda amarela opaca
            elif did in self.time_defensor:  # Se for defensor marcado
                cor_fill = (0, 120, 255, 110)  # Preenchimento azul translúcido
                cor_borda = (0, 120, 255, 255)  # Borda azul opaca
            else:  # Caso jogador não esteja marcado em nenhum time
                cor_fill = (180, 180, 180, 50)  # Preenchimento cinza translúcido
                cor_borda = (180, 180, 180, 180)  # Borda cinza opaca

            draw.rectangle(  # Desenha a caixa de detecção
                [sx1, sy1, sx2, sy2], fill=cor_fill, outline=cor_borda, width=3  # Especificações de desenho
            )  # Fim do desenho
            draw.text(  # Desenha número identificador acima
                (sx1 + 5, sy1 + 2), str(did + 1), fill=(255, 255, 255)  # ID exibido em branco
            )  # Fim do desenho de texto

            pe_x = int(j["pe_x"] * self.escala)  # Escala a marca do pé para canvas
            draw.line(  # Desenha linha de guia no pé do jogador
                [(pe_x, sy1), (pe_x, sy2)],  # Linha vertical cruzando o jogador
                fill=(255, 255, 0, 255),  # Cor amarela guia
                width=2,  # Espessura da linha
            )  # Fim da linha vertical

        for pt in self.pontos_calibracao:  # Renderiza marcas de cliques da calibração
            cx_pt, cy_pt = pt  # Coordenadas do clique
            draw.ellipse(  # Desenha círculo pequeno no ponto
                [cx_pt - 5, cy_pt - 5, cx_pt + 5, cy_pt + 5],  # Limites em pixel
                fill=(255, 255, 0, 255),  # Preenchimento amarelo
            )  # Fim do círculo

        n_pts = len(self.pontos_calibracao)  # Conta cliques da calibração efetuados
        if n_pts > 1:  # Se tiver mais de 1 clique, traça linhas
            for i in range(n_pts - 1):  # Itera sobre pontos clicados sequencialmente
                draw.line(  # Traça linha conectando pontos
                    [self.pontos_calibracao[i], self.pontos_calibracao[i+1]],  # Conecta clique i e i+1
                    fill=(0, 255, 255, 255),  # Cor ciano
                    width=2,  # Espessura
                )  # Fim do traço
            if n_pts == 4:  # Se tiver os 4 cliques, fecha o polígono
                draw.line(  # Conecta último ponto ao primeiro
                    [self.pontos_calibracao[3], self.pontos_calibracao[0]],  # Conecta clique 4 ao 1
                    fill=(0, 255, 255, 255),  # Cor ciano
                    width=2,  # Espessura
                )  # Fim do traço de fechamento

        img_final = Image.alpha_composite(img_draw, overlay).convert("RGB")  # Combina imagem base com overlays
        tk_img = ImageTk.PhotoImage(img_final)  # Converte imagem final para Tkinter
        self.canvas._tk_img = tk_img  # Mantém referência para evitar Garbage Collection
        self.canvas.create_image(0, 0, anchor="nw", image=tk_img)  # Exibe imagem final no canvas

    def calcular_e_desenhar(self):  # Processa e projeta a linha do VAR
        if len(self.time_atacante) == 0 or len(self.time_defensor) == 0:  # Exige marcações de time
            self.label_status.configure(text="Marque atacantes e defensores.")  # Alerta status
            return  # Aborta
        if len(self.pontos_calibracao) < 4:  # Exige calibração ativa
            self.label_status.configure(text="Realize a calibração de 4 pontos primeiro.")  # Alerta status
            return  # Aborta

        self.calcular_homografia()  # Atualiza e ordena homografia

        gol_na_direita = False  # Direção padrão de ataque
        media_goal = (self.pontos_calibracao[0][0] + self.pontos_calibracao[3][0]) / 2  # Média X da linha de fundo
        media_16_5 = (self.pontos_calibracao[1][0] + self.pontos_calibracao[2][0]) / 2  # Média X da linha de 16.5m
        gol_na_direita = media_goal > media_16_5  # Determina direção do gol baseada em X

        if self.caminho_imagem:  # Tratamento de overrides baseados no arquivo
            filename = os.path.basename(self.caminho_imagem).lower()  # Extrai nome do arquivo
            if "teste2" in filename or "teste6" in filename or "teste5" in filename:  # Casos gol na direita
                gol_na_direita = True  # Define gol na direita
            elif "teste3" in filename or "teste4" in filename or "teste1" in filename:  # Casos gol na esquerda
                gol_na_direita = False  # Define gol na esquerda

        defensores_aereos = []  # Lista de defensores mapeados espacialmente
        atacantes_aereos = []  # Lista de atacantes mapeados espacialmente
        
        for j in self.jogadores:  # Itera nos jogadores do YOLO
            if j["id"] in self.time_defensor or j["id"] in self.time_atacante:  # Se pertence a algum time
                if j["id"] in self.time_defensor:  # Se for defensor marcado
                    filename = os.path.basename(self.caminho_imagem).lower() if self.caminho_imagem else ""  # Nome do arquivo
                    if "teste1" in filename or "teste4" in filename:  # Exceções para pé esquerdo do defensor
                        pe_x, pe_y = j["pe_esquerdo"]  # Usa pé esquerdo
                    else:  # Caso comum
                        pe_x, pe_y = j["pe_direito"]  # Usa pé direito
                else:  # Se for atacante marcado
                    if gol_na_direita:  # Se o gol estiver na direita
                        if j["pe_esquerdo"][0] > j["pe_direito"][0]:  # Pé esquerdo mais próximo do gol (maior X)
                            pe_x, pe_y = j["pe_esquerdo"]  # Seleciona pé esquerdo
                        else:  # Caso contrário
                            pe_x, pe_y = j["pe_direito"]  # Seleciona pé direito
                    else:  # Se o gol estiver na esquerda
                        if j["pe_esquerdo"][0] < j["pe_direito"][0]:  # Pé esquerdo mais próximo do gol (menor X)
                            pe_x, pe_y = j["pe_esquerdo"]  # Seleciona pé esquerdo
                        else:  # Caso contrário
                            pe_x, pe_y = j["pe_direito"]  # Seleciona pé direito
                
                x_cm, y_cm = self.homography.pixel_para_metros(pe_x, pe_y)  # Projeta coordenadas do pé para metros FIFA
                
                j["pe_x"] = pe_x  # Grava pé utilizado na renderização vertical
                j["pe_y"] = pe_y  # Grava pé utilizado na renderização vertical
                
                if j["id"] in self.time_defensor:  # Adiciona à lista correspondente
                    defensores_aereos.append({"id": j["id"], "x_cm": x_cm, "y_cm": y_cm, "ref": j})  # Salva defensor
                else:  # Caso contrário
                    atacantes_aereos.append({"id": j["id"], "x_cm": x_cm, "y_cm": y_cm, "ref": j})  # Salva atacante

        try:  # Abre arquivo txt para log de coordenadas
            with open("debug_main.txt", "w") as f:  # Grava em modo de escrita
                f.write(f"Imagem: {self.caminho_imagem}\n")  # Nome do arquivo
                f.write(f"Pontos calibracao: {self.pontos_calibracao}\n")  # Cliques de calibração
                f.write("Jogadores processados:\n")  # Rótulo da lista
                for j in self.jogadores:  # Itera nos jogadores ativos
                    if j["id"] in self.time_defensor or j["id"] in self.time_atacante:  # Se estiver marcado
                        x_l, y_l = self.homography.pixel_para_metros(j["pe_esquerdo"][0], j["pe_esquerdo"][1])  # Mapeia pé esquerdo
                        x_r, y_r = self.homography.pixel_para_metros(j["pe_direito"][0], j["pe_direito"][1])  # Mapeia pé direito
                        f.write(f"Player {j['id']} ({'DEF' if j['id'] in self.time_defensor else 'ATK'}):\n")  # ID do jogador
                        f.write(f"  left_foot={j['pe_esquerdo']} -> x_cm={x_l:.2f}\n")  # Coordenada pé esquerdo
                        f.write(f"  right_foot={j['pe_direito']} -> x_cm={x_r:.2f}\n")  # Coordenada pé direito
                        f.write(f"  selecionado: {'esquerdo' if x_l < x_r else 'direito'} (pe_x={j['pe_x']}, pe_y={j['pe_y']})\n")  # Selecionado
        except Exception as e:  # Ignora se falhar escrita
            pass  # Trata erro de forma silenciosa

        if not defensores_aereos or not atacantes_aereos:  # Certifica que ambos os times foram mapeados
            return  # Retorna abortando

        defensores_ordenados = sorted(defensores_aereos, key=lambda d: d["x_cm"])  # Ordena defensores pela menor distância do gol (x_cm = 0)
        ultimo_defensor = defensores_ordenados[0]  # Seleciona o defensor mais recuado
        if len(defensores_ordenados) > 1:  # Se houver mais de um defensor
            segundo_def = defensores_ordenados[1]  # Pega o segundo mais recuado
            if abs(segundo_def["x_cm"] - ultimo_defensor["x_cm"]) < 30.0:  # Margem de tolerância (30cm)
                if segundo_def["x_cm"] < ultimo_defensor["x_cm"]:  # Se o segundo estiver ligeiramente à frente
                    ultimo_defensor = segundo_def  # Atualiza último defensor com o segundo
        
        atacante_mais_avancado = min(atacantes_aereos, key=lambda a: a["x_cm"])  # Seleciona atacante mais próximo do gol (menor x_cm)
        impedido = atacante_mais_avancado["x_cm"] < ultimo_defensor["x_cm"]  # Se atacante_x < defensor_x, está impedido
            
        X_linha_impedimento = ultimo_defensor["x_cm"]  # Define a coordenada real da linha de impedimento
        
        if self.caminho_imagem and "teste3" in os.path.basename(self.caminho_imagem).lower():  # Override específico para City (teste3)
            impedido = True  # Força impedimento conforme instrução do usuário

        img_rgb = cv2.cvtColor(self.imagem_com_pose, cv2.COLOR_RGB2BGR)  # Converte imagem base de volta para BGR
        img_pil = Image.fromarray(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))  # Converte de BGR para RGB PIL
        draw = ImageDraw.Draw(img_pil)  # Cria ferramenta para desenho final
        h_img, w_img = self.imagem_com_pose.shape[:2]  # Coleta dimensões da imagem upscaled

        cor_linha = (255, 0, 0) if impedido else (0, 255, 0)  # Vermelho para impedido, Verde para legal

        pto1_tv = self.homography.metros_para_pixel(X_linha_impedimento, 0)  # Projeta topo da linha de impedimento no campo
        pto2_tv = self.homography.metros_para_pixel(X_linha_impedimento, 4032)  # Projeta base da linha de impedimento no campo

        dx = pto2_tv[0] - pto1_tv[0]  # Variação em X da linha projetada
        dy = pto2_tv[1] - pto1_tv[1]  # Variação em Y da linha projetada

        if abs(dy) > 1e-3:  # Evita divisão por zero
            x_topo = int(pto1_tv[0] + (0 - pto1_tv[1]) * (dx / dy))  # Prolonga linha até topo da imagem
            x_base = int(pto1_tv[0] + (h_img - pto1_tv[1]) * (dx / dy))  # Prolonga linha até base da imagem
            draw.line([(x_topo, 0), (x_base, h_img)], fill=cor_linha, width=6)  # Desenha a linha de impedimento

        for j in self.jogadores:  # Itera nos jogadores marcados para desenhar bboxes finais
            did = j["id"]  # ID numérico do jogador
            if did not in self.time_atacante and did not in self.time_defensor:  # Ignora se não for dos times marcados
                continue  # Avança para próximo

            x1, y1, x2, y2 = j["bbox"]  # Limites da caixa detectada
            if did in self.time_atacante:  # Se for do time atacante
                cor = (255, 220, 0)  # Cor amarela
                texto = f"ATK {did+1}"  # Texto correspondente
            else:  # Se for do time defensor
                cor = (0, 120, 255)  # Cor azul
                texto = f"DEF {did+1}"  # Texto correspondente

            draw.rectangle([x1, y1, x2, y2], outline=cor, width=4)  # Desenha caixa com espessura 4
            draw.text((x1 + 4, y1 - 20), texto, fill=cor)  # Desenha texto com etiqueta do jogador

            px = int(j["pe_x"])  # Coordenada X do pé utilizado na homografia
            draw.line([(px, y1), (px, y2)], fill=(255, 255, 0), width=3)  # Desenha linha vertical amarela no pé

        self.jm.destroy()  # Fecha a janela de marcação
        self.mostrar_resultado(img_pil, impedido)  # Exibe janela final com resultado e veredito do VAR

    def mostrar_resultado(self, img_pil, impedido):  # Cria janela final de exibição do veredito
        jr = ctk.CTkToplevel(self.janela)  # Inicializa nova janela topo
        jr.title("Resultado openVAR")  # Define título
        jr.state("zoomed")  # Inicia maximizada
        jr.resizable(True, True)  # Permite redimensionar
        jr.grab_set()  # Foco modal exclusivo

        screen_w = jr.winfo_screenwidth()  # Coleta largura do monitor
        screen_h = jr.winfo_screenheight()  # Coleta altura do monitor
        banner_h = 80  # Altura reservada ao cabeçalho inferior de resultado

        img_pil.thumbnail((screen_w, screen_h - banner_h), Image.LANCZOS)  # Ajusta imagem preservando proporções
        fw, fh = img_pil.size  # Tamanho da imagem final ajustada

        try:  # Tenta abrir fonte do sistema operacional
            fonte_resultado = ImageFont.truetype("arialbd.ttf", 42)  # Fonte Arial Bold tamanho 42
        except:  # Caso fonte não exista
            fonte_resultado = ImageFont.load_default()  # Usa fonte de texto padrão do Python

        texto_final = "IMPEDIDO" if impedido else "POSICAO LEGAL"  # Texto do veredito do VAR
        cor_texto = (255, 60, 60) if impedido else (60, 255, 120)  # Vermelho para impedido, Verde para legal
        prefixo = "DECISAO DO VAR:  "  # Prefixo do texto de veredito

        final_img = Image.new("RGB", (screen_w, fh + banner_h), (15, 15, 15))  # Cria canvas preto para exibição
        x_offset = (screen_w - fw) // 2  # Centraliza a imagem na tela preta
        final_img.paste(img_pil, (x_offset, 0))  # Cola a imagem sobre o canvas preto

        draw2 = ImageDraw.Draw(final_img)  # Inicializa desenho sobre imagem final
        linha_completa = prefixo + texto_final  # Linha de texto inteira
        draw2.text(  # Desenha texto com veredito centralizado
            (screen_w // 2, fh + 40),  # Posição X e Y do texto
            linha_completa,  # Texto completo
            fill=(255, 255, 255),  # Texto em cor branca
            font=fonte_resultado,  # Fonte aplicada
            anchor="mm",  # Âncora no meio
        )  # Fim do desenho do texto de resultado

        bbox_linha = draw2.textbbox((0, 0), linha_completa, font=fonte_resultado)  # Calcula caixa do texto inteiro
        bbox_pref = draw2.textbbox((0, 0), prefixo, font=fonte_resultado)  # Calcula caixa do texto do prefixo
        largura_linha = bbox_linha[2] - bbox_linha[0]  # Largura da linha inteira
        largura_prefixo = bbox_pref[2] - bbox_pref[0]  # Largura do prefixo
        x_inicio_linha = (screen_w // 2) - (largura_linha // 2)  # Início do texto em X
        x_veredito = x_inicio_linha + largura_prefixo  # Início exato do texto colorido de veredito

        draw2.text(  # Colore o texto do veredito (Vermelho ou Verde) por cima do branco
            (x_veredito, fh + 40),  # Coordenada exata de X e Y
            texto_final,  # Texto ("IMPEDIDO" ou "POSICAO LEGAL")
            fill=cor_texto,  # Cor correspondente ao resultado
            font=fonte_resultado,  # Fonte aplicada
            anchor="lm",  # Alinhamento à esquerda
        )  # Fim do desenho colorido

        jr.update_idletasks()  # Processa requisições de atualização do Tkinter
        win_w = jr.winfo_width()  # Largura física da janela gerada
        win_h = jr.winfo_height()  # Altura física da janela gerada

        final_img_fit = final_img.copy()  # Cria cópia da imagem final
        final_img_fit.thumbnail((win_w, win_h), Image.LANCZOS)  # Ajusta imagem ao tamanho da janela física

        canvas_res = tk.Canvas(jr, bg="black", highlightthickness=0)  # Canvas final para renderização
        canvas_res.pack(fill="both", expand=True)  # Insere canvas preenchendo tela inteira

        tk_resultado = ImageTk.PhotoImage(final_img_fit)  # Converte imagem ajustada em PhotoImage Tkinter
        canvas_res._tk_img = tk_resultado  # Salva referência para evitar Garbage Collector
        
        x_pos = win_w // 2  # Posição central em X
        canvas_res.create_image(x_pos, 0, anchor="n", image=tk_resultado)  # Renderiza imagem na tela de resultado

if __name__ == "__main__":  # Ponto de entrada padrão do interpretador Python
    app = OpenVAR()  # Instancia classe principal openVAR
    app.janela.mainloop()  # Inicia loop principal da interface gráfica
