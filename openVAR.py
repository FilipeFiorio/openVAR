import os
import cv2
import numpy as np
import tkinter as tk
import customtkinter

from PIL import Image, ImageDraw, ImageFont, ImageTk
from ultralytics import YOLO

# Carrega o modelo YOLOv8 especializado em estimativa de pose humana
modelo_pose = YOLO("yolov8x-pose.pt")

# Variáveis globais para armazenar as imagens e os dados do fluxo
imagem_original = None
imagem_com_pose = None
deteccoes = []

# Conjuntos para armazenar os IDs dos jogadores marcados pelo usuário
time_atacante = set()
time_defensor = set()


def parte_mais_avancada_x(kps, bbox, atacando_para_direita=True):
    """Calcula a coordenada X do ponto mais avançado do corpo do jogador."""
    x1, y1, x2, y2 = bbox
    # Índices dos keypoints: nariz, ombros, quadris, joelhos e tornozelos
    indices = [0, 5, 6, 11, 12, 13, 14, 15, 16]
    xs = []

    # Extrai os pontos com confiança de detecção aceitável (> 0.25)
    for i in indices:
        if i < len(kps):
            x, y, conf = kps[i]
            if conf > 0.25:
                xs.append(x)

    # Retorna o X extremo baseado na direção que o time está atacando
    if xs:
        if atacando_para_direita:
            return max(xs)  # Maior X se ataca para a direita
        else:
            return min(xs)  # Menor X se ataca para a esquerda

    # Fallback caso nenhum keypoint tenha confiança alta: usa a borda da Bounding Box
    if atacando_para_direita:
        return x2
    else:
        return x1


def gerarJanela():
    """Gera e configura a interface da janela principal do openVAR."""
    janela = customtkinter.CTk()
    janela.title("openVAR")
    janela.geometry("680x560")
    janela.configure(fg_color="#0057ae")
    janela.resizable(False, False)

    label_status = customtkinter.CTkLabel(
        janela, text="Carregue uma imagem.", text_color="white", font=("Arial", 14)
    )

    # Variável padrão para armazenar a inclinação da perspectiva (dx/dy)
    dx_per_dy_calib = 0.24

    def getImagem():
        """Abre a caixa de diálogo para o usuário carregar uma foto do computador."""
        global imagem_original, imagem_com_pose, deteccoes, time_atacante, time_defensor

        caminho_arquivo = customtkinter.filedialog.askopenfilename(
            parent=janela,
            title="Escolha uma imagem",
            filetypes=[("Image files", "*.jpg *.png *.jpeg")],
        )

        if not caminho_arquivo:
            return

        # Normaliza o caminho do arquivo e lê a imagem em formato BGR com OpenCV
        caminho_normalizado = os.path.normpath(caminho_arquivo)
        img_bgr = cv2.imread(caminho_normalizado)

        if img_bgr is None:
            label_status.configure(text="Erro ao carregar imagem.")
            return

        # Reseta os dados globais para processar a nova imagem limpa
        imagem_original = img_bgr
        imagem_com_pose = None
        deteccoes = []
        time_atacante = set()
        time_defensor = set()

        label_status.configure(
            text=f"Imagem carregada: {os.path.basename(caminho_normalizado)}"
        )

    def verificar():
        """Roda a detecção de jogadores e esqueletos via IA YOLOv8."""
        global imagem_com_pose, deteccoes

        if imagem_original is None:
            label_status.configure(text="Nenhuma imagem carregada.")
            return

        label_status.configure(text="Detectando jogadores...")
        janela.update()

        # Aplica filtro bilateral para reduzir ruído mantendo as bordas nítidas
        imagem_blur = cv2.bilateralFilter(imagem_original, 5, 50, 50)

        # Dobra o tamanho da imagem para melhorar a precisão em jogadores distantes
        imagem_upscale = cv2.resize(
            imagem_blur, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
        )

        # Executa a inferência de Pose da IA YOLOv8
        resultados = modelo_pose(
            imagem_upscale, conf=0.02, iou=0.25, imgsz=1280, verbose=False
        )

        resultado = resultados[0]
        # Converte a imagem de BGR (OpenCV) para RGB (Pillow/Tkinter)
        imagem_com_pose = cv2.cvtColor(imagem_upscale.copy(), cv2.COLOR_BGR2RGB)

        boxes = resultado.boxes
        kps_all = resultado.keypoints
        deteccoes = []

        # Filtra os objetos detectados para extrair apenas pessoas (classe 0)
        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            if cls != 0:
                continue

            conf_box = float(box.conf[0])
            if conf_box < 0.02:
                continue

            # Converte coordenadas da caixa limitadora (Bounding Box) para inteiros
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            largura = x2 - x1
            altura = y2 - y1

            # Ignora caixas excessivamente pequenas (ruídos de detecção)
            if largura < 25 or altura < 60:
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            kps = []

            # Extrai os keypoints correspondentes àquele jogador específico
            if kps_all is not None and i < len(kps_all.data):
                for kp in kps_all.data[i]:
                    kps.append((float(kp[0]), float(kp[1]), float(kp[2])))
            else:
                kps = [(0, 0, 0)] * 17

            # Calcula o X limite inicial assumindo ataque para a direita
            perna_x = parte_mais_avancada_x(kps, (x1, y1, x2, y2), True)

            # Salva o dicionário com os metadados estruturados do jogador
            deteccoes.append(
                {
                    "id": len(deteccoes),
                    "bbox": (x1, y1, x2, y2),
                    "cx": cx,
                    "cy": cy,
                    "kps": kps,
                    "perna_x": perna_x,
                }
            )

        if len(deteccoes) == 0:
            label_status.configure(text="Nenhum jogador detectado.")
            return

        label_status.configure(text=f"{len(deteccoes)} jogadores detectados.")
        abrir_janela_marcacao()

    def abrir_janela_marcacao():
        """Abre a janela de tela cheia para calibração e seleção manual de times."""
        LABEL_H = 50
        BTN_H = 70

        # Obtém o tamanho da tela do monitor para ajustar o Layout
        screen_w = janela.winfo_screenwidth()
        screen_h = janela.winfo_screenheight()

        area_img_h = screen_h - LABEL_H - BTN_H

        # Calcula o redimensionamento mantendo a proporção original da imagem
        h_orig, w_orig = imagem_com_pose.shape[:2]
        escala = min(screen_w / w_orig, area_img_h / h_orig)
        disp_w = int(w_orig * escala)
        disp_h = int(h_orig * escala)

        # Redimensiona a imagem com filtro de alta qualidade Lanczos
        pil_base = Image.fromarray(imagem_com_pose).resize(
            (disp_w, disp_h), Image.LANCZOS
        )

        # Cria uma janela Toplevel (sobreposta) em modo maximizado
        jm = customtkinter.CTkToplevel(janela)
        jm.title("Marcação")
        jm.state("zoomed")
        jm.grab_set()

        # Lista local para guardar os 2 pontos clicados na calibração
        pontos_calibracao = []

        label_instrucao = customtkinter.CTkLabel(
            jm,
            text="Calibração: Clique com o botão esquerdo em dois pontos do campo que definem a linha vertical correta (ex: a linha de fundo ou a linha da grande área) para calibrar a perspectiva.",
            font=("Arial", 14),
        )
        label_instrucao.pack(pady=6)

        frame_canvas = tk.Frame(jm, bg="black")
        frame_canvas.pack(fill="both", expand=True)

        # Cria o Canvas do Tkinter para receber desenhos vetoriais dinâmicos
        canvas = tk.Canvas(
            frame_canvas,
            width=disp_w,
            height=disp_h,
            bg="black",
            highlightthickness=0,
        )
        canvas.place(relx=0.5, rely=0.5, anchor="center")

        def jogador_no_clique(mx, my):
            """Verifica se a coordenada do clique caiu dentro da caixa de algum jogador."""
            cx_off = canvas.winfo_x()
            cy_off = canvas.winfo_y()
            rx = mx - cx_off
            ry = my - cy_off
            for det in deteccoes:
                x1, y1, x2, y2 = det["bbox"]
                # Converte os limites da caixa para a escala atual do Canvas
                sx1 = int(x1 * escala)
                sy1 = int(y1 * escala)
                sx2 = int(x2 * escala)
                sy2 = int(y2 * escala)
                if sx1 <= rx <= sx2 and sy1 <= ry <= sy2:
                    return det["id"]
            return None

        def redesenhar():
            """Atualiza os elementos gráficos (caixas, linhas, cliques) no Canvas."""
            canvas.delete("all")
            img_draw = pil_base.copy().convert("RGBA")
            overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            # Desenha as caixas dos jogadores com cores baseadas nos times
            for det in deteccoes:
                x1, y1, x2, y2 = det["bbox"]
                sx1 = int(x1 * escala)
                sy1 = int(y1 * escala)
                sx2 = int(x2 * escala)
                sy2 = int(y2 * escala)
                did = det["id"]

                if did in time_atacante:
                    cor_fill = (255, 220, 0, 110)  # Amarelo translúcido para ATK
                    cor_borda = (255, 220, 0, 255)
                elif did in time_defensor:
                    cor_fill = (0, 120, 255, 110)  # Azul translúcido para DEF
                    cor_borda = (0, 120, 255, 255)
                else:
                    cor_fill = (180, 180, 180, 50)  # Cinza para não selecionados
                    cor_borda = (180, 180, 180, 180)

                draw.rectangle(
                    [sx1, sy1, sx2, sy2], fill=cor_fill, outline=cor_borda, width=3
                )
                draw.text(
                    (sx1 + 5, sy1 + 2), str(did + 1), fill=(255, 255, 255)
                )

                # Desenha a linha amarela individual indicando 
                # o ponto avançado daquele esqueleto
                perna_x = int(det["perna_x"] * escala)
                draw.line(
                    [(perna_x, sy1), (perna_x, sy2)],
                    fill=(255, 0, 0, 255),
                    width=2,
                )

            # Desenha círculos amarelos nos pontos de calibração clicados
            for pt in pontos_calibracao:
                cx_pt, cy_pt = pt
                draw.ellipse(
                    [cx_pt - 5, cy_pt - 5, cx_pt + 5, cy_pt + 5],
                    fill=(255, 255, 0, 255),
                )

            # Conecta os dois pontos com uma linha ciano para conferência visual
            if len(pontos_calibracao) == 2:
                c1_x, c1_y = pontos_calibracao[0]
                c2_x, c2_y = pontos_calibracao[1]
                draw.line(
                    [(c1_x, c1_y), (c2_x, c2_y)],
                    fill=(0, 255, 255, 255),
                    width=2,
                )

            # Renderiza o frame composto final de volta no Tkinter Canvas
            img_final = Image.alpha_composite(img_draw, overlay).convert("RGB")
            tk_img = ImageTk.PhotoImage(img_final)
            canvas._tk_img = tk_img
            canvas.create_image(0, 0, anchor="nw", image=tk_img)

        def clique_esquerdo(event):
            """Gerencia calibração (dois primeiros cliques) ou marcação de atacantes."""
            nonlocal dx_per_dy_calib

            cx_off = canvas.winfo_x()
            cy_off = canvas.winfo_y()
            mx = event.x_root - frame_canvas.winfo_rootx()
            my = event.y_root - frame_canvas.winfo_rooty()
            rx = mx - cx_off
            ry = my - cy_off

            # Fase de calibração ativa se houver menos de dois cliques salvos
            if len(pontos_calibracao) < 2:
                pontos_calibracao.append((rx, ry))
                redesenhar()

                if len(pontos_calibracao) == 1:
                    label_instrucao.configure(
                        text="Calibração: Clique no segundo ponto para finalizar."
                    )
                elif len(pontos_calibracao) == 2:
                    c1_x, c1_y = pontos_calibracao[0]
                    c2_x, c2_y = pontos_calibracao[1]

                    # Calcula a taxa de variação dx/dy baseada na reta humana criada
                    if abs(c2_y - c1_y) > 1e-3:
                        dx_per_dy_calib = (c2_x - c1_x) / (c2_y - c1_y)
                    else:
                        dx_per_dy_calib = 0.0

                    label_instrucao.configure(
                        text="Calibração Concluída! Clique esquerdo = atacante | clique direito = defensor"
                    )
                return

            # Fase de marcação ativa após calibração concluída
            did = jogador_no_clique(mx, my)
            if did is None:
                return
            time_atacante.add(did)
            time_defensor.discard(did)  # Garante que não está em dois times
            redesenhar()

        def clique_direito(event):
            """Adiciona o jogador clicado ao grupo de defensores."""
            if len(pontos_calibracao) < 2:
                return  # Bloqueia cliques secundários durante a calibração
            did = jogador_no_clique(
                event.x_root - frame_canvas.winfo_rootx(),
                event.y_root - frame_canvas.winfo_rooty(),
            )
            if did is None:
                return
            time_defensor.add(did)
            time_atacante.discard(did)
            redesenhar()

        # Vincula os eventos físicos do mouse às funções de processamento
        canvas.bind("<Button-1>", clique_esquerdo)
        canvas.bind("<Button-3>", clique_direito)

        customtkinter.CTkButton(
            jm,
            text="Analisar Impedimento",
            command=lambda: analisar_impedimento(jm),
            width=280,
            height=50,
            fg_color="#cae332",
            text_color="#0057ae",
            hover_color="#d4f04a",
            font=("Arial", 18, "bold"),
        ).pack(pady=8)

        redesenhar()

    def analisar_impedimento(janela_marcacao):
        """Calcula o veredito final do VAR e desenha a projeção tridimensional."""
        if len(time_atacante) == 0 or len(time_defensor) == 0:
            label_status.configure(text="Marque atacantes e defensores.")
            return

        xs_atk_temp = [deteccoes[i]["perna_x"] for i in time_atacante]
        xs_def_temp = [deteccoes[i]["perna_x"] for i in time_defensor]

        # Descobre de forma inteligente o lado do ataque comparando as médias posicionais
        media_atk = np.mean(xs_atk_temp)
        media_def = np.mean(xs_def_temp)
        atacando_para_direita = media_atk > media_def

        # Força o esqueleto a recalcular seu limite X com o lado correto estabelecido
        for det in deteccoes:
            det["perna_x"] = parte_mais_avancada_x(
                det["kps"], det["bbox"], atacando_para_direita
            )

        xs_atk = [deteccoes[i]["perna_x"] for i in time_atacante]
        xs_def = [deteccoes[i]["perna_x"] for i in time_defensor]

        # Encontra a posição da linha física e do atacante crítico 
        # conforme a direção do ataque
        if atacando_para_direita:
            linha_impedimento = max(xs_def)
            atacante_mais_avancado = max(xs_atk)
            impedido = atacante_mais_avancado > linha_impedimento
        else:
            linha_impedimento = min(xs_def)
            atacante_mais_avancado = min(xs_atk)
            impedido = atacante_mais_avancado < linha_impedimento

        img_pil = Image.fromarray(imagem_com_pose.copy())
        draw = ImageDraw.Draw(img_pil)
        h, w = imagem_com_pose.shape[:2]

        cor_linha = (255, 0, 0) if impedido else (0, 255, 0)

        # Encontra qual defensor específico estabeleceu a linha de impedimento
        if atacando_para_direita:
            ultimo_defensor_id = max(
                time_defensor, key=lambda i: deteccoes[i]["perna_x"]
            )
        else:
            ultimo_defensor_id = min(
                time_defensor, key=lambda i: deteccoes[i]["perna_x"]
            )

        x1, y1, x2, y2 = deteccoes[ultimo_defensor_id]["bbox"]
        x_ref = int(linha_impedimento)
        y_ref = int(y2)  # Usa o Y da base (pé) como âncora de projeção

        # Projeção com a Perspectiva Calibrada Manualmente
        x_topo = int(x_ref - y_ref * dx_per_dy_calib)
        x_base = int(x_ref + (h - y_ref) * dx_per_dy_calib)

        label_status.configure(
            text=f"Sucesso: Linha de impedimento desenhada usando calibração manual (dx_per_dy = {dx_per_dy_calib:.4f})."
        )
        draw.line([(x_topo, 0), (x_base, h)], fill=cor_linha, width=6)

        # Desenha os rótulos de texto e Bounding Boxes dos jogadores analisados
        for det in deteccoes:
            did = det["id"]
            if did not in time_atacante and did not in time_defensor:
                continue

            x1, y1, x2, y2 = det["bbox"]

            if did in time_atacante:
                cor = (255, 220, 0)
                texto = f"ATK {did+1}"
            else:
                cor = (0, 120, 255)
                texto = f"DEF {did+1}"

            draw.rectangle([x1, y1, x2, y2], outline=cor, width=4)
            draw.text((x1 + 4, y1 - 20), texto, fill=cor)

            # Desenha uma linha guia amarela vertical sobre 
            # a parte do corpo mapeada daquele atleta
            px = int(det["perna_x"])
            draw.line([(px, y1), (px, y2)], fill=(255, 255, 0), width=3)

        # Configura as proporções da tela final do relatório gráfico
        screen_w = janela.winfo_screenwidth()
        screen_h = janela.winfo_screenheight()
        banner_h = 80

        img_pil.thumbnail((screen_w, screen_h - banner_h), Image.LANCZOS)
        fw, fh = img_pil.size

        try:
            fonte_resultado = ImageFont.truetype("arialbd.ttf", 42)
        except:
            fonte_resultado = ImageFont.load_default()

        texto_final = "IMPEDIDO" if impedido else "POSICAO LEGAL"
        cor_texto = (255, 60, 60) if impedido else (60, 255, 120)
        prefixo = "DECISAO DO VAR:  "

        # Gera o banner escuro inferior para exibir o veredito textual centralizado
        final_img = Image.new("RGB", (screen_w, fh + banner_h), (15, 15, 15))
        x_offset = (screen_w - fw) // 2
        final_img.paste(img_pil, (x_offset, 0))

        draw2 = ImageDraw.Draw(final_img)
        linha_completa = prefixo + texto_final
        draw2.text(
            (screen_w // 2, fh + 40),
            linha_completa,
            fill=(255, 255, 255),
            font=fonte_resultado,
            anchor="mm",
        )

        # Mede as caixas de texto com precisão para colorir
        # a palavra do veredito separadamente
        bbox_linha = draw2.textbbox((0, 0), linha_completa, font=fonte_resultado)
        bbox_pref = draw2.textbbox((0, 0), prefixo, font=fonte_resultado)
        largura_linha = bbox_linha[2] - bbox_linha[0]
        largura_prefixo = bbox_pref[2] - bbox_pref[0]
        x_inicio_linha = (screen_w // 2) - (largura_linha // 2)
        x_veredito = x_inicio_linha + largura_prefixo

        draw2.text(
            (x_veredito, fh + 40),
            texto_final,
            fill=cor_texto,
            font=fonte_resultado,
            anchor="lm",
        )

        janela_marcacao.destroy()

        # Monta e inicializa a tela cheia de encerramento contendo o veredito oficial
        # do openVAR
        jr = customtkinter.CTkToplevel(janela)
        jr.title("Resultado")
        jr.state("zoomed")
        jr.resizable(True, True)
        jr.grab_set()

        jr.update_idletasks()
        win_w = jr.winfo_width()
        win_h = jr.winfo_height()

        final_img_fit = final_img.copy()
        final_img_fit.thumbnail((win_w, win_h), Image.LANCZOS)

        canvas_res = tk.Canvas(jr, bg="black", highlightthickness=0)
        canvas_res.pack(fill="both", expand=True)

        tk_resultado = ImageTk.PhotoImage(final_img_fit)
        canvas_res._tk_img = tk_resultado

        x_pos = win_w // 2
        canvas_res.create_image(x_pos, 0, anchor="n", image=tk_resultado)

    # Renderização estrutural da Tela Principal (Menu Inicial)
    try:
        pil_logo = Image.open("openVAR_logo.png")
        ctk_logo = customtkinter.CTkImage(
            light_image=pil_logo, dark_image=pil_logo, size=(220, 220)
        )
        customtkinter.CTkLabel(janela, image=ctk_logo, text="").pack(
            pady=(25, 10)
        )
    except:
        pass

    customtkinter.CTkButton(
        janela,
        text="Adicionar Imagem",
        command=getImagem,
        width=320,
        height=60,
        fg_color="#cae332",
        text_color="#0057ae",
        hover_color="#d4f04a",
        font=("Arial", 22, "bold"),
    ).pack(pady=10)

    customtkinter.CTkButton(
        janela,
        text="Iniciar Checagem",
        command=verificar,
        width=320,
        height=60,
        fg_color="#cae332",
        text_color="#0057ae",
        hover_color="#d4f04a",
        font=("Arial", 22, "bold"),
    ).pack()

    label_status.pack(pady=15)
    janela.mainloop()


# Inicializa a execução do ecossistema do aplicativo openVAR
gerarJanela()