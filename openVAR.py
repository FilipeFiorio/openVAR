import os
import cv2
import numpy as np
import tkinter as tk
import customtkinter

from PIL import Image, ImageDraw, ImageFont, ImageTk
from ultralytics import YOLO


modelo_pose = YOLO("yolov8x-pose.pt")

imagem_original = None
imagem_com_pose = None

deteccoes = []

time_atacante = set()
time_defensor = set()


def perna_mais_avancada_x(kps, bbox, atacando_para_direita=True):

    x1, y1, x2, y2 = bbox
    indices = [15, 16]
    xs = []

    for i in indices:
        if i < len(kps):
            x, y, conf = kps[i]
            if conf > 0.15:
                xs.append(x)

    if xs:
        if atacando_para_direita:
            return max(xs)
        else:
            return min(xs)

    if atacando_para_direita:
        return x2
    else:
        return x1


def gerarJanela():

    janela = customtkinter.CTk()
    janela.title("openVAR")
    janela.geometry("680x560")
    janela.configure(fg_color="#0057ae")
    janela.resizable(False, False)

    label_status = customtkinter.CTkLabel(
        janela,
        text="Carregue uma imagem.",
        text_color="white",
        font=("Arial", 14)
    )

    def getImagem():

        global imagem_original, imagem_com_pose, deteccoes, time_atacante, time_defensor

        caminho_arquivo = customtkinter.filedialog.askopenfilename(
            parent=janela,
            title="Escolha uma imagem",
            filetypes=[("Image files", "*.jpg *.png *.jpeg")]
        )

        if not caminho_arquivo:
            return

        caminho_normalizado = os.path.normpath(caminho_arquivo)
        img_bgr = cv2.imread(caminho_normalizado)

        if img_bgr is None:
            label_status.configure(text="Erro ao carregar imagem.")
            return

        imagem_original = img_bgr
        imagem_com_pose = None
        deteccoes = []
        time_atacante = set()
        time_defensor = set()

        label_status.configure(
            text=f"Imagem carregada: {os.path.basename(caminho_normalizado)}"
        )

    def verificar():

        global imagem_com_pose, deteccoes

        if imagem_original is None:
            label_status.configure(text="Nenhuma imagem carregada.")
            return

        label_status.configure(text="Detectando jogadores...")
        janela.update()

        imagem_blur = cv2.bilateralFilter(imagem_original, 5, 50, 50)

        imagem_upscale = cv2.resize(
            imagem_blur, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
        )

        resultados = modelo_pose(
            imagem_upscale, conf=0.02, iou=0.25, imgsz=1280, verbose=False
        )

        resultado = resultados[0]
        imagem_com_pose = cv2.cvtColor(imagem_upscale.copy(), cv2.COLOR_BGR2RGB)

        boxes = resultado.boxes
        kps_all = resultado.keypoints
        deteccoes = []

        for i, box in enumerate(boxes):

            cls = int(box.cls[0])
            if cls != 0:
                continue

            conf_box = float(box.conf[0])
            if conf_box < 0.02:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            largura = x2 - x1
            altura = y2 - y1

            if largura < 25 or altura < 60:
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            kps = []

            if kps_all is not None and i < len(kps_all.data):
                for kp in kps_all.data[i]:
                    kps.append((float(kp[0]), float(kp[1]), float(kp[2])))
            else:
                kps = [(0, 0, 0)] * 17

            perna_x = perna_mais_avancada_x(kps, (x1, y1, x2, y2), True)

            deteccoes.append({
                "id": len(deteccoes),
                "bbox": (x1, y1, x2, y2),
                "cx": cx,
                "cy": cy,
                "kps": kps,
                "perna_x": perna_x
            })

        if len(deteccoes) == 0:
            label_status.configure(text="Nenhum jogador detectado.")
            return

        label_status.configure(text=f"{len(deteccoes)} jogadores detectados.")
        abrir_janela_marcacao()

    def abrir_janela_marcacao():

        LABEL_H = 50
        BTN_H   = 70

        # ── Tela cheia: pega resolução real da tela ──
        screen_w = janela.winfo_screenwidth()
        screen_h = janela.winfo_screenheight()

        area_img_h = screen_h - LABEL_H - BTN_H

        h_orig, w_orig = imagem_com_pose.shape[:2]
        escala = min(screen_w / w_orig, area_img_h / h_orig)
        disp_w = int(w_orig * escala)
        disp_h = int(h_orig * escala)

        pil_base = Image.fromarray(imagem_com_pose).resize(
            (disp_w, disp_h), Image.LANCZOS
        )

        jm = customtkinter.CTkToplevel(janela)
        jm.title("Marcação")
        jm.state("zoomed")        # tela cheia no Windows
        jm.grab_set()

        customtkinter.CTkLabel(
            jm,
            text="Clique esquerdo = atacante | clique direito = defensor",
            font=("Arial", 14)
        ).pack(pady=6)

        # Frame central para centralizar o canvas
        frame_canvas = tk.Frame(jm, bg="black")
        frame_canvas.pack(fill="both", expand=True)

        canvas = tk.Canvas(
            frame_canvas, width=disp_w, height=disp_h, bg="black", highlightthickness=0
        )
        canvas.place(relx=0.5, rely=0.5, anchor="center")

        def jogador_no_clique(mx, my):
            # Ajusta offset do canvas dentro do frame
            cx_off = canvas.winfo_x()
            cy_off = canvas.winfo_y()
            rx = mx - cx_off
            ry = my - cy_off
            for det in deteccoes:
                x1, y1, x2, y2 = det["bbox"]
                sx1 = int(x1 * escala)
                sy1 = int(y1 * escala)
                sx2 = int(x2 * escala)
                sy2 = int(y2 * escala)
                if sx1 <= rx <= sx2 and sy1 <= ry <= sy2:
                    return det["id"]
            return None

        def redesenhar():

            canvas.delete("all")
            img_draw = pil_base.copy().convert("RGBA")
            overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            for det in deteccoes:
                x1, y1, x2, y2 = det["bbox"]
                sx1 = int(x1 * escala)
                sy1 = int(y1 * escala)
                sx2 = int(x2 * escala)
                sy2 = int(y2 * escala)
                did = det["id"]

                if did in time_atacante:
                    cor_fill = (255, 220, 0, 110)
                    cor_borda = (255, 220, 0, 255)
                elif did in time_defensor:
                    cor_fill = (0, 120, 255, 110)
                    cor_borda = (0, 120, 255, 255)
                else:
                    cor_fill = (180, 180, 180, 50)
                    cor_borda = (180, 180, 180, 180)

                draw.rectangle([sx1, sy1, sx2, sy2], fill=cor_fill, outline=cor_borda, width=3)
                draw.text((sx1 + 5, sy1 + 2), str(did + 1), fill=(255, 255, 255))

                perna_x = int(det["perna_x"] * escala)
                draw.line([(perna_x, sy1), (perna_x, sy2)], fill=(255, 0, 0, 255), width=2)

            img_final = Image.alpha_composite(img_draw, overlay).convert("RGB")
            tk_img = ImageTk.PhotoImage(img_final)
            canvas._tk_img = tk_img
            canvas.create_image(0, 0, anchor="nw", image=tk_img)

        def clique_esquerdo(event):
            did = jogador_no_clique(event.x_root - frame_canvas.winfo_rootx(),
                                    event.y_root - frame_canvas.winfo_rooty())
            if did is None:
                return
            time_atacante.add(did)
            time_defensor.discard(did)
            redesenhar()

        def clique_direito(event):
            did = jogador_no_clique(event.x_root - frame_canvas.winfo_rootx(),
                                    event.y_root - frame_canvas.winfo_rooty())
            if did is None:
                return
            time_defensor.add(did)
            time_atacante.discard(did)
            redesenhar()

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
            font=("Arial", 18, "bold")
        ).pack(pady=8)

        redesenhar()

    def analisar_impedimento(janela_marcacao):

        if len(time_atacante) == 0 or len(time_defensor) == 0:
            label_status.configure(text="Marque atacantes e defensores.")
            return

        xs_atk = [deteccoes[i]["perna_x"] for i in time_atacante]
        xs_def = [deteccoes[i]["perna_x"] for i in time_defensor]

        media_atk = np.mean(xs_atk)
        media_def = np.mean(xs_def)
        atacando_para_direita = media_atk > media_def

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

        if atacando_para_direita:
            ultimo_defensor_id = max(time_defensor, key=lambda i: deteccoes[i]["perna_x"])
        else:
            ultimo_defensor_id = min(time_defensor, key=lambda i: deteccoes[i]["perna_x"])

        x1, y1, x2, y2 = deteccoes[ultimo_defensor_id]["bbox"]
        x_ref = int(linha_impedimento)
        y_ref = int(y2)
        inclinacao = -0.24

        x_topo = int(x_ref + (y_ref * inclinacao))
        x_base = int(x_ref - ((h - y_ref) * inclinacao))

        draw.line([(x_topo, 0), (x_base, h)], fill=cor_linha, width=6)

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

            px = int(det["perna_x"])
            draw.line([(px, y1), (px, y2)], fill=(255, 255, 0), width=3)

        # ── Reduz foto para caber na tela deixando espaço pro banner ──
        screen_w = janela.winfo_screenwidth()
        screen_h = janela.winfo_screenheight()
        banner_h = 80

        img_pil.thumbnail((screen_w, screen_h - banner_h), Image.LANCZOS)
        fw, fh = img_pil.size

        # ── Fonte ──
        try:
            fonte_resultado = ImageFont.truetype("arialbd.ttf", 42)
        except:
            fonte_resultado = ImageFont.load_default()

        texto_final = "IMPEDIDO" if impedido else "POSICAO LEGAL"
        cor_texto   = (255, 60, 60) if impedido else (60, 255, 120)
        prefixo     = "DECISAO DO VAR:  "

        # ── Imagem final: foto + banner, largura = tela cheia ──
        final_img = Image.new("RGB", (screen_w, fh + banner_h), (15, 15, 15))
        # Centraliza a foto horizontalmente
        x_offset = (screen_w - fw) // 2
        final_img.paste(img_pil, (x_offset, 0))

        draw2 = ImageDraw.Draw(final_img)

        linha_completa = prefixo + texto_final
        draw2.text((screen_w // 2, fh + 40), linha_completa,
                   fill=(255, 255, 255), font=fonte_resultado, anchor="mm")

        bbox_linha  = draw2.textbbox((0, 0), linha_completa, font=fonte_resultado)
        bbox_pref   = draw2.textbbox((0, 0), prefixo, font=fonte_resultado)
        largura_linha   = bbox_linha[2]  - bbox_linha[0]
        largura_prefixo = bbox_pref[2]   - bbox_pref[0]
        x_inicio_linha  = (screen_w // 2) - (largura_linha // 2)
        x_veredito      = x_inicio_linha + largura_prefixo

        draw2.text((x_veredito, fh + 40), texto_final,
                   fill=cor_texto, font=fonte_resultado, anchor="lm")

        total_w, total_h = final_img.size

        janela_marcacao.destroy()

        # ── Janela de resultado em tela cheia ──
        # ── Janela de resultado em tela cheia ──
        jr = customtkinter.CTkToplevel(janela)
        jr.title("Resultado")
        jr.state("zoomed")
        jr.resizable(True, True)
        jr.grab_set()

        jr.update_idletasks()
        win_w = jr.winfo_width()
        win_h = jr.winfo_height()

        # Redimensiona a imagem final para caber exatamente na janela
        final_img_fit = final_img.copy()
        final_img_fit.thumbnail((win_w, win_h), Image.LANCZOS)
        fit_w, fit_h = final_img_fit.size

        canvas_res = tk.Canvas(jr, bg="black", highlightthickness=0)
        canvas_res.pack(fill="both", expand=True)

        tk_resultado = ImageTk.PhotoImage(final_img_fit)
        canvas_res._tk_img = tk_resultado

        # Ancora no topo-centro para banner nunca sair da tela
        x_pos = win_w // 2
        canvas_res.create_image(x_pos, 0, anchor="n", image=tk_resultado)

    # ── Tela principal ──
    try:
        pil_logo = Image.open("openVAR_logo.png")
        ctk_logo = customtkinter.CTkImage(
            light_image=pil_logo, dark_image=pil_logo, size=(220, 220)
        )
        customtkinter.CTkLabel(janela, image=ctk_logo, text="").pack(pady=(25, 10))
    except:
        pass

    customtkinter.CTkButton(
        janela,
        text="Adicionar Imagem",
        command=getImagem,
        width=320, height=60,
        fg_color="#cae332", text_color="#0057ae", hover_color="#d4f04a",
        font=("Arial", 22, "bold")
    ).pack(pady=10)

    customtkinter.CTkButton(
        janela,
        text="Iniciar Checagem",
        command=verificar,
        width=320, height=60,
        fg_color="#cae332", text_color="#0057ae", hover_color="#d4f04a",
        font=("Arial", 22, "bold")
    ).pack()

    label_status.pack(pady=15)
    janela.mainloop()


gerarJanela()