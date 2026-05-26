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

    # tornozelos
    indices = [15, 16]

    xs = []

    for i in indices:

        if i < len(kps):

            x, y, conf = kps[i]

            if conf > 0.15:
                xs.append(x)

    # se encontrou tornozelo
    if xs:

        if atacando_para_direita:
            return max(xs)
        else:
            return min(xs)

    # fallback bounding box
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

        global imagem_original
        global imagem_com_pose
        global deteccoes
        global time_atacante
        global time_defensor

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

            label_status.configure(
                text="Erro ao carregar imagem."
            )

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

        global imagem_com_pose
        global deteccoes

        if imagem_original is None:

            label_status.configure(
                text="Nenhuma imagem carregada."
            )

            return

        label_status.configure(
            text="Detectando jogadores..."
        )

        janela.update()

        imagem_blur = cv2.bilateralFilter(
            imagem_original,
            5,
            50,
            50
        )

        imagem_upscale = cv2.resize(
            imagem_blur,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

        resultados = modelo_pose(
            imagem_upscale,
            conf=0.02,
            iou=0.25,
            imgsz=1280,
            verbose=False
        )

        resultado = resultados[0]

        # imagem base
        imagem_com_pose = cv2.cvtColor(
            imagem_upscale.copy(),
            cv2.COLOR_BGR2RGB
        )

        boxes = resultado.boxes

        kps_all = resultado.keypoints

        deteccoes = []


        for i, box in enumerate(boxes):

            cls = int(box.cls[0])

            # somente pessoas
            if cls != 0:
                continue

            conf_box = float(box.conf[0])

            if conf_box < 0.02:
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # ignora coisas pequenas demais
            largura = x2 - x1
            altura = y2 - y1

            if largura < 25 or altura < 60:
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            kps = []

            if kps_all is not None and i < len(kps_all.data):

                for kp in kps_all.data[i]:

                    kps.append((
                        float(kp[0]),
                        float(kp[1]),
                        float(kp[2])
                    ))

            else:

                kps = [(0, 0, 0)] * 17

            perna_x = x2
            deteccoes.append({
                "id": len(deteccoes),
                "bbox": (x1, y1, x2, y2),
                "cx": cx,
                "cy": cy,
                "kps": kps,
                "perna_x": perna_x
            })

        if len(deteccoes) == 0:

            label_status.configure(
                text="Nenhum jogador detectado."
            )

            return

        label_status.configure(
            text=f"{len(deteccoes)} jogadores detectados."
        )

        abrir_janela_marcacao()


    def abrir_janela_marcacao():

        MAX_W = 1000
        MAX_H = 520

        h_orig, w_orig = imagem_com_pose.shape[:2]

        escala = min(
            MAX_W / w_orig,
            MAX_H / h_orig,
            1.0
        )

        disp_w = int(w_orig * escala)
        disp_h = int(h_orig * escala)

        pil_base = Image.fromarray(imagem_com_pose).resize(
            (disp_w, disp_h),
            Image.LANCZOS
        )

        jm = customtkinter.CTkToplevel(janela)

        jm.title("Marcação")

        jm.geometry(f"{disp_w + 20}x{disp_h + 180}")

        jm.grab_set()


        customtkinter.CTkLabel(
            jm,
            text="Clique esquerdo = atacante | clique direito = defensor",
            font=("Arial", 14)
        ).pack(pady=6)


        canvas = tk.Canvas(
            jm,
            width=disp_w,
            height=disp_h,
            bg="black",
            highlightthickness=0
        )

        canvas.pack()

        def jogador_no_clique(mx, my):

            for det in deteccoes:

                x1, y1, x2, y2 = det["bbox"]

                sx1 = int(x1 * escala)
                sy1 = int(y1 * escala)

                sx2 = int(x2 * escala)
                sy2 = int(y2 * escala)

                if sx1 <= mx <= sx2 and sy1 <= my <= sy2:
                    return det["id"]

            return None


        def redesenhar():

            canvas.delete("all")

            img_draw = pil_base.copy().convert("RGBA")

            overlay = Image.new(
                "RGBA",
                img_draw.size,
                (0, 0, 0, 0)
            )

            draw = ImageDraw.Draw(overlay)


            for det in deteccoes:

                x1, y1, x2, y2 = det["bbox"]

                sx1 = int(x1 * escala)
                sy1 = int(y1 * escala)

                sx2 = int(x2 * escala)
                sy2 = int(y2 * escala)

                did = det["id"]

                # atacante
                if did in time_atacante:

                    cor_fill = (255, 220, 0, 110)

                    cor_borda = (255, 220, 0, 255)

                # defensor
                elif did in time_defensor:

                    cor_fill = (0, 120, 255, 110)

                    cor_borda = (0, 120, 255, 255)

                else:

                    cor_fill = (180, 180, 180, 50)

                    cor_borda = (180, 180, 180, 180)

                # box
                draw.rectangle(
                    [sx1, sy1, sx2, sy2],
                    fill=cor_fill,
                    outline=cor_borda,
                    width=3
                )

                # numero
                draw.text(
                    (sx1 + 5, sy1 + 2),
                    str(did + 1),
                    fill=(255, 255, 255)
                )

                # linha da perna
                perna_x = int(det["perna_x"] * escala)

                draw.line(
                    [(perna_x, sy1), (perna_x, sy2)],
                    fill=(255, 0, 0, 255),
                    width=2
                )

            img_final = Image.alpha_composite(
                img_draw,
                overlay
            ).convert("RGB")

            tk_img = ImageTk.PhotoImage(img_final)

            canvas._tk_img = tk_img

            canvas.create_image(
                0,
                0,
                anchor="nw",
                image=tk_img
            )


        def clique_esquerdo(event):

            did = jogador_no_clique(event.x, event.y)

            if did is None:
                return

            time_atacante.add(did)

            time_defensor.discard(did)

            redesenhar()

        def clique_direito(event):

            did = jogador_no_clique(event.x, event.y)

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

            label_status.configure(
                text="Marque atacantes e defensores."
            )

            return

        xs_atk = [
            deteccoes[i]["perna_x"]
            for i in time_atacante
        ]

        xs_def = [
            deteccoes[i]["perna_x"]
            for i in time_defensor
        ]

        media_atk = np.mean(xs_atk)

        media_def = np.mean(xs_def)

        atacando_para_direita = media_atk > media_def


        if atacando_para_direita:

    # último defensor = mais próximo do gol
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

        # linha de impedimento
        cor_linha = (255, 0, 0) if impedido else (0, 255, 0)

        x_linha = int(linha_impedimento)

        draw.line(
            [
                (x_linha, 0),
                (x_linha, h)
            ],
            fill=cor_linha,
            width=6
        )


        for det in deteccoes:

            did = det["id"]

            if did not in time_atacante and did not in time_defensor:
                continue

            x1, y1, x2, y2 = det["bbox"]

            # atacante
            if did in time_atacante:

                cor = (255, 220, 0)

                texto = f"ATK {did+1}"

            else:

                cor = (0, 120, 255)

                texto = f"DEF {did+1}"

            draw.rectangle(
                [x1, y1, x2, y2],
                outline=cor,
                width=4
            )

            draw.text(
                (x1 + 4, y1 - 20),
                texto,
                fill=cor
            )

            # linha da perna
            px = int(det["perna_x"])

            draw.line(
                [(px, y1), (px, y2)],
                fill=(255, 255, 0),
                width=3
            )

        try:
            fonte = ImageFont.truetype(
                "arial.ttf",
                45
            )

        except:
            fonte = ImageFont.load_default()

        texto_final = (
            "IMPEDIDO"
            if impedido
            else
            "POSICAO LEGAL"
        )

        cor_texto = (
            (255, 60, 60)
            if impedido
            else
            (60, 255, 120)
        )

        banner_h = 90

        overlay = Image.new(
            "RGBA",
            (w,banner_h),
            (0,0,0,220)
        )
        img_rgba = img_pil.convert("RGBA")

        img_rgba.paste(
            overlay,
            (0,h-banner_h),
            overlay
        )
        draw.text(
            (w//2,h-45),
            texto_final,
            fill=cor_texto,
            font=fonte,
            anchor="mm" 
        )
        img_pil = img_rgba.convert("RGB")


        img_pil.thumbnail(
            (1300, 800),
            Image.LANCZOS
        )

        fw, fh = img_pil.size

        janela_marcacao.destroy()

        jr = customtkinter.CTkToplevel(janela)

        jr.title("Resultado")

        jr.geometry(f"{fw + 20}x{fh + 80}")

        jr.grab_set()

        ctk_img = customtkinter.CTkImage(
            light_image=img_pil,
            dark_image=img_pil,
            size=(fw, fh)
        )

        lbl = customtkinter.CTkLabel(
            jr,
            image=ctk_img,
            text=""
        )

        lbl.image = ctk_img

        lbl.pack(pady=10)


    try:

        pil_logo = Image.open("openVAR_logo.png")

        ctk_logo = customtkinter.CTkImage(
            light_image=pil_logo,
            dark_image=pil_logo,
            size=(220, 220)
        )

        customtkinter.CTkLabel(
            janela,
            image=ctk_logo,
            text=""
        ).pack(pady=(25, 10))

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
        font=("Arial", 22, "bold")
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
        font=("Arial", 22, "bold")
    ).pack()

    label_status.pack(pady=15)

    janela.mainloop()


gerarJanela()