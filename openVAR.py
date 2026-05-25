import cv2
import customtkinter
from PIL import Image
from ultralytics import YOLO

modelo_pose = YOLO('yolov8n-pose.pt')

imagem = None

def gerarJanela():
    janela = customtkinter.CTk()

    janela.title('openVAR')
    janela.configure(fg_color='#0057ae')
    janela.geometry('640x480')
    janela.resizable(False, False)

    def getImagem():
        global imagem
        caminho_arquivo = customtkinter.filedialog.askopenfilename(
            parent=janela,
            title="Escolha uma imagem da partida",
            filetypes=[("Image files", "*.jpg *.png *.jpeg")]
        )

        if not caminho_arquivo:
            return
        
        label_caminho = customtkinter.CTkLabel(
            janela,
            text_color="white", 
            text=f"Arquivo selecionado: {caminho_arquivo.split('/')[-1]}")

        imagem = cv2.imread(caminho_arquivo)

    def verificar():
        global imagem

        if imagem is None:
            print("Nenhuma imagem selecionada!")
            return

        resultados = modelo_pose(imagem)
        
        imagem_processada = resultados[0].plot()

        imagem_rgb = cv2.cvtColor(imagem_processada, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(imagem_rgb)

        max_w, max_h = 1280, 720
        pil_img.thumbnail((max_w, max_h), Image.LANCZOS)

        img_w, img_h = pil_img.size

        ctk_img = customtkinter.CTkImage(
            light_image=pil_img,
            dark_image=pil_img,
            size=(img_w, img_h)
        )

        janela2 = customtkinter.CTkToplevel(janela)
        janela2.title("Cabine do openVAR - Análise de Campo")
        janela2.geometry(f"{img_w + 40}x{img_h + 40}")
        janela2.resizable(False, False)

        label_imagem = customtkinter.CTkLabel(
            janela2,
            image=ctk_img,
            text=''
        )

        label_imagem.pack(pady=(20, 20))

        label_imagem.image = ctk_img
        janela2.update()
        janela2.grab_set()

    try:
        pil_logo = Image.open('openVAR_logo.png')
        ctk_logo = customtkinter.CTkImage(
            light_image=pil_logo,
            dark_image=pil_logo,
            size=(200, 200)
        )

        label_logo = customtkinter.CTkLabel(
            janela,
            image=ctk_logo,
            text=''
        )
        label_logo.pack(pady=(40, 20))
    except FileNotFoundError:
        print("Aviso: Logo 'openVAR_logo.png' não encontrada na pasta.")

    botao_imagem = customtkinter.CTkButton(
        janela,
        text='Adicionar Imagem',
        command=getImagem,
        fg_color='#cae332',
        text_color='#0057ae',
        hover_color='#d4f04a',
        width=300,
        height=60,
        font=('Arial', 22, 'bold')
    )

    botao_imagem.pack(anchor='center', pady=(10, 8))

    botao_verificacao = customtkinter.CTkButton(
        janela,
        text='Iniciar Checagem',
        command=verificar,
        fg_color='#cae332',
        text_color='#0057ae',
        hover_color='#d4f04a',
        width=300,
        height=60,
        font=('Arial', 22, 'bold')
    )

    botao_verificacao.pack(anchor='center')

    janela.mainloop()

gerarJanela()



