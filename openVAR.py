import cv2
import customtkinter
from PIL import Image

imagem = None

def gerarJanela():
    janela = customtkinter.CTk()

    janela.title('openVAR')
    janela.configure(fg_color='#0057ae')
    janela.geometry('640x480')

    def getImagem():
        caminho_arquivo = customtkinter.filedialog.askopenfilename(
            parent=janela,
            title="Escolha uma imagem",
            filetypes=[("Image files", "*.jpg *.png *.jpeg")]
        )

        if not caminho_arquivo:
            return

        global imagem
        imagem = cv2.imread(caminho_arquivo)

    def verificar():
        global imagem

        if imagem is None:
            return

        imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(imagem_rgb)

        max_w, max_h = 600, 400
        pil_img.thumbnail((max_w, max_h), Image.LANCZOS)

        img_w, img_h = pil_img.size

        ctk_img = customtkinter.CTkImage(
            light_image=pil_img,
            dark_image=pil_img,
            size=(img_w, img_h)
        )

        janela2 = customtkinter.CTkToplevel(janela)

        janela2.title("Cabine do openVAR")
        janela2.geometry(f"{max(img_w + 40, 400)}x{img_h + 120}")

        label_imagem = customtkinter.CTkLabel(
            janela2,
            image=ctk_img,
            text=''
        )

        label_imagem.pack(pady=(20, 10))

        label_imagem.image = ctk_img

        janela2.grab_set()

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

    label_logo.pack(pady=(80, 20))

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

    botao_imagem.pack(anchor='center', pady=(0, 8))

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