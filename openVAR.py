import cv2
import numpy as np
from tkinter import *
from tkinter import ttk

def criarInterface():
    janela = Tk()
    janela.title("openVAR")
    janela.geometry("800x600")
    janela.configure(bg="#dddddd")

    botao = Button(janela, text="Checar Impedimento", command=lambda : print("clicou"))
    botao.pack()

    estilo = ttk.Style()
    estilo.theme_use("clam")

    janela.mainloop()

criarInterface()
