import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

def carregar_imagens(diretorio, tamanho=(64, 64)):
    imagens = []
    rotulos = []

    for rotulo, classe in enumerate(os.listdir(diretorio)):
        caminho_classe = os.path.join(diretorio, classe)
        for arquivo in os.listdir(caminho_classe):
            caminho_imagem = os.path.join(caminho_classe, arquivo)

            # Carregar imagem em cores
            imagem = cv2.imread(caminho_imagem)
            if imagem is not None:
                imagem = cv2.resize(imagem, tamanho)
                imagens.append(imagem)
                rotulos.append(rotulo)

    return np.array(imagens), np.array(rotulos)

def carregar_bases_de_dados():
    # Carregar imagens de treinamento
    diretorio_treinamento = r'C:\Users\J. Lucas\.vscode-cli\Desktop\Natalie\CROHME_dataset\Train'
    imagens_treinamento, rotulos_treinamento = carregar_imagens(diretorio_treinamento)

    # Carregar imagens de teste
    diretorio_teste = r'C:\Users\J. Lucas\.vscode-cli\Desktop\Natalie\CROHME_dataset\Test'
    imagens_teste, rotulos_teste = carregar_imagens(diretorio_teste)

    # Normalizar as imagens
    imagens_treinamento = imagens_treinamento / 255.0
    imagens_teste = imagens_teste / 255.0

    return imagens_treinamento, to_categorical(rotulos_treinamento), imagens_teste, to_categorical(rotulos_teste)
