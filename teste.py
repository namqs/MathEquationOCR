import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Função de carregamento personalizada para o otimizador
def custom_optimizer():
    return tf.keras.optimizers.Adam()

# Carregar o modelo treinado
modelo = load_model('modelo_treinado.h5', custom_objects={'Adam': custom_optimizer})


import cv2
import numpy as np

# Carregar a imagem de teste
caminho_imagem = 'C:\\Users\\J. Lucas\\Pictures\\Testenatalie.png'  # Substitua pelo caminho real do arquivo
imagem_teste = cv2.imread(caminho_imagem)

# Verificar se a imagem foi carregada corretamente
if imagem_teste is not None:
    print("A imagem foi carregada com sucesso.")
else:
    print("Não foi possível carregar a imagem.")
    exit()

# Redimensionar a imagem
imagem_teste = cv2.resize(imagem_teste, (64, 64))
imagem_teste = np.expand_dims(imagem_teste, axis=0)  # Adicionar dimensão extra para o lote

# Normalizar a imagem
imagem_teste = imagem_teste / 255.0

# Fazer previsões
previsoes = modelo.predict(imagem_teste)
classe_predita = np.argmax(previsoes)

print(f"Classe predita: {classe_predita}")
