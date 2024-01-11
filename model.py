import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import optimizers


Adam = optimizers.Adam

def construir_rede_neural(input_shape):
    # Carregar classes do data_loader
    from data_loader import carregar_bases_de_dados
    _, rotulos_treinamento, _, _, _ = carregar_bases_de_dados()
    num_classes = len(rotulos_treinamento[0])

    # Construção da rede neural
    rede_neural = Sequential()

    # Adicionar camadas convolucionais e de pooling
    rede_neural.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'))
    rede_neural.add(MaxPooling2D(pool_size=(2, 2)))

    rede_neural.add(Conv2D(32, (3, 3), activation='relu'))
    rede_neural.add(MaxPooling2D(pool_size=(2, 2)))

    # Adicionar camada de achatamento (Flatten)
    rede_neural.add(Flatten())

    # Adicionar camadas densas (fully connected)
    rede_neural.add(Dense(128, activation='relu'))
    rede_neural.add(Dropout(0.5))

    rede_neural.add(Dense(64, activation='relu'))

    # Camada de saída
    rede_neural.add(Dense(num_classes, activation='softmax'))

    # Compilar a rede neural
    otimizador = optimizers.Adam(learning_rate=0.001)
    rede_neural.compile(optimizer=otimizador, loss='categorical_crossentropy', metrics=['accuracy'])

    return rede_neural