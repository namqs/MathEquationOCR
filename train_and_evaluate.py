from data_loader import carregar_bases_de_dados
from model import construir_rede_neural
import matplotlib.pyplot as plt

def treinar_e_avaliar(rede_neural, imagens_treinamento, rotulos_treinamento, imagens_teste, rotulos_teste):
    # Treinar a rede neural
    historico = rede_neural.fit(imagens_treinamento, rotulos_treinamento, epochs=20, validation_data=(imagens_teste, rotulos_teste))

    # Avaliar o desempenho do modelo nos dados de teste
    resultado_teste = rede_neural.evaluate(imagens_teste, rotulos_teste)
    print(f"Acurácia nos dados de teste: {resultado_teste[1] * 100:.2f}%")

    # Salvar o modelo treinado
    rede_neural.save('modelo_treinado.h5')

    # Plotar curvas de aprendizado
    plt.figure(figsize=(12, 4))

    # Plotar a perda durante o treinamento
    plt.subplot(1, 2, 1)
    plt.plot(historico.history['loss'], label='Perda de Treinamento')
    plt.plot(historico.history['val_loss'], label='Perda de Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.title('Curva de Aprendizado - Perda')
    plt.legend()

    # Plotar a acurácia durante o treinamento
    plt.subplot(1, 2, 2)
    plt.plot(historico.history['accuracy'], label='Acurácia de Treinamento')
    plt.plot(historico.history['val_accuracy'], label='Acurácia de Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.title('Curva de Aprendizado - Acurácia')
    plt.legend()

    plt.show()

# Carregar bases de dados
imagens_treinamento, rotulos_treinamento, imagens_teste, rotulos_teste, _ = carregar_bases_de_dados()

# Construir a rede neural
input_shape = (64, 128, 3)  # Ajuste conforme necessário
rede_neural = construir_rede_neural(input_shape)

# Treinar e avaliar a rede neural
treinar_e_avaliar(rede_neural, imagens_treinamento, rotulos_treinamento, imagens_teste, rotulos_teste)
