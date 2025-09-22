import os
# Configurações para evitar segmentation fault
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
import joblib

# Configurar TensorFlow para usar apenas CPU e evitar problemas de threading
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

print("TensorFlow version:", tf.__version__)
print("GPU disponível:", tf.config.list_physical_devices('GPU'))

# Carregar dados
print("Carregando dados...")
df = pd.read_csv("dataset_carga_milimetros.csv")

# Pré-processamento
X = df['distancia_mm'].values.reshape(-1, 1)
y = LabelEncoder().fit_transform(df['estado'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Dados processados: X shape {X_scaled.shape}, y shape {y.shape}")
print(f"Classes: {np.unique(y)}")

# Modelo MLP
print("\nCriando modelo...")
model = Sequential([
    Dense(8, activation='relu', input_shape=(1,)),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: CHEIO, PARCIAL, VAZIO
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Modelo criado com sucesso!")
print("Iniciando treinamento...")

# Treinar com configurações mais conservadoras
model.fit(X_scaled, y, epochs=100, batch_size=16, verbose=1)

# Salvar modelo
model.save('modelo_tf.h5')
print("Modelo salvo como 'modelo_tf.h5'")

# Salvar scaler e label encoder
joblib.dump(scaler, 'scaler.pkl')
label_encoder = LabelEncoder().fit(df['estado'])
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Scaler salvo como 'scaler.pkl'")
print("LabelEncoder salvo como 'label_encoder.pkl'")

# Exemplo de predição
print(f"\nExemplo de predição:")
print(f"Classes disponíveis: {label_encoder.classes_}")
print(f"Para uma distância de 500mm: {label_encoder.inverse_transform([np.argmax(model.predict(scaler.transform([[500]]), verbose=0))])[0]}")
print(f"Para uma distância de 1500mm: {label_encoder.inverse_transform([np.argmax(model.predict(scaler.transform([[1500]]), verbose=0))])[0]}")
print(f"Para uma distância de 2500mm: {label_encoder.inverse_transform([np.argmax(model.predict(scaler.transform([[2500]]), verbose=0))])[0]}")

print("\nTreinamento concluído com sucesso!")