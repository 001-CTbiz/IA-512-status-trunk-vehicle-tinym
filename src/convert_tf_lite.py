import os
# Configurações para evitar segmentation fault
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

import tensorflow as tf
from tensorflow.keras.models import load_model

# Configurar TensorFlow para usar apenas CPU e evitar problemas de threading
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

print("Carregando modelo TensorFlow...")
try:
    # Carregar o modelo treinado
    model = load_model('modelo_tf.h5')
    print("Modelo carregado com sucesso!")
    
    # Conversão para TFLite
    print("Convertendo para TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Salvar o modelo
    with open("modelo_quantizado.tflite", "wb") as f:
        f.write(tflite_model)
    
    print("Modelo TFLite salvo como 'modelo_quantizado.tflite'")
    print("Conversão concluída com sucesso!")
    
except FileNotFoundError:
    print("Erro: Arquivo 'modelo_tf.h5' não encontrado!")
    print("Execute primeiro o main.py para treinar o modelo.")
except Exception as e:
    print(f"Erro durante a conversão: {e}")