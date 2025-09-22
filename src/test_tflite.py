import os
# Configurações para evitar segmentation fault
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

import tensorflow as tf
import numpy as np
import joblib

# Configurar TensorFlow para usar apenas CPU e evitar problemas de threading
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

def test_tflite_model():
    """Testa o modelo TFLite convertido"""
    try:
        # Carregar o modelo TFLite
        print("Carregando modelo TFLite...")
        interpreter = tf.lite.Interpreter(model_path="modelo_quantizado.tflite")
        interpreter.allocate_tensors()
        
        # Obter detalhes de entrada e saída
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Output shape: {output_details[0]['shape']}")
        
        # Carregar scaler e label encoder
        print("Carregando scaler e label encoder...")
        scaler = joblib.load('scaler.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        
        print(f"Classes disponíveis: {label_encoder.classes_}")
        
        # Testar algumas predições
        test_distances = [300, 500, 800, 1200, 1500, 1800, 2000]
        
        print("\n" + "="*50)
        print("TESTE DO MODELO TFLite")
        print("="*50)
        
        for distance in test_distances:
            # Normalizar entrada
            input_data = scaler.transform([[distance]])
            input_data = input_data.astype(np.float32)
            
            # Fazer predição
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            # Obter resultado
            output_data = interpreter.get_tensor(output_details[0]['index'])
            prediction = np.argmax(output_data[0])
            
            # Converter para nome da classe
            predicted_class = label_encoder.inverse_transform([prediction])[0]
            probabilities = output_data[0]
            
            print(f"\nDistância: {distance}mm")
            print(f"Estado previsto: {predicted_class}")
            print("Probabilidades:")
            for i, classe in enumerate(label_encoder.classes_):
                print(f"  {classe}: {probabilities[i]:.3f}")
        
        print("\n" + "="*50)
        print("TESTE CONCLUÍDO COM SUCESSO!")
        print("="*50)
        
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado - {e}")
        print("Execute primeiro o main.py e convert_tf_lite.py")
    except Exception as e:
        print(f"Erro durante o teste: {e}")

if __name__ == "__main__":
    test_tflite_model()
