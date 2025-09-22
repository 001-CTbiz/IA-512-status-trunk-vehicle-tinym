import joblib
import numpy as np

def carregar_modelo():
    """Carrega o modelo treinado e seus componentes"""
    try:
        model = joblib.load('modelo_mlp.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        return model, scaler, label_encoder
    except FileNotFoundError as e:
        print(f"Erro ao carregar arquivos: {e}")
        print("Execute primeiro o main.py para treinar o modelo")
        return None, None, None

def prever_estado_trunk(distancia_mm, model, scaler, label_encoder):
    """Preve o estado do trunk baseado na distância em mm"""
    # Normalizar a entrada
    X_scaled = scaler.transform([[distancia_mm]])
    
    # Fazer predição
    predicao = model.predict(X_scaled)[0]
    
    # Converter de volta para o nome da classe
    estado = label_encoder.inverse_transform([predicao])[0]
    
    # Obter probabilidades
    probabilidades = model.predict_proba(X_scaled)[0]
    
    return estado, probabilidades

def main():
    print("Carregando modelo...")
    model, scaler, label_encoder = carregar_modelo()
    
    if model is None:
        return
    
    print("Modelo carregado com sucesso!")
    print(f"Classes disponíveis: {label_encoder.classes_}")
    
    # Exemplos de uso
    distancias_teste = [300, 500, 800, 1200, 1500, 1800, 2000]
    
    print("\n" + "="*50)
    print("EXEMPLOS DE PREDIÇÃO")
    print("="*50)
    
    for distancia in distancias_teste:
        estado, probs = prever_estado_trunk(distancia, model, scaler, label_encoder)
        
        print(f"\nDistância: {distancia}mm")
        print(f"Estado previsto: {estado}")
        print("Probabilidades:")
        for i, classe in enumerate(label_encoder.classes_):
            print(f"  {classe}: {probs[i]:.3f}")
    
    # Interface interativa
    print("\n" + "="*50)
    print("INTERFACE INTERATIVA")
    print("="*50)
    print("Digite uma distância em mm para prever o estado do trunk")
    print("Digite 'quit' para sair")
    
    while True:
        try:
            entrada = input("\nDistância (mm): ").strip()
            
            if entrada.lower() == 'quit':
                break
                
            distancia = float(entrada)
            estado, probs = prever_estado_trunk(distancia, model, scaler, label_encoder)
            
            print(f"Estado previsto: {estado}")
            print("Probabilidades:")
            for i, classe in enumerate(label_encoder.classes_):
                print(f"  {classe}: {probs[i]:.3f}")
                
        except ValueError:
            print("Por favor, digite um número válido ou 'quit' para sair")
        except KeyboardInterrupt:
            print("\nSaindo...")
            break

if __name__ == "__main__":
    main()
