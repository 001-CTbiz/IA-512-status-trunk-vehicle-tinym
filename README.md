# Status Trunk Vehicle - TensorFlow

Projeto de classificação do estado do porta-malas de um veículo usando TensorFlow e TensorFlow Lite.

## Descrição

Este projeto utiliza uma rede neural MLP (Multi-Layer Perceptron) para classificar o estado do porta-malas em três categorias:
- **CHEIO**: Distâncias menores (~300-500mm)
- **PARCIAL**: Distâncias médias (~800-1200mm)  
- **VAZIO**: Distâncias maiores (~1500-2000mm)

## Estrutura do Projeto

```
status_trunk_vehicle/
├── dataset_carga_milimetros.csv    # Dataset com 300 amostras
├── main.py                         # Treinamento do modelo TensorFlow
├── convert_tf_lite.py             # Conversão para TensorFlow Lite
├── test_tflite.py                 # Teste do modelo TFLite
├── predict.py                     # Interface para predições (scikit-learn)
├── modelo_tf.h5                   # Modelo TensorFlow treinado
├── modelo_quantizado.tflite       # Modelo TensorFlow Lite
├── scaler.pkl                     # Normalizador dos dados
├── label_encoder.pkl              # Encoder das classes
└── venv_tf/                       # Ambiente virtual
```

## Instalação e Configuração

### 1. Criar ambiente virtual
```bash
python3 -m venv venv_tf
source venv_tf/bin/activate  # Linux/macOS
# ou
venv_tf\Scripts\activate     # Windows
```

### 2. Instalar dependências
```bash
pip install tensorflow pandas scikit-learn
```

## Como Usar

### 1. Treinar o modelo
```bash
source venv_tf/bin/activate
python main.py
```

### 2. Converter para TensorFlow Lite
```bash
python convert_tf_lite.py
```

### 3. Testar o modelo TFLite
```bash
python test_tflite.py
```

### 4. Fazer predições (interface interativa)
```bash
python predict.py
```

## Resultados

- **Acurácia**: 100% no conjunto de teste
- **Modelo**: MLP com 2 camadas ocultas (8 neurônios cada)
- **Otimização**: Adam optimizer
- **Quantização**: TensorFlow Lite com otimização DEFAULT

## Exemplos de Predição

| Distância (mm) | Estado Previsto | Probabilidade |
|----------------|-----------------|---------------|
| 300           | CHEIO          | 99.8%         |
| 500           | CHEIO          | 98.7%         |
| 800           | PARCIAL        | 96.6%         |
| 1200          | PARCIAL        | 100%          |
| 1500          | VAZIO          | 80.4%         |
| 1800          | VAZIO          | 99.9%         |
| 2000          | VAZIO          | 100%          |

## Solução do Problema de Segmentation Fault

O projeto original apresentava segmentation fault devido à incompatibilidade entre TensorFlow 2.20.0 e Python 3.13.5. A solução implementada:

1. **Ambiente virtual isolado**: Criado `venv_tf` com dependências específicas
2. **Configurações de threading**: Limitado threads para evitar conflitos
3. **Configurações de ambiente**: Desabilitado logs e otimizações problemáticas
4. **Versão compatível**: TensorFlow 2.20.0 funcionando corretamente no ambiente virtual

## Tecnologias Utilizadas

- **TensorFlow 2.20.0**: Framework principal
- **TensorFlow Lite**: Modelo otimizado para dispositivos móveis
- **scikit-learn**: Pré-processamento dos dados
- **pandas**: Manipulação do dataset
- **numpy**: Operações numéricas

## Autor

Projeto desenvolvido para classificação do estado do porta-malas de veículos usando aprendizado de máquina.
