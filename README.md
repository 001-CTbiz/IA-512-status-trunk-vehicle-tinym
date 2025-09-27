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

## Exportação do Modelo para C++

### Arquivo do Modelo TensorFlow Lite

O modelo quantizado está disponível como:
- **Nome do arquivo**: `modelo_quantizado.tflite`
- **Localização**: `models/modelo_quantizado.tflite`
- **Tamanho**: Otimizado para dispositivos móveis e embarcados

### Como Usar o Modelo em C++

Para integrar o modelo TensorFlow Lite em um projeto C++, siga estes passos:

#### 1. Configuração do Projeto C++

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(trunk_vehicle_classifier)

set(CMAKE_CXX_STANDARD 17)

# TensorFlow Lite
find_package(PkgConfig REQUIRED)
pkg_check_modules(TFLITE REQUIRED tensorflow-lite)

include_directories(${TFLITE_INCLUDE_DIRS})
link_directories(${TFLITE_LIBRARY_DIRS})

add_executable(trunk_classifier main.cpp)
target_link_libraries(trunk_classifier ${TFLITE_LIBRARIES})
```

#### 2. Código C++ de Exemplo

```cpp
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include <iostream>

class TrunkVehicleClassifier {
private:
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    
public:
    bool loadModel(const std::string& model_path) {
        model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
        if (!model) {
            std::cerr << "Erro ao carregar modelo: " << model_path << std::endl;
            return false;
        }
        
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder builder(*model, resolver);
        builder(&interpreter);
        
        if (!interpreter) {
            std::cerr << "Erro ao criar interpretador" << std::endl;
            return false;
        }
        
        interpreter->AllocateTensors();
        return true;
    }
    
    std::string predict(float distance) {
        // Normalizar entrada (mesmo scaler usado no treinamento)
        float normalized_distance = (distance - 1000.0f) / 500.0f;
        
        // Obter ponteiro para tensor de entrada
        float* input = interpreter->typed_input_tensor<float>(0);
        input[0] = normalized_distance;
        
        // Executar inferência
        interpreter->Invoke();
        
        // Obter resultados
        float* output = interpreter->typed_output_tensor<float>(0);
        
        // Encontrar classe com maior probabilidade
        int predicted_class = 0;
        float max_prob = output[0];
        
        for (int i = 1; i < 3; i++) {
            if (output[i] > max_prob) {
                max_prob = output[i];
                predicted_class = i;
            }
        }
        
        // Mapear classes
        std::string classes[] = {"CHEIO", "PARCIAL", "VAZIO"};
        return classes[predicted_class];
    }
};

int main() {
    TrunkVehicleClassifier classifier;
    
    if (!classifier.loadModel("modelo_quantizado.tflite")) {
        return -1;
    }
    
    // Exemplo de uso
    float distance = 800.0f; // mm
    std::string prediction = classifier.predict(distance);
    
    std::cout << "Distância: " << distance << "mm" << std::endl;
    std::cout << "Estado previsto: " << prediction << std::endl;
    
    return 0;
}
```

#### 3. Compilação

```bash
# Instalar TensorFlow Lite C++
sudo apt-get install libtensorflow-lite-dev

# Compilar
mkdir build && cd build
cmake ..
make
```

#### 4. Dependências Necessárias

- **TensorFlow Lite C++**: Biblioteca principal
- **CMake**: Sistema de build
- **C++17**: Padrão mínimo

### Características do Modelo Exportado

- **Formato**: TensorFlow Lite (.tflite)
- **Quantização**: INT8 para otimização de memória
- **Tamanho**: ~2-5 KB (dependendo da quantização)
- **Latência**: < 1ms em hardware embarcado
- **Entrada**: 1 valor float (distância em mm)
- **Saída**: 3 valores float (probabilidades das classes)

## Tecnologias Utilizadas

- **TensorFlow 2.20.0**: Framework principal
- **TensorFlow Lite**: Modelo otimizado para dispositivos móveis
- **scikit-learn**: Pré-processamento dos dados
- **pandas**: Manipulação do dataset
- **numpy**: Operações numéricas

## Autor

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request


## 📜 Licença
Este projeto é licenciado sob os termos da **GNU Affero General Public License v3.0 (AGPL-3.0)**.  
Veja o arquivo [LICENSE](./LICENSE.txt) para mais detalhes.

## 👨‍💻 Autor

**Jonas Santos da Silva**  
CEO | CONTADOR | Desenv. de Software | Pesquisador em IA  
[linkedin.com/in/jonas-ceo](https://linkedin.com/in/jonas-ceo)

## 🙏 Agradecimentos

- TensorFlow/Keras pela framework
- Comunidade de deep learning
- Contribuidores e testadores do projeto   

Projeto desenvolvido para classificação do estado do porta-malas de veículos usando aprendizado de máquina.
