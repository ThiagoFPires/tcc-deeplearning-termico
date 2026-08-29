# ThermoScan AI: Classificação e Explicabilidade em Termografia Mamária com Deep Learning

> **Trabalho de Conclusão de Curso (TCC)**  
> Diagnóstico Assistido por Computador (CADe) para Detecção de Patologias em Termogramas Mamários via Redes Neurais Convolucionais e Grad-CAM.

---

## 📌 Visão Geral do Projeto

Este projeto desenvolve uma solução completa e ponta-a-ponta para a análise automática de imagens de termografia mamária (Dataset **DMR-IR**).

Utilizando **Transfer Learning**, comparamos a arquitetura moderna e eficiente **EfficientNet-B0** com a clássica **ResNet-50**, implementamos explicabilidade visual com **Grad-CAM** (*Gradient-weighted Class Activation Mapping*) e disponibilizamos uma **API REST (FastAPI)** integrada a uma **Interface Web interativa**.

---

## 📊 Resultados Científicos no Conjunto de Teste (Dados Inéditos / 24 Pacientes)

| Métrica Clínica / Computacional | EfficientNet-B0 *(Modelo Principal)* | ResNet-50 *(Modelo Comparativo)* | Vantagem para o TCC |
| :--- | :---: | :---: | :---: |
| **Acurácia Geral** | **93,54%** | 90,62% | **+2,92%** para EfficientNet |
| **Sensibilidade (Recall)** | **95,00%** | 92,86% | **+2,14%** (Menos falsos negativos) |
| **Especificidade** | **91,50%** | 87,50% | **+4,00%** (Menos alarmes falsos) |
| **Precisão** | **93,99%** | 91,23% | **+2,76%** |
| **F1-Score** | **0,9449** | 0,9204 | **+0,0245** |
| **AUC-ROC** | **0,9865** | 0,9555 | **+0,0310** (Curva ROC superior) |
| **Parâmetros Totais** | **4,01 Milhões** | 23,51 Milhões | **5,8x mais leve** |
| **Tamanho dos Pesos** | **~15,3 MB** | ~89,7 MB | **Ideal para deploy** |

---

## 📁 Estrutura do Repositório

```text
classificador-termico/
├── api/
│   └── main.py                     # Servidor FastAPI (Endpoints de inferência e Grad-CAM)
├── interface/
│   ├── index.html                  # Interface Web moderna
│   ├── styles.css                  # Folha de estilos Dark Mode de alto padrão
│   └── app.js                      # Lógica de upload, comunicação assíncrona e renderização
├── modelos_salvos/
│   ├── melhor_efficientnet_b0.pth  # Pesos treinados do modelo principal (93,54%)
│   └── melhor_resnet50.pth         # Pesos treinados do modelo comparativo (90,62%)
├── treinamento/
│   ├── splits.json                 # Divisão estratificada por paciente (sem data leakage)
│   ├── dividir_dados.py            # Gerador da divisão de dados
│   ├── dataset.py                  # PyTorch Dataset + Data Augmentation Térmico
│   ├── modelos.py                  # Fábrica das redes neurais
│   ├── metricas.py                 # Métricas diagnósticas e matriz de confusão
│   ├── treinar.py                  # Pipeline de treino com AMP CUDA na RTX 5060
│   ├── gradcam.py                  # Módulo de explicabilidade visual Grad-CAM
│   ├── avaliar_comparativo.py      # Gerador de gráficos em 300 DPI para a monografia
│   └── relatorios/                 # Figuras em 300 DPI prontas para o documento do TCC
├── iniciar_sistema.bat             # Executável para subir API + Web + Navegador com 1 clique
└── README.md                       # Documentação completa
```

---

## 🚀 Como Executar o Sistema Completo

### 1. Inicialização Automática (1 Clique):
Basta dar um duplo clique no arquivo:
```text
classificador-termico/iniciar_sistema.bat
```

### 2. Inicialização Manual:

#### **Backend (FastAPI)**:
```powershell
cd classificador-termico
.\venv\Scripts\activate
uvicorn main:app --app-dir api --host 127.0.0.1 --port 8000
```
* Acesso à documentação Swagger da API: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

#### **Frontend (Interface Web)**:
```powershell
cd classificador-termico
.\venv\Scripts\activate
python -m http.server 3000 --directory interface
```
* Acesso à aplicação: [http://127.0.0.1:3000/](http://127.0.0.1:3000/)

---

## 🔬 Aceleração por Hardware

* **GPU**: NVIDIA GeForce RTX 5060 (Arquitetura Blackwell)
* **CUDA**: 12.8
* **Precisão Mista (AMP)**: FP16 / Tensor Cores
* **Tempo de Inferência Médio**: ~80 a 120 ms por exame (incluindo geração do mapa Grad-CAM).
