# DeepVision CADe — Classificação e Explicabilidade em Termografia Mamária com Deep Learning

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8%20(RTX%205060)-76B900.svg?style=flat&logo=nvidia)](https://developer.nvidia.com/cuda-zone)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141.1-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?style=flat&logo=docker)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Trabalho de Conclusão de Curso (TCC)**  
> **Autor:** Thiago Freitas Pires (`contato.thiagofreitasp@gmail.com`)  
> **DeepVision CADe**: Sistema Computacional de Detecção Assistida por Computador (CADe) em Termografia Mamária utilizando **EfficientNet-B0**, **ResNet-50**, **Grad-CAM**, **Validação Cruzada 5-Fold** e **Docker**.


---

## 📌 Visão Geral do Projeto

Este projeto desenvolve uma solução completa e ponta-a-ponta para a análise e classificação de imagens de termografia mamária (Dataset **DMR-IR** — 149 pacientes e 2.979 termogramas).

O pipeline conta com:
1. **Modelagem Deep Learning**: Comparativo entre a arquitetura moderna **EfficientNet-B0** e a clássica **ResNet-50** com *Transfer Learning*.
2. **Validação Estatística Rigorosa**: Divisão estratificada a nível de paciente (*patient-level split*) e **Validação Cruzada 5-Fold** para eliminação de vazamento de dados (*data leakage*).
3. **Explicabilidade Visual (Grad-CAM)**: Mapeamento de atenção em mapa de calor para auditoria médica de focos de hipertermia e assimetria vascular.
4. **Segmentação Automática de ROI**: Algoritmo de visão computacional para isolar a região torácica/mamária e remover ruídos de fundo.
5. **Auditoria e Persistência**: Banco de dados **SQLite** para histórico clínico de exames processados.
6. **Deploy Multiplataforma**: Execução nativa acelerada por GPU (NVIDIA RTX 5060) e containerização completa com **Docker**.

---

## 📊 Resultados Científicos

### 1. Teste Cego com Pacientes Inéditos (Split 70/15/15)

| Métrica Diagnóstica / Computacional | EfficientNet-B0 *(Proposto)* | ResNet-50 *(Baseline)* | Vantagem para o TCC |
| :--- | :---: | :---: | :---: |
| **Acurácia Geral** | **93,54%** | 90,62% | **+2,92%** |
| **Sensibilidade (Recall)** | **95,00%** | 92,86% | **+2,14%** (Menos falsos negativos) |
| **Especificidade** | **91,50%** | 87,50% | **+4,00%** (Menos alarmes falsos) |
| **Precisão** | **93,99%** | 91,23% | **+2,76%** |
| **F1-Score** | **0,9449** | 0,9204 | **+0,0245** |
| **AUC-ROC** | **0,9865** | 0,9555 | **+0,0310** |
| **Total de Parâmetros** | **4,01 Milhões** | 23,51 Milhões | **5,8x mais leve** |
| **Tamanho em Disco** | **~15,3 MB** | ~89,7 MB | **83% menor** |
| **Tempo Médio de Inferência** | **~84 ms** (GPU) | ~190 ms (GPU) | **Mais de 2x mais rápido** |

---

### 2. Validação Cruzada Estratificada (5-Fold Cross Validation)

Treinamento em 5 partições independentes na GPU **NVIDIA GeForce RTX 5060**:

* **Acurácia Média:** **95,85% ± 2,24%**
* **Sensibilidade Média:** **94,31% ± 4,40%**
* **Especificidade Média:** **98,08% ± 2,98%**
* **Precisão Média:** **98,56% ± 2,17%**
* **F1-Score Médio:** **0,9630 ± 0,0208**
* **AUC-ROC Médio:** **0,9828 ± 0,0184**

---

## 📁 Estrutura do Repositório

```text
tcc-deeplearning-termico/
├── classificador-termico/
│   ├── api/
│   │   ├── main.py                     # API FastAPI (Inferência, Grad-CAM e Histórico)
│   │   └── banco.py                    # Gerenciador do banco de dados SQLite
│   ├── interface/
│   │   ├── index.html                  # Interface Web Médica (CADe PACS)
│   │   ├── styles.css                  # Folha de estilos Dark Slate Hospitalar
│   │   └── app.js                      # Comunicação assíncrona, slider blend e histórico
│   ├── modelos_salvos/                 # Checkpoints treinados (.pth)
│   ├── treinamento/
│   │   ├── dataset.py                  # Dataloaders e Data Augmentation térmico
│   │   ├── modelos.py                  # Arquiteturas (EfficientNet-B0 e ResNet-50)
│   │   ├── metricas.py                 # Cálculo de matriz de confusão, F1, AUC
│   │   ├── treinar.py                  # Pipeline de treino com AMP CUDA
│   │   ├── validacao_cruzada.py        # Módulo de 5-Fold Cross Validation
│   │   ├── segmentador_roi.py          # Segmentação térmica de Região de Interesse
│   │   ├── gradcam.py                  # Explicabilidade visual Grad-CAM
│   │   └── relatorios/                 # Figuras em 300 DPI (ROC, Boxplots, Grad-CAM)
│   ├── Dockerfile                      # Imagem do container
│   ├── docker-compose.yml              # Orquestração do container
│   └── iniciar_sistema.bat             # Executável para Windows em 1 clique
├── docker-compose.yml                  # Compose na raiz do repositório
└── README.md                           # Documentação oficial
```

---

## 🚀 Como Executar o Projeto

### Opção 1: Execução com Docker (Recomendado para Deploy)
Com o Docker Desktop aberto, execute na raiz do projeto:
```bash
docker compose up -d
```
Acesse a aplicação no navegador:  
👉 **[http://localhost:8000/web/](http://localhost:8000/web/)**

---

### Opção 2: Execução Nativa no Windows (Aceleração Máxima na GPU)
Dê um duplo clique no arquivo:
```text
classificador-termico/iniciar_sistema.bat
```
A aplicação abrirá automaticamente:
* **Interface Web:** [http://127.0.0.1:3000/](http://127.0.0.1:3000/)
* **API Swagger / Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## ⚖️ Aviso Clínico (Medical Disclaimer)

> Esta ferramenta computacional (CADe) não emite diagnóstico médico soberano e possui finalidade exclusiva de triagem e apoio à decisão de profissionais de saúde habilitados. A análise térmica deve ser correlacionada com a história clínica e exames complementares de imagem (como mamografia e ultrassonografia).
