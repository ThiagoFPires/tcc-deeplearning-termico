# Relatório Consolidado de Resultados do TCC

**Título do Trabalho:** Classificação e Explicabilidade Visual (Grad-CAM) em Termografia Mamária Utilizando Redes Neurais Convolucionais e Transfer Learning  
**Autor:** Thiago Freitas Pires  
**Hardware de Aceleração:** NVIDIA GeForce RTX 5060 (Arquitetura Blackwell, CUDA 12.8)  
**Dataset Utilizado:** DMR-IR (*Database for Mastology Research with Infrared Image*)  
**Repositório Oficial:** [https://github.com/ThiagoFPires/tcc-deeplearning-termico](https://github.com/ThiagoFPires/tcc-deeplearning-termico)

---

## 1. Resumo e Contextualização do Projeto

O câncer de mama é uma das principais causas de mortalidade feminina no mundo. A termografia mamária por infravermelho dinâmico (DIT) destaca-se como um método complementar não invasivo, indolor e livre de radiação ionizante, capaz de identificar alterações fisiológicas e hipertermia tecidual decorrentes de angiogênese precoce antes mesmo de alterações anatômicas estruturais visíveis.

Este trabalho desenvolveu um sistema completo de **Diagnóstico Assistido por Computador (CADe)** baseado em *Deep Learning*, comparando a arquitetura moderna **EfficientNet-B0** com a clássica **ResNet-50**, incorporando técnicas de inteligência artificial explicável (**Grad-CAM**) e disponibilizando a aplicação através de uma **API REST (FastAPI)** e **Interface Web interativa**.

---

## 2. Metodologia Científica

### 2.1. Estruturação do Dataset e Prevenção de *Data Leakage*
O conjunto de dados DMR-IR foi composto por **149 pacientes** (62 saudáveis e 87 com patologias confirmadas), totalizando **2.979 imagens térmicas** (~20 frames por paciente).

Para evitar o **vazamento de dados (*data leakage*)** — erro metodológico comum em que frames de um mesmo exame são distribuídos inadvertidamente entre treino e teste, inflando a acurácia —, realizou-se uma **divisão estratificada a nível de paciente** com semente pseudoaleatória fixa (`seed=42`):

| Partição | Proporção | Qtd. Pacientes | Saudáveis (0) | Doentes (1) | Total de Termogramas |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Treinamento** | 70% | **103** | 43 | 60 | **2.059** imagens |
| **Validação** | 15% | **22** | 9 | 13 | **440** imagens |
| **Teste Isolado (Inédito)** | 15% | **24** | 10 | 14 | **480** imagens |

### 2.2. Pré-processamento e *Data Augmentation* Térmico
1. **Redimensionamento:** Imagens ajustadas para $224 \times 224$ pixels.
2. **Normalização:** Médias e desvios padrão calibrados para *Transfer Learning* no ImageNet ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
3. **Data Augmentation:** Rotações sutis ($\pm 10^\circ$), espelhamento horizontal ($p=0.5$) e ajuste leve de brilho e contraste ($\pm 10\%$), preservando os gradientes térmicos naturais.

### 2.3. Hiperparâmetros de Treinamento
* **Otimizador:** AdamW ($\text{LR} = 3 \times 10^{-4}$, $\text{Weight Decay} = 10^{-2}$).
* **Agendador de Taxa de Aprendizado:** *Cosine Annealing LR* ($\eta_{\min} = 10^{-6}$).
* **Função de Perda:** *Cross-Entropy Loss* ponderada ($\text{peso}_{\text{saudável}} = 1.197$, $\text{peso}_{\text{doente}} = 0.858$) para tratar o leve desbalanceamento de classes.
* **Tamanho de Lote (Batch Size):** 32.
* **Aceleração:** *Automatic Mixed Precision* (AMP - FP16) em Tensor Cores.
* **Critério de Parada:** *Early Stopping* com paciência de 8 épocas monitorando o F1-Score de validação.

---

## 3. Resultados Experimentais e Comparativo de Desempenho

A avaliação dos modelos foi conduzida no **Conjunto de Teste Isolado** (480 termogramas de 24 pacientes inéditos que a rede nunca viu durante o treinamento):

### 3.1. Tabela Comparativa de Métricas Clínicas e Computacionais

| Métrica Diagnóstica | EfficientNet-B0 *(Modelo Proposto)* | ResNet-50 *(Modelo Comparativo)* | Diferença Absoluta |
| :--- | :---: | :---: | :---: |
| **Acurácia Geral** | **93,54%** | 90,62% | **+2,92%** |
| **Sensibilidade / Recall (Detecção de Doentes)** | **95,00%** | 92,86% | **+2,14%** |
| **Especificidade (Confirmação de Saudáveis)** | **91,50%** | 87,50% | **+4,00%** |
| **Precisão (Valor Preditivo Positivo)** | **93,99%** | 91,23% | **+2,76%** |
| **F1-Score** | **0,9449** | 0,9204 | **+0,0245** |
| **AUC-ROC** | **0,9865** | 0,9555 | **+0,0310** |
| **Número de Parâmetros** | **4,01 Milhões** | 23,51 Milhões | **5,8x menor** |
| **Tamanho em Disco dos Pesos** | **~15,3 MB** | ~89,7 MB | **83% de redução** |
| **Tempo de Treinamento** | **4,15 min** (15 épocas) | 2,67 min (11 épocas) | Rápida convergência |

---

### 3.2. Análise das Matrizes de Confusão

```text
                  EFFICIENTNET-B0                          RESNET-50
       Pred: Saudável (0) | Pred: Doente (1)     Pred: Saudável (0) | Pred: Doente (1)
Real: 0     183 (TN)      |     17 (FP)               175 (TN)      |     25 (FP)
Real: 1      14 (FN)      |    266 (TP)                20 (FN)      |    260 (TP)
```

* **Falsos Negativos (Casos Doentes Perdidos):** A EfficientNet-B0 obteve **apenas 14 falsos negativos** contra **20** da ResNet-50. Na medicina diagnóstica, minimizar os falsos negativos é a prioridade crítica para que pacientes com lesões não deixem de receber encaminhamento.
* **Falsos Positivos (Alarmes Falsos):** A EfficientNet-B0 reduziu os alarmes falsos de 25 para 17, diminuindo estresse desnecessário a pacientes saudáveis.

---

### 3.3. Gráficos Gerados em Alta Resolução (300 DPI)
Os gráficos científicos prontos para a monografia encontram-se no diretório `treinamento/relatorios/`:
1. `comparativo_curvas_aprendizado.png`: Demonstra a convergência suave e estabilização da função de perda e da acurácia.
2. `comparativo_matrizes_confusao.png`: Representação gráfica das matrizes de confusão lado a lado.
3. `comparativo_metricas_barras.png`: Comparativo direto de todas as métricas percentuais.

---

## 4. Explicabilidade Visual com Grad-CAM

Para mitigar a característica de "caixa preta" das redes neurais profundas, implementou-se o método **Grad-CAM** (*Gradient-weighted Class Activation Mapping*) na última camada convolucional da EfficientNet-B0 (`features[-1]`).

* **Funcionamento:** O Grad-CAM calcula a média global dos gradientes da classe predita em relação aos mapas de ativação da última camada convolucional, gerando um mapa de calor normalizado $L_{\text{Grad-CAM}} \in [0, 1]$.
* **Validação Clínica:** Os mapas térmicos gerados em `gradcam_explicabilidade_efficientnet_b0.png` comprovaram que a rede neural ativa fortemente (tons vermelhos e amarelos) sobre as **regiões de hipertermia focal e assimetria vascular na mama**, e não em artefatos de fundo ou áreas anatomicamente neutras.

---

## 5. Implementação da Aplicação CADe (FastAPI + Interface Web)

A solução foi empacotada em uma aplicação prática:
1. **Backend (FastAPI):** Microsserviço de alta performance que expõe a rota `POST /api/diagnosticar`, executando a inferência na GPU e retornando a predição probabilística e a imagem Grad-CAM codificada em Base64 em **~80 ms**.
2. **Frontend (Web):** Interface interativa *Dark Mode* que permite arrastar termogramas, testar amostras pré-configuradas e visualizar o laudo diagnóstico com mapa térmico e botão para impressão de relatório clínico.
3. **Automação:** Script `iniciar_sistema.bat` para inicialização completa com 1 clique.

---

## 6. Conclusões e Destaques para a Banca

1. **Superioridade do *Compound Scaling*:** A EfficientNet-B0 superou a ResNet-50 em acurácia (+2,92%), sensibilidade (+2,14%) e AUC (+0,0310), mesmo sendo quase 6 vezes menor em quantidade de parâmetros.
2. **Rigor Metodológico:** A divisão a nível de paciente garantiu que a acurácia de 93,54% seja genuína e representativa da capacidade de generalização do modelo em novos pacientes.
3. **Explicabilidade Pronta para Uso Clínico:** A integração do Grad-CAM na interface web transforma o modelo em uma ferramenta real de apoio à decisão clínica médica.
