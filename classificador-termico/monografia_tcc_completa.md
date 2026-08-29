# CLASSIFICAÇÃO E EXPLICABILIDADE VISUAL (GRAD-CAM) EM TERMOGRAFIA MAMÁRIA UTILIZANDO REDES NEURAIS CONVOLUCIONAIS E TRANSFER LEARNING

**Autor:** Thiago Freitas Pires  
**Instituição:** Trabalho de Conclusão de Curso (TCC)  
**Repositório Oficial:** https://github.com/ThiagoFPires/tcc-deeplearning-termico  

---

## RESUMO

O câncer de mama é um dos tipos de câncer mais incidentes entre as mulheres, tornando a detecção precoce um fator fundamental para aumentar as chances de tratamento e reduzir a mortalidade associada à doença. Nesse contexto, a termografia mamária destaca-se como uma técnica não invasiva, indolor e livre de radiação ionizante, capaz de identificar variações térmicas relacionadas a alterações fisiológicas nos tecidos mamários. Entretanto, a interpretação dos termogramas pode apresentar subjetividade, evidenciando a necessidade de métodos computacionais que auxiliem na análise dessas imagens. Diante desse cenário, o presente trabalho desenvolveu e avaliou um modelo baseado na arquitetura EfficientNet-B0, utilizando transfer learning e explicabilidade visual por meio do método Grad-CAM para a classificação de termogramas mamários, comparando seu desempenho com a arquitetura de referência ResNet-50. Para isso, utilizou-se a base de dados DMR-IR, composta por 2.979 imagens termográficas provenientes de 149 pacientes, divididas em nível de paciente para evitar vazamento de dados (*data leakage*). O modelo foi implementado em Python com o framework PyTorch e acelerado em GPU. Nos testes com pacientes inéditos, a EfficientNet-B0 alcançou acurácia de 93,54%, sensibilidade de 95,00%, especificidade de 91,50% e AUC-ROC de 0,9865, superando a ResNet-50 e gerando mapas térmicos explicáveis condizentes com as anomalias clínicas, além de ser integrada a uma interface web com banco de dados SQLite para apoio à decisão médica.

**Palavras-chave:** câncer de mama; termografia mamária; Deep Learning; EfficientNet; Grad-CAM; transfer learning.

---

## 1. INTRODUÇÃO

O câncer de mama é um dos tipos de câncer mais incidentes no mundo, sendo considerado um importante problema de saúde pública. Segundo a Organização Mundial da Saúde, a doença é o câncer mais frequentemente diagnosticado entre as mulheres e uma das principais causas de morte por câncer na população feminina em escala global (WHO, 2024). No Brasil, são estimados 73.610 novos casos da doença por ano para o triênio 2023–2025, correspondendo a uma taxa bruta de incidência de 66,54 casos por 100 mil mulheres, o que reforça a relevância da detecção precoce para aumentar as chances de tratamento e reduzir a mortalidade associada à doença (INCA, 2023).

Entre os métodos utilizados para auxiliar na detecção precoce, a termografia mamária caracteriza-se por ser uma técnica não invasiva, indolor e livre de radiação ionizante (MILOŠEVIĆ; JANKOVIĆ; PEULIĆ, 2014). Essa característica torna sua utilização relevante na área médica, uma vez que permite a realização de exames repetitivos sem os riscos associados à exposição radiológica, além de proporcionar maior conforto ao paciente durante o procedimento. Esse método é capaz de identificar variações térmicas associadas a alterações fisiológicas do paciente, a partir da captação da radiação infravermelha emitida pelo corpo humano. Com isso, torna-se possível analisar diferenças de temperatura na superfície das mamas, que podem estar relacionadas a inflamações ou até mesmo a alterações metabólicas decorrentes da angiogênese tumoral (LUBKOWSKA et al., 2021). 

Apesar dessas vantagens, a análise dos termogramas mamários ainda apresenta limitações, pois depende diretamente da interpretação do profissional responsável pela avaliação, o que pode levar a conclusões distintas diante de uma mesma imagem. Fatores como experiência clínica e qualidade das imagens analisadas podem influenciar diretamente na identificação de padrões suspeitos, evidenciando a necessidade de métodos computacionais que forneçam suporte à tomada de decisões durante a análise dos termogramas mamários.

Diante da subjetividade presente na interpretação manual dos termogramas mamários e da influência da experiência do avaliador nesse processo, surge o seguinte problema de pesquisa: **é possível desenvolver um modelo baseado em Deep Learning, utilizando a arquitetura EfficientNet-B0 e técnicas de transfer learning aliadas à explicabilidade visual com Grad-CAM, capaz de classificar termogramas mamários com desempenho satisfatório, reduzindo a subjetividade da análise e apoiando a identificação de padrões suspeitos associados ao câncer de mama?**

Técnicas baseadas em Deep Learning, especialmente as Redes Neurais Convolucionais (CNNs), têm demonstrado grande eficácia no reconhecimento automatizado de padrões complexos em imagens médicas (LITJENS et al., 2017). Dessa forma, o presente trabalho tem como **objetivo geral** desenvolver e avaliar uma ferramenta computacional baseada em Deep Learning para a detecção de padrões suspeitos em termogramas mamários, utilizando a arquitetura EfficientNet-B0 comparada à ResNet-50.

Como **objetivos específicos**, este trabalho busca:
1. Realizar uma pesquisa bibliográfica sobre câncer de mama, termografia mamária, Deep Learning e inteligência artificial explicável;
2. Estruturar a base de dados DMR-IR por meio de divisão estratificada a nível de paciente (*patient-level split*) para mitigar o vazamento de dados (*data leakage*);
3. Implementar e treinar os modelos neurais EfficientNet-B0 e ResNet-50 utilizando *transfer learning* a partir do ImageNet;
4. Aplicar a técnica de explicabilidade visual Grad-CAM para identificar as regiões anatômicas determinantes na classificação da rede;
5. Avaliar o desempenho dos modelos por meio de métricas diagnósticas (Acurácia, Sensibilidade, Especificidade, Precisão, F1-Score e AUC-ROC);
6. Desenvolver uma interface web interativa conectada a uma API REST com persistência em banco de dados SQLite para apoio à decisão clínica.

---

## 2. REFERENCIAL TEÓRICO

### 2.1 Câncer de Mama e Detecção Precoce
O câncer de mama é originado da multiplicação desordenada de células anormais da mama, formando um tumor com potencial de invadir tecidos adjacentes e órgãos distantes. Em escala mundial, a neoplasia é uma das principais causas de óbito feminino (OMS, 2021). A detecção em fases iniciais é determinante para prognósticos favoráveis, possibilitando intervenções cirúrgicas menos invasivas e redução da mortalidade (BRASIL, 2023). Os métodos convencionais incluem o exame clínico e a mamografia de rastreamento, a qual apresenta limitações em mamas densas e envolve exposição a radiação ionizante (MIGOWSKI et al., 2018).

### 2.2 Fundamentos da Termografia Infravermelha
A termografia por infravermelho capta a radiação eletromagnética emitida pela superfície corporal na faixa do infravermelho termal. O crescimento tumoral é acompanhado pelo fenômeno da angiogênese — neoformação de vasos sanguíneos induzida por fatores de crescimento endotelial (VEGF) —, que eleva localmente o fluxo sanguíneo e a taxa metabólica, gerando assimetrias e hipertermia tecidual detectáveis antes de manifestações anatômicas palpáveis (CÔRTE et al., 2016; LUBKOWSKA et al., 2021).

### 2.3 Deep Learning e Redes Neurais Convolucionais
Deep Learning é uma vertente do aprendizado de máquina baseada em redes neurais artificiais com múltiplas camadas de processamento hierárquico (LECUN; BENGIO; HINTON, 2015). As Redes Neurais Convolucionais (CNNs) são especialmente projetadas para dados bidimensionais, aplicando filtros convolucionais capazes de extrair automaticamente desde bordas e texturas até padrões morfológicos complexos de lesões em imagens médicas (LITJENS et al., 2017).

### 2.4 A Arquitetura EfficientNet
A família EfficientNet (TAN; LE, 2019) introduziu o conceito de escalonamento composto (*compound scaling*), que balanceia simultaneamente a profundidade da rede (número de camadas), a largura (número de canais) e a resolução espacial de entrada por meio de um coeficiente constante. A variante EfficientNet-B0 utiliza blocos convolucionais invertidos (*MBConv*) com módulos de atenção *Squeeze-and-Excitation*, alcançando acurácias superiores com uma fração dos parâmetros exigidos por arquiteturas clássicas (RASEL et al., 2023).

### 2.5 Transfer Learning
O *Transfer Learning* reaproveita os pesos e filtros de representação visual aprendidos por modelos treinados previamente em conjuntos massivos de dados, como o ImageNet (KIM et al., 2022). Em imagens médicas, onde a disponibilidade de amostras anotadas é historicamente limitada, a transferência de aprendizado acelera a convergência do treino e previne o sobreajuste (*overfitting*) (MOHAMED et al., 2022).

### 2.6 Inteligência Artificial Explicável e o Método Grad-CAM
Em sistemas de apoio à decisão clínica, a confiabilidade depende da capacidade de auditar as decisões dos modelos. O método Grad-CAM (*Gradient-weighted Class Activation Mapping*) (SELVARAJU et al., 2017) utiliza os gradientes da classe predita calculados em relação aos mapas de ativação da última camada convolucional para mapear as regiões de maior influência na tomada de decisão da rede. Isso permite validar se a classificação foi fundamentada em alterações térmicas mamárias reais.

---

## 3. TRABALHOS RELACIONADOS

Pesquisas recentes têm explorado o uso de CNNs na termografia mamária:
* **Abdel-Nasser et al. (2022):** Propuseram um sistema automatizado combinando segmentação por U-Net e classificação profunda na base DMR-IR, obtendo 99,33% de acurácia com segmentação prévia.
* **Bani Ahmad et al. (2025):** Desenvolveram a arquitetura híbrida StackVDRNet (integrando VGG16, ResNet e DenseNet), evidenciando o potencial das arquiteturas profundas na redução da subjetividade da interpretação humana.

O presente trabalho diferencia-se ao investigar o escalonamento composto da EfficientNet-B0 em comparação direta com a ResNet-50, integrando explicabilidade por Grad-CAM e uma aplicação CADe acessível via navegador.

---

## 4. METODOLOGIA E DESENVOLVIMENTO

### 4.1 Base de Dados DMR-IR e Divisão por Paciente
Utilizou-se a base DMR-IR (*Database for Mastology Research with Infrared Image*), composta por 149 pacientes (62 saudáveis e 87 com patologias confirmadas), totalizando 2.979 imagens térmicas.

Para evitar o vazamento de dados (*data leakage*), a divisão foi realizada estritamente a nível de paciente (*patient-level split*) com semente fixa (*seed* = 42):
* **Treinamento (70%):** 103 pacientes (2.059 termogramas);
* **Validação (15%):** 22 pacientes (440 termogramas);
* **Teste Isolado (15%):** 24 pacientes (480 termogramas).

### 4.2 Pré-Processamento e Aumento de Dados
As imagens foram redimensionadas para 224 × 224 pixels e normalizadas conforme o padrão ImageNet ($\mu = [0,485, 0,456, 0,406]$, $\sigma = [0,229, 0,224, 0,225]$). O *Data Augmentation* incluiu espelhamentos horizontais ($p = 0,5$), rotações suaves (±10°) e variações discretas de brilho e contraste (±10%), preservando os gradientes de temperatura teciduais.

### 4.3 Treinamento e Configuração Experimental
Os modelos EfficientNet-B0 e ResNet-50 foram implementados em PyTorch com pesos pré-treinados do ImageNet. Empregou-se:
* Otimizador AdamW ($\text{LR} = 3 \times 10^{-4}$, $\text{Weight Decay} = 10^{-2}$);
* Agendador *Cosine Annealing LR* ($\eta_{\min} = 10^{-6}$);
* Função de perda *Cross-Entropy* com pesos de classe ($\text{peso}_0 = 1,197$, $\text{peso}_1 = 0,858$);
* *Automatic Mixed Precision* (AMP FP16) acelerado em GPU NVIDIA GeForce RTX 5060;
* *Early Stopping* com paciência de 8 épocas monitorando o F1-Score de validação.

---

## 5. RESULTADOS E DISCUSSÃO

### 5.1 Comparativo Quantitativo de Desempenho
A avaliação foi conduzida no conjunto de teste isolado (480 termogramas de 24 pacientes inéditos).

**Tabela 1 – Comparativo de desempenho entre EfficientNet-B0 e ResNet-50 no conjunto de teste.**

| Métrica Diagnóstica / Computacional | EfficientNet-B0 (Proposto) | ResNet-50 (Baseline) | Vantagem / Ganho |
| :--- | :---: | :---: | :---: |
| **Acurácia Geral (%)** | **93,54%** | 90,62% | **+2,92%** |
| **Sensibilidade / Recall (%)** | **95,00%** | 92,86% | **+2,14%** |
| **Especificidade (%)** | **91,50%** | 87,50% | **+4,00%** |
| **Precisão (%)** | **93,99%** | 91,23% | **+2,76%** |
| **F1-Score** | **0,9449** | 0,9204 | **+0,0245** |
| **AUC-ROC** | **0,9865** | 0,9555 | **+0,0310** |
| **Parâmetros Totais (Milhões)** | **4,01 M** | 23,51 M | **5,8x mais leve** |
| **Tamanho do Arquivo de Pesos** | **~15,3 MB** | ~89,7 MB | **83% menor** |
| **Tempo Médio de Inferência por Exame** | **~84 ms** | ~190 ms | **Mais de 2x mais rápido** |

*Fonte: Elaborado pelo autor (2026).*

A EfficientNet-B0 superou a ResNet-50 em todas as métricas clínicas, demonstrando que o escalonamento composto é altamente eficaz na extração de padrões térmicos com reduzida complexidade paramétrica.

### 5.2 Análise das Matrizes de Confusão e Falsos Negativos
A matriz de confusão revelou:
* **EfficientNet-B0:** 183 Verdadeiros Negativos (TN), 17 Falsos Positivos (FP), 14 Falsos Negativos (FN) e 266 Verdadeiros Positivos (TP).
* **ResNet-50:** 175 Verdadeiros Negativos (TN), 25 Falsos Positivos (FP), 20 Falsos Negativos (FN) e 260 Verdadeiros Positivos (TP).

A redução dos **falsos negativos de 20 para 14 casos** (redução de 30%) é de extrema relevância clínica, pois minimiza o risco de pacientes com lesões não serem identificadas precocemente.

### 5.3 Explicabilidade Visual com Grad-CAM
Os mapas de calor gerados com Grad-CAM comprovaram que a EfficientNet-B0 concentra sua atenção em focos assimétricos de hipertermia na mama nas amostras patológicas, mantendo ativação difusa e homogênea nas amostras normais. Isso confirma a fundamentação fisiológica do aprendizado da rede.

### 5.4 Aplicação CADe e Persistência em Banco de Dados
Desenvolveu-se uma aplicação completa com API REST (FastAPI), banco de dados SQLite para auditoria de exames e interface web interativa com controle de opacidade do Grad-CAM. A aplicação inclui aviso explícito de que atua como ferramenta de apoio à decisão (CADe), devendo ser correlacionada com a avaliação médica soberana.

---

## 6. CONSIDERAÇÕES FINAIS

O trabalho desenvolveu com êxito um classificador automatizado de termogramas mamários baseado na EfficientNet-B0 com transfer learning e Grad-CAM. O modelo alcançou 93,54% de acurácia e 95,00% de sensibilidade no conjunto de teste, superando a ResNet-50 em desempenho e eficiência computacional.

A metodologia rigorosa com divisão a nível de paciente atestou a generalização do modelo, enquanto o Grad-CAM e a interface web proporcionaram transparência e praticidade clínica. Como trabalhos futuros, sugere-se a validação com dados multicêntricos e a exploração de segmentação automática de regiões anatômicas.

---

## REFERÊNCIAS

ABDEL-NASSER, Mohamed et al. An automated breast cancer detection system in thermal images based on deep learning. **Expert Systems with Applications**, v. 195, p. 116584, 2022.

BANI AHMAD, Firas et al. A hybrid deep learning framework for breast cancer diagnosis using thermography. **Biomedical Signal Processing and Control**, v. 100, p. 107050, 2025.

BRASIL. Ministério da Saúde. **Controle do câncer de mama: documento de consenso**. Brasília: Ministério da Saúde, 2023.

CÔRTE, Ana Lúcia et al. Termografia médica infravermelha: princípios e aplicações clínicas. **Revista Brasileira de Medicina do Esporte**, v. 22, n. 4, p. 320-325, 2016.

INCA. Instituto Nacional de Câncer. **Estimativa 2023: incidência de câncer no Brasil**. Rio de Janeiro: INCA, 2023. Disponível em: https://www.inca.gov.br. Acesso em: 29 ago. 2026.

KIM, Dong-Hyun et al. Transfer learning for medical image classification: a review. **Computers in Biology and Medicine**, v. 142, p. 105214, 2022.

LECUN, Yann; BENGIO, Yoshua; HINTON, Geoffrey. Deep learning. **Nature**, v. 521, n. 7553, p. 436-444, 2015. DOI: https://doi.org/10.1038/nature14539.

LITJENS, Geert et al. A survey on deep learning in medical image analysis. **Medical Image Analysis**, v. 42, p. 60-88, 2017.

LUBKOWSKA, Anna et al. Application of infrared thermography in medicine. **Sensors**, v. 21, n. 15, p. 5044, 2021.

MIGOWSKI, Arn et al. Diretrizes para a detecção precoce do câncer de mama no Brasil. **Cadernos de Saúde Pública**, v. 34, n. 6, p. e00180217, 2018.

MILOŠEVIĆ, Nenad; JANKOVIĆ, Dragan; PEULIĆ, Aleksandar. Thermography based breast cancer detection using machine learning. **Computers in Biology and Medicine**, v. 46, p. 12-19, 2014.

MOHAMED, Asmaa et al. Deep learning and transfer learning for thermal imaging analysis in breast cancer: a systematic review. **Journal of Healthcare Engineering**, v. 2022, p. 1-18, 2022.

OMS. Organização Mundial da Saúde. **Breast cancer: fact sheet**. Genebra: OMS, 2021.

RASEL, Md et al. EfficientNet architectures for medical image analysis: a comprehensive survey. **IEEE Access**, v. 11, p. 89210-89230, 2023.

SELVARAJU, Ramprasaath R. et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. In: **IEEE INTERNATIONAL CONFERENCE ON COMPUTER VISION (ICCV)**, 2017, Venice. Proceedings [...]. Venice: IEEE, 2017. p. 618-626. Disponível em: https://arxiv.org/abs/1610.02391.

SILVA, Lucas et al. A new database for breast research with infrared image. **Journal of Medical Imaging and Health Informatics**, v. 4, n. 1, p. 92-100, 2014.

TAN, Mingxing; LE, Quoc V. EfficientNet: rethinking model scaling for convolutional neural networks. In: **INTERNATIONAL CONFERENCE ON MACHINE LEARNING (ICML)**, 2019, Long Beach. Proceedings [...]. Long Beach: PMLR, 2019. p. 6105-6114.

WHO. World Health Organization. **Global Cancer Observatory: Breast cancer statistics**. Genebra: WHO, 2024. Disponível em: https://gco.iarc.who.int. Acesso em: 29 ago. 2026.
