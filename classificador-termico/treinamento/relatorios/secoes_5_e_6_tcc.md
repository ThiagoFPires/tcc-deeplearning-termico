# SEÇÕES 5 E 6 — FORMATO CLÁSSICO DE ARTIGO (REVISTA CIENTÍFICA UNIFAGOC)
**Título do Artigo:** DETECÇÃO DE PADRÕES SUSPEITOS EM TERMOGRAMAS MAMÁRIOS UTILIZANDO DEEP LEARNING  
**Autor:** Thiago de Freitas Pires  
**Orientador:** Marcelo Daibert  

---

## 5. RESULTADOS E DISCUSSÃO

A avaliação experimental dos modelos foi conduzida sobre o conjunto de teste independente, composto por 24 pacientes inéditos da base DMR-IR (SILVA et al., 2014), totalizando 480 termogramas (200 de pacientes saudáveis e 280 com alterações patológicas confirmadas). A divisão estrita em nível de paciente (*patient-level split*) assegurou a ausência de vazamento de dados (*data leakage*), refletindo com fidedignidade a capacidade de generalização dos modelos em ambiente de inferência clínica real. A Tabela 1 sintetiza o desempenho diagnóstico e os parâmetros computacionais da arquitetura proposta EfficientNet-B0 em comparação direta com a *baseline* ResNet-50 sob aceleração por GPU (NVIDIA GeForce RTX 5060).

**Tabela 1 – Comparativo de desempenho diagnóstico e computacional no conjunto de teste independente**

| Métrica Clínica / Computacional | EfficientNet-B0 *(Proposto)* | ResNet-50 *(Baseline)* | Variação ($\Delta$) | Impacto Clínico / Engenharia |
| :--- | :---: | :---: | :---: | :--- |
| **Acurácia Global** | **93,54%** | 90,62% | **+2,92%** | Maior taxa global de diagnósticos concordantes. |
| **Sensibilidade (*Recall*)** | **95,00%** | 92,86% | **+2,14%** | Minimização de falsos negativos em triagem médica. |
| **Especificidade** | **91,50%** | 87,50% | **+4,00%** | Redução substancial de falsos positivos e biópsias desnecessárias. |
| **Precisão** | **93,99%** | 91,23% | **+2,76%** | Alta confiabilidade preditiva perante achados alterados. |
| **F1-Score** | **0,9449** | 0,9204 | **+0,0245** | Harmonia superior entre sensibilidade e precisão. |
| **AUC-ROC** | **0,9865** | 0,9555 | **+0,0310** | Excelente discriminação de classes em múltiplos limiares. |
| **Parâmetros Totais** | **4,01 Milhões** | 23,51 Milhões | **5,8× menor** | Estrutura compacta; menor propensão a *overfitting*. |
| **Tamanho dos Pesos** | **~15,3 MB** | ~89,7 MB | **-82,9%** | Facilidade de embarque em servidores locais e hospitais. |
| **Tempo Médio de Inferência** | **~85 ms** | ~115 ms | **26,1% mais rápida** | Resposta interativa instantânea para o especialista. |

*Fonte: Elaborado pelo autor (2026).*

Os dados da Tabela 1 demonstram que a EfficientNet-B0 superou a ResNet-50 em todas as métricas diagnósticas e computacionais. A acurácia global atingiu 93,54% contra 90,62% (+2,92%), confirmando os benefícios do método de escalonamento composto proposto por Tan e Le (2019), que otimiza de maneira proporcional a largura, a profundidade e a resolução da rede. Além da superioridade preditiva, a EfficientNet-B0 demandou apenas 4,01 milhões de parâmetros — uma redução de 82,9% em relação aos 23,51 milhões da ResNet-50 —, resultando em um modelo substancialmente mais leve (~15,3 MB contra ~89,7 MB) e com tempo médio de inferência de aproximadamente 85 ms por termograma. Essa compacidade reduz a propensão ao sobreajuste (*overfitting*) em imagens médicas e viabiliza a implantação em sistemas computacionais hospitalares convencionais sem a necessidade de infraestruturas de alto custo.

No contexto oncológico, a análise das matrizes de confusão (Gráfico 1) evidencia a relevância clínica do modelo proposto, no qual a sensibilidade (*recall*) foi priorizada para mitigar ao máximo as classificações falso-negativas (BRASIL, 2023). Dentre as 280 imagens patológicas avaliadas, a EfficientNet-B0 cometeu apenas 14 falsos negativos (sensibilidade de 95,00%), enquanto a ResNet-50 incorreu em 20 classificações falso-negativas (sensibilidade de 92,86%). Essa diferença representa uma redução de 30% nos casos de lesões não identificadas pela ferramenta, aspecto crucial para assegurar o encaminhamento oportuno de pacientes em estágio inicial da doença. Paralelamente, a especificidade de 91,50% alcançada pela EfficientNet-B0 (183 verdadeiros negativos e 17 falsos positivos em 200 amostras saudáveis), superior aos 87,50% da ResNet-50 (25 falsos positivos), atenua substancialmente a ocorrência de alarmes falsos, prevenindo a solicitação desnecessária de exames complementares invasivos e o consequente impacto psicológico sobre as pacientes (MIGOWSKI et al., 2018).

**Gráfico 1 – Matrizes de confusão comparativas no conjunto de teste independente**

*(Inserir aqui: `grafico2_matrizes_confusao.png`)*

*Fonte: Elaborado pelo autor (2026).*

A capacidade de discriminação probabilística entre os perfis saudável e doente foi avaliada por meio da análise das curvas ROC e da Área sob a Curva (AUC-ROC), apresentadas no Gráfico 2. A EfficientNet-B0 obteve AUC-ROC de 0,9865, superando a ResNet-50 (AUC de 0,9555) em 3,10 pontos percentuais. A trajetória da curva do modelo proposto demonstra acentuada inclinação em direção à taxa de 100% de verdadeiros positivos mesmo sob baixas taxas de falsos positivos, caracterizando um teste diagnóstico de discriminação excelente em múltiplos limiares de corte operacional. A estabilidade desses resultados foi corroborada por experimentos complementares de validação cruzada estratificada em 5 dobras (*5-fold cross-validation*) a nível de paciente, que registraram acurácia média de 95,85% ± 2,24%, sensibilidade média de 94,31% ± 4,40% e AUC-ROC média de 0,9828 ± 0,0184, afastando a hipótese de desempenho casual decorrente de uma partição pontual favorável.

**Gráfico 2 – Curvas ROC comparativas no conjunto de teste (EfficientNet-B0 vs. ResNet-50)**

*(Inserir aqui: `grafico1_curvas_roc_comparativas.png`)*

*Fonte: Elaborado pelo autor (2026).*

Para superar a natureza de "caixa-preta" frequentemente atribuída às redes neurais profundas, geraram-se mapas de calor com a técnica de explicabilidade visual Grad-CAM (SELVARAJU et al., 2017) aplicada à última camada convolucional de ambos os modelos, conforme ilustrado na Figura 1. No caso saudável, caracterizado por isotermia e distribuição uniforme de temperatura cutânea (LUBKOWSKA; CHUDECKA, 2021), a EfficientNet-B0 não produziu ativações focais intensas sobre o parênquima mamário, mantendo tonalidades frias condizentes com a ausência de gradientes térmicos anormais. Já no caso com patologia confirmada, o mapa de ativação convergiu com precisão milimétrica sobre a área de assimetria e hipertermia tumoral (núcleo vermelho/laranja), guardando estrita coerência com os princípios fisiológicos da angiogênese e do hipermetabolismo neoplásico descritos na literatura (CÔRTE; HERNANDEZ, 2016; PERCON, 2016). Em comparação, a ResNet-50 apresentou mapas mais dispersos, ativando regiões do torso e axilas, o que corrobora a superioridade espacial e interpretativa da arquitetura proposta para fins de auditoria médica.

**Figura 1 – Mapas de calor comparativos com Grad-CAM para casos saudável e patológico**

*(Inserir aqui: `figura1_gradcam_comparativo.png`)*

*Fonte: Elaborado pelo autor (2026).*

A validação prática da solução computacional concretizou-se na implementação do sistema web interativo **DeepVision CADe**, desenvolvido para apoiar a rotina do corpo clínico. Integrando o *back-end* em FastAPI com acelerador PyTorch FP16 e uma interface responsiva nativa, o sistema disponibiliza funcionalidades essenciais para triagem: submissão individual ou em lote de termogramas, visualização comparativa lado a lado (*side-by-side*), ferramenta de sobreposição com controle deslizante de transparência (*blend slider* de 0% a 100%), seleção dinâmica de mapas térmicos (*Jet*, *Turbo*, *Inferno*, *Magma* e *Plasma*), magnificação anatômica com pan e zoom, e registro histórico permanente de auditoria em banco SQLite local com exportação em CSV. Para assegurar a conformidade ética em saúde digital, a ferramenta exibe alerta mandatório ressaltando seu caráter de suporte e triagem (CADe), preservando a autoridade soberana e a responsabilidade exclusiva do profissional de saúde na emissão do laudo diagnóstico final.

---

## 6. CONSIDERAÇÕES FINAIS

O presente estudo desenvolveu e avaliou um modelo computacional baseado em Deep Learning para detecção de padrões suspeitos em termogramas mamários, empregando a arquitetura EfficientNet-B0 com aprendizagem por transferência (*transfer learning*) e explicabilidade visual por Grad-CAM, comparando seu desempenho com a ResNet-50 a partir da base DMR-IR.

Os resultados obtidos responderam positivamente ao problema de pesquisa proposto. Avaliada em pacientes inéditos mediante partição estrita em nível de paciente (*patient-level split*), a EfficientNet-B0 alcançou acurácia global de 93,54%, sensibilidade de 95,00%, especificidade de 91,50% e AUC-ROC de 0,9865, superando a ResNet-50 em todas as métricas diagnósticas. O modelo reduziu em 30% os falsos negativos em relação à *baseline*, garantindo alto índice de segurança clínica para triagem oncológica precoce, aliada a uma especificidade capaz de mitigar alarmes falsos e procedimentos invasivos dispensáveis.

Do ponto de vista computacional, a arquitetura proposta comprovou que modelos compactos baseados em escalonamento composto superam redes sobreparametrizadas na interpretação de padrões térmicos sutis. Com apenas 4,01 milhões de parâmetros (cerca de 5,8 vezes menor que a ResNet-50), a rede demandou apenas ~15,3 MB de armazenamento e alcançou tempo médio de resposta de aproximadamente 85 ms, viabilizando sua implantação em dispositivos convencionais de baixo custo. Além disso, a aplicação do Grad-CAM conferiu transparência interpretativa ao demonstrar que a atenção da rede coincide com áreas de angiogênese e hipertermia tumoral, e sua integração na interface web interativa DeepVision CADe comprovou a viabilidade prática da ferramenta para auxílio à tomada de decisão médica.

Como limitações da pesquisa, apontam-se o uso de uma base de dados adquirida sob condições ambientais estritamente homogêneas e controladas, a dimensão amostral finita (149 pacientes) e a ausência de variáveis clínicas associadas (como idade, fatores hormonais e antecedentes familiares). Para trabalhos futuros, recomenda-se a validação do classificador em estudos clínicos prospectivos multicêntricos, o desenvolvimento de modelos multimodais que integrem termogramas a prontuários eletrônicos, a incorporação de rotinas de segmentação automática de mamas em tempo real e a condução de testes clínicos cegos com médicos mastologistas para mensurar o impacto da ferramenta na acurácia e agilidade dos laudos em ambientes hospitalares de rotina.
