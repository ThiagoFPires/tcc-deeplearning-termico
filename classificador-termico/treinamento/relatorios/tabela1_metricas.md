# Tabela 1: Comparativo Diagnóstico e Computacional dos Modelos no Conjunto de Teste

| Métrica Clínica / Computacional | EfficientNet-B0 *(Modelo Proposto)* | ResNet-50 *(Baseline Comparativo)* | Diferença Absoluta | Impacto Clínico / Engenharia |
| :--- | :---: | :---: | :---: | :--- |
| **Acurácia Global** | **93.54%** | 90.62% | **+2.92%** | Maior taxa global de diagnósticos corretos. |
| **Sensibilidade (Recall)** | **95.00%** | 92.86% | **+2.14%** | Menor probabilidade de falsos negativos (vital na triagem). |
| **Especificidade** | **91.50%** | 87.50% | **+4.00%** | Redução significativa de alarmes falsos e biópsias desnecessárias. |
| **Precisão** | **93.99%** | 91.23% | **+2.76%** | Alta confiabilidade quando a IA aponta alteração. |
| **F1-Score** | **0.9449** | 0.9204 | **+0.0246** | Harmonia superior entre sensibilidade e precisão. |
| **AUC-ROC** | **0.9865** | 0.9555 | **+0.0310** | Discriminação probabilística excelente em qualquer limiar. |
| **Parâmetros Totais** | **4,01 Milhões** | 23,51 Milhões | **5,8x menor** | Rede muito mais leve contra overfitting. |
| **Tamanho dos Pesos** | **~15,3 MB** | ~89,7 MB | **-82,9%** | Facilidade de embarque em servidores locais e hospitais. |
| **Tempo Médio de Inferência** | **~85 ms** | ~115 ms | **26% mais rápida** | Resposta interativa instantânea para o corpo clínico. |
