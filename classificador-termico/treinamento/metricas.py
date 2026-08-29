# pyre-ignore-all-errors
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from typing import Dict, List, Tuple

def calcular_metricas_completas(
    y_true: List[int],
    y_pred: List[int],
    y_probs: List[float]
) -> Dict[str, float]:
    """
    Calcula as métricas diagnósticas essenciais para o TCC na área médica:
    - Acurácia
    - Sensibilidade / Recall (capacidade de detectar pacientes doentes)
    - Especificidade (capacidade de confirmar pacientes saudáveis)
    - Precisão
    - F1-Score
    - AUC-ROC (Área sob a curva ROC)
    - Matriz de Confusão (TN, FP, FN, TP)
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_probs_arr = np.array(y_probs)

    # Matriz de Confusão: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    acc = accuracy_score(y_true_arr, y_pred_arr)
    sensibilidade = recall_score(y_true_arr, y_pred_arr, zero_division=0)  # TP / (TP + FN)
    especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0.0      # TN / (TN + FP)
    precisao = precision_score(y_true_arr, y_pred_arr, zero_division=0)   # TP / (TP + FP)
    f1 = f1_score(y_true_arr, y_pred_arr, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true_arr, y_probs_arr)
    except Exception:
        auc = 0.5

    return {
        'acuracia': round(float(acc), 4),
        'sensibilidade': round(float(sensibilidade), 4),
        'especificidade': round(float(especificidade), 4),
        'precisao': round(float(precisao), 4),
        'f1_score': round(float(f1), 4),
        'auc_roc': round(float(auc), 4),
        'matriz_confusao': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
    }

def formatar_relatorio_metricas(metricas: Dict[str, float], nome_modelo: str) -> str:
    """Gera uma string formatada em tabela para relatórios."""
    mc = metricas['matriz_confusao']
    relatorio = f"""
============================================================
RESULTADOS DE AVALIAÇÃO: {nome_modelo.upper()}
============================================================
- Acurácia Geral:    {metricas['acuracia'] * 100:.2f}%
- Sensibilidade:     {metricas['sensibilidade'] * 100:.2f}%  (Recall / Detecção de Doentes)
- Especificidade:    {metricas['especificidade'] * 100:.2f}%  (Confirmação de Saudáveis)
- Precisão:          {metricas['precisao'] * 100:.2f}%
- F1-Score:          {metricas['f1_score']:.4f}
- AUC-ROC:           {metricas['auc_roc']:.4f}
------------------------------------------------------------
MATRIZ DE CONFUSÃO:
  [Verdadeiros Negativos (Saudáveis corretos)]: {mc['tn']}
  [Falsos Positivos (Alarme falso)]:             {mc['fp']}
  [Falsos Negativos (Doentes perdidos)]:        {mc['fn']}
  [Verdadeiros Positivos (Doentes corretos)]:    {mc['tp']}
============================================================
"""
    return relatorio
