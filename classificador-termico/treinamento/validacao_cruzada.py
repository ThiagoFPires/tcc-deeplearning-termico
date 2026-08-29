# pyre-ignore-all-errors
import os
import sys
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import Dict, List, Any

# Garante acesso aos módulos locais
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from dataset import TermografiaDataset, obter_transformacoes
from modelos import criar_modelo
from metricas import calcular_metricas_completas

def carregar_lista_pacientes(dataset_dir: str) -> List[Dict[str, Any]]:
    """Carrega todos os pacientes e suas respectivas classes a partir da pasta dataset."""
    pacientes = []
    classes = {'saudavel': 0, 'doente': 1}
    
    for subpasta, rotulo in classes.items():
        caminho_sub = os.path.join(dataset_dir, subpasta)
        if os.path.exists(caminho_sub):
            for p_id in os.listdir(caminho_sub):
                p_path = os.path.join(caminho_sub, p_id)
                if os.path.isdir(p_path):
                    imgs = [f for f in os.listdir(p_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                    if imgs:
                        pacientes.append({
                            'id': p_id,
                            'classe': rotulo,
                            'qtd_imagens': len(imgs)
                        })
    return pacientes

def treinar_epoca_fold(
    modelo: nn.Module,
    loader: DataLoader,
    criterio: nn.Module,
    otimizador: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device
) -> Dict[str, Any]:
    modelo.train()
    total_loss = 0.0
    y_true_list = []
    y_pred_list = []
    y_probs_list = []

    for imagens, rotulos, _ in loader:
        imagens = imagens.to(device, non_blocking=True)
        rotulos = rotulos.to(device, non_blocking=True)

        otimizador.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, dtype=torch.float16):
            logits = modelo(imagens)
            loss = criterio(logits, rotulos)

        scaler.scale(loss).backward()
        scaler.step(otimizador)
        scaler.update()

        total_loss += loss.item() * imagens.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        y_true_list.extend(rotulos.cpu().tolist())
        y_pred_list.extend(preds.cpu().tolist())
        y_probs_list.extend(probs.detach().cpu().tolist())

    metricas = calcular_metricas_completas(y_true_list, y_pred_list, y_probs_list)
    metricas['loss'] = round(total_loss / len(loader.dataset), 4)
    return metricas

@torch.no_grad()
def avaliar_fold(
    modelo: nn.Module,
    loader: DataLoader,
    criterio: nn.Module,
    device: torch.device
) -> Dict[str, Any]:
    modelo.eval()
    total_loss = 0.0
    y_true_list = []
    y_pred_list = []
    y_probs_list = []

    for imagens, rotulos, _ in loader:
        imagens = imagens.to(device, non_blocking=True)
        rotulos = rotulos.to(device, non_blocking=True)

        with autocast(device_type=device.type, dtype=torch.float16):
            logits = modelo(imagens)
            loss = criterio(logits, rotulos)

        total_loss += loss.item() * imagens.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        y_true_list.extend(rotulos.cpu().tolist())
        y_pred_list.extend(preds.cpu().tolist())
        y_probs_list.extend(probs.cpu().tolist())

    metricas = calcular_metricas_completas(y_true_list, y_pred_list, y_probs_list)
    metricas['loss'] = round(total_loss / len(loader.dataset), 4)
    metricas['y_true'] = y_true_list
    metricas['y_probs'] = y_probs_list
    return metricas

def executar_validacao_cruzada(
    n_splits: int = 5,
    nome_modelo: str = 'efficientnet_b0',
    epocas: int = 15,
    batch_size: int = 32,
    lr: float = 3e-4,
    patience: int = 5
) -> Dict[str, Any]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "=" * 65)
    print(f"INICIANDO VALIDAÇÃO CRUZADA ESTRATIFICADA EM {n_splits} FOLDS")
    print(f"Arquitetura: {nome_modelo.upper()} | Dispositivo: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 65)

    dataset_dir = os.path.join(BASE_DIR, 'dataset')
    relatorios_dir = os.path.join(BASE_DIR, 'relatorios')
    os.makedirs(relatorios_dir, exist_ok=True)

    todos_pacientes = carregar_lista_pacientes(dataset_dir)
    p_ids = np.array([p['id'] for p in todos_pacientes])
    p_classes = np.array([p['classe'] for p in todos_pacientes])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    transform_treino, transform_val = obter_transformacoes(224)

    resultados_folds = []
    roc_data = []

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    t_inicio_total = time.time()

    for fold_idx, (idx_treino, idx_val) in enumerate(skf.split(p_ids, p_classes), start=1):
        t_fold_inicio = time.time()
        print(f"\n--- [FOLD {fold_idx}/{n_splits}] ---")
        
        pacientes_treino = [todos_pacientes[i] for i in idx_treino]
        pacientes_val = [todos_pacientes[i] for i in idx_val]

        ds_treino = TermografiaDataset(dataset_dir, pacientes_treino, transform=transform_treino)
        ds_val = TermografiaDataset(dataset_dir, pacientes_val, transform=transform_val)

        loader_treino = DataLoader(ds_treino, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        # Cálculo de pesos de classe
        rotulos_treino = [a[1] for a in ds_treino.amostras]
        c0, c1 = rotulos_treino.count(0), rotulos_treino.count(1)
        w0 = len(rotulos_treino) / (2.0 * c0) if c0 > 0 else 1.0
        w1 = len(rotulos_treino) / (2.0 * c1) if c1 > 0 else 1.0
        class_weights = torch.tensor([w0, w1], dtype=torch.float).to(device)

        # Instancia modelo limpo
        modelo = criar_modelo(nome_modelo=nome_modelo, num_classes=2, pretrained=True).to(device)
        criterio = nn.CrossEntropyLoss(weight=class_weights)
        otimizador = AdamW(modelo.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = CosineAnnealingLR(otimizador, T_max=epocas, eta_min=1e-6)
        scaler = GradScaler(device=device.type)

        melhor_f1_fold = 0.0
        epocas_sem_melhora = 0
        melhor_metricas_fold = None

        for ep in range(1, epocas + 1):
            m_tr = treinar_epoca_fold(modelo, loader_treino, criterio, otimizador, scaler, device)
            m_vl = avaliar_fold(modelo, loader_val, criterio, device)
            scheduler.step()

            if m_vl['f1_score'] > melhor_f1_fold:
                melhor_f1_fold = m_vl['f1_score']
                melhor_metricas_fold = m_vl
                epocas_sem_melhora = 0
            else:
                epocas_sem_melhora += 1
                if epocas_sem_melhora >= patience:
                    break

        duracao_fold = time.time() - t_fold_inicio
        print(f"Fold {fold_idx} finalizado ({duracao_fold:.1f}s) | "
              f"Acurácia: {melhor_metricas_fold['acuracia']*100:.2f}% | "
              f"Sensibilidade: {melhor_metricas_fold['sensibilidade']*100:.2f}% | "
              f"Especificidade: {melhor_metricas_fold['especificidade']*100:.2f}% | "
              f"F1: {melhor_metricas_fold['f1_score']:.4f} | "
              f"AUC: {melhor_metricas_fold['auc_roc']:.4f}")

        # Cálculo da Curva ROC do Fold
        fpr, tpr, _ = roc_curve(melhor_metricas_fold['y_true'], melhor_metricas_fold['y_probs'])
        roc_auc = auc(fpr, tpr)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        resultados_folds.append({
            'fold': fold_idx,
            'acuracia': melhor_metricas_fold['acuracia'],
            'sensibilidade': melhor_metricas_fold['sensibilidade'],
            'especificidade': melhor_metricas_fold['especificidade'],
            'precisao': melhor_metricas_fold['precisao'],
            'f1_score': melhor_metricas_fold['f1_score'],
            'auc_roc': melhor_metricas_fold['auc_roc'],
            'matriz_confusao': melhor_metricas_fold['matriz_confusao'],
            'tempo_segundos': round(duracao_fold, 1)
        })

    tempo_total = time.time() - t_inicio_total

    # Estatísticas Consolidadas (Média e Desvio Padrão)
    metricas_chaves = ['acuracia', 'sensibilidade', 'especificidade', 'precisao', 'f1_score', 'auc_roc']
    resumo_estatistico = {}

    for k in metricas_chaves:
        vals = [rf[k] for rf in resultados_folds]
        resumo_estatistico[k] = {
            'media': round(float(np.mean(vals)), 4),
            'desvio_padrao': round(float(np.std(vals)), 4),
            'valores_por_fold': vals
        }

    # Salva relatório em JSON
    relatorio_completo = {
        'modelo': nome_modelo,
        'n_splits': n_splits,
        'tempo_total_minutos': round(tempo_total / 60, 2),
        'resumo_estatistico': resumo_estatistico,
        'detalhes_folds': resultados_folds
    }

    caminho_json = os.path.join(relatorios_dir, 'validacao_cruzada_5folds_resultado.json')
    with open(caminho_json, 'w', encoding='utf-8') as f:
        json.dump(relatorio_completo, f, indent=4)

    # 1. Gráfico das Curvas ROC dos 5 Folds com Intervalo de Confiança (300 DPI)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    for i, tpr_f in enumerate(tprs, start=1):
        ax.plot(mean_fpr, tpr_f, alpha=0.4, linestyle='--', label=f'Fold {i} (AUC = {resultados_folds[i-1]["auc_roc"]:.3f})')

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std([rf['auc_roc'] for rf in resultados_folds])

    ax.plot(mean_fpr, mean_tpr, color='#0284C7', linewidth=2.5,
            label=f'Curva ROC Média (AUC = {mean_auc:.3f} ± {std_auc:.3f})')

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='#38BDF8', alpha=0.2, label='Desvio Padrão (±1σ)')

    ax.plot([0, 1], [0, 1], linestyle='--', color='#94A3B8', alpha=0.8, label='Aleatório (AUC = 0.500)')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Taxa de Falsos Positivos (1 - Especificidade)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)', fontsize=11, fontweight='bold')
    ax.set_title(f'Validação Cruzada 5-Fold Estratificada ({nome_modelo.upper()})\nCurvas ROC com Intervalo de Confiança', fontsize=12, fontweight='bold')
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho_fig_roc = os.path.join(relatorios_dir, 'validacao_cruzada_5folds_roc.png')
    plt.savefig(caminho_fig_roc, dpi=300)
    plt.close(fig)

    # 2. Gráfico de Boxplot das Métricas dos 5 Folds
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    dados_box = [[rf[k] * 100 for rf in resultados_folds] for k in ['acuracia', 'sensibilidade', 'especificidade', 'precisao', 'f1_score']]
    rotulos_box = ['Acurácia', 'Sensibilidade\n(Recall)', 'Especificidade', 'Precisão', 'F1-Score']

    bp = ax.boxplot(dados_box, patch_artist=True, tick_labels=rotulos_box)
    for patch in bp['boxes']:
        patch.set_facecolor('#E0F2FE')
        patch.set_edgecolor('#0284C7')
    for median in bp['medians']:
        median.set(color='#0369A1', linewidth=2)

    ax.set_ylabel('Pontuação (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'Distribuição das Métricas nos 5 Folds ({nome_modelo.upper()})', fontsize=12, fontweight='bold')
    ax.set_ylim(80, 102)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    caminho_fig_box = os.path.join(relatorios_dir, 'validacao_cruzada_boxplots.png')
    plt.savefig(caminho_fig_box, dpi=300)
    plt.close(fig)

    print("\n" + "=" * 65)
    print("VALIDAÇÃO CRUZADA 5-FOLD CONCLUÍDA COM SUCESSO!")
    print(f"Tempo total: {tempo_total/60:.2f} minutos")
    print("--- RESULTADOS MÉDIOS CONSOLIDADOS (MÉDIA ± DESVIO PADRÃO) ---")
    print(f" - Acurácia:      {resumo_estatistico['acuracia']['media']*100:.2f}% ± {resumo_estatistico['acuracia']['desvio_padrao']*100:.2f}%")
    print(f" - Sensibilidade: {resumo_estatistico['sensibilidade']['media']*100:.2f}% ± {resumo_estatistico['sensibilidade']['desvio_padrao']*100:.2f}%")
    print(f" - Especificidade:{resumo_estatistico['especificidade']['media']*100:.2f}% ± {resumo_estatistico['especificidade']['desvio_padrao']*100:.2f}%")
    print(f" - Precisão:      {resumo_estatistico['precisao']['media']*100:.2f}% ± {resumo_estatistico['precisao']['desvio_padrao']*100:.2f}%")
    print(f" - F1-Score:      {resumo_estatistico['f1_score']['media']:.4f} ± {resumo_estatistico['f1_score']['desvio_padrao']:.4f}")
    print(f" - AUC-ROC:       {resumo_estatistico['auc_roc']['media']:.4f} ± {resumo_estatistico['auc_roc']['desvio_padrao']:.4f}")
    print("=" * 65)
    print(f"Gráficos 300 DPI gerados:")
    print(f" 1. {caminho_fig_roc}")
    print(f" 2. {caminho_fig_box}")
    print("=" * 65)

    return relatorio_completo

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validação Cruzada 5-Fold")
    parser.add_argument('--modelo', type=str, default='efficientnet_b0', help="Nome da arquitetura")
    parser.add_argument('--epocas', type=int, default=15, help="Épocas por fold")
    parser.add_argument('--batch_size', type=int, default=32, help="Tamanho do lote")
    args = parser.parse_args()

    executar_validacao_cruzada(
        n_splits=5,
        nome_modelo=args.modelo,
        epocas=args.epocas,
        batch_size=args.batch_size
    )
