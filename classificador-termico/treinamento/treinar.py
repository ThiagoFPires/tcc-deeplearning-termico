# pyre-ignore-all-errors
import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from typing import Dict, Any

from dataset import obter_dataloaders
from modelos import criar_modelo
from metricas import calcular_metricas_completas, formatar_relatorio_metricas

def treinar_epoca(
    modelo: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterio: nn.Module,
    otimizador: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device
) -> Dict[str, Any]:
    """Executa uma época de treinamento com Automatic Mixed Precision (AMP)."""
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
def avaliar_epoca(
    modelo: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterio: nn.Module,
    device: torch.device
) -> Dict[str, Any]:
    """Executa a avaliação em validação ou teste."""
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
    return metricas

def executar_treinamento(
    nome_modelo: str = 'efficientnet_b0',
    epocas: int = 25,
    batch_size: int = 32,
    lr: float = 3e-4,
    patience: int = 8
) -> Dict[str, Any]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n" + "=" * 60)
    print(f"INICIANDO EXPERIMENTO COM: {nome_modelo.upper()}")
    print(f"Dispositivo: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Épocas: {epocas} | Batch Size: {batch_size} | LR: {lr}")
    print("=" * 60)

    # Diretórios
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    splits_path = os.path.join(base_dir, 'splits.json')
    dataset_path = os.path.join(base_dir, 'dataset')
    modelos_salvos_dir = os.path.join(project_root, 'modelos_salvos')
    os.makedirs(modelos_salvos_dir, exist_ok=True)

    # Carrega dados
    loader_treino, loader_val, loader_teste, class_weights = obter_dataloaders(
        splits_path, dataset_path, batch_size=batch_size
    )
    class_weights = class_weights.to(device)

    # Cria Modelo
    modelo = criar_modelo(nome_modelo=nome_modelo, num_classes=2, pretrained=True).to(device)
    criterio = nn.CrossEntropyLoss(weight=class_weights)
    otimizador = AdamW(modelo.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(otimizador, T_max=epocas, eta_min=1e-6)
    scaler = GradScaler(device=device.type)

    historico = {
        'modelo': nome_modelo,
        'treino_loss': [],
        'treino_acc': [],
        'treino_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'val_sensibilidade': [],
        'val_especificidade': [],
        'val_auc': []
    }

    melhor_val_f1 = 0.0
    epocas_sem_melhora = 0
    caminho_melhor_modelo = os.path.join(modelos_salvos_dir, f"melhor_{nome_modelo}.pth")

    tempo_inicio = time.time()

    for epoca in range(1, epocas + 1):
        t_epoca_inicio = time.time()
        m_treino = treinar_epoca(modelo, loader_treino, criterio, otimizador, scaler, device)
        m_val = avaliar_epoca(modelo, loader_val, criterio, device)
        scheduler.step()

        t_duracao = time.time() - t_epoca_inicio

        # Atualiza histórico
        historico['treino_loss'].append(m_treino['loss'])
        historico['treino_acc'].append(m_treino['acuracia'])
        historico['treino_f1'].append(m_treino['f1_score'])
        historico['val_loss'].append(m_val['loss'])
        historico['val_acc'].append(m_val['acuracia'])
        historico['val_f1'].append(m_val['f1_score'])
        historico['val_sensibilidade'].append(m_val['sensibilidade'])
        historico['val_especificidade'].append(m_val['especificidade'])
        historico['val_auc'].append(m_val['auc_roc'])

        print(
            f"Época [{epoca:02d}/{epocas:02d}] ({t_duracao:.1f}s) | "
            f"Treino Loss: {m_treino['loss']:.4f} Acc: {m_treino['acuracia']*100:.1f}% F1: {m_treino['f1_score']:.4f} | "
            f"Val Loss: {m_val['loss']:.4f} Acc: {m_val['acuracia']*100:.1f}% F1: {m_val['f1_score']:.4f} Sens: {m_val['sensibilidade']*100:.1f}%"
        )

        # Checkpoint baseado no F1-Score de validação
        if m_val['f1_score'] > melhor_val_f1:
            melhor_val_f1 = m_val['f1_score']
            epocas_sem_melhora = 0
            torch.save({
                'epoca': epoca,
                'nome_modelo': nome_modelo,
                'state_dict': modelo.state_dict(),
                'metricas_val': m_val,
                'otimizador_state': otimizador.state_dict()
            }, caminho_melhor_modelo)
            print(f"  --> Novo melhor modelo salvo! (Val F1: {melhor_val_f1:.4f})")
        else:
            epocas_sem_melhora += 1
            if epocas_sem_melhora >= patience:
                print(f"\n[Early Stopping] Treinamento interrompido após {epoca} épocas sem melhora.")
                break

    tempo_total = time.time() - tempo_inicio
    print(f"\nTreinamento de {nome_modelo} finalizado em {tempo_total/60:.2f} minutos.")

    # Salva o arquivo de histórico
    caminho_historico = os.path.join(base_dir, f"historico_{nome_modelo}.json")
    with open(caminho_historico, 'w', encoding='utf-8') as f:
        json.dump(historico, f, indent=4)

    # Avaliação Final no Conjunto de Teste (Dados Inéditos / Isolados)
    print("\n" + "=" * 60)
    print(f"AVALIAÇÃO FINAL NO CONJUNTO DE TESTE ISOLADO: {nome_modelo.upper()}")
    print("=" * 60)
    
    checkpoint = torch.load(caminho_melhor_modelo, map_location=device, weights_only=False)
    modelo.load_state_dict(checkpoint['state_dict'])
    m_teste = avaliar_epoca(modelo, loader_teste, criterio, device)
    
    relatorio_teste = formatar_relatorio_metricas(m_teste, f"{nome_modelo} (CONJUNTO DE TESTE)")
    print(relatorio_teste)

    # Salva métricas de teste no arquivo
    caminho_metricas_teste = os.path.join(base_dir, f"teste_{nome_modelo}.json")
    with open(caminho_metricas_teste, 'w', encoding='utf-8') as f:
        json.dump(m_teste, f, indent=4)

    return m_teste

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Treinamento de Modelos Térmicos")
    parser.add_argument('--modelo', type=str, default='efficientnet_b0', choices=['efficientnet_b0', 'resnet50'], help="Nome da arquitetura")
    parser.add_argument('--epocas', type=int, default=25, help="Número máximo de épocas")
    parser.add_argument('--batch_size', type=int, default=32, help="Tamanho do lote")
    parser.add_argument('--lr', type=float, default=3e-4, help="Taxa de aprendizado")
    parser.add_argument('--patience', type=int, default=8, help="Paciência para Early Stopping")

    args = parser.parse_args()
    executar_treinamento(
        nome_modelo=args.modelo,
        epocas=args.epocas,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience
    )
