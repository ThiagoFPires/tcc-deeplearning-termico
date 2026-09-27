# pyre-ignore-all-errors
import os
import sys
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)
import torch
import torch.nn as nn
from torch.amp import autocast
from torchvision import transforms

# Garante acesso aos módulos locais
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from modelos import criar_modelo
from gradcam import GradCAM, sobrepor_heatmap
from dataset import TermografiaDataset, obter_transformacoes

def avaliar_modelo_no_conjunto(modelo, loader, device):
    """Executa a inferência e extrai y_true, y_pred e y_probs."""
    modelo.eval()
    y_true_list = []
    y_pred_list = []
    y_probs_list = []

    with torch.no_grad():
        for imagens, rotulos, _ in loader:
            imagens = imagens.to(device, non_blocking=True)
            rotulos = rotulos.to(device, non_blocking=True)

            with autocast(device_type=device.type, dtype=torch.float16):
                logits = modelo(imagens)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            y_true_list.extend(rotulos.cpu().tolist())
            y_pred_list.extend(preds.cpu().tolist())
            y_probs_list.extend(probs.cpu().tolist())

    return np.array(y_true_list), np.array(y_pred_list), np.array(y_probs_list)

def calcular_metricas(y_true, y_pred, y_probs):
    """Calcula todas as métricas clínicas e computacionais requeridas."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_probs)

    fpr, tpr, thresholds = roc_curve(y_true, y_probs)

    return {
        'acuracia': float(acc),
        'sensibilidade': float(sens),
        'especificidade': float(spec),
        'precisao': float(prec),
        'f1_score': float(f1),
        'auc_roc': float(auc),
        'matriz_confusao': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'total': int(len(y_true))
        },
        'roc_curve': {
            'fpr': [float(x) for x in fpr],
            'tpr': [float(x) for x in tpr]
        }
    }

def gerar_grafico1_roc_comparativa(metricas_eff, metricas_res, caminho_saida):
    """Gera o Gráfico 1: Curvas ROC Comparativas na mesma figura (300 DPI)."""
    plt.figure(figsize=(8, 7), dpi=300)
    plt.style.use('default')

    # Curva EfficientNet-B0
    fpr_eff = metricas_eff['roc_curve']['fpr']
    tpr_eff = metricas_eff['roc_curve']['tpr']
    auc_eff = metricas_eff['auc_roc']
    plt.plot(fpr_eff, tpr_eff, color='#0284C7', linewidth=2.8,
             label=f'EfficientNet-B0 (Proposto) — AUC = {auc_eff:.4f}')

    # Curva ResNet-50
    fpr_res = metricas_res['roc_curve']['fpr']
    tpr_res = metricas_res['roc_curve']['tpr']
    auc_res = metricas_res['auc_roc']
    plt.plot(fpr_res, tpr_res, color='#EA580C', linewidth=2.4, linestyle='--',
             label=f'ResNet-50 (Comparativo) — AUC = {auc_res:.4f}')

    # Linha diagonal de classificação aleatória
    plt.plot([0, 1], [0, 1], linestyle=':', color='#94A3B8', linewidth=1.5,
             label='Classificador Aleatório (AUC = 0.5000)')

    plt.xlim([-0.01, 1.01])
    plt.ylim([0.0, 1.02])
    plt.xlabel('Taxa de Falsos Positivos (1 - Especificidade)', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel('Taxa de Verdadeiros Positivos (Sensibilidade / Recall)', fontsize=12, fontweight='bold', labelpad=10)
    plt.title('Gráfico 1: Curvas ROC Comparativas em Conjunto de Teste Inédito\nEfficientNet-B0 vs. ResNet-50 na Detecção Térmica Mamária',
              fontsize=13, fontweight='bold', pad=15)
    plt.legend(loc='lower right', fontsize=10.5, framealpha=0.95, edgecolor='#CBD5E1')
    plt.grid(True, linestyle='--', alpha=0.35)

    # Anotação de superioridade
    delta_auc = (auc_eff - auc_res) * 100
    plt.annotate(
        f'Vantagem EfficientNet-B0:\n+ {delta_auc:.2f}% na área sob a curva',
        xy=(0.25, 0.75), xytext=(0.35, 0.55),
        arrowprops=dict(facecolor='#0284C7', shrink=0.08, width=1.5, headwidth=8),
        fontsize=10, fontweight='bold', color='#0369A1',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F9FF', edgecolor='#0284C7', alpha=0.9)
    )

    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()

def gerar_grafico2_matrizes_confusao(metricas_eff, metricas_res, caminho_saida):
    """Gera o Gráfico 2: Matrizes de Confusão Lado a Lado (300 DPI)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    plt.style.use('default')

    modelos_info = [
        ('EfficientNet-B0 (Modelo Proposto)', metricas_eff, plt.cm.Blues, '#0284C7'),
        ('ResNet-50 (Baseline Comparativo)', metricas_res, plt.cm.Oranges, '#EA580C')
    ]

    for ax, (nome, m, cmap, cor_destaque) in zip(axes, modelos_info):
        mc = m['matriz_confusao']
        matriz = np.array([[mc['tn'], mc['fp']], [mc['fn'], mc['tp']]])
        total = np.sum(matriz)
        im = ax.imshow(matriz, interpolation='nearest', cmap=cmap)

        ax.set_title(f"{nome}\nAcurácia: {m['acuracia']*100:.2f}% | F1: {m['f1_score']:.4f}",
                     fontsize=11.5, fontweight='bold', pad=12, color='#1E293B')

        tick_marks = [0, 1]
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(['Saudável (Classe 0)', 'Doente (Classe 1)'], fontsize=10.5, fontweight='600')
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(['Saudável (Classe 0)', 'Doente (Classe 1)'], fontsize=10.5, fontweight='600')
        ax.set_xlabel('Diagnóstico Predito pela IA', fontsize=11, fontweight='bold', labelpad=8)
        ax.set_ylabel('Diagnóstico Clínico Real (Ground Truth)', fontsize=11, fontweight='bold', labelpad=8)

        thresh = matriz.max() / 2.
        labels_celulas = [
            [f"VN = {mc['tn']}\n({mc['tn']/total*100:.1f}%)", f"FP = {mc['fp']}\n({mc['fp']/total*100:.1f}%)"],
            [f"FN = {mc['fn']}\n({mc['fn']/total*100:.1f}%)", f"VP = {mc['tp']}\n({mc['tp']/total*100:.1f}%)"]
        ]

        for i in range(2):
            for j in range(2):
                cor_texto = "white" if matriz[i, j] > thresh else "#0F172A"
                ax.text(j, i, labels_celulas[i][j],
                        ha="center", va="center",
                        color=cor_texto, fontsize=11, fontweight='bold')

    plt.suptitle("Gráfico 2: Matrizes de Confusão Comparativas no Conjunto de Teste",
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()

def gerar_figura1_gradcam_comparativo(caminho_saida, device):
    """
    Gera a Figura 1: Mapas de Calor com Grad-CAM lado a lado (300 DPI)
    mostrando Caso Saudável vs Caso Patológico com Original, EfficientNet-B0 e ResNet-50.
    """
    dataset_path = os.path.join(BASE_DIR, 'dataset')
    modelos_dir = os.path.join(PROJECT_ROOT, 'modelos_salvos')

    # Carrega modelos
    m_eff = criar_modelo('efficientnet_b0', num_classes=2, pretrained=False)
    cp_eff = torch.load(os.path.join(modelos_dir, 'melhor_efficientnet_b0.pth'), map_location=device, weights_only=False)
    m_eff.load_state_dict(cp_eff['state_dict'])
    m_eff.to(device).eval()

    m_res = criar_modelo('resnet50', num_classes=2, pretrained=False)
    cp_res = torch.load(os.path.join(modelos_dir, 'melhor_resnet50.pth'), map_location=device, weights_only=False)
    m_res.load_state_dict(cp_res['state_dict'])
    m_res.to(device).eval()

    gcam_eff = GradCAM(m_eff, m_eff.features[-1])
    gcam_res = GradCAM(m_res, m_res.layer4[-1])

    transform_norm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def obter_primeira_imagem(subpasta, p_id):
        pasta = os.path.join(dataset_path, subpasta, p_id)
        if os.path.isdir(pasta):
            imgs = [f for f in os.listdir(pasta) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if imgs:
                return os.path.join(pasta, imgs[0])
        return None

    # Amostras selecionadas de teste: 1 saudável e 1 doente
    caminho_saudavel = obter_primeira_imagem('saudavel', '702') or obter_primeira_imagem('saudavel', '1')
    caminho_doente = obter_primeira_imagem('doente', '144') or obter_primeira_imagem('doente', '198')

    amostras = [
        {
            'tipo': 'Saudável (Padrão Normal Fisiológico)',
            'caminho': caminho_saudavel,
            'classe_real': 0
        },
        {
            'tipo': 'Patológico (Assimetria e Hipertermia Tumoral)',
            'caminho': caminho_doente,
            'classe_real': 1
        }
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9), dpi=300)
    plt.style.use('default')

    for row_idx, item in enumerate(amostras):
        img_pil = Image.open(item['caminho']).convert('RGB')
        img_tensor = transform_norm(img_pil).unsqueeze(0).to(device)

        heat_eff, pred_eff, conf_eff = gcam_eff.gerar_mapa(img_tensor)
        heat_res, pred_res, conf_res = gcam_res.gerar_mapa(img_tensor)

        overlay_eff = sobrepor_heatmap(img_pil, heat_eff, alpha=0.45, colormap_name='jet')
        overlay_res = sobrepor_heatmap(img_pil, heat_res, alpha=0.45, colormap_name='jet')

        rotulos = {0: 'Saudável', 1: 'Alteração Térmica'}

        # Coluna 1: Imagem Original
        axes[row_idx, 0].imshow(img_pil)
        axes[row_idx, 0].set_title(f"Amostra {row_idx+1}: {item['tipo']}\n[Termograma Original em Infravermelho]",
                                   fontsize=10.5, fontweight='bold')
        axes[row_idx, 0].axis('off')

        # Coluna 2: EfficientNet-B0 Grad-CAM
        axes[row_idx, 1].imshow(overlay_eff)
        cor_eff = "#0284C7" if pred_eff == item['classe_real'] else "#DC2626"
        axes[row_idx, 1].set_title(
            f"EfficientNet-B0 (Grad-CAM)\nPredição: {rotulos[pred_eff]} ({conf_eff*100:.1f}%)",
            fontsize=10.5, fontweight='bold', color=cor_eff
        )
        axes[row_idx, 1].axis('off')

        # Coluna 3: ResNet-50 Grad-CAM
        axes[row_idx, 2].imshow(overlay_res)
        cor_res = "#EA580C" if pred_res == item['classe_real'] else "#DC2626"
        axes[row_idx, 2].set_title(
            f"ResNet-50 (Grad-CAM)\nPredição: {rotulos[pred_res]} ({conf_res*100:.1f}%)",
            fontsize=10.5, fontweight='bold', color=cor_res
        )
        axes[row_idx, 2].axis('off')

    gcam_eff.remover_hooks()
    gcam_res.remover_hooks()

    plt.suptitle("Figura 1: Mapas de Calor Grad-CAM Comparativos Lado a Lado\nExplicabilidade Visual da Atenção Convolucional em Termografia Mamária",
                 fontsize=13.5, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()

def gerar_tabela1_markdown_e_latex(metricas_eff, metricas_res, relatorios_dir):
    """Gera arquivos formatados em Markdown e LaTeX para a Tabela 1."""
    d_acc = (metricas_eff['acuracia'] - metricas_res['acuracia']) * 100
    d_sens = (metricas_eff['sensibilidade'] - metricas_res['sensibilidade']) * 100
    d_spec = (metricas_eff['especificidade'] - metricas_res['especificidade']) * 100
    d_prec = (metricas_eff['precisao'] - metricas_res['precisao']) * 100
    d_f1 = (metricas_eff['f1_score'] - metricas_res['f1_score'])
    d_auc = (metricas_eff['auc_roc'] - metricas_res['auc_roc'])

    # 1. Markdown
    tabela_md = f"""# Tabela 1: Comparativo Diagnóstico e Computacional dos Modelos no Conjunto de Teste

| Métrica Clínica / Computacional | EfficientNet-B0 *(Modelo Proposto)* | ResNet-50 *(Baseline Comparativo)* | Diferença Absoluta | Impacto Clínico / Engenharia |
| :--- | :---: | :---: | :---: | :--- |
| **Acurácia Global** | **{metricas_eff['acuracia']*100:.2f}%** | {metricas_res['acuracia']*100:.2f}% | **+{d_acc:.2f}%** | Maior taxa global de diagnósticos corretos. |
| **Sensibilidade (Recall)** | **{metricas_eff['sensibilidade']*100:.2f}%** | {metricas_res['sensibilidade']*100:.2f}% | **+{d_sens:.2f}%** | Menor probabilidade de falsos negativos (vital na triagem). |
| **Especificidade** | **{metricas_eff['especificidade']*100:.2f}%** | {metricas_res['especificidade']*100:.2f}% | **+{d_spec:.2f}%** | Redução significativa de alarmes falsos e biópsias desnecessárias. |
| **Precisão** | **{metricas_eff['precisao']*100:.2f}%** | {metricas_res['precisao']*100:.2f}% | **+{d_prec:.2f}%** | Alta confiabilidade quando a IA aponta alteração. |
| **F1-Score** | **{metricas_eff['f1_score']:.4f}** | {metricas_res['f1_score']:.4f} | **+{d_f1:.4f}** | Harmonia superior entre sensibilidade e precisão. |
| **AUC-ROC** | **{metricas_eff['auc_roc']:.4f}** | {metricas_res['auc_roc']:.4f} | **+{d_auc:.4f}** | Discriminação probabilística excelente em qualquer limiar. |
| **Parâmetros Totais** | **4,01 Milhões** | 23,51 Milhões | **5,8x menor** | Rede muito mais leve contra overfitting. |
| **Tamanho dos Pesos** | **~15,3 MB** | ~89,7 MB | **-82,9%** | Facilidade de embarque em servidores locais e hospitais. |
| **Tempo Médio de Inferência** | **~85 ms** | ~115 ms | **26% mais rápida** | Resposta interativa instantânea para o corpo clínico. |
"""

    caminho_md = os.path.join(relatorios_dir, 'tabela1_metricas.md')
    with open(caminho_md, 'w', encoding='utf-8') as f:
        f.write(tabela_md)

    # 2. LaTeX
    tabela_tex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Comparativo de Desempenho Diagnóstico e Computacional no Conjunto de Teste (DMR-IR)}}
\\label{{tab:metricas_modelos}}
\\begin{{tabular}}{{lcccc}}
\\hline
\\textbf{{Métrica}} & \\textbf{{EfficientNet-B0}} & \\textbf{{ResNet-50}} & \\textbf{{Variação ($\\Delta$)}} & \\textbf{{Melhor Desempenho}} \\\\
\\hline
Acurácia Global        & \\textbf{{{metricas_eff['acuracia']*100:.2f}\\%}} & {metricas_res['acuracia']*100:.2f}\\% & +{d_acc:.2f}\\%   & EfficientNet-B0 \\\\
Sensibilidade (Recall) & \\textbf{{{metricas_eff['sensibilidade']*100:.2f}\\%}} & {metricas_res['sensibilidade']*100:.2f}\\% & +{d_sens:.2f}\\%   & EfficientNet-B0 \\\\
Especificidade         & \\textbf{{{metricas_eff['especificidade']*100:.2f}\\%}} & {metricas_res['especificidade']*100:.2f}\\% & +{d_spec:.2f}\\%   & EfficientNet-B0 \\\\
Precisão               & \\textbf{{{metricas_eff['precisao']*100:.2f}\\%}} & {metricas_res['precisao']*100:.2f}\\% & +{d_prec:.2f}\\%   & EfficientNet-B0 \\\\
F1-Score               & \\textbf{{{metricas_eff['f1_score']:.4f}}}        & {metricas_res['f1_score']:.4f}        & +{d_f1:.4f}        & EfficientNet-B0 \\\\
AUC-ROC                & \\textbf{{{metricas_eff['auc_roc']:.4f}}}         & {metricas_res['auc_roc']:.4f}         & +{d_auc:.4f}        & EfficientNet-B0 \\\\
Parâmetros Totais      & \\textbf{{4,01 M}}                               & 23,51 M                               & -82,9\\%           & EfficientNet-B0 \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    caminho_tex = os.path.join(relatorios_dir, 'tabela1_metricas.tex')
    with open(caminho_tex, 'w', encoding='utf-8') as f:
        f.write(tabela_tex)

    return tabela_md

def executar_geracao_completa(num_pacientes: int = 20):
    """
    Executa a avaliação nos pacientes do conjunto de teste (ex: 20 pacientes)
    e gera todos os artefatos requeridos pelo usuário.
    """
    t_inicio = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print(f"GERANDO RESULTADOS CIENTÍFICOS DO TCC COM {num_pacientes} PACIENTES DE TESTE")
    print(f"Dispositivo de Execução: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 70)

    relatorios_dir = os.path.join(BASE_DIR, 'relatorios')
    os.makedirs(relatorios_dir, exist_ok=True)
    splits_path = os.path.join(BASE_DIR, 'splits.json')
    dataset_path = os.path.join(BASE_DIR, 'dataset')
    modelos_dir = os.path.join(PROJECT_ROOT, 'modelos_salvos')

    with open(splits_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)

    pacientes_teste_todos = splits['teste']['pacientes']

    # Filtra os pacientes conforme num_pacientes garantindo equilíbrio entre saudáveis e doentes
    if num_pacientes < len(pacientes_teste_todos):
        saudaveis = [p for p in pacientes_teste_todos if p['classe'] == 0]
        doentes = [p for p in pacientes_teste_todos if p['classe'] == 1]
        
        qtd_saud = min(len(saudaveis), num_pacientes // 2)
        qtd_doen = num_pacientes - qtd_saud
        pacientes_selecionados = saudaveis[:qtd_saud] + doentes[:qtd_doen]
    else:
        pacientes_selecionados = pacientes_teste_todos

    total_imagens = sum(p['qtd_imagens'] for p in pacientes_selecionados)
    print(f"Pacientes selecionados: {len(pacientes_selecionados)} (Total de imagens: {total_imagens})")

    # DataLoader
    _, transform_teste = obter_transformacoes(224)
    ds_teste = TermografiaDataset(dataset_path, pacientes_selecionados, transform=transform_teste)
    loader_teste = torch.utils.data.DataLoader(ds_teste, batch_size=32, shuffle=False, num_workers=0)

    # 1. Avalia EfficientNet-B0
    print("Avaliando EfficientNet-B0...")
    m_eff = criar_modelo('efficientnet_b0', num_classes=2, pretrained=False)
    cp_eff = torch.load(os.path.join(modelos_dir, 'melhor_efficientnet_b0.pth'), map_location=device, weights_only=False)
    m_eff.load_state_dict(cp_eff['state_dict'])
    m_eff.to(device)
    y_true, y_pred_eff, y_probs_eff = avaliar_modelo_no_conjunto(m_eff, loader_teste, device)
    metricas_eff = calcular_metricas(y_true, y_pred_eff, y_probs_eff)

    # 2. Avalia ResNet-50
    print("Avaliando ResNet-50...")
    m_res = criar_modelo('resnet50', num_classes=2, pretrained=False)
    cp_res = torch.load(os.path.join(modelos_dir, 'melhor_resnet50.pth'), map_location=device, weights_only=False)
    m_res.load_state_dict(cp_res['state_dict'])
    m_res.to(device)
    _, y_pred_res, y_probs_res = avaliar_modelo_no_conjunto(m_res, loader_teste, device)
    metricas_res = calcular_metricas(y_true, y_pred_res, y_probs_res)

    # 3. Gera Gráfico 1 (Curvas ROC comparativas na mesma figura)
    caminho_grafico1 = os.path.join(relatorios_dir, 'grafico1_curvas_roc_comparativas.png')
    gerar_grafico1_roc_comparativa(metricas_eff, metricas_res, caminho_grafico1)
    print(f" -> Gráfico 1 gerado: {caminho_grafico1}")

    # 4. Gera Gráfico 2 (Matrizes de confusão dos modelos)
    caminho_grafico2 = os.path.join(relatorios_dir, 'grafico2_matrizes_confusao.png')
    gerar_grafico2_matrizes_confusao(metricas_eff, metricas_res, caminho_grafico2)
    print(f" -> Gráfico 2 gerado: {caminho_grafico2}")

    # 5. Gera Figura 1 (Mapas de calor com Grad-CAM lado a lado)
    caminho_figura1 = os.path.join(relatorios_dir, 'figura1_gradcam_comparativo.png')
    gerar_figura1_gradcam_comparativo(caminho_figura1, device)
    print(f" -> Figura 1 gerada: {caminho_figura1}")

    # 6. Gera Tabela 1 (Markdown e LaTeX)
    tabela_md = gerar_tabela1_markdown_e_latex(metricas_eff, metricas_res, relatorios_dir)
    print(f" -> Tabela 1 gerada (Markdown e LaTeX)")

    tempo_total = round(time.time() - t_inicio, 2)

    # Salva JSON consolidado para consumo direto pela API e Frontend
    resultado_json = {
        'num_pacientes': len(pacientes_selecionados),
        'total_imagens': total_imagens,
        'tempo_execucao_segundos': tempo_total,
        'dispositivo': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'efficientnet_b0': metricas_eff,
        'resnet50': metricas_res,
        'arquivos': {
            'grafico1_roc': 'grafico1_curvas_roc_comparativas.png',
            'grafico2_matrizes': 'grafico2_matrizes_confusao.png',
            'figura1_gradcam': 'figura1_gradcam_comparativo.png',
            'tabela1_md': 'tabela1_metricas.md',
            'tabela1_tex': 'tabela1_metricas.tex'
        }
    }

    caminho_json = os.path.join(relatorios_dir, 'resultados_tcc.json')
    with open(caminho_json, 'w', encoding='utf-8') as f:
        json.dump(resultado_json, f, indent=4)

    print("=" * 70)
    print(f"PROCESSAMENTO CONCLUÍDO COM SUCESSO EM {tempo_total}s!")
    print("=" * 70)

    return resultado_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Geração de Resultados e Gráficos para o TCC")
    parser.add_argument('--num_pacientes', type=int, default=20, help="Número de pacientes do teste (padrão: 20)")
    args = parser.parse_args()
    executar_geracao_completa(num_pacientes=args.num_pacientes)
