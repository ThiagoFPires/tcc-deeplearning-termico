# pyre-ignore-all-errors
import os
import json
import matplotlib.pyplot as plt
import numpy as np

def gerar_graficos_e_relatorio_comparativo(treinamento_dir: str):
    """
    Gera gráficos comparativos em alta resolução (300 DPI) para inserção direta
    no documento do TCC e compila uma tabela comparativa detalhada.
    """
    relatorios_dir = os.path.join(treinamento_dir, 'relatorios')
    os.makedirs(relatorios_dir, exist_ok=True)

    caminho_hist_eff = os.path.join(treinamento_dir, 'historico_efficientnet_b0.json')
    caminho_hist_res = os.path.join(treinamento_dir, 'historico_resnet50.json')
    caminho_teste_eff = os.path.join(treinamento_dir, 'teste_efficientnet_b0.json')
    caminho_teste_res = os.path.join(treinamento_dir, 'teste_resnet50.json')

    if not (os.path.exists(caminho_teste_eff) and os.path.exists(caminho_teste_res)):
        print("Aviso: É necessário ter treinado e avaliado ambos os modelos para gerar o comparativo completo.")
        return

    with open(caminho_hist_eff, 'r', encoding='utf-8') as f:
        hist_eff = json.load(f)
    with open(caminho_hist_res, 'r', encoding='utf-8') as f:
        hist_res = json.load(f)
    with open(caminho_teste_eff, 'r', encoding='utf-8') as f:
        teste_eff = json.load(f)
    with open(caminho_teste_res, 'r', encoding='utf-8') as f:
        teste_res = json.load(f)

    plt.style.use('default')

    # 1. Gráfico de Curvas de Aprendizado (Loss e Acurácia)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    # Loss
    axes[0].plot(hist_eff['treino_loss'], label='EfficientNet-B0 (Treino)', color='#1f77b4', linestyle='--')
    axes[0].plot(hist_eff['val_loss'], label='EfficientNet-B0 (Validação)', color='#1f77b4', linewidth=2)
    axes[0].plot(hist_res['treino_loss'], label='ResNet-50 (Treino)', color='#ff7f0e', linestyle='--')
    axes[0].plot(hist_res['val_loss'], label='ResNet-50 (Validação)', color='#ff7f0e', linewidth=2)
    axes[0].set_title('Convergência da Função de Perda (Loss)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Acurácia
    axes[1].plot([a*100 for a in hist_eff['val_acc']], label='EfficientNet-B0 (Validação)', color='#1f77b4', linewidth=2)
    axes[1].plot([a*100 for a in hist_res['val_acc']], label='ResNet-50 (Validação)', color='#ff7f0e', linewidth=2)
    axes[1].set_title('Evolução da Acurácia de Validação (%)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Acurácia (%)')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    grafico_curvas = os.path.join(relatorios_dir, 'comparativo_curvas_aprendizado.png')
    plt.savefig(grafico_curvas, dpi=300)
    plt.close(fig)

    # 2. Matrizes de Confusão Lado a Lado
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    
    for ax, modelo_nome, m_teste, cor in zip(
        axes,
        ['EfficientNet-B0', 'ResNet-50'],
        [teste_eff, teste_res],
        ['Blues', 'Oranges']
    ):
        mc = m_teste['matriz_confusao']
        matriz = np.array([[mc['tn'], mc['fp']], [mc['fn'], mc['tp']]])
        ax.imshow(matriz, interpolation='nearest', cmap=cor)
        ax.set_title(f'Matriz de Confusão - {modelo_nome}\n(Conjunto de Teste: 480 Imagens)', fontsize=11, fontweight='bold')
        
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(['Saudável (0)', 'Doente (1)'], fontsize=10)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(['Saudável (0)', 'Doente (1)'], fontsize=10)
        ax.set_xlabel('Rótulo Predito pelo Modelo', fontsize=10)
        ax.set_ylabel('Rótulo Real (Diagnóstico)', fontsize=10)

        thresh = matriz.max() / 2.
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matriz[i, j]}",
                        horizontalalignment="center",
                        color="white" if matriz[i, j] > thresh else "black",
                        fontsize=12, fontweight='bold')

    plt.tight_layout()
    grafico_matrizes = os.path.join(relatorios_dir, 'comparativo_matrizes_confusao.png')
    plt.savefig(grafico_matrizes, dpi=300)
    plt.close(fig)

    # 3. Gráfico de Barras Comparativo das Métricas Clínicas
    metricas_nomes = ['Acurácia', 'Sensibilidade\n(Recall)', 'Especificidade', 'Precisão', 'F1-Score', 'AUC-ROC']
    valores_eff = [
        teste_eff['acuracia'] * 100,
        teste_eff['sensibilidade'] * 100,
        teste_eff['especificidade'] * 100,
        teste_eff['precisao'] * 100,
        teste_eff['f1_score'] * 100,
        teste_eff['auc_roc'] * 100
    ]
    valores_res = [
        teste_res['acuracia'] * 100,
        teste_res['sensibilidade'] * 100,
        teste_res['especificidade'] * 100,
        teste_res['precisao'] * 100,
        teste_res['f1_score'] * 100,
        teste_res['auc_roc'] * 100
    ]

    x = np.arange(len(metricas_nomes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    rects1 = ax.bar(x - width/2, valores_eff, width, label='EfficientNet-B0 (Principal)', color='#1f77b4', edgecolor='black', alpha=0.85)
    rects2 = ax.bar(x + width/2, valores_res, width, label='ResNet-50 (Comparativo)', color='#ff7f0e', edgecolor='black', alpha=0.85)

    ax.set_ylabel('Pontuação (%)', fontsize=11, fontweight='bold')
    ax.set_title('Comparativo Final no Conjunto de Teste Não Visto (TCC)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metricas_nomes, fontsize=10)
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)

    ax.bar_label(rects1, fmt='%.1f%%', padding=3, fontsize=9, fontweight='bold')
    ax.bar_label(rects2, fmt='%.1f%%', padding=3, fontsize=9, fontweight='bold')

    plt.tight_layout()
    grafico_barras = os.path.join(relatorios_dir, 'comparativo_metricas_barras.png')
    plt.savefig(grafico_barras, dpi=300)
    plt.close(fig)

    print("\n" + "=" * 60)
    print("GRÁFICOS COMPARATIVOS GERADOS EM ALTA RESOLUÇÃO (300 DPI)")
    print("=" * 60)
    print(f"1. {grafico_curvas}")
    print(f"2. {grafico_matrizes}")
    print(f"3. {grafico_barras}")
    print("=" * 60)

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    gerar_graficos_e_relatorio_comparativo(base_dir)
