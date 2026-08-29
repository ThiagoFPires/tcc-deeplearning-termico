# pyre-ignore-all-errors
import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from typing import Tuple, Optional, List, Dict, Any

# Garante a importação dos módulos locais independentemente do diretório de execução
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modelos import criar_modelo

class GradCAM:
    """
    Implementação de Grad-CAM (Gradient-weighted Class Activation Mapping)
    para explicabilidade visual de redes neurais profundas em termografia médica.
    """
    def __init__(self, modelo: nn.Module, target_layer: nn.Module) -> None:
        self.modelo: nn.Module = modelo
        self.target_layer: nn.Module = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.hook_handles: List[Any] = []
        self._registrar_hooks()

    def _registrar_hooks(self) -> None:
        def forward_hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            self.activations = output.detach()

        def backward_hook(module: nn.Module, grad_in: Any, grad_out: Tuple[torch.Tensor, ...]) -> None:
            if len(grad_out) > 0 and grad_out[0] is not None:
                self.gradients = grad_out[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles.extend([h1, h2])

    def remover_hooks(self) -> None:
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

    def gerar_mapa(
        self,
        imagem_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Gera o mapa de calor Grad-CAM (2D normalizado entre 0 e 1)
        junto com a classe predita e a probabilidade de confiança.
        """
        self.modelo.eval()
        self.modelo.zero_grad()

        # Forward pass
        logits = self.modelo(imagem_tensor)
        probs = torch.softmax(logits, dim=1)[0].detach()
        
        pred_class = int(torch.argmax(logits, dim=1)[0].item())
        confianca = float(probs[pred_class].item())

        if target_class is None:
            target_class = pred_class

        # Backward pass para a classe alvo
        score = logits[0, target_class]
        score.backward()

        if self.gradients is None or self.activations is None:
            return np.zeros((224, 224), dtype=np.float32), pred_class, confianca

        # Cálculo dos pesos alpha através da média global dos gradientes
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        for i in range(activations.size(0)):
            activations[i, :, :] *= pooled_gradients[i]

        # Combinação linear ponderada com ReLU
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        
        # Normalização entre 0 e 1
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap, pred_class, confianca

def sobrepor_heatmap(
    imagem_pil: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = 'jet'
) -> np.ndarray:
    """
    Redimensiona o heatmap para o tamanho da imagem original e aplica
    o mapa de cores sobreposto à foto original.
    """
    img_array = np.array(imagem_pil.convert('RGB')) / 255.0
    h, w = img_array.shape[:2]

    # Redimensiona o heatmap usando matplotlib/PIL
    heatmap_pil = Image.fromarray(np.uint8(255 * heatmap)).resize((w, h), Image.Resampling.BICUBIC)
    heatmap_resized = np.array(heatmap_pil) / 255.0

    # Aplica Colormap
    cmap = plt.get_cmap(colormap_name)
    heatmap_color = cmap(heatmap_resized)[:, :, :3]

    # Sobreposição (Overlay)
    overlay = (1.0 - alpha) * img_array + alpha * heatmap_color
    overlay = np.clip(overlay, 0.0, 1.0)

    return overlay

def executar_analise_gradcam_em_lote(
    nome_modelo: str = 'efficientnet_b0',
    num_amostras: int = 6
) -> None:
    """
    Executa a análise de explicabilidade em lote para termogramas de teste
    e salva uma figura de alta resolução para o TCC.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    splits_path = os.path.join(base_dir, 'splits.json')
    dataset_path = os.path.join(base_dir, 'dataset')
    modelos_dir = os.path.join(project_root, 'modelos_salvos')
    relatorios_dir = os.path.join(base_dir, 'relatorios')
    os.makedirs(relatorios_dir, exist_ok=True)

    caminho_modelo = os.path.join(modelos_dir, f"melhor_{nome_modelo}.pth")
    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(f"Modelo não encontrado em: {caminho_modelo}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Carrega modelo e pesos
    modelo = criar_modelo(nome_modelo=nome_modelo, num_classes=2, pretrained=False)
    checkpoint = torch.load(caminho_modelo, map_location=device, weights_only=False)
    modelo.load_state_dict(checkpoint['state_dict'])
    modelo.to(device)
    modelo.eval()

    # Define a camada alvo para o Grad-CAM
    if nome_modelo == 'efficientnet_b0':
        target_layer = modelo.features[-1]  # Último bloco convolucional da EfficientNet
    else:
        target_layer = modelo.layer4[-1]    # Último bloco Bottleneck da ResNet-50

    gradcam = GradCAM(modelo, target_layer)

    # Carrega splits de teste
    with open(splits_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)

    pacientes_teste = splits['teste']['pacientes']
    
    # Seleciona metade saudáveis e metade doentes para ilustrar o comparativo
    saudaveis = [p for p in pacientes_teste if p['classe'] == 0]
    doentes = [p for p in pacientes_teste if p['classe'] == 1]
    
    qtd_cada = max(1, num_amostras // 2)
    selecionados = saudaveis[:qtd_cada] + doentes[:qtd_cada]

    # Transformação padrão
    transform_norm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    classes_nomes = {0: 'Saudável', 1: 'Doente (Anomalia Térmica)'}

    fig, axes = plt.subplots(len(selecionados), 3, figsize=(13, 3.5 * len(selecionados)), dpi=300)

    for i, p_info in enumerate(selecionados):
        p_id = str(p_info['id'])
        p_classe = int(p_info['classe'])
        subpasta = 'saudavel' if p_classe == 0 else 'doente'
        pasta_p = os.path.join(dataset_path, subpasta, p_id)
        
        # Pega a primeira imagem do paciente
        imgs = [f for f in os.listdir(pasta_p) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        caminho_img = os.path.join(pasta_p, imgs[0])
        
        img_pil = Image.open(caminho_img).convert('RGB')
        img_tensor = transform_norm(img_pil).unsqueeze(0).to(device)

        heatmap, pred_classe, confianca = gradcam.gerar_mapa(img_tensor)
        overlay = sobrepor_heatmap(img_pil, heatmap, alpha=0.45, colormap_name='jet')

        status_acerto = "CORRETO" if pred_classe == p_classe else "ERRO"
        cor_titulo = "#008000" if status_acerto == "CORRETO" else "#D00000"

        # 1. Imagem Original
        axes[i, 0].imshow(img_pil)
        axes[i, 0].set_title(f"Paciente #{p_id}\nDiagnóstico Real: {classes_nomes[p_classe]}", fontsize=10, fontweight='bold')
        axes[i, 0].axis('off')

        # 2. Mapa Grad-CAM puro
        axes[i, 1].imshow(heatmap, cmap='jet')
        axes[i, 1].set_title("Grad-CAM da IA\n(Intensidade de Atenção)", fontsize=10, fontweight='bold')
        axes[i, 1].axis('off')

        # 3. Sobreposição Explicável
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(
            f"Previsão IA: {classes_nomes[pred_classe]} ({confianca*100:.1f}%)\n[{status_acerto}]",
            fontsize=10,
            fontweight='bold',
            color=cor_titulo
        )
        axes[i, 2].axis('off')

    plt.suptitle(
        f"Explicabilidade Visual com Grad-CAM ({nome_modelo.upper()})\nAnálise de Atenção em Termogramas Mamários Inéditos",
        fontsize=14,
        fontweight='bold',
        y=0.995
    )
    plt.tight_layout()

    caminho_saida = os.path.join(relatorios_dir, f"gradcam_explicabilidade_{nome_modelo}.png")
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close(fig)
    gradcam.remover_hooks()

    print("\n" + "=" * 60)
    print("ANÁLISE GRAD-CAM GERADA COM SUCESSO EM ALTA RESOLUÇÃO (300 DPI)")
    print(f"Salvo em: {caminho_saida}")
    print("=" * 60)

if __name__ == '__main__':
    executar_analise_gradcam_em_lote(nome_modelo='efficientnet_b0', num_amostras=6)
