# pyre-ignore-all-errors
import os
import sys
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from typing import Tuple, Optional

def segmentar_roi_termica(
    imagem_pil: Image.Image,
    limiar_relativo: float = 0.15,
    margem_seguranca: float = 0.05
) -> Tuple[Image.Image, np.ndarray, Tuple[int, int, int, int]]:
    """
    Segmenta automaticamente a Região de Interesse (ROI) das mamas em termogramas,
    isolando o tecido térmico ativo e eliminando o fundo frio/ambiente.
    
    Retorna:
    - imagem_recortada_pil: Imagem recortada na ROI das mamas.
    - mascara_binaria: Array 2D (0 e 1) da máscara segmentada.
    - bbox: Tupla (xmin, ymin, xmax, ymax) com as coordenadas de recorte.
    """
    img_rgb = imagem_pil.convert('RGB')
    largura, altura = img_rgb.size
    
    # Converte para escala de cinza e array numpy
    img_gray = img_rgb.convert('L')
    arr_gray = np.array(img_gray, dtype=np.float32) / 255.0

    # Suavização Gaussiana para reduzir ruídos pontuais do sensor térmico
    img_blur = img_gray.filter(ImageFilter.GaussianBlur(radius=3))
    arr_blur = np.array(img_blur, dtype=np.float32) / 255.0

    # Limiarização térmica baseada em Otsu simplificado / limiar adaptativo
    val_min = np.min(arr_blur)
    val_max = np.max(arr_blur)
    limiar = val_min + limiar_relativo * (val_max - val_min)
    
    mascara_binaria = (arr_blur > limiar).astype(np.uint8)

    # Identificação das coordenadas ativas do corpo
    coords_y, coords_x = np.where(mascara_binaria > 0)
    
    if len(coords_x) == 0 or len(coords_y) == 0:
        # Fallback caso a imagem seja muito escura
        return img_rgb, mascara_binaria, (0, 0, largura, altura)

    xmin, xmax = np.min(coords_x), np.max(coords_x)
    ymin, ymax = np.min(coords_y), np.max(coords_y)

    # Adiciona margem de segurança
    pad_x = int((xmax - xmin) * margem_seguranca)
    pad_y = int((ymax - ymin) * margem_seguranca)

    xmin_crop = max(0, xmin - pad_x)
    xmax_crop = min(largura, xmax + pad_x)
    ymin_crop = max(0, ymin - pad_y)
    ymax_crop = min(altura, ymax + pad_y)

    bbox = (xmin_crop, ymin_crop, xmax_crop, ymax_crop)
    imagem_recortada = img_rgb.crop(bbox)

    return imagem_recortada, mascara_binaria, bbox

def gerar_figura_demonstrativa_segmentacao(
    caminho_amostra: str,
    caminho_saida: str
) -> None:
    """Gera uma figura em 300 DPI demonstrando as etapas da segmentação de ROI."""
    img_original = Image.open(caminho_amostra).convert('RGB')
    img_roi, mascara, bbox = segmentar_roi_termica(img_original)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=300)

    # 1. Imagem Original com Bounding Box
    arr_orig = np.array(img_original)
    axes[0].imshow(arr_orig)
    xmin, ymin, xmax, ymax = bbox
    rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                         fill=False, edgecolor='#EF4444', linewidth=2, linestyle='--')
    axes[0].add_patch(rect)
    axes[0].set_title("1. Termograma com BBox da ROI", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # 2. Máscara Térmica Binária
    axes[1].imshow(mascara, cmap='gray')
    axes[1].set_title("2. Máscara Térmica Filtrada", fontsize=11, fontweight='bold')
    axes[1].axis('off')

    # 3. ROI Isolada e Redimensionada
    axes[2].imshow(img_roi)
    axes[2].set_title(f"3. ROI Mamária Segmentada\n({img_roi.size[0]}x{img_roi.size[1]} px)", fontsize=11, fontweight='bold')
    axes[2].axis('off')

    plt.suptitle("Segmentação Automática de Região de Interesse (ROI) em Termografia Mamária", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("Demonstração de segmentação de ROI gerada com sucesso em:", caminho_saida)

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    amostra = os.path.join(base_dir, 'dataset', 'doente', '144', 'T0144.1.1.D.2013-01-29.00.jpg')
    saida = os.path.join(base_dir, 'relatorios', 'segmentacao_roi_exemplo.png')

    if os.path.exists(amostra):
        gerar_figura_demonstrativa_segmentacao(amostra, saida)
