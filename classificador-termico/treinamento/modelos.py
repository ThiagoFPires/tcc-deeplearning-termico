# pyre-ignore-all-errors
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights, 
    ResNet50_Weights
)
from typing import Dict, Any

def criar_modelo(
    nome_modelo: str = 'efficientnet_b0',
    num_classes: int = 2,
    pretrained: bool = True,
    dropout_rate: float = 0.3
) -> nn.Module:
    """
    Instancia a arquitetura selecionada com pesos pré-treinados do ImageNet
    e adapta a camada de saída final para a classificação binária de termografia.
    """
    nome_modelo = nome_modelo.lower()

    if nome_modelo == 'efficientnet_b0':
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        modelo = models.efficientnet_b0(weights=weights)
        
        # In_features da EfficientNet-B0 no bloco classificador (nn.Sequential)
        classifier_layer = modelo.classifier[1]
        in_features = getattr(classifier_layer, 'in_features', 1280)
        
        modelo.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    elif nome_modelo == 'resnet50':
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        modelo = models.resnet50(weights=weights)
        
        # In_features da camada fc da ResNet-50
        in_features = getattr(modelo.fc, 'in_features', 2048)
        
        modelo.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    else:
        raise ValueError(
            f"Modelo '{nome_modelo}' não suportado. Opções válidas: 'efficientnet_b0', 'resnet50'"
        )

    return modelo

def obter_resumo_modelo(modelo: nn.Module, nome_modelo: str) -> Dict[str, Any]:
    """
    Retorna métricas de complexidade da rede (parâmetros totais e treináveis).
    """
    total_params = sum(p.numel() for p in modelo.parameters())
    trainable_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    tamanho_mb = (total_params * 4) / (1024 ** 2)

    return {
        'nome': nome_modelo,
        'total_parametros': total_params,
        'parametros_treinaveis': trainable_params,
        'tamanho_estimado_mb': round(tamanho_mb, 2)
    }

if __name__ == '__main__':
    for nome in ['efficientnet_b0', 'resnet50']:
        m = criar_modelo(nome, num_classes=2)
        info = obter_resumo_modelo(m, nome)
        print("=" * 50)
        print(f"Arquitetura: {info['nome'].upper()}")
        print(f" - Total de Parâmetros: {info['total_parametros']:,}")
        print(f" - Parâmetros Treináveis: {info['parametros_treinaveis']:,}")
        print(f" - Tamanho Estimado: {info['tamanho_estimado_mb']} MB")
