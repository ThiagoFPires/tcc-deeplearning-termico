# pyre-ignore-all-errors
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Tuple, Dict, List, Optional, Any

class TermografiaDataset(Dataset):
    """
    Dataset customizado para imagens termográficas mamárias.
    Carrega as imagens com base nos IDs de pacientes definidos no splits.json.
    """
    def __init__(
        self,
        dataset_dir: str,
        lista_pacientes: List[Dict[str, Any]],
        transform: Optional[transforms.Compose] = None
    ) -> None:
        super().__init__()
        self.dataset_dir: str = dataset_dir
        self.transform: Optional[transforms.Compose] = transform
        self.amostras: List[Tuple[str, int, str]] = []  # (caminho_imagem, rotulo_classe, id_paciente)
        
        classes_map = {0: 'saudavel', 1: 'doente'}
        
        for p in lista_pacientes:
            p_id = str(p.get('id', ''))
            p_classe = int(p.get('classe', 0))
            subpasta_classe = classes_map.get(p_classe, 'saudavel')
            caminho_pasta_paciente = os.path.join(dataset_dir, subpasta_classe, p_id)
            
            if os.path.isdir(caminho_pasta_paciente):
                imgs = [
                    f for f in os.listdir(caminho_pasta_paciente)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
                ]
                for img_nome in imgs:
                    caminho_completo = os.path.join(caminho_pasta_paciente, img_nome)
                    self.amostras.append((caminho_completo, p_classe, p_id))

    def __len__(self) -> int:
        return len(self.amostras)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        caminho_img, rotulo, paciente_id = self.amostras[idx]
        imagem_pil = Image.open(caminho_img).convert('RGB')
        
        if self.transform is not None:
            imagem_tensor = self.transform(imagem_pil)
        else:
            imagem_tensor = transforms.ToTensor()(imagem_pil)
            
        return imagem_tensor, rotulo, paciente_id

def obter_transformacoes(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Retorna os pipelines de transformação e data augmentation adequados
    para termografia médica (sem distorções que alterem o gradiente térmico).
    """
    # Médias e Desvios padrão pré-definidos do ImageNet para modelos pré-treinados
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    transform_treino = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    transform_val_teste = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    return transform_treino, transform_val_teste

def obter_dataloaders(
    splits_json: str,
    dataset_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Cria DataLoaders para Treino, Validação e Teste, além de calcular
    os pesos de classe (class weights) para compensar desbalanceamentos.
    """
    with open(splits_json, 'r', encoding='utf-8') as f:
        splits = json.load(f)

    transform_treino, transform_val_teste = obter_transformacoes(img_size)

    ds_treino = TermografiaDataset(dataset_dir, splits['treino']['pacientes'], transform=transform_treino)
    ds_val = TermografiaDataset(dataset_dir, splits['validacao']['pacientes'], transform=transform_val_teste)
    ds_teste = TermografiaDataset(dataset_dir, splits['teste']['pacientes'], transform=transform_val_teste)

    loader_treino = DataLoader(ds_treino, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    loader_teste = DataLoader(ds_teste, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Cálculo dos pesos de classe para a Loss Function (Balanceamento)
    rotulos_treino = [amostra[1] for amostra in ds_treino.amostras]
    contagem_0 = rotulos_treino.count(0)
    contagem_1 = rotulos_treino.count(1)
    total = len(rotulos_treino)

    # Pesos inversamente proporcionais às frequências
    peso_0 = total / (2.0 * contagem_0) if contagem_0 > 0 else 1.0
    peso_1 = total / (2.0 * contagem_1) if contagem_1 > 0 else 1.0
    class_weights = torch.tensor([peso_0, peso_1], dtype=torch.float)

    return loader_treino, loader_val, loader_teste, class_weights

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    splits_path = os.path.join(base_dir, 'splits.json')
    dataset_path = os.path.join(base_dir, 'dataset')

    loader_treino, loader_val, loader_teste, pesos = obter_dataloaders(splits_path, dataset_path)
    print("DataLoaders criados com sucesso!")
    print(f" - Lotes de Treino: {len(loader_treino)} (Total imagens: {len(loader_treino.dataset)})")
    print(f" - Lotes de Validação: {len(loader_val)} (Total imagens: {len(loader_val.dataset)})")
    print(f" - Lotes de Teste: {len(loader_teste)} (Total imagens: {len(loader_teste.dataset)})")
    print(f" - Pesos de Classe calculados: {pesos.tolist()}")
