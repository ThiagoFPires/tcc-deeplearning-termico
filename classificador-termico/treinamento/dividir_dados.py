# pyre-ignore-all-errors
import os
import json
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split

def gerar_splits_estratificados(
    dataset_dir: str,
    output_json: str,
    proporcoes=(0.70, 0.15, 0.15),
    seed=42
):
    """
    Realiza a divisão estratificada a nível de PACIENTE (evita data leakage).
    Proporções padrão: 70% Treino, 15% Validação, 15% Teste.
    """
    random.seed(seed)
    
    classes = {'saudavel': 0, 'doente': 1}
    pacientes_por_classe = defaultdict(list)
    contagem_imagens = defaultdict(dict)
    
    for nome_classe, rotulo in classes.items():
        caminho_classe = os.path.join(dataset_dir, nome_classe)
        if not os.path.exists(caminho_classe):
            raise FileNotFoundError(f"Diretório não encontrado: {caminho_classe}")
            
        pastas_pacientes = [
            f for f in os.listdir(caminho_classe) 
            if os.path.isdir(os.path.join(caminho_classe, f))
        ]
        
        for pasta in pastas_pacientes:
            caminho_paciente = os.path.join(caminho_classe, pasta)
            imgs = [
                img for img in os.listdir(caminho_paciente)
                if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ]
            pacientes_por_classe[rotulo].append(pasta)
            contagem_imagens[rotulo][pasta] = len(imgs)
            
    splits = {
        'treino': {'pacientes': [], 'rotulos': [], 'total_imagens': 0},
        'validacao': {'pacientes': [], 'rotulos': [], 'total_imagens': 0},
        'teste': {'pacientes': [], 'rotulos': [], 'total_imagens': 0}
    }
    
    # Divisão estratificada por classe
    p_treino, p_val, p_teste = proporcoes
    val_teste_ratio = p_val + p_teste  # 0.30
    teste_rel_ratio = p_teste / val_teste_ratio  # 0.15 / 0.30 = 0.50
    
    for rotulo, lista_pacientes in pacientes_por_classe.items():
        # Primeiro separa Treino (70%) e Resto (30%)
        treino_p, resto_p = train_test_split(
            lista_pacientes, 
            test_size=val_teste_ratio, 
            random_state=seed,
            shuffle=True
        )
        # Depois divide o Resto igualmente entre Validação (15%) e Teste (15%)
        val_p, teste_p = train_test_split(
            resto_p, 
            test_size=teste_rel_ratio, 
            random_state=seed,
            shuffle=True
        )
        
        for p in treino_p:
            splits['treino']['pacientes'].append({'id': p, 'classe': rotulo, 'qtd_imagens': contagem_imagens[rotulo][p]})
            splits['treino']['total_imagens'] += contagem_imagens[rotulo][p]
            
        for p in val_p:
            splits['validacao']['pacientes'].append({'id': p, 'classe': rotulo, 'qtd_imagens': contagem_imagens[rotulo][p]})
            splits['validacao']['total_imagens'] += contagem_imagens[rotulo][p]
            
        for p in teste_p:
            splits['teste']['pacientes'].append({'id': p, 'classe': rotulo, 'qtd_imagens': contagem_imagens[rotulo][p]})
            splits['teste']['total_imagens'] += contagem_imagens[rotulo][p]

    # Salva o arquivo JSON para garantir reprodutibilidade
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=4, ensure_ascii=False)
        
    print("=" * 60)
    print("DIVISÃO ESTRATIFICADA POR PACIENTE CONCLUÍDA COM SUCESSO")
    print("=" * 60)
    
    for nome_split in ['treino', 'validacao', 'teste']:
        dados_split = splits[nome_split]
        qtd_pacientes = len(dados_split['pacientes'])
        qtd_saudaveis = sum(1 for p in dados_split['pacientes'] if p['classe'] == 0)
        qtd_doentes = sum(1 for p in dados_split['pacientes'] if p['classe'] == 1)
        print(f"[{nome_split.upper()}]")
        print(f" - Pacientes: {qtd_pacientes} (Saudáveis: {qtd_saudaveis}, Doentes: {qtd_doentes})")
        print(f" - Imagens Térmicas: {dados_split['total_imagens']}")
        print("-" * 60)
        
    print(f"Arquivo de particionamento salvo em: {output_json}\n")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, 'dataset')
    output_path = os.path.join(base_dir, 'splits.json')
    gerar_splits_estratificados(dataset_path, output_path)
