# pyre-ignore-all-errors
import os
import sys
import io
import base64
import time
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

# Adiciona pastas necessárias ao path
API_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(API_DIR)
TREINAMENTO_DIR = os.path.join(PROJECT_ROOT, 'treinamento')
MODELOS_DIR = os.path.join(PROJECT_ROOT, 'modelos_salvos')

sys.path.insert(0, TREINAMENTO_DIR)
from modelos import criar_modelo
from gradcam import GradCAM, sobrepor_heatmap

app = FastAPI(
    title="API - Classificador Térmico Mamário",
    description="Serviço de Inferência e Explicabilidade Visual (Grad-CAM) para Detecção de Patologias em Termogramas Mamários via Deep Learning.",
    version="1.0.0"
)

# Habilita CORS total para conexão com a Interface Web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variáveis globais do modelo
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELO = None
GRADCAM = None

TRANSFORMACOES = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.on_event("startup")
def carregar_modelo_ao_iniciar():
    global MODELO, GRADCAM
    caminho_pesos = os.path.join(MODELOS_DIR, "melhor_efficientnet_b0.pth")
    
    if not os.path.exists(caminho_pesos):
        print(f"[AVISO] Pesos do modelo não encontrados em {caminho_pesos}")
        return

    print(f"Carregando EfficientNet-B0 no dispositivo: {DEVICE}...")
    MODELO = criar_modelo(nome_modelo='efficientnet_b0', num_classes=2, pretrained=False)
    checkpoint = torch.load(caminho_pesos, map_location=DEVICE, weights_only=False)
    MODELO.load_state_dict(checkpoint['state_dict'])
    MODELO.to(DEVICE)
    MODELO.eval()

    # Inicializa Grad-CAM na última camada convolucional
    target_layer = MODELO.features[-1]
    GRADCAM = GradCAM(MODELO, target_layer)
    print("Modelo e Grad-CAM carregados e prontos para inferência!")

@app.get("/")
def raiz():
    return {
        "sistema": "Classificador Térmico Mamário IA",
        "status": "online",
        "dispositivo": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "modelo_carregado": MODELO is not None
    }

@app.get("/api/info")
def informacoes_modelo():
    return {
        "arquitetura": "EfficientNet-B0",
        "acuracia_teste": 93.54,
        "sensibilidade_recall": 95.00,
        "especificidade": 91.50,
        "precisao": 93.99,
        "f1_score": 0.9449,
        "auc_roc": 0.9865,
        "parametros_totais": "4,01 Milhões",
        "dataset": "DMR-IR (Database for Mastology Research with Infrared Image)",
        "aceleracao_hardware": "NVIDIA GeForce RTX 5060 (CUDA 12.8)"
    }

@app.post("/api/diagnosticar")
async def diagnosticar_termograma(arquivo: UploadFile = File(...)):
    global MODELO, GRADCAM
    if MODELO is None or GRADCAM is None:
        carregar_modelo_ao_iniciar()
        if MODELO is None:
            raise HTTPException(status_code=500, detail="Modelo neural não carregado.")

    # Validação de formato
    ext = os.path.splitext(arquivo.filename)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
        raise HTTPException(status_code=400, detail="Formato de imagem inválido. Use JPG ou PNG.")

    try:
        t_inicio = time.time()
        conteudo_bytes = await arquivo.read()
        imagem_pil = Image.open(io.BytesIO(conteudo_bytes)).convert('RGB')

        # Pré-processamento
        imagem_tensor = TRANSFORMACOES(imagem_pil).unsqueeze(0).to(DEVICE)

        # Inferência com Grad-CAM
        with torch.enable_grad():
            heatmap, pred_classe, confianca = GRADCAM.gerar_mapa(imagem_tensor)

        # Cálculo de probabilidades das 2 classes
        with torch.no_grad():
            logits = MODELO(imagem_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            prob_saudavel = float(probs[0]) * 100
            prob_doente = float(probs[1]) * 100

        # Gera Sobreposição Grad-CAM
        overlay_array = sobrepor_heatmap(imagem_pil, heatmap, alpha=0.45, colormap_name='jet')
        overlay_pil = Image.fromarray((overlay_array * 255).astype(np.uint8))

        # Converte a imagem de sobreposição para Base64 para envio direto ao frontend
        buffer = io.BytesIO()
        overlay_pil.save(buffer, format="PNG")
        overlay_b64 = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode('utf-8')

        tempo_inferencia_ms = round((time.time() - t_inicio) * 1000, 2)

        diagnostico_texto = "Doente (Anomalia Térmica Detectada)" if pred_classe == 1 else "Saudável (Sem Anomalias)"
        nivel_alerta = "alto" if (pred_classe == 1 and confianca >= 0.8) else ("medio" if pred_classe == 1 else "baixo")

        return {
            "status": "sucesso",
            "tempo_ms": tempo_inferencia_ms,
            "arquivo": arquivo.filename,
            "predicao": {
                "classe_id": int(pred_classe),
                "diagnostico": diagnostico_texto,
                "confianca_percentual": round(confianca * 100, 2),
                "probabilidade_saudavel": round(prob_saudavel, 2),
                "probabilidade_doente": round(prob_doente, 2),
                "nivel_alerta": nivel_alerta
            },
            "gradcam_base64": overlay_b64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar imagem: {str(e)}")

@app.get("/api/exemplos")
def listar_exemplos():
    """Retorna uma lista de imagens de exemplo para testes na interface."""
    dataset_dir = os.path.join(TREINAMENTO_DIR, 'dataset')
    exemplos = []

    for classe, nome in [('saudavel', 'Saudável (Classe 0)'), ('doente', 'Doente (Classe 1)')]:
        caminho_classe = os.path.join(dataset_dir, classe)
        if os.path.exists(caminho_classe):
            pastas = [p for p in os.listdir(caminho_classe) if os.path.isdir(os.path.join(caminho_classe, p))]
            if pastas:
                # Pega 2 pacientes de cada classe
                for p in pastas[:2]:
                    pasta_p = os.path.join(caminho_classe, p)
                    imgs = [f for f in os.listdir(pasta_p) if f.lower().endswith(('.jpg', '.png'))]
                    if imgs:
                        exemplos.append({
                            "categoria": classe,
                            "paciente_id": p,
                            "arquivo": imgs[0],
                            "rotulo_esperado": nome,
                            "url": f"/api/amostra/{classe}/{p}/{imgs[0]}"
                        })

    return {"exemplos": exemplos}

@app.get("/api/amostra/{classe}/{paciente_id}/{arquivo}")
def obter_amostra(classe: str, paciente_id: str, arquivo: str):
    caminho = os.path.join(TREINAMENTO_DIR, 'dataset', classe, paciente_id, arquivo)
    if os.path.exists(caminho):
        return FileResponse(caminho)
    raise HTTPException(status_code=404, detail="Amostra não encontrada.")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
