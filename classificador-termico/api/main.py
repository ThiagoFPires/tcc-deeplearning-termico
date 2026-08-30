# pyre-ignore-all-errors
import os
import sys
import io
import base64
import time
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
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
sys.path.insert(0, API_DIR)

from modelos import criar_modelo
from gradcam import GradCAM, sobrepor_heatmap
from banco import inicializar_banco, salvar_exame, listar_historico, limpar_historico

app = FastAPI(
    title="DeepVision CADe API",
    description="Serviço de Detecção Assistida por Computador (CADe) e Explicabilidade Visual (Grad-CAM) para Termografia Mamária.",
    version="2.2.0"
)


# Habilita CORS total para conexão com a Interface Web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dicionários globais de modelos e Grad-CAMs
MODELOS = {}
GRADCAMS = {}

TRANSFORMACOES = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def carregar_modelos():
    global MODELOS, GRADCAMS
    inicializar_banco()
    
    modelos_info = [
        ('efficientnet_b0', 'melhor_efficientnet_b0.pth'),
        ('resnet50', 'melhor_resnet50.pth')
    ]

    for nome, arquivo in modelos_info:
        caminho = os.path.join(MODELOS_DIR, arquivo)
        if os.path.exists(caminho):
            print(f"Carregando {nome.upper()} no dispositivo: {DEVICE}...")
            m = criar_modelo(nome_modelo=nome, num_classes=2, pretrained=False)
            checkpoint = torch.load(caminho, map_location=DEVICE, weights_only=False)
            m.load_state_dict(checkpoint['state_dict'])
            m.to(DEVICE)
            m.eval()
            MODELOS[nome] = m

            # Configura Grad-CAM para cada modelo
            target_layer = m.features[-1] if nome == 'efficientnet_b0' else m.layer4[-1]
            GRADCAMS[nome] = GradCAM(m, target_layer)
            print(f" -> {nome.upper()} pronto!")

@app.on_event("startup")
def startup_event():
    carregar_modelos()

@app.get("/")
def raiz():
    return {
        "sistema": "ThermoScan Clinical Suite",
        "status": "online",
        "dispositivo": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "modelos_disponiveis": list(MODELOS.keys())
    }

@app.get("/api/info")
def informacoes_modelos():
    return {
        "efficientnet_b0": {
            "nome_exibicao": "EfficientNet-B0 (Modelo Proposto)",
            "acuracia_teste": 93.54,
            "sensibilidade": 95.00,
            "especificidade": 91.50,
            "precisao": 93.99,
            "f1_score": 0.9449,
            "auc_roc": 0.9865,
            "parametros": "4,01 M",
            "destaque": "Mais leve, rápido e com maior sensibilidade clínica."
        },
        "resnet50": {
            "nome_exibicao": "ResNet-50 (Baseline Comparativo)",
            "acuracia_teste": 90.62,
            "sensibilidade": 92.86,
            "especificidade": 87.50,
            "precisao": 91.23,
            "f1_score": 0.9204,
            "auc_roc": 0.9555,
            "parametros": "23,51 M",
            "destaque": "Arquitetura clássica com blocos residuais profundos."
        }
    }

@app.post("/api/diagnosticar")
async def diagnosticar_termograma(
    arquivo: UploadFile = File(...),
    modelo_escolhido: str = Form("efficientnet_b0"),
    paleta_cor: str = Form("jet")
):
    global MODELOS, GRADCAMS
    if not MODELOS:
        carregar_modelos()

    nome_mod = modelo_escolhido.lower().strip()
    if nome_mod not in ['efficientnet_b0', 'resnet50', 'ensemble']:
        nome_mod = 'efficientnet_b0'

    # Validação de formato
    ext = os.path.splitext(arquivo.filename)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
        raise HTTPException(status_code=400, detail="Formato inválido. Use JPG ou PNG.")

    try:
        t_inicio = time.time()
        conteudo_bytes = await arquivo.read()
        imagem_pil = Image.open(io.BytesIO(conteudo_bytes)).convert('RGB')
        imagem_tensor = TRANSFORMACOES(imagem_pil).unsqueeze(0).to(DEVICE)

        if nome_mod == 'ensemble':
            m_eff = MODELOS.get('efficientnet_b0')
            m_res = MODELOS.get('resnet50')

            with torch.no_grad():
                probs_eff = torch.softmax(m_eff(imagem_tensor), dim=1)[0]
                probs_res = torch.softmax(m_res(imagem_tensor), dim=1)[0]
                probs = (0.6 * probs_eff + 0.4 * probs_res).cpu().numpy()

            pred_classe = int(np.argmax(probs))
            confianca = float(probs[pred_classe])
            prob_saudavel = float(probs[0]) * 100
            prob_doente = float(probs[1]) * 100

            with torch.enable_grad():
                heatmap, _, _ = GRADCAMS['efficientnet_b0'].gerar_mapa(imagem_tensor)

            nome_exibicao_modelo = "Ensemble (EfficientNet-B0 + ResNet-50)"

        else:
            modelo = MODELOS[nome_mod]
            gradcam = GRADCAMS[nome_mod]

            with torch.enable_grad():
                heatmap, pred_classe, confianca = gradcam.gerar_mapa(imagem_tensor)

            with torch.no_grad():
                logits = modelo(imagem_tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                prob_saudavel = float(probs[0]) * 100
                prob_doente = float(probs[1]) * 100

            nome_exibicao_modelo = "EfficientNet-B0" if nome_mod == 'efficientnet_b0' else "ResNet-50"

        # Gera Mapa Grad-CAM Puro e Sobreposição
        paleta_valida = paleta_cor if paleta_cor in ['jet', 'inferno', 'turbo', 'magma', 'plasma'] else 'jet'
        overlay_array = sobrepor_heatmap(imagem_pil, heatmap, alpha=0.45, colormap_name=paleta_valida)
        overlay_pil = Image.fromarray((overlay_array * 255).astype(np.uint8))

        cmap = plt.get_cmap(paleta_valida)
        heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize(imagem_pil.size, Image.Resampling.BICUBIC)
        heatmap_color = (cmap(np.array(heatmap_resized) / 255.0)[:, :, :3] * 255).astype(np.uint8)
        heatmap_pil = Image.fromarray(heatmap_color)

        buf_overlay = io.BytesIO()
        overlay_pil.save(buf_overlay, format="PNG")
        overlay_b64 = "data:image/png;base64," + base64.b64encode(buf_overlay.getvalue()).decode('utf-8')

        buf_heat = io.BytesIO()
        heatmap_pil.save(buf_heat, format="PNG")
        heatmap_b64 = "data:image/png;base64," + base64.b64encode(buf_heat.getvalue()).decode('utf-8')

        tempo_inferencia_ms = round((time.time() - t_inicio) * 1000, 2)

        diagnostico_texto = "Alteração Térmica Detectada" if pred_classe == 1 else "Padrão Fisiológico Normal"
        confianca_pct = round(confianca * 100, 2)

        # Salva o exame no banco de dados SQLite para histórico
        registro_id = salvar_exame(
            nome_arquivo=arquivo.filename,
            modelo_utilizado=nome_exibicao_modelo,
            classe_id=int(pred_classe),
            diagnostico=diagnostico_texto,
            confianca=confianca_pct,
            prob_saudavel=round(prob_saudavel, 2),
            prob_doente=round(prob_doente, 2),
            tempo_ms=tempo_inferencia_ms
        )

        return {
            "status": "sucesso",
            "registro_id": registro_id,
            "tempo_ms": tempo_inferencia_ms,
            "arquivo": arquivo.filename,
            "modelo_utilizado": nome_exibicao_modelo,
            "predicao": {
                "classe_id": int(pred_classe),
                "diagnostico": diagnostico_texto,
                "confianca_percentual": confianca_pct,
                "probabilidade_saudavel": round(prob_saudavel, 2),
                "probabilidade_doente": round(prob_doente, 2)
            },
            "gradcam_overlay_base64": overlay_b64,
            "gradcam_pure_base64": heatmap_b64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar imagem: {str(e)}")

@app.get("/api/historico")
def obter_historico():
    """Retorna o histórico de exames salvos no banco SQLite."""
    return {"historico": listar_historico(limite=50)}

@app.delete("/api/historico")
def deletar_historico():
    """Limpa a tabela de histórico de exames."""
    limpar_historico()
    return {"status": "sucesso", "mensagem": "Histórico limpo com sucesso."}

@app.get("/api/exemplos")
def listar_exemplos():
    dataset_dir = os.path.join(TREINAMENTO_DIR, 'dataset')
    exemplos = []

    casos = [
        ('saudavel', '1', 'Paciente #01 (Saudável)', 'T0001.1.1.D.2012-10-08.00.jpg'),
        ('saudavel', '10', 'Paciente #10 (Saudável)', 'T0010.1.1.D.2012-10-24.00.jpg'),
        ('doente', '144', 'Paciente #144 (Patologia)', 'T0144.1.1.D.2013-01-29.00.jpg'),
        ('doente', '156', 'Paciente #156 (Patologia)', 'TFRON_V159_18-2-2013_00.jpg')
    ]

    for cat, p_id, rotulo, arq in casos:
        caminho = os.path.join(dataset_dir, cat, p_id, arq)
        if os.path.exists(caminho):
            exemplos.append({
                "categoria": cat,
                "paciente_id": p_id,
                "arquivo": arq,
                "rotulo_exibicao": rotulo,
                "url": f"/api/amostra/{cat}/{p_id}/{arq}"
            })

    return {"exemplos": exemplos}

@app.get("/api/amostra/{classe}/{paciente_id}/{arquivo}")
def obter_amostra(classe: str, paciente_id: str, arquivo: str):
    caminho = os.path.join(TREINAMENTO_DIR, 'dataset', classe, paciente_id, arquivo)
    if os.path.exists(caminho):
        return FileResponse(caminho)
    raise HTTPException(status_code=404, detail="Amostra não encontrada.")

# Monta a interface web diretamente no FastAPI (porta 8000)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

INTERFACE_DIR = os.path.join(PROJECT_ROOT, 'interface')
if os.path.exists(INTERFACE_DIR):
    app.mount("/web", StaticFiles(directory=INTERFACE_DIR, html=True), name="web")

    @app.get("/app")
    def redirecionar_app():
        return RedirectResponse(url="/web/")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

