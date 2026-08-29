// ==========================================================================
// ThermoScan AI — Frontend Application Logic
// ==========================================================================

const API_BASE_URL = 'http://127.0.0.1:8000';

let arquivoSelecionado = null;
let listaExemplos = [];

// DOM Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const dropZonePrompt = document.getElementById('drop-zone-prompt');
const previewContainer = document.getElementById('preview-container');
const imagePreview = document.getElementById('image-preview');
const btnRemove = document.getElementById('btn-remove');
const btnAnalisar = document.getElementById('btn-analisar');
const btnSpinner = document.getElementById('btn-spinner');

const emptyState = document.getElementById('empty-state');
const loadingState = document.getElementById('loading-state');
const resultsContent = document.getElementById('results-content');
const inferenceTimeTag = document.getElementById('inference-time-tag');

const diagnosisBanner = document.getElementById('diagnosis-banner');
const diagnosisBadge = document.getElementById('diagnosis-badge');
const diagnosisTitle = document.getElementById('diagnosis-title');
const diagnosisDesc = document.getElementById('diagnosis-desc');
const confidenceVal = document.getElementById('confidence-val');

const resOriginal = document.getElementById('res-original');
const resGradcam = document.getElementById('res-gradcam');

const probSaudavelVal = document.getElementById('prob-saudavel-val');
const probSaudavelBar = document.getElementById('prob-saudavel-bar');
const probDoenteVal = document.getElementById('prob-doente-val');
const probDoenteBar = document.getElementById('prob-doente-bar');
const gradcamExplanation = document.getElementById('gradcam-explanation');
const btnNovaAnalise = document.getElementById('btn-nova-analise');

// Sample Buttons
const sampleSaudavel1 = document.getElementById('sample-saudavel-1');
const sampleSaudavel2 = document.getElementById('sample-saudavel-2');
const sampleDoente1 = document.getElementById('sample-doente-1');
const sampleDoente2 = document.getElementById('sample-doente-2');

// ==========================================================================
// 1. Initial Status Check & Samples Fetch
// ==========================================================================
async function inicializarApp() {
    const gpuElement = document.getElementById('gpu-name');
    try {
        const res = await fetch(`${API_BASE_URL}/`);
        if (res.ok) {
            const data = await res.json();
            gpuElement.textContent = data.gpu !== 'N/A' ? `${data.gpu} (Ativa)` : 'CPU (Online)';
        }
    } catch (e) {
        gpuElement.textContent = 'API Offline (Inicie o Servidor)';
        document.getElementById('status-gpu').style.borderColor = 'rgba(244, 63, 94, 0.4)';
    }

    try {
        const resEx = await fetch(`${API_BASE_URL}/api/exemplos`);
        if (resEx.ok) {
            const data = await resEx.json();
            listaExemplos = data.exemplos || [];
            configurarBotoesAmostras();
        }
    } catch (e) {
        console.warn('Não foi possível carregar lista dinâmica de exemplos:', e);
    }
}
inicializarApp();

function configurarBotoesAmostras() {
    if (listaExemplos.length >= 4) {
        sampleSaudavel1.onclick = () => carregarAmostraPorUrl(`${API_BASE_URL}${listaExemplos[0].url}`, listaExemplos[0].arquivo);
        sampleSaudavel2.onclick = () => carregarAmostraPorUrl(`${API_BASE_URL}${listaExemplos[1].url}`, listaExemplos[1].arquivo);
        sampleDoente1.onclick = () => carregarAmostraPorUrl(`${API_BASE_URL}${listaExemplos[2].url}`, listaExemplos[2].arquivo);
        sampleDoente2.onclick = () => carregarAmostraPorUrl(`${API_BASE_URL}${listaExemplos[3].url}`, listaExemplos[3].arquivo);
    }
}

// ==========================================================================
// 2. Drag & Drop & File Selection Handlers
// ==========================================================================
dropZone.addEventListener('click', () => {
    if (!arquivoSelecionado) {
        fileInput.click();
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        carregarArquivo(e.target.files[0]);
    }
});

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
        carregarArquivo(e.dataTransfer.files[0]);
    }
});

btnRemove.addEventListener('click', (e) => {
    e.stopPropagation();
    limparArquivo();
});

btnNovaAnalise.addEventListener('click', () => {
    limparArquivo();
    resultsContent.classList.add('hidden');
    emptyState.classList.remove('hidden');
    inferenceTimeTag.textContent = 'Aguardando exame';
});

function carregarArquivo(file) {
    if (!file.type.match('image.*')) {
        alert('Por favor, selecione um arquivo de imagem válido (JPG ou PNG).');
        return;
    }
    arquivoSelecionado = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        dropZonePrompt.classList.add('hidden');
        previewContainer.classList.remove('hidden');
        btnAnalisar.disabled = false;
    };
    reader.readAsDataURL(file);
}

function limparArquivo() {
    arquivoSelecionado = null;
    fileInput.value = '';
    imagePreview.src = '';
    previewContainer.classList.add('hidden');
    dropZonePrompt.classList.remove('hidden');
    btnAnalisar.disabled = true;
}

// ==========================================================================
// 3. Quick Sample Testers
// ==========================================================================
async function carregarAmostraPorUrl(url, nomeArquivo) {
    try {
        const res = await fetch(url);
        const blob = await res.blob();
        const file = new File([blob], nomeArquivo, { type: blob.type || 'image/jpeg' });
        carregarArquivo(file);
    } catch (e) {
        console.error('Erro ao carregar amostra:', e);
    }
}

// ==========================================================================
// 4. Execution & API Communication
// ==========================================================================
btnAnalisar.addEventListener('click', async () => {
    if (!arquivoSelecionado) return;

    // UI Loading State
    btnAnalisar.disabled = true;
    btnSpinner.classList.remove('hidden');
    emptyState.classList.add('hidden');
    resultsContent.classList.add('hidden');
    loadingState.classList.remove('hidden');

    const formData = new FormData();
    formData.append('arquivo', arquivoSelecionado);

    try {
        const response = await fetch(`${API_BASE_URL}/api/diagnosticar`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Erro na inferência da API');
        }

        const data = await response.json();
        exibirResultados(data);

    } catch (error) {
        alert(`Falha na análise: ${error.message}\nCertifique-se de que a API FastAPI está em execução (porta 8000).`);
        loadingState.classList.add('hidden');
        emptyState.classList.remove('hidden');
    } finally {
        btnAnalisar.disabled = false;
        btnSpinner.classList.add('hidden');
    }
});

// ==========================================================================
// 5. Render Results to Screen
// ==========================================================================
function exibirResultados(data) {
    loadingState.classList.add('hidden');
    resultsContent.classList.remove('hidden');

    const pred = data.predicao;
    inferenceTimeTag.textContent = `Inferência: ${data.tempo_ms} ms (GPU)`;

    // Diagnostic Card Status
    if (pred.classe_id === 1) {
        // Doente / Anomalia
        diagnosisBanner.className = 'diagnosis-banner alert-high';
        diagnosisBadge.textContent = 'SUSPEITA DE ANOMALIA TÉRMICA';
        diagnosisTitle.textContent = 'Padrão Hiper-Radiante Detectado (Doente)';
        diagnosisDesc.textContent = 'Identificada assimetria vascular e gradiente térmico significativo na região mamária destacada pelo Grad-CAM.';
        gradcamExplanation.textContent = 'O mapa de calor Grad-CAM evidencia em tons quentes (vermelho/amarelo) a área de anomalia térmica com hipertermia focal que determinou a classificação positiva.';
    } else {
        // Saudável
        diagnosisBanner.className = 'diagnosis-banner';
        diagnosisBadge.textContent = 'SAUDÁVEL';
        diagnosisTitle.textContent = 'Sem Anomalias Térmicas Detectadas';
        diagnosisDesc.textContent = 'Distribuição de temperatura simétrica e padrão fisiológico compatível com normalidade.';
        gradcamExplanation.textContent = 'O mapa de calor Grad-CAM demonstra ausência de focos hiper-radiantes assimétricos, confirmando a estabilidade térmica bilateral.';
    }

    confidenceVal.textContent = `${pred.confianca_percentual.toFixed(1)}%`;

    // Images
    resOriginal.src = imagePreview.src;
    resGradcam.src = data.gradcam_base64;

    // Probabilities Bar Breakdown
    probSaudavelVal.textContent = `${pred.probabilidade_saudavel.toFixed(1)}%`;
    probSaudavelBar.style.width = `${pred.probabilidade_saudavel}%`;

    probDoenteVal.textContent = `${pred.probabilidade_doente.toFixed(1)}%`;
    probDoenteBar.style.width = `${pred.probabilidade_doente}%`;
}
