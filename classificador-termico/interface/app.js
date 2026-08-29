// ==========================================================================
// ThermoScan AI — Clean Application Logic (v2.1)
// ==========================================================================

const API_BASE_URL = 'http://127.0.0.1:8000';

let arquivoSelecionado = null;
let modeloSelecionado = 'efficientnet_b0';
let paletaSelecionada = 'jet';
let listaExemplos = [];
let resultadoAtual = null;

// DOM Elements - Header
const statusLabel = document.getElementById('status-label');
const switchButtons = document.querySelectorAll('.switch-btn');
const cmapChoices = document.querySelectorAll('.cmap-choice');

// DOM Elements - Input Panel
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const dropzoneIdle = document.getElementById('dropzone-idle');
const dropzonePreview = document.getElementById('dropzone-preview');
const imagePreview = document.getElementById('image-preview');
const btnRemovePreview = document.getElementById('btn-remove-preview');
const btnExecute = document.getElementById('btn-execute');
const spinnerWheel = document.getElementById('spinner-wheel');

// DOM Elements - Samples
const sampleSaudavel1 = document.getElementById('sample-saudavel-1');
const sampleSaudavel2 = document.getElementById('sample-saudavel-2');
const sampleDoente1 = document.getElementById('sample-doente-1');
const sampleDoente2 = document.getElementById('sample-doente-2');

// DOM Elements - Viewer
const tabSide = document.getElementById('tab-side');
const tabBlend = document.getElementById('tab-blend');
const modeSide = document.getElementById('mode-side');
const modeBlend = document.getElementById('mode-blend');
const runtimeBadge = document.getElementById('runtime-badge');

const viewEmpty = document.getElementById('view-empty');
const viewLoading = document.getElementById('view-loading');
const loadingStatusText = document.getElementById('loading-status-text');
const viewResults = document.getElementById('view-results');

// Images & Sliders
const resOrigImg = document.getElementById('res-orig-img');
const resGradImg = document.getElementById('res-grad-img');
const frameModelBadge = document.getElementById('frame-model-badge');
const blendBaseImg = document.getElementById('blend-base-img');
const blendHeatImg = document.getElementById('blend-heat-img');
const blendSlider = document.getElementById('blend-slider');
const sliderPercentPill = document.getElementById('slider-percent-pill');

// Report Card
const statusBadge = document.getElementById('status-badge');
const reportMainTitle = document.getElementById('report-main-title');
const reportSubTitle = document.getElementById('report-sub-title');
const confValue = document.getElementById('conf-value');
const probValHealthy = document.getElementById('prob-val-healthy');
const barHealthy = document.getElementById('bar-healthy');
const probValSick = document.getElementById('prob-val-sick');
const barSick = document.getElementById('bar-sick');
const noteExplanationText = document.getElementById('note-explanation-text');
const btnResetExam = document.getElementById('btn-reset-exam');

// ==========================================================================
// 1. Initialization
// ==========================================================================
async function inicializar() {
    try {
        const res = await fetch(`${API_BASE_URL}/`);
        if (res.ok) {
            const data = await res.json();
            statusLabel.textContent = data.gpu !== 'N/A' ? `${data.gpu} (Ativo)` : 'CPU (Online)';
        }
    } catch (e) {
        statusLabel.textContent = 'API Offline (Inicie o Servidor)';
    }

    try {
        const resEx = await fetch(`${API_BASE_URL}/api/exemplos`);
        if (resEx.ok) {
            const data = await resEx.json();
            listaExemplos = data.exemplos || [];
            configurarAmostras();
        }
    } catch (e) {
        console.warn('Falha ao obter lista de exemplos:', e);
    }
}
inicializar();

// ==========================================================================
// 2. Model & Palette Selectors
// ==========================================================================
switchButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        switchButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        modeloSelecionado = btn.getAttribute('data-model');
        
        if (arquivoSelecionado && resultadoAtual) {
            processarDiagnostico();
        }
    });
});

cmapChoices.forEach(btn => {
    btn.addEventListener('click', () => {
        cmapChoices.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        paletaSelecionada = btn.getAttribute('data-cmap');

        if (arquivoSelecionado && resultadoAtual) {
            processarDiagnostico();
        }
    });
});

// ==========================================================================
// 3. Tab Switching & Blend Slider
// ==========================================================================
tabSide.addEventListener('click', () => {
    tabSide.classList.add('active');
    tabBlend.classList.remove('active');
    modeSide.classList.remove('hidden');
    modeBlend.classList.add('hidden');
});

tabBlend.addEventListener('click', () => {
    tabBlend.classList.add('active');
    tabSide.classList.remove('active');
    modeBlend.classList.remove('hidden');
    modeSide.classList.add('hidden');
});

blendSlider.addEventListener('input', (e) => {
    const val = e.target.value;
    sliderPercentPill.textContent = `${val}%`;
    blendHeatImg.style.opacity = val / 100.0;
});

// ==========================================================================
// 4. Drag & Drop and File Selection Handlers
// ==========================================================================
dropZone.addEventListener('click', () => {
    if (!arquivoSelecionado) {
        fileInput.click();
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        selecionarArquivo(e.target.files[0]);
    }
});

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-active');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-active');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-active');
    if (e.dataTransfer.files.length > 0) {
        selecionarArquivo(e.dataTransfer.files[0]);
    }
});

btnRemovePreview.addEventListener('click', (e) => {
    e.stopPropagation();
    limparExame();
});

btnResetExam.addEventListener('click', () => {
    limparExame();
});

function selecionarArquivo(file) {
    if (!file.type.match('image.*')) {
        alert('Por favor, envie um arquivo de imagem válido (JPG, PNG ou BMP).');
        return;
    }
    arquivoSelecionado = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        dropzoneIdle.classList.add('hidden');
        dropzonePreview.classList.remove('hidden');
        btnExecute.disabled = false;
    };
    reader.readAsDataURL(file);
}

function limparExame() {
    arquivoSelecionado = null;
    resultadoAtual = null;
    fileInput.value = '';
    imagePreview.src = '';
    dropzonePreview.classList.add('hidden');
    dropzoneIdle.classList.remove('hidden');
    btnExecute.disabled = true;
    viewResults.classList.add('hidden');
    viewEmpty.classList.remove('hidden');
    runtimeBadge.textContent = 'Aguardando exame';
}

// ==========================================================================
// 5. Presets
// ==========================================================================
async function carregarAmostra(url, nome) {
    try {
        const res = await fetch(url);
        const blob = await res.blob();
        const file = new File([blob], nome, { type: blob.type || 'image/jpeg' });
        selecionarArquivo(file);
        setTimeout(() => processarDiagnostico(), 100);
    } catch (e) {
        console.error('Erro ao carregar amostra:', e);
    }
}

function configurarAmostras() {
    if (listaExemplos.length >= 4) {
        sampleSaudavel1.onclick = () => carregarAmostra(`${API_BASE_URL}${listaExemplos[0].url}`, listaExemplos[0].arquivo);
        sampleSaudavel2.onclick = () => carregarAmostra(`${API_BASE_URL}${listaExemplos[1].url}`, listaExemplos[1].arquivo);
        sampleDoente1.onclick = () => carregarAmostra(`${API_BASE_URL}${listaExemplos[2].url}`, listaExemplos[2].arquivo);
        sampleDoente2.onclick = () => carregarAmostra(`${API_BASE_URL}${listaExemplos[3].url}`, listaExemplos[3].arquivo);
    }
}

// ==========================================================================
// 6. Clinical Inference Execution
// ==========================================================================
btnExecute.addEventListener('click', () => {
    processarDiagnostico();
});

async function processarDiagnostico() {
    if (!arquivoSelecionado) return;

    btnExecute.disabled = true;
    spinnerWheel.classList.remove('hidden');
    viewEmpty.classList.add('hidden');
    viewResults.classList.add('hidden');
    viewLoading.classList.remove('hidden');
    loadingStatusText.textContent = `Executando inferência com ${modeloSelecionado.toUpperCase()} e paleta ${paletaSelecionada.toUpperCase()}...`;

    const formData = new FormData();
    formData.append('arquivo', arquivoSelecionado);
    formData.append('modelo_escolhido', modeloSelecionado);
    formData.append('paleta_cor', paletaSelecionada);

    try {
        const response = await fetch(`${API_BASE_URL}/api/diagnosticar`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Falha ao processar o exame');
        }

        const data = await response.json();
        resultadoAtual = data;
        renderizarLaudo(data);

    } catch (error) {
        alert(`Erro na análise: ${error.message}\nCertifique-se de que o backend FastAPI está em execução.`);
        viewLoading.classList.add('hidden');
        viewEmpty.classList.remove('hidden');
    } finally {
        btnExecute.disabled = false;
        spinnerWheel.classList.add('hidden');
    }
}

// ==========================================================================
// 7. Render Report to UI
// ==========================================================================
function renderizarLaudo(data) {
    viewLoading.classList.add('hidden');
    viewResults.classList.remove('hidden');

    const pred = data.predicao;
    const isDoente = pred.classe_id === 1;

    // Badges & Meta
    runtimeBadge.innerHTML = `Modelo: <strong>${data.modelo_utilizado}</strong> | Tempo: <strong>${data.tempo_ms} ms</strong> (GPU)`;
    frameModelBadge.textContent = data.modelo_utilizado;

    // Images
    resOrigImg.src = imagePreview.src;
    resGradImg.src = data.gradcam_overlay_base64;

    blendBaseImg.src = imagePreview.src;
    blendHeatImg.src = data.gradcam_pure_base64;

    // Clinical Status
    if (isDoente) {
        statusBadge.className = 'status-badge alert';
        statusBadge.textContent = 'ALTERAÇÃO TÉRMICA DETECTADA';
        reportMainTitle.textContent = 'Suspeita de Padrão Patológico (Classe 1)';
        reportSubTitle.textContent = 'Identificada hiper-radiação e assimetria vascular com gradiente térmico significativo.';
        noteExplanationText.textContent = 'O mapa Grad-CAM revelou foco de intensa ativação hiper-radiante na região mamária destacada. Recomenda-se correlação com exames de imagem complementares (mamografia/ultrassonografia).';
    } else {
        statusBadge.className = 'status-badge healthy';
        statusBadge.textContent = 'PADRÃO FISIOLÓGICO NORMAL';
        reportMainTitle.textContent = 'Sem Evidências de Anomalias Térmicas (Classe 0)';
        reportSubTitle.textContent = 'Distribuição de temperatura simétrica e homogênea bilateralmente.';
        noteExplanationText.textContent = 'O algoritmo não identificou gradientes assimétricos anormais. Os mapas de ativação neural mantiveram-se uniformes e estáveis, compatíveis com a normalidade.';
    }

    confValue.textContent = `${pred.confianca_percentual.toFixed(1)}%`;

    // Probability Bars
    probValHealthy.textContent = `${pred.probabilidade_saudavel.toFixed(1)}%`;
    barHealthy.style.width = `${pred.probabilidade_saudavel}%`;

    probValSick.textContent = `${pred.probabilidade_doente.toFixed(1)}%`;
    barSick.style.width = `${pred.probabilidade_doente}%`;
}
