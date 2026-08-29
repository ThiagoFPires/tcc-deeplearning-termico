// ==========================================================================
// ThermoScan PACS — Frontend Application Logic (v2.0)
// ==========================================================================

const API_BASE_URL = 'http://127.0.0.1:8000';

let arquivoSelecionado = null;
let modeloSelecionado = 'efficientnet_b0';
let paletaSelecionada = 'jet';
let listaExemplos = [];
let resultadoAtual = null;

// DOM Elements
const statusText = document.getElementById('status-text');
const modelSegButtons = document.querySelectorAll('.seg-btn');
const cmapButtons = document.querySelectorAll('.cmap-btn');

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const dropPrompt = document.getElementById('drop-prompt');
const dropPreview = document.getElementById('drop-preview');
const imagePreview = document.getElementById('image-preview');
const btnClearImg = document.getElementById('btn-clear-img');
const btnDiagnosticar = document.getElementById('btn-diagnosticar');
const loaderSpinner = document.getElementById('loader-spinner');

const emptyState = document.getElementById('empty-state');
const loadingState = document.getElementById('loading-state');
const loadingDesc = document.getElementById('loading-desc');
const resultsViewport = document.getElementById('results-viewport');
const inferenceMeta = document.getElementById('inference-meta');

// Tabs & Views
const tabSideBySide = document.getElementById('tab-side-by-side');
const tabBlend = document.getElementById('tab-blend');
const viewSideBySide = document.getElementById('view-side-by-side');
const viewBlend = document.getElementById('view-blend');

// Images
const imgOrigSide = document.getElementById('img-orig-side');
const imgGradSide = document.getElementById('img-grad-side');
const modelTagGradcam = document.getElementById('model-tag-gradcam');
const imgBlendBase = document.getElementById('img-blend-base');
const imgBlendHeat = document.getElementById('img-blend-heat');
const opacitySlider = document.getElementById('opacity-slider');
const sliderValPill = document.getElementById('slider-val-pill');

// Report Card Elements
const reportBadge = document.getElementById('report-badge');
const reportTitle = document.getElementById('report-title');
const reportSubtitle = document.getElementById('report-subtitle');
const confidenceNum = document.getElementById('confidence-num');
const probNormalTxt = document.getElementById('prob-normal-txt');
const probNormalBar = document.getElementById('prob-normal-bar');
const probLesaoTxt = document.getElementById('prob-lesao-txt');
const probLesaoBar = document.getElementById('prob-lesao-bar');
const obsText = document.getElementById('obs-text');
const btnReset = document.getElementById('btn-reset');

// Presets
const presetSaudavel1 = document.getElementById('preset-saudavel-1');
const presetSaudavel2 = document.getElementById('preset-saudavel-2');
const presetDoente1 = document.getElementById('preset-doente-1');
const presetDoente2 = document.getElementById('preset-doente-2');

// ==========================================================================
// 1. Initialization
// ==========================================================================
async function inicializarWorkstation() {
    try {
        const res = await fetch(`${API_BASE_URL}/`);
        if (res.ok) {
            const data = await res.json();
            statusText.textContent = data.gpu !== 'N/A' ? `${data.gpu} (Ativa)` : 'CPU (Online)';
        }
    } catch (e) {
        statusText.textContent = 'API Offline (Inicie o Servidor)';
    }

    try {
        const resEx = await fetch(`${API_BASE_URL}/api/exemplos`);
        if (resEx.ok) {
            const data = await resEx.json();
            listaExemplos = data.exemplos || [];
            configurarPresets();
        }
    } catch (e) {
        console.warn('Erro ao carregar exemplos:', e);
    }
}
inicializarWorkstation();

// ==========================================================================
// 2. Model & Colormap Selectors
// ==========================================================================
modelSegButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        modelSegButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        modeloSelecionado = btn.getAttribute('data-model');
        
        // Se já tiver uma imagem carregada e analisada, re-executa a análise com o novo modelo
        if (arquivoSelecionado && resultadoAtual) {
            executarDiagnostico();
        }
    });
});

cmapButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        cmapButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        paletaSelecionada = btn.getAttribute('data-cmap');

        if (arquivoSelecionado && resultadoAtual) {
            executarDiagnostico();
        }
    });
});

// ==========================================================================
// 3. Tab Switching & Blend Slider
// ==========================================================================
tabSideBySide.addEventListener('click', () => {
    tabSideBySide.classList.add('active');
    tabBlend.classList.remove('active');
    viewSideBySide.classList.remove('hidden');
    viewBlend.classList.add('hidden');
});

tabBlend.addEventListener('click', () => {
    tabBlend.classList.add('active');
    tabSideBySide.classList.remove('active');
    viewBlend.classList.remove('hidden');
    viewSideBySide.classList.add('hidden');
});

opacitySlider.addEventListener('input', (e) => {
    const val = e.target.value;
    sliderValPill.textContent = `${val}%`;
    imgBlendHeat.style.opacity = val / 100.0;
});

// ==========================================================================
// 4. Drag & Drop and File Selection
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
    dropZone.classList.add('drag-active');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-active');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-active');
    if (e.dataTransfer.files.length > 0) {
        carregarArquivo(e.dataTransfer.files[0]);
    }
});

btnClearImg.addEventListener('click', (e) => {
    e.stopPropagation();
    limparExame();
});

btnReset.addEventListener('click', () => {
    limparExame();
});

function carregarArquivo(file) {
    if (!file.type.match('image.*')) {
        alert('Por favor, selecione uma imagem válida (JPG, PNG ou BMP).');
        return;
    }
    arquivoSelecionado = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        dropPrompt.classList.add('hidden');
        dropPreview.classList.remove('hidden');
        btnDiagnosticar.disabled = false;
    };
    reader.readAsDataURL(file);
}

function limparExame() {
    arquivoSelecionado = null;
    resultadoAtual = null;
    fileInput.value = '';
    imagePreview.src = '';
    dropPreview.classList.add('hidden');
    dropPrompt.classList.remove('hidden');
    btnDiagnosticar.disabled = true;
    resultsViewport.classList.add('hidden');
    emptyState.classList.remove('hidden');
    inferenceMeta.textContent = 'Aguardando processamento...';
}

// ==========================================================================
// 5. Presets
// ==========================================================================
async function carregarAmostra(url, nome) {
    try {
        const res = await fetch(url);
        const blob = await res.blob();
        const file = new File([blob], nome, { type: blob.type || 'image/jpeg' });
        carregarArquivo(file);
        // Dispara o diagnóstico automaticamente
        setTimeout(() => executarDiagnostico(), 100);
    } catch (e) {
        console.error('Erro ao carregar amostra:', e);
    }
}

function configurarPresets() {
    if (listaExemplos.length >= 4) {
        presetSaudavel1.onclick = () => carregarAmostra(`${API_BASE_URL}${listaExemplos[0].url}`, listaExemplos[0].arquivo);
        presetSaudavel2.onclick = () => carregarAmostra(`${API_BASE_URL}${listaExemplos[1].url}`, listaExemplos[1].arquivo);
        presetDoente1.onclick = () => carregarAmostra(`${API_BASE_URL}${listaExemplos[2].url}`, listaExemplos[2].arquivo);
        presetDoente2.onclick = () => carregarAmostra(`${API_BASE_URL}${listaExemplos[3].url}`, listaExemplos[3].arquivo);
    }
}

// ==========================================================================
// 6. Clinical Inference Execution
// ==========================================================================
btnDiagnosticar.addEventListener('click', () => {
    executarDiagnostico();
});

async function executarDiagnostico() {
    if (!arquivoSelecionado) return;

    btnDiagnosticar.disabled = true;
    loaderSpinner.classList.remove('hidden');
    emptyState.classList.add('hidden');
    resultsViewport.classList.add('hidden');
    loadingState.classList.remove('hidden');
    loadingDesc.textContent = `Executando inferência com ${modeloSelecionado.toUpperCase()} e mapa Grad-CAM (${paletaSelecionada.toUpperCase()})...`;

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
            throw new Error(err.detail || 'Erro no processamento da API');
        }

        const data = await response.json();
        resultadoAtual = data;
        renderizarLaudoClinico(data);

    } catch (error) {
        alert(`Falha no exame: ${error.message}\nVerifique se o backend FastAPI está em execução.`);
        loadingState.classList.add('hidden');
        emptyState.classList.remove('hidden');
    } finally {
        btnDiagnosticar.disabled = false;
        loaderSpinner.classList.add('hidden');
    }
}

// ==========================================================================
// 7. Render Clinical Report & PACS Visuals
// ==========================================================================
function renderizarLaudoClinico(data) {
    loadingState.classList.add('hidden');
    resultsViewport.classList.remove('hidden');

    const pred = data.predicao;
    const isDoente = pred.classe_id === 1;

    // Metadata
    inferenceMeta.innerHTML = `Modelo: <strong>${data.modelo_utilizado}</strong> | Tempo: <strong>${data.tempo_ms} ms</strong> (GPU)`;
    modelTagGradcam.textContent = data.modelo_utilizado;

    // Images Side-by-Side
    imgOrigSide.src = imagePreview.src;
    imgGradSide.src = data.gradcam_overlay_base64;

    // Images Blend Slider
    imgBlendBase.src = imagePreview.src;
    imgBlendHeat.src = data.gradcam_pure_base64;

    // Clinical Status & Badges
    if (isDoente) {
        reportBadge.className = 'report-badge badge-status-alert';
        reportBadge.textContent = 'ALTERAÇÃO TÉRMICA DETECTADA';
        reportTitle.textContent = 'Suspeita de Padrão Patológico (Classe 1)';
        reportSubtitle.textContent = 'Hiper-radiação e assimetria vascular com gradiente térmico significativo.';
        obsText.textContent = 'O mapa Grad-CAM revelou foco de alta ativação (foco hiper-radiante) na região mamária, recomendando correlação com exames de imagem complementares (mamografia/ultrassonografia).';
    } else {
        reportBadge.className = 'report-badge badge-status-normal';
        reportBadge.textContent = 'PADRÃO FISIOLÓGICO NORMAL';
        reportTitle.textContent = 'Sem Evidências de Anomalias Térmicas (Classe 0)';
        reportSubtitle.textContent = 'Distribuição de temperatura bilateralmente homogênea e simétrica.';
        obsText.textContent = 'O algoritmo não identificou gradientes assimétricos anormais. Os mapas de ativação neural mantiveram-se estáveis e difusos, compatíveis com a fisiologia esperada.';
    }

    confidenceNum.textContent = `${pred.confianca_percentual.toFixed(1)}%`;

    // Probability Bars
    probNormalTxt.textContent = `${pred.probabilidade_saudavel.toFixed(1)}%`;
    probNormalBar.style.width = `${pred.probabilidade_saudavel}%`;

    probLesaoTxt.textContent = `${pred.probabilidade_doente.toFixed(1)}%`;
    probLesaoBar.style.width = `${pred.probabilidade_doente}%`;
}
