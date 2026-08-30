// ==========================================================================
// DeepVision CADe — Medical Diagnostic Workstation Logic with PACS Zoom (v2.2)
// ==========================================================================

const API_BASE_URL = 'http://127.0.0.1:8000';

let arquivoSelecionado = null;
let modeloSelecionado = 'efficientnet_b0';
let paletaSelecionada = 'jet';
let resultadoAtual = null;

// Dynamic Session Worklist Array (In-Memory)
const examesSessao = [];
let exameSessaoAtivoId = null;

// PACS Zoom & Pan Engine State
let zoomLevel = 1.0;
let panX = 0;
let panY = 0;
let isPanning = false;
let startPanX = 0;
let startPanY = 0;

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

// DOM Elements - Session Worklist
const sessionList = document.getElementById('session-list');
const sessionEmptyMsg = document.getElementById('session-empty-msg');
const sessionCountBadge = document.getElementById('session-count-badge');

// DOM Elements - Viewer Tabs
const tabSide = document.getElementById('tab-side');
const tabBlend = document.getElementById('tab-blend');
const tabHistory = document.getElementById('tab-history');

const modeSide = document.getElementById('mode-side');
const modeBlend = document.getElementById('mode-blend');
const viewHistory = document.getElementById('view-history');
const runtimeBadge = document.getElementById('runtime-badge');

const viewEmpty = document.getElementById('view-empty');
const viewLoading = document.getElementById('view-loading');
const loadingStatusText = document.getElementById('loading-status-text');
const viewResults = document.getElementById('view-results');

// PACS Zoom Toolbar Elements
const btnZoomIn = document.getElementById('btn-zoom-in');
const btnZoomOut = document.getElementById('btn-zoom-out');
const btnZoomReset = document.getElementById('btn-zoom-reset');
const zoomFactorLabel = document.getElementById('zoom-factor-label');
const pacsViewports = document.querySelectorAll('.pacs-viewport');
const zoomableImages = document.querySelectorAll('.zoomable-img');

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

// History Table (SQLite)
const historyTbody = document.getElementById('history-tbody');
const btnRefreshHistory = document.getElementById('btn-refresh-history');
const btnClearHistory = document.getElementById('btn-clear-history');

// ==========================================================================
// 1. Initialization
// ==========================================================================
async function inicializar() {
    try {
        const res = await fetch(`${API_BASE_URL}/`);
        if (res.ok) {
            const data = await res.json();
            statusLabel.textContent = data.gpu !== 'N/A' ? `${data.gpu} (Ativo)` : 'Sistema Online';
        }
    } catch (e) {
        statusLabel.textContent = 'API Offline';
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
// 3. Tab Navigation Handlers
// ==========================================================================
tabSide.addEventListener('click', () => {
    ativarTab(tabSide);
    viewHistory.classList.add('hidden');
    if (resultadoAtual) {
        viewResults.classList.remove('hidden');
        modeSide.classList.remove('hidden');
        modeBlend.classList.add('hidden');
    } else {
        viewEmpty.classList.remove('hidden');
    }
    resetarZoom();
});

tabBlend.addEventListener('click', () => {
    ativarTab(tabBlend);
    viewHistory.classList.add('hidden');
    if (resultadoAtual) {
        viewResults.classList.remove('hidden');
        modeBlend.classList.remove('hidden');
        modeSide.classList.add('hidden');
    } else {
        viewEmpty.classList.remove('hidden');
    }
    resetarZoom();
});

tabHistory.addEventListener('click', () => {
    ativarTab(tabHistory);
    viewEmpty.classList.add('hidden');
    viewLoading.classList.add('hidden');
    viewResults.classList.add('hidden');
    viewHistory.classList.remove('hidden');
    carregarHistorico();
});

function ativarTab(tabAtiva) {
    [tabSide, tabBlend, tabHistory].forEach(t => t.classList.remove('active'));
    tabAtiva.classList.add('active');
}

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
    limparUploadAtual();
});

btnResetExam.addEventListener('click', () => {
    limparUploadAtual();
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

function limparUploadAtual() {
    arquivoSelecionado = null;
    resultadoAtual = null;
    exameSessaoAtivoId = null;
    fileInput.value = '';
    imagePreview.src = '';
    dropzonePreview.classList.add('hidden');
    dropzoneIdle.classList.remove('hidden');
    btnExecute.disabled = true;
    viewResults.classList.add('hidden');
    viewHistory.classList.add('hidden');
    viewEmpty.classList.remove('hidden');
    ativarTab(tabSide);
    runtimeBadge.textContent = 'Aguardando exame';
    resetarZoom();
    atualizarDestaqueSessao();
}

// ==========================================================================
// 5. Clinical Diagnostic Execution (GPU Backend)
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
    viewHistory.classList.add('hidden');
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
        
        // Adiciona à lista de exames da sessão atual
        adicionarExameSessao(arquivoSelecionado.name, imagePreview.src, data);

        // Renderiza laudo na tela
        resetarZoom();
        renderizarLaudo(data, imagePreview.src);

    } catch (error) {
        alert(`Erro na análise: ${error.message}\nCertifique-se de que o backend está em execução.`);
        viewLoading.classList.add('hidden');
        viewEmpty.classList.remove('hidden');
    } finally {
        btnExecute.disabled = false;
        spinnerWheel.classList.add('hidden');
    }
}

// ==========================================================================
// 6. Dynamic Session Worklist Management
// ==========================================================================
function adicionarExameSessao(nomeArquivo, previewSrc, dadosResultado) {
    const novoExame = {
        id: examesSessao.length + 1,
        nomeArquivo: nomeArquivo,
        previewSrc: previewSrc,
        dadosResultado: dadosResultado,
        timestamp: new Date().toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    };

    examesSessao.unshift(novoExame); // Insere no topo da lista
    exameSessaoAtivoId = novoExame.id;
    renderizarListaSessao();
}

function renderizarListaSessao() {
    sessionCountBadge.textContent = `${examesSessao.length} ${examesSessao.length === 1 ? 'exame' : 'exames'}`;

    if (examesSessao.length === 0) {
        sessionList.innerHTML = `
            <div class="session-empty" id="session-empty-msg">
                <div class="session-empty-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="24" height="24">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <polyline points="14 2 14 8 20 8"/>
                        <line x1="12" y1="18" x2="12" y2="12"/>
                        <line x1="9" y1="15" x2="15" y2="15"/>
                    </svg>
                </div>
                <p>Nenhum exame analisado nesta sessão.</p>
                <span>Os exames processados aparecerão aqui para alternância rápida.</span>
            </div>
        `;
        return;
    }

    sessionList.innerHTML = examesSessao.map(ex => {
        const isDoente = ex.dadosResultado.predicao.classe_id === 1;
        const tagClass = isDoente ? 'tag-alert' : 'tag-healthy';
        const tagText = isDoente ? 'Alteração' : 'Normal';
        const isActive = ex.id === exameSessaoAtivoId ? 'active' : '';

        return `
            <div class="session-item ${isActive}" onclick="carregarExameSessao(${ex.id})">
                <img class="session-thumb" src="${ex.previewSrc}" alt="Miniatura">
                <div class="session-info">
                    <div class="session-name" title="${ex.nomeArquivo}">${ex.nomeArquivo}</div>
                    <div class="session-meta">
                        <span class="session-badge ${tagClass}">${tagText}</span>
                        <span>${ex.dadosResultado.predicao.confianca_percentual.toFixed(1)}%</span>
                        <span>• ${ex.timestamp}</span>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

window.carregarExameSessao = function(id) {
    const exame = examesSessao.find(e => e.id === id);
    if (!exame) return;

    exameSessaoAtivoId = id;
    resultadoAtual = exame.dadosResultado;

    // Atualiza preview no dropzone
    imagePreview.src = exame.previewSrc;
    dropzoneIdle.classList.add('hidden');
    dropzonePreview.classList.remove('hidden');

    resetarZoom();
    atualizarDestaqueSessao();
    renderizarLaudo(exame.dadosResultado, exame.previewSrc);
};

function atualizarDestaqueSessao() {
    document.querySelectorAll('.session-item').forEach((el, idx) => {
        if (examesSessao[idx] && examesSessao[idx].id === exameSessaoAtivoId) {
            el.classList.add('active');
        } else {
            el.classList.remove('active');
        }
    });
}

// ==========================================================================
// 7. Render Report to UI
// ==========================================================================
function renderizarLaudo(data, previewSrc) {
    viewLoading.classList.add('hidden');
    viewEmpty.classList.add('hidden');
    viewHistory.classList.add('hidden');
    viewResults.classList.remove('hidden');

    if (tabBlend.classList.contains('active')) {
        modeBlend.classList.remove('hidden');
        modeSide.classList.add('hidden');
    } else {
        modeSide.classList.remove('hidden');
        modeBlend.classList.add('hidden');
    }

    const pred = data.predicao;
    const isDoente = pred.classe_id === 1;

    // Badges & Meta
    runtimeBadge.innerHTML = `Modelo: <strong>${data.modelo_utilizado}</strong> | Tempo: <strong>${data.tempo_ms} ms</strong>`;
    frameModelBadge.textContent = data.modelo_utilizado;

    // Images
    resOrigImg.src = previewSrc;
    resGradImg.src = data.gradcam_overlay_base64;

    blendBaseImg.src = previewSrc;
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

// ==========================================================================
// 8. PACS Interactive Zoom & Synchronized Pan Engine
// ==========================================================================
function aplicarTransformacaoZoom() {
    zoomFactorLabel.textContent = `${Math.round(zoomLevel * 100)}%`;

    const transStr = `translate(${panX}px, ${panY}px) scale(${zoomLevel})`;

    document.querySelectorAll('.zoomable-img').forEach(img => {
        img.style.transform = transStr;
    });

    document.querySelectorAll('.pacs-viewport').forEach(vp => {
        if (zoomLevel > 1.01) {
            vp.classList.add('is-zoomed');
        } else {
            vp.classList.remove('is-zoomed');
            vp.classList.remove('is-dragging');
        }
    });
}

function ajustarZoom(delta) {
    zoomLevel = Math.max(1.0, Math.min(5.0, zoomLevel + delta));
    if (zoomLevel <= 1.01) {
        panX = 0;
        panY = 0;
    }
    aplicarTransformacaoZoom();
}

function resetarZoom() {
    zoomLevel = 1.0;
    panX = 0;
    panY = 0;
    isPanning = false;
    aplicarTransformacaoZoom();
}

btnZoomIn.addEventListener('click', () => ajustarZoom(0.25));
btnZoomOut.addEventListener('click', () => ajustarZoom(-0.25));
btnZoomReset.addEventListener('click', () => resetarZoom());

// Mouse Wheel Zoom & Drag Panning on PACS Viewports
document.querySelectorAll('.pacs-viewport').forEach(vp => {
    // Wheel Zoom
    vp.addEventListener('wheel', (e) => {
        e.preventDefault();
        const delta = e.deltaY < 0 ? 0.2 : -0.2;
        ajustarZoom(delta);
    }, { passive: false });

    // Double Click to Reset / Quick 2x Zoom
    vp.addEventListener('dblclick', () => {
        if (zoomLevel > 1.01) {
            resetarZoom();
        } else {
            zoomLevel = 2.0;
            aplicarTransformacaoZoom();
        }
    });

    // Pan (Drag) Start
    vp.addEventListener('mousedown', (e) => {
        if (zoomLevel > 1.01) {
            isPanning = true;
            startPanX = e.clientX - panX;
            startPanY = e.clientY - panY;
            vp.classList.add('is-dragging');
        }
    });
});

window.addEventListener('mousemove', (e) => {
    if (!isPanning) return;
    panX = e.clientX - startPanX;
    panY = e.clientY - startPanY;

    // Constrain Pan limits based on zoom level
    const maxPan = 140 * (zoomLevel - 1.0);
    panX = Math.max(-maxPan, Math.min(maxPan, panX));
    panY = Math.max(-maxPan, Math.min(maxPan, panY));

    aplicarTransformacaoZoom();
});

window.addEventListener('mouseup', () => {
    if (isPanning) {
        isPanning = false;
        document.querySelectorAll('.pacs-viewport').forEach(vp => {
            vp.classList.remove('is-dragging');
        });
    }
});

// ==========================================================================
// 9. History Database Management (SQLite)
// ==========================================================================
btnRefreshHistory.addEventListener('click', () => carregarHistorico());

btnClearHistory.addEventListener('click', async () => {
    if (confirm('Deseja realmente limpar todo o histórico de exames salvos no banco SQLite?')) {
        try {
            await fetch(`${API_BASE_URL}/api/historico`, { method: 'DELETE' });
            carregarHistorico();
        } catch (e) {
            console.error('Erro ao limpar histórico:', e);
        }
    }
});

async function carregarHistorico() {
    try {
        const res = await fetch(`${API_BASE_URL}/api/historico`);
        if (res.ok) {
            const data = await res.json();
            renderizarTabelaHistorico(data.historico || []);
        }
    } catch (e) {
        historyTbody.innerHTML = `<tr><td colspan="7" class="table-empty">Falha ao carregar histórico da API.</td></tr>`;
    }
}

function renderizarTabelaHistorico(lista) {
    if (lista.length === 0) {
        historyTbody.innerHTML = `<tr><td colspan="7" class="table-empty">Nenhum exame gravado no banco de dados ainda.</td></tr>`;
        return;
    }

    historyTbody.innerHTML = lista.map(item => {
        const isDoente = item.classe_id === 1;
        const tagClass = isDoente ? 'badge-alert' : 'badge-normal';
        const tagText = isDoente ? 'Alteração' : 'Normal';

        return `
            <tr>
                <td>#${item.id}</td>
                <td>${item.data_hora}</td>
                <td><strong>${item.nome_arquivo}</strong></td>
                <td>${item.modelo_utilizado}</td>
                <td><span class="table-badge ${tagClass}">${tagText}</span></td>
                <td><strong>${item.confianca.toFixed(1)}%</strong></td>
                <td>${item.tempo_ms} ms</td>
            </tr>
        `;
    }).join('');
}
