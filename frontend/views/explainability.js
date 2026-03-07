/**
 * NeuralScribe v2 — Live Neural Network (Production)
 *
 * Layout (all viewport-fit, zero scroll):
 * ┌─────────┬──────────────────────────┬──────────┐
 * │  DRAW   │     NN DIAGRAM           │ GRADCAM  │
 * │         │                          │ PREPROC  │
 * │ PREDICT │                          │ CONFUSE  │
 * │         │                          │ PROB EVO │
 * ├─────────┴────────────┬─────────────┼──────────┤
 * │   FEATURE MAPS       │ LAYER DIVE  │ ROBUST   │
 * │                      │             │ CALIB    │
 * │   STROKE TIMELINE    │             │ EMBED    │
 * └──────────────────────┴─────────────┴──────────┘
 */
import { createCanvas } from '../components/canvas.js';
import { createConfidenceBars, updateConfidenceBars } from '../components/confidenceBars.js';
import { ARCH, VIS_DEFAULT, computePositions, drawNN, applyLiveData } from '../components/nnDiagram.js';
import { drawEvo, drawStrokeTimeline, drawEmbedding, resetEmbeddingCache } from '../components/lnnCharts.js';
import { drawGradCAM, updateFeatureMaps, updateConfusion, updateRobustness, updateCalibration } from '../components/lnnPanels.js';

let _af = null, _dead = false;

export async function renderExplainability(container) {
    if (_af) { cancelAnimationFrame(_af); _af = null; }
    _dead = false;
    container.innerHTML = '';
    container.className = 'view-fit';

    let ready = false;
    try { ready = (await (await fetch('/api/inference/status')).json()).ready; } catch(e) {}
    if (!ready) {
        container.innerHTML = '<div class="view-title">Live Neural Network</div><div class="panel"><div class="panel-body text-muted">No model loaded. Go to Models → Load.</div></div>';
        return;
    }

    // ═══════════════════════════════════════
    // BUILD DOM
    // ═══════════════════════════════════════

    // Header
    const hdr = mk('div', 'lnn-hdr');
    hdr.innerHTML = '<span class="lnn-title">Live Neural Network</span><span class="lnn-status" id="ex-status">Draw a letter or digit to begin</span>';
    container.appendChild(hdr);

    const layout = mk('div', 'lnn-layout');
    container.appendChild(layout);

    const topRow = mk('div', 'lnn-top');
    layout.appendChild(topRow);
    const botRow = mk('div', 'lnn-bot');
    layout.appendChild(botRow);

    // ────────── TOP LEFT: Draw + Predict ──────────
    const colL = mk('div', 'lnn-col-l');
    topRow.appendChild(colL);

    const drawPanel = panel('Draw', 'Real-time recognition as you draw');
    drawPanel.el.classList.add('lnn-draw-panel');
    const canvasObj = createCanvas(160);
    drawPanel.body.classList.add('lnn-draw-body');
    const canvasWrapper = canvasObj.element;
    canvasWrapper.style.cssText = 'display:flex;flex-direction:column;align-items:center;width:100%;';
    drawPanel.body.appendChild(canvasWrapper);
    colL.appendChild(drawPanel.el);

    const predPanel = panel('Prediction', 'Top-5 most likely characters');
    predPanel.el.classList.add('lnn-pred-panel');
    const bigPred = mk('div', 'lnn-big');
    bigPred.textContent = '?';
    const confTxt = mk('div', 'lnn-conf');
    confTxt.textContent = '—';
    const barsEl = createConfidenceBars([]);
    predPanel.body.append(bigPred, confTxt, barsEl);
    colL.appendChild(predPanel.el);

    // ────────── TOP CENTER: NN Diagram ──────────
    const colC = mk('div', 'lnn-col-c');
    topRow.appendChild(colC);

    const nnPanel = panel('Neural Network', 'Signal flow — brighter = stronger activation');
    nnPanel.el.classList.add('lnn-nn-panel');
    nnPanel.body.classList.add('lnn-nn-body');
    const nnWrap = mk('div', 'lnn-cvwrap');
    const nnCv = document.createElement('canvas');
    nnCv.classList.add('lnn-cv');
    nnWrap.appendChild(nnCv);
    nnPanel.body.appendChild(nnWrap);
    colC.appendChild(nnPanel.el);

    // ────────── TOP RIGHT: GradCAM / Preprocess / Confusion / ProbEvo ──────────
    const colR = mk('div', 'lnn-col-r');
    topRow.appendChild(colR);

    const gcPanel = panel('Grad-CAM Saliency', 'Pixels most influencing the prediction', true);
    gcPanel.el.classList.add('lnn-gc-panel');
    gcPanel.body.innerHTML = '<div class="lnn-gc"><div class="lnn-gc-img"><canvas id="gc-cv" width="56" height="56"></canvas></div><div class="lnn-gc-side"><div class="lnn-gc-bar"></div><div class="lnn-gc-labels"><span>Low</span><span>High</span></div></div></div>';
    colR.appendChild(gcPanel.el);

    const ppPanel = panel('Preprocessing', 'What the model actually sees', true);
    ppPanel.el.classList.add('lnn-pp-panel');
    ppPanel.body.innerHTML = '<div class="lnn-pp"><div class="lnn-pp-step"><img id="pp-raw" class="lnn-pp-img" /><div class="lnn-pp-lbl">Your input</div></div><div class="lnn-pp-arrow">→</div><div class="lnn-pp-step"><img id="pp-proc" class="lnn-pp-img" /><div class="lnn-pp-lbl">Model sees</div></div></div>';
    colR.appendChild(ppPanel.el);

    const cnPanel = panel('Confusion', 'Classes the model is deciding between', true);
    cnPanel.el.classList.add('lnn-cn-panel');
    cnPanel.body.id = 'cn-body';
    cnPanel.body.innerHTML = '<div class="lnn-placeholder">Draw to see</div>';
    colR.appendChild(cnPanel.el);

    const evoPanel = panel('Probability Evolution', 'Confidence changes as you draw');
    evoPanel.el.classList.add('lnn-evo-panel');
    const evoWrap = mk('div', 'lnn-cvwrap');
    const evoCv = document.createElement('canvas');
    evoCv.classList.add('lnn-cv');
    evoWrap.appendChild(evoCv);
    evoPanel.body.classList.add('lnn-cv-body');
    evoPanel.body.appendChild(evoWrap);
    colR.appendChild(evoPanel.el);

    // ────────── BOTTOM LEFT: Feature Maps + Stroke Timeline ──────────
    const botL = mk('div', 'lnn-bot-l');
    botRow.appendChild(botL);

    const fmPanel = panel('Feature Maps', 'Top activation channels — patterns the network detects');
    fmPanel.el.classList.add('lnn-fm-panel');
    fmPanel.body.id = 'fmap-body';
    fmPanel.body.classList.add('lnn-fm-body');
    fmPanel.body.innerHTML = '<div class="lnn-placeholder">Draw to see activations</div>';
    botL.appendChild(fmPanel.el);

    const stPanel = panel('Stroke Timeline', 'Confidence over time — dashed lines mark each stroke', true);
    stPanel.el.classList.add('lnn-st-panel');
    const stWrap = mk('div', 'lnn-cvwrap');
    const stCv = document.createElement('canvas');
    stCv.classList.add('lnn-cv');
    stWrap.appendChild(stCv);
    stPanel.body.classList.add('lnn-cv-body');
    stPanel.body.appendChild(stWrap);
    botL.appendChild(stPanel.el);

    // ────────── BOTTOM CENTER: Reserved Space ──────────
    const ldPanel = panel('Layer Inspector', 'Reserved for future use');
    ldPanel.el.classList.add('lnn-ld-panel');
    ldPanel.body.innerHTML = '<div class="lnn-placeholder" style="display:flex;align-items:center;justify-content:center;flex:1;font-size:11px;color:var(--muted);">Coming soon</div>';
    botRow.appendChild(ldPanel.el);

    // ────────── BOTTOM RIGHT: Robustness + Calibration + Embedding ──────────
    const botR = mk('div', 'lnn-bot-r');
    botRow.appendChild(botR);

    const rbPanel = panel('Robustness', 'Stability across rotations and shifts', true);
    rbPanel.el.classList.add('lnn-rb-panel');
    rbPanel.body.innerHTML = '<div class="lnn-rb"><div class="lnn-rb-track"><div class="lnn-rb-fill" id="rb-fill"></div></div><span class="lnn-rb-val" id="rb-val">—</span></div><div class="lnn-tta" id="tta-grid"></div>';
    botR.appendChild(rbPanel.el);

    const calPanel = panel('Calibration', 'Is the confidence trustworthy?', true);
    calPanel.el.classList.add('lnn-cal-panel');
    calPanel.body.innerHTML = '<div class="lnn-cal"><div class="lnn-cal-ring" id="cal-dial">—</div><div class="lnn-cal-txt" id="cal-info"><span class="lnn-placeholder">Draw to see</span></div></div>';
    botR.appendChild(calPanel.el);

    const emPanel = panel('Embedding Space', 'Your input vs class clusters', true);
    emPanel.el.classList.add('lnn-em-panel');
    const emWrap = mk('div', 'lnn-cvwrap');
    const emCv = document.createElement('canvas');
    emCv.classList.add('lnn-cv');
    emWrap.appendChild(emCv);
    emPanel.body.classList.add('lnn-cv-body');
    emPanel.body.appendChild(emWrap);
    botR.appendChild(emPanel.el);

    // ═══════════════════════════════════════
    // STATE
    // ═══════════════════════════════════════
    let nodeAct = ARCH.map(l => new Array(l.vis || VIS_DEFAULT).fill(0));
    let outPreds = [], winIdx = -1, evoHistory = [], strokeHistory = [];
    let lastFull = 0, pending = false, dirty = false, sysText = '';
    let cachedPixels = null, strokeCount = 0, lastStrokeTime = 0;
    const nnCtx = nnCv.getContext('2d');
    const evoCtx = evoCv.getContext('2d');
    const stCtx = stCv.getContext('2d');
    const emCtx = emCv.getContext('2d');
    let sysTimer = null;
    let layerPos = [];

    // Stroke tracking
    canvasObj.onChange(() => {
        dirty = true;
        const now = Date.now();
        if (now - lastStrokeTime > 200) { strokeCount++; lastStrokeTime = now; }
        if (debounceTimer) clearTimeout(debounceTimer);
        debounceTimer = setTimeout(runInference, 50);
    });

    drawGradCAM(null);

    // ═══════════════════════════════════════
    // RESET
    // ═══════════════════════════════════════
    function reset() {
        nodeAct = ARCH.map(l => new Array(l.vis || VIS_DEFAULT).fill(0));
        outPreds = [];
        winIdx = -1;
        evoHistory = [];
        strokeHistory = [];
        strokeCount = 0;
        resetEmbeddingCache();
        bigPred.textContent = '?';
        confTxt.textContent = '—';
        updateConfidenceBars(barsEl, []);
        const ids = ['fmap-body', 'cn-body', 'ld-grid', 'tta-grid'];
        ids.forEach(id => {
            const el = document.getElementById(id);
            if (el) el.innerHTML = '<div class="lnn-placeholder">Draw to see</div>';
        });
        const rbFill = document.getElementById('rb-fill');
        if (rbFill) rbFill.style.width = '0%';
        const rbVal = document.getElementById('rb-val');
        if (rbVal) rbVal.textContent = '—';
        const calDial = document.getElementById('cal-dial');
        if (calDial) calDial.textContent = '—';
        const calInfo = document.getElementById('cal-info');
        if (calInfo) calInfo.innerHTML = '<span class="lnn-placeholder">Draw to see</span>';
        const status = document.getElementById('ex-status');
        if (status) status.textContent = 'Draw a letter or digit to begin';
        drawGradCAM(null);
    }

    // ═══════════════════════════════════════
    // RESIZE
    // ═══════════════════════════════════════
    function resize() {
        const dpr = window.devicePixelRatio || 1;
        [nnWrap, evoWrap, stWrap, emWrap].forEach(wrapper => {
            const cv = wrapper.querySelector('canvas');
            if (!cv) return;
            const w = wrapper.clientWidth;
            const h = wrapper.clientHeight;
            if (w > 0 && h > 0) {
                cv.width = w * dpr;
                cv.height = h * dpr;
                cv.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
            }
        });
        const w = nnWrap.clientWidth;
        const h = nnWrap.clientHeight;
        layerPos = computePositions(w, h, nodeAct);
    }

    const resizeObserver = new ResizeObserver(resize);
    [nnWrap, evoWrap, stWrap, emWrap].forEach(w => resizeObserver.observe(w));
    setTimeout(resize, 120);

    // System polling
    async function pollSystem() {
        try {
            const [gpuRes, sysRes] = await Promise.all([
                fetch('/api/system/gpu'),
                fetch('/api/system/stats'),
            ]);
            const gpu = await gpuRes.json();
            const sys = await sysRes.json();
            sysText = 'CPU:' + sys.cpu_percent + '% RAM:' + sys.ram_used_gb + '/' + sys.ram_total_gb + 'GB';
            if (gpu.available) sysText += ' GPU:' + gpu.gpu_util_percent + '% VRAM:' + gpu.memory_used_mb + 'MB';
        } catch (e) { /* silent */ }
    }
    pollSystem();
    sysTimer = setInterval(pollSystem, 5000);

    // ═══════════════════════════════════════
    // INFERENCE LOOP
    // ═══════════════════════════════════════
    let debounceTimer = null;

    async function runInference() {
        if (_dead || pending) {
            if (pending) dirty = true;
            return;
        }

        const pixels = canvasObj.getPixels();
        if (pixels.reduce((a, b) => a + b, 0) < 0.5) {
            reset();
            return;
        }

        pending = true;
        cachedPixels = pixels;

        try {
            // Fast live endpoint
            const liveRes = await fetch('/api/explain/live', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pixels: pixels }),
            });
            const live = await liveRes.json();
            if (live.error) { pending = false; return; }

            const result = applyLiveData(live, nodeAct, layerPos);
            outPreds = result.outPreds;
            winIdx = result.winIdx;

            const preds = live.predictions || [];
            updateConfidenceBars(barsEl, preds);
            updateConfusion(preds);
            updateRobustness(preds);
            updateCalibration(preds);
            drawGradCAM(pixels);

            const topPred = preds[0];
            if (topPred) {
                bigPred.textContent = topPred.display;
                confTxt.textContent = (topPred.confidence * 100).toFixed(1) + '% confidence';
            }

            evoHistory.push(preds);
            if (evoHistory.length > 60) evoHistory.shift();

            strokeHistory.push({
                conf: topPred ? topPred.confidence : 0,
                stroke: strokeCount,
                label: topPred ? topPred.display : '',
            });
            if (strokeHistory.length > 120) strokeHistory.shift();

            const statusEl = document.getElementById('ex-status');
            if (statusEl) {
                statusEl.textContent = topPred
                    ? topPred.display + ' · ' + (topPred.confidence * 100).toFixed(1) + '% · ' + (live.inference_time_ms || 0).toFixed(1) + 'ms · ' + sysText
                    : '—';
            }

            // Full explain (feature maps + preprocessed image) every 600ms
            const now = Date.now();
            if (now - lastFull > 600) {
                lastFull = now;
                try {
                    const fullRes = await fetch('/api/explain/full', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ pixels: pixels }),
                    });
                    const full = await fullRes.json();
                    if (!full.error) {
                        updateFeatureMaps(full);
                        if (full.input_image) {
                            const procImg = document.getElementById('pp-proc');
                            if (procImg) procImg.src = 'data:image/png;base64,' + full.input_image;
                        }
                    }
                } catch (e) { /* silent */ }

                // Debug preview for raw
                try {
                    const dbRes = await fetch('/api/inference/debug-preview', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ pixels: pixels }),
                    });
                    const db = await dbRes.json();
                    if (db.image_b64) {
                        const rawImg = document.getElementById('pp-raw');
                        if (rawImg) rawImg.src = 'data:image/png;base64,' + db.image_b64;
                    }
                } catch (e) { /* silent */ }
            }

        } catch (e) {
            console.error('Explain error:', e);
        }

        pending = false;
        if (dirty) {
            dirty = false;
            runInference();
        }
    }

    // ── Animation Loop ──
    function animate() {
        if (_dead) return;
        const w = nnWrap.clientWidth;
        const h = nnWrap.clientHeight;
        drawNN(nnCtx, w, h, layerPos, outPreds, winIdx);
        drawEvo(evoCtx, evoWrap, evoHistory);
        drawStrokeTimeline(stCtx, stWrap, strokeHistory);
        drawEmbedding(emCtx, emWrap, outPreds);
        _af = requestAnimationFrame(animate);
    }
    animate();

    // ── Cleanup ──
    container._cleanup = () => {
        _dead = true;
        if (_af) { cancelAnimationFrame(_af); _af = null; }
        if (debounceTimer) clearTimeout(debounceTimer);
        resizeObserver.disconnect();
        if (sysTimer) clearInterval(sysTimer);
    };
}

// ═══════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════

function mk(tag, className) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    return el;
}

function panel(title, subtitle, isNew) {
    const el = mk('div', 'lnn-panel');
    const header = mk('div', 'lnn-panel-hd');
    header.innerHTML = '<div class="lnn-panel-title">' + title + '</div>' +
        (subtitle ? '<div class="lnn-panel-sub">' + subtitle + '</div>' : '');
    el.appendChild(header);
    const body = mk('div', 'lnn-panel-bd');
    el.appendChild(body);
    return { el: el, body: body };
}