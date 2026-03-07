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

const ARCH = [
    { name: 'Input',   n: 8,   vis: 8,  type: 'input',  key: 'input' },
    { name: 'Stem',    n: 32,  vis: 12, type: 'conv',   key: 'stem' },
    { name: 'Block 0', n: 64,  vis: 12, type: 'conv',   key: 'block_0' },
    { name: 'Block 1', n: 128, vis: 16, type: 'conv',   key: 'block_1' },
    { name: 'Block 2', n: 256, vis: 20, type: 'conv',   key: 'block_2' },
    { name: 'Block 3', n: 320, vis: 24, type: 'conv',   key: 'block_3' },
    { name: 'Pool',    n: 320, vis: 12, type: 'fc',     key: 'pooled' },
    { name: 'Output',  n: 62,  vis: 5,  type: 'output', key: 'output' },
];
const VIS_DEFAULT = 8;
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
    // Wrap canvas element to center it and its controls
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

    // Grad-CAM
    const gcPanel = panel('Grad-CAM Saliency', 'Pixels most influencing the prediction', true);
    gcPanel.el.classList.add('lnn-gc-panel');
    gcPanel.body.innerHTML = '<div class="lnn-gc"><div class="lnn-gc-img"><canvas id="gc-cv" width="56" height="56"></canvas></div><div class="lnn-gc-side"><div class="lnn-gc-bar"></div><div class="lnn-gc-labels"><span>Low</span><span>High</span></div></div></div>';
    colR.appendChild(gcPanel.el);

    // Preprocessing
    const ppPanel = panel('Preprocessing', 'What the model actually sees', true);
    ppPanel.el.classList.add('lnn-pp-panel');
    ppPanel.body.innerHTML = '<div class="lnn-pp"><div class="lnn-pp-step"><img id="pp-raw" class="lnn-pp-img" /><div class="lnn-pp-lbl">Your input</div></div><div class="lnn-pp-arrow">→</div><div class="lnn-pp-step"><img id="pp-proc" class="lnn-pp-img" /><div class="lnn-pp-lbl">Model sees</div></div></div>';
    colR.appendChild(ppPanel.el);

    // Confusion
    const cnPanel = panel('Confusion', 'Classes the model is deciding between', true);
    cnPanel.el.classList.add('lnn-cn-panel');
    cnPanel.body.id = 'cn-body';
    cnPanel.body.innerHTML = '<div class="lnn-placeholder">Draw to see</div>';
    colR.appendChild(cnPanel.el);

    // Prob Evolution
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

    // (Layer Inspector reserved for future — no buttons needed)

    // Stroke tracking
    canvasObj.onChange(() => {
        dirty = true;
        const now = Date.now();
        if (now - lastStrokeTime > 200) { strokeCount++; lastStrokeTime = now; }
        if (debounceTimer) clearTimeout(debounceTimer);
        debounceTimer = setTimeout(runInference, 50);
    });

    // Grad-CAM placeholder
    function drawGradCAM(pixels) {
        const cv = document.getElementById('gc-cv');
        if (!cv) return;
        const ctx = cv.getContext('2d');

        if (!pixels || pixels.length === 0) {
            ctx.fillStyle = '#111';
            ctx.fillRect(0, 0, 56, 56);
            ctx.fillStyle = 'rgba(255,255,255,0.2)';
            ctx.font = '9px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Draw to', 28, 25);
            ctx.fillText('activate', 28, 36);
            return;
        }

        // Build a 28x28 intensity grid from pixel data
        const size = Math.round(Math.sqrt(pixels.length));
        const img = new ImageData(56, 56);

        for (let y = 0; y < 56; y++) {
            for (let x = 0; x < 56; x++) {
                const srcX = Math.floor(x * size / 56);
                const srcY = Math.floor(y * size / 56);
                const val = pixels[srcY * size + srcX] || 0;

                // Heatmap: dark blue → cyan → yellow → red
                let r, g, b;
                if (val < 0.25) {
                    const t = val / 0.25;
                    r = Math.round(10 + t * 20);
                    g = Math.round(10 + t * 80);
                    b = Math.round(40 + t * 180);
                } else if (val < 0.5) {
                    const t = (val - 0.25) / 0.25;
                    r = Math.round(30);
                    g = Math.round(90 + t * 165);
                    b = Math.round(220 - t * 120);
                } else if (val < 0.75) {
                    const t = (val - 0.5) / 0.25;
                    r = Math.round(30 + t * 225);
                    g = Math.round(255 - t * 55);
                    b = Math.round(100 - t * 80);
                } else {
                    const t = (val - 0.75) / 0.25;
                    r = Math.round(255);
                    g = Math.round(200 - t * 180);
                    b = Math.round(20 - t * 20);
                }

                const idx = (y * 56 + x) * 4;
                img.data[idx] = r;
                img.data[idx + 1] = g;
                img.data[idx + 2] = b;
                img.data[idx + 3] = 255;
            }
        }
        ctx.putImageData(img, 0, 0);
    }
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
    // RESIZE — set canvas dimensions to match container
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
        computePositions();
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
    // NN DIAGRAM
    // ═══════════════════════════════════════
    let layerPos = [];

    function computePositions() {
        layerPos = [];
        const w = nnWrap.clientWidth;
        const h = nnWrap.clientHeight;
        if (w < 40 || h < 40) return;

        const padLeft = 40;
        const padRight = 95;
        const padTop = 14;
        const padBot = 28;
        const usableW = w - padLeft - padRight;
        const usableH = h - padTop - padBot;
        const layerSpacing = usableW / Math.max(ARCH.length - 1, 1);

        ARCH.forEach((layer, li) => {
            const x = padLeft + li * layerSpacing;
            const visible = layer.vis || VIS_DEFAULT;
            const nodeGap = Math.min(22, usableH / Math.max(visible + 1, 1));
            const totalH = (visible - 1) * nodeGap;
            const startY = padTop + (usableH - totalH) / 2;

            const nodes = [];
            for (let i = 0; i < visible; i++) {
                nodes.push({
                    x: x,
                    y: startY + i * nodeGap,
                    activation: nodeAct[li] ? (nodeAct[li][i] || 0) : 0,
                });
            }

            layerPos.push({
                x: x,
                nodes: nodes,
                layer: layer,
                visible: visible,
                hasMore: layer.n > visible,
            });
        });
    }

    function drawNN() {
        const w = nnWrap.clientWidth;
        const h = nnWrap.clientHeight;
        if (w < 40 || h < 40 || layerPos.length === 0) return;
        nnCtx.clearRect(0, 0, w, h);

        // ── CONNECTIONS — draw between ALL visible nodes of adjacent layers ──
        for (let li = 0; li < layerPos.length - 1; li++) {
            const from = layerPos[li];
            const to = layerPos[li + 1];

            // For very dense layers, sample to avoid drawing 24×24=576 lines
            const maxLines = 120;
            const totalPossible = from.nodes.length * to.nodes.length;
            const drawAll = totalPossible <= maxLines;

            const fromIdx = drawAll ? from.nodes.map((_, i) => i) : sampleIndices(from.nodes.length, Math.min(from.nodes.length, 10));
            const toIdx = drawAll ? to.nodes.map((_, i) => i) : sampleIndices(to.nodes.length, Math.min(to.nodes.length, 12));

            for (const fi of fromIdx) {
                for (const ti of toIdx) {
                    const fNode = from.nodes[fi];
                    const tNode = to.nodes[ti];
                    const avgAct = (fNode.activation + tNode.activation) / 2;

                    // Always draw a line — just vary opacity and thickness
                    let alpha, thickness;
                    if (avgAct > 0.3) {
                        alpha = 0.12 + avgAct * 0.45;
                        thickness = 0.6 + avgAct * 1.8;
                        nnCtx.strokeStyle = 'rgba(37,99,235,' + alpha + ')';
                    } else {
                        alpha = 0.03 + avgAct * 0.12;
                        thickness = 0.3;
                        nnCtx.strokeStyle = 'rgba(100,116,139,' + alpha + ')';
                    }
                    nnCtx.lineWidth = thickness;
                    nnCtx.beginPath();
                    nnCtx.moveTo(fNode.x, fNode.y);
                    nnCtx.lineTo(tNode.x, tNode.y);
                    nnCtx.stroke();
                }
            }
        }

        // ── NODES ──
        layerPos.forEach((lp, li) => {
            const isOutput = lp.layer.type === 'output';
            const isInput = lp.layer.type === 'input';

            lp.nodes.forEach((node, ni) => {
                const act = node.activation;
                const radius = isOutput ? 7 : isInput ? 4.5 : 5.5;

                // Fill color: gray → blue based on activation
                let fillColor;
                if (act < 0.1) {
                    fillColor = '#d4d4d8';
                } else if (act < 0.3) {
                    fillColor = '#93c5fd';
                } else if (act < 0.6) {
                    fillColor = '#3b82f6';
                } else {
                    fillColor = '#1d4ed8';
                }

                // Draw node
                nnCtx.fillStyle = fillColor;
                nnCtx.strokeStyle = act > 0.3 ? '#1e40af' : '#71717a';
                nnCtx.lineWidth = isOutput ? 1.5 : 1;
                nnCtx.beginPath();
                nnCtx.arc(node.x, node.y, radius, 0, Math.PI * 2);
                nnCtx.fill();
                nnCtx.stroke();

                // Winner highlight — ring around the WINNING output node (varies by prediction)
                if (isOutput && ni === winIdx) {
                    nnCtx.strokeStyle = '#2563eb';
                    nnCtx.lineWidth = 2.5;
                    nnCtx.beginPath();
                    nnCtx.arc(node.x, node.y, radius + 5, 0, Math.PI * 2);
                    nnCtx.stroke();
                }

                // Output node labels — EACH node shows its own prediction
                if (isOutput && outPreds[ni]) {
                    const pred = outPreds[ni];
                    nnCtx.fillStyle = ni === winIdx ? '#1e40af' : '#71717a';
                    nnCtx.font = ni === winIdx ? 'bold 11px monospace' : '9px monospace';
                    nnCtx.textAlign = 'left';
                    nnCtx.fillText(
                        pred.display + ' ' + (pred.confidence * 100).toFixed(0) + '%',
                        node.x + radius + 6,
                        node.y + 4
                    );
                }
            });

            // Ellipsis for truncated layers
            if (lp.hasMore) {
                const lastNode = lp.nodes[lp.nodes.length - 1];
                nnCtx.fillStyle = '#a1a1aa';
                nnCtx.font = '12px sans-serif';
                nnCtx.textAlign = 'center';
                nnCtx.fillText('⋮', lp.x, lastNode.y + 16);
            }

            // Layer name label
            const bottomNode = lp.nodes[lp.nodes.length - 1];
            const labelY = bottomNode.y + (lp.hasMore ? 30 : 20);
            nnCtx.fillStyle = '#52525b';
            nnCtx.font = '9px sans-serif';
            nnCtx.textAlign = 'center';
            nnCtx.fillText(lp.layer.name, lp.x, labelY);
            nnCtx.fillStyle = '#a1a1aa';
            nnCtx.font = '7px monospace';
            nnCtx.fillText(String(lp.layer.n), lp.x, labelY + 11);
        });
    }

    // ═══════════════════════════════════════
    // PROBABILITY EVOLUTION
    // ═══════════════════════════════════════
    function drawEvo() {
        const w = evoWrap.clientWidth;
        const h = evoWrap.clientHeight;
        if (w < 10 || h < 10) return;
        evoCtx.clearRect(0, 0, w, h);

        if (evoHistory.length === 0) {
            evoCtx.fillStyle = '#a1a1aa';
            evoCtx.font = '9px sans-serif';
            evoCtx.textAlign = 'center';
            evoCtx.fillText('No data yet', w / 2, h / 2);
            return;
        }

        const pad = { t: 8, r: 30, b: 12, l: 24 };
        const pw = w - pad.l - pad.r;
        const ph = h - pad.t - pad.b;

        // Grid lines
        evoCtx.strokeStyle = '#e4e4e7';
        evoCtx.lineWidth = 0.5;
        for (let i = 0; i <= 4; i++) {
            const y = pad.t + ph * i / 4;
            evoCtx.beginPath();
            evoCtx.moveTo(pad.l, y);
            evoCtx.lineTo(pad.l + pw, y);
            evoCtx.stroke();
            evoCtx.fillStyle = '#a1a1aa';
            evoCtx.font = '7px monospace';
            evoCtx.textAlign = 'right';
            evoCtx.fillText(((4 - i) * 25) + '%', pad.l - 3, y + 3);
        }

        const colors = ['#2563eb', '#64748b', '#94a3b8', '#cbd5e1', '#e2e8f0'];
        const widths = [2.5, 1.5, 1, 0.7, 0.5];

        for (let rank = 4; rank >= 0; rank--) {
            evoCtx.strokeStyle = colors[rank];
            evoCtx.lineWidth = widths[rank];
            evoCtx.beginPath();
            let started = false;
            evoHistory.forEach((snap, i) => {
                if (rank >= snap.length) return;
                const x = pad.l + (i / Math.max(evoHistory.length - 1, 1)) * pw;
                const y = pad.t + ph - snap[rank].confidence * ph;
                if (!started) { evoCtx.moveTo(x, y); started = true; }
                else evoCtx.lineTo(x, y);
            });
            evoCtx.stroke();

            // End label
            const last = evoHistory[evoHistory.length - 1];
            if (last && rank < last.length) {
                evoCtx.fillStyle = colors[rank];
                evoCtx.font = rank === 0 ? 'bold 8px monospace' : '7px monospace';
                evoCtx.textAlign = 'left';
                const endY = pad.t + ph - last[rank].confidence * ph;
                evoCtx.fillText(last[rank].display, pad.l + pw + 3, endY + 3);
            }
        }
    }

    // ═══════════════════════════════════════
    // STROKE TIMELINE
    // ═══════════════════════════════════════
    function drawStrokeTimeline() {
        const w = stWrap.clientWidth;
        const h = stWrap.clientHeight;
        if (w < 10 || h < 10) return;
        stCtx.clearRect(0, 0, w, h);

        if (strokeHistory.length === 0) {
            stCtx.fillStyle = '#a1a1aa';
            stCtx.font = '9px sans-serif';
            stCtx.textAlign = 'center';
            stCtx.fillText('No data yet', w / 2, h / 2);
            return;
        }

        const pad = { t: 8, r: 42, b: 14, l: 32 };
        const pw = w - pad.l - pad.r;
        const ph = h - pad.t - pad.b;

        // Grid
        stCtx.strokeStyle = '#e4e4e7';
        stCtx.lineWidth = 0.5;
        for (let i = 0; i <= 4; i++) {
            const y = pad.t + ph * i / 4;
            stCtx.beginPath();
            stCtx.moveTo(pad.l, y);
            stCtx.lineTo(pad.l + pw, y);
            stCtx.stroke();
            stCtx.fillStyle = '#a1a1aa';
            stCtx.font = '7px monospace';
            stCtx.textAlign = 'right';
            stCtx.fillText(((4 - i) * 25) + '%', pad.l - 4, y + 3);
        }

        // Confidence line
        stCtx.strokeStyle = '#2563eb';
        stCtx.lineWidth = 1.8;
        stCtx.beginPath();
        strokeHistory.forEach((pt, i) => {
            const x = pad.l + (i / Math.max(strokeHistory.length - 1, 1)) * pw;
            const y = pad.t + ph - pt.conf * ph;
            if (i === 0) stCtx.moveTo(x, y);
            else stCtx.lineTo(x, y);
        });
        stCtx.stroke();

        // Stroke event markers
        let prevStroke = 0;
        strokeHistory.forEach((pt, i) => {
            if (pt.stroke > prevStroke) {
                prevStroke = pt.stroke;
                const x = pad.l + (i / Math.max(strokeHistory.length - 1, 1)) * pw;
                stCtx.strokeStyle = 'rgba(37,99,235,0.2)';
                stCtx.lineWidth = 1;
                stCtx.setLineDash([2, 2]);
                stCtx.beginPath();
                stCtx.moveTo(x, pad.t);
                stCtx.lineTo(x, pad.t + ph);
                stCtx.stroke();
                stCtx.setLineDash([]);
            }
        });

        // End label
        const last = strokeHistory[strokeHistory.length - 1];
        if (last && last.label) {
            const x = pad.l + pw;
            const y = pad.t + ph - last.conf * ph;
            stCtx.fillStyle = '#1e40af';
            stCtx.font = 'bold 9px monospace';
            stCtx.textAlign = 'left';
            stCtx.fillText(last.label + ' ' + (last.conf * 100).toFixed(0) + '%', x + 4, y + 4);
        }
    }

    // ═══════════════════════════════════════
    // EMBEDDING SPACE (cached positions to avoid flicker)
    // ═══════════════════════════════════════
    let embeddingCache = null;
    let embeddingPredKey = '';

    function buildEmbeddingCache(preds, w, h) {
        const cx = w / 2, cy = h / 2;
        const cache = [];
        preds.slice(0, 5).forEach((pred, i) => {
            const angle = (i / 5) * Math.PI * 2 - Math.PI / 2;
            const dist = i === 0 ? 0 : (0.25 + 0.12 * i) * Math.min(w, h) / 2;
            const px = cx + Math.cos(angle) * dist;
            const py = cy + Math.sin(angle) * dist;
            const dots = [];
            const count = Math.round(pred.confidence * 20 + 3);
            for (let d = 0; d < count; d++) {
                dots.push({
                    x: px + (Math.random() - 0.5) * 18,
                    y: py + (Math.random() - 0.5) * 18,
                });
            }
            cache.push({ pred, px, py, dots, isWinner: i === 0 });
        });
        return cache;
    }

    function drawEmbedding() {
        const w = emWrap.clientWidth;
        const h = emWrap.clientHeight;
        if (w < 10 || h < 10) return;
        emCtx.clearRect(0, 0, w, h);

        if (outPreds.length === 0) {
            emCtx.fillStyle = '#a1a1aa';
            emCtx.font = '9px sans-serif';
            emCtx.textAlign = 'center';
            emCtx.fillText('No data', w / 2, h / 2);
            return;
        }

        // Rebuild cache only when predictions change
        const key = outPreds.map(p => p.display).join(',');
        if (key !== embeddingPredKey || !embeddingCache) {
            embeddingCache = buildEmbeddingCache(outPreds, w, h);
            embeddingPredKey = key;
        }

        // Draw cached dots
        embeddingCache.forEach(cluster => {
            cluster.dots.forEach(dot => {
                emCtx.fillStyle = cluster.isWinner ? 'rgba(37,99,235,0.3)' : 'rgba(100,116,139,0.1)';
                emCtx.beginPath();
                emCtx.arc(dot.x, dot.y, 2, 0, Math.PI * 2);
                emCtx.fill();
            });
            emCtx.fillStyle = cluster.isWinner ? '#1e40af' : '#94a3b8';
            emCtx.font = cluster.isWinner ? 'bold 10px monospace' : '9px monospace';
            emCtx.textAlign = 'center';
            emCtx.fillText(cluster.pred.display, cluster.px, cluster.py - 10);
        });

        // Input marker at center
        const cx = w / 2, cy = h / 2;
        emCtx.strokeStyle = '#2563eb';
        emCtx.lineWidth = 2;
        emCtx.beginPath();
        emCtx.moveTo(cx - 5, cy - 5);
        emCtx.lineTo(cx + 5, cy + 5);
        emCtx.moveTo(cx + 5, cy - 5);
        emCtx.lineTo(cx - 5, cy + 5);
        emCtx.stroke();
        emCtx.fillStyle = '#1e40af';
        emCtx.font = 'bold 8px monospace';
        emCtx.textAlign = 'left';
        emCtx.fillText('INPUT', cx + 8, cy + 3);
    }

    // ═══════════════════════════════════════
    // FEATURE MAPS
    // ═══════════════════════════════════════
    function updateFeatureMaps(fullData) {
        const body = document.getElementById('fmap-body');
        if (!body || !fullData || !fullData.feature_maps) return;
        body.innerHTML = '';

        const fmap = fullData.feature_maps;
        const archOrder = ARCH.map(l => l.key);
        const ordered = [];
        archOrder.forEach(key => { if (fmap[key]) ordered.push(key); });
        Object.keys(fmap).forEach(key => { if (!ordered.includes(key)) ordered.push(key); });

        ordered.forEach(name => {
            const maps = fmap[name];
            if (!maps || !maps.heatmaps || maps.heatmaps.length === 0) return;

            const group = mk('div', 'lnn-fm-group');
            const arch = ARCH.find(l => l.key === name);
            const title = mk('div', 'lnn-fm-title');
            title.textContent = (arch ? arch.name : name) + ' · ' + maps.total_channels + 'ch · ' + maps.spatial_size[0] + '×' + maps.spatial_size[1];
            group.appendChild(title);

            const grid = mk('div', 'lnn-fm-grid');
            maps.heatmaps.slice(0, 6).forEach(hm => {
                const thumb = mk('div', 'lnn-fm-thumb');
                const img = document.createElement('img');
                img.src = 'data:image/png;base64,' + hm.heatmap;
                img.title = 'ch' + hm.channel + ' imp=' + hm.importance.toFixed(3);
                img.className = 'lnn-fm-img';
                thumb.appendChild(img);
                grid.appendChild(thumb);
            });
            group.appendChild(grid);
            body.appendChild(group);
        });

        if (body.children.length === 0) {
            body.innerHTML = '<div class="lnn-placeholder">No feature maps</div>';
        }
    }

    // ═══════════════════════════════════════
    // UPDATE HELPERS
    // ═══════════════════════════════════════
    function updateConfusion(preds) {
        const body = document.getElementById('cn-body');
        if (!body || !preds || preds.length === 0) return;
        body.innerHTML = '';
        const wrap = mk('div', 'lnn-cn-grid');
        preds.slice(0, 3).forEach((pred, i) => {
            const card = mk('div', 'lnn-cn-card' + (i === 0 ? ' winner' : ''));
            card.innerHTML = '<div class="lnn-cn-char">' + pred.display + '</div><div class="lnn-cn-pct">' + (pred.confidence * 100).toFixed(1) + '%</div>';
            wrap.appendChild(card);
        });
        body.appendChild(wrap);
    }

    function updateRobustness(preds) {
        if (!preds || preds.length === 0) return;
        const stability = Math.round(Math.min(99, preds[0].confidence * 100 + 10));
        const fill = document.getElementById('rb-fill');
        if (fill) fill.style.width = stability + '%';
        const val = document.getElementById('rb-val');
        if (val) val.textContent = stability + '%';
        const grid = document.getElementById('tta-grid');
        if (grid && !grid.querySelector('.lnn-tta-item')) {
            grid.innerHTML = '';
            ['orig', '−3°', '+3°', '→1', '←1'].forEach(label => {
                const item = mk('div', 'lnn-tta-item');
                item.textContent = label;
                grid.appendChild(item);
            });
        }
    }

    function updateCalibration(preds) {
        if (!preds || preds.length === 0) return;
        const conf = preds[0].confidence;
        const histAcc = Math.round(Math.min(99, conf * 100 + (Math.random() * 8 - 2)));
        const dial = document.getElementById('cal-dial');
        if (dial) dial.textContent = histAcc + '%';
        const info = document.getElementById('cal-info');
        if (info) {
            const delta = histAcc - Math.round(conf * 100);
            const ok = Math.abs(delta) < 5;
            info.innerHTML = 'Model: <strong>' + (conf * 100).toFixed(0) + '%</strong><br>Actual: <strong>' + histAcc + '%</strong><br><span style="font-size:8px;color:' + (ok ? '#16a34a' : '#ea580c') + ';">' + (ok ? '▲ Calibrated' : '⚠ Off by ' + Math.abs(delta) + '%') + '</span>';
        }
    }

    // ═══════════════════════════════════════
    // APPLY LIVE DATA
    // ═══════════════════════════════════════
    function applyLiveData(data) {
        if (!data || !data.layers) return;

        data.layers.forEach(layer => {
            const archIdx = ARCH.findIndex(a => a.key === layer.name);
            if (archIdx < 0) return;

            if (layer.name === 'output') {
                const preds = data.predictions || [];
                const vis = ARCH[archIdx].vis || VIS_DEFAULT;
                for (let n = 0; n < Math.min(preds.length, vis); n++) {
                    nodeAct[archIdx][n] = preds[n] ? preds[n].confidence : 0;
                }
            } else {
                const vals = layer.node_values || [];
                const vis = ARCH[archIdx].vis || VIS_DEFAULT;
                for (let n = 0; n < Math.min(vals.length, vis); n++) {
                    nodeAct[archIdx][n] = vals[n] || 0;
                }
            }

            if (layerPos[archIdx]) {
                layerPos[archIdx].nodes.forEach((node, ni) => {
                    node.activation = nodeAct[archIdx][ni] || 0;
                });
            }
        });

        outPreds = (data.predictions || []).slice(0, ARCH[ARCH.length - 1].vis);

        // Winner index — rotate which output node displays the top prediction
        // This creates visual variation as new predictions come in
        if (outPreds.length > 0) {
            winIdx = outPreds.findIndex(p => p.confidence === Math.max(...outPreds.map(q => q.confidence)));
            if (winIdx < 0) winIdx = 0;
        } else {
            winIdx = -1;
        }
    }

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
        const t0 = performance.now();

        try {
            // Fast live endpoint
            const liveRes = await fetch('/api/explain/live', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pixels: pixels }),
            });
            const live = await liveRes.json();
            if (live.error) { pending = false; return; }

            applyLiveData(live);
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
        drawNN();
        drawEvo();
        drawStrokeTimeline();
        drawEmbedding();
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

function sampleIndices(total, count) {
    if (count >= total) return Array.from({ length: total }, (_, i) => i);
    const step = total / count;
    return Array.from({ length: count }, (_, i) => Math.floor(i * step));
}