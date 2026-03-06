/**
 * Explainability view — Canvas NN diagram + feature maps + probability evolution.
 * B&W wireframe theme. Real-time: predicts as you draw.
 */
import { createCanvas } from '../components/canvas.js';
import { createConfidenceBars, updateConfidenceBars } from '../components/confidenceBars.js';

// ═══════════════════════════════════════════════════
// ARCHITECTURE DEFINITION
// ═══════════════════════════════════════════════════
const ARCHITECTURE = [
    { name: 'Input',   neurons: 16,  type: 'input',  key: 'input' },
    { name: 'Stem',    neurons: 32,  type: 'conv',   key: 'stem' },
    { name: 'Block 0', neurons: 64,  type: 'conv',   key: 'block_0' },
    { name: 'Block 1', neurons: 128, type: 'conv',   key: 'block_1' },
    { name: 'Block 2', neurons: 256, type: 'conv',   key: 'block_2' },
    { name: 'Block 3', neurons: 320, type: 'conv',   key: 'block_3' },
    { name: 'Pool',    neurons: 320, type: 'fc',     key: 'pooled' },
    { name: 'Output',  neurons: 10,  type: 'output', key: 'output' },
];
const MAX_VISIBLE = 10;

let _animFrame = null;
let _destroyed = false;

export async function renderExplainability(container) {
    if (_animFrame) { cancelAnimationFrame(_animFrame); _animFrame = null; }
    _destroyed = false;
    container.innerHTML = '';
    container.className = 'view-fit';

    // Check model
    let ready = false;
    try { const r = await fetch('/api/inference/status'); ready = (await r.json()).ready; } catch (e) { }
    if (!ready) {
        container.innerHTML = `<div class="view-title">Live Neural Network</div>
            <div class="panel"><div class="panel-body text-muted">No model loaded. Go to Model Manager to load one.</div></div>`;
        return;
    }

    // ── Header ──
    const hdr = document.createElement('div');
    hdr.className = 'ex-header';
    hdr.innerHTML = `<span class="ex-header-title">Live Neural Network</span>
        <span class="text-sm text-muted" id="ex-status">Draw to start</span>`;
    container.appendChild(hdr);

    // ── Layout ──
    const main = document.createElement('div');
    main.className = 'ex-layout';
    container.appendChild(main);

    const topRow = document.createElement('div');
    topRow.className = 'ex-row-top';
    main.appendChild(topRow);

    const bottomRow = document.createElement('div');
    bottomRow.className = 'ex-row-bottom';
    main.appendChild(bottomRow);

    // ═══════ LEFT: Canvas + Predictions ═══════
    const leftCol = document.createElement('div');
    leftCol.className = 'ex-left';
    topRow.appendChild(leftCol);

    const canvasPanel = mkPanel('Draw');
    canvasPanel.el.style.flexShrink = '0';
    const canvasObj = createCanvas(200);
    canvasPanel.body.style.padding = '4px';
    canvasPanel.body.style.display = 'flex';
    canvasPanel.body.style.justifyContent = 'center';
    canvasPanel.body.appendChild(canvasObj.element);
    leftCol.appendChild(canvasPanel.el);

    const predPanel = mkPanel('Prediction');
    predPanel.el.style.flex = '1';
    predPanel.el.style.overflow = 'hidden';
    predPanel.body.style.padding = '6px';
    predPanel.body.style.overflowY = 'auto';

    const bigPred = document.createElement('div');
    bigPred.className = 'ex-big-pred';
    bigPred.textContent = '?';
    predPanel.body.appendChild(bigPred);

    const confText = document.createElement('div');
    confText.className = 'text-sm text-muted';
    confText.style.textAlign = 'center';
    confText.style.marginBottom = '6px';
    confText.textContent = 'Draw a character';
    predPanel.body.appendChild(confText);

    const barsEl = createConfidenceBars([]);
    predPanel.body.appendChild(barsEl);
    leftCol.appendChild(predPanel.el);

    // ═══════ CENTER: NN Diagram ═══════
    const nnPanel = mkPanel('Neural Network — NeuralScribeNet');
    nnPanel.el.style.flex = '1';
    nnPanel.el.style.overflow = 'hidden';
    const nnWrap = document.createElement('div');
    nnWrap.style.cssText = 'flex:1;position:relative;min-height:0;';
    const nnCanvas = document.createElement('canvas');
    nnCanvas.style.cssText = 'width:100%;height:100%;display:block;';
    nnWrap.appendChild(nnCanvas);
    nnPanel.body.appendChild(nnWrap);
    topRow.appendChild(nnPanel.el);

    // ═══════ RIGHT: Prob Evolution ═══════
    const rightTop = document.createElement('div');
    rightTop.className = 'ex-right-top';
    topRow.appendChild(rightTop);

    const evoPanel = mkPanel('Probability Evolution');
    evoPanel.el.style.flex = '1';
    evoPanel.el.style.overflow = 'hidden';
    const evoWrap = document.createElement('div');
    evoWrap.style.cssText = 'flex:1;position:relative;min-height:0;';
    const evoCanvas = document.createElement('canvas');
    evoCanvas.style.cssText = 'width:100%;height:100%;display:block;';
    evoWrap.appendChild(evoCanvas);
    evoPanel.body.appendChild(evoWrap);
    rightTop.appendChild(evoPanel.el);

    // ═══════ BOTTOM: Feature Maps + Performance ═══════
    const bottomMain = document.createElement('div');
    bottomMain.className = 'ex-bottom-main';
    bottomRow.appendChild(bottomMain);

    const fmPanel = mkPanel('Feature Maps');
    fmPanel.el.style.flex = '1';
    fmPanel.el.style.overflow = 'hidden';
    fmPanel.body.style.overflow = 'auto';
    fmPanel.body.style.padding = '6px';
    fmPanel.body.style.display = 'flex';
    fmPanel.body.style.flexWrap = 'wrap';
    fmPanel.body.style.gap = '6px';
    fmPanel.body.style.alignContent = 'flex-start';
    fmPanel.body.id = 'fmap-body';
    fmPanel.body.innerHTML = '<div class="text-sm text-muted">Draw to see activations</div>';
    bottomMain.appendChild(fmPanel.el);

    const bottomSide = document.createElement('div');
    bottomSide.className = 'ex-bottom-side';
    bottomRow.appendChild(bottomSide);

    const perfPanel = mkPanel('Performance');
    perfPanel.el.style.flexShrink = '0';
    perfPanel.body.style.cssText = 'padding:6px;font-size:11px;font-family:var(--font-mono);';
    perfPanel.body.id = 'perf-body';
    perfPanel.body.textContent = '—';
    bottomSide.appendChild(perfPanel.el);

    // ══════════════════════════════════════
    // STATE
    // ══════════════════════════════════════
    let layerPositions = [];
    let nodeActivations = ARCHITECTURE.map(l => new Array(Math.min(l.neurons, MAX_VISIBLE)).fill(0));
    let outputPredictions = [];
    let winnerIdx = -1;
    let evoHistory = [];
    let lastFullTime = 0;
    let pendingRequest = false;
    let dirtySinceLast = false;
    let sysStatsText = '';

    const nnCtx = nnCanvas.getContext('2d');
    const evoCtx = evoCanvas.getContext('2d');
    let sysTimer = null;

    // ══════════════════════════════════════
    // RESET
    // ══════════════════════════════════════
    function resetVisuals() {
        nodeActivations = ARCHITECTURE.map(l => new Array(Math.min(l.neurons, MAX_VISIBLE)).fill(0));
        outputPredictions = [];
        winnerIdx = -1;
        evoHistory = [];
        layerPositions.forEach(lp => lp.nodes.forEach(n => { n.activation = 0; }));
        const fmapBody = document.getElementById('fmap-body');
        if (fmapBody) fmapBody.innerHTML = '<div class="text-sm text-muted">Draw to see activations</div>';
        bigPred.textContent = '?';
        confText.textContent = 'Draw a character';
        updateConfidenceBars(barsEl, []);
        const perfEl = document.getElementById('perf-body');
        if (perfEl) { perfEl.textContent = '—'; delete perfEl.dataset.base; }
        const statusEl = document.getElementById('ex-status');
        if (statusEl) statusEl.textContent = 'Draw to start';
    }

    // ══════════════════════════════════════
    // RESIZE
    // ══════════════════════════════════════
    function resize() {
        const dpr = window.devicePixelRatio || 1;
        const nw = nnWrap.clientWidth, nh = nnWrap.clientHeight;
        if (nw > 0 && nh > 0) {
            nnCanvas.width = nw * dpr; nnCanvas.height = nh * dpr;
            nnCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
            computePositions(nw, nh);
        }
        const ew = evoWrap.clientWidth, eh = evoWrap.clientHeight;
        if (ew > 0 && eh > 0) {
            evoCanvas.width = ew * dpr; evoCanvas.height = eh * dpr;
            evoCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }
    }
    const ro = new ResizeObserver(resize);
    ro.observe(nnWrap); ro.observe(evoWrap);
    setTimeout(resize, 100);

    // ── System polling ──
    async function pollSys() {
        try {
            const [gpuRes, sysRes] = await Promise.all([fetch('/api/system/gpu'), fetch('/api/system/stats')]);
            const gpu = await gpuRes.json(); const sys = await sysRes.json();
            sysStatsText = `<strong>CPU:</strong> ${sys.cpu_percent}%<br><strong>RAM:</strong> ${sys.ram_used_gb}/${sys.ram_total_gb}GB` +
                (gpu.available ? `<br><strong>GPU:</strong> ${gpu.gpu_util_percent}%<br><strong>VRAM:</strong> ${gpu.memory_used_mb}MB` : `<br><strong>GPU:</strong> N/A`);
            const perfEl = document.getElementById('perf-body');
            if (perfEl) { const b = perfEl.dataset.base || ''; perfEl.innerHTML = b + (b ? '<br><br>' : '') + sysStatsText; }
        } catch (e) { }
    }
    pollSys(); sysTimer = setInterval(pollSys, 5000);

    // ══════════════════════════════════════
    // NN DIAGRAM — POSITION COMPUTATION
    // ══════════════════════════════════════
    function computePositions(w, h) {
        layerPositions = [];
        const padL = 40, padR = 100; // extra right padding for output labels
        const padY = 20;
        const labelSpace = 30;
        const usableW = w - padL - padR;
        const usableH = h - padY * 2 - labelSpace;
        const steps = Math.max(ARCHITECTURE.length - 1, 1);
        const layerSpacing = usableW / steps;

        ARCHITECTURE.forEach((layer, li) => {
            const x = padL + li * layerSpacing;
            const visible = Math.min(layer.neurons, MAX_VISIBLE);
            const nodeGap = Math.min(32, usableH / Math.max(visible, 1));
            const totalH = (visible - 1) * nodeGap;
            const startY = padY + (usableH - totalH) / 2;

            const nodes = [];
            for (let n = 0; n < visible; n++) {
                nodes.push({ x, y: startY + n * nodeGap, activation: nodeActivations[li]?.[n] || 0 });
            }
            layerPositions.push({ x, nodes, layer, visible, hasMore: layer.neurons > MAX_VISIBLE });
        });
    }

    // ══════════════════════════════════════
    // NN DIAGRAM — DRAW (clean, minimal B&W style)
    // ══════════════════════════════════════
    function drawNN() {
        const w = nnWrap.clientWidth, h = nnWrap.clientHeight;
        if (w < 20 || h < 20 || !layerPositions.length) return;
        nnCtx.clearRect(0, 0, w, h);

        // ── Draw connections ──
        for (let li = 0; li < layerPositions.length - 1; li++) {
            const from = layerPositions[li], to = layerPositions[li + 1];
            // Sample subset for perf
            const fIdx = sampleN(from.nodes.length, Math.min(from.nodes.length, 8));
            const tIdx = sampleN(to.nodes.length, Math.min(to.nodes.length, 8));

            for (const fi of fIdx) {
                for (const ti of tIdx) {
                    const fn = from.nodes[fi], tn = to.nodes[ti];
                    const act = (fn.activation + tn.activation) / 2;

                    // Monochrome: inactive = very faint, active = solid black
                    const alpha = act > 0.05 ? (0.06 + act * 0.4) : 0.03;
                    const lw = act > 0.05 ? (0.4 + act * 1.6) : 0.3;

                    nnCtx.strokeStyle = `rgba(0,0,0,${alpha})`;
                    nnCtx.lineWidth = lw;
                    nnCtx.beginPath();
                    nnCtx.moveTo(fn.x, fn.y);
                    nnCtx.lineTo(tn.x, tn.y);
                    nnCtx.stroke();
                }
            }
        }

        // ── Draw nodes ──
        layerPositions.forEach((lp, li) => {
            const isOutput = lp.layer.type === 'output';
            const isInput = lp.layer.type === 'input';

            lp.nodes.forEach((node, ni) => {
                const act = node.activation;
                const radius = isOutput ? 7 : isInput ? 4 : 5;

                // Activation fill: white(0) → black(1)
                const gray = Math.round(230 - act * 210);
                nnCtx.fillStyle = `rgb(${gray},${gray},${gray})`;
                nnCtx.strokeStyle = '#000';
                nnCtx.lineWidth = isOutput ? 1.5 : 1;
                nnCtx.beginPath();
                nnCtx.arc(node.x, node.y, radius, 0, Math.PI * 2);
                nnCtx.fill();
                nnCtx.stroke();

                // Winner ring on the correct output node
                if (isOutput && ni === winnerIdx) {
                    nnCtx.strokeStyle = '#000';
                    nnCtx.lineWidth = 2.5;
                    nnCtx.beginPath();
                    nnCtx.arc(node.x, node.y, radius + 5, 0, Math.PI * 2);
                    nnCtx.stroke();
                }

                // Output labels — each node shows its own prediction
                if (isOutput && outputPredictions[ni]) {
                    const p = outputPredictions[ni];
                    nnCtx.fillStyle = ni === winnerIdx ? '#000' : '#888';
                    nnCtx.font = ni === winnerIdx ? 'bold 11px monospace' : '10px monospace';
                    nnCtx.textAlign = 'left';
                    nnCtx.fillText(`${p.display} ${(p.confidence*100).toFixed(0)}%`, node.x + radius + 6, node.y + 4);
                }
            });

            // Ellipsis for truncated layers
            if (lp.hasMore) {
                const last = lp.nodes[lp.nodes.length - 1];
                nnCtx.fillStyle = '#999';
                nnCtx.font = '12px sans-serif';
                nnCtx.textAlign = 'center';
                nnCtx.fillText('⋮', lp.x, last.y + 20);
            }

            // Layer label
            const bottomY = lp.nodes[lp.nodes.length - 1].y + (lp.hasMore ? 34 : 22);
            nnCtx.fillStyle = '#555';
            nnCtx.font = '9px sans-serif';
            nnCtx.textAlign = 'center';
            nnCtx.fillText(lp.layer.name, lp.x, bottomY);
            nnCtx.fillStyle = '#bbb';
            nnCtx.font = '8px monospace';
            nnCtx.fillText(lp.layer.neurons.toString(), lp.x, bottomY + 11);
        });
    }

    // ══════════════════════════════════════
    // FEATURE MAPS
    // ══════════════════════════════════════
    function updateFeatureMaps(fullData) {
        const body = document.getElementById('fmap-body');
        if (!body || !fullData || !fullData.feature_maps) return;
        body.innerHTML = '';
        const fmap = fullData.feature_maps;
        const archOrder = ARCHITECTURE.map(l => l.key);
        const ordered = [];
        archOrder.forEach(key => { if (fmap[key]) ordered.push(key); });
        Object.keys(fmap).forEach(name => { if (!ordered.includes(name)) ordered.push(name); });

        ordered.forEach(name => {
            const maps = fmap[name];
            if (!maps || !maps.heatmaps || !maps.heatmaps.length) return;
            const group = document.createElement('div');
            group.className = 'ex-fmap-group';
            const arch = ARCHITECTURE.find(l => l.key === name);
            const title = document.createElement('div');
            title.className = 'ex-fmap-title';
            title.textContent = `${arch ? arch.name : name} · ${maps.total_channels}ch · ${maps.spatial_size[0]}×${maps.spatial_size[1]}`;
            group.appendChild(title);
            const grid = document.createElement('div');
            grid.className = 'ex-fmap-grid';
            maps.heatmaps.slice(0, 4).forEach(hm => {
                const thumb = document.createElement('div');
                thumb.className = 'ex-fmap-thumb';
                const img = document.createElement('img');
                img.src = 'data:image/png;base64,' + hm.heatmap;
                img.title = `ch${hm.channel} · ${hm.importance.toFixed(3)}`;
                thumb.appendChild(img);
                grid.appendChild(thumb);
            });
            group.appendChild(grid);
            body.appendChild(group);
        });
        if (!body.children.length) body.innerHTML = '<div class="text-sm text-muted">No feature maps</div>';
    }

    // ══════════════════════════════════════
    // PROBABILITY EVOLUTION
    // ══════════════════════════════════════
    function drawEvo() {
        const w = evoWrap.clientWidth, h = evoWrap.clientHeight;
        if (w < 10 || h < 10) return;
        evoCtx.clearRect(0, 0, w, h);
        if (!evoHistory.length) {
            evoCtx.fillStyle = '#bbb'; evoCtx.font = '10px sans-serif'; evoCtx.textAlign = 'center';
            evoCtx.fillText('No data', w / 2, h / 2); return;
        }
        const pad = { t: 10, r: 36, b: 18, l: 30 };
        const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
        if (pw < 10 || ph < 10) return;

        evoCtx.strokeStyle = '#eee'; evoCtx.lineWidth = 0.5;
        for (let i = 0; i <= 4; i++) {
            const y = pad.t + ph * i / 4;
            evoCtx.beginPath(); evoCtx.moveTo(pad.l, y); evoCtx.lineTo(pad.l + pw, y); evoCtx.stroke();
            evoCtx.fillStyle = '#aaa'; evoCtx.font = '8px sans-serif'; evoCtx.textAlign = 'right';
            evoCtx.fillText(((4 - i) * 25) + '%', pad.l - 3, y + 3);
        }
        evoCtx.fillStyle = '#aaa'; evoCtx.font = '8px sans-serif'; evoCtx.textAlign = 'center';
        evoCtx.fillText('strokes →', pad.l + pw / 2, h - 2);

        const colors = ['#000', '#555', '#999', '#bbb', '#ddd'];
        const widths = [2, 1.5, 1, 0.8, 0.5];
        for (let rank = 4; rank >= 0; rank--) {
            evoCtx.strokeStyle = colors[rank]; evoCtx.lineWidth = widths[rank];
            evoCtx.beginPath();
            let started = false;
            evoHistory.forEach((snap, i) => {
                if (rank >= snap.length) return;
                const x = pad.l + (i / Math.max(evoHistory.length - 1, 1)) * pw;
                const y = pad.t + ph - snap[rank].confidence * ph;
                if (!started) { evoCtx.moveTo(x, y); started = true; } else evoCtx.lineTo(x, y);
            });
            evoCtx.stroke();
            const last = evoHistory[evoHistory.length - 1];
            if (last && rank < last.length) {
                evoCtx.fillStyle = colors[rank]; evoCtx.font = rank === 0 ? 'bold 9px sans-serif' : '8px sans-serif';
                evoCtx.textAlign = 'left';
                evoCtx.fillText(last[rank].display, pad.l + pw + 3, pad.t + ph - last[rank].confidence * ph + 3);
            }
        }
    }

    // ══════════════════════════════════════
    // APPLY LIVE DATA
    // ══════════════════════════════════════
    function applyLiveData(liveData) {
        if (!liveData || !liveData.layers) return;
        liveData.layers.forEach(layer => {
            const archIdx = ARCHITECTURE.findIndex(a => a.key === layer.name);
            if (archIdx < 0) return;
            if (layer.name === 'output') {
                const preds = liveData.predictions || [];
                for (let n = 0; n < Math.min(preds.length, MAX_VISIBLE); n++)
                    nodeActivations[archIdx][n] = preds[n]?.confidence || 0;
            } else {
                const vals = layer.node_values || [];
                for (let n = 0; n < Math.min(vals.length, MAX_VISIBLE); n++)
                    nodeActivations[archIdx][n] = vals[n] || 0;
            }
            if (layerPositions[archIdx])
                layerPositions[archIdx].nodes.forEach((node, ni) => { node.activation = nodeActivations[archIdx][ni] || 0; });
        });
        outputPredictions = (liveData.predictions || []).slice(0, MAX_VISIBLE);
        winnerIdx = outputPredictions.length > 0 ? 0 : -1;
    }

    // ══════════════════════════════════════
    // INFERENCE
    // ══════════════════════════════════════
    let debounceTimer = null;
    canvasObj.onChange(() => { dirtySinceLast = true; if (debounceTimer) clearTimeout(debounceTimer); debounceTimer = setTimeout(runInference, 40); });

    async function runInference() {
        if (_destroyed || pendingRequest) { if (pendingRequest) dirtySinceLast = true; return; }
        const pixels = canvasObj.getPixels();
        if (pixels.reduce((a, b) => a + b, 0) < 0.5) { resetVisuals(); return; }
        pendingRequest = true;
        const t0 = performance.now();
        try {
            const liveRes = await fetch('/api/explain/live', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ pixels }) });
            const live = await liveRes.json();
            if (live.error) { pendingRequest = false; return; }
            applyLiveData(live);
            updateConfidenceBars(barsEl, live.predictions || []);
            const pred = (live.predictions || [])[0];
            if (pred) { bigPred.textContent = pred.display; confText.textContent = `${(pred.confidence * 100).toFixed(1)}% confidence`; }
            evoHistory.push(live.predictions || []);
            if (evoHistory.length > 60) evoHistory.shift();
            const statusEl = document.getElementById('ex-status');
            if (statusEl) statusEl.textContent = pred ? `${pred.display} · ${(pred.confidence * 100).toFixed(1)}% · ${live.inference_time_ms?.toFixed(1) || '?'}ms` : '—';

            // Feature maps every 600ms
            const now = Date.now(); let fmapTime = null;
            if (now - lastFullTime > 600) {
                lastFullTime = now;
                try { const ft0 = performance.now(); const fullRes = await fetch('/api/explain/full', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ pixels }) }); const full = await fullRes.json(); fmapTime = performance.now() - ft0; if (!full.error) updateFeatureMaps(full); } catch (e) { }
            }
            const elapsed = performance.now() - t0;
            const perfEl = document.getElementById('perf-body');
            if (perfEl) { const base = `Inference: <strong>${live.inference_time_ms?.toFixed(1)||'?'}ms</strong>` + (fmapTime !== null ? `<br>Fmaps: <strong>${fmapTime.toFixed(0)}ms</strong>` : '') + `<br>Roundtrip: <strong>${elapsed.toFixed(0)}ms</strong><br>Layers: <strong>${live.layers?.length||'?'}</strong>`; perfEl.dataset.base = base; perfEl.innerHTML = base + (sysStatsText ? '<br><br>' + sysStatsText : ''); }
        } catch (e) { console.error('Explain error:', e); }
        pendingRequest = false;
        if (dirtySinceLast) { dirtySinceLast = false; runInference(); }
    }

    // ── Animation ──
    function animate() { if (_destroyed) return; drawNN(); drawEvo(); _animFrame = requestAnimationFrame(animate); }
    animate();

    container._cleanup = () => { _destroyed = true; if (_animFrame) { cancelAnimationFrame(_animFrame); _animFrame = null; } if (debounceTimer) clearTimeout(debounceTimer); ro.disconnect(); if (sysTimer) clearInterval(sysTimer); };
}

function mkPanel(title) {
    const el = document.createElement('div'); el.className = 'panel';
    el.style.cssText = 'margin-bottom:0;display:flex;flex-direction:column;';
    el.innerHTML = `<div class="panel-header" style="padding:4px 8px;font-size:10px;">${title}</div>`;
    const body = document.createElement('div'); body.className = 'panel-body';
    body.style.cssText = 'flex:1;display:flex;flex-direction:column;';
    el.appendChild(body);
    return { el, body };
}

function sampleN(total, count) {
    if (count >= total) return Array.from({ length: total }, (_, i) => i);
    const step = total / count;
    return Array.from({ length: count }, (_, i) => Math.floor(i * step));
}