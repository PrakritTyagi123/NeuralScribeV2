/**
 * Explainability view — Canvas NN diagram + feature maps + probability evolution.
 * Adapted from user mockup. B&W wireframe theme.
 * Real-time: predicts as you draw.
 */
import { createCanvas } from '../components/canvas.js';
import { createConfidenceBars, updateConfidenceBars } from '../components/confidenceBars.js';

// ═══════════════════════════════════════════════════
// ARCHITECTURE DEFINITION
// ═══════════════════════════════════════════════════
const ARCHITECTURE = [
    { name: 'Input', neurons: 16, type: 'input', key: 'input' },
    { name: 'Stem', neurons: 32, type: 'conv', key: 'stem' },
    { name: 'Block 0', neurons: 64, type: 'conv', key: 'block_0' },
    { name: 'Block 1', neurons: 128, type: 'conv', key: 'block_1' },
    { name: 'Block 2', neurons: 256, type: 'conv', key: 'block_2' },
    { name: 'Block 3', neurons: 320, type: 'conv', key: 'block_3' },
    { name: 'Pool', neurons: 320, type: 'fc', key: 'pooled' },
    { name: 'Output', neurons: 120, type: 'output', key: 'output' },
];
const MAX_VISIBLE = 16;

let _animFrame = null;
let _destroyed = false;

export async function renderExplainability(container) {
    // Cleanup previous
    if (_animFrame) { cancelAnimationFrame(_animFrame); _animFrame = null; }
    _destroyed = false;
    container.innerHTML = '';
    container.className = 'view-fit';

    // Check model
    let ready = false;
    try {
        const r = await fetch('/api/inference/status');
        ready = (await r.json()).ready;
    } catch (e) { }
    if (!ready) {
        container.innerHTML = `<div class="view-title">Explainability</div>
            <div class="panel"><div class="panel-body text-muted">No model loaded. Go to Model Manager to load one.</div></div>`;
        return;
    }

    // ── Header ──
    const hdr = document.createElement('div');
    hdr.className = 'ex-header';
    hdr.innerHTML = `
        <span class="ex-header-title">Live Neural Network</span>
        <span class="text-sm text-muted" id="ex-status">Draw to start</span>
    `;
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

    // ═══════ LEFT (top): Canvas + Predictions ═══════
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

    // ═══════ CENTER (top): Canvas NN Diagram ═══════
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

    // ═══════ RIGHT (top): Prob Evolution ═══════
    const rightTop = document.createElement('div');
    rightTop.className = 'ex-right-top';
    topRow.appendChild(rightTop);

    const evoPanel = mkPanel('Probability Evolution');
    evoPanel.el.style.flex = '1';
    evoPanel.el.style.flexShrink = '0';
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
    fmPanel.body.style.overflow = 'hidden';
    fmPanel.body.style.padding = '6px';
    fmPanel.body.style.display = 'grid';
    fmPanel.body.style.gridTemplateColumns = 'repeat(3, 1fr)';
    fmPanel.body.style.gridAutoRows = '1fr';
    fmPanel.body.style.gap = '6px';
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
    let particles = [];
    let nodeActivations = ARCHITECTURE.map(l => new Array(Math.min(l.neurons, MAX_VISIBLE)).fill(0));
    let outputNodeLabels = new Array(Math.min(ARCHITECTURE[ARCHITECTURE.length - 1].neurons, MAX_VISIBLE)).fill('');
    let winnerIdx = -1;
    let evoHistory = [];
    let lastFullTime = 0;
    let pendingRequest = false;
    let dirtySinceLast = false;
    let sysStatsText = '';
    let pulseFrame = 0;

    const nnCtx = nnCanvas.getContext('2d');
    const evoCtx = evoCanvas.getContext('2d');

    let sysTimer = null;

    // ══════════════════════════════════════
    // RESET VISUALS (after Clear / empty canvas)
    // ══════════════════════════════════════
    function resetVisuals() {
        nodeActivations = ARCHITECTURE.map(l => new Array(Math.min(l.neurons, MAX_VISIBLE)).fill(0));
        outputNodeLabels = outputNodeLabels.map(() => '');
        winnerIdx = -1;
        evoHistory = [];
        particles = [];

        // Zero out cached positions
        layerPositions.forEach((lp, li) => {
            lp.nodes.forEach((n, ni) => {
                n.activation = 0;
            });
        });

        // Reset feature maps
        const fmapBody = document.getElementById('fmap-body');
        if (fmapBody) {
            fmapBody.innerHTML = '<div class="text-sm text-muted">Draw to see activations</div>';
        }

        // Reset prediction text
        bigPred.textContent = '?';
        confText.textContent = 'Draw a character';
        updateConfidenceBars(barsEl, []);

        // Reset perf + status
        const perfEl = document.getElementById('perf-body');
        if (perfEl) {
            perfEl.textContent = '—';
            delete perfEl.dataset.base;
        }
        const statusEl = document.getElementById('ex-status');
        if (statusEl) statusEl.textContent = 'Draw to start';
    }

    // ══════════════════════════════════════
    // RESIZE
    // ══════════════════════════════════════
    function resize() {
        const dpr = window.devicePixelRatio || 1;

        // NN canvas
        const nw = nnWrap.clientWidth, nh = nnWrap.clientHeight;
        if (nw > 0 && nh > 0) {
            nnCanvas.width = nw * dpr;
            nnCanvas.height = nh * dpr;
            nnCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
            computePositions(nw, nh);
        }

        // Evo canvas
        const ew = evoWrap.clientWidth, eh = evoWrap.clientHeight;
        if (ew > 0 && eh > 0) {
            evoCanvas.width = ew * dpr;
            evoCanvas.height = eh * dpr;
            evoCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }
    }

    const ro = new ResizeObserver(resize);
    ro.observe(nnWrap);
    ro.observe(evoWrap);
    setTimeout(resize, 100);

    // ─═════════════════════════════════════
    // SYSTEM PERFORMANCE (CPU / RAM / GPU)
    // ─═════════════════════════════════════
    async function pollSys() {
        try {
            const [gpuRes, sysRes] = await Promise.all([
                fetch('/api/system/gpu'),
                fetch('/api/system/stats'),
            ]);
            const gpu = await gpuRes.json();
            const sys = await sysRes.json();
            sysStatsText =
                `<strong>CPU:</strong> ${sys.cpu_percent}%` +
                `<br><strong>RAM:</strong> ${sys.ram_used_gb}/${sys.ram_total_gb}GB (${sys.ram_percent}%)` +
                `<br><strong>Disk:</strong> ${sys.disk_used_gb}/${sys.disk_total_gb}GB` +
                (gpu.available
                    ? `<br><strong>GPU:</strong> ${gpu.gpu_util_percent}%` +
                      `<br><strong>VRAM:</strong> ${gpu.memory_used_mb}MB`
                    : `<br><strong>GPU:</strong> N/A`);

            const perfEl = document.getElementById('perf-body');
            if (perfEl) {
                const base = perfEl.dataset.base || '';
                perfEl.innerHTML = base + (base ? '<br><br>' : '') + sysStatsText;
            }
        } catch (e) {
            // ignore
        }
    }

    pollSys();
    sysTimer = setInterval(pollSys, 5000);

    // ══════════════════════════════════════
    // NN DIAGRAM — POSITION COMPUTATION
    // ══════════════════════════════════════
    function computePositions(w, h) {
        layerPositions = [];
        const padX = 40, padY = 10;
        const labelSpace = 32; // reserved space under bottom row for layer names
        const usableW = w - padX * 2;
        const usableH = h - padY * 2 - labelSpace;
        const steps = Math.max(ARCHITECTURE.length - 1, 1);
        const layerSpacing = usableW / steps;
        // Center whole stack horizontally
        const totalSpan = layerSpacing * steps;
        const startX = (w - totalSpan) / 2;

        ARCHITECTURE.forEach((layer, li) => {
            const x = startX + li * layerSpacing;
            const visible = Math.min(layer.neurons, MAX_VISIBLE);
            const totalH = visible > 1 ? Math.min(usableH, visible * 22) : 0;
            // Slight bias toward the top so labels are always visible
            const startY = padY + (usableH - totalH) * 0.3;
            const gap = visible > 1 ? totalH / (visible - 1) : 0;

            const nodes = [];
            for (let n = 0; n < visible; n++) {
                nodes.push({
                    x: x,
                    y: startY + n * gap,
                    activation: nodeActivations[li][n] || 0,
                });
            }

            layerPositions.push({
                x, nodes, layer, visible,
                hasMore: layer.neurons > MAX_VISIBLE,
            });
        });
    }

    // ══════════════════════════════════════
    // NN DIAGRAM — DRAW
    // ══════════════════════════════════════
    function drawNN() {
        const w = nnWrap.clientWidth, h = nnWrap.clientHeight;
        if (w < 20 || h < 20) return;
        nnCtx.clearRect(0, 0, w, h);

        if (!layerPositions.length) return;

        // ── Connections ──
        for (let li = 0; li < layerPositions.length - 1; li++) {
            const from = layerPositions[li], to = layerPositions[li + 1];
            const fSample = sampleN(from.nodes.length, Math.min(from.nodes.length, 8));
            const tSample = sampleN(to.nodes.length, Math.min(to.nodes.length, 8));

            // Base grid — always-visible light black lines
            nnCtx.strokeStyle = 'rgba(0,0,0,0.18)';
            nnCtx.lineWidth = 0.35;
            for (const fi of fSample) {
                for (const ti of tSample) {
                    nnCtx.beginPath();
                    nnCtx.moveTo(from.nodes[fi].x, from.nodes[fi].y);
                    nnCtx.lineTo(to.nodes[ti].x, to.nodes[ti].y);
                    nnCtx.stroke();
                }
            }

            // Activation overlay — pure red (low) or pure green (high)
            for (const fi of fSample) {
                for (const ti of tSample) {
                    const act = (from.nodes[fi].activation + to.nodes[ti].activation) / 2;
                    if (act <= 0.1) continue;
                    const t = Math.max(0, Math.min(1, act));
                    // Below midpoint → pure red, above → pure green
                    const isHigh = t >= 0.5;
                    const r = isHigh ? 0 : 255;
                    const g = isHigh ? 255 : 0;
                    const alpha = 0.15 + 0.35 * t;
                    nnCtx.strokeStyle = `rgba(${r},${g},0,${alpha})`;
                    nnCtx.lineWidth = 0.6 + t * 0.8;
                    nnCtx.beginPath();
                    nnCtx.moveTo(from.nodes[fi].x, from.nodes[fi].y);
                    nnCtx.lineTo(to.nodes[ti].x, to.nodes[ti].y);
                    nnCtx.stroke();
                }
            }
        }

        // ── Particles ──
        particles.forEach(p => {
            const alpha = Math.max(0, 1 - Math.abs(p.progress - 0.5) * 2);
            nnCtx.fillStyle = `rgba(0,0,0,${alpha * 0.7})`;
            nnCtx.beginPath();
            nnCtx.arc(p.x, p.y, 2, 0, Math.PI * 2);
            nnCtx.fill();
        });

        // ── Nodes ──
        layerPositions.forEach((lp, li) => {
                lp.nodes.forEach((node, ni) => {
                    const act = node.activation;
                    const isOutput = lp.layer.type === 'output';
                    const isInput = lp.layer.type === 'input';
                    const r = isOutput ? 6 : isInput ? 3.5 : 4.5;

                    // Base dot — grey fill with black outline
                    const baseGray = 210;
                    const fillGray = act > 0.25 ? 40 : baseGray;
                    nnCtx.fillStyle = `rgb(${fillGray},${fillGray},${fillGray})`;
                    nnCtx.strokeStyle = '#000';
                    nnCtx.lineWidth = 1;
                nnCtx.beginPath();
                nnCtx.arc(node.x, node.y, r, 0, Math.PI * 2);
                nnCtx.fill();
                nnCtx.stroke();

                // Small index labels for input nodes
                if (isInput) {
                    nnCtx.fillStyle = '#666';
                    nnCtx.font = '8px monospace';
                    nnCtx.textAlign = 'right';
                    nnCtx.fillText(String(ni + 1), node.x - r - 2, node.y + 2);
                }

                // Dynamic labels for visible output nodes
                if (isOutput && outputNodeLabels[ni]) {
                    nnCtx.fillStyle = '#000';
                    nnCtx.font = '9px sans-serif';
                    nnCtx.textAlign = 'left';
                    nnCtx.fillText(outputNodeLabels[ni], node.x + r + 4, node.y + 3);
                }

                // Winner highlight
                if (isOutput && ni === winnerIdx) {
                    nnCtx.strokeStyle = '#000';
                    nnCtx.lineWidth = 2.5;
                    nnCtx.beginPath();
                    nnCtx.arc(node.x, node.y, r + 4, 0, Math.PI * 2);
                    nnCtx.stroke();
                }
            });

            // Layer label
            nnCtx.fillStyle = '#888';
            nnCtx.font = '10px sans-serif';
            nnCtx.textAlign = 'center';
            const labelY = lp.nodes[lp.nodes.length - 1].y + 20;
            const pretty = lp.layer.name + (lp.layer.neurons ? '' : '');
            nnCtx.fillText(pretty, lp.x, labelY);

            // Neuron count
            if (lp.layer.neurons) {
                nnCtx.fillStyle = '#bbb';
                nnCtx.font = '8px monospace';
                nnCtx.fillText(`(${lp.layer.neurons})`, lp.x, labelY + 11);
            }
        });
    }

    // ── Particles ──
    function spawnParticles() {
        // Pulses disabled — no new particles are spawned.
        return;
    }

    function updateParticles() {
        particles = particles.filter(p => {
            p.progress += p.speed;
            p.x = p.fromX + (p.toX - p.fromX) * p.progress;
            p.y = p.fromY + (p.toY - p.fromY) * p.progress;
            return p.progress < 1;
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

        // Order layers to match the high-level architecture so we
        // consistently show: Input → Stem → Block 0… → Pool → Output.
        const archOrder = ARCHITECTURE.map(l => l.key);
        const ordered = [];
        archOrder.forEach(key => {
            if (Object.prototype.hasOwnProperty.call(fmap, key)) ordered.push(key);
        });
        Object.keys(fmap).forEach(name => {
            if (!ordered.includes(name)) ordered.push(name);
        });

        ordered.forEach(name => {
            const maps = fmap[name];
            if (!maps || !maps.heatmaps || !maps.heatmaps.length) return;

            const group = document.createElement('div');
            group.className = 'ex-fmap-group';

            const arch = ARCHITECTURE.find(l => l.key === name);
            const label = arch ? arch.name : name;

            const title = document.createElement('div');
            title.className = 'ex-fmap-title';
            title.textContent = `${label} · ${maps.total_channels}ch · ${maps.spatial_size[0]}×${maps.spatial_size[1]}`;
            group.appendChild(title);

            const grid = document.createElement('div');
            grid.className = 'ex-fmap-grid';
            maps.heatmaps.slice(0, 6).forEach(hm => {
                const thumb = document.createElement('div');
                thumb.className = 'ex-fmap-thumb';

                const img = document.createElement('img');
                img.src = 'data:image/png;base64,' + hm.heatmap;
                img.alt = `ch${hm.channel}`;
                img.title = `ch${hm.channel} · imp ${hm.importance.toFixed(3)}`;

                thumb.appendChild(img);
                grid.appendChild(thumb);
            });
            group.appendChild(grid);
            body.appendChild(group);
        });

        if (!body.children.length) {
            body.innerHTML = '<div class="text-sm text-muted">No feature maps</div>';
        }
    }

    // ══════════════════════════════════════
    // PROBABILITY EVOLUTION
    // ══════════════════════════════════════
    function drawEvo() {
        const w = evoWrap.clientWidth, h = evoWrap.clientHeight;
        if (w < 10 || h < 10) return;
        evoCtx.clearRect(0, 0, w, h);

        if (!evoHistory.length) {
            evoCtx.fillStyle = '#bbb';
            evoCtx.font = '10px sans-serif';
            evoCtx.textAlign = 'center';
            evoCtx.fillText('No data', w / 2, h / 2);
            return;
        }

        const pad = { t: 10, r: 32, b: 10, l: 26 };
        const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
        if (pw < 10 || ph < 10) return;

        // Grid
        evoCtx.strokeStyle = '#eee';
        evoCtx.lineWidth = 0.5;
        for (let i = 0; i <= 4; i++) {
            const y = pad.t + ph * i / 4;
            evoCtx.beginPath();
            evoCtx.moveTo(pad.l, y);
            evoCtx.lineTo(pad.l + pw, y);
            evoCtx.stroke();
            evoCtx.fillStyle = '#aaa';
            evoCtx.font = '8px sans-serif';
            evoCtx.textAlign = 'right';
            evoCtx.fillText(((4 - i) * 25) + '%', pad.l - 3, y + 3);
        }

        const colors = ['#000', '#444', '#888', '#bbb', '#ddd'];
        const widths = [2.5, 1.5, 1, 1, 0.5];

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
                evoCtx.font = rank === 0 ? 'bold 9px sans-serif' : '8px sans-serif';
                evoCtx.textAlign = 'left';
                const y = pad.t + ph - last[rank].confidence * ph;
                evoCtx.fillText(last[rank].display, pad.l + pw + 3, y + 3);
            }
        }
    }

    // ══════════════════════════════════════
    // APPLY LIVE DATA TO NN DIAGRAM
    // ══════════════════════════════════════
    function applyLiveData(liveData) {
        if (!liveData || !liveData.layers) return;

        liveData.layers.forEach(layer => {
            const archIdx = ARCHITECTURE.findIndex(a => a.key === layer.name);
            if (archIdx < 0) return;

            // For the output layer we want the visible nodes to reflect the
            // top‑k predictions directly so that the rightmost connections
            // stay bright (green) for confident classes regardless of their
            // raw class_id index.
            if (layer.name === 'output') {
                const preds = liveData.predictions || [];
                const visible = Math.min(preds.length, MAX_VISIBLE);
                for (let n = 0; n < visible; n++) {
                    nodeActivations[archIdx][n] = preds[n]?.confidence || 0;
                }
            } else {
                const vals = layer.node_values || [];
                const visible = Math.min(vals.length, MAX_VISIBLE);
                for (let n = 0; n < visible; n++) {
                    nodeActivations[archIdx][n] = vals[n] || 0;
                }
            }

            if (layerPositions[archIdx]) {
                layerPositions[archIdx].nodes.forEach((node, ni) => {
                    node.activation = nodeActivations[archIdx][ni] || 0;
                });
            }
        });

        // Winner on output — highlight the top-ranked prediction node.
        const pred = liveData.predictions?.[0];
        if (pred) {
            // The first visible output node corresponds to the top prediction.
            winnerIdx = 0;
        } else {
            winnerIdx = -1;
        }
    }

    // ══════════════════════════════════════
    // REAL-TIME INFERENCE
    // ══════════════════════════════════════
    let debounceTimer = null;

    canvasObj.onChange(() => {
        dirtySinceLast = true;
        if (debounceTimer) clearTimeout(debounceTimer);
        debounceTimer = setTimeout(runInference, 40);
    });

    async function runInference() {
        if (_destroyed) return;
        // If a request is already in flight, just mark that we need
        // another pass when it finishes to stay real-time.
        if (pendingRequest) {
            dirtySinceLast = true;
            return;
        }

        const pixels = canvasObj.getPixels();
        const sum = pixels.reduce((a, b) => a + b, 0);
        if (sum < 0.5) {
            resetVisuals();
            return;
        }
        pendingRequest = true;

        const t0 = performance.now();
        try {
            // Live endpoint
            const liveRes = await fetch('/api/explain/live', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pixels }),
            });
            const live = await liveRes.json();
            if (live.error) { pendingRequest = false; return; }

            // Update NN diagram activations
            applyLiveData(live);
            spawnParticles();

            // Update predictions
            updateConfidenceBars(barsEl, live.predictions || []);

            const predList = live.predictions || [];
            const pred = predList[0];
            if (pred) {
                bigPred.textContent = pred.display;
                confText.textContent = `${(pred.confidence * 100).toFixed(1)}% confidence`;
            }

            // Update dynamic labels for visible output nodes using top predictions
            // The first visible node shows the top prediction, the next node the
            // second-best, and so on, avoiding mismatches between class IDs and
            // diagram node indices.
            outputNodeLabels = outputNodeLabels.map(() => '');
            predList.slice(0, MAX_VISIBLE).forEach((p, i) => {
                if (i < outputNodeLabels.length) {
                    outputNodeLabels[i] = p.display;
                }
            });

            // Prob evolution
            evoHistory.push(live.predictions || []);
            if (evoHistory.length > 60) evoHistory.shift();

            // Status
            const elapsed = performance.now() - t0;
            const statusEl = document.getElementById('ex-status');
            if (statusEl) {
                statusEl.textContent = pred
                    ? `${pred.display} · ${(pred.confidence * 100).toFixed(1)}% · ${live.inference_time_ms?.toFixed(1) || '?'}ms`
                    : '—';
            }

            // Full explain (feature maps) every 600ms
            const now = Date.now();
            let fmapTime = null;
            if (now - lastFullTime > 600) {
                lastFullTime = now;
                try {
                    const ft0 = performance.now();
                    const fullRes = await fetch('/api/explain/full', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ pixels }),
                    });
                    const full = await fullRes.json();
                    fmapTime = performance.now() - ft0;
                    if (!full.error) updateFeatureMaps(full);
                } catch (e) { }
            }

            // Performance
            const perfEl = document.getElementById('perf-body');
            if (perfEl) {
                const base =
                    `Inference: <strong>${live.inference_time_ms?.toFixed(1) || '?'}ms</strong>` +
                    (fmapTime !== null ? `<br>Feature maps: <strong>${fmapTime.toFixed(0)}ms</strong>` : '') +
                    `<br>Round-trip: <strong>${elapsed.toFixed(0)}ms</strong>` +
                    `<br>Layers: <strong>${live.layers?.length || '?'}</strong>`;
                perfEl.dataset.base = base;
                perfEl.innerHTML = base + (sysStatsText ? `<br><br>${sysStatsText}` : '');
            }

        } catch (e) {
            console.error('Explain error:', e);
        }
        pendingRequest = false;
        // If the user kept drawing while we were waiting on the
        // network, immediately schedule a new pass so the UI
        // catches up without waiting.
        if (dirtySinceLast) {
            dirtySinceLast = false;
            runInference();
        }
    }

    // ══════════════════════════════════════
    // ANIMATION LOOP
    // ══════════════════════════════════════
    function animate() {
        if (_destroyed) return;
        // Spawn new particles while we have an active prediction,
        // but only every 3rd frame to keep density low.
        if (winnerIdx !== -1) {
            pulseFrame = (pulseFrame + 1) % 3;
            if (pulseFrame === 0) {
                spawnParticles();
            }
        }
        updateParticles();
        drawNN();
        drawEvo();
        _animFrame = requestAnimationFrame(animate);
    }
    animate();

    // Cleanup hook
    const origCleanup = container._cleanup;
    container._cleanup = () => {
        _destroyed = true;
        if (_animFrame) { cancelAnimationFrame(_animFrame); _animFrame = null; }
        if (debounceTimer) clearTimeout(debounceTimer);
        ro.disconnect();
        if (sysTimer) clearInterval(sysTimer);
        if (origCleanup) origCleanup();
    };
}

// ── Helpers ──

function mkPanel(title) {
    const el = document.createElement('div');
    el.className = 'panel';
    el.style.cssText = 'margin-bottom:0;display:flex;flex-direction:column;';
    el.innerHTML = `<div class="panel-header" style="padding:4px 8px;font-size:10px;">${title}</div>`;
    const body = document.createElement('div');
    body.className = 'panel-body';
    body.style.cssText = 'flex:1;display:flex;flex-direction:column;';
    el.appendChild(body);
    return { el, body };
}

function sampleN(total, count) {
    if (count >= total) return Array.from({ length: total }, (_, i) => i);
    const step = total / count;
    return Array.from({ length: count }, (_, i) => Math.floor(i * step));
}