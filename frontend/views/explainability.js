/**
 * Explainability view — Network flow + feature maps at top,
 * live NN diagram filling remaining space below.
 * Real-time: predicts as you draw.
 */
import { createCanvas } from '../components/canvas.js';
import { createConfidenceBars, updateConfidenceBars } from '../components/confidenceBars.js';

let animFrame = null;

export async function renderExplainability(container) {
    if (animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }
    container.innerHTML = '';
    container.className = 'view-fit';

    // Check model
    let ready = false;
    try {
        const r = await fetch('/api/inference/status');
        ready = (await r.json()).ready;
    } catch (e) {}
    if (!ready) {
        container.innerHTML = `<div class="view-title">Explainability</div>
            <div class="panel"><div class="panel-body text-muted">No model loaded.</div></div>`;
        return;
    }

    // ── Header ──
    const hdr = document.createElement('div');
    hdr.style.cssText = 'display:flex;justify-content:space-between;align-items:center;flex-shrink:0;padding-bottom:4px;border-bottom:2px solid var(--border);margin-bottom:6px;';
    hdr.innerHTML = `
        <span style="font-size:14px;font-weight:bold;text-transform:uppercase;letter-spacing:2px;">Explainability</span>
        <div style="display:flex;gap:8px;align-items:center;">
            <button class="btn" id="nn-orient" style="font-size:10px;padding:2px 8px;">↔ Horizontal</button>
            <span class="text-sm text-muted" id="ex-status">Draw to start</span>
        </div>
    `;
    container.appendChild(hdr);

    // ── 3-column main ──
    const main = document.createElement('div');
    main.style.cssText = 'display:flex;flex:1;gap:6px;min-height:0;overflow:hidden;';
    container.appendChild(main);

    // ══════════ LEFT: Canvas + Predictions ══════════
    const leftCol = document.createElement('div');
    leftCol.style.cssText = 'width:210px;flex-shrink:0;display:flex;flex-direction:column;gap:4px;overflow:hidden;';
    main.appendChild(leftCol);

    const canvasPanel = mkPanel('Draw');
    canvasPanel.el.style.flexShrink = '0';
    const canvasObj = createCanvas(190);
    canvasPanel.body.style.padding = '4px';
    canvasPanel.body.style.display = 'flex';
    canvasPanel.body.style.justifyContent = 'center';
    canvasPanel.body.appendChild(canvasObj.element);
    leftCol.appendChild(canvasPanel.el);

    const predPanel = mkPanel('Predictions');
    predPanel.el.style.flex = '1';
    predPanel.el.style.overflow = 'hidden';
    predPanel.body.style.cssText = 'padding:4px;overflow-y:auto;';
    const barsEl = createConfidenceBars([]);
    predPanel.body.appendChild(barsEl);
    leftCol.appendChild(predPanel.el);

    // ══════════ CENTER: Network Flow (top) + NN Diagram (bottom) ══════════
    const centerCol = document.createElement('div');
    centerCol.style.cssText = 'flex:1;display:flex;flex-direction:column;gap:4px;min-width:0;overflow:hidden;';
    main.appendChild(centerCol);

    // Network Flow — scrollable list of layers with inline feature maps
    const flowPanel = mkPanel('Network Flow');
    flowPanel.el.style.cssText += 'height:220px;flex-shrink:0;overflow:hidden;';
    flowPanel.body.style.cssText = 'overflow-y:auto;padding:4px;';
    flowPanel.body.id = 'flow-body';
    centerCol.appendChild(flowPanel.el);

    // NN Diagram — fills remaining space
    const nnPanel = mkPanel('Neural Network');
    nnPanel.el.style.cssText += 'flex:1;overflow:hidden;';
    const nnWrap = document.createElement('div');
    nnWrap.style.cssText = 'flex:1;position:relative;min-height:0;';
    const nnCanvas = document.createElement('canvas');
    nnCanvas.style.cssText = 'width:100%;height:100%;display:block;';
    nnWrap.appendChild(nnCanvas);
    nnPanel.el.appendChild(nnWrap);
    centerCol.appendChild(nnPanel.el);

    // ══════════ RIGHT: Prob Evolution + Performance ══════════
    const rightCol = document.createElement('div');
    rightCol.style.cssText = 'width:180px;flex-shrink:0;display:flex;flex-direction:column;gap:4px;overflow:hidden;';
    main.appendChild(rightCol);

    const evoPanel = mkPanel('Prob. Evolution');
    evoPanel.el.style.cssText += 'flex:1;overflow:hidden;';
    const evoWrap = document.createElement('div');
    evoWrap.style.cssText = 'flex:1;position:relative;min-height:0;';
    const evoCanvas = document.createElement('canvas');
    evoCanvas.style.cssText = 'width:100%;height:100%;display:block;';
    evoWrap.appendChild(evoCanvas);
    evoPanel.el.appendChild(evoWrap);
    rightCol.appendChild(evoPanel.el);

    const perfPanel = mkPanel('Performance');
    perfPanel.el.style.flexShrink = '0';
    perfPanel.body.style.cssText = 'padding:6px;font-size:11px;';
    perfPanel.body.id = 'perf-body';
    perfPanel.body.textContent = '—';
    rightCol.appendChild(perfPanel.el);

    // ══════════════════════════════════════════════════
    // STATE
    // ══════════════════════════════════════════════════
    let isVertical = true;
    let layerData = [];       // from /explain/live → layers[]
    let particles = [];
    let evoHistory = [];
    let lastFullTime = 0;
    const nnCtx = nnCanvas.getContext('2d');
    const evoCtx = evoCanvas.getContext('2d');
    const MAX_NODES = 16;

    document.getElementById('nn-orient').addEventListener('click', () => {
        isVertical = !isVertical;
        document.getElementById('nn-orient').textContent = isVertical ? '↔ Horizontal' : '↕ Vertical';
    });

    // ══════════════════════════════════════════════════
    // RESIZE
    // ══════════════════════════════════════════════════
    function resize() {
        const dpr = window.devicePixelRatio || 1;
        [[nnCanvas, nnWrap], [evoCanvas, evoWrap]].forEach(([cv, wrap]) => {
            const w = wrap.clientWidth, h = wrap.clientHeight;
            if (w > 0 && h > 0) {
                cv.width = w * dpr; cv.height = h * dpr;
                cv.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
            }
        });
    }
    const ro = new ResizeObserver(resize);
    ro.observe(nnWrap);
    ro.observe(evoWrap);
    setTimeout(resize, 100);

    // ══════════════════════════════════════════════════
    // NETWORK FLOW (feature maps per layer, top center)
    // ══════════════════════════════════════════════════
    function updateNetworkFlow(fullData) {
        const body = document.getElementById('flow-body');
        if (!body) return;
        body.innerHTML = '';
        if (!fullData || !fullData.feature_maps) return;

        const layerNames = fullData.layers || Object.keys(fullData.feature_maps);

        layerNames.forEach(name => {
            const maps = fullData.feature_maps[name];
            if (!maps) return;

            const row = document.createElement('div');
            row.style.cssText = 'display:flex;align-items:center;gap:8px;margin-bottom:6px;padding:4px;border:1px solid var(--border-light);';

            // Layer info
            const info = document.createElement('div');
            info.style.cssText = 'width:80px;flex-shrink:0;';
            info.innerHTML = `<div style="font-weight:600;font-size:12px;">${name}</div>
                <div style="font-size:9px;color:var(--muted);">${maps.total_channels}ch · ${maps.spatial_size[0]}×${maps.spatial_size[1]}</div>`;
            row.appendChild(info);

            // Activation bar
            const actBar = document.createElement('div');
            actBar.style.cssText = 'width:40px;height:6px;background:#eee;flex-shrink:0;overflow:hidden;';
            const actFill = document.createElement('div');
            const avgImp = maps.heatmaps.reduce((s, h) => s + h.importance, 0) / Math.max(maps.heatmaps.length, 1);
            actFill.style.cssText = `height:100%;width:${Math.min(100, avgImp * 200)}%;background:#000;`;
            actBar.appendChild(actFill);
            row.appendChild(actBar);

            // Feature map thumbnails
            const fmGrid = document.createElement('div');
            fmGrid.style.cssText = 'display:flex;flex-wrap:wrap;gap:2px;flex:1;min-width:0;';
            (maps.heatmaps || []).slice(0, 16).forEach(hm => {
                const img = document.createElement('img');
                img.src = 'data:image/png;base64,' + hm.heatmap;
                img.title = `ch${hm.channel} imp:${hm.importance.toFixed(3)}`;
                img.style.cssText = 'width:28px;height:28px;image-rendering:pixelated;border:1px solid #ddd;';
                fmGrid.appendChild(img);
            });
            row.appendChild(fmGrid);

            body.appendChild(row);
        });

        // Arrow between layers
        const arrows = body.querySelectorAll('.flow-arrow');
        // Already inline — no separate arrows needed, each row is a layer
    }

    // ══════════════════════════════════════════════════
    // NN DIAGRAM — nodes, connections, particles
    // ══════════════════════════════════════════════════
    function computePositions() {
        const w = nnWrap.clientWidth, h = nnWrap.clientHeight;
        if (!layerData.length || w < 20 || h < 20) return [];

        const n = layerData.length;
        const padX = 50, padY = 30;
        const positions = [];

        layerData.forEach((layer, li) => {
            const nodeVals = layer.node_values || [];
            const visible = Math.min(nodeVals.length, MAX_NODES);
            const vals = nodeVals.slice(0, visible);

            let cx, cy;
            if (isVertical) {
                // Top-to-bottom
                cx = w / 2;
                cy = padY + (li / Math.max(n - 1, 1)) * (h - padY * 2);
            } else {
                // Left-to-right
                cx = padX + (li / Math.max(n - 1, 1)) * (w - padX * 2);
                cy = h / 2;
            }

            const maxSpread = isVertical
                ? Math.min(w - 120, visible * 20)
                : Math.min(h - padY * 2, visible * 18);
            const gap = visible > 1 ? maxSpread / (visible - 1) : 0;

            const nodes = [];
            for (let ni = 0; ni < visible; ni++) {
                const offset = (ni - (visible - 1) / 2) * gap;
                nodes.push({
                    x: isVertical ? cx + offset : cx,
                    y: isVertical ? cy : cy + offset,
                    val: vals[ni] || 0,
                    isWinner: layer.type === 'output' && layer.winner !== undefined && ni === layer.winner,
                });
            }
            positions.push({
                cx, cy, nodes, layer, visible,
                hasMore: (layer.channels || 0) > visible,
            });
        });
        return positions;
    }

    function drawNN() {
        const w = nnWrap.clientWidth, h = nnWrap.clientHeight;
        nnCtx.clearRect(0, 0, w, h);

        if (!layerData.length) {
            nnCtx.fillStyle = '#bbb';
            nnCtx.font = '12px sans-serif';
            nnCtx.textAlign = 'center';
            nnCtx.fillText('Draw on the canvas to see the network activate', w / 2, h / 2);
            return;
        }

        const positions = computePositions();

        // ── Connections ──
        for (let li = 0; li < positions.length - 1; li++) {
            const from = positions[li], to = positions[li + 1];
            const fSample = sampleN(from.nodes.length, Math.min(from.nodes.length, 8));
            const tSample = sampleN(to.nodes.length, Math.min(to.nodes.length, 8));
            for (const fi of fSample) {
                for (const ti of tSample) {
                    const act = (from.nodes[fi].val + to.nodes[ti].val) / 2;
                    if (act > 0.5) {
                        const t = (act - 0.5) * 2;
                        nnCtx.strokeStyle = `rgba(0,${Math.round(100 * t)},0,${0.03 + t * 0.1})`;
                    } else {
                        const t = (0.5 - act) * 2;
                        nnCtx.strokeStyle = `rgba(${Math.round(100 * t)},0,0,${0.03 + t * 0.06})`;
                    }
                    nnCtx.lineWidth = 0.5;
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
            nnCtx.fillStyle = p.val > 0.5
                ? `rgba(0,150,0,${alpha * 0.9})`
                : `rgba(180,0,0,${alpha * 0.7})`;
            nnCtx.beginPath();
            nnCtx.arc(p.x, p.y, 2, 0, Math.PI * 2);
            nnCtx.fill();
        });

        // ── Nodes ──
        positions.forEach(lp => {
            lp.nodes.forEach(node => {
                const r = node.isWinner ? 7 : lp.layer.type === 'output' ? 5 : 4;
                const v = node.val;

                // Color: 0=red → 0.5=black → 1=green
                let cr, cg, cb;
                if (v > 0.5) {
                    const t = (v - 0.5) * 2;
                    cr = Math.round(20 * (1 - t));
                    cg = Math.round(20 + 160 * t);
                    cb = Math.round(20 * (1 - t));
                } else {
                    const t = v * 2;
                    cr = Math.round(180 * (1 - t) + 20 * t);
                    cg = Math.round(20 * t);
                    cb = Math.round(20 * t);
                }

                nnCtx.fillStyle = `rgb(${cr},${cg},${cb})`;
                nnCtx.strokeStyle = '#000';
                nnCtx.lineWidth = node.isWinner ? 2.5 : 1;
                nnCtx.beginPath();
                nnCtx.arc(node.x, node.y, r, 0, Math.PI * 2);
                nnCtx.fill();
                nnCtx.stroke();

                // Winner double ring
                if (node.isWinner) {
                    nnCtx.strokeStyle = '#000';
                    nnCtx.lineWidth = 1.5;
                    nnCtx.beginPath();
                    nnCtx.arc(node.x, node.y, r + 4, 0, Math.PI * 2);
                    nnCtx.stroke();
                }
            });

            // Label
            nnCtx.fillStyle = '#888';
            nnCtx.font = '9px sans-serif';
            if (isVertical) {
                nnCtx.textAlign = 'left';
                const rx = lp.nodes[lp.nodes.length - 1].x + 14;
                let label = lp.layer.name;
                if (lp.hasMore) label += ` (${lp.layer.channels}ch)`;
                nnCtx.fillText(label, rx, lp.cy + 3);
            } else {
                nnCtx.textAlign = 'center';
                const by = lp.nodes[lp.nodes.length - 1].y + 16;
                nnCtx.fillText(lp.layer.name, lp.cx, by);
                if (lp.hasMore) {
                    nnCtx.font = '8px sans-serif';
                    nnCtx.fillText(`${lp.layer.channels}ch`, lp.cx, by + 10);
                }
            }
        });
    }

    function spawnParticles() {
        const positions = computePositions();
        if (positions.length < 2) return;
        for (let li = 0; li < positions.length - 1; li++) {
            const from = positions[li], to = positions[li + 1];
            for (let p = 0; p < 3; p++) {
                const fi = Math.floor(Math.random() * from.nodes.length);
                const ti = Math.floor(Math.random() * to.nodes.length);
                particles.push({
                    fromX: from.nodes[fi].x, fromY: from.nodes[fi].y,
                    toX: to.nodes[ti].x, toY: to.nodes[ti].y,
                    x: from.nodes[fi].x, y: from.nodes[fi].y,
                    progress: 0,
                    speed: 0.012 + Math.random() * 0.02,
                    val: (from.nodes[fi].val + to.nodes[ti].val) / 2,
                });
            }
        }
    }

    function updateParticles() {
        particles = particles.filter(p => {
            p.progress += p.speed;
            p.x = p.fromX + (p.toX - p.fromX) * p.progress;
            p.y = p.fromY + (p.toY - p.fromY) * p.progress;
            return p.progress < 1;
        });
    }

    // ══════════════════════════════════════════════════
    // PROBABILITY EVOLUTION CHART
    // ══════════════════════════════════════════════════
    function drawEvo() {
        const w = evoWrap.clientWidth, h = evoWrap.clientHeight;
        evoCtx.clearRect(0, 0, w, h);
        if (!evoHistory.length) {
            evoCtx.fillStyle = '#bbb';
            evoCtx.font = '10px sans-serif';
            evoCtx.textAlign = 'center';
            evoCtx.fillText('No data yet', w / 2, h / 2);
            return;
        }

        const pad = { t: 10, r: 30, b: 10, l: 26 };
        const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;

        // Grid
        evoCtx.strokeStyle = '#ddd';
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

        const colors = ['#000', '#555', '#999', '#bbb', '#ddd'];
        for (let rank = 4; rank >= 0; rank--) {
            evoCtx.strokeStyle = colors[rank];
            evoCtx.lineWidth = rank === 0 ? 2 : 1;
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

    // ══════════════════════════════════════════════════
    // REAL-TIME INFERENCE
    // ══════════════════════════════════════════════════
    let debounceTimer = null;

    canvasObj.onChange(() => {
        if (debounceTimer) clearTimeout(debounceTimer);
        debounceTimer = setTimeout(runInference, 30);
    });

    async function runInference() {
        const pixels = canvasObj.getPixels();
        const sum = pixels.reduce((a, b) => a + b, 0);
        if (sum < 0.5) return;
        if (pendingRequest) return; // don't stack requests
        pendingRequest = true;

        const t0 = performance.now();
        try {
            // Live endpoint — lightweight, for NN diagram
            const liveRes = await fetch('/api/explain/live', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pixels }),
            });
            const live = await liveRes.json();
            if (live.error) { pendingRequest = false; return; }

            // Update NN diagram data
            layerData = live.layers || [];

            // Update predictions
            updateConfidenceBars(barsEl, live.predictions || []);

            // Prob evolution
            evoHistory.push(live.predictions || []);
            if (evoHistory.length > 80) evoHistory.shift();

            // Particles
            spawnParticles();

            // Status
            const elapsed = performance.now() - t0;
            const pred = live.predictions?.[0];
            document.getElementById('ex-status').textContent = pred
                ? `${pred.display} · ${(pred.confidence * 100).toFixed(1)}%`
                : '—';
            document.getElementById('perf-body').innerHTML =
                `Inference: <strong>${live.inference_time_ms?.toFixed(1) || '?'}ms</strong><br>` +
                `Round-trip: <strong>${elapsed.toFixed(0)}ms</strong><br>` +
                `Layers: <strong>${layerData.length}</strong>`;

            // Full explain (with feature maps) every 500ms
            const now = Date.now();
            if (now - lastFullTime > 500) {
                lastFullTime = now;
                try {
                    const fullRes = await fetch('/api/explain/full', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ pixels }),
                    });
                    const full = await fullRes.json();
                    if (!full.error) updateNetworkFlow(full);
                } catch (e) {}
            }
        } catch (e) {
            console.error('Explain error:', e);
        }
        pendingRequest = false;
    }

    // ══════════════════════════════════════════════════
    // ANIMATION LOOP
    // ══════════════════════════════════════════════════
    function animate() {
        updateParticles();
        drawNN();
        drawEvo();
        animFrame = requestAnimationFrame(animate);
    }
    animate();
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