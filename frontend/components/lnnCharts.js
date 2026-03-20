/**
 * NeuralScribe v2 — LNN Chart Renderers
 * Probability evolution, stroke timeline, and embedding space visualizations.
 */

// ═══════════════════════════════════════
// PROBABILITY EVOLUTION
// ═══════════════════════════════════════
export function drawEvo(ctx, wrap, evoHistory) {
    const w = wrap.clientWidth;
    const h = wrap.clientHeight;
    if (w < 10 || h < 10) return;
    ctx.clearRect(0, 0, w, h);

    if (evoHistory.length === 0) {
        ctx.fillStyle = '#a1a1aa';
        ctx.font = '9px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No data yet', w / 2, h / 2);
        return;
    }

    const pad = { t: 8, r: 30, b: 12, l: 24 };
    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;

    // Grid lines
    ctx.strokeStyle = '#e4e4e7';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const y = pad.t + ph * i / 4;
        ctx.beginPath();
        ctx.moveTo(pad.l, y);
        ctx.lineTo(pad.l + pw, y);
        ctx.stroke();
        ctx.fillStyle = '#a1a1aa';
        ctx.font = '7px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(((4 - i) * 25) + '%', pad.l - 3, y + 3);
    }

    const colors = ['#2563eb', '#64748b', '#94a3b8', '#cbd5e1', '#e2e8f0'];
    const widths = [2.5, 1.5, 1, 0.7, 0.5];

    for (let rank = 4; rank >= 0; rank--) {
        ctx.strokeStyle = colors[rank];
        ctx.lineWidth = widths[rank];
        ctx.beginPath();
        let started = false;
        evoHistory.forEach((snap, i) => {
            if (rank >= snap.length) return;
            const x = pad.l + (i / Math.max(evoHistory.length - 1, 1)) * pw;
            const y = pad.t + ph - snap[rank].confidence * ph;
            if (!started) { ctx.moveTo(x, y); started = true; }
            else ctx.lineTo(x, y);
        });
        ctx.stroke();

        // End label
        const last = evoHistory[evoHistory.length - 1];
        if (last && rank < last.length) {
            ctx.fillStyle = colors[rank];
            ctx.font = rank === 0 ? 'bold 8px monospace' : '7px monospace';
            ctx.textAlign = 'left';
            const endY = pad.t + ph - last[rank].confidence * ph;
            ctx.fillText(last[rank].display, pad.l + pw + 3, endY + 3);
        }
    }
}

// ═══════════════════════════════════════
// STROKE TIMELINE
// ═══════════════════════════════════════
export function drawStrokeTimeline(ctx, wrap, strokeHistory) {
    const w = wrap.clientWidth;
    const h = wrap.clientHeight;
    if (w < 10 || h < 10) return;
    ctx.clearRect(0, 0, w, h);

    if (strokeHistory.length === 0) {
        ctx.fillStyle = '#a1a1aa';
        ctx.font = '9px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No data yet', w / 2, h / 2);
        return;
    }

    const pad = { t: 8, r: 42, b: 14, l: 32 };
    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;

    // Grid
    ctx.strokeStyle = '#e4e4e7';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const y = pad.t + ph * i / 4;
        ctx.beginPath();
        ctx.moveTo(pad.l, y);
        ctx.lineTo(pad.l + pw, y);
        ctx.stroke();
        ctx.fillStyle = '#a1a1aa';
        ctx.font = '7px monospace';
        ctx.textAlign = 'right';
        ctx.fillText(((4 - i) * 25) + '%', pad.l - 4, y + 3);
    }

    // Confidence line
    ctx.strokeStyle = '#2563eb';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    strokeHistory.forEach((pt, i) => {
        const x = pad.l + (i / Math.max(strokeHistory.length - 1, 1)) * pw;
        const y = pad.t + ph - pt.conf * ph;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Stroke event markers
    let prevStroke = 0;
    strokeHistory.forEach((pt, i) => {
        if (pt.stroke > prevStroke) {
            prevStroke = pt.stroke;
            const x = pad.l + (i / Math.max(strokeHistory.length - 1, 1)) * pw;
            ctx.strokeStyle = 'rgba(37,99,235,0.2)';
            ctx.lineWidth = 1;
            ctx.setLineDash([2, 2]);
            ctx.beginPath();
            ctx.moveTo(x, pad.t);
            ctx.lineTo(x, pad.t + ph);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    });

    // End label
    const last = strokeHistory[strokeHistory.length - 1];
    if (last && last.label) {
        const x = pad.l + pw;
        const y = pad.t + ph - last.conf * ph;
        ctx.fillStyle = '#1e40af';
        ctx.font = 'bold 9px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(last.label + ' ' + (last.conf * 100).toFixed(0) + '%', x + 4, y + 4);
    }
}

// ═══════════════════════════════════════
// EMBEDDING SPACE
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

export function drawEmbedding(ctx, wrap, outPreds) {
    const w = wrap.clientWidth;
    const h = wrap.clientHeight;
    if (w < 10 || h < 10) return;
    ctx.clearRect(0, 0, w, h);

    if (outPreds.length === 0) {
        ctx.fillStyle = '#a1a1aa';
        ctx.font = '9px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No data', w / 2, h / 2);
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
            ctx.fillStyle = cluster.isWinner ? 'rgba(37,99,235,0.3)' : 'rgba(100,116,139,0.1)';
            ctx.beginPath();
            ctx.arc(dot.x, dot.y, 2, 0, Math.PI * 2);
            ctx.fill();
        });
        ctx.fillStyle = cluster.isWinner ? '#1e40af' : '#94a3b8';
        ctx.font = cluster.isWinner ? 'bold 10px monospace' : '9px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(cluster.pred.display, cluster.px, cluster.py - 10);
    });

    // Input marker at center
    const cx = w / 2, cy = h / 2;
    ctx.strokeStyle = '#2563eb';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx - 5, cy - 5);
    ctx.lineTo(cx + 5, cy + 5);
    ctx.moveTo(cx + 5, cy - 5);
    ctx.lineTo(cx - 5, cy + 5);
    ctx.stroke();
    ctx.fillStyle = '#1e40af';
    ctx.font = 'bold 8px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('INPUT', cx + 8, cy + 3);
}

/** Reset embedding cache (call when canvas is cleared). */
export function resetEmbeddingCache() {
    embeddingCache = null;
    embeddingPredKey = '';
}
