/**
 * NeuralScribe v2 — LNN Chart Renderers
 * Probability evolution and stroke timeline.
 * Embedding is now in embedding3d.js (Three.js).
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
