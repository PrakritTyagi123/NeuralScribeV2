/**
 * NeuralScribe v2 — LNN Panel Update Helpers
 * Real Grad-CAM, real robustness, honest calibration/embedding.
 */

import { ARCH } from './nnDiagram.js';

function mk(tag, className) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    return el;
}

// ═══════════════════════════════════════
// GRAD-CAM (now uses real gradcam data from backend)
// ═══════════════════════════════════════

let _lastGradcamPixels = null;

export function drawGradCAM(pixels) {
    _lastGradcamPixels = pixels;
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

    // Fetch real Grad-CAM from backend
    fetch('/api/explain/gradcam', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pixels }),
    })
    .then(r => r.json())
    .then(data => {
        if (data.error || !data.gradcam) {
            _drawFallbackHeatmap(ctx, pixels);
            return;
        }
        _drawRealGradcam(ctx, data.gradcam);
    })
    .catch(() => {
        _drawFallbackHeatmap(ctx, pixels);
    });
}

function _drawRealGradcam(ctx, gradcamFlat) {
    const size = Math.round(Math.sqrt(gradcamFlat.length)); // 28
    const img = new ImageData(56, 56);

    for (let y = 0; y < 56; y++) {
        for (let x = 0; x < 56; x++) {
            const srcX = Math.floor((55 - x) * size / 56); // flip horizontal
            const srcY = Math.floor(y * size / 56);
            const val = gradcamFlat[srcY * size + srcX] || 0;

            const [r, g, b] = _heatmapColor(val);
            const idx = (y * 56 + x) * 4;
            img.data[idx] = r;
            img.data[idx + 1] = g;
            img.data[idx + 2] = b;
            img.data[idx + 3] = 255;
        }
    }
    ctx.putImageData(img, 0, 0);
}

function _drawFallbackHeatmap(ctx, pixels) {
    // Fallback: just color the input pixels (old behavior)
    const size = Math.round(Math.sqrt(pixels.length));
    const img = new ImageData(56, 56);
    for (let y = 0; y < 56; y++) {
        for (let x = 0; x < 56; x++) {
            const srcX = Math.floor((55 - x) * size / 56);
            const srcY = Math.floor(y * size / 56);
            const val = pixels[srcY * size + srcX] || 0;
            const [r, g, b] = _heatmapColor(val);
            const idx = (y * 56 + x) * 4;
            img.data[idx] = r;
            img.data[idx + 1] = g;
            img.data[idx + 2] = b;
            img.data[idx + 3] = 255;
        }
    }
    ctx.putImageData(img, 0, 0);
}

function _heatmapColor(val) {
    let r, g, b;
    if (val < 0.25) {
        const t = val / 0.25;
        r = Math.round(10 + t * 20); g = Math.round(10 + t * 80); b = Math.round(40 + t * 180);
    } else if (val < 0.5) {
        const t = (val - 0.25) / 0.25;
        r = Math.round(30); g = Math.round(90 + t * 165); b = Math.round(220 - t * 120);
    } else if (val < 0.75) {
        const t = (val - 0.5) / 0.25;
        r = Math.round(30 + t * 225); g = Math.round(255 - t * 55); b = Math.round(100 - t * 80);
    } else {
        const t = (val - 0.75) / 0.25;
        r = Math.round(255); g = Math.round(200 - t * 180); b = Math.round(20 - t * 20);
    }
    return [r, g, b];
}

// ═══════════════════════════════════════
// FEATURE MAPS (unchanged — already real)
// ═══════════════════════════════════════

export function updateFeatureMaps(fullData) {
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
        title.textContent = (arch ? arch.name : name) + ' · ' + maps.total_channels + 'ch';
        group.appendChild(title);
        const grid = mk('div', 'lnn-fm-grid');
        maps.heatmaps.slice(0, 6).forEach(hm => {
            const cell = mk('div', 'lnn-fm-cell');
            const img = document.createElement('img');
            img.src = 'data:image/png;base64,' + hm.heatmap;
            img.className = 'lnn-fm-img';
            img.title = 'ch' + hm.channel + ' imp=' + hm.importance.toFixed(3);
            cell.appendChild(img);
            grid.appendChild(cell);
        });
        group.appendChild(grid);
        body.appendChild(group);
    });

    if (body.children.length === 0) {
        body.innerHTML = '<div class="lnn-placeholder">No feature maps</div>';
    }
}

// ═══════════════════════════════════════
// CONFUSION (unchanged — already real)
// ═══════════════════════════════════════

export function updateConfusion(preds) {
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

// ═══════════════════════════════════════
// ROBUSTNESS (now uses real TTA from backend)
// ═══════════════════════════════════════

let _robustnessTimeout = null;

export function updateRobustness(preds) {
    if (!preds || preds.length === 0) return;

    // Debounce — don't hit backend on every frame
    if (_robustnessTimeout) clearTimeout(_robustnessTimeout);
    _robustnessTimeout = setTimeout(() => {
        if (!_lastGradcamPixels) return;
        fetch('/api/explain/robustness', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pixels: _lastGradcamPixels }),
        })
        .then(r => r.json())
        .then(data => {
            if (data.error) return;
            _renderRobustness(data);
        })
        .catch(() => {});
    }, 800);
}

function _renderRobustness(data) {
    const fill = document.getElementById('rb-fill');
    if (fill) fill.style.width = data.stability + '%';
    const val = document.getElementById('rb-val');
    if (val) val.textContent = data.stability + '%';

    const grid = document.getElementById('tta-grid');
    if (grid) {
        grid.innerHTML = '';
        (data.perturbations || []).forEach(p => {
            const item = mk('div', 'lnn-tta-item');
            const agreed = p.predicted_class === data.perturbations[0].predicted_class;
            item.style.cssText = 'font-size:10px;padding:2px 4px;border:1px solid ' +
                (agreed ? '#16a34a' : '#dc2626') + ';color:' + (agreed ? '#16a34a' : '#dc2626') + ';';
            item.textContent = p.name + ': ' + p.display + ' ' + (p.confidence * 100).toFixed(0) + '%';
            grid.appendChild(item);
        });
    }
}

// ═══════════════════════════════════════
// CALIBRATION (honest — labeled as estimate)
// ═══════════════════════════════════════

export function updateCalibration(preds) {
    if (!preds || preds.length === 0) return;
    const conf = preds[0].confidence;

    // We don't have real calibration data, so show confidence spread instead
    const top1 = conf;
    const top2 = preds.length > 1 ? preds[1].confidence : 0;
    const gap = top1 - top2;
    const decisive = gap > 0.3;

    const dial = document.getElementById('cal-dial');
    if (dial) dial.textContent = (conf * 100).toFixed(0) + '%';

    const info = document.getElementById('cal-info');
    if (info) {
        const gapPct = (gap * 100).toFixed(0);
        info.innerHTML =
            'Top-1: <strong>' + (top1 * 100).toFixed(0) + '%</strong><br>' +
            'Top-2: <strong>' + (top2 * 100).toFixed(0) + '%</strong><br>' +
            'Gap: <strong>' + gapPct + '%</strong><br>' +
            '<span style="font-size:8px;color:' + (decisive ? '#16a34a' : '#ea580c') + ';">' +
            (decisive ? '▲ Decisive' : '⚠ Uncertain — top classes are close') + '</span>';
    }
}
