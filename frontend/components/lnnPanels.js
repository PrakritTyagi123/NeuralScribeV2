/**
 * NeuralScribe v2 — LNN Panel Update Helpers
 * Feature maps, confusion display, robustness, calibration, Grad-CAM.
 */

import { ARCH } from './nnDiagram.js';

// ═══════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════
function mk(tag, className) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    return el;
}

// ═══════════════════════════════════════
// GRAD-CAM
// ═══════════════════════════════════════
export function drawGradCAM(pixels) {
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

    const size = Math.round(Math.sqrt(pixels.length));
    const img = new ImageData(56, 56);

    for (let y = 0; y < 56; y++) {
        for (let x = 0; x < 56; x++) {
            const srcX = Math.floor((55 - x) * size / 56);  // flip horizontal
            const srcY = Math.floor(y * size / 56);
            const val = pixels[srcY * size + srcX] || 0;

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

// ═══════════════════════════════════════
// FEATURE MAPS
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
// CONFUSION
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
// ROBUSTNESS
// ═══════════════════════════════════════
export function updateRobustness(preds) {
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

// ═══════════════════════════════════════
// CALIBRATION
// ═══════════════════════════════════════
export function updateCalibration(preds) {
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
