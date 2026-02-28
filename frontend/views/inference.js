/** Inference view — fast prediction mode. */
import { createCanvas } from '../components/canvas.js';
import { createConfidenceBars, updateConfidenceBars } from '../components/confidenceBars.js';

let predictTimer = null;

export async function renderInference(container) {
    container.innerHTML = `<div class="view-title">Inference</div>`;

    // Check model status
    let modelReady = false;
    try {
        const res = await fetch('/api/inference/status');
        const data = await res.json();
        modelReady = data.ready;
        if (!modelReady) {
            container.innerHTML += `<div class="panel"><div class="panel-body text-muted">No model loaded. Go to Model Manager to load one.</div></div>`;
            return;
        }
    } catch (e) {
        container.innerHTML += `<div class="panel"><div class="panel-body text-muted">Cannot reach backend.</div></div>`;
        return;
    }

    const layout = document.createElement('div');
    layout.className = 'grid-3';
    container.appendChild(layout);

    // ── Canvas ──
    const canvasPanel = document.createElement('div');
    canvasPanel.className = 'panel';
    canvasPanel.innerHTML = '<div class="panel-header">Draw</div>';
    const canvasBody = document.createElement('div');
    canvasBody.className = 'panel-body';

    const canvasObj = createCanvas(280);
    canvasBody.appendChild(canvasObj.element);
    canvasPanel.appendChild(canvasBody);
    layout.appendChild(canvasPanel);

    // ── Prediction output ──
    const predPanel = document.createElement('div');
    predPanel.className = 'panel';
    predPanel.innerHTML = '<div class="panel-header">Prediction</div>';
    const predBody = document.createElement('div');
    predBody.className = 'panel-body';

    const bigPred = document.createElement('div');
    bigPred.style.cssText = 'font-size:64px;font-weight:bold;text-align:center;padding:16px 0;';
    bigPred.textContent = '?';
    predBody.appendChild(bigPred);

    const confText = document.createElement('div');
    confText.className = 'text-sm text-muted';
    confText.style.textAlign = 'center';
    confText.textContent = 'Draw a character';
    predBody.appendChild(confText);

    const topK = createConfidenceBars([]);
    predBody.appendChild(topK);

    const timeText = document.createElement('div');
    timeText.className = 'text-sm text-muted mt-8';
    timeText.textContent = '';
    predBody.appendChild(timeText);

    predPanel.appendChild(predBody);
    layout.appendChild(predPanel);

    // ── Options + category grid ──
    const optPanel = document.createElement('div');
    optPanel.className = 'panel';
    optPanel.innerHTML = '<div class="panel-header">Options</div>';
    const optBody = document.createElement('div');
    optBody.className = 'panel-body';

    optBody.innerHTML = `
        <div class="form-group">
            <label>TTA (Test-Time Augmentation)</label>
            <select id="inf-tta">
                <option value="false">Off (fast)</option>
                <option value="true">On (robust)</option>
            </select>
        </div>
        <div class="form-group">
            <label>Top K</label>
            <input type="number" id="inf-topk" value="5" min="1" max="20">
        </div>
    `;

    const catGrid = document.createElement('div');
    catGrid.id = 'inf-categories';
    catGrid.className = 'mt-16 text-sm';
    catGrid.innerHTML = '<div class="text-muted">Category breakdown appears after prediction</div>';
    optBody.appendChild(catGrid);

    optPanel.appendChild(optBody);
    layout.appendChild(optPanel);

    // ── Auto-predict on draw ──
    canvasObj.onChange(() => {
        if (predictTimer) clearTimeout(predictTimer);
        predictTimer = setTimeout(() => runPrediction(), 300);
    });

    async function runPrediction() {
        const pixels = canvasObj.getPixels();
        const useTTA = container.querySelector('#inf-tta').value === 'true';
        const topKVal = parseInt(container.querySelector('#inf-topk').value) || 5;

        // Check if canvas is mostly empty
        const sum = pixels.reduce((a, b) => a + b, 0);
        if (sum < 1) {
            bigPred.textContent = '?';
            confText.textContent = 'Draw a character';
            updateConfidenceBars(topK, []);
            return;
        }

        try {
            const res = await fetch('/api/inference/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pixels, top_k: topKVal, use_tta: useTTA }),
            });
            const data = await res.json();

            if (data.error) {
                bigPred.textContent = '!';
                confText.textContent = data.error;
                return;
            }

            bigPred.textContent = data.predicted_label;
            confText.textContent = `${(data.confidence * 100).toFixed(1)}% confidence | ${data.inference_time_ms.toFixed(1)}ms`;
            updateConfidenceBars(topK, data.top_k || []);
            timeText.textContent = `TTA: ${data.used_tta ? 'on' : 'off'}`;

            // Category breakdown
            if (data.category_probabilities) {
                let html = '';
                for (const [cat, items] of Object.entries(data.category_probabilities)) {
                    const topItem = items[0];
                    if (topItem && topItem.probability > 0.01) {
                        html += `<div><strong>${cat}:</strong> ${topItem.display} (${(topItem.probability * 100).toFixed(1)}%)</div>`;
                    }
                }
                catGrid.innerHTML = html || '<div class="text-muted">No significant categories</div>';
            }
        } catch (e) {
            confText.textContent = `Error: ${e.message}`;
        }
    }
}