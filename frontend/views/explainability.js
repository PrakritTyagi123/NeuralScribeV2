/** Explainability view — live neural network visualization. */
import { createCanvas } from '../components/canvas.js';
import { createNetworkFlow } from '../components/networkFlow.js';
import { createFeatureMapViewer } from '../components/featureMapViewer.js';
import { createProbEvolution } from '../components/probEvolution.js';
import { createConfidenceBars, updateConfidenceBars } from '../components/confidenceBars.js';

let cachedPixels = null;

export async function renderExplainability(container) {
    container.innerHTML = `<div class="view-title">Explainability</div>`;

    // Check model
    try {
        const res = await fetch('/api/inference/status');
        const data = await res.json();
        if (!data.ready) {
            container.innerHTML += `<div class="panel"><div class="panel-body text-muted">No model loaded.</div></div>`;
            return;
        }
    } catch (e) {
        container.innerHTML += `<div class="panel"><div class="panel-body text-muted">Cannot reach backend.</div></div>`;
        return;
    }

    // Mode toggle
    const modeBar = document.createElement('div');
    modeBar.className = 'flex-between mb-16';
    modeBar.innerHTML = `
        <button class="btn btn-primary" id="explain-run">Analyze</button>
        <span class="text-sm text-muted" id="explain-time"></span>
    `;
    container.appendChild(modeBar);

    const layout = document.createElement('div');
    layout.className = 'grid-3';
    container.appendChild(layout);

    // ── Input panel ──
    const inputPanel = document.createElement('div');
    inputPanel.className = 'panel';
    inputPanel.innerHTML = '<div class="panel-header">Input</div>';
    const inputBody = document.createElement('div');
    inputBody.className = 'panel-body';

    const canvasObj = createCanvas(224);
    inputBody.appendChild(canvasObj.element);

    const processedImg = document.createElement('div');
    processedImg.className = 'mt-8';
    processedImg.innerHTML = '<div class="text-sm text-muted">Processed 28×28 appears after analysis</div>';
    inputBody.appendChild(processedImg);

    inputPanel.appendChild(inputBody);
    layout.appendChild(inputPanel);

    // ── Network flow panel ──
    const flowPanel = document.createElement('div');
    flowPanel.className = 'panel';
    flowPanel.innerHTML = '<div class="panel-header">Network Flow</div>';
    const flowBody = document.createElement('div');
    flowBody.className = 'panel-body';
    flowBody.id = 'explain-flow';

    const featureMapArea = document.createElement('div');
    featureMapArea.id = 'explain-featuremaps';
    featureMapArea.className = 'mt-8';

    const networkFlow = createNetworkFlow(
        ['stem', 'block_0', 'block_1', 'block_2', 'block_3', 'output'],
        async (layerName) => {
            if (!cachedPixels) return;
            featureMapArea.innerHTML = '<div class="text-sm text-muted">Loading...</div>';
            try {
                const res = await fetch('/api/explain/layer', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ pixels: cachedPixels, layer_name: layerName, max_channels: 16 }),
                });
                const data = await res.json();
                featureMapArea.innerHTML = '';
                if (data.error) {
                    featureMapArea.innerHTML = `<div class="text-sm text-muted">${data.error}</div>`;
                } else {
                    featureMapArea.appendChild(createFeatureMapViewer(data));
                }
            } catch (e) {
                featureMapArea.innerHTML = `<div class="text-sm text-muted">Error: ${e.message}</div>`;
            }
        }
    );

    flowBody.appendChild(networkFlow);
    flowBody.appendChild(featureMapArea);
    flowPanel.appendChild(flowBody);
    layout.appendChild(flowPanel);

    // ── Output panel ──
    const outPanel = document.createElement('div');
    outPanel.className = 'panel';
    outPanel.innerHTML = '<div class="panel-header">Output</div>';
    const outBody = document.createElement('div');
    outBody.className = 'panel-body';

    const predBars = createConfidenceBars([]);
    outBody.appendChild(predBars);

    const evolTitle = document.createElement('div');
    evolTitle.className = 'text-bold text-sm mt-16 mb-8';
    evolTitle.textContent = 'Probability Evolution';
    outBody.appendChild(evolTitle);

    const evolArea = document.createElement('div');
    evolArea.id = 'explain-evolution';
    evolArea.innerHTML = '<div class="text-sm text-muted">Run analysis to see evolution</div>';
    outBody.appendChild(evolArea);

    outPanel.appendChild(outBody);
    layout.appendChild(outPanel);

    // ── Analyze button ──
    container.querySelector('#explain-run').addEventListener('click', async () => {
        const pixels = canvasObj.getPixels();
        const sum = pixels.reduce((a, b) => a + b, 0);
        if (sum < 1) return;

        cachedPixels = pixels;
        const btn = container.querySelector('#explain-run');
        btn.disabled = true;
        btn.textContent = 'Analyzing...';

        try {
            const res = await fetch('/api/explain/full', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pixels }),
            });
            const data = await res.json();

            if (data.error) {
                btn.textContent = 'Analyze';
                btn.disabled = false;
                return;
            }

            // Timing
            container.querySelector('#explain-time').textContent = `${data.inference_time_ms.toFixed(1)}ms`;

            // Processed input
            if (data.input_image) {
                processedImg.innerHTML = `
                    <div class="text-sm text-bold mb-8">Processed 28×28</div>
                    <img src="data:image/png;base64,${data.input_image}" 
                         style="border:1px solid #000;image-rendering:pixelated;width:112px;height:112px;">
                `;
            }

            // Predictions
            updateConfidenceBars(predBars, data.predictions || []);

            // Feature maps for first available layer
            featureMapArea.innerHTML = '';
            const firstLayer = Object.keys(data.feature_maps || {})[0];
            if (firstLayer && data.feature_maps[firstLayer]) {
                featureMapArea.appendChild(createFeatureMapViewer(data.feature_maps[firstLayer]));
            }

            // Probability evolution
            evolArea.innerHTML = '';
            if (data.probability_evolution) {
                evolArea.appendChild(createProbEvolution(data.probability_evolution));
            }

        } catch (e) {
            console.error(e);
        }

        btn.textContent = 'Analyze';
        btn.disabled = false;
    });
}