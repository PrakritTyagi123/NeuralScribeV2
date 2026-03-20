/**
 * NeuralScribe v2 — Data Preparation
 * Reads EMNIST directly, preprocesses, augments, caches.
 * User configures: augmentation factor.
 */
import { createProgressBar, updateProgressBar } from '../components/progressBar.js';
import { createLogConsole, appendLog } from '../components/logConsole.js';
import { onWsEvent } from '../src/ws.js';
import { showToast } from '../components/toast.js';

let unsubs = [];
let sysPoll = null;

export async function renderDataPrep(container) {
    unsubs.forEach(u => u()); unsubs = [];
    if (sysPoll) clearInterval(sysPoll);

    container.innerHTML = '';
    container.className = 'view-fit';

    const title = document.createElement('div');
    title.className = 'view-title';
    title.textContent = 'Data Preparation';
    container.appendChild(title);

    const desc = document.createElement('div');
    desc.className = 'text-sm text-muted';
    desc.style.marginBottom = '8px';
    desc.textContent = 'Step 2: Preprocess and augment EMNIST for training. Reads directly from the downloaded data.';
    container.appendChild(desc);

    // Top row
    const topRow = document.createElement('div');
    topRow.className = 'grid-2';
    topRow.style.flexShrink = '0';
    container.appendChild(topRow);

    // Config panel
    const configPanel = document.createElement('div');
    configPanel.className = 'panel';
    configPanel.style.marginBottom = '8px';
    configPanel.innerHTML = `
        <div class="panel-header">Configuration</div>
        <div class="panel-body" style="padding:8px;">
            <div class="form-group" style="margin-bottom:6px;">
                <label>Dataset</label>
                <span class="text-sm" id="prep-info">Checking...</span>
            </div>
            <div class="form-group" style="margin-bottom:6px;">
                <label>Augmentation Factor</label>
                <input type="number" id="prep-aug" value="3" min="0" max="10" style="padding:2px 4px;width:60px;">
                <span class="text-sm text-muted">× multiplier (0 = none, 3 = recommended)</span>
            </div>
            <div class="form-group" style="margin-bottom:6px;">
                <label>Expected Samples</label>
                <span class="text-sm" id="prep-estimate">--</span>
            </div>
            <div class="btn-group mt-8">
                <button class="btn btn-primary" id="prep-start">Start Preparation</button>
                <button class="btn" id="prep-cancel" disabled>Cancel</button>
            </div>
        </div>
    `;
    topRow.appendChild(configPanel);

    // Progress panel
    const progressPanel = document.createElement('div');
    progressPanel.className = 'panel';
    progressPanel.style.marginBottom = '8px';
    progressPanel.innerHTML = '<div class="panel-header">Progress</div>';
    const progressBody = document.createElement('div');
    progressBody.className = 'panel-body';
    progressBody.style.padding = '8px';

    const stages = [
        { id: 'loading', label: '1. Loading EMNIST' },
        { id: 'preprocessing', label: '2. Preprocessing' },
        { id: 'augmenting', label: '3. Augmentation' },
        { id: 'caching', label: '4. Split & Cache' },
    ];
    const bars = {};
    stages.forEach(s => {
        const row = document.createElement('div');
        row.style.marginBottom = '4px';
        row.innerHTML = `<div class="text-sm" style="margin-bottom:2px;">${s.label}</div>`;
        const bar = createProgressBar(0, 'Waiting');
        row.appendChild(bar);
        progressBody.appendChild(row);
        bars[s.id] = bar;
    });

    const sysInfo = document.createElement('div');
    sysInfo.className = 'text-sm mt-8';
    progressBody.appendChild(sysInfo);
    progressPanel.appendChild(progressBody);
    topRow.appendChild(progressPanel);

    // Log panel
    const logPanel = document.createElement('div');
    logPanel.className = 'panel';
    logPanel.style.cssText = 'flex:1;display:flex;flex-direction:column;overflow:hidden;margin-bottom:0;';
    logPanel.innerHTML = '<div class="panel-header" style="flex-shrink:0;">Log</div>';
    const logConsole = createLogConsole();
    logConsole.style.cssText = 'flex:1;height:auto;min-height:0;border:none;overflow-y:auto;';
    logPanel.appendChild(logConsole);
    container.appendChild(logPanel);

    const startBtn = container.querySelector('#prep-start');
    const cancelBtn = container.querySelector('#prep-cancel');
    const augInput = container.querySelector('#prep-aug');
    const infoEl = container.querySelector('#prep-info');
    const estimateEl = container.querySelector('#prep-estimate');

    // Estimate updater
    const BASE_SAMPLES = 131600;
    function updateEstimate() {
        const aug = parseInt(augInput.value) || 0;
        const total = BASE_SAMPLES * (1 + aug);
        estimateEl.textContent = `~${total.toLocaleString()} total (${BASE_SAMPLES.toLocaleString()} base × ${1 + aug})`;
    }
    augInput.addEventListener('input', updateEstimate);
    updateEstimate();

    // Load status
    try {
        const st = await (await fetch('/api/dataset/status')).json();
        if (st.cache_exists) {
            infoEl.textContent = `English — 47 classes — Cache: ${st.cache_size}`;
            appendLog(logConsole, `Cache exists: ${st.cache_size} (${st.num_classes} classes). Re-run to rebuild.`);
        } else if (st.emnist_downloaded) {
            infoEl.textContent = 'English — 47 classes — EMNIST downloaded, ready to prepare';
            appendLog(logConsole, 'EMNIST downloaded. Click Start to preprocess and cache.');
        } else {
            infoEl.textContent = 'English — 47 classes — EMNIST NOT downloaded';
            appendLog(logConsole, '⚠ EMNIST not downloaded. Go to Download page first.');
            startBtn.disabled = true;
        }
    } catch (e) { appendLog(logConsole, 'Cannot reach backend.'); }

    // System polling
    async function pollSys() {
        try {
            const [g, s] = await Promise.all([fetch('/api/system/gpu'), fetch('/api/system/stats')]);
            const gpu = await g.json(), sys = await s.json();
            sysInfo.innerHTML = `<strong>CPU:</strong> ${sys.cpu_percent}% | <strong>RAM:</strong> ${sys.ram_used_gb}/${sys.ram_total_gb}GB` +
                (gpu.available ? ` | <strong>GPU:</strong> ${gpu.gpu_util_percent}%` : '');
        } catch {}
    }
    pollSys(); sysPoll = setInterval(pollSys, 5000);

    // Start
    startBtn.addEventListener('click', async () => {
        startBtn.disabled = true; cancelBtn.disabled = false;
        Object.values(bars).forEach(b => updateProgressBar(b, 0, 'Waiting'));

        const aug = parseInt(augInput.value) || 0;
        appendLog(logConsole, `Starting: augmentation ×${aug}`);

        try {
            const res = await fetch('/api/dataset/prepare', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ overrides: { 'augmentation.precompute_factor': aug } }),
            });
            const d = await res.json();
            if (d.error) {
                appendLog(logConsole, `Error: ${d.error}`);
                startBtn.disabled = false; cancelBtn.disabled = true;
            }
        } catch (e) {
            appendLog(logConsole, `Error: ${e.message}`);
            startBtn.disabled = false; cancelBtn.disabled = true;
        }
    });

    cancelBtn.addEventListener('click', async () => {
        await fetch('/api/dataset/cancel', { method: 'POST' });
        appendLog(logConsole, 'Cancellation requested.');
    });

    // WS progress
    const stageMap = {
        'loading_emnist': 'loading',
        'preprocessing': 'preprocessing',
        'augmenting': 'augmenting',
        'splitting': 'caching',
        'caching': 'caching',
        'complete': 'caching',
    };

    unsubs.push(onWsEvent('dataset_progress', (d) => {
        const key = stageMap[d.stage];
        if (key && bars[key]) {
            const pct = d.total > 0 ? Math.min(100, d.processed / d.total * 100) : 0;
            let text = d.message || `${Math.round(pct)}%`;
            updateProgressBar(bars[key], pct, text);

            // Mark earlier stages complete
            const order = ['loading', 'preprocessing', 'augmenting', 'caching'];
            const idx = order.indexOf(key);
            for (let i = 0; i < idx; i++) updateProgressBar(bars[order[i]], 100, 'Complete');
        }
        if (d.message) appendLog(logConsole, d.message);
    }));

    unsubs.push(onWsEvent('dataset_complete', (d) => {
        Object.values(bars).forEach(b => updateProgressBar(b, 100, 'Complete'));
        startBtn.disabled = false; cancelBtn.disabled = true;
        if (d.error) {
            appendLog(logConsole, `Error: ${d.error}`);
        } else {
            appendLog(logConsole, `Complete! ${d.total_samples?.toLocaleString()} samples, cache: ${d.cache_size}, time: ${d.elapsed_seconds}s`);
            infoEl.textContent = `English — 47 classes — Cache: ${d.cache_size}`;
            showToast('Dataset prepared! Go to Training.');
        }
    }));
}
