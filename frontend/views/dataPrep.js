/** Data Preparation view. */
import { createProgressBar, updateProgressBar } from '../components/progressBar.js';
import { createLogConsole, appendLog } from '../components/logConsole.js';
import { createStatCard } from '../components/statCard.js';
import { onWsEvent } from '../src/ws.js';
import { showToast } from '../components/toast.js';

let unsubscribe = null;

export async function renderDataPrep(container) {
    if (unsubscribe) unsubscribe();

    container.innerHTML = `<div class="view-title">Data Preparation</div>`;

    const layout = document.createElement('div');
    layout.className = 'grid-3';
    container.appendChild(layout);

    // ── Config panel ──
    const configPanel = document.createElement('div');
    configPanel.className = 'panel';
    configPanel.innerHTML = `
        <div class="panel-header">Config</div>
        <div class="panel-body" id="prep-config">
            <div class="form-group">
                <label>EMNIST Split</label>
                <input type="text" value="balanced" disabled>
            </div>
            <div class="form-group">
                <label>Synthetic Enabled</label>
                <select id="prep-synthetic"><option value="true">Yes</option><option value="false">No</option></select>
            </div>
            <div class="form-group">
                <label>Samples/Symbol</label>
                <input type="number" id="prep-samples" value="1500">
            </div>
            <div class="form-group">
                <label>Augment Factor</label>
                <input type="number" id="prep-augfactor" value="3">
            </div>
            <div class="btn-group mt-16">
                <button class="btn btn-primary" id="prep-start">Start</button>
                <button class="btn" id="prep-cancel" disabled>Cancel</button>
            </div>
        </div>
    `;
    layout.appendChild(configPanel);

    // ── Progress panel ──
    const progressPanel = document.createElement('div');
    progressPanel.className = 'panel';
    progressPanel.innerHTML = `<div class="panel-header">Progress</div>`;
    const progressBody = document.createElement('div');
    progressBody.className = 'panel-body';

    const progressBar = createProgressBar(0, 'Idle');
    progressBody.appendChild(progressBar);

    const progressInfo = document.createElement('div');
    progressInfo.className = 'mt-8 text-sm';
    progressInfo.id = 'prep-info';
    progressInfo.textContent = 'Ready to prepare dataset.';
    progressBody.appendChild(progressInfo);

    progressPanel.appendChild(progressBody);
    layout.appendChild(progressPanel);

    // ── Stats panel ──
    const statsPanel = document.createElement('div');
    statsPanel.className = 'panel';
    statsPanel.innerHTML = `<div class="panel-header">System</div>`;
    const statsBody = document.createElement('div');
    statsBody.className = 'panel-body';
    statsBody.id = 'prep-stats';
    statsBody.innerHTML = '<div class="text-sm text-muted">Loading...</div>';
    statsPanel.appendChild(statsBody);
    layout.appendChild(statsPanel);

    // ── Log console ──
    const logPanel = document.createElement('div');
    logPanel.className = 'panel mt-16';
    logPanel.innerHTML = `<div class="panel-header">Log</div>`;
    const logConsole = createLogConsole();
    logPanel.appendChild(logConsole);
    container.appendChild(logPanel);

    // ── Load current status ──
    try {
        const res = await fetch('/api/dataset/status');
        const status = await res.json();
        progressInfo.textContent = status.cache_exists
            ? `Cache exists: ${status.cache_size} (${status.num_classes} classes)`
            : 'No cache found. Click Start to prepare.';

        const sysRes = await fetch('/api/system/stats');
        const sys = await sysRes.json();
        statsBody.innerHTML = '';
        statsBody.appendChild(createStatCard('CPU', `${sys.cpu_percent}%`, `${sys.cpu_count} cores`));
        statsBody.appendChild(createStatCard('RAM', `${sys.ram_percent}%`, `${sys.ram_used_gb}/${sys.ram_total_gb} GB`));
    } catch (e) {
        progressInfo.textContent = 'Cannot reach backend.';
    }

    // ── Buttons ──
    const startBtn = container.querySelector('#prep-start');
    const cancelBtn = container.querySelector('#prep-cancel');

    startBtn.addEventListener('click', async () => {
        startBtn.disabled = true;
        cancelBtn.disabled = false;
        appendLog(logConsole, 'Starting dataset preparation...');

        try {
            const res = await fetch('/api/dataset/prepare', { method: 'POST' });
            const data = await res.json();
            appendLog(logConsole, data.message || JSON.stringify(data));
        } catch (e) {
            appendLog(logConsole, `Error: ${e.message}`);
            startBtn.disabled = false;
            cancelBtn.disabled = true;
        }
    });

    cancelBtn.addEventListener('click', async () => {
        await fetch('/api/dataset/cancel', { method: 'POST' });
        appendLog(logConsole, 'Cancellation requested.');
    });

    // ── WS updates ──
    unsubscribe = onWsEvent('dataset_progress', (data) => {
        const pct = data.total > 0 ? (data.processed / data.total * 100) : 0;
        updateProgressBar(progressBar, pct, data.message || `${Math.round(pct)}%`);
        progressInfo.innerHTML = `
            Stage: ${data.stage || '--'}<br>
            ${data.processed?.toLocaleString() || 0} / ${data.total?.toLocaleString() || 0}
            ${data.samples_per_sec ? ` | ${data.samples_per_sec.toLocaleString()}/s` : ''}
            ${data.eta_seconds ? ` | ETA: ${Math.round(data.eta_seconds)}s` : ''}
        `;
        if (data.message) appendLog(logConsole, data.message);
    });

    const unsub2 = onWsEvent('dataset_complete', (data) => {
        updateProgressBar(progressBar, 100, 'Complete');
        startBtn.disabled = false;
        cancelBtn.disabled = true;
        appendLog(logConsole, `Complete! ${data.total_samples?.toLocaleString()} samples, ${data.cache_size}`);
        showToast('Dataset preparation complete!');
    });

    const origUnsub = unsubscribe;
    unsubscribe = () => { origUnsub(); unsub2(); };
}