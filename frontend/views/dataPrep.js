/** Data Preparation view — language-aware, fits viewport. */
import { createProgressBar, updateProgressBar } from '../components/progressBar.js';
import { createLogConsole, appendLog } from '../components/logConsole.js';
import { createLanguageDropdown } from '../components/languageDropdown.js';
import { onWsEvent } from '../src/ws.js';
import { showToast } from '../components/toast.js';

let unsubscribers = [];
let sysPollTimer = null;

export async function renderDataPrep(container) {
    unsubscribers.forEach(u => u());
    unsubscribers = [];
    if (sysPollTimer) clearInterval(sysPollTimer);

    container.innerHTML = '';
    container.className = 'view-fit';

    const title = document.createElement('div');
    title.className = 'view-title';
    title.textContent = 'Data Preparation';
    container.appendChild(title);

    // ── Language toolbar ──
    const toolbar = document.createElement('div');
    toolbar.className = 'lang-toolbar';
    container.appendChild(toolbar);

    let logConsole; // declare early for use in dropdown callback

    const langDropdown = await createLanguageDropdown(async (language) => {
        // Switch backend language
        await fetch('/api/dataset/set-language', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ language }),
        });
        // Reload status
        await reloadStatus();
    }, 'prep_language');
    toolbar.appendChild(langDropdown.element);

    // ── Top row: config + progress ──
    const topRow = document.createElement('div');
    topRow.className = 'grid-2';
    topRow.style.cssText = 'flex-shrink:0;';
    container.appendChild(topRow);

    // Config panel
    const configPanel = document.createElement('div');
    configPanel.className = 'panel';
    configPanel.style.marginBottom = '8px';
    configPanel.innerHTML = `
        <div class="panel-header">Configuration</div>
        <div class="panel-body" style="padding:8px;">
            <div class="form-group" style="margin-bottom:4px;">
                <label>Dataset Source</label>
                <input type="text" id="prep-source" value="EMNIST Balanced" disabled style="padding:2px 4px;">
                <div class="text-sm text-muted">Source dataset for current language. Auto-downloads on first run.</div>
            </div>
            <div class="form-group" style="margin-bottom:4px;">
                <label>Augmentation Factor</label>
                <input type="number" id="prep-augfactor" value="3" style="padding:2px 4px;">
                <div class="text-sm text-muted">Multiplier for augmentations (rotation, shear, elastic).</div>
            </div>
            <div class="form-group" style="margin-bottom:4px;">
                <label>Classes</label>
                <span id="prep-classes" class="text-sm text-muted">--</span>
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
    progressPanel.innerHTML = `<div class="panel-header">Progress</div>`;
    const progressBody = document.createElement('div');
    progressBody.className = 'panel-body';
    progressBody.style.padding = '8px';

    const stages = [
        { id: 'emnist', label: '1. Loading Dataset' },
        { id: 'preprocessing', label: '2. Preprocessing' },
        { id: 'augmentation', label: '3. Augmentation' },
        { id: 'splitting', label: '4. Split & Cache' },
    ];

    const stageBars = {};
    stages.forEach(s => {
        const row = document.createElement('div');
        row.style.marginBottom = '4px';
        row.innerHTML = `<div class="text-sm" style="margin-bottom:2px;">${s.label}</div>`;
        const bar = createProgressBar(0, 'Waiting');
        row.appendChild(bar);
        progressBody.appendChild(row);
        stageBars[s.id] = bar;
    });

    const sysInfo = document.createElement('div');
    sysInfo.className = 'text-sm mt-8';
    sysInfo.id = 'prep-sys';
    progressBody.appendChild(sysInfo);
    progressPanel.appendChild(progressBody);
    topRow.appendChild(progressPanel);

    // ── Log panel ──
    const logPanel = document.createElement('div');
    logPanel.className = 'panel';
    logPanel.style.cssText = 'flex:1;display:flex;flex-direction:column;overflow:hidden;margin-bottom:0;';
    logPanel.innerHTML = `<div class="panel-header" style="flex-shrink:0;">Log</div>`;
    logConsole = createLogConsole();
    logConsole.style.cssText = 'flex:1;height:auto;min-height:0;border:none;overflow-y:auto;';
    logPanel.appendChild(logConsole);
    container.appendChild(logPanel);

    // ── Load status for current language ──
    async function reloadStatus() {
        try {
            const lang = langDropdown.getValue();
            const res = await fetch(`/api/dataset/status?language=${lang}`);
            const status = await res.json();
            const classesEl = container.querySelector('#prep-classes');
            if (classesEl) classesEl.textContent = `${status.num_classes} classes (${lang})`;
            if (status.cache_exists) {
                appendLog(logConsole, `[${lang}] Cache exists: ${status.cache_size} (${status.num_classes} classes)`);
            } else {
                appendLog(logConsole, `[${lang}] No cache found. Click Start to prepare dataset.`);
            }
            if (status.registry_status === 'placeholder') {
                appendLog(logConsole, `[${lang}] ⚠ This language is a placeholder — dataset not available yet.`);
                container.querySelector('#prep-start').disabled = true;
            } else {
                container.querySelector('#prep-start').disabled = false;
            }
        } catch (e) { appendLog(logConsole, 'Cannot reach backend.'); }
    }
    await reloadStatus();

    // ── System polling ──
    async function pollSys() {
        try {
            const [gpuRes, sysRes] = await Promise.all([fetch('/api/system/gpu'), fetch('/api/system/stats')]);
            const gpu = await gpuRes.json();
            const sys = await sysRes.json();
            sysInfo.innerHTML = `<strong>CPU:</strong> ${sys.cpu_percent}% | <strong>RAM:</strong> ${sys.ram_used_gb}/${sys.ram_total_gb}GB${gpu.available ? ` | <strong>GPU:</strong> ${gpu.gpu_util_percent}% VRAM:${gpu.memory_used_mb}MB` : ''}`;
        } catch (e) {}
    }
    pollSys();
    sysPollTimer = setInterval(pollSys, 5000);

    // ── Buttons ──
    const startBtn = container.querySelector('#prep-start');
    const cancelBtn = container.querySelector('#prep-cancel');

    startBtn.addEventListener('click', async () => {
        startBtn.disabled = true;
        cancelBtn.disabled = false;
        Object.values(stageBars).forEach(bar => updateProgressBar(bar, 0, 'Waiting'));

        const augVal = parseInt(container.querySelector('#prep-augfactor').value) || 3;
        const lang = langDropdown.getValue();

        appendLog(logConsole, `[${lang}] Starting preparation, augment factor=${augVal}`);
        try {
            const res = await fetch('/api/dataset/prepare', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ language: lang, overrides: { 'augmentation.precompute_factor': augVal } }),
            });
            const data = await res.json();
            if (data.error) {
                appendLog(logConsole, `Error: ${data.error}`);
                startBtn.disabled = false;
                cancelBtn.disabled = true;
            } else {
                appendLog(logConsole, data.message || JSON.stringify(data));
            }
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

    // ── WS events ──
    const stageMap = {
        'loading_emnist': 'emnist',
        'generating_synthetic': 'emnist',
        'merging': 'preprocessing',
        'preprocessing': 'preprocessing',
        'augmenting': 'augmentation',
        'splitting': 'splitting',
        'caching': 'splitting',
        'complete': 'splitting',
    };

    unsubscribers.push(onWsEvent('dataset_progress', (data) => {
        const stageKey = stageMap[data.stage] || null;
        const pct = data.total > 0 ? Math.min(100, data.processed / data.total * 100) : 0;

        if (stageKey && stageBars[stageKey]) {
            let text = `${Math.round(pct)}%`;
            if (data.samples_per_sec) text += ` | ${data.samples_per_sec.toLocaleString()}/s`;
            if (data.eta_seconds) text += ` | ETA ${Math.round(data.eta_seconds)}s`;
            updateProgressBar(stageBars[stageKey], pct, text);

            const order = ['emnist', 'preprocessing', 'augmentation', 'splitting'];
            const idx = order.indexOf(stageKey);
            for (let i = 0; i < idx; i++) updateProgressBar(stageBars[order[i]], 100, 'Complete');
        }
        if (data.message) appendLog(logConsole, data.message);
    }));

    unsubscribers.push(onWsEvent('dataset_complete', (data) => {
        Object.values(stageBars).forEach(bar => updateProgressBar(bar, 100, 'Complete'));
        startBtn.disabled = false;
        cancelBtn.disabled = true;
        appendLog(logConsole, `Complete! ${data.total_samples?.toLocaleString()} samples, ${data.cache_size}`);
        showToast(`Dataset ready for ${data.language || 'current language'}!`);
        langDropdown.refresh();
    }));
}