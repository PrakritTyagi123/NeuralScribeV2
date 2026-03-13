/** Training view — language-aware, fits viewport. */
import { createChart } from '../components/chart.js';
import { createProgressBar, updateProgressBar } from '../components/progressBar.js';
import { createLanguageDropdown, createModelDropdown } from '../components/languageDropdown.js';
import { onWsEvent } from '../src/ws.js';
import { showToast } from '../components/toast.js';

let unsubscribers = [];
let systemPollTimer = null;

export async function renderTraining(container) {
    unsubscribers.forEach(u => u());
    unsubscribers = [];
    if (systemPollTimer) clearInterval(systemPollTimer);

    container.innerHTML = '';
    container.className = 'view-fit';

    const title = document.createElement('div');
    title.className = 'view-title';
    title.textContent = 'Training';
    container.appendChild(title);

    // ── Language + Model toolbar ──
    const toolbar = document.createElement('div');
    toolbar.className = 'lang-toolbar';
    container.appendChild(toolbar);

    let modelDropdown;

    const langDropdown = await createLanguageDropdown(async (language) => {
        await fetch('/api/training/set-language', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ language }),
        });
        // Reload everything for new language
        if (modelDropdown) await modelDropdown.refresh(language);
        await reloadHistory();
    }, 'training_language');
    toolbar.appendChild(langDropdown.element);

    modelDropdown = await createModelDropdown(async (modelName) => {
        await fetch('/api/models/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: modelName, language: langDropdown.getValue() }),
        });
        showToast(`Model "${modelName}" loaded`);
    }, langDropdown.getValue());
    toolbar.appendChild(modelDropdown.element);

    // ── Status + progress ──
    const statusBar = document.createElement('div');
    statusBar.className = 'flex-between';
    statusBar.style.flexShrink = '0';
    statusBar.innerHTML = '<span class="text-sm text-muted">Loading...</span>';
    container.appendChild(statusBar);

    const epochProgress = createProgressBar(0, 'Idle');
    epochProgress.style.flexShrink = '0';
    container.appendChild(epochProgress);

    const sysBar = document.createElement('div');
    sysBar.className = 'flex gap-16 text-sm';
    sysBar.style.cssText = 'flex-shrink:0;margin:4px 0;';
    sysBar.innerHTML = '<span class="text-muted">System: loading...</span>';
    container.appendChild(sysBar);

    // ── Charts + Controls row ──
    const midRow = document.createElement('div');
    midRow.style.cssText = 'display:flex;gap:8px;flex-shrink:0;margin-bottom:8px;';
    container.appendChild(midRow);

    const lossPanel = document.createElement('div');
    lossPanel.className = 'panel';
    lossPanel.style.cssText = 'flex:1;margin-bottom:0;';
    lossPanel.innerHTML = '<div class="panel-header">Loss</div>';
    const lossBody = document.createElement('div');
    lossBody.className = 'panel-body';
    lossBody.style.padding = '4px';
    const lossChart = createChart(400, 180);
    lossBody.appendChild(lossChart.element);
    lossPanel.appendChild(lossBody);
    midRow.appendChild(lossPanel);

    const accPanel = document.createElement('div');
    accPanel.className = 'panel';
    accPanel.style.cssText = 'flex:1;margin-bottom:0;';
    accPanel.innerHTML = '<div class="panel-header">Accuracy</div>';
    const accBody = document.createElement('div');
    accBody.className = 'panel-body';
    accBody.style.padding = '4px';
    const accChart = createChart(400, 180);
    accBody.appendChild(accChart.element);
    accPanel.appendChild(accBody);
    midRow.appendChild(accPanel);

    const ctrlPanel = document.createElement('div');
    ctrlPanel.className = 'panel';
    ctrlPanel.style.cssText = 'width:220px;flex-shrink:0;margin-bottom:0;';
    ctrlPanel.innerHTML = `
        <div class="panel-header">Controls</div>
        <div class="panel-body" style="padding:8px;">
            <div class="form-group" style="margin-bottom:4px;">
                <label>Epochs</label>
                <input type="number" id="train-epochs" value="100" style="padding:2px 4px;">
            </div>
            <div class="form-group" style="margin-bottom:4px;">
                <label>Batch Size</label>
                <input type="number" id="train-bs" value="256" style="padding:2px 4px;">
            </div>
            <div class="btn-group mt-8" style="flex-wrap:wrap;gap:4px;">
                <button class="btn btn-primary" id="train-start">Start</button>
                <button class="btn" id="train-pause" disabled>Pause</button>
                <button class="btn" id="train-resume" disabled>Resume</button>
                <button class="btn" id="train-stop" disabled>Stop</button>
            </div>
        </div>
    `;
    midRow.appendChild(ctrlPanel);

    // ── Epoch history table ──
    const histPanel = document.createElement('div');
    histPanel.className = 'panel';
    histPanel.style.cssText = 'flex:1;display:flex;flex-direction:column;overflow:hidden;margin-bottom:0;';
    const histHeader = document.createElement('div');
    histHeader.className = 'panel-header';
    histHeader.style.cssText = 'flex-shrink:0;display:flex;justify-content:space-between;align-items:center;';
    histHeader.innerHTML = '<span>Epoch History</span>';
    const dlBtn = document.createElement('button');
    dlBtn.className = 'btn';
    dlBtn.style.cssText = 'font-size:10px;padding:1px 6px;';
    dlBtn.textContent = '↓ CSV';
    dlBtn.addEventListener('click', () => {
        const tbody = document.getElementById('hist-tbody');
        if (!tbody || !tbody.rows.length) return;
        let csv = 'Epoch,Train Loss,Train Acc,Val Loss,Val Acc,LR,Time,Best\n';
        for (const row of tbody.rows) csv += [...row.cells].map(c => c.textContent.trim()).join(',') + '\n';
        const a = document.createElement('a');
        a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
        a.download = `training_history_${langDropdown.getValue()}.csv`;
        a.click();
    });
    histHeader.appendChild(dlBtn);
    histPanel.appendChild(histHeader);
    const histBody = document.createElement('div');
    histBody.style.cssText = 'flex:1;overflow-y:auto;min-height:0;';
    const histTable = document.createElement('table');
    histTable.className = 'table';
    histTable.style.fontSize = '11px';
    histTable.innerHTML = `<thead><tr>
        <th>Epoch</th><th>Train Loss</th><th>Train Acc</th>
        <th>Val Loss</th><th>Val Acc</th><th>LR</th><th>Time</th><th></th>
    </tr></thead><tbody id="hist-tbody"></tbody>`;
    histBody.appendChild(histTable);
    histPanel.appendChild(histBody);
    container.appendChild(histPanel);

    // ── Data ──
    let trainLosses = [], valLosses = [], trainAccs = [], valAccs = [];

    async function reloadHistory() {
        trainLosses = []; valLosses = []; trainAccs = []; valAccs = [];
        const tbody = document.getElementById('hist-tbody');
        if (tbody) tbody.innerHTML = '';

        try {
            const lang = langDropdown.getValue();
            const res = await fetch(`/api/training/status?language=${lang}`);
            const status = await res.json();
            statusBar.innerHTML = `
                <span>${lang.toUpperCase()} — Epoch ${(status.current_epoch || 0) + 1} | Best: ${(status.best_val_acc * 100).toFixed(1)}% | ${status.device}</span>
                <span class="badge">${status.is_training ? 'TRAINING' : 'IDLE'}</span>
            `;
            (status.history || []).forEach(h => {
                trainLosses.push(h.train_loss); valLosses.push(h.val_loss);
                trainAccs.push(h.train_acc); valAccs.push(h.val_acc);
                addHistRow(h);
            });
            if (trainLosses.length > 0) {
                lossChart.setData([{ label: 'Train', data: trainLosses }, { label: 'Val', data: valLosses }]);
                accChart.setData([{ label: 'Train', data: trainAccs }, { label: 'Val', data: valAccs }]);
            } else {
                lossChart.setData([]);
                accChart.setData([]);
            }
            const cfgRes = await fetch(`/api/training/config?language=${lang}`);
            const cfg = await cfgRes.json();
            const ei = container.querySelector('#train-epochs');
            const bi = container.querySelector('#train-bs');
            if (ei) ei.value = cfg.training?.epochs || 100;
            if (bi) bi.value = cfg.training?.batch_size || 256;
        } catch (e) {}
    }
    await reloadHistory();

    // ── System polling ──
    async function pollSystem() {
        try {
            const [gpuRes, sysRes] = await Promise.all([fetch('/api/system/gpu'), fetch('/api/system/stats')]);
            const gpu = await gpuRes.json();
            const sys = await sysRes.json();
            sysBar.innerHTML = `<strong>CPU:</strong> ${sys.cpu_percent}% &nbsp; <strong>RAM:</strong> ${sys.ram_used_gb}/${sys.ram_total_gb}GB${gpu.available ? ` &nbsp; <strong>GPU:</strong> ${gpu.gpu_util_percent}% VRAM:${gpu.memory_used_mb}/${gpu.memory_total_mb}MB` : ''}`;
        } catch (e) {}
    }
    pollSystem();
    systemPollTimer = setInterval(pollSystem, 5000);

    function addHistRow(h) {
        const tbody = document.getElementById('hist-tbody');
        if (!tbody) return;
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${(h.epoch || 0) + 1}/${h.total_epochs || '?'}</td>
            <td>${h.train_loss?.toFixed(4) || '--'}</td>
            <td>${h.train_acc != null ? (h.train_acc * 100).toFixed(2) + '%' : '--'}</td>
            <td>${h.val_loss?.toFixed(4) || '--'}</td>
            <td>${h.val_acc != null ? (h.val_acc * 100).toFixed(2) + '%' : '--'}</td>
            <td>${h.lr || '--'}</td>
            <td>${h.epoch_time ? h.epoch_time.toFixed(1) + 's' : '--'}</td>
            <td>${h.is_best ? '★' : ''}</td>
        `;
        tbody.appendChild(tr);
        const parent = tbody.closest('div');
        if (parent) parent.scrollTop = parent.scrollHeight;
    }

    // ── Buttons ──
    const startBtn = container.querySelector('#train-start');
    const pauseBtn = container.querySelector('#train-pause');
    const resumeBtn = container.querySelector('#train-resume');
    const stopBtn = container.querySelector('#train-stop');

    function setUI(running) {
        startBtn.disabled = running;
        pauseBtn.disabled = !running;
        stopBtn.disabled = !running;
    }

    startBtn.addEventListener('click', async () => {
        const epochs = parseInt(container.querySelector('#train-epochs').value) || 100;
        const bs = parseInt(container.querySelector('#train-bs').value) || 256;
        const lang = langDropdown.getValue();
        const res = await fetch('/api/training/start', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ language: lang, overrides: { 'training.epochs': epochs, 'training.batch_size': bs } }),
        });
        const data = await res.json();
        if (data.error) {
            showToast(data.error);
        } else {
            setUI(true);
        }
    });
    pauseBtn.addEventListener('click', async () => { await fetch('/api/training/pause', { method: 'POST' }); resumeBtn.disabled = false; });
    resumeBtn.addEventListener('click', async () => { await fetch('/api/training/resume', { method: 'POST' }); resumeBtn.disabled = true; });
    stopBtn.addEventListener('click', async () => { await fetch('/api/training/stop', { method: 'POST' }); });

    // ── WS ──
    unsubscribers.push(onWsEvent('training_epoch', (data) => {
        // Only update if this is for our language
        if (data.language && data.language !== langDropdown.getValue()) return;

        trainLosses.push(data.train_loss); valLosses.push(data.val_loss);
        trainAccs.push(data.train_acc); valAccs.push(data.val_acc);
        lossChart.setData([{ label: 'Train', data: trainLosses }, { label: 'Val', data: valLosses }]);
        accChart.setData([{ label: 'Train', data: trainAccs }, { label: 'Val', data: valAccs }]);
        const pct = ((data.epoch + 1) / data.total_epochs * 100);
        updateProgressBar(epochProgress, pct, `Epoch ${data.epoch + 1}/${data.total_epochs} | val_acc=${(data.val_acc * 100).toFixed(1)}%`);
        statusBar.innerHTML = `
            <span>${langDropdown.getValue().toUpperCase()} — Epoch ${data.epoch + 1}/${data.total_epochs} | Best: ${(data.best_val_acc * 100).toFixed(1)}%</span>
            <span class="badge">TRAINING${data.is_best ? ' ★' : ''}</span>
        `;
        addHistRow(data);
    }));

    unsubscribers.push(onWsEvent('training_batch', (data) => {
        if (data.language && data.language !== langDropdown.getValue()) return;
        updateProgressBar(epochProgress, data.batch / data.total_batches * 100,
            `Epoch ${data.epoch + 1} — batch ${data.batch}/${data.total_batches} loss=${data.loss}`);
    }));

    unsubscribers.push(onWsEvent('training_complete', (data) => {
        if (data.language && data.language !== langDropdown.getValue()) return;
        setUI(false);
        updateProgressBar(epochProgress, 100, 'Complete');
        showToast(`Training complete for ${data.language || langDropdown.getValue()}!`);
        modelDropdown.refresh(langDropdown.getValue());
    }));
}