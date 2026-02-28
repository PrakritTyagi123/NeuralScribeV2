/** Training view. */
import { createChart } from '../components/chart.js';
import { createStatCard } from '../components/statCard.js';
import { createLogConsole, appendLog } from '../components/logConsole.js';
import { createProgressBar, updateProgressBar } from '../components/progressBar.js';
import { onWsEvent } from '../src/ws.js';
import { showToast } from '../components/toast.js';

let unsubscribers = [];

export async function renderTraining(container) {
    unsubscribers.forEach(u => u());
    unsubscribers = [];

    container.innerHTML = `<div class="view-title">Training</div>`;

    // ── Status bar ──
    const statusBar = document.createElement('div');
    statusBar.className = 'flex-between mb-16';
    statusBar.id = 'train-status-bar';
    statusBar.innerHTML = '<span class="text-sm text-muted">Loading...</span>';
    container.appendChild(statusBar);

    // ── Progress ──
    const epochProgress = createProgressBar(0, 'Idle');
    container.appendChild(epochProgress);

    // ── Charts + controls ──
    const mainGrid = document.createElement('div');
    mainGrid.className = 'grid-2 mt-16';
    container.appendChild(mainGrid);

    // Loss chart
    const lossPanel = document.createElement('div');
    lossPanel.className = 'panel';
    lossPanel.innerHTML = '<div class="panel-header">Loss</div>';
    const lossBody = document.createElement('div');
    lossBody.className = 'panel-body';
    const lossChart = createChart(400, 200);
    lossBody.appendChild(lossChart.element);
    lossPanel.appendChild(lossBody);
    mainGrid.appendChild(lossPanel);

    // Right: accuracy chart + controls
    const rightCol = document.createElement('div');

    const accPanel = document.createElement('div');
    accPanel.className = 'panel';
    accPanel.innerHTML = '<div class="panel-header">Accuracy</div>';
    const accBody = document.createElement('div');
    accBody.className = 'panel-body';
    const accChart = createChart(400, 200);
    accBody.appendChild(accChart.element);
    accPanel.appendChild(accBody);
    rightCol.appendChild(accPanel);

    // Controls
    const ctrlPanel = document.createElement('div');
    ctrlPanel.className = 'panel mt-16';
    ctrlPanel.innerHTML = `
        <div class="panel-header">Controls</div>
        <div class="panel-body">
            <div class="grid-2 mb-8">
                <div class="form-group">
                    <label>Epochs</label>
                    <input type="number" id="train-epochs" value="100">
                </div>
                <div class="form-group">
                    <label>Batch Size</label>
                    <input type="number" id="train-bs" value="256">
                </div>
            </div>
            <div class="btn-group">
                <button class="btn btn-primary" id="train-start">Start</button>
                <button class="btn" id="train-pause" disabled>Pause</button>
                <button class="btn" id="train-resume" disabled>Resume</button>
                <button class="btn" id="train-stop" disabled>Stop</button>
            </div>
        </div>
    `;
    rightCol.appendChild(ctrlPanel);
    mainGrid.appendChild(rightCol);

    // ── Log ──
    const logPanel = document.createElement('div');
    logPanel.className = 'panel mt-16';
    logPanel.innerHTML = '<div class="panel-header">Live Logs</div>';
    const logConsole = createLogConsole();
    logPanel.appendChild(logConsole);
    container.appendChild(logPanel);

    // ── Load history ──
    const trainLosses = [];
    const valLosses = [];
    const trainAccs = [];
    const valAccs = [];

    try {
        const res = await fetch('/api/training/status');
        const status = await res.json();

        statusBar.innerHTML = `
            <span>Epoch ${status.current_epoch} | Best: ${(status.best_val_acc * 100).toFixed(1)}% | ${status.device}</span>
            <span class="badge">${status.is_training ? 'TRAINING' : 'IDLE'}</span>
        `;

        (status.history || []).forEach(h => {
            trainLosses.push(h.train_loss);
            valLosses.push(h.val_loss);
            trainAccs.push(h.train_acc);
            valAccs.push(h.val_acc);
        });

        if (trainLosses.length > 0) {
            lossChart.setData([{ label: 'Train', data: trainLosses }, { label: 'Val', data: valLosses }]);
            accChart.setData([{ label: 'Train', data: trainAccs }, { label: 'Val', data: valAccs }]);
        }

        // Load config
        const cfgRes = await fetch('/api/training/config');
        const cfg = await cfgRes.json();
        const epochsInput = container.querySelector('#train-epochs');
        const bsInput = container.querySelector('#train-bs');
        if (epochsInput) epochsInput.value = cfg.training?.epochs || 100;
        if (bsInput) bsInput.value = cfg.training?.batch_size || 256;
    } catch (e) {
        statusBar.innerHTML = '<span class="text-muted">Cannot reach backend</span>';
    }

    // ── Button handlers ──
    const startBtn = container.querySelector('#train-start');
    const pauseBtn = container.querySelector('#train-pause');
    const resumeBtn = container.querySelector('#train-resume');
    const stopBtn = container.querySelector('#train-stop');

    function setTrainingUI(running) {
        startBtn.disabled = running;
        pauseBtn.disabled = !running;
        stopBtn.disabled = !running;
    }

    startBtn.addEventListener('click', async () => {
        const epochs = parseInt(container.querySelector('#train-epochs').value) || 100;
        const bs = parseInt(container.querySelector('#train-bs').value) || 256;

        const res = await fetch('/api/training/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ overrides: { 'training.epochs': epochs, 'training.batch_size': bs } }),
        });
        const data = await res.json();
        appendLog(logConsole, data.message || JSON.stringify(data));
        if (!data.error) setTrainingUI(true);
    });

    pauseBtn.addEventListener('click', async () => {
        await fetch('/api/training/pause', { method: 'POST' });
        appendLog(logConsole, 'Paused');
        resumeBtn.disabled = false;
    });

    resumeBtn.addEventListener('click', async () => {
        await fetch('/api/training/resume', { method: 'POST' });
        appendLog(logConsole, 'Resumed');
        resumeBtn.disabled = true;
    });

    stopBtn.addEventListener('click', async () => {
        await fetch('/api/training/stop', { method: 'POST' });
        appendLog(logConsole, 'Stop requested');
    });

    // ── WS events ──
    unsubscribers.push(onWsEvent('training_epoch', (data) => {
        trainLosses.push(data.train_loss);
        valLosses.push(data.val_loss);
        trainAccs.push(data.train_acc);
        valAccs.push(data.val_acc);
        lossChart.setData([{ label: 'Train', data: trainLosses }, { label: 'Val', data: valLosses }]);
        accChart.setData([{ label: 'Train', data: trainAccs }, { label: 'Val', data: valAccs }]);

        const pct = ((data.epoch + 1) / data.total_epochs * 100);
        updateProgressBar(epochProgress, pct,
            `Epoch ${data.epoch}/${data.total_epochs} | val_acc=${(data.val_acc * 100).toFixed(1)}%`);

        statusBar.innerHTML = `
            <span>Epoch ${data.epoch}/${data.total_epochs} | Best: ${(data.best_val_acc * 100).toFixed(1)}% | LR: ${data.lr}</span>
            <span class="badge">TRAINING${data.is_best ? ' ★' : ''}</span>
        `;

        appendLog(logConsole,
            `E${data.epoch} train=${data.train_acc.toFixed(4)} val=${data.val_acc.toFixed(4)}${data.is_best ? ' ★' : ''}`);
    }));

    unsubscribers.push(onWsEvent('training_batch', (data) => {
        const batchPct = data.batch / data.total_batches * 100;
        updateProgressBar(epochProgress, batchPct,
            `Epoch ${data.epoch} — batch ${data.batch}/${data.total_batches} loss=${data.loss}`);
    }));

    unsubscribers.push(onWsEvent('training_complete', (data) => {
        setTrainingUI(false);
        updateProgressBar(epochProgress, 100, 'Complete');
        appendLog(logConsole, `Training ${data.status}! Best: ${(data.best_val_acc * 100).toFixed(1)}%`);
        showToast('Training complete!');
    }));
}