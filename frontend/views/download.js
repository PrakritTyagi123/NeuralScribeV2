/**
 * NeuralScribe v2 — Download Page
 * Downloads EMNIST dataset. One button, one progress bar.
 */
import { createProgressBar, updateProgressBar } from '../components/progressBar.js';
import { onWsEvent } from '../src/ws.js';
import { showToast } from '../components/toast.js';

let unsubs = [];

export async function renderDownload(container) {
    unsubs.forEach(u => u()); unsubs = [];
    container.innerHTML = '';
    container.className = 'view-fit';

    const title = document.createElement('div');
    title.className = 'view-title';
    title.textContent = 'Download Dataset';
    container.appendChild(title);

    const desc = document.createElement('div');
    desc.className = 'text-sm text-muted';
    desc.style.marginBottom = '12px';
    desc.textContent = 'Step 1: Download the EMNIST handwriting dataset. This is a one-time download (~550 MB).';
    container.appendChild(desc);

    const card = document.createElement('div');
    card.className = 'panel';
    card.innerHTML = `
        <div class="panel-header" style="display:flex;justify-content:space-between;align-items:center;">
            <span><strong>EMNIST Balanced</strong> — 47 classes (0-9, A-Z, 11 lowercase)</span>
            <span class="badge" id="dl-badge">Checking...</span>
        </div>
        <div class="panel-body" style="padding:12px;">
            <div class="text-sm text-muted" style="margin-bottom:10px;">
                ~100,800 handwritten character images collected by NIST. Downloads via torchvision (~550 MB).
            </div>
            <div id="dl-bar-wrap" style="margin-bottom:10px;"></div>
            <div style="display:flex;gap:8px;align-items:center;">
                <button class="btn btn-primary" id="dl-btn" style="width:160px;">↓ Download EMNIST</button>
                <span class="text-sm text-muted" id="dl-detail"></span>
            </div>
        </div>
    `;
    container.appendChild(card);

    const bar = createProgressBar(0, 'Click Download to start');
    card.querySelector('#dl-bar-wrap').appendChild(bar);

    const badge = card.querySelector('#dl-badge');
    const detail = card.querySelector('#dl-detail');
    const btn = card.querySelector('#dl-btn');

    // Info
    const info = document.createElement('div');
    info.className = 'panel';
    info.style.marginTop = '12px';
    info.innerHTML = `
        <div class="panel-header">What happens next</div>
        <div class="panel-body" style="padding:12px;font-size:12px;line-height:1.6;color:var(--muted);">
            After downloading, go to <strong>Data Prep</strong> to preprocess and augment the dataset.
            You can configure how many augmented samples to generate.
            Then go to <strong>Training</strong> to train the neural network.
        </div>
    `;
    container.appendChild(info);

    // Check status
    async function refresh() {
        try {
            const st = await (await fetch('/api/dataset/download-status')).json();
            if (st.downloaded) {
                badge.textContent = '✓ Downloaded'; badge.style.color = '#16a34a';
                btn.disabled = true; btn.textContent = '✓ Downloaded';
                updateProgressBar(bar, 100, 'EMNIST ready — go to Data Prep');
                detail.textContent = st.message;
            } else {
                badge.textContent = '○ Not downloaded'; badge.style.color = '#d97706';
                btn.disabled = false; btn.textContent = '↓ Download EMNIST';
                updateProgressBar(bar, 0, 'Click Download to start');
            }
        } catch (e) {
            badge.textContent = '? Error'; detail.textContent = 'Cannot reach backend';
        }
    }
    await refresh();

    // Download button
    btn.addEventListener('click', async () => {
        btn.disabled = true; btn.textContent = 'Downloading...';
        badge.textContent = '↓ Downloading'; badge.style.color = '#2563eb';
        updateProgressBar(bar, 1, 'Starting...');
        try {
            await fetch('/api/dataset/download', { method: 'POST' });
        } catch (e) {
            btn.disabled = false; btn.textContent = '↓ Retry';
            updateProgressBar(bar, 0, 'Error: ' + e.message);
        }
    });

    // WS progress
    unsubs.push(onWsEvent('download_progress', (d) => {
        updateProgressBar(bar, d.pct || 0, d.message || '');
        detail.textContent = d.message || '';
    }));

    unsubs.push(onWsEvent('download_complete', (d) => {
        if (d.status === 'success' || d.status === 'exists') {
            updateProgressBar(bar, 100, 'Download complete — go to Data Prep');
            btn.disabled = true; btn.textContent = '✓ Downloaded';
            badge.textContent = '✓ Downloaded'; badge.style.color = '#16a34a';
            showToast('EMNIST downloaded! Go to Data Prep.');
        } else {
            updateProgressBar(bar, 0, 'Error: ' + (d.message || 'Failed'));
            btn.disabled = false; btn.textContent = '↓ Retry';
            badge.textContent = '✗ Error'; badge.style.color = '#dc2626';
        }
        detail.textContent = d.message || '';
    }));
}
