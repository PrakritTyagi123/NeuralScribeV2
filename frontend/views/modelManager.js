/** Model Manager — list, load, unload, export, delete, clear all. */
import { showToast } from '../components/toast.js';

export async function renderModelManager(container) {
    container.innerHTML = `<div class="view-title">Model Manager</div>`;

    // ── Action bar ──
    const actionBar = document.createElement('div');
    actionBar.className = 'flex gap-8 mb-16';
    actionBar.innerHTML = `
        <button class="btn" id="mm-unload">Unload Model</button>
        <button class="btn" id="mm-clear-all" style="margin-left:auto;">Clear All Data</button>
    `;
    container.appendChild(actionBar);

    // ── Loaded model info ──
    const loadedInfo = document.createElement('div');
    loadedInfo.className = 'text-sm mb-16';
    loadedInfo.id = 'mm-loaded';
    container.appendChild(loadedInfo);

    // ── Models table ──
    const tablePanel = document.createElement('div');
    tablePanel.className = 'panel';
    tablePanel.innerHTML = `<div class="panel-header">Saved Models</div>`;
    const tableBody = document.createElement('div');
    tableBody.className = 'panel-body';
    tableBody.id = 'model-table-body';
    tableBody.innerHTML = '<div class="text-sm text-muted">Loading...</div>';
    tablePanel.appendChild(tableBody);
    container.appendChild(tablePanel);

    // ── Details ──
    const detailPanel = document.createElement('div');
    detailPanel.className = 'panel mt-16';
    detailPanel.innerHTML = `<div class="panel-header">Model Details</div>`;
    const detailBody = document.createElement('div');
    detailBody.className = 'panel-body';
    detailBody.id = 'model-detail';
    detailBody.innerHTML = '<div class="text-sm text-muted">Select a model to see details</div>';
    detailPanel.appendChild(detailBody);
    container.appendChild(detailPanel);

    // ── Load initial state ──
    await refreshLoadedInfo(loadedInfo);
    await loadModelList(tableBody, detailBody, loadedInfo);

    // ── Unload button ──
    container.querySelector('#mm-unload').addEventListener('click', async () => {
        try {
            const res = await fetch('/api/models/unload', { method: 'POST' });
            const data = await res.json();
            showToast(data.message || data.error || 'Model unloaded');
            await refreshLoadedInfo(loadedInfo);
        } catch (e) {
            showToast('Failed to unload');
        }
    });

    // ── Clear all data ──
    container.querySelector('#mm-clear-all').addEventListener('click', async () => {
        if (!confirm('This will delete ALL saved models and cached dataset. Are you sure?')) return;
        if (!confirm('Really? This cannot be undone.')) return;
        try {
            const res = await fetch('/api/system/clear-all', { method: 'POST' });
            const data = await res.json();
            showToast(data.message || 'All data cleared');
            await loadModelList(tableBody, detailBody, loadedInfo);
            await refreshLoadedInfo(loadedInfo);
        } catch (e) {
            showToast('Failed to clear');
        }
    });
}

async function refreshLoadedInfo(el) {
    try {
        const res = await fetch('/api/inference/status');
        const data = await res.json();
        el.innerHTML = data.ready
            ? `<strong>Loaded:</strong> ${data.loaded_model || 'unknown'} — ready for inference`
            : '<span class="text-muted">No model loaded</span>';
    } catch (e) {
        el.innerHTML = '<span class="text-muted">Cannot reach backend</span>';
    }
}

function formatTimestamp(ts) {
    if (!ts) return '--';
    if (ts.includes('_') && ts.length === 15) {
        const y = ts.slice(0, 4), mo = ts.slice(4, 6), d = ts.slice(6, 8);
        const h = ts.slice(9, 11), mi = ts.slice(11, 13), s = ts.slice(13, 15);
        return `${y}-${mo}-${d} ${h}:${mi}:${s}`;
    }
    try { return new Date(ts).toLocaleString(); } catch (e) { return ts; }
}

async function loadModelList(tableBody, detailBody, loadedInfo) {
    try {
        const res = await fetch('/api/models/list');
        const data = await res.json();
        const models = data.models || [];

        if (models.length === 0) {
            tableBody.innerHTML = '<div class="text-sm text-muted">No models saved yet. Train a model first.</div>';
            return;
        }

        const table = document.createElement('table');
        table.className = 'table';
        table.innerHTML = `
            <thead><tr>
                <th>Name</th><th>Date/Time</th><th>Val Acc</th><th>Epoch</th><th>Size</th><th>Actions</th>
            </tr></thead>
            <tbody>
                ${models.map(m => `
                    <tr data-name="${m.name}" style="cursor:pointer;">
                        <td>${m.name}${m.is_best ? ' ★' : ''}</td>
                        <td>${formatTimestamp(m.timestamp)}</td>
                        <td>${m.val_acc != null ? (m.val_acc * 100).toFixed(1) + '%' : '--'}</td>
                        <td>${m.epoch != null ? m.epoch + 1 : '--'}</td>
                        <td>${m.size || '--'}</td>
                        <td>
                            <button class="btn" data-action="load" data-name="${m.name}">Load</button>
                            <button class="btn" data-action="export" data-name="${m.name}">ONNX</button>
                            <button class="btn" data-action="download" data-name="${m.name}">↓</button>
                            ${!m.is_best ? `<button class="btn" data-action="delete" data-name="${m.name}">×</button>` : ''}
                        </td>
                    </tr>
                `).join('')}
            </tbody>
        `;

        tableBody.innerHTML = '';
        tableBody.appendChild(table);

        table.addEventListener('click', async (e) => {
            const btn = e.target.closest('[data-action]');
            if (btn) {
                const action = btn.dataset.action;
                const name = btn.dataset.name;
                if (action === 'load') {
                    await fetch('/api/models/load', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ name }),
                    });
                    showToast(`Model "${name}" loaded`);
                    await refreshLoadedInfo(loadedInfo);
                    const metaRes = await fetch(`/api/models/metadata/${name}`);
                    renderModelDetail(detailBody, await metaRes.json());
                }
                if (action === 'export') {
                    btn.disabled = true; btn.textContent = '...';
                    const res = await fetch(`/api/models/export-onnx/${name}`, { method: 'POST' });
                    showToast((await res.json()).error || `Exported to ONNX`);
                    btn.disabled = false; btn.textContent = 'ONNX';
                }
                if (action === 'download') window.open(`/api/models/download/${name}`, '_blank');
                if (action === 'delete') {
                    if (!confirm(`Delete "${name}"?`)) return;
                    await fetch(`/api/models/${name}`, { method: 'DELETE' });
                    showToast(`Deleted ${name}`);
                    await loadModelList(tableBody, detailBody, loadedInfo);
                }
                return;
            }
            const row = e.target.closest('tr[data-name]');
            if (row) {
                try {
                    const meta = await (await fetch(`/api/models/metadata/${row.dataset.name}`)).json();
                    renderModelDetail(detailBody, meta);
                } catch (err) {}
            }
        });
    } catch (e) {
        tableBody.innerHTML = '<div class="text-sm text-muted">Cannot reach backend</div>';
    }
}

function renderModelDetail(detailBody, meta) {
    if (meta.error) { detailBody.innerHTML = `<div class="text-sm text-muted">${meta.error}</div>`; return; }
    detailBody.innerHTML = `
        <div class="grid-3">
            <div><strong>Name:</strong> ${meta.name || '--'}</div>
            <div><strong>Saved:</strong> ${formatTimestamp(meta.timestamp)}</div>
            <div><strong>Params:</strong> ${meta.n_params?.toLocaleString() || '--'}</div>
            <div><strong>Epoch:</strong> ${meta.epoch != null ? meta.epoch + 1 : '--'}</div>
            <div><strong>Val Acc:</strong> ${meta.val_acc != null ? (meta.val_acc * 100).toFixed(2) + '%' : '--'}</div>
            <div><strong>Val Loss:</strong> ${meta.val_loss != null ? meta.val_loss.toFixed(4) : '--'}</div>
            <div><strong>Train Loss:</strong> ${meta.train_loss != null ? meta.train_loss.toFixed(4) : '--'}</div>
            <div><strong>Classes:</strong> ${meta.num_classes || '--'}</div>
            <div><strong>Device:</strong> ${meta.device || '--'}</div>
        </div>
    `;
}