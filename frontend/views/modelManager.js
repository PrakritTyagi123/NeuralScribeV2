/** Model Manager — list, load, unload, export, delete. */
import { showToast } from '../components/toast.js';

export async function renderModelManager(container) {
    container.innerHTML = '<div class="view-title">Models</div>';

    const actionBar = document.createElement('div');
    actionBar.className = 'flex gap-8 mb-16';
    actionBar.innerHTML = `
        <button class="btn" id="mm-unload">Unload Model</button>
        <button class="btn" id="mm-clear-all" style="margin-left:auto;">Clear All Data</button>
    `;
    container.appendChild(actionBar);

    const loadedInfo = document.createElement('div');
    loadedInfo.className = 'text-sm mb-16';
    container.appendChild(loadedInfo);

    const tablePanel = document.createElement('div');
    tablePanel.className = 'panel';
    tablePanel.innerHTML = '<div class="panel-header">Saved Models</div>';
    const tableBody = document.createElement('div');
    tableBody.className = 'panel-body';
    tableBody.innerHTML = '<div class="text-sm text-muted">Loading...</div>';
    tablePanel.appendChild(tableBody);
    container.appendChild(tablePanel);

    const detailPanel = document.createElement('div');
    detailPanel.className = 'panel mt-16';
    detailPanel.innerHTML = '<div class="panel-header">Model Details</div>';
    const detailBody = document.createElement('div');
    detailBody.className = 'panel-body';
    detailBody.innerHTML = '<div class="text-sm text-muted">Select a model to see details</div>';
    detailPanel.appendChild(detailBody);
    container.appendChild(detailPanel);

    await refreshLoaded(loadedInfo);
    await loadList(tableBody, detailBody, loadedInfo);

    container.querySelector('#mm-unload').addEventListener('click', async () => {
        try {
            const r = await fetch('/api/models/unload', { method: 'POST' });
            showToast((await r.json()).message || 'Unloaded');
            await refreshLoaded(loadedInfo);
        } catch (e) { showToast('Failed'); }
    });

    container.querySelector('#mm-clear-all').addEventListener('click', async () => {
        if (!confirm('Delete ALL models and cached dataset?')) return;
        if (!confirm('Really? Cannot be undone.')) return;
        try {
            const r = await fetch('/api/system/clear-all', { method: 'POST' });
            showToast((await r.json()).message || 'Cleared');
            await loadList(tableBody, detailBody, loadedInfo);
            await refreshLoaded(loadedInfo);
        } catch (e) { showToast('Failed'); }
    });
}

async function refreshLoaded(el) {
    try {
        const d = await (await fetch('/api/inference/status')).json();
        el.innerHTML = d.ready
            ? `<strong>Loaded:</strong> ${d.loaded_model || 'unknown'} — ready for inference`
            : '<span class="text-muted">No model loaded</span>';
    } catch (e) { el.innerHTML = '<span class="text-muted">Cannot reach backend</span>'; }
}

function fmtTs(ts) {
    if (!ts) return '--';
    if (ts.includes('_') && ts.length === 15) {
        return `${ts.slice(0,4)}-${ts.slice(4,6)}-${ts.slice(6,8)} ${ts.slice(9,11)}:${ts.slice(11,13)}:${ts.slice(13,15)}`;
    }
    try { return new Date(ts).toLocaleString(); } catch { return ts; }
}

async function loadList(tableBody, detailBody, loadedInfo) {
    try {
        const d = await (await fetch('/api/models/list')).json();
        const models = d.models || [];
        if (!models.length) {
            tableBody.innerHTML = '<div class="text-sm text-muted">No models saved. Train a model first.</div>';
            return;
        }
        const table = document.createElement('table');
        table.className = 'table';
        table.innerHTML = `<thead><tr><th>Name</th><th>Date</th><th>Val Acc</th><th>Epoch</th><th>Size</th><th>Actions</th></tr></thead>
        <tbody>${models.map(m => `<tr data-name="${m.name}" style="cursor:pointer;">
            <td>${m.name}${m.is_best?' ★':''}</td>
            <td>${fmtTs(m.timestamp)}</td>
            <td>${m.val_acc!=null?(m.val_acc*100).toFixed(1)+'%':'--'}</td>
            <td>${m.epoch!=null?m.epoch+1:'--'}</td>
            <td>${m.size||'--'}</td>
            <td>
                <button class="btn" data-action="load" data-name="${m.name}">Load</button>
                <button class="btn" data-action="export" data-name="${m.name}">ONNX</button>
                <button class="btn" data-action="download" data-name="${m.name}">↓</button>
                ${!m.is_best?`<button class="btn" data-action="delete" data-name="${m.name}">×</button>`:''}
            </td></tr>`).join('')}</tbody>`;
        tableBody.innerHTML = '';
        tableBody.appendChild(table);

        table.addEventListener('click', async (e) => {
            const btn = e.target.closest('[data-action]');
            if (btn) {
                const { action, name } = btn.dataset;
                if (action === 'load') {
                    await fetch('/api/models/load', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({name}) });
                    showToast(`"${name}" loaded`);
                    await refreshLoaded(loadedInfo);
                    const m = await (await fetch(`/api/models/metadata/${name}`)).json();
                    showDetail(detailBody, m);
                }
                if (action === 'export') {
                    btn.disabled = true; btn.textContent = '...';
                    const r = await fetch(`/api/models/export-onnx/${name}`, { method: 'POST' });
                    showToast((await r.json()).error || 'Exported');
                    btn.disabled = false; btn.textContent = 'ONNX';
                }
                if (action === 'download') window.open(`/api/models/download/${name}`, '_blank');
                if (action === 'delete') {
                    if (!confirm(`Delete "${name}"?`)) return;
                    await fetch(`/api/models/${name}`, { method: 'DELETE' });
                    showToast(`Deleted ${name}`);
                    await loadList(tableBody, detailBody, loadedInfo);
                }
                return;
            }
            const row = e.target.closest('tr[data-name]');
            if (row) {
                try { showDetail(detailBody, await (await fetch(`/api/models/metadata/${row.dataset.name}`)).json()); } catch {}
            }
        });
    } catch (e) { tableBody.innerHTML = '<div class="text-sm text-muted">Cannot reach backend</div>'; }
}

function showDetail(el, m) {
    if (m.error) { el.innerHTML = `<div class="text-sm text-muted">${m.error}</div>`; return; }
    el.innerHTML = `<div class="grid-3">
        <div><strong>Name:</strong> ${m.name||'--'}</div>
        <div><strong>Saved:</strong> ${fmtTs(m.timestamp)}</div>
        <div><strong>Params:</strong> ${m.n_params?.toLocaleString()||'--'}</div>
        <div><strong>Epoch:</strong> ${m.epoch!=null?m.epoch+1:'--'}</div>
        <div><strong>Val Acc:</strong> ${m.val_acc!=null?(m.val_acc*100).toFixed(2)+'%':'--'}</div>
        <div><strong>Val Loss:</strong> ${m.val_loss!=null?m.val_loss.toFixed(4):'--'}</div>
        <div><strong>Train Loss:</strong> ${m.train_loss!=null?m.train_loss.toFixed(4):'--'}</div>
        <div><strong>Classes:</strong> ${m.num_classes||'--'}</div>
    </div>`;
}
