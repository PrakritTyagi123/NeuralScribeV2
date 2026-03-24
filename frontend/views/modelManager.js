/** Model Manager — list, load, export, delete + training history graphs/table. */
import { createChart } from '../components/chart.js';
import { showToast } from '../components/toast.js';

export async function renderModelManager(container) {
    container.innerHTML = '<div class="view-title">Models</div>';
    container.className = 'view-fit';

    const actionBar = document.createElement('div');
    actionBar.className = 'flex gap-8';
    actionBar.style.cssText = 'flex-shrink:0;margin-bottom:8px;';
    actionBar.innerHTML = `
        <button class="btn" id="mm-refresh">Refresh</button>
        <button class="btn" id="mm-unload">Unload Model</button>
        <button class="btn" id="mm-clear-all" style="margin-left:auto;">Clear All Data</button>
    `;
    container.appendChild(actionBar);

    const loadedInfo = document.createElement('div');
    loadedInfo.className = 'text-sm';
    loadedInfo.style.cssText = 'flex-shrink:0;margin-bottom:8px;';
    container.appendChild(loadedInfo);

    // Top section: table
    const tablePanel = document.createElement('div');
    tablePanel.className = 'panel';
    tablePanel.style.cssText = 'flex-shrink:0;max-height:200px;overflow:hidden;display:flex;flex-direction:column;';
    tablePanel.innerHTML = '<div class="panel-header" style="flex-shrink:0;">Saved Models</div>';
    const tableBody = document.createElement('div');
    tableBody.className = 'panel-body';
    tableBody.style.cssText = 'overflow-y:auto;flex:1;min-height:0;';
    tableBody.innerHTML = '<div class="text-sm text-muted">Loading...</div>';
    tablePanel.appendChild(tableBody);
    container.appendChild(tablePanel);

    // Bottom section: detail + graphs (fills remaining space)
    const detailSection = document.createElement('div');
    detailSection.style.cssText = 'flex:1;display:flex;gap:8px;min-height:0;overflow:hidden;margin-top:8px;';
    container.appendChild(detailSection);

    // Detail info panel (left)
    const detailPanel = document.createElement('div');
    detailPanel.className = 'panel';
    detailPanel.style.cssText = 'width:260px;flex-shrink:0;display:flex;flex-direction:column;overflow:hidden;';
    detailPanel.innerHTML = '<div class="panel-header" style="flex-shrink:0;">Model Details</div>';
    const detailBody = document.createElement('div');
    detailBody.className = 'panel-body';
    detailBody.style.cssText = 'overflow-y:auto;flex:1;min-height:0;';
    detailBody.innerHTML = '<div class="text-sm text-muted">Select a model to see details</div>';
    detailPanel.appendChild(detailBody);
    detailSection.appendChild(detailPanel);

    // Graphs + table panel (right, fills remaining)
    const graphSection = document.createElement('div');
    graphSection.style.cssText = 'flex:1;display:flex;flex-direction:column;gap:8px;min-height:0;overflow:hidden;';
    detailSection.appendChild(graphSection);

    // Charts row
    const chartsRow = document.createElement('div');
    chartsRow.style.cssText = 'display:flex;gap:8px;flex-shrink:0;';
    graphSection.appendChild(chartsRow);

    const lossPanel = document.createElement('div');
    lossPanel.className = 'panel';
    lossPanel.style.cssText = 'flex:1;margin-bottom:0;';
    lossPanel.innerHTML = '<div class="panel-header">Loss History</div>';
    const lossBody = document.createElement('div');
    lossBody.className = 'panel-body';
    lossBody.style.padding = '4px';
    const lossChart = createChart(400, 150);
    lossBody.appendChild(lossChart.element);
    lossPanel.appendChild(lossBody);
    chartsRow.appendChild(lossPanel);

    const accPanel = document.createElement('div');
    accPanel.className = 'panel';
    accPanel.style.cssText = 'flex:1;margin-bottom:0;';
    accPanel.innerHTML = '<div class="panel-header">Accuracy History</div>';
    const accBody = document.createElement('div');
    accBody.className = 'panel-body';
    accBody.style.padding = '4px';
    const accChart = createChart(400, 150);
    accBody.appendChild(accChart.element);
    accPanel.appendChild(accBody);
    chartsRow.appendChild(accPanel);

    // Epoch table
    const histPanel = document.createElement('div');
    histPanel.className = 'panel';
    histPanel.style.cssText = 'flex:1;display:flex;flex-direction:column;overflow:hidden;margin-bottom:0;';
    const histHeader = document.createElement('div');
    histHeader.className = 'panel-header';
    histHeader.style.cssText = 'flex-shrink:0;display:flex;justify-content:space-between;align-items:center;';
    histHeader.innerHTML = '<span>Epoch History</span>';
    const csvBtn = document.createElement('button');
    csvBtn.className = 'btn';
    csvBtn.style.cssText = 'font-size:10px;padding:1px 6px;';
    csvBtn.textContent = '↓ CSV';
    csvBtn.addEventListener('click', () => exportCSV());
    histHeader.appendChild(csvBtn);
    histPanel.appendChild(histHeader);

    const histBody = document.createElement('div');
    histBody.style.cssText = 'flex:1;overflow-y:auto;min-height:0;';
    const histTable = document.createElement('table');
    histTable.className = 'table';
    histTable.style.fontSize = '11px';
    histTable.innerHTML = '<thead><tr><th>Epoch</th><th>Train Loss</th><th>Train Acc</th><th>Val Loss</th><th>Val Acc</th><th>LR</th><th>Time</th><th></th></tr></thead><tbody id="mm-hist-tbody"></tbody>';
    histBody.appendChild(histTable);
    histPanel.appendChild(histBody);
    graphSection.appendChild(histPanel);

    // Placeholder text for graphs
    const noData = '<div class="text-sm text-muted" style="padding:12px;text-align:center;">Select a model to see training history</div>';
    if (!lossChart.element.querySelector('canvas')) lossBody.innerHTML = noData;
    if (!accChart.element.querySelector('canvas')) accBody.innerHTML = noData;

    await refreshLoaded(loadedInfo);
    await loadList(tableBody, detailBody, loadedInfo, lossChart, accChart);

    container.querySelector('#mm-refresh').addEventListener('click', async () => {
        await loadList(tableBody, detailBody, loadedInfo, lossChart, accChart);
        showToast('Refreshed');
    });

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
            await loadList(tableBody, detailBody, loadedInfo, lossChart, accChart);
            await refreshLoaded(loadedInfo);
        } catch (e) { showToast('Failed'); }
    });

    function exportCSV() {
        const tbody = document.getElementById('mm-hist-tbody');
        if (!tbody || !tbody.rows.length) return;
        let csv = 'Epoch,Train Loss,Train Acc,Val Loss,Val Acc,LR,Time,Best\n';
        for (const row of tbody.rows) csv += [...row.cells].map(c => c.textContent.trim()).join(',') + '\n';
        const a = document.createElement('a');
        a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
        a.download = 'training_history.csv'; a.click();
    }
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

async function loadList(tableBody, detailBody, loadedInfo, lossChart, accChart) {
    try {
        const d = await (await fetch('/api/models/list')).json();
        const models = d.models || [];
        if (!models.length) {
            tableBody.innerHTML = '<div class="text-sm text-muted" style="padding:8px;">No models saved. Train a model first.</div>';
            return;
        }
        const table = document.createElement('table');
        table.className = 'table';
        table.style.fontSize = '11px';
        table.innerHTML = `<thead><tr><th>Name</th><th>Date</th><th>Val Acc</th><th>Epoch</th><th>Size</th><th>Actions</th></tr></thead>
        <tbody>${models.map(m => `<tr data-name="${m.name}" style="cursor:pointer;">
            <td>${m.name}${m.is_best?' ★':''}</td>
            <td>${fmtTs(m.timestamp)}</td>
            <td>${m.val_acc!=null?(m.val_acc*100).toFixed(1)+'%':'--'}</td>
            <td>${m.epoch!=null?m.epoch+1:'--'}</td>
            <td>${m.size||'--'}</td>
            <td>
                <button class="btn" data-action="load" data-name="${m.name}" style="font-size:10px;padding:1px 6px;">Load</button>
                <button class="btn" data-action="export" data-name="${m.name}" style="font-size:10px;padding:1px 6px;">ONNX</button>
                <button class="btn" data-action="download" data-name="${m.name}" style="font-size:10px;padding:1px 6px;">↓</button>
                ${!m.is_best?`<button class="btn" data-action="delete" data-name="${m.name}" style="font-size:10px;padding:1px 6px;">×</button>`:''}
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
                    showModelDetail(name, detailBody, lossChart, accChart);
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
                    await loadList(tableBody, detailBody, loadedInfo, lossChart, accChart);
                }
                return;
            }
            const row = e.target.closest('tr[data-name]');
            if (row) {
                // Highlight selected row
                table.querySelectorAll('tr').forEach(r => r.style.background = '');
                row.style.background = 'var(--bg-active)';
                showModelDetail(row.dataset.name, detailBody, lossChart, accChart);
            }
        });

        // Auto-select first model
        if (models.length > 0) {
            showModelDetail(models[0].name, detailBody, lossChart, accChart);
            const firstRow = table.querySelector('tr[data-name]');
            if (firstRow) firstRow.style.background = 'var(--bg-active)';
        }

    } catch (e) { tableBody.innerHTML = '<div class="text-sm text-muted">Cannot reach backend</div>'; }
}

async function showModelDetail(name, detailBody, lossChart, accChart) {
    // Fetch model metadata
    try {
        const m = await (await fetch(`/api/models/metadata/${name}`)).json();
        if (m.error) {
            detailBody.innerHTML = `<div class="text-sm text-muted">${m.error}</div>`;
            return;
        }
        detailBody.innerHTML = `
            <div style="font-size:12px;line-height:1.8;">
                <div><strong>Name:</strong> ${m.name || '--'}</div>
                <div><strong>Saved:</strong> ${fmtTs(m.timestamp)}</div>
                <div><strong>Parameters:</strong> ${m.n_params?.toLocaleString() || '--'}</div>
                <div><strong>Epoch:</strong> ${m.epoch != null ? m.epoch + 1 : '--'}</div>
                <div><strong>Val Accuracy:</strong> ${m.val_acc != null ? (m.val_acc * 100).toFixed(2) + '%' : '--'}</div>
                <div><strong>Val Loss:</strong> ${m.val_loss != null ? m.val_loss.toFixed(4) : '--'}</div>
                <div><strong>Train Loss:</strong> ${m.train_loss != null ? m.train_loss.toFixed(4) : '--'}</div>
                <div><strong>Classes:</strong> ${m.num_classes || '--'}</div>
            </div>
        `;
    } catch (e) {
        detailBody.innerHTML = '<div class="text-sm text-muted">Cannot load metadata</div>';
    }

    // Fetch training history and populate graphs + table
    try {
        const st = await (await fetch('/api/training/status')).json();
        const history = st.history || [];

        if (history.length > 0) {
            const trainLosses = history.map(h => h.train_loss);
            const valLosses = history.map(h => h.val_loss);
            const trainAccs = history.map(h => h.train_acc);
            const valAccs = history.map(h => h.val_acc);

            lossChart.setData([
                { label: 'Train', data: trainLosses },
                { label: 'Val', data: valLosses },
            ]);
            accChart.setData([
                { label: 'Train', data: trainAccs },
                { label: 'Val', data: valAccs },
            ]);

            // Populate epoch table
            const tbody = document.getElementById('mm-hist-tbody');
            if (tbody) {
                tbody.innerHTML = '';
                history.forEach(h => {
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
                });
            }
        } else {
            // No history — clear graphs
            lossChart.setData([]);
            accChart.setData([]);
            const tbody = document.getElementById('mm-hist-tbody');
            if (tbody) tbody.innerHTML = '<tr><td colspan="8" class="text-muted" style="text-align:center;padding:8px;">No training history found</td></tr>';
        }
    } catch (e) {
        console.warn('Could not load training history:', e);
    }
}