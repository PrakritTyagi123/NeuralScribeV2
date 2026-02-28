/** Model Manager view. */
import { showModal } from '../components/modal.js';
import { showToast } from '../components/toast.js';

export async function renderModelManager(container) {
    container.innerHTML = `<div class="view-title">Model Manager</div>`;

    const tablePanel = document.createElement('div');
    tablePanel.className = 'panel';
    tablePanel.innerHTML = `<div class="panel-header">Saved Models</div>`;
    const tableBody = document.createElement('div');
    tableBody.className = 'panel-body';
    tableBody.id = 'model-table-body';
    tableBody.innerHTML = '<div class="text-sm text-muted">Loading...</div>';
    tablePanel.appendChild(tableBody);
    container.appendChild(tablePanel);

    // Details panel
    const detailPanel = document.createElement('div');
    detailPanel.className = 'panel mt-16';
    detailPanel.innerHTML = `<div class="panel-header">Model Details</div>`;
    const detailBody = document.createElement('div');
    detailBody.className = 'panel-body';
    detailBody.id = 'model-detail';
    detailBody.innerHTML = '<div class="text-sm text-muted">Select a model above</div>';
    detailPanel.appendChild(detailBody);
    container.appendChild(detailPanel);

    await loadModelList(tableBody, detailBody);
}

async function loadModelList(tableBody, detailBody) {
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
            <thead>
                <tr>
                    <th>Name</th><th>Acc</th><th>Epoch</th><th>Size</th><th>Actions</th>
                </tr>
            </thead>
            <tbody>
                ${models.map(m => `
                    <tr data-name="${m.name}">
                        <td>${m.name}${m.is_best ? ' ★' : ''}</td>
                        <td>${m.val_acc != null ? (m.val_acc * 100).toFixed(1) + '%' : '--'}</td>
                        <td>${m.epoch ?? '--'}</td>
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

        // Action handlers
        table.addEventListener('click', async (e) => {
            const btn = e.target.closest('[data-action]');
            if (!btn) return;

            const action = btn.dataset.action;
            const name = btn.dataset.name;

            if (action === 'load') {
                const res = await fetch('/api/models/load', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name }),
                });
                const data = await res.json();
                showToast(data.error || `Loaded ${name}`);

                // Show details
                const metaRes = await fetch(`/api/models/metadata/${name}`);
                const meta = await metaRes.json();
                renderModelDetail(detailBody, meta);
            }

            if (action === 'export') {
                btn.disabled = true;
                btn.textContent = '...';
                const res = await fetch(`/api/models/export-onnx/${name}`, { method: 'POST' });
                const data = await res.json();
                showToast(data.error || `Exported: ${data.name}`);
                btn.disabled = false;
                btn.textContent = 'ONNX';
            }

            if (action === 'download') {
                window.open(`/api/models/download/${name}`, '_blank');
            }

            if (action === 'delete') {
                if (!confirm(`Delete ${name}?`)) return;
                await fetch(`/api/models/${name}`, { method: 'DELETE' });
                showToast(`Deleted ${name}`);
                await loadModelList(tableBody, detailBody);
            }
        });

        // Row click → show details
        table.querySelectorAll('tbody tr').forEach(row => {
            row.style.cursor = 'pointer';
            row.addEventListener('click', async (e) => {
                if (e.target.closest('[data-action]')) return;
                const name = row.dataset.name;
                try {
                    const metaRes = await fetch(`/api/models/metadata/${name}`);
                    const meta = await metaRes.json();
                    renderModelDetail(detailBody, meta);
                } catch (err) {
                    detailBody.innerHTML = '<div class="text-sm text-muted">No metadata available</div>';
                }
            });
        });

    } catch (e) {
        tableBody.innerHTML = '<div class="text-sm text-muted">Cannot reach backend</div>';
    }
}

function renderModelDetail(detailBody, meta) {
    if (meta.error) {
        detailBody.innerHTML = `<div class="text-sm text-muted">${meta.error}</div>`;
        return;
    }

    detailBody.innerHTML = `
        <div class="grid-3">
            <div><strong>Name:</strong> ${meta.name || '--'}</div>
            <div><strong>Epoch:</strong> ${meta.epoch ?? '--'}</div>
            <div><strong>Params:</strong> ${meta.n_params?.toLocaleString() || '--'}</div>
            <div><strong>Val Acc:</strong> ${meta.val_acc != null ? (meta.val_acc * 100).toFixed(2) + '%' : '--'}</div>
            <div><strong>Train Loss:</strong> ${meta.train_loss ?? '--'}</div>
            <div><strong>Val Loss:</strong> ${meta.val_loss ?? '--'}</div>
            <div><strong>Classes:</strong> ${meta.num_classes || '--'}</div>
            <div><strong>Timestamp:</strong> ${meta.timestamp || '--'}</div>
        </div>
    `;
}