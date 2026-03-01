/** Settings view. */
import { createStatCard } from '../components/statCard.js';
import { showToast } from '../components/toast.js';

export async function renderSetting(container) {
    container.innerHTML = `<div class="view-title">Settings</div>`;

    const grid = document.createElement('div');
    grid.className = 'grid-2';
    container.appendChild(grid);

    // ── System info ──
    const sysPanel = document.createElement('div');
    sysPanel.className = 'panel';
    sysPanel.innerHTML = '<div class="panel-header">System Info</div>';
    const sysBody = document.createElement('div');
    sysBody.className = 'panel-body';
    sysBody.id = 'settings-sys';
    sysBody.innerHTML = '<div class="text-sm text-muted">Loading...</div>';
    sysPanel.appendChild(sysBody);
    grid.appendChild(sysPanel);

    // ── PyTorch info ──
    const torchPanel = document.createElement('div');
    torchPanel.className = 'panel';
    torchPanel.innerHTML = '<div class="panel-header">PyTorch / CUDA</div>';
    const torchBody = document.createElement('div');
    torchBody.className = 'panel-body';
    torchBody.id = 'settings-torch';
    torchBody.innerHTML = '<div class="text-sm text-muted">Loading...</div>';
    torchPanel.appendChild(torchBody);
    grid.appendChild(torchPanel);

    // ── Class registry ──
    const regPanel = document.createElement('div');
    regPanel.className = 'panel mt-16';
    regPanel.innerHTML = '<div class="panel-header">Class Registry</div>';
    const regBody = document.createElement('div');
    regBody.className = 'panel-body';
    regBody.id = 'settings-registry';
    regBody.innerHTML = '<div class="text-sm text-muted">Loading...</div>';
    regPanel.appendChild(regBody);
    container.appendChild(regPanel);

    // Fetch data
    try {
        const [sysRes, torchRes, regRes] = await Promise.all([
            fetch('/api/system/stats'),
            fetch('/api/system/torch'),
            fetch('/api/system/class-registry'),
        ]);

        const sys = await sysRes.json();
        const torch = await torchRes.json();
        const reg = await regRes.json();

        // System
        sysBody.innerHTML = `
            <div class="grid-2">
                ${statHTML('Platform', sys.platform)}
                ${statHTML('Python', sys.python_version)}
                ${statHTML('CPU Cores', sys.cpu_count)}
                ${statHTML('RAM', `${sys.ram_used_gb}/${sys.ram_total_gb} GB (${sys.ram_percent}%)`)}
                ${statHTML('Disk', `${sys.disk_used_gb}/${sys.disk_total_gb} GB (${sys.disk_percent}%)`)}
            </div>
        `;

        // Torch
        torchBody.innerHTML = `
            <div class="grid-2">
                ${statHTML('PyTorch', torch.torch_version)}
                ${statHTML('CUDA', torch.cuda_available ? torch.cuda_version : 'Not available')}
                ${statHTML('cuDNN', torch.cudnn_version || 'N/A')}
                ${statHTML('Devices', torch.device_count)}
            </div>
            ${torch.devices?.length ? torch.devices.map(d => `
                <div class="text-sm mt-8">GPU ${d.index}: ${d.name} (${d.capability.join('.')})</div>
            `).join('') : ''}
        `;

        // Registry
        const cats = reg.category_order || [];
        const classes = reg.classes || [];
        let regHTML = `<div class="text-sm mb-8">${reg.num_classes} classes across ${cats.length} categories</div>`;
        regHTML += '<table class="table"><thead><tr><th>Category</th><th>Count</th><th>All Classes</th></tr></thead><tbody>';
        cats.forEach(cat => {
            const catClasses = classes.filter(c => c.category === cat);
            const all = catClasses.map(c => `<span title="ID ${c.id}: ${c.label}">${c.display}</span>`).join('  ');
            regHTML += `<tr><td>${cat}</td><td>${catClasses.length}</td><td>${all}</td></tr>`;
        });
        regHTML += '</tbody></table>';
        regBody.innerHTML = regHTML;

    } catch (e) {
        sysBody.innerHTML = '<div class="text-sm text-muted">Cannot reach backend</div>';
    }
}

function statHTML(label, value) {
    return `<div class="text-sm"><strong>${label}:</strong> ${value}</div>`;
}