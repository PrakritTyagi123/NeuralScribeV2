/** Dashboard view — system overview. */
import { createStatCard } from '../components/statCard.js';
import { createChart } from '../components/chart.js';
import { state } from '../src/state/appState.js';

export async function renderDashboard(container) {
    container.innerHTML = `<div class="view-title">Dashboard</div>`;

    const cards = document.createElement('div');
    cards.className = 'grid-4';
    cards.id = 'dash-cards';
    container.appendChild(cards);

    const chartsRow = document.createElement('div');
    chartsRow.className = 'grid-2 mt-16';
    container.appendChild(chartsRow);

    // Mini charts
    const lossChart = createChart(400, 180);
    const accChart = createChart(400, 180);

    const lossPanel = document.createElement('div');
    lossPanel.className = 'panel';
    lossPanel.innerHTML = '<div class="panel-header">Loss (recent)</div>';
    const lossBody = document.createElement('div');
    lossBody.className = 'panel-body';
    lossBody.appendChild(lossChart.element);
    lossPanel.appendChild(lossBody);
    chartsRow.appendChild(lossPanel);

    const accPanel = document.createElement('div');
    accPanel.className = 'panel';
    accPanel.innerHTML = '<div class="panel-header">Accuracy (recent)</div>';
    const accBody = document.createElement('div');
    accBody.className = 'panel-body';
    accBody.appendChild(accChart.element);
    accPanel.appendChild(accBody);
    chartsRow.appendChild(accPanel);

    // Fetch data
    try {
        const res = await fetch('/api/system/dashboard');
        const data = await res.json();

        cards.innerHTML = '';
        cards.appendChild(createStatCard(
            'GPU',
            data.gpu.available ? `${data.gpu.gpu_util_percent}%` : 'N/A',
            data.gpu.available ? `${data.gpu.memory_used_mb}/${data.gpu.memory_total_mb} MB` : 'CPU mode'
        ));
        cards.appendChild(createStatCard(
            'Dataset',
            data.dataset.prepared ? 'Ready' : 'Not prepared',
            data.dataset.prepared ? data.dataset.cache_size : 'Run Data Prep'
        ));
        cards.appendChild(createStatCard(
            'Best Model',
            data.model.best_accuracy != null ? `${(data.model.best_accuracy * 100).toFixed(1)}%` : '--',
            data.model.best_epoch != null ? `Epoch ${data.model.best_epoch}` : 'No model yet'
        ));
        cards.appendChild(createStatCard(
            'Training',
            data.training.is_training ? 'Running' : 'Idle',
            data.training.is_training ? `Epoch ${data.training.current_epoch}` : ''
        ));

        // Load history for charts
        const histRes = await fetch('/api/training/history');
        const histData = await histRes.json();
        const history = histData.history || [];

        if (history.length > 0) {
            lossChart.setData([
                { label: 'Train', data: history.map(h => h.train_loss) },
                { label: 'Val', data: history.map(h => h.val_loss) },
            ]);
            accChart.setData([
                { label: 'Train', data: history.map(h => h.train_acc) },
                { label: 'Val', data: history.map(h => h.val_acc) },
            ]);
        }
    } catch (e) {
        cards.innerHTML = '';
        cards.appendChild(createStatCard('Status', 'Offline', 'Cannot reach backend'));
    }
}