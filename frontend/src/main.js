import { initRouter } from './router.js';
import { initWebSocket } from './ws.js';
import { state } from './state/appState.js';

async function init() {
    initWebSocket();
    await initRouter();
    pollGPU();
    setInterval(pollGPU, 30000);
    document.getElementById('quit-btn').addEventListener('click', async () => {
        if (!confirm('Shut down NeuralScribe server?')) return;
        try { await fetch('/api/system/shutdown', { method: 'POST' }); } catch (e) {}
        document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;font-family:monospace;">Server shut down.</div>';
    });
}

async function pollGPU() {
    try {
        const res = await fetch('/api/system/gpu');
        const data = await res.json();
        state.gpu = data;
        document.getElementById('gpu-badge').textContent = data.available ? `GPU: ${data.gpu_util_percent}%` : 'GPU: N/A';
        document.getElementById('device-badge').textContent = data.available ? (data.name || 'CUDA') : 'CPU';
    } catch (e) {}
}

document.addEventListener('DOMContentLoaded', init);
