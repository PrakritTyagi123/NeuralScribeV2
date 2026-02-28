/**
 * NeuralScribe v2 — Main entry point.
 * Initializes WebSocket, router, state, and GPU badge polling.
 */

import { initRouter } from './router.js';
import { initWebSocket } from './ws.js';
import { state } from './state/appState.js';

async function init() {
    // Start WebSocket
    initWebSocket();

    // Initialize router (renders first view)
    initRouter();

    // Poll GPU stats for header badge (every 30s — quiet)
    pollGPU();
    setInterval(pollGPU, 30000);

    // Quit button
    document.getElementById('quit-btn').addEventListener('click', async () => {
        if (!confirm('Shut down NeuralScribe server?')) return;
        try {
            await fetch('/api/system/shutdown', { method: 'POST' });
        } catch (e) {
            // Connection will close
        }
        document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;font-family:monospace;font-size:16px;">Server shut down. You can close this tab.</div>';
    });
}

async function pollGPU() {
    try {
        const res = await fetch('/api/system/gpu');
        const data = await res.json();
        state.gpu = data;

        const badge = document.getElementById('gpu-badge');
        const deviceBadge = document.getElementById('device-badge');

        if (data.available) {
            badge.textContent = `GPU: ${data.gpu_util_percent}%`;
            deviceBadge.textContent = data.name || 'CUDA';
        } else {
            badge.textContent = 'GPU: N/A';
            deviceBadge.textContent = 'CPU';
        }
    } catch (e) {
        // Server not ready yet
    }
}

document.addEventListener('DOMContentLoaded', init);