/**
 * NeuralScribe v2 — WebSocket client.
 * Auto-reconnects and dispatches events to registered listeners.
 */

import { state } from './state/appState.js';

let ws = null;
let reconnectTimer = null;
const listeners = {};

export function initWebSocket() {
    connect();
}

function connect() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${location.host}/ws`;

    try {
        ws = new WebSocket(url);
    } catch (e) {
        scheduleReconnect();
        return;
    }

    ws.onopen = () => {
        state.wsConnected = true;
        emit('ws_connected', {});
    };

    ws.onclose = () => {
        state.wsConnected = false;
        emit('ws_disconnected', {});
        scheduleReconnect();
    };

    ws.onerror = () => {
        state.wsConnected = false;
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            const type = data.type || 'unknown';

            // Update state
            if (type === 'dataset_progress') state.datasetProgress = data;
            if (type === 'training_batch') state.trainingBatch = data;
            if (type === 'training_epoch') state.trainingEpoch = data;
            if (type === 'training_complete') state.trainingComplete = data;
            if (type === 'gpu_stats') state.gpu = data;
            if (type === 'ping') {
                wsSend({ type: 'pong' });
                return;
            }

            // Dispatch to listeners
            emit(type, data);
            emit('*', data); // wildcard
        } catch (e) {
            // Ignore parse errors
        }
    };
}

function scheduleReconnect() {
    if (reconnectTimer) return;
    reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        connect();
    }, 3000);
}

export function wsSend(data) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(data));
    }
}

export function onWsEvent(type, callback) {
    if (!listeners[type]) listeners[type] = [];
    listeners[type].push(callback);
    return () => {
        listeners[type] = listeners[type].filter(cb => cb !== callback);
    };
}

function emit(type, data) {
    if (listeners[type]) {
        listeners[type].forEach(cb => cb(data));
    }
}