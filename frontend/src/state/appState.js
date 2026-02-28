/**
 * NeuralScribe v2 — Application state.
 * Simple shared state object. Views read from here and WS updates write to it.
 */

export const state = {
    // WebSocket
    wsConnected: false,

    // GPU
    gpu: { available: false, gpu_util_percent: 0, memory_percent: 0, name: '' },

    // Dataset
    datasetProgress: {},
    datasetStatus: null,

    // Training
    trainingBatch: null,
    trainingEpoch: null,
    trainingComplete: null,
    trainingHistory: [],

    // Models
    modelList: [],
    loadedModel: null,

    // Inference
    lastPrediction: null,
};