/** Confidence bars — shows top-k predictions. */
export function createConfidenceBars(predictions = []) {
    const el = document.createElement('div');
    el.className = 'confidence-bars';
    updateConfidenceBars(el, predictions);
    return el;
}

export function updateConfidenceBars(el, predictions = []) {
    el.innerHTML = predictions.map(p => `
        <div class="confidence-bar">
            <span class="cb-label">${p.display || p.label || '?'}</span>
            <div class="cb-track">
                <div class="cb-fill" style="width: ${(p.confidence * 100).toFixed(1)}%"></div>
            </div>
            <span class="cb-value">${(p.confidence * 100).toFixed(1)}%</span>
        </div>
    `).join('');
}