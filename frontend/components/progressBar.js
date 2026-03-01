/** Progress bar component. */
export function createProgressBar(percent = 0, text = '') {
    const el = document.createElement('div');
    el.className = 'progress-bar';
    el.innerHTML = `
        <div class="progress-fill" style="width: ${percent}%"></div>
        <div class="progress-text">${text || Math.round(percent) + '%'}</div>
    `;
    return el;
}

export function updateProgressBar(el, percent, text = '') {
    const clamped = Math.min(100, Math.max(0, percent));
    const fill = el.querySelector('.progress-fill');
    const label = el.querySelector('.progress-text');
    if (fill) fill.style.width = clamped + '%';
    if (label) label.textContent = text || Math.round(clamped) + '%';
}