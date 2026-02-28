/** Probability evolution — shows how predictions change through layers. */
export function createProbEvolution(evolution = []) {
    const el = document.createElement('div');

    if (!evolution || evolution.length === 0) {
        el.innerHTML = '<div class="text-muted text-sm">No data</div>';
        return el;
    }

    evolution.forEach((frame, i) => {
        const frameEl = document.createElement('div');
        frameEl.className = 'prob-evolution-frame';

        const label = document.createElement('div');
        label.className = 'frame-label';
        label.textContent = frame.layer;
        frameEl.appendChild(label);

        (frame.top_5 || []).forEach(entry => {
            const bar = document.createElement('div');
            bar.className = 'confidence-bar';
            bar.innerHTML = `
                <span class="cb-label">${entry.display}</span>
                <div class="cb-track"><div class="cb-fill" style="width:${(entry.probability * 100)}%"></div></div>
                <span class="cb-value">${(entry.probability * 100).toFixed(1)}%</span>
            `;
            frameEl.appendChild(bar);
        });

        el.appendChild(frameEl);
    });

    return el;
}