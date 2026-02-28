/** Stat card component. */
export function createStatCard(label, value, sub = '') {
    const el = document.createElement('div');
    el.className = 'stat-card';
    el.innerHTML = `
        <div class="stat-label">${label}</div>
        <div class="stat-value">${value}</div>
        ${sub ? `<div class="stat-sub">${sub}</div>` : ''}
    `;
    return el;
}