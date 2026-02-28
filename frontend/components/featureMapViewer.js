/** Feature map viewer — displays base64 heatmap grid. */
export function createFeatureMapViewer(featureMapData) {
    const el = document.createElement('div');

    if (!featureMapData || !featureMapData.heatmaps) {
        el.innerHTML = '<div class="text-muted text-sm">No feature maps</div>';
        return el;
    }

    const info = document.createElement('div');
    info.className = 'text-sm text-muted mb-8';
    info.textContent = `${featureMapData.total_channels} channels, ${featureMapData.spatial_size.join('×')} spatial`;
    el.appendChild(info);

    const grid = document.createElement('div');
    grid.className = 'feature-map-grid';

    featureMapData.heatmaps.forEach(hm => {
        const img = document.createElement('img');
        img.src = `data:image/png;base64,${hm.heatmap}`;
        img.title = `Ch ${hm.channel} (${hm.importance.toFixed(3)})`;
        grid.appendChild(img);
    });

    el.appendChild(grid);

    // Importance bars
    if (featureMapData.importance && featureMapData.importance.length > 0) {
        const impTitle = document.createElement('div');
        impTitle.className = 'text-sm text-bold mt-8 mb-8';
        impTitle.textContent = 'Channel Importance';
        el.appendChild(impTitle);

        const maxImp = Math.max(...featureMapData.importance.map(i => i.importance));
        featureMapData.importance.slice(0, 16).forEach(imp => {
            const bar = document.createElement('div');
            bar.className = 'confidence-bar';
            const pct = maxImp > 0 ? (imp.importance / maxImp * 100) : 0;
            bar.innerHTML = `
                <span class="cb-label">${imp.channel}</span>
                <div class="cb-track"><div class="cb-fill" style="width:${pct}%"></div></div>
                <span class="cb-value">${imp.importance.toFixed(3)}</span>
            `;
            el.appendChild(bar);
        });
    }

    return el;
}