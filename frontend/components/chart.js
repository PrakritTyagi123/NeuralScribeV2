/** Responsive line chart — auto-sizes to container, HiDPI support. */
export function createChart(defaultWidth = 500, height = 220) {
    const wrapper = document.createElement('div');
    wrapper.className = 'chart-container';

    const canvas = document.createElement('canvas');
    wrapper.appendChild(canvas);

    const dpr = window.devicePixelRatio || 1;
    let datasets = [];
    let currentWidth = defaultWidth;

    function resize() {
        const w = wrapper.clientWidth || defaultWidth;
        currentWidth = w;
        canvas.width = w * dpr;
        canvas.height = height * dpr;
        canvas.style.width = w + 'px';
        canvas.style.height = height + 'px';
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        render();
    }

    // Auto-resize when container changes
    const ro = new ResizeObserver(() => resize());

    // Start observing after append
    setTimeout(() => {
        ro.observe(wrapper);
        resize();
    }, 50);

    function render() {
        const ctx = canvas.getContext('2d');
        const w = currentWidth;
        const h = height;
        const pad = { top: 20, right: 15, bottom: 25, left: 55 };
        const plotW = w - pad.left - pad.right;
        const plotH = h - pad.top - pad.bottom;

        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = '#fff';
        ctx.fillRect(0, 0, w, h);

        if (datasets.length === 0 || datasets.every(d => d.data.length === 0)) {
            ctx.fillStyle = '#ccc';
            ctx.font = '13px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('No data', w / 2, h / 2);
            return;
        }

        let allValues = datasets.flatMap(d => d.data);
        let minVal = Math.min(...allValues);
        let maxVal = Math.max(...allValues);
        let range = maxVal - minVal;
        if (range < 0.001) { range = 0.2; minVal -= 0.1; maxVal += 0.1; }
        else { minVal -= range * 0.05; maxVal += range * 0.05; }
        let maxLen = Math.max(...datasets.map(d => d.data.length));

        // Grid
        ctx.strokeStyle = '#eee';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = pad.top + (plotH * i / 4);
            ctx.beginPath();
            ctx.moveTo(pad.left, y);
            ctx.lineTo(pad.left + plotW, y);
            ctx.stroke();
            const val = maxVal - (maxVal - minVal) * i / 4;
            ctx.fillStyle = '#999';
            ctx.font = '11px sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText(val.toFixed(3), pad.left - 4, y + 3);
        }

        // X labels
        if (maxLen >= 1) {
            ctx.fillStyle = '#999';
            ctx.font = '11px sans-serif';
            ctx.textAlign = 'center';
            const step = Math.max(1, Math.floor(maxLen / 10));
            for (let i = 0; i < maxLen; i += step) {
                const x = pad.left + (i / Math.max(maxLen - 1, 1)) * plotW;
                ctx.fillText((i + 1).toString(), x, pad.top + plotH + 14);
            }
        }

        // Axes
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(pad.left, pad.top);
        ctx.lineTo(pad.left, pad.top + plotH);
        ctx.lineTo(pad.left + plotW, pad.top + plotH);
        ctx.stroke();

        const colors = ['#000', '#888'];
        const dashes = [[], [5, 3]];

        datasets.forEach((ds, di) => {
            if (ds.data.length === 0) return;
            const color = colors[di % colors.length];
            ctx.strokeStyle = color;
            ctx.fillStyle = color;
            ctx.lineWidth = 1.5;
            ctx.setLineDash(dashes[di % dashes.length]);
            const divisor = maxLen <= 1 ? 1 : maxLen - 1;

            if (ds.data.length >= 2) {
                ctx.beginPath();
                ds.data.forEach((val, i) => {
                    const x = pad.left + (i / divisor) * plotW;
                    const y = pad.top + plotH - ((val - minVal) / (maxVal - minVal)) * plotH;
                    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                });
                ctx.stroke();
            }

            ds.data.forEach((val, i) => {
                const x = pad.left + (i / divisor) * plotW;
                const y = pad.top + plotH - ((val - minVal) / (maxVal - minVal)) * plotH;
                ctx.beginPath();
                ctx.arc(x, y, 2.5, 0, Math.PI * 2);
                ctx.fill();
            });
            ctx.setLineDash([]);
        });

        // Legend
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'left';
        datasets.forEach((ds, di) => {
            const x = pad.left + 8 + di * 90;
            ctx.setLineDash(dashes[di % dashes.length]);
            ctx.strokeStyle = colors[di % colors.length];
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(x, pad.top + 4);
            ctx.lineTo(x + 16, pad.top + 4);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = '#333';
            ctx.fillText(ds.label || `Series ${di}`, x + 20, pad.top + 7);
        });

        ctx.fillStyle = '#999';
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Epoch', pad.left + plotW / 2, h - 2);
    }

    return {
        element: wrapper,
        setData(newDatasets) { datasets = newDatasets; render(); },
        addPoint(di, val) { if (datasets[di]) { datasets[di].data.push(val); render(); } },
        clear() { datasets = []; render(); },
        render,
        resize,
    };
}