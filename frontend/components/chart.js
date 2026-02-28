/** Simple line chart using canvas 2D. No external dependencies. */
export function createChart(width = 400, height = 200) {
    const wrapper = document.createElement('div');
    wrapper.className = 'chart-container';

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    wrapper.appendChild(canvas);

    const ctx = canvas.getContext('2d');
    let datasets = [];

    function render() {
        const w = canvas.width;
        const h = canvas.height;
        const pad = { top: 20, right: 10, bottom: 25, left: 50 };
        const plotW = w - pad.left - pad.right;
        const plotH = h - pad.top - pad.bottom;

        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = '#fff';
        ctx.fillRect(0, 0, w, h);

        if (datasets.length === 0 || datasets.every(d => d.data.length === 0)) {
            ctx.fillStyle = '#ccc';
            ctx.font = '12px Courier New';
            ctx.textAlign = 'center';
            ctx.fillText('No data', w / 2, h / 2);
            return;
        }

        // Find range
        let allValues = datasets.flatMap(d => d.data);
        let minVal = Math.min(...allValues);
        let maxVal = Math.max(...allValues);
        let range = maxVal - minVal;
        if (range < 0.001) { range = 0.2; minVal -= 0.1; maxVal += 0.1; }
        else { minVal -= range * 0.05; maxVal += range * 0.05; }
        let maxLen = Math.max(...datasets.map(d => d.data.length));

        // Grid lines
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
            ctx.font = '10px Courier New';
            ctx.textAlign = 'right';
            ctx.fillText(val.toFixed(3), pad.left - 4, y + 3);
        }

        // X-axis epoch labels
        if (maxLen > 1) {
            ctx.fillStyle = '#999';
            ctx.font = '10px Courier New';
            ctx.textAlign = 'center';
            const step = Math.max(1, Math.floor(maxLen / 8));
            for (let i = 0; i < maxLen; i += step) {
                const x = pad.left + (i / Math.max(maxLen - 1, 1)) * plotW;
                ctx.fillText(i.toString(), x, pad.top + plotH + 14);
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

        // Lines + dots
        const colors = ['#000', '#888'];
        const dashes = [[], [6, 3]];

        datasets.forEach((ds, di) => {
            if (ds.data.length === 0) return;
            const color = colors[di % colors.length];
            ctx.strokeStyle = color;
            ctx.fillStyle = color;
            ctx.lineWidth = 1.5;
            ctx.setLineDash(dashes[di % dashes.length]);

            const divisor = maxLen <= 1 ? 1 : maxLen - 1;

            // Draw line
            if (ds.data.length >= 2) {
                ctx.beginPath();
                ds.data.forEach((val, i) => {
                    const x = pad.left + (i / divisor) * plotW;
                    const y = pad.top + plotH - ((val - minVal) / (maxVal - minVal)) * plotH;
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                });
                ctx.stroke();
            }

            // Draw dots
            ds.data.forEach((val, i) => {
                const x = pad.left + (i / divisor) * plotW;
                const y = pad.top + plotH - ((val - minVal) / (maxVal - minVal)) * plotH;
                ctx.beginPath();
                ctx.arc(x, y, 3, 0, Math.PI * 2);
                ctx.fill();
            });

            ctx.setLineDash([]);
        });

        // Legend
        ctx.font = '10px Courier New';
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

        // X-axis label
        ctx.fillStyle = '#999';
        ctx.font = '10px Courier New';
        ctx.textAlign = 'center';
        ctx.fillText('Epoch', pad.left + plotW / 2, h - 2);
    }

    return {
        element: wrapper,
        setData(newDatasets) {
            datasets = newDatasets;
            render();
        },
        addPoint(datasetIndex, value) {
            if (datasets[datasetIndex]) {
                datasets[datasetIndex].data.push(value);
                render();
            }
        },
        clear() {
            datasets = [];
            render();
        },
        render,
    };
}