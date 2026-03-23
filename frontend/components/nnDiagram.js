/**
 * NeuralScribe v2 — Neural Network Diagram Renderer
 * Handles layout computation, node positioning, connection drawing,
 * and layer label rendering for the live NN visualization.
 */

export const ARCH = [
    { name: 'Input',   n: 8,   vis: 8,  type: 'input',  key: 'input' },
    { name: 'Stem',    n: 32,  vis: 12, type: 'conv',   key: 'stem' },
    { name: 'Block 0', n: 64,  vis: 12, type: 'conv',   key: 'block_0' },
    { name: 'Block 1', n: 128, vis: 14, type: 'conv',   key: 'block_1' },
    { name: 'Block 2', n: 256, vis: 16, type: 'conv',   key: 'block_2' },
    { name: 'Block 3', n: 320, vis: 18, type: 'conv',   key: 'block_3' },
    { name: 'Pool',    n: 320, vis: 12, type: 'fc',     key: 'pooled' },
    { name: 'Output',  n: 47,  vis: 5,  type: 'output', key: 'output' },
];

export const VIS_DEFAULT = 8;

/**
 * Compute layer positions for the NN diagram.
 * @param {number} w - container width
 * @param {number} h - container height
 * @param {Array} nodeAct - per-layer activation arrays
 * @returns {Array} layerPos
 */
export function computePositions(w, h, nodeAct) {
    const layerPos = [];
    if (w < 40 || h < 40) return layerPos;

    const padLeft = 40;
    const padRight = 95;
    const padTop = 14;
    const padBot = 48;
    const usableW = w - padLeft - padRight;
    const usableH = h - padTop - padBot;
    const layerSpacing = usableW / Math.max(ARCH.length - 1, 1);

    // Find the max visible node count to set a consistent node gap
    const maxVis = Math.max(...ARCH.map(l => l.vis || VIS_DEFAULT));
    const globalNodeGap = Math.min(22, usableH / Math.max(maxVis + 1, 1));

    ARCH.forEach((layer, li) => {
        const x = padLeft + li * layerSpacing;
        const visible = layer.vis || VIS_DEFAULT;
        const nodeGap = globalNodeGap;
        const totalH = (visible - 1) * nodeGap;
        const startY = padTop + (usableH - totalH) / 2;

        const nodes = [];
        for (let i = 0; i < visible; i++) {
            nodes.push({
                x: x,
                y: startY + i * nodeGap,
                activation: nodeAct[li] ? (nodeAct[li][i] || 0) : 0,
            });
        }

        layerPos.push({
            x: x,
            nodes: nodes,
            layer: layer,
            visible: visible,
            hasMore: layer.n > visible,
        });
    });
    return layerPos;
}

/**
 * Draw the full NN diagram: connections, nodes, labels.
 */
export function drawNN(ctx, w, h, layerPos, outPreds, winIdx) {
    if (w < 40 || h < 40 || layerPos.length === 0) return;
    ctx.clearRect(0, 0, w, h);

    // ── CONNECTIONS ──
    for (let li = 0; li < layerPos.length - 1; li++) {
        const from = layerPos[li];
        const to = layerPos[li + 1];
        const fLen = from.nodes.length;
        const tLen = to.nodes.length;

        // Build the set of (fi, ti) pairs to draw.
        // Strategy: first ensure every node on BOTH sides has at least
        // one connection (nearest-neighbor mapping), then add sampled
        // cross-connections for density.
        const pairs = new Set();

        // 1. Map each 'from' node to its nearest 'to' node (by position)
        for (let fi = 0; fi < fLen; fi++) {
            const ti = Math.round(fi / Math.max(fLen - 1, 1) * Math.max(tLen - 1, 1));
            pairs.add(fi + ',' + ti);
        }
        // 2. Map each 'to' node to its nearest 'from' node
        for (let ti = 0; ti < tLen; ti++) {
            const fi = Math.round(ti / Math.max(tLen - 1, 1) * Math.max(fLen - 1, 1));
            pairs.add(fi + ',' + ti);
        }

        // 3. Add sampled cross-connections for visual fullness
        const maxExtra = 150;
        const budget = maxExtra - pairs.size;
        if (budget > 0) {
            const fromSample = sampleIndices(fLen, Math.min(fLen, 10));
            const toSample = sampleIndices(tLen, Math.min(tLen, 12));
            outer:
            for (const fi of fromSample) {
                for (const ti of toSample) {
                    pairs.add(fi + ',' + ti);
                    if (pairs.size >= maxExtra) break outer;
                }
            }
        }

        // Draw all collected pairs
        for (const key of pairs) {
            const [fi, ti] = key.split(',').map(Number);
            const fNode = from.nodes[fi];
            const tNode = to.nodes[ti];
            const avgAct = (fNode.activation + tNode.activation) / 2;

            let alpha, thickness;
            if (avgAct > 0.3) {
                alpha = 0.12 + avgAct * 0.45;
                thickness = 0.6 + avgAct * 1.8;
                ctx.strokeStyle = 'rgba(37,99,235,' + alpha + ')';
            } else {
                alpha = 0.03 + avgAct * 0.12;
                thickness = 0.3;
                ctx.strokeStyle = 'rgba(100,116,139,' + alpha + ')';
            }
            ctx.lineWidth = thickness;
            ctx.beginPath();
            ctx.moveTo(fNode.x, fNode.y);
            ctx.lineTo(tNode.x, tNode.y);
            ctx.stroke();
        }
    }

    // ── NODES ──
    layerPos.forEach((lp, li) => {
        const isOutput = lp.layer.type === 'output';
        const isInput = lp.layer.type === 'input';

        lp.nodes.forEach((node, ni) => {
            const act = node.activation;
            const radius = isOutput ? 7 : isInput ? 4.5 : 5.5;

            let fillColor;
            if (act < 0.1) fillColor = '#d4d4d8';
            else if (act < 0.3) fillColor = '#93c5fd';
            else if (act < 0.6) fillColor = '#3b82f6';
            else fillColor = '#1d4ed8';

            ctx.fillStyle = fillColor;
            ctx.strokeStyle = act > 0.3 ? '#1e40af' : '#71717a';
            ctx.lineWidth = isOutput ? 1.5 : 1;
            ctx.beginPath();
            ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();

            // Winner highlight
            if (isOutput && ni === winIdx) {
                ctx.strokeStyle = '#2563eb';
                ctx.lineWidth = 2.5;
                ctx.beginPath();
                ctx.arc(node.x, node.y, radius + 5, 0, Math.PI * 2);
                ctx.stroke();
            }

            // Output node labels
            if (isOutput && outPreds[ni]) {
                const pred = outPreds[ni];
                ctx.fillStyle = ni === winIdx ? '#1e40af' : '#71717a';
                ctx.font = ni === winIdx ? 'bold 11px monospace' : '9px monospace';
                ctx.textAlign = 'left';
                ctx.fillText(
                    pred.display + ' ' + (pred.confidence * 100).toFixed(0) + '%',
                    node.x + radius + 6,
                    node.y + 4
                );
            }
        });

        // Ellipsis
        if (lp.hasMore) {
            const lastNode = lp.nodes[lp.nodes.length - 1];
            ctx.fillStyle = '#a1a1aa';
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('⋮', lp.x, lastNode.y + 16);
        }

        // Layer name label
        const bottomNode = lp.nodes[lp.nodes.length - 1];
        const labelY = bottomNode.y + (lp.hasMore ? 30 : 20);
        ctx.fillStyle = '#52525b';
        ctx.font = '9px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(lp.layer.name, lp.x, labelY);
        ctx.fillStyle = '#a1a1aa';
        ctx.font = '7px monospace';
        ctx.fillText(String(lp.layer.n), lp.x, labelY + 11);
    });
}

/**
 * Apply live data from the API to the node activation arrays and layer positions.
 */
export function applyLiveData(data, nodeAct, layerPos) {
    if (!data || !data.layers) return { outPreds: [], winIdx: -1 };

    data.layers.forEach(layer => {
        const archIdx = ARCH.findIndex(a => a.key === layer.name);
        if (archIdx < 0) return;

        if (layer.name === 'output') {
            const preds = data.predictions || [];
            const vis = ARCH[archIdx].vis || VIS_DEFAULT;
            for (let n = 0; n < Math.min(preds.length, vis); n++) {
                nodeAct[archIdx][n] = preds[n] ? preds[n].confidence : 0;
            }
        } else {
            const vals = layer.node_values || [];
            const vis = ARCH[archIdx].vis || VIS_DEFAULT;
            for (let n = 0; n < Math.min(vals.length, vis); n++) {
                nodeAct[archIdx][n] = vals[n] || 0;
            }
        }

        if (layerPos[archIdx]) {
            layerPos[archIdx].nodes.forEach((node, ni) => {
                node.activation = nodeAct[archIdx][ni] || 0;
            });
        }
    });

    const outPreds = (data.predictions || []).slice(0, ARCH[ARCH.length - 1].vis);
    let winIdx = -1;
    if (outPreds.length > 0) {
        winIdx = outPreds.findIndex(p => p.confidence === Math.max(...outPreds.map(q => q.confidence)));
        if (winIdx < 0) winIdx = 0;
    }

    return { outPreds, winIdx };
}

function sampleIndices(total, count) {
    if (count >= total) return Array.from({ length: total }, (_, i) => i);
    const step = total / count;
    return Array.from({ length: count }, (_, i) => Math.floor(i * step));
}
