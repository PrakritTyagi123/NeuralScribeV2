/** Network flow — vertical node diagram for explainability. */
export function createNetworkFlow(layers = [], onNodeClick = null) {
    const el = document.createElement('div');
    el.className = 'network-flow';

    const defaultLayers = ['Input', 'Conv Stem', 'Block 0', 'Block 1', 'Block 2', 'Block 3', 'Classifier', 'Softmax'];
    const nodeList = layers.length > 0 ? layers : defaultLayers;

    nodeList.forEach((name, i) => {
        if (i > 0) {
            const arrow = document.createElement('div');
            arrow.className = 'network-arrow';
            arrow.textContent = '↓';
            el.appendChild(arrow);
        }

        const node = document.createElement('div');
        node.className = 'network-node';
        node.textContent = name;
        node.dataset.layer = name;

        if (onNodeClick) {
            node.addEventListener('click', () => {
                el.querySelectorAll('.network-node').forEach(n => n.classList.remove('active'));
                node.classList.add('active');
                onNodeClick(name, i);
            });
        }

        el.appendChild(node);
    });

    return el;
}