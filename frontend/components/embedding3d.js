/**
 * NeuralScribe v2 — 3D Embedding Visualization
 * Scatter mode: Three.js 3D, draggable, auto-rotate, contained.
 * Radar mode: flat 2D canvas (no Three.js) with crisp HTML-like text.
 */

let THREE = null;
let _instance = null;

const CDN = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js';

function loadThree() {
    return new Promise((resolve, reject) => {
        if (THREE) { resolve(THREE); return; }
        if (window.THREE) { THREE = window.THREE; resolve(THREE); return; }
        const s = document.createElement('script');
        s.src = CDN;
        s.onload = () => { THREE = window.THREE; resolve(THREE); };
        s.onerror = () => reject(new Error('Failed to load Three.js'));
        document.head.appendChild(s);
    });
}

export async function initEmbedding3D(container) {
    if (_instance) { _instance.destroy(); _instance = null; }
    await loadThree();

    container.innerHTML = '';
    container.style.cssText = 'display:flex;flex-direction:column;height:100%;';

    // Toolbar
    const toolbar = document.createElement('div');
    toolbar.style.cssText = 'display:flex;gap:4px;padding:4px 6px;flex-shrink:0;align-items:center;';
    const btn3d = document.createElement('button');
    btn3d.textContent = '3D Scatter';
    btn3d.className = 'btn';
    btn3d.style.cssText = 'font-size:10px;padding:1px 8px;background:var(--fg);color:var(--bg);';
    const btnRadar = document.createElement('button');
    btnRadar.textContent = 'Radar';
    btnRadar.className = 'btn';
    btnRadar.style.cssText = 'font-size:10px;padding:1px 8px;';
    toolbar.appendChild(btn3d);
    toolbar.appendChild(btnRadar);
    container.appendChild(toolbar);

    // Content area — holds either 3D canvas or 2D radar canvas
    const cvWrap = document.createElement('div');
    cvWrap.style.cssText = 'flex:1;min-height:0;position:relative;overflow:hidden;';
    container.appendChild(cvWrap);

    // ════════════════════════════════════
    // 3D SCATTER (Three.js)
    // ════════════════════════════════════
    const canvas3d = document.createElement('canvas');
    canvas3d.style.cssText = 'width:100%;height:100%;display:block;cursor:grab;';
    cvWrap.appendChild(canvas3d);

    // 2D RADAR canvas (hidden initially)
    const canvas2d = document.createElement('canvas');
    canvas2d.style.cssText = 'width:100%;height:100%;display:none;position:absolute;top:0;left:0;';
    cvWrap.appendChild(canvas2d);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    camera.position.set(0, 0, 6);
    camera.lookAt(0, 0, 0);
    const renderer = new THREE.WebGLRenderer({ canvas: canvas3d, antialias: true, alpha: true });
    renderer.setClearColor(0x000000, 0);

    const group = new THREE.Group();
    scene.add(group);

    let mode = 'scatter';
    let autoRotate = true, angle = 0;
    let dragging = false, lastMX = 0, lastMY = 0, rotX = 0, rotY = 0;
    let destroyed = false;
    let currentPreds = [];

    let scatterNodes = [], scatterLabels = [], scatterDots = [], scatterAxes = [];

    // Drag (3D only)
    canvas3d.addEventListener('mousedown', e => { dragging = true; lastMX = e.clientX; lastMY = e.clientY; autoRotate = false; });
    canvas3d.addEventListener('mousemove', e => {
        if (!dragging) return;
        rotY += (e.clientX - lastMX) * 0.008;
        rotX += (e.clientY - lastMY) * 0.008;
        rotX = Math.max(-1.2, Math.min(1.2, rotX)); // clamp vertical
        lastMX = e.clientX; lastMY = e.clientY;
    });
    canvas3d.addEventListener('mouseup', () => dragging = false);
    canvas3d.addEventListener('mouseleave', () => dragging = false);

    btn3d.addEventListener('click', () => setMode('scatter'));
    btnRadar.addEventListener('click', () => setMode('radar'));

    function setMode(m) {
        mode = m;
        btn3d.style.background = m === 'scatter' ? 'var(--fg)' : 'transparent';
        btn3d.style.color = m === 'scatter' ? 'var(--bg)' : 'var(--fg)';
        btnRadar.style.background = m === 'radar' ? 'var(--fg)' : 'transparent';
        btnRadar.style.color = m === 'radar' ? 'var(--bg)' : 'var(--fg)';
        canvas3d.style.display = m === 'scatter' ? 'block' : 'none';
        canvas2d.style.display = m === 'radar' ? 'block' : 'none';
        if (m === 'scatter') autoRotate = true;
        if (m === 'radar') drawRadar2D();
    }

    // ── Scatter positions — CONTAINED within radius 1.8 ──
    function scatterPos(i, total, conf) {
        if (i === 0) return new THREE.Vector3(0, 0, 0);
        const phi = (i / Math.max(total - 1, 1)) * Math.PI * 2 + 0.5;
        const theta = 0.5 + (i * 0.25) % 1.0;
        const r = 0.8 + (1 - conf) * 1.0; // max ~1.8, stays in view
        return new THREE.Vector3(
            r * Math.sin(theta) * Math.cos(phi),
            r * Math.sin(theta) * Math.sin(phi),
            r * Math.cos(theta) - 0.3
        );
    }

    function makeLabel3D(text, isWin) {
        const c = document.createElement('canvas');
        c.width = 512; c.height = 128;
        const ctx = c.getContext('2d');
        ctx.clearRect(0, 0, 512, 128);
        ctx.font = isWin ? 'bold 72px monospace' : '56px monospace';
        ctx.fillStyle = isWin ? '#1d4ed8' : '#000000';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, 256, 64);
        const tex = new THREE.CanvasTexture(c);
        tex.minFilter = THREE.LinearFilter;
        tex.magFilter = THREE.LinearFilter;
        const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
        const sp = new THREE.Sprite(mat);
        sp.scale.set(isWin ? 1.6 : 1.2, isWin ? 0.4 : 0.3, 1);
        return sp;
    }

    function clearScatter() {
        while (group.children.length > 0) {
            const child = group.children[0];
            group.remove(child);
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (child.material.map) child.material.map.dispose();
                child.material.dispose();
            }
        }
        scatterNodes = []; scatterLabels = []; scatterDots = []; scatterAxes = [];
    }

    function buildScatter(preds) {
        clearScatter();
        if (!preds || preds.length === 0) return;
        const N = preds.length;

        preds.forEach((p, i) => {
            const isWin = i === 0;
            const geo = new THREE.SphereGeometry(isWin ? 0.18 : 0.10, 16, 16);
            const color = isWin ? 0x2563eb : (p.confidence > 0.02 ? 0x16a34a : 0x333333);
            const mat = new THREE.MeshBasicMaterial({ color, transparent: !isWin, opacity: isWin ? 1 : 0.7 });
            const mesh = new THREE.Mesh(geo, mat);
            const pos = scatterPos(i, N, p.confidence);
            mesh.position.copy(pos);
            group.add(mesh);
            scatterNodes.push(mesh);

            const labelText = p.display + ' ' + (p.confidence * 100).toFixed(0) + '%';
            const sp = makeLabel3D(labelText, isWin);
            sp.position.copy(pos);
            sp.position.y += 0.32;
            group.add(sp);
            scatterLabels.push(sp);

            // Dots
            const count = Math.max(3, Math.round(p.confidence * 120 + 3));
            const spread = isWin ? 0.3 : 0.25;
            for (let d = 0; d < count; d++) {
                const dg = new THREE.SphereGeometry(0.025, 6, 6);
                const dm = new THREE.MeshBasicMaterial({
                    color: isWin ? 0x2563eb : 0x555555,
                    transparent: true, opacity: isWin ? 0.25 : 0.12
                });
                const dot = new THREE.Mesh(dg, dm);
                dot.position.set(
                    pos.x + (Math.random() - 0.5) * spread,
                    pos.y + (Math.random() - 0.5) * spread,
                    pos.z + (Math.random() - 0.5) * spread
                );
                group.add(dot);
                scatterDots.push(dot);
            }
        });

        // Axes — black
        [[[1.6,0,0],[-1.6,0,0]],[[0,1.6,0],[0,-1.6,0]],[[0,0,1.6],[0,0,-1.6]]].forEach(pts => {
            const g = new THREE.BufferGeometry().setFromPoints(pts.map(p => new THREE.Vector3(...p)));
            const l = new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.4 }));
            group.add(l);
            scatterAxes.push(l);
        });
    }

    // ════════════════════════════════════
    // 2D RADAR (pure Canvas 2D — crisp text)
    // ════════════════════════════════════
    function drawRadar2D() {
        const w = cvWrap.clientWidth;
        const h = cvWrap.clientHeight;
        if (w < 10 || h < 10) return;
        const dpr = window.devicePixelRatio || 1;
        canvas2d.width = w * dpr;
        canvas2d.height = h * dpr;
        canvas2d.style.width = w + 'px';
        canvas2d.style.height = h + 'px';
        const ctx = canvas2d.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, w, h);

        const preds = currentPreds;
        if (!preds || preds.length === 0) {
            ctx.fillStyle = '#888';
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Draw to see', w / 2, h / 2);
            return;
        }

        const cx = w / 2;
        const cy = h / 2;
        const maxR = Math.min(w, h) * 0.36;
        const N = preds.length;

        // Grid rings — black
        for (let ring = 1; ring <= 4; ring++) {
            const r = maxR * ring / 4;
            ctx.beginPath();
            for (let i = 0; i <= 64; i++) {
                const a = (i / 64) * Math.PI * 2;
                const x = cx + Math.cos(a) * r;
                const y = cy + Math.sin(a) * r;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.strokeStyle = '#000000';
            ctx.lineWidth = 0.5;
            ctx.stroke();

            // Ring labels
            ctx.fillStyle = '#888';
            ctx.font = '9px monospace';
            ctx.textAlign = 'left';
            ctx.fillText((ring * 25) + '%', cx + 3, cy - r + 10);
        }

        // Spokes — black
        for (let i = 0; i < N; i++) {
            const a = (i / N) * Math.PI * 2 - Math.PI / 2;
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(cx + Math.cos(a) * maxR, cy + Math.sin(a) * maxR);
            ctx.strokeStyle = '#000000';
            ctx.lineWidth = 0.5;
            ctx.stroke();
        }

        // Confidence polygon — sqrt scale for visibility
        ctx.beginPath();
        preds.forEach((p, i) => {
            const a = (i / N) * Math.PI * 2 - Math.PI / 2;
            const r = Math.max(Math.sqrt(p.confidence) * maxR, maxR * 0.05);
            const x = cx + Math.cos(a) * r;
            const y = cy + Math.sin(a) * r;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.fillStyle = 'rgba(37,99,235,0.15)';
        ctx.fill();
        ctx.strokeStyle = '#2563eb';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Dots on polygon vertices
        preds.forEach((p, i) => {
            const a = (i / N) * Math.PI * 2 - Math.PI / 2;
            const r = Math.max(Math.sqrt(p.confidence) * maxR, maxR * 0.05);
            const x = cx + Math.cos(a) * r;
            const y = cy + Math.sin(a) * r;

            ctx.beginPath();
            ctx.arc(x, y, i === 0 ? 5 : 3, 0, Math.PI * 2);
            ctx.fillStyle = i === 0 ? '#2563eb' : '#333';
            ctx.fill();
        });

        // Labels at OUTER EDGE — always readable
        preds.forEach((p, i) => {
            const a = (i / N) * Math.PI * 2 - Math.PI / 2;
            const lx = cx + Math.cos(a) * (maxR + 20);
            const ly = cy + Math.sin(a) * (maxR + 20);

            ctx.font = i === 0 ? 'bold 14px monospace' : '12px monospace';
            ctx.fillStyle = i === 0 ? '#1d4ed8' : '#000000';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(p.display, lx, ly - 7);

            ctx.font = '10px monospace';
            ctx.fillStyle = i === 0 ? '#2563eb' : '#555';
            ctx.fillText((p.confidence * 100).toFixed(1) + '%', lx, ly + 7);
        });
    }

    // ════════════════════════════════════
    // RESIZE
    // ════════════════════════════════════
    function resize() {
        const w = cvWrap.clientWidth;
        const h = cvWrap.clientHeight;
        if (w < 10 || h < 10) return;
        renderer.setSize(w, h);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        if (mode === 'radar') drawRadar2D();
    }
    const ro = new ResizeObserver(resize);
    ro.observe(cvWrap);
    setTimeout(resize, 100);

    // ════════════════════════════════════
    // ANIMATE (3D only)
    // ════════════════════════════════════
    function animate() {
        if (destroyed) return;
        requestAnimationFrame(animate);
        if (mode !== 'scatter') return; // skip rendering when radar is shown

        if (autoRotate && !dragging) angle += 0.003;
        group.rotation.y = angle + rotY;
        group.rotation.x = rotX;

        scatterLabels.forEach(sp => sp.lookAt(camera.position));
        renderer.render(scene, camera);
    }
    animate();

    // ════════════════════════════════════
    // PUBLIC API
    // ════════════════════════════════════
    function predKey(preds) {
        return preds.map(p => p.display + ':' + p.confidence.toFixed(3)).join('|');
    }
    let lastKey = '';

    _instance = {
        update(preds) {
            if (!preds || preds.length === 0) return;
            const key = predKey(preds);
            if (key === lastKey) return;
            lastKey = key;
            currentPreds = preds;
            buildScatter(preds);
            if (mode === 'radar') drawRadar2D();
        },
        reset() {
            clearScatter();
            lastKey = '';
            currentPreds = [];
            if (mode === 'radar') {
                const ctx = canvas2d.getContext('2d');
                ctx.clearRect(0, 0, canvas2d.width, canvas2d.height);
            }
        },
        destroy() {
            destroyed = true;
            ro.disconnect();
            clearScatter();
            renderer.dispose();
            _instance = null;
        }
    };

    return _instance;
}

export function resetEmbeddingCache() {
    if (_instance) _instance.reset();
}
