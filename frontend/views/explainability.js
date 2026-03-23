/**
 * NeuralScribe v2 — Live Neural Network View (English only)
 */
import { createCanvas } from '../components/canvas.js';
import { createConfidenceBars, updateConfidenceBars } from '../components/confidenceBars.js';
import { ARCH, VIS_DEFAULT, computePositions, drawNN, applyLiveData } from '../components/nnDiagram.js';
import { drawEvo, drawStrokeTimeline } from '../components/lnnCharts.js';
import { drawGradCAM, updateFeatureMaps, updateConfusion, updateRobustness, updateCalibration } from '../components/lnnPanels.js';
import { initEmbedding3D, resetEmbeddingCache } from '../components/embedding3d.js';

let _af = null, _dead = false;

export async function renderExplainability(container) {
    if (_af) { cancelAnimationFrame(_af); _af = null; }
    _dead = false;
    container.innerHTML = '';
    container.className = 'view-fit';

    let ready = false;
    try { ready = (await (await fetch('/api/inference/status')).json()).ready; } catch(e) {}
    if (!ready) {
        container.innerHTML = '<div class="view-title">Live View</div><div class="panel"><div class="panel-body text-muted">No model loaded. Go to Models → Load a model first.</div></div>';
        return;
    }

    const hdr = mk('div', 'lnn-hdr');
    hdr.innerHTML = '<span class="lnn-title">Live Neural Network</span><span class="lnn-status" id="ex-status">Draw a letter or digit to begin</span>';
    container.appendChild(hdr);

    const layout = mk('div', 'lnn-layout');
    container.appendChild(layout);
    const topRow = mk('div', 'lnn-top'); layout.appendChild(topRow);
    const botRow = mk('div', 'lnn-bot'); layout.appendChild(botRow);

    // TOP LEFT: Draw + Predict
    const colL = mk('div', 'lnn-col-l'); topRow.appendChild(colL);
    const drawPanel = panel('Draw', 'Real-time recognition as you draw');
    drawPanel.el.classList.add('lnn-draw-panel');
    const canvasObj = createCanvas(160);
    drawPanel.body.classList.add('lnn-draw-body');
    const cw = canvasObj.element; cw.style.cssText = 'display:flex;flex-direction:column;align-items:center;width:100%;';
    drawPanel.body.appendChild(cw); colL.appendChild(drawPanel.el);

    const predPanel = panel('Prediction', 'Top-5 most likely characters');
    predPanel.el.classList.add('lnn-pred-panel');
    const bigPred = mk('div', 'lnn-big'); bigPred.textContent = '?';
    const confTxt = mk('div', 'lnn-conf'); confTxt.textContent = '—';
    const barsEl = createConfidenceBars([]);
    predPanel.body.append(bigPred, confTxt, barsEl); colL.appendChild(predPanel.el);

    // TOP CENTER: NN Diagram
    const colC = mk('div', 'lnn-col-c'); topRow.appendChild(colC);
    const nnPanel = panel('Neural Network', 'Signal flow — brighter = stronger activation');
    nnPanel.el.classList.add('lnn-nn-panel'); nnPanel.body.classList.add('lnn-nn-body');
    const nnWrap = mk('div', 'lnn-cvwrap');
    const nnCv = document.createElement('canvas'); nnCv.classList.add('lnn-cv');
    nnWrap.appendChild(nnCv); nnPanel.body.appendChild(nnWrap); colC.appendChild(nnPanel.el);

    // TOP RIGHT
    const colR = mk('div', 'lnn-col-r'); topRow.appendChild(colR);
    const gcPanel = panel('Grad-CAM Saliency', 'Where the model looks');
    gcPanel.el.classList.add('lnn-gc-panel');
    gcPanel.body.innerHTML = '<div class="lnn-gc"><div class="lnn-gc-img"><canvas id="gc-cv" width="56" height="56"></canvas></div><div class="lnn-gc-side"><div class="lnn-gc-bar"></div><div class="lnn-gc-labels"><span>Low</span><span>High</span></div></div></div>';
    colR.appendChild(gcPanel.el);

    const ppPanel = panel('Preprocessing', 'What the model actually sees');
    ppPanel.el.classList.add('lnn-pp-panel');
    ppPanel.body.innerHTML = '<div class="lnn-pp"><div class="lnn-pp-step"><img id="pp-raw" class="lnn-pp-img" /><div class="lnn-pp-lbl">Your input</div></div><div class="lnn-pp-arrow">→</div><div class="lnn-pp-step"><img id="pp-proc" class="lnn-pp-img" /><div class="lnn-pp-lbl">Model sees</div></div></div>';
    colR.appendChild(ppPanel.el);

    const cnPanel = panel('Confusion', 'Classes the model is deciding between');
    cnPanel.el.classList.add('lnn-cn-panel');
    cnPanel.body.id = 'cn-body'; cnPanel.body.innerHTML = '<div class="lnn-placeholder">Draw to see</div>';
    colR.appendChild(cnPanel.el);

    const evoPanel = panel('Probability Evolution', 'Confidence changes as you draw');
    evoPanel.el.classList.add('lnn-evo-panel');
    const evoWrap = mk('div', 'lnn-cvwrap');
    const evoCv = document.createElement('canvas'); evoCv.classList.add('lnn-cv');
    evoWrap.appendChild(evoCv); evoPanel.body.classList.add('lnn-cv-body');
    evoPanel.body.appendChild(evoWrap); colR.appendChild(evoPanel.el);

    // BOTTOM LEFT: Feature Maps + Stroke Timeline
    const botL = mk('div', 'lnn-bot-l'); botRow.appendChild(botL);
    const fmPanel = panel('Feature Maps', 'Top activation channels');
    fmPanel.el.classList.add('lnn-fm-panel'); fmPanel.body.id = 'fmap-body';
    fmPanel.body.classList.add('lnn-fm-body');
    fmPanel.body.innerHTML = '<div class="lnn-placeholder">Draw to see activations</div>';
    botL.appendChild(fmPanel.el);

    const stPanel = panel('Stroke Timeline', 'Confidence over time');
    stPanel.el.classList.add('lnn-st-panel');
    const stWrap = mk('div', 'lnn-cvwrap');
    const stCv = document.createElement('canvas'); stCv.classList.add('lnn-cv');
    stWrap.appendChild(stCv); stPanel.body.classList.add('lnn-cv-body');
    stPanel.body.appendChild(stWrap); botL.appendChild(stPanel.el);

    // BOTTOM CENTER: 3D Embedding Space
    const emPanel = panel('Embedding Space', '3D scatter / radar — drag to rotate');
    emPanel.el.classList.add('lnn-em-panel');
    emPanel.el.style.cssText = 'flex:2;min-height:0;';
    const emContainer = mk('div', '');
    emContainer.style.cssText = 'flex:1;min-height:0;display:flex;flex-direction:column;';
    emPanel.body.style.cssText = 'padding:0;flex:1;display:flex;flex-direction:column;min-height:0;';
    emPanel.body.appendChild(emContainer);
    botRow.appendChild(emPanel.el);

    // BOTTOM RIGHT: Robustness + Calibration
    const botR = mk('div', 'lnn-bot-r'); botRow.appendChild(botR);
    const rbPanel = panel('Robustness', 'Stability across perturbations');
    rbPanel.el.classList.add('lnn-rb-panel');
    rbPanel.body.innerHTML = '<div class="lnn-rb"><div class="lnn-rb-track"><div class="lnn-rb-fill" id="rb-fill"></div></div><span class="lnn-rb-val" id="rb-val">—</span></div><div class="lnn-tta" id="tta-grid"></div>';
    botR.appendChild(rbPanel.el);

    const calPanel = panel('Calibration', 'Confidence analysis');
    calPanel.el.classList.add('lnn-cal-panel');
    calPanel.body.innerHTML = '<div class="lnn-cal"><div class="lnn-cal-ring" id="cal-dial">—</div><div class="lnn-cal-txt" id="cal-info"><span class="lnn-placeholder">Draw to see</span></div></div>';
    botR.appendChild(calPanel.el);

    // Init 3D embedding (async — loads Three.js from CDN)
    let emb3d = null;
    initEmbedding3D(emContainer).then(inst => { emb3d = inst; }).catch(e => {
        console.warn('3D embedding init failed:', e);
        emContainer.innerHTML = '<div class="lnn-placeholder">3D not available</div>';
    });

    // STATE
    let nodeAct = ARCH.map(l => new Array(l.vis || VIS_DEFAULT).fill(0));
    let outPreds = [], winIdx = -1, evoHistory = [], strokeHistory = [];
    let lastFull = 0, pending = false, dirty = false, sysText = '';
    let strokeCount = 0, lastStrokeTime = 0;
    const nnCtx = nnCv.getContext('2d'), evoCtx = evoCv.getContext('2d');
    const stCtx = stCv.getContext('2d');
    let sysTimer = null, layerPos = [], debounceTimer = null;

    canvasObj.onChange(() => {
        dirty = true;
        const now = Date.now();
        if (now - lastStrokeTime > 200) { strokeCount++; lastStrokeTime = now; }
        if (debounceTimer) clearTimeout(debounceTimer);
        debounceTimer = setTimeout(runInference, 50);
    });

    drawGradCAM(null);

    function reset() {
        nodeAct = ARCH.map(l => new Array(l.vis || VIS_DEFAULT).fill(0));
        outPreds = []; winIdx = -1; evoHistory = []; strokeHistory = []; strokeCount = 0;
        if (emb3d) emb3d.reset();
        bigPred.textContent = '?'; confTxt.textContent = '—';
        updateConfidenceBars(barsEl, []);
        ['fmap-body','cn-body','tta-grid'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.innerHTML = '<div class="lnn-placeholder">Draw to see</div>';
        });
        const rf = document.getElementById('rb-fill'); if (rf) rf.style.width = '0%';
        const rv = document.getElementById('rb-val'); if (rv) rv.textContent = '—';
        const cd = document.getElementById('cal-dial'); if (cd) cd.textContent = '—';
        const ci = document.getElementById('cal-info'); if (ci) ci.innerHTML = '<span class="lnn-placeholder">Draw to see</span>';
        const st = document.getElementById('ex-status'); if (st) st.textContent = 'Draw a letter or digit to begin';
        drawGradCAM(null);
    }

    function resize() {
        const dpr = window.devicePixelRatio || 1;
        [nnWrap, evoWrap, stWrap].forEach(w => {
            const cv = w.querySelector('canvas');
            if (!cv) return;
            const ww = w.clientWidth, hh = w.clientHeight;
            if (ww > 0 && hh > 0) {
                cv.width = ww * dpr; cv.height = hh * dpr;
                cv.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
            }
        });
        layerPos = computePositions(nnWrap.clientWidth, nnWrap.clientHeight, nodeAct);
    }

    const ro = new ResizeObserver(resize);
    [nnWrap, evoWrap, stWrap].forEach(w => ro.observe(w));
    setTimeout(resize, 120);

    async function pollSys() {
        try {
            const [g,s] = await Promise.all([fetch('/api/system/gpu'),fetch('/api/system/stats')]);
            const gpu = await g.json(), sys = await s.json();
            sysText = 'CPU:'+sys.cpu_percent+'% RAM:'+sys.ram_used_gb+'/'+sys.ram_total_gb+'GB';
            if (gpu.available) sysText += ' GPU:'+gpu.gpu_util_percent+'%';
        } catch {}
    }
    pollSys(); sysTimer = setInterval(pollSys, 5000);

    async function runInference() {
        if (_dead || pending) { if (pending) dirty = true; return; }
        const pixels = canvasObj.getPixels();
        if (pixels.reduce((a,b)=>a+b,0) < 0.5) { reset(); return; }
        pending = true;
        try {
            const r = await fetch('/api/explain/live', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({pixels}) });
            const live = await r.json();
            if (live.error) { pending = false; return; }
            const result = applyLiveData(live, nodeAct, layerPos);
            outPreds = result.outPreds; winIdx = result.winIdx;
            const preds = live.predictions || [];
            updateConfidenceBars(barsEl, preds);
            updateConfusion(preds);
            updateRobustness(preds);
            updateCalibration(preds);
            drawGradCAM(pixels);
            if (emb3d) emb3d.update(preds);
            const top = preds[0];
            if (top) { bigPred.textContent = top.display; confTxt.textContent = (top.confidence*100).toFixed(1)+'% confidence'; }
            evoHistory.push(preds); if (evoHistory.length > 60) evoHistory.shift();
            strokeHistory.push({conf:top?top.confidence:0, stroke:strokeCount, label:top?top.display:''});
            if (strokeHistory.length > 120) strokeHistory.shift();
            const stEl = document.getElementById('ex-status');
            if (stEl) stEl.textContent = top ? top.display+' · '+(top.confidence*100).toFixed(1)+'% · '+(live.inference_time_ms||0).toFixed(1)+'ms · '+sysText : '—';
            const now = Date.now();
            if (now - lastFull > 600) {
                lastFull = now;
                try {
                    const full = await (await fetch('/api/explain/full', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pixels})})).json();
                    if (!full.error) { updateFeatureMaps(full); if (full.input_image) { const p = document.getElementById('pp-proc'); if (p) p.src = 'data:image/png;base64,'+full.input_image; } }
                } catch {}
                try {
                    const db = await (await fetch('/api/inference/debug-preview', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pixels})})).json();
                    if (db.image_b64) { const r = document.getElementById('pp-raw'); if (r) r.src = 'data:image/png;base64,'+db.image_b64; }
                } catch {}
            }
        } catch (e) { console.error('Explain error:', e); }
        pending = false;
        if (dirty) { dirty = false; runInference(); }
    }

    function animate() {
        if (_dead) return;
        drawNN(nnCtx, nnWrap.clientWidth, nnWrap.clientHeight, layerPos, outPreds, winIdx);
        drawEvo(evoCtx, evoWrap, evoHistory);
        drawStrokeTimeline(stCtx, stWrap, strokeHistory);
        _af = requestAnimationFrame(animate);
    }
    animate();
}

function mk(tag, cls) { const e = document.createElement(tag); if (cls) e.className = cls; return e; }
function panel(title, sub) {
    const el = mk('div','lnn-panel');
    const hd = mk('div','lnn-panel-hd');
    hd.innerHTML = '<div class="lnn-panel-title">'+title+'</div>'+(sub?'<div class="lnn-panel-sub">'+sub+'</div>':'');
    el.appendChild(hd);
    const bd = mk('div','lnn-panel-bd'); el.appendChild(bd);
    return {el, body: bd};
}
