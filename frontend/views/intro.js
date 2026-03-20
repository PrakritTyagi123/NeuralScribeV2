/**
 * NeuralScribe v2 — Intro Page
 * Project introduction, workflow steps, and creator info.
 */
export async function renderIntro(container) {
    container.innerHTML = '';
    container.style.cssText = 'overflow-y:auto;padding:20px;';

    const card = document.createElement('div');
    card.style.cssText = 'max-width:900px;width:100%;';
    card.innerHTML = `
        <div style="text-align:center;margin-bottom:32px;">
            <h1 style="font-size:28px;font-weight:700;margin:0 0 4px 0;">NeuralScribe v2</h1>
            <p style="color:var(--muted);font-size:14px;margin:0;">Handwriting Recognition with Live Neural Network Visualization</p>
        </div>

        <div class="panel" style="margin-bottom:16px;">
            <div class="panel-header">About</div>
            <div class="panel-body" style="padding:12px;font-size:13px;line-height:1.6;">
                NeuralScribe v2 is an end-to-end handwriting recognition system that lets you
                train a convolutional neural network to recognize handwritten characters, then
                watch it think in real-time as you draw. The model uses a compact residual CNN
                architecture with Squeeze-and-Excitation attention, trained on the EMNIST dataset
                covering digits (0-9), uppercase letters (A-Z), and 11 lowercase letters — 47 classes total.
                <br><br>
                The live visualization shows signal flow through every layer, feature map activations,
                probability evolution, Grad-CAM saliency, robustness analysis, and more — all
                updating in real-time as your pen moves.
            </div>
        </div>

        <div class="panel" style="margin-bottom:16px;">
            <div class="panel-header">Workflow — Follow These Steps</div>
            <div class="panel-body" style="padding:12px;">
                <div style="display:flex;flex-direction:column;gap:12px;">
                    ${step(1, 'Download', 'Download the EMNIST dataset (~550 MB) and convert it to the standard image format. This is a one-time setup.', 'download')}
                    ${step(2, 'Data Prep', 'Preprocess the images (crop, align, smooth) and apply augmentation. Adjust augmentation factor to control dataset size.', 'dataPrep')}
                    ${step(3, 'Training', 'Train the neural network. Configure epochs, batch size, and watch loss/accuracy graphs update live.', 'training')}
                    ${step(4, 'Models', 'View saved model checkpoints, load models for inference, export to ONNX format.', 'modelManager')}
                    ${step(5, 'Live View', 'Draw characters on the canvas and watch the neural network recognize them in real-time with full visualization.', 'explainability')}
                </div>
            </div>
        </div>

        <div class="panel" style="margin-bottom:16px;">
            <div class="panel-header">Technical Details</div>
            <div class="panel-body" style="padding:12px;font-size:12px;line-height:1.6;color:var(--muted);">
                <strong>Architecture:</strong> Residual CNN with SE attention — Stem Conv → [ResBlock + SE] × 4 → Global Avg Pool → Linear(36)<br>
                <strong>Dataset:</strong> EMNIST Balanced — 36 classes (digits + uppercase letters), ~100K samples<br>
                <strong>Training:</strong> AdamW optimizer, cosine annealing with warm restarts, mixup regularization, mixed precision<br>
                <strong>Stack:</strong> PyTorch (backend ML), FastAPI (API server), vanilla JS (frontend), WebSocket (real-time updates)
            </div>
        </div>

        <div style="text-align:center;font-size:12px;color:var(--muted);padding:16px 0;">
            Created by <strong>Prathamesh Minde</strong>
        </div>
    `;
    container.appendChild(card);

    // Make step buttons navigate
    card.querySelectorAll('[data-goto]').forEach(btn => {
        btn.addEventListener('click', () => {
            const { navigateTo } = require_router();
            navigateTo(btn.dataset.goto);
        });
    });
}

function step(num, title, desc, gotoView) {
    return `
        <div style="display:flex;gap:12px;align-items:flex-start;">
            <div style="width:28px;height:28px;border-radius:50%;border:2px solid var(--fg);display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px;flex-shrink:0;">${num}</div>
            <div style="flex:1;">
                <div style="font-weight:600;font-size:13px;margin-bottom:2px;">${title}</div>
                <div style="font-size:12px;color:var(--muted);line-height:1.5;">${desc}</div>
            </div>
        </div>
    `;
}

function require_router() {
    // Dynamic import workaround for click handlers
    return { navigateTo: (v) => {
        document.querySelector(`.nav-item[data-view="${v}"]`)?.click();
    }};
}
