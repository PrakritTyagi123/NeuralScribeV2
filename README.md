# NeuralScribe v2

**Handwriting Recognition with Live Neural Network Visualization**

![NeuralScribe Banner](screenshots/banner.png)

NeuralScribe v2 is an end-to-end handwriting recognition system. Train a convolutional neural network on the EMNIST dataset, then draw characters and watch the model think in real-time — every layer activation, every probability shift, every decision the network makes, visualized live as your pen moves.

![Demo](screenshots/demo.gif)

---

## What it does

You draw a letter or digit. The neural network recognizes it. But instead of just showing you the answer, NeuralScribe opens up the black box and shows you *everything* happening inside — signal flow through every layer, which pixels the model is focusing on (real Grad-CAM), how confident it is across perturbations, and how its predictions evolve stroke by stroke.

**47 classes**: digits 0-9, uppercase A-Z, and 11 lowercase letters (a, b, d, e, f, g, h, n, q, r, t)

---

## The Pipeline

NeuralScribe walks you through the full ML pipeline in order:

### 1. Download

One-click EMNIST download (~550 MB). Progress bar updates in real-time.

![Download Page](screenshots/download.png)

### 2. Data Preparation

Preprocess raw images (crop, center-of-mass alignment, smoothing) and apply configurable augmentation. Set the multiplier — 3× recommended for good results.

![Data Prep](screenshots/dataprep.png)

### 3. Training

Train with live loss and accuracy charts. Configure epochs and batch size. The epoch history table fills in as training progresses — exportable to CSV.

![Training](screenshots/training.png)

GPU utilization is tracked in the header throughout.

![Training Charts](screenshots/training_charts.png)

### 4. Model Manager

All checkpoints saved automatically. Load any model, export to ONNX, download weights, or delete old runs.

![Models](screenshots/models.png)

### 5. Live Neural Network View

The main event. Draw on the canvas and watch 10+ visualization panels update in real-time:

![Live View Full](screenshots/liveview_full.png)

**Signal flow through the network** — node brightness maps to actual layer activations:

![NN Diagram](screenshots/nn_diagram.png)

**Grad-CAM saliency** — real gradient-weighted activation mapping showing where the model looks, not just where ink is:

![Grad-CAM](screenshots/gradcam.png)

**Feature maps** — top activation channels from each convolutional block:

![Feature Maps](screenshots/feature_maps.png)

**Probability evolution** — how the model's confidence shifts as you draw each stroke:

![Prob Evolution](screenshots/prob_evolution.png)

**3D Embedding Space** — interactive Three.js scatter plot. Switch to radar mode for a flat confidence chart:

![3D Scatter](screenshots/embedding_3d.png)

![Radar Mode](screenshots/embedding_radar.png)

**Robustness testing** — real test-time augmentation. The model runs 5 perturbed versions (rotations, shifts) and reports stability:

![Robustness](screenshots/robustness.png)

**Preprocessing comparison** — your raw input vs what the model actually sees after crop, align, and smooth:

![Preprocessing](screenshots/preprocessing.png)

---

## Architecture

```
Input (28×28 grayscale)
  │
  ├─ Stem Conv (1 → 32ch)
  │
  ├─ ResBlock 0 + SE Attention (32 → 64ch)
  ├─ ResBlock 1 + SE Attention (64 → 128ch)
  ├─ ResBlock 2 + SE Attention (128 → 256ch)
  ├─ ResBlock 3 + SE Attention (256 → 320ch)
  │
  ├─ Global Average Pooling → 320-dim vector
  │
  └─ Linear → 47 classes
```

~3M parameters. Residual connections, Squeeze-and-Excitation channel attention, drop path regularization, mixup training, mixed precision (FP16) on GPU, cosine annealing with warm restarts.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML | PyTorch + torchvision |
| API | FastAPI + Uvicorn + WebSocket |
| Frontend | Vanilla JS (no framework) |
| 3D Viz | Three.js |
| Desktop | Eel (Chrome/Edge app mode) |
| GPU | pynvml + CUDA |
| Build | Nuitka → native binary |
| Installer | Inno Setup |

---

## Quick Start

### Requirements
- Python 3.10+
- NVIDIA GPU with drivers 535+
- Chrome or Edge

### Install

```bash
git clone https://github.com/yourusername/NeuralScribeV2.git
cd NeuralScribeV2
python -m venv .venv
.venv\Scripts\activate

# PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Everything else
pip install -r requirements.txt
```

### Run

```bash
python launcher.py
```

A Chrome/Edge window opens automatically. If Eel isn't installed, falls back to default browser.

For server-only mode (no Eel window):
```bash
python run_backend.py
# Open http://localhost:8000
```

![Intro Page](screenshots/intro.png)

---

## Building a Distributable .exe

Compile Python to native machine code with Nuitka. Source files are NOT included in the output — only compiled binaries.

### Install build tools

```bash
pip install nuitka ordered-set zstandard eel
```

### Compile

```
python -m nuitka --standalone --output-dir=dist --output-filename=NeuralScribe.exe --windows-console-mode=attach --enable-plugin=torch --include-data-dir=frontend=frontend --include-data-dir=configs=configs --include-data-files=project_config.json=project_config.json --include-module=uvicorn --include-module=uvicorn.logging --include-module=uvicorn.protocols --include-module=uvicorn.protocols.http --include-module=uvicorn.protocols.http.httptools_impl --include-module=uvicorn.protocols.http.h11_impl --include-module=uvicorn.protocols.websockets --include-module=uvicorn.protocols.websockets.websockets_impl --include-module=uvicorn.lifespan --include-module=uvicorn.lifespan.on --include-module=fastapi --include-module=pydantic --include-module=eel --include-module=pynvml --include-module=psutil --include-module=yaml --include-module=PIL --include-module=scipy --include-module=matplotlib --include-module=torchvision --include-module=onnx --include-module=onnxruntime launcher.py
```

Takes 10-30 minutes. Output: `dist/launcher.dist/`

Or use the batch file:
```bash
build.bat
```

### Ship with a trained model

```bash
copy models\english\best_model.pth dist\launcher.dist\models\english\
copy models\english\index.json dist\launcher.dist\models\english\
```

### Create Windows installer

1. Install [Inno Setup](https://jrsoftware.org/isdl.php)
2. Open `installer.iss` → Build → Compile
3. Output: `installer_output/NeuralScribeSetup.exe`

The recipient double-clicks the installer, gets a desktop shortcut, clicks it, and NeuralScribe opens.

---

## Configuration

Training hyperparameters live in `configs/languages/english/train.yaml`. Key settings:

| Parameter | Default | What it does |
|-----------|---------|-------------|
| `training.epochs` | 10 | Training epochs |
| `training.batch_size` | 1024 | Batch size (lower if GPU OOM) |
| `optimizer.lr` | 0.002 | Learning rate |
| `loss.label_smoothing` | 0.04 | Smoothing factor |
| `regularization.mixup_alpha` | 0.15 | Mixup strength |
| `model.dropout` | 0.2 | Dropout rate |
| `model.block_channels` | [64,128,256,320] | Channels per block |

Preprocessing config in `configs/languages/english/prep.yaml`:

| Parameter | Default | What it does |
|-----------|---------|-------------|
| `augmentation.precompute_factor` | 3 | Augmentation multiplier |
| `augmentation.rotation_range` | [-12, 12] | Random rotation range |
| `preprocessing.center_of_mass` | true | Center by mass |
| `preprocessing.smoothing.sigma` | 0.5 | Gaussian smooth |

---

## Project Structure

```
NeuralScribeV2/
├── launcher.py                  # Eel window + server
├── run_backend.py               # Server-only entry
├── build.bat                    # Nuitka build
├── installer.iss                # Inno Setup config
├── requirements.txt
├── .gitignore
│
├── backend/
│   ├── api/                     # FastAPI app + routes + WebSocket
│   ├── ml/                      # Model, losses, augmentation, preprocessing
│   └── services/                # Dataset, training, model, inference, system
│
├── frontend/
│   ├── index.html               # SPA shell
│   ├── style.css / lnn.css      # Styles
│   ├── components/              # Canvas, charts, 3D embedding, NN diagram
│   ├── views/                   # 7 pages (intro → settings)
│   └── src/                     # Router, WebSocket, state
│
├── configs/languages/english/   # Class registry + YAML configs
├── datasets/                    # Downloaded + cached data (gitignored)
├── models/                      # Checkpoints (gitignored)
└── training_logs/               # Training history (gitignored)
```

---

## License

See [LICENSE](LICENSE) for details.

---

**Built by Prakrit Tyagi**