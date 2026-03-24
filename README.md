# NeuralScribe v2

**Handwriting Recognition with Live Neural Network Visualization**

![NeuralScribe Banner](screenshots/banner.png)

NeuralScribe v2 is an end-to-end handwriting recognition system. Train a convolutional neural network on the EMNIST dataset, then draw characters and watch the model think in real-time — every layer activation, every probability shift, every decision the network makes, visualized live as your pen moves.

**47 classes**: digits 0-9, uppercase A-Z, and 11 lowercase letters (a, b, d, e, f, g, h, n, q, r, t)

![Demo](screenshots/demo.gif)

---

## Features

- Full ML pipeline: download → preprocess → augment → train → infer
- Real-time neural network visualization with 10+ interactive panels
- Real Grad-CAM saliency (gradient-based, not fake heatmaps)
- Real robustness testing via test-time augmentation
- 3D embedding space with scatter/radar toggle (Three.js)
- GPU-accelerated training and inference (CUDA)
- Native desktop window (Eel + Chrome/Edge app mode)
- Auto-shutdown when browser window closes
- Compilable to standalone .exe (Nuitka) with Windows installer (Inno Setup)

---

## Installation

### Requirements

- Windows 10/11
- Python 3.10+
- NVIDIA GPU with drivers 535+ (for CUDA training)
- Chrome or Edge browser

### Setup

```bash
git clone https://github.com/yourusername/NeuralScribeV2.git
cd NeuralScribeV2
python -m venv .venv
.venv\Scripts\activate
```

Install PyTorch with CUDA support:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Install everything else:

```bash
pip install -r requirements.txt
```

### Run

```bash
python launcher.py
```

A Chrome/Edge window opens automatically. Close the window to stop the server.

For server-only mode (opens in default browser, auto-shuts down after 10s idle):

```bash
python run_backend.py
```

---

## Walkthrough

### Step 1: Intro

The landing page. Shows project overview, workflow steps, and technical details.

![Intro Page](screenshots/intro.png)

### Step 2: Download EMNIST

Click "Download EMNIST". The progress bar updates in real-time as torchvision fetches the dataset (~550 MB). This is a one-time download.

![Download Page](screenshots/download.png)

![Download Complete](screenshots/download_done.png)

### Step 3: Data Preparation

Configure the augmentation factor (3× recommended) and click "Start Preparation". Four progress bars track each stage: Loading → Preprocessing → Augmentation → Split & Cache.

When you reopen the app, the page remembers that data is already prepared — all bars show "Complete" and the cache size is displayed.

![Data Prep Running](screenshots/dataprep_running.png)

![Data Prep Complete](screenshots/dataprep_done.png)

### Step 4: Training

Set epochs and batch size, then click Start. Loss and accuracy charts update live after each epoch. The epoch history table fills in with every metric — scrollable even with 100+ epochs.

![Training Start](screenshots/training_start.png)

![Training Progress](screenshots/training_progress.gif)

![Training Charts](screenshots/training_charts.png)

GPU utilization is tracked in the header bar throughout training.

### Step 5: Model Manager

All model checkpoints are saved automatically. Click any model to see:
- Model metadata (parameters, accuracy, loss, date)
- Loss and accuracy history charts
- Full epoch table (exportable to CSV)

Load a model to use it in Live View. Export to ONNX for deployment.

![Models Page](screenshots/models.png)

![Model Detail](screenshots/model_detail.png)

### Step 6: Live Neural Network View

The main event. Load a model, draw on the canvas, and watch 10+ panels update in real-time.

![Live View Full](screenshots/liveview_full.png)

![Live View Demo](screenshots/liveview_demo.gif)

#### Draw & Predict

Draw a character on the canvas. The model predicts in real-time with confidence scores for the top-5 candidates.

![Draw Panel](screenshots/draw_predict.png)

#### Neural Network Diagram

Signal flow through every layer. Node brightness = actual activation strength from the model's forward pass.

![NN Diagram](screenshots/nn_diagram.png)

#### Grad-CAM Saliency

Real gradient-weighted class activation mapping. Shows which pixels the model is actually looking at — computed using PyTorch autograd, not a fake heatmap of the input.

![Grad-CAM](screenshots/gradcam.png)

#### Feature Maps

Top activation channels from each convolutional block, ranked by importance.

![Feature Maps](screenshots/feature_maps.png)

#### Probability Evolution

How the model's confidence across top-5 predictions shifts as you draw each stroke.

![Probability Evolution](screenshots/prob_evolution.png)

#### 3D Embedding Space

Interactive Three.js visualization. Toggle between 3D scatter (drag to rotate) and radar chart mode. Cluster size and position reflect actual prediction confidence.

![3D Scatter](screenshots/embedding_3d.png)

![Radar Mode](screenshots/embedding_radar.png)

#### Robustness Testing

Real test-time augmentation — the model runs 5 perturbed versions (±5° rotation, ±1px shift) and reports how stable the prediction is. Green = agrees with original, red = different prediction.

![Robustness](screenshots/robustness.png)

#### Other Panels

- **Preprocessing**: Side-by-side comparison of your raw input vs what the model actually sees after crop, center-of-mass alignment, and smoothing
- **Confusion**: Top-3 competing predictions the model is deciding between
- **Calibration**: Top-1 vs Top-2 confidence gap analysis (decisive vs uncertain)
- **Stroke Timeline**: Confidence over time with stroke event markers

![Other Panels](screenshots/other_panels.png)

### Step 7: Settings

System info, CUDA status, PyTorch version, GPU details.

![Settings](screenshots/settings.png)

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

~3M parameters. Residual connections, Squeeze-and-Excitation channel attention, drop path regularization, mixup training, mixed precision (FP16), cosine annealing with warm restarts.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML | PyTorch + torchvision |
| API | FastAPI + Uvicorn + WebSocket |
| Frontend | Vanilla JavaScript |
| 3D Visualization | Three.js |
| Desktop Window | Eel (Chrome/Edge app mode) |
| GPU Monitoring | pynvml + CUDA |
| Build | Nuitka (compiles to native C) |
| Installer | Inno Setup |

---

## Building a Distributable .exe

Compile to native machine code. Source `.py` files are NOT included in the output.

### Install build tools

```bash
pip install nuitka ordered-set zstandard eel
```

### Compile

```
python -m nuitka --standalone --output-dir=dist --output-filename=NeuralScribe.exe --windows-console-mode=attach --enable-plugin=torch --include-data-dir=frontend=frontend --include-data-dir=configs=configs --include-data-files=project_config.json=project_config.json --include-module=uvicorn --include-module=uvicorn.logging --include-module=uvicorn.protocols --include-module=uvicorn.protocols.http --include-module=uvicorn.protocols.http.httptools_impl --include-module=uvicorn.protocols.http.h11_impl --include-module=uvicorn.protocols.websockets --include-module=uvicorn.protocols.websockets.websockets_impl --include-module=uvicorn.lifespan --include-module=uvicorn.lifespan.on --include-module=fastapi --include-module=pydantic --include-module=eel --include-module=pynvml --include-module=psutil --include-module=yaml --include-module=PIL --include-module=scipy --include-module=matplotlib --include-module=torchvision --include-module=onnx --include-module=onnxruntime launcher.py
```

Takes 10-30 minutes. Output: `dist/launcher.dist/`

### Ship with trained model

```bash
copy models\english\best_model.pth dist\launcher.dist\models\english\
copy models\english\index.json dist\launcher.dist\models\english\
mkdir dist\launcher.dist\datasets\english\cache
mkdir dist\launcher.dist\datasets\english\raw
mkdir dist\launcher.dist\training_logs\english
```

### Create Windows installer

1. Install [Inno Setup](https://jrsoftware.org/isdl.php)
2. Open `installer.iss` in Inno Setup Compiler
3. Build → Compile
4. Output: `installer_output/NeuralScribeSetup.exe`

The recipient double-clicks the installer → Next → Install → desktop shortcut → done.

---

## Configuration

Training hyperparameters: `configs/languages/english/train.yaml`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.epochs` | 10 | Number of epochs |
| `training.batch_size` | 1024 | Batch size |
| `optimizer.lr` | 0.002 | Learning rate |
| `loss.label_smoothing` | 0.04 | Smoothing factor |
| `regularization.mixup_alpha` | 0.15 | Mixup strength |
| `model.dropout` | 0.2 | Dropout rate |

Preprocessing: `configs/languages/english/prep.yaml`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `augmentation.precompute_factor` | 3 | Augmentation multiplier |
| `augmentation.rotation_range` | [-12, 12] | Random rotation |
| `preprocessing.center_of_mass` | true | Center by mass |
| `preprocessing.smoothing.sigma` | 0.5 | Gaussian smooth |

---

## Project Structure

```
NeuralScribeV2/
├── launcher.py              # Eel window + server (auto-shutdown)
├── run_backend.py           # Server-only (auto-shutdown default 10s)
├── build.bat                # Nuitka build script
├── installer.iss            # Inno Setup config
├── requirements.txt
├── .gitignore
│
├── backend/
│   ├── api/                 # FastAPI app, routes, WebSocket
│   ├── ml/                  # Model, losses, augmentation, preprocessing
│   └── services/            # Dataset, training, model, inference, system
│
├── frontend/
│   ├── index.html           # SPA shell (SVG icons)
│   ├── style.css / lnn.css  # Styles
│   ├── components/          # Canvas, charts, 3D embedding, NN diagram
│   ├── views/               # 7 pages
│   └── src/                 # Router, WebSocket, state
│
├── configs/                 # Class registry + YAML configs
├── datasets/                # Downloaded + cached data (gitignored)
├── models/                  # Checkpoints (gitignored)
└── training_logs/           # History (gitignored)
```

---

## License

See [LICENSE](LICENSE) for details.

---

**Built by Prakrit Tyagi**