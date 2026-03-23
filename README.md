# NeuralScribe v2

**Handwriting Recognition with Live Neural Network Visualization**

![NeuralScribe Banner](screenshots/banner.png)
<!-- Take a screenshot of the Live View page with a drawn character and all panels active. Save as screenshots/banner.png -->

---

## Overview

NeuralScribe v2 is an end-to-end handwriting recognition system that lets you train a convolutional neural network to recognize handwritten characters, then watch it think in real-time as you draw. The model uses a compact residual CNN architecture with Squeeze-and-Excitation attention blocks, trained on the EMNIST Balanced dataset.

**47 classes**: digits (0-9), uppercase letters (A-Z), and 11 lowercase letters (a, b, d, e, f, g, h, n, q, r, t)

![Live View Demo](screenshots/demo.gif)
<!-- Record a GIF of: drawing a letter on the canvas → all panels lighting up in real-time. Use ScreenToGif or similar tool. Save as screenshots/demo.gif -->

---

## Features

### Full Training Pipeline
- One-click EMNIST dataset download (~550 MB via torchvision)
- Configurable preprocessing: bounding box crop, center-of-mass alignment, Gaussian smoothing
- Adjustable data augmentation (rotation, translation, scaling, shearing)
- Real-time training with live loss/accuracy charts and epoch history
- Automatic model checkpointing with best model tracking
- ONNX export support

### Live Neural Network Visualization
Draw a character and watch the model think in real-time:

| Panel | Description |
|-------|-------------|
| **NN Diagram** | Signal flow through all layers — node brightness = activation strength |
| **Grad-CAM** | Real gradient-weighted saliency — where the model actually looks |
| **Feature Maps** | Top activation channels from each convolutional block |
| **Probability Evolution** | How confidence changes as you draw each stroke |
| **Robustness** | Real test-time augmentation — stability across rotations and shifts |
| **3D Embedding** | Interactive 3D scatter plot + radar chart of prediction confidence |
| **Stroke Timeline** | Confidence tracking over time with stroke markers |
| **Confusion** | Top-3 competing predictions the model is deciding between |
| **Calibration** | Top-1 vs Top-2 confidence gap analysis |
| **Preprocessing** | Side-by-side: your input vs what the model actually sees |

### Desktop Application
- Native window via Eel (Chrome/Edge app mode — no address bar)
- GPU-accelerated training and inference (CUDA)
- Real-time system monitoring (CPU, RAM, GPU utilization)
- Compilable to standalone .exe via Nuitka

---

## Screenshots

### Intro Page
![Intro](screenshots/intro.png)
<!-- Screenshot of the Intro page -->

### Download Page
![Download](screenshots/download.png)
<!-- Screenshot showing EMNIST download in progress or completed -->

### Data Preparation
![Data Prep](screenshots/dataprep.png)
<!-- Screenshot showing Data Prep with progress bars active -->

### Training
![Training](screenshots/training.png)
<!-- Screenshot showing training in progress with loss/accuracy charts -->

### Model Manager
![Models](screenshots/models.png)
<!-- Screenshot showing saved models list -->

### Live Neural Network View
![Live View](screenshots/liveview.png)
<!-- Screenshot of Live View with all panels active after drawing a character -->

### 3D Embedding Space
![Embedding 3D](screenshots/embedding3d.png)
<!-- Screenshot of the 3D scatter mode -->

![Embedding Radar](screenshots/embedding_radar.png)
<!-- Screenshot of the radar mode -->

---

## Architecture

```
Input (28×28 grayscale)
  │
  ├─ Stem Conv (1 → 32 channels)
  │
  ├─ ResBlock 0 + SE (32 → 64)
  ├─ ResBlock 1 + SE (64 → 128)
  ├─ ResBlock 2 + SE (128 → 256)
  ├─ ResBlock 3 + SE (256 → 320)
  │
  ├─ Global Average Pooling (320-dim vector)
  │
  └─ Linear Classifier → 47 classes
```

**Total parameters**: ~3M

**Key components**:
- Residual connections with pre-activation BatchNorm
- Squeeze-and-Excitation attention (channel-wise recalibration)
- Drop path regularization
- Mixup training
- Mixed precision (FP16) on GPU
- Cosine annealing with warm restarts

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Framework | PyTorch + torchvision |
| Backend API | FastAPI + Uvicorn |
| Real-time Updates | WebSocket |
| Frontend | Vanilla JavaScript (no framework) |
| 3D Visualization | Three.js |
| Desktop Window | Eel (Chrome/Edge app mode) |
| GPU Monitoring | pynvml |
| Build Tool | Nuitka (compiles to native C) |
| Installer | Inno Setup |

---

## Quick Start (Development)

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA-compatible drivers (535+)
- Chrome or Edge browser

### Setup

```bash
git clone https://github.com/yourusername/NeuralScribeV2.git
cd NeuralScribeV2
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Important**: Install PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Run

```bash
python launcher.py
```

Or without Eel (opens in default browser):
```bash
python run_backend.py
```

### Workflow

1. **Download** — Click "Download EMNIST" (one-time, ~550 MB)
2. **Data Prep** — Set augmentation factor (3 recommended), click Start
3. **Training** — Set epochs (10+), click Start, watch live charts
4. **Models** — Load the best model
5. **Live View** — Draw and watch the neural network think

---

## Building the Executable

### Prerequisites
```bash
pip install nuitka ordered-set zstandard eel
```

### Compile

```
python -m nuitka --standalone --output-dir=dist --output-filename=NeuralScribe.exe --windows-console-mode=attach --enable-plugin=torch --include-data-dir=frontend=frontend --include-data-dir=configs=configs --include-data-files=project_config.json=project_config.json --include-module=uvicorn --include-module=uvicorn.logging --include-module=uvicorn.protocols --include-module=uvicorn.protocols.http --include-module=uvicorn.protocols.http.httptools_impl --include-module=uvicorn.protocols.http.h11_impl --include-module=uvicorn.protocols.websockets --include-module=uvicorn.protocols.websockets.websockets_impl --include-module=uvicorn.lifespan --include-module=uvicorn.lifespan.on --include-module=fastapi --include-module=pydantic --include-module=eel --include-module=pynvml --include-module=psutil --include-module=yaml --include-module=PIL --include-module=scipy --include-module=matplotlib --include-module=torchvision --include-module=onnx --include-module=onnxruntime launcher.py
```

Takes 10-30 minutes. Output: `dist/launcher.dist/NeuralScribe.exe`

Or use the batch file:
```
build.bat
```

### Ship a pre-trained model

```bash
copy models\english\best_model.pth dist\launcher.dist\models\english\
copy models\english\index.json dist\launcher.dist\models\english\
```

### Create Windows installer

1. Install [Inno Setup](https://jrsoftware.org/isdl.php)
2. Open `installer.iss` in Inno Setup Compiler
3. Build → Compile
4. Output: `installer_output/NeuralScribeSetup.exe` (~1-1.5 GB)

---

## Project Structure

```
NeuralScribeV2/
├── launcher.py              # Entry point (Eel window + server)
├── run_backend.py           # Server-only entry point
├── project_config.json      # Project state
├── requirements.txt
├── build.bat                # Nuitka build script
├── installer.iss            # Inno Setup config
├── .gitignore
│
├── backend/
│   ├── api/
│   │   ├── app.py           # FastAPI application
│   │   ├── ws.py            # WebSocket handler
│   │   └── routes/
│   │       ├── dataset.py   # Download + prepare endpoints
│   │       ├── training.py  # Train start/stop/pause
│   │       ├── models.py    # Model CRUD + export
│   │       ├── inference.py # Prediction endpoint
│   │       ├── explainability.py  # Grad-CAM, robustness, feature maps
│   │       └── system.py    # GPU, stats, shutdown
│   │
│   ├── ml/
│   │   ├── model.py         # NeuralScribeNet (ResCNN + SE)
│   │   ├── losses.py        # Label smoothing, focal loss
│   │   ├── augmentation.py  # Augmentation pipeline + TTA
│   │   ├── preprocess.py    # Crop, align, smooth
│   │   ├── synthetic.py     # Synthetic data generation
│   │   └── dataset_downloader.py  # EMNIST auto-download
│   │
│   ├── services/
│   │   ├── dataset_service.py    # Data pipeline orchestrator
│   │   ├── training_service.py   # Training loop + checkpointing
│   │   ├── model_service.py      # Model loading/saving
│   │   ├── interface_service.py  # Inference + Grad-CAM + robustness
│   │   └── system_service.py     # System monitoring
│   │
│   └── utils/
│       ├── config.py        # YAML config, paths, registry
│       ├── gpu_monitor.py   # NVIDIA GPU stats
│       ├── helpers.py       # Utilities
│       ├── logging.py       # Logging setup
│       └── metrics.py       # Accuracy, confusion matrix
│
├── frontend/
│   ├── index.html           # SPA shell (SVG sidebar icons)
│   ├── style.css            # Core styles (black & white wireframe)
│   ├── lnn.css              # Live NN page layout
│   │
│   ├── components/
│   │   ├── canvas.js        # Drawing canvas
│   │   ├── chart.js         # Line chart renderer
│   │   ├── confidenceBars.js
│   │   ├── embedding3d.js   # Three.js 3D scatter + 2D radar
│   │   ├── featureMapViewer.js
│   │   ├── lnnCharts.js     # Prob evolution, stroke timeline
│   │   ├── lnnPanels.js     # Grad-CAM, feature maps, robustness, calibration
│   │   ├── nnDiagram.js     # Neural network signal flow diagram
│   │   ├── progressBar.js
│   │   ├── toast.js
│   │   └── ...
│   │
│   ├── views/
│   │   ├── intro.js         # Welcome page
│   │   ├── download.js      # EMNIST download
│   │   ├── dataPrep.js      # Preprocessing + augmentation
│   │   ├── training.js      # Training with live charts
│   │   ├── modelManager.js  # Model list + actions
│   │   ├── explainability.js # Live NN visualization
│   │   └── setting.js       # System settings
│   │
│   └── src/
│       ├── main.js          # App entry point
│       ├── router.js        # SPA routing
│       ├── ws.js            # WebSocket client
│       └── theme.js         # Theme manager
│
├── configs/languages/english/
│   ├── class_registry.yaml  # 47 class definitions
│   ├── prep.yaml            # Preprocessing config
│   └── train.yaml           # Training hyperparameters
│
├── datasets/english/        # (gitignored)
│   ├── raw/                 # EMNIST download
│   └── cache/               # Preprocessed .pt cache
│
├── models/english/          # (gitignored)
│   ├── best_model.pth       # Best checkpoint
│   └── exports/             # ONNX exports
│
└── training_logs/english/   # (gitignored)
    ├── training_history.json
    ├── loss_graph.json
    └── accuracy_graph.json
```

---

## Configuration

### Training Hyperparameters (`configs/languages/english/train.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.epochs` | 10 | Number of training epochs |
| `training.batch_size` | 1024 | Batch size (adjust for GPU memory) |
| `optimizer.lr` | 0.002 | Learning rate |
| `optimizer.weight_decay` | 0.0001 | L2 regularization |
| `loss.label_smoothing` | 0.04 | Label smoothing factor |
| `regularization.mixup_alpha` | 0.15 | Mixup interpolation strength |
| `model.dropout` | 0.2 | Dropout rate |
| `model.block_channels` | [64, 128, 256, 320] | Channels per residual block |

### Preprocessing (`configs/languages/english/prep.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `augmentation.precompute_factor` | 3 | Augmentation multiplier (0 = none) |
| `augmentation.rotation_range` | [-12, 12] | Random rotation degrees |
| `augmentation.scale_range` | [0.9, 1.1] | Random scaling |
| `preprocessing.center_of_mass` | true | Center character by mass |
| `preprocessing.smoothing.sigma` | 0.5 | Gaussian smoothing |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/system/gpu` | GPU status |
| GET | `/api/system/stats` | CPU/RAM/disk stats |
| POST | `/api/dataset/download` | Start EMNIST download |
| GET | `/api/dataset/download-status` | Check download status |
| POST | `/api/dataset/prepare` | Start preprocessing |
| POST | `/api/training/start` | Start training |
| POST | `/api/training/stop` | Stop training |
| GET | `/api/training/status` | Training progress |
| GET | `/api/models/list` | List saved models |
| POST | `/api/models/load` | Load model for inference |
| POST | `/api/inference/predict` | Run prediction |
| POST | `/api/explain/live` | Live layer activations |
| POST | `/api/explain/full` | Full feature maps |
| POST | `/api/explain/gradcam` | Grad-CAM saliency map |
| POST | `/api/explain/robustness` | TTA robustness test |
| WS | `/ws` | Real-time training/progress updates |

---

## Adding Screenshots

Create a `screenshots/` folder in the project root and add:

| Filename | What to capture |
|----------|----------------|
| `banner.png` | Live View with a drawn character, all panels active |
| `demo.gif` | Record drawing a letter → panels lighting up (use ScreenToGif) |
| `intro.png` | The Intro page |
| `download.png` | Download page with progress bar |
| `dataprep.png` | Data Prep with progress bars active |
| `training.png` | Training in progress with charts |
| `models.png` | Model Manager with a saved model |
| `liveview.png` | Full Live View page |
| `embedding3d.png` | 3D Scatter mode |
| `embedding_radar.png` | Radar mode |

---

## License

See [LICENSE](LICENSE) for details.

---

## Author

**Prakrit Tyagi**

---