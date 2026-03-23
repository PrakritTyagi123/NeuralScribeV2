# NeuralScribe v2 — Build & Distribution Guide

## Prerequisites

1. **Python 3.10+** with your venv activated
2. **All pip dependencies installed** (`pip install -r requirements.txt`)
3. **Nuitka**: `pip install nuitka ordered-set zstandard`
4. **Inno Setup** (optional, for installer): https://jrsoftware.org/isdl.php
5. **An icon file**: Save a 256x256 `.ico` file as `icon.ico` in the project root
   - If you don't have one, remove `--windows-icon-from-ico=icon.ico` from the build command

## Build Steps

### Step 1: Install build tools

```
pip install nuitka ordered-set zstandard eel
```

### Step 2: Compile with Nuitka

**Option A — Batch file:**
```
build.bat
```

**Option B — Terminal command:**
```
python -m nuitka --standalone --output-dir=dist --output-filename=NeuralScribe.exe --windows-console-mode=attach --enable-plugin=torch --include-data-dir=frontend=frontend --include-data-dir=configs=configs --include-data-files=project_config.json=project_config.json --include-module=uvicorn --include-module=uvicorn.logging --include-module=uvicorn.protocols --include-module=uvicorn.protocols.http --include-module=uvicorn.protocols.http.httptools_impl --include-module=uvicorn.protocols.http.h11_impl --include-module=uvicorn.protocols.websockets --include-module=uvicorn.protocols.websockets.websockets_impl --include-module=uvicorn.lifespan --include-module=uvicorn.lifespan.on --include-module=fastapi --include-module=pydantic --include-module=eel --include-module=pynvml --include-module=psutil --include-module=yaml --include-module=PIL --include-module=scipy --include-module=matplotlib --include-module=torchvision --include-module=onnx --include-module=onnxruntime launcher.py
```

This takes 10-30 minutes. Output: `dist/launcher.dist/NeuralScribe.exe` (~2-3 GB folder)

### Step 3: Test the build

```
cd dist\launcher.dist
NeuralScribe.exe
```

Should open a Chrome/Edge window with NeuralScribe. Test all pages.

### Step 4: Ship a pre-trained model (recommended)

Copy your trained model so the recipient can skip Download/Prep/Training:

```
copy models\english\best_model.pth dist\launcher.dist\models\english\
copy models\english\index.json dist\launcher.dist\models\english\
```

Optional — copy cached dataset (so they can retrain without downloading):
```
copy datasets\english\cache\cached_dataset.pt dist\launcher.dist\datasets\english\cache\
```

### Step 5: Create installer (optional)

1. Install Inno Setup from https://jrsoftware.org/isdl.php
2. Open `installer.iss` in Inno Setup Compiler
3. Click Build > Compile
4. Output: `installer_output/NeuralScribeSetup.exe` (~1-1.5 GB compressed)

## What the user gets

1. Double-click `NeuralScribeSetup.exe`
2. Standard Windows installer (Next, Next, Install)
3. Desktop shortcut: "NeuralScribe"
4. Click shortcut → console window opens → Chrome/Edge app window opens
5. Close the console window to stop

## Notes

- **CUDA**: The build includes CUDA runtime from your PyTorch installation. The recipient needs a compatible NVIDIA driver (535+), but NOT a separate CUDA install.
- **Chrome/Edge**: Eel tries Chrome first, then Edge, then default browser. At least one must be installed.
- **Antivirus**: Large unsigned exes may trigger Windows Defender. The recipient may need to allow it. For production, consider code signing with a certificate.
- **First run**: If you didn't ship the dataset, EMNIST download (~550 MB) happens on first use. The recipient needs internet for this.
- **Code protection**: Nuitka compiles Python to C, then to native machine code. Source `.py` files are NOT bundled — only compiled binaries exist in the output.

## Folder structure after install

```
C:\Program Files\NeuralScribe\
  NeuralScribe.exe
  frontend\          (HTML, CSS, JS)
  configs\           (YAML configs)
  datasets\          (downloaded data)
  models\            (trained models)
  training_logs\     (logs)
  torch\, ...        (bundled dependencies)
```