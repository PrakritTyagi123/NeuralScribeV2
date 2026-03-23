# NeuralScribe v2 — Build & Distribution Guide

## Prerequisites

1. **Python 3.10+** with your venv activated
2. **All pip dependencies installed** (`pip install -r requirements.txt`)
3. **Nuitka**: `pip install nuitka ordered-set zstandard`
4. **Inno Setup** (optional, for installer): https://jrsoftware.org/isdl.php
5. **An icon file**: Save a 256x256 `.ico` file as `icon.ico` in the project root
   - If you don't have one, remove the `--windows-icon-from-ico=icon.ico` line from `build.bat`
   - And remove the `SetupIconFile=icon.ico` line from `installer.iss`

## Build Steps

### Step 1: Compile with Nuitka

```
build.bat
```

This takes 10-30 minutes. It:
- Compiles `launcher.py` + all backend Python to native C code
- Bundles PyTorch, torchvision, CUDA libraries
- Copies frontend/, configs/ as data files
- Output: `dist/launcher.dist/NeuralScribe.exe` (~2-3 GB folder)

### Step 2: Test the build

```
cd dist\launcher.dist
NeuralScribe.exe
```

Should open a Chrome/Edge window with NeuralScribe. Test:
- Download EMNIST
- Data Prep
- Training (verify GPU usage)
- Live View

### Step 3: Create installer (optional)

1. Install Inno Setup from https://jrsoftware.org/isdl.php
2. Open `installer.iss` in Inno Setup Compiler
3. Click Build > Compile
4. Output: `installer_output/NeuralScribeSetup.exe`

The installer will be ~1-1.5 GB (compressed from ~3 GB).

## What the user gets

1. Double-click `NeuralScribeSetup.exe`
2. Standard Windows installer (Next, Next, Install)
3. Desktop shortcut: "NeuralScribe"
4. Click shortcut → console window opens → Chrome/Edge app window opens
5. Close the console window to stop

## Notes

- **CUDA**: The build includes CUDA runtime from your PyTorch installation.
  The recipient needs a compatible NVIDIA driver (535+), but NOT a separate CUDA install.
- **Chrome/Edge**: Eel tries Chrome first, then Edge, then default browser. At least one must be installed.
- **Antivirus**: Large unsigned exes may trigger Windows Defender. The recipient may need to allow it.
  For production, consider code signing with a certificate.
- **First run**: EMNIST download (~550 MB) happens on first use. The recipient needs internet for this.
- **Models**: If you want to ship a pre-trained model, copy your `models/english/best_model.pth` into 
  `dist/launcher.dist/models/english/` before creating the installer.

## Shipping a pre-trained model

To include your trained model so the user can skip Download/Prep/Training:

1. After building, copy these into `dist/launcher.dist/`:
   - `models/english/best_model.pth`
   - `models/english/index.json`
   - `datasets/english/cache/cached_dataset.pt` (optional — only if they might retrain)
2. Then build the installer

The user can go straight to Live View.

## Folder structure after install

```
C:\Program Files\NeuralScribe\
  NeuralScribe.exe
  frontend\          (HTML, CSS, JS)
  configs\           (YAML configs)
  backend\           (compiled Python — not readable)
  datasets\          (downloaded data — user writable)
  models\            (trained models — user writable)
  training_logs\     (logs — user writable)
  torch\, torchvision\, ... (bundled dependencies)
```
