# MECHATRIX AngleC — Setup Guide

For Windows 10/11.

## 1. Install Python

If Python is not installed:

1. Visit https://www.python.org/
2. Install Python
3. Open PowerShell and verify:

```powershell
py --version
```

## 2. Install Required Libraries

Update pip:

```powershell
py -m pip install --upgrade pip
```

Install NumPy and OpenCV using the tested versions:

```powershell
py -m pip install "numpy==1.26.4" "opencv-python==4.9.0.80"
```

Install the additional libraries:

```powershell
py -m pip install scipy pillow matplotlib
```

## 3. Verify Installation

```powershell
py -c "import numpy, cv2, scipy; from PIL import Image; import matplotlib; print('NumPy:', numpy.__version__); print('OpenCV:', cv2.__version__); print('All imports successful')"
```

## 4. Run AngleC

1. Download the repository using **Code → Download ZIP** on GitHub.
2. Extract the ZIP file.
3. Open the extracted folder containing `AngleC.py`.
4. Click the File Explorer address bar, type `powershell`, and press Enter.
5. Run:

```powershell
py -m AngleC
```

Connect your webcam and follow the on-screen instructions.

## Main Controls

| Key | Action |
|---|---|
| `C` | Capture image |
| `U` | Undo last surface point |
| `X` | Clear selected points |
| `E` | Export results |
| `R` | Reset measurement |
| `Q` / `ESC` | Quit |

## Troubleshooting

**NumPy/OpenCV import error:** Verify the installed versions:

```powershell
py -c "import numpy, cv2; print(numpy.__version__, cv2.__version__)"
```

Expected: NumPy `1.26.4`, OpenCV `4.9.0`.

**Webcam not detected:** If the webcam is not detected, try connecting it to a different USB port. Some laptops may have ports that work differently with external webcams, so try each available port to identify one that works.

## Development Credit

**Concept and system design:** Mir Md Minhaajuddin  
**Coding assistance:** ChatGPT and Claude

AngleC uses local image processing and does not require a paid API key or cloud service for its measurement workflow.
