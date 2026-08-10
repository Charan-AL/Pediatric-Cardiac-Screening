# Pediatric Cardiac Screening — Quick Start

**Run the entire system in 2 commands (2 separate terminals)**

---

## Prerequisites
- Python 3.12.5 (venv in parent directory: `../.venv`)
- Node.js 18+ (for npm)
- CUDA GPU (recommended) or CPU

---

## Terminal 1 — Backend API

```powershell
cd "C:\Users\hrish\OneDrive\Documents\6th sem notes\MajorProject\pediatric_cardiac_screening"
$py = "C:\Users\hrish\OneDrive\Documents\6th sem notes\MajorProject\.venv\Scripts\python.exe"
$env:AUDIO_CKPT = "checkpoints/audio/audio_best.pth"
$env:US_CKPT = "checkpoints/ultrasound/ultrasound_best.pth"
$env:XRAY_CKPT = "checkpoints/xray/xray_best.pth"
$env:GMU_CKPT = "checkpoints/gmu/gmu_best.pth"
& $py -m uvicorn inference.api:app --reload --host 0.0.0.0 --port 8000
```

✅ Wait for: `INFO:     Application startup complete.`

---

## Terminal 2 — React Frontend

```powershell
cd "C:\Users\hrish\OneDrive\Documents\6th sem notes\MajorProject\pediatric_cardiac_screening\frontend"
npm run dev
```

✅ Wait for: `➜  Local:   http://localhost:5173/`

---

## Open Browser

**http://localhost:5173**

Upload audio (WAV) + images (JPG/PNG), click "Run Screening" → AI decision + Grad-CAM visualizations.

---

## System Architecture

```
┌─────────────────────────────────────────────────┐
│     React Frontend (localhost:5173)             │
│  - Upload: audio (WAV), ultrasound, X-ray       │
│  - Display: decision, confidence, heatmaps      │
└────────────────┬────────────────────────────────┘
                 │ FormData POST
                 ▼
┌─────────────────────────────────────────────────┐
│     FastAPI Backend (localhost:8000)            │
│  - Process files & preprocess                   │
│  - Inference: CRNN + NTS + EfficientNet         │
│  - Fusion: GMU gating network                   │
│  - Return: JSON + base64 Grad-CAM images        │
└─────────────────────────────────────────────────┘
```

**Models (Frozen Specialists):**
- Audio (CRNN): 97.6% val acc, 12.2M params → embedding
- Ultrasound (NTS-Net): 97.6% val acc, 26M params → embedding
- X-Ray (EfficientNetV2): 89.4% val acc, 37.6M params → embedding

**GMU Fusion:**
- Gating network: 2.5M trainable params
- MLP classifier: binary (PASS/REFER)
- Val accuracy: 94.6% | F1: 0.891 | Sensitivity: 92.3% | Specificity: 95.1%

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| ModuleNotFoundError: cv2 | Ensure `.venv` has PyTorch + OpenCV installed |
| Port 8000/5173 already in use | Kill old processes: `netstat -ano \| findstr :8000` → `taskkill /PID <pid> /F` |
| Checkpoint not found | Verify files exist in `checkpoints/` subdirectories |
| CORS error | Backend CORS is enabled; check API is running |

---

## Files Reference

- **Backend**: `inference/api.py` (FastAPI endpoints)
- **Frontend**: `frontend/src/UploadForm.jsx` (React upload UI)
- **Models**: `models/crnn_heart_sound.py`, `models/nts_net_ultrasound.py`, `models/efficientnet_xray.py`, `models/gmu_fusion.py`
- **Preprocessing**: `preprocessing/audio_preprocessing.py`, `preprocessing/image_preprocessing.py`
- **Explainability**: `explainability/gradcam.py` (Grad-CAM visualization)



**For detailed setup & troubleshooting, see [frontend/README.md](frontend/README.md)**
