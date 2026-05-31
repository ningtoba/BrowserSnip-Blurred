# BrowserSnip — Face Blur

AI face blurring module for BrowserSnip. Detects, recognizes, and blurs faces in video — 100% client-side using WebGPU.

## Quick Start

```bash
npm install
npm run dev
```

Requires a WebGPU-capable browser (Chrome 113+, Edge 113+).

## AI Models

Place two ONNX models in `public/models/`:

| File | Source | Size |
|------|--------|------|
| `yolo26n-face.onnx` | Community face-detection YOLO model or custom-trained on WIDER Face → `python scripts/export_yolo.py` | ~4.8 MB |
| `w600k_mbf.onnx` | [InsightFace buffalo_sc release](https://github.com/deepinsight/insightface/releases) | ~10 MB |

Standard COCO YOLO has no face class — you need a face-specific model from HuggingFace, Ultralytics Hub, or custom training.

## Pipeline

Upload → sample frames → YOLO detects faces → MobileFaceNet generates embeddings → cosine-similarity clustering groups identities → pick face(s) and blur type → full-frame processing → ffmpeg-wasm reconstructs video.

## License

MIT © BrowserSnip Contributors
