# BrowserSnip — Face Blur

AI face blurring module for BrowserSnip. Detects, recognizes, and blurs faces in video — 100% client-side using WebGPU.

## Quick Start

```bash
npm install
npm run dev
```

Requires a WebGPU-capable browser (Chrome 113+, Edge 113+).

## AI Models

| File | Source | Size |
|------|--------|------|
| `yolo26n-face.onnx` | [akanametov/yolo-face](https://github.com/akanametov/yolo-face) — YOLOv26n trained on WIDER Face | 9.9 MB |
| `w600k_mbf.onnx` | [InsightFace buffalo_sc](https://github.com/deepinsight/insightface/releases) — MobileFaceNet | 13 MB |

## Pipeline

Upload → sample frames → YOLO detects faces → MobileFaceNet generates embeddings → cosine-similarity clustering groups identities → pick face(s) and blur type → full-frame processing with zero-copy GPU pipeline → ffmpeg-wasm reconstructs video.

Preprocessing and blur run as WebGPU compute shaders. Video frames are imported directly from the decoder via `importExternalTexture` — zero CPU copies for detection input. Detection runs every 2nd frame with bbox reuse for intermediate frames.

## Acknowledgments

- [akanametov/yolo-face](https://github.com/akanametov/yolo-face) — YOLOv26n-face model
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [InsightFace](https://github.com/deepinsight/insightface)
- [ONNX Runtime Web](https://github.com/microsoft/onnxruntime)
- [ffmpeg.wasm](https://github.com/ffmpegwasm/ffmpeg.wasm)
- [coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker)
- [BrowserSnip](https://github.com/ningtoba/BrowserSnip)

## License

MIT © BrowserSnip Contributors
