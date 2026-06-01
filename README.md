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
| `scrfd_2.5g_bnkps.onnx` | [InsightFace SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd) — high-accuracy face detector with 5-point landmarks | 3.3 MB |
| `w600k_mbf.onnx` | [InsightFace buffalo_sc](https://github.com/deepinsight/insightface/releases) — MobileFaceNet | 13 MB |

## Pipeline

Upload → sample frames → SCRFD detects faces with 5-point landmarks → MobileFaceNet generates embeddings → cosine-similarity clustering groups identities → pick face(s) and blur type → full-frame processing → ffmpeg-wasm reconstructs video.

Detection runs every 2nd frame with Kalman filter tracking for intermediate frames. Blur types: Mosaic, Eye Bar, Black Box.

## Performance

| Stage | Speed |
|-------|-------|
| Face detection (SCRFD 2.5G) | ~30ms/frame (WASM) |
| Face recognition (MFN) | ~50ms/face |
| Blur (CPU Canvas2D) | ~5ms/frame |
| Video decode (WebCodecs) | ~5ms/frame |
| Video encode (ffmpeg.wasm) | varies |

## Acknowledgments

- [SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd) — high-accuracy face detection (Guo et al., ICLR 2022)
- [InsightFace](https://github.com/deepinsight/insightface) — face detection and recognition models
- [OpenCV Zoo](https://github.com/opencv/opencv_zoo) — model reference and benchmarks
- [ONNX Runtime Web](https://github.com/microsoft/onnxruntime)
- [ffmpeg.wasm](https://github.com/ffmpegwasm/ffmpeg.wasm)
- [mp4box.js](https://github.com/gpac/mp4box.js) — MP4 demuxing
- [coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker)
- [BrowserSnip](https://github.com/ningtoba/BrowserSnip)

## License

MIT © BrowserSnip Contributors
