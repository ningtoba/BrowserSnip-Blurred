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
| `face_detection_yunet_2023mar.onnx` | [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo) — fast, lightweight face detector with 5-point landmarks | 228 KB |
| `w600k_mbf.onnx` | [InsightFace buffalo_sc](https://github.com/deepinsight/insightface/releases) — MobileFaceNet | 13 MB |

## Pipeline

Upload → sample frames → YuNet detects faces with 5-point landmarks → MobileFaceNet generates embeddings → cosine-similarity clustering groups identities → pick face(s) and blur type → full-frame processing → ffmpeg-wasm reconstructs video.

Detection runs every 2nd frame with bbox reuse for intermediate frames. Blur runs as WebGPU compute shaders with CPU fallback.

## Acknowledgments

- [YuNet](https://github.com/ShiqiYu/libfacedetection) — lightweight face detection (Wu, Peng & Yu, 2023)
- [OpenCV Zoo](https://github.com/opencv/opencv_zoo) — model export and reference implementation
- [InsightFace](https://github.com/deepinsight/insightface)
- [ONNX Runtime Web](https://github.com/microsoft/onnxruntime)
- [ffmpeg.wasm](https://github.com/ffmpegwasm/ffmpeg.wasm)
- [coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker)
- [BrowserSnip](https://github.com/ningtoba/BrowserSnip)

## License

MIT © BrowserSnip Contributors
