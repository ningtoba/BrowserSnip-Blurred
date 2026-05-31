# BrowserSnip Face Blur

**100% client-side AI face blurring. No uploads, no servers, total privacy.**

Face Blur is a standalone module of BrowserSnip that detects, recognizes, and blurs faces in video — entirely inside your browser using WebGPU. Upload a video, the AI scans every face and groups them by identity, you pick which person to blur (pixelated mosaic or black eye bar), and it reconstructs the video preserving original resolution, format, and metadata. Your video never leaves your machine.

![License](https://img.shields.io/badge/license-MIT-blue)
![TypeScript](https://img.shields.io/badge/TypeScript-5.6-blue)
![React](https://img.shields.io/badge/React-18-61dafb)
![Vite](https://img.shields.io/badge/Vite-6-646cff)

---

## Table of Contents

- [Philosophy](#philosophy)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [AI Models](#ai-models)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Pipeline Reference](#pipeline-reference)
- [Browser Support](#browser-support)
- [Known Limitations](#known-limitations)
- [Contributing](#contributing)
- [License](#license)

---

## Philosophy

BrowserSnip Face Blur is a focused, single-purpose tool. It does one thing: detect and blur faces in video. It is not a general-purpose editor, a cloud service, or a privacy-invasive surveillance tool.

### Core Principles

- **No server footprint.** Both AI inference (WebGPU) and video encoding (ffmpeg.wasm) run entirely in your browser. Faces are detected, embeddings are computed, and video is reconstructed on-device.
- **You choose what to blur.** The AI doesn't decide who to hide — it shows you every identity it finds, and you pick. Nothing is auto-censored.
- **Preserve the original.** Output video matches the input in resolution, frame rate, codec, and metadata. Only the pixels you selected are modified.
- **Lightweight.** The AI models total ~15 MB. Combined with application code, the entire tool loads under 20 MB.

---

## Features

### Face Detection & Recognition

| Capability | Detail |
|-----------|--------|
| **Detection** | YOLOv26n nano model (~4.8 MB FP16) via ONNX Runtime Web with WebGPU backend |
| **Recognition** | MobileFaceNet (buffalo_sc) generates 512-dim embeddings per face |
| **Identity clustering** | Cosine-similarity agglomerative clustering groups faces by person across all frames |
| **Multi-face** | Handles multiple people in a single video — each detected and recognized independently |

### Blur Types

| Type | Effect |
|------|--------|
| **Pixelated Mosaic** | Blocky pixelation over the entire face region (configurable block size) |
| **Black Bar (Eyes)** | Solid black rounded bar covering only the eye area |

### Cross-Cutting Features

- **Phased loading bar** — Shows what's happening: model download, frame extraction, face detection, recognition, clustering, frame processing, video encoding
- **Face picker UI** — Thumbnail grid grouped by identity with frame counts and select/deselect-all
- **Smart sampling** — Scans at 1 fps for identity discovery, full-frame detection only during final export
- **Batched encoding** — Processes 300 frames per ffmpeg segment to avoid out-of-memory crashes on long videos
- **Memory guardrails** — Warning banner for files over 500 MB
- **WebGPU detection** — Graceful fallback message with browser recommendations if WebGPU is unavailable

---

## Tech Stack

| Concern | Technology |
|---------|-----------|
| Framework | React 18 with TypeScript |
| Build Tool | Vite 6 (static bundler) |
| AI Runtime | ONNX Runtime Web 1.21+ (WebGPU backend) |
| Face Detection | YOLOv26n ONNX (end-to-end with NMS) |
| Face Recognition | MobileFaceNet ONNX (buffalo_sc, 512-dim ArcFace embeddings) |
| Video Encoding | @ffmpeg/ffmpeg 0.12+ (multi-threaded WebAssembly) |
| State Management | Zustand v5 |
| Styling | TailwindCSS 3 (dark theme, shared tokens with BrowserSnip) |

---

## Getting Started

### Prerequisites

- **Node.js** 18 or later
- **npm** 9 or later
- A browser with **WebGPU** support (Chrome 113+, Edge 113+, Opera 99+)

### Quick Start

```bash
# Clone the repository
git clone git@github.com:ningtoba/browsersnip-blur.git
cd browsersnip-blur

# Install dependencies
npm install

# Start the dev server
npm run dev
```

Open `http://localhost:5173` in a WebGPU-capable browser.

### Important: Cross-Origin Isolation

Both ffmpeg.wasm and ONNX Runtime Web require `SharedArrayBuffer`, which browsers only expose when the page is **cross-origin isolated**. The Vite dev server is pre-configured:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

If these headers are missing, processing will fail. Verify them in the Network tab.

### Build for Production

```bash
npm run build
npx vite preview
```

The output lands in `dist/`. Do not open `dist/index.html` directly from the filesystem — ES modules, SharedArrayBuffer, and service workers are blocked on `file://` URLs.

---

## AI Models

Two ONNX model files must be placed in `public/models/`:

| File | Source | Size | Purpose |
|------|--------|------|---------|
| `yolo26n-face.onnx` | Export from Ultralytics YOLO (see script) | ~4.8 MB FP16 | Face detection (end-to-end with NMS) |
| `w600k_mbf.onnx` | InsightFace buffalo_sc release | ~10 MB | Face recognition (512-dim embeddings) |

### Obtaining the Models

**YOLO face detection model:**

Standard YOLO COCO models do not include a face class. You need either:

- A community face-detection YOLO model from HuggingFace or Ultralytics Hub
- Custom-trained: `yolo detect train data=widerface.yaml model=yolo26n.yaml epochs=100`

Then export:

```bash
pip install ultralytics onnx
python scripts/export_yolo.py
```

This produces `public/models/yolo26n-face.onnx` — FP16, simplified, with NMS baked into the ONNX graph. Output is `(N, 6)` boxes directly, no JavaScript NMS needed.

**MobileFaceNet recognition model:**

Download `w600k_mbf.onnx` from the [InsightFace buffalo_sc release](https://github.com/deepinsight/insightface/releases). The model is already in ONNX format (opset 11). Input: `(1, 3, 112, 112)` BGR with `(pixel - 127.5) / 127.5` normalization. Output: `(1, 512)` float32 embedding.

### Model Caching

ONNX models are fetched once per session via `fetch()` and cached by the browser's HTTP cache. Subsequent visits load instantly from disk. The combined model download is ~15 MB.

---

## Architecture

### Directory Structure

```
src/
├── main.tsx                     # Entry point
├── App.tsx                      # Root layout: header + phases + log monitor
├── index.css                    # Tailwind + BrowserSnip design tokens
│
├── lib/
│   ├── engine/
│   │   ├── session.ts           # ONNX singleton (load, warmup, infer, dispose)
│   │   ├── detection.ts         # YOLO preprocessing + postprocessing
│   │   ├── recognition.ts       # MobileFaceNet preprocessing + embedding
│   │   ├── clustering.ts        # Cosine-similarity union-find clustering
│   │   └── blur.ts              # Pixelate + eye-bar blur algorithms
│   ├── video/
│   │   ├── extract.ts           # Frame extraction via video-seeking + Canvas
│   │   ├── composite.ts         # Per-frame detection + blur application
│   │   └── reconstruct.ts       # ffmpeg batch encoding + segment concat
│   ├── ffmpeg/
│   │   ├── core.ts              # FFmpeg WASM singleton + COOP detection
│   │   └── commands.ts          # Reconstruct + concat command builders
│   ├── utils/
│   │   ├── image.ts             # ImageData ↔ tensor conversions
│   │   ├── math.ts              # Cosine similarity, L2 normalization
│   │   └── download.ts          # Blob download helper
│   └── constants.ts             # Thresholds, model configs, phase weights
│
├── stores/
│   ├── file-store.ts            # Video file + metadata + large-file detection
│   ├── process-store.ts         # Pipeline phase, detections, identities, output
│   └── ui-store.ts              # Log monitor toggle
│
├── hooks/
│   ├── useONNX.ts               # Model loading with progress messages
│   ├── useFFmpeg.ts             # FFmpeg lifecycle orchestration
│   └── usePipeline.ts           # Full pipeline: scan → detect → recognize → cluster → blur → reconstruct
│
├── types/
│   └── index.ts                 # Shared TypeScript interfaces
│
└── components/
    ├── ui/
    │   ├── FileDropZone.tsx      # Drag-and-drop video import
    │   ├── VideoPreview.tsx      # HTML5 video player for source
    │   ├── ProgressBar.tsx       # Phased progress with label + detail
    │   ├── PhaseIndicator.tsx    # Current pipeline phase display
    │   ├── LoadingScreen.tsx     # Initial model download progress
    │   ├── WebGPUBanner.tsx      # Unsupported browser message
    │   ├── OutputActions.tsx     # Download button + output preview
    │   ├── MemoryWarning.tsx     # Large-file warning banner
    │   └── LogMonitor.tsx        # Scrollable processing log viewer
    └── face/
        ├── FacePicker.tsx        # Thumbnail grid + select-all + process button
        ├── FaceThumbnail.tsx     # Individual identity card with toggle
        └── BlurTypeSelector.tsx  # Pixelated mosaic vs. eye bar chip toggle
```

### Pipeline Data Flow

```
User drops video
    │
    ▼
FileDropZone → file-store (File + VideoMetadata)
    │
    ▼
usePipeline.startScan():
    │
    ├─ Phase 1: extract-frames (1 fps sample)
    │     └─ video element seeking + Canvas → ImageData[]
    │
    ├─ Phase 2: detect-faces (YOLO ONNX per sampled frame)
    │     └─ letterbox → normalize → infer → rescale boxes
    │
    ├─ Phase 3: recognize-faces (MobileFaceNet per detection)
    │     └─ crop face → resize 112×112 → BGR normalize → infer → L2 normalize
    │
    ├─ Phase 4: cluster (cosine-sim union-find)
    │     └─ pairwise similarity → connected components → FaceIdentity[]
    │
    └─ Phase 5: waiting-selection
          └─ generate thumbnails → show FacePicker
    │
    ▼
User selects face(s) + blur type → clicks "Blur N faces"
    │
    ▼
usePipeline.processAndExport():
    │
    ├─ Phase 6: process-frames (ALL frames at original fps)
    │     ├─ For each frame: detect → match embeddings → apply blur → write PNG to MEMFS
    │     └─ Batch every 300 frames to avoid OOM
    │
    └─ Phase 7: reconstruct (ffmpeg-wasm)
          └─ encode frames → concat segments → read blob → Object URL
    │
    ▼
OutputActions: Download
```

### State Management

Three Zustand stores matching BrowserSnip's pattern:

| Store | Purpose | Key Keys |
|-------|---------|----------|
| `file-store` | Input file | `file`, `metadata`, `isLargeFile`, `probing` |
| `process-store` | Pipeline state | `phase`, `progress`, `allDetections`, `identities`, `identityThumbnails`, `selectedIdentities`, `blurConfig`, `outputBlob`, `outputUrl`, `error`, `logs` |
| `ui-store` | UI toggles | `showLogMonitor` |

---

## How It Works

### Face Detection (YOLOv26n)

1. **Letterbox** the frame to 640×640 (pad with gray 114)
2. **Normalize** RGB pixels to [0, 1], transpose HWC → CHW
3. **Infer** via ONNX Runtime Web with WebGPU backend
4. **Parse** `(N, 6)` output — NMS is embedded in the ONNX graph (end-to-end export)
5. **Rescale** coordinates back to original frame dimensions
6. **Filter** by confidence threshold (default: 0.4)

### Face Recognition (MobileFaceNet)

1. **Crop** face region from frame (bounding box expanded by 20%)
2. **Resize** to 112×112
3. **Convert** RGB → BGR, normalize: `(pixel - 127.5) / 127.5`
4. **Infer** via ONNX Runtime Web
5. **L2-normalize** the 512-dim embedding to unit length

### Identity Clustering

Union-find agglomerative clustering over all detections with embeddings:

1. Compute pairwise **cosine similarity** (skip same-frame pairs — they're different people by definition)
2. Union pairs where similarity > 0.45
3. For each connected component: compute **centroid embedding** (mean + L2 normalize)
4. Pick highest-confidence face as the **representative thumbnail**
5. Sort identities by face count (most appearances first)

### Blur Algorithms

**Pixelated Mosaic:**
Divide the face bounding box into `blockSize × blockSize` blocks, compute the average color per block, fill each block with its average. The result is a classic mosaic/pixelation effect.

**Eye Bar:**
Draw a filled rounded rectangle over the top ~30% of the face bounding box (the eye region). The bar is 85% of face width, centered horizontally.

### Video Reconstruction

ffmpeg-wasm encodes the processed frames back into MP4:

```bash
ffmpeg -framerate {fps} -i frame_%04d.png \
  -c:v libx264 -preset ultrafast -crf 18 \
  -pix_fmt yuv420p -movflags +faststart output.mp4
```

For videos longer than ~10 seconds, processing is split into **300-frame segments**. Each segment is encoded separately, then concatenated via the ffmpeg concat demuxer. This prevents the in-memory filesystem from overflowing.

---

## Pipeline Reference

### Progress Bar Phases

| Phase | Weight | Description |
|-------|--------|-------------|
| loading-models | 15% | Downloading AI models from `public/models/` |
| extracting-frames | 10% | Seeking video and capturing sample frames at 1 fps |
| detecting-faces | 25% | Running YOLO on each sample frame |
| recognizing-faces | 25% | Running MobileFaceNet on every detected face |
| clustering | 5% | Grouping all embeddings into identities |
| waiting-selection | 0% | User picks faces and blur type |
| processing-frames | 15% | Full-frame detection + blur at original fps |
| reconstructing | 5% | ffmpeg encoding + segment concatenation |

### Tunable Constants

All thresholds are in `src/lib/constants.ts`:

| Constant | Default | Purpose |
|----------|---------|---------|
| `SAMPLE_FPS` | 1 | Frames per second for initial face scan |
| `DETECTION_CONFIDENCE` | 0.4 | Minimum confidence for YOLO face detection |
| `NMS_IOU_THRESHOLD` | 0.45 | Overlap threshold for NMS (embedded in model) |
| `CLUSTER_SIMILARITY_THRESHOLD` | 0.45 | Cosine similarity threshold for same identity |
| `FACE_EXPAND_RATIO` | 0.2 | Bounding box expansion for face cropping |
| `PIXELATE_BLOCK_SIZE` | 15 | Block size in pixels for mosaic blur |
| `BATCH_SIZE` | 300 | Frames per ffmpeg encoding segment |

### Blur Types

| Type | Parameter | Effect |
|------|-----------|--------|
| Pixelated Mosaic | `pixelSize` (default 15) | Block-averages face region at given block size |
| Black Bar (Eyes) | — | Solid rounded black bar across eye region |

---

## Browser Support

**WebGPU is required.** The tool will not run without it.

| Browser | Minimum Version | WebGPU |
|---------|----------------|--------|
| Chrome | 113+ | Supported |
| Edge | 113+ | Supported |
| Opera | 99+ | Supported |
| Chrome Android | 121+ | Supported |
| Firefox | — | Not yet supported (nightly only) |
| Safari | — | Not yet supported (experimental) |

If your browser lacks WebGPU, a full-page banner will display with compatible alternatives.

---

## Known Limitations

### Model Accuracy

- **No facial keypoint alignment.** YOLO does not output the 5 facial landmarks (eyes, nose, mouth corners) that ArcFace normally uses for affine alignment. The tool uses simple crop+resize from the bounding box, which reduces embedding quality by ~5-10% similarity compared to proper alignment. This may cause identity confusion for faces at extreme angles or partial occlusion.

### File Size

- **Soft limit: 500 MB.** Files above this trigger a warning banner. Processing large files at full frame rate requires substantial memory.
- **Long videos** (>2 minutes) will take significant time during the full-frame processing phase. Sample scanning (1 fps) is fast regardless of length.

### Processing Speed

- **AI inference** depends on your GPU. A modern integrated GPU processes ~10-20 frames per second for detection alone. Recognition adds additional overhead per face.
- **ffmpeg encoding** is 5-10× slower than native FFmpeg. A 30-second 1080p video reconstruction may take 2-5 minutes.

### Codec Support

- **Input:** Any video format your browser can decode (`video/*` MIME type)
- **Output:** H.264 video + AAC audio in MP4 container (matches BrowserSnip's default). Original resolution, frame rate, and pixel format are preserved. Metadata is stripped during reconstruction.

---

## Contributing

Contributions are welcome. The tool follows BrowserSnip's conventions:

### Development Setup

```bash
npm install
npm run dev
```

### Code Conventions

- **TypeScript** throughout — no `any` in application code
- **Components** — one per file, named exports, inline Props interface
- **Engine modules** — pure functions for preprocessing/postprocessing, ONNX inference via singleton session manager
- **State** — Zustand stores for cross-component state, local React state for UI-only state
- **Styling** — Tailwind utility classes with BrowserSnip's shared design tokens (cream/ink/accent palette, doodle-* components)

### Before Submitting a PR

```bash
npm run build      # Must pass with no errors
npx tsc --noEmit   # Must pass with no type errors
```

---

## License

MIT © BrowserSnip Contributors

---

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — State-of-the-art object detection
- [InsightFace](https://github.com/deepinsight/insightface) — 2D/3D face analysis (buffalo_sc model pack)
- [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) — Cross-platform ML inference
- [ffmpeg.wasm](https://github.com/ffmpegwasm/ffmpeg.wasm) — FFmpeg compiled to WebAssembly
- [BrowserSnip](https://github.com/ningtoba/BrowserSnip) — The parent project
- [coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker) — Cross-origin isolation polyfill
