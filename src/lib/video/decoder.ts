import { createFile, MP4BoxBuffer } from 'mp4box';

interface DecodedFrame {
  imageData: ImageData;
  timestamp: number;
  index: number;
}

async function* decodeAndYield(
  config: VideoDecoderConfig,
  chunks: { data: ArrayBuffer; key: boolean }[],
  signal: AbortSignal
): AsyncGenerator<DecodedFrame> {
  const support = await VideoDecoder.isConfigSupported(config);
  if (!support.supported) throw new Error('VideoDecoder config not supported');

  let frameIndex = 0;
  const frameQueue: VideoFrame[] = [];
  let decodeError: Error | null = null;

  const decoder = new VideoDecoder({
    output(frame: VideoFrame) { frameQueue.push(frame); },
    error(err: Error) { decodeError = err; },
  });

  decoder.configure(config);

  const frameIntervalUs = 33_333;
  let timestamp = 0;

  for (let i = 0; i < chunks.length; i++) {
    if (signal.aborted) break;
    try {
      decoder.decode(new EncodedVideoChunk({
        type: chunks[i].key ? 'key' : 'delta',
        timestamp, duration: frameIntervalUs,
        data: chunks[i].data,
      }));
    } catch {
      await new Promise(r => setTimeout(r, 10));
      decoder.decode(new EncodedVideoChunk({
        type: chunks[i].key ? 'key' : 'delta',
        timestamp, duration: frameIntervalUs,
        data: chunks[i].data,
      }));
    }
    timestamp += frameIntervalUs;
    if (i % 30 === 0) {
      await new Promise(r => setTimeout(r, 0));
      if (decodeError) throw decodeError;
    }
  }

  if (decodeError) throw decodeError;

  const flushOk = await Promise.race([
    decoder.flush().then(() => true),
    new Promise<boolean>(r => setTimeout(() => r(false), 30000)),
  ]);
  if (!flushOk) { decoder.close(); throw new Error('Decoder flush timed out'); }

  const canvas = new OffscreenCanvas(1, 1);
  let ctx: OffscreenCanvasRenderingContext2D | null = null;

  for (const frame of frameQueue) {
    if (signal.aborted) { frame.close(); continue; }
    const w = frame.displayWidth, h = frame.displayHeight;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w; canvas.height = h;
      ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    }
    if (!ctx) ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    ctx.drawImage(frame, 0, 0);
    yield { imageData: ctx.getImageData(0, 0, w, h), timestamp: frameIndex / 30, index: frameIndex };
    frameIndex++;
    frame.close();
  }

  decoder.close();
  if (decodeError) throw decodeError;
}

/**
 * Use ffmpeg.wasm to transcode any input to H.264 MP4.
 * Returns the MP4 file bytes.
 */
async function transcodeToH264MP4(
  ffmpeg: import('@ffmpeg/ffmpeg').FFmpeg,
  videoFile: File
): Promise<Uint8Array> {
  const inputBuf = new Uint8Array(await videoFile.arrayBuffer());
  await ffmpeg.writeFile('input.bin', inputBuf);

  console.debug('[WebCodecs] transcoding to H.264 MP4...');
  await ffmpeg.exec(
    [
      '-i', 'input.bin',
      '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
      '-an', '-movflags', '+faststart',
      'output.mp4',
    ],
    300_000
  );

  const mp4Data = await ffmpeg.readFile('output.mp4');

  try { await ffmpeg.deleteFile('input.bin'); } catch { /* ignore */ }
  try { await ffmpeg.deleteFile('output.mp4'); } catch { /* ignore */ }

  if (typeof mp4Data === 'string') throw new Error('Expected binary data from ffmpeg, got string');
  return mp4Data;
}

/**
 * Demux an MP4 buffer using mp4box.js — the right tool for MP4 demuxing.
 * Returns the VideoDecoderConfig (with avcC description) and encoded samples.
 */
function demuxMP4(mp4Buf: ArrayBuffer): Promise<{
  config: VideoDecoderConfig;
  chunks: { data: ArrayBuffer; key: boolean }[];
}> {
  return new Promise((resolve, reject) => {
    const file = createFile();
    const samples: { data: ArrayBuffer; key: boolean }[] = [];
    let config: VideoDecoderConfig | null = null;
    let ready = false;

    file.onReady = (info: any) => {
      const track = info.videoTracks?.[0];
      if (!track) { reject(new Error('No video track')); return; }

      config = {
        codec: track.codec,
        codedWidth: track.track_width,
        codedHeight: track.track_height,
      };

      // Get avcC description — mp4box provides it via track.description
      const desc = (track as any).description as Uint8Array | undefined;
      if (desc && desc.length > 0) {
        const descBuf = new ArrayBuffer(desc.length);
        new Uint8Array(descBuf).set(desc);
        (config as any).description = descBuf;
        console.debug('[mp4box] description:', descBuf.byteLength, 'bytes');
      } else {
        console.warn('[mp4box] no description from track — decoder may fail');
      }

      file.setExtractionOptions(track.id, 'video');
      file.start();
      ready = true;
    };

    file.onSamples = (_trackId: number, _user: any, s: any[]) => {
      for (const sample of s) {
        samples.push({ data: sample.data.buffer.slice(0), key: sample.is_sync });
      }
    };

    file.onError = (e: any) => reject(new Error('mp4box error: ' + (e?.message || e)));

    const buf = MP4BoxBuffer.fromArrayBuffer(mp4Buf, 0);
    (buf as any).fileStart = 0;
    file.appendBuffer(buf);
    file.flush();

    const check = setInterval(() => {
      if (ready && config && samples.length > 0) {
        clearInterval(check);
        console.debug('[mp4box] samples:', samples.length,
          'keyframes:', samples.filter(s => s.key).length,
          'codec:', config.codec);
        resolve({ config, chunks: samples });
      }
    }, 50);
    setTimeout(() => { clearInterval(check); reject(new Error('mp4box extraction timed out')); }, 30000);
  });
}

export async function* decodeFramesWebCodecs(
  videoFile: File,
  signal: AbortSignal
): AsyncGenerator<DecodedFrame> {
  const { getFFmpeg } = await import('@/lib/ffmpeg/core');
  const ffmpeg = await getFFmpeg();

  // Step 1: ffmpeg transcodes any input to H.264 MP4
  const mp4Data = await transcodeToH264MP4(ffmpeg, videoFile);
  console.debug('[WebCodecs] MP4:', mp4Data.length, 'bytes');
  if (mp4Data.length === 0) throw new Error('ffmpeg produced empty MP4');

  // Step 2: mp4box demuxes the MP4 into codec config + encoded samples
  const mp4Buf = mp4Data.buffer.slice(mp4Data.byteOffset, mp4Data.byteOffset + mp4Data.byteLength);
  const { config, chunks } = await demuxMP4(mp4Buf);

  // Step 3: VideoDecoder decodes the samples
  yield* decodeAndYield(config, chunks, signal);
}
