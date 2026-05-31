import { createFile, MP4BoxBuffer } from 'mp4box';

interface DecodedFrame {
  imageData: ImageData;
  timestamp: number;
  index: number;
}

interface PendingSample {
  data: ArrayBuffer;
  key: boolean;
  cts: number;
  duration: number;
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

function demuxWithMP4Box(arrayBuf: ArrayBuffer): Promise<{
  config: VideoDecoderConfig;
  chunks: { data: ArrayBuffer; key: boolean }[];
}> {
  return new Promise((resolve, reject) => {
    const file = createFile();
    const pendingSamples: PendingSample[] = [];
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
      // Extract avcC/hvcC from track description if available
      if (track.description) {
        const descBuf = new ArrayBuffer(track.description.length);
        new Uint8Array(descBuf).set(track.description);
        (config as any).description = descBuf;
      }
      console.debug('[MP4Box] codec:', track.codec, 'size:', track.track_width, 'x', track.track_height);
      file.setExtractionOptions(track.id, 'video');
      ready = true;
    };

    file.onSamples = (_trackId: number, _user: any, samples: any[]) => {
      for (const s of samples) {
        pendingSamples.push({
          data: s.data.buffer.slice(0),
          key: s.is_sync,
          cts: s.cts,
          duration: s.duration,
        });
      }
    };

    file.onError = (e: any) => {
      reject(new Error('MP4Box error: ' + (e?.message || e)));
    };

    // Feed the buffer as MP4BoxBuffer
    const mp4Buf = MP4BoxBuffer.fromArrayBuffer(arrayBuf, 0);
    (mp4Buf as any).fileStart = 0;
    file.appendBuffer(mp4Buf);
    file.flush();

    // Wait a bit for onReady to fire, then resolve
    const check = () => {
      if (ready && config) {
        const chunks = pendingSamples.map(s => ({ data: s.data, key: s.key }));
        console.debug('[MP4Box]', chunks.length, 'samples extracted');
        resolve({ config, chunks });
      } else if (pendingSamples.length > 0 && config) {
        // Samples arrived before onReady flag
        const chunks = pendingSamples.map(s => ({ data: s.data, key: s.key }));
        console.debug('[MP4Box]', chunks.length, 'samples extracted (late ready)');
        resolve({ config, chunks });
      } else {
        setTimeout(check, 100);
      }
    };
    setTimeout(check, 200);
  });
}

export async function* decodeFramesWebCodecs(
  videoFile: File,
  signal: AbortSignal
): AsyncGenerator<DecodedFrame> {
  const { getFFmpeg } = await import('@/lib/ffmpeg/core');
  const ffmpeg = await getFFmpeg();
  const inputBuf = new Uint8Array(await videoFile.arrayBuffer());

  // ffmpeg remux any format to clean MP4
  await ffmpeg.writeFile('input.bin', inputBuf);
  try {
    await ffmpeg.exec(
      ['-i', 'input.bin', '-c:v', 'copy', '-an', '-movflags', '+faststart', 'clean.mp4'],
      120_000
    );
  } catch {
    // -c:v copy failed (codec not MP4-compatible), transcode
    console.debug('[WebCodecs] remux copy failed, transcoding...');
    await ffmpeg.exec(
      ['-i', 'input.bin', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23', '-an', '-movflags', '+faststart', 'clean.mp4'],
      300_000
    );
  }

  const mp4Data = new Uint8Array((await ffmpeg.readFile('clean.mp4')) as Uint8Array);
  try { await ffmpeg.deleteFile('input.bin'); } catch { /* ignore */ }
  try { await ffmpeg.deleteFile('clean.mp4'); } catch { /* ignore */ }

  console.debug('[WebCodecs] remuxed MP4:', mp4Data.length, 'bytes');
  // Ensure we pass a properly-sized ArrayBuffer
  const mp4Buffer = mp4Data.buffer.slice(mp4Data.byteOffset, mp4Data.byteOffset + mp4Data.byteLength);
  const { config, chunks } = await demuxWithMP4Box(mp4Buffer);
  yield* decodeAndYield(config, chunks, signal);
}
