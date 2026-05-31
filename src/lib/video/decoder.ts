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

function extractAvcC(mp4Buf: ArrayBuffer): Uint8Array | null {
  const buf = new Uint8Array(mp4Buf);
  try {
    // Find moov → trak → stbl → stsd → avc1/avc3 → avcC
    let off = 0;
    while (off < buf.length - 8) {
      const size = ((buf[off] << 24) | (buf[off+1] << 16) | (buf[off+2] << 8) | buf[off+3]) >>> 0;
      const tag = String.fromCharCode(buf[off+4], buf[off+5], buf[off+6], buf[off+7]);
      if (size < 8 || off + size > buf.length) break;
      if (tag === 'moov') {
        // Search for avcC fourCC within the trak/stsd subtree
        const moovEnd = off + size;
        for (let i = off + 8; i < moovEnd - 8; i++) {
          if (buf[i] === 0x61 && buf[i+1] === 0x76 && buf[i+2] === 0x63 && buf[i+3] === 0x43) {
            // avcC found — read the box
            const boxSize = ((buf[i-4] << 24) | (buf[i-3] << 16) | (buf[i-2] << 8) | buf[i-1]) >>> 0;
            if (boxSize > 8 && boxSize < 1000 && i - 4 + boxSize <= moovEnd) {
              return buf.slice(i + 4, i - 4 + boxSize);
            }
          }
        }
        break;
      }
      off += size;
    }
  } catch {}
  return null;
}

function demuxWithMP4Box(arrayBuf: ArrayBuffer): Promise<{
  config: VideoDecoderConfig;
  chunks: { data: ArrayBuffer; key: boolean }[];
}> {
  return new Promise((resolve, reject) => {
    const file = createFile();
    const pendingSamples: { data: ArrayBuffer; key: boolean }[] = [];
    let config: VideoDecoderConfig | null = null;
    let ready = false;

    file.onReady = (info: any) => {
      const track = info.videoTracks?.[0];
      if (!track) { reject(new Error('No video track')); return; }
      config = { codec: track.codec, codedWidth: track.track_width, codedHeight: track.track_height };
      // Get avcC from track description or fallback to buffer scan
      let desc: Uint8Array | undefined = track.description as Uint8Array | undefined;
      if (!desc || desc.length === 0) {
        desc = extractAvcC(arrayBuf) ?? undefined;
        console.debug('[MP4Box] avcC from buffer:', desc?.length, 'bytes');
      }
      if (desc && desc.length > 0) {
        const descBuf = new ArrayBuffer(desc.length);
        new Uint8Array(descBuf).set(desc);
        (config as any).description = descBuf;
        console.debug('[MP4Box] desc set:', descBuf.byteLength, 'bytes');
      }
      file.setExtractionOptions(track.id, 'video');
      file.start();
      ready = true;
    };

    file.onSamples = (_trackId: number, _user: any, samples: any[]) => {
      for (const s of samples) {
        pendingSamples.push({ data: s.data.buffer.slice(0), key: s.is_sync });
      }
      if (pendingSamples.length <= 3) {
        const s = samples[0];
        console.debug('[MP4Box] sample', pendingSamples.length - 1,
          'size:', s?.data?.byteLength ?? s?.data?.length,
          'is_sync:', s?.is_sync,
          'desc:', config ? 'present' : 'missing');
      }
    };

    file.onError = (e: any) => reject(new Error('MP4Box error: ' + (e?.message || e)));

    const mp4Buf = MP4BoxBuffer.fromArrayBuffer(arrayBuf, 0);
    (mp4Buf as any).fileStart = 0;
    file.appendBuffer(mp4Buf);
    file.flush();

    const checkInterval = setInterval(() => {
      if (ready && config && pendingSamples.length > 0) {
        clearInterval(checkInterval);
        console.debug('[MP4Box]', pendingSamples.length, 'samples,',
          'firstKey:', pendingSamples[0]?.key,
          'desc:', config.description ? (config.description as ArrayBuffer).byteLength + ' bytes' : 'MISSING',
          'codec:', config.codec);
        resolve({ config, chunks: pendingSamples });
      }
    }, 50);
    setTimeout(() => { clearInterval(checkInterval); reject(new Error('MP4Box extraction timed out')); }, 30000);
  });
}

export async function* decodeFramesWebCodecs(
  videoFile: File,
  signal: AbortSignal
): AsyncGenerator<DecodedFrame> {
  const { getFFmpeg } = await import('@/lib/ffmpeg/core');
  const ffmpeg = await getFFmpeg();
  const inputBuf = new Uint8Array(await videoFile.arrayBuffer());

  // ffmpeg remux to clean MP4
  await ffmpeg.writeFile('input.bin', inputBuf);
  try {
    await ffmpeg.exec(
      ['-i', 'input.bin', '-c:v', 'copy', '-an', '-movflags', '+faststart', 'clean.mp4'],
      120_000
    );
  } catch {
    console.debug('[WebCodecs] copy failed, transcoding...');
    await ffmpeg.exec(
      ['-i', 'input.bin', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23', '-an', '-movflags', '+faststart', 'clean.mp4'],
      300_000
    );
  }

  const mp4Data = new Uint8Array((await ffmpeg.readFile('clean.mp4')) as Uint8Array);
  try { await ffmpeg.deleteFile('input.bin'); } catch { /* ignore */ }
  try { await ffmpeg.deleteFile('clean.mp4'); } catch { /* ignore */ }

  console.debug('[WebCodecs] remuxed MP4:', mp4Data.length, 'bytes');
  const mp4Buf = mp4Data.buffer.slice(mp4Data.byteOffset, mp4Data.byteOffset + mp4Data.byteLength);
  const { config, chunks } = await demuxWithMP4Box(mp4Buf);
  yield* decodeAndYield(config, chunks, signal);
}
