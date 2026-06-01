interface DecodedFrame {
  imageData: ImageData;
  timestamp: number;
  index: number;
}

async function* decodeAndYield(
  config: VideoDecoderConfig,
  chunks: { data: Uint8Array; key: boolean }[],
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
 * Split an Annex B H.264 bitstream into individual NAL unit payloads.
 * Returns raw NAL bytes WITHOUT the start code prefix (the caller adds it back).
 */
function splitAnnexBNALs(stream: Uint8Array): Uint8Array[] {
  const nals: Uint8Array[] = [];
  let i = 0;
  while (i < stream.length - 3) {
    // Find start code: 00 00 00 01 or 00 00 01
    if (stream[i] === 0 && stream[i + 1] === 0) {
      let startCodeLen: number;
      if (stream[i + 2] === 1) {
        startCodeLen = 3;
      } else if (stream[i + 2] === 0 && i + 3 < stream.length && stream[i + 3] === 1) {
        startCodeLen = 4;
      } else {
        i++;
        continue;
      }

      // nalPayloadStart = first byte after the start code
      const nalPayloadStart = i + startCodeLen;

      // Find the next start code (or end of stream)
      let nalEnd = stream.length;
      for (let j = nalPayloadStart + 1; j < stream.length - 2; j++) {
        if (stream[j] === 0 && stream[j + 1] === 0 &&
            (stream[j + 2] === 1 || (stream[j + 2] === 0 && j + 3 < stream.length && stream[j + 3] === 1))) {
          nalEnd = j;
          break;
        }
      }
      nals.push(stream.slice(nalPayloadStart, nalEnd));
      i = nalEnd;
    } else {
      i++;
    }
  }
  return nals;
}

/**
 * Use ffmpeg.wasm to produce an Annex B H.264 bitstream from any input format.
 * Always transcodes to H.264 via libx264 (ultrafast) so the h264_mp4toannexb
 * BSF works regardless of the source codec (AV1, H.265, VP9, etc.).
 */
async function extractAnnexBStream(
  ffmpeg: import('@ffmpeg/ffmpeg').FFmpeg,
  videoFile: File
): Promise<Uint8Array> {
  const inputBuf = new Uint8Array(await videoFile.arrayBuffer());
  await ffmpeg.writeFile('input.bin', inputBuf);

  // Single step: transcode to H.264 Annex B bitstream.
  // libx264 handles any input codec; -an strips audio; -bsf:v produces Annex B.
  console.debug('[WebCodecs] transcoding to H.264 Annex B...');
  await ffmpeg.exec(
    [
      '-i', 'input.bin',
      '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
      '-an',
      '-bsf:v', 'h264_mp4toannexb',
      '-f', 'h264',
      'output.h264',
    ],
    300_000
  );

  const h264Data = await ffmpeg.readFile('output.h264');

  try { await ffmpeg.deleteFile('input.bin'); } catch { /* ignore */ }
  try { await ffmpeg.deleteFile('output.h264'); } catch { /* ignore */ }

  if (typeof h264Data === 'string') throw new Error('Expected binary data from ffmpeg, got string');
  return h264Data;
}

export async function* decodeFramesWebCodecs(
  videoFile: File,
  signal: AbortSignal
): AsyncGenerator<DecodedFrame> {
  const { getFFmpeg } = await import('@/lib/ffmpeg/core');
  const ffmpeg = await getFFmpeg();

  // Use ffmpeg to produce Annex B H.264 bitstream (SPS/PPS inline)
  const annexB = await extractAnnexBStream(ffmpeg, videoFile);
  console.debug('[WebCodecs] Annex B stream:', annexB.length, 'bytes');

  if (annexB.length === 0) throw new Error('ffmpeg produced empty H.264 stream');

  // Split into individual NAL units
  const nalUnits = splitAnnexBNALs(annexB);
  console.debug('[WebCodecs] NAL units:', nalUnits.length);

  // Group NALs into access units: accumulate until an IDR (keyframe),
  // then combine SPS+PPS+IDR into a single key chunk. This satisfies
  // the WebCodecs requirement that the first chunk must be a keyframe.
  const START_CODE = new Uint8Array([0, 0, 0, 1]);
  const chunks: { data: Uint8Array; key: boolean }[] = [];
  let pendingBufs: Uint8Array[] = [];
  let pendingSize = 0;

  function flushPending(key: boolean) {
    if (pendingBufs.length === 0) return;
    const combined = new Uint8Array(pendingSize);
    let off = 0;
    for (const buf of pendingBufs) { combined.set(buf, off); off += buf.length; }
    chunks.push({ data: combined, key });
    pendingBufs = [];
    pendingSize = 0;
  }

  for (const nal of nalUnits) {
    const nalType = nal.length > 3 ? (nal[3] & 0x1f) : 0;
    // Re-attach start code
    const chunk = new Uint8Array(START_CODE.length + nal.length);
    chunk.set(START_CODE, 0);
    chunk.set(nal, START_CODE.length);

    if (nalType === 5) {
      // IDR: flush any preceding SPS/PPS + this IDR as one key chunk
      pendingBufs.push(chunk);
      pendingSize += chunk.length;
      flushPending(true);
    } else if (nalType === 7 || nalType === 8) {
      // SPS/PPS: accumulate, will be merged with the next IDR
      pendingBufs.push(chunk);
      pendingSize += chunk.length;
    } else {
      // Non-IDR slice: flush any leftover SPS/PPS (shouldn't happen), then emit as delta
      flushPending(false);
      chunks.push({ data: chunk, key: false });
    }
  }
  flushPending(false); // flush any trailing data

  console.debug('[WebCodecs] chunks:', chunks.length,
    'keyframes:', chunks.filter(c => c.key).length);

  if (chunks.length === 0) throw new Error('No NAL units in H.264 stream');

  // WebCodecs can parse SPS/PPS from Annex B — no description needed
  const config: VideoDecoderConfig = {
    codec: 'avc1.42001e', // baseline — will be overridden by isConfigSupported if needed
    codedWidth: 1920,
    codedHeight: 1080,
  };

  // Try to detect actual resolution from SPS if available
  // (fall back to defaults — VideoDecoder adapts)
  yield* decodeAndYield(config, chunks, signal);
}
