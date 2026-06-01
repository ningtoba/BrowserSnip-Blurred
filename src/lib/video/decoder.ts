interface DecodedFrame {
  imageData: ImageData;
  timestamp: number;
  index: number;
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

/** Find the next Annex B start code and return the position of the first byte of the start code. */
function findStartCode(stream: Uint8Array, from: number): number {
  for (let i = from; i < stream.length - 3; i++) {
    if (stream[i] === 0 && stream[i + 1] === 0) {
      if (stream[i + 2] === 1) return i;
      if (stream[i + 2] === 0 && stream[i + 3] === 1) return i;
    }
  }
  return -1;
}

/**
 * Split the Annex B stream into access units (chunks), each starting with
 * a start code. Group SPS+PPS+IDR into a single key chunk.
 */
function splitIntoAccessUnits(stream: Uint8Array): { data: Uint8Array; key: boolean }[] {
  // Step 1: split at every start code boundary
  const rawUnits: Uint8Array[] = [];
  let pos = findStartCode(stream, 0);
  while (pos >= 0) {
    const nextPos = findStartCode(stream, pos + 3);
    rawUnits.push(stream.slice(pos, nextPos >= 0 ? nextPos : stream.length));
    pos = nextPos;
  }

  // Step 2: group into access units — SPS/PPS accumulate until IDR
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

  for (const unit of rawUnits) {
    // Determine NAL type from byte after start code
    let nalType = 0;
    if (unit[2] === 1) nalType = unit[3] & 0x1f;
    else if (unit[3] === 1) nalType = unit[4] & 0x1f;

    if (nalType === 5) {
      // IDR: flush any preceding SPS/PPS + this IDR as one key chunk
      pendingBufs.push(unit);
      pendingSize += unit.length;
      flushPending(true);
    } else if (nalType === 7 || nalType === 8) {
      // SPS/PPS: accumulate
      pendingBufs.push(unit);
      pendingSize += unit.length;
    } else {
      // Non-IDR slice: flush leftover, then emit as delta
      flushPending(false);
      chunks.push({ data: unit, key: false });
    }
  }
  flushPending(false);

  return chunks;
}

/**
 * Extract avcC from the SPS/PPS found at the start of an Annex B stream.
 * Returns the AVCDecoderConfigurationRecord bytes.
 */
function buildAvcCFromAnnexB(stream: Uint8Array): { avcC: Uint8Array; profileIdc: number; levelIdc: number } | null {
  const nals: { type: number; data: Uint8Array }[] = [];
  let pos = findStartCode(stream, 0);
  while (pos >= 0) {
    const nextPos = findStartCode(stream, pos + 3);
    const unit = stream.slice(pos, nextPos >= 0 ? nextPos : stream.length);

    // Extract NAL payload (after start code)
    let nalStart: number;
    if (unit[2] === 1) nalStart = 3;
    else if (unit[3] === 1) nalStart = 4;
    else { pos = nextPos; continue; }

    const nalType = unit[nalStart] & 0x1f;
    if (nalType === 7 || nalType === 8) {
      nals.push({ type: nalType, data: unit.slice(nalStart) });
    } else if (nalType === 5 || nalType === 1) {
      break; // reached picture data — stop
    }
    pos = nextPos >= 0 ? nextPos : -1;
  }

  const sps = nals.find(n => n.type === 7);
  const pps = nals.find(n => n.type === 8);
  if (!sps || !pps) return null;

  const spsBytes = sps.data;
  const ppsBytes = pps.data;
  const profileIdc = spsBytes[1];
  const constraintFlags = spsBytes[2];
  const levelIdc = spsBytes[3];

  const avcC = new Uint8Array(
    5 + 1 + 2 + spsBytes.length + 1 + 2 + ppsBytes.length
  );
  let off = 0;
  avcC[off++] = 1;              // configurationVersion
  avcC[off++] = profileIdc;
  avcC[off++] = constraintFlags;
  avcC[off++] = levelIdc;
  avcC[off++] = 0xFF;           // lengthSizeMinusOne = 3 (4-byte NAL lengths)
  avcC[off++] = 0xE1;           // numOfSPS = 1
  avcC[off++] = (spsBytes.length >> 8) & 0xFF;
  avcC[off++] = spsBytes.length & 0xFF;
  avcC.set(spsBytes, off); off += spsBytes.length;
  avcC[off++] = 1;              // numOfPPS = 1
  avcC[off++] = (ppsBytes.length >> 8) & 0xFF;
  avcC[off++] = ppsBytes.length & 0xFF;
  avcC.set(ppsBytes, off);

  return { avcC, profileIdc, levelIdc };
}

export async function* decodeFramesWebCodecs(
  videoFile: File,
  signal: AbortSignal
): AsyncGenerator<DecodedFrame> {
  const { getFFmpeg } = await import('@/lib/ffmpeg/core');
  const ffmpeg = await getFFmpeg();

  const annexB = await extractAnnexBStream(ffmpeg, videoFile);
  console.debug('[WebCodecs] Annex B stream:', annexB.length, 'bytes');
  if (annexB.length === 0) throw new Error('ffmpeg produced empty H.264 stream');

  // Build avcC + derive codec string from actual SPS profile/level
  const avcCInfo = buildAvcCFromAnnexB(annexB);
  console.debug('[WebCodecs] avcC:', avcCInfo ? avcCInfo.avcC.length + ' bytes' : 'MISSING');
  if (!avcCInfo) throw new Error('No SPS/PPS found in H.264 stream');

  const { avcC, profileIdc, levelIdc } = avcCInfo;

  // Derive codec string from actual SPS profile/level/constraints
  const codec = `avc1.${profileIdc.toString(16).padStart(2, '0')}${avcC[2].toString(16).padStart(2, '0')}${levelIdc.toString(16).padStart(2, '0')}`;
  console.debug('[WebCodecs] codec:', codec);

  // Split into access units with SPS+PPS+IDR grouping
  const chunks = splitIntoAccessUnits(annexB);
  console.debug('[WebCodecs] access units:', chunks.length,
    'keyframes:', chunks.filter(c => c.key).length);
  if (chunks.length === 0) throw new Error('No access units in H.264 stream');

  // First chunk MUST be a keyframe
  if (!chunks[0].key) {
    console.error('[WebCodecs] first chunk is NOT a keyframe! NAL types in first 200 bytes:',
      Array.from(annexB.slice(0, 200)).map(b => b.toString(16).padStart(2, '0')).join(' '));
    throw new Error('First access unit is not a keyframe');
  }

  const config: VideoDecoderConfig = {
    codec,
    codedWidth: 1920,
    codedHeight: 1080,
    description: avcC.buffer.slice(avcC.byteOffset, avcC.byteOffset + avcC.byteLength),
  };

  const support = await VideoDecoder.isConfigSupported(config);
  console.debug('[WebCodecs] config supported:', support.supported);
  if (!support.supported) throw new Error('VideoDecoder config not supported');

  const frameQueue: VideoFrame[] = [];
  let decodeError: Error | null = null;

  const decoder = new VideoDecoder({
    output(frame: VideoFrame) { frameQueue.push(frame); },
    error(err: Error) { decodeError = err; },
  });

  decoder.configure(config);

  // Feed access units one at a time
  const frameIntervalUs = 33_333;
  let timestamp = 0;
  for (let i = 0; i < chunks.length; i++) {
    if (signal.aborted) break;
    decoder.decode(new EncodedVideoChunk({
      type: chunks[i].key ? 'key' : 'delta',
      timestamp,
      duration: frameIntervalUs,
      data: chunks[i].data,
    }));
    timestamp += frameIntervalUs;
    // Yield to event loop periodically
    if (i % 30 === 0) {
      await new Promise(r => setTimeout(r, 0));
      if (decodeError) throw decodeError;
    }
  }

  if (decodeError) throw decodeError;

  const flushOk = await Promise.race([
    decoder.flush().then(() => true),
    new Promise<boolean>(r => setTimeout(() => r(false), 120_000)),
  ]);
  if (!flushOk) { decoder.close(); throw new Error('Decoder flush timed out'); }
  if (decodeError) throw decodeError;

  console.debug('[WebCodecs] decoded', frameQueue.length, 'frames');

  const canvas = new OffscreenCanvas(1, 1);
  let ctx: OffscreenCanvasRenderingContext2D | null = null;

  for (let frameIndex = 0; frameIndex < frameQueue.length; frameIndex++) {
    const frame = frameQueue[frameIndex];
    if (signal.aborted) { frame.close(); continue; }
    const w = frame.displayWidth, h = frame.displayHeight;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w; canvas.height = h;
      ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    }
    if (!ctx) ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    ctx.drawImage(frame, 0, 0);
    yield { imageData: ctx.getImageData(0, 0, w, h), timestamp: frameIndex / 30, index: frameIndex };
    frame.close();
  }

  decoder.close();
  if (decodeError) throw decodeError;
}
