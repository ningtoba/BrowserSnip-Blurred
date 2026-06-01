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

/**
 * Extract the avcC (AVCDecoderConfigurationRecord) from an Annex B H.264
 * bitstream by finding the SPS and PPS NAL units at the start of the stream
 * and packaging them into the avcC binary format that WebCodecs expects.
 */
function buildAvcCFromAnnexB(stream: Uint8Array): Uint8Array | null {
  // Find all NAL units before the first non-SPS/PPS NAL
  const nals: { type: number; data: Uint8Array }[] = [];
  let i = 0;
  while (i < stream.length - 3) {
    if (stream[i] === 0 && stream[i + 1] === 0) {
      let scLen: number;
      if (stream[i + 2] === 1) scLen = 3;
      else if (stream[i + 2] === 0 && i + 3 < stream.length && stream[i + 3] === 1) scLen = 4;
      else { i++; continue; }

      const nalStart = i + scLen;
      let nalEnd = stream.length;
      for (let j = nalStart + 1; j < stream.length - 2; j++) {
        if (stream[j] === 0 && stream[j + 1] === 0 &&
            (stream[j + 2] === 1 || (stream[j + 2] === 0 && j + 3 < stream.length && stream[j + 3] === 1))) {
          nalEnd = j;
          break;
        }
      }

      const nalType = stream[nalStart] & 0x1f;
      if (nalType === 7 || nalType === 8) {
        nals.push({ type: nalType, data: stream.slice(nalStart, nalEnd) });
      } else if (nalType === 5 || nalType === 1) {
        break; // reached IDR or non-IDR slice — stop collecting
      } else {
        // Skip other NAL types (AUD, SEI, etc.)
      }
      i = nalEnd;
    } else {
      i++;
    }
  }

  const sps = nals.find(n => n.type === 7);
  const pps = nals.find(n => n.type === 8);
  if (!sps || !pps) return null;

  // Build AVCDecoderConfigurationRecord (ISO 14496-15)
  const spsBytes = sps.data;
  const ppsBytes = pps.data;

  // Parse minimal SPS fields
  const profileIdc = spsBytes[1];
  const constraintFlags = spsBytes[2];
  const levelIdc = spsBytes[3];

  const avcC = new Uint8Array(
    5 +                       // header (configVersion + profile + constraints + level + lengthSizeMinusOne)
    1 + 2 + spsBytes.length + // numOfSPS + SPS length + SPS data
    1 + 2 + ppsBytes.length   // numOfPPS + PPS length + PPS data
  );
  let off = 0;
  avcC[off++] = 1;            // configurationVersion
  avcC[off++] = profileIdc;   // AVCProfileIndication
  avcC[off++] = constraintFlags; // profile_compatibility
  avcC[off++] = levelIdc;     // AVCLevelIndication
  avcC[off++] = 0xFF;         // lengthSizeMinusOne = 3 (4-byte NAL lengths)

  // SPS
  avcC[off++] = 0xE1;         // numOfSequenceParameterSets = 1
  avcC[off++] = (spsBytes.length >> 8) & 0xFF;
  avcC[off++] = spsBytes.length & 0xFF;
  avcC.set(spsBytes, off); off += spsBytes.length;

  // PPS
  avcC[off++] = 1;            // numOfPictureParameterSets = 1
  avcC[off++] = (ppsBytes.length >> 8) & 0xFF;
  avcC[off++] = ppsBytes.length & 0xFF;
  avcC.set(ppsBytes, off);

  return avcC;
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

  // Build avcC description from the SPS/PPS in the Annex B stream
  const avcC = buildAvcCFromAnnexB(annexB);
  console.debug('[WebCodecs] avcC:', avcC ? avcC.length + ' bytes' : 'MISSING');

  // Feed the entire Annex B stream as a single keyframe chunk.
  // Chrome's built-in Annex B converter handles all NAL parsing internally.
  const config: VideoDecoderConfig = {
    codec: 'avc1.640028', // High@L4.0 — common for 1080p H.264
    codedWidth: 1920,
    codedHeight: 1080,
    ...(avcC ? { description: avcC.buffer.slice(avcC.byteOffset, avcC.byteOffset + avcC.byteLength) } : {}),
  };

  const support = await VideoDecoder.isConfigSupported(config);
  console.debug('[WebCodecs] config supported:', support.supported, 'config:', support.config);
  if (!support.supported) throw new Error('VideoDecoder config not supported');

  const frameQueue: VideoFrame[] = [];
  let decodeError: Error | null = null;

  const decoder = new VideoDecoder({
    output(frame: VideoFrame) { frameQueue.push(frame); },
    error(err: Error) { decodeError = err; },
  });

  decoder.configure(config);

  // Single keyframe chunk containing the entire Annex B bitstream
  decoder.decode(new EncodedVideoChunk({
    type: 'key',
    timestamp: 0,
    duration: 33_333,
    data: annexB,
  }));

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
