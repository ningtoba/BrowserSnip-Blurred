import { getFFmpeg } from '@/lib/ffmpeg/core';

interface H264Sample {
  data: Uint8Array;
  type: 'sps' | 'pps' | 'idr' | 'non-idr';
}

interface DecodedFrame {
  imageData: ImageData;
  timestamp: number;
  index: number;
}

function parseAnnexB(buffer: Uint8Array): H264Sample[] {
  const samples: H264Sample[] = [];
  let i = 0;
  const len = buffer.length;

  while (i < len - 4) {
    // Find start code: 0x00 0x00 0x00 0x01 or 0x00 0x00 0x01
    let startLen = 0;
    if (buffer[i] === 0 && buffer[i + 1] === 0) {
      if (buffer[i + 2] === 0 && buffer[i + 3] === 1) {
        startLen = 4;
      } else if (buffer[i + 2] === 1) {
        startLen = 3;
      }
    }

    if (startLen > 0) {
      const nalStart = i + startLen;
      // Find next start code
      let nextStart = -1;
      for (let j = nalStart; j < len - 3; j++) {
        if (buffer[j] === 0 && buffer[j + 1] === 0) {
          if ((buffer[j + 2] === 0 && buffer[j + 3] === 1) || buffer[j + 2] === 1) {
            nextStart = j;
            break;
          }
        }
      }
      if (nextStart === -1) nextStart = len;

      const nalData = buffer.slice(nalStart, nextStart);
      if (nalData.length > 0) {
        const nalType = nalData[0] & 0x1f;
        let type: H264Sample['type'];
        if (nalType === 7) type = 'sps';
        else if (nalType === 8) type = 'pps';
        else if (nalType === 5) type = 'idr';
        else type = 'non-idr';

        samples.push({ data: nalData, type });
      }
      i = nextStart;
    } else {
      i++;
    }
  }

  return samples;
}

function buildCodecConfig(sps: Uint8Array, pps: Uint8Array): { config: VideoDecoderConfig; descBytes: Uint8Array } {
  const desc = new Uint8Array(8 + sps.length + 3 + pps.length);
  desc[0] = 1;
  desc[1] = sps[1];
  desc[2] = sps[2];
  desc[3] = sps[3];
  desc[4] = 0xff;
  desc[5] = 0xe1;
  desc[6] = (sps.length >> 8) & 0xff;
  desc[7] = sps.length & 0xff;
  desc.set(sps, 8);
  const off = 8 + sps.length;
  desc[off] = 1;
  desc[off + 1] = (pps.length >> 8) & 0xff;
  desc[off + 2] = pps.length & 0xff;
  desc.set(pps, off + 3);

  const codec = 'avc1.' +
    sps[1].toString(16).padStart(2, '0') +
    sps[2].toString(16).padStart(2, '0') +
    sps[3].toString(16).padStart(2, '0');

  // Copy to a standalone ArrayBuffer (not backed by the Uint8Array)
  const buf = new ArrayBuffer(desc.length);
  new Uint8Array(buf).set(desc);

  return { config: { codec, description: buf }, descBytes: desc };
}

export async function* decodeFramesWebCodecs(
  videoFile: File,
  signal: AbortSignal
): AsyncGenerator<DecodedFrame> {
  const ffmpeg = await getFFmpeg();

  // Write input video to MEMFS and extract Annex B bitstream
  const inputBuf = new Uint8Array(await videoFile.arrayBuffer());
  await ffmpeg.writeFile('input.mp4', inputBuf);

  try {
    await ffmpeg.exec(
      ['-i', 'input.mp4', '-c:v', 'copy', '-bsf:v', 'h264_mp4toannexb', '-an', '-f', 'h264', 'video.h264'],
      120_000
    );
  } catch {
    // Try without the bitstream filter (for non-H.264 codecs)
    await ffmpeg.exec(
      ['-i', 'input.mp4', '-c:v', 'copy', '-an', '-f', 'h264', 'video.h264'],
      120_000
    );
  }

  // Read the raw H.264 bitstream
  const h264Data = new Uint8Array((await ffmpeg.readFile('video.h264')) as Uint8Array);

  try { await ffmpeg.deleteFile('input.mp4'); } catch { /* ignore */ }
  try { await ffmpeg.deleteFile('video.h264'); } catch { /* ignore */ }

  // Parse Annex B
  const samples = parseAnnexB(h264Data);
  if (samples.length === 0) throw new Error('No H.264 samples found');

  const counts = { sps: 0, pps: 0, idr: 0, 'non-idr': 0 };
  for (const s of samples) counts[s.type]++;
  console.debug(`[WebCodecs] ${samples.length} NALs:`, counts);

  // Find SPS and PPS for codec config
  const spsSample = samples.find((s) => s.type === 'sps');
  const ppsSample = samples.find((s) => s.type === 'pps');
  if (!spsSample || !ppsSample) throw new Error('SPS/PPS not found in bitstream');

  // Build avcC description from SPS/PPS (required for AVC/H.264)
  const sps = spsSample.data;
  const pps = ppsSample.data;
  const codec = 'avc1.' +
    sps[1].toString(16).padStart(2, '0') +
    sps[2].toString(16).padStart(2, '0') +
    sps[3].toString(16).padStart(2, '0');

  const avcC = new Uint8Array(8 + sps.length + 3 + pps.length);
  avcC[0] = 1; avcC[1] = sps[1]; avcC[2] = sps[2]; avcC[3] = sps[3];
  avcC[4] = 0xff; avcC[5] = 0xe1;
  avcC[6] = (sps.length >> 8) & 0xff; avcC[7] = sps.length & 0xff;
  avcC.set(sps, 8);
  const off = 8 + sps.length;
  avcC[off] = 1; avcC[off + 1] = (pps.length >> 8) & 0xff; avcC[off + 2] = pps.length & 0xff;
  avcC.set(pps, off + 3);

  const config: VideoDecoderConfig = { codec, description: avcC.buffer.slice(0) as ArrayBuffer };
  console.debug('[WebCodecs] codec:', codec, 'avcC:', avcC.length, 'bytes');

  const support = await VideoDecoder.isConfigSupported(config);
  console.debug('[WebCodecs] isConfigSupported:', support.supported);
  if (!support.supported) {
    throw new Error(`VideoDecoder config not supported for ${codec}`);
  }

  // Prepare frames for decoding
  let frameIndex = 0;
  const frameQueue: VideoFrame[] = [];
  let decodeError: Error | null = null;

  const decoder = new VideoDecoder({
    output(frame: VideoFrame) {
      frameQueue.push(frame);
    },
    error(err: Error) {
      decodeError = err;
    },
  });

  decoder.configure(config);

  // Feed NAL units in length-prefixed format (avcC mode)
  // Each chunk: 4-byte BE length + NAL data (no start code)
  let timestamp = 0;
  const frameIntervalUs = 33_333;
  let foundKeyframe = false;

  for (const sample of samples) {
    if (signal.aborted) break;
    if (sample.type === 'sps' || sample.type === 'pps') continue;
    if (sample.data.length === 0) continue;

    if (!foundKeyframe && sample.type !== 'idr') continue;
    foundKeyframe = true;

    // 4-byte big-endian length prefix + NAL data
    const nalLen = sample.data.length;
    const chunkBuf = new ArrayBuffer(4 + nalLen);
    const view = new DataView(chunkBuf);
    view.setUint32(0, nalLen, false); // big-endian
    new Uint8Array(chunkBuf, 4).set(sample.data);

    decoder.decode(new EncodedVideoChunk({
      type: sample.type === 'idr' ? 'key' : 'delta',
      timestamp,
      duration: frameIntervalUs,
      data: chunkBuf,
    }));
    timestamp += frameIntervalUs;
  }

  if (!foundKeyframe) throw new Error('No keyframe found in bitstream');
  if (decodeError) throw decodeError;

  console.debug(`[WebCodecs] ${timestamp / frameIntervalUs} chunks sent, flushing...`);
  await decoder.flush();
  console.debug(`[WebCodecs] flush done, ${frameQueue.length} frames decoded`);

  if (frameQueue.length === 0) {
    decoder.close();
    if (decodeError) throw decodeError;
    throw new Error('Decoder produced 0 frames');
  }

  // Read decoded frames
  const canvas = new OffscreenCanvas(1, 1);
  let ctx: OffscreenCanvasRenderingContext2D | null = null;

  for (const frame of frameQueue) {
    if (signal.aborted) {
      frame.close();
      continue;
    }

    const w = frame.displayWidth;
    const h = frame.displayHeight;

    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    } else {
      ctx = ctx ?? canvas.getContext('2d', { willReadFrequently: true })!;
    }

    ctx.drawImage(frame, 0, 0);
    const imageData = ctx.getImageData(0, 0, w, h);

    const result: DecodedFrame = {
      imageData,
      timestamp: frame.timestamp / 1_000_000,
      index: frameIndex,
    };

    frameIndex++;
    frame.close();
    yield result;
  }

  decoder.close();

  if (decodeError) throw decodeError;
}
