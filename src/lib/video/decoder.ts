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

function buildCodecConfig(sps: Uint8Array, pps: Uint8Array): VideoDecoderConfig {
  // Build avcC description: https://www.w3.org/TR/webcodecs-avc-codec-registration/
  const desc = new Uint8Array(7 + sps.length + 1 + 2 + pps.length);
  desc[0] = 1;       // configurationVersion
  desc[1] = sps[1];  // AVCProfileIndication
  desc[2] = sps[2];  // profile_compatibility
  desc[3] = sps[3];  // AVCLevelIndication
  desc[4] = 0xff;    // lengthSizeMinusOne | reserved
  desc[5] = 0xe1;    // numOfSequenceParameterSets | reserved
  desc[6] = (sps.length >> 8) & 0xff;
  desc[7] = sps.length & 0xff;
  desc.set(sps, 8);
  const ppsOffset = 8 + sps.length;
  desc[ppsOffset] = 1; // numOfPictureParameterSets
  desc[ppsOffset + 1] = (pps.length >> 8) & 0xff;
  desc[ppsOffset + 2] = pps.length & 0xff;
  desc.set(pps, ppsOffset + 3);

  return {
    codec: 'avc1.' + [
      sps[1].toString(16).padStart(2, '0'),
      sps[2].toString(16).padStart(2, '0'),
      sps[3].toString(16).padStart(2, '0'),
    ].join(''),
    description: desc.buffer as AllowSharedBufferSource,
  };
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

  // Find SPS and PPS for codec config
  const spsSample = samples.find((s) => s.type === 'sps');
  const ppsSample = samples.find((s) => s.type === 'pps');
  if (!spsSample || !ppsSample) throw new Error('SPS/PPS not found in bitstream');

  const config = buildCodecConfig(spsSample.data, ppsSample.data);

  // Prepare frames for decoding
  let frameIndex = 0;
  const frameQueue: VideoFrame[] = [];
  let decodeDone = false;
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

  // Feed all NAL units as chunks, grouping non-IDR after an IDR
  let currentChunkType: EncodedVideoChunkType = 'key';
  let chunkData: Uint8Array[] = [];
  let timestamp = 0;
  const frameIntervalUs = 33_333; // ~30fps

  for (const sample of samples) {
    if (signal.aborted) break;

    if (sample.type === 'sps' || sample.type === 'pps') continue;

    if (sample.type === 'idr') {
      // Flush previous chunk
      if (chunkData.length > 0) {
        const totalLen = chunkData.reduce((s, d) => s + d.length, 0);
        const merged = new Uint8Array(totalLen);
        let offset = 0;
        for (const d of chunkData) { merged.set(d, offset); offset += d.length; }
        decoder.decode(new EncodedVideoChunk({
          type: currentChunkType,
          timestamp,
          duration: frameIntervalUs,
          data: merged.buffer as AllowSharedBufferSource,
        }));
        chunkData = [];
        timestamp += frameIntervalUs;
      }
      currentChunkType = 'key';
      chunkData.push(sample.data);
      // Also include subsequent non-IDR up to the next IDR or slice
    } else if (chunkData.length > 0) {
      // Check if this is a new frame (first slice of a new picture)
      const nalType = sample.data[0] & 0x1f;
      const isSlice = nalType === 1 || nalType === 5;
      const isFirstSlice = isSlice && (sample.data[1] & 0x80);

      if (isSlice && isFirstSlice && chunkData.length > 0) {
        // Flush as a new delta frame
        const totalLen = chunkData.reduce((s, d) => s + d.length, 0);
        const merged = new Uint8Array(totalLen);
        let offset = 0;
        for (const d of chunkData) { merged.set(d, offset); offset += d.length; }
        decoder.decode(new EncodedVideoChunk({
          type: 'delta',
          timestamp,
          duration: frameIntervalUs,
          data: merged.buffer as AllowSharedBufferSource,
        }));
        chunkData = [];
        timestamp += frameIntervalUs;
      }
      chunkData.push(sample.data);
    }
  }

  // Flush remaining
  if (chunkData.length > 0) {
    const totalLen = chunkData.reduce((s, d) => s + d.length, 0);
    const merged = new Uint8Array(totalLen);
    let offset = 0;
    for (const d of chunkData) { merged.set(d, offset); offset += d.length; }
    decoder.decode(new EncodedVideoChunk({
      type: currentChunkType,
      timestamp,
      duration: frameIntervalUs,
      data: merged.buffer as AllowSharedBufferSource,
    }));
  }

  await decoder.flush();

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
