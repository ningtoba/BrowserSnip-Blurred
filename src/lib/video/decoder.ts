interface DecodedFrame {
  imageData: ImageData;
  timestamp: number;
  index: number;
}

interface MP4Sample {
  offset: number;
  size: number;
  key: boolean;
}

function readUint32(buf: Uint8Array, off: number): number {
  return ((buf[off] << 24) | (buf[off + 1] << 16) | (buf[off + 2] << 8) | buf[off + 3]) >>> 0;
}

function findBox(buf: Uint8Array, type: string, start: number, end: number): { offset: number; size: number } | null {
  let off = start;
  while (off < end - 8) {
    let size = readUint32(buf, off);
    if (size === 1) {
      // 64-bit extended size
      const hi = readUint32(buf, off + 8);
      const lo = readUint32(buf, off + 12);
      size = hi * 0x100000000 + lo - 8;
      if (size <= 0 || off + 16 + size > end) { off += 16 + size; continue; }
      const tag = String.fromCharCode(buf[off + 4], buf[off + 5], buf[off + 6], buf[off + 7]);
      if (tag === type) return { offset: off + 16, size };
      off += 16 + size;
    } else if (size === 0) {
      const tag = String.fromCharCode(buf[off + 4], buf[off + 5], buf[off + 6], buf[off + 7]);
      if (tag === type) return { offset: off + 8, size: end - off - 8 };
      break;
    } else if (size < 8 || off + size > end) {
      break;
    } else {
      const tag = String.fromCharCode(buf[off + 4], buf[off + 5], buf[off + 6], buf[off + 7]);
      if (tag === type) return { offset: off + 8, size: size - 8 };
      off += size;
    }
  }
  return null;
}

function demuxMP4(buf: Uint8Array): { samples: MP4Sample[]; avcC: Uint8Array } | null {
  try {
    const moov = findBox(buf, 'moov', 0, buf.length);
    if (!moov) return null;

    const moovEnd = moov.offset + moov.size;
    let trakOff = moov.offset;
    let avcC: Uint8Array | null = null;
    let stco: number[] = [], stsc: [number, number, number][] = [], stsz: number[] = [], stss = new Set<number>();

    while (trakOff < moovEnd - 8) {
      const tSize = readUint32(buf, trakOff);
      const tTag = String.fromCharCode(buf[trakOff + 4], buf[trakOff + 5], buf[trakOff + 6], buf[trakOff + 7]);
      if (tSize <= 0 || trakOff + tSize > moovEnd) break;

      if (tTag === 'trak') {
        const stbl = findBox(buf, 'stbl', trakOff + 8, trakOff + tSize);
        if (!stbl) { trakOff += tSize; continue; }
        const stblEnd = stbl.offset + stbl.size;

        const stsdBox = findBox(buf, 'stsd', stbl.offset, stblEnd);
        if (stsdBox) {
          const entryCount = readUint32(buf, stsdBox.offset + 4);
          if (entryCount > 0) {
            const entryStart = stsdBox.offset + 8;
            const entrySize = readUint32(buf, entryStart);
            const entryTag = String.fromCharCode(buf[entryStart + 4], buf[entryStart + 5], buf[entryStart + 6], buf[entryStart + 7]);
            if (entryTag === 'avc1' || entryTag === 'avc3' || entryTag === 'hvc1' || entryTag === 'hev1') {
              for (let s = entryStart + 8; s < entryStart + entrySize - 12; s++) {
                if ((buf[s]===0x61 && buf[s+1]===0x76 && buf[s+2]===0x63 && buf[s+3]===0x43) ||
                    (buf[s]===0x68 && buf[s+1]===0x76 && buf[s+2]===0x63 && buf[s+3]===0x43)) {
                  const cfgSize = readUint32(buf, s - 4);
                  if (cfgSize > 8 && cfgSize < entrySize) avcC = buf.slice(s + 4, s - 4 + cfgSize);
                  break;
                }
              }
            }
          }
        }

        let stcoBox = findBox(buf, 'stco', stbl.offset, stblEnd);
        if (!stcoBox) stcoBox = findBox(buf, 'co64', stbl.offset, stblEnd);
        if (stcoBox) {
          const count = readUint32(buf, stcoBox.offset + 4);
          for (let i = 0; i < count; i++) stco.push(readUint32(buf, stcoBox.offset + 8 + i * 4));
        }

        const stscBox = findBox(buf, 'stsc', stbl.offset, stblEnd);
        if (stscBox) {
          const count = readUint32(buf, stscBox.offset + 4);
          for (let i = 0; i < count; i++) {
            const base = stscBox.offset + 8 + i * 12;
            stsc.push([readUint32(buf, base), readUint32(buf, base + 4), readUint32(buf, base + 8)]);
          }
        }

        const stszBox = findBox(buf, 'stsz', stbl.offset, stblEnd);
        if (stszBox) {
          const count = readUint32(buf, stszBox.offset + 8);
          for (let i = 0; i < count; i++) stsz.push(readUint32(buf, stszBox.offset + 12 + i * 4));
        }

        const stssBox = findBox(buf, 'stss', stbl.offset, stblEnd);
        if (stssBox) {
          const count = readUint32(buf, stssBox.offset + 4);
          for (let i = 0; i < count; i++) stss.add(readUint32(buf, stssBox.offset + 8 + i * 4) - 1);
        }

        if (avcC && stco.length > 0 && stsz.length > 0) break;
      }
      trakOff += tSize;
    }

    if (!avcC || stco.length === 0 || stsz.length === 0) return null;

    const samples: MP4Sample[] = [];
    let stscIdx = 0;
    for (let ci = 0; ci < stco.length; ci++) {
      while (stscIdx + 1 < stsc.length && ci + 1 >= stsc[stscIdx + 1][0]) stscIdx++;
      const perChunk = stsc[stscIdx][1];
      let off = stco[ci];
      for (let s = 0; s < perChunk; s++) {
        const si = samples.length;
        if (si >= stsz.length) break;
        samples.push({ offset: off, size: stsz[si], key: stss.size === 0 || stss.has(si) });
        off += stsz[si];
      }
    }
    return { samples, avcC };
  } catch { return null; }
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
    error(err: Error) { console.debug('[WebCodecs] error:', err.message); decodeError = err; },
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

export async function* decodeFramesWebCodecs(
  videoFile: File,
  signal: AbortSignal
): AsyncGenerator<DecodedFrame> {
  const { getFFmpeg } = await import('@/lib/ffmpeg/core');
  const ffmpeg = await getFFmpeg();
  const inputBuf = new Uint8Array(await videoFile.arrayBuffer());

  // Use ffmpeg to remux ANY format to clean MP4 with H.264
  await ffmpeg.writeFile('input.bin', inputBuf);
  await ffmpeg.exec(
    ['-i', 'input.bin', '-c:v', 'copy', '-an', '-movflags', '+faststart', 'clean.mp4'],
    120_000
  );

  const mp4Data = new Uint8Array((await ffmpeg.readFile('clean.mp4')) as Uint8Array);
  try { await ffmpeg.deleteFile('input.bin'); } catch { /* ignore */ }
  try { await ffmpeg.deleteFile('clean.mp4'); } catch { /* ignore */ }

  const demuxed = demuxMP4(mp4Data);
  if (!demuxed) throw new Error('Failed to parse remuxed MP4');

  const codec = 'avc1.' +
    demuxed.avcC[1].toString(16).padStart(2, '0') +
    demuxed.avcC[2].toString(16).padStart(2, '0') +
    demuxed.avcC[3].toString(16).padStart(2, '0');
  const desc = new ArrayBuffer(demuxed.avcC.length);
  new Uint8Array(desc).set(demuxed.avcC);
  const chunks = demuxed.samples.map(s => ({
    data: mp4Data.buffer.slice(s.offset, s.offset + s.size),
    key: s.key,
  }));

  console.debug(`[WebCodecs] ffmpeg remux → ${chunks.length} samples, codec: ${codec}`);
  yield* decodeAndYield({ codec, description: desc }, chunks, signal);
}
