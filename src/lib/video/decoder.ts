interface DecodedFrame {
  imageData: ImageData;
  timestamp: number;
  index: number;
}

// Minimal MP4 demuxer — extracts H.264 samples in length-prefixed format
interface MP4Sample {
  offset: number;
  size: number;
  key: boolean;
}

function readUint32(buf: Uint8Array, off: number): number {
  return (buf[off] << 24) | (buf[off + 1] << 16) | (buf[off + 2] << 8) | buf[off + 3];
}

function findBox(buf: Uint8Array, type: string, start: number, end: number): { offset: number; size: number } | null {
  let off = start;
  while (off < end - 8) {
    const size = readUint32(buf, off);
    const tag = String.fromCharCode(buf[off + 4], buf[off + 5], buf[off + 6], buf[off + 7]);
    if (size <= 0 || off + size > end) break;
    if (tag === type) return { offset: off + 8, size: size - 8 };
    off += size;
  }
  return null;
}

function demuxMP4(buf: Uint8Array): { samples: MP4Sample[]; avcC: Uint8Array; width: number; height: number } | null {
  try {
    // Log top-level boxes
    const topBoxes: string[] = [];
    let scanOff = 0;
    while (scanOff < Math.min(buf.length, 200) - 8) {
      const sz = readUint32(buf, scanOff);
      const tg = String.fromCharCode(buf[scanOff + 4], buf[scanOff + 5], buf[scanOff + 6], buf[scanOff + 7]);
      if (sz <= 0 || scanOff + sz > buf.length) break;
      topBoxes.push(tg + ':' + sz);
      scanOff += sz;
    }
    console.debug('[demux] top-level boxes:', topBoxes.join(', '), '| file size:', (buf.length / 1024 / 1024).toFixed(1), 'MB');

    const moov = findBox(buf, 'moov', 0, buf.length);
    if (!moov) { console.debug('[demux] moov not found in', buf.length, 'bytes'); return null; }

    const moovEnd = moov.offset + moov.size;
    let trakOff = moov.offset;
    let avcC: Uint8Array | null = null;
    let width = 0, height = 0;
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
            console.debug('[demux] stsd entry:', entryTag, 'size:', entrySize);
            if (entryTag === 'avc1' || entryTag === 'avc3' || entryTag === 'hvc1' || entryTag === 'hev1') {
              width = (buf[entryStart + 24] << 8) | buf[entryStart + 25];
              height = (buf[entryStart + 26] << 8) | buf[entryStart + 27];
              for (let s = entryStart + 8; s < entryStart + entrySize - 12; s++) {
                if (buf[s] === 0x61 && buf[s+1] === 0x76 && buf[s+2] === 0x63 && buf[s+3] === 0x43) {
                  const avcCSize = readUint32(buf, s - 4);
                  if (avcCSize > 8 && avcCSize < entrySize) {
                    avcC = buf.slice(s + 4, s - 4 + avcCSize);
                    console.debug('[demux] found avcC at offset', s - 4, 'size:', avcC.length);
                  }
                  break;
                }
                // Also check for hvcC (HEVC)
                if (buf[s] === 0x68 && buf[s+1] === 0x76 && buf[s+2] === 0x63 && buf[s+3] === 0x43) {
                  const hvcCSize = readUint32(buf, s - 4);
                  if (hvcCSize > 8 && hvcCSize < entrySize) {
                    avcC = buf.slice(s + 4, s - 4 + hvcCSize);
                    console.debug('[demux] found hvcC at offset', s - 4, 'size:', avcC.length);
                  }
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
          for (let i = 0; i < count; i++) {
            stco.push(readUint32(buf, stcoBox.offset + 8 + i * 4));
          }
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
          for (let i = 0; i < count; i++) {
            stsz.push(readUint32(buf, stszBox.offset + 12 + i * 4));
          }
        }

        const stssBox = findBox(buf, 'stss', stbl.offset, stblEnd);
        if (stssBox) {
          const count = readUint32(buf, stssBox.offset + 4);
          for (let i = 0; i < count; i++) {
            stss.add(readUint32(buf, stssBox.offset + 8 + i * 4) - 1);
          }
        }

        console.debug('[demux] stco:', stco.length, 'stsc:', stsc.length, 'stsz:', stsz.length, 'stss:', stss.size);
        if (avcC && stco.length > 0 && stsc.length > 0 && stsz.length > 0) break;
      }
      trakOff += tSize;
    }

    if (!avcC) { console.debug('[demux] no avcC found'); return null; }
    if (stco.length === 0) { console.debug('[demux] no stco'); return null; }
    if (stsc.length === 0) { console.debug('[demux] no stsc'); return null; }
    if (stsz.length === 0) { console.debug('[demux] no stsz'); return null; }

    const samples: MP4Sample[] = [];
    let stscIdx = 0;

    for (let chunkIdx = 0; chunkIdx < stco.length; chunkIdx++) {
      while (stscIdx + 1 < stsc.length && chunkIdx + 1 >= stsc[stscIdx + 1][0]) {
        stscIdx++;
      }
      const [, samplesPerChunk] = stsc[stscIdx];
      let chunkOffset = stco[chunkIdx];

      for (let s = 0; s < samplesPerChunk; s++) {
        const sampleIdx = samples.length;
        if (sampleIdx >= stsz.length) break;
        samples.push({
          offset: chunkOffset,
          size: stsz[sampleIdx],
          key: stss.size === 0 || stss.has(sampleIdx),
        });
        chunkOffset += stsz[sampleIdx];
      }
    }

    console.debug('[demux] built', samples.length, 'samples');
    return { samples, avcC, width, height };
  } catch (e) {
    console.debug('[demux] error:', e instanceof Error ? e.message : e);
    return null;
  }
}

export async function* decodeFramesWebCodecs(
  videoFile: File,
  signal: AbortSignal
): AsyncGenerator<DecodedFrame> {
  const buf = new Uint8Array(await videoFile.arrayBuffer());
  const demuxed = demuxMP4(buf);
  if (!demuxed) throw new Error('Failed to demux MP4 — unsupported format');

  const { samples, avcC } = demuxed;
  console.debug(`[WebCodecs] demuxed ${samples.length} samples, avcC: ${avcC.length} bytes`);

  // Build codec string from avcC
  const codec = 'avc1.' +
    avcC[1].toString(16).padStart(2, '0') +
    avcC[2].toString(16).padStart(2, '0') +
    avcC[3].toString(16).padStart(2, '0');

  // avcC is already in the correct format — just need a standalone ArrayBuffer
  const descBuf = new ArrayBuffer(avcC.length);
  new Uint8Array(descBuf).set(avcC);

  const config: VideoDecoderConfig = { codec, description: descBuf };
  console.debug('[WebCodecs] codec:', codec);

  const support = await VideoDecoder.isConfigSupported(config);
  console.debug('[WebCodecs] isConfigSupported:', support.supported);
  if (!support.supported) throw new Error(`VideoDecoder config not supported for ${codec}`);

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

  // Feed MP4 samples directly (already in length-prefixed format)
  let timestamp = 0;
  const frameIntervalUs = 33_333;

  for (let i = 0; i < samples.length; i++) {
    if (signal.aborted) break;

    const sample = samples[i];
    const sampleData = buf.subarray(sample.offset, sample.offset + sample.size);
    const chunkBuf = new ArrayBuffer(sampleData.length);
    new Uint8Array(chunkBuf).set(sampleData);

    try {
      decoder.decode(new EncodedVideoChunk({
        type: sample.key ? 'key' : 'delta',
        timestamp,
        duration: frameIntervalUs,
        data: chunkBuf,
      }));
    } catch {
      await new Promise(r => setTimeout(r, 10));
      decoder.decode(new EncodedVideoChunk({
        type: sample.key ? 'key' : 'delta',
        timestamp,
        duration: frameIntervalUs,
        data: chunkBuf,
      }));
    }
    timestamp += frameIntervalUs;

    if (i % 30 === 0) {
      await new Promise(r => setTimeout(r, 0));
      if (decodeError) throw decodeError;
    }
  }

  if (decodeError) throw decodeError;

  console.debug(`[WebCodecs] ${samples.length} samples sent, flushing...`);
  const flushOk = await Promise.race([
    decoder.flush().then(() => true),
    new Promise<boolean>(r => setTimeout(() => r(false), 30000)),
  ]);

  if (!flushOk) {
    decoder.close();
    throw new Error('Decoder flush timed out after 30s');
  }

  console.debug(`[WebCodecs] flush done, ${frameQueue.length} frames`);

  const canvas = new OffscreenCanvas(1, 1);
  let ctx: OffscreenCanvasRenderingContext2D | null = null;

  for (const frame of frameQueue) {
    if (signal.aborted) { frame.close(); continue; }
    const w = frame.displayWidth, h = frame.displayHeight;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w; canvas.height = h;
      ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    } else {
      ctx = ctx ?? canvas.getContext('2d', { willReadFrequently: true })!;
    }
    ctx.drawImage(frame, 0, 0);
    const imageData = ctx.getImageData(0, 0, w, h);
    frame.close();
    yield { imageData, timestamp: frameIndex / 30, index: frameIndex };
    frameIndex++;
  }

  decoder.close();
  if (decodeError) throw decodeError;
}
