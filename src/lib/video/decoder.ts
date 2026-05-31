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
  return ((buf[off] << 24) | (buf[off + 1] << 16) | (buf[off + 2] << 8) | buf[off + 3]) >>> 0;
}

function readUintBE(buf: Uint8Array, off: number, size: number): number {
  let val = 0;
  for (let i = 0; i < size; i++) {
    val = (val << 8) | buf[off + i];
  }
  return val >>> 0;
}

function findBox(buf: Uint8Array, type: string, start: number, end: number): { offset: number; size: number } | null {
  let off = start;
  while (off < end - 8) {
    let size = readUint32(buf, off);
    if (size === 1) {
      // 64-bit extended size: read high 4 bytes and low 4 bytes
      const hi = readUint32(buf, off + 8);
      const lo = readUint32(buf, off + 12);
      size = hi * 0x100000000 + lo - 8; // subtract header for consistency
      if (size <= 0 || off + 16 + size > end) { off += 16 + size; continue; }
      const tag = String.fromCharCode(buf[off + 4], buf[off + 5], buf[off + 6], buf[off + 7]);
      if (tag === type) return { offset: off + 16, size };
      off += 16 + size;
    } else if (size === 0) {
      // Box extends to end of file
      const tag = String.fromCharCode(buf[off + 4], buf[off + 5], buf[off + 6], buf[off + 7]);
      if (tag === type) return { offset: off + 8, size: end - off - 8 };
      break; // No more boxes after an EOF-sized box
    } else if (size < 8 || off + size > end) {
      break; // Invalid size — stop parsing
    } else {
      const tag = String.fromCharCode(buf[off + 4], buf[off + 5], buf[off + 6], buf[off + 7]);
      if (tag === type) return { offset: off + 8, size: size - 8 };
      off += size;
    }
  }
  return null;
}

function demuxMP4(buf: Uint8Array): { samples: MP4Sample[]; avcC: Uint8Array; width: number; height: number } | null {
  try {
    // Log first bytes to identify format and structure
    const header64 = Array.from(buf.slice(0, 64)).map(b => b.toString(16).padStart(2, '0')).join(' ');
    console.debug('[demux] first 64 bytes:', header64);

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

// Minimal WebM/Matroska demuxer for VP8/VP9
function readVINT(buf: Uint8Array, off: number): { value: number; len: number } | null {
  if (off >= buf.length) return null;
  const first = buf[off];
  if (first === 0) return null;
  let len = 0;
  let mask = 0x80;
  while (!(first & mask) && mask > 0) { len++; mask >>= 1; }
  len++;
  if (off + len > buf.length) return null;
  let value = first & (mask - 1);
  for (let i = 1; i < len; i++) {
    value = (value << 8) | buf[off + i];
  }
  return { value, len };
}

function readVINTValue(buf: Uint8Array, off: number): number {
  const r = readVINT(buf, off);
  return r ? r.value : 0;
}

function readVINTLen(buf: Uint8Array, off: number): number {
  const r = readVINT(buf, off);
  return r ? r.len : 1;
}

interface WebMTrack {
  codec: string;
  codecPrivate?: Uint8Array;
  width: number;
  height: number;
}

function parseWebM(buf: Uint8Array): { tracks: WebMTrack[]; clusters: { timestamp: number; blocks: Uint8Array[] }[] } | null {
  try {
    // Find Segment (0x18538067) — skip EBML header and other top-level elements
    let off = 0;
    const scanned: string[] = [];
    while (off < buf.length - 4 && scanned.length < 20) {
      const idVint = readVINT(buf, off);
      if (!idVint) break;
      const szVint = readVINT(buf, off + idVint.len);
      if (!szVint) break;
      scanned.push(`0x${idVint.value.toString(16)}:${szVint.value}`);
      if (idVint.value === 0x08538067) { console.debug('[WebM] found Segment at', off, 'scanned:', scanned.join(', ')); break; }
      off += idVint.len + szVint.len + szVint.value;
    }
    if (off >= buf.length - 4) { console.debug('[WebM] segment not found, scanned:', scanned.join(', ')); return null; }

    const segStart = off;
    const segVint = readVINT(buf, off + readVINTLen(buf, off));
    const segSize = segVint ? segVint.value : (buf.length - segStart);
    const segEnd = Math.min(segStart + readVINTLen(buf, off) + segVint!.len + segSize, buf.length);

    off += readVINTLen(buf, off) + segVint!.len;

    const tracks: WebMTrack[] = [];
    const clusters: { timestamp: number; blocks: Uint8Array[] }[] = [];
    let currentTrack: WebMTrack | null = null;
    let clusterTime = 0;
    let currentBlock: { data: Uint8Array }[] = [];

    while (off < segEnd - 2) {
      const elVint = readVINT(buf, off);
      if (!elVint) break;
      const elId = elVint.value;
      off += elVint.len;
      const szVint = readVINT(buf, off);
      if (!szVint) break;
      const elSize = szVint.value;
      off += szVint.len;
      const elEnd = off + elSize;
      if (elEnd > segEnd) break;

      if (elId === 0x0654AE6B) {
        // Tracks
        let toff = off;
        while (toff < elEnd - 2) {
          const tv = readVINT(buf, toff); if (!tv) break;
          toff += tv.len;
          const ts = readVINT(buf, toff); if (!ts) break;
          toff += ts.len;
          const te = toff + ts.value;
          if (te > elEnd) break;

          if (tv.value === 0x2E) {
            // TrackEntry
            currentTrack = { codec: '', width: 0, height: 0 };
            let eoff = toff;
            while (eoff < te - 2) {
              const ev = readVINT(buf, eoff); if (!ev) break;
              eoff += ev.len;
              const es = readVINT(buf, eoff); if (!es) break;
              eoff += es.len;
              const ee = eoff + es.value;
              if (ee > te) break;
              if (ev.value === 0x06) currentTrack.codec = new TextDecoder().decode(buf.slice(eoff, ee));
              else if (ev.value === 0x23A2) { currentTrack.codecPrivate = buf.slice(eoff, ee); console.debug('[WebM] CodecPrivate:', currentTrack.codecPrivate.length, 'bytes, first bytes:', Array.from(currentTrack.codecPrivate.slice(0, 8)).map(b => b.toString(16).padStart(2,'0')).join(' ')); }
              else if (ev.value === 0x60) {
                // Video
                let voff = eoff;
                while (voff < ee - 2) {
                  const vv = readVINT(buf, voff); if (!vv) break;
                  voff += vv.len;
                  const vs = readVINT(buf, voff); if (!vs) break;
                  voff += vs.len;
                  const ve = voff + vs.value;
                  if (ve > ee) break;
                  if (vv.value === 0x30) currentTrack!.width = readUintBE(buf, voff, vs.value);
                  else if (vv.value === 0x3A) currentTrack!.height = readUintBE(buf, voff, vs.value);
                  voff = ve;
                }
              }
              eoff = ee;
            }
            if (currentTrack && currentTrack.codec) tracks.push(currentTrack);
          }
          toff = te;
        }
      } else if (elId === 0x0F43B675) {
        // Cluster
        clusterTime = 0;
        currentBlock = [];
        let coff = off;
        while (coff < elEnd - 2) {
          const cv = readVINT(buf, coff); if (!cv) break;
          coff += cv.len;
          const cs = readVINT(buf, coff); if (!cs) break;
          coff += cs.len;
          const ce = coff + cs.value;
          if (ce > elEnd) break;
          if (cv.value === 0x67) {
            clusterTime = readVINTValue(buf, coff);
          } else if (cv.value === 0x23) {
            const blockData = buf.slice(coff, ce);
            currentBlock.push({ data: blockData });
          }
          coff = ce;
        }
        if (currentBlock.length > 0) {
          clusters.push({ timestamp: clusterTime, blocks: currentBlock.map(b => b.data) });
        }
      }
      off = elEnd;
    }

    console.debug(`[WebM] ${tracks.length} tracks (${tracks.map(t => t.codec).join(', ')}), ${clusters.length} clusters, ${clusters.reduce((s, c) => s + c.blocks.length, 0)} blocks`);
    if (tracks.length === 0) { console.debug('[WebM] no video tracks found'); return null; }
    if (clusters.length === 0) { console.debug('[WebM] no clusters found'); return null; }
    return { tracks, clusters };
  } catch (e) {
    console.debug('[WebM] parse error:', e instanceof Error ? e.message : e);
    return null;
  }
}

async function* decodeAndYield(
  config: VideoDecoderConfig,
  chunks: { data: ArrayBuffer; key: boolean }[],
  signal: AbortSignal
): AsyncGenerator<DecodedFrame> {
  const support = await VideoDecoder.isConfigSupported(config);
  if (!support.supported) throw new Error(`VideoDecoder config not supported`);

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
    } else {
      ctx = ctx ?? canvas.getContext('2d', { willReadFrequently: true })!;
    }
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
  const buf = new Uint8Array(await videoFile.arrayBuffer());
  const isMP4 = buf.length > 8 && buf[4] === 0x66 && buf[5] === 0x74 && buf[6] === 0x79 && buf[7] === 0x70;

  if (isMP4) {
    const demuxed = demuxMP4(buf);
    if (!demuxed) throw new Error('Failed to parse MP4');
    const codec = 'avc1.' +
      demuxed.avcC[1].toString(16).padStart(2, '0') +
      demuxed.avcC[2].toString(16).padStart(2, '0') +
      demuxed.avcC[3].toString(16).padStart(2, '0');
    const desc = new ArrayBuffer(demuxed.avcC.length);
    new Uint8Array(desc).set(demuxed.avcC);
    const chunks = demuxed.samples.map(s => ({
      data: buf.buffer.slice(s.offset, s.offset + s.size),
      key: s.key,
    }));
    console.debug(`[WebCodecs] MP4: ${chunks.length} samples, codec: ${codec}`);
    yield* decodeAndYield({ codec, description: desc }, chunks, signal);
  } else {
    // Try WebM
    const webm = parseWebM(buf);
    console.debug('[WebCodecs] WebM parse result:', webm ? `tracks=${webm.tracks.length} clusters=${webm.clusters.length}` : 'null');
    if (!webm || webm.tracks.length === 0) throw new Error('Unsupported format — could not parse video track');
    const track = webm.tracks[0];
    console.debug('[WebCodecs] track codec:', track.codec, 'size:', track.width, 'x', track.height);
    let config: VideoDecoderConfig;
    if (track.codec.startsWith('V_VP8')) {
      config = { codec: 'vp8' };
    } else if (track.codec.startsWith('V_VP9')) {
      config = { codec: 'vp09.00.10.08' };
    } else if (track.codec.startsWith('V_AV1')) {
      config = { codec: 'av01.0.04M.08' };
    } else if (track.codec === 'V_MPEG4/ISO/AVC') {
      // H.264 in MKV — use CodecPrivate as avcC
      if (!track.codecPrivate || track.codecPrivate.length < 7) throw new Error('H.264 in MKV missing CodecPrivate');
      const desc = new ArrayBuffer(track.codecPrivate.length);
      new Uint8Array(desc).set(track.codecPrivate);
      config = {
        codec: 'avc1.' +
          track.codecPrivate[1].toString(16).padStart(2, '0') +
          track.codecPrivate[2].toString(16).padStart(2, '0') +
          track.codecPrivate[3].toString(16).padStart(2, '0'),
        description: desc,
      };
    } else {
      throw new Error(`Unsupported codec: ${track.codec}`);
    }
    const chunks: { data: ArrayBuffer; key: boolean }[] = [];
    for (const cluster of webm.clusters) {
      for (const block of cluster.blocks) {
        if (block.length < 4) continue;
        // SimpleBlock: track(VINT) + timecode(2B) + flags(1B) + data
        const tkVint = readVINT(block, 0);
        if (!tkVint) continue;
        const flagsOff = tkVint.len + 2;
        const dataOff = flagsOff + 1;
        if (dataOff >= block.length) continue;
        const key = (block[flagsOff] & 0x80) !== 0;
        const frameData = block.slice(dataOff);
        const ab = new ArrayBuffer(frameData.length);
        new Uint8Array(ab).set(frameData);
        chunks.push({ data: ab, key });
      }
    }
    console.debug(`[WebCodecs] WebM/MKV: ${chunks.length} blocks, codec: ${JSON.stringify(config.codec)}`);
    yield* decodeAndYield(config, chunks, signal);
  }
}
