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

/** Read a box header (size + fourCC) at the given offset */
function parseBox(buf: Uint8Array, off: number): { size: number; tag: string; dataOff: number } | null {
  if (off + 8 > buf.length) return null;
  const size = ((buf[off] << 24) | (buf[off+1] << 16) | (buf[off+2] << 8) | buf[off+3]) >>> 0;
  const tag = String.fromCharCode(buf[off+4], buf[off+5], buf[off+6], buf[off+7]);
  if (size < 8 || off + size > buf.length) return null;
  return { size, tag, dataOff: off + 8 };
}

/**
 * Find a named box within a byte range. Walks the top-level children
 * of the region [start, start+len) and returns the first match.
 */
function findBox(buf: Uint8Array, start: number, len: number, target: string): { off: number; size: number } | null {
  let off = start;
  const end = start + len;
  while (off + 8 <= end) {
    const box = parseBox(buf, off);
    if (!box) break;
    if (box.tag === target) return { off, size: box.size };
    off += box.size;
  }
  return null;
}

/**
 * Extract the avcC box bytes from an MP4 buffer by traversing:
 *   top-level → moov → trak → stbl → stsd → avc1 → avcC
 * Returns the FULL avcC box (8-byte header + payload) so the description
 * field matches the format mp4box.js returns via track.description.
 */
function extractAvcC(mp4Buf: ArrayBuffer): Uint8Array | null {
  const buf = new Uint8Array(mp4Buf);
  try {
    // Find top-level moov box (may not be at offset 0 if ftyp comes first)
    const moov = findBox(buf, 0, buf.length, 'moov');
    if (!moov) { console.debug('[extractAvcC] no moov box found'); return null; }
    console.debug('[extractAvcC] moov at offset', moov.off, 'size', moov.size);

    // moov → trak
    const trak = findBox(buf, moov.off + 8, moov.size - 8, 'trak');
    if (!trak) { console.debug('[extractAvcC] no trak inside moov'); return null; }
    console.debug('[extractAvcC] trak at offset', trak.off, 'size', trak.size);

    // trak → mdia
    const mdia = findBox(buf, trak.off + 8, trak.size - 8, 'mdia');
    if (!mdia) { console.debug('[extractAvcC] no mdia inside trak'); return null; }

    // mdia → minf
    const minf = findBox(buf, mdia.off + 8, mdia.size - 8, 'minf');
    if (!minf) { console.debug('[extractAvcC] no minf inside mdia'); return null; }

    // minf → stbl
    const stbl = findBox(buf, minf.off + 8, minf.size - 8, 'stbl');
    if (!stbl) { console.debug('[extractAvcC] no stbl inside minf'); return null; }

    // stbl → stsd
    const stsd = findBox(buf, stbl.off + 8, stbl.size - 8, 'stsd');
    if (!stsd) { console.debug('[extractAvcC] no stsd inside stbl'); return null; }
    console.debug('[extractAvcC] stsd at offset', stsd.off, 'size', stsd.size);

    // Inside stsd, skip the 8-byte version+flags + 4-byte entry_count to reach the first entry
    // stsd content: [version(1) + flags(3)] + [entry_count(4)] + [entries...]
    const entryOff = stsd.off + 8 + 4 + 4; // box header + full-box header + entry count
    const entry = parseBox(buf, entryOff);
    if (!entry) { console.debug('[extractAvcC] no sample entry in stsd'); return null; }
    console.debug('[extractAvcC] sample entry:', entry.tag, 'at offset', entryOff, 'size', entry.size);

    // The entry should be avc1, avc3, hev1, hvc1, etc.
    // Find avcC inside the entry's child boxes (skip the fixed VisualSampleEntry fields)
    // VisualSampleEntry fixed fields = 78 bytes after the 8-byte box header
    const VIDEO_ENTRY_FIXED = 78;
    const childStart = entryOff + 8 + VIDEO_ENTRY_FIXED;
    const childLen = entry.size - 8 - VIDEO_ENTRY_FIXED;

    const avcC = findBox(buf, childStart, childLen, 'avcC');
    if (!avcC) {
      // Fallback: scan the entire entry for avcC fourCC (handles non-standard layouts)
      console.debug('[extractAvcC] avcC not at expected offset, scanning entry...');
      for (let i = entryOff + 8; i < entryOff + entry.size - 8; i++) {
        if (buf[i] === 0x61 && buf[i+1] === 0x76 && buf[i+2] === 0x63 && buf[i+3] === 0x43) {
          const boxSize = ((buf[i-4] << 24) | (buf[i-3] << 16) | (buf[i-2] << 8) | buf[i-1]) >>> 0;
          if (boxSize >= 8 && boxSize < 1000 && i - 4 + boxSize <= entryOff + entry.size) {
            const result = buf.slice(i - 4, i - 4 + boxSize);
            console.debug('[extractAvcC] found avcC via scan:', result.length, 'bytes');
            return result;
          }
        }
      }
      console.debug('[extractAvcC] no avcC found in sample entry');
      return null;
    }

    // Return the FULL avcC box (header + payload) — matches what mp4box.js returns
    const result = buf.slice(avcC.off, avcC.off + avcC.size);
    console.debug('[extractAvcC] found avcC via traversal:', result.length, 'bytes');
    return result;
  } catch (e) {
    console.debug('[extractAvcC] error:', e);
    return null;
  }
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

      // track.description may exist at runtime even though it's not in the mp4box types
      const trackAny = track as any;
      console.debug('[MP4Box] track:', {
        codec: track.codec,
        width: track.track_width,
        height: track.track_height,
        nb_samples: track.nb_samples,
        description: trackAny.description ? `${(trackAny.description as Uint8Array).length} bytes` : 'undefined',
      });

      // Try multiple sources for the avcC description
      let desc: Uint8Array | undefined = trackAny.description as Uint8Array | undefined;

      // Fallback 2: extract avcC directly from the MP4 buffer via box traversal
      if (!desc || desc.length === 0) {
        desc = extractAvcC(arrayBuf) ?? undefined;
        console.debug('[MP4Box] avcC from buffer scan:', desc?.length, 'bytes');
      }

      if (desc && desc.length > 0) {
        const descBuf = new ArrayBuffer(desc.length);
        new Uint8Array(descBuf).set(desc);
        (config as any).description = descBuf;
        console.debug('[MP4Box] description set:', descBuf.byteLength, 'bytes, first 4:',
          Array.from(new Uint8Array(descBuf).slice(0, 4)).map(b => '0x' + b.toString(16)).join(' '));
      } else {
        console.warn('[MP4Box] NO avcC description found — VideoDecoder will likely fail');
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
          'is_sync:', s?.is_sync);
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
        console.debug('[MP4Box] resolved:', pendingSamples.length, 'samples,',
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
