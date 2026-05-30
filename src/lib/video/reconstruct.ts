import { getFFmpeg } from '@/lib/ffmpeg/core';
import { reconstructCommand, concatCommand } from '@/lib/ffmpeg/commands';
import type { VideoMetadata } from '@/types';
import type { FFmpeg } from '@ffmpeg/ffmpeg';
import { BATCH_SIZE } from '@/lib/constants';

export async function reconstructVideo(
  totalFrames: number,
  metadata: VideoMetadata,
  onProgress?: (phase: string, percent: number) => void
): Promise<Blob> {
  const ffmpeg = await getFFmpeg();
  const fps = metadata.fps;

  if (totalFrames <= BATCH_SIZE) {
    onProgress?.('encoding', 50);
    await ffmpeg.exec(
      reconstructCommand('frame_%04d.png', 'output.mp4', fps),
      300_000
    );
    const data = await ffmpeg.readFile('output.mp4');
    return new Blob([data], { type: 'video/mp4' });
  }

  const numSegments = Math.ceil(totalFrames / BATCH_SIZE);
  const segmentFiles: string[] = [];
  const concatListLines: string[] = [];

  for (let seg = 0; seg < numSegments; seg++) {
    onProgress?.(`encoding segment ${seg + 1}/${numSegments}`, (seg / numSegments) * 80);

    const startFrame = seg * BATCH_SIZE + 1;
    const endFrame = Math.min(startFrame + BATCH_SIZE - 1, totalFrames);
    const segName = `seg_${seg + 1}.mp4`;

    const segPattern = `frame_%04d.png`;
    await ffmpeg.exec(
      [
        '-start_number', startFrame.toString(),
        '-framerate', fps.toString(),
        '-i', segPattern,
        '-frames:v', (endFrame - startFrame + 1).toString(),
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        segName,
      ],
      300_000
    );

    segmentFiles.push(segName);
    concatListLines.push(`file '${segName}'`);

    // Clean up processed frame PNGs for this segment
    for (let f = startFrame; f <= endFrame; f++) {
      try {
        await ffmpeg.deleteFile(`frame_${String(f).padStart(4, '0')}.png`);
      } catch { /* ignore delete errors */ }
    }
  }

  onProgress?.('concatenating segments', 90);

  const concatContent = concatListLines.join('\n');
  await ffmpeg.writeFile('concat_list.txt', concatContent);
  await ffmpeg.exec(concatCommand(segmentFiles, 'output.mp4'), 300_000);

  const data = await ffmpeg.readFile('output.mp4');
  const blob = new Blob([data], { type: 'video/mp4' });

  // Cleanup
  await cleanupFFmpegFiles(ffmpeg, ['output.mp4', 'concat_list.txt', ...segmentFiles]);

  return blob;
}

async function cleanupFFmpegFiles(ffmpeg: FFmpeg, files: string[]): Promise<void> {
  for (const file of files) {
    try {
      await ffmpeg.deleteFile(file);
    } catch { /* ignore */ }
  }
}
