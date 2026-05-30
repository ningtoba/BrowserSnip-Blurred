import { useCallback, useRef } from 'react';
import { getFFmpeg, terminateFFmpeg } from '@/lib/ffmpeg/core';
import { useProcessStore } from '@/stores/process-store';

export function useFFmpeg() {
  const logBufferRef = useRef<string[]>([]);

  const process = useCallback(
    async (
      commandArgs: string[],
      extraSetup?: (ffmpeg: Awaited<ReturnType<typeof getFFmpeg>>) => Promise<void>
    ): Promise<Blob | null> => {
      const store = useProcessStore.getState();
      logBufferRef.current = [];

      try {
        const ffmpeg = await getFFmpeg();

        ffmpeg.on('log', ({ message }: { message: string }) => {
          logBufferRef.current.push(message);
          if (logBufferRef.current.length % 10 === 0) {
            useProcessStore.getState().appendLog(message);
          }
        });

        if (extraSetup) {
          await extraSetup(ffmpeg);
        }

        await ffmpeg.exec(commandArgs, 600_000);

        const data = await ffmpeg.readFile('output.mp4');
        const blob = new Blob([data], { type: 'video/mp4' });
        const url = URL.createObjectURL(blob);

        useProcessStore.getState().setOutput(blob, url);

        return blob;
      } catch (err) {
        await terminateFFmpeg();

        const allLogs = logBufferRef.current.join('\n');
        let message: string;

        if (err instanceof Error) {
          message = err.message;
        } else if (typeof err === 'string') {
          message = err;
        } else {
          message = 'Processing failed';
        }

        useProcessStore.getState().setError(message);
        return null;
      }
    },
    []
  );

  return { process };
}
