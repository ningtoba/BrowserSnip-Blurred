function threads(): string {
  return Math.min(4, navigator.hardwareConcurrency || 2).toString();
}

export function reconstructCommand(
  inputPattern: string,
  outputName: string,
  fps: number
): string[] {
  return [
    '-framerate', fps.toString(),
    '-i', inputPattern,
    '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
    '-pix_fmt', 'yuv420p',
    '-threads', threads(),
    '-movflags', '+faststart',
    outputName,
  ];
}

export function concatCommand(
  segmentList: string[],
  outputName: string
): string[] {
  const concatFile = 'concat_list.txt';
  const listContent = segmentList.map((s) => `file '${s}'`).join('\n');

  return [
    '-f', 'concat',
    '-safe', '0',
    '-i', concatFile,
    '-c', 'copy',
    '-movflags', '+faststart',
    outputName,
  ];
}

export function probeCommand(inputName: string): string[] {
  return ['-i', inputName, '-f', 'null', '-'];
}
