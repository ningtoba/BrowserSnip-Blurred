/**
 * Hungarian algorithm (Munkres) for optimal one-to-one assignment.
 * Minimizes total cost — we use (1 - IOU) as cost to maximize total IOU.
 *
 * This prevents identity swaps that happen with greedy matching:
 * Greedy: face A matches to detection 1 (IOU 0.6), face B can't match detection 2 (IOU 0.5)
 * Hungarian: face A matches to detection 2 (IOU 0.4), face B matches to detection 1 (IOU 0.7) — BETTER total
 */
export function hungarianMatch(
  costMatrix: number[][]  // costMatrix[row][col], lower = better
): { row: number; col: number }[] {
  const n = costMatrix.length;
  const m = costMatrix[0]?.length ?? 0;
  if (n === 0 || m === 0) return [];

  const INF = 1e18;
  const maxDim = Math.max(n, m);
  const u = new Float64Array(maxDim + 1);
  const v = new Float64Array(maxDim + 1);
  const p = new Int32Array(maxDim + 1);
  const way = new Int32Array(maxDim + 1);

  for (let i = 1; i <= n; i++) {
    p[0] = i;
    const minv = new Float64Array(m + 1).fill(INF);
    const used = new Uint8Array(m + 1);
    let j0 = 0;

    do {
      used[j0] = 1;
      const i0 = p[j0];
      let delta = INF;
      let j1 = 0;

      for (let j = 1; j <= m; j++) {
        if (used[j]) continue;
        const cost = (i0 <= n && j <= m) ? costMatrix[i0 - 1][j - 1] : INF;
        const cur = cost - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }

      for (let j = 0; j <= m; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);

    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 !== 0);
  }

  const matches: { row: number; col: number }[] = [];
  for (let j = 1; j <= m; j++) {
    if (p[j] > 0 && p[j] <= n && j <= m) {
      const row = p[j] - 1;
      const col = j - 1;
      if (row < n && col < m && costMatrix[row][col] < INF) {
        matches.push({ row, col });
      }
    }
  }
  return matches;
}
