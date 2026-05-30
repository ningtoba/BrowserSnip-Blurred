import type { FaceDetection, FaceIdentity } from '@/types';
import { cosineSimilarity, averageEmbeddings } from '@/lib/utils/math';
import { CLUSTER_SIMILARITY_THRESHOLD } from '@/lib/constants';

class UnionFind {
  private parent: number[];

  constructor(n: number) {
    this.parent = new Array(n);
    for (let i = 0; i < n; i++) this.parent[i] = i;
  }

  find(x: number): number {
    while (this.parent[x] !== x) {
      this.parent[x] = this.parent[this.parent[x]];
      x = this.parent[x];
    }
    return x;
  }

  union(a: number, b: number): void {
    const ra = this.find(a);
    const rb = this.find(b);
    if (ra !== rb) this.parent[rb] = ra;
  }

  groups(): Map<number, number[]> {
    const map = new Map<number, number[]>();
    for (let i = 0; i < this.parent.length; i++) {
      const root = this.find(i);
      if (!map.has(root)) map.set(root, []);
      map.get(root)!.push(i);
    }
    return map;
  }
}

export function clusterFaces(detections: FaceDetection[]): FaceIdentity[] {
  const validDetections = detections.filter((d) => d.embedding);
  if (validDetections.length === 0) return [];

  const n = validDetections.length;
  const uf = new UnionFind(n);

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (validDetections[i].frameIndex === validDetections[j].frameIndex) continue;

      const sim = cosineSimilarity(
        validDetections[i].embedding!,
        validDetections[j].embedding!
      );

      if (sim > CLUSTER_SIMILARITY_THRESHOLD) {
        uf.union(i, j);
      }
    }
  }

  const groups = uf.groups();
  const identities: FaceIdentity[] = [];
  let nextId = 0;

  for (const [, indices] of groups) {
    const clusterFaces = indices.map((i) => validDetections[i]);
    const embeddings = clusterFaces
      .map((f) => f.embedding!)
      .filter(Boolean);

    const averageEmbedding = averageEmbeddings(embeddings);

    let bestIdx = indices[0];
    let bestConf = 0;
    for (const idx of indices) {
      if (validDetections[idx].confidence > bestConf) {
        bestConf = validDetections[idx].confidence;
        bestIdx = idx;
      }
    }

    const id = nextId++;
    for (const idx of indices) {
      validDetections[idx].clusterId = id;
    }

    identities.push({
      id,
      representativeFace: bestIdx,
      faces: clusterFaces,
      averageEmbedding,
    });
  }

  identities.sort((a, b) => b.faces.length - a.faces.length);

  return identities;
}

export function matchDetectionsToIdentities(
  detections: FaceDetection[],
  identities: FaceIdentity[]
): Map<number, number> {
  const matchMap = new Map<number, number>();

  for (let i = 0; i < detections.length; i++) {
    const det = detections[i];
    if (!det.embedding) continue;

    let bestSim = -1;
    let bestId = -1;

    for (const identity of identities) {
      const sim = cosineSimilarity(det.embedding, identity.averageEmbedding);
      if (sim > bestSim && sim > CLUSTER_SIMILARITY_THRESHOLD) {
        bestSim = sim;
        bestId = identity.id;
      }
    }

    if (bestId >= 0) {
      matchMap.set(i, bestId);
    }
  }

  return matchMap;
}
