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

function bboxIOU(a: FaceDetection, b: FaceDetection): number {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  if (x2 <= x1 || y2 <= y1) return 0;
  const inter = (x2 - x1) * (y2 - y1);
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter);
}

export function clusterFaces(detections: FaceDetection[]): FaceIdentity[] {
  const validDetections = detections.filter((d) => d.embedding);
  if (validDetections.length === 0) return [];

  const n = validDetections.length;
  const uf = new UnionFind(n);

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      // Never merge faces from the same frame — they're different people
      if (validDetections[i].frameIndex === validDetections[j].frameIndex) continue;

      // Spatial constraint: faces must plausibly be in the same region
      // to be the same person across frames. If their bboxes never overlap,
      // they occupy different parts of the frame and must be different people.
      const iou = bboxIOU(validDetections[i], validDetections[j]);
      if (iou === 0) continue;

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
