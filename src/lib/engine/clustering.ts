import type { FaceDetection, FaceIdentity } from '@/types';
import { cosineSimilarity, averageEmbeddings } from '@/lib/utils/math';
import { CLUSTER_SIMILARITY_THRESHOLD } from '@/lib/constants';

export function clusterFaces(detections: FaceDetection[]): FaceIdentity[] {
  const validDetections = detections.filter((d) => d.embedding);
  if (validDetections.length === 0) return [];

  // Group detections by frame, sort each frame's faces by x-position
  const frameGroups = new Map<number, FaceDetection[]>();
  for (const d of validDetections) {
    const list = frameGroups.get(d.frameIndex) || [];
    list.push(d);
    frameGroups.set(d.frameIndex, list);
  }

  // Determine max faces per frame (typically 2)
  let maxPerFrame = 0;
  for (const faces of frameGroups.values()) {
    faces.sort((a, b) => a.x1 - b.x1);
    if (faces.length > maxPerFrame) maxPerFrame = faces.length;
  }

  // Seed identities by position: face at position 0 in each frame = identity 0, etc.
  const positionClusters: FaceDetection[][] = Array.from({ length: maxPerFrame }, () => []);

  for (const faces of frameGroups.values()) {
    for (let pos = 0; pos < faces.length; pos++) {
      positionClusters[pos].push(faces[pos]);
    }
  }

  // Build identities from position clusters
  const identities: FaceIdentity[] = [];

  for (let pos = 0; pos < maxPerFrame; pos++) {
    const cluster = positionClusters[pos];
    if (cluster.length === 0) continue;

    const embeddings = cluster.map((f) => f.embedding!);
    const avgEmbedding = averageEmbeddings(embeddings);

    let bestIdx = 0;
    let bestConf = 0;
    for (let i = 0; i < cluster.length; i++) {
      if (cluster[i].confidence > bestConf) {
        bestConf = cluster[i].confidence;
        bestIdx = i;
      }
    }

    // Find this face in the original validDetections array for the representative
    const repFace = cluster[bestIdx];
    const repIdx = validDetections.indexOf(repFace);

    identities.push({
      id: pos,
      representativeFace: repIdx >= 0 ? repIdx : 0,
      faces: cluster,
      averageEmbedding: avgEmbedding,
    });

    // Assign cluster ID
    for (const face of cluster) {
      face.clusterId = pos;
    }
  }

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
