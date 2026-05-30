export function l2Norm(vec: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < vec.length; i++) {
    sum += vec[i] * vec[i];
  }
  return Math.sqrt(sum);
}

export function l2Normalize(vec: Float32Array): Float32Array {
  const norm = l2Norm(vec);
  if (norm < 1e-12) return vec;
  const result = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) {
    result[i] = vec[i] / norm;
  }
  return result;
}

export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
  }
  return dot;
}

export function averageEmbeddings(embeddings: Float32Array[]): Float32Array {
  const dim = embeddings[0].length;
  const result = new Float32Array(dim);
  for (const emb of embeddings) {
    for (let i = 0; i < dim; i++) {
      result[i] += emb[i] / embeddings.length;
    }
  }
  return l2Normalize(result);
}
