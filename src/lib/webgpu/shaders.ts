export const PIXELATE_SHADER = /* wgsl */ `
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;

struct BlurParams {
  faceX: f32,
  faceY: f32,
  faceW: f32,
  faceH: f32,
  blockSize: f32,
};

@group(0) @binding(2) var<uniform> params: BlurParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let px = gid.x;
  let py = gid.y;

  let fx = u32(params.faceX);
  let fy = u32(params.faceY);
  let fw = u32(params.faceW);
  let fh = u32(params.faceH);

  // Only process pixels within the face region
  if (px < fx || px >= fx + fw || py < fy || py >= fy + fh) {
    let orig = textureLoad(inputTex, vec2u(px, py), 0);
    textureStore(outputTex, vec2u(px, py), orig);
    return;
  }

  let bs = u32(params.blockSize);
  let bx = (px - fx) / bs;
  let by = (py - fy) / bs;
  let sx = fx + bx * bs;
  let sy = fy + by * bs;
  let ex = min(sx + bs, fx + fw);
  let ey = min(sy + bs, fy + fh);

  var sum = vec4f(0.0);
  var count: f32 = 0.0;
  for (var y = sy; y < ey; y++) {
    for (var x = sx; x < ex; x++) {
      sum += textureLoad(inputTex, vec2u(x, y), 0);
      count += 1.0;
    }
  }
  let avg = sum / count;

  for (var y = sy; y < ey; y++) {
    for (var x = sx; x < ex; x++) {
      textureStore(outputTex, vec2u(x, y), avg);
    }
  }
}
`;

export const EYEBAR_SHADER = /* wgsl */ `
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;

struct EyebarParams {
  barX: f32,
  barY: f32,
  barW: f32,
  barH: f32,
  radius: f32,
};

@group(0) @binding(2) var<uniform> params: EyebarParams;

fn roundedRectSDF(p: vec2f, b: vec2f, r: f32) -> f32 {
  let d = abs(p) - b + vec2f(r);
  return length(max(d, vec2f(0.0))) + min(max(d.x, d.y), 0.0) - r;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let px = gid.x;
  let py = gid.y;

  let bx = u32(params.barX);
  let by = u32(params.barY);
  let bw = u32(params.barW);
  let bh = u32(params.barH);
  let r = params.radius;

  let orig = textureLoad(inputTex, vec2u(px, py), 0);

  if (px >= bx && px < bx + bw && py >= by && py < by + bh) {
    let center = vec2f(f32(bx) + f32(bw) * 0.5, f32(by) + f32(bh) * 0.5);
    let halfSize = vec2f(f32(bw) * 0.5, f32(bh) * 0.5);
    let d = roundedRectSDF(vec2f(f32(px), f32(py)) - center, halfSize, r);
    if (d <= 0.0) {
      textureStore(outputTex, vec2u(px, py), vec4f(0.0, 0.0, 0.0, 1.0));
      return;
    }
  }

  textureStore(outputTex, vec2u(px, py), orig);
}
`;

export const NORMALIZE_CHW_SHADER = /* wgsl */ `
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var texSampler: sampler;
@group(0) @binding(2) var<storage, read_write> outputBuf: array<f32>;

struct NormalizeParams {
  srcW: f32,
  srcH: f32,
  dstW: f32,
  dstH: f32,
  scale: f32,
  padLeft: f32,
  padTop: f32,
  meanR: f32,
  meanG: f32,
  meanB: f32,
  stdR: f32,
  stdG: f32,
  stdB: f32,
};

@group(0) @binding(3) var<uniform> params: NormalizeParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;

  if (x >= u32(params.dstW) || y >= u32(params.dstH)) { return; }

  let planeSize = u32(params.dstW) * u32(params.dstH);

  // Map output pixel to source via letterbox
  let srcX = (f32(x) - params.padLeft) / params.scale;
  let srcY = (f32(y) - params.padTop) / params.scale;

  var color = vec4f(0.447, 0.447, 0.447, 1.0); // gray 114 padding

  if (srcX >= 0.0 && srcX < params.srcW && srcY >= 0.0 && srcY < params.srcH) {
    let uv = vec2f(srcX / params.srcW, srcY / params.srcH);
    color = textureSampleLevel(inputTex, texSampler, uv, 0.0);
  }

  // Normalize: (pixel - mean) / std
  let r = (color.r - params.meanR / 255.0) / (params.stdR / 255.0);
  let g = (color.g - params.meanG / 255.0) / (params.stdG / 255.0);
  let b = (color.b - params.meanB / 255.0) / (params.stdB / 255.0);

  // Write CHW layout
  outputBuf[y * u32(params.dstW) + x] = f32(r);
  outputBuf[planeSize + y * u32(params.dstW) + x] = f32(g);
  outputBuf[2u * planeSize + y * u32(params.dstW) + x] = f32(b);
}
`;
