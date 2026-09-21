/**
 * Grayscale image primitives.
 *
 * Dependency-free, so the whole pipeline can be unit-tested with no device or
 * native modules; only `decode.ts` touches Skia. These mirror the OpenCV calls
 * in micr/segment.py, and that correspondence is load-bearing: the model was
 * trained on crops OpenCV produced. Approximations are called out where used.
 */

export interface GrayImage {
  /** Row-major, one byte per pixel, length === width * height. */
  data: Uint8Array;
  width: number;
  height: number;
}

export interface Rect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/** A binary ink mask: 1 where there is ink, 0 where there is paper. */
export interface InkMask {
  data: Uint8Array;
  width: number;
  height: number;
}

export function makeGray(width: number, height: number, fill = 0): GrayImage {
  const data = new Uint8Array(width * height);
  if (fill) {
    data.fill(fill);
  }
  return { data, width, height };
}

export function cropImage(image: GrayImage, rect: Rect): GrayImage {
  const x0 = clamp(Math.round(rect.x0), 0, image.width);
  const y0 = clamp(Math.round(rect.y0), 0, image.height);
  const x1 = clamp(Math.round(rect.x1), x0 + 1, image.width);
  const y1 = clamp(Math.round(rect.y1), y0 + 1, image.height);

  const width = x1 - x0;
  const height = y1 - y0;
  const out = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    const src = (y0 + y) * image.width + x0;
    out.set(image.data.subarray(src, src + width), y * width);
  }
  // Always a fresh buffer, never a subarray view. An earlier version returned a
  // view, so a later in-place pass silently edited the parent image too.
  return { data: out, width, height };
}

export function cropRows(image: GrayImage, top: number, bottom: number): GrayImage {
  return cropImage(image, { x0: 0, y0: top, x1: image.width, y1: bottom });
}

/** Rotate by `quarterTurns` * 90 degrees clockwise. */
export function rotate90(image: GrayImage, quarterTurns: number): GrayImage {
  const k = ((quarterTurns % 4) + 4) % 4;
  if (k === 0) {
    return { data: Uint8Array.from(image.data), width: image.width, height: image.height };
  }

  const { data, width: w, height: h } = image;
  if (k === 2) {
    const out = new Uint8Array(data.length);
    for (let i = 0, j = data.length - 1; i < data.length; i++, j--) {
      out[i] = data[j];
    }
    return { data: out, width: w, height: h };
  }

  const outW = h;
  const outH = w;
  const out = new Uint8Array(data.length);
  if (k === 1) {
    // Clockwise: output (x, y) comes from input row (h - 1 - x), column y.
    for (let y = 0; y < outH; y++) {
      const row = y * outW;
      for (let x = 0; x < outW; x++) {
        out[row + x] = data[(h - 1 - x) * w + y];
      }
    }
  } else {
    // Counter-clockwise.
    for (let y = 0; y < outH; y++) {
      const row = y * outW;
      for (let x = 0; x < outW; x++) {
        out[row + x] = data[x * w + (w - 1 - y)];
      }
    }
  }
  return { data: out, width: outW, height: outH };
}

/** Mirror horizontally. Undoes a front-facing or otherwise flipped sensor. */
export function mirrorImage(image: GrayImage): GrayImage {
  const { data, width, height } = image;
  const out = new Uint8Array(data.length);
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      out[row + x] = data[row + width - 1 - x];
    }
  }
  return { data: out, width, height };
}

// Resampling

interface Taps {
  /** Absolute, pre-clamped source index per tap. */
  indices: Int32Array;
  counts: Int32Array;
  weights: Float32Array;
  stride: number;
}

/**
 * Per-output-pixel source ranges and weights for a 1-D resize. Downscaling
 * averages over the source interval as cv2.INTER_AREA does; upscaling falls back
 * to bilinear, since INTER_AREA degenerates to nearest-neighbour there.
 */
function buildTaps(srcSize: number, outSize: number): Taps {
  const scale = srcSize / outSize;
  const stride = scale >= 1 ? Math.ceil(scale) + 1 : 2;
  const indices = new Int32Array(outSize * stride);
  const counts = new Int32Array(outSize);
  const weights = new Float32Array(outSize * stride);

  for (let i = 0; i < outSize; i++) {
    const base = i * stride;
    if (scale >= 1) {
      const from = i * scale;
      const to = (i + 1) * scale;
      const first = Math.floor(from);
      const last = Math.min(srcSize - 1, Math.ceil(to) - 1);
      let total = 0;
      let n = 0;
      for (let s = first; s <= last && n < stride; s++, n++) {
        // Partial coverage at both ends of the interval, full in the middle.
        const weight = Math.max(0, Math.min(to, s + 1) - Math.max(from, s));
        // Clamped here, once, so the resampling loop needs no bounds check.
        indices[base + n] = clamp(s, 0, srcSize - 1);
        weights[base + n] = weight;
        total += weight;
      }
      counts[i] = n;
      if (total > 0) {
        for (let t = 0; t < n; t++) {
          weights[base + t] /= total;
        }
      } else {
        indices[base] = clamp(first, 0, srcSize - 1);
        weights[base] = 1;
        counts[i] = 1;
      }
    } else {
      const centre = (i + 0.5) * scale - 0.5;
      const floor = Math.floor(centre);
      const frac = centre - floor;
      const first = clamp(floor, 0, srcSize - 1);
      const second = clamp(floor + 1, 0, srcSize - 1);
      indices[base] = first;
      indices[base + 1] = second;
      counts[i] = second === first ? 1 : 2;
      weights[base] = second === first ? 1 : 1 - frac;
      weights[base + 1] = second === first ? 0 : frac;
    }
  }
  return { indices, counts, weights, stride };
}

/**
 * Resize to an exact size. Separable: horizontal pass, then vertical.
 */
export function resample(image: GrayImage, outW: number, outH: number): GrayImage {
  const width = Math.max(1, Math.round(outW));
  const height = Math.max(1, Math.round(outH));
  if (width === image.width && height === image.height) {
    return { data: Uint8Array.from(image.data), width, height };
  }

  // Everything the inner loops touch is hoisted into a local. Hermes interprets
  // bytecode rather than JIT-compiling it, so a property load left inside a loop
  // over millions of pixels is a real cost: hoisting took a 1600 px rescale from
  // ~790 ms to 31 ms.
  const src = image.data;
  const srcW = image.width;
  const srcH = image.height;

  const xTaps = buildTaps(srcW, width);
  const xIndices = xTaps.indices;
  const xWeights = xTaps.weights;
  const xCounts = xTaps.counts;
  const xStride = xTaps.stride;

  const horizontal = new Float32Array(width * srcH);
  for (let y = 0; y < srcH; y++) {
    const srcRow = y * srcW;
    const dstRow = y * width;
    for (let x = 0; x < width; x++) {
      const base = x * xStride;
      const count = xCounts[x];
      let acc = 0;
      for (let t = 0; t < count; t++) {
        acc += src[srcRow + xIndices[base + t]] * xWeights[base + t];
      }
      horizontal[dstRow + x] = acc;
    }
  }

  const yTaps = buildTaps(srcH, height);
  const yIndices = yTaps.indices;
  const yWeights = yTaps.weights;
  const yCounts = yTaps.counts;
  const yStride = yTaps.stride;

  const out = new Uint8Array(width * height);
  // A float accumulator per output row. Summing straight into the Uint8Array
  // would round and clamp at every tap instead of once at the end.
  const rowAcc = new Float32Array(width);
  for (let y = 0; y < height; y++) {
    const base = y * yStride;
    const count = yCounts[y];
    rowAcc.fill(0);
    for (let t = 0; t < count; t++) {
      // Accumulate a whole source row at a time, so the tap lookups leave the
      // per-pixel loop entirely.
      const srcRow = yIndices[base + t] * width;
      const weight = yWeights[base + t];
      for (let x = 0; x < width; x++) {
        rowAcc[x] += horizontal[srcRow + x] * weight;
      }
    }
    const dstRow = y * width;
    for (let x = 0; x < width; x++) {
      const value = rowAcc[x];
      out[dstRow + x] = value < 0 ? 0 : value > 255 ? 255 : value + 0.5;
    }
  }
  return { data: out, width, height };
}

/** Scale so the longer side is at most `maxSide`, preserving aspect ratio. */
export function fitWithin(image: GrayImage, maxSide: number): GrayImage {
  const longest = Math.max(image.width, image.height);
  if (longest <= maxSide) {
    return image;
  }
  const scale = maxSide / longest;
  return resample(image, Math.round(image.width * scale), Math.round(image.height * scale));
}

/**
 * The three resolutions a read works at, each needing far fewer pixels than the
 * last: `probe` decides which way up the cheque is, `scout` locates the band,
 * and `work` is the only level glyph crops are cut from.
 *
 * decode.ts builds these with Skia so the scaling happens in native code. The JS
 * fallback here exists for tests and for a caller that already has pixels.
 */
export interface ImagePyramid {
  work: GrayImage;
  scout: GrayImage;
  probe: GrayImage;
}

/**
 * Chosen against what each stage needs, not for headroom. At `work` = 1800 a
 * cheque is ~293 px/inch, so E-13B's 8 characters per inch land on a 37 px pitch
 * and every glyph crop is a downscale to 32x48, the direction the training crops
 * were resampled in. At 2400 a read took 7.0 s on device.
 */
export const PYRAMID_SIZES = { work: 1800, scout: 1000, probe: 700 } as const;

export function buildPyramid(
  photo: GrayImage,
  sizes: { work: number; scout: number; probe: number } = PYRAMID_SIZES,
): ImagePyramid {
  const work = fitWithin(photo, sizes.work);
  const scout = fitWithin(work, sizes.scout);
  return { work, scout, probe: fitWithin(scout, sizes.probe) };
}

export function isPyramid(value: GrayImage | ImagePyramid): value is ImagePyramid {
  return (value as ImagePyramid).work !== undefined;
}

// Thresholding

/** Separable 3x3 blur with a [1 2 1] kernel, matching cv2.GaussianBlur(3, 3). */
export function blur3(image: GrayImage): GrayImage {
  const { data, width, height } = image;
  const tmp = new Float32Array(data.length);
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      const left = data[row + (x > 0 ? x - 1 : 0)];
      const right = data[row + (x < width - 1 ? x + 1 : width - 1)];
      tmp[row + x] = (left + 2 * data[row + x] + right) / 4;
    }
  }
  const out = new Uint8Array(data.length);
  for (let y = 0; y < height; y++) {
    const row = y * width;
    const above = (y > 0 ? y - 1 : 0) * width;
    const below = (y < height - 1 ? y + 1 : height - 1) * width;
    for (let x = 0; x < width; x++) {
      out[row + x] = (tmp[above + x] + 2 * tmp[row + x] + tmp[below + x]) / 4 + 0.5;
    }
  }
  return { data: out, width, height };
}

/** Otsu's threshold: the grey level that best separates ink from paper. */
export function otsuThreshold(image: GrayImage): number {
  const histogram = new Int32Array(256);
  const data = image.data;
  for (let i = 0; i < data.length; i++) {
    histogram[data[i]]++;
  }
  const total = data.length;

  let sum = 0;
  for (let t = 0; t < 256; t++) {
    sum += t * histogram[t];
  }

  let sumBackground = 0;
  let weightBackground = 0;
  let best = 0;
  let bestVariance = -1;

  for (let t = 0; t < 256; t++) {
    weightBackground += histogram[t];
    if (weightBackground === 0) {
      continue;
    }
    const weightForeground = total - weightBackground;
    if (weightForeground === 0) {
      break;
    }
    sumBackground += t * histogram[t];
    const meanBackground = sumBackground / weightBackground;
    const meanForeground = (sum - sumBackground) / weightForeground;
    const between =
      weightBackground * weightForeground * (meanBackground - meanForeground) ** 2;
    if (between > bestVariance) {
      bestVariance = between;
      best = t;
    }
  }
  return best;
}

export type ThresholdMode = 'otsu' | 'adaptive';

/**
 * Ink mask. Ink is dark, so a pixel is ink when it sits below the threshold.
 *
 * `adaptive` approximates cv2.ADAPTIVE_THRESH_GAUSSIAN_C with a box mean over an
 * integral image, O(n) regardless of window size. It rescues a cheque shot under
 * a side light, where one global threshold loses one end of the band.
 */
export function inkMask(
  image: GrayImage,
  mode: ThresholdMode = 'otsu',
  options: { blockFrac?: number; offset?: number } = {},
): InkMask {
  const blurred = blur3(image);
  const { width, height } = image;
  const out = new Uint8Array(width * height);

  if (mode === 'otsu') {
    const threshold = otsuThreshold(blurred);
    for (let i = 0; i < out.length; i++) {
      out[i] = blurred.data[i] <= threshold ? 1 : 0;
    }
    return { data: out, width, height };
  }

  // Window is tied to band height, not a fixed pixel count: the same photo at a
  // different resolution must threshold the same way.
  const blockFrac = options.blockFrac ?? 0.6;
  const offset = options.offset ?? 10;
  const half = Math.max(3, Math.round(height * blockFrac * 0.5));

  // Int32, not Float64. The largest possible sum is width * height * 255, which
  // for any image this pipeline handles stays well inside a signed 32-bit int,
  // and halving the width of the table halves the memory traffic of the four
  // lookups per pixel below, which is what this function actually spends its
  // time on.
  const src = blurred.data;
  const stride = width + 1;
  const integral = new Int32Array(stride * (height + 1));
  for (let y = 0; y < height; y++) {
    const srcRow = y * width;
    const above = y * stride;
    const here = (y + 1) * stride;
    let rowSum = 0;
    for (let x = 0; x < width; x++) {
      rowSum += src[srcRow + x];
      integral[here + x + 1] = integral[above + x + 1] + rowSum;
    }
  }

  // Pre-clamped window bounds, so the per-pixel loop has no Math.min/Math.max.
  const left = new Int32Array(width);
  const right = new Int32Array(width);
  for (let x = 0; x < width; x++) {
    left[x] = x - half < 0 ? 0 : x - half;
    right[x] = x + half > width - 1 ? width - 1 : x + half;
  }

  for (let y = 0; y < height; y++) {
    const y0 = y - half < 0 ? 0 : y - half;
    const y1 = y + half > height - 1 ? height - 1 : y + half;
    const rowsInWindow = y1 - y0 + 1;
    const topRow = y0 * stride;
    const bottomRow = (y1 + 1) * stride;
    const srcRow = y * width;
    for (let x = 0; x < width; x++) {
      const x0 = left[x];
      const x1 = right[x] + 1;
      const sum =
        integral[bottomRow + x1] -
        integral[topRow + x1] -
        integral[bottomRow + x0] +
        integral[topRow + x0];
      const mean = sum / ((x1 - x0) * rowsInWindow);
      out[srcRow + x] = src[srcRow + x] <= mean - offset ? 1 : 0;
    }
  }
  return { data: out, width, height };
}

// Projections

export function columnInk(mask: InkMask): Int32Array {
  const out = new Int32Array(mask.width);
  for (let y = 0; y < mask.height; y++) {
    const row = y * mask.width;
    for (let x = 0; x < mask.width; x++) {
      if (mask.data[row + x]) {
        out[x]++;
      }
    }
  }
  return out;
}

export function rowInk(mask: InkMask): Int32Array {
  const out = new Int32Array(mask.height);
  for (let y = 0; y < mask.height; y++) {
    const row = y * mask.width;
    let count = 0;
    for (let x = 0; x < mask.width; x++) {
      if (mask.data[row + x]) {
        count++;
      }
    }
    out[y] = count;
  }
  return out;
}

/**
 * Close along x: bridge horizontal gaps up to `kernel` pixels. The 1-D
 * equivalent of the MORPH_CLOSE in band_candidates(), smearing a line of glyphs
 * into one blob while leaving the cheque's printed border a thin line.
 */
export function closeHorizontal(mask: InkMask, kernel: number): InkMask {
  const k = Math.max(1, Math.round(kernel));
  const { width, height } = mask;
  const out = new Uint8Array(mask.data.length);
  for (let y = 0; y < height; y++) {
    const row = y * width;
    let lastInk = -1;
    for (let x = 0; x < width; x++) {
      if (mask.data[row + x]) {
        if (lastInk >= 0 && x - lastInk <= k) {
          for (let f = lastInk + 1; f < x; f++) {
            out[row + f] = 1;
          }
        }
        out[row + x] = 1;
        lastInk = x;
      }
    }
  }
  return { data: out, width, height };
}

// Deskew

/**
 * Slope (dy/dx) that makes the text baseline horizontal.
 *
 * DIVERGES: the offline segmenter uses cv2.minAreaRect. This shears the ink by a
 * candidate slope and scores how tightly it piles into a few rows, which is more
 * stable on a band that has picked up a fragment of the cheque border.
 */
export function estimateShear(mask: InkMask, maxSlope?: number, steps = 21): number {
  const { width, height, data } = mask;
  if (width < 8 || height < 8) {
    return 0;
  }

  // A shear moves the outermost column by slope * width/2, so allowing more
  // than the strip can hold slides the text out of its own crop. A 46 px band
  // across 1600 px tolerates barely 1.6 degrees.
  const geometric = (0.6 * height) / width;
  const limit = Math.min(maxSlope ?? 0.18, geometric);
  if (limit <= 1e-3) {
    return 0;
  }

  const centreX = width / 2;

  // Gather the ink once rather than re-walking every pixel for all 21 slopes.
  // Ink is a few percent of a cheque, so this turns the search from tens of
  // millions of iterations into tens of thousands. It was the single most
  // expensive thing in a read.
  let inkCount = 0;
  for (let i = 0; i < data.length; i++) {
    if (data[i]) {
      inkCount++;
    }
  }
  if (inkCount < 16) {
    return 0;
  }
  const xs = new Int32Array(inkCount);
  const ys = new Int32Array(inkCount);
  for (let y = 0, n = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      if (data[row + x]) {
        xs[n] = x;
        ys[n] = y;
        n++;
      }
    }
  }

  let bestSlope = 0;
  let bestScore = -1;
  const profile = new Float64Array(height);

  for (let s = 0; s < steps; s++) {
    const slope = -limit + (2 * limit * s) / (steps - 1);
    profile.fill(0);
    let kept = 0;
    for (let i = 0; i < inkCount; i++) {
      const target = Math.round(ys[i] - slope * (xs[i] - centreX));
      // Discard what shears out of the strip. Clamping instead piles every
      // out-of-range pixel onto the first and last rows, which manufactures two
      // huge spikes, so the most extreme slope always won and every band
      // came back sheared to the limit and unreadable.
      if (target < 0 || target >= height) {
        continue;
      }
      profile[target]++;
      kept++;
    }
    if (kept === 0) {
      continue;
    }
    // Concentration: ink piled into few rows beats ink spread over many.
    // Normalised by the sample count so that slopes which shear pixels out of
    // the strip are not rewarded for having fewer of them left.
    let score = 0;
    for (let y = 0; y < height; y++) {
      score += profile[y] * profile[y];
    }
    score /= kept;
    if (score > bestScore) {
      bestScore = score;
      bestSlope = slope;
    }
  }
  return bestSlope;
}

/**
 * Apply a vertical shear, which straightens a mildly rotated band. At the few
 * degrees a guided capture produces, a shear and a rotation are the same
 * transform to within a fraction of a pixel, and a shear is cheaper.
 */
export function shearVertical(image: GrayImage, slope: number): GrayImage {
  if (!isFinite(slope) || Math.abs(slope) < 1e-3) {
    return image;
  }
  const { width, height, data } = image;
  const centreX = width / 2;
  const out = new Uint8Array(data.length);

  for (let x = 0; x < width; x++) {
    const shift = slope * (x - centreX);
    for (let y = 0; y < height; y++) {
      const source = y + shift;
      const base = Math.floor(source);
      const frac = source - base;
      const a = data[clamp(base, 0, height - 1) * width + x];
      const b = data[clamp(base + 1, 0, height - 1) * width + x];
      out[y * width + x] = a + (b - a) * frac + 0.5;
    }
  }
  return { data: out, width, height };
}

export function clamp(value: number, low: number, high: number): number {
  return value < low ? low : value > high ? high : value;
}

export function median(values: ArrayLike<number>): number {
  if (values.length === 0) {
    return 0;
  }
  const sorted = Array.from(values).sort((a, b) => a - b);
  // Integer halving.
  // eslint-disable-next-line no-bitwise
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}
