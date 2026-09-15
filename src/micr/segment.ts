/**
 * Cut a MICR band into glyph crops, in plain JS so it can run inside a
 * vision-camera frame processor worklet.
 *
 * This is a port of micr/segment.py from the training repo. Keeping the two in
 * step matters: the model is trained on crops produced by the Python version,
 * so if this one cuts differently the model sees inputs it was never shown.
 *
 * The guided-capture overlay does the heavy lifting the Python version has to
 * do itself -- the user aligns the band inside the guide, so there is no
 * document to detect, no perspective to rectify and no orientation to work
 * out. What is left is: binarize, find boundaries, cut.
 */

import { INPUT_HEIGHT, INPUT_WIDTH } from './classes';

export interface GrayImage {
  data: Uint8Array; // row-major, one byte per pixel
  width: number;
  height: number;
}

export interface GlyphBox {
  x0: number;
  x1: number;
}

/** Otsu's threshold. Returns the grey level that best splits ink from paper. */
export function otsuThreshold(image: GrayImage): number {
  const histogram = new Int32Array(256);
  for (let i = 0; i < image.data.length; i++) {
    histogram[image.data[i]]++;
  }
  const total = image.data.length;

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

/** Column-wise ink counts. Ink is darker than the threshold. */
export function inkProjection(image: GrayImage, threshold: number): Int32Array {
  const projection = new Int32Array(image.width);
  for (let y = 0; y < image.height; y++) {
    const row = y * image.width;
    for (let x = 0; x < image.width; x++) {
      if (image.data[row + x] <= threshold) {
        projection[x]++;
      }
    }
  }
  return projection;
}

function median(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export interface SegmentOptions {
  /** Centre hops below this fraction of a run's width are inside one glyph. */
  pitchMinFrac?: number;
  /** Lone marks further than this many pitches from a neighbour are not glyphs. */
  maxNeighbourPitch?: number;
  /** Runs narrower than this fraction of a glyph are specks. */
  minGlyphWidthFrac?: number;
}

/**
 * Find glyph boundaries by fitting a fixed-pitch character grid.
 *
 * Gap-based merging cannot work here: the transit and on-us symbols are drawn
 * as several separate vertical strokes, so any threshold loose enough to join
 * one symbol's strokes also joins two adjacent digits. E-13B's constant pitch
 * resolves it -- strokes of one symbol land in one cell, adjacent characters
 * do not, and the blank cells between fields simply hold no ink.
 */
export function findGlyphBoxes(
  image: GrayImage,
  threshold: number,
  options: SegmentOptions = {},
): GlyphBox[] {
  const pitchMinFrac = options.pitchMinFrac ?? 0.6;
  const maxNeighbourPitch = options.maxNeighbourPitch ?? 3.0;
  const minGlyphWidthFrac = options.minGlyphWidthFrac ?? 0.12;

  const projection = inkProjection(image, threshold);

  // Contiguous runs of inked columns, dropping anything touching the edge --
  // that is the guide border or the neighbouring field, not a glyph.
  const runs: GlyphBox[] = [];
  let start: number | null = null;
  for (let x = 0; x < projection.length; x++) {
    const inked = projection[x] > 0;
    if (inked && start === null) {
      start = x;
    } else if (!inked && start !== null) {
      runs.push({ x0: start, x1: x });
      start = null;
    }
  }
  if (start !== null) {
    runs.push({ x0: start, x1: projection.length });
  }

  const inner = runs.filter(r => r.x0 > 0 && r.x1 < projection.length);
  if (inner.length < 2) {
    return inner;
  }

  const estimate = (list: GlyphBox[]) => {
    const widths = list.map(r => r.x1 - r.x0);
    const base = median(widths);
    const centres = list.map(r => (r.x0 + r.x1) / 2);
    const deltas: number[] = [];
    for (let i = 1; i < centres.length; i++) {
      deltas.push(centres[i] - centres[i - 1]);
    }
    const between = deltas.filter(d => d >= base * pitchMinFrac);
    return { base, centres, pitch: median(between.length ? between : deltas) };
  };

  let { base, centres, pitch } = estimate(inner);
  if (!isFinite(pitch) || pitch <= 1) {
    return inner;
  }

  // Drop isolated marks: the guide edge, dust, a stray pen line. Every real
  // glyph has a neighbour within a couple of pitches, even across the blank
  // cells between fields.
  let kept = inner;
  if (inner.length >= 3) {
    const nearest = centres.map((c, i) => {
      const left = i > 0 ? c - centres[i - 1] : Infinity;
      const right = i < centres.length - 1 ? centres[i + 1] - c : Infinity;
      return Math.min(left, right);
    });
    const filtered = inner.filter((_, i) => nearest[i] <= pitch * maxNeighbourPitch);
    if (filtered.length >= 2 && filtered.length < inner.length) {
      kept = filtered;
      ({ base, centres, pitch } = estimate(kept));
    }
  }

  // Fit the grid origin, then group runs by cell.
  let origin = centres[0];
  for (let pass = 0; pass < 3; pass++) {
    let residual = 0;
    for (const c of centres) {
      residual += c - origin - Math.round((c - origin) / pitch) * pitch;
    }
    origin += residual / centres.length;
  }

  const cells = new Map<number, GlyphBox>();
  kept.forEach((run, i) => {
    const cell = Math.round((centres[i] - origin) / pitch);
    const existing = cells.get(cell);
    cells.set(
      cell,
      existing
        ? { x0: Math.min(existing.x0, run.x0), x1: Math.max(existing.x1, run.x1) }
        : { ...run },
    );
  });

  const minWidth = Math.max(1, base * minGlyphWidthFrac);
  return [...cells.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, box]) => box)
    .filter(box => box.x1 - box.x0 >= minWidth);
}

/**
 * Cut one glyph to the model's input size.
 *
 * Full band height, not the glyph's own bounding box: the training renderer
 * places every glyph on one shared baseline at its true relative height, so a
 * dash is short within its crop. Cropping tight here would rescale the dash to
 * full height and hand the model something it never saw in training.
 *
 * Returns float32 in [0, 1] -- normalisation is inside the model.
 */
export function cropGlyph(
  image: GrayImage,
  box: GlyphBox,
  padFrac = 0.18,
): Float32Array {
  const pad = Math.round((box.x1 - box.x0) * padFrac);
  const left = Math.max(0, box.x0 - pad);
  const right = Math.min(image.width, box.x1 + pad);
  const cropWidth = Math.max(1, right - left);

  const out = new Float32Array(INPUT_WIDTH * INPUT_HEIGHT);
  for (let y = 0; y < INPUT_HEIGHT; y++) {
    // Nearest-neighbour is enough: the source band is already close to the
    // target height, so this is a mild resample, not a big downscale.
    const srcY = Math.min(
      image.height - 1,
      Math.floor((y * image.height) / INPUT_HEIGHT),
    );
    for (let x = 0; x < INPUT_WIDTH; x++) {
      const srcX = left + Math.min(cropWidth - 1, Math.floor((x * cropWidth) / INPUT_WIDTH));
      out[y * INPUT_WIDTH + x] = image.data[srcY * image.width + srcX] / 255;
    }
  }
  return out;
}

/** Is this band worth running the model over at all? */
export function bandQuality(boxes: GlyphBox[]): number {
  if (boxes.length < 8 || boxes.length > 50) {
    return 0;
  }
  const widths = boxes.map(b => b.x1 - b.x0);
  const mean = widths.reduce((a, b) => a + b, 0) / widths.length;
  if (mean <= 0) {
    return 0;
  }
  const variance =
    widths.reduce((acc, w) => acc + (w - mean) ** 2, 0) / widths.length;
  const cv = Math.sqrt(variance) / mean;
  return boxes.length * Math.max(0, 1 - cv);
}
