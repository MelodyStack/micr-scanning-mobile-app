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

/**
 * Narrow a crop down to the rows the character line actually occupies.
 *
 * The guide box is deliberately taller than a MICR band, so a crop of it also
 * contains whatever sits above and below -- a signature stroke, the memo rule,
 * the edge of the cheque. Thresholding across all of that makes a full-width
 * dark row read as one ink run spanning the whole crop, and everything after
 * that behaves as if there were no characters at all.
 *
 * Rows are scored by ink, then the densest contiguous stretch is kept. This is
 * the same job locate_band does in the offline Python segmenter, at a smaller
 * scale: it is looking for the line inside the frame, this for the line inside
 * the guide.
 */
export function locateBandRows(
  image: GrayImage,
  threshold: number,
): { top: number; bottom: number } {
  const rowInk = new Int32Array(image.height);
  let peak = 0;
  for (let y = 0; y < image.height; y++) {
    const row = y * image.width;
    let count = 0;
    for (let x = 0; x < image.width; x++) {
      if (image.data[row + x] <= threshold) {
        count++;
      }
    }
    rowInk[y] = count;
    if (count > peak) {
      peak = count;
    }
  }
  if (peak === 0) {
    return { top: 0, bottom: image.height };
  }

  // A row belongs to the line if it carries a reasonable share of the peak.
  // Low enough to keep the thin waist of a glyph, high enough to exclude a
  // faint background rule.
  const cutoff = peak * 0.18;
  let bestTop = 0;
  let bestLen = 0;
  let runStart = -1;
  for (let y = 0; y <= image.height; y++) {
    const inked = y < image.height && rowInk[y] >= cutoff;
    if (inked && runStart < 0) {
      runStart = y;
    } else if (!inked && runStart >= 0) {
      if (y - runStart > bestLen) {
        bestLen = y - runStart;
        bestTop = runStart;
      }
      runStart = -1;
    }
  }
  if (bestLen === 0) {
    return { top: 0, bottom: image.height };
  }

  const pad = Math.max(2, Math.round(bestLen * 0.18));
  return {
    top: Math.max(0, bestTop - pad),
    bottom: Math.min(image.height, bestTop + bestLen + pad),
  };
}

export function cropRows(image: GrayImage, top: number, bottom: number): GrayImage {
  const height = Math.max(1, bottom - top);
  return {
    data: image.data.subarray(top * image.width, (top + height) * image.width),
    width: image.width,
    height,
  };
}

/**
 * Every horizontal strip in the image that might be a line of text.
 *
 * Rows carrying ink are grouped into runs, with small vertical gaps bridged so
 * the dot of a glyph does not split a line in two. Anything too thin to be
 * print or tall enough to be a block of handwriting is dropped. The caller
 * scores what survives.
 *
 * This is band_candidates from the offline Python segmenter. Ink alone is a
 * weak signal -- a signature or a printed caption carries more of it than the
 * MICR line -- so candidates are ranked later by how well they actually parse
 * as E-13B, not by how dark they are.
 */
export function findBandCandidates(
  image: GrayImage,
  threshold: number,
  options: { maxCandidates?: number } = {},
): { top: number; bottom: number }[] {
  const maxCandidates = options.maxCandidates ?? 10;

  const rowInk = new Int32Array(image.height);
  for (let y = 0; y < image.height; y++) {
    const row = y * image.width;
    let count = 0;
    for (let x = 0; x < image.width; x++) {
      if (image.data[row + x] <= threshold) {
        count++;
      }
    }
    rowInk[y] = count;
  }

  // A row counts as "inked" if a small fraction of it is dark. Too high and a
  // sparse line of digits is missed; too low and paper texture joins up.
  const minInk = Math.max(3, Math.round(image.width * 0.01));
  const bridge = Math.max(1, Math.round(image.height * 0.006));

  const runs: { top: number; bottom: number; ink: number }[] = [];
  let start = -1;
  let gap = 0;
  let ink = 0;
  for (let y = 0; y <= image.height; y++) {
    const inked = y < image.height && rowInk[y] >= minInk;
    if (inked) {
      if (start < 0) {
        start = y;
        ink = 0;
      }
      gap = 0;
      ink += rowInk[y];
    } else if (start >= 0) {
      gap++;
      if (gap > bridge || y === image.height) {
        runs.push({ top: start, bottom: y - gap + 1, ink });
        start = -1;
      }
    }
  }

  const minHeight = Math.max(4, Math.round(image.height * 0.012));
  const maxHeight = Math.round(image.height * 0.22);
  return runs
    .filter(r => {
      const h = r.bottom - r.top;
      return h >= minHeight && h <= maxHeight;
    })
    .sort((a, b) => b.ink - a.ink)
    .slice(0, maxCandidates)
    .map(r => {
      const pad = Math.max(2, Math.round((r.bottom - r.top) * 0.2));
      return {
        top: Math.max(0, r.top - pad),
        bottom: Math.min(image.height, r.bottom + pad),
      };
    });
}

/** Horizontally mirror an image. Used to undo a flipped sensor. */
export function mirrorImage(image: GrayImage): GrayImage {
  const out = new Uint8Array(image.data.length);
  for (let y = 0; y < image.height; y++) {
    const row = y * image.width;
    for (let x = 0; x < image.width; x++) {
      out[row + x] = image.data[row + image.width - 1 - x];
    }
  }
  return { data: out, width: image.width, height: image.height };
}

export interface BandFind {
  band: GrayImage;
  boxes: GlyphBox[];
  quality: number;
  rows: { top: number; bottom: number };
  mirrored: boolean;
}

/**
 * Search a whole cheque image for the MICR line.
 *
 * Every candidate strip is segmented and scored, and the best-parsing one
 * wins. That is what makes this robust to where the cheque sits in frame: no
 * alignment, no guide box, no mapping between preview and sensor coordinates.
 *
 * Mirroring is deliberately NOT considered here. A mirrored band segments
 * exactly as well as an upright one -- same glyph count, same pitch, same
 * widths -- so no amount of geometry tells the two apart. Only classifying the
 * glyphs and checking the ABA digit does, which is why the mirror retry lives
 * in recognizeDocument instead.
 */
export function findMicrBand(image: GrayImage): BandFind | null {
  let best: BandFind | null = null;
  const coarse = otsuThreshold(image);

  for (const rows of findBandCandidates(image, coarse)) {
    const band = cropRows(image, rows.top, rows.bottom);
    const boxes = findGlyphBoxes(band, otsuThreshold(band));
    const quality = bandQuality(boxes);
    if (quality > 0 && (!best || quality > best.quality)) {
      best = { band, boxes, quality, rows, mirrored: false };
    }
  }
  return best;
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

  // Runs touching the edge are usually the guide border or a neighbouring
  // field bleeding in -- but only drop them if something is left. When the
  // whole crop binarises to one edge-to-edge run, discarding it returned zero
  // glyphs and the caller could not tell "nothing here" from "one big blob".
  const trimmed = runs.filter(r => r.x0 > 0 && r.x1 < projection.length);
  const inner = trimmed.length >= 2 ? trimmed : runs;
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
