/**
 * Find the MICR band in a cheque photo and cut it into glyph crops.
 *
 * A port of micr/segment.py from the training repo. The model only ever saw
 * crops the Python segmenter produced, so a crop cut to different proportions
 * here is an input it was never trained on. Deliberate differences are marked
 * DIVERGES. Nothing here imports a native module.
 */

import { INPUT_HEIGHT, INPUT_WIDTH } from './classes';
import {
  clamp,
  closeHorizontal,
  columnInk,
  cropImage,
  cropRows,
  estimateShear,
  type GrayImage,
  type InkMask,
  inkMask,
  median,
  otsuThreshold,
  type Rect,
  resample,
  rowInk,
  shearVertical,
  type ThresholdMode,
} from './image';

export type { GrayImage, Rect } from './image';

export interface GlyphBox {
  x0: number;
  x1: number;
}

export interface SegmentConfig {
  /** Fraction of the region's height, measured from the bottom, to search. */
  searchFrac: number;
  /** A band must span at least this fraction of the region width. */
  minBandWidthFrac: number;
  /** ... and be no taller than this fraction of the region height. */
  maxBandHeightFrac: number;
  /** Padding added above and below the detected line, as a fraction of it. */
  bandPadFrac: number;
  /** Centre hops below this fraction of a run's width are inside one glyph. */
  pitchMinFrac: number;
  /**
   * Gap, in pitches, that ends one chain of print and starts another. Real
   * cheques run 1.24 to 1.46 between fields; edge micro-print sat 2.96 out.
   */
  maxNeighbourPitch: number;
  /** Runs narrower than this fraction of a glyph are specks. */
  minGlyphWidthFrac: number;
  /** Side padding on each glyph crop, as a fraction of its width. */
  cropPadFrac: number;
  /** A column counts as inked at this fraction of the band height. */
  columnNoiseFrac: number;
  minGlyphs: number;
  maxGlyphs: number;
  maxCandidates: number;
  deskew: boolean;
}

/**
 * Defaults are the SegmentConfig values from the training repo, with two
 * additions noted in the type above.
 */
export const DEFAULT_CONFIG: SegmentConfig = {
  searchFrac: 1.0,
  minBandWidthFrac: 0.3,
  maxBandHeightFrac: 0.16,
  bandPadFrac: 0.22,
  pitchMinFrac: 0.6,
  maxNeighbourPitch: 2.0,
  minGlyphWidthFrac: 0.12,
  cropPadFrac: 0.18,
  columnNoiseFrac: 0.05,
  minGlyphs: 8,
  maxGlyphs: 50,
  maxCandidates: 8,
  deskew: true,
};

// Finding the sheet

/**
 * Fractional bounds of the sheet of paper within the frame.
 *
 * Must run before anything looks for a band. A photo of a cheque is mostly
 * desk, so Otsu over the whole frame separates desk from paper rather than ink
 * from paper, and every candidate is discarded on a frame where the band is
 * legible. With this step chk001 segments to all 32 glyphs; without it, zero.
 */
export function findSheet(image: GrayImage): Rect {
  const { data, width, height } = image;
  const threshold = otsuThreshold(image);

  const colBright = new Int32Array(width);
  const rowBright = new Int32Array(height);
  for (let y = 0; y < height; y++) {
    const row = y * width;
    let count = 0;
    for (let x = 0; x < width; x++) {
      if (data[row + x] > threshold) {
        colBright[x]++;
        count++;
      }
    }
    rowBright[y] = count;
  }

  const span = (counts: Int32Array, extent: number): { lo: number; hi: number } => {
    let peak = 0;
    for (let i = 0; i < counts.length; i++) {
      if (counts[i] > peak) {
        peak = counts[i];
      }
    }
    if (peak === 0) {
      return { lo: 0, hi: extent };
    }
    const cutoff = peak * 0.35;
    let lo = 0;
    let hi = counts.length - 1;
    while (lo < counts.length && counts[lo] < cutoff) {
      lo++;
    }
    while (hi > lo && counts[hi] < cutoff) {
      hi--;
    }
    // A sliver is not a sheet. Falling back to the whole frame beats cropping
    // to a highlight on the desk.
    return hi - lo < extent * 0.25 ? { lo: 0, hi: extent } : { lo, hi: hi + 1 };
  };

  const cols = span(colBright, width);
  const rows = span(rowBright, height);

  // A little margin, because the MICR band sits close to the bottom edge and
  // clipping it is worse than keeping a sliver of desk.
  const padX = width * 0.01;
  const padY = height * 0.01;
  return {
    x0: clamp(cols.lo - padX, 0, width) / width,
    y0: clamp(rows.lo - padY, 0, height) / height,
    x1: clamp(cols.hi + padX, 0, width) / width,
    y1: clamp(rows.hi + padY, 0, height) / height,
  };
}

/** Apply a fractional rect to an image of any size. */
export function cropFraction(image: GrayImage, rect: Rect): GrayImage {
  return cropImage(image, {
    x0: rect.x0 * image.width,
    y0: rect.y0 * image.height,
    x1: rect.x1 * image.width,
    y1: rect.y1 * image.height,
  });
}

// Band location

export interface BandCandidate {
  top: number;
  bottom: number;
  /** Fraction of the region width the line of ink spans. */
  coverage: number;
}

/**
 * Horizontal strips that might be a line of print. Ink is smeared along x, so a
 * line of glyphs becomes one blob while the printed border stays a thin rule.
 * Ranked later by how well they parse as E-13B, never by how much ink they
 * carry, since the signature line and memo rule carry more.
 */
export function findBandCandidates(
  region: GrayImage,
  mask: InkMask,
  config: SegmentConfig = DEFAULT_CONFIG,
): BandCandidate[] {
  const { width, height } = region;
  const searchTop = Math.floor(height * (1 - clamp(config.searchFrac, 0.05, 1)));

  // Kernel width from the training repo: max(15, width / 40).
  const smeared = closeHorizontal(mask, Math.max(15, Math.round(width / 40)));
  const ink = rowInk(smeared);

  const minRowInk = Math.max(2, Math.round(width * config.minBandWidthFrac * 0.4));
  const minHeight = 4;
  const maxHeight = Math.max(minHeight + 1, Math.round(height * config.maxBandHeightFrac));

  // Pass one: the dense core of every line of print.
  const cores: { top: number; bottom: number; peak: number }[] = [];
  let start = -1;
  for (let y = searchTop; y <= height; y++) {
    const inked = y < height && ink[y] >= minRowInk;
    if (inked && start < 0) {
      start = y;
    } else if (!inked && start >= 0) {
      let peak = 0;
      for (let r = start; r < y; r++) {
        peak = Math.max(peak, ink[r]);
      }
      cores.push({ top: start, bottom: y, peak });
      start = -1;
    }
  }

  // Pass two: gate each core and pad it out to a band. Growing the run outwards
  // over faint rows was tried and made things worse.
  const candidates: BandCandidate[] = [];
  for (const core of cores) {
    // Same gates as the connected-component filter in the Python: wide enough
    // to be a line of print, short enough not to be handwriting or a dark patch.
    const lineHeight = core.bottom - core.top;
    if (
      lineHeight >= minHeight &&
      lineHeight <= maxHeight &&
      core.peak >= width * config.minBandWidthFrac
    ) {
      const pad = Math.round(lineHeight * config.bandPadFrac);
      candidates.push({
        top: Math.max(0, core.top - pad),
        bottom: Math.min(height, core.bottom + pad),
        coverage: core.peak / width,
      });
    }
  }

  // Bottom-most first: the MICR line is the lowest line of print on a cheque.
  // This only orders the work, it never excludes anything.
  return candidates
    .sort((a, b) => b.top - a.top)
    .slice(0, config.maxCandidates);
}

// Glyph boundaries

/**
 * Glyph boundaries, from fitting a fixed-pitch character grid.
 *
 * Gap-based merging cannot work on E-13B: the transit and on-us symbols are
 * drawn as separate strokes, so any threshold loose enough to join them also
 * joins two adjacent digits. The font's constant pitch resolves it, letting
 * cell membership decide instead.
 */
export function findGlyphBoxes(
  mask: InkMask,
  config: SegmentConfig = DEFAULT_CONFIG,
): GlyphBox[] {
  const projection = columnInk(mask);
  const width = projection.length;

  // DIVERGES: the Python treats any non-zero column as inked, which is safe on
  // a clean Otsu binarisation of an already-cropped band. Here the band comes
  // straight out of a phone photo, so a noise floor tied to band height stops
  // a single speckled pixel from welding two glyphs into one run.
  const noiseFloor = Math.max(1, Math.round(mask.height * config.columnNoiseFrac));

  const runs: GlyphBox[] = [];
  let start: number | null = null;
  for (let x = 0; x < width; x++) {
    const inked = projection[x] >= noiseFloor;
    if (inked && start === null) {
      start = x;
    } else if (!inked && start !== null) {
      runs.push({ x0: start, x1: x });
      start = null;
    }
  }
  if (start !== null) {
    runs.push({ x0: start, x1: width });
  }

  // Runs touching the edge are the printed border or a neighbouring field
  // bleeding in. Only dropped if something is left, so a band that binarises to
  // one edge-to-edge blob still reports that blob rather than nothing.
  const trimmed = runs.filter(r => r.x0 > 0 && r.x1 < width);
  let kept = trimmed.length >= 2 ? trimmed : runs;
  if (kept.length < 2) {
    return kept;
  }

  let stats = estimatePitch(kept, config);
  if (!isFinite(stats.pitch) || stats.pitch <= 1) {
    return kept;
  }

  // A run spanning much more than a cell is a rule, a border or a dark patch.
  // Catches a 182 px blob against a 16 px pitch, which would otherwise corrupt
  // the pitch estimate and the character it lands on.
  {
    const limit = stats.pitch * 1.4;
    const narrow = kept.filter(r => r.x1 - r.x0 <= limit);
    if (narrow.length >= 2 && narrow.length < kept.length) {
      kept = narrow;
      stats = estimatePitch(kept, config);
      if (!isFinite(stats.pitch) || stats.pitch <= 1) {
        return kept;
      }
    }
  }

  // Drop junk chains, keep the fields. Chains are judged on length, not
  // distance: keeping only the longest is wrong, because the gap before the
  // on-us field is a genuine boundary that measured 2.21 pitches on a real
  // cheque, so the account number split off and was discarded. Junk arrives in
  // ones and twos; a MICR field never does.
  if (kept.length >= 3) {
    const chains = splitIntoChains(kept, stats.pitch * config.maxNeighbourPitch);
    const substantial = chains.map(c => c.length >= MIN_CHAIN_GLYPHS);
    const first = substantial.indexOf(true);
    const last = substantial.lastIndexOf(true);
    if (first >= 0 && (first > 0 || last < chains.length - 1)) {
      const merged = chains.slice(first, last + 1).flat();
      if (merged.length >= 2 && merged.length < kept.length) {
        kept = merged;
        stats = estimatePitch(kept, config);
        if (!isFinite(stats.pitch) || stats.pitch <= 1) {
          return kept;
        }
      }
    }
  }

  const { base, centres } = stats;
  const grid = fitGrid(centres, stats.pitch);

  const cells = new Map<number, GlyphBox>();
  for (let i = 0; i < kept.length; i++) {
    const cell = Math.round((centres[i] - grid.origin) / grid.pitch);
    const existing = cells.get(cell);
    cells.set(
      cell,
      existing
        ? { x0: Math.min(existing.x0, kept[i].x0), x1: Math.max(existing.x1, kept[i].x1) }
        : { ...kept[i] },
    );
  }

  const minWidth = Math.max(1, base * config.minGlyphWidthFrac);
  const ordered = [...cells.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, box]) => box)
    .filter(box => box.x1 - box.x0 >= minWidth);

  return mergeSplitGlyphs(ordered, grid.pitch);
}

/**
 * Rejoin a character the grid cut in half, which leaves the remainder correctly
 * classified and the orphan a phantom character. This was the most common
 * failure on real cheques.
 *
 * Fixed pitch makes the repair safe: two neighbours that together fit inside a
 * pitch were never two characters. At a 24 px pitch the halves of a split symbol
 * span 18 px together, while adjacent characters span 36 px.
 */
function mergeSplitGlyphs(boxes: GlyphBox[], pitch: number): GlyphBox[] {
  if (boxes.length < 2 || !isFinite(pitch) || pitch <= 1) {
    return boxes;
  }

  const maxSpan = pitch * 0.95;
  const maxGap = pitch * 0.35;
  const merged: GlyphBox[] = [boxes[0]];
  for (let i = 1; i < boxes.length; i++) {
    const previous = merged[merged.length - 1];
    const box = boxes[i];
    const gap = box.x0 - previous.x1;
    const span = box.x1 - previous.x0;
    if (gap <= maxGap && span <= maxSpan) {
      merged[merged.length - 1] = { x0: previous.x0, x1: box.x1 };
    } else {
      merged.push(box);
    }
  }
  return merged;
}

interface PitchStats {
  base: number;
  centres: number[];
  pitch: number;
}

/**
 * Break a left-to-right list of runs wherever the gap exceeds `maxGap`. Measured
 * between edges, not centres, since two wide runs with distant centres may still
 * be touching.
 */
function splitIntoChains(runs: GlyphBox[], maxGap: number): GlyphBox[][] {
  const chains: GlyphBox[][] = [[runs[0]]];
  for (let i = 1; i < runs.length; i++) {
    if (runs[i].x0 - runs[i - 1].x1 > maxGap) {
      chains.push([runs[i]]);
    } else {
      chains[chains.length - 1].push(runs[i]);
    }
  }
  return chains;
}

/**
 * Typical run width, and the character pitch: the median spacing between run
 * centres, ignoring the short hops between strokes inside one symbol.
 */
function estimatePitch(runs: GlyphBox[], config: SegmentConfig): PitchStats {
  const base = median(runs.map(r => r.x1 - r.x0));
  const centres = runs.map(r => (r.x0 + r.x1) / 2);
  const deltas: number[] = [];
  for (let i = 1; i < centres.length; i++) {
    deltas.push(centres[i] - centres[i - 1]);
  }
  const between = deltas.filter(d => d >= base * config.pitchMinFrac);
  return { base, centres, pitch: median(between.length ? between : deltas) };
}

/**
 * Fit the character grid: both where it starts and how wide its cells are.
 *
 * DIVERGES: the training code refines only the origin, which is wrong over
 * thirty characters, as a pitch out by 3% drifts nearly a whole cell and gives a
 * line with the right number of boxes and the wrong contents. Instead, assign
 * each run to its nearest cell, least-squares regress centre against cell index,
 * and repeat until the assignments stop moving.
 */
interface Grid {
  origin: number;
  pitch: number;
}

function fitGrid(centres: number[], pitch: number): Grid {
  let grid: Grid = { origin: centres[0], pitch };
  if (centres.length < 3) {
    return grid;
  }

  for (let pass = 0; pass < 4; pass++) {
    const cells = centres.map(c => Math.round((c - grid.origin) / grid.pitch));

    let meanCell = 0;
    let meanCentre = 0;
    for (let i = 0; i < centres.length; i++) {
      meanCell += cells[i];
      meanCentre += centres[i];
    }
    meanCell /= centres.length;
    meanCentre /= centres.length;

    let covariance = 0;
    let variance = 0;
    for (let i = 0; i < centres.length; i++) {
      const dCell = cells[i] - meanCell;
      covariance += dCell * (centres[i] - meanCentre);
      variance += dCell * dCell;
    }
    if (variance <= 0) {
      return grid;
    }

    const fitted = covariance / variance;
    // Refuse a fit that has collapsed or run away: the assignment stepped to a
    // different multiple of the true pitch, so the previous round is better.
    if (!isFinite(fitted) || fitted < pitch * 0.5 || fitted > pitch * 2) {
      return grid;
    }
    grid = { origin: meanCentre - fitted * meanCell, pitch: fitted };
  }
  return grid;
}

/**
 * How much does this look like a real MICR line rather than texture? Counting
 * boxes alone rewards noise, so a bounded character count, clustered widths and
 * a constant pitch are all required.
 */
export function bandQuality(
  boxes: GlyphBox[],
  config: SegmentConfig = DEFAULT_CONFIG,
): number {
  const count = boxes.length;
  if (count < config.minGlyphs || count > config.maxGlyphs) {
    return 0;
  }

  const widths = boxes.map(b => b.x1 - b.x0);
  const centres = boxes.map(b => (b.x0 + b.x1) / 2);
  const meanWidth = widths.reduce((a, b) => a + b, 0) / count;
  if (meanWidth <= 0) {
    return 0;
  }

  const deltas: number[] = [];
  for (let i = 1; i < centres.length; i++) {
    deltas.push(centres[i] - centres[i - 1]);
  }
  const pitch = median(deltas);
  if (!isFinite(pitch) || pitch <= 0) {
    return 0;
  }

  const grid = fitGrid(centres, pitch);
  let squared = 0;
  for (const c of centres) {
    const offset = c - grid.origin;
    const residual = offset - Math.round(offset / grid.pitch) * grid.pitch;
    squared += residual * residual;
  }
  const gridFit = Math.max(0, 1 - (2 * Math.sqrt(squared / count)) / grid.pitch);

  const variance =
    widths.reduce((acc, w) => acc + (w - meanWidth) ** 2, 0) / count;
  const widthFit = Math.max(0, 1 - Math.sqrt(variance) / meanWidth);

  return count * gridFit * widthFit;
}

/**
 * Spacing between consecutive glyphs, in character cells. Neighbours are one
 * cell apart and a blank cell between fields reads as two; anything larger is a
 * hole where characters should have been.
 */
export function cellSteps(boxes: GlyphBox[]): number[] {
  if (boxes.length < 2) {
    return [];
  }
  const centres = boxes.map(b => (b.x0 + b.x1) / 2);
  const deltas: number[] = [];
  for (let i = 1; i < centres.length; i++) {
    deltas.push(centres[i] - centres[i - 1]);
  }
  const pitch = median(deltas);
  if (!isFinite(pitch) || pitch <= 1) {
    return deltas.map(() => 1);
  }
  return deltas.map(d => Math.max(1, Math.round(d / pitch)));
}

// Glyph crops

/**
 * Cut one glyph to the model's input size, at full band height and never the
 * glyph's own bounding box. The training renderer puts every glyph on a shared
 * baseline, so cropping tight would rescale the dash to full height and it would
 * come back as a digit. Returns float32 in [0, 1].
 */
export function cropGlyph(
  band: GrayImage,
  box: GlyphBox,
  config: SegmentConfig = DEFAULT_CONFIG,
): Float32Array {
  const pad = Math.round((box.x1 - box.x0) * config.cropPadFrac);
  const rect: Rect = {
    x0: box.x0 - pad,
    y0: 0,
    x1: box.x1 + pad,
    y1: band.height,
  };
  // Area-weighted, matching the cv2.INTER_AREA the training crops went through.
  // Nearest-neighbour aliases the thin strokes of E-13B to a different thickness
  // at every crop position, which is the cue separating 8 from 0.
  const scaled = resample(cropImage(band, rect), INPUT_WIDTH, INPUT_HEIGHT);

  const out = new Float32Array(INPUT_WIDTH * INPUT_HEIGHT);
  for (let i = 0; i < out.length; i++) {
    out[i] = scaled.data[i] / 255;
  }
  return out;
}

// Driver

export interface BandReading {
  band: GrayImage;
  boxes: GlyphBox[];
  /** Raw E-13B plausibility of the segmentation. */
  quality: number;
  /** Ordering score: quality, weighted by where the band sits and how long it is. */
  rank: number;
  rows: { top: number; bottom: number };
  threshold: ThresholdMode;
  /** Deskew slope applied, dy/dx. */
  slope: number;
}

/**
 * Shortest chain of runs treated as part of the line rather than junk. The
 * auxiliary field is the shortest real field, at six characters.
 */
const MIN_CHAIN_GLYPHS = 6;

/** A MICR line is 2 transit symbols + 9 routing digits + an account, and up. */
const PLAUSIBLE_MIN_GLYPHS = 19;
const PLAUSIBLE_MAX_GLYPHS = 40;

/**
 * Order candidate bands so the likeliest MICR line is classified first, using
 * two cheap priors: the MICR line is the bottom-most print on a cheque, and it
 * runs about 19 to 40 characters. Neither is a filter; the checksum settles it.
 */
function rankBand(quality: number, boxes: number, centreFrac: number): number {
  const bottomness = 0.55 + 0.45 * clamp(centreFrac, 0, 1);
  const plausible =
    boxes >= PLAUSIBLE_MIN_GLYPHS && boxes <= PLAUSIBLE_MAX_GLYPHS ? 1 : 0.55;
  return quality * bottomness * plausible;
}

/**
 * Every plausible MICR band in a region, best-scoring first. Both thresholds are
 * scored and the better wins, which is affordable because scoring involves no
 * model, only projections and a median.
 */
export function readBands(
  region: GrayImage,
  config: SegmentConfig = DEFAULT_CONFIG,
  modes: ThresholdMode[] = ['otsu', 'adaptive'],
  full?: GrayImage,
): BandReading[] {
  const readings: BandReading[] = [];

  // Locating the band does not need full resolution, but cutting the strip
  // does. `region` is a downscaled scout; `full` carries the crop pixels.
  const source = full ?? region;
  const scale = source.height / Math.max(1, region.height);

  // Adaptive exists for uneven lighting within the band. Running both over the
  // whole photo was pure cost, so it is only reached when Otsu finds nothing.
  let candidates = findBandCandidates(region, inkMask(region, 'otsu'), config);
  if (candidates.length === 0 && modes.includes('adaptive')) {
    candidates = findBandCandidates(region, inkMask(region, 'adaptive'), config);
  }

  for (const candidate of candidates) {
    const top = Math.round(candidate.top * scale);
    const bottom = Math.round(candidate.bottom * scale);
    const raw = cropRows(source, top, bottom);
    if (raw.height < 8 || raw.width < 32) {
      continue;
    }
    const centreFrac =
      (candidate.top + candidate.bottom) / 2 / Math.max(1, region.height);

    // On the strip alone, which is a few hundred thousand pixels against the
    // photo's several million.
    for (const mode of modes) {
      let band = raw;
      let mask = inkMask(raw, mode);
      let slope = 0;
      if (config.deskew) {
        slope = estimateShear(mask);
        // Only re-threshold if the shear moved anything; doing it for a slope
        // of zero doubled the cost of the search.
        if (Math.abs(slope) > 1e-3) {
          band = shearVertical(raw, slope);
          mask = inkMask(band, mode);
        }
      }

      const boxes = findGlyphBoxes(mask, config);
      const quality = bandQuality(boxes, config);
      if (quality > 0) {
        readings.push({
          band,
          boxes,
          quality,
          rank: rankBand(quality, boxes.length, centreFrac),
          // In `full` coordinates, so the caller can draw the band back over
          // the photo it handed in.
          rows: { top, bottom },
          threshold: mode,
          slope,
        });
      }
    }
  }

  return readings.sort((a, b) => b.rank - a.rank);
}
