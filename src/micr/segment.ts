/**
 * Find the MICR band in a cheque photo and cut it into glyph crops.
 *
 * A port of micr/segment.py from the training repo, and the correspondence is
 * the point of the file. The model only ever saw crops that the Python
 * segmenter produced; a crop cut to different proportions here is an input it
 * was never trained on, and it will guess. Where this diverges from the Python
 * it is marked DIVERGES and justified.
 *
 * Nothing here imports a native module. The whole file is exercised by
 * __tests__/micr.test.ts on synthetic images, with no device involved.
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
   * Gap, in pitches, that ends one chain of print and starts another.
   *
   * Measured across the real cheque photos in the training repo: every cheque
   * that segments to exactly the right glyph count has a largest internal gap
   * of 1.24 to 1.46 pitches, those being the blank cells between MICR fields.
   * A capture whose band picked up micro-print from the sheet's left edge put
   * that junk 2.96 pitches out. 2.0 sits between the two with room either side;
   * the training repo's 3.0 was tuned before any of this was measured and let
   * the junk through.
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

// --- finding the sheet -----------------------------------------------------

/**
 * Fractional bounds of the sheet of paper within the frame.
 *
 * This has to happen before anything looks for a band, and skipping it is not a
 * small loss of quality -- it is total failure. A photo of a cheque is mostly
 * desk, and Otsu over the whole frame separates *desk from paper*, not *ink
 * from paper*. Every pixel of the desk then counts as ink, the rows above and
 * below the cheque come out solid, and they merge into one run far too tall to
 * be a line of text. Every candidate is discarded and the search reports
 * nothing, on a frame where the band is perfectly legible.
 *
 * Measured on the real cheque photos in the training repo at preview
 * resolution: with this step, chk001 segments to all 32 glyphs; without it, to
 * zero. The synthetic test cheque hides the problem because the image *is* the
 * sheet, with no desk around it.
 *
 * Brightness profile rather than contour finding: a cheque is a bright
 * rectangle on a darker surface, which is the whole of the signal needed, and
 * it costs two passes instead of an edge detector.
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

// --- band location ---------------------------------------------------------

export interface BandCandidate {
  top: number;
  bottom: number;
  /** Fraction of the region width the line of ink spans. */
  coverage: number;
}

/**
 * Horizontal strips that might be a line of print.
 *
 * Ink is smeared along x first, so a line of separate glyphs becomes one
 * continuous blob while the cheque's printed border stays a thin rule. The
 * strips that survive are ranked later by how well they parse as E-13B, never
 * by how much ink they carry: on a real cheque the signature line, the memo
 * rule and a printed caption all carry more ink than the MICR line, and picking
 * the heaviest strip reliably grabs one of those instead.
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

  // Pass two: gate each core and pad it out to a band.
  //
  // Growing the run outwards over faint rows first was tried, on the theory
  // that the row threshold clips the sparse top and bottom of a line and hands
  // the model crops cut through the glyph. The data refuted it: without any
  // growth the reference cheque reads exactly right at 0.95 confidence, and
  // with it the band doubles in height, swallows the rule above, and the read
  // falls apart. `bandPadFrac` is already doing this job.
  const candidates: BandCandidate[] = [];
  for (const core of cores) {
    // Same two gates as the connected-component filter in the Python: the blob
    // must be wide enough to be a line of print, and short enough not to be a
    // block of handwriting or a dark region of the photo.
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

  // Bottom-most first. The MICR line is always the lowest line of print on a
  // cheque, so this is the order most likely to hit it on the first classify --
  // but it only orders the work, it never excludes anything.
  return candidates
    .sort((a, b) => b.top - a.top)
    .slice(0, config.maxCandidates);
}

// --- glyph boundaries ------------------------------------------------------

/**
 * Glyph boundaries, from fitting a fixed-pitch character grid.
 *
 * Gap-based merging cannot work on E-13B. The transit and on-us symbols are
 * drawn as several separate vertical strokes, so any gap threshold loose enough
 * to join one symbol's strokes also joins two adjacent digits, and any
 * threshold tight enough to keep digits apart shatters the symbols. Both
 * failure modes showed up on the first real cheque the training code was run
 * against.
 *
 * The font's constant pitch resolves it: estimate the pitch, fit a grid, and
 * let cell membership decide. Strokes of one symbol share a cell, adjacent
 * characters do not, and the blank cells between fields simply hold no ink.
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

  // Runs touching the edge are the cheque's printed border or a neighbouring
  // field bleeding in -- but only drop them if something is left, so that a
  // band which binarises to one edge-to-edge blob still reports that blob
  // rather than reporting nothing.
  const trimmed = runs.filter(r => r.x0 > 0 && r.x1 < width);
  let kept = trimmed.length >= 2 ? trimmed : runs;
  if (kept.length < 2) {
    return kept;
  }

  let stats = estimatePitch(kept, config);
  if (!isFinite(stats.pitch) || stats.pitch <= 1) {
    return kept;
  }

  // Drop anything too wide to be one character. E-13B is fixed pitch, so a
  // single glyph's ink cannot span much more than one cell -- a run that does
  // is a rule, a border, or a dark patch of the photo. On a real cheque this
  // catches a 182 px blob against a 16 px pitch, which otherwise merges into a
  // cell and corrupts both the pitch estimate and the character it lands on.
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

  // Drop junk chains, keep the fields.
  //
  // A cheque's border rules, corner specks and edge micro-print sit apart from
  // the band, and the obvious filter -- drop anything with no near neighbour --
  // does not remove them, because they arrive in clusters that protect each
  // other. On one capture the leading junk was three marks 24 and 33 px apart
  // sitting 132 px out: every one had a close neighbour, so every one survived
  // and the read came back seven characters too long.
  //
  // Splitting into chains is the global version of that idea. What the chains
  // are then judged on is **length, not distance**. Keeping the longest chain
  // was the first attempt and it is wrong: the gap before the on-us field is a
  // genuine field boundary and on a real cheque it measured 2.21 pitches, so
  // the account number split off into its own chain and was thrown away --
  // `O010454OT111000614T` with all eleven characters of the account missing.
  //
  // Junk arrives in ones and twos, occasionally five. A MICR field never does:
  // the transit field is eleven characters, the on-us nine to thirteen. So the
  // substantial chains are all kept and rejoined, and only short ones at the
  // very ends are discarded. A short chain *between* two kept ones stays, since
  // it is part of the line whatever it is.
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
 * Rejoin a character the grid cut in half.
 *
 * The grid assigns each run of ink to a cell, and a cell boundary landing
 * inside a multi-stroke symbol splits it. The on-us symbol is the one that
 * suffers: it is two thin bars and a block, so a boundary falling after the
 * first bar leaves a 3 px orphan beside a 13 px remainder, against a 17 px
 * median. The remainder still classifies as on-us; the orphan becomes a phantom
 * character, and the line comes back one glyph too long with a `D` bolted on
 * the front. It was the single most common failure on real cheques.
 *
 * Fixed pitch is what makes the repair safe. One character's ink cannot span
 * more than one cell, so two neighbours that *together* still fit inside a
 * pitch were never two characters. Measured on a real cheque at a 24 px pitch:
 * the two halves of a split symbol sit 2 px apart and span 18 px together,
 * while genuinely adjacent characters sit 6 px apart and span 36 px -- the two
 * cases are nowhere near each other.
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
 * Break a left-to-right list of runs wherever the gap exceeds `maxGap`.
 *
 * Gaps are measured between edges, not centres: two wide runs whose centres sit
 * far apart may still be touching, and it is the blank space between them that
 * says whether they belong to the same line of print.
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
 * Typical run width, and the character pitch.
 *
 * The pitch is the median spacing between run centres, ignoring the short hops
 * between the strokes inside one symbol. With only a handful of multi-stroke
 * symbols in a 30-odd glyph line, the median still lands on a digit.
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
 * The training code refines only the origin and keeps the pitch as first
 * estimated. That is fine over a handful of characters and wrong over thirty:
 * a pitch out by 3% drifts nearly a whole cell across a MICR line, and once the
 * grid slips, characters at one end start sharing a cell or splitting across
 * two. The symptom is a line with the right number of boxes but the wrong
 * contents -- on real cheques, a narrow `1` swallowed by its neighbour and a
 * stray `D` conjured out of the leftover fragment.
 *
 * So: assign each run to its nearest cell, then least-squares regress centre
 * against cell index to get pitch (slope) and origin (intercept), and repeat.
 * Assignments stop moving after two or three rounds.
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
    // Refuse a fit that has collapsed or run away -- it means the assignment
    // stepped to a different multiple of the true pitch, and the previous
    // round is the better answer.
    if (!isFinite(fitted) || fitted < pitch * 0.5 || fitted > pitch * 2) {
      return grid;
    }
    grid = { origin: meanCentre - fitted * meanCell, pitch: fitted };
  }
  return grid;
}

/**
 * How much does this look like a real MICR line rather than texture?
 *
 * Counting boxes alone is not enough -- it rewards noise. In the training repo
 * an upside-down cheque once outscored the right way up, because a band of
 * texture shattered into 70 fragments. A genuine E-13B line has a bounded
 * number of characters, widths that cluster (the glyphs differ, but only within
 * about 2x), and centres that sit on a constant pitch. All three are required.
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
 * Spacing between consecutive glyphs, measured in character cells.
 *
 * E-13B is fixed pitch, so neighbouring characters are one cell apart and a
 * blank cell between fields reads as two. Anything larger is a hole where
 * characters should have been -- which is the only evidence there is that the
 * segmenter dropped some, because the model cannot be unsure about a crop it
 * was never handed.
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

// --- glyph crops -----------------------------------------------------------

/**
 * Cut one glyph to the model's input size.
 *
 * Full band height, never the glyph's own bounding box. The training renderer
 * places every glyph on one shared baseline at its true relative height, so the
 * dash is short *within its crop*. Cropping tight here would rescale the dash to
 * full height and hand the model something that looks nothing like what it was
 * trained on -- and the dash would come back as a digit.
 *
 * Returns float32 in [0, 1]. Normalisation is a layer inside the model, so
 * nothing else is applied.
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
  // Nearest-neighbour here cost real accuracy: it aliases the thin strokes of
  // E-13B into a different thickness at every crop position, which is exactly
  // the cue separating 8 from 0.
  const scaled = resample(cropImage(band, rect), INPUT_WIDTH, INPUT_HEIGHT);

  const out = new Float32Array(INPUT_WIDTH * INPUT_HEIGHT);
  for (let i = 0; i < out.length; i++) {
    out[i] = scaled.data[i] / 255;
  }
  return out;
}

// --- driver ----------------------------------------------------------------

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
 * Shortest chain of runs still treated as part of the line rather than junk.
 *
 * The auxiliary field is the shortest real field, at six characters including
 * its delimiters. Junk clusters on the cheques seen so far run from one mark to
 * five.
 */
const MIN_CHAIN_GLYPHS = 6;

/** A MICR line is 2 transit symbols + 9 routing digits + an account, and up. */
const PLAUSIBLE_MIN_GLYPHS = 19;
const PLAUSIBLE_MAX_GLYPHS = 40;

/**
 * Order candidate bands so the likeliest MICR line is classified first.
 *
 * Raw segmentation quality is not enough on its own. A block of clean sans
 * headline text segments into well-pitched, similar-width boxes and can outrank
 * the real band -- running the training repo's segmenter over the synthetic test
 * cheque, it settled on an upside-down "ACME MANUFACTURING LLC". Two cheap
 * priors fix the ordering:
 *
 *   * the MICR line is the bottom-most line of print on a cheque, and
 *   * it is between about 19 and 40 characters long.
 *
 * Neither is a filter. Both only decide what is tried first, because the thing
 * that actually settles it is the ABA checksum a few steps later.
 */
function rankBand(quality: number, boxes: number, centreFrac: number): number {
  const bottomness = 0.55 + 0.45 * clamp(centreFrac, 0, 1);
  const plausible =
    boxes >= PLAUSIBLE_MIN_GLYPHS && boxes <= PLAUSIBLE_MAX_GLYPHS ? 1 : 0.55;
  return quality * bottomness * plausible;
}

/**
 * Every plausible MICR band in a region, best-scoring first.
 *
 * Both threshold strategies are tried on every candidate. Otsu is right for an
 * evenly-lit cheque and adaptive is right for one lit from the side, and which
 * applies cannot be known in advance -- so both are scored and the better one
 * wins. This is affordable precisely because scoring involves no model: it is
 * projections and a median, and only the handful of survivors ever reach the
 * network.
 */
export function readBands(
  region: GrayImage,
  config: SegmentConfig = DEFAULT_CONFIG,
  modes: ThresholdMode[] = ['otsu', 'adaptive'],
  full?: GrayImage,
): BandReading[] {
  const readings: BandReading[] = [];

  // Locating the band is a coarse job -- row sums over a smeared mask -- and
  // does not need full resolution. Cutting the strip does. So `region` can be a
  // downscaled scout while `full` carries the pixels the crops come from, which
  // is what keeps a 2400 px photo from being thresholded end to end several
  // times over. Passing neither is the same image for both.
  const source = full ?? region;
  const scale = source.height / Math.max(1, region.height);

  // Locating a line of print is a coarse job and Otsu does it; the adaptive
  // pass exists for uneven lighting *within* the band. Running both over the
  // whole photo was pure cost -- so adaptive is only reached for the search if
  // Otsu turns up nothing at all, which is the genuinely badly-lit case.
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

    // Both thresholds are tried here, on the strip alone. A band is a few
    // hundred thousand pixels against the photo's several million, so trying
    // two ways of binarising it costs almost nothing -- and which one is right
    // genuinely cannot be known in advance.
    for (const mode of modes) {
      let band = raw;
      let mask = inkMask(raw, mode);
      let slope = 0;
      if (config.deskew) {
        slope = estimateShear(mask);
        // Only re-threshold if the shear actually moved anything. On a guided
        // capture the band is usually already level, and thresholding a strip
        // twice for a slope of zero was doubling the cost of the search.
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

/** The single best-scoring band in a region, or null if none parses as E-13B. */
export function findMicrBand(
  region: GrayImage,
  config: SegmentConfig = DEFAULT_CONFIG,
): BandReading | null {
  return readBands(region, config)[0] ?? null;
}
