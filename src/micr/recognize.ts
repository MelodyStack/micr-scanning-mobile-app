/**
 * Cheque photo to validated fields.
 *
 * The only ML step is classifying one 48x32 crop at a time; everything either
 * side of it is ordinary code. The checksum is a hard gate, so a bad read is
 * rejected rather than sent on.
 *
 * Orientation is resolved by trying rather than guessing, because sensor
 * mounting, EXIF tags and `isMirrored` disagree across devices.
 */

import type { TfliteModel } from 'react-native-fast-tflite';

import {
  classAt,
  CLASSES,
  INPUT_HEIGHT,
  INPUT_WIDTH,
  type MicrClass,
  SUBSTITUTION,
  toSymbols,
} from './classes';
import {
  buildPyramid,
  cropImage,
  estimateShear,
  type GrayImage,
  type ImagePyramid,
  inkMask,
  isPyramid,
  mirrorImage,
  PYRAMID_SIZES,
  type Rect,
  rotate90,
  shearVertical,
  type ThresholdMode,
} from './image';
import { parseMicr, type ParseResult } from './parse';
import {
  type BandReading,
  bandQuality,
  cellSteps,
  cropFraction,
  cropGlyph,
  DEFAULT_CONFIG,
  findGlyphBoxes,
  findSheet,
  type GlyphBox,
  readBands,
  type SegmentConfig,
} from './segment';

export interface Recognition extends ParseResult {
  /** How many glyphs were cut from the band that produced this result. */
  glyphCount: number;
  /** Softmax confidence per glyph, in reading order. */
  confidences: number[];
  /** Lowest per-glyph confidence. */
  minConfidence: number;
  /** Quarter turns clockwise applied to the photo before reading it. */
  quarterTurns: number;
  /** True if the band had to be mirrored, i.e. a flipped sensor. */
  mirrored: boolean;
  /** Which threshold strategy produced the winning band. */
  threshold: 'otsu' | 'adaptive' | null;
  /** Where the band sat, for drawing it back over the photo. */
  bandRows?: { top: number; bottom: number };
  /** How many bands were classified before this result. Diagnostic. */
  attempts: number;
  /** Every distinct string that was read, best first. Diagnostic. */
  candidates: string[];
}

export interface RecognizeOptions {
  segment?: Partial<SegmentConfig>;
  /**
   * Region of the photo to search, in fractions of its width and height.
   * Only applies to a bare image; a pyramid is taken as already framed.
   */
  region?: Rect;
  /** Longest side to cut glyph crops at. Below ~1200 they stop separating cleanly. */
  workMaxSide?: number;
  /** Longest side to *search* for the band at. Coarse work; cheaper is fine. */
  scoutMaxSide?: number;
  /** Longest side to rank rotations at. Coarser still. */
  probeMaxSide?: number;
  /** Rotations to score before committing to one. */
  probeRotations?: number[];
  /** How many rotations survive the probe and get read at full resolution. */
  keepRotations?: number;
  /** Hard cap on bands handed to the model, across all rotations. */
  maxClassify?: number;
}

const EMPTY: Omit<Recognition, 'error'> = {
  ok: false,
  raw: '',
  glyphCount: 0,
  confidences: [],
  minConfidence: 0,
  quarterTurns: 0,
  mirrored: false,
  threshold: null,
  attempts: 0,
  candidates: [],
};

export function noBand(error: string): Recognition {
  return { ...EMPTY, error };
}

export function describeModel(model: TfliteModel): string {
  const input = model.inputs[0];
  const output = model.outputs[0];
  return (
    `in ${input?.dataType} [${input?.shape?.join(', ')}] · ` +
    `out ${output?.dataType} [${output?.shape?.join(', ')}]`
  );
}

/**
 * Check the loaded model is the one this code was written against. The wrong
 * build, or one retrained with a different class order, produces confident
 * nonsense that the checksum rejects with no clue why.
 */
export function checkModelContract(model: TfliteModel): string | null {
  const input = model.inputs[0];
  const output = model.outputs[0];
  if (!input || !output) {
    return 'model exposes no input/output tensors';
  }

  const shape = input.shape ?? [];
  const asNhwc =
    shape.length === 4 && shape[1] === INPUT_HEIGHT && shape[2] === INPUT_WIDTH &&
    shape[3] === 1;
  if (!asNhwc) {
    return (
      `model input is [${shape.join(', ')}], expected [1, ${INPUT_HEIGHT}, ` +
      `${INPUT_WIDTH}, 1] NHWC. This looks like the Core ML build or a model ` +
      'exported at a different crop size.'
    );
  }
  if (input.dataType !== 'float32') {
    return (
      `model input is ${input.dataType}, expected float32. The int8 build is ` +
      'exported with float I/O; an all-integer model needs its own scale and ' +
      'zero point applied, which this code does not do.'
    );
  }

  const outShape = output.shape ?? [];
  const classes = outShape[outShape.length - 1];
  if (classes !== CLASSES.length) {
    return (
      `model returns ${classes} classes, expected ${CLASSES.length}. ` +
      'classes.ts and micr_labels.json have to agree with the checkpoint.'
    );
  }
  return null;
}

/** Full softmax, so a runner-up can be reconsidered without re-running the model. */
export function softmaxAll(logits: ArrayLike<number>): number[] {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > max) {
      max = logits[i];
    }
  }
  let sum = 0;
  const out: number[] = [];
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - max);
    out.push(e);
    sum += e;
  }
  return out.map(e => e / sum);
}

const SYMBOLS = new Set(['T', 'A', 'O', 'D']);

/** Confidence below which a symbol is worth reconsidering. */
const UNSURE = 0.9;

/**
 * Probability a class needs before a glyph is worth reconsidering as that class.
 * Keeps the transit repair honest: trying every position would mean thirty
 * attempts, and the ABA checksum passes by luck about one time in ten.
 */
const MIN_ALTERNATIVE = 0.05;

/**
 * Read a cheque photo. Fails by returning, never by throwing: a photo that
 * caught a thumb should leave the user with a hint and a shutter button.
 */
export function recognizeCheque(
  model: TfliteModel,
  source: GrayImage | ImagePyramid,
  options: RecognizeOptions = {},
): Recognition {
  const config: SegmentConfig = { ...DEFAULT_CONFIG, ...options.segment };
  const maxClassify = options.maxClassify ?? 6;
  const keepRotations = options.keepRotations ?? 2;
  const probeRotations = options.probeRotations ?? [0, 1, 2, 3];

  // A pyramid built by Skia arrives pre-scaled; a bare image is scaled here.
  // `work` is the only level glyph crops are cut from, `scout` is where the band
  // is located, `probe` is where the rotation is decided.
  let pyramid: ImagePyramid;
  if (isPyramid(source)) {
    pyramid = source;
  } else {
    const region = options.region
      ? cropImage(source, toPixels(options.region, source))
      : source;
    pyramid = buildPyramid(region, {
      work: options.workMaxSide ?? PYRAMID_SIZES.work,
      scout: options.scoutMaxSide ?? PYRAMID_SIZES.scout,
      probe: options.probeMaxSide ?? PYRAMID_SIZES.probe,
    });
  }

  if (pyramid.work.width < 200 || pyramid.work.height < 60) {
    return noBand('the framed area is too small to hold a MICR line');
  }

  // Rotation-invariant, so done once and applied to all three levels by the
  // same fractions, keeping the scout-to-work scale factor unchanged.
  const sheet = findSheet(pyramid.probe);
  const work = cropFraction(pyramid.work, sheet);
  const scout = cropFraction(pyramid.scout, sheet);
  const probe = cropFraction(pyramid.probe, sheet);

  // Mirroring is not probed: a mirrored band segments exactly as well as an
  // upright one, so only the checksum tells them apart. Retried per candidate
  // below.
  const ranked = probeRotations
    .map(turns => ({
      turns,
      score: bestQuality(rotate90(probe, turns), config),
    }))
    .sort((a, b) => b.score - a.score);

  // Keep any rotation close to the best, since geometry cannot separate a MICR
  // line from upside-down headline text.
  const viable = ranked.filter(r => r.score > 0);
  if (viable.length === 0) {
    return noBand('no line of print that looks like a MICR band');
  }
  const cutoff = viable[0].score * 0.4;
  const chosen = viable.filter(r => r.score >= cutoff).slice(0, keepRotations);

  // Straighten over the whole sheet, not per band. A skewed line is a tall
  // smear rather than a short row (at 2 degrees across 1800 px the band climbs
  // ~63 px) so it fails maxBandHeightFrac and is discarded before there is any
  // band left to deskew.
  const levelled = chosen.map(({ turns }) => {
    let scouted = turns === 0 ? scout : rotate90(scout, turns);
    let full = turns === 0 ? work : rotate90(work, turns);
    if (config.deskew) {
      const slope = estimateShear(inkMask(scouted, 'otsu'), 0.25);
      if (Math.abs(slope) > 4e-3) {
        // Slope is dimensionless, so the same value applies at both scales.
        scouted = shearVertical(scouted, slope);
        full = shearVertical(full, slope);
      }
    }
    return { turns, scouted, full };
  });

  const seen: string[] = [];
  let bestFailure: Recognition | null = null;
  let attempts = 0;

  const attempt = (modes: ThresholdMode[]): Recognition | null => {
    const readings: { reading: BandReading; turns: number }[] = [];
    for (const { turns, scouted, full } of levelled) {
      for (const reading of readBands(scouted, config, modes, full)) {
        readings.push({ reading, turns });
      }
    }
    readings.sort((a, b) => b.reading.rank - a.reading.rank);

    for (const { reading, turns } of readings.slice(0, maxClassify)) {
      attempts++;

      const upright = classifyBand(model, reading.band, reading.boxes, config);
      record(seen, upright.raw);
      const uprightResult = finish(upright, reading, turns, false, attempts, seen);
      if (uprightResult.ok) {
        return uprightResult;
      }
      bestFailure = better(bestFailure, uprightResult);

      // A flipped sensor segments and classifies perfectly, then assembles
      // backwards. Only tried after an upright read has failed.
      const flipped = mirrorImage(reading.band);
      const flippedBoxes = findGlyphBoxes(inkMask(flipped, reading.threshold), config);
      if (bandQuality(flippedBoxes, config) > 0) {
        const mirrored = classifyBand(model, flipped, flippedBoxes, config);
        record(seen, mirrored.raw);
        const mirroredResult = finish(mirrored, reading, turns, true, attempts, seen);
        if (mirroredResult.ok) {
          return mirroredResult;
        }
        bestFailure = better(bestFailure, mirroredResult);
      }
    }
    return null;
  };

  // Otsu suits most cheques and is the cheaper of the two; adaptive is for one
  // lit from the side. Running both unconditionally roughly doubled the time for
  // a photo Otsu alone already read correctly.
  return (
    attempt(['otsu']) ??
    attempt(['adaptive']) ??
    bestFailure ??
    noBand('found a line of print, but it does not segment as E-13B')
  );
}

function toPixels(region: Rect, image: GrayImage): Rect {
  return {
    x0: region.x0 * image.width,
    y0: region.y0 * image.height,
    x1: region.x1 * image.width,
    y1: region.y1 * image.height,
  };
}

/**
 * How well does this rotation parse as E-13B? No model calls, just geometry.
 *
 * Both threshold modes deliberately: Otsu alone made the probe blind to the
 * glare-lit images that need adaptive, scoring every rotation zero.
 */
function bestQuality(image: GrayImage, config: SegmentConfig): number {
  return readBands(image, config, ['otsu', 'adaptive'])[0]?.rank ?? 0;
}

function record(seen: string[], raw: string): void {
  if (raw && !seen.includes(raw)) {
    seen.push(raw);
  }
}

/** Prefer the failure that got furthest, which makes a better hint. */
function better(current: Recognition | null, candidate: Recognition): Recognition {
  if (!current) {
    return candidate;
  }
  return candidate.glyphCount > current.glyphCount ? candidate : current;
}

interface BandClassification {
  raw: string;
  confidences: number[];
  glyphCount: number;
  /** Spacing to the next glyph, in character cells. */
  steps: number[];
  /** Runner-up symbol for each position, where the model was unsure. */
  swaps: { index: number; to: string }[];
  /** Positions the model thought might be a transit, best first. */
  transitCandidates: { index: number; p: number }[];
}

/**
 * Did the segmenter drop characters out of the middle of a number?
 *
 * Nothing else can see this: the checksum covers only the routing number, and
 * the model answers the crops it is given correctly, so a read missing two
 * account digits still scored 0.93. Fixed pitch supplies the evidence, since a
 * gap between two digits is a hole where characters used to be.
 */
export function hasMissingDigits(raw: string, steps: number[]): boolean {
  for (let i = 0; i + 1 < raw.length && i < steps.length; i++) {
    const isDigit = raw[i] >= '0' && raw[i] <= '9';
    const nextIsDigit = raw[i + 1] >= '0' && raw[i + 1] <= '9';
    if (isDigit && nextIsDigit && steps[i] > 1) {
      return true;
    }
  }
  return false;
}

function classifyBand(
  model: TfliteModel,
  band: GrayImage,
  boxes: GlyphBox[],
  config: SegmentConfig,
): BandClassification {
  const classes: MicrClass[] = [];
  const confidences: number[] = [];
  const swaps: { index: number; to: string }[] = [];
  const transitCandidates: { index: number; p: number }[] = [];
  const transitIndex = CLASSES.indexOf('transit');

  boxes.forEach((box, position) => {
    const input = cropGlyph(band, box, config);
    // Normalisation is a layer inside the model, so the crop goes straight
    // across with no mean/std applied here.
    const [logits] = model.runSync([input.buffer as ArrayBuffer]);
    const probabilities = softmaxAll(new Float32Array(logits));

    let best = 0;
    for (let i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > probabilities[best]) {
        best = i;
      }
    }
    const label = SUBSTITUTION[classAt(best)];
    classes.push(classAt(best));
    confidences.push(probabilities[best]);

    // Symbol for symbol only: digits are never reconsidered, so the routing
    // and account numbers stay exactly as read.
    if (SYMBOLS.has(label) && probabilities[best] < UNSURE) {
      let alternative = -1;
      for (let i = 0; i < probabilities.length; i++) {
        if (i === best || !SYMBOLS.has(SUBSTITUTION[classAt(i)])) {
          continue;
        }
        if (alternative < 0 || probabilities[i] > probabilities[alternative]) {
          alternative = i;
        }
      }
      if (alternative >= 0) {
        swaps.push({ index: position, to: SUBSTITUTION[classAt(alternative)] });
      }
    }

    // Anywhere the model gave transit a real chance, digits included. The
    // transit pair is mandatory, so a line carrying one has lost one.
    if (
      label !== 'T' &&
      transitIndex >= 0 &&
      probabilities[transitIndex] >= MIN_ALTERNATIVE
    ) {
      transitCandidates.push({ index: position, p: probabilities[transitIndex] });
    }
  });
  transitCandidates.sort((a, b) => b.p - a.p);

  return {
    raw: toSymbols(classes),
    confidences,
    glyphCount: boxes.length,
    steps: cellSteps(boxes),
    swaps,
    transitCandidates,
  };
}

/**
 * Retry a failed read with a few leading characters discarded, for when the band
 * search picks up print to the left of the line.
 *
 * Leading side only. The trailing side holds the account number, so trimming
 * there would quietly shorten it.
 */
const MAX_LEADING_TRIM = 3;

function parseAllowingLeadingJunk(
  raw: string,
  steps: number[],
): { parsed: ParseResult; trimmed: number } {
  const direct = parseMicr(raw);
  if (direct.ok && !hasMissingDigits(raw, steps)) {
    return { parsed: direct, trimmed: 0 };
  }
  for (let lead = 1; lead <= MAX_LEADING_TRIM && lead < raw.length; lead++) {
    const candidate = raw.slice(lead);
    const parsed = parseMicr(candidate);
    if (parsed.ok && !hasMissingDigits(candidate, steps.slice(lead))) {
      return { parsed: { ...parsed, raw }, trimmed: lead };
    }
  }
  return { parsed: direct, trimmed: 0 };
}

/** Most swap combinations to try before giving up on a line. */
const MAX_SHORTLIST = 12;

/**
 * Retry with an unsure symbol replaced by the model's second choice. The symbols
 * delimit the fields, so one wrong symbol invalidates an otherwise perfect read.
 * A swap still has to satisfy the checksum, the field structure and the
 * missing-digit check.
 */
function repairSymbols(
  raw: string,
  steps: number[],
  swaps: { index: number; to: string }[],
  transitCandidates: { index: number; p: number }[],
): { parsed: ParseResult; repaired: string } | null {
  if (swaps.length === 0 && transitCandidates.length === 0) {
    return null;
  }
  const attempt = (candidate: string) => {
    const result = parseAllowingLeadingJunk(candidate, steps);
    return result.parsed.ok && !hasMissingDigits(candidate, steps)
      ? { parsed: { ...result.parsed, raw }, repaired: candidate }
      : null;
  };

  // Confidence does not help here: on a tilted capture the wrong reading scored
  // above 0.90, so a threshold that caught it would have reconsidered half the
  // line. The nine characters a promotion exposes still have to satisfy the
  // checksum.
  const structural: { index: number; to: string }[] = [];
  if ((raw.match(/T/g) ?? []).length === 1) {
    for (const candidate of transitCandidates.slice(0, 4)) {
      structural.push({ index: candidate.index, to: 'T' });
    }
  }

  const shortlist = [...swaps, ...structural].slice(0, MAX_SHORTLIST);
  for (const swap of shortlist) {
    const hit = attempt(raw.slice(0, swap.index) + swap.to + raw.slice(swap.index + 1));
    if (hit) {
      return hit;
    }
  }
  for (let a = 0; a < shortlist.length; a++) {
    for (let b = a + 1; b < shortlist.length; b++) {
      let candidate = raw;
      for (const swap of [shortlist[a], shortlist[b]]) {
        candidate =
          candidate.slice(0, swap.index) + swap.to + candidate.slice(swap.index + 1);
      }
      const hit = attempt(candidate);
      if (hit) {
        return hit;
      }
    }
  }
  return null;
}

function finish(
  classification: BandClassification,
  reading: BandReading,
  quarterTurns: number,
  mirrored: boolean,
  attempts: number,
  candidates: string[],
): Recognition {
  const { parsed: settled } = parseAllowingLeadingJunk(
    classification.raw,
    classification.steps,
  );
  const repaired = settled.ok
    ? null
    : repairSymbols(
        classification.raw,
        classification.steps,
        classification.swaps,
        classification.transitCandidates,
      );

  const parsed: ParseResult = settled.ok
    ? settled
    : repaired
      ? repaired.parsed
      : parseMicr(classification.raw).ok
        ? {
            ok: false,
            raw: classification.raw,
            error: 'a character is missing from the middle of a number',
          }
        : settled;
  return {
    ...parsed,
    raw: classification.raw,
    glyphCount: classification.glyphCount,
    confidences: classification.confidences,
    minConfidence: classification.confidences.length
      ? Math.min(...classification.confidences)
      : 0,
    quarterTurns,
    mirrored,
    threshold: reading.threshold,
    bandRows: reading.rows,
    attempts,
    candidates: [...candidates],
  };
}

/**
 * Confidence below which a read is flagged even though it passed the checksum.
 * A random misread passes the ABA check by luck roughly one time in ten.
 */
export const LOW_CONFIDENCE = 0.75;
