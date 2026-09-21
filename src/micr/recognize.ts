/**
 * Cheque photo to validated fields.
 *
 * The only ML step is classifying one 48x32 crop at a time. Everything either
 * side of it is ordinary code: find the band, cut the glyphs, join the
 * predictions, check the ABA digit. The checksum is a hard gate, so a bad read
 * is rejected rather than sent on.
 *
 * Orientation is resolved by trying rather than guessing. Sensor mounting, EXIF
 * tags and `isMirrored` flags disagree across devices, so each rotation is
 * scored on how well it parses as E-13B (which costs no model calls) and only
 * the best one or two are classified.
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
 * Check the loaded model is the one this code was written against.
 *
 * `export/` contains float32, fp16 and int8 builds, and the app bundles one of
 * them under a name that does not say which. The wrong file, or a model
 * retrained with a different class order, produces confident nonsense that the
 * checksum rejects with no clue why.
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

/**
 * Confidence below which a symbol is worth reconsidering. The four E-13B
 * symbols are built from separate strokes, so a soft or tilted crop blurs them
 * into each other; they are what the model is least sure about.
 */
const UNSURE = 0.9;

/**
 * Probability a class needs before a glyph is worth reconsidering as that class.
 *
 * This keeps the transit repair honest. A line with one transit is missing one,
 * but trying every position means thirty attempts and the ABA checksum passes by
 * luck about one time in ten. Restricting it to positions the model itself
 * ranked as a possible transit collapses that to one or two candidates.
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

  // Crop to the sheet first. Rotation-invariant, so it is done once and applied
  // to all three levels by the same fractions, which keeps the scale factor
  // between scout and work unchanged.
  const sheet = findSheet(pyramid.probe);
  const work = cropFraction(pyramid.work, sheet);
  const scout = cropFraction(pyramid.scout, sheet);
  const probe = cropFraction(pyramid.probe, sheet);

  // Mirroring is not probed. A mirrored band segments exactly as well as an
  // upright one, so no amount of geometry tells them apart; only classifying and
  // checking the ABA digit does. That retry happens below, per candidate.
  const ranked = probeRotations
    .map(turns => ({
      turns,
      score: bestQuality(rotate90(probe, turns), config),
    }))
    .sort((a, b) => b.score - a.score);

  // Keep any rotation close to the best, not just the winner. Geometry cannot
  // separate a MICR line from a block of upside-down headline text, so pruning
  // hard here would discard the right answer before the checksum saw it.
  const viable = ranked.filter(r => r.score > 0);
  if (viable.length === 0) {
    return noBand('no line of print that looks like a MICR band');
  }
  const cutoff = viable[0].score * 0.4;
  const chosen = viable.filter(r => r.score >= cutoff).slice(0, keepRotations);

  // Straighten each surviving rotation once, over the whole sheet.
  //
  // The band search groups rows of ink, and a skewed line of print is a tall
  // smear rather than a short row: at 2 degrees across an 1800 px cheque the
  // band climbs ~63 px, comparable to its own height, so it fails the
  // maxBandHeightFrac test and is discarded. Deskewing per band cannot rescue
  // that because there is no band left to deskew. Every line of print on a
  // cheque shares the same skew, so the sheet-wide estimate has more to work
  // with anyway.
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

      // A flipped sensor produces a band that segments and classifies perfectly
      // then assembles backwards, failing the checksum with nothing to indicate
      // why. Only tried after an upright read has already failed.
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

  // Otsu first, on its own: it is right for an evenly-lit cheque, which is most
  // of them, and much the cheaper of the two. Adaptive is for a cheque lit from
  // one side, where a single global threshold either loses the shaded end of the
  // band or floods the lit end. Running both unconditionally roughly doubled the
  // time for a photo Otsu alone already read correctly.
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
 * Both threshold modes deliberately. `readBands` only reaches for adaptive when
 * Otsu finds no candidate at all, so this stays cheap on an ordinary cheque, but
 * passing Otsu alone made the probe blind to the images that need adaptive: on a
 * capture with glare, every rotation scored zero and the read gave up before the
 * adaptive path was tried.
 */
function bestQuality(image: GrayImage, config: SegmentConfig): number {
  return readBands(image, config, ['otsu', 'adaptive'])[0]?.rank ?? 0;
}

function record(seen: string[], raw: string): void {
  if (raw && !seen.includes(raw)) {
    seen.push(raw);
  }
}

/**
 * Prefer the failure that got furthest. A read that assembled 32 glyphs and
 * missed the checksum by one digit is more useful to show than one that found
 * four smudges.
 */
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
 * The rest of the pipeline cannot see this. The ABA checksum only covers the
 * routing number, and confidence says nothing: the model answers the crops it is
 * given correctly, so a read missing two account digits came back at 0.93
 * minimum confidence.
 *
 * Fixed pitch supplies the missing evidence. Adjacent characters are one cell
 * apart and a blank cell between fields makes two, so a gap between two *digits*
 * is a hole where characters used to be.
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
    // runSync takes and returns raw ArrayBuffers. The model wants
    // [1, 48, 32, 1] NHWC float32 in 0..1; normalisation is a layer inside the
    // model, so the crop goes straight across with no mean/std applied here.
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

    // Note the runner-up while the probabilities are to hand. Symbol for symbol
    // only, and only where the model was unsure: digits are never reconsidered,
    // so the routing and account numbers are exactly as read.
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

    // Separately, note anywhere the model gave transit a real chance, digits
    // included. The transit pair is mandatory, so a line carrying one has lost
    // one, and these are the only places worth looking.
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
 * Retry a failed read with a few leading characters discarded.
 *
 * Cheques carry print to the left of the band (vertical micro-text, border
 * rules, specks) and the band search keeps it when it falls within a couple of
 * pitches of the first character. The result is a line that is right with junk
 * on the front.
 *
 * Only the leading side is trimmed. The trailing side holds the account number,
 * so dropping characters there would quietly shorten it; the leading side holds
 * the auxiliary cheque number, which has to match `O<digits>O` exactly for the
 * trimmed string to parse at all.
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
 * Retry with an unsure symbol replaced by the model's second choice.
 *
 * The symbols delimit the fields, so one wrong symbol invalidates an otherwise
 * perfect read. Only symbols are reconsidered: digits stay exactly as
 * classified, and a swap still has to satisfy the ABA checksum, the field
 * structure and the missing-digit check before it is accepted. The runner-up
 * came out of the same softmax as the winner, so trying costs nothing.
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

  // Structural candidates on top of the confidence-based ones. Exactly two
  // transits delimit the routing number, so a line carrying one has lost a
  // transit. Confidence does not help here: on a tilted capture the wrong
  // reading scored above 0.90, so a threshold that caught it would have
  // reconsidered half the line. Promotion is not trusted on its own, as the nine
  // characters it exposes still have to be digits and satisfy the checksum.
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
 * A random misread has roughly a 1-in-10 chance of passing the ABA check by
 * luck, and requiring two transits in plausible positions narrows that a long
 * way further but not to zero.
 */
export const LOW_CONFIDENCE = 0.75;
