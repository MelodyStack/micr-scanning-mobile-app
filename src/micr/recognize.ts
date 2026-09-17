/**
 * Cheque photo -> validated fields.
 *
 * The only ML step in the whole pipeline is classifying one 48x32 crop at a
 * time. Everything either side of it is ordinary code: find the band, cut the
 * glyphs, join the predictions, check the ABA digit. That is the design in spec
 * section 3 -- the checksum is a hard correctness gate, so a bad read is
 * rejected rather than sent on.
 *
 * The orientation and mirror problem is solved by trying, not by guessing.
 * Sensor mounting, EXIF tags and `isMirrored` flags all disagree across
 * devices, and every one of those mappings is a chance to be wrong in a way
 * that looks exactly like a camera seeing nothing. Instead each rotation is
 * scored by how well it parses as E-13B -- which costs no model calls -- and
 * only the best one or two are ever classified.
 */

import type { TfliteModel } from 'react-native-fast-tflite';

import { classAt, INPUT_HEIGHT, INPUT_WIDTH, type MicrClass, toSymbols } from './classes';
import { CLASSES } from './classes';
import {
  buildPyramid,
  cropImage,
  type GrayImage,
  type ImagePyramid,
  estimateShear,
  inkMask,
  isPyramid,
  mirrorImage,
  PYRAMID_SIZES,
  type Rect,
  rotate90,
  shearVertical,
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
import type { ThresholdMode } from './image';


export interface Recognition extends ParseResult {
  /** How many glyphs were cut from the band that produced this result. */
  glyphCount: number;
  /** Softmax confidence per glyph, in reading order. */
  confidences: number[];
  /** Lowest per-glyph confidence -- the weakest link in the read. */
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
   *
   * Only applies when a bare image is passed; a pre-built pyramid is taken as
   * already framed. The scanner does not set this -- the band is located by
   * searching, so no coordinate has to be mapped from the screen's guide box
   * into sensor space, which is where the previous version of this app spent
   * most of its bugs.
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

/**
 * Check the loaded model really is the one this code was written against.
 *
 * Worth doing on every load. `export/` contains a float32, an fp16 and an int8
 * build, and the app bundles one of them under a name that does not say which.
 * Shipping the wrong file, or a model retrained with a different class order,
 * produces confident nonsense that the checksum rejects with no clue why.
 */
export function describeModel(model: TfliteModel): string {
  const input = model.inputs[0];
  const output = model.outputs[0];
  return (
    `in ${input?.dataType} [${input?.shape?.join(', ')}] · ` +
    `out ${output?.dataType} [${output?.shape?.join(', ')}]`
  );
}

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
      'exported with float I/O -- an all-integer model needs its own scale and ' +
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

/** Peak softmax probability, and the index it belongs to. */
export function softmaxPeak(logits: ArrayLike<number>): { index: number; p: number } {
  let best = 0;
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > max) {
      max = logits[i];
      best = i;
    }
  }
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    sum += Math.exp(logits[i] - max);
  }
  // exp(max - max) / sum === 1 / sum.
  return { index: best, p: 1 / sum };
}

/**
 * Read a cheque photo.
 *
 * Fails by returning, never by throwing: a photo that caught a thumb should
 * leave the user with a hint and a shutter button, not a red screen.
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
  // Either way `work` is the only level glyph crops are cut from, `scout` is
  // where the band is located, and `probe` is where the rotation is decided.
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

  // Crop to the sheet before anything else. Rotation-invariant, so it is done
  // once, up front, and applied to all three levels by the same fractions --
  // which keeps the scale factor between scout and work exactly as it was.
  const sheet = findSheet(pyramid.probe);
  const work = cropFraction(pyramid.work, sheet);
  const scout = cropFraction(pyramid.scout, sheet);
  const probe = cropFraction(pyramid.probe, sheet);
  // Mirroring is deliberately not probed: a mirrored band segments exactly as
  // well as an upright one -- same glyph count, same pitch, same widths -- so
  // no amount of geometry tells the two apart. Only classifying and checking
  // the ABA digit does, so that retry happens below, per candidate.
  const ranked = probeRotations
    .map(turns => ({
      turns,
      score: bestQuality(rotate90(probe, turns), config),
    }))
    .sort((a, b) => b.score - a.score);

  // Keep any rotation that scored within striking distance of the best, not
  // just the winner. Geometry genuinely cannot separate a MICR line from a
  // block of upside-down headline text -- the training repo's segmenter picks
  // the headline on the synthetic test cheque -- so pruning hard here would
  // throw away the right answer before the checksum ever got a look at it.
  const viable = ranked.filter(r => r.score > 0);
  if (viable.length === 0) {
    return noBand('no line of print that looks like a MICR band');
  }
  const cutoff = viable[0].score * 0.4;
  const chosen = viable.filter(r => r.score >= cutoff).slice(0, keepRotations);

  // Read every surviving rotation at full resolution, then rank all the bands
  // together so the best one wins regardless of which rotation produced it.
  // Straighten each surviving rotation once, up front.
  //
  // The band search groups rows of ink, and a skewed line of print is a tall
  // smear rather than a short row -- at 2 degrees across an 1800 px cheque the
  // band climbs ~63 px, comparable to its own height, so it fails the
  // maxBandHeightFrac test and is discarded. Deskewing per band cannot rescue
  // that, because there is no band left to deskew. Estimating the angle over
  // the whole sheet also has far more to work with: every line of print on a
  // cheque shares the same skew.
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

      // A flipped sensor produces a band that segments perfectly and classifies
      // perfectly, then assembles backwards -- so it fails the checksum with
      // nothing to indicate why. Only the band is mirrored, and only after an
      // upright read has already failed.
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

  // Otsu first, on its own. It is right for an evenly-lit cheque, which is most
  // of them, and it is much the cheaper of the two. The adaptive pass exists
  // for a cheque lit from one side, where a single global threshold either
  // loses the shaded end of the band or floods the lit end -- so it is worth
  // having, but not worth paying for on every read. Running both passes
  // unconditionally roughly doubled the time for a photo that Otsu alone
  // already read correctly.
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

function bestQuality(image: GrayImage, config: SegmentConfig): number {
  return readBands(image, config, ['otsu'])[0]?.rank ?? 0;
}

function record(seen: string[], raw: string): void {
  if (raw && !seen.includes(raw)) {
    seen.push(raw);
  }
}

/**
 * Prefer the failure that got furthest. A read that assembled 32 glyphs and
 * missed the checksum by one digit is a far more useful thing to show the user
 * than one that found four smudges.
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
}

/**
 * Did the segmenter drop characters out of the middle of a number?
 *
 * This is the one failure the rest of the pipeline cannot see. The ABA checksum
 * only covers the routing number, and confidence says nothing at all -- the
 * model is asked about the crops it is given and answers those correctly, so a
 * read missing two account digits came back at 0.93 minimum confidence, higher
 * than two reads that were right.
 *
 * Fixed pitch supplies the missing evidence. Adjacent characters are one cell
 * apart; a blank cell between fields makes two. A gap between two *digits* is
 * neither -- it is a hole where characters used to be. Measured across the real
 * cheques: every correct read stepped 1, with 2s only ever beside a `T` or `O`
 * field delimiter, while the one read that passed the checksum with a wrong
 * account number stepped 3 between two digits, exactly where two digits had
 * been lost.
 *
 * Rejecting here costs a retry. Not rejecting pays the wrong account.
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

  for (const box of boxes) {
    const input = cropGlyph(band, box, config);
    // runSync takes and returns raw ArrayBuffers. The model wants
    // [1, 48, 32, 1] NHWC float32 in 0..1; normalisation is a layer inside the
    // model, so the crop goes straight across with no mean/std applied here.
    const [raw] = model.runSync([input.buffer as ArrayBuffer]);
    const { index, p } = softmaxPeak(new Float32Array(raw));
    classes.push(classAt(index));
    confidences.push(p);
  }

  return {
    raw: toSymbols(classes),
    confidences,
    glyphCount: boxes.length,
    steps: cellSteps(boxes),
  };
}

function finish(
  classification: BandClassification,
  reading: BandReading,
  quarterTurns: number,
  mirrored: boolean,
  attempts: number,
  candidates: string[],
): Recognition {
  let parsed = parseMicr(classification.raw);
  if (parsed.ok && hasMissingDigits(classification.raw, classification.steps)) {
    parsed = {
      ok: false,
      raw: classification.raw,
      error: 'a character is missing from the middle of a number',
    };
  }
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
 * Confidence below which a read is worth flagging even though it passed the
 * checksum.
 *
 * A random misread has roughly a 1-in-10 chance of passing the ABA check by
 * luck. Requiring two transit symbols in plausible positions narrows that a
 * long way further, but not to zero, so a read carrying a genuinely unsure
 * glyph is shown with a warning rather than presented as settled.
 */
export const LOW_CONFIDENCE = 0.75;

export const EXPECTED_INPUT = { width: INPUT_WIDTH, height: INPUT_HEIGHT };
