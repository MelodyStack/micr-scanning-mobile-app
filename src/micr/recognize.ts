/**
 * Band pixels -> validated fields.
 *
 * The only ML step in the whole pipeline is classifying one 48x32 crop at a
 * time. Everything either side of it is ordinary code: cut the glyphs, join
 * the predictions, check the ABA digit. That is the point of the design in
 * spec section 3 -- the checksum is a hard correctness gate, so a bad read is
 * rejected rather than sent to the backend.
 */

import type { TensorflowModel } from 'react-native-fast-tflite';

import { classAt, INPUT_HEIGHT, INPUT_WIDTH, MicrClass, toSymbols } from './classes';
import { parseMicr, ParseResult } from './parse';
import {
  bandQuality,
  cropGlyph,
  cropRows,
  findGlyphBoxes,
  findMicrBand,
  GlyphBox,
  mirrorImage,
  GrayImage,
  locateBandRows,
  otsuThreshold,
} from './segment';

export interface Recognition extends ParseResult {
  glyphCount: number;
  /** Softmax confidence per glyph, in reading order. */
  confidences: number[];
  /** Lowest per-glyph confidence -- the weakest link in the read. */
  minConfidence: number;
  /** True if the band was only found after mirroring, i.e. a flipped sensor. */
  mirrored?: boolean;
  /** Where the band was found, for drawing it back over the preview. */
  bandRows?: { top: number; bottom: number };
}

function softmaxMax(logits: ArrayLike<number>): { index: number; p: number } {
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
  return { index: best, p: 1 / sum };
}

export const NO_BAND: Recognition = {
  ok: false,
  raw: '',
  error: 'no MICR line in the guide',
  glyphCount: 0,
  confidences: [],
  minConfidence: 0,
};

/**
 * Read one band image. Returns a failed result rather than throwing: this runs
 * per frame, and a frame that happens to catch a thumb should just be the next
 * frame's problem.
 */
/**
 * Read a whole cheque image: find the MICR line in it, then classify it.
 *
 * The caller hands over the entire frame, not a pre-aligned strip. Locating
 * the band is this function's job -- which is the point, because every
 * coordinate mapping between preview and sensor is one more thing to get
 * wrong, and getting it wrong looks identical to a camera that sees nothing.
 */
export function recognizeDocument(
  model: TensorflowModel,
  frame: GrayImage,
): Recognition {
  const found = findMicrBand(frame);
  if (!found) {
    return { ...NO_BAND, error: 'no MICR line found in view' };
  }

  const upright = classifyBand(model, found.band, found.boxes);
  if (upright.ok) {
    return { ...upright, bandRows: found.rows };
  }

  // Retry mirrored. A flipped sensor produces a band that segments perfectly
  // and classifies perfectly, then assembles backwards -- so it fails the
  // checksum with nothing to indicate why. Geometry cannot tell the two apart,
  // but the ABA digit can, so let it decide. Only the band is mirrored, not the
  // whole frame, and only after an upright read has already failed.
  const flipped = mirrorImage(found.band);
  const boxes = findGlyphBoxes(flipped, otsuThreshold(flipped));
  if (bandQuality(boxes) > 0) {
    const mirrored = classifyBand(model, flipped, boxes);
    if (mirrored.ok) {
      return { ...mirrored, mirrored: true, bandRows: found.rows };
    }
  }
  return { ...upright, bandRows: found.rows };
}

/** Read a strip already known to contain the band. */
export function recognizeBand(model: TensorflowModel, crop: GrayImage): Recognition {
  const coarse = otsuThreshold(crop);
  const rows = locateBandRows(crop, coarse);
  const band = cropRows(crop, rows.top, rows.bottom);
  const boxes = findGlyphBoxes(band, otsuThreshold(band));

  if (bandQuality(boxes) <= 0) {
    return { ...NO_BAND, glyphCount: boxes.length };
  }
  return classifyBand(model, band, boxes);
}

function classifyBand(
  model: TensorflowModel,
  band: GrayImage,
  boxes: GlyphBox[],
): Recognition {
  const classes: MicrClass[] = [];
  const confidences: number[] = [];

  for (const box of boxes) {
    const input = cropGlyph(band, box);
    // runSync takes and returns raw ArrayBuffers. The model wants
    // [1, 48, 32, 1] NHWC float32 in 0..1; normalisation is baked in, so the
    // crop goes straight across with no mean/std applied here.
    const [raw] = model.runSync([input.buffer as ArrayBuffer]);
    const { index, p } = softmaxMax(new Float32Array(raw));
    classes.push(classAt(index));
    confidences.push(p);
  }

  const raw = toSymbols(classes);
  const parsed = parseMicr(raw);
  const minConfidence = confidences.length ? Math.min(...confidences) : 0;

  return {
    ...parsed,
    raw,
    glyphCount: boxes.length,
    confidences,
    minConfidence,
  };
}

export const EXPECTED_INPUT = { width: INPUT_WIDTH, height: INPUT_HEIGHT };

/**
 * Multi-frame voting.
 *
 * A single frame that passes the checksum is already strong evidence -- a
 * random misread has a 1-in-10 chance of passing, so two independent frames
 * agreeing on the same digits is about as good as this gets without a second
 * sensor. Requiring agreement costs one extra frame and removes the 10% case.
 */
export class FrameVoter {
  private readonly counts = new Map<string, number>();
  private best: Recognition | null = null;

  constructor(private readonly required = 2) {}

  /** Returns the agreed reading once `required` frames concur, else null. */
  push(result: Recognition): Recognition | null {
    if (!result.ok || !result.fields) {
      return null;
    }
    const key = result.raw;
    const next = (this.counts.get(key) ?? 0) + 1;
    this.counts.set(key, next);

    if (!this.best || result.minConfidence > this.best.minConfidence) {
      this.best = result;
    }
    return next >= this.required ? (this.best?.raw === key ? this.best : result) : null;
  }

  reset(): void {
    this.counts.clear();
    this.best = null;
  }

  get agreement(): number {
    return Math.max(0, ...this.counts.values());
  }
}
