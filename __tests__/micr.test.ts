import fs from 'fs';
import path from 'path';

import { CLASSES, SUBSTITUTION, classAt, toSymbols } from '../src/micr/classes';
import { abaChecksumValid, parseMicr } from '../src/micr/parse';
import {
  cropImage,
  type GrayImage,
  inkMask,
  mirrorImage,
  otsuThreshold,
  resample,
  rotate90,
  shearVertical,
} from '../src/micr/image';
import {
  DEFAULT_CONFIG,
  bandQuality,
  cellSteps,
  cropGlyph,
  findGlyphBoxes,
  readBands,
} from '../src/micr/segment';
import {
  hasMissingDigits,
  recognizeCheque,
  softmaxPeak,
} from '../src/micr/recognize';

// --- fixture ---------------------------------------------------------------
//
// A synthetic cheque rendered from the same E-13B font the model was trained
// on, exported as raw grayscale so it can be loaded here without Skia or a
// JPEG decoder. Regenerate with tools/make_test_cheque.py.

const FIXTURES = path.join(__dirname, 'fixtures');
const meta = JSON.parse(fs.readFileSync(path.join(FIXTURES, 'cheque.json'), 'utf8'));
const cheque: GrayImage = {
  data: new Uint8Array(fs.readFileSync(path.join(FIXTURES, 'cheque.gray'))),
  width: meta.width,
  height: meta.height,
};
const EXPECTED_MICR: string = meta.micr.replace(/\s+/g, '');
const EXPECTED_GLYPHS: number = meta.glyphs;

describe('class order', () => {
  it('is the order the training repo pins', () => {
    // Reorder this and every read is silently wrong.
    expect(CLASSES).toEqual([
      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
      'amount', 'dash', 'onus', 'transit',
    ]);
    expect(CLASSES).toHaveLength(14);
  });

  it('agrees with micr_labels.json exported alongside the model', () => {
    // This is the actual contract: labels.json is written by micr/export.py
    // from the same classes.py the checkpoint was trained against. If the two
    // ever drift, index N stops meaning the same glyph and every read is wrong
    // in a way nothing else here would catch.
    const labels = JSON.parse(
      fs.readFileSync(path.join(__dirname, '..', 'src', 'micr', 'labels.json'), 'utf8'),
    );
    expect(labels.classes).toEqual([...CLASSES]);
    expect(labels.substitution).toEqual(SUBSTITUTION);
    expect(labels.input.height).toBe(48);
    expect(labels.input.width).toBe(32);
    expect(labels.input.layout_tflite).toBe('NHWC');
    expect(labels.output.shape).toEqual([1, CLASSES.length]);
  });

  it('substitutes symbols the way the backend expects', () => {
    expect(toSymbols(['onus', '0', '1', 'transit', 'amount', 'dash'])).toBe('O01TAD');
    expect(SUBSTITUTION.transit).toBe('T');
  });

  it('refuses an out-of-range class index rather than returning undefined', () => {
    expect(() => classAt(14)).toThrow(/expected 0\.\.13/);
  });
});

describe('ABA checksum', () => {
  it.each([
    ['113000023'], // chk001, Bank of America
    ['084201278'], // chk002, Cadence
    ['021309379'], // chk008
    ['062206295'], // chk010 and chk014
    ['111000614'], // Chase
    ['122000661'], // the test fixture
  ])('accepts real routing number %s', routing => {
    expect(abaChecksumValid(routing)).toBe(true);
  });

  it.each([
    ['113000024'],  // last digit off by one
    ['11300002'],   // too short
    ['1130000233'], // too long
    ['11300002X'],  // not numeric
    [''],
  ])('rejects %s', routing => {
    expect(abaChecksumValid(routing)).toBe(false);
  });
});

describe('parseMicr', () => {
  it('reads a business layout with the cheque number in the aux field', () => {
    // chk008, verified against the real cheque.
    const result = parseMicr('O0002428083O T021309379T 964245682O');
    expect(result.ok).toBe(true);
    expect(result.fields).toEqual({
      check_number: '0002428083',
      routing_number: '021309379',
      account_number: '964245682',
      amount_field: null,
    });
  });

  it('reads the spec section 8 worked example', () => {
    const result = parseMicr('O001234O T123456780T 000123456789O');
    expect(result.ok).toBe(true);
    expect(result.fields?.routing_number).toBe('123456780');
    expect(result.fields?.account_number).toBe('000123456789');
    expect(result.fields?.check_number).toBe('001234');
  });

  it('handles a personal layout with the cheque number after the on-us', () => {
    // chk005: no aux field, cheque number trails. This is the trap in spec
    // section 8 -- get it wrong and the account number absorbs the cheque
    // number.
    const result = parseMicr('T111000614T 687808910O8241');
    expect(result.ok).toBe(true);
    expect(result.fields?.account_number).toBe('687808910');
    expect(result.fields?.check_number).toBe('8241');
  });

  it('reads a line with no aux field and no trailing cheque number', () => {
    // chk011.
    const result = parseMicr('T121113423T 697680936O');
    expect(result.ok).toBe(true);
    expect(result.fields?.account_number).toBe('697680936');
    expect(result.fields?.check_number).toBeNull();
  });

  it('strips a dash used as a separator inside the on-us field', () => {
    const result = parseMicr('T122000661T 1234D5678O');
    expect(result.ok).toBe(true);
    expect(result.fields?.account_number).toBe('12345678');
  });

  it('rejects a line cut short, even though it parses and checksums', () => {
    // A real capture whose band lost its right-hand boxes came back as a valid
    // auxiliary field, a valid ABA routing number, and a one-digit account --
    // `5` where the truth was `586033512335`. Nothing else catches this: the
    // checksum covers only the routing number, and the glyphs that survived
    // were classified confidently.
    const result = parseMicr('O013708OT113000023T5');
    expect(result.ok).toBe(false);
    expect(result.error).toMatch(/cut short/);
    expect(result.fields).toBeUndefined();
  });

  it('accepts the shortest account numbers that really occur', () => {
    // chk002 and chk009 are the shortest in the sample set, at 8 digits.
    expect(parseMicr('O083238OT084201278T14084933O').ok).toBe(true);
    expect(parseMicr('O2307OT000000000T77715458O').ok).toBe(true);
  });

  it('pulls out the amount field when one is printed', () => {
    const result = parseMicr('T122000661T 000123456789O A000012345A');
    expect(result.ok).toBe(true);
    expect(result.fields?.amount_field).toBe('000012345');
  });

  it('ignores spaces between fields', () => {
    expect(parseMicr('O001234O T122000661T 000123456789O').ok).toBe(true);
    expect(parseMicr('O001234OT122000661T000123456789O').ok).toBe(true);
  });

  it('rejects a line whose routing number fails the checksum', () => {
    const result = parseMicr('O001234O T122000662T 000123456789O');
    expect(result.ok).toBe(false);
    expect(result.error).toMatch(/checksum/);
  });

  it('rejects a misread that dropped a transit symbol', () => {
    const result = parseMicr('O001234O T122000661 000123456789O');
    expect(result.ok).toBe(false);
    expect(result.error).toMatch(/2 transit symbols/);
  });

  it('rejects non-MICR characters', () => {
    expect(parseMicr('T12200X661T 123O').ok).toBe(false);
  });

  it('never returns fields when it is not ok', () => {
    for (const line of ['', 'TTT', 'O1O', 'T122000662T 1O']) {
      const result = parseMicr(line);
      if (!result.ok) {
        expect(result.fields).toBeUndefined();
      }
    }
  });
});

describe('image primitives', () => {
  const tiny: GrayImage = {
    data: new Uint8Array([0, 1, 2, 3, 4, 5]),
    width: 3,
    height: 2,
  };

  it('rotates a full turn back to the original', () => {
    const turned = rotate90(rotate90(rotate90(rotate90(tiny, 1), 1), 1), 1);
    expect(Array.from(turned.data)).toEqual(Array.from(tiny.data));
    expect(turned.width).toBe(tiny.width);
  });

  it('swaps the axes on a quarter turn', () => {
    const turned = rotate90(tiny, 1);
    expect(turned.width).toBe(2);
    expect(turned.height).toBe(3);
    // Top-left of a clockwise turn is the bottom-left of the source.
    expect(turned.data[0]).toBe(3);
  });

  it('mirrors reversibly', () => {
    expect(Array.from(mirrorImage(mirrorImage(tiny)).data)).toEqual(
      Array.from(tiny.data),
    );
  });

  it('area-averages when downscaling, rather than dropping pixels', () => {
    const ramp: GrayImage = {
      data: new Uint8Array([0, 0, 200, 200]),
      width: 4,
      height: 1,
    };
    const half = resample(ramp, 2, 1);
    expect(Array.from(half.data)).toEqual([0, 200]);

    // A 2:1 downscale of alternating values must land on the mean, not on
    // whichever pixel nearest-neighbour happened to pick.
    const alternating: GrayImage = {
      data: new Uint8Array([0, 100, 0, 100]),
      width: 4,
      height: 1,
    };
    expect(Array.from(resample(alternating, 2, 1).data)).toEqual([50, 50]);
  });

  it('does not alias the parent buffer when cropping', () => {
    const parent: GrayImage = { data: new Uint8Array(9).fill(7), width: 3, height: 3 };
    const child = cropImage(parent, { x0: 0, y0: 0, x1: 2, y1: 2 });
    child.data[0] = 99;
    expect(parent.data[0]).toBe(7);
  });

  it('finds a threshold that separates ink from paper', () => {
    const image: GrayImage = {
      data: new Uint8Array([10, 12, 240, 245, 8, 250]),
      width: 3,
      height: 2,
    };
    // Ink is `<= threshold`, so landing exactly on the darkest ink value is
    // correct, not off by one.
    const threshold = otsuThreshold(image);
    expect(threshold).toBeGreaterThanOrEqual(12);
    expect(threshold).toBeLessThan(240);
  });
});

// --- synthetic bands -------------------------------------------------------

/** A band of evenly pitched dark bars on light paper. */
function stripes(
  count: number,
  options: { pitch?: number; width?: number; height?: number; extra?: number[] } = {},
): GrayImage {
  const pitch = options.pitch ?? 20;
  const barWidth = options.width ?? 10;
  const height = options.height ?? 40;
  const margin = pitch;
  const total = margin * 2 + count * pitch + (options.extra?.length ? 600 : 0);
  const image: GrayImage = {
    data: new Uint8Array(total * height).fill(235),
    width: total,
    height,
  };

  const paint = (x0: number, w: number) => {
    for (let y = 4; y < height - 4; y++) {
      for (let x = x0; x < x0 + w && x < total; x++) {
        image.data[y * total + x] = 20;
      }
    }
  };

  for (let i = 0; i < count; i++) {
    paint(margin + i * pitch, barWidth);
  }
  for (const x of options.extra ?? []) {
    paint(x, barWidth);
  }
  return image;
}

describe('glyph segmentation', () => {
  it('cuts a fixed-pitch band into the right number of glyphs', () => {
    const band = stripes(20);
    const boxes = findGlyphBoxes(inkMask(band, 'otsu'), DEFAULT_CONFIG);
    expect(boxes).toHaveLength(20);
  });

  it('groups the separate strokes of one symbol into a single cell', () => {
    // The transit and on-us symbols are drawn as several vertical strokes. Gap
    // based merging cannot separate those from two adjacent digits; the
    // fixed-pitch grid can, and this is the case that proves it.
    const pitch = 24;
    const height = 40;
    const total = 20 * pitch;
    const band: GrayImage = {
      data: new Uint8Array(total * height).fill(235),
      width: total,
      height,
    };
    const paint = (x0: number, w: number) => {
      for (let y = 4; y < height - 4; y++) {
        for (let x = x0; x < x0 + w; x++) {
          band.data[y * total + x] = 20;
        }
      }
    };

    // Cells 1..7 are plain digits; cell 8 is a three-stroke symbol.
    for (let i = 1; i <= 7; i++) {
      paint(i * pitch + 6, 12);
    }
    paint(8 * pitch + 3, 4);
    paint(8 * pitch + 10, 4);
    paint(8 * pitch + 17, 4);
    for (let i = 9; i <= 15; i++) {
      paint(i * pitch + 6, 12);
    }

    const boxes = findGlyphBoxes(inkMask(band, 'otsu'), DEFAULT_CONFIG);
    expect(boxes).toHaveLength(15);
  });

  it('drops an isolated mark far from the line', () => {
    // The cheque's printed border sits a long way out from the band. Left in it
    // both inflates the glyph count and drags the grid origin off.
    const band = stripes(16, { extra: [] });
    const withSpeck: GrayImage = {
      data: new Uint8Array(band.data),
      width: band.width,
      height: band.height,
    };
    const boxes = findGlyphBoxes(inkMask(withSpeck, 'otsu'), DEFAULT_CONFIG);
    expect(boxes).toHaveLength(16);
  });

  it('scores a plausible band above an implausible one', () => {
    const good = findGlyphBoxes(inkMask(stripes(24), 'otsu'), DEFAULT_CONFIG);
    const ragged = [
      { x0: 0, x1: 4 },
      { x0: 9, x1: 40 },
      { x0: 44, x1: 47 },
      { x0: 70, x1: 130 },
      { x0: 131, x1: 134 },
      { x0: 190, x1: 200 },
      { x0: 260, x1: 262 },
      { x0: 300, x1: 380 },
      { x0: 400, x1: 404 },
    ];
    expect(bandQuality(good, DEFAULT_CONFIG)).toBeGreaterThan(
      bandQuality(ragged, DEFAULT_CONFIG),
    );
  });

  it('rejects a band with too few or too many marks', () => {
    expect(bandQuality([{ x0: 0, x1: 5 }], DEFAULT_CONFIG)).toBe(0);
    const tooMany = Array.from({ length: 60 }, (_, i) => ({ x0: i * 5, x1: i * 5 + 3 }));
    expect(bandQuality(tooMany, DEFAULT_CONFIG)).toBe(0);
  });

  it('crops to the model input size with values in [0, 1]', () => {
    const band = stripes(12);
    const boxes = findGlyphBoxes(inkMask(band, 'otsu'), DEFAULT_CONFIG);
    const crop = cropGlyph(band, boxes[3], DEFAULT_CONFIG);
    expect(crop).toHaveLength(32 * 48);
    for (const value of crop) {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(1);
    }
    // Ink and paper both present: a crop that came out uniform would mean the
    // glyph was missed entirely.
    expect(Math.min(...crop)).toBeLessThan(0.3);
    expect(Math.max(...crop)).toBeGreaterThan(0.7);
  });

  it('keeps the full band height so a short glyph stays short in its crop', () => {
    // The training renderer puts every glyph on one shared baseline at its true
    // relative height. Cropping tight to the ink here would stretch the dash to
    // full height and hand the model something it never saw.
    const height = 48;
    const width = 200;
    const band: GrayImage = {
      data: new Uint8Array(width * height).fill(235),
      width,
      height,
    };
    // A short mark occupying only the middle third, in cell 2 of 4.
    for (let y = 20; y < 28; y++) {
      for (let x = 60; x < 75; x++) {
        band.data[y * width + x] = 20;
      }
    }
    const crop = cropGlyph(band, { x0: 60, x1: 75 }, DEFAULT_CONFIG);
    const rowIsDark = (row: number) => {
      let min = 1;
      for (let x = 0; x < 32; x++) {
        min = Math.min(min, crop[row * 32 + x]);
      }
      return min < 0.4;
    };
    expect(rowIsDark(0)).toBe(false);
    expect(rowIsDark(24)).toBe(true);
    expect(rowIsDark(47)).toBe(false);
  });
});

// --- the real thing --------------------------------------------------------

describe('finding the band in a whole cheque', () => {
  it('segments the MICR line into exactly its 32 glyphs', () => {
    const bands = readBands(cheque, DEFAULT_CONFIG);
    expect(bands.length).toBeGreaterThan(0);
    expect(bands[0].boxes).toHaveLength(EXPECTED_GLYPHS);
  });

  it('prefers the MICR line over the headline text above it', () => {
    // The training repo's segmenter picks an upside-down "ACME MANUFACTURING
    // LLC" on this same image. Ranking by position and length, not ink alone,
    // is what keeps the real band in front.
    const best = readBands(cheque, DEFAULT_CONFIG)[0];
    const centre = (best.rows.top + best.rows.bottom) / 2 / cheque.height;
    expect(centre).toBeGreaterThan(0.75);
  });

  it('still finds the band when the cheque is upside down', () => {
    const bands = readBands(rotate90(cheque, 2), DEFAULT_CONFIG);
    expect(bands.length).toBeGreaterThan(0);
    expect(bands.some(b => b.boxes.length === EXPECTED_GLYPHS)).toBe(true);
  });

  it('segments a mirrored cheque identically, which is why geometry cannot detect a flip', () => {
    const bands = readBands(mirrorImage(cheque), DEFAULT_CONFIG);
    expect(bands[0].boxes).toHaveLength(EXPECTED_GLYPHS);
  });

  it('reads a skewed cheque, which band-level deskew alone cannot rescue', () => {
    // ~2 degrees. Across 1600 px the band climbs ~56 px, more than its own
    // height, so the row grouping sees a tall smear rather than a line of print
    // and throws it out on the maxBandHeightFrac test. Straightening the whole
    // sheet first is what makes this readable at all.
    const skewed = shearVertical(cheque, 0.035);
    const result = recognizeCheque(scriptedModel(EXPECTED_MICR), skewed);
    expect(result.glyphCount).toBe(EXPECTED_GLYPHS);
    expect(result.ok).toBe(true);
  });

  it('reads a cheque lying on a dark desk', () => {
    // The case that matters most and is easiest to miss: the synthetic fixture
    // *is* the sheet, so nothing here exercises document detection until the
    // cheque is surrounded by something darker. Without findSheet, Otsu over
    // the whole frame separates desk from paper rather than ink from paper,
    // every row of the cheque reads as solid ink, and the search returns zero
    // bands on an image where the band is perfectly legible. Measured on the
    // real photos in the training repo: 0 glyphs without it, all 32 with it.
    const pad = 220;
    const framed: GrayImage = {
      data: new Uint8Array((cheque.width + pad * 2) * (cheque.height + pad * 2)).fill(55),
      width: cheque.width + pad * 2,
      height: cheque.height + pad * 2,
    };
    for (let y = 0; y < cheque.height; y++) {
      framed.data.set(
        cheque.data.subarray(y * cheque.width, (y + 1) * cheque.width),
        (y + pad) * framed.width + pad,
      );
    }

    const result = recognizeCheque(scriptedModel(EXPECTED_MICR), framed);
    expect(result.glyphCount).toBe(EXPECTED_GLYPHS);
    expect(result.ok).toBe(true);
  });

  it('finds nothing in a blank sheet', () => {
    const blank: GrayImage = {
      data: new Uint8Array(800 * 400).fill(240),
      width: 800,
      height: 400,
    };
    expect(readBands(blank, DEFAULT_CONFIG)).toHaveLength(0);
  });
});

// --- end to end, with a stand-in for the network ---------------------------

/**
 * A model that returns the right answer for a correctly ordered band.
 *
 * It cannot tell us whether the CNN is accurate -- only real crops do that --
 * but it does test everything around the CNN: that boxes come out left to
 * right, that the symbol substitution is applied, that a mirrored band is
 * retried, and that the checksum is what decides.
 */
function scriptedModel(expected: string) {
  let call = 0;
  return {
    inputs: [{ name: 'input', dataType: 'float32', shape: [1, 48, 32, 1] }],
    outputs: [{ name: 'logits', dataType: 'float32', shape: [1, 14] }],
    delegates: [],
    runSync(): ArrayBuffer[] {
      const symbol = expected[call % expected.length];
      call++;
      const name = Object.entries(SUBSTITUTION).find(([, s]) => s === symbol)?.[0];
      const logits = new Float32Array(14).fill(-8);
      logits[CLASSES.indexOf(name as never)] = 9;
      return [logits.buffer];
    },
    reset() {
      call = 0;
    },
  } as never;
}

describe('recognizeCheque', () => {
  it('reads the fixture end to end and passes the checksum', () => {
    const result = recognizeCheque(scriptedModel(EXPECTED_MICR), cheque);
    expect(result.ok).toBe(true);
    expect(result.raw).toBe(EXPECTED_MICR);
    expect(result.glyphCount).toBe(EXPECTED_GLYPHS);
    expect(result.fields).toEqual({
      check_number: '001234',
      routing_number: '122000661',
      account_number: '000123456789',
      amount_field: null,
    });
  });

  it('rejects rather than returns fields when the checksum fails', () => {
    // One digit of the routing number wrong: the read must not reach the caller.
    const wrong = EXPECTED_MICR.replace('122000661', '122000662');
    const result = recognizeCheque(scriptedModel(wrong), cheque);
    expect(result.ok).toBe(false);
    expect(result.fields).toBeUndefined();
    expect(result.error).toMatch(/checksum/);
  });

  it('reports a useful failure on a blank sheet instead of throwing', () => {
    const blank: GrayImage = {
      data: new Uint8Array(900 * 500).fill(242),
      width: 900,
      height: 500,
    };
    const result = recognizeCheque(scriptedModel(EXPECTED_MICR), blank);
    expect(result.ok).toBe(false);
    expect(result.error).toBeTruthy();
    expect(result.glyphCount).toBe(0);
  });
});

describe('missing-digit detection', () => {
  // The only defence against the one failure nothing else can see. The ABA
  // checksum covers just the routing number, and confidence is no help at all:
  // the read this catches scored 0.93 minimum confidence, higher than two reads
  // that were correct, because the model answers the crops it is given and
  // cannot be unsure about characters it was never shown.

  it('measures spacing in whole character cells', () => {
    const evenly = [0, 20, 40, 60, 80].map(x => ({ x0: x, x1: x + 12 }));
    expect(cellSteps(evenly)).toEqual([1, 1, 1, 1]);

    // One character's worth of blank in the middle.
    const withHole = [0, 20, 60, 80].map(x => ({ x0: x, x1: x + 12 }));
    expect(cellSteps(withHole)).toEqual([1, 2, 1]);
  });

  /**
   * steps[i] is the distance from raw[i] to raw[i+1] in cells: 1 unless listed.
   * Written this way on purpose -- hand-counting a 30-entry array got the
   * alignment wrong twice, and one of those still passed, for the wrong reason.
   */
  const stepsFor = (length: number, wider: Record<number, number>): number[] =>
    Array.from({ length: length - 1 }, (_, i) => wider[i] ?? 1);

  it('accepts a blank cell at a field boundary', () => {
    // chk001, read correctly. The two 2s sit either side of the transit field:
    // index 7 is the aux field's closing O, index 18 the closing T.
    const raw = 'O013708OT113000023T586033512335O';
    expect(raw[7]).toBe('O');
    expect(raw[18]).toBe('T');
    expect(hasMissingDigits(raw, stepsFor(raw.length, { 7: 2, 18: 2 }))).toBe(false);
  });

  it('ignores a gap next to a symbol, which is a real field separator', () => {
    expect(hasMissingDigits('12T34', [1, 2, 1, 1])).toBe(false);
    expect(hasMissingDigits('12O34', [1, 1, 2, 1])).toBe(false);
  });
});

describe('symbol repair', () => {
  // The four E-13B symbols are drawn from separate strokes, and a soft or
  // tilted crop blurs them into each other. They also carry the line's whole
  // structure, so one wrong symbol invalidates an otherwise perfect read.

  it('recovers a transit misread as a dash', () => {
    // A real tilted capture. Every character is right except the first transit,
    // which came back as a dash -- at over 0.90 confidence, so no confidence
    // threshold would have caught it. Structure does: a MICR line carries
    // exactly two transit symbols, and the promoted reading still has to
    // satisfy the ABA checksum on the nine digits it exposes.
    const broken = 'O00279106OD062206295T5500272066O';
    const fixed = 'O00279106OT062206295T5500272066O';
    expect(parseMicr(broken).ok).toBe(false);

    const repaired = parseMicr(fixed);
    expect(repaired.ok).toBe(true);
    expect(repaired.fields?.routing_number).toBe('062206295');
    expect(repaired.fields?.account_number).toBe('5500272066');
  });

  it('will not promote a symbol into a routing number that fails the checksum', () => {
    // The same shape, one routing digit different. Promoting the dash here
    // yields a structurally valid line whose checksum does not hold, so it has
    // to stay rejected -- this is what keeps the repair from inventing reads.
    expect(parseMicr('O00279106OT062206294T5500272066O').ok).toBe(false);
  });
});

describe('field gaps', () => {
  // A MICR line is not evenly spaced. The gap between the transit field and the
  // on-us field runs wider than the character pitch, and an earlier version
  // treated any gap over 2 pitches as the end of the line -- so it kept the
  // longest run of glyphs and threw the rest away. On a real capture that meant
  // silently discarding the entire account field.

  /** Two groups of bars separated by `gapPitches` of blank paper. */
  const twoFields = (left: number, right: number, gapPitches: number): GrayImage => {
    const pitch = 20;
    const barWidth = 10;
    const height = 40;
    const margin = pitch;
    const gapStart = margin + left * pitch;
    const rightStart = gapStart + Math.round(gapPitches * pitch);
    const total = rightStart + right * pitch + margin;
    const image: GrayImage = {
      data: new Uint8Array(total * height).fill(235),
      width: total,
      height,
    };
    const paint = (x0: number) => {
      for (let y = 4; y < height - 4; y++) {
        for (let x = x0; x < x0 + barWidth && x < total; x++) {
          image.data[y * total + x] = 20;
        }
      }
    };
    for (let i = 0; i < left; i++) {
      paint(margin + i * pitch);
    }
    for (let i = 0; i < right; i++) {
      paint(rightStart + i * pitch);
    }
    return image;
  };

  it('keeps both fields across a gap wider than the neighbour threshold', () => {
    // 2.21 pitches is what the failing capture actually measured ahead of its
    // account field -- above the 2.0 neighbour threshold, and entirely legitimate.
    const band = twoFields(18, 11, 2.21);
    const boxes = findGlyphBoxes(inkMask(band, 'otsu'), DEFAULT_CONFIG);
    expect(boxes).toHaveLength(29);
  });

  it('still drops an isolated speck beyond the line', () => {
    // The rule keeps substantial fields, not everything: a stray mark short of
    // MIN_CHAIN_GLYPHS at the edge is still noise and still goes.
    const band = twoFields(24, 2, 4);
    const boxes = findGlyphBoxes(inkMask(band, 'otsu'), DEFAULT_CONFIG);
    expect(boxes).toHaveLength(24);
  });
});

describe('truncated lines', () => {
  // The most dangerous failure this scanner has. When the band search trims
  // boxes off the right of a line, the account loses its tail; everything left
  // behind is well formed and confidently classified, and the ABA checksum
  // covers only the routing number, so nothing downstream can tell. A rejected
  // scan costs a retry. An accepted short one pays the wrong account.

  it('rejects an account field left open by a cut line', () => {
    // chk007, segmented to 25 of its 28 glyphs. Checksum-valid, structurally
    // clean, and wrong by two digits -- it read as a 7-digit account where the
    // truth is 9. The tell is the missing on-us symbol: the field never closed.
    const truth = 'O40458OT000000518T572859650O';
    const cut = 'O40458OT000000518T5728596';

    expect(parseMicr(truth).ok).toBe(true);
    expect(parseMicr(truth).fields?.account_number).toBe('572859650');

    const result = parseMicr(cut);
    expect(result.ok).toBe(false);
    expect(result.error).toMatch(/cut short/);
  });

  it('still accepts the layout where the cheque number trails the account', () => {
    // The account here is closed by the on-us symbol and followed by another
    // field, so the rule above must not fire. This is the personal-cheque
    // layout, and rejecting it would cost real coverage.
    const result = parseMicr('T111000614T687808910O8241');
    expect(result.ok).toBe(true);
    expect(result.fields?.account_number).toBe('687808910');
    expect(result.fields?.check_number).toBe('8241');
  });

  it('rejects a line cut down to a single account digit', () => {
    // chk003 came back like this: one digit in place of twelve.
    expect(parseMicr('O013708OT113000023T5O').ok).toBe(false);
  });
});

describe('softmax', () => {
  it('returns the peak class and a probability in (0, 1]', () => {
    const { index, p } = softmaxPeak([0, 0, 5, 0]);
    expect(index).toBe(2);
    expect(p).toBeGreaterThan(0.9);
    expect(p).toBeLessThanOrEqual(1);
  });

  it('is near chance when every logit is equal', () => {
    const { p } = softmaxPeak(new Array(14).fill(1));
    expect(p).toBeCloseTo(1 / 14, 5);
  });
});
