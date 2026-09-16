import { abaChecksumValid, parseMicr } from '../src/micr/parse';
import { CLASSES, SUBSTITUTION, toSymbols } from '../src/micr/classes';
import {
  findGlyphBoxes,
  otsuThreshold,
  bandQuality,
  cropRows,
  locateBandRows,
  findMicrBand,
  mirrorImage,
  findDocumentRect,
  cropGlyph,
  GrayImage,
} from '../src/micr/segment';

describe('class order', () => {
  it('matches micr_labels.json shipped with the model', () => {
    // Reorder this and every read is silently wrong.
    expect(CLASSES).toEqual([
      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
      'amount', 'dash', 'onus', 'transit',
    ]);
    expect(CLASSES).toHaveLength(14);
  });

  it('substitutes symbols the way the backend expects', () => {
    expect(toSymbols(['onus', '0', '1', 'transit', 'amount', 'dash'])).toBe('O01TAD');
    expect(SUBSTITUTION.transit).toBe('T');
  });
});

describe('ABA checksum', () => {
  it.each([
    ['113000023', true],  // chk001, Bank of America
    ['084201278', true],  // chk002, Cadence
    ['021309379', true],  // chk008
    ['062206295', true],  // chk010 and chk014
    ['111000614', true],  // Chase
    ['111000025', true],  // the ACH number printed on chk001, also valid
  ])('accepts real routing number %s', (routing, expected) => {
    expect(abaChecksumValid(routing)).toBe(expected);
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
  it('reads a business layout with the check number in the aux field', () => {
    // chk008, verified against the real check.
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

  it('handles a personal layout with the check number after the on-us', () => {
    // chk005: no aux field, check number trails. This is the trap in spec
    // section 8 -- get it wrong and the account number absorbs the check number.
    const result = parseMicr('T111000614T 687808910O8241');
    expect(result.ok).toBe(true);
    expect(result.fields?.account_number).toBe('687808910');
    expect(result.fields?.check_number).toBe('8241');
  });

  it('ignores spaces between fields', () => {
    expect(parseMicr('O001234O   T123456780T   000123456789O').ok).toBe(true);
  });

  it('rejects a line whose routing number fails the checksum', () => {
    const result = parseMicr('O001234O T123456781T 000123456789O');
    expect(result.ok).toBe(false);
    expect(result.error).toMatch(/ABA checksum/);
  });

  it('rejects a misread that dropped a transit symbol', () => {
    const result = parseMicr('O001234O T123456780 000123456789O');
    expect(result.ok).toBe(false);
    expect(result.error).toMatch(/transit/);
  });

  it('rejects non-MICR characters', () => {
    expect(parseMicr('O12X4O T123456780T 999O').ok).toBe(false);
  });

  it('never returns fields when it is not ok', () => {
    for (const bad of ['', 'TTT', 'O1O', 'T123456781T 9O']) {
      const result = parseMicr(bad);
      if (!result.ok) {
        expect(result.fields).toBeUndefined();
      }
    }
  });
});

/** Build a synthetic band: `count` bars on a fixed pitch. */
function syntheticBand(count: number, pitch = 20, barWidth = 10): GrayImage {
  const width = pitch * (count + 1);
  const height = 48;
  const data = new Uint8Array(width * height).fill(240); // paper
  for (let i = 0; i < count; i++) {
    const x0 = pitch * (i + 0.5);
    for (let y = 6; y < height - 6; y++) {
      for (let x = x0; x < x0 + barWidth; x++) {
        data[y * width + Math.floor(x)] = 20; // ink
      }
    }
  }
  return { data, width, height };
}

describe('segmentation', () => {
  it('finds a threshold that separates ink from paper', () => {
    const band = syntheticBand(10);
    const t = otsuThreshold(band);
    // Otsu puts [0..t] in the dark class, and inkProjection tests `<= t`, so
    // ink at 20 must land on or above the threshold and paper at 240 above it.
    expect(t).toBeGreaterThanOrEqual(20);
    expect(t).toBeLessThan(240);
  });

  it('cuts a fixed-pitch band into the right number of glyphs', () => {
    for (const count of [8, 21, 28, 32]) {
      const band = syntheticBand(count);
      const boxes = findGlyphBoxes(band, otsuThreshold(band));
      expect(boxes).toHaveLength(count);
    }
  });

  it('groups the separate strokes of one symbol into a single cell', () => {
    // The on-us symbol is drawn as two bars plus a block inside one character
    // cell. Gap-based merging splits it; the grid must not.
    const pitch = 24;
    const width = pitch * 6;
    const height = 48;
    const data = new Uint8Array(width * height).fill(240);
    const ink = (x0: number, w: number) => {
      for (let y = 8; y < height - 8; y++) {
        for (let x = x0; x < x0 + w; x++) {
          data[y * width + x] = 20;
        }
      }
    };
    // Three plain digits, then one symbol made of three thin strokes.
    ink(pitch * 0 + 6, 12);
    ink(pitch * 1 + 6, 12);
    ink(pitch * 2 + 6, 12);
    ink(pitch * 3 + 5, 3);
    ink(pitch * 3 + 10, 3);
    ink(pitch * 3 + 15, 3);

    const band: GrayImage = { data, width, height };
    const boxes = findGlyphBoxes(band, otsuThreshold(band));
    expect(boxes).toHaveLength(4);
  });

  it('drops an isolated mark far from the line', () => {
    const band = syntheticBand(12);
    // A speck a long way to the right of the last glyph.
    const strayX = band.width - 4;
    for (let y = 20; y < 28; y++) {
      band.data[y * band.width + strayX] = 10;
    }
    const boxes = findGlyphBoxes(band, otsuThreshold(band));
    expect(boxes).toHaveLength(12);
  });

  it('scores a plausible band above an implausible one', () => {
    const good = syntheticBand(28);
    const goodBoxes = findGlyphBoxes(good, otsuThreshold(good));
    expect(bandQuality(goodBoxes)).toBeGreaterThan(0);
    expect(bandQuality([])).toBe(0);
    // 70 fragments is texture, not a MICR line.
    expect(bandQuality(Array.from({ length: 70 }, (_, i) => ({ x0: i, x1: i + 1 })))).toBe(0);
  });

  it('crops to the model input size with values in [0, 1]', () => {
    const band = syntheticBand(10);
    const boxes = findGlyphBoxes(band, otsuThreshold(band));
    const crop = cropGlyph(band, boxes[0]);
    expect(crop).toHaveLength(32 * 48);
    for (const v of crop) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
    // The crop must contain ink, otherwise the box is in the wrong place.
    expect(Math.min(...crop)).toBeLessThan(0.3);
  });
});

describe('band row localisation', () => {
  /** A band crop with the line in the middle and clutter above it. */
  function cluttered(): GrayImage {
    const width = 400;
    const height = 90;
    const data = new Uint8Array(width * height).fill(235);
    // A full-width dark rule near the top -- a signature line or cheque edge.
    for (let x = 0; x < width; x++) {
      data[8 * width + x] = 30;
      data[9 * width + x] = 30;
    }
    // The actual glyph row, further down.
    for (let i = 0; i < 12; i++) {
      const x0 = 20 + i * 30;
      for (let y = 40; y < 70; y++) {
        for (let x = x0; x < x0 + 14; x++) {
          data[y * width + x] = 25;
        }
      }
    }
    return { data, width, height };
  }

  it('finds the glyph rows and excludes the rule above them', () => {
    const img = cluttered();
    const rows = locateBandRows(img, otsuThreshold(img));
    expect(rows.top).toBeGreaterThan(12);
    expect(rows.bottom).toBeLessThanOrEqual(90);
    expect(rows.bottom - rows.top).toBeGreaterThan(20);
  });

  it('segments the cluttered crop that used to yield nothing', () => {
    const img = cluttered();
    const rows = locateBandRows(img, otsuThreshold(img));
    const band = cropRows(img, rows.top, rows.bottom);
    expect(findGlyphBoxes(band, otsuThreshold(band))).toHaveLength(12);
  });

  it('keeps edge-touching runs rather than returning nothing', () => {
    // One blob spanning the full width: dropping it left zero boxes before.
    const width = 200;
    const height = 40;
    const data = new Uint8Array(width * height).fill(20);
    const boxes = findGlyphBoxes({ data, width, height }, 128);
    expect(boxes.length).toBeGreaterThanOrEqual(1);
  });
});

describe('finding the band in a whole cheque', () => {
  /**
   * A cheque-like image: printed text near the top, a long handwritten-ish
   * scrawl in the middle, and a fixed-pitch MICR line near the bottom.
   */
  function cheque(micrGlyphs = 32): GrayImage {
    const width = 960;
    const height = 436; // 2.2:1
    const data = new Uint8Array(width * height).fill(232);

    const block = (x0: number, y0: number, w: number, h: number, v = 40) => {
      for (let y = y0; y < y0 + h; y++) {
        for (let x = x0; x < x0 + w; x++) {
          data[y * width + x] = v;
        }
      }
    };

    // Payee / bank lines: irregular word-like blobs.
    for (let i = 0; i < 6; i++) {
      block(80 + i * 70, 60, 34 + (i % 3) * 18, 14);
    }
    for (let i = 0; i < 4; i++) {
      block(120 + i * 130, 150, 90 + (i % 2) * 40, 20);
    }
    // A long signature stroke -- lots of ink, no fixed pitch.
    for (let x = 520; x < 900; x++) {
      const y = 250 + Math.round(18 * Math.sin(x / 26));
      block(x, y, 2, 4, 20);
    }

    // The MICR line: constant pitch near the bottom.
    const pitch = 24;
    const startX = 90;
    for (let i = 0; i < micrGlyphs; i++) {
      block(startX + i * pitch, 370, 13, 30, 25);
    }
    return { data, width, height };
  }

  it('locates the MICR line rather than the signature or the text', () => {
    const img = cheque(32);
    const found = findMicrBand(img);
    expect(found).not.toBeNull();
    expect(found!.boxes).toHaveLength(32);
    // It must have chosen the strip near the bottom, not the lines above.
    expect(found!.rows.top).toBeGreaterThan(300);
  });

  it('segments a mirrored frame just as well, which is why geometry cannot detect the flip', () => {
    // A mirrored band has the same glyph count, pitch and widths as an upright
    // one, so the search cannot tell them apart and must not pretend to. Only
    // classifying the glyphs and checking the ABA digit distinguishes them,
    // which is what recognizeDocument retries on.
    const found = findMicrBand(mirrorImage(cheque(32)));
    expect(found).not.toBeNull();
    expect(found!.boxes).toHaveLength(32);
    expect(found!.mirrored).toBe(false);
  });

  it('returns null when there is no MICR line in view', () => {
    const width = 400;
    const height = 300;
    const data = new Uint8Array(width * height).fill(230);
    // Scattered blobs, nothing on a regular pitch.
    for (let i = 0; i < 5; i++) {
      for (let y = 40 + i * 40; y < 55 + i * 40; y++) {
        for (let x = 30 + i * 61; x < 30 + i * 61 + 25; x++) {
          data[y * width + x] = 30;
        }
      }
    }
    expect(findMicrBand({ data, width, height })).toBeNull();
  });
});

describe('cheque photographed on a dark surface', () => {
  /**
   * The situation from the device: a bright cheque occupying the middle of a
   * frame, surrounded by a dark desk. Otsu over the whole frame splits desk
   * from paper, so without cropping to the sheet first every dark background
   * row reads as solid ink and the band search finds nothing.
   */
  function chequeOnDesk(): GrayImage {
    const width = 960;
    const height = 720;
    const data = new Uint8Array(width * height).fill(58); // dark desk

    const px = 70;
    const py = 150;
    const pw = 820;
    const ph = 420;
    for (let y = py; y < py + ph; y++) {
      data.fill(230, y * width + px, y * width + px + pw); // paper
    }

    const block = (x0: number, y0: number, w: number, h: number) => {
      for (let y = y0; y < y0 + h; y++) {
        data.fill(35, y * width + x0, y * width + x0 + w);
      }
    };
    // Printed lines on the cheque.
    for (let i = 0; i < 5; i++) {
      block(140 + i * 90, 210, 46 + (i % 3) * 20, 15);
    }
    for (let i = 0; i < 4; i++) {
      block(160 + i * 140, 330, 96, 18);
    }
    // The MICR line, fixed pitch, near the bottom of the paper.
    for (let i = 0; i < 32; i++) {
      block(130 + i * 23, 500, 12, 28);
    }
    return { data, width, height };
  }

  it('crops to the sheet and excludes the desk', () => {
    const rect = findDocumentRect(chequeOnDesk());
    expect(rect.x0).toBeGreaterThan(40);
    expect(rect.x1).toBeLessThan(940);
    expect(rect.y0).toBeGreaterThan(120);
    expect(rect.y1).toBeLessThan(600);
  });

  it('finds the MICR line despite the dark background', () => {
    const found = findMicrBand(chequeOnDesk());
    expect(found).not.toBeNull();
    expect(found!.boxes).toHaveLength(32);
  });

  it('falls back to the whole frame when there is no distinct sheet', () => {
    // Uniform image: nothing to crop to, so it must not crop to a sliver.
    const width = 300;
    const height = 200;
    const rect = findDocumentRect({
      data: new Uint8Array(width * height).fill(200),
      width,
      height,
    });
    expect(rect).toEqual({ x0: 0, y0: 0, x1: width, y1: height });
  });
});
