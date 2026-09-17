"""Regenerate the synthetic cheque the unit tests segment.

    python tools/make_test_cheque.py

Writes __tests__/fixtures/cheque.gray -- raw 8-bit grayscale, because Jest has
no JPEG decoder and no Skia, so the tests read the bytes straight in -- plus
cheque.json describing it.

Deliberately synthetic. No customer data goes in this repo. The routing number
is a real, public ABA number (routing numbers identify banks, not people); the
account and cheque numbers are invented, and the rest is drawn here.

Needs a micr-training checkout beside this one, for the E-13B font and the
verified codepoint mapping. Adjust REPO if yours lives elsewhere.
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent.parent / "micr-training"
APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from micr.classes import E13B_CODEPOINTS  # noqa: E402

FONT = REPO / "fonts" / "E13B-OFL.ttf"
FIXTURE_DIR = APP / "__tests__" / "fixtures"

# ABA check on 122000661: 3+14+2+0+0+0+18+42+1 = 80, and 80 % 10 == 0 -> valid.
SUBSTITUTED = "O001234O T122000661T 000123456789O"
TRANSIT = chr(E13B_CODEPOINTS["transit"])
ONUS = chr(E13B_CODEPOINTS["onus"])
MICR_LINE = (
    SUBSTITUTED.replace("O", ONUS).replace("T", TRANSIT)
)

# A US cheque is 6.14 x 2.75 inches. Rendering at ~390 px/inch puts it in the
# same range a phone photo lands in after decode.ts downscales to WORK_MAX_SIDE.
DPI = 390
W, H = int(6.14 * DPI), int(2.75 * DPI)

# E-13B is 8 characters per inch, and E13B-OFL is monospaced with an advance of
# exactly the font size -- so the font size IS the pitch. Getting this wrong is
# what clipped the first attempt off the right-hand edge of the sheet.
PITCH = DPI // 8


def body(points):
    """A plain face for the non-MICR printing."""
    size = max(8, int(points * DPI / 72))
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def inch(value):
    return int(value * DPI)


sheet = Image.new("L", (W, H), 246)
draw = ImageDraw.Draw(sheet)

# Faint security tint, so a single global threshold is not trivially correct.
for y in range(0, H, max(2, DPI // 60)):
    draw.line([(0, y), (W, y)], fill=240)
draw.rectangle([2, 2, W - 3, H - 3], outline=170, width=2)

# Clutter above the band. The segmenter has to pick the MICR line out from among
# other lines of print, which is the whole point of scoring candidates rather
# than taking the strip with the most ink.
draw.text((inch(0.4), inch(0.22)), "ACME MANUFACTURING LLC", font=body(13), fill=40)
draw.text((inch(0.4), inch(0.45)), "1200 INDUSTRIAL PARKWAY", font=body(9), fill=90)
draw.text((inch(0.4), inch(0.62)), "SPRINGFIELD, IL 62701", font=body(9), fill=90)
draw.text((W - inch(1.5), inch(0.22)), "No. 001234", font=body(13), fill=40)
draw.text((W - inch(1.5), inch(0.48)), "DATE ____________", font=body(9), fill=90)

draw.text((inch(0.4), inch(1.02)), "PAY TO THE", font=body(8), fill=110)
draw.text((inch(0.4), inch(1.16)), "ORDER OF", font=body(8), fill=110)
draw.line([(inch(1.3), inch(1.3)), (inch(4.6), inch(1.3))], fill=60, width=2)
draw.text((inch(4.75), inch(1.12)), "$", font=body(12), fill=40)
draw.line([(inch(4.95), inch(1.3)), (W - inch(0.4), inch(1.3))], fill=60, width=2)

draw.line([(inch(0.4), inch(1.62)), (W - inch(0.4), inch(1.62))], fill=60, width=2)
draw.text((W - inch(0.95), inch(1.48)), "DOLLARS", font=body(8), fill=110)

draw.text((inch(0.4), inch(1.74)), "FIRST NATIONAL BANK", font=body(10), fill=60)
draw.text((inch(0.4), inch(1.90)), "SPRINGFIELD, IL", font=body(8), fill=110)

# These sit well clear of the band. On the first attempt the memo rule was 0.1"
# above it, so the band crop picked up a continuous horizontal line -- which
# inks every column and welds neighbouring glyphs into one run. The segmenter
# found 25 characters instead of 32. A real cheque leaves far more room, and the
# spacing here now reflects that.
draw.line([(inch(3.7), inch(2.02)), (W - inch(0.4), inch(2.02))], fill=60, width=2)
draw.text((inch(3.7), inch(2.06)), "AUTHORIZED SIGNATURE", font=body(7), fill=120)
draw.text((inch(0.4), inch(2.04)), "MEMO ____________", font=body(8), fill=110)

# The MICR band, drawn glyph by glyph on an exact pitch grid rather than as one
# string. That is how a real MICR printer lays it out, and it is what the
# segmenter's grid fit assumes. Spaces are simply empty cells.
micr_font = ImageFont.truetype(str(FONT), PITCH)
band_y = H - inch(0.37)
for index, glyph in enumerate(MICR_LINE):
    if glyph == " ":
        continue
    draw.text((inch(0.4) + index * PITCH, band_y), glyph, font=micr_font, fill=18)

# --- raw fixture for the Jest tests ----------------------------------------
#
# Jest has no Skia and no JPEG decoder, so the same sheet is exported as plain
# grayscale bytes. The TypeScript segmenter can then be tested against a real
# E-13B image rather than against hand-built stripes.
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
fixture = sheet if max(sheet.size) <= 1600 else sheet.resize(
    (1600, int(sheet.height * 1600 / sheet.width)), Image.LANCZOS
)
(FIXTURE_DIR / "cheque.gray").write_bytes(fixture.tobytes())
(FIXTURE_DIR / "cheque.json").write_text(
    json.dumps(
        {
            "width": fixture.width,
            "height": fixture.height,
            "micr": SUBSTITUTED,
            "glyphs": len(SUBSTITUTED.replace(" ", "")),
            "note": "Synthetic. Generated by scratchpad/make_test_cheque.py.",
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(f"wrote {FIXTURE_DIR / 'cheque.gray'}  ({fixture.width}x{fixture.height})")
print(f"substituted: {SUBSTITUTED}  ({len(SUBSTITUTED.replace(' ', ''))} glyphs)")
