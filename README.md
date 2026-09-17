# MICR cheque scanner — React Native

Reads the MICR (E-13B) line from the bottom of a cheque using the phone camera
and returns the routing number, account number and cheque number. Everything
runs on the device: no network call, no per-scan cost.

The model is trained separately in
[micr-training](https://github.com/MelodyStack/micr-training); this repo is the
app that ships it.

## The pipeline

```
tap the shutter
  → takePhoto()                    full sensor resolution, ~4032x3024
     ↳ takeSnapshot() on failure   preview-sized, for virtual cameras
  → Skia: decode + scale           native; emits 1800 / 1000 / 700 px levels
  → luminance                      ┐
  → crop to the sheet of paper     │
  → rank the 4 rotations           │
  → locate the band                │  plain TypeScript, unit-tested
  → Otsu / adaptive threshold      │
  → fixed-pitch grid segmentation  │
  → TFLite, one glyph at a time    │  ← the only ML step
  → assemble the T/A/O/D string    │
  → ABA checksum                   ┘
  → show the fields
```

**There are no worklets in this app.** Capture is a promise, decoding is Skia,
and recognition is ordinary code on the JS thread. That is a deliberate
reversal of the earlier frame-processor design — see below.

**The checksum is a hard gate.** A read that fails it is never shown and never
sent anywhere. A read that passes but contains a glyph the model was unsure
about is shown with a warning, because a random misread still has roughly a
1-in-10 chance of passing the checksum by luck.

## Why tap-to-capture and not a live frame processor

The app used to run a vision-camera frame processor over the video stream. Two
things were wrong with it, and only one of them was fixable by tuning.

**It could not start.** VisionCamera v4's frame processors require
`react-native-worklets-core`, which has no build for this React Native version.
`librnworklets.so` fails to resolve the libc++ symbol
`__cxa_init_primary_exception`, and the failure lands inside
`CameraViewModule`'s static initialiser — during TurboModule registration,
before any JavaScript runs:

```
FATAL EXCEPTION: pool-2-thread-1
java.lang.UnsatisfiedLinkError: dlopen failed: cannot locate symbol
  "__cxa_init_primary_exception" referenced by ".../lib/x86_64/librnworklets.so"
  at com.mrousavy.camera.react.CameraViewModule.<clinit>(CameraViewModule.kt:48)
```

`VisionCamera_enableFrameProcessors=false` in `android/gradle.properties` is
what fixes it: the camera still previews and still takes photos, but
`libVisionCamera.so` is no longer linked against worklets at all.

**And it was starved of pixels.** The frame processor worked at 960 px wide,
which is about 30 px per MICR glyph, then upscaled each crop to the model's
32×48 input. The model was trained on crops cut from full-resolution photos. A
still capture worked at 1800 px gives a 37 px pitch and an ~85 px band, so every
crop is a *downscale* to 32×48 — the same direction the training crops were
resampled in, which is the property that matters.

## Layout

| path | |
|---|---|
| `src/micr/image.ts` | grayscale primitives: rotate, mirror, crop, area resample, Otsu, adaptive threshold, deskew |
| `src/micr/segment.ts` | band search and the fixed-pitch grid — a port of `micr/segment.py` |
| `src/micr/recognize.ts` | rotation ranking, model invocation, mirror retry |
| `src/micr/parse.ts` | ABA checksum and field extraction |
| `src/micr/decode.ts` | the only module that touches a native library (Skia) |
| `src/screens/ScanScreen.tsx` | camera, permissions, capture, states |
| `assets/micr_cnn_v1_fp32.tflite` | the model, 486 KB |
| `tools/make_test_cheque.py` | regenerates the synthetic cheque the tests segment |

## Running it

```bash
npm install
npm run android
```

```bash
npm test               # 54 tests, no device needed
npx tsc --noEmit
```

### On the emulator

The emulator's camera renders a synthetic scene rather than a cheque, so the
capture path cannot be exercised there. Point the AVD's back camera at
`webcam0` in its advanced settings and hold a cheque up to the webcam, or build
to a real device.

The pipeline itself needs no camera: `npm test` runs the whole segmenter over
`__tests__/fixtures/cheque.gray`, a synthetic cheque rendered from the same
E-13B font the model was trained on, and asserts the fields parsed out of it.
That runs on every commit rather than needing someone to tap a button.

## Things that will bite you

**Crop to the sheet before thresholding anything.** A photo of a cheque is
mostly desk, and Otsu over a whole frame separates *desk from paper*, not *ink
from paper*. Every desk pixel then counts as ink, the cheque reads as solid
rows, and the band search returns nothing on a frame where the band is
perfectly legible. `findSheet()` is not a refinement — without it, the real
cheque photos in the training repo segment to **zero** glyphs; with it, chk001
segments to all 32. The synthetic test cheque hides this entirely, because the
image *is* the sheet. A regression test now frames it on a dark background.

**Some cameras cannot produce a photo CameraX will accept.** CameraX finishes
every capture by writing EXIF orientation back into the file, and
`ExifInterface` rejects anything it cannot parse as JPEG/PNG/WebP. Emulators
with a virtual camera — LDPlayer among them — produce exactly that, so the
capture succeeds, the bytes reach disk, and the whole thing is discarded over a
metadata write:

```
androidx.camera.core.ImageCaptureException: Failed to update Exif data
Caused by: java.io.IOException: ExifInterface only supports saving attributes
           on JPEG, PNG, or WebP formats.
```

`takeSnapshot()` compresses the preview view's bitmap itself and never touches
ExifInterface, so it is used as a fallback. It is preview-sized rather than
sensor-sized — a real loss of resolution, which is why it is second choice —
and the debug readout says `preview snapshot` when it is in play.

**Both MICR layouts exist.** Some cheques put the cheque number in a leading
auxiliary field; many personal cheques leave that empty and append it to the
on-us field *after* the account number. Parse by position, or account numbers
come out with the cheque number glued on. Both are covered by tests.

**Class order is a contract.** `src/micr/classes.ts` must match
`micr_labels.json` shipped beside the model. Reorder it and every read is
silently wrong — a test asserts it. `checkModelContract()` also verifies the
loaded `.tflite` on every launch: `[1, 48, 32, 1]` float32 in, 14 classes out.

**The model file says which build it is, on purpose.** The training repo's
`export/` holds float32, fp16 and int8 builds of the same weights. This app
bundled the int8 one under the plain `micr_cnn_v1.tflite` name, so copying the
fp32 export over it would have swapped models silently. It now ships
`micr_cnn_v1_fp32.tflite`: 100% argmax parity with PyTorch against int8's
99.67%, for 350 KB.

**Hermes has no optimising JIT.** It interprets bytecode, so a property load
left inside a per-pixel loop is a real cost, and the whole pipeline runs roughly
7× slower on device than the same code under Node. Three things mattered, in
order: hoisting the tap tables out of `resample`'s inner loops (a 1600 px
rescale went 790 ms → 31 ms), switching the adaptive threshold's integral image
from `Float64Array` to `Int32Array` (338 ms → 20 ms), and not running the
adaptive pass at all unless Otsu has already failed. Together with trimming the
working resolutions, a read went from 7.0 s to 4.5 s on an x86_64 emulator.
Measure before changing anything in `image.ts`; the obvious-looking version is
often 15× slower.

A read is a few seconds, not instant. That is the trade tap-to-capture makes:
one high-resolution read instead of many starved ones.

**Orientation is decided by trying, not by asking.** Sensor mounting, EXIF tags
and `isMirrored` all disagree across devices, and each mapping is a chance to be
wrong in a way that looks exactly like a camera seeing nothing. All four
rotations are scored by how well they parse as E-13B — which costs no model
calls — and the best one or two are classified. Mirroring is *not* decided that
way: a mirrored band segments identically to an upright one, so only the
checksum can tell them apart, and the flipped retry happens per candidate.

**Geometry alone cannot find the MICR line.** A block of clean sans headline
text segments into well-pitched, similar-width boxes and can outscore the real
band — the training repo's own segmenter picks an upside-down "ACME
MANUFACTURING LLC" on the synthetic test cheque. Candidates are therefore ranked
by position (the MICR line is the bottom-most print on a cheque) and length (19
to 40 characters), and several are classified until one passes the checksum.

**`android/build.gradle` lists Maven mirrors before `google()`.** `dl.google.com`
was unreachable from the machine this was built on. They are read-only mirrors
of the same artefacts and `google()` is still there as a fallback; delete the
block if your network reaches Google directly.

## Where this actually stands

Benchmarked by running the shipped `.tflite` over the crops this segmenter
produces, on all 14 real cheque photos in the training repo, taking the first
candidate that parses:

```
accepted 5/14,  correct 5/14,  wrong-but-accepted 0/14
```

Every read that got through was right. The other nine were rejected, which is
the behaviour to want: a rejected scan costs a retry, an accepted wrong one
pays the wrong account.

### The two things that got it there

**Rejoin characters the grid cut in half.** The on-us symbol `⑈` is two thin
bars and a block, and a cell boundary landing after the first bar left a 3 px
orphan beside a 13 px remainder, against a 17 px median. The remainder still
classified as on-us; the orphan became a phantom `D` on the front of the line.
This was the most common failure on real cheques, and it is what made the
symbol look untrained when every digit around it was correct. Fixed pitch makes
the repair safe: one character cannot span more than one cell, so two
neighbours that together still fit inside a pitch were never two characters.
See `mergeSplitGlyphs`.

**Reject a hole between two digits.** Merging the split symbols removed the
leading junk on a second cheque too -- and that cheque had *also* silently lost
two account digits, so it went straight from safely rejected to passing the
checksum with `5033512335` in place of `586033512335`. Nothing else in the
pipeline can see that: the ABA digit covers only the routing number, and
confidence is no help, because the model answers the crops it is handed and
cannot be unsure about characters it was never shown -- that read scored 0.93
minimum confidence, higher than two reads that were correct.

Fixed pitch supplies the missing evidence. Adjacent characters are one cell
apart and a field separator makes two, so a gap between two *digits* is a hole
where characters used to be. Across the real cheques every correct read stepped
1, with 2s only ever beside a `T` or `O`; the bad read stepped 3 between two
digits, exactly where the two were lost. See `hasMissingDigits`.

### Measured and rejected — do not retry these blind

* Growing each band over its faint top and bottom rows. Doubled the band height
  and broke a cheque that had been reading exactly right.
* Cropping the sheet at full resolution before downscaling. No change.
* Re-cutting the band so the ink fills the same fraction as in training. The
  premise was a measurement error -- band-level ink extent compared against
  per-crop ink extent -- and the crops were already within a couple of percent
  of the offline segmenter's. Applying it halved the benchmark.
* A minimum-confidence gate. Measured twice, at different stages: it costs
  correct reads (two scored 0.48 and 0.50) and catches nothing.
* Letting the parser skip leading junk. Produces exactly the false accept the
  gap check now blocks.

### What would move it further

The nine rejections are mostly creased, scribbled-over or badly lit photos.
More real cheques through the training repo's `micr.segment --debug-dir`,
tuning `SegmentConfig` against the debug images, and carrying the values back
into `DEFAULT_CONFIG` here -- the two configs are deliberately the same shape.
That is the 100-to-200-cheque exercise the spec asks for in section 12, and it
is a data exercise rather than a code one.

## Not in this MVP

Manual-entry fallback, submitting to a backend, and
iOS — the iOS project is configured but has not been run.
