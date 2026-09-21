# MICR cheque scanner (React Native)

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
and recognition is ordinary code on the JS thread. This is a deliberate reversal
of the earlier frame-processor design (see below).

**The checksum is a hard gate.** A read that fails it is never shown and never
sent anywhere. A read that passes but contains a glyph the model was unsure
about is shown with a warning, because a random misread still has roughly a
1-in-10 chance of passing the checksum by luck.

## Why tap-to-capture and not a live frame processor

The app used to run a vision-camera frame processor over the video stream. Two
things were wrong with it, and only one was fixable by tuning.

**It could not start.** VisionCamera v4's frame processors require
`react-native-worklets-core`, which has no build for this React Native version.
`librnworklets.so` fails to resolve the libc++ symbol
`__cxa_init_primary_exception`, and the failure lands inside `CameraViewModule`'s
static initialiser, during TurboModule registration and before any JavaScript
runs:

```
FATAL EXCEPTION: pool-2-thread-1
java.lang.UnsatisfiedLinkError: dlopen failed: cannot locate symbol
  "__cxa_init_primary_exception" referenced by ".../lib/x86_64/librnworklets.so"
  at com.mrousavy.camera.react.CameraViewModule.<clinit>(CameraViewModule.kt:48)
```

`VisionCamera_enableFrameProcessors=false` in `android/gradle.properties` fixes
it: the camera still previews and still takes photos, but `libVisionCamera.so` is
no longer linked against worklets.

**And it was starved of pixels.** The frame processor worked at 960 px wide,
about 30 px per MICR glyph, then upscaled each crop to the model's 32x48 input.
The model was trained on crops cut from full-resolution photos. A still capture
worked at 1800 px gives a 37 px pitch and an ~85 px band, so every crop is a
downscale to 32x48, the same direction the training crops were resampled in.

## Layout

| path | |
|---|---|
| `src/micr/image.ts` | grayscale primitives: rotate, mirror, crop, area resample, Otsu, adaptive threshold, deskew |
| `src/micr/segment.ts` | band search and the fixed-pitch grid; a port of `micr/segment.py` |
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
npm test               # 63 tests, no device needed
npx tsc --noEmit
```

### Building a release APK

Release signing reads its credentials from `~/.gradle/gradle.properties`
(`MICR_STORE_FILE`, `MICR_STORE_PASSWORD`, `MICR_KEY_ALIAS`, `MICR_KEY_PASSWORD`),
so neither the keystore nor its passwords live in the repo. Without them the
release build stays debug-signed, which installs but cannot be published.

```bash
# Per-ABI APKs, smallest download:
cd android && ./gradlew assembleRelease

# One APK that installs on any phone. Use this for anything you hand to
# someone else; picking the wrong per-ABI file fails as "App not installed".
cd android && ./gradlew assembleRelease -PMICR_UNIVERSAL=1 \
    -PreactNativeArchitectures=arm64-v8a,armeabi-v7a
```

Bump `versionCode` in `android/app/build.gradle` before shipping an update. An
APK signed with a different key than the installed copy also fails as "App not
installed"; the old app has to be uninstalled first.

### On the emulator

The emulator's camera renders a synthetic scene rather than a cheque, so the
capture path cannot be exercised there. Point the AVD's back camera at `webcam0`
in its advanced settings and hold a cheque up to the webcam, or build to a real
device.

The pipeline itself needs no camera: `npm test` runs the whole segmenter over
`__tests__/fixtures/cheque.gray`, a synthetic cheque rendered from the same
E-13B font the model was trained on, and asserts the fields parsed out of it.

## Things that will bite you

**Crop to the sheet before thresholding anything.** A photo of a cheque is
mostly desk, and Otsu over a whole frame separates desk from paper rather than
ink from paper. Every desk pixel then counts as ink, the cheque reads as solid
rows, and the band search returns nothing on a frame where the band is legible.
`findSheet()` is not a refinement: without it the real cheque photos segment to
**zero** glyphs; with it, chk001 segments to all 32. The synthetic test cheque
hides this entirely because the image *is* the sheet, so a regression test frames
it on a dark background.

**Some cameras cannot produce a photo CameraX will accept.** CameraX finishes
every capture by writing EXIF orientation back into the file, and `ExifInterface`
rejects anything it cannot parse as JPEG/PNG/WebP. Emulators with a virtual
camera, LDPlayer among them, produce exactly that: the capture succeeds, the
bytes reach disk, and the whole thing is discarded over a metadata write.

```
androidx.camera.core.ImageCaptureException: Failed to update Exif data
Caused by: java.io.IOException: ExifInterface only supports saving attributes
           on JPEG, PNG, or WebP formats.
```

`takeSnapshot()` compresses the preview view's bitmap itself and never touches
ExifInterface, so it is used as a fallback. It is preview-sized rather than
sensor-sized, which is a real loss of resolution and why it is second choice.

**Both MICR layouts exist.** Some cheques put the cheque number in a leading
auxiliary field; many personal cheques leave that empty and append it to the
on-us field *after* the account number. Parse by position, or account numbers
come out with the cheque number glued on. Both are covered by tests.

**Class order is a contract.** `src/micr/classes.ts` must match
`micr_labels.json` shipped beside the model. Reorder it and every read is
silently wrong, so a test asserts it. `checkModelContract()` also verifies the
loaded `.tflite` on every launch: `[1, 48, 32, 1]` float32 in, 14 classes out.

**The model file says which build it is, on purpose.** The training repo's
`export/` holds float32, fp16 and int8 builds of the same weights. This app
bundled the int8 one under the plain `micr_cnn_v1.tflite` name, so copying the
fp32 export over it would have swapped models silently. It now ships
`micr_cnn_v1_fp32.tflite`: 100% argmax parity with PyTorch against int8's 99.67%,
for 350 KB.

**Hermes has no optimising JIT.** It interprets bytecode, so a property load left
inside a per-pixel loop is a real cost, and the whole pipeline runs roughly 7x
slower on device than the same code under Node. Three things mattered, in order:
hoisting the tap tables out of `resample`'s inner loops (a 1600 px rescale went
790 ms to 31 ms), switching the adaptive threshold's integral image from
`Float64Array` to `Int32Array` (338 ms to 20 ms), and not running the adaptive
pass at all unless Otsu has already failed. With trimmed working resolutions, a
read went from 7.0 s to 4.5 s on an x86_64 emulator. Measure before changing
anything in `image.ts`; the obvious-looking version is often 15x slower.

A read takes a few seconds, not an instant. That is the trade tap-to-capture
makes: one high-resolution read instead of many starved ones.

**Orientation is decided by trying, not by asking.** Sensor mounting, EXIF tags
and `isMirrored` all disagree across devices, and each mapping is a chance to be
wrong in a way that looks like a camera seeing nothing. All four rotations are
scored by how well they parse as E-13B, which costs no model calls, and the best
one or two are classified. Mirroring is not decided that way: a mirrored band
segments identically to an upright one, so only the checksum can tell them apart
and the flipped retry happens per candidate.

**Geometry alone cannot find the MICR line.** A block of clean sans headline text
segments into well-pitched, similar-width boxes and can outscore the real band;
the training repo's own segmenter picks an upside-down "ACME MANUFACTURING LLC"
on the synthetic test cheque. Candidates are ranked by position (the MICR line is
the bottom-most print on a cheque) and length (19 to 40 characters), and several
are classified until one passes the checksum.

**`android/build.gradle` lists Maven mirrors before `google()`.** `dl.google.com`
was unreachable from the machine this was built on. They are read-only mirrors of
the same artefacts and `google()` is still there as a fallback; delete the block
if your network reaches Google directly.

## Where this actually stands

Benchmarked by running the shipped `.tflite` over the crops this segmenter
produces, across all 14 real cheque photos in the training repo, taking the first
candidate that parses:

```
correct 4/14,  rejected 10/14,  wrong-but-accepted 0/14
```

Measured against the previous revision on identical inputs, which scored 3
correct with 2 wrong-but-accepted. Every read that gets through is right, and the
rest are rejected. That is the behaviour to want: a rejected scan costs a retry,
an accepted wrong one pays the wrong account.

### What gets a line rejected

Four guards, in the order they catch things:

* **ABA checksum.** Covers the routing number only, which is why the rest exist.
* **Field structure.** Two transit symbols, a parseable auxiliary field, a
  numeric account.
* **`hasMissingDigits`.** E-13B is fixed pitch, so adjacent characters are one
  cell apart and a field separator makes two. A gap between two *digits* is a
  hole where characters used to be. Nothing else can see this: the model answers
  the crops it is handed and cannot be unsure about characters it was never
  shown, so one read that had silently lost two account digits still scored 0.93
  minimum confidence.
* **Truncation guards in `parse.ts`.** An account field has to be at least six
  digits and has to be closed by an on-us symbol. A line trimmed at the right
  loses its account tail and its closing symbol together, and is otherwise
  perfectly well formed: chk007 read as a 7-digit account against a true 9.

Digits are never repaired. Only symbols are reconsidered, and only where the
model itself ranked an alternative as plausible.

### Measured and rejected, do not retry these blind

* Growing each band over its faint top and bottom rows. Doubled the band height
  and broke a cheque that had been reading correctly.
* Cropping the sheet at full resolution before downscaling. No change.
* Re-cutting the band so the ink fills the same fraction as in training. The
  premise was a measurement error, comparing band-level ink extent against
  per-crop ink extent, and the crops were already within a couple of percent of
  the offline segmenter's. Applying it halved the benchmark.
* A minimum-confidence gate. Measured twice at different stages: it costs correct
  reads (two scored 0.48 and 0.50) and catches nothing.
* Keeping only the longest chain of glyphs. The gap before the on-us field is a
  genuine field boundary that measured 2.21 pitches on a real cheque, so this
  discarded whole account numbers.

### What would move it further

The ten rejections are mostly creased, scribbled-over or badly lit photos, and
several fail because the band is never located at all rather than because it is
misread. More real cheques through the training repo's `micr.segment --debug-dir`,
tuning `SegmentConfig` against the debug images, and carrying the values back into
`DEFAULT_CONFIG` here (the two configs are deliberately the same shape). That is
the 100-to-200-cheque exercise the spec asks for in section 12, and it is a data
exercise rather than a code one.

## Not in this MVP

Manual-entry fallback, submitting to a backend, and iOS. The iOS project is
configured but has not been run.
