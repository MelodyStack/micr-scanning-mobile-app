# MICR cheque scanner — React Native

Reads the MICR (E-13B) line from the bottom of a cheque using the phone camera,
and returns the routing number, account number and cheque number. Everything
runs on the device: no network call, no per-scan cost.

The model is trained separately in
[micr-training](https://github.com/MelodyStack/micr-training); this repo is the
app that ships it.

## The pipeline

```
camera frame
  → crop to the guide box          worklet: vision-camera + resize-plugin
  → greyscale
  → Otsu threshold                 ┐
  → fixed-pitch grid segmentation  │  plain TypeScript, unit-tested
  → TFLite, one glyph at a time    │  ← the only ML step
  → assemble the T/A/O/D string    │
  → ABA checksum                   ┘
  → two frames must agree          → show the fields
```

**The guide box is load-bearing, not decoration.** Asking the user to line the
MICR band up inside it fixes the band's position, scale and rotation, which is
why the runtime never has to detect the cheque, correct perspective, or work out
which way up it is. Those three are what make freehand cheque photos hard: on a
batch of 14 real photos taken on a desk, the offline segmenter handled 4. Inside
the guide the same algorithm has a far easier problem.

**The checksum is a hard gate.** A read that fails it is never shown and never
sent anywhere — the user just keeps scanning. A random misread has roughly a
1-in-10 chance of passing the checksum by luck, so two independent frames must
agree on the same digits before anything is displayed.

## Layout

| path | |
|---|---|
| `src/micr/classes.ts` | the 14 classes, pinned to the order in `micr_labels.json` |
| `src/micr/segment.ts` | Otsu, projection, fixed-pitch grid — a port of the Python segmenter |
| `src/micr/parse.ts` | ABA checksum and field extraction |
| `src/micr/recognize.ts` | model invocation and multi-frame voting |
| `src/screens/ScanScreen.tsx` | camera, permissions, states |
| `src/components/` | guide overlay and the result sheet |
| `assets/micr_cnn_v1.tflite` | the int8 model, 135 KB |

## Running it

```bash
npm install
npm run android        # a physical device: the frame processor needs a real camera
```

```bash
npm test               # 27 tests, no device needed
npx tsc --noEmit
```

The tests cover the parts where being wrong is expensive: the ABA checksum
against real routing numbers, both MICR field layouts, and the segmenter's two
awkward cases — grouping the separate strokes of one symbol into a single
character cell, and rejecting isolated marks.

## Things that will bite you

**Both MICR layouts exist.** Some cheques put the cheque number in a leading
auxiliary field; many personal cheques leave that empty and append it to the
on-us field *after* the account number. Parse by position, or account numbers
come out with the cheque number glued on the end. Both are covered by tests.

**Class order is a contract.** `src/micr/classes.ts` must match
`micr_labels.json` shipped beside the model. Reorder it and every read is
silently wrong — a test asserts it.

**vision-camera is pinned to v4 on purpose.** v5 is a Nitro rewrite with no
`useFrameProcessor`, and `vision-camera-resize-plugin` targets the v4 API.

**Metro needs `assetExts` appended, not replaced.** Setting it outright drops
`png`, `ttf` and everything else.

**`react-native-nitro-modules` is a direct dependency on purpose.**
`fast-tflite` needs it as a Gradle subproject, and autolinking only registers
direct dependencies — as a transitive one the Android build fails to configure.

**`android/build.gradle` lists Maven mirrors before `google()`.** `dl.google.com`
was unreachable from the machine this was built on. They are read-only mirrors
of the same artefacts and `google()` is still there as a fallback; delete the
block if your network reaches Google directly.

## Not in this MVP

Manual-entry fallback (the button is stubbed), submitting to a backend, and
iOS — the iOS project builds but has not been run.
