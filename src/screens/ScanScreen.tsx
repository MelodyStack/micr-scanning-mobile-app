/**
 * The one screen: frame a cheque, tap, get validated fields back.
 *
 * Tap-to-capture rather than a live frame processor, for two reasons that both
 * turned out to matter more than the convenience of continuous scanning:
 *
 *  * Resolution. A still is 4032x3024, which is ~90 px per MICR glyph. A video
 *    frame processed at 960 px gives ~30, and the model was trained on crops
 *    cut from full-resolution photos.
 *  * Frame processors on VisionCamera v4 require react-native-worklets-core,
 *    which has no build for this React Native version. It fails to link
 *    `__cxa_init_primary_exception` and takes the process down during
 *    TurboModule init, before any JavaScript runs at all.
 *
 * So there are no worklets anywhere in this app. Capture is a promise, decoding
 * is Skia, and recognition is ordinary TypeScript on the JS thread.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Linking,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraFormat,
  useCameraPermission,
} from 'react-native-vision-camera';
import { loadTensorflowModel, type TfliteModel } from 'react-native-fast-tflite';

import { decodePyramid } from '../micr/decode';
import type { MicrFields } from '../micr/parse';
import {
  checkModelContract,
  describeModel,
  LOW_CONFIDENCE,
  type Recognition,
  recognizeCheque,
} from '../micr/recognize';
import ResultCard from '../components/ResultCard';
import ScanOverlay from '../components/ScanOverlay';

/**
 * The float32 build, named for what it is.
 *
 * `export/` in the training repo holds float32, fp16 and int8 builds of the
 * same weights. This app previously bundled the int8 one under the plain
 * `micr_cnn_v1.tflite` name, so nothing on disk said which was shipping and
 * copying the fp32 export over it would have silently swapped models. The int8
 * build measures 99.67% argmax parity against PyTorch versus 100% for float32,
 * and the difference costs 350 KB of APK, which is not a trade worth making
 * when the whole design leans on reads being right.
 */
const MODEL = require('../../assets/micr_cnn_v1_fp32.tflite');

type Phase = 'loading' | 'ready' | 'working' | 'done' | 'error';

/**
 * Step markers, dev builds only.
 *
 * A read is several seconds of native decode plus several of JS. When one of
 * those stalls the screen just sits there, and from the outside a hang in Skia
 * looks identical to a slow segmentation. These are what tell the two apart.
 */
function trace(message: string): void {
  if (__DEV__) {
    console.log(`[scanner] ${message}`);
  }
}

export default function ScanScreen() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');
  // The largest photo the sensor offers. Every extra pixel across the band is
  // one the segmenter does not have to invent.
  const format = useCameraFormat(device, [{ photoResolution: 'max' }]);
  // The camera LED. Lighting is the biggest lever on whether a band segments
  // cleanly, so it earns its place on real hardware. Emulators have no LED, and
  // asking CameraX for a flash that does not exist throws FlashUnavailableError
  // straight out of takePhoto.
  const hasTorch = device?.hasTorch ?? false;

  const camera = useRef<Camera>(null);
  const [model, setModel] = useState<TfliteModel | null>(null);
  const [phase, setPhase] = useState<Phase>('loading');
  const [fields, setFields] = useState<MicrFields | null>(null);
  const [reading, setReading] = useState<Recognition | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [torch, setTorch] = useState(false);
  const [note, setNote] = useState('Fill the frame with the cheque, then tap');
  // True when the capture came from the preview snapshot rather than the sensor.
  const lowResRef = useRef(false);

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  useEffect(() => {
    let cancelled = false;

    /**
     * Load on CPU first, then try to upgrade to a hardware delegate.
     *
     * NNAPI is missing or broken on plenty of devices, emulators especially,
     * and asking for it up front takes the whole app down instead of falling
     * back. The model is 123k parameters on a 48x32 crop, so the CPU is
     * perfectly quick; the delegate is a bonus, never a requirement.
     */
    loadTensorflowModel(MODEL, [])
      .then(loaded => {
        if (cancelled) {
          return;
        }
        // Worth checking every launch: export/ holds a float32, an fp16 and an
        // int8 build, and the app bundles one of them under a name that does
        // not say which. The wrong file returns confident nonsense that the
        // checksum rejects with no clue why.
        const mismatch = checkModelContract(loaded);
        if (mismatch) {
          setError(mismatch);
          setPhase('error');
          return;
        }
        trace(`model: ${describeModel(loaded)}`);
        setModel(loaded);
        setPhase('ready');

        loadTensorflowModel(MODEL, ['nnapi'])
          .then(accelerated => {
            if (!cancelled && !checkModelContract(accelerated)) {
              setModel(accelerated);
            }
          })
          .catch(() => {});
      })
      .catch(e => {
        if (!cancelled) {
          setError(`Could not load the model: ${e?.message ?? e}`);
          setPhase('error');
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const read = useCallback(
    async (load: () => Promise<Awaited<ReturnType<typeof decodePyramid>>>) => {
      if (!model) {
        return;
      }
      setPhase('working');
      setNote('Reading the number line…');
      const started = Date.now();
      try {
        // Decoding is native and quick; recognition is a second or two of plain
        // JavaScript. Yielding first lets the spinner actually paint.
        trace('decode: start');
        const pyramid = await load();
        trace(
          `decode: done work=${pyramid.work.width}x${pyramid.work.height} ` +
            `scout=${pyramid.scout.width}x${pyramid.scout.height} ` +
            `probe=${pyramid.probe.width}x${pyramid.probe.height}`,
        );
        await new Promise(resolve => setTimeout(resolve, 0));

        trace('recognize: start');
        const result = recognizeCheque(model, pyramid);
        trace(
          `recognize: done ok=${result.ok} glyphs=${result.glyphCount} ` +
            `${result.quarterTurns * 90}deg ${result.threshold ?? '-'} ` +
            `tried=${result.attempts} ` +
            `minconf=${(result.minConfidence * 100).toFixed(0)}% ` +
            `${Date.now() - started}ms${result.raw ? ` raw=${result.raw}` : ''}`,
        );
        setReading(result);

        if (result.ok && result.fields) {
          setFields(result.fields);
          setNote('Read and checksummed');
          setPhase('done');
          return;
        }
        setFields(null);
        setPhase('ready');
        setNote(hintFor(result, lowResRef.current));
      } catch (e: any) {
        trace(`read: threw ${e?.message ?? e}`);
        setFields(null);
        setPhase('ready');
        setNote(`Could not read that photo: ${e?.message ?? e}`);
      }
    },
    [model],
  );

  /**
   * Full-resolution capture, falling back to a preview snapshot.
   *
   * `takePhoto` is the one worth having: it is the full sensor frame, and
   * pixels across the MICR band are the whole reason this app captures stills
   * rather than video frames.
   *
   * But CameraX finishes every photo by writing EXIF orientation back into the
   * file, and `ExifInterface` rejects anything it cannot parse as JPEG/PNG/WebP.
   * Emulators with a virtual camera, LDPlayer among them, produce exactly that:
   * the capture succeeds, the bytes reach disk, and the whole thing is thrown
   * away over a metadata write.
   *
   *   androidx.camera.core.ImageCaptureException: Failed to update Exif data
   *   Caused by: java.io.IOException: ExifInterface only supports saving
   *     attributes on JPEG, PNG, or WebP formats.
   *
   * `takeSnapshot` grabs the preview view's bitmap and compresses it itself, so
   * it never goes near ExifInterface. It is limited to the size of the preview
   * on screen, which is a real loss of resolution, hence second choice rather
   * than first. It is still the difference between a usable scanner and a dead
   * button on a virtual device.
   */
  const capture = useCallback(async () => {
    const device_ = camera.current;
    if (!device_) {
      return;
    }
    try {
      let path: string;
      let degraded = false;
      try {
        const photo = await device_.takePhoto({
          flash: torch && hasTorch ? 'on' : 'off',
          enableShutterSound: false,
        });
        path = photo.path;
      } catch (photoError: any) {
        const snapshot = await device_.takeSnapshot({ quality: 100 });
        path = snapshot.path;
        degraded = true;
        trace(`capture: snapshot at ${path}`);
        console.warn(
          '[scanner] takePhoto failed, fell back to a preview snapshot:',
          photoError?.message ?? photoError,
        );
      }
      lowResRef.current = degraded;
      await read(() => decodePyramid(path));
    } catch (e: any) {
      setPhase('ready');
      setNote(`Capture failed: ${e?.message ?? e}`);
    }
  }, [read, torch, hasTorch]);

  const rescan = useCallback(() => {
    setFields(null);
    setReading(null);
    setPhase('ready');
    setNote('Fill the frame with the cheque, then tap');
  }, []);


  if (!hasPermission) {
    return (
      <Centered>
        <Text style={styles.title}>Camera access needed</Text>
        <Text style={styles.body}>
          The scanner reads the number line printed along the bottom of a cheque.
          Nothing leaves the device until a read passes its checksum.
        </Text>
        <Pressable style={styles.button} onPress={() => requestPermission()}>
          <Text style={styles.buttonText}>Allow camera</Text>
        </Pressable>
        <Pressable onPress={() => Linking.openSettings()}>
          <Text style={styles.link}>Open settings</Text>
        </Pressable>
      </Centered>
    );
  }

  if (phase === 'error') {
    return (
      <Centered>
        <Text style={styles.title}>Scanner unavailable</Text>
        <Text style={styles.body}>{error}</Text>
      </Centered>
    );
  }

  if (phase === 'loading') {
    return (
      <Centered>
        <ActivityIndicator color="#fff" />
        <Text style={styles.body}>Loading the reader…</Text>
      </Centered>
    );
  }

  const busy = phase === 'working';

  return (
    <View style={styles.root}>
      {device ? (
        <Camera
          ref={camera}
          style={StyleSheet.absoluteFill}
          device={device}
          format={format}
          isActive={phase !== 'done'}
          photo
          torch={torch && hasTorch ? 'on' : 'off'}
          enableZoomGesture={false}
        />
      ) : (
        // No usable camera, which is the emulator's normal state.
        <Centered>
          <Text style={styles.title}>No camera on this device</Text>
          <Text style={styles.body}>
            The scanner needs a rear camera to capture a cheque.
          </Text>
        </Centered>
      )}

      {/* Overlay and controls are a column above the camera, so the guide is
          centred in whatever space is left between the header and the control
          bar. Floating the controls over the preview put the torch and the
          shutter on top of both ends of the MICR band. */}
      <View style={styles.stack} pointerEvents="box-none">
        <View style={styles.overlaySlot} pointerEvents="none">
          <ScanOverlay
            note={note}
            busy={busy}
            warn={!!reading && reading.ok && reading.minConfidence < LOW_CONFIDENCE}
          />
        </View>

        {phase !== 'done' && (
          <View style={styles.controls}>
            <View style={styles.side}>
              {hasTorch && (
                <Pressable
                  style={[styles.chip, torch && styles.chipOn]}
                  onPress={() => setTorch(t => !t)}
                >
                  <Text style={[styles.chipText, torch && styles.chipTextOn]}>
                    Torch
                  </Text>
                </Pressable>
              )}
            </View>

            <Pressable
              style={[styles.shutter, busy && styles.shutterBusy]}
              onPress={capture}
              disabled={busy || !device}
              accessibilityLabel="Capture the cheque"
            >
              {busy ? (
                <ActivityIndicator color="#000" />
              ) : (
                <View style={styles.shutterCore} />
              )}
            </Pressable>

            {/* Balances the slot on the left so the shutter stays centred. */}
            <View style={styles.side} />
          </View>
        )}
      </View>

      {fields && (
        <ResultCard
          fields={fields}
          lowConfidence={!!reading && reading.minConfidence < LOW_CONFIDENCE}
          onRescan={rescan}
        />
      )}
    </View>
  );
}

/** Turn a failed read into something the user can act on. */
function hintFor(result: Recognition, lowRes: boolean): string {
  const suffix = lowRes
    ? ' (this device captured the preview, not the sensor, so the band is low-resolution)'
    : '';
  if (result.glyphCount === 0) {
    return `No number line found. Fill the frame with the cheque and keep it flat.${suffix}`;
  }
  if (result.error?.includes('checksum')) {
    return `Read the line but the checksum failed. Try again with more light.${suffix}`;
  }
  if (result.error?.includes('transit')) {
    return `Part of the number line was missed. Move closer and keep it level.${suffix}`;
  }
  return result.error ?? 'Could not read it. Try again.';
}

function Centered({ children }: { children: React.ReactNode }) {
  return <View style={[styles.root, styles.centered]}>{children}</View>;
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#000' },
  centered: { alignItems: 'center', justifyContent: 'center', padding: 32, gap: 12 },
  title: { color: '#fff', fontSize: 20, fontWeight: '600', textAlign: 'center' },
  body: { color: '#b9bec7', fontSize: 15, lineHeight: 21, textAlign: 'center' },
  button: {
    marginTop: 12,
    backgroundColor: '#3b82f6',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 10,
  },
  buttonText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  link: { color: '#7aa2f7', fontSize: 14, marginTop: 4 },

  // Fills the screen above the camera. `box-none` so taps fall through to the
  // preview everywhere except the controls themselves.
  stack: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    flexDirection: 'column',
  },
  overlaySlot: { flex: 1 },

  // A real bar with a background, not buttons floating over the preview. The
  // cheque guide is centred in the space left above it, so the controls can
  // never sit on top of the MICR band.
  controls: {
    paddingVertical: 14,
    paddingHorizontal: 24,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.55)',
  },
  side: { flex: 1 },
  shutter: {
    width: 74,
    height: 74,
    borderRadius: 37,
    backgroundColor: 'rgba(255,255,255,0.28)',
    borderWidth: 3,
    borderColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  shutterBusy: { backgroundColor: 'rgba(255,255,255,0.85)' },
  shutterCore: {
    width: 58,
    height: 58,
    borderRadius: 29,
    backgroundColor: '#fff',
  },
  chip: {
    alignSelf: 'flex-start',
    minWidth: 92,
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 999,
    backgroundColor: 'rgba(0,0,0,0.5)',
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: 'rgba(255,255,255,0.25)',
    alignItems: 'center',
  },
  chipOn: { backgroundColor: '#f5d90a', borderColor: '#f5d90a' },
  chipText: { color: '#e6e9ef', fontSize: 13, fontWeight: '600' },
  chipTextOn: { color: '#000' },
});
