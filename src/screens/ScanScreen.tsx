/**
 * The one screen: point the camera at a check, get validated fields back.
 *
 * The guide rectangle is doing real work, not decoration. It constrains the
 * user to put the MICR band in a known place at a known scale, which removes
 * document detection, perspective correction and orientation from the runtime
 * problem entirely -- the three things that are hardest on a handheld photo.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Linking,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
} from 'react-native-vision-camera';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { loadTensorflowModel, TensorflowModel } from 'react-native-fast-tflite';
import { useRunOnJS, useSharedValue } from 'react-native-worklets-core';

import { FrameVoter, recognizeDocument } from '../micr/recognize';
import { MicrFields } from '../micr/parse';
import ResultCard from '../components/ResultCard';
import ScanOverlay from '../components/ScanOverlay';

/**
 * Working resolution for the whole frame, in display orientation.
 *
 * The guide is 92% of this wide, so a 30-glyph MICR line lands at roughly 30 px
 * per character -- comfortably above what the segmenter needs to separate
 * them, without making the per-frame luminance pass expensive.
 */
const WORK_WIDTH = 960;

/**
 * How the sensor frame has to be turned to match what the preview draws.
 *
 * Sensor mounting differs by device, so the same code can come out upright on
 * one phone and mirrored or upside down on another. A mirrored frame is the
 * nastiest of those: the glyphs still segment, the model still classifies, and
 * the line assembles backwards -- so it fails the checksum with no hint as to
 * why. Tapping the preview cycles these, and the readout shows which is
 * active, so the right one can be found on the device in a few seconds rather
 * than guessed at from here.
 */
export const ORIENTATIONS: ('auto' | '0deg' | '90deg' | '180deg' | '270deg')[] = [
  'auto',
  '90deg',
  '270deg',
  '180deg',
  '0deg',
];

type Status = 'loading' | 'scanning' | 'done' | 'error';

export default function ScanScreen() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');
  const { resize } = useResizePlugin();

  const [model, setModel] = useState<TensorflowModel | null>(null);
  const [status, setStatus] = useState<Status>('loading');
  const [fields, setFields] = useState<MicrFields | null>(null);
  const [hint, setHint] = useState('Fit the whole cheque in the frame');
  const [error, setError] = useState<string | null>(null);
  // Shown only in dev builds. Guessing at frame geometry from the outside cost
  // a whole debugging round; this makes it visible on the device.
  const [debug, setDebug] = useState('');

  const voter = useRef(new FrameVoter(2));
  const busy = useRef(false);
  // Shared with the camera thread: a ref is not visible from a worklet, so
  // without this the processor would pile up frames faster than they are read.
  const busyShared = useSharedValue(false);
  // Primitives, not an index into a module-level array. A worklet only sees
  // values captured into its closure; reaching for a module object from inside
  // one leaves it undefined, and the resulting throw was being swallowed by the
  // catch below -- the processor span forever without ever calling onBand, so
  // the readout simply went blank.
  const rotationShared =
    useSharedValue<'auto' | '0deg' | '90deg' | '180deg' | '270deg'>('auto');
  const modeIndex = useRef(0);

  const cycleOrientation = useCallback(() => {
    modeIndex.current = (modeIndex.current + 1) % ORIENTATIONS.length;
    rotationShared.value = ORIENTATIONS[modeIndex.current];
    voter.current.reset();
  }, [rotationShared]);
  // Same reason in reverse -- the worklet closure would capture a stale status.
  const statusRef = useRef<Status>('loading');
  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  useEffect(() => {
    let cancelled = false;

    /**
     * Load on CPU, and only then try to upgrade to a hardware delegate.
     *
     * NNAPI is absent or broken on plenty of devices -- emulators especially --
     * and asking for it up front takes the whole app down rather than falling
     * back. The model is 123k parameters on a 48x32 crop, so CPU is perfectly
     * fast; the delegate is a bonus, never a requirement.
     */
    const asset = require('../../assets/micr_cnn_v1.tflite');
    loadTensorflowModel(asset, [])
      .then(loaded => {
        if (cancelled) {
          return;
        }
        setModel(loaded);
        setStatus('scanning');
        // Opportunistic upgrade. If it throws, we keep the CPU model already
        // running and the user never notices.
        loadTensorflowModel(asset, ['nnapi'])
          .then(accelerated => {
            if (!cancelled) {
              setModel(accelerated);
            }
          })
          .catch(() => {});
      })
      .catch(e => {
        if (!cancelled) {
          setError(`Could not load the model: ${e?.message ?? e}`);
          setStatus('error');
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  /**
   * Recognition runs here, on the JS thread, not in the worklet.
   *
   * The frame processor only does what genuinely needs the Frame object --
   * crop and greyscale -- then hands over a 48 KB buffer. Segmentation and
   * inference are ordinary code that way: no 'worklet' directives threaded
   * through the whole pipeline, and the same functions the unit tests cover.
   */
  const onFrame = useRunOnJS(
    (
      grey: Uint8Array,
      width: number,
      height: number,
      fw: number,
      fh: number,
      rotation: string,
    ) => {
      if (!model || statusRef.current !== 'scanning') {
        busy.current = false;
        return;
      }
      try {
        const result = recognizeDocument(model, { data: grey, width, height });
        setDebug(
          `${fw}x${fh} ${rotation} · glyphs ${result.glyphCount}` +
            (result.mirrored ? ' · mirrored' : '') +
            (result.raw ? `\n${result.raw}` : ''),
        );

        if (!result.ok) {
          setHint(
            result.glyphCount === 0
              ? 'Show the whole cheque, number line included'
              : result.error ?? 'Hold steady',
          );
          return;
        }

        const agreed = voter.current.push(result);
        if (agreed?.fields) {
          setFields(agreed.fields);
          setStatus('done');
        } else {
          setHint('Almost — hold steady');
        }
      } finally {
        busy.current = false;
      }
    },
    [model],
  );

  const onWorkletError = useRunOnJS((message: string) => {
    setDebug(`frame processor error: ${message}`);
  }, []);

  const frameProcessor = useFrameProcessor(
    frame => {
      'worklet';
      if (busyShared.value) {
        return;
      }
      busyShared.value = true;

      try {
        // Rotate the frame to display orientation, keeping its aspect ratio,
        // then crop the guide out of the region the preview is actually
        // showing.
        // 'auto' until the user overrides by tapping. The screen is locked to
        // landscape, so a portrait sensor frame always needs a quarter turn --
        // without it the cheque sits sideways in the work image, the MICR band
        // runs vertically, and a search that scans rows finds nothing at all.
        const requested = rotationShared.value;
        const rotation =
          requested === 'auto'
            ? frame.height > frame.width
              ? '90deg'
              : '0deg'
            : requested;
        const quarterTurn = rotation === '90deg' || rotation === '270deg';
        // A quarter turn swaps the axes, so the output aspect flips with it.
        const srcW = quarterTurn ? frame.height : frame.width;
        const srcH = quarterTurn ? frame.width : frame.height;

        const workW = WORK_WIDTH;
        const workH = Math.max(2, Math.round((workW * srcH) / srcW));

        const full = resize(frame, {
          scale: { width: workW, height: workH },
          rotation,
          pixelFormat: 'rgb',
          dataType: 'uint8',
        });

        // Whole frame to luminance (ITU-R 601 weights). No crop: recognition
        // locates the band itself, so there is nothing to map between preview
        // and sensor coordinates and nothing for the user to line up.
        const grey = new Uint8Array(workW * workH);
        for (let i = 0, p = 0; i < grey.length; i++, p += 3) {
          grey[i] = (full[p] * 77 + full[p + 1] * 150 + full[p + 2] * 29) >> 8;
        }

        onFrame(grey, workW, workH, srcW, srcH, rotation).then(() => {
          busyShared.value = false;
        });
      } catch (e: any) {
        // Never swallow this. A silent catch here already hid one bug for a
        // whole round: the processor threw on every frame, reset itself, and
        // looked exactly like a camera that was simply not seeing anything.
        onWorkletError(String(e?.message ?? e));
        busyShared.value = false;
      }
    },
    [
      onFrame, onWorkletError, resize, busyShared, rotationShared,
    ],
  );

  const reset = useCallback(() => {
    voter.current.reset();
    setFields(null);
    setHint('Hold the check so the number line fills the box');
    setStatus('scanning');
  }, []);

  const body = useMemo(() => {
    if (!hasPermission) {
      return (
        <Centered>
          <Text style={styles.title}>Camera access needed</Text>
          <Text style={styles.body}>
            The scanner reads the number line printed along the bottom of a
            cheque. Nothing leaves the device until a read passes its checksum.
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

    if (status === 'error') {
      return (
        <Centered>
          <Text style={styles.title}>Scanner unavailable</Text>
          <Text style={styles.body}>{error}</Text>
        </Centered>
      );
    }

    if (!device || status === 'loading') {
      return (
        <Centered>
          <ActivityIndicator color="#fff" />
          <Text style={styles.body}>
            {device ? 'Loading the reader…' : 'No camera found'}
          </Text>
        </Centered>
      );
    }

    return (
      <>
        <Camera
          style={StyleSheet.absoluteFill}
          device={device}
          isActive={status === 'scanning'}
          frameProcessor={status === 'scanning' ? frameProcessor : undefined}
          enableZoomGesture={false}
          photo={false}
          video={false}
        />
        <ScanOverlay
          hint={hint}
          active={status === 'scanning'}
          debug={__DEV__ ? debug : ''}
        />
        {/* Tap anywhere to try the next sensor orientation. Sits under the
            result sheet so it cannot swallow those buttons. */}
        {status === 'scanning' && (
          <Pressable
            style={StyleSheet.absoluteFill}
            onPress={cycleOrientation}
            accessibilityLabel="Change camera orientation"
          />
        )}
        {fields && <ResultCard fields={fields} onRescan={reset} />}
      </>
    );
  }, [
    device, error, fields, frameProcessor, hasPermission, hint,
    requestPermission, reset, status, cycleOrientation, debug,
  ]);

  return <View style={styles.root}>{body}</View>;
}

function Centered({ children }: { children: React.ReactNode }) {
  return <View style={styles.centered}>{children}</View>;
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#000' },
  centered: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 32,
    gap: 12,
  },
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
  ...(Platform.OS === 'ios' ? {} : {}),
});
