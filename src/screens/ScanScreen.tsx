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

import { FrameVoter, recognizeBand } from '../micr/recognize';
import { MicrFields } from '../micr/parse';
import ResultCard from '../components/ResultCard';
import ScanOverlay, { GUIDE } from '../components/ScanOverlay';

/**
 * Working resolution for the whole frame, in display orientation.
 *
 * The guide is 92% of this wide, so a 30-glyph MICR line lands at roughly 30 px
 * per character -- comfortably above what the segmenter needs to separate
 * them, without making the per-frame luminance pass expensive.
 */
const WORK_WIDTH = 960;
const WORK_HEIGHT = 540;

type Status = 'loading' | 'scanning' | 'done' | 'error';

export default function ScanScreen() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');
  const { resize } = useResizePlugin();

  const [model, setModel] = useState<TensorflowModel | null>(null);
  const [status, setStatus] = useState<Status>('loading');
  const [fields, setFields] = useState<MicrFields | null>(null);
  const [hint, setHint] = useState('Hold the check so the number line fills the box');
  const [error, setError] = useState<string | null>(null);
  // Shown only in dev builds. Guessing at frame geometry from the outside cost
  // a whole debugging round; this makes it visible on the device.
  const [debug, setDebug] = useState('');

  const voter = useRef(new FrameVoter(2));
  const busy = useRef(false);
  // Shared with the camera thread: a ref is not visible from a worklet, so
  // without this the processor would pile up frames faster than they are read.
  const busyShared = useSharedValue(false);
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
  const onBand = useRunOnJS(
    (grey: Uint8Array, width: number, height: number, fw: number, fh: number) => {
      if (!model || statusRef.current !== 'scanning') {
        busy.current = false;
        return;
      }
      try {
        const result = recognizeBand(model, { data: grey, width, height });
        setDebug(
          `frame ${fw}x${fh} → band ${width}x${height} · glyphs ${result.glyphCount}`,
        );

        if (!result.ok) {
          setHint(
            result.glyphCount === 0
              ? 'Line the number row up inside the box'
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

  const frameProcessor = useFrameProcessor(
    frame => {
      'worklet';
      if (busyShared.value) {
        return;
      }
      busyShared.value = true;

      try {
        // Rotate and scale the WHOLE frame to display orientation, then crop
        // the guide out of that in plain array arithmetic.
        //
        // resize()'s crop rectangle is in sensor coordinates while GUIDE is in
        // display coordinates, and the preview is rotated between the two.
        // Cropping there sampled a different part of the frame than the box the
        // user is aiming with -- the band was visibly inside the guide and the
        // segmenter still saw zero glyphs, because it was being handed a patch
        // of blank paper. Doing the crop after rotation removes the mapping
        // entirely: what the buffer holds is what the preview shows.
        const portraitFrame = frame.height > frame.width;
        const full = resize(frame, {
          scale: { width: WORK_WIDTH, height: WORK_HEIGHT },
          rotation: portraitFrame ? '90deg' : '0deg',
          pixelFormat: 'rgb',
          dataType: 'uint8',
        });

        const x0 = Math.round(WORK_WIDTH * GUIDE.x);
        const y0 = Math.round(WORK_HEIGHT * GUIDE.y);
        const bandW = Math.round(WORK_WIDTH * GUIDE.width);
        const bandH = Math.round(WORK_HEIGHT * GUIDE.height);

        // Crop and convert to luminance in one pass (ITU-R 601 weights).
        const grey = new Uint8Array(bandW * bandH);
        for (let y = 0; y < bandH; y++) {
          let src = ((y0 + y) * WORK_WIDTH + x0) * 3;
          let dst = y * bandW;
          for (let x = 0; x < bandW; x++, src += 3, dst++) {
            grey[dst] = (full[src] * 77 + full[src + 1] * 150 + full[src + 2] * 29) >> 8;
          }
        }

        onBand(grey, bandW, bandH, frame.width, frame.height).then(() => {
          busyShared.value = false;
        });
      } catch {
        busyShared.value = false;
      }
    },
    [onBand, resize, busyShared],
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
        {fields && <ResultCard fields={fields} onRescan={reset} />}
      </>
    );
  }, [
    device, error, fields, frameProcessor, hasPermission, hint,
    requestPermission, reset, status,
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
