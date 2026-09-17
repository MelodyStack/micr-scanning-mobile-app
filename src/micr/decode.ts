/**
 * JPEG on disk -> grayscale pixels in memory.
 *
 * This is the only module in the pipeline that touches a native library.
 * Everything downstream is plain TypeScript over a Uint8Array, which is what
 * lets the segmenter and the parser be unit-tested without a device.
 *
 * Skia is used rather than a frame processor for two reasons:
 *
 *  1. A still photo is 4032x3024 on the phones this targets. Across a cheque
 *     that is roughly 90 px per MICR glyph. The previous frame-processor design
 *     worked from a 960 px video frame -- about 30 px per glyph -- and then
 *     upscaled each one to the model's 32x48 input. The model was trained on
 *     crops cut from full-resolution photos, so that gap alone would cost
 *     accuracy no amount of tuning could recover.
 *
 *  2. Surface.Make() is a CPU raster surface. It needs no GL context, which
 *     matters on the emulator, where the GL stack is unreliable.
 */

import {
  AlphaType,
  ColorType,
  FilterMode,
  MipmapMode,
  Skia,
  type SkImage,
} from '@shopify/react-native-skia';

import { type GrayImage, type ImagePyramid, PYRAMID_SIZES } from './image';

export class DecodeError extends Error {}

/** Prefix a bare filesystem path with file:// -- Skia needs a URI. */
export function toUri(path: string): string {
  if (/^[a-z][a-z0-9+.-]*:/i.test(path)) {
    return path;
  }
  return `file://${path.startsWith('/') ? '' : '/'}${path}`;
}

/**
 * Decode once and emit all three working resolutions.
 *
 * Scaling is what a read spends much of its time on, and Skia does it in native
 * code -- so the alternative, decoding at full size and rescaling twice in
 * JavaScript, pays for the same work in a bytecode interpreter. One decode,
 * three native draws.
 */
export async function decodePyramid(
  pathOrUri: string,
  sizes: { work: number; scout: number; probe: number } = PYRAMID_SIZES,
): Promise<ImagePyramid> {
  return withDecoded(pathOrUri, image => ({
    work: imageToGrayscale(image, sizes.work),
    scout: imageToGrayscale(image, sizes.scout),
    probe: imageToGrayscale(image, sizes.probe),
  }));
}

async function withDecoded<T>(
  pathOrUri: string,
  read: (image: SkImage) => T,
): Promise<T> {
  const data = await Skia.Data.fromURI(toUri(pathOrUri));
  const image = Skia.Image.MakeImageFromEncoded(data);
  if (!image) {
    data.dispose?.();
    throw new DecodeError(`could not decode ${pathOrUri} -- unsupported or corrupt image`);
  }
  try {
    return read(image);
  } finally {
    // Skia handles are reference counted natively and are not released by the
    // JS garbage collector promptly. A 12 MP decode held across a few captures
    // is enough to get the app killed for memory.
    image.dispose?.();
    data.dispose?.();
  }
}

export function imageToGrayscale(image: SkImage, maxSide: number): GrayImage {
  const srcW = image.width();
  const srcH = image.height();
  if (srcW < 2 || srcH < 2) {
    throw new DecodeError(`decoded image is ${srcW}x${srcH}`);
  }

  const scale = Math.min(1, maxSide / Math.max(srcW, srcH));
  const width = Math.max(2, Math.round(srcW * scale));
  const height = Math.max(2, Math.round(srcH * scale));

  const surface = Skia.Surface.Make(width, height);
  if (!surface) {
    throw new DecodeError(`could not allocate a ${width}x${height} surface`);
  }

  try {
    const canvas = surface.getCanvas();
    const paint = Skia.Paint();
    paint.setAntiAlias(true);
    canvas.drawImageRectOptions(
      image,
      Skia.XYWHRect(0, 0, srcW, srcH),
      Skia.XYWHRect(0, 0, width, height),
      FilterMode.Linear,
      MipmapMode.Linear,
      paint,
    );
    surface.flush();

    const snapshot = surface.makeImageSnapshot();
    try {
      const pixels = snapshot.readPixels(0, 0, {
        width,
        height,
        colorType: ColorType.RGBA_8888,
        alphaType: AlphaType.Unpremul,
      });
      if (!pixels) {
        throw new DecodeError('readPixels returned nothing');
      }
      return toLuminance(pixels as Uint8Array, width, height);
    } finally {
      snapshot.dispose?.();
    }
  } finally {
    surface.dispose?.();
  }
}

/**
 * RGBA -> luminance, ITU-R BT.601 weights in fixed point.
 *
 * The green weight dominating is not incidental here: cheque security tints are
 * usually blue or green pastels, and weighting green heavily flattens a blue
 * tint into the paper rather than letting it read as ink.
 */
export function toLuminance(rgba: Uint8Array, width: number, height: number): GrayImage {
  const out = new Uint8Array(width * height);
  for (let i = 0, p = 0; i < out.length; i++, p += 4) {
    // eslint-disable-next-line no-bitwise -- fixed point: /256 without a divide
    out[i] = (rgba[p] * 77 + rgba[p + 1] * 150 + rgba[p + 2] * 29) >> 8;
  }
  return { data: out, width, height };
}
