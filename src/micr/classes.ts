/**
 * The 14 MICR E-13B classes.
 *
 * This MUST stay in the same order as micr/classes.py in the training repo and
 * as export/micr_labels.json shipped beside the model. Index N is the Nth
 * output of the network; reorder this and every read is silently wrong.
 *
 *   0 1 2 3 4 5 6 7 8 9 amount dash onus transit
 */

export const CLASSES = [
  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
  'amount', 'dash', 'onus', 'transit',
] as const;

export type MicrClass = (typeof CLASSES)[number];

/** Export substitution: T = transit, A = amount, O = on-us, D = dash. */
export const SUBSTITUTION: Record<MicrClass, string> = {
  '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
  '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
  amount: 'A',
  dash: 'D',
  onus: 'O',
  transit: 'T',
};

/** Model input geometry. Normalisation is baked into the model itself. */
export const INPUT_WIDTH = 32;
export const INPUT_HEIGHT = 48;

export function classAt(index: number): MicrClass {
  const name = CLASSES[index];
  if (!name) {
    throw new Error(`model returned class index ${index}, expected 0..${CLASSES.length - 1}`);
  }
  return name;
}

export function toSymbols(classes: MicrClass[]): string {
  return classes.map(c => SUBSTITUTION[c]).join('');
}
