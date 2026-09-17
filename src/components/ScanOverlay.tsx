/**
 * Framing guide and status line.
 *
 * Two rules, both learned from testing against real cheques:
 *
 *  1. **Nothing is drawn over the number line.** An early version put a dashed
 *     box and the caption "NUMBER LINE" exactly where the MICR band falls --
 *     the one part of the cheque the user has to see to line it up.
 *  2. **But it still has to say where the numbers go.** Removing the marker
 *     entirely left the user guessing.
 *
 * So the band is *bracketed*, never covered: a red line above it and another
 * below, with the row between them left completely clear. White marks the
 * cheque's edges, red means "put the numbers here" -- two colours, one job
 * each.
 *
 * The outline is an aiming aid, nothing more. Recognition searches the whole
 * photo and finds the band itself, so no coordinate ever crosses between screen
 * space and sensor space -- which is where an earlier version of this app spent
 * most of its bugs. What the guide is genuinely for is getting the cheque close
 * to filling the frame, because that decides how many pixels land on each
 * glyph.
 */

import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

interface Props {
  note: string;
  busy: boolean;
  /** The read passed its checksum but carried an unsure glyph. */
  warn?: boolean;
}

export default function ScanOverlay({ note, busy, warn }: Props) {
  return (
    <View style={styles.root} pointerEvents="none">
      <View style={styles.header}>
        <Text style={[styles.note, busy && styles.noteBusy]} numberOfLines={2}>
          {note}
        </Text>
        {warn ? (
          <Text style={styles.warn}>
            One character was borderline — check the digits carefully.
          </Text>
        ) : (
          <Text style={styles.sub}>
            Keep the number row between the red lines
          </Text>
        )}
      </View>

      <View style={styles.centre}>
        <View style={styles.cheque}>
          <Corner style={styles.tl} />
          <Corner style={styles.tr} />
          <Corner style={styles.bl} />
          <Corner style={styles.br} />

          {/* The band goes between these. Nothing is drawn across the row
              itself, so the printed characters stay fully legible. */}
          <View style={styles.bandTop} />
          <View style={styles.bandBottom} />

          {/* End markers, so the zone still reads as a band and not as two
              unrelated rules. */}
          <View style={[styles.bandEnd, styles.bandEndLeft]} />
          <View style={[styles.bandEnd, styles.bandEndRight]} />
        </View>
      </View>
    </View>
  );
}

function Corner({ style }: { style: object }) {
  return <View style={[styles.corner, style]} />;
}

// White for the sheet, red for the band. A cheque is cream paper with dark ink
// and often a pastel security tint; green sat inside that range and disappeared
// into it, which is why this is not green any more.
const SHEET = '#ffffff';
const BAND = '#ff3b30';

// A MICR band sits in the bottom 5/8 inch of a 2.75 inch cheque, and the
// characters themselves occupy roughly 84% to 91% of its height. The brackets
// are set a little outside that so the row has room to breathe.
const BAND_TOP = '79%';
const BAND_BOTTOM = '4%';

const styles = StyleSheet.create({
  root: { flex: 1 },
  header: {
    paddingTop: 30,
    paddingBottom: 10,
    paddingHorizontal: 24,
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.55)',
  },
  centre: { flex: 1, alignItems: 'center', justifyContent: 'center' },

  // A US cheque is 6.14 x 2.75 inches, so about 2.23:1.
  cheque: {
    width: '86%',
    aspectRatio: 2.23,
    maxHeight: '88%',
  },

  corner: { position: 'absolute', width: 30, height: 30, borderColor: SHEET },
  tl: { top: 0, left: 0, borderTopWidth: 3, borderLeftWidth: 3 },
  tr: { top: 0, right: 0, borderTopWidth: 3, borderRightWidth: 3 },
  bl: { bottom: 0, left: 0, borderBottomWidth: 3, borderLeftWidth: 3 },
  br: { bottom: 0, right: 0, borderBottomWidth: 3, borderRightWidth: 3 },

  bandTop: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: BAND_TOP,
    height: 2,
    backgroundColor: BAND,
  },
  bandBottom: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: BAND_BOTTOM,
    height: 2,
    backgroundColor: BAND,
  },
  // Short uprights joining the two lines at each end.
  bandEnd: {
    position: 'absolute',
    top: BAND_TOP,
    bottom: BAND_BOTTOM,
    width: 2,
    backgroundColor: BAND,
  },
  bandEndLeft: { left: 0 },
  bandEndRight: { right: 0 },

  note: { color: '#fff', fontSize: 15, textAlign: 'center', fontWeight: '500' },
  noteBusy: { color: '#7aa2f7' },
  sub: { color: '#ff8a80', fontSize: 11, textAlign: 'center', marginTop: 4 },
  warn: { color: '#fbbf24', fontSize: 12, textAlign: 'center', marginTop: 4 },
});
