/**
 * Framing guide and status line.
 *
 * The rule this follows, learned the hard way: **nothing is drawn over the
 * cheque**. An earlier version put a dashed box and the caption "NUMBER LINE"
 * right where the MICR band sits, which is the one part of the cheque the user
 * needs to see well enough to line up. The band indicator is now a pair of
 * ticks outside the frame, and the controls live in their own bar below it.
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
  /** Dev-only readout of what the reader actually saw. */
  debug?: string;
}

export default function ScanOverlay({ note, busy, warn, debug }: Props) {
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
            Nothing is sent until the read passes its checksum
          </Text>
        )}
        {!!debug && <Text style={styles.debug}>{debug}</Text>}
      </View>

      <View style={styles.centre}>
        <View style={styles.cheque}>
          <Corner style={styles.tl} />
          <Corner style={styles.tr} />
          <Corner style={styles.bl} />
          <Corner style={styles.br} />

          {/* Where the band sits, marked from outside the frame so the print
              underneath stays legible. */}
          <View style={[styles.bandTick, styles.bandTickLeft]} />
          <View style={[styles.bandTick, styles.bandTickRight]} />
        </View>
      </View>
    </View>
  );
}

function Corner({ style }: { style: object }) {
  return <View style={[styles.corner, style]} />;
}

// White, not a colour. A cheque is cream paper with dark ink and often a
// pastel security tint, and the previous green sat right on top of that range.
const GUIDE = '#ffffff';

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

  corner: { position: 'absolute', width: 30, height: 30, borderColor: GUIDE },
  tl: { top: 0, left: 0, borderTopWidth: 3, borderLeftWidth: 3 },
  tr: { top: 0, right: 0, borderTopWidth: 3, borderRightWidth: 3 },
  bl: { bottom: 0, left: 0, borderBottomWidth: 3, borderLeftWidth: 3 },
  br: { bottom: 0, right: 0, borderBottomWidth: 3, borderRightWidth: 3 },

  // Short ticks just outside the left and right edges, at the height the MICR
  // band occupies on a cheque. They say "put the number line here" without
  // covering it.
  bandTick: {
    position: 'absolute',
    bottom: '9%',
    width: 18,
    height: 2,
    backgroundColor: GUIDE,
    opacity: 0.8,
  },
  bandTickLeft: { left: -22 },
  bandTickRight: { right: -22 },

  note: { color: '#fff', fontSize: 15, textAlign: 'center', fontWeight: '500' },
  noteBusy: { color: '#7aa2f7' },
  sub: { color: '#9aa2b1', fontSize: 11, textAlign: 'center', marginTop: 4 },
  warn: { color: '#fbbf24', fontSize: 12, textAlign: 'center', marginTop: 4 },
  debug: {
    color: '#93c5fd',
    fontSize: 10,
    fontFamily: 'monospace',
    textAlign: 'center',
    marginTop: 6,
  },
});
