/**
 * Framing guide and status line.
 *
 * The outline is an aiming aid, nothing more. Recognition searches the whole
 * photo and finds the band itself, so no coordinate ever crosses between screen
 * space and sensor space -- which is where an earlier version of this app spent
 * most of its bugs. Every one of them failed the same silent way: the cheque
 * looked perfectly placed and the reader saw nothing.
 *
 * What the guide is genuinely for is getting the cheque close to filling the
 * frame. That is what decides how many pixels land on each MICR glyph, and it
 * is the one thing the user controls that actually matters.
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
    <View style={StyleSheet.absoluteFill} pointerEvents="none">
      {/* Status sits at the top. At the bottom it covered the guide's own band
          marker and collided with the shutter row on a short landscape screen. */}
      <View style={styles.header}>
        <Text style={[styles.note, busy && styles.noteBusy]}>{note}</Text>
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
          {/* Where the band sits on a cheque. A hint for the user; nothing is
              measured from it. */}
          <View style={styles.bandHint}>
            <Text style={styles.bandLabel}>number line</Text>
          </View>
        </View>
      </View>
    </View>
  );
}

function Corner({ style }: { style: object }) {
  return <View style={[styles.corner, style]} />;
}

const BORDER = '#4ade80';

const styles = StyleSheet.create({
  header: {
    paddingTop: 34,
    paddingBottom: 10,
    paddingHorizontal: 24,
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.45)',
  },
  centre: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  // A US cheque is 6.14 x 2.75 inches, so about 2.23:1.
  cheque: {
    width: '88%',
    aspectRatio: 2.23,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.3)',
    borderRadius: 6,
  },
  bandHint: {
    position: 'absolute',
    left: '4%',
    right: '4%',
    bottom: '7%',
    height: '15%',
    borderWidth: 1,
    borderStyle: 'dashed',
    borderColor: 'rgba(74,222,128,0.45)',
    borderRadius: 3,
    alignItems: 'center',
    justifyContent: 'center',
  },
  bandLabel: {
    color: 'rgba(74,222,128,0.7)',
    fontSize: 10,
    letterSpacing: 1,
    textTransform: 'uppercase',
  },
  corner: { position: 'absolute', width: 26, height: 26, borderColor: BORDER },
  tl: { top: -1, left: -1, borderTopWidth: 3, borderLeftWidth: 3, borderTopLeftRadius: 6 },
  tr: { top: -1, right: -1, borderTopWidth: 3, borderRightWidth: 3, borderTopRightRadius: 6 },
  bl: {
    bottom: -1, left: -1, borderBottomWidth: 3, borderLeftWidth: 3,
    borderBottomLeftRadius: 6,
  },
  br: {
    bottom: -1, right: -1, borderBottomWidth: 3, borderRightWidth: 3,
    borderBottomRightRadius: 6,
  },
  note: { color: '#fff', fontSize: 15, textAlign: 'center', fontWeight: '500' },
  noteBusy: { color: '#7aa2f7' },
  sub: { color: '#8b93a1', fontSize: 11, textAlign: 'center', marginTop: 5 },
  warn: { color: '#fbbf24', fontSize: 12, textAlign: 'center', marginTop: 5 },
  debug: {
    color: '#4ade80',
    fontSize: 10,
    fontFamily: 'monospace',
    textAlign: 'center',
    marginTop: 8,
  },
});
