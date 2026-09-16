/**
 * Guided capture overlay.
 *
 * The box is the whole trick. Asking the user to put the number line inside it
 * fixes the band's position, scale and rotation, which is why the runtime never
 * has to detect the cheque, correct perspective or work out which way up it is.
 */

import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

/**
 * Guide rectangle as a fraction of the frame. Must match the frame processor.
 *
 * The screen is locked to landscape, so this is a wide, shallow strip sized
 * like the MICR band itself: the band runs almost the full width of a cheque
 * and is only a few millimetres tall. Sitting it slightly below centre matches
 * where it falls when the whole cheque is in view, since the band is near the
 * bottom edge.
 */
export const GUIDE = { x: 0.04, y: 0.55, width: 0.92, height: 0.14 };

interface Props {
  hint: string;
  active: boolean;
}

export default function ScanOverlay({ hint, active }: Props) {
  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="none">
      {/* Dim everything outside the guide so the eye goes to the band. */}
      <View style={[styles.scrim, { height: `${GUIDE.y * 100}%` }]} />
      <View style={styles.middleRow}>
        <View style={[styles.scrim, { width: `${GUIDE.x * 100}%` }]} />
        <View style={styles.guide}>
          <Corner style={styles.tl} />
          <Corner style={styles.tr} />
          <Corner style={styles.bl} />
          <Corner style={styles.br} />
        </View>
        <View style={[styles.scrim, { flex: 1 }]} />
      </View>
      <View style={[styles.scrim, styles.bottom]}>
        <Text style={styles.hint}>{hint}</Text>
        {active && (
          <Text style={styles.sub}>
            Nothing is sent until the read passes its checksum
          </Text>
        )}
      </View>
    </View>
  );
}

function Corner({ style }: { style: object }) {
  return <View style={[styles.corner, style]} />;
}

const BORDER = '#4ade80';

const styles = StyleSheet.create({
  scrim: { backgroundColor: 'rgba(0,0,0,0.62)' },
  middleRow: { flexDirection: 'row', height: `${GUIDE.height * 100}%` },
  guide: {
    width: `${GUIDE.width * 100}%`,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.35)',
    borderRadius: 4,
  },
  corner: {
    position: 'absolute',
    width: 22,
    height: 22,
    borderColor: BORDER,
  },
  tl: { top: -1, left: -1, borderTopWidth: 3, borderLeftWidth: 3, borderTopLeftRadius: 4 },
  tr: { top: -1, right: -1, borderTopWidth: 3, borderRightWidth: 3, borderTopRightRadius: 4 },
  bl: { bottom: -1, left: -1, borderBottomWidth: 3, borderLeftWidth: 3, borderBottomLeftRadius: 4 },
  br: { bottom: -1, right: -1, borderBottomWidth: 3, borderRightWidth: 3, borderBottomRightRadius: 4 },
  bottom: { flex: 1, paddingTop: 28, paddingHorizontal: 32, alignItems: 'center' },
  hint: { color: '#fff', fontSize: 16, textAlign: 'center', fontWeight: '500' },
  sub: { color: '#8b93a1', fontSize: 12, textAlign: 'center', marginTop: 8 },
});
