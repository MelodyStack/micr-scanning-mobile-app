/**
 * Framing guide for the whole cheque.
 *
 * Earlier this was a thin slot the user had to line the MICR band up inside.
 * That put the burden of alignment on them and, worse, made recognition depend
 * on mapping the box from screen coordinates back into sensor coordinates --
 * through a rotation and a cover crop. Every one of those mappings was a bug,
 * and each failed the same silent way: the band looked perfectly placed and the
 * reader saw nothing.
 *
 * Now the whole frame goes to the recogniser and it finds the band itself. The
 * outline is only an aiming aid, so it does not have to be exact.
 */

import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

interface Props {
  hint: string;
  active: boolean;
  /** Dev-only readout of what the frame processor is actually seeing. */
  debug?: string;
}

export default function ScanOverlay({ hint, active, debug }: Props) {
  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="none">
      <View style={styles.centre}>
        <View style={styles.cheque}>
          <Corner style={styles.tl} />
          <Corner style={styles.tr} />
          <Corner style={styles.bl} />
          <Corner style={styles.br} />
          {/* Where the MICR band sits on a cheque, as a hint only -- nothing
              is measured from it. */}
          <View style={styles.bandHint} />
        </View>
      </View>

      <View style={styles.footer}>
        <Text style={styles.hint}>{hint}</Text>
        {active && (
          <Text style={styles.sub}>
            Nothing is sent until the read passes its checksum
          </Text>
        )}
        {!!debug && <Text style={styles.debug}>{debug}</Text>}
      </View>
    </View>
  );
}

function Corner({ style }: { style: object }) {
  return <View style={[styles.corner, style]} />;
}

const BORDER = '#4ade80';

const styles = StyleSheet.create({
  centre: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  // A US cheque is about 2.2:1.
  cheque: {
    width: '86%',
    aspectRatio: 2.2,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.3)',
    borderRadius: 6,
  },
  bandHint: {
    position: 'absolute',
    left: '4%',
    right: '4%',
    bottom: '8%',
    height: '13%',
    borderWidth: 1,
    borderStyle: 'dashed',
    borderColor: 'rgba(74,222,128,0.45)',
    borderRadius: 3,
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
  footer: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    paddingBottom: 18,
    paddingHorizontal: 24,
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.45)',
    paddingTop: 12,
  },
  hint: { color: '#fff', fontSize: 15, textAlign: 'center', fontWeight: '500' },
  sub: { color: '#8b93a1', fontSize: 11, textAlign: 'center', marginTop: 5 },
  debug: {
    color: '#4ade80',
    fontSize: 10,
    fontFamily: 'monospace',
    textAlign: 'center',
    marginTop: 8,
  },
});
