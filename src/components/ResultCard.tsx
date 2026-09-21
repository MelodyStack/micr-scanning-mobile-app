/**
 * The read, once it has passed the ABA checksum.
 *
 * Account and routing numbers are shown in full deliberately: the user is
 * about to confirm them against the cheque in their hand, and a masked number
 * cannot be checked. The manual-entry route from spec section 10 lives here
 * too, because a scanner with no fallback just strands people.
 */

import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';

import { MicrFields } from '../micr/parse';

interface Props {
  fields: MicrFields;
  onRescan: () => void;
  onConfirm?: (fields: MicrFields) => void;
  /** Passed the checksum, but at least one glyph was read without confidence. */
  lowConfidence?: boolean;
}

export default function ResultCard({
  fields,
  onRescan,
  onConfirm,
  lowConfidence,
}: Props) {
  return (
    <View style={styles.sheet}>
      <View style={[styles.badge, lowConfidence && styles.badgeWarn]}>
        <Text style={[styles.badgeText, lowConfidence && styles.badgeTextWarn]}>
          {lowConfidence ? 'Checksum passed, low confidence' : 'Checksum passed'}
        </Text>
      </View>

      <Row label="Routing" value={fields.routing_number} />
      <Row label="Account" value={fields.account_number} />
      <Row label="Cheque no." value={fields.check_number ?? 'none'} />
      {fields.amount_field && <Row label="Amount field" value={fields.amount_field} />}

      <Text style={styles.verify}>Check these against the cheque before continuing.</Text>

      <View style={styles.actions}>
        <Pressable style={[styles.button, styles.secondary]} onPress={onRescan}>
          <Text style={styles.secondaryText}>Scan again</Text>
        </Pressable>
        <Pressable
          style={[styles.button, styles.primary]}
          onPress={() => onConfirm?.(fields)}
        >
          <Text style={styles.primaryText}>Looks right</Text>
        </Pressable>
      </View>
    </View>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.row}>
      <Text style={styles.label}>{label}</Text>
      <Text style={styles.value} selectable>
        {value}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  sheet: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: '#12151b',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    padding: 24,
    paddingBottom: 34,
    gap: 2,
  },
  badge: {
    alignSelf: 'flex-start',
    backgroundColor: 'rgba(74,222,128,0.14)',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 999,
    marginBottom: 14,
  },
  badgeWarn: { backgroundColor: 'rgba(251,191,36,0.16)' },
  badgeText: { color: '#4ade80', fontSize: 12, fontWeight: '600' },
  badgeTextWarn: { color: '#fbbf24' },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'baseline',
    paddingVertical: 9,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#252a34',
  },
  label: { color: '#8b93a1', fontSize: 14 },
  value: {
    color: '#fff',
    fontSize: 17,
    fontWeight: '600',
    fontVariant: ['tabular-nums'],
    letterSpacing: 0.5,
  },
  verify: { color: '#8b93a1', fontSize: 13, marginTop: 14, lineHeight: 18 },
  actions: { flexDirection: 'row', gap: 12, marginTop: 18 },
  button: { flex: 1, paddingVertical: 14, borderRadius: 12, alignItems: 'center' },
  secondary: { backgroundColor: '#1e2430' },
  secondaryText: { color: '#cbd2dd', fontSize: 16, fontWeight: '600' },
  primary: { backgroundColor: '#3b82f6' },
  primaryText: { color: '#fff', fontSize: 16, fontWeight: '600' },
});
