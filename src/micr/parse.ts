/**
 * Turn a recognised MICR string into the fields the backend expects, and gate
 * it on the ABA checksum.
 *
 * Spec section 8: the routing number carries a checksum, so every read can be
 * validated in code. A failed checksum means reject and retry -- never return
 * garbage to the backend.
 */

export interface MicrFields {
  check_number: string | null;
  routing_number: string;
  account_number: string;
  amount_field: string | null;
}

export interface ParseResult {
  ok: boolean;
  fields?: MicrFields;
  raw: string;
  error?: string;
}

/**
 * ABA checksum: weights 3, 7, 1 across the nine digits, sum mod 10 must be 0.
 */
export function abaChecksumValid(routing: string): boolean {
  if (!/^\d{9}$/.test(routing)) {
    return false;
  }
  const weights = [3, 7, 1, 3, 7, 1, 3, 7, 1];
  let sum = 0;
  for (let i = 0; i < 9; i++) {
    sum += Number(routing[i]) * weights[i];
  }
  return sum % 10 === 0;
}

/**
 * Parse a substituted MICR line (T/A/O/D plus digits).
 *
 * Handles both layouts, which is the trap spec section 8 calls out: many
 * personal checks leave the leading auxiliary field empty and append the check
 * number to the on-us field after the account number. Splitting on position
 * rather than on "the digits before the first O" is what stops account numbers
 * coming out with the check number glued on.
 *
 *   O013708O T113000023T 586033512335O   -> aux holds the check number
 *   T111000614T 687808910O8241           -> check number trails the on-us
 */
export function parseMicr(raw: string): ParseResult {
  const line = raw.replace(/\s+/g, '');

  if (!/^[0-9TAOD]*$/.test(line)) {
    return { ok: false, raw, error: 'line contains non-MICR characters' };
  }

  const transitPositions = [...line].reduce<number[]>(
    (acc, ch, i) => (ch === 'T' ? [...acc, i] : acc),
    [],
  );
  if (transitPositions.length !== 2) {
    return {
      ok: false,
      raw,
      error: `expected 2 transit symbols, found ${transitPositions.length}`,
    };
  }

  const [tStart, tEnd] = transitPositions;
  const routing = line.slice(tStart + 1, tEnd);
  if (!abaChecksumValid(routing)) {
    return {
      ok: false,
      raw,
      error: `routing ${routing || '(empty)'} fails the ABA checksum`,
    };
  }

  // Leading auxiliary field, if present: O<digits>O before the transit field.
  const head = line.slice(0, tStart);
  let checkNumber: string | null = null;
  const auxMatch = head.match(/^O(\d+)O$/);
  if (auxMatch) {
    checkNumber = auxMatch[1];
  } else if (head.length > 0 && !/^O?$/.test(head)) {
    return { ok: false, raw, error: `unrecognised leading field "${head}"` };
  }

  // Everything after the transit field is the on-us portion.
  const tail = line.slice(tEnd + 1);
  const amountMatch = tail.match(/A(\d+)A/);
  const amountField = amountMatch ? amountMatch[1] : null;
  const onus = amountMatch ? tail.replace(amountMatch[0], '') : tail;

  const parts = onus.split('O').filter(p => p.length > 0);
  if (parts.length === 0) {
    return { ok: false, raw, error: 'no account number found after the routing field' };
  }

  const account = parts[0];
  if (checkNumber === null && parts.length > 1) {
    // Personal-check layout: account, on-us symbol, then the check number.
    checkNumber = parts[parts.length - 1];
  }

  if (!/^\d+$/.test(account)) {
    return { ok: false, raw, error: `account ${account} is not numeric` };
  }

  return {
    ok: true,
    raw,
    fields: {
      check_number: checkNumber,
      routing_number: routing,
      account_number: account,
      amount_field: amountField,
    },
  };
}
