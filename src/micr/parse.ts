/**
 * Turn a recognised MICR string into the fields the backend expects, and gate
 * it on the ABA checksum.
 *
 * Spec section 8: the routing number carries a checksum, so every read can be
 * validated in code. A failed checksum means reject and retry, never return
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
 * ABA checksum: weights 3, 7, 1 repeating across the nine digits; the weighted
 * sum must be a multiple of 10.
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

/** Longest on-us field the ANSI layout allows, plus room for a misread. */
const MAX_ACCOUNT_DIGITS = 17;

/**
 * Shortest account number treated as a real read.
 *
 * A plausibility floor against truncation. When the band search trims boxes off
 * the right of a line it amputates the account and nothing else notices: the
 * checksum covers only the routing number, the structure still parses, and the
 * surviving glyphs are classified confidently. One cheque came back with a
 * one-digit account, `5` in place of `586033512335`, and was otherwise well
 * formed.
 *
 * The 14 reference cheques run 8 to 12 digits. Six leaves room below anything
 * realistic while still catching a line that lost most of itself.
 */
const MIN_ACCOUNT_DIGITS = 6;

/**
 * Parse a substituted MICR line (T/A/O/D plus digits).
 *
 * Handles both layouts, which is the trap spec section 8 calls out: many
 * personal cheques leave the leading auxiliary field empty and append the
 * cheque number to the on-us field *after* the account number. Splitting by
 * position rather than by "the digits before the first O" is what stops account
 * numbers coming out with the cheque number glued on.
 *
 *   O013708O T113000023T 586033512335O   aux field holds the cheque number
 *   T111000614T 687808910O8241           cheque number trails the on-us field
 */
export function parseMicr(raw: string): ParseResult {
  const line = raw.replace(/\s+/g, '');

  if (line.length === 0) {
    return { ok: false, raw, error: 'nothing was read' };
  }
  if (!/^[0-9TAOD]*$/.test(line)) {
    return { ok: false, raw, error: 'line contains non-MICR characters' };
  }

  const transitPositions: number[] = [];
  for (let i = 0; i < line.length; i++) {
    if (line[i] === 'T') {
      transitPositions.push(i);
    }
  }
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

  // Leading auxiliary field, if present: O<digits>O ahead of the transit field.
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

  const parts = onus.split('O').filter(part => part.length > 0);
  if (parts.length === 0) {
    return { ok: false, raw, error: 'no account number after the routing field' };
  }

  // The account field has to be closed by an on-us symbol.
  //
  // The second half of the truncation guard, catching what a length floor
  // cannot. A truncated line loses its account tail and its closing symbol
  // together: chk007 came back as `...T5728596` for a true `...T572859650O`,
  // which is well formed, checksum-valid and wrong by two digits.
  //
  // A digit run that stops at the end of the line was never terminated, so
  // there is no evidence the account ended there rather than the reading did.
  // Either the run is followed by another field (`687808910O8241`, the layout
  // where the cheque number trails) or the line closes on the symbol itself.
  // All 14 reference cheques do one or the other.
  if (parts.length === 1 && !onus.endsWith('O')) {
    return {
      ok: false,
      raw,
      error: 'the account field is not closed by an on-us symbol; the line was cut short',
    };
  }

  // The dash is a separator inside the on-us field on plenty of cheques, not a
  // digit. It is kept in `raw` for traceability and dropped from the value the
  // backend receives, which is the convention the existing web app follows.
  const account = parts[0].replace(/D/g, '');
  if (checkNumber === null && parts.length > 1) {
    // Personal-cheque layout: account, on-us symbol, then the cheque number.
    checkNumber = parts[parts.length - 1].replace(/D/g, '');
  }

  if (!/^\d+$/.test(account)) {
    return { ok: false, raw, error: `account "${parts[0]}" is not numeric` };
  }
  if (account.length > MAX_ACCOUNT_DIGITS) {
    return {
      ok: false,
      raw,
      error: `account number is ${account.length} digits, longer than the layout allows`,
    };
  }
  if (account.length < MIN_ACCOUNT_DIGITS) {
    return {
      ok: false,
      raw,
      error:
        `account number is only ${account.length} digit(s); the line was cut short`,
    };
  }
  if (checkNumber !== null && !/^\d+$/.test(checkNumber)) {
    return { ok: false, raw, error: `cheque number "${checkNumber}" is not numeric` };
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
