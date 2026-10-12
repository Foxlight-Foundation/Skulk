import { describe, expect, it } from 'vitest';
import { serverTimeFromResponse } from './steward';

describe('serverTimeFromResponse', () => {
  it('reads the server clock from the Date header', () => {
    const response = new Response(null, { headers: { date: 'Sun, 11 Oct 2026 12:00:00 GMT' } });
    expect(serverTimeFromResponse(response)).toBe(Date.parse('2026-10-11T12:00:00Z'));
  });

  it('gives null when there is no response, no header, or an unreadable one', () => {
    expect(serverTimeFromResponse(undefined)).toBeNull();
    expect(serverTimeFromResponse(new Response(null))).toBeNull();
    expect(serverTimeFromResponse(new Response(null, { headers: { date: 'not a date' } }))).toBeNull();
  });
});
