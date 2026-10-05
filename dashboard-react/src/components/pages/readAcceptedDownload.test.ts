import { describe, expect, it } from 'vitest';

import { readAcceptedDownload } from './ModelSearchModal';

function response(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } });
}

describe('readAcceptedDownload', () => {
  it('reports a node download when the node has no model store', async () => {
    expect(await readAcceptedDownload(response({ status: 'downloading', destination: 'node' }))).toEqual({
      rejected: false,
      reason: null,
      destination: 'node',
    });
  });

  it('reports a store download by default', async () => {
    expect(await readAcceptedDownload(response({ status: 'pending' }))).toEqual({
      rejected: false,
      reason: null,
      destination: 'store',
    });
  });

  it('carries a store rejection reason', async () => {
    expect(await readAcceptedDownload(response({ status: 'error', error: 'disk full' }))).toEqual({
      rejected: true,
      reason: 'disk full',
      destination: 'store',
    });
  });
});
