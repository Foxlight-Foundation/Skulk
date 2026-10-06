import { describe, expect, it } from 'vitest';

import { extractErrorDetail, readAcceptedDownload } from './ModelSearchModal';

function jsonResponse(payload: object, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

describe('readAcceptedDownload', () => {
  it('treats a store-host rejection riding an HTTP 200 as rejected with its reason', async () => {
    // The API answers 200 even when the store host refused the request; the
    // rejection is encoded in the body. Judging by res.ok alone showed a
    // false "Downloading to store" success toast.
    const result = await readAcceptedDownload(jsonResponse({
      status: 'error',
      error: 'HTTP 409: Requested immutable card is not available for this alias',
    }));

    expect(result.rejected).toBe(true);
    expect(result.reason).toContain('immutable card');
  });

  it('accepts a normal transfer-state response', async () => {
    const result = await readAcceptedDownload(jsonResponse({ modelId: 'org/m', status: 'pending' }));
    expect(result.rejected).toBe(false);
  });

  it('reports a rejection without usable text as rejected with no reason', async () => {
    const result = await readAcceptedDownload(jsonResponse({ status: 'error', error: '   ' }));
    expect(result.rejected).toBe(true);
    expect(result.reason).toBeNull();
  });

  it('treats an unparseable body as accepted (historical shape)', async () => {
    const result = await readAcceptedDownload(new Response('not json', { status: 200 }));
    expect(result.rejected).toBe(false);
  });
});

describe('extractErrorDetail', () => {
  it("reads the API's error envelope (error.message), the shape HTTPExceptions actually produce", async () => {
    // API.http_exception_handler serializes HTTPException as an OpenAI-style
    // {"error": {"message", "type", "code"}} envelope, not FastAPI's default
    // top-level detail; a helper reading only `detail` would return null and
    // the toast would stay generic.
    const detail = await extractErrorDetail(jsonResponse({
      error: {
        message: "Access to 'meta-llama/gated' is restricted and this node sent no Hugging Face token.",
        type: 'Bad Request',
        code: 400,
      },
    }, 400));

    expect(detail).toContain('Hugging Face token');
  });

  it('falls back to a top-level FastAPI detail string', async () => {
    const detail = await extractErrorDetail(jsonResponse({
      detail: 'Requested immutable card is not available for this alias',
    }, 409));

    expect(detail).toContain('immutable card');
  });

  it('returns null for a body without usable detail in either shape', async () => {
    expect(await extractErrorDetail(jsonResponse({ detail: 42 }, 400))).toBeNull();
    expect(await extractErrorDetail(jsonResponse({ error: { message: '  ' } }, 400))).toBeNull();
    expect(await extractErrorDetail(new Response('oops', { status: 500 }))).toBeNull();
  });
});
