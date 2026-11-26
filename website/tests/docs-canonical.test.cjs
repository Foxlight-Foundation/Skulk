const { test } = require("node:test");
const assert = require("node:assert/strict");
const { preferredCanonical } = require("../docs-canonical.cjs");
const legacy = "https://foxlight-foundation.github.io";

test("maps stable and next pages to their matching preferred channel", () => {
  for (const [source, expected] of [
    ["/Skulk/", "/skulk/"],
    ["/Skulk/install", "/skulk/install/"],
    ["/Skulk/next/", "/skulk/next/"],
    ["/Skulk/next/install/", "/skulk/next/install/"],
    ["/Skulk/next/api/list-models-get", "/skulk/next/api/list-models-get/"],
  ])
    assert.equal(
      preferredCanonical(source, legacy),
      "https://docs.foxlight.ai" + expected,
    );
});

test("keeps non-mirrored references and other deployments independent", () => {
  for (const path of [
    "/Skulk/typedoc/",
    "/Skulk/typedoc/classes/Store.html",
    "/Skulk/next/typedoc/",
    "/Skulk/404.html",
    "/Other/",
  ]) {
    assert.equal(preferredCanonical(path, legacy), null);
  }
  assert.equal(
    preferredCanonical("/skulk/install/", "https://docs.foxlight.ai"),
    null,
  );
  assert.equal(
    preferredCanonical("/Skulk/install/", "https://preview.example.com"),
    null,
  );
});
