const LEGACY_ORIGIN = "https://foxlight-foundation.github.io";

/** Map only mirrored product pages; TypeDoc remains owned by GitHub Pages. */
function preferredCanonical(pathname, origin) {
  if (origin !== LEGACY_ORIGIN || !/^\/Skulk(?:\/|$)/.test(pathname))
    return null;
  const suffix = pathname.replace(/^\/Skulk\/?/, "").replace(/\/+$/, "");
  if (/^(?:next\/)?(?:typedoc(?:\/|$)|404\.html$)/.test(suffix)) return null;
  return `https://docs.foxlight.ai/skulk/${suffix ? suffix + "/" : ""}`;
}

module.exports = { preferredCanonical };
