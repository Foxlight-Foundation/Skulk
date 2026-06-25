# Tolgee Import

The dashboard keeps its English Tolgee import file at:

```text
src/i18n/en/skulk.json
```

This file uses Tolgee native JSON with ICU-style placeholders and belongs to the
`skulk` namespace. It is the English fallback bundled into the app, and it is
also the file to upload when seeding the Tolgee project.

Regenerate the file after adding or changing localized dashboard strings:

```bash
npm run tolgee:export
```

The exporter scans dashboard source files for `t("key", "English fallback")`
calls, rejects missing fallbacks, rejects conflicting fallback text for the same
key, and writes sorted key/value pairs to `src/i18n/en/skulk.json`.

## Import into Tolgee

1. Open the Tolgee project and go to **Import**.
2. Upload `dashboard-react/src/i18n/en/skulk.json`.
3. Select language `en`.
4. Select namespace `skulk`.
5. Import the file.

When publishing translations back to the dashboard CDN, export/publish each
language in the same namespace path expected by the runtime:

```text
{VITE_TOLGEE_CDN_PREFIX}/skulk/{language}.json
```

The default prefix is `/i18n`, so the default English-compatible CDN shape is
`/i18n/skulk/en.json`. Other languages use the same namespace path, for example
`/i18n/skulk/es.json`.
