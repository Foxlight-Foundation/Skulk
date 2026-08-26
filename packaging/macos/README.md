# Frozen macOS runtime probe

This directory builds the first-party macOS runtime used to prove that the
process opening Skulk's local-network sockets has a Skulk-owned application
identity. It is a packaging probe, not the eventual desktop supervisor.

## Build

Requirements are Apple Silicon macOS, `uv`, Node/npm, Xcode command-line tools,
and `mactop` 2.1.5 or newer on `PATH`. Earlier `mactop` releases can trigger an
unnecessary Screen Recording request while running headlessly.

```bash
./packaging/macos/build-frozen-app.sh
```

The ignored artifact is written to `dist/macos/Skulk.app`. Rebuild without
reinstalling dashboard dependencies when its existing output is current:

```bash
./packaging/macos/build-frozen-app.sh --skip-dashboard
```

The bundle uses `packaging/macos/Skulk.icns`, derived from the canonical
1024-pixel Skulk application artwork.

The default bundle identifier is
`foundation.foxlight.skulk.desktop.probe`. Release automation may provide
`SKULK_MACOS_BUNDLE_IDENTIFIER`, `SKULK_MACOS_BUNDLE_VERSION`, and
`SKULK_CODESIGN_IDENTITY`. Without a signing identity, PyInstaller creates an
ad-hoc signature suitable only for local validation. A Developer ID build must
also pass notarization and Gatekeeper checks before distribution.

macOS binds Local Network privacy grants to the application's code
requirement. Rebuilding an ad-hoc-signed app changes that requirement: the
**Skulk** row may remain visible in System Settings while the replacement
binary is denied. Use a stable Developer ID identity for repeatable permission
and release qualification; never interpret a stale enabled toggle as proof.

## Isolated runtime probe

Never launch this artifact in a shared cluster's default namespace. A local
probe should use a dedicated data root, namespace, API port, and offline mode;
disable the Zenoh data plane unless that transport is itself under test.

```bash
open -n -g -W \
  --env SKULK_HOME=/tmp/skulk-desktop-probe-data \
  --env SKULK_LIBP2P_NAMESPACE=desktop-macos-probe \
  --env SKULK_ZENOH_DATA_PLANE=0 \
  dist/macos/Skulk.app \
  --args --api-port 53415 --libp2p-port 0 --offline --no-downloads
```

The expected permission identity is **Skulk** in System Settings under
Privacy & Security -> Local Network. Do not change privacy settings
programmatically.
