/**
 * Theme palettes for the Skulk dashboard.
 *
 * Two palettes (`darkTheme`, `lightTheme`) share the same `Theme` shape so
 * components reference tokens by name and the active palette swaps the values.
 * Components must never branch on theme name — all variation lives here.
 */

import valleyNight from '../assets/scene/valley-night.webp';

const sharedFonts = {
  body: "'Outfit', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
  mono: "'JetBrains Mono', 'Fira Code', monospace",
} as const;

const sharedFontSizes = {
  xs: '12px',
  sm: '14px',
  md: '16px',
  lg: '18px',
  xl: '22px',
  xxl: '30px',
  label: '13px',
  tableHead: '13px',
  tableBody: '15px',
  nav: '14px',
} as const;

const sharedRadii = {
  sm: '4px',
  md: '8px',
  lg: '12px',
  xl: '16px',
} as const;

const sharedSpacing = {
  xs: '4px',
  sm: '8px',
  md: '16px',
  lg: '24px',
  xl: '32px',
} as const;

/** Color tokens. Both palettes must define every key. */
interface ColorTokens {
  // Surfaces
  bg: string;
  bgGradient: string; // full `background:` value for body
  surface: string;
  surfaceHover: string;
  surfaceElevated: string;
  surfaceSunken: string;
  header: string;
  headerBorder: string;
  overlay: string;
  shadow: string;
  shadowStrong: string;

  // Borders
  border: string;
  borderLight: string;
  borderStrong: string;

  // Text
  text: string;
  textSecondary: string;
  textMuted: string;
  textOnAccent: string; // text drawn on top of the accent/gold/error fills

  // Brand
  gold: string;
  goldDim: string;
  goldBg: string;
  goldStrong: string; // readable on goldBg

  /**
   * The living colour (Den design language): marks work actually in flight —
   * a loading instance, a download in progress, RAM a model is holding.
   * Never decoration; if it appears in three places on a screen, two are
   * wrong. Distinct from `gold`, the everyday interactive accent.
   */
  live: string;
  liveBg: string;
  onLive: string;

  // Semantic
  accent: string;
  accentHover: string;
  accentBg: string;
  error: string;
  errorBg: string;
  errorText: string;
  // Error callout fill (saturated solid bg) + on-fill foreground. Use these
  // when the message is meant to *grab* attention (cluster warnings, blocking
  // toasts) rather than tint a surface. Identical in light and dark.
  errorFill: string;
  errorOnFill: string;
  /**
   * Body-text-on-a-regular-surface color for errors. Palette-aware so the
   * text reads as "error" in both modes without needing a saturated fill.
   * Distinct from `errorText` (used in section banners) — `errorOnSurface`
   * is intended for callout lists like the cluster-warnings popover.
   */
  errorOnSurface: string;
  warning: string;
  warningBg: string;
  warningText: string;
  // Warning callout fill + on-fill foreground. Same intent as errorFill but
  // for non-blocking advisories. Identical in light and dark.
  warningFill: string;
  warningOnFill: string;
  /**
   * Body-text-on-a-regular-surface color for warnings. Stays semantically
   * "amber" in both modes (unlike `warningText`, which is intentionally
   * slate-grey in the light palette to avoid clashing with the blue brand
   * inside section banners).
   */
  warningOnSurface: string;
  info: string;
  infoBg: string;

  // Chat surfaces
  chatBubbleUser: string;
  chatBubbleAssistant: string;
  chatBubbleBorder: string;
  chatCodeBg: string;

  // Heatmap (token visualization)
  heatmapLow: string;
  heatmapMid: string;
  heatmapHigh: string;

  // Topology / SVG
  deviceIconStroke: string;
  deviceIconFill: string;
  deviceBody: string; // background fill of the device "case" in the topology icon
  ramFill: string;    // RAM-fullness fill drawn on top of deviceBody
  deviceLabel: string; // wordmark drawn on a device front (e.g. "AMD"), theme-aware
  gpuBarBg: string;   // empty/background fill of the GPU stats bar
  meshLine: string;
  meshNode: string;
  // Fullscreen background NetworkMesh — must be much subtler than the topology mesh.
  bgMeshLine: string;
  bgMeshNode: string;

  /**
   * Scene image behind the app (a CSS background-image value), or 'none'.
   * The night palette crowns the viewport with the star field from the
   * brand valley painting (shared with foxlight.ai and the operator app),
   * fading to nothing on the way down; a palette without a scene keeps the
   * abstract mesh instead. Components branch on this token's value, never
   * on the theme name.
   */
  scene: string;
  /** Structural scrim over the scene: sinks the top for the header and
   * buries the base so dense content keeps its ground. */
  sceneScrim: string;

  // Status (always-on, palette-independent severity colors are ok inside semantic.*)
  healthy: string;
  unhealthy: string;
}

const darkColors: ColorTokens = {
  // Den (night on the ridge), from the operator app design system: indigo
  // surfaces over a deep night canvas, starlight for hairlines and the
  // everyday accent, amber strictly for whatever is alive. Values track
  // skulk-app/src/theme/tokens.ts (the 1a Den palette).
  bg: '#080C1A',
  // A CSS-only night: den glow crowning the header, and the fire's warmth
  // (the palette's horizon amber) breathing up from just below the frame,
  // so the canvas carries the valley's atmosphere without the painting.
  bgGradient: `
    radial-gradient(ellipse 90% 55% at 50% -12%, rgba(43, 58, 99, 0.55) 0%, transparent 62%),
    radial-gradient(ellipse 68% 38% at 82% 108%, rgba(168, 86, 12, 0.28) 0%, transparent 66%),
    radial-gradient(ellipse 90% 30% at 12% 112%, rgba(43, 58, 99, 0.38) 0%, transparent 58%),
    #080C1A
  `,
  surface: '#12192E',
  surfaceHover: '#1B2540',
  surfaceElevated: 'rgba(16, 22, 42, 0.96)',
  surfaceSunken: 'rgba(43, 58, 99, 0.16)',
  header: 'rgba(8, 12, 26, 0.78)',
  headerBorder: 'linear-gradient(to right, rgba(147, 174, 223, 0.22), rgba(147, 174, 223, 0.03))',
  overlay: 'rgba(5, 7, 15, 0.65)',
  shadow: 'rgba(0, 0, 0, 0.5)',
  shadowStrong: 'rgba(0, 0, 0, 0.7)',

  border: 'rgba(147, 174, 223, 0.13)',
  borderLight: 'rgba(147, 174, 223, 0.10)',
  borderStrong: 'rgba(147, 174, 223, 0.22)',

  text: '#F4F6FB',
  textSecondary: 'rgba(147, 174, 223, 0.78)',
  textMuted: 'rgba(232, 237, 247, 0.52)',
  textOnAccent: '#07101E',

  // The night accent is starlight, not gold: interactive emphasis borrows the
  // Den's secondary accent so amber (see `live`) stays scarce and alive.
  gold: '#93AEDF',
  goldDim: 'rgba(147, 174, 223, 0.45)',
  goldBg: 'rgba(147, 174, 223, 0.12)',
  goldStrong: '#B9CBEC',

  live: '#F2A03D',
  liveBg: 'rgba(242, 160, 61, 0.14)',
  onLive: '#1C1206',

  accent: '#54C79A',
  accentHover: '#3FB287',
  accentBg: 'rgba(84, 199, 154, 0.14)',
  error: '#F2707E',
  errorBg: 'rgba(242, 112, 126, 0.14)',
  errorText: '#F8A9B1',
  errorFill: '#dc2626',
  errorOnFill: '#ffffff',
  errorOnSurface: '#F8A9B1',
  warning: '#F2A03D',
  warningBg: 'rgba(242, 160, 61, 0.13)',
  warningText: '#FFD79A',
  // Solid attention badging is one of the few places amber earns a fill:
  // it marks something the operator must act on, not decoration.
  warningFill: '#F2A03D',
  warningOnFill: '#1C1206',
  warningOnSurface: '#F2A03D',
  info: '#93AEDF',
  infoBg: 'rgba(147, 174, 223, 0.14)',

  chatBubbleUser: 'rgba(43, 58, 99, 0.36)',
  chatBubbleAssistant: 'rgba(43, 58, 99, 0.14)',
  chatBubbleBorder: 'rgba(147, 174, 223, 0.13)',
  chatCodeBg: 'rgba(5, 7, 15, 0.55)',

  heatmapLow: '#16203F',
  heatmapMid: '#93AEDF',
  heatmapHigh: '#F2A03D',

  deviceIconStroke: '#E8EDF7',
  deviceIconFill: 'rgba(147, 174, 223, 0.08)',
  deviceBody: '#131B30',
  // RAM a model is holding is work in flight: the one place the topology
  // legitimately burns amber (the Den's "node actually holding work").
  ramFill: 'rgba(242, 160, 61, 0.75)',
  deviceLabel: '#E8EDF7',
  gpuBarBg: 'rgba(43, 58, 99, 0.65)',
  meshLine: 'rgba(147, 174, 223, 0.30)',
  meshNode: 'rgba(147, 174, 223, 0.55)',
  bgMeshLine: 'rgba(147, 174, 223, 0.10)',
  bgMeshNode: 'rgba(147, 174, 223, 0.08)',

  // The star field crowns the viewport and dissolves on the way down; the
  // CSS night gradient beneath it carries the rest of the atmosphere.
  scene: `url(${valleyNight})`,
  sceneScrim: 'none',

  healthy: '#54C79A',
  unhealthy: '#F2707E',
};

const lightColors: ColorTokens = {
  // Noon Ridge (the valley at midday), from the operator app design system:
  // crisp white cards over an open-sky canvas, the deep-ink primary blue for
  // the everyday accent, and amber (deepened so it reads on white) still
  // reserved for whatever is alive. Values track
  // skulk-app/src/theme/tokens.ts (the 2a Noon palette).
  bg: '#EEF4FB',
  // The same three-stop atmosphere as the night, at noon: open sky crowning
  // the header, the warm horizon breathing up from below the frame, and the
  // far ridge's haze on the left.
  bgGradient: `
    radial-gradient(ellipse 90% 55% at 50% -12%, rgba(120, 168, 228, 0.42) 0%, transparent 62%),
    radial-gradient(ellipse 68% 38% at 82% 108%, rgba(255, 230, 188, 0.55) 0%, transparent 66%),
    radial-gradient(ellipse 90% 30% at 12% 112%, rgba(190, 212, 232, 0.5) 0%, transparent 58%),
    #EEF4FB
  `,
  surface: '#FFFFFF',
  surfaceHover: '#EEF4FB',
  surfaceElevated: 'rgba(255, 255, 255, 0.97)',
  surfaceSunken: 'rgba(17, 33, 60, 0.04)',
  header: 'rgba(238, 244, 251, 0.9)',
  headerBorder: 'linear-gradient(to right, rgba(69, 111, 176, 0.24), rgba(69, 111, 176, 0.04))',
  overlay: 'rgba(17, 33, 60, 0.42)',
  shadow: 'rgba(17, 33, 60, 0.10)',
  shadowStrong: 'rgba(17, 33, 60, 0.20)',

  border: 'rgba(17, 33, 60, 0.10)',
  borderLight: 'rgba(17, 33, 60, 0.07)',
  borderStrong: 'rgba(17, 33, 60, 0.18)',

  text: '#11213C',
  textSecondary: 'rgba(17, 33, 60, 0.74)',
  textMuted: '#5F7086',
  textOnAccent: '#FFFFFF',

  // The dark palette's `gold` family carries the everyday accent; at noon
  // that is the primary blue, so the rest of the codebase doesn't need to
  // know which palette is active.
  gold: '#456FB0',
  goldDim: 'rgba(69, 111, 176, 0.55)',
  goldBg: 'rgba(69, 111, 176, 0.10)',
  goldStrong: '#29446F',

  // Noon identity amber (the deeper value that reads on white).
  live: '#AC580A',
  liveBg: '#FCF2E6',
  onLive: '#FFF6EA',

  accent: '#1C7A54',
  accentHover: '#166245',
  accentBg: '#E1F1EA',
  error: '#B23A44',
  errorBg: '#F8E6E8',
  errorText: '#8E2E36',
  // Solid-callout pair: on-fill contrast (white-on-red) guaranteed on any
  // surface.
  errorFill: '#B23A44',
  errorOnFill: '#FFFFFF',
  errorOnSurface: '#B23A44',
  warning: '#96601A',
  warningBg: '#FAEEDC',
  warningText: '#6B4512',
  // Solid attention badging: the identity amber, deepened for daylight, so
  // "act on this" speaks the same language at both hours.
  warningFill: '#AC580A',
  warningOnFill: '#FFF6EA',
  warningOnSurface: '#96601A',
  info: '#456FB0',
  infoBg: 'rgba(69, 111, 176, 0.10)',

  chatBubbleUser: 'rgba(69, 111, 176, 0.10)',
  chatBubbleAssistant: '#FFFFFF',
  chatBubbleBorder: 'rgba(17, 33, 60, 0.10)',
  chatCodeBg: 'rgba(17, 33, 60, 0.05)',

  heatmapLow: '#DCE8F6',
  heatmapMid: '#456FB0',
  heatmapHigh: '#1C2B4A',

  deviceIconStroke: '#1C2B4A',
  deviceIconFill: 'rgba(69, 111, 176, 0.08)',
  deviceBody: '#DCE8F6',          // near-void sky tint: the "empty RAM" case
  // Memory pressure is the one gauge that legitimately burns amber at noon
  // too — same "node actually holding work" semantic as the night palette.
  ramFill: 'rgba(172, 88, 10, 0.70)',
  deviceLabel: '#5F7086',
  gpuBarBg: '#BED4E8',             // the far ridge's haze; separates the bar from the case
  meshLine: 'rgba(69, 111, 176, 0.30)',
  meshNode: 'rgba(69, 111, 176, 0.55)',
  bgMeshLine: 'rgba(69, 111, 176, 0.14)',
  bgMeshNode: 'rgba(69, 111, 176, 0.10)',

  // Noon has no star field; the mesh carries the daylight atmosphere.
  scene: 'none',
  sceneScrim: 'none',

  healthy: '#1C7A54',
  unhealthy: '#B23A44',
};

function buildTheme(colors: ColorTokens) {
  return {
    colors,
    fonts: sharedFonts,
    fontSizes: sharedFontSizes,
    radii: sharedRadii,
    spacing: sharedSpacing,
  } as const;
}

export const darkTheme = buildTheme(darkColors);
export const lightTheme = buildTheme(lightColors);

export type ThemeName = 'light' | 'dark';
export type Theme = typeof darkTheme;

/** @deprecated Use `darkTheme` directly. Kept for backward-compat imports. */
export const theme = darkTheme;
