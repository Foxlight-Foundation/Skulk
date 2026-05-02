/**
 * Theme palettes for the Skulk dashboard.
 *
 * Two palettes (`darkTheme`, `lightTheme`) share the same `Theme` shape so
 * components reference tokens by name and the active palette swaps the values.
 * Components must never branch on theme name — all variation lives here.
 */

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
  gpuBarBg: string;   // empty/background fill of the GPU stats bar
  meshLine: string;
  meshNode: string;
  // Fullscreen background NetworkMesh — must be much subtler than the topology mesh.
  bgMeshLine: string;
  bgMeshNode: string;

  // Status (always-on, palette-independent severity colors are ok inside semantic.*)
  healthy: string;
  unhealthy: string;
}

const darkColors: ColorTokens = {
  bg: '#000000',
  bgGradient: `
    radial-gradient(ellipse at 0% 0%, #141428 0%, transparent 50%),
    radial-gradient(ellipse at 100% 100%, #141428 0%, transparent 50%),
    #000000
  `,
  surface: '#111111',
  surfaceHover: '#1a1a1a',
  surfaceElevated: 'rgba(17, 17, 17, 0.95)',
  surfaceSunken: 'rgba(0, 0, 0, 0.4)',
  header: 'rgba(5, 2, 31, 0.16)',
  headerBorder: 'linear-gradient(to right, rgba(255, 255, 255, 0.16), rgba(255, 255, 255, 0.03))',
  overlay: 'rgba(0, 0, 0, 0.6)',
  shadow: 'rgba(0, 0, 0, 0.4)',
  shadowStrong: 'rgba(0, 0, 0, 0.6)',

  border: 'rgba(255, 255, 255, 0.21)',
  borderLight: 'rgba(255, 255, 255, 0.18)',
  borderStrong: 'rgba(255, 255, 255, 0.35)',

  text: '#ffffff',
  textSecondary: 'rgba(255, 255, 255, 0.7)',
  textMuted: 'rgba(255, 255, 255, 0.45)',
  textOnAccent: '#000000',

  gold: '#FFD700',
  goldDim: 'rgba(255, 215, 0, 0.5)',
  goldBg: 'rgba(255, 215, 0, 0.08)',
  goldStrong: '#FFD700',

  accent: '#22c55e',
  accentHover: '#16a34a',
  accentBg: 'rgba(34, 197, 94, 0.12)',
  error: '#ef4444',
  errorBg: 'rgba(239, 68, 68, 0.12)',
  errorText: '#fca5a5',
  errorFill: '#dc2626',
  errorOnFill: '#ffffff',
  errorOnSurface: '#fca5a5',                 // light red, readable on dark surface
  warning: '#f59e0b',
  warningBg: 'rgba(245, 158, 11, 0.12)',
  warningText: '#fcd34d',
  warningFill: '#ffcc33',
  warningOnFill: '#000000',
  warningOnSurface: '#fcd34d',               // light amber, readable on dark surface
  info: '#3b82f6',
  infoBg: 'rgba(59, 130, 246, 0.12)',

  chatBubbleUser: 'rgba(255, 215, 0, 0.08)',
  chatBubbleAssistant: 'rgba(255, 255, 255, 0.04)',
  chatBubbleBorder: 'rgba(255, 255, 255, 0.12)',
  chatCodeBg: 'rgba(0, 0, 0, 0.5)',

  heatmapLow: '#1e3a8a',
  heatmapMid: '#FFD700',
  heatmapHigh: '#ef4444',

  deviceIconStroke: '#ffffff',
  deviceIconFill: 'rgba(255, 255, 255, 0.08)',
  deviceBody: '#1a1a1a',
  ramFill: 'rgba(255, 215, 0, 0.75)',
  gpuBarBg: 'rgba(80, 80, 90, 0.7)',
  meshLine: 'rgba(255, 215, 0, 0.35)',
  meshNode: 'rgba(255, 215, 0, 0.6)',
  bgMeshLine: 'rgba(255, 215, 0, 0.21)',
  bgMeshNode: 'rgba(255, 215, 0, 0.10)',

  healthy: '#4ade80',
  unhealthy: '#ef4444',
};

const lightColors: ColorTokens = {
  bg: '#eef3fb',
  bgGradient: `
    radial-gradient(ellipse at 0% 0%, #dbeafe 0%, transparent 50%),
    radial-gradient(ellipse at 100% 100%, #e0e7ff 0%, transparent 50%),
    #eef3fb
  `,
  surface: '#ffffff',
  surfaceHover: '#e6edf8',
  surfaceElevated: 'rgba(255, 255, 255, 0.96)',
  surfaceSunken: 'rgba(15, 23, 42, 0.04)',
  header: 'rgba(255, 255, 255, 0.78)',
  headerBorder: 'linear-gradient(to right, rgba(30, 64, 175, 0.18), rgba(30, 64, 175, 0.03))',
  overlay: 'rgba(15, 23, 42, 0.42)',
  shadow: 'rgba(15, 23, 42, 0.10)',
  shadowStrong: 'rgba(15, 23, 42, 0.18)',

  border: 'rgba(30, 64, 175, 0.16)',
  borderLight: 'rgba(30, 64, 175, 0.10)',
  borderStrong: 'rgba(30, 64, 175, 0.32)',

  text: '#0f172a',
  textSecondary: 'rgba(15, 23, 42, 0.72)',
  textMuted: 'rgba(15, 23, 42, 0.5)',
  textOnAccent: '#ffffff',

  // The dark palette uses gold as the brand accent. Light mode reuses the same
  // token names but maps them to a dominant blue so the rest of the codebase
  // doesn't need to know which palette is active.
  gold: '#1d4ed8',
  goldDim: 'rgba(29, 78, 216, 0.55)',
  goldBg: 'rgba(29, 78, 216, 0.10)',
  goldStrong: '#1e3a8a',

  accent: '#0ea5e9',
  accentHover: '#0284c7',
  accentBg: 'rgba(14, 165, 233, 0.12)',
  error: '#dc2626',
  errorBg: 'rgba(220, 38, 38, 0.10)',
  errorText: '#991b1b',
  // Same solid-callout pair as the dark palette — palette-independent so
  // the on-fill contrast (white-on-red) is guaranteed regardless of mode.
  errorFill: '#dc2626',
  errorOnFill: '#ffffff',
  errorOnSurface: '#b91c1c',                 // red-700, readable on white
  // Light-theme warnings stay greyscale rather than borrowing the amber
  // palette the dark theme uses — amber clashed with the cool blue accents
  // and read as a stain on the surface. The semantic ("this is a warning")
  // is carried by the section heading and the surrounding context; the body
  // just needs to be legible and not draw the eye away from the brand.
  warning: '#475569',                       // slate-600 (border/accent)
  warningBg: 'rgba(71, 85, 105, 0.08)',     // slate-600 at 8%
  warningText: '#1e293b',                   // slate-800
  // Solid-callout pair stays palette-independent — same yellow + black in
  // both modes — because the intent is high-attention badging, not subtle
  // tinting. Mirrors the dark-mode values verbatim.
  warningFill: '#ffcc33',
  warningOnFill: '#000000',
  warningOnSurface: '#b45309',               // amber-700, readable on white
  info: '#1d4ed8',
  infoBg: 'rgba(29, 78, 216, 0.10)',

  chatBubbleUser: 'rgba(29, 78, 216, 0.10)',
  chatBubbleAssistant: '#ffffff',
  chatBubbleBorder: 'rgba(30, 64, 175, 0.16)',
  chatCodeBg: 'rgba(15, 23, 42, 0.06)',

  heatmapLow: '#dbeafe',
  heatmapMid: '#3b82f6',
  heatmapHigh: '#1e3a8a',

  deviceIconStroke: '#1e3a8a',
  deviceIconFill: 'rgba(29, 78, 216, 0.08)',
  deviceBody: '#dbeafe',          // light-blue "empty RAM" case background
  ramFill: 'rgba(29, 78, 216, 0.75)', // darker blue RAM fullness
  gpuBarBg: '#bccfe8',             // a touch darker than the device case so the bar reads as a separate element
  meshLine: 'rgba(29, 78, 216, 0.30)',
  meshNode: 'rgba(29, 78, 216, 0.55)',
  bgMeshLine: 'rgba(29, 78, 216, 0.16)',
  bgMeshNode: 'rgba(29, 78, 216, 0.12)',

  healthy: '#0ea5e9',
  unhealthy: '#dc2626',
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
