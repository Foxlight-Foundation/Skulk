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
  warning: string;
  warningBg: string;
  warningText: string;
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
  meshLine: string;
  meshNode: string;

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
  warning: '#f59e0b',
  warningBg: 'rgba(245, 158, 11, 0.12)',
  warningText: '#fcd34d',
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
  meshLine: 'rgba(255, 215, 0, 0.35)',
  meshNode: 'rgba(255, 215, 0, 0.6)',

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
  warning: '#b45309',
  warningBg: 'rgba(180, 83, 9, 0.12)',
  warningText: '#78350f',
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
  meshLine: 'rgba(29, 78, 216, 0.30)',
  meshNode: 'rgba(29, 78, 216, 0.55)',

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
