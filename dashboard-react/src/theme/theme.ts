/**
 * Theme palettes for the Skulk dashboard.
 *
 * Two palettes (`darkTheme`, `lightTheme`) share the same `Theme` shape so
 * components reference tokens by name and the active palette swaps the values.
 * Components must never branch on theme name — all variation lives here.
 */

import valleyNight from '../assets/scene/valley-night.webp';

/**
 * Build-time opt-in for the night-sky scene (`VITE_NIGHT_SKY=1`): the star
 * field from the brand valley painting crowns dark mode, shooting stars
 * included, and the abstract mesh stands down. Without the flag, dark mode
 * ships the plain CSS night. Everything downstream keys off the `scene`
 * token, so the flag decides it in exactly one place.
 */
const nightSkyEnabled = import.meta.env.VITE_NIGHT_SKY === '1';

const sharedFonts = {
  body: "'Instrument Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
  mono: "'JetBrains Mono', 'Fira Code', monospace",
} as const;

const sharedFontSizes = {
  xs: '12px',
  sm: '14px',
  md: '15px',
  lg: '18px',
  xl: '22px',
  xxl: '34px',
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
  /** Studio Night semantic surfaces and interaction states. */
  body: string;
  borderControl: string;
  borderDanger: string;
  borderHealthy: string;
  borderLive: string;
  deepAccent: string;
  focusRing: string;
  idle: string;
  liveDeep: string;
  liveHover: string;
  liveStrong: string;
  pressed: string;
  selected: string;
  shadowCard: string;
  shadowPop: string;

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
  /** Muted specimen text; aliases the fourth text level. */
  subtleText: string;
  textOnAccent: string; // text drawn on top of the accent/gold/error fills

  // Brand
  gold: string;
  goldDim: string;
  goldBg: string;
  /** Everyday accent text; aliases starlight. */
  accentText: string;
  actionFill: string;
  approvalFill: string;
  liveText: string;
  goldStrong: string; // readable on goldBg
  /**
   * De-emphasized accent-family FOREGROUND text (metric values, thinking
   * headers, code-language labels). Unlike `goldDim`, which dims far enough
   * to serve as borders and glows, this stays readable at small sizes.
   */
  goldTextDim: string;

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
  /** Raised node surface from skulk-app's native topology palette. */
  topologyNodeSurface: string;
  /** Primary-colour memory fill, rendered at 42% opacity by the node. */
  topologyNodeMemory: string;
  topologyNodeComputeTrack: string;
  topologyNodeCompute: string;
  topologyNodeSelection: string;
  topologyNodeText: string;
  topologyNodeLabel: string;
  topologyNodeDetail: string;
  topologyNodeHealthy: string;
  topologyNodeSyncing: string;
  topologyNodeWarning: string;
  topologyNodeDanger: string;
  topologyNodeDotBorder: string;
  /** Native topology hue at desktop-readable contrast. */
  topologyConnectionLine: string;
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

  /**
   * 'on' to wrap topology/placement SVG marks in their soft glow filters
   * (reads as luminance on a dark canvas); 'none' where the same filter
   * reads as a hard drop-shadow (light surfaces). Replaces the old habit of
   * sniffing `bg === '#000000'`, which broke the moment the canvas moved.
   */
  svgGlow: string;

  /**
   * Vision capability-chip tint. Palette-aware because the chip renders
   * 10px text over its own faint tint: the night palette can afford a
   * pastel cyan, while daylight needs a deep cyan to stay readable on
   * white. Deliberately not the warning token (amber must never repeat
   * down a list as decoration).
   */
  tagVision: string;

  // Status (always-on, palette-independent severity colors are ok inside semantic.*)
  healthy: string;
  unhealthy: string;
}

const darkColors: ColorTokens = {
  bg: "#070a14",
  bgGradient: "radial-gradient(ellipse 90% 55% at 50% -12%,rgba(43,58,99,.55) 0%,transparent 62%),radial-gradient(ellipse 68% 38% at 82% 108%,rgba(168,86,12,.28) 0%,transparent 66%),#070a14",
  surface: "#0d1226",
  surfaceHover: "#131a33",
  surfaceElevated: "rgba(13,18,38,.96)",
  surfaceSunken: "rgba(147,174,223,.08)",
  header: "rgba(7,10,20,.78)",
  overlay: "rgba(5,7,15,.65)",
  border: "rgba(147,174,223,.14)",
  borderLight: "rgba(147,174,223,.10)",
  borderStrong: "rgba(147,174,223,.30)",
  text: "#e8edf7",
  textSecondary: "#8a9ab8",
  textMuted: "#6c7ea3",
  subtleText: "#6c7ea3",
  textOnAccent: "#070a14",
  gold: "#93aedf",
  goldDim: "rgba(147,174,223,.30)",
  goldTextDim: "#93aedf",
  goldBg: "rgba(147,174,223,.12)",
  accentText: "#93aedf",
  actionFill: "#93aedf",
  approvalFill: "#f2a03d",
  liveText: "#f2a03d",
  goldStrong: "#4d7cc4",
  live: "#f2a03d",
  liveBg: "rgba(242,160,61,.08)",
  onLive: "#1b1200",
  accent: "#54c79a",
  accentHover: "#54c79a",
  accentBg: "rgba(84,199,154,.12)",
  error: "#e5655f",
  errorBg: "rgba(229,101,95,.10)",
  errorText: "#e5655f",
  errorFill: "#e5655f",
  errorOnFill: "#fff",
  errorOnSurface: "#e5655f",
  warning: "#f2a03d",
  warningBg: "rgba(242,160,61,.08)",
  warningText: "#f2a03d",
  warningFill: "#f2a03d",
  warningOnFill: "#1b1200",
  warningOnSurface: "#f2a03d",
  info: "#93aedf",
  infoBg: "rgba(147,174,223,.12)",
  chatBubbleUser: "#16203f",
  chatBubbleAssistant: "rgba(147,174,223,.08)",
  chatBubbleBorder: "rgba(147,174,223,.14)",
  chatCodeBg: "#070a14",
  heatmapLow: "#16203f",
  heatmapMid: "#93aedf",
  heatmapHigh: "#f2a03d",
  deviceIconStroke: "#e8edf7",
  deviceIconFill: "rgba(147,174,223,.08)",
  deviceBody: "#131a33",
  ramFill: "#f2a03d",
  deviceLabel: "#e8edf7",
  gpuBarBg: "#2b3a63",
  meshLine: "rgba(147,174,223,.30)",
  meshNode: "#93aedf",
  topologyNodeSurface: "#131a33",
  topologyNodeMemory: "#93aedf",
  topologyNodeComputeTrack: "rgba(147,174,223,.18)",
  topologyNodeCompute: "#54c79a",
  topologyNodeSelection: "#93aedf",
  topologyNodeText: "#e8edf7",
  topologyNodeLabel: "#8a9ab8",
  topologyNodeDetail: "#6c7ea3",
  topologyNodeHealthy: "#54c79a",
  topologyNodeSyncing: "#93aedf",
  topologyNodeWarning: "#f2a03d",
  topologyNodeDanger: "#e5655f",
  topologyNodeDotBorder: "#0d1226",
  topologyConnectionLine: "#93aedf",
  bgMeshLine: "rgba(147,174,223,.10)",
  bgMeshNode: "rgba(147,174,223,.10)",
  tagVision: "#22d3ee",
  healthy: "#54c79a",
  unhealthy: "#e5655f",
  body: "#c9d3e8",
  idle: "#4c5a7a",
  selected: "#16203f",
  deepAccent: "#2b3a63",
  borderControl: "rgba(147,174,223,.18)",
  borderLive: "rgba(242,160,61,.35)",
  borderDanger: "rgba(229,101,95,.40)",
  borderHealthy: "rgba(84,199,154,.40)",
  pressed: "rgba(147,174,223,.14)",
  liveStrong: "rgba(242,160,61,.14)",
  liveHover: "#f7b04f",
  liveDeep: "#a8560c",
  shadowCard: "0 14px 34px rgba(0,0,0,.35)",
  shadowPop: "0 22px 50px rgba(0,0,0,.55)",
  focusRing: "0 0 0 3px rgba(147,174,223,.22)",
  headerBorder: "linear-gradient(to right, rgba(147,174,223,.30), rgba(147,174,223,.10))",
  shadow: "rgba(0,0,0,.35)",
  shadowStrong: "rgba(0,0,0,.55)",
  sceneScrim: "none",
  svgGlow: "on",
  scene: nightSkyEnabled ? `url(${valleyNight})` : 'none',
};

const lightColors: ColorTokens = {
  bg: "#eef4fb",
  bgGradient: "radial-gradient(120% 52% at 74% 0%,rgba(120,168,228,.42) 0%,rgba(238,244,251,0) 64%),#eef4fb",
  surface: "#ffffff",
  surfaceHover: "#f5f8fc",
  surfaceElevated: "rgba(255,255,255,.96)",
  surfaceSunken: "rgba(17,33,60,.05)",
  header: "rgba(255,255,255,.78)",
  overlay: "rgba(17,33,60,.42)",
  border: "rgba(17,33,60,.12)",
  borderLight: "rgba(17,33,60,.08)",
  borderStrong: "rgba(17,33,60,.30)",
  text: "#11213c",
  textSecondary: "#5f7086",
  textMuted: "#7a8aa3",
  subtleText: "#7a8aa3",
  textOnAccent: "#ffffff",
  gold: "#4d7cc4",
  goldDim: "rgba(17,33,60,.30)",
  goldTextDim: "#4d7cc4",
  goldBg: "rgba(77,124,196,.12)",
  accentText: "#4d7cc4",
  actionFill: "#4d7cc4",
  approvalFill: "#b35c0a",
  liveText: "#b35c0a",
  goldStrong: "#1c2b4a",
  live: "#b35c0a",
  liveBg: "rgba(179,92,10,.08)",
  onLive: "#fff6ea",
  accent: "#1c7a54",
  accentHover: "#1c7a54",
  accentBg: "rgba(28,122,84,.10)",
  error: "#b23a44",
  errorBg: "rgba(178,58,68,.10)",
  errorText: "#b23a44",
  errorFill: "#b23a44",
  errorOnFill: "#fff",
  errorOnSurface: "#b23a44",
  warning: "#b35c0a",
  warningBg: "rgba(179,92,10,.08)",
  warningText: "#b35c0a",
  warningFill: "#b35c0a",
  warningOnFill: "#fff6ea",
  warningOnSurface: "#b35c0a",
  info: "#4d7cc4",
  infoBg: "rgba(77,124,196,.12)",
  chatBubbleUser: "#e6eef9",
  chatBubbleAssistant: "rgba(17,33,60,.05)",
  chatBubbleBorder: "rgba(17,33,60,.12)",
  chatCodeBg: "#eef4fb",
  heatmapLow: "#e6eef9",
  heatmapMid: "#4d7cc4",
  heatmapHigh: "#b35c0a",
  deviceIconStroke: "#11213c",
  deviceIconFill: "rgba(17,33,60,.05)",
  deviceBody: "#f5f8fc",
  ramFill: "#b35c0a",
  deviceLabel: "#11213c",
  gpuBarBg: "#c9d9f0",
  meshLine: "rgba(17,33,60,.30)",
  meshNode: "#4d7cc4",
  topologyNodeSurface: "#f5f8fc",
  topologyNodeMemory: "#4d7cc4",
  topologyNodeComputeTrack: "rgba(17,33,60,.16)",
  topologyNodeCompute: "#1c7a54",
  topologyNodeSelection: "#4d7cc4",
  topologyNodeText: "#11213c",
  topologyNodeLabel: "#5f7086",
  topologyNodeDetail: "#7a8aa3",
  topologyNodeHealthy: "#1c7a54",
  topologyNodeSyncing: "#4d7cc4",
  topologyNodeWarning: "#b35c0a",
  topologyNodeDanger: "#b23a44",
  topologyNodeDotBorder: "#ffffff",
  topologyConnectionLine: "#4d7cc4",
  bgMeshLine: "rgba(77,124,196,.16)",
  bgMeshNode: "rgba(77,124,196,.16)",
  tagVision: "#0e7490",
  healthy: "#1c7a54",
  unhealthy: "#b23a44",
  body: "#2c3b52",
  idle: "#a2afc4",
  selected: "#e6eef9",
  deepAccent: "#c9d9f0",
  borderControl: "rgba(17,33,60,.16)",
  borderLive: "rgba(179,92,10,.40)",
  borderDanger: "rgba(178,58,68,.40)",
  borderHealthy: "rgba(28,122,84,.40)",
  pressed: "rgba(17,33,60,.09)",
  liveStrong: "rgba(179,92,10,.14)",
  liveHover: "#c9700f",
  liveDeep: "#8a4406",
  shadowCard: "0 10px 26px rgba(28,43,74,.12)",
  shadowPop: "0 22px 50px rgba(28,43,74,.22)",
  focusRing: "0 0 0 3px rgba(77,124,196,.25)",
  headerBorder: "linear-gradient(to right, rgba(17,33,60,.30), rgba(17,33,60,.08))",
  shadow: "rgba(28,43,74,.12)",
  shadowStrong: "rgba(28,43,74,.22)",
  sceneScrim: "none",
  svgGlow: "none",
  scene: 'none',
};

function buildTheme(colors: ColorTokens) {
  return {
    colors,
    fonts: sharedFonts,
    // The capability specimens intentionally retain their distinct tints in both palettes.
    capabilityTints: {
      optiq: '#a78bfa', embedding: '#f472b6', tts: '#38bdf8', stt: '#34d399',
      code: '#818cf8', image_gen: '#fb923c', image_edit: '#fb923c',
    },
    fontSizes: sharedFontSizes,
    radii: sharedRadii,
    spacing: sharedSpacing,
    motion: { fast: '120ms', normal: '200ms', slow: '320ms', easing: 'cubic-bezier(.2,.7,.2,1)' },
  } as const;
}

export const darkTheme = buildTheme(darkColors);
export const lightTheme = buildTheme(lightColors);

export type ThemeName = 'light' | 'dark';
export type Theme = typeof darkTheme;

/** @deprecated Use `darkTheme` directly. Kept for backward-compat imports. */
export const theme = darkTheme;
