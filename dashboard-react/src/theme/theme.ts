export const theme = {
  colors: {
    bg: '#000000',
    surface: '#111111',
    surfaceHover: '#1a1a1a',
    border: '#222222',
    borderLight: '#333333',
    text: '#ffffff',
    textSecondary: '#999999',
    textMuted: '#666666',
    accent: '#22c55e',
    accentHover: '#16a34a',
    error: '#ef4444',
    warning: '#f59e0b',
    info: '#3b82f6',
  },
  fonts: {
    body: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    mono: "'JetBrains Mono', 'Fira Code', monospace",
  },
  radii: {
    sm: '4px',
    md: '8px',
    lg: '12px',
    xl: '16px',
  },
  spacing: {
    xs: '4px',
    sm: '8px',
    md: '16px',
    lg: '24px',
    xl: '32px',
  },
} as const;

export type Theme = typeof theme;
