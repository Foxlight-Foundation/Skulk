import type { Preview } from '@storybook/react-vite';
import { ThemeProvider } from 'styled-components';
import { darkTheme, lightTheme, GlobalStyle } from '../src/theme';
import { useEffect, type ReactNode } from 'react';

type ThemeName = 'light' | 'dark';

/** Hooks must run inside a React component, not directly in a Storybook
 *  decorator function — extract the side effect into a wrapper component. */
const ThemeWrapper = ({ themeName, children }: { themeName: ThemeName; children: ReactNode }) => {
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', themeName);
  }, [themeName]);
  const activeTheme = themeName === 'light' ? lightTheme : darkTheme;
  return (
    <ThemeProvider theme={activeTheme}>
      <GlobalStyle />
      {children}
    </ThemeProvider>
  );
};

const withTheme = (Story: () => ReactNode, context: { globals: { theme?: string } }) => {
  const themeName: ThemeName = context.globals.theme === 'light' ? 'light' : 'dark';
  return (
    <ThemeWrapper themeName={themeName}>
      <Story />
    </ThemeWrapper>
  );
};

const preview: Preview = {
  decorators: [withTheme],
  globalTypes: {
    theme: {
      name: 'Theme',
      description: 'Color theme',
      defaultValue: 'dark',
      toolbar: {
        icon: 'circlehollow',
        items: [
          { value: 'dark', icon: 'circle', title: 'Dark' },
          { value: 'light', icon: 'circlehollow', title: 'Light' },
        ],
        dynamicTitle: true,
      },
    },
  },
  parameters: {
    backgrounds: {
      default: 'dark',
      values: [
        { name: 'dark', value: '#000000' },
        { name: 'light', value: '#f5f4ef' },
      ],
    },
    controls: {
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/i,
      },
    },
    a11y: {
      test: 'todo',
    },
  },
};

export default preview;
