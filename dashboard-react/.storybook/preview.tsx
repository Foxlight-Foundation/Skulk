import type { Preview } from '@storybook/react-vite';
import { ThemeProvider } from 'styled-components';
import { darkTheme, lightTheme, GlobalStyle } from '../src/theme';
import { useEffect, type ReactNode } from 'react';

const withTheme = (Story: () => ReactNode, context: { globals: { theme?: string } }) => {
  const themeName = context.globals.theme === 'light' ? 'light' : 'dark';
  const activeTheme = themeName === 'light' ? lightTheme : darkTheme;
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', themeName);
  }, [themeName]);
  return (
    <ThemeProvider theme={activeTheme}>
      <GlobalStyle />
      <Story />
    </ThemeProvider>
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
