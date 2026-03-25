import type { Preview } from '@storybook/react-vite';
import { ThemeProvider } from 'styled-components';
import { theme, GlobalStyle } from '../src/theme';
import type { ReactNode } from 'react';

const withTheme = (Story: () => ReactNode) => (
  <ThemeProvider theme={theme}>
    <GlobalStyle />
    <Story />
  </ThemeProvider>
);

const preview: Preview = {
  decorators: [withTheme],
  parameters: {
    backgrounds: {
      default: 'dark',
      values: [{ name: 'dark', value: '#000000' }],
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
