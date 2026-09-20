import { FixtureProvider, configureFixtureScreen } from './fixtures';
import type { Preview } from '@storybook/react-vite';
import { createGlobalStyle, ThemeProvider } from 'styled-components';
import { TolgeeProvider, Tolgee, FormatSimple } from '@tolgee/react';
import { darkTheme, lightTheme, GlobalStyle } from '../src/theme';
import english from '../src/i18n/en/skulk.json';

// Screen reviews use bundled English without the in-context editor's DOM
// observer or browser language persistence. Both can mutate fixture content.
const galleryTranslations = Tolgee().use(FormatSimple()).init({ language: 'en', defaultLanguage: 'en', defaultNs: 'skulk', ns: ['skulk'], staticData: { 'en:skulk': english } });
import { useEffect, type ReactNode } from 'react';

type ThemeName = 'light' | 'dark';

// The specimen canvas uses the reference void, independently of app scene effects.
const GalleryCanvas = createGlobalStyle<{ $screen: boolean }>`
  body { background: ${({ theme, $screen }) => $screen ? theme.colors.bgGradient : theme.colors.bg}; }
  #storybook-root { min-width: 0; max-width: 100%; }
  .sb-main-centered #storybook-root { width: 100%; box-sizing: border-box; }
`;

/** Hooks must run inside a React component, not directly in a Storybook
 *  decorator function — extract the side effect into a wrapper component. */
// eslint-disable-next-line react-refresh/only-export-components -- Storybook exports configuration; the local decorator supplies its theme context.
const ThemeWrapper = ({ themeName, children, screen }: { themeName: ThemeName; children: ReactNode; screen: boolean }) => {
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', themeName);
  }, [themeName]);
  const activeTheme = themeName === 'light' ? lightTheme : darkTheme;
  return (
    <ThemeProvider theme={activeTheme}>
      <GlobalStyle />
      <GalleryCanvas $screen={screen} />
      {/* Components use Tolgee's t(); provide the instance so i18n-using stories
          render instead of throwing "no TolgeeProvider". Falls back to the
          bundled English namespace. */}
      <TolgeeProvider tolgee={galleryTranslations} fallback={null}>
        <div style={{ color: activeTheme.colors.body }}>{children}</div>
      </TolgeeProvider>
    </ThemeProvider>
  );
};

const withTheme = (Story: () => ReactNode, context: { globals: { theme?: string }; parameters: { screenRoute?: string } }) => {
  const themeName: ThemeName = context.globals.theme === 'light' ? 'light' : 'dark';
  return (
    <ThemeWrapper themeName={themeName} screen={!!context.parameters.screenRoute}>
      <Story />
    </ThemeWrapper>
  );
};

const preview: Preview = {
  loaders: [context => { configureFixtureScreen(!!context.parameters.screenRoute, !!context.parameters.telemetryConsent, !!context.parameters.pluginInventoryUnavailable); return {}; }],
  decorators: [withTheme, (Story, context) => <FixtureProvider screenRoute={context.parameters.screenRoute} storyId={context.id} theme={context.globals.theme === 'light' ? 'light' : 'dark'}><Story /></FixtureProvider>],
  globalTypes: {
    theme: {
      name: 'Theme',
      description: 'Color theme',
      defaultValue: 'dark',
      toolbar: {
        icon: 'circlehollow',
        items: [
          { value: 'dark', icon: 'circle', title: 'Night' },
          { value: 'light', icon: 'circlehollow', title: 'Noon Ridge' },
        ],
        dynamicTitle: true,
      },
    },
  },
  parameters: {
    backgrounds: { disable: true },
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
