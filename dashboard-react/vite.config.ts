/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vite.dev/config/
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { storybookTest } from '@storybook/addon-vitest/vitest-plugin';
import { playwright } from '@vitest/browser-playwright';
const dirname = typeof __dirname !== 'undefined' ? __dirname : path.dirname(fileURLToPath(import.meta.url));

/**
 * The Skulk version shown in the dashboard header. Read from the repo root
 * pyproject.toml at build time so there is exactly ONE version source of
 * truth: the release process bumps pyproject and the UI follows. (The
 * dashboard package.json version was the source before and drifted
 * silently through two releases.)
 */
function skulkVersion(): string {
  const pyproject = readFileSync(path.resolve(dirname, '../pyproject.toml'), 'utf8');
  const match = pyproject.match(/^version\s*=\s*"([^"]+)"/m);
  if (!match) throw new Error('could not read version from ../pyproject.toml');
  return match[1];
}

// More info at: https://storybook.js.org/docs/next/writing-tests/integrations/vitest-addon
export default defineConfig({
  define: {
    __APP_VERSION__: JSON.stringify(skulkVersion()),
  },
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/state': 'http://localhost:52415',
      '/config': 'http://localhost:52415',
      '/store': 'http://localhost:52415',
      '/download': 'http://localhost:52415',
      '/models': 'http://localhost:52415',
      '/place_instance': 'http://localhost:52415',
      '/v1': {
        target: 'http://localhost:52415',
        ws: true,
      },
      '/instance': 'http://localhost:52415',
    },
  },
  test: {
    projects: [{
      extends: true,
      test: {
        name: 'unit',
        include: ['src/**/*.test.{ts,tsx}'],
        browser: {
          enabled: true,
          headless: true,
          provider: playwright({}),
          instances: [{
            browser: 'chromium'
          }]
        }
      }
    }, {
      extends: true,
      plugins: [
      // The plugin will run tests for the stories defined in your Storybook config
      // See options at: https://storybook.js.org/docs/next/writing-tests/integrations/vitest-addon#storybooktest
      storybookTest({
        configDir: path.join(dirname, '.storybook')
      })],
      test: {
        name: 'storybook',
        exclude: ['src/**/*.test.{ts,tsx}'],
        browser: {
          enabled: true,
          headless: true,
          provider: playwright({}),
          instances: [{
            browser: 'chromium'
          }]
        }
      }
    }]
  }
});
