// For more info, see https://github.com/storybookjs/eslint-plugin-storybook#configuration-flat-config-format
import storybook from "eslint-plugin-storybook";

import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([globalIgnores(['dist', 'storybook-static']), {
  files: ['**/*.{ts,tsx}'],
  extends: [
    js.configs.recommended,
    tseslint.configs.recommended,
    reactHooks.configs.flat.recommended,
    reactRefresh.configs.vite,
  ],
  languageOptions: {
    ecmaVersion: 2020,
    globals: globals.browser,
  },
  rules: {
    // crypto.randomUUID exists only in secure contexts, so it is missing on a
    // dashboard opened over plain HTTP from another machine on the LAN.
    'no-restricted-syntax': ['error', {
      selector: "CallExpression[callee.property.name='randomUUID']",
      message: 'randomUUID is missing on plain-HTTP LAN dashboards; use randomHex32 from utils/randomIds (or generateCallId for a UUID).',
    }],
  },
}, ...storybook.configs["flat/recommended"]])
