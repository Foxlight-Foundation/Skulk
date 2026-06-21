import {
  BackendFetch,
  DevTools,
  FormatSimple,
  LanguageDetector,
  LanguageStorage,
  Tolgee,
  useTranslate,
  type UseTranslateResult,
} from '@tolgee/react';

export const SKULK_NAMESPACE = 'skulk';
const DEFAULT_LANGUAGE = 'en';
const DEFAULT_CDN_PREFIX = '/i18n';
const CDN_TIMEOUT_MS = 10_000;

function uniqueLanguages(languages: string[]): string[] {
  const unique = new Set<string>();
  for (const language of languages) {
    const normalized = language.trim();
    if (normalized.length > 0) {
      unique.add(normalized);
    }
  }
  unique.add(DEFAULT_LANGUAGE);
  return [...unique];
}

function parseConfiguredLanguages(value: string | undefined): string[] {
  if (value == null || value.trim().length === 0) {
    return [DEFAULT_LANGUAGE];
  }
  return uniqueLanguages(value.split(','));
}

const availableLanguages = parseConfiguredLanguages(
  import.meta.env.VITE_TOLGEE_AVAILABLE_LANGUAGES,
);

const tolgeeBuilder = Tolgee()
  .use(LanguageStorage())
  .use(LanguageDetector())
  .use(BackendFetch({
    prefix: import.meta.env.VITE_TOLGEE_CDN_PREFIX ?? DEFAULT_CDN_PREFIX,
    fallbackOnFail: true,
    timeout: CDN_TIMEOUT_MS,
  }));

if (import.meta.env.DEV) {
  tolgeeBuilder.use(DevTools());
}

export const tolgee = tolgeeBuilder
  .use(FormatSimple())
  .init({
    defaultLanguage: DEFAULT_LANGUAGE,
    fallbackLanguage: DEFAULT_LANGUAGE,
    availableLanguages,
    defaultNs: SKULK_NAMESPACE,
    ns: [SKULK_NAMESPACE],
    availableNs: [SKULK_NAMESPACE],
    staticData: {
      [`${DEFAULT_LANGUAGE}:${SKULK_NAMESPACE}`]: () =>
        import('./en/skulk.json').then((module) => module.default),
    },
  });

export function useSkulkTranslation() {
  return useTranslate(SKULK_NAMESPACE);
}

export type SkulkTranslate = UseTranslateResult['t'];
