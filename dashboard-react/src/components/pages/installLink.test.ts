import { expect, it } from 'vitest';
import { installRequestFromSearch, parseInstallLink } from './installLink';

it('reads the storefront link and the dashboard address it becomes', () => {
  expect(parseInstallLink('skulk://install?bundle=foxlight.video-studio&version=0.1.0')).toEqual({ bundleId: 'foxlight.video-studio', version: '0.1.0' });
  expect(parseInstallLink('  SKULK://install/?version=1.2.3&bundle=Pub%3Atool%40x_1-2 ')).toEqual({ bundleId: 'Pub:tool@x_1-2', version: '1.2.3' });
  expect(installRequestFromSearch('?install=foxlight.runpod&version=0.6.0')).toEqual({ bundleId: 'foxlight.runpod', version: '0.6.0' });
  expect(installRequestFromSearch('?install=foxlight.runpod')).toEqual({ bundleId: 'foxlight.runpod', version: null });
});

it('refuses anything that is not exactly an install link', () => {
  for (const text of [
    'skulk://pair?z=eNqVkE8LgkAQxb9K',
    'skulk://install?bundle=a.b',
    'skulk://install?bundle=a.b&version=1.0',
    'skulk://install?bundle=a.b&version=1.0.0&feed=https://example.invalid/',
    'skulk://install?bundle=a.b&bundle=c.d&version=1.0.0',
    'skulk://install?bundle=a%2Fb&version=1.0.0',
    'skulk://install?bundle=&version=1.0.0',
    'skulk://install?bundle=a.b&version=1.0.0#x',
    'skulk://installer?bundle=a.b&version=1.0.0',
    'skulk://install?bundle=a.b&version=01.0.0',
    'https://apps.foxlight.ai/capabilities/a.b/',
    `skulk://install?bundle=${'a'.repeat(129)}&version=1.0.0`,
  ]) expect(parseInstallLink(text), text).toBeNull();
  expect(installRequestFromSearch('?view=browse')).toBeNull();
  expect(installRequestFromSearch('?install=a%2Fb')).toBeNull();
  expect(installRequestFromSearch('?install=a.b&version=latest')).toBeNull();
  expect(installRequestFromSearch('?install=a.b&install=c.d')).toBeNull();
});
