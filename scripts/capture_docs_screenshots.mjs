// Copyright 2026 Foxlight Foundation
// Capture a running dashboard without submitting operations; see CONTRIBUTING.md.
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(new URL('../dashboard-react/package.json', import.meta.url));
const { chromium } = require('playwright');
const target = process.env.SKULK_DOCS_URL;
if (!target) throw new Error('Set SKULK_DOCS_URL to the dashboard origin to capture.');
const origin = new URL(target);
if (!['http:', 'https:'].includes(origin.protocol) || origin.username || origin.password) {
  throw new Error('Use an HTTP(S) dashboard origin without embedded credentials.');
}
const browser = await chromium.launch({ headless: true });
const imageDirectory = fileURLToPath(new URL('../website/docs/imgs', import.meta.url));
const records = [];

/** Capture actual rendered state; permit only read requests and local UI changes. */
async function capture(route, filename, { mobile = false, action, dark = false } = {}) {
  const page = await browser.newPage({
    viewport: mobile ? { width: 390, height: 844 } : { width: 1440, height: 960 },
    deviceScaleFactor: 1,
    reducedMotion: 'reduce',
    serviceWorkers: 'block',
  });
  const errors = [];
  const blockedMethods = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/*', request => {
    const method = request.request().method();
    if (['GET', 'HEAD', 'OPTIONS'].includes(method)) return request.continue();
    blockedMethods.push(method);
    return request.abort('blockedbyclient');
  });
  await page.goto(new URL(route, origin).href);
  await page.locator('header').waitFor({ timeout: 60000 });
  await page.evaluate(() => document.fonts.ready);
  // Allow initial state subscriptions and responsive drawer state to settle.
  await page.waitForTimeout(2500);
  if (mobile) {
    for (const label of ['Toggle instances panel', 'Toggle sidebar']) {
      const control = page.getByRole('button', { name: label, exact: true });
      if (await control.count() && await control.getAttribute('aria-pressed') === 'true') {
        await control.click();
      }
    }
  }
  if (dark && !mobile) {
    const control = page.getByRole('button', { name: 'Switch to dark theme', exact: true });
    if (await control.count()) await control.click();
  }
  if (action) await action(page);
  // Topology simulation and drawer transitions need a settled capture frame.
  await page.waitForTimeout(900);
  await page.screenshot({ path: `${imageDirectory}/${filename}` });
  records.push({ filename, errors, blockedMethods });
  await page.close();
}

try {
  await capture('/', 'dash-1.png');
  await capture('/model-store', 'dash-2.png');
  await capture('/chat', 'dash-3.png', { dark: true });
  await capture('/', 'mobile-topology.png', { mobile: true });
  await capture('/', 'mobile-menu.png', {
    mobile: true,
    action: page => page.getByRole('button', { name: 'Toggle mobile menu' }).click(),
  });
  await capture('/chat', 'mobile-chat.png', { mobile: true });
  await capture('/', 'mobile-observability.png', {
    mobile: true,
    action: async page => {
      await page.getByRole('button', { name: 'Toggle mobile menu' }).click();
      await page.getByRole('button', { name: 'Observability', exact: true }).click();
      await page.getByText('Runners', { exact: true }).waitFor({ timeout: 30000 });
    },
  });
  await capture('/integrations', 'dashboard-integrations.png');
  await capture('/plugins', 'dashboard-plugins.png');
  await capture('/', 'dashboard-settings.png', {
    action: async page => {
      await page.getByRole('button', { name: 'Settings', exact: true }).click();
      await page.getByRole('dialog').waitFor();
    },
  });
  await capture('/', 'dashboard-steward.png', {
    dark: true,
    action: async page => {
      await page.getByRole('button', { name: 'Ask Skulk', exact: true }).click();
      await page.getByRole('dialog', { name: 'Skulk Steward' }).waitFor();
    },
  });
  console.log(JSON.stringify(records, null, 2));
  if (records.some(record => record.errors.length || record.blockedMethods.length)) {
    throw new Error('Capture encountered page errors or attempted non-read requests.');
  }
} finally {
  await browser.close();
}
