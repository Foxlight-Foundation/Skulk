// Copyright 2026 Foxlight Foundation
// Start offline Storybook on loopback port 6017; see CONTRIBUTING.md.
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(new URL('../dashboard-react/package.json', import.meta.url));
const { chromium } = require('playwright');
const browser = await chromium.launch({ headless: true });
const imageDirectory = fileURLToPath(new URL('../website/docs/imgs', import.meta.url));
const records = [];

/** Capture a real screen with fictional data, checking rendering and page errors. */
async function capture(story, filename, { mobile = false, action, theme = 'dark' } = {}) {
  const page = await browser.newPage({
    viewport: mobile ? { width: 390, height: 844 } : { width: 1440, height: 960 },
    deviceScaleFactor: 1,
    reducedMotion: 'reduce',
  });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(`http://127.0.0.1:6017/iframe.html?id=screens-dashboard--${story}&viewMode=story&globals=theme:${theme}`);
  await page.locator('#storybook-root header').waitFor({ timeout: 60000 });
  await page.evaluate(() => document.fonts.ready);
  if (mobile) {
    // Let responsive layout settle before closing either overlaid side panel.
    await page.waitForTimeout(500);
    for (const label of ['Toggle instances panel', 'Toggle sidebar']) {
      const control = page.getByRole('button', { name: label, exact: true });
      if (await control.count() && await control.getAttribute('aria-pressed') === 'true') {
        await control.click();
      }
    }
  }
  if (action) await action(page);
  // Topology simulation and drawer transitions need a settled capture frame.
  await page.waitForTimeout(900);
  await page.screenshot({ path: `${imageDirectory}/${filename}` });
  records.push({ filename, errors });
  await page.close();
}

/** Exercise fixture-only chat through the same model selector and composer as users. */
async function converse(page) {
  await page.getByRole('button', { name: 'Select chat model' }).click();
  await page.getByRole('option', { name: /Chat-32B/ }).click();
  const composer = page.getByRole('textbox', { name: 'Chat message' });
  await composer.fill('How should I investigate a failed runner?');
  await composer.press('Enter');
  await page.getByText(/The example cluster has three nodes/).waitFor();
}

try {
  await capture('cluster', 'dash-1.png');
  await capture('model-store', 'dash-2.png', { theme: 'light' });
  await capture('chat', 'dash-3.png', { action: converse });
  await capture('cluster', 'mobile-topology.png', { mobile: true });
  await capture('cluster', 'mobile-menu.png', {
    mobile: true,
    action: page => page.getByRole('button', { name: 'Toggle mobile menu' }).click(),
  });
  await capture('chat', 'mobile-chat.png', { mobile: true, action: converse });
  await capture('cluster', 'mobile-observability.png', {
    mobile: true,
    action: async page => {
      await page.getByRole('button', { name: 'Toggle mobile menu' }).click();
      await page.getByRole('button', { name: 'Observability', exact: true }).click();
    },
  });
  await capture('integrations', 'dashboard-integrations.png', { theme: 'light' });
  await capture('plugins', 'dashboard-plugins.png');
  await capture('settings', 'dashboard-settings.png', {
    theme: 'light',
    action: page => page.getByRole('dialog').waitFor(),
  });
  await capture('steward-conversation', 'dashboard-steward.png', {
    action: page => page.getByText(/The example cluster has three nodes/).waitFor(),
  });
  console.log(JSON.stringify(records, null, 2));
  if (records.some(record => record.errors.length)) {
    throw new Error('Screenshot capture encountered a page error.');
  }
} finally {
  await browser.close();
}
