import { chromium } from 'playwright';
const OUT = process.env.OUT;
const b = await chromium.launch({ headless: true });
const p = await b.newPage({ viewport: { width: 1440, height: 900 } });
await p.goto('http://localhost:3000', { waitUntil: 'networkidle' });
const notNow = p.locator('text=Not now');
if (await notNow.count()) await notNow.click();
await p.evaluate(() => localStorage.setItem('skulk-theme', 'dark'));
await p.reload({ waitUntil: 'networkidle' });
await p.waitForTimeout(1500);

// Settings panel
await p.locator('button[title="Settings"], button:has-text("Settings")').first().click().catch(() => {});
await p.waitForTimeout(700);
await p.screenshot({ path: `${OUT}/den-settings.png` });
await p.keyboard.press('Escape'); await p.waitForTimeout(400);

// Observability panel
await p.locator('button[title="Observability"]').first().click().catch(() => {});
await p.waitForTimeout(1200);
await p.screenshot({ path: `${OUT}/den-observability.png` });
await p.keyboard.press('Escape'); await p.waitForTimeout(400);

// Store: placement manager + purge dialog
await p.locator('text=Model Store').first().click(); await p.waitForTimeout(700);
const tune = p.locator('button[title="Configure placement"]').first();
if (await tune.count()) { await tune.click(); await p.waitForTimeout(700); await p.screenshot({ path: `${OUT}/den-placement.png` }); await p.keyboard.press('Escape'); await p.waitForTimeout(300); }
await p.locator('text=Purge Node Caches').first().click(); await p.waitForTimeout(400);
await p.screenshot({ path: `${OUT}/den-purge.png` });
await p.locator('text=Cancel').first().click().catch(() => {});
await b.close();
console.log('done');
