// Copyright 2026 Foxlight Foundation

import { defineConfig } from 'playwright/test';

/** Configuration for explicit live-cluster dashboard vision qualification. */
export default defineConfig({
  testDir: './e2e',
  testMatch: 'vision-live.spec.ts',
  outputDir: process.env.SKULK_VISION_OUTPUT_DIR ?? 'test-results/vision-live',
  timeout: 10 * 60 * 1000,
  expect: {
    timeout: 2 * 60 * 1000,
  },
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: process.env.SKULK_DASHBOARD_URL ?? 'http://localhost:52415',
    headless: true,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
});
