// Copyright 2026 Foxlight Foundation

import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { basename } from 'node:path';

import { expect, test, type Page, type Request } from 'playwright/test';

const dashboardUrl = process.env.SKULK_DASHBOARD_URL;
const modelId = process.env.SKULK_VISION_MODEL;
const imagePath = process.env.SKULK_VISION_IMAGE_PATH;
const assertionMode = process.env.SKULK_VISION_ASSERTION_MODE ?? 'identity';

type VisionAssertionMode = 'identity' | 'structured-identity' | 'visible-details';

interface ImageUrlPart {
  type: 'image_url';
  image_url: {
    url: string;
  };
}

interface TextPart {
  type: 'text';
  text: string;
}

interface ChatCompletionBody {
  model?: unknown;
  max_tokens?: unknown;
  temperature?: unknown;
  top_p?: unknown;
  enable_thinking?: unknown;
  messages?: Array<{
    role?: unknown;
    content?: unknown;
  }>;
}

function requiredEnvironment(name: string, value: string | undefined): string {
  if (!value) {
    throw new Error(`${name} is required for live dashboard vision qualification.`);
  }
  return value;
}

function exactAssertionMode(value: string): VisionAssertionMode {
  if (value !== 'identity' && value !== 'structured-identity' && value !== 'visible-details') {
    throw new Error(
      `SKULK_VISION_ASSERTION_MODE must be identity, structured-identity, or visible-details, got ${value}.`,
    );
  }
  return value;
}

async function selectExactModel(page: Page, exactModelId: string): Promise<void> {
  const textarea = page.locator('textarea');
  await expect(textarea).toBeVisible({ timeout: 2 * 60 * 1000 });
  const label = exactModelId.split('/').at(-1) ?? exactModelId;
  const escapedLabel = label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const exactOption = page.locator('select option').filter({
    hasText: new RegExp(`^${escapedLabel}$`),
  });
  if (await exactOption.count() > 0) {
    const selector = exactOption.first().locator('..');
    await selector.selectOption(exactModelId);
  }

  await expect(textarea).toHaveAttribute(
    'placeholder',
    new RegExp(label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')),
  );
}

function completionBody(request: Request): ChatCompletionBody {
  const body = request.postDataJSON() as unknown;
  if (typeof body !== 'object' || body === null) {
    throw new Error('Dashboard completion request did not contain a JSON object.');
  }
  return body as ChatCompletionBody;
}

function finalUserContent(body: ChatCompletionBody): Array<ImageUrlPart | TextPart> {
  const finalMessage = body.messages?.at(-1);
  if (!finalMessage || !Array.isArray(finalMessage.content)) {
    throw new Error('Dashboard completion request did not contain structured user content.');
  }
  return finalMessage.content as Array<ImageUrlPart | TextPart>;
}

test('the built-in dashboard sends and understands a real portrait', async ({ page }, testInfo) => {
  const exactDashboardUrl = requiredEnvironment('SKULK_DASHBOARD_URL', dashboardUrl);
  const exactModelId = requiredEnvironment('SKULK_VISION_MODEL', modelId);
  const exactImagePath = requiredEnvironment('SKULK_VISION_IMAGE_PATH', imagePath);
  const exactMode = exactAssertionMode(assertionMode);

  await page.addInitScript(() => {
    localStorage.setItem('skulk-telemetry-consent-seen', new Date().toISOString());
  });
  await page.goto(new URL('/chat', exactDashboardUrl).toString(), { waitUntil: 'domcontentloaded' });
  await selectExactModel(page, exactModelId);

  const attachButton = page.getByRole('button', { name: 'Attach file' });
  await expect(attachButton).toBeEnabled();
  await page.locator('input[type="file"][accept="image/*"]').setInputFiles(exactImagePath);
  await expect(page.getByRole('img', { name: basename(exactImagePath) })).toBeVisible();

  const prompt = exactMode === 'identity'
    ? 'Identify the person in this image. Then report the clothing colors and the background color. Be concise.'
    : exactMode === 'structured-identity'
      ? 'Who is this person? What colors are the jacket, shirt, and background? Answer exactly: NAME; JACKET; SHIRT; BACKGROUND.'
      : 'How many people are visible? Describe the jacket color, the shirt or collar color, and the background color. Do not identify the person.';
  await page.locator('textarea').fill(prompt);
  const completionResponsePromise = page.waitForResponse(
    (response) => response.request().method() === 'POST'
      && new URL(response.url()).pathname === '/v1/chat/completions',
  );
  await page.getByRole('button', { name: 'Send message' }).click();

  const response = await completionResponsePromise;
  await response.finished();
  expect(response.ok()).toBe(true);
  await expect(page.getByRole('button', { name: 'Send message' })).toBeVisible();
  const request = response.request();
  const body = completionBody(request);
  expect(body.model).toBe(exactModelId);
  const content = finalUserContent(body);
  expect(content.at(-1)).toEqual({ type: 'text', text: prompt });
  const image = content.find((part): part is ImageUrlPart => part.type === 'image_url');
  expect(image).toBeDefined();
  const dataUrlMatch = image?.image_url.url.match(/^data:image\/png;base64,([A-Za-z0-9+/=]+)$/);
  expect(dataUrlMatch).not.toBeNull();
  expect(Buffer.from(dataUrlMatch![1], 'base64').equals(readFileSync(exactImagePath))).toBe(true);

  const assistantMessage = page.locator('[data-message-role="assistant"]').last();
  const assistantResponse = assistantMessage.locator('[data-testid="assistant-response-content"]').last();
  await expect(assistantResponse).toContainText(/\S/, { timeout: 2 * 60 * 1000 });
  const answer = await assistantResponse.innerText();

  writeFileSync(
    testInfo.outputPath('vision-answer.json'),
    `${JSON.stringify({
      modelId: exactModelId,
      assertionMode: exactMode,
      prompt,
      answer,
      requestControls: {
        maxTokens: body.max_tokens,
        temperature: body.temperature,
        topP: body.top_p,
        enableThinking: body.enable_thinking,
      },
      imageSha256: createHash('sha256').update(readFileSync(exactImagePath)).digest('hex'),
    }, null, 2)}\n`,
  );

  if (exactMode !== 'visible-details') {
    expect(answer).toMatch(/Elon\s+Musk/i);
  } else {
    expect(answer).toMatch(
      /(?:\b(?:one|1|single)\b.{0,30}\b(?:person|man)\b|\b(?:person|man)\b.{0,30}\b(?:one|1|single)\b)/i,
    );
  }
  expect(answer).toMatch(/(?:dark|black|navy).{0,40}(?:suit|jacket|blazer)|(?:suit|jacket|blazer).{0,40}(?:dark|black|navy)/is);
  expect(answer).toMatch(/(?:white|light(?:-colored)?).{0,30}(?:shirt|collar)|(?:shirt|collar).{0,30}(?:white|light(?:-colored)?)/is);
  expect(answer).toMatch(/(?:black|dark).{0,30}(?:background|backdrop)|(?:background|backdrop).{0,30}(?:black|dark)/is);
  expect(answer).not.toMatch(/bedroom|furniture|potted plant/i);
  expect(answer).not.toMatch(
    /<\|(?:im_start|im_end|image|video|assistant|user)\|>|<(?:start|end)_of_turn>/i,
  );

  await page.screenshot({
    path: testInfo.outputPath(`vision-${exactModelId.replace(/[^A-Za-z0-9.-]+/g, '-')}.png`),
    fullPage: true,
  });
});
