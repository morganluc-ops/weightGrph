import { test, expect } from '@playwright/test';

test('Verify Prompts md feature', async ({ page }) => {
  await page.goto('http://localhost:5173');

  // Wait for the prompts to load in the select dropdown
  await expect(page.locator('select#custom-prompt-input')).toBeVisible();
  await page.waitForTimeout(2000); // Give it time to fetch

  // Check that the prompts are correctly loaded
  const options = await page.locator('select#custom-prompt-input option').allTextContents();
  expect(options.length).toBeGreaterThan(0);

  // Select a prompt
  await page.locator('select#custom-prompt-input').selectOption({ index: 0 });

  // Click on Analyze
  await page.click('button:has-text("Analyser le Prompt")');

  // Verify that an analysis result appears
  await expect(page.locator('text=Routage Gating Network par Token')).toBeVisible({ timeout: 15000 });

  await page.screenshot({ path: 'feature1-screenshot.png', fullPage: true });
});
