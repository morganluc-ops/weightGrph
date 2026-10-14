import { test, expect } from '@playwright/test';

test.describe('Copy button', () => {
  // Use a mock to allow clipboard usage
  test.use({ permissions: ['clipboard-read', 'clipboard-write'] });

  test('Verify copy analysis button works', async ({ page }) => {
    await page.goto('http://localhost:5173');

    // Wait for the table to appear (meaning data is loaded)
    await expect(page.locator('table[aria-label="Experts Table"]')).toBeVisible({ timeout: 15000 });

    const copyBtn = page.locator('button:has-text("📋 Copy Analysis")');
    await expect(copyBtn).toBeVisible();

    // Click the button
    await copyBtn.click();

    // Verify it changes to Copié
    await expect(page.locator('button:has-text("✅ Copié !")')).toBeVisible();

    await page.screenshot({ path: 'feature3-screenshot.png', fullPage: true });
  });
});
