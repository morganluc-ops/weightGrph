import { test, expect } from '@playwright/test';

test('Verify Local MoE tooltip', async ({ page }) => {
  await page.goto('http://localhost:5173');

  // Verify the tooltip icon exists
  const tooltipIcon = page.locator('div[title="1. Placez les dossiers du modèle sur le serveur. 2. Entrez le chemin absolu vers le dossier (ex: /chemin/vers/le/modele)."]');
  await expect(tooltipIcon).toBeVisible();
  await expect(tooltipIcon).toHaveText('?');

  await page.screenshot({ path: 'feature2-screenshot.png', fullPage: true });
});
