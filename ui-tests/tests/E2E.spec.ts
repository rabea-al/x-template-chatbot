import { test, expect } from '@playwright/test';

const TARGET_URL = process.env.TARGET_URL ?? 'http://localhost:8888';

test('Should complete E2E test', async ({
  page,
}) => {
  const logs: string[] = [];

  page.on('console', (message) => {
    logs.push(message.text());
  });

  await page.goto(`${TARGET_URL}`);
  await page.waitForSelector('#jupyterlab-splash', { state: 'detached' });
  
  await page.locator('[aria-label="File\\ Browser\\ Section"] >> text=xircuits-workflows').dblclick();
  
  // Check First Workflow
  // Click text=TrainChatModel.xircuits
  await page.locator('text=TrainChatModel.xircuits').dblclick()

  page.once('dialog', dialog => {
    console.log(`Dialog message: ${dialog.message()}`);
    dialog.dismiss().catch(() => {});
  });

  await page.locator("xpath=//*[contains(@title, 'Save (Ctrl+S)')]").first().click();
  await page.locator("xpath=//*[contains(@title, 'Compile Xircuits')]").first().click();
  await page.locator("xpath=//*[contains(@title, 'Compile and Run Xircuits')]").first().click();

  // Start Xircuits
  await page.locator('button:has-text("Start")').click();
  // Select First Kernel
  await page.locator('button:has-text("Select")').click();

  const content = await page.locator("text=Finish Executing").innerHTML()

  expect(content).toContain('Finish Executing')

  // Check Second Workflow
  // Click text=Inference.xircuits
  await page.locator('text=Inference.xircuits').dblclick()

  page.once('dialog', dialog => {
    console.log(`Dialog message: ${dialog.message()}`);
    dialog.dismiss().catch(() => {});
  });

  await page.locator("xpath=//*[contains(@title, 'Save (Ctrl+S)')]").first().click();
  await page.locator("xpath=//*[contains(@title, 'Compile Xircuits')]").first().click();
  await page.locator("xpath=//*[contains(@title, 'Compile and Run Xircuits')]").first().click();

  // Start Xircuits
  await page.locator('button:has-text("Start")').click();
  // Select First Kernel
  await page.locator('button:has-text("Select")').click();

  const content2 = await page.locator("text=Finish Executing").innerHTML()

  expect(content2).toContain('Finish Executing')

});
