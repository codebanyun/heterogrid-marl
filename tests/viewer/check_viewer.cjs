// Optional browser smoke test. Set PLAYWRIGHT_MODULE when Playwright is bundled elsewhere.
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const path = require('path');
const fs = require('fs');

(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: process.env.PLAYWRIGHT_EXECUTABLE || undefined });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  const output = path.resolve('output');
  fs.mkdirSync(output, { recursive: true });
  try {
    await page.goto(process.env.VIEWER_URL || 'http://127.0.0.1:8765', { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => document.getElementById('tick').textContent === '0');
    await page.getByRole('button', { name: '单步', exact: true }).click();
    await page.waitForFunction(() => document.getElementById('tick').textContent === '1');
    await page.getByRole('button', { name: '开始', exact: true }).click();
    await page.waitForFunction(() => Number(document.getElementById('tick').textContent) >= 3);
    await page.getByRole('button', { name: '暂停', exact: true }).click();
    await page.getByRole('button', { name: '重置', exact: true }).click();
    await page.waitForFunction(() => document.getElementById('tick').textContent === '0');
    await page.getByText('red_0', { exact: true }).click();
    await page.locator('#showVision').check();
    await page.screenshot({ path: path.join(output, 'viewer-preview.png'), fullPage: true });
    await page.locator('#mapX').fill('16');
    await page.locator('#mapY').fill('16');
    for (const team of ['red', 'blue']) {
      for (const role of ['Scout', 'Strike', 'Hybrid']) await page.locator('#' + team + role).fill('1');
    }
    await page.locator('#commMode').selectOption('random');
    await page.locator('#dropout').fill('1');
    await page.getByRole('button', { name: '应用设置并重置', exact: true }).click();
    await page.waitForFunction(() => document.getElementById('redCount').textContent === '3 / 3');
    const label = await page.locator('#network').textContent();
    if (label !== '随机失联') throw new Error('Communication setting did not apply');
    await page.locator('#replayFile').setInputFiles(path.join(output, 'replays', 'episode_000.json'));
    await page.waitForFunction(() => document.getElementById('mode').textContent === '回放模式');
    await page.locator('#replaySlider').fill('12');
    await page.waitForFunction(() => document.getElementById('tick').textContent === '12');
    await page.getByRole('button', { name: '返回实时', exact: true }).click();
    await page.waitForFunction(() => document.getElementById('mode').textContent === '规则策略演示');
    await page.locator('#mapX').fill('24');
    await page.locator('#mapY').fill('24');
    for (const team of ['red', 'blue']) {
      for (const role of ['Scout', 'Strike', 'Hybrid']) await page.locator('#' + team + role).fill('2');
    }
    await page.locator('#commMode').selectOption('distance');
    await page.locator('#dropout').fill('0.25');
    await page.getByRole('button', { name: '应用设置并重置', exact: true }).click();
    await page.waitForFunction(() => document.getElementById('redCount').textContent === '6 / 6');
    await page.locator('#showVision').uncheck();
    await page.screenshot({ path: path.join(output, 'viewer-preview.png'), fullPage: true });
    if (errors.length) throw new Error(errors.join('\n'));
    console.log(JSON.stringify({ browser_test: 'passed', checked: ['page rendering', 'single step', 'play/pause', 'reset', 'unit detail', 'layers', 'configuration', 'replay'], javascript_errors: errors, screenshot: path.join(output, 'viewer-preview.png') }));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exit(1); });
