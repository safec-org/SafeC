// Headless-Chromium check of a safec-doc reference (Playwright).
//
//   node browser_check.mjs <base-url> <screenshot-dir> <query> <expected-path> [page...]
//
// <base-url> serves the reference (e.g. 'safec-doc serve build/doc'). For
// each page: it loads without console errors or failed requests, the wasm
// client starts (the search dialog opens on '/'), and a light and a dark
// screenshot are saved. Then searching <query> must list <expected-path>
// (an item path such as 'std::Vec') and Enter must navigate to its page;
// the theme toggle must switch html.dark. Exit status 1 on any failure.
import { createRequire } from 'node:module';
import { mkdirSync } from 'node:fs';
import { execSync } from 'node:child_process';

const require = createRequire(import.meta.url);
let playwright;
try {
  playwright = require('playwright');
} catch {
  const root = execSync('npm root -g').toString().trim();
  playwright = require(`${root}/playwright`);
}

const [base, shots, query, expected, ...pages] = process.argv.slice(2);
if (!base || !shots || !query || !expected) {
  console.error('usage: node browser_check.mjs <base-url> <screenshot-dir> <query> <expected-path> [page...]');
  process.exit(2);
}
mkdirSync(shots, { recursive: true });
const failures = [];
const fail = (m) => { failures.push(m); console.log(`  [FAIL] ${m}`); };
const pass = (m) => console.log(`  [PASS] ${m}`);

const browser = await playwright.chromium.launch();
const context = await browser.newContext({ viewport: { width: 1360, height: 900 } });
const page = await context.newPage();
const problems = [];
page.on('console', (m) => { if (m.type() === 'error') problems.push(`console: ${m.text()}`); });
page.on('pageerror', (e) => problems.push(`page error: ${e.message}`));
page.on('requestfailed', (r) => problems.push(`request failed: ${r.url()}`));
page.on('response', (r) => { if (r.status() >= 400) problems.push(`HTTP ${r.status()}: ${r.url()}`); });

const list = pages.length ? pages : ['index.html'];
for (const p of list) {
  problems.length = 0;
  const url = new URL(p, base.endsWith('/') ? base : base + '/').href;
  await page.goto(url, { waitUntil: 'networkidle' });
  // The wasm client is up when '/' opens the search dialog.
  await page.keyboard.press('/');
  const opened = await page.waitForSelector('#search:not([hidden])', { timeout: 5000 }).then(() => true, () => false);
  await page.keyboard.press('Escape');
  const name = p.replace(/[^a-zA-Z0-9]+/g, '_');
  await page.evaluate(() => document.documentElement.classList.remove('dark'));
  await page.screenshot({ path: `${shots}/${name}-light.png`, fullPage: false });
  await page.evaluate(() => document.documentElement.classList.add('dark'));
  await page.screenshot({ path: `${shots}/${name}-dark.png`, fullPage: false });
  await page.evaluate(() => document.documentElement.classList.remove('dark'));
  if (!opened) fail(`${p}: the wasm client did not start (search did not open)`);
  if (problems.length) fail(`${p}: ${problems.join('; ')}`);
  if (opened && !problems.length) pass(`${p}: loads cleanly, client running (screenshots ${name}-light/dark.png)`);
}

// Search.
await page.goto(new URL(list[0], base.endsWith('/') ? base : base + '/').href, { waitUntil: 'networkidle' });
await page.click('.search-btn');
await page.fill('.search-input', query);
const hit = await page.waitForFunction(
  (want) => [...document.querySelectorAll('.search-results .title')].some((t) => t.textContent === want),
  expected, { timeout: 5000 }).then(() => true, () => false);
await page.screenshot({ path: `${shots}/search-${query.replace(/\W+/g, '_')}.png` });
if (!hit) {
  fail(`search '${query}' does not list ${expected}`);
} else {
  const first = await page.textContent('.search-results a.selected .title');
  pass(`search '${query}' lists ${expected} (first: ${first})`);
  // Navigate to the expected result.
  const links = await page.$$('.search-results a');
  for (const l of links) {
    if ((await l.$eval('.title', (t) => t.textContent)) === expected) { await l.click(); break; }
  }
  await page.waitForLoadState('networkidle');
  const h1 = await page.textContent('h1');
  if (h1 && h1.includes(expected)) pass(`result navigates to '${h1.trim()}'`);
  else fail(`result navigated to '${h1}'`);
  await page.screenshot({ path: `${shots}/result-page.png` });
}

// Theme toggle.
const before = await page.evaluate(() => document.documentElement.classList.contains('dark'));
await page.click('.theme-toggle');
const after = await page.evaluate(() => document.documentElement.classList.contains('dark'));
if (before !== after) pass(`theme toggle switches to ${after ? 'dark' : 'light'}`);
else fail('theme toggle does nothing');

// Mobile layout.
await page.setViewportSize({ width: 390, height: 844 });
await page.screenshot({ path: `${shots}/mobile.png` });

await browser.close();
console.log(failures.length ? `${failures.length} browser check(s) failed` : 'all browser checks passed');
process.exit(failures.length ? 1 : 0);
