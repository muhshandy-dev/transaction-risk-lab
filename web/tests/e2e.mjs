// End-to-end checks against `vite preview` (port 4173). Run: npm run build && npm run preview & node tests/e2e.mjs
import { createRequire } from 'node:module'
import { readFileSync } from 'node:fs'
const require = createRequire(import.meta.url)
const { chromium } = require('playwright')
const axeSource = readFileSync(require.resolve('axe-core/axe.min.js'), 'utf8')

const BASE = process.env.BASE_URL ?? 'http://localhost:4173'
const TABS = ['meja', 'budget', 'evaluasi', 'penjelasan', 'kegagalan', 'metode']
let failures = 0
let checks = 0
const ok = (cond, msg) => {
  checks++
  if (!cond) {
    failures++
    console.log('GAGAL', msg)
  }
}

const browser = await chromium.launch()
for (const scheme of ['light', 'dark']) {
  for (const width of [320, 375, 768, 1440]) {
    const page = await browser.newPage({ viewport: { width, height: 900 }, colorScheme: scheme })
    const errors = []
    page.on('pageerror', (e) => errors.push(String(e)))
    page.on('console', (m) => m.type() === 'error' && errors.push(m.text()))
    for (const tab of TABS) {
      await page.goto(`${BASE}/#${tab}`)
      await page.waitForSelector('main h2', { timeout: 15000 })
      await page.waitForTimeout(300)
      const sw = await page.evaluate(() => document.documentElement.scrollWidth)
      ok(sw <= width, `${tab} ${width}px ${scheme}: horizontal scroll ${sw}`)
      if (width === 1440 || width === 375) {
        await page.addScriptTag({ content: axeSource })
        const res = await page.evaluate(async () => (await window.axe.run(document, { resultTypes: ['violations'] })).violations)
        ok(res.length === 0, `${tab} ${width}px ${scheme}: axe ${res.map((v) => `${v.id}(${v.nodes.length})`).join(', ')}`)
      }
    }
    ok(errors.length === 0, `${width}px ${scheme}: console errors ${errors.join(' | ')}`)
    await page.close()
  }
}

// interaction: what-if changes the live score and switches provenance label
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
await page.goto(`${BASE}/#meja`)
await page.waitForSelector('.big-num')
const before = await page.textContent('.big-num')
ok((await page.textContent('.provenance')).includes('Python'), 'provenance shows parity with Python')
await page.fill('input[type=number] >> nth=0', '3')
await page.waitForTimeout(400)
const after = await page.textContent('.big-num')
ok(before !== after, `what-if changes score (${before} -> ${after})`)
ok((await page.textContent('.provenance')).includes('simulasi'), 'edited result is labelled as simulation')
await page.click('text=Kembalikan nilai asli')
ok((await page.textContent('.big-num')) === before, 'reset restores score')
// filter chips and selecting another case
await page.click('button.chip:has-text("Alarm palsu")')
const items = await page.$$('.queue-item')
ok(items.length > 0, 'false-alert filter shows cases')
await items[0].click()
ok((await page.textContent('.pair')).includes('sah'), 'false alert case has synthetic label sah')
await page.click('button:has-text("Logistik · kontribusi linear")')
ok((await page.$$('.factors li')).length >= 5, 'logistic factors render')
// budget slider moves the marker
await page.goto(`${BASE}/#budget`)
await page.waitForSelector('#budget')
const r1 = await page.textContent('.tile-num')
await page.focus('#budget')
for (let i = 0; i < 40; i++) await page.keyboard.press('ArrowLeft')
const r2 = await page.textContent('.tile-num')
ok(r1 !== r2, `budget slider updates recall (${r1} -> ${r2})`)
// keyboard: the skip link is the first tab stop
await page.reload()
await page.waitForSelector('#budget')
await page.keyboard.press('Tab')
ok(await page.evaluate(() => document.activeElement?.classList.contains('skip')), 'skip link is first tab stop')
await browser.close()

// no personal data or accusations in shipped data
const cases = readFileSync(new URL('../public/data/cases.json', import.meta.url), 'utf8')
ok(!/"(first|last|street|cc_num|ssn|card_id|dob|job)"/.test(cases), 'cases.json has no personal fields')
ok(!/(?<![.\d])\d{13,19}(?![.\d])/.test(cases), 'cases.json has no card-number-like digits')
const dist = readFileSync(new URL('../dist/index.html', import.meta.url), 'utf8')
ok(!/https?:\/\/(?!www\.w3\.org)/.test(dist), 'index.html loads nothing from other hosts')

console.log(`${checks - failures}/${checks} pemeriksaan lulus`)
process.exit(failures ? 1 : 0)
