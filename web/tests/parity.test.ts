import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { encode, margin, shap } from '../src/model/lgbm'
import { margin as logitMargin } from '../src/model/logistic'
import type { FeatureRow, LgbmModel, LogisticSpec } from '../src/model/types'

const load = (p: string) => JSON.parse(readFileSync(new URL(p, import.meta.url), 'utf8'))
const fixture = load('./fixtures/parity.json') as {
  rows: FeatureRow[]
  lgbm_margin: number[]
  lgbm_contrib: number[][]
  logistic_margin: number[]
}
const lgbm = load('../public/data/models/lgbm.json') as LgbmModel
const logit = load('../public/data/models/logistic.json') as LogisticSpec

describe('browser models match Python on test rows', () => {
  it('LightGBM margin', () => {
    fixture.rows.forEach((row, i) => {
      expect(Math.abs(margin(lgbm, encode(lgbm, row)) - fixture.lgbm_margin[i])).toBeLessThan(1e-9)
    })
  })
  it('TreeSHAP values equal LightGBM pred_contrib and add up', () => {
    let worst = 0
    fixture.rows.forEach((row, i) => {
      const x = encode(lgbm, row)
      const { phi, base } = shap(lgbm, x)
      const ref = fixture.lgbm_contrib[i]
      phi.forEach((v, j) => {
        worst = Math.max(worst, Math.abs(v - ref[j]))
      })
      worst = Math.max(worst, Math.abs(base - ref[ref.length - 1]))
      const total = phi.reduce((a, b) => a + b, base)
      expect(Math.abs(total - fixture.lgbm_margin[i])).toBeLessThan(1e-8)
    })
    expect(worst).toBeLessThan(1e-6)
  })
  it('logistic margin', () => {
    fixture.rows.forEach((row, i) => {
      expect(Math.abs(logitMargin(logit, row) - fixture.logistic_margin[i])).toBeLessThan(1e-9)
    })
  })
})
