// Same preprocessing and attribution as txrisk.models.LinearSpec / LogisticModel.contributions.
import type { FeatureRow, LogisticSpec } from './types'

function raw(spec: { log1p: boolean }, v: unknown): number {
  let x = v === null || v === undefined ? NaN : Number(v)
  if (spec.log1p && !Number.isNaN(x)) x = Math.log1p(Math.max(x, 0))
  return x
}

export function design(spec: LogisticSpec, row: FeatureRow): Float64Array {
  const z: number[] = []
  for (const n of spec.numeric) {
    const x = raw(n, row[n.name])
    const miss = Number.isNaN(x)
    z.push(((miss ? n.median : x) - n.mean) / n.std)
    if (n.missing_indicator) z.push(miss ? 1 : 0)
  }
  for (const [key, vals] of Object.entries(spec.onehot)) {
    const v = row[key]
    for (const level of vals) z.push(String(v) === String(level) ? 1 : 0)
  }
  if (z.length !== spec.coef.length) throw new Error(`design has ${z.length} columns, model ${spec.coef.length}`)
  return Float64Array.from(z)
}

export function margin(spec: LogisticSpec, row: FeatureRow): number {
  const z = design(spec, row)
  let s = spec.intercept
  for (let j = 0; j < z.length; j++) s += spec.coef[j] * z[j]
  return s
}

/** Group a design column back to its original feature name. */
export function columnGroup(col: string): string {
  const [kind, name] = col.split(':', 2)
  return kind === 'oh' ? name.split('=')[0] : name
}

export function contributions(spec: LogisticSpec, row: FeatureRow): { contrib: Record<string, number>; base: number } {
  const z = design(spec, row)
  const contrib: Record<string, number> = {}
  let base = spec.intercept
  spec.columns.forEach((col, j) => {
    const g = columnGroup(col)
    contrib[g] = (contrib[g] ?? 0) + (z[j] - spec.design_mean[j]) * spec.coef[j]
    base += spec.design_mean[j] * spec.coef[j]
  })
  return { contrib, base }
}

export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z))
