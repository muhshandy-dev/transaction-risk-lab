const nf = (d: number) => new Intl.NumberFormat('id-ID', { minimumFractionDigits: d, maximumFractionDigits: d })
const cache = new Map<number, Intl.NumberFormat>()

export function num(x: number | null | undefined, d = 2): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return '–'
  let f = cache.get(d)
  if (!f) {
    f = nf(d)
    cache.set(d, f)
  }
  return f.format(x)
}

export const pct = (x: number | null | undefined, d = 1) =>
  x === null || x === undefined || !Number.isFinite(x) ? '–' : `${num(x * 100, d)}%`

export const usd = (x: number | null | undefined) => (x === null || x === undefined ? '–' : `$${num(x, 2)}`)

export const int = (x: number | null | undefined) => num(x, 0)

export const signed = (x: number, d = 2) => `${x >= 0 ? '+' : '−'}${num(Math.abs(x), d)}`

export const ciText = (lo: number, hi: number, f: (v: number) => string) => `${f(lo)} – ${f(hi)}`

export const budgetLabel = (b: number) => pct(b, b < 0.01 ? 2 : 1)
