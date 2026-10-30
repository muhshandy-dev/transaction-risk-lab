import { scaleBand, scaleLinear, scaleLog } from 'd3-scale'
import { num } from '../format'
import { useWidth } from './useWidth'

interface Pt {
  x: number | string
  shap: number
  fraud: number
}

export function Scatter({ points, xLabel, title, logX }: { points: Pt[]; xLabel: string; title: string; logX?: boolean }) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const h = 300
  const m = { t: 10, r: 14, b: 70, l: 54 }
  const categorical = typeof points[0]?.x === 'string'
  const ys = points.map((p) => p.shap)
  const y = scaleLinear()
    .domain([Math.min(...ys, 0), Math.max(...ys, 0)])
    .nice()
    .range([h - m.b, m.t])
  let xPos: (p: Pt, i: number) => number
  let ticks: { pos: number; label: string }[] = []
  if (categorical) {
    const cats = [...new Set(points.map((p) => String(p.x)))].sort()
    const band = scaleBand().domain(cats).range([m.l, width - m.r]).padding(0.2)
    xPos = (p, i) => (band(String(p.x)) ?? 0) + band.bandwidth() * (0.15 + 0.7 * (((i * 2654435761) % 1000) / 1000))
    ticks = cats.map((c) => ({ pos: (band(c) ?? 0) + band.bandwidth() / 2, label: c }))
  } else {
    const xs = points.map((p) => Number(p.x)).filter((v) => Number.isFinite(v))
    const minPos = Math.max(Math.min(...xs.filter((v) => v > 0)), 1e-3)
    const useLog = logX && xs.every((v) => v >= 0)
    const sx = useLog
      ? scaleLog().domain([minPos, Math.max(...xs)]).range([m.l, width - m.r]).clamp(true)
      : scaleLinear().domain([Math.min(...xs), Math.max(...xs)]).nice().range([m.l, width - m.r])
    xPos = (p) => sx(Math.max(Number(p.x), useLog ? minPos : -Infinity))
    let tv = sx.ticks(width < 420 ? 3 : 6) as number[]
    if (useLog) {
      const [d0, d1] = sx.domain()
      tv = []
      for (let e = Math.floor(Math.log10(d0)); e <= Math.ceil(Math.log10(d1)); e++) {
        for (const m of width < 560 ? [1] : [1, 3]) {
          const v = m * 10 ** e
          if (v >= d0 && v <= d1) tv.push(v)
        }
      }
    }
    ticks = tv.map((t) => ({ pos: sx(t), label: num(t, t < 1 ? 2 : t < 10 && t % 1 ? 1 : 0) }))
  }
  return (
    <div ref={ref} className="chart">
      <svg width={width} height={h} role="img" aria-label={title}>
        <title>{title}</title>
        {y.ticks(5).map((t) => (
          <g key={t}>
            <line x1={m.l} x2={width - m.r} y1={y(t)} y2={y(t)} className={t === 0 ? 'axis' : 'grid'} />
            <text x={m.l - 8} y={y(t)} dy="0.32em" textAnchor="end" className="tick">
              {num(t, 1)}
            </text>
          </g>
        ))}
        {ticks.map((t) => (
          <text
            key={t.label + t.pos}
            x={t.pos}
            y={h - m.b + 14}
            textAnchor={categorical ? 'end' : 'middle'}
            transform={categorical ? `rotate(-35 ${t.pos} ${h - m.b + 14})` : undefined}
            className="tick"
          >
            {t.label}
          </text>
        ))}
        <text x={(m.l + width - m.r) / 2} y={h - 4} textAnchor="middle" className="axis-label">
          {xLabel}
        </text>
        <text transform={`translate(14 ${(m.t + h - m.b) / 2}) rotate(-90)`} textAnchor="middle" className="axis-label">
          SHAP (log-odds)
        </text>
        {points.map((p, i) => (
          <circle key={i} cx={xPos(p, i)} cy={y(p.shap)} r={p.fraud ? 3.2 : 2.2} className={p.fraud ? 'pt-fraud' : 'pt-legit'} />
        ))}
      </svg>
      <ul className="legend">
        <li>
          <span className="dot" style={{ background: 'var(--rust)', borderColor: 'var(--rust)' }} /> transaksi berlabel fraud
        </li>
        <li>
          <span className="dot" style={{ background: 'var(--sage)', borderColor: 'var(--sage)' }} /> transaksi sah
        </li>
      </ul>
    </div>
  )
}
