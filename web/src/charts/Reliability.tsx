import { scaleLog } from 'd3-scale'
import type { CalBin } from '../data'
import { pct } from '../format'
import { useWidth } from './useWidth'

interface Props {
  title: string
  layers: { key: string; label: string; color: string; bins: CalBin[]; hollow?: boolean }[]
}

/** Reliability diagram on log-log axes: fraud is rare, so most mass sits near zero. */
export function Reliability({ title, layers }: Props) {
  const [ref, width] = useWidth<HTMLDivElement>(360)
  const size = Math.min(width, 380)
  const h = size
  const m = { t: 10, r: 12, b: 44, l: 54 }
  const lo = 1e-4
  const x = scaleLog().domain([lo, 1]).range([m.l, size - m.r]).clamp(true)
  const y = scaleLog().domain([lo, 1]).range([h - m.b, m.t]).clamp(true)
  const ticks = [1e-4, 1e-3, 1e-2, 1e-1, 1]
  const maxN = Math.max(...layers.flatMap((l) => l.bins.map((b) => b.n)))
  return (
    <div ref={ref} className="chart">
      <svg width={size} height={h} role="img" aria-label={title}>
        <title>{title}</title>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={m.t} y2={h - m.b} className="grid" />
            <line x1={m.l} x2={size - m.r} y1={y(t)} y2={y(t)} className="grid" />
            <text x={x(t)} y={h - m.b + 18} textAnchor="middle" className="tick">
              {pct(t, t < 0.01 ? 2 : 0)}
            </text>
            <text x={m.l - 6} y={y(t)} dy="0.32em" textAnchor="end" className="tick">
              {pct(t, t < 0.01 ? 2 : 0)}
            </text>
          </g>
        ))}
        <line x1={x(lo)} y1={y(lo)} x2={x(1)} y2={y(1)} className="diagonal" />
        <text x={(m.l + size - m.r) / 2} y={h - 6} textAnchor="middle" className="axis-label">
          Peluang yang diprediksi
        </text>
        <text transform={`translate(14 ${(m.t + h - m.b) / 2}) rotate(-90)`} textAnchor="middle" className="axis-label">
          Porsi fraud teramati
        </text>
        {layers.map((l) =>
          l.bins.map((b) => (
            <circle
              key={`${l.key}${b.lo}`}
              cx={x(Math.max(b.mean_pred, lo))}
              cy={y(Math.max(b.frac_pos, lo))}
              r={3 + 7 * Math.sqrt(b.n / maxN)}
              fill={l.hollow ? 'none' : l.color}
              fillOpacity={0.75}
              stroke={l.color}
              strokeWidth={1.5}
            >
              <title>{`${l.label}: prediksi ${pct(b.mean_pred, 2)}, teramati ${pct(b.frac_pos, 2)}, n=${Math.round(b.n)}`}</title>
            </circle>
          )),
        )}
      </svg>
      <ul className="legend">
        {layers.map((l) => (
          <li key={l.key}>
            <span className={`dot${l.hollow ? ' hollow' : ''}`} style={{ borderColor: l.color, background: l.hollow ? 'transparent' : l.color }} />
            {l.label}
          </li>
        ))}
        <li>
          <span className="swatch dashed" style={{ borderColor: 'var(--muted)' }} /> kalibrasi sempurna
        </li>
      </ul>
    </div>
  )
}
