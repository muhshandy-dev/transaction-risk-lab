import { scaleLinear, scaleLog } from 'd3-scale'
import { line } from 'd3-shape'
import { useWidth } from './useWidth'

export interface Series {
  key: string
  label: string
  color: string
  dashed?: boolean
  points: { x: number; y: number }[]
}

interface Props {
  series: Series[]
  xLabel: string
  yLabel: string
  logX?: boolean
  xDomain?: [number, number]
  yDomain?: [number, number]
  xFormat: (v: number) => string
  yFormat: (v: number) => string
  marker?: number
  height?: number
  title: string
  xTicks?: number[]
}

export function LineChart(p: Props) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const h = p.height ?? 300
  const m = { t: 12, r: 16, b: 44, l: 54 }
  const allX = p.series.flatMap((s) => s.points.map((d) => d.x))
  const xd = p.xDomain ?? [Math.min(...allX), Math.max(...allX)]
  const x = (p.logX ? scaleLog() : scaleLinear()).domain(xd).range([m.l, width - m.r])
  const y = scaleLinear()
    .domain(p.yDomain ?? [0, 1])
    .range([h - m.b, m.t])
  const mk = line<{ x: number; y: number }>()
    .x((d) => x(d.x))
    .y((d) => y(d.y))
  const xticks = p.xTicks ?? (x.ticks(width < 420 ? 4 : 6) as number[])
  return (
    <div ref={ref} className="chart">
      <svg width={width} height={h} role="img" aria-label={p.title}>
        <title>{p.title}</title>
        {y.ticks(5).map((t) => (
          <g key={`y${t}`}>
            <line x1={m.l} x2={width - m.r} y1={y(t)} y2={y(t)} className="grid" />
            <text x={m.l - 8} y={y(t)} dy="0.32em" textAnchor="end" className="tick">
              {p.yFormat(t)}
            </text>
          </g>
        ))}
        {xticks.map((t) => (
          <text key={`x${t}`} x={x(t)} y={h - m.b + 18} textAnchor="middle" className="tick">
            {p.xFormat(t)}
          </text>
        ))}
        <line x1={m.l} x2={width - m.r} y1={h - m.b} y2={h - m.b} className="axis" />
        <text x={(m.l + width - m.r) / 2} y={h - 6} textAnchor="middle" className="axis-label">
          {p.xLabel}
        </text>
        <text transform={`translate(14 ${(m.t + h - m.b) / 2}) rotate(-90)`} textAnchor="middle" className="axis-label">
          {p.yLabel}
        </text>
        {p.marker !== undefined && (
          <line x1={x(p.marker)} x2={x(p.marker)} y1={m.t} y2={h - m.b} className="marker" />
        )}
        {p.series.map((s) => (
          <path
            key={s.key}
            d={mk(s.points) ?? ''}
            fill="none"
            stroke={s.color}
            strokeWidth={2}
            strokeDasharray={s.dashed ? '5 4' : undefined}
          />
        ))}
      </svg>
      <ul className="legend">
        {p.series.map((s) => (
          <li key={s.key}>
            <span className={`swatch${s.dashed ? ' dashed' : ''}`} style={{ borderColor: s.color }} />
            {s.label}
          </li>
        ))}
      </ul>
    </div>
  )
}
