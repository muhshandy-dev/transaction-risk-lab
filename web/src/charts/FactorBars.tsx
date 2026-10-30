import { signed } from '../format'

export interface Factor {
  key: string
  label: string
  value: string
  contrib: number
}

/** Diverging bars of additive contributions (log-odds). */
export function FactorBars({ factors, title }: { factors: Factor[]; title: string }) {
  const max = Math.max(0.5, ...factors.map((f) => Math.abs(f.contrib)))
  return (
    <ol className="factors" aria-label={title}>
      {factors.map((f) => {
        const w = (Math.abs(f.contrib) / max) * 50
        const up = f.contrib >= 0
        return (
          <li key={f.key}>
            <div className="factor-text">
              <span className="factor-label">{f.label}</span>
              <span className="factor-value">{f.value}</span>
            </div>
            <div className="factor-track" aria-hidden="true">
              <span className="factor-mid" />
              <span
                className={`factor-bar ${up ? 'up' : 'down'}`}
                style={up ? { left: '50%', width: `${w}%` } : { right: '50%', width: `${w}%` }}
              />
            </div>
            <span className={`factor-num ${up ? 'up' : 'down'}`}>{signed(f.contrib)}</span>
          </li>
        )
      })}
    </ol>
  )
}
