export function HBar({
  rows,
  color,
  format,
  title,
}: {
  rows: { key: string; label: string; value: number }[]
  color: string
  format: (v: number) => string
  title: string
}) {
  const max = Math.max(...rows.map((r) => r.value), 1e-12)
  return (
    <ul className="hbar" aria-label={title}>
      {rows.map((r) => (
        <li key={r.key}>
          <span className="hbar-label">{r.label}</span>
          <span className="hbar-track" aria-hidden="true">
            <span className="hbar-fill" style={{ width: `${(r.value / max) * 100}%`, background: color }} />
          </span>
          <span className="hbar-num">{format(r.value)}</span>
        </li>
      ))}
    </ul>
  )
}
