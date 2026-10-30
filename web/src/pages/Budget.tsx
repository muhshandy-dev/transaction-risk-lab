import { useState } from 'react'
import { LineChart } from '../charts/LineChart'
import type { BudgetPoint, Bundle } from '../data'
import { budgetLabel, int, num, pct } from '../format'
import { MODEL_COLOR, MODEL_SHORT } from '../labels'

const KEYS = ['lgbm', 'logistic', 'rule_amount', 'random'] as const

function interp(points: BudgetPoint[], b: number): BudgetPoint {
  const i = points.findIndex((p) => p.budget >= b)
  if (i <= 0) return points[Math.max(i, 0)]
  const a = points[i - 1]
  const c = points[i]
  const t = (Math.log(b) - Math.log(a.budget)) / (Math.log(c.budget) - Math.log(a.budget))
  const mix = (u: number, v: number) => u + t * (v - u)
  return { budget: b, recall: mix(a.recall, c.recall), precision: mix(a.precision, c.precision), amount_recall: mix(a.amount_recall, c.amount_recall) }
}

export function Budget({ bundle }: { bundle: Bundle }) {
  const curves = bundle.metrics.budget_curves
  const test = bundle.meta.data.test
  const days = (Date.parse(test.end.slice(0, 10)) - Date.parse(test.start.slice(0, 10))) / 86400000 + 1
  const perDay = test.rows / days
  const [logB, setLogB] = useState(Math.log10(bundle.meta.protocol.primary_budget))
  const b = 10 ** logB
  const rows = KEYS.map((k) => ({ k, p: interp(curves[k], b) }))
  return (
    <div className="page">
      <header className="page-head">
        <p className="eyebrow">Simulator budget tinjauan</p>
        <h2>Berapa banyak fraud tertangkap bila petugas hanya sanggup meninjau sebagian transaksi?</h2>
        <p className="lead-small">
          Geser kapasitas tinjauan. Angka di bawah dihitung offline pada {int(test.rows)} transaksi periode uji ({test.start.slice(0, 10)} sampai{' '}
          {test.end.slice(0, 10)}), dengan transaksi berskor tertinggi ditinjau lebih dulu. Ini hasil evaluasi tersimpan, bukan perhitungan live.
        </p>
      </header>
      <div className="slider-card">
        <label htmlFor="budget">
          Kapasitas tinjauan: <strong>{budgetLabel(b)}</strong> dari transaksi, sekitar <strong>{int(b * perDay)}</strong> transaksi per hari
        </label>
        <input
          id="budget"
          type="range"
          min={Math.log10(0.0005)}
          max={Math.log10(0.05)}
          step={0.01}
          value={logB}
          onChange={(e) => setLogB(Number(e.target.value))}
          aria-valuetext={`${budgetLabel(b)} transaksi`}
        />
        <div className="slider-scale small muted">
          <span>0,05%</span>
          <span>0,5%</span>
          <span>5%</span>
        </div>
      </div>
      <div className="tiles">
        {rows.map(({ k, p }) => (
          <article key={k} className="tile" style={{ borderTopColor: MODEL_COLOR[k] }}>
            <h3>{MODEL_SHORT[k]}</h3>
            <p className="tile-num">{pct(p.recall, 1)}</p>
            <p className="small">fraud tertangkap (recall)</p>
            <dl className="tile-dl">
              <div>
                <dt>Presisi antrean</dt>
                <dd>{pct(p.precision, 1)}</dd>
              </div>
              <div>
                <dt>Nilai fraud tertangkap</dt>
                <dd>{pct(p.amount_recall, 1)}</dd>
              </div>
              <div>
                <dt>Fraud terlewat</dt>
                <dd>{int((1 - p.recall) * test.fraud)} dari {int(test.fraud)}</dd>
              </div>
            </dl>
          </article>
        ))}
      </div>
      <LineChart
        title="Recall terhadap budget tinjauan, periode uji"
        series={KEYS.map((k) => ({
          key: k,
          label: MODEL_SHORT[k],
          color: MODEL_COLOR[k],
          dashed: k === 'random',
          points: curves[k].map((p) => ({ x: p.budget, y: p.recall })),
        }))}
        logX
        xDomain={[0.0005, 0.05]}
        xTicks={[0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]}
        xLabel="Budget tinjauan (porsi transaksi, skala log)"
        yLabel="Recall"
        xFormat={(v) => budgetLabel(v)}
        yFormat={(v) => pct(v, 0)}
        marker={b}
      />
      <p className="small muted">
        Fraud di periode uji: {int(test.fraud)} transaksi ({pct(test.fraud_rate, 2)}) dari {int(test.fraud_cards)} kartu. Rata-rata{' '}
        {num(perDay, 0)} transaksi per hari. Interval kepercayaan untuk budget 0,25 / 0,5 / 1 / 2 persen ada di halaman Evaluasi. Biaya
        tinjauan dan kerugian tidak diasumsikan karena data sintetis tidak punya angka biaya yang nyata.
      </p>
    </div>
  )
}
