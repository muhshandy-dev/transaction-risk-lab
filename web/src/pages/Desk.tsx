import { useMemo, useState } from 'react'
import { FactorBars, type Factor } from '../charts/FactorBars'
import type { Bundle, Case, Models } from '../data'
import { budgetLabel, num, pct, usd } from '../format'
import { BUCKET_LABEL, CATEGORY_LABEL, FEATURE_LABEL, featureValue } from '../labels'
import { encode, margin as lgbmMargin, shap } from '../model/lgbm'
import { contributions, margin as logitMargin, sigmoid } from '../model/logistic'
import type { FeatureRow } from '../model/types'

type Filter = 'all' | 'alert' | 'missed' | 'false_alert' | 'quiet'
const FILTERS: { key: Filter; label: string }[] = [
  { key: 'all', label: 'Semua contoh' },
  { key: 'alert', label: 'Masuk antrean' },
  { key: 'missed', label: 'Fraud terlewat' },
  { key: 'false_alert', label: 'Alarm palsu' },
  { key: 'quiet', label: 'Tenang' },
]

type Overrides = Partial<Record<'amt' | 'hour' | 'n_24h' | 'amt_24h' | 'n_1h', number>>

export interface Scored {
  row: FeatureRow
  lgbm: { margin: number; prob: number; phi: Float64Array; base: number }
  logistic: { margin: number; prob: number; contrib: Record<string, number>; base: number }
  ms: number
}

export function applyOverrides(c: Case, o: Overrides): FeatureRow {
  const row: FeatureRow = { ...c.features }
  for (const [k, v] of Object.entries(o)) if (v !== undefined && Number.isFinite(v)) row[k] = v
  if (o.amt !== undefined) {
    row.amt_ratio_card = c.card_mean_90d ? o.amt / c.card_mean_90d : null
    row.amt_ratio_card_cat = c.card_cat_mean_90d ? o.amt / c.card_cat_mean_90d : null
  }
  return row
}

export function score(models: Models, row: FeatureRow): Scored {
  const t0 = performance.now()
  const x = encode(models.lgbm, row)
  const m = lgbmMargin(models.lgbm, x)
  const { phi, base } = shap(models.lgbm, x)
  const pl = models.calibration.platt
  const lm = logitMargin(models.logistic, row)
  const lc = contributions(models.logistic, row)
  return {
    row,
    lgbm: { margin: m, prob: sigmoid(pl.lgbm.a * m + pl.lgbm.b), phi, base },
    logistic: { margin: lm, prob: sigmoid(pl.logistic.a * lm + pl.logistic.b), contrib: lc.contrib, base: lc.base },
    ms: performance.now() - t0,
  }
}

function factorsLgbm(models: Models, s: Scored, top = 7): Factor[] {
  const names = models.lgbm.feature_names
  const all = names.map((n, j) => ({ key: n, contrib: s.lgbm.phi[j] }))
  return groupTop(all, s.row, top)
}

function factorsLogit(s: Scored, top = 7): Factor[] {
  const all = Object.entries(s.logistic.contrib).map(([key, contrib]) => ({ key, contrib }))
  return groupTop(all, s.row, top)
}

function groupTop(all: { key: string; contrib: number }[], row: FeatureRow, top: number): Factor[] {
  const sorted = [...all].sort((a, b) => Math.abs(b.contrib) - Math.abs(a.contrib))
  const head = sorted.slice(0, top).map((f) => ({
    key: f.key,
    label: FEATURE_LABEL[f.key] ?? f.key,
    value: featureValue(f.key, row[f.key] ?? null),
    contrib: f.contrib,
  }))
  const rest = sorted.slice(top)
  if (rest.length) {
    head.push({
      key: '_rest',
      label: `${rest.length} faktor lain`,
      value: 'digabung',
      contrib: rest.reduce((a, b) => a + b.contrib, 0),
    })
  }
  return head
}

export function probText(p: number): string {
  if (p > 0.999) return '> 99,9%'
  if (p < 0.0001) return '< 0,01%'
  return pct(p, p < 0.01 ? 2 : 1)
}

const round2 = (v: unknown) => Math.round(Number(v) * 100) / 100

function narrative(f: Factor[]): string {
  const up = f.filter((x) => x.key !== '_rest' && x.contrib > 0.05).slice(0, 2)
  const down = f.filter((x) => x.key !== '_rest' && x.contrib < -0.05).slice(0, 1)
  const parts: string[] = []
  if (up.length) parts.push(`Skor naik terutama karena ${up.map((x) => `${x.label.toLowerCase()} (${x.value})`).join(' dan ')}`)
  if (down.length) parts.push(`yang menahan skor adalah ${down[0].label.toLowerCase()} (${down[0].value})`)
  return parts.length ? `${parts.join('; ')}.` : 'Tidak ada faktor yang menonjol; skor dekat dengan rata-rata.'
}

export function Desk({ bundle, models }: { bundle: Bundle; models: Models }) {
  const cases = bundle.cases.cases
  const primary = models.calibration.primary_budget
  const thr = {
    lgbm: models.calibration.thresholds.lgbm[String(primary)],
    logistic: models.calibration.thresholds.logistic[String(primary)],
  }
  const [filter, setFilter] = useState<Filter>('all')
  const [selected, setSelected] = useState(cases[0]?.id)
  const [overrides, setOverrides] = useState<Overrides>({})
  const [explainModel, setExplainModel] = useState<'lgbm' | 'logistic'>('lgbm')

  // queue needs only the score; SHAP is computed for the selected case
  const queueProb = useMemo(() => {
    const pl = models.calibration.platt.lgbm
    return new Map(cases.map((c) => [c.id, sigmoid(pl.a * lgbmMargin(models.lgbm, encode(models.lgbm, c.features)) + pl.b)]))
  }, [cases, models])
  const shown = cases.filter((c) => {
    if (filter === 'all') return true
    if (filter === 'alert') return queueProb.get(c.id)! >= thr.lgbm
    return c.bucket === filter
  })
  const current = cases.find((c) => c.id === selected) ?? cases[0]
  const edited = Object.keys(overrides).length > 0
  const original = useMemo(() => score(models, current.features), [current, models])
  const s = useMemo(
    () => (edited ? score(models, applyOverrides(current, overrides)) : original),
    [current, overrides, edited, models, original],
  )
  const parity = useMemo(() => {
    let worst = Math.abs(original.lgbm.margin - current.python.lgbm_margin)
    original.lgbm.phi.forEach((v, j) => (worst = Math.max(worst, Math.abs(v - current.python.lgbm_shap[j]))))
    return worst
  }, [current, original])

  const inQueue = s.lgbm.prob >= thr.lgbm
  const factors = explainModel === 'lgbm' ? factorsLgbm(models, s) : factorsLogit(s)
  const pick = (id: string) => {
    setSelected(id)
    setOverrides({})
  }
  const setO = (k: keyof Overrides, v: string) => {
    const x = Number(v)
    setOverrides((o) => ({ ...o, [k]: Number.isFinite(x) ? x : undefined }))
  }

  return (
    <div className="desk">
      <aside className="queue" aria-label="Antrean contoh transaksi">
        <div className="queue-head">
          <h2>Antrean contoh</h2>
          <p className="small muted">
            {cases.length} transaksi periode uji, dipilih acak dari tiap kelompok hasil. Diurutkan dari skor tertinggi.
          </p>
          <div className="chips" role="group" aria-label="Saring antrean">
            {FILTERS.map((f) => (
              <button
                key={f.key}
                type="button"
                className={`chip${filter === f.key ? ' on' : ''}`}
                aria-pressed={filter === f.key}
                onClick={() => setFilter(f.key)}
              >
                {f.label}
              </button>
            ))}
          </div>
        </div>
        <ul className="queue-list">
          {shown.map((c) => {
            const prob = queueProb.get(c.id)!
            return (
              <li key={c.id}>
                <button
                  type="button"
                  className={`queue-item${c.id === current.id ? ' active' : ''}`}
                  aria-current={c.id === current.id}
                  onClick={() => pick(c.id)}
                >
                  <span className="qi-top">
                    <span className="mono">{c.id}</span>
                    <span className={`qi-score${prob >= thr.lgbm ? ' hot' : ''}`}>{probText(prob)}</span>
                  </span>
                  <span className="qi-meta">
                    {CATEGORY_LABEL[c.category]} · {usd(c.amt)}
                  </span>
                  <span className="qi-bar" aria-hidden="true">
                    <span style={{ width: `${Math.max(2, Math.min(100, (Math.log10(prob) + 5) * 20))}%` }} />
                  </span>
                </button>
              </li>
            )
          })}
          {!shown.length && <li className="muted small pad">Tidak ada contoh untuk saringan ini.</li>}
        </ul>
      </aside>

      <section className="scorecard" aria-labelledby="sc-title">
        <p className="eyebrow">Kartu skor</p>
        <h2 id="sc-title" className="mono">
          {current.id}
        </h2>
        <p className="muted small">
          {current.ts} UTC · {CATEGORY_LABEL[current.category]} · {usd(current.amt)}
        </p>
        <div className={`verdict ${inQueue ? 'review' : 'calm'}`}>
          <div>
            <span className="big-num">{probText(s.lgbm.prob)}</span>
            <span className="small">perkiraan peluang fraud (LightGBM, kalibrasi Platt)</span>
          </div>
          <p>
            {inQueue
              ? `Masuk antrean tinjauan manusia. Ambang budget ${budgetLabel(primary)} adalah ${pct(thr.lgbm, 2)}.`
              : `Di bawah ambang budget ${budgetLabel(primary)} (${pct(thr.lgbm, 2)}), tidak masuk antrean.`}
          </p>
        </div>
        <p className="note">
          Skor ini hanya urutan prioritas untuk ditinjau petugas, bukan keputusan bahwa pemilik kartu berbuat curang.
        </p>
        <div className="pair">
          <div>
            <span className="small muted">Regresi logistik</span>
            <strong>{probText(s.logistic.prob)}</strong>
            <span className="small">{s.logistic.prob >= thr.logistic ? 'masuk antrean' : 'di bawah ambang'}</span>
          </div>
          <div>
            <span className="small muted">Label di data sintetis</span>
            <strong>{current.is_fraud ? 'fraud' : 'sah'}</strong>
            <span className="small">{BUCKET_LABEL[current.bucket]} (pada model asli)</span>
          </div>
        </div>
        <details className="facts">
          <summary>Nilai fitur yang dilihat model</summary>
          <dl>
            {models.lgbm.feature_names.map((n) => (
              <div key={n}>
                <dt>{FEATURE_LABEL[n]}</dt>
                <dd>{featureValue(n, s.row[n] ?? null)}</dd>
              </div>
            ))}
          </dl>
        </details>

        <fieldset className="whatif">
          <legend>Simulasi what-if</legend>
          <p className="small muted">
            Ubah satu nilai untuk melihat skor dan faktor dihitung ulang di browser. Riwayat lain kartu tetap; rasio nominal ikut
            berubah bila nominal diubah.
          </p>
          <div className="whatif-grid">
            <label>
              Nominal (USD)
              <input
                type="number"
                min={1}
                step="any"
                value={overrides.amt ?? current.amt}
                onChange={(e) => setO('amt', e.target.value)}
              />
            </label>
            <label>
              Jam (UTC)
              <select value={overrides.hour ?? Number(current.features.hour)} onChange={(e) => setO('hour', e.target.value)}>
                {Array.from({ length: 24 }, (_, h) => (
                  <option key={h} value={h}>
                    {String(h).padStart(2, '0')}.00
                  </option>
                ))}
              </select>
            </label>
            <label>
              Transaksi 1 jam terakhir
              <input type="number" min={0} step={1} value={overrides.n_1h ?? Number(current.features.n_1h)} onChange={(e) => setO('n_1h', e.target.value)} />
            </label>
            <label>
              Transaksi 24 jam terakhir
              <input type="number" min={0} step={1} value={overrides.n_24h ?? Number(current.features.n_24h)} onChange={(e) => setO('n_24h', e.target.value)} />
            </label>
            <label>
              Total nominal 24 jam (USD)
              <input
                type="number"
                min={0}
                step="any"
                value={overrides.amt_24h ?? round2(current.features.amt_24h)}
                onChange={(e) => setO('amt_24h', e.target.value)}
              />
            </label>
          </div>
          <button type="button" className="ghost" onClick={() => setOverrides({})} disabled={!edited}>
            Kembalikan nilai asli
          </button>
        </fieldset>
      </section>

      <section className="explain-panel" aria-labelledby="ex-title">
        <p className="eyebrow">Faktor keputusan</p>
        <h2 id="ex-title">Mengapa skornya begini</h2>
        <div className="seg" role="group" aria-label="Model yang dijelaskan">
          <button type="button" aria-pressed={explainModel === 'lgbm'} className={explainModel === 'lgbm' ? 'on' : ''} onClick={() => setExplainModel('lgbm')}>
            LightGBM · TreeSHAP
          </button>
          <button
            type="button"
            aria-pressed={explainModel === 'logistic'}
            className={explainModel === 'logistic' ? 'on' : ''}
            onClick={() => setExplainModel('logistic')}
          >
            Logistik · kontribusi linear
          </button>
        </div>
        <p className="lead-small">{narrative(factors)}</p>
        <FactorBars factors={factors} title="Kontribusi faktor terhadap skor (log-odds)" />
        <p className="small muted">
          Batang ke kanan menaikkan skor, ke kiri menurunkan, dalam satuan log-odds model sebelum kalibrasi. Nilai dasar{' '}
          {num(explainModel === 'lgbm' ? s.lgbm.base : s.logistic.base, 2)} + jumlah kontribusi ={' '}
          {num(explainModel === 'lgbm' ? s.lgbm.margin : s.logistic.margin, 2)}.
        </p>
        <div className={`provenance${edited ? ' edited' : ''}`}>
          <strong>{edited ? 'Hasil simulasi, dihitung langsung di browser' : 'Dihitung langsung di browser'}</strong>
          <span>
            TreeSHAP eksak atas {models.lgbm.num_trees} pohon dalam {num(s.ms, 0)} ms.{' '}
            {edited
              ? 'Nilai ini tidak ada di data uji; jangan dibaca sebagai hasil evaluasi.'
              : parity === 0
                ? 'Hasilnya identik dengan nilai Python untuk transaksi ini.'
                : `Selisih maksimum dengan nilai Python untuk transaksi ini: ${parity.toExponential(1)}.`}
          </span>
        </div>
      </section>
    </div>
  )
}
