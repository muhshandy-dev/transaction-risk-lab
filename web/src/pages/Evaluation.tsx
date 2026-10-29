import { LineChart } from '../charts/LineChart'
import { Reliability } from '../charts/Reliability'
import type { Bundle, Interval, Paired } from '../data'
import { budgetLabel, num, pct, signed } from '../format'
import { MODEL_COLOR, MODEL_SHORT } from '../labels'

const ORDER = ['random', 'rule_amount', 'logistic', 'logistic_ix', 'lgbm']

function Cell({ v, ci, f }: { v: number | undefined; ci?: Interval; f: (x: number) => string }) {
  return (
    <td>
      <span className="cell-main">{v === undefined ? '–' : f(v)}</span>
      {ci && <span className="cell-ci">{`${f(ci.lo)} – ${f(ci.hi)}`}</span>}
    </td>
  )
}

function Delta({ title, d, f, better }: { title: string; d: Paired; f: (x: number) => string; better: 'up' | 'down' }) {
  const excludesZero = d.lo > 0 || d.hi < 0
  const good = better === 'up' ? d.lo > 0 : d.hi < 0
  return (
    <article className={`delta ${excludesZero ? (good ? 'good' : 'bad') : 'flat'}`}>
      <h3>{title}</h3>
      <p className="delta-num">{f(d.mean)}</p>
      <p className="small">
        CI 95%: {f(d.lo)} sampai {f(d.hi)}
      </p>
      <p className="small muted">
        {excludesZero ? (good ? 'Interval tidak memuat nol: LightGBM lebih baik.' : 'Interval tidak memuat nol: LightGBM lebih buruk.') : 'Interval memuat nol: tidak terbukti berbeda.'}
      </p>
    </article>
  )
}

export function Evaluation({ bundle }: { bundle: Bundle }) {
  const { models, ci } = bundle.metrics
  const p = bundle.meta.protocol
  const pb = String(p.primary_budget)
  const pairs = bundle.metrics.paired_lgbm_minus_logistic
  const pairsIx = bundle.metrics.paired_lgbm_minus_logistic_ix
  const pp = (x: number) => signed(x * 100, 1) + ' pp'
  return (
    <div className="page">
      <header className="page-head">
        <p className="eyebrow">Evaluasi pada periode uji</p>
        <h2>Logistik sebagai baseline, LightGBM sebagai pembanding</h2>
        <p className="lead-small">
          Semua angka dari periode uji Juli sampai Desember 2020, dipakai satu kali setelah model, kalibrasi, dan ambang dikunci di data
          validasi. Interval adalah CI 95% dari {p.n_boot} bootstrap yang mengambil ulang kartu, bukan baris, karena transaksi satu kartu
          saling berkaitan.
        </p>
      </header>

      <div className="table-wrap" tabIndex={0}>
        <table className="metrics">
          <caption>Metrik peringkat dan kalibrasi (setelah Platt)</caption>
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col">PR-AUC</th>
              <th scope="col">ROC-AUC</th>
              <th scope="col">Recall @ {budgetLabel(p.primary_budget)}</th>
              <th scope="col">Presisi @ {budgetLabel(p.primary_budget)}</th>
              <th scope="col">Brier</th>
              <th scope="col">ECE</th>
            </tr>
          </thead>
          <tbody>
            {ORDER.map((k) => {
              const m = models[k]
              const c = ci[k] ?? {}
              const cal = m.calibration?.platt
              return (
                <tr key={k} className={k === 'lgbm' ? 'hl' : ''}>
                  <th scope="row">
                    <span className="dot" style={{ background: MODEL_COLOR[k], borderColor: MODEL_COLOR[k] }} />
                    {m.label}
                  </th>
                  <Cell v={m.ap} ci={c.ap} f={(x) => num(x, 3)} />
                  <Cell v={m.roc_auc} ci={c.auc} f={(x) => num(x, 4)} />
                  <Cell v={m.at_budget[pb].recall} ci={c[`recall@${pb}`]} f={(x) => pct(x, 1)} />
                  <Cell v={m.at_budget[pb].precision} ci={c[`precision@${pb}`]} f={(x) => pct(x, 1)} />
                  <Cell v={cal?.brier} ci={c.brier} f={(x) => num(x * 1000, 2) + '‰'} />
                  <Cell v={cal?.ece} ci={c.ece} f={(x) => num(x * 1000, 2) + '‰'} />
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="small muted">
        Acak memberi PR-AUC sama dengan porsi fraud ({pct(bundle.meta.data.test.fraud_rate, 2)}); CI tidak dihitung untuk baris ini. Brier dan
        ECE ditulis dalam per mil (‰).
      </p>

      <h3 className="section-title">Uji beda berpasangan</h3>
      <p className="small muted">
        Selisih LightGBM dikurangi pembanding pada sampel bootstrap kartu yang sama. Aturan keputusan dikunci sebelum uji: LightGBM dinyatakan
        lebih baik bila batas bawah CI selisih PR-AUC di atas nol.
      </p>
      <div className="deltas">
        <Delta title="PR-AUC, LightGBM − logistik" d={pairs.ap} f={(x) => signed(x, 3)} better="up" />
        <Delta title={`Recall @ ${budgetLabel(p.primary_budget)}, LightGBM − logistik`} d={pairs[`recall@${pb}`]} f={pp} better="up" />
        <Delta title="Brier, LightGBM − logistik" d={pairs.brier} f={(x) => signed(x * 1000, 2) + '‰'} better="down" />
        <Delta title="PR-AUC, LightGBM − logistik + interaksi" d={pairsIx.ap} f={(x) => signed(x, 3)} better="up" />
      </div>

      <div className="two">
        <div>
          <h3 className="section-title">Kurva precision-recall</h3>
          <LineChart
            title="Kurva precision-recall periode uji"
            series={['rule_amount', 'logistic', 'lgbm'].map((k) => ({
              key: k,
              label: MODEL_SHORT[k],
              color: MODEL_COLOR[k],
              points: (models[k].pr_curve ?? []).map((d) => ({ x: d.recall, y: d.precision })),
            }))}
            xDomain={[0, 1]}
            xLabel="Recall"
            yLabel="Presisi"
            xFormat={(v) => pct(v, 0)}
            yFormat={(v) => pct(v, 0)}
          />
          <p className="small muted">Aturan nominal tidak punya kurva karena tidak dikalibrasi; nilainya ada di tabel.</p>
        </div>
        <div>
          <h3 className="section-title">Ambang dari validasi B, diterapkan ke uji</h3>
          <div className="table-wrap" tabIndex={0}>
            <table className="metrics compact">
              <caption>Laju alarm yang direncanakan vs yang terjadi</caption>
              <thead>
                <tr>
                  <th scope="col">Budget</th>
                  <th scope="col">Model</th>
                  <th scope="col">Alarm nyata</th>
                  <th scope="col">Recall</th>
                  <th scope="col">Presisi</th>
                </tr>
              </thead>
              <tbody>
                {p.budgets.flatMap((b) =>
                  ['logistic', 'lgbm'].map((k) => {
                    const o = models[k].operational?.[String(b)]
                    return (
                      <tr key={`${b}${k}`}>
                        <td>{budgetLabel(b)}</td>
                        <td>{MODEL_SHORT[k]}</td>
                        <td>{pct(o?.alert_rate, 2)}</td>
                        <td>{pct(o?.recall, 1)}</td>
                        <td>{pct(o?.precision, 1)}</td>
                      </tr>
                    )
                  }),
                )}
              </tbody>
            </table>
          </div>
          <p className="small muted">
            Di produksi ambang harus dipilih sebelum data baru datang. Selisih antara laju alarm yang direncanakan dan yang terjadi menunjukkan
            seberapa stabil skor antar periode.
          </p>
        </div>
      </div>

      <h3 className="section-title">Kalibrasi</h3>
      <p className="small muted">
        Platt scaling dipasang di validasi B (April sampai Juni 2020). Titik di atas garis putus berarti model meremehkan risiko; ukuran titik
        sebanding dengan jumlah transaksi di bin.
      </p>
      <div className="two">
        {['logistic', 'lgbm'].map((k) => {
          const cal = models[k].calibration!
          return (
            <div key={k}>
              <Reliability
                title={`Diagram reliabilitas ${MODEL_SHORT[k]}`}
                layers={[
                  { key: 'raw', label: `${MODEL_SHORT[k]} mentah`, color: 'var(--muted)', bins: cal.raw.bins, hollow: true },
                  { key: 'platt', label: `${MODEL_SHORT[k]} + Platt`, color: MODEL_COLOR[k], bins: cal.platt.bins },
                ]}
              />
            </div>
          )
        })}
      </div>
      <div>
        <div className="table-wrap" tabIndex={0}>
          <table className="metrics compact">
            <caption>Kalibrasi pada uji: mentah, Platt, isotonik</caption>
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col">Versi</th>
                <th scope="col">Brier ‰</th>
                <th scope="col">ECE ‰</th>
                <th scope="col">ECE zona alarm</th>
                <th scope="col">Rata-rata prediksi</th>
              </tr>
            </thead>
            <tbody>
              {['logistic', 'lgbm'].flatMap((k) =>
                (['raw', 'platt', 'isotonic'] as const).map((v) => {
                  const c = models[k].calibration![v]
                  return (
                    <tr key={k + v}>
                      <td>{MODEL_SHORT[k]}</td>
                      <td>{{ raw: 'mentah', platt: 'Platt', isotonic: 'isotonik' }[v]}</td>
                      <td>{num(c.brier * 1000, 2)}</td>
                      <td>{num(c.ece * 1000, 2)}</td>
                      <td>{pct(c.ece_alert_region, 1)}</td>
                      <td>
                        {pct(c.mean_pred, 2)} vs {pct(c.frac_pos, 2)}
                      </td>
                    </tr>
                  )
                }),
              )}
            </tbody>
          </table>
          <p className="small muted">
            Platt tidak selalu membantu: untuk logistik, ECE sesudah Platt sedikit lebih buruk daripada skor mentah, dan hal itu dilaporkan apa
            adanya. ECE keseluruhan kecil karena hampir semua transaksi berpeluang mendekati nol. ECE zona alarm hanya menghitung bin dengan prediksi
            minimal 1%, tempat keputusan tinjauan diambil.
          </p>
        </div>
      </div>
    </div>
  )
}
