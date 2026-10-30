import { HBar } from '../charts/HBar'
import { LineChart } from '../charts/LineChart'
import type { Bundle, Paired, StressModel, StressValue } from '../data'
import { budgetLabel, int, num, pct, signed } from '../format'
import { CATEGORY_LABEL, MODEL_COLOR, MODEL_SHORT } from '../labels'

const SCENARIO_LABEL: Record<string, string> = {
  base: 'Kondisi normal (run 2)',
  label_missing_10: 'Label fraud hilang, 10% kartu',
  label_missing_30: 'Label fraud hilang, 30% kartu',
  label_missing_50: 'Label fraud hilang, 50% kartu',
  novel_pattern: 'Pola fraud baru (online tidak pernah dilatih)',
  amount_up_25: 'Nominal naik 25%',
  clock_shift_3h: 'Jam bergeser +3 jam',
  cold_start: 'Semua kartu baru, tanpa riwayat',
}

function Verdict({ d }: { d: Paired }) {
  const sig = d.lo > 0 || d.hi < 0
  return (
    <article className={`delta ${sig ? (d.lo > 0 ? 'good' : 'bad') : 'flat'}`}>
      <h3>Selisih PR-AUC, LightGBM − logistik</h3>
      <p className="delta-num">{signed(d.mean, 3)}</p>
      <p className="small">
        CI 95%: {signed(d.lo, 3)} sampai {signed(d.hi, 3)}
      </p>
      <p className="small muted">
        {sig
          ? d.lo > 0
            ? 'Interval tidak memuat nol: LightGBM lebih baik.'
            : 'Interval tidak memuat nol: LightGBM lebih buruk.'
          : 'Interval memuat nol: di data nyata LightGBM tidak terbukti lebih baik dari logistik.'}
      </p>
    </article>
  )
}

const ci3 = (v: { lo: number; hi: number }) => `${num(v.lo, 3)} – ${num(v.hi, 3)}`
const pp = (x: number) => `${signed(x * 100, 1)} pp`

function StressCell({ m, k }: { m: StressValue; k: 'ap' | 'pct' }) {
  return (
    <>
      <span className="cell-main">{k === 'ap' ? num(m.value, 3) : pct(m.value, 1)}</span>
      <span className="cell-ci">{k === 'ap' ? ci3(m) : `${pct(m.lo, 1)} – ${pct(m.hi, 1)}`}</span>
    </>
  )
}

function DeltaCell({ d, ap }: { d?: StressValue; ap: boolean }) {
  if (!d) return <span className="muted">–</span>
  const sig = d.lo > 0 || d.hi < 0
  return (
    <>
      <span className={`cell-main ${sig && d.hi < 0 ? 'neg' : ''}`}>{ap ? signed(d.value, 3) : pp(d.value)}</span>
      <span className="cell-ci">{ap ? `${signed(d.lo, 3)} – ${signed(d.hi, 3)}` : `${pp(d.lo)} – ${pp(d.hi)}`}</span>
    </>
  )
}

function StressRow({ name, m, model }: { name: string; m: StressModel; model: string }) {
  return (
    <tr className={name === 'base' ? 'hl' : ''}>
      <th scope="row">{SCENARIO_LABEL[name] ?? name}</th>
      <td>{MODEL_SHORT[model]}</td>
      <td>
        <StressCell m={m.ap} k="ap" />
      </td>
      <td>
        <DeltaCell d={m.delta_vs_base?.ap} ap />
      </td>
      <td>
        <StressCell m={m.op_recall} k="pct" />
      </td>
      <td>
        <DeltaCell d={m.delta_vs_base?.op_recall} ap={false} />
      </td>
      <td>
        <span className="cell-main">{pct(m.op_alert_rate.value, 2)}</span>
        <span className="cell-ci">rencana 1%</span>
      </td>
    </tr>
  )
}

export function Robustness({ bundle }: { bundle: Bundle }) {
  const x = bundle.external
  const st = bundle.stress
  const syn = bundle.metrics
  const pb = String(bundle.meta.protocol.primary_budget)
  if (!x || !st) return <p className="error">Hasil uji ketahanan belum diekspor.</p>
  const xs = x.data.splits
  const sc = (name: string) => st.scenarios.find((s) => s.scenario === name)
  const novel = sc('novel_pattern')
  const base = sc('base')
  const lm50 = sc('label_missing_50')
  const amt = sc('amount_up_25')
  const clock = sc('clock_shift_3h')
  const cold = sc('cold_start')
  return (
    <div className="page">
      <header className="page-head">
        <p className="eyebrow">Uji ketahanan</p>
        <h2>Apakah hasilnya bertahan di luar data sintetis?</h2>
        <p className="lead-small">
          Angka utama situs ini datang dari data sintetis yang pola fraud-nya sengaja mencolok. Halaman ini menguji klaim itu dengan dua cara:
          protokol yang sama dijalankan pada transaksi kartu nyata, dan model sintetis diberi gangguan yang wajar terjadi di produksi. Semua
          protokol dikunci sebelum hasil dilihat, dan tidak ada tuning ulang.
        </p>
      </header>

      <h3 className="section-title">1. Data kartu nyata</h3>
      <p>
        {x.dataset.name}: {int(x.data.rows)} transaksi kartu Eropa, {x.dataset.period}, {int(x.data.fraud)} fraud. Fitur V1 sampai V28 adalah
        komponen PCA yang dirilis bank untuk menyamarkan data asli, jadi hanya nominal dan waktu yang bisa dibaca. Data ini tidak punya ID
        kartu, sehingga fitur perilaku per kartu dari model utama tidak bisa dibuat. Yang diuji di sini adalah protokolnya: baseline
        logistik vs LightGBM, seleksi di validasi A, Platt dan ambang di validasi B, satu kali uji.
      </p>
      <div className="table-wrap" tabIndex={0}>
        <table className="metrics compact">
          <caption>Pembagian menurut urutan waktu (jam sejak transaksi pertama)</caption>
          <thead>
            <tr>
              <th scope="col">Bagian</th>
              <th scope="col">Jam</th>
              <th scope="col">Transaksi</th>
              <th scope="col">Fraud</th>
              <th scope="col">Porsi fraud</th>
            </tr>
          </thead>
          <tbody>
            {(
              [
                ['train', 'Latih'],
                ['val_a', 'Validasi A: seleksi'],
                ['val_b', 'Validasi B: Platt + ambang'],
                ['test', 'Uji, sekali'],
              ] as const
            ).map(([k, label]) => (
              <tr key={k}>
                <td>{label}</td>
                <td>
                  {num(xs[k].time_from_h, 1)} – {num(xs[k].time_to_h, 1)}
                </td>
                <td>{int(xs[k].rows)}</td>
                <td>{int(xs[k].fraud)}</td>
                <td>{pct(xs[k].fraud_rate, 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="two">
        <div>
          <div className="table-wrap" tabIndex={0}>
            <table className="metrics">
              <caption>Hasil uji pada data nyata (CI 95%, {x.protocol.n_boot} bootstrap blok 10 menit)</caption>
              <thead>
                <tr>
                  <th scope="col">Model</th>
                  <th scope="col">PR-AUC</th>
                  <th scope="col">ROC-AUC</th>
                  <th scope="col">Recall @ {budgetLabel(Number(pb))}</th>
                </tr>
              </thead>
              <tbody>
                {['rule_amount', 'logistic', 'lgbm'].map((k) => (
                  <tr key={k} className={k === 'lgbm' ? 'hl' : ''}>
                    <th scope="row">
                      <span className="dot" style={{ background: MODEL_COLOR[k], borderColor: MODEL_COLOR[k] }} />
                      {MODEL_SHORT[k]}
                    </th>
                    <td>
                      <span className="cell-main">{num(x.models[k].ap, 3)}</span>
                      <span className="cell-ci">{ci3(x.ci[k].ap)}</span>
                    </td>
                    <td>
                      <span className="cell-main">{num(x.models[k].roc_auc, 3)}</span>
                      <span className="cell-ci">{ci3(x.ci[k].auc)}</span>
                    </td>
                    <td>
                      <span className="cell-main">{pct(x.models[k].at_budget[pb].recall, 1)}</span>
                      <span className="cell-ci">
                        {pct(x.ci[k][`recall@${pb}`].lo, 1)} – {pct(x.ci[k][`recall@${pb}`].hi, 1)}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="small muted">
            Uji hanya berisi {xs.test.fraud} fraud dalam {x.protocol.n_blocks_test} blok 10 menit, jadi intervalnya lebar. Aturan nominal hampir
            tidak berguna di sini: fraud di data ini justru cenderung bernominal kecil.
          </p>
        </div>
        <div className="deltas single">
          <Verdict d={x.paired_lgbm_minus_logistic.ap} />
        </div>
      </div>

      <h3 className="section-title">Sintetis vs nyata, berdampingan</h3>
      <div className="table-wrap" tabIndex={0}>
        <table className="metrics compact">
          <caption>Protokol sama, data berbeda</caption>
          <thead>
            <tr>
              <th scope="col">Ukuran</th>
              <th scope="col">Sintetis (Sparkov)</th>
              <th scope="col">Nyata (ULB)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>PR-AUC LightGBM</td>
              <td>{num(syn.models.lgbm.ap, 3)}</td>
              <td>{num(x.models.lgbm.ap, 3)}</td>
            </tr>
            <tr>
              <td>PR-AUC logistik</td>
              <td>{num(syn.models.logistic.ap, 3)}</td>
              <td>{num(x.models.logistic.ap, 3)}</td>
            </tr>
            <tr>
              <td>Selisih PR-AUC (CI 95%)</td>
              <td>
                {signed(syn.paired_lgbm_minus_logistic.ap.mean, 3)} ({signed(syn.paired_lgbm_minus_logistic.ap.lo, 3)} sampai{' '}
                {signed(syn.paired_lgbm_minus_logistic.ap.hi, 3)})
              </td>
              <td>
                {signed(x.paired_lgbm_minus_logistic.ap.mean, 3)} ({signed(x.paired_lgbm_minus_logistic.ap.lo, 3)} sampai{' '}
                {signed(x.paired_lgbm_minus_logistic.ap.hi, 3)})
              </td>
            </tr>
            <tr>
              <td>Recall @ 1%, LightGBM vs logistik</td>
              <td>
                {pct(syn.models.lgbm.at_budget[pb].recall, 1)} vs {pct(syn.models.logistic.at_budget[pb].recall, 1)}
              </td>
              <td>
                {pct(x.models.lgbm.at_budget[pb].recall, 1)} vs {pct(x.models.logistic.at_budget[pb].recall, 1)}
              </td>
            </tr>
            <tr>
              <td>Laju alarm nyata pada ambang 1% (LightGBM)</td>
              <td>{pct(syn.models.lgbm.operational?.[pb].alert_rate, 2)}</td>
              <td>{pct(x.models.lgbm.operational?.[pb].alert_rate, 2)}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        Keunggulan besar LightGBM di data sintetis datang dari interaksi yang sengaja ditanam generator (nominal khas per kategori, jam malam).
        Di data nyata, fitur PCA bank sudah memisahkan fraud cukup baik secara linear, dan selisih kedua model tidak bermakna. Kesimpulan yang
        jujur: pilihan model harus diputuskan dari data institusi sendiri; logistik yang sederhana tetap baseline yang layak.
      </p>

      <div className="two">
        <div>
          <h3 className="section-title">Kurva precision-recall data nyata</h3>
          <LineChart
            title="Kurva precision-recall data nyata"
            series={['logistic', 'lgbm'].map((k) => ({
              key: k,
              label: MODEL_SHORT[k],
              color: MODEL_COLOR[k],
              points: x.models[k].pr_curve.map((d) => ({ x: d.recall, y: d.precision })),
            }))}
            xDomain={[0, 1]}
            xLabel="Recall"
            yLabel="Presisi"
            xFormat={(v) => pct(v, 0)}
            yFormat={(v) => pct(v, 0)}
          />
        </div>
        <div>
          <h3 className="section-title">Faktor terbesar LightGBM (rata-rata |SHAP|)</h3>
          <HBar
            title="Kepentingan global LightGBM pada data nyata"
            color="var(--forest)"
            rows={x.explain.global.slice(0, 10).map((g) => ({
              key: g.feature,
              label: g.feature === 'log_amount' ? 'Nominal (log)' : `${g.feature} (komponen PCA)`,
              value: g.mean_abs_shap,
            }))}
            format={(v) => num(v, 3)}
          />
          <p className="small muted">
            Uji aditivitas TreeSHAP: selisih terbesar {x.explain.checks.lgbm_additivity_max_abs.toExponential(1)} pada {int(x.explain.sample_size)}{' '}
            transaksi uji ({x.explain.test_frauds_in_sample} fraud). Karena V1 sampai V28 hasil PCA, faktor ini menjelaskan model tetapi tidak bisa
            dibaca sebagai alasan bisnis untuk petugas.
          </p>
        </div>
      </div>

      <h3 className="section-title">2. Stress test pada data sintetis</h3>
      <p>
        Pengaturan model dikunci ({MODEL_SHORT.lgbm} {st.settings.num_leaves} daun, {st.settings.rounds} pohon; logistik C={st.settings.C}).
        Skenario label hilang dan pola baru melatih ulang model dengan pengaturan itu; tiga skenario pergeseran memakai model, kalibrasi, dan
        ambang run 2 apa adanya. Semua dinilai pada {int(st.test_rows)} transaksi uji yang sama ({int(st.test_fraud)} fraud), CI dari{' '}
        {st.n_boot} bootstrap kartu, dan selisih terhadap kondisi normal dihitung berpasangan.
      </p>
      <div className="table-wrap" tabIndex={0}>
        <table className="metrics stress">
          <caption>Kinerja di bawah gangguan</caption>
          <thead>
            <tr>
              <th scope="col">Skenario</th>
              <th scope="col">Model</th>
              <th scope="col">PR-AUC</th>
              <th scope="col">Δ PR-AUC</th>
              <th scope="col">Recall pada ambang</th>
              <th scope="col">Δ recall</th>
              <th scope="col">Laju alarm</th>
            </tr>
          </thead>
          <tbody>
            {st.scenarios.flatMap((s) =>
              (['lgbm', 'logistic'] as const).map((k) => <StressRow key={s.scenario + k} name={s.scenario} m={s.models[k]} model={k} />),
            )}
          </tbody>
        </table>
      </div>
      {base && lm50 && amt && clock && cold && (
        <ul className="notes">
          <li>
            <strong>Label hilang tidak terlalu merusak.</strong> Bahkan bila separuh kartu fraud tidak pernah dilaporkan, PR-AUC LightGBM hanya
            turun dari {num(base.models.lgbm.ap.value, 3)} ke {num(lm50.models.lgbm.ap.value, 3)}, karena pola yang tersisa masih sama.
          </li>
          <li>
            <strong>Pola baru adalah titik lemah utama.</strong> Fraud online yang tidak pernah dilatih hanya tertangkap{' '}
            {pct(novel?.models.lgbm.op_recall_novel?.value, 1)} oleh LightGBM dan {pct(novel?.models.logistic.op_recall_novel?.value, 1)} oleh
            logistik, padahal biasanya hampir semuanya tertangkap.
          </li>
          <li>
            <strong>Pergeseran membuat ambang meleset.</strong> Saat nominal naik 25%, laju alarm LightGBM pada ambang 1% menjadi{' '}
            {pct(amt.models.lgbm.op_alert_rate.value, 2)}, dan presisinya turun dari {pct(base.models.lgbm.op_precision.value, 0)} ke{' '}
            {pct(amt.models.lgbm.op_precision.value, 0)}. Jam bergeser 3 jam menurunkan PR-AUC LightGBM ke {num(clock.models.lgbm.ap.value, 3)}.
          </li>
          <li>
            <strong>Kartu tanpa riwayat paling sulit.</strong> Recall LightGBM tetap {pct(cold.models.lgbm.op_recall.value, 1)}, tetapi alarm
            membengkak ke {pct(cold.models.lgbm.op_alert_rate.value, 2)} dan PR-AUC jatuh ke {num(cold.models.lgbm.ap.value, 3)}, karena tanpa
            riwayat banyak transaksi sah terlihat tidak biasa.
          </li>
          <li>
            Pelajarannya untuk produksi: pantau laju alarm harian, kalibrasi ulang ambang setelah pergeseran, dan latih ulang saat muncul pola
            fraud baru. Skor tinggi di kondisi normal tidak menjamin apa pun di luar kondisi itu.
          </li>
        </ul>
      )}
      <ul className="small muted notes">
        {st.scenarios
          .filter((s) => s.scenario !== 'base')
          .map((s) => (
            <li key={s.scenario}>
              <strong>{SCENARIO_LABEL[s.scenario]}:</strong> {s.note}
            </li>
          ))}
      </ul>

      {novel && base && (
        <>
          <h3 className="section-title">Pola fraud yang tidak pernah dilihat</h3>
          <p>
            Fraud kategori {st.novel_categories.map((c) => CATEGORY_LABEL[c] ?? c).join(' dan ')} dihapus dari data latih dan validasi, lalu model
            diuji pada {int(st.test_fraud_novel)} fraud kategori itu di periode uji.
          </p>
          <div className="tiles">
            {(['lgbm', 'logistic'] as const).map((k) => (
              <article key={k} className="tile" style={{ borderTopColor: MODEL_COLOR[k] }}>
                <h3>{MODEL_SHORT[k]}</h3>
                <p className="tile-num">{pct(novel.models[k].op_recall_novel?.value, 1)}</p>
                <p className="small">recall fraud pola baru (normal: {pct(base.models[k].op_recall_novel?.value, 1)})</p>
                <p className="small muted">
                  fraud kategori lain: {pct(novel.models[k].op_recall_other?.value, 1)} (normal: {pct(base.models[k].op_recall_other?.value, 1)})
                </p>
              </article>
            ))}
          </div>
        </>
      )}

      <h3 className="section-title">Yang tetap tidak bisa diklaim</h3>
      <ul>
        <li>Data nyata ini dari 2013, dua hari, satu penerbit di Eropa. Pola fraud dan sistem pembayaran sudah berubah.</li>
        <li>Fitur PCA membuat model data nyata tidak bisa dipakai di meja tinjauan; skor live dan penjelasan per transaksi tetap dari model sintetis.</li>
        <li>Tanpa ID kartu, bootstrap blok waktu hanya mendekati ketergantungan antar transaksi; interval yang sebenarnya bisa lebih lebar.</li>
        <li>Stress test sintetis menguji ketahanan terhadap gangguan yang dirancang, bukan semua kemungkinan pergeseran di produksi.</li>
        <li>
          Logistik di data nyata memilih C={x.selection.best.logistic.C}, ujung bawah grid. Grid tidak diubah setelah hasil terlihat, jadi
          logistik mungkin sedikit lebih baik lagi dengan regularisasi lebih kuat.
        </li>
      </ul>
    </div>
  )
}
