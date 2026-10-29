import { HBar } from '../charts/HBar'
import type { Bundle, GroupRate } from '../data'
import { budgetLabel, int, num, pct } from '../format'
import { AUDIT_LABEL, CATEGORY_LABEL } from '../labels'

function rateRows(rows: GroupRate[], labeler: (g: string) => string = (g) => g) {
  return rows.map((r) => ({ key: r.group, label: `${labeler(r.group)} (n=${int(r.n)})`, value: r.rate }))
}

export function Failures({ bundle }: { bundle: Bundle }) {
  const f = bundle.failures
  const fa = bundle.fairness
  const ep = f.episodes
  const cat = (g: string) => CATEGORY_LABEL[g] ?? g
  return (
    <div className="page">
      <header className="page-head">
        <p className="eyebrow">Kegagalan dan audit kelompok</p>
        <h2>Di mana model gagal, dan apakah bebannya merata</h2>
        <p className="lead-small">
          LightGBM pada ambang budget {budgetLabel(f.budget)} yang dipilih di validasi B, diterapkan ke periode uji. Dari {int(f.fraud_total)}{' '}
          transaksi fraud, <strong>{int(f.fraud_missed)}</strong> terlewat; dari {int(f.alerts)} alarm, <strong>{int(f.false_alerts)}</strong> adalah
          transaksi sah.
        </p>
      </header>

      <div className="tiles">
        <article className="tile">
          <h3>Episode fraud terdeteksi</h3>
          <p className="tile-num">{pct(ep.detected_share, 1)}</p>
          <p className="small">
            {int(ep.detected)} dari {int(ep.n)} kartu yang mengalami fraud di periode uji punya minimal satu alarm
          </p>
        </article>
        <article className="tile">
          <h3>Tertangkap paling lambat transaksi ke-2</h3>
          <p className="tile-num">{pct(ep.share_caught_by_2nd_tx, 1)}</p>
          <p className="small">median posisi alarm pertama dalam episode: {num(ep.median_first_alert_pos, 0)}</p>
        </article>
        <article className="tile">
          <h3>Nilai fraud tertahan</h3>
          <p className="tile-num">{pct(ep.amount_caught_share, 1)}</p>
          <p className="small">porsi nominal fraud yang transaksinya masuk antrean</p>
        </article>
        <article className="tile">
          <h3>Saling melengkapi</h3>
          <p className="tile-num">{int(f.missed_by_lgbm_caught_by_logistic)}</p>
          <p className="small">
            fraud terlewat LightGBM tetapi ditangkap logistik; kebalikannya {int(f.caught_by_lgbm_missed_by_logistic)}
          </p>
        </article>
      </div>

      <div className="two">
        <section>
          <h3 className="section-title">Recall per rentang nominal</h3>
          <HBar title="Recall per rentang nominal" color="var(--forest)" format={(v) => pct(v, 0)} rows={rateRows(f.recall_by_amount_band, (g) => `$${g}`)} />
          <h3 className="section-title">Recall malam vs siang</h3>
          <HBar title="Recall malam vs siang" color="var(--forest)" format={(v) => pct(v, 0)} rows={rateRows(f.recall_night_vs_day)} />
        </section>
        <section>
          <h3 className="section-title">Recall per kategori</h3>
          <HBar title="Recall per kategori" color="var(--forest)" format={(v) => pct(v, 0)} rows={rateRows(f.recall_by_category, cat)} />
        </section>
      </div>
      <div className="two">
        <section>
          <h3 className="section-title">Laju alarm palsu per kategori</h3>
          <HBar title="Laju alarm palsu per kategori" color="var(--rust)" format={(v) => pct(v, 2)} rows={rateRows(f.false_alert_rate_by_category, cat)} />
        </section>
        <section>
          <h3 className="section-title">Laju alarm palsu per rentang nominal</h3>
          <HBar
            title="Laju alarm palsu per nominal"
            color="var(--rust)"
            format={(v) => pct(v, 2)}
            rows={rateRows(f.false_alert_rate_by_amount_band, (g) => `$${g}`)}
          />
        </section>
      </div>

      {bundle.ablations && (
        <section>
          <h3 className="section-title">Uji ketergantungan: latih ulang tanpa kelompok fitur</h3>
          <p className="small muted">
            Dinilai di validasi B (diagnostik, bukan seleksi). Penurunan besar berarti model sangat bergantung pada pola itu, yang di Sparkov
            sengaja dibuat mencolok.
          </p>
          <div className="table-wrap" tabIndex={0}>
            <table className="metrics compact">
              <thead>
                <tr>
                  <th scope="col">Set fitur</th>
                  <th scope="col">PR-AUC</th>
                  <th scope="col">Recall @ 1%</th>
                </tr>
              </thead>
              <tbody>
                {bundle.ablations.results.map((r) => (
                  <tr key={r.set}>
                    <td>{r.set}</td>
                    <td>{num(r.val_b_ap, 3)}</td>
                    <td>{pct(r.val_b_recall_at_1pct, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <h3 className="section-title">Audit kelompok</h3>
      <p className="small muted">
        Gender, umur, dan kota/desa tidak dipakai sebagai fitur. Tabel menunjukkan apakah recall (TPR) dan laju alarm palsu (FPR) pada ambang
        yang sama berbeda antar kelompok. CI 95% dari bootstrap kartu di dalam kelompok. Kelompok kecil punya interval lebar; jangan menarik
        kesimpulan dari selisih yang intervalnya tumpang tindih.
      </p>
      {Object.entries(fa.groups).map(([attr, rows]) => (
        <div className="table-wrap" key={attr} tabIndex={0}>
          <table className="metrics compact">
            <caption>{AUDIT_LABEL[attr] ?? attr}</caption>
            <thead>
              <tr>
                <th scope="col">Kelompok</th>
                <th scope="col">Kartu</th>
                <th scope="col">Fraud</th>
                <th scope="col">TPR LightGBM</th>
                <th scope="col">FPR LightGBM</th>
                <th scope="col">TPR logistik</th>
                <th scope="col">FPR logistik</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.group}>
                  <td>{r.group}</td>
                  <td>{int(r.cards)}</td>
                  <td>{int(r.fraud)}</td>
                  {(['lgbm', 'logistic'] as const).flatMap((k) => [
                    <td key={k + 't'}>
                      <span className="cell-main">{pct(r[k].tpr, 1)}</span>
                      <span className="cell-ci">
                        {pct(r[k].tpr_ci.lo, 1)} – {pct(r[k].tpr_ci.hi, 1)}
                      </span>
                    </td>,
                    <td key={k + 'f'}>
                      <span className="cell-main">{pct(r[k].fpr, 2)}</span>
                      <span className="cell-ci">
                        {pct(r[k].fpr_ci.lo, 2)} – {pct(r[k].fpr_ci.hi, 2)}
                      </span>
                    </td>,
                  ])}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  )
}
