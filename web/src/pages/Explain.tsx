import { useState } from 'react'
import { HBar } from '../charts/HBar'
import { Scatter } from '../charts/Scatter'
import type { Bundle } from '../data'
import { num, pct } from '../format'
import { FEATURE_LABEL } from '../labels'

const LOG_X = new Set(['amt', 'amt_24h', 'amt_ratio_card', 'amt_ratio_card_cat', 'secs_since_prev', 'distance_km'])

export function Explain({ bundle }: { bundle: Bundle }) {
  const ex = bundle.explain
  const depKeys = Object.keys(ex.dependence)
  const [feat, setFeat] = useState(depKeys[0])
  const st = ex.seed_stability
  return (
    <div className="page">
      <header className="page-head">
        <p className="eyebrow">Penjelasan model</p>
        <h2>Apa yang dipakai model untuk mengurutkan risiko</h2>
        <p className="lead-small">
          Penjelasan global di halaman ini dihitung offline (precomputed) dengan pustaka shap pada {num(ex.sample_size, 0)} transaksi acak periode
          uji, model versi {bundle.meta.version}, dievaluasi {bundle.meta.evaluated}. Penjelasan per transaksi di Meja tinjauan dihitung langsung di
          browser.
        </p>
      </header>
      <div className="two">
        <section>
          <h3 className="section-title">LightGBM: rata-rata |SHAP|</h3>
          <HBar
            title="Kepentingan global LightGBM"
            color="var(--forest)"
            format={(v) => num(v, 2)}
            rows={ex.global.lgbm.map((r) => ({ key: r.feature, label: FEATURE_LABEL[r.feature] ?? r.label, value: r.mean_abs_shap }))}
          />
        </section>
        <section>
          <h3 className="section-title">Logistik: rata-rata |kontribusi|</h3>
          <HBar
            title="Kepentingan global regresi logistik"
            color="var(--ochre)"
            format={(v) => num(v, 2)}
            rows={ex.global.logistic.map((r) => ({ key: r.feature, label: FEATURE_LABEL[r.feature] ?? r.label, value: r.mean_abs_contrib }))}
          />
        </section>
      </div>
      <p className="small muted">
        Satuan keduanya log-odds. Kontribusi logistik dihitung eksak sebagai koefisien × (nilai − rata-rata latih), sama dengan SHAP linear
        dengan latar belakang independen; one-hot kategori, jam, dan hari dijumlahkan per fitur asal.
      </p>

      <h3 className="section-title">Bentuk pengaruh satu fitur (LightGBM)</h3>
      <div className="row-controls">
        <label>
          Fitur{' '}
          <select value={feat} onChange={(e) => setFeat(e.target.value)}>
            {depKeys.map((k) => (
              <option key={k} value={k}>
                {FEATURE_LABEL[k] ?? k}
              </option>
            ))}
          </select>
        </label>
      </div>
      <Scatter
        title={`Dependence plot SHAP untuk ${FEATURE_LABEL[feat] ?? feat}`}
        points={ex.dependence[feat]}
        xLabel={FEATURE_LABEL[feat] ?? feat}
        logX={LOG_X.has(feat)}
      />

      <div className="two">
        <section className="panel">
          <h3>Pemeriksaan penjelasan</h3>
          <ul className="checks">
            <li>
              Aditivitas TreeSHAP: |Σ SHAP + nilai dasar − skor| maksimum <strong>{ex.checks.lgbm_additivity_max_abs.toExponential(1)}</strong>
            </li>
            <li>
              shap.TreeExplainer vs pred_contrib LightGBM: selisih maksimum <strong>{ex.checks.shap_vs_lightgbm_pred_contrib_max_abs.toExponential(1)}</strong>
            </li>
            <li>
              Aditivitas kontribusi logistik: <strong>{ex.checks.logistic_additivity_max_abs.toExponential(1)}</strong>
            </li>
          </ul>
        </section>
        <section className="panel">
          <h3>Stabilitas antar seed latih</h3>
          <p className="small">
            Model terpilih dilatih ulang dengan {st.seeds.length} seed (bagging dan subsampling fitur berubah). Korelasi Spearman peringkat
            kepentingan: rata-rata <strong>{num(st.spearman_mean, 3)}</strong>, terendah <strong>{num(st.spearman_min, 3)}</strong>. Lima fitur
            teratas yang sama: rata-rata <strong>{pct(st.top5_overlap_mean, 0)}</strong>, terendah <strong>{pct(st.top5_overlap_min, 0)}</strong>.
          </p>
        </section>
      </div>
    </div>
  )
}
