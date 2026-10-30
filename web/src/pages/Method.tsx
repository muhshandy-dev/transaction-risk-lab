import type { Bundle } from '../data'
import { int, pct } from '../format'

const SPLITS = [
  { key: 'warmup', label: 'Pemanasan fitur', from: '2019-01-01', to: '2019-04-01', cls: 'sp-warm' },
  { key: 'train', label: 'Latih', from: '2019-04-01', to: '2020-01-01', cls: 'sp-train' },
  { key: 'val_a', label: 'Validasi A: seleksi', from: '2020-01-01', to: '2020-04-01', cls: 'sp-va' },
  { key: 'val_b', label: 'Validasi B: kalibrasi + ambang', from: '2020-04-01', to: '2020-07-01', cls: 'sp-vb' },
  { key: 'test', label: 'Uji, sekali', from: '2020-07-01', to: '2021-01-01', cls: 'sp-test' },
]

export function Method({ bundle }: { bundle: Bundle }) {
  const start = Date.parse('2019-01-01')
  const span = Date.parse('2021-01-01') - start
  const d = bundle.meta.data
  const man = bundle.meta.manifest
  const best = bundle.meta.selection_best
  return (
    <div className="page prose">
      <header className="page-head">
        <p className="eyebrow">Metode dan batasan</p>
        <h2>Cara angka di situs ini dibuat, dan apa yang tidak boleh disimpulkan</h2>
      </header>

      <h3 className="section-title">Pembagian waktu</h3>
      <div className="timeline" role="img" aria-label="Garis waktu pembagian data 2019 sampai 2020">
        {SPLITS.map((s) => (
          <div
            key={s.key}
            className={`span ${s.cls}`}
            style={{ left: `${((Date.parse(s.from) - start) / span) * 100}%`, width: `${((Date.parse(s.to) - Date.parse(s.from)) / span) * 100}%` }}
          >
            <span>{s.label}</span>
          </div>
        ))}
      </div>
      <div className="timeline-axis small muted" aria-hidden="true">
        <span>Jan 2019</span>
        <span>Jan 2020</span>
        <span>Des 2020</span>
      </div>
      <div className="table-wrap" tabIndex={0}>
        <table className="metrics compact">
          <thead>
            <tr>
              <th scope="col">Bagian</th>
              <th scope="col">Transaksi</th>
              <th scope="col">Fraud</th>
              <th scope="col">Porsi fraud</th>
              <th scope="col">Kartu dengan fraud</th>
            </tr>
          </thead>
          <tbody>
            {(['train', 'val_a', 'val_b', 'test'] as const).map((k) => (
              <tr key={k}>
                <td>{SPLITS.find((s) => s.key === k)?.label}</td>
                <td>{int(d[k].rows)}</td>
                <td>{int(d[k].fraud)}</td>
                <td>{pct(d[k].fraud_rate, 2)}</td>
                <td>{int(d[k].fraud_cards)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3 className="section-title">Data</h3>
      <p>
        Transaksi kartu sintetis dari generator Sparkov (Brandon Harris, lisensi MIT), generator yang juga dipakai dataset Kaggle "Credit Card
        Transactions Fraud Detection", dengan satu perbaikan bug (lihat di bawah). Konfigurasi: {man ? int(Number(man.n_customers)) : '1.000'} nasabah, 1 Januari 2019 sampai 31 Desember
        2020, seed tetap, commit generator <code>{man ? String(man.generator_commit).slice(0, 7) : 'b5eb45c'}</code>. Total{' '}
        {man ? int(Number(man.rows)) : '–'} transaksi. Nama, alamat, nomor kartu, dan SSN palsu buatan Faker dibuang sebelum apa pun disimpan untuk
        model; kartu hanya dikenali dengan nomor pseudonim, dan situs ini tidak memuat nomor kartu sama sekali.
      </p>
      <p>
        Fitur dihitung per kartu hanya dari transaksi sebelumnya: jumlah dan total nominal dalam 1 jam, 24 jam, 7 hari; jeda dari transaksi
        terakhir; perbandingan nominal dengan kebiasaan kartu 90 hari terakhir; porsi kategori dan rentang jam dalam riwayat; jarak rumah ke
        merchant; jam, hari, kategori. Tes otomatis mengubah dan menghapus transaksi masa depan lalu memeriksa fitur masa lalu tidak berubah.
      </p>

      <h3 className="section-title">Model</h3>
      <ul>
        <li>
          Baseline: regresi logistik (L2, C={best.logistic?.C}) dengan log-transform, imputasi median + indikator kosong, standardisasi, one-hot
          kategori, jam, hari.
        </li>
        <li>Varian: logistik yang sama ditambah interaksi kategori × nominal (C={best.logistic_ix?.C}), untuk melihat seberapa besar selisih yang berasal dari interaksi.</li>
        <li>
          Pembanding: LightGBM ({best.lgbm?.num_leaves} daun, min_child_samples {best.lgbm?.min_child_samples}, {best.lgbm?.best_iteration} pohon,
          laju belajar 0,05) dengan kategori native.
        </li>
        <li>Pembanding sederhana: aturan nominal terbesar dulu, dan urutan acak.</li>
        <li>Hiperparameter dan jumlah pohon (early stopping) dipilih hanya dengan PR-AUC validasi A. Kalibrasi Platt dan ambang budget hanya dari validasi B.</li>
      </ul>

      <h3 className="section-title">Keputusan pengembangan yang tercatat</h3>
      <p>
        Versi awal memakai rata-rata dan porsi dari seluruh riwayat kartu. Di validasi A terlihat fitur itu bergeser: kartu makin lama makin
        lengkap riwayatnya, sehingga arti "kategori baru bagi kartu ini" berubah antar periode dan regresi logistik turun tajam. Fitur diganti
        dengan jendela 90 hari dan tiga bulan pertama dijadikan pemanasan. Keputusan ini diambil dari data validasi, sebelum data uji dibuka.
      </p>
      <p>
        Run penuh pertama dibuang. Analisis kegagalan memperlihatkan transaksi sah di periode uji hanya muncul di 6 dari 14 kategori. Penyebabnya
        bug di generator Sparkov: kategori dan pagi/sore ikut ditentukan oleh angka acak tanggal, sehingga kategori berputar mengikuti kalender.
        Bug diperbaiki (patch P5, dijaga tes), data dibangkitkan ulang, dan protokol dijalankan lagi tanpa perubahan lain. Angka di situs ini dari run
        kedua. Temuan pergeseran fitur di atas diamati sebelum perbaikan itu.
      </p>

      <h3 className="section-title">Batas klaim</h3>
      <ul>
        <li>
          Pola fraud di Sparkov sengaja mencolok: jam malam, nominal khas per kategori, ledakan transaksi dalam 1 sampai 2 hari, satu episode per
          kartu. Skor tinggi di sini tidak berarti model siap untuk data bank nyata. Lihat <a href="#ketahanan">Uji ketahanan</a>: di
          data kartu nyata, selisih LightGBM dan logistik tidak bermakna.
        </li>
        <li>Porsi fraud {pct(d.test.fraud_rate, 2)} dan pola waktunya ditentukan generator, bukan pengamatan dunia nyata.</li>
        <li>Skor adalah urutan prioritas tinjauan manusia, bukan keputusan bahwa seseorang berbuat curang.</li>
        <li>Gender dan umur di data ini juga sintetis; audit kelompok menguji prosedur, bukan keadilan di populasi nyata.</li>
        <li>Penjelasan SHAP menjelaskan model, bukan penyebab fraud.</li>
        <li>
          C terbaik untuk kedua varian logistik jatuh di ujung grid (C=1). Untuk logistik biasa kenaikannya sudah datar, tetapi varian dengan
          interaksi masih naik, jadi selisih LightGBM terhadap varian itu bisa sedikit lebih kecil dengan grid yang lebih lebar. Grid tidak diubah
          setelah uji dibuka.
        </li>
      </ul>

      <h3 className="section-title">Reproduksi</h3>
      <pre tabIndex={0}>
        <code>{`pip install -e ".[dev]"
txrisk run          # bangkitkan data, latih, kalibrasi, uji, analisis
txrisk export-web   # tulis web/public/data`}</code>
      </pre>
      <p className="small muted">
        Lingkungan run: Python {bundle.meta.environment.python}, LightGBM {bundle.meta.environment.lightgbm}, scikit-learn{' '}
        {bundle.meta.environment.sklearn}, shap {bundle.meta.environment.shap}. Model versi {bundle.meta.version}, dievaluasi {bundle.meta.evaluated}.
      </p>
    </div>
  )
}
