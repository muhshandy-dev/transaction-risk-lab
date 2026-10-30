import { useEffect, useState } from 'react'
import { useBundle, useModels } from './data'
import { Budget } from './pages/Budget'
import { Desk } from './pages/Desk'
import { Evaluation } from './pages/Evaluation'
import { Explain } from './pages/Explain'
import { Failures } from './pages/Failures'
import { Method } from './pages/Method'
import { Robustness } from './pages/Robustness'

const TABS = [
  { key: 'meja', label: 'Meja tinjauan' },
  { key: 'budget', label: 'Simulator budget' },
  { key: 'evaluasi', label: 'Evaluasi' },
  { key: 'ketahanan', label: 'Uji ketahanan' },
  { key: 'penjelasan', label: 'Penjelasan model' },
  { key: 'kegagalan', label: 'Kegagalan & audit' },
  { key: 'metode', label: 'Metode & batasan' },
] as const
type Tab = (typeof TABS)[number]['key']

function readHash(): Tab {
  const h = window.location.hash.replace('#', '')
  return (TABS.find((t) => t.key === h)?.key ?? 'meja') as Tab
}

function Mark() {
  return (
    <svg width="34" height="34" viewBox="0 0 34 34" aria-hidden="true" className="mark">
      <rect x="1" y="1" width="32" height="32" rx="7" fill="var(--forest)" />
      <path d="M9 23 L15 16 L19 19.5 L25 11" stroke="var(--ivory)" strokeWidth="2.4" fill="none" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="25" cy="11" r="2.6" fill="var(--rust-bright)" />
    </svg>
  )
}

export default function App() {
  const [tab, setTab] = useState<Tab>(readHash)
  const { data, error } = useBundle()
  const { models, error: modelError } = useModels()
  useEffect(() => {
    const on = () => setTab(readHash())
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  useEffect(() => {
    document.title = `${TABS.find((t) => t.key === tab)?.label} · Transaction Risk Lab`
  }, [tab])

  return (
    <>
      <a className="skip" href="#main">
        Langsung ke isi
      </a>
      <header className="masthead">
        <div className="masthead-inner">
          <div className="brand">
            <Mark />
            <div>
              <h1>Transaction Risk Lab</h1>
              <p>Meja tinjauan risiko transaksi yang bisa dijelaskan</p>
            </div>
          </div>
          <p className="badge">
            Data sintetis Sparkov · model {data?.meta.version ?? '…'} · dievaluasi {data?.meta.evaluated ?? '…'}
          </p>
        </div>
        <nav aria-label="Bagian">
          <ul>
            {TABS.map((t) => (
              <li key={t.key}>
                <a href={`#${t.key}`} aria-current={tab === t.key ? 'page' : undefined}>
                  {t.label}
                </a>
              </li>
            ))}
          </ul>
        </nav>
        <p className="strip">
          Transaksi yang ditampilkan di situs ini sintetis; halaman Uji ketahanan hanya memuat hasil agregat dari data kartu nyata. Skor
          menentukan urutan tinjauan oleh petugas, bukan tuduhan kepada pemilik kartu.
        </p>
      </header>
      <main id="main" tabIndex={-1}>
        {error && <p className="error">Data evaluasi gagal dimuat: {error}</p>}
        {!data && !error && <p className="loading">Memuat hasil evaluasi…</p>}
        {data && tab === 'meja' && (models ? <Desk bundle={data} models={models} /> : <p className="loading">{modelError ? `Model gagal dimuat: ${modelError}` : 'Memuat model…'}</p>)}
        {data && tab === 'budget' && <Budget bundle={data} />}
        {data && tab === 'evaluasi' && <Evaluation bundle={data} />}
        {data && tab === 'ketahanan' && <Robustness bundle={data} />}
        {data && tab === 'penjelasan' && <Explain bundle={data} />}
        {data && tab === 'kegagalan' && <Failures bundle={data} />}
        {data && tab === 'metode' && <Method bundle={data} />}
      </main>
      <footer className="foot">
        <p>
          Transaction Risk Lab · regresi logistik vs LightGBM, PR-AUC, recall@budget, kalibrasi, SHAP. Data: Sparkov (MIT); uji eksternal: ULB Credit Card Fraud (DbCL). Kode: lisensi MIT.
        </p>
      </footer>
    </>
  )
}
