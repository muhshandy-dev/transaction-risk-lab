# Transaction Risk Lab

Meja tinjauan risiko transaksi kartu yang bisa dijelaskan: regresi logistik sebagai baseline, LightGBM sebagai pembanding, dievaluasi dengan PR-AUC, recall@budget, dan kalibrasi, lalu dijelaskan dengan SHAP. Data: transaksi kartu **sintetis** dari generator Sparkov (MIT).

Website (statis, model berjalan di browser): belum dideploy. Lihat `docs/DEPLOY_CLOUDFLARE.md`.

## Hasil (periode uji Jul-Des 2020, 469.626 transaksi, 2.323 fraud dari 244 kartu)

CI 95% dari 1.000 bootstrap yang mengambil ulang kartu.

| Model | PR-AUC | Recall @ 1% transaksi | Presisi @ 1% | Brier (Platt) |
|---|---|---|---|---|
| Acak | 0,005 | 0,7% | 0,4% | – |
| Aturan: nominal terbesar | 0,151 (0,133-0,172) | 47,6% (45,1-50,1) | 23,6% | – |
| Regresi logistik | 0,709 (0,673-0,741) | 77,0% (74,3-79,5) | 38,1% | 2,22‰ |
| Logistik + interaksi kategori × nominal | 0,862 (0,839-0,883) | 89,8% (87,9-91,7) | 44,4% | 1,39‰ |
| **LightGBM** | **0,972 (0,964-0,979)** | **98,5% (97,8-99,0)** | 48,7% | 0,55‰ |

- Selisih berpasangan LightGBM − logistik: PR-AUC **+0,262 (0,234 sampai 0,293)**, recall @ 1% +21,5 pp (19,2 sampai 24,1). Aturan keputusan yang dikunci sebelum uji (batas bawah CI > 0) terpenuhi.
- Terhadap logistik + interaksi: PR-AUC +0,110 (0,093 sampai 0,128). Sekitar 58% selisih terhadap logistik biasa hilang hanya dengan menambah interaksi kategori × nominal.
- Kalibrasi Platt (dipasang di validasi B) menurunkan ECE LightGBM dari 0,29‰ ke 0,15‰; untuk logistik Platt tidak membantu (0,57‰ → 0,67‰).
- Ambang 1% yang dipilih di validasi B menghasilkan laju alarm nyata 0,84% di uji, recall 97,9%.
- Semua 244 episode fraud mendapat alarm, umumnya pada transaksi fraud pertama. 49 transaksi fraud terlewat, terutama nominal $20-100 dan kategori makan & minum, anak & hewan, lain-lain (toko).
- Audit kelompok (atribut tidak dipakai sebagai fitur): recall LightGBM umur 25-49 tahun 97,0% (95,5-98,3) vs 50+ 99,1% (98,5-99,6), selisih kecil tetapi intervalnya tidak tumpang tindih. Logistik jauh lebih timpang (25-49: 63,5%, 50+: 85,6%).

**Batas klaim.** Pola fraud Sparkov sengaja mencolok (jam malam, nominal khas per kategori, ledakan transaksi 1-2 hari). Angka setinggi ini **tidak** berarti model siap untuk bank nyata. Skor adalah urutan prioritas tinjauan petugas, bukan tuduhan.

## Yang ditemukan di sepanjang jalan

1. **Pergeseran fitur riwayat penuh.** Di validasi A, fitur seperti "kategori baru bagi kartu ini" berubah arti seiring riwayat kartu bertambah. Diganti jendela 90 hari + tiga bulan pemanasan.
2. **Bug di generator Sparkov.** Analisis kegagalan run pertama memperlihatkan transaksi sah di periode uji hanya ada di 6 dari 14 kategori. Penyebabnya `closest_rand_parallel` memakai angka acak tanggal untuk kategori dan pagi/sore, sehingga kategori berputar mengikuti kalender. Diperbaiki (patch P5, dijaga tes), data dibangkitkan ulang, run pertama dibuang dan diarsipkan di `docs/riwayat/run1-generator-bug/`.

## Struktur

```
src/txrisk/            paket Python
  sparkov/             salinan generator Sparkov (MIT) + PATCHES.md
  generate.py          driver deterministik
  ingest.py            buang data pribadi palsu, ID kartu pseudonim
  features.py          fitur kausal per kartu
  models.py            logistik (preprocessing eksplisit) dan LightGBM
  metrics.py           PR-AUC, recall@budget, kalibrasi, bootstrap kartu
  pipeline.py          protokol lengkap
  export_web.py        JSON untuk website
  kaggle.py            adapter dataset Kaggle kartik2112/fraud-detection
reports/               hasil run yang dilaporkan
artifacts/             model terlatih, kalibrasi, ambang
web/                   React + TypeScript + Vite; LightGBM dan TreeSHAP di browser
notebooks/             notebook Kaggle (belum dijalankan)
docs/                  protokol, kartu data, kartu model, deploy
```

## Menjalankan

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q                  # 16 tes
txrisk run                 # ~25 menit di 4 CPU: bangkitkan data, pilih, kalibrasi, uji, analisis
txrisk export-web          # tulis web/public/data
cd web && npm ci && npm test && npm run build
npm run preview & npm run e2e   # 92 pemeriksaan termasuk axe
```

Workflow GitHub: `CI` (tes, run kecil, paritas browser, build) dan `Reproduksi penuh` (manual, ketik JALANKAN; membandingkan angka dengan `reports/`).

## Dokumen

- `docs/EVALUASI.md` protokol dan riwayat keputusan
- `docs/KARTU_DATA.md`, `docs/KARTU_MODEL.md`
- `docs/DEPLOY_CLOUDFLARE.md`
- `docs/REPRODUKSI.md` bukti dua run dari nol menghasilkan angka identik
- `NOTICE.md` lisensi pihak ketiga
