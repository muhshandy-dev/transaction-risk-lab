# Bukti reproduksi

29 Oktober 2026, mesin pengembangan (4 CPU, Python 3.13.17, LightGBM 4.7.0, scikit-learn 1.9.1, shap 0.54.1, pandas 3.1.0, numpy 2.5.4).

`txrisk run` dijalankan dua kali dari nol di folder terpisah (data dibangkitkan ulang dari generator):

| Yang dibandingkan | Hasil |
|---|---|
| SHA-256 tabel mentah dan bersih (`data/manifest.json`) | identik (`aba3a408…`, `cac46b81…`) |
| `artifacts/lgbm.txt` | identik byte demi byte |
| `reports/explain.json`, `failures.json`, `fairness.json`, `ablations.json` | identik |
| `reports/results.json`, `selection.json` | identik kecuali kolom durasi (`seconds`) |
| `tools/compare_reports.py` | selisih terbesar 0 |

Di mesin lain (misalnya runner GitHub) urutan penjumlahan floating point LightGBM bisa sedikit berbeda; workflow `Reproduksi penuh` memakai toleransi 0,005 pada angka utama.
