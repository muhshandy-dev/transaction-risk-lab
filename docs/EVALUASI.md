# Protokol evaluasi

Dikunci di `src/txrisk/config.py`. Mengubah nilai di sana berarti protokol berubah dan harus dicatat di bagian "Riwayat" di bawah sebelum angka dibuat ulang.

## Pertanyaan

Untuk mengurutkan transaksi kartu yang perlu ditinjau petugas, apakah LightGBM lebih baik daripada regresi logistik, dan seberapa baik probabilitasnya dikalibrasi?

## Data dan pembagian waktu

Sparkov, 1.000 nasabah, seed 42, 1 Jan 2019 sampai 31 Des 2020 (lihat `KARTU_DATA.md`).

| Bagian | Rentang (UTC) | Dipakai untuk |
|---|---|---|
| Pemanasan | Jan-Mar 2019 | Hanya riwayat untuk fitur jendela 90 hari; tidak dilatih |
| Latih | Apr-Des 2019 | Fit model |
| Validasi A | Jan-Mar 2020 | Pilih hiperparameter dan jumlah pohon (PR-AUC) |
| Validasi B | Apr-Jun 2020 | Platt scaling, ambang budget, ablasi diagnostik |
| Uji | Jul-Des 2020 | Sekali, setelah semua di atas dikunci |

Kartu yang sama muncul di semua periode, seperti di bank nyata. Model tidak melihat ID kartu.

## Model

- Acak (prevalensi) dan aturan "nominal terbesar dulu" sebagai pembanding sederhana.
- Regresi logistik L2, C dari {0,001; 0,003; 0,01; 0,03; 0,1; 0,3; 1}.
- Regresi logistik + interaksi kategori × log nominal, grid C yang sama (untuk mengukur berapa bagian selisih berasal dari interaksi).
- LightGBM, laju belajar 0,05, num_leaves {15, 31, 63} × min_child_samples {50, 200}, early stopping 200 putaran di validasi A, maksimal 5.000 pohon. Model akhir dilatih ulang di data latih dengan jumlah pohon terbaik.

## Metrik di uji

- PR-AUC (average precision, sama dengan scikit-learn), ROC-AUC.
- Recall, presisi, dan recall nominal @budget: transaksi berskor tertinggi sebanyak b dari seluruh transaksi uji, b ∈ {0,25%; 0,5%; 1%; 2%}. **Budget utama 1%.**
- Operasional: ambang probabilitas yang memberi budget b di validasi B, diterapkan ke uji; dilaporkan laju alarm nyata, recall, presisi, FPR.
- Kalibrasi: Brier, log loss, ECE dengan 12 bin tetap (rapat dekat nol), ECE 10 bin standar, ECE zona alarm (bin ≥ 1%), reliability diagram; mentah vs Platt (dipakai) vs isotonik (sensitivitas).

## Ketidakpastian

Bootstrap 1.000 kali dengan mengambil ulang **kartu** (seluruh transaksi satu kartu ikut), karena transaksi satu kartu dan satu episode fraud saling berkaitan. CI 95% persentil. Selisih berpasangan LightGBM − logistik dihitung pada sampel bootstrap yang sama.

**Aturan keputusan (sebelum uji):** LightGBM dinyatakan lebih baik dari logistik bila batas bawah CI 95% selisih PR-AUC > 0. Hasil dilaporkan apa adanya.

## Penjelasan dan analisis

- TreeSHAP (pustaka shap) pada 5.000 transaksi uji acak; cek aditivitas dan kesamaan dengan `pred_contrib` LightGBM.
- Kontribusi logistik eksak: koefisien × (nilai − rata-rata latih) per kolom, dijumlah per fitur asal; dites sama dengan `shap.LinearExplainer`.
- Stabilitas: model terpilih dilatih dengan 5 seed, korelasi Spearman dan irisan 5 fitur teratas.
- Kegagalan pada ambang operasional 1%: recall per kategori, nominal, malam/siang; alarm palsu per kategori/nominal; deteksi per episode kartu.
- Audit kelompok: TPR dan FPR per gender, kelompok umur, kota/desa (atribut tidak dipakai sebagai fitur), CI bootstrap kartu dalam kelompok.
- Ablasi (validasi B): latih ulang tanpa jam, tanpa nominal, tanpa velocity, tanpa kebiasaan kartu.

## Uji eksternal pada data nyata (tambahan 30 Okt 2026)

Tujuan: angka utama berasal dari data sintetis, jadi protokol yang sama diuji pada transaksi kartu nyata. Dikunci sebelum hasil dilihat (kriteria K16 sampai K19), kode di `src/txrisk/external.py`.

- Data: Credit Card Fraud Detection (ULB Machine Learning Group dan Worldline), 284.807 transaksi kartu Eropa selama dua hari pada September 2013, 492 fraud. Diunduh dari salinan publik TensorFlow, SHA-256 dicek, tidak di-commit. Fitur V1-V28 adalah komponen PCA dari bank; hanya `Time` dan `Amount` yang asli; tidak ada ID kartu.
- Pembagian: baris diurutkan menurut `Time`, lalu 50% pertama latih, 15% validasi A, 15% validasi B, 20% terakhir uji (batas di nilai `Time`, sehingga detik yang sama tidak terbelah).
- Masukan: V1-V28 dan log(1 + nominal). Jam tidak dipakai karena data hanya dua hari; jam di periode uji tidak pernah muncul dua kali di latih.
- Model dan grid sama dengan protokol utama: logistik (standardisasi, C dari 7 nilai) dan LightGBM (6 kombinasi, early stopping 200 di validasi A). Platt dan ambang budget dari validasi B. Uji dipakai sekali.
- Ketidakpastian: tidak ada ID kartu, dan fraud datang berkelompok dalam waktu, jadi bootstrap mengambil ulang blok 10 menit (1.000 kali, seed sama). Selisih berpasangan pada sampel yang sama.
- Aturan keputusan sama: LightGBM lebih baik hanya bila batas bawah CI selisih PR-AUC > 0.
- Cek kebocoran: baris uji yang punya salinan persis sebelum periode uji dihitung (hasil: 0).

## Stress test pada data sintetis (tambahan 30 Okt 2026)

Kode di `src/txrisk/stress.py`, kriteria K20. Pengaturan model dikunci dari seleksi run 2 (logistik C=1; LightGBM 15 daun, min_child_samples 50, 892 pohon); tidak ada tuning ulang dan tidak ada early stopping baru.

| Skenario | Yang diubah | Latih ulang? |
|---|---|---|
| Label hilang 10/30/50% | Label fraud seluruh transaksi milik sebagian kartu fraud (dipilih acak, seed tetap) dianggap sah di latih, validasi A, dan validasi B. Label uji tetap bersih. | Ya, lalu Platt dan ambang dari validasi B yang labelnya juga hilang |
| Pola fraud baru | Transaksi fraud kategori `shopping_net` dan `misc_net` dihapus dari latih dan validasi | Ya |
| Nominal naik 25% | Semua nominal periode uji × 1,25; fitur dihitung ulang dari data bersih | Tidak |
| Jam bergeser +3 jam | Semua waktu periode uji + 3 jam; fitur dihitung ulang | Tidak |
| Kartu baru | Semua transaksi sebelum 1 Juli 2020 dihapus, fitur dihitung ulang, sehingga riwayat setiap kartu mulai kosong | Tidak |

Semua skenario dinilai pada transaksi uji yang sama (urut `tx_id`) dengan 1.000 bootstrap kartu yang sama, sehingga selisih terhadap kondisi normal berpasangan. Metrik: PR-AUC, recall pada 1% teratas, serta laju alarm, recall, dan presisi pada ambang operasional 1% yang dipilih di validasi B. Kondisi normal dicek sama dengan run 2.

## Riwayat

| Tanggal | Perubahan | Dasar |
|---|---|---|
| 29 Okt 2026 | Fitur kebiasaan kartu dari seluruh riwayat diganti jendela 90 hari; Jan-Mar 2019 jadi pemanasan; data latih mulai Apr 2019 | Di validasi A, porsi fraud pada "kategori baru bagi kartu" turun dari ~50% (awal 2019) ke ~8% (akhir 2020) karena riwayat kartu terus bertambah; regresi logistik turun tajam. Diputuskan sebelum data uji dibuka. |
| 29 Okt 2026 | Run penuh 1 dibuang | Analisis kegagalan run 1 menunjukkan transaksi sah di periode uji hanya ada di 6 dari 14 kategori. Penyebabnya bug di Sparkov (`closest_rand_parallel` memakai kolom acak tanggal untuk kategori dan AM/PM). Generator diperbaiki (patch P5, dites), data dibangkitkan ulang. Protokol, grid, dan fitur tidak diubah. Laporan run 1 disimpan di `docs/riwayat/run1-generator-bug/`. |
| 29 Okt 2026 | Run penuh 2 pada data yang sudah diperbaiki: angka yang dilaporkan | — |
| 30 Okt 2026 | Tambahan uji eksternal pada data nyata ULB dan stress test sintetis; protokol keduanya ditulis di kriteria K16-K20 sebelum dijalankan; masing-masing dijalankan sekali | Angka utama berasal dari data sintetis. Angka run 2 tidak diubah. Satu-satunya perubahan pada laporan run 2 adalah penambahan tanggal run (`evaluated`) ke `results.json`, agar tanggal di situs tidak bergantung pada waktu checkout berkas. |

Catatan jujur: temuan pergeseran fitur riwayat penuh (baris pertama) diamati pada data sebelum perbaikan P5, yang kategorinya berputar mengikuti kalender dan kemungkinan memperbesar pergeseran itu. Jendela 90 hari tetap dipertahankan karena memang desain yang lebih stabil untuk fitur kebiasaan, bukan karena hasil uji.
