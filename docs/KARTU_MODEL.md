# Kartu model

## Kegunaan

Mengurutkan transaksi kartu menurut prioritas tinjauan oleh petugas. Bukan untuk menolak transaksi atau menilai orang secara otomatis. Model yang dirilis dilatih dan diuji pada data sintetis Sparkov. Protokol yang sama (bukan model ini) juga diuji pada data kartu nyata ULB, lihat bagian Uji ketahanan.

## Model yang dirilis

| | Regresi logistik (baseline) | LightGBM (pembanding) |
|---|---|---|
| Berkas | `artifacts/logistic.json` | `artifacts/lgbm.txt`, `web/public/data/models/lgbm.json` |
| Hiperparameter | L2, C=1 (dipilih dari 7 nilai) | 15 daun, min_child_samples 50, 892 pohon, laju belajar 0,05, subsample 0,8 baris dan fitur |
| Fitur | 14 fitur perilaku; log-transform, imputasi median + indikator kosong, standardisasi, one-hot kategori/jam/hari | 14 fitur yang sama, kategori native |
| Kalibrasi | Platt, dipasang di validasi B | Platt, dipasang di validasi B |
| Penjelasan | Kontribusi linear eksak per fitur | TreeSHAP eksak (Python dan browser) |

Fitur tidak memuat gender, umur, lokasi, pekerjaan, atau ID kartu.

## Kinerja (uji Jul-Des 2020)

Lihat tabel di README dan `reports/results.json`. Ringkasnya LightGBM PR-AUC 0,972 (0,964-0,979) vs logistik 0,709 (0,673-0,741).

## Uji ketahanan (tambahan 30 Okt 2026)

- Data nyata ULB (fitur PCA, tanpa ID kartu): LightGBM PR-AUC 0,803 (0,684-0,897) vs logistik 0,771 (0,627-0,871); selisih +0,032 (−0,010 sampai +0,094) tidak bermakna. Kesimpulan run 2 bahwa LightGBM jauh lebih baik **tidak** berlaku umum.
- Stress test sintetis dengan pengaturan terkunci: label fraud hilang pada 50% kartu hanya menurunkan PR-AUC LightGBM 0,028; pola fraud baru menurunkan recall fraud online dari 99,6% ke 76,0%; nominal naik 25% membuat laju alarm pada ambang 1% menjadi 1,77%; kartu tanpa riwayat membuat PR-AUC jatuh ke 0,510 dengan alarm 2,03%.
- Detail di `reports/external_ulb.json`, `reports/stress.json`, README, dan halaman Uji ketahanan.

## Penjelasan

- Kepentingan global LightGBM (rata-rata |SHAP|): nominal, kategori, total nominal 24 jam, jam, porsi rentang jam dalam riwayat kartu.
- Logistik terutama memakai total nominal 24 jam, rasio nominal terhadap kebiasaan kartu, dan jumlah transaksi 7 hari.
- Cek: aditivitas TreeSHAP 7e-14; shap.TreeExplainer identik dengan `pred_contrib` LightGBM; TreeSHAP di browser cocok dengan Python sampai < 1e-6 pada 400 transaksi uji.
- Stabilitas antar 5 seed: Spearman rata-rata 0,977 (terendah 0,956), irisan lima fitur teratas rata-rata 92%.
- Ablasi (validasi B): tanpa fitur nominal PR-AUC turun dari 0,979 ke 0,591; tanpa velocity ke 0,918; tanpa jam ke 0,949; tanpa fitur kebiasaan kartu 0,967.

## Risiko dan batasan

- Pola fraud sintetis jauh lebih mudah dari kenyataan; ketergantungan kuat pada nominal dan jam adalah sifat generator.
- Ambang dari periode sebelumnya menghasilkan laju alarm yang berbeda (rencana 1%, nyata 0,84%); setelah pergeseran nominal atau pada kartu baru laju alarm bisa dua kali lipat. Di produksi laju alarm harian perlu dipantau dan ambang dikalibrasi ulang.
- Pola fraud yang tidak pernah dilatih tertangkap jauh lebih sedikit; latih ulang perlu dijadwalkan saat pola baru dilaporkan.
- Selisih recall antar kelompok umur kecil tetapi nyata di data ini; harus diaudit ulang pada data nyata.
- Varian logistik + interaksi mencapai ujung grid C; selisihnya terhadap LightGBM mungkin sedikit menyempit dengan grid lebih lebar.
