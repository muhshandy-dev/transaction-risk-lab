# Kartu model

## Kegunaan

Mengurutkan transaksi kartu menurut prioritas tinjauan oleh petugas. Bukan untuk menolak transaksi atau menilai orang secara otomatis. Dilatih dan diuji hanya pada data sintetis Sparkov; tidak divalidasi pada data bank mana pun.

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

## Penjelasan

- Kepentingan global LightGBM (rata-rata |SHAP|): nominal, kategori, total nominal 24 jam, jam, porsi rentang jam dalam riwayat kartu.
- Logistik terutama memakai total nominal 24 jam, rasio nominal terhadap kebiasaan kartu, dan jumlah transaksi 7 hari.
- Cek: aditivitas TreeSHAP 7e-14; shap.TreeExplainer identik dengan `pred_contrib` LightGBM; TreeSHAP di browser cocok dengan Python sampai < 1e-6 pada 400 transaksi uji.
- Stabilitas antar 5 seed: Spearman rata-rata 0,977 (terendah 0,956), irisan lima fitur teratas rata-rata 92%.
- Ablasi (validasi B): tanpa fitur nominal PR-AUC turun dari 0,979 ke 0,591; tanpa velocity ke 0,918; tanpa jam ke 0,949; tanpa fitur kebiasaan kartu 0,967.

## Risiko dan batasan

- Pola fraud sintetis jauh lebih mudah dari kenyataan; ketergantungan kuat pada nominal dan jam adalah sifat generator.
- Ambang dari periode sebelumnya menghasilkan laju alarm yang berbeda (rencana 1%, nyata 0,84%); di produksi perlu pemantauan.
- Selisih recall antar kelompok umur kecil tetapi nyata di data ini; harus diaudit ulang pada data nyata.
- Varian logistik + interaksi mencapai ujung grid C; selisihnya terhadap LightGBM mungkin sedikit menyempit dengan grid lebih lebar.
