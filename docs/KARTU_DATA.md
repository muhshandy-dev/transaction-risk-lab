# Kartu data

## Sumber

- Generator: Sparkov_Data_Generation, Brandon Harris, lisensi MIT, commit `b5eb45c89d36f2aa4ef16044a42945bed8b96d93`. Disalin ke `src/txrisk/sparkov/` dengan patch yang tercatat di `PATCHES.md` (path berkas, tanggal acuan umur tetap 2021-01-01, tanpa berkas log). Logika profil dan fraud tidak diubah.
- Konfigurasi: 1.000 nasabah, 1 Jan 2019 sampai 31 Des 2020, seed 42. Driver `txrisk/generate.py` memberi seed tersendiri untuk tiap profil dan menjalankan dengan TZ=UTC, sehingga keluaran identik antar run dan jumlah proses.
- Checksum hasil: `data/manifest.json` (dibuat oleh `txrisk run`, SHA-256 atas isi tabel mentah dan bersih).

## Isi

Satu baris = satu transaksi kartu. Hasil: 1.748.617 transaksi, 993 kartu (7 nasabah tidak punya transaksi), 9.410 berlabel fraud (0,54%). Angka per bagian waktu ada di `reports/results.json` bagian `data`.

## Cara Sparkov membuat fraud (dibaca dari kode)

- Hampir setiap nasabah (99 dari 101) mendapat tepat satu jendela fraud 1 sampai 2 hari pada tanggal acak.
- Selama jendela itu transaksi sah pada tanggal yang sama dihapus; transaksi fraud diambil dari profil fraud: 2 sampai 25 transaksi per hari, 80% di jam 22.00-04.00, nominal dengan rata-rata khas per kategori dan simpangan kecil (misalnya belanja online sekitar $1.000 ± 100, BBM sekitar $10 ± 2).
- Lokasi merchant fraud dibuat sama seperti transaksi sah (sekitar rumah), jadi jarak tidak informatif.

Artinya pola fraud di sini jauh lebih mudah daripada fraud nyata. Model yang bagus di data ini belum tentu bagus di bank.

## Data pribadi

Sparkov mengisi nama, alamat, nomor kartu, SSN, pekerjaan, dan tanggal lahir palsu dengan Faker. `txrisk/ingest.py` membuang semuanya sebelum fitur dibuat; kartu diganti ID `K0001` dan seterusnya; koordinat diringkas menjadi jarak km lalu dibuang. Gender, kelompok umur, dan kota/desa disimpan sebagai kolom `audit_*` dan tidak pernah dipakai sebagai fitur (dites di `tests/test_data.py`). Website tidak memuat ID kartu sama sekali.

## Batasan

- Sintetis; porsi fraud, jam, dan nominal ditentukan konfigurasi generator.
- Satu episode per kartu; tidak ada fraud berulang, tidak ada adaptasi pelaku.
- Gender dan umur juga sintetis.
