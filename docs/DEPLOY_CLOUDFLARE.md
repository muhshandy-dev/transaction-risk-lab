# Deploy ke Cloudflare Pages

Situs statis penuh; tidak butuh server, database, atau API key. Model berjalan di browser.

1. Cloudflare dashboard → Workers & Pages → Create → Pages → Connect to Git → pilih `muhshandy-dev/transaction-risk-lab`.
2. Project name: `transaction-risk-lab` (alamat `transaction-risk-lab.pages.dev` bila nama tersedia).
3. Build settings:
   - Framework preset: None
   - Root directory: `web`
   - Build command: `npm ci && npm run build`
   - Build output directory: `dist`
   - Environment variable: `NODE_VERSION` = `22`
4. Save and Deploy. Header keamanan (CSP tanpa sumber luar) ada di `web/public/_headers`.

Setelah live, periksa: buka di jendela incognito dan di ponsel, pilih beberapa transaksi di Meja tinjauan (skor dan faktor muncul dalam kurang dari 1 detik), dan pastikan tidak ada galat di konsol.

Ukuran: model LightGBM (`web/public/data/models/lgbm.json`) adalah berkas terbesar; batas Pages 25 MiB per berkas.
