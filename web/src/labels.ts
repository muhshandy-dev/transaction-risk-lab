import { int, num, pct, usd } from './format'
import type { FeatureValue } from './model/types'

export const CATEGORY_LABEL: Record<string, string> = {
  entertainment: 'Hiburan',
  food_dining: 'Makan & minum',
  gas_transport: 'BBM & transportasi',
  grocery_net: 'Kebutuhan harian (online)',
  grocery_pos: 'Kebutuhan harian (toko)',
  health_fitness: 'Kesehatan & kebugaran',
  home: 'Rumah tangga',
  kids_pets: 'Anak & hewan peliharaan',
  misc_net: 'Lain-lain (online)',
  misc_pos: 'Lain-lain (toko)',
  personal_care: 'Perawatan diri',
  shopping_net: 'Belanja (online)',
  shopping_pos: 'Belanja (toko)',
  travel: 'Perjalanan',
}

const DAYS = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']

export const FEATURE_LABEL: Record<string, string> = {
  amt: 'Nominal transaksi',
  category: 'Kategori merchant',
  hour: 'Jam transaksi',
  dow: 'Hari',
  distance_km: 'Jarak rumah ke merchant',
  secs_since_prev: 'Jeda sejak transaksi sebelumnya',
  n_1h: 'Transaksi kartu 1 jam terakhir',
  n_24h: 'Transaksi kartu 24 jam terakhir',
  n_7d: 'Transaksi kartu 7 hari terakhir',
  amt_24h: 'Total nominal kartu 24 jam terakhir',
  amt_ratio_card: 'Nominal dibanding rata-rata kartu (90 hari)',
  amt_ratio_card_cat: 'Nominal dibanding rata-rata kartu di kategori ini',
  cat_share_card: 'Porsi kategori ini dalam riwayat kartu',
  hourband_share_card: 'Porsi rentang jam ini dalam riwayat kartu',
  'category*amt': 'Interaksi kategori x nominal',
}

export function duration(secs: number): string {
  if (secs < 90) return `${int(secs)} detik`
  if (secs < 5400) return `${int(secs / 60)} menit`
  if (secs < 172800) return `${num(secs / 3600, 1)} jam`
  return `${num(secs / 86400, 1)} hari`
}

export function featureValue(name: string, v: FeatureValue): string {
  if (v === null || v === undefined || (typeof v === 'number' && !Number.isFinite(v))) {
    return name === 'secs_since_prev' ? 'transaksi pertama' : 'belum ada riwayat'
  }
  const x = Number(v)
  switch (name) {
    case 'amt':
    case 'amt_24h':
      return usd(x)
    case 'category':
      return CATEGORY_LABEL[String(v)] ?? String(v)
    case 'hour':
      return `${String(x).padStart(2, '0')}.00–${String(x).padStart(2, '0')}.59 UTC`
    case 'dow':
      return DAYS[x] ?? String(v)
    case 'distance_km':
      return `${int(x)} km`
    case 'secs_since_prev':
      return duration(x)
    case 'n_1h':
    case 'n_24h':
    case 'n_7d':
      return `${int(x)} transaksi`
    case 'amt_ratio_card':
    case 'amt_ratio_card_cat':
      return `${num(x, x < 10 ? 2 : 1)}×`
    case 'cat_share_card':
    case 'hourband_share_card':
      return pct(x, 0)
    default:
      return num(x, 2)
  }
}

export const MODEL_COLOR: Record<string, string> = {
  lgbm: 'var(--forest)',
  logistic: 'var(--ochre)',
  logistic_ix: 'var(--plum-muted)',
  rule_amount: 'var(--slate)',
  random: 'var(--muted)',
}

export const MODEL_SHORT: Record<string, string> = {
  lgbm: 'LightGBM',
  logistic: 'Logistik',
  logistic_ix: 'Logistik + interaksi',
  rule_amount: 'Aturan nominal',
  random: 'Acak',
}

export const AUDIT_LABEL: Record<string, string> = {
  audit_gender: 'Gender (di data sintetis)',
  audit_age_band: 'Kelompok umur',
  audit_area: 'Kota atau desa',
}

export const BUCKET_LABEL: Record<string, string> = {
  caught: 'Fraud tertangkap',
  missed: 'Fraud terlewat',
  false_alert: 'Alarm palsu',
  quiet: 'Sah, tidak ditandai',
}
