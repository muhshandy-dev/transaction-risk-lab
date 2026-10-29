import { useEffect, useState } from 'react'
import type { FeatureRow, LgbmModel, LogisticSpec, Platt } from './model/types'

export interface Interval {
  lo: number
  hi: number
  n_boot: number
}
export interface CalBin {
  lo: number
  hi: number
  n: number
  mean_pred: number
  frac_pos: number
}
export interface Calibration {
  brier: number
  log_loss: number
  ece: number
  ece10: number
  ece_alert_region: number | null
  mean_pred: number
  frac_pos: number
  bins: CalBin[]
}
export interface BudgetPoint {
  budget: number
  recall: number
  precision: number
  amount_recall: number
}
export interface ModelResult {
  label: string
  ap: number
  roc_auc: number
  at_budget: Record<string, { recall: number; precision: number; amount_recall: number }>
  calibration?: { raw: Calibration; platt: Calibration; isotonic: Calibration }
  platt?: Platt
  operational?: Record<
    string,
    { threshold: number; alert_rate: number; recall: number; precision: number; fpr: number; amount_recall: number }
  >
  pr_curve?: { recall: number; precision: number; threshold: number }[]
}
export interface Paired {
  mean: number
  lo: number
  hi: number
  n_boot: number
  share_gt_0: number
}
export interface Metrics {
  models: Record<string, ModelResult>
  ci: Record<string, Record<string, Interval>>
  paired_lgbm_minus_logistic: Record<string, Paired>
  paired_lgbm_minus_logistic_ix: Record<string, Paired>
  budget_curves: Record<string, BudgetPoint[]>
}
export interface SplitInfo {
  rows: number
  fraud: number
  fraud_rate: number
  cards: number
  fraud_cards: number
  start: string
  end: string
}
export interface Meta {
  version: string
  evaluated: string
  protocol: Record<string, unknown> & { budgets: number[]; primary_budget: number; n_boot: number }
  data: Record<string, SplitInfo>
  manifest: Record<string, string | number> | null
  environment: Record<string, string>
  selection_best: Record<string, Record<string, number>>
  features: { name: string; label: string }[]
}
export interface Case {
  id: string
  bucket: 'missed' | 'false_alert' | 'caught' | 'quiet'
  ts: string
  category: string
  amt: number
  is_fraud: number
  features: FeatureRow
  card_mean_90d: number | null
  card_cat_mean_90d: number | null
  python: {
    lgbm_margin: number
    lgbm_shap: number[]
    lgbm_base: number
    logistic_margin: number
    logistic_contrib: Record<string, number>
    logistic_base: number
    lgbm_prob: number
    logistic_prob: number
  }
}
export interface Explain {
  sample_size: number
  lgbm_base_value: number
  logistic_base_value: number
  checks: Record<string, number>
  global: {
    lgbm: { feature: string; label: string; mean_abs_shap: number }[]
    logistic: { feature: string; label: string; mean_abs_contrib: number }[]
  }
  seed_stability: {
    seeds: number[]
    spearman_min: number
    spearman_mean: number
    top5_overlap_min: number
    top5_overlap_mean: number
    rank_by_seed: Record<string, number[]>
  }
  dependence: Record<string, { x: number | string; shap: number; fraud: number }[]>
  logistic_coefficients: { column: string; coef: number }[]
}
export interface GroupRate {
  group: string
  rate: number
  n: number
}
export interface Failures {
  budget: number
  threshold: number
  fraud_missed: number
  fraud_total: number
  false_alerts: number
  alerts: number
  missed_by_lgbm_caught_by_logistic: number
  caught_by_lgbm_missed_by_logistic: number
  recall_by_category: GroupRate[]
  recall_by_amount_band: GroupRate[]
  recall_night_vs_day: GroupRate[]
  false_alert_rate_by_category: GroupRate[]
  false_alert_rate_by_amount_band: GroupRate[]
  episodes: Record<string, number>
}
export interface FairRow {
  group: string
  n: number
  fraud: number
  cards: number
  logistic: { tpr: number; tpr_ci: Interval; fpr: number; fpr_ci: Interval; alert_rate: number }
  lgbm: { tpr: number; tpr_ci: Interval; fpr: number; fpr_ci: Interval; alert_rate: number }
}
export interface Fairness {
  budget: number
  groups: Record<string, FairRow[]>
}
export interface Ablations {
  results: { set: string; dropped: string[]; val_b_ap: number; val_b_recall_at_1pct: number }[]
}
export interface Selection {
  grid: Record<string, Record<string, number>[]>
  best: Record<string, Record<string, number>>
}
export interface CalibrationFile {
  platt: Record<string, Platt>
  thresholds: Record<string, Record<string, number>>
  primary_budget: number
}

export interface Bundle {
  meta: Meta
  metrics: Metrics
  explain: Explain
  failures: Failures
  fairness: Fairness
  ablations: Ablations | null
  selection: Selection
  cases: { model_version: string; evaluated: string; cases: Case[] }
}

const base = `${import.meta.env.BASE_URL}data/`

async function get<T>(path: string): Promise<T> {
  const r = await fetch(base + path)
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`)
  return (await r.json()) as T
}

export function useBundle() {
  const [state, setState] = useState<{ data?: Bundle; error?: string }>({})
  useEffect(() => {
    Promise.all([
      get<Meta>('meta.json'),
      get<Metrics>('metrics.json'),
      get<Explain>('explain.json'),
      get<Failures>('failures.json'),
      get<Fairness>('fairness.json'),
      get<Ablations>('ablations.json').catch(() => null),
      get<Selection>('selection.json'),
      get<Bundle['cases']>('cases.json'),
    ])
      .then(([meta, metrics, explain, failures, fairness, ablations, selection, cases]) =>
        setState({ data: { meta, metrics, explain, failures, fairness, ablations, selection, cases } }),
      )
      .catch((e: Error) => setState({ error: e.message }))
  }, [])
  return state
}

export interface Models {
  lgbm: LgbmModel
  logistic: LogisticSpec
  calibration: CalibrationFile
}

let modelsPromise: Promise<Models> | null = null
export function loadModels(): Promise<Models> {
  modelsPromise ??= Promise.all([
    get<LgbmModel>('models/lgbm.json'),
    get<LogisticSpec>('models/logistic.json'),
    get<CalibrationFile>('models/calibration.json'),
  ]).then(([lgbm, logistic, calibration]) => ({ lgbm, logistic, calibration }))
  return modelsPromise
}

export function useModels() {
  const [state, setState] = useState<{ models?: Models; error?: string }>({})
  useEffect(() => {
    loadModels()
      .then((models) => setState({ models }))
      .catch((e: Error) => setState({ error: e.message }))
  }, [])
  return state
}
