export type FeatureValue = number | string | null

export type FeatureRow = Record<string, FeatureValue>

export interface CompactTree {
  f: number[]
  t: (number | null)[]
  cats?: (number[] | null)[]
  dl: number[]
  mt: number[]
  l: number[]
  r: number[]
  v: (number | null)[]
  c: number[]
}

export interface LgbmModel {
  type: 'lgbm'
  feature_names: string[]
  categories: string[]
  num_trees: number
  trees: CompactTree[]
}

export interface NumericSpec {
  name: string
  log1p: boolean
  median: number
  missing_indicator: boolean
  mean: number
  std: number
}

export interface LogisticSpec {
  type: 'logistic'
  C: number
  numeric: NumericSpec[]
  onehot: Record<string, (string | number)[]>
  columns: string[]
  coef: number[]
  intercept: number
  design_mean: number[]
}

export interface Platt {
  a: number
  b: number
}
