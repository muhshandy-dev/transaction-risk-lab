// LightGBM inference and exact TreeSHAP (path-dependent, Lundberg et al. 2018) in the browser.
// The decision rule mirrors LightGBM's tree.h (NumericalDecision / CategoricalDecision).
import type { CompactTree, FeatureRow, LgbmModel } from './types'

const ZERO_THRESHOLD = 1e-35
const MISSING_ZERO = 1
const MISSING_NAN = 2

/** Encode a feature row as the numeric vector LightGBM sees (category -> code, null -> NaN). */
export function encode(model: LgbmModel, row: FeatureRow): Float64Array {
  const x = new Float64Array(model.feature_names.length)
  model.feature_names.forEach((name, j) => {
    const v = row[name]
    if (name === 'category') {
      const idx = typeof v === 'string' ? model.categories.indexOf(v) : -1
      x[j] = idx < 0 ? NaN : idx
    } else {
      x[j] = v === null || v === undefined ? NaN : Number(v)
    }
  })
  return x
}

/** Which child a value goes to at an internal node. */
export function goesLeft(tree: CompactTree, node: number, fval: number): boolean {
  const cats = tree.cats?.[node]
  if (cats) {
    if (Number.isNaN(fval)) return false
    const iv = Math.trunc(fval)
    if (iv < 0) return false
    return cats.includes(iv)
  }
  const mt = tree.mt[node]
  let v = fval
  if (Number.isNaN(v) && mt !== MISSING_NAN) v = 0
  if ((mt === MISSING_ZERO && Math.abs(v) <= ZERO_THRESHOLD) || (mt === MISSING_NAN && Number.isNaN(v))) {
    return tree.dl[node] === 1
  }
  return v <= (tree.t[node] as number)
}

function leafOf(tree: CompactTree, x: Float64Array): number {
  let n = 0
  while (tree.f[n] >= 0) n = goesLeft(tree, n, x[tree.f[n]]) ? tree.l[n] : tree.r[n]
  return n
}

export function margin(model: LgbmModel, x: Float64Array): number {
  let s = 0
  for (const tree of model.trees) s += tree.v[leafOf(tree, x)] as number
  return s
}

// ---------------------------------------------------------------- TreeSHAP
// Port of the recursive algorithm in shap's C++ (tree_shap.h). The unique path of every
// recursion level lives in one preallocated buffer; level k starts right after level k-1.
interface Buf {
  feature: Int32Array
  zero: Float64Array
  one: Float64Array
  pw: Float64Array
}

function makeBuf(maxDepth: number): Buf {
  const n = ((maxDepth + 2) * (maxDepth + 3)) / 2 + 2
  return { feature: new Int32Array(n), zero: new Float64Array(n), one: new Float64Array(n), pw: new Float64Array(n) }
}

function extend(b: Buf, o: number, d: number, zero: number, one: number, feature: number) {
  b.feature[o + d] = feature
  b.zero[o + d] = zero
  b.one[o + d] = one
  b.pw[o + d] = d === 0 ? 1 : 0
  for (let i = d - 1; i >= 0; i--) {
    b.pw[o + i + 1] += (one * b.pw[o + i] * (i + 1)) / (d + 1)
    b.pw[o + i] = (zero * b.pw[o + i] * (d - i)) / (d + 1)
  }
}

function unwind(b: Buf, o: number, d: number, idx: number) {
  const one = b.one[o + idx]
  const zero = b.zero[o + idx]
  let next = b.pw[o + d]
  for (let i = d - 1; i >= 0; i--) {
    if (one !== 0) {
      const tmp = b.pw[o + i]
      b.pw[o + i] = (next * (d + 1)) / ((i + 1) * one)
      next = tmp - (b.pw[o + i] * zero * (d - i)) / (d + 1)
    } else {
      b.pw[o + i] = (b.pw[o + i] * (d + 1)) / (zero * (d - i))
    }
  }
  for (let i = idx; i < d; i++) {
    b.feature[o + i] = b.feature[o + i + 1]
    b.zero[o + i] = b.zero[o + i + 1]
    b.one[o + i] = b.one[o + i + 1]
  }
}

function unwoundSum(b: Buf, o: number, d: number, idx: number): number {
  const one = b.one[o + idx]
  const zero = b.zero[o + idx]
  let next = b.pw[o + d]
  let total = 0
  for (let i = d - 1; i >= 0; i--) {
    if (one !== 0) {
      const tmp = (next * (d + 1)) / ((i + 1) * one)
      total += tmp
      next = b.pw[o + i] - (tmp * zero * (d - i)) / (d + 1)
    } else {
      total += b.pw[o + i] / zero / ((d - i) / (d + 1))
    }
  }
  return total
}

function recurse(
  tree: CompactTree,
  x: Float64Array,
  phi: Float64Array,
  b: Buf,
  node: number,
  parentOffset: number,
  depth: number,
  parentZero: number,
  parentOne: number,
  parentFeature: number,
) {
  // copy the parent's path (depth entries) into this level's slot, then extend it
  const o = parentOffset + depth
  if (depth > 0) {
    b.feature.copyWithin(o, parentOffset, parentOffset + depth)
    b.zero.copyWithin(o, parentOffset, parentOffset + depth)
    b.one.copyWithin(o, parentOffset, parentOffset + depth)
    b.pw.copyWithin(o, parentOffset, parentOffset + depth)
  }
  extend(b, o, depth, parentZero, parentOne, parentFeature)
  const split = tree.f[node]
  if (split < 0) {
    const value = tree.v[node] as number
    for (let i = 1; i <= depth; i++) {
      const w = unwoundSum(b, o, depth, i)
      phi[b.feature[o + i]] += w * (b.one[o + i] - b.zero[o + i]) * value
    }
    return
  }
  const left = tree.l[node]
  const right = tree.r[node]
  const hot = goesLeft(tree, node, x[split]) ? left : right
  const cold = hot === left ? right : left
  const w = tree.c[node]
  const hotZero = tree.c[hot] / w
  const coldZero = tree.c[cold] / w
  let inZero = 1
  let inOne = 1
  let d = depth
  let k = 0
  while (k <= d && b.feature[o + k] !== split) k++
  if (k !== d + 1) {
    inZero = b.zero[o + k]
    inOne = b.one[o + k]
    unwind(b, o, d, k)
    d -= 1
  }
  recurse(tree, x, phi, b, hot, o, d + 1, hotZero * inZero, inOne, split)
  recurse(tree, x, phi, b, cold, o, d + 1, coldZero * inZero, 0, split)
}

function treeDepth(tree: CompactTree, node = 0): number {
  if (tree.f[node] < 0) return 0
  return 1 + Math.max(treeDepth(tree, tree.l[node]), treeDepth(tree, tree.r[node]))
}

/** Expected value of a tree under its training cover. */
function treeMean(tree: CompactTree): number {
  let s = 0
  for (let i = 0; i < tree.f.length; i++) if (tree.f[i] < 0) s += (tree.v[i] as number) * tree.c[i]
  return s / tree.c[0]
}

const prepared = new WeakMap<LgbmModel, { buf: Buf; means: number[] }>()

/** SHAP values per feature plus the base value; sum(phi) + base equals margin(). */
export function shap(model: LgbmModel, x: Float64Array): { phi: Float64Array; base: number } {
  let prep = prepared.get(model)
  if (!prep) {
    const maxDepth = Math.max(0, ...model.trees.map((t) => treeDepth(t)))
    prep = { buf: makeBuf(maxDepth), means: model.trees.map((t) => (t.f.length === 1 ? (t.v[0] as number) : treeMean(t))) }
    prepared.set(model, prep)
  }
  const phi = new Float64Array(model.feature_names.length)
  let base = 0
  model.trees.forEach((tree, ti) => {
    base += prep.means[ti]
    if (tree.f.length > 1) recurse(tree, x, phi, prep.buf, 0, 0, 0, 1, 1, -1)
  })
  return { phi, base }
}
