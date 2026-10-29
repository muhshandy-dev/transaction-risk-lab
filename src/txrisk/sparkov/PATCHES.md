# Sparkov vendored copy

Source: https://github.com/namebrandon/Sparkov_Data_Generation, commit
`b5eb45c89d36f2aa4ef16044a42945bed8b96d93` (22 Jun 2022), MIT licence (see LICENSE.md).

Files copied: `datagen_customer.py` (as `customer.py`), `datagen_transaction.py` (as
`transaction.py`), `profile_weights.py`, `main_config.py`, `profiles/*.json` (without
`template.json`), `data/merchants.csv`, `demographic_data/age_gender_demographics.csv`,
`demographic_data/locations_partitions.csv`.

The generation logic (profiles, fraud windows, amounts, hours) is unchanged. Changes:

| ID | File | Change | Why |
|----|------|--------|-----|
| P1 | customer.py, transaction.py | Data files are read relative to the package | Run from any directory |
| P2 | customer.py | `date.today()` replaced by fixed `REF_DATE = 2021-01-01` | Ages otherwise depend on the day the generator runs |
| P3 | customer.py | Profile-overlap warnings no longer appended to a log file | No side effects in the working directory |
| P4 | customer.py, transaction.py | Package-relative imports | Packaging |
| P5 | profile_weights.py | `closest_rand_parallel` compares column `i` instead of always column 1 | Bug fix, see below |

### P5: category and time of day were tied to the date

`sample_from` draws two uniform columns: column 1 for the date and column 2 for category and
AM/PM. `closest_rand_parallel(r, i, j, obj)` is meant to map column `i`, but upstream line
`np.where(r2[:,1] <= x, ...)` always uses column 1. So the category and the AM/PM part of every
transaction were a deterministic function of its date: in any month a card's legitimate
transactions came from only two or three categories, rotating through the year, while fraud
windows (sampled separately) spread over many categories. A model could then flag fraud by
"category that is unusual for this time of year", which is a generator artefact, not a fraud
pattern. Found on 29 Oct 2026 during failure analysis (legitimate test transactions only
appeared in 6 of 14 categories). The first full run used the buggy data and was discarded;
its reports are kept in `docs/riwayat/run1-generator-bug/` for transparency.
`tests/test_data.py::test_category_does_not_depend_on_date` guards the fix.

Determinism is added by the driver `txrisk/generate.py`, not by editing the generator:
it seeds `random`, `numpy.random` and Faker before each profile task, runs with `TZ=UTC`,
and redirects stdout to a file that is closed properly (upstream leaves it open).
