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

Determinism is added by the driver `txrisk/generate.py`, not by editing the generator:
it seeds `random`, `numpy.random` and Faker before each profile task, runs with `TZ=UTC`,
and redirects stdout to a file that is closed properly (upstream leaves it open).
