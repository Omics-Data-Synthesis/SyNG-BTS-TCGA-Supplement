# Data

Nothing in `data/` is committed. `data/raw/`, `data/processed/`, and `data/generated/` are ignored by Git. This directory is the local workspace for the workflow. **The `data/` directory is populated by the scripts, but none of its contents are committed.**

## Released bundles

The augmented datasets are published as **GitHub release assets in the SyNG-BTS package repository**, which should be their only fixed location. This repository may link to them and never uploads copies.

Released bundles:

| Modality | Tag | Cohorts | Total size |
|---|---|---|---|
| `mirna` | [`data-v1.0`](https://github.com/Omics-Data-Synthesis/SyNG-BTS/releases/tag/data-v1.0) | 24 | ~0.5 GB |
| `rnaseq` | [`data-rnaseq-v1.0`](https://github.com/Omics-Data-Synthesis/SyNG-BTS/releases/tag/data-rnaseq-v1.0) | 5 | ~1.6 GB |

Download the augmented datasets using the `syng-bts` package:

```python
from syng_bts import list_tcga_datasets, load_tcga_dataset, tcga_cache_dir

list_tcga_datasets(short=True, modality="rnaseq")   # cohorts in that manifest
ds = load_tcga_dataset("COAD", modality="rnaseq")   # downloads, verifies, caches
```

SyNG-BTS caches files under `~/.cache/syng-bts/tcga/<version>/`, or wherever `SYNG_BTS_CACHE_DIR` points. Set that variable if you would rather keep the bundles inside `data/`:

```bash
export SYNG_BTS_CACHE_DIR="$PWD/data/generated"
```


## Pre-generated versus re-generated

The bundles already ship synthetic sample sets. Stage 04 can also produce them. These are not automatically the same data.

TODO: state which one the published results use. If the results come from the released sets, stage 04 documents how those were produced and is not rerun for the figures. If they come from a rerun, say so, because then stage 04's seed determines the published numbers.

