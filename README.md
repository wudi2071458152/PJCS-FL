# PJCS-FL

Reproduction materials for **Joint Client Sampling with Reusable Synthetic Data for Non-IID Federated Learning**, by Di Wu.

**Source release published:** [v1.0.0](https://github.com/wudi2071458152/PJCS-FL/releases/tag/v1.0.0). This repository is public. The initialization attachment and checksum file are available and match the prepared SHA256 values; the CIFAR-10 and Fashion-MNIST data attachments are still missing from the release and must be uploaded before the full download workflow is available. All data files have already been accepted and packaged locally. See [download status](docs/downloads.md).

The primary experiment uses real and proxy local learning rates of **0.01**, synthetic data optimized for **32,000 iterations**, two datasets (CIFAR-10 and Fashion-MNIST), five Dirichlet configurations, and five federated training seeds. Learning-rate-0.1 experiments are sensitivity results and must remain separate.

The recorded experiment specification is [configs/paper_lr001.json](configs/paper_lr001.json). Its ten methods are FedAvg, Power-of-Choice, Practical IS-HT, Practical DELTA-HT, DivFL, static equal-weight Gram HCSFed, uniform HT, proxy-score HT, proxy-score plus clustering K=3, and PJCS (eta=1). The unrelated wireless study is outside this release.

Pre-generated 32k proxy datasets are distributed as separate release attachments with checksums once their uploads are complete. Raw benchmark datasets are obtained from their original sources. Training checkpoints, machine-specific deployment archives, and private working notes are excluded from source commits.

## Source snapshot and checks

`legacy/` retains 51 Python files and four JSON files without numerical edits. [The source manifest](assets/source_manifest.json) records exact SHA256 fingerprints and source paths. [The source guide](docs/frozen_source.md) explains the entry points and passive historical dependencies. `run_fl.py` adds portable input binding while reusing the retained numerical kernels and checkpoint loop; historical output protocols remain separate.

`requirements.txt` lists the direct dependencies. `requirements-historical.txt` records the observed paper environment rather than claiming a complete environment lockfile. Downloading and file-only packaging/verification use the Python standard library (Python 3.9 or later); FL and initialization generation/acceptance require PyTorch and the retained implementation's dependencies.

From this repository root:

```bash
python -B tools/verify_release.py
python -B -m unittest discover -s tests -p "test_*.py"
```

The first command checks frozen source bytes and inventory coverage; it does not inspect absent tensor files unless an artifact root is supplied. Tests use small generated file or CPU fixtures and do not run formal training. The numerical snapshot was also checked for Python compilation, JSON parsing and four FL CLI help invocations.

## Proxy data recovery

[The proxy inventory](assets/proxy_manifest.json) records expected hashes from the formal LR=0.01 PJCS protocols. All 320 snapshots and 54 support files have now passed local acceptance; the combined dataset contains 110,000 synthetic images. See [the acceptance record](docs/data_acceptance.md), [the inventory guide](docs/proxy_inventory.md), and [the export instructions](docs/export_proxy_assets.md). Raw benchmark data are separate.

The export tools default to in-memory/read-only output preflight. Only `--export` writes an archive; neither tool starts synthesis or FL training. The eight supplemental initial states for seeds 2027/2028/2029/2030 have also passed [acceptance](docs/initialization_acceptance.md). Their formal tensor fingerprints are fixed in [the initialization manifest](assets/initialization_manifest.json); initializers from a different environment must not silently replace them. [The download guide](docs/downloads.md) describes the three prepared attachments and checksum-verified installation; URLs remain planned until publication.

## Run and report

See [the reproduction workflow](docs/reproduce.md) and [single-run guide](docs/portable_fl.md). Both `run_fl.py` and `tools/schedule_fl.py` default to preparation/plan mode; training requires `--run`. The default matrix is 500 runs, with the 100 seed2025 runs prioritized before the other seeds. The scheduler creates [verified compact result archives](docs/results_packaging.md) after the first seed and final matrix. No formal training was started while preparing this repository.

[The recorded table](results/table.md) and [five-seed values](results/lr001_tail300.json) summarize the already audited LR=0.01 experiments. Run `python -B tools/report_results.py` to regenerate the table from those records. Stars identify the highest unrounded mean within each dataset/alpha column and do not denote statistical significance.

## Synthesis lineage

For each data partition, two shared FedAvg reference trajectories are generated with seeds 2025 and 2026. Each client optimizes its own synthetic dataset using local real data, evaluated at model states sampled from these reference trajectories. This description follows the recorded input protocols; it does not describe independently trained reference trajectories for each client.

See [the release preparation record](docs/release_preparation.md) for the evidence and remaining work. The manuscript itself has not been modified. The retained synthesis source is available; a complete portable from-scratch synthesis scheduler is outside the accepted quick-FL binding described here.

## Citation and license

Author: **Di Wu**. Software citation metadata is in [CITATION.cff](CITATION.cff). No DOI or publication status is assumed in this draft.

Original contributions are intended to be released under the [MIT license](LICENSE). Third-party material retains its own notices and license conditions; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The manuscript's publication license does not determine the software or dataset license.

The author's synthetic-data and initial-model asset contributions use [CC BY 4.0](DATA_LICENSE.txt); original benchmark rights and attribution remain with their sources. Raw benchmarks are downloaded separately, not redistributed in the PJCS attachments.
