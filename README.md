# PJCS-FL

Reproduction materials for **Joint Client Sampling with Reusable Synthetic Data for Non-IID Federated Learning**, by Di Wu.

**Preparation in progress.** This private preparation snapshot contains the retained numerical sources, source and input inventories, licenses, and a verified server export tool. The end-to-end portable training entry point and downloadable data assets are not yet complete. Repository: <https://github.com/wudi2071458152/PJCS-FL>.

The primary experiment uses real and proxy local learning rates of **0.01**, synthetic data optimized for **32,000 iterations**, two datasets (CIFAR-10 and Fashion-MNIST), five Dirichlet configurations, and five federated training seeds. Learning-rate-0.1 experiments are sensitivity results and must remain separate.

The recorded experiment specification is [configs/paper_lr001.json](configs/paper_lr001.json). Its ten methods are FedAvg, Power-of-Choice, Practical IS-HT, Practical DELTA-HT, DivFL, static equal-weight Gram HCSFed, uniform HT, proxy-score HT, proxy-score plus clustering K=3, and PJCS (eta=1). The unrelated wireless study is outside this release.

Pre-generated 32k proxy datasets will be distributed as separate download assets with checksums. Raw benchmark datasets will be obtained from their original sources. Training checkpoints, machine-specific deployment archives, and private working notes will not be committed to the repository.

## Source snapshot and checks

`legacy/` retains 51 Python files and four JSON files without numerical edits. [The source manifest](assets/source_manifest.json) records exact SHA256 fingerprints and source paths. [The source guide](docs/frozen_source.md) explains the entry points and passive historical dependencies. These retained workers still enforce their historical input bindings; copying the source alone is insufficient for an end-to-end run.

`requirements.txt` lists the direct dependencies. `requirements-historical.txt` records the observed paper environment rather than claiming a complete environment lockfile. The preparation tools use only the Python standard library (Python 3.9 or later).

From this repository root:

```bash
python -B tools/verify_release.py
python -B -m unittest discover -s tests -p "test_*.py"
```

The first command checks frozen source bytes and inventory coverage; it does not imply that absent proxy tensors have been validated. Tests use small generated file fixtures and do not run formal training. The numerical snapshot was also checked for Python compilation, JSON parsing and four FL CLI help invocations.

## Proxy data recovery

[The proxy inventory](assets/proxy_manifest.json) records expected hashes from the formal LR=0.01 PJCS protocols. There are 320 snapshots: 96 CIFAR-10 snapshots are already available locally; 224 snapshots and 54 required support files must be recovered from the server. See [the inventory guide](docs/proxy_inventory.md) and [the export instructions](docs/export_proxy_assets.md). Raw benchmark data are not part of that export.

The export tool defaults to read-only verification. Only `--export` writes an archive; the tool never starts synthesis or FL training. Public dataset download links will be added after asset acceptance.

## Synthesis lineage

For each data partition, two shared FedAvg reference trajectories are generated with seeds 2025 and 2026. Each client optimizes its own synthetic dataset using local real data, evaluated at model states sampled from these reference trajectories. This description follows the recorded input protocols; it does not describe independently trained reference trajectories for each client.

See [the release preparation record](docs/release_preparation.md) for the evidence and remaining work. The manuscript itself has not been modified. Training quick-start commands will be added after the portable input binding and returned data assets have been checked.

## Citation and license

Author: **Di Wu**. Software citation metadata is in [CITATION.cff](CITATION.cff). No DOI or publication status is assumed in this draft.

Original contributions are intended to be released under the [MIT license](LICENSE). Third-party material retains its own notices and license conditions; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The manuscript's publication license does not determine the software or dataset license.
