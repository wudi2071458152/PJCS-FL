# PJCS-FL

Reproduction materials for **Joint Client Sampling with Reusable Synthetic Data for Non-IID Federated Learning**, by Di Wu.

**Preparation in progress.** This repository currently contains release metadata and preparation documentation. It is not yet an executable reproduction package or a completed release. Repository: <https://github.com/wudi2071458152/PJCS-FL>.

The primary experiment uses real and proxy local learning rates of **0.01**, synthetic data optimized for **32,000 iterations**, two datasets (CIFAR-10 and Fashion-MNIST), five Dirichlet configurations, and five federated training seeds. Learning-rate-0.1 experiments are sensitivity results and must remain separate.

The release will contain the PJCS sampling and HT aggregation code, data preparation and synthesis, the ten client-selection methods used in the paper, exact partition metadata, experiment configurations, and scripts to validate and summarize results. The unrelated wireless study is outside this release.

Pre-generated 32k proxy datasets will be distributed as separate download assets with checksums. Raw benchmark datasets will be obtained from their original sources. Training checkpoints, machine-specific deployment archives, and private working notes will not be committed to the repository.

See [the release preparation record](docs/release_preparation.md) for confirmed settings, source tracing, and remaining work. This draft deliberately provides no training commands until the portable entry points have been checked.

## Citation and license

Author: **Di Wu**. Software citation metadata is in [CITATION.cff](CITATION.cff). No DOI or publication status is assumed in this draft.

Original contributions are intended to be released under the [MIT license](LICENSE). Third-party material retains its own notices and license conditions; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The manuscript's publication license does not determine the software or dataset license.
