# Release preparation record

Updated: 2026-11-27 (Asia/Shanghai). This is a local working record, not a completed public release.

## Confirmed scope

- GitHub account: `wudi2071458152` (authenticated login verified on 2026-11-27).
- Repository: `wudi2071458152/PJCS-FL`; verified private with read/write access. Empty at the initial connection check.
- Copyright holder: Di Wu.
- Primary real/proxy local learning rate: 0.01 for both.
- Synthetic optimization: existing old-loss 32,000-iteration data, not regenerated to accommodate documentation changes.
- Datasets: CIFAR-10 and Fashion-MNIST.
- Dirichlet alpha: 0.05, 0.07, 0.1, 0.2, 0.3.
- Federated seeds: 2025, 2027, 2028, 2029, 2030; exact partitions and synthetic images are shared between training seeds.
- Federated training: 32 clients, 3 selected per round, 1,500 rounds, 3 consecutive local SGD steps, batch size capped at 256, no momentum or weight decay.
- Real loss: ordinary cross entropy. Proxy loss: the existing class-weighted cross entropy. Do not replace either during packaging.
- Publish pre-generated final 32k proxy data as separately checksummed assets.

## Methods and invariants

The ten client-selection methods are FedAvg, Power-of-Choice, Practical IS-HT, Practical DELTA-HT, DivFL, static equal-weight Gram HCSFed, uniform HT, proxy-score HT, proxy-score plus clustering K=3, and PJCS (eta=1).

This is not a license or naming claim that the adapted baselines are byte-identical to their authors' programs. Their adaptation details must be recorded separately.

In HT routes, the aggregate is `sum_selected(w_i / Pi_i * R_i)` with global real-sample fractions `w_i` and the actual inclusion probability `Pi_i`, without an additional normalization. FedAvg and the publicly implemented baseline-specific aggregations retain their confirmed rules. PJCS preserves the current AOCS marginals and optimizes the joint distribution of client triples; it does not change those marginals or substitute single-draw probabilities for inclusion probabilities.

SCAFFOLD/residual extensions and the separate wireless study are outside the first release scope unless the author adds them explicitly.

## Required release files

1. Numerical kernels for local updates, scoring, fixed-size sampling, clustering, PJCS LP and its certificate checks, and aggregation; preserve historical numerical definitions.
2. Network/data preprocessing, raw-data download checksums, exact train/test partition indices, class counts and capacity allocation.
3. Reference-trajectory, coefficient-calibration, and old-loss synthesis programs; independent portable paths rather than references to the author's filesystem.
4. A primary LR=0.01 configuration and scheduler, plus validation and result-analysis scripts; LR=0.1 belongs in separate sensitivity configurations.
5. Compact per-seed results and provenance linking every reported result to its configuration, source code, initialization and proxy fingerprints.
6. README with tested commands, dependency/environment record, citation metadata, MIT for original work, and exact third-party notices and license texts.
7. Checksummed data-download assets and an index of their contents. Include only inputs actually needed to run FL; synthesis optimizer state is a separate resume-only asset if offered.

Do not upload the entire historical experiment directory. Raw benchmarks, full result/deployment archives, arbitrary checkpoints, account credentials, private correspondence, review/submission materials and unrelated projects are excluded.

## Source correspondence requiring resolution

The manuscript Appendix A describes reference trajectories trained independently on each client's real data. The historical reference generator and the checked source protocols for all five CIFAR-10 configurations and all five Fashion-MNIST configurations instead record a shared FedAvg reference pool (`reference_rounds=1500`, `reference_selected=3`, selected-sample-weighted FedAvg aggregation).

The reference seeds are 2025 and 2026. Synthetic optimization samples the state groups `[0,10,30,75]`, `[150,250,400,600]`, and `[750,1000,1200]`. The last sampled round, 1200, is not a per-client training-step budget; the reference trajectories themselves run for 1500 rounds. Each client's synthetic loss still uses that client's own real data.

Proposed factual wording: "For each data partition, two shared FedAvg reference trajectories are generated with seeds 2025 and 2026. Each client optimizes its own synthetic dataset using local real data, evaluated at model states sampled from these reference trajectories."

Evidence paths below are relative to the separate historical experiment project, not requirements to preserve the author's machine paths in the public program:

| Scope | Recorded evidence |
| --- | --- |
| CIFAR-10 alpha=0.05 | `received_results/dirichlet_oldserver_28_20261012/dirichlet_oldserver_results.tar.gz`, member `a0p05_b20/source24000/protocol.json` |
| CIFAR-10 alpha=0.07 | `received_results/dirichlet_a007_20261015/dirichlet_a007_results.tar.gz`, source `protocol.json` and `synthesis_source/reference/0p07/2025/COMPLETED.json` |
| CIFAR-10 alpha=0.1/0.2/0.3 | `synthesis_label2_study/dirichlet_continue32000_source_protocol.json` |
| Fashion-MNIST five configurations | `analysis_outputs/fmnist_continue32000_20261021/audit_sources/continue32k__protocol.json`, nested `source_protocol` |
| Fashion-MNIST source mapping | `analysis_outputs/fmnist_joint_mechanism_20261021/extracted/input_snapshot/source_map.json` |

The frozen reference script SHA256 shared by checked alpha=0.07 and Fashion-MNIST source protocols is `3e95728c684b0c5f743734d4bb4a2c953a5226badab303c71986afa9b34b9c21`.

The 96 local final proxy snapshots for CIFAR-10 alpha=0.1/0.2/0.3 under `inputs/dirichlet_pracis_windows_20261012/original32000/synthesis/` have been matched byte-for-byte by SHA256 to `files_sha256["proxy/client_XX/snapshot.pt"]` in the formal seed2025 PJCS protocols under `inputs/dirichlet_baseline191_20261015/reused/seed2025/{alpha}/representative_real_eta1p0/protocol.json`. All 96 matched. This verifies a reusable subset; it does not constitute acceptance of every asset planned for publication. The remaining CIFAR-10 configurations and Fashion-MNIST image files still need complete release-asset inventory and checksum verification.

Present this source correspondence to the author before finalizing the public reproduction description or revising the manuscript. Do not change numerical code or regenerate results to make them resemble the prose. No manuscript file has been modified.

## Third-party review

- The historical project license identifies Copyright (c) 2021 Visual Computing (ViCo) Group under MIT. Preserve that license and notice with applicable DatasetCondensation-derived files.
- The Practical IS/DELTA implementation records `T-Lab-CUHKSZ/DELTA_FL` at commit `bc8e3a7aed6691ea6a954837656437721edf81e7`.
- DivFL records `melodi-lab/divfl` at commit `4a5b80df85a72b97751ce1130ee5c7d76a2c1540` and contains an explicit port of the author's lazy-greedy routine.
- The old research tree mentions another HCSFed implementation with GPL licensing. That does not establish the license of the final HCS runtime; inspect the actual files included in the release rather than copying that old implementation by default.

The local `networks.py` is byte-identical to the DatasetCondensation copy (SHA256 `fd213761f5edc38e84d6ef653ec60e66fee6f786ddfd406fa6c7cb2734a2dfe5`); `utils.py` is modified from that codebase. The exact pinned-commit DivFL and DELTA_FL MIT license texts were retrieved through the GitHub connection on 2026-11-27 and retained under `licenses/`. Copyright holders are respectively 2022 Melodi Lab @ The University of Washington and 2023 Lin Wang; see `THIRD_PARTY_NOTICES.md` for the commit and blob identifiers. Other source-specific inclusion checks remain pending.

Acquire and retain the relevant exact upstream license texts and source mappings before incorporating those files. A generic acknowledgment is not a replacement for their notices. Dataset asset licensing must be stated separately from original software licensing. No third-party source has yet been copied into this staging folder.

## Current status

This folder now contains a byte-identical snapshot of 55 historical source/configuration files, retained license texts, the ten-method paper specification, a 320-snapshot expected input inventory with 54 support files, and a server export tool with focused fixture tests. These preparation files are being synchronized to the private repository.

The server export has been received and accepted: all 320 snapshots, 54 support files and 448 client metadata files have passed the relevant recorded file checks. Tensor/partition inspection also passed: 110,000 proxy images and ten complete exact partitions. These are independent copies under the ignored `artifacts/` directory, not edits or hardlinks to original experiment inputs.

`run_fl.py` now provides a separate portable input protocol while retaining the historical one-round dispatch, training/checkpoint loop and numerical rules. The matrix scheduler plans 500 runs, with the first 100 seed2025 routes completed before the remaining seeds. The LR=0.01 result-table JSON/CSV and reporting script contain the existing audited 500 seed-level values; they are not new runs.

One additional portability prerequisite was identified: locally regenerated initializers can differ across PyTorch/platform versions. The original seed2025 states are accepted, and `assets/initialization_manifest.json` pins all ten dataset/seed states from formal records. The separate initializer exporter generates and validates the other eight states in the compatible server environment without SGD or training. That supplemental bundle still needs to be returned before a portable five-seed binary asset release can be finalized. Public hosting and a portable from-scratch synthesis schedule remain separate release items. No full synthesis or FL rerun is needed for file recovery.

The connected repository remains private. No formal training has been started and no historical code, protocol, checkpoint, manuscript, or result has been altered.
