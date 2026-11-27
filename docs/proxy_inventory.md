# Proxy asset inventory

This inventory describes the final **32,000-iteration**, old-loss proxy inputs for the primary LR=0.01 experiment. It contains expected SHA256 checksums from the recorded **seed2025 PJCS protocols**, not newly generated proxy data. The five federated seeds reuse these same partitions and proxy images. All recovery and acceptance steps below are complete; ordinary users should obtain the published attachments through [downloads.md](downloads.md). The table records the author's original recovery plan.

| Dataset | Dirichlet alpha | Final snapshots | Already available locally | To export |
| --- | --- | ---: | ---: | ---: |
| CIFAR-10 | 0.05, 0.07 | 64 | 0 | 64 |
| CIFAR-10 | 0.1, 0.2, 0.3 | 96 | 96 | 0 |
| Fashion-MNIST | 0.05, 0.07, 0.1, 0.2, 0.3 | 160 | 0 | 160 |
| Total | | 320 | 96 | 224 |

The 96 local CIFAR-10 snapshots were matched to the formal protocol hashes in the preceding audit; this preparation does not repeat that check. The other 224 snapshots are absent from the compact result archives, by design. Export the original files from the server; no synthesis or training rerun is needed.

[proxy_manifest.json](../assets/proxy_manifest.json) records each dataset, alpha, client, expected file hash, project-relative server path, release asset path, and protocol evidence. `required_export=false` identifies the 96 snapshots that can be reused locally. The manifest's `support_files` additionally lists exact partition indices, class-count and input manifests, source protocols, and the seed2025 initial model files needed for reproducible input binding. They remain subject to the same file-hash verification during export. Repeated small metadata files are deliberately named by configuration for a self-contained release layout.

Use the actual server project root as an argument. Recorded `/home/wd` paths have been represented as paths relative to the project root; the current root is expected to be `/data1/wd/project/Cifar10-experiments`. This mapping is explicit and does not change any expected checksum. A missing path or mismatch must stop the export rather than substitute a newer file.

The synthesis lineage uses **two shared FedAvg reference trajectories per data partition**, with seeds 2025 and 2026. Each client uses its own real data when optimizing its own synthetic images at sampled reference states. The reference pool is not a separately trained local trajectory for every client.

No raw CIFAR-10 or Fashion-MNIST images, synthesis optimizer states, or unrelated checkpoints are included in this requested export. The complete historical synthesis/continuation pipeline can require further reference checkpoints and coefficient files beyond this quick-FL asset inventory; the inventory does not claim to contain a full synthesis-resume bundle.
