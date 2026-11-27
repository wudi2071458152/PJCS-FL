# Accepted proxy assets

The received archive `pjcs_missing_proxy_assets.tar.gz` was accepted on 2026-11-27. Its SHA256 is `9326534bd62d796bb8210a984f51d12755a11976c5fcca8016c9bf0d924ac4c5`, matching the transferred checksum file.

The acceptance tool verified the archive's explicit file index, regular-file-only safe paths, member sizes and hashes, and the expected checksums in `assets/proxy_manifest.json`. Its 224 snapshot identities match the formal experiment inventory. All 54 declared support files are present and match the recorded input hashes. The remaining 448 files are explicitly indexed client configuration and completion JSON metadata.

The 96 already audited local CIFAR-10 snapshots were independently copied into the same asset layout. No hardlink to an original experiment file was created. Each new copy was checked against the formal input checksum. Different existing destination files are refused rather than replaced.

The accepted layout under the ignored `artifacts/` directory contains:

| Contents | Count |
| --- | ---: |
| Received final 32k proxy snapshots | 224 |
| Reused CIFAR-10 final 32k proxy snapshots | 96 |
| Exact partitions, count/input manifests, initial models and source protocols | 54 |
| Received client configuration/completion JSON files | 448 |
| Total accepted files | 822 |

The combined accepted files occupy 819,481,019 uncompressed bytes. The detailed report is [assets/accepted_proxy_report.json](../assets/accepted_proxy_report.json). The report lists relative asset paths rather than machine-specific locations.

Snapshots are stored as `artifacts/proxies/{dataset}/{alpha}/client_XX/iter_32000.pt`. Support files are under `artifacts/metadata/{dataset}/{alpha}/`, including `counts.json`, `manifest.json`, `initial_model.pt`, `splits/alpha{alpha}.pt`, and `source/protocol.json`. Some CIFAR configurations also retain `proxy/protocol.json`.

The first acceptance verifies file identity and completeness. A subsequent distinct content check loaded the already accepted proxy tensors and the ten exact partition files, without repeating file hashes. It verified all 320 snapshots (110,000 synthetic images): float32 images, finite values, CIFAR-10 shape `3x32x32`, Fashion-MNIST shape `1x28x28`, int64 labels in `[0,9]`, iteration 32,000, and each client's exact synthetic class histogram and sample count against the recorded source budget. All 320 client configurations matched the metadata and label histograms; the 96 reused configurations were read from their original local metadata without modifying them.

Each CIFAR-10 configuration has exactly 10,000 synthetic images; each Fashion-MNIST configuration has exactly 12,000. All ten partitions have 32 nonempty clients, complete unique coverage of the 50,000 or 60,000 training indices, and per-client lengths equal to recorded real class-count totals. The detailed content report is [assets/tensor_acceptance_report.json](../assets/tensor_acceptance_report.json).

These checks did not load raw benchmark images or initial-model weights, run model forward passes, optimize images, start formal training or reproduce an accuracy value. The portable FL input binding is checked separately. Original experiment code, data, checkpoints and results were not modified. The binary assets remain outside normal Git source commits.

## Repeating acceptance on another machine

The standard-library tool accepts explicit source and destination locations:

```bash
python -B tools/accept_proxy_assets.py \
  --archive /path/to/pjcs_missing_proxy_assets.tar.gz \
  --checksum /path/to/pjcs_missing_proxy_assets.tar.gz.sha256 \
  --manifest assets/proxy_manifest.json \
  --artifacts-root artifacts \
  --local-project-root /path/to/historical-project \
  --report assets/accepted_proxy_report.json
```

The historical project argument supplies only the 96 entries with a recorded `local_relative_path`. This is a recovery workflow for the author's preparation stage, not the final public data-download workflow.
