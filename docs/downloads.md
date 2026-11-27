# Data attachments

Release tag: [`v1.0.0`](https://github.com/wudi2071458152/PJCS-FL/releases/tag/v1.0.0). The repository and complete attachment set are public. On 2026-11-27, the three archives and `SHA256SUMS` were checked against GitHub's uploaded-asset metadata: names, sizes, download URLs and server-reported SHA256 values match the prepared files. `assets/download_manifest.json` records the file checksums and complete member inventories with `published=true`.

| Attachment | Size (decimal MB) | Contents |
| --- | ---: | --- |
| [pjcs_cifar10_32k_v1.tar.gz](https://github.com/wudi2071458152/PJCS-FL/releases/download/v1.0.0/pjcs_cifar10_32k_v1.tar.gz) | 572.57 | Five alpha configurations, 160 proxy snapshots, exact partitions and required metadata, seed2025 model states and asset license |
| [pjcs_fmnist_32k_v1.tar.gz](https://github.com/wudi2071458152/PJCS-FL/releases/download/v1.0.0/pjcs_fmnist_32k_v1.tar.gz) | 172.34 | Five alpha configurations, 160 proxy snapshots, exact partitions and required metadata, seed2025 model states and asset license |
| [pjcs_initializations_v1.tar.gz](https://github.com/wudi2071458152/PJCS-FL/releases/download/v1.0.0/pjcs_initializations_v1.tar.gz) | 9.36 | The remaining four FL seeds for both datasets: eight accepted initial model states, license and member index |

Obtain the dataset of interest and the initialization attachment. For both datasets, obtain all three attachments. Raw CIFAR-10 and Fashion-MNIST data are separate downloads from the original sources. The generated data/initialization contributions use CC BY 4.0 as described in `DATA_LICENSE.txt`, including preserved upstream notices.

The release also provides [SHA256SUMS](https://github.com/wudi2071458152/PJCS-FL/releases/download/v1.0.0/SHA256SUMS). Inspect the expected files without writing or downloading anything:

```bash
python -B tools/get_assets.py
```

To install already downloaded attachments, place them in a directory and run:

```bash
python -B tools/get_assets.py --archive-dir /path/to/attachments \
  --artifacts-root artifacts --install
```

For one dataset, add `--dataset cifar10` or `--dataset fmnist`; the shared initialization attachment is included automatically. Installation verifies the compressed archive and every regular member against the release inventory before publishing files. It rejects unsafe paths and different existing files. It does not load tensors or start training.

Network download and installation can be combined:

```bash
python -B tools/get_assets.py --download --install
```

The source snapshot's `published` field records hosting verification at preparation time. The original `v1.0.0` tag predates the completed uploads and records false; its explicit download command still works with the published URLs, verifies the same pinned archive/member hashes, and may print the older verification status. No tag movement is needed. The default branch records the completed verification. A failed download never installs unverified bytes. Keep the attachments outside normal Git source commits.
