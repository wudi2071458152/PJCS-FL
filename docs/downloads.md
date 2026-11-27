# Data attachments

Release tag: `v1.0.0`. **The files below are prepared locally; their public GitHub hosting has not yet been verified.** The authoritative file sizes, SHA256 checksums, member inventories and planned URLs are in `assets/download_manifest.json`. Its `published` field remains false until the release is actually available.

| Attachment | Size (decimal MB) | Contents |
| --- | ---: | --- |
| `pjcs_cifar10_32k_v1.tar.gz` | 572.57 | Five alpha configurations, 160 proxy snapshots, exact partitions and required metadata, seed2025 model states and asset license |
| `pjcs_fmnist_32k_v1.tar.gz` | 172.34 | Five alpha configurations, 160 proxy snapshots, exact partitions and required metadata, seed2025 model states and asset license |
| `pjcs_initializations_v1.tar.gz` | 9.36 | The remaining four FL seeds for both datasets: eight accepted initial model states, license and member index |

Obtain the dataset of interest and the initialization attachment. For both datasets, obtain all three attachments. Raw CIFAR-10 and Fashion-MNIST data are separate downloads from the original sources. The generated data/initialization contributions use CC BY 4.0 as described in `DATA_LICENSE.txt`, including preserved upstream notices.

Inspect the planned files without writing or downloading anything:

```bash
python -B tools/get_assets.py
```

To install already downloaded attachments, place them in a directory and run:

```bash
python -B tools/get_assets.py --archive-dir /path/to/attachments \
  --artifacts-root artifacts --install
```

For one dataset, add `--dataset cifar10` or `--dataset fmnist`; the shared initialization attachment is included automatically. Installation verifies the compressed archive and every regular member against the release inventory before publishing files. It rejects unsafe paths and different existing files. It does not load tensors or start training.

After public hosting is verified and the manifest is updated, network download and installation can be combined:

```bash
python -B tools/get_assets.py --download --install
```

The source snapshot's `published` field records hosting verification at preparation time. If it is false, the explicit network command displays that status and still attempts the requested release URLs; this allows the tagged source to work once its release attachments have been published. A missing/private attachment remains an HTTP error, and no unverified bytes are installed. Public URLs are planned under `https://github.com/wudi2071458152/PJCS-FL/releases/download/v1.0.0/`; creating a private draft release alone does not make those URLs publicly usable. Keep the attachments outside normal Git source commits.
