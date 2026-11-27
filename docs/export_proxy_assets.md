# Recover missing 32k proxy assets

The public release needs 320 client snapshots. The 96 CIFAR-10 snapshots for alpha=0.1/0.2/0.3 are already available locally and were matched to the formal PJCS input fingerprints. Export only the missing 224 from the server: CIFAR-10 alpha=0.05/0.07 (64) and all five Fashion-MNIST configurations (160).

Transfer `tools/export_proxy_assets.py` and `assets/proxy_manifest.json` together to the server, preserving their relative locations. The script requires Python 3.9 or later, uses only its standard library, and does not load models, modify experiment inputs, download data, synthesize images, or start training.

From the uploaded release-preparation folder, first verify all required files without writing an archive:

```bash
python -B tools/export_proxy_assets.py \
  --project-root /data1/wd/project/Cifar10-experiments \
  --manifest assets/proxy_manifest.json \
  --output pjcs_missing_proxy_assets.tar.gz
```

Only after `preflight_ok` reports 224 snapshots, run the same command with `--export`:

```bash
python -B tools/export_proxy_assets.py \
  --project-root /data1/wd/project/Cifar10-experiments \
  --manifest assets/proxy_manifest.json \
  --output pjcs_missing_proxy_assets.tar.gz --export
```

`--project-root` is an explicit path prefix; if the server project is elsewhere, change that argument. A missing file or unexpected snapshot fingerprint stops the operation. Do not edit expected fingerprints to suppress a mismatch. Existing output, checksum, or partial files are not overwritten.

The archive contains the selected final snapshots, each client's `config.json` and `COMPLETED_32000.json` when present, explicitly listed `support_files` (such as exact partition indices or required initialization files), and `export_manifest.json` with member hashes and sizes. Every declared support file must also match its recorded fingerprint; it is not found through a directory scan. Optimizer states, unlisted checkpoints, raw datasets, and training logs are excluded. The generated archive is read back and checked before it receives its final filename; its sibling `.sha256` file identifies the completed archive.

Verify and transfer both completed files back to the Windows receiving folder:

```bash
sha256sum -c pjcs_missing_proxy_assets.tar.gz.sha256
```

The archive is a recovery input for release preparation. It is not itself a public dataset release or a complete from-scratch reproduction bundle. Metadata files absent from the server are omitted and can be recovered separately if required; snapshot integrity is always checked against the supplied manifest.
