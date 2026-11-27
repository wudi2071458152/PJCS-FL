# Package the accepted public dataset assets

`tools/package_data.py` packages only the expected proxy and support files in `assets/proxy_manifest.json`, using the accepted `asset_path` layout. It does not scan the historical project, include raw benchmark images, client logs, optimizer states or unrelated checkpoints, deserialize tensors, or train a model.

The publisher must provide the confirmed data-license text at `DATA_LICENSE.txt`. It is preserved under `metadata/{dataset}/DATA_LICENSE.txt`. Each archive also contains `metadata/{dataset}/release_manifest.json`, so installing both datasets does not create an index-name collision.

Inspect the selected input files without writing an archive:

```bash
python -B tools/package_data.py --output-dir /path/to/release-assets
```

Explicitly package the accepted CIFAR-10 and Fashion-MNIST asset sets:

```bash
python -B tools/package_data.py --output-dir /path/to/release-assets --package
```

The tool verifies each source stream against its expected SHA256 while writing the new archive, then reads back all members and verifies names, sizes and hashes. Publication uses same-directory atomic no-overwrite links. A successful asset has a sibling `.sha256` file. Existing final or partial files and an existing download manifest are never overwritten.

`assets/download_manifest.json` records the compressed archive hashes, sizes and complete member checksums. The two dataset attachments and the separately accepted initialization supplement are now public in `v1.0.0`; see [downloads.md](downloads.md). The default branch records `published: true` after matching all uploaded attachment sizes and GitHub-reported SHA256 values to the prepared files. Historical tagged preparation metadata may still record the earlier status.
