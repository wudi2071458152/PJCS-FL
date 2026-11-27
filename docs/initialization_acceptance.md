# Initial model supplement acceptance

Accepted on 2026-11-27. The received `pjcs_initializations.tar.gz` matched SHA256 `b766b74dea6f103f274659d37c8029c4b8db5bee5d003cd1065b3f7d10944369` and its sibling checksum. It contained exactly eight regular model-state members and `initialization_export_manifest.json`.

The export index binds `assets/initialization_manifest.json` by SHA256 `571d343ce00d7901649b63996011c03f5cfb6f8e31c5bea201b87f4b518b341a`. All eight file hashes, sizes, dataset/seed identities, tensor shapes/dtypes, finite values and canonical tensor-content fingerprints matched. Tensor deserialization occurred only after archive and file checks, using `torch.load(..., weights_only=True, map_location='cpu')`.

The accepted states cover CIFAR-10 and Fashion-MNIST, each with FL seeds 2027, 2028, 2029 and 2030. They were independently copied to:

```text
artifacts/metadata/{dataset}/initializations/seed_{seed}.pt
```

The public `run_fl.initial_model` helper then loaded all eight accepted states and matched their formal canonical fingerprints in the local Windows/PyTorch 2.7 environment. This confirms that it uses the recorded server weights rather than silently replacing them with locally regenerated states. Seed2025 continues to use the original per-alpha states accepted in the proxy/support bundle.

`assets/accepted_initializations_report.json` records the member hashes, tensor metadata and eight public-loader checks. No original experiment file was changed, no proxy hashes were repeated, and no benchmark/proxy data, model forward pass, SGD or formal training was run for this supplement acceptance.

## Accept a transferred supplement

From the repository root, run:

```bash
python -B tools/accept_initializations.py \
  --archive /path/to/pjcs_initializations.tar.gz \
  --checksum /path/to/pjcs_initializations.tar.gz.sha256
```

The default artifact root is `artifacts/`; `--artifacts-root` selects another explicit root. The tool refuses unexpected, duplicate or nonregular archive members, different existing target files, incorrect checksums, changed formal manifests and canonical state mismatches. An already accepted byte-identical target is reused. This tool performs file acceptance only; it does not train or regenerate models.

Five focused fixture tests passed, covering valid/idempotent acceptance, archive and member hash rejection before deserialization, missing/unsafe members, canonical-state mismatch and preservation of a foreign existing target. Fixture states are small generated tensors and are unrelated to the benchmark experiments.
