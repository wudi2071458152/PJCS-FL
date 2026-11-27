# Recover the remaining initial model states

This author-side recovery step is complete. All eight supplemental states are accepted and published; ordinary users should follow [downloads.md](downloads.md) rather than regenerate them on the original server.

The proxy recovery bundle contains the original seed2025 initial models for all ten dataset/configuration groups. The other four FL seeds originally generated their initial models in the server environment. A prior audit found that local Windows/PyTorch 2.7 generation differs from the recorded Linux/PyTorch 2.14 states, even with the same seed. Do not substitute those different weights for the paper initialization.

`assets/initialization_manifest.json` records the tensor-content fingerprints from the formal five-seed protocols. The initializer exporter uses the unchanged retained `dirichlet_screen_common.model(seed)` and `fmnist_common.model(seed)` functions, and checks the unchanged `run_dirichlet_multiseed.state_sha256` fingerprint before exporting anything. The recorded environment is Python 3.11.16 and PyTorch 2.14.0+cu130. Using the same compatible environment is still subject to the fingerprint checks.

This operation creates **eight initial models**, not eight training experiments: two datasets times seeds 2027/2028/2029/2030. It does not load benchmark or proxy data, run SGD, or start FL training.

Upload the dedicated preparation package containing `legacy/`, `tools/export_initializations.py`, and `assets/initialization_manifest.json`, and verify its checksum. From its extracted root, activate the original server environment and optionally run read-only output preflight:

```bash
python -B tools/export_initializations.py
```

After all eight fingerprints match, export the small state bundle:

```bash
python -B tools/export_initializations.py \
  --output pjcs_initializations.tar.gz --export
sha256sum -c pjcs_initializations.tar.gz.sha256
```

The `--export` command performs the checks itself; a separate preflight invocation is optional. Default behavior generates and checks models in memory without writing outputs. A mismatch stops before an archive is published. Do not edit expected fingerprints to suppress a mismatch.

The archive contains eight state dictionaries at `metadata/{dataset}/initializations/seed_{seed}.pt` and `initialization_export_manifest.json`. The manifest records file hashes, expected tensor-content hashes, tensor shapes/dtypes, and the hash of the expected initialization manifest. Every member is read back before publishing the archive and its sibling `.sha256`. Existing output or partial files are never overwritten.

Transfer both completed files back to the Windows receiving directory. The portable runner can then load these exact new-seed states under its artifact root rather than relying on platform-dependent fresh initialization. Seed2025 continues to use the original per-alpha `metadata/{dataset}/{alpha}/initial_model.pt`.
