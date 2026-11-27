# Portable federated-learning entry point

`run_fl.py` binds the accepted public asset layout to the retained numerical implementation. It restricts experiments to the ten routes in `configs/paper_lr001.json`. It does not edit the legacy sources, change sampling or aggregation, replace weighted proxy CE, or change the real/proxy SGD learning rate of 0.01.

The public input layout is the asset manifest's `asset_path`, under an explicit artifact root. It contains exact partition indices, real class counts, benchmark-file fingerprints, the recorded seed2025 initial model for each configuration, and final 32k proxy tensors. Raw benchmark data are supplied separately in the same layout expected by their original dataset readers. Files are verified before tensor deserialization. Proxies are used only for proxy-score HT, K=3 and PJCS; the other routes do not need to deserialize proxy tensors.

## Check inputs

For example, from the repository root:

```bash
python -B run_fl.py --dataset cifar10 --alpha 0p05 \
  --route representative_real_eta1p0 --seed 2025 \
  --artifacts-root artifacts --data-root /path/to/benchmark-data \
  --output outputs/cifar10/0p05/seed2025/pjcs
```

The default is preflight. It checks the source snapshot, raw data, exact partition/class counts, required proxies and initialization, then reports `formal_training_started: false`; it does not create an output directory or run a local SGD step.

Append `--run` to explicitly start training. Formal runs require CUDA. The default target is 1500 rounds; a deliberately smaller `--rounds` value is a diagnostic run and must not be reported as a completed paper experiment. Use `--dataset fmnist` for Fashion-MNIST. The alpha and federated-seed choices are the recorded five values in the paper configuration. This worker does not create a new data partition or new synthetic data.

## Initialization portability

Seed2025 always loads the accepted original `metadata/{dataset}/{alpha}/initial_model.pt`. The other seeds first look for an accepted server initialization at `metadata/{dataset}/initializations/seed_{seed}.pt`; otherwise they use the original seeded initializer. In both cases their canonical tensor-state fingerprint must match `assets/initialization_manifest.json`.

An initializer mismatch is a hard error. Different PyTorch/platform versions have been observed to generate different initial parameters for the same seed, so replacing expected hashes with locally generated hashes would change the experiment. Use the recorded environment or the verified server initialization export instead.

## Records and resume

The training loop is the retained `baseline191.train`, with only dispatch, state validation and final output verification rebound to the retained common workers and the public input protocol. It therefore preserves update application, evaluation, algorithm costs, FP32 logical communication, checkpoint integrity, RNG restoration and summaries. The paper accuracy statistic is `mean_last300_test_accuracy` in `summary.json`.

Outputs must be disjoint from benchmark data, artifact and frozen-source directories. A foreign nonempty directory without a bound protocol is rejected. Repeating exactly the same command resumes a matching checkpoint; a changed configuration, code or bound input is rejected rather than silently continuing. A `COMPLETED.json` marker is created only after the recorded target, checksums and numerical-record checks pass.

Portable outputs use a separate `pjcs_portable_lr001` protocol. They cannot be spliced into historical output directories or used to resume old experimental checkpoints. Packaging and fixture checks do not constitute a new five-seed accuracy experiment. Formal training is started by the user.
