# Reproduction workflow

The main configuration is `configs/paper_lr001.json`: two datasets, five alpha values, five FL seeds and ten methods, for 500 runs. These are five training seeds with fixed client partitions and fixed proxies, not five independent partition draws.

## Prepare code and data

Install a compatible PyTorch/torchvision environment and the direct dependencies in `requirements.txt`. The observed paper environment is recorded separately in `requirements-historical.txt`; it is not a complete lockfile. Retained numerical source bytes are verified with:

```bash
python -B tools/verify_release.py
```

Place the accepted dataset artifacts under `artifacts/`, with `proxies/` and `metadata/` directly below that directory. Raw benchmarks are separate. CIFAR-10's raw directory must contain `cifar-10-batches-py/`; Fashion-MNIST's must contain the four original gzip files below `FashionMNIST/raw/`. The source manifests contain the exact benchmark-file hashes, and the portable runner verifies them before reading data. The benchmark source locations are <https://www.cs.toronto.edu/~kriz/cifar.html> and <https://github.com/zalandoresearch/fashion-mnist>.

The three accepted binary attachments and their installation procedure are listed in `docs/downloads.md`. The source release is public; the two dataset attachments are still awaiting upload, while the initialization attachment and checksum file are available. The code's `published` flag records whether the complete attachment set has been verified in that source snapshot. Downloads require an explicit option and remain subject to the pinned archive/member hashes. `docs/data_acceptance.md` and `assets/accepted_proxy_report.json` describe the accepted snapshot set. The supplemental eight initialization states have also been accepted; see `docs/initialization_acceptance.md`.

For seeds 2027/2028/2029/2030, use the separately verified initialization assets described in `docs/export_initializations.md`, or a compatible environment whose regenerated state passes the formal tensor fingerprint. Seed2025 loads the original accepted per-alpha initial model. Do not replace expected hashes to accommodate a different initializer.

## Preflight and one run

Follow `docs/portable_fl.md` for the complete single-run interface. A seed2025 example is:

```bash
python -B run_fl.py --dataset cifar10 --alpha 0p05 \
  --route representative_real_eta1p0 --seed 2025 \
  --artifacts-root artifacts --data-root /path/to/cifar-data \
  --output outputs/cifar10/0p05/2025/pjcs
```

This checks inputs without training or creating the output directory. Add `--run` only when training is intended. Formal training requires CUDA; it is not started by installation or preflight.

## Full matrix

Inspect the schedule without starting workers:

```bash
python -B tools/schedule_fl.py
```

The default plan is 500 jobs. All 100 seed2025 jobs finish before the other 400 are launched. To explicitly run the matrix:

```bash
python -B tools/schedule_fl.py --run --workers 4 --gpus 0,1 \
  --artifacts-root artifacts \
  --cifar-data-root /path/to/cifar-data \
  --fmnist-data-root /path/to/fmnist-data \
  --output-root outputs/paper_lr001
```

Concurrency is configurable. The four-worker default assigns two workers to each of two GPUs, sets each numerical-library environment to two threads, and retains the existing single-thread LP setting. It is a validated starting convention, not a guarantee of speed on other hardware. A worker failure stops the scheduler's own remaining workers and records a failed status. Already accepted matching outputs are verified/resumed by the worker; changed inputs or protocols are rejected. No other users' processes are managed.

The scheduler records status, logs, and phase-completion lists below its output directory. After validating the exact route identities, 1500-round coverage and result fingerprints, it produces `first_seed_results.tar.gz` and `final_results.tar.gz`, each with a SHA256 file. These archives contain compact JSON records and manifests, excluding model checkpoints, tensors and logs. Packaging failure prevents the next phase or final completion; see `docs/results_packaging.md`.

## Recorded paper results

`results/lr001_tail300.json` and `.csv` contain the 500 already audited seed-level values. Each value is the mean test ACC over rounds 1201–1500. `results/table.md` reports their mean and sample standard deviation across five seeds. Regenerate the table without training:

```bash
python -B tools/report_results.py
```

The highest unrounded method mean in each dataset/alpha column is bold with a star. A star does not denote statistical significance. These files reproduce the recorded result table; they do not claim that new executions have already achieved those accuracy values.

## Synthesis scope

The original reference, calibration, old-loss synthesis and 32k continuation implementations are preserved in `legacy/`, with their entry points described in `docs/frozen_source.md`. They use shared FedAvg reference trajectories and each client's own real data for synthetic optimization. The released quick-FL asset layout is not a full historical synthesis-resume tree; reference checkpoints, calibration records and optimizer states are separate prerequisites for resuming those historical synthesis jobs. A portable from-scratch synthesis scheduler remains a separate packaging item; do not present the quick-FL commands as from-scratch synthesis commands.
