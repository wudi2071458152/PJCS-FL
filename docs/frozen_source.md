# Frozen numerical source snapshot

`legacy/` contains a selected, byte-identical copy of the historical experiment implementation. The SHA256, original project-relative path, byte count and inclusion role of every copied file are recorded in `assets/source_manifest.json`. The original research project was not edited.

This snapshot contains **51 Python files and four frozen JSON configurations** (624,520 bytes total). The dependency closure of the four FL entry points and four synthesis/reference entry points contains 47 Python files. Four additional Python files are retained because protocol functions fingerprint or refer to them even though they are not ordinary runtime imports. These are the two Fashion-MNIST extension adapters and the historical cluster-history and old-server schedulers. The four JSON files are the fixed CIFAR continuation source protocol and the original three Fashion-MNIST suite configurations.

This is an implementation snapshot, not a claim that copying these files alone reproduces the paper. The public `run_fl.py` binds the accepted exact partitions, proxy tensors and initial states to the retained FL kernels; see [the portable entry guide](portable_fl.md). Historical synthesis/resume prerequisites remain separate. Paths inside historical defaults are preserved as provenance and are not required locations on a reader's machine.

## Why historical modules remain

The import chain for the Fashion-MNIST 32k worker passes through the common baseline worker, the CIFAR alpha=0.05/0.07 adapters, multiseed sampling, control variates, HCSFed and synthesis utilities. Consequently, this folder also includes passively imported SCAFFOLD, history/residual and alternative-loss modules. Their presence does not add those methods to the first public experiment matrix. The public entry point restricts the available methods to the agreed ten client-selection routes.

The legacy implementation uses `FunctionType` to rebind functions and contexts. In particular, the 32k Fashion-MNIST adapter replaces specific constants in the frozen 8k function code objects. Reformatting, extracting or rewriting those functions without an equivalence check may invalidate this mechanism or alter a recorded fingerprint. The release therefore retains their original bytes.

The 47-file import closure also imports the scheduler module for its lock and integrity helpers. No copy of the separately downloaded third-party source-review trees under `archive/` is included. Their paths in the baseline provenance records are citations to the review evidence, not runtime imports.

## Included entry points

The paths below identify the retained implementations. For the executable quick-FL workflow, use the public `run_fl.py` and `tools/schedule_fl.py` described in [reproduce.md](reproduce.md); the historical entry points retain their original input protocols.

| Purpose | Retained implementation under `legacy/synthesis_label2_study/` |
| --- | --- |
| CIFAR PoC/IS-HT/DELTA-HT/DivFL/HCSFed | `run_dirichlet_baseline191.py` |
| CIFAR proxy selection and PJCS | `run_dirichlet_multiseed.py`, with the alpha=0.05/0.07 adapters |
| Fashion-MNIST nonproxy methods and original 8k methods | `run_fmnist_fl.py` |
| Fashion-MNIST 32k proxy methods | `run_fmnist_continue32000_fl.py` |
| Shared FedAvg reference trajectories | `run_dirichlet_reference.py` |
| CIFAR calibration and synthesis primitives | `run_dirichlet_screen.py` and `dirichlet_dsa4000_worker.py` |
| Fashion-MNIST reference/calibration/8k synthesis | `run_fmnist_synthesis.py` |
| Fashion-MNIST continuation with retained image-SGD momentum | `run_fmnist_continue32000_synthesis.py` |

For example, run `python legacy/synthesis_label2_study/run_dirichlet_baseline191.py --help` from the repository root to inspect its arguments. Do not invoke the historical reference generator with a normal argument set merely to inspect files: its retained normal invocation starts reference training. The public FL entry point performs preparation by default and requires `--run` for training.

## Dependencies and validation scope

The direct third-party imports in this snapshot are NumPy, SciPy, PyTorch, torchvision and scikit-learn. `requirements.txt` lists them without inventing missing version records. `requirements-historical.txt` records the available reference-environment information: Python 3.11.16, torch 2.14.0+cu130 and SciPy 1.17.1. It is not a full environment lockfile, and the appropriate torch/torchvision installation source depends on the target CUDA environment.

Every copied file has been checked against its original bytes. All Python files were compiled in memory and the four JSON files were parsed. Each of the four retained FL entry points also completed `--help` with exit code 0 in the local Python environment, exercising the transitive imports and function adapters without loading experiment data. Portable-entry fixture tests and real-input preflights are recorded in [validation.md](validation.md) and `assets/portable_preflight_report.json`; no new formal training or accuracy reproduction is claimed.

The DatasetCondensation MIT notice is retained in `licenses/DatasetCondensation-MIT.txt` (Copyright (c) 2021 Visual Computing (ViCo) Group). `networks.py` is the unchanged retained upstream copy; `utils.py` contains historical modifications. Additional baseline and network notices remain governed by `THIRD_PARTY_NOTICES.md`. The top-level Di Wu license does not replace upstream notices.
