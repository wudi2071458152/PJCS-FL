# Third-party notices

This preparation snapshot includes historical source code under `legacy/`. Its exact inventory and original file fingerprints are recorded in `assets/source_manifest.json`. The paths in the table below are relative to `legacy/`; these notices supplement the original source acknowledgments and do not replace them.

| Local source being reviewed | Upstream | License and adaptation |
| --- | --- | --- |
| `networks.py`; modified `utils.py` | [DatasetCondensation](https://github.com/VICO-UoE/DatasetCondensation) | [MIT](licenses/DatasetCondensation-MIT.txt); Copyright (c) 2021 Visual Computing (ViCo) Group. `networks.py` is byte-identical to the retained historical upstream copy; `utils.py` has project modifications. |
| Practical IS/DELTA scoring and feedback in `synthesis_label2_study/baseline191_algorithms.py` | [DELTA_FL, fixed commit](https://github.com/T-Lab-CUHKSZ/DELTA_FL/tree/bc8e3a7aed6691ea6a954837656437721edf81e7) | [MIT](licenses/DELTA_FL-MIT.txt); Copyright (c) 2023 Lin Wang. The experimental methods use the confirmed HT sampling/aggregation adaptation rather than claiming identity with the author's complete program. |
| DivFL lazy-greedy routine in `synthesis_label2_study/baseline191_algorithms.py` | [DivFL, fixed commit](https://github.com/melodi-lab/divfl/tree/4a5b80df85a72b97751ce1130ee5c7d76a2c1540) | [MIT](licenses/DivFL-MIT.txt); Copyright (c) 2022 Melodi Lab @ The University of Washington. The local function explicitly records that it is a port of the author's `fedbase.py`. |

The DELTA_FL and DivFL license files were retrieved from the exact commits linked above on 2026-11-27. Their Git blob SHAs are respectively `d8fc52e503ad4abc83aa44dbd9939716a5e09647` and `b99d87d00cd1cb7c2fb7484f65684262da35d612`.

The network source also acknowledges [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) and [BIGBALLON/CIFAR-ZOO](https://github.com/BIGBALLON/CIFAR-ZOO). Their MIT notices are retained in [pytorch-cifar-MIT.txt](licenses/pytorch-cifar-MIT.txt) (Copyright (c) 2017 liukuang) and [CIFAR-ZOO-MIT.txt](licenses/CIFAR-ZOO-MIT.txt) (Copyright (c) 2019 Wei Li). The retrieved license blob SHAs are respectively `2e229faa39851c4ddf71b0284c7e56a02dfd577a` and `05a1e3f7bb5b502b5e122c1de5ebef4fee0ead03`; these are retained current-upstream notices, not invented historical commit identifiers.

Power-of-Choice is recorded as a fresh implementation from the paper. The final HCSFed adaptation uses the local Gram/cluster kernels; the historical project also contains other HCSFed experiments and downloaded third-party code. Those historical files are not automatically included in this release and their licenses must not be generalized to all HCSFed implementations.

Benchmark data and synthetic-data assets require separate provenance and usage terms; the top-level software license does not relicense the original datasets.
