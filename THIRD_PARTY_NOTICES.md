# Third-party notices

This preparation snapshot does not yet include third-party source code. The following records apply to the files being reviewed for the executable release; they do not assert that the final release inventory has already been accepted.

| Local source being reviewed | Upstream | License and adaptation |
| --- | --- | --- |
| `networks.py`; modified `utils.py` | [DatasetCondensation](https://github.com/VICO-UoE/DatasetCondensation) | MIT; Copyright (c) 2021 Visual Computing (ViCo) Group. `networks.py` is byte-identical to the retained historical upstream copy; `utils.py` has project modifications. Preserve the upstream license with these files. |
| Practical IS/DELTA scoring and feedback in `baseline191_algorithms.py` | [DELTA_FL, fixed commit](https://github.com/T-Lab-CUHKSZ/DELTA_FL/tree/bc8e3a7aed6691ea6a954837656437721edf81e7) | [MIT](licenses/DELTA_FL-MIT.txt); Copyright (c) 2023 Lin Wang. The experimental methods use the confirmed HT sampling/aggregation adaptation rather than claiming identity with the author's complete program. |
| DivFL lazy-greedy routine in `baseline191_algorithms.py` | [DivFL, fixed commit](https://github.com/melodi-lab/divfl/tree/4a5b80df85a72b97751ce1130ee5c7d76a2c1540) | [MIT](licenses/DivFL-MIT.txt); Copyright (c) 2022 Melodi Lab @ The University of Washington. The local function explicitly records that it is a port of the author's `fedbase.py`. |

The DELTA_FL and DivFL license files were retrieved from the exact commits linked above on 2026-11-27. Their Git blob SHAs are respectively `d8fc52e503ad4abc83aa44dbd9939716a5e09647` and `b99d87d00cd1cb7c2fb7484f65684262da35d612`.

The network source also acknowledges [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) and [BIGBALLON/CIFAR-ZOO](https://github.com/BIGBALLON/CIFAR-ZOO). Their applicable notices must be retained if the corresponding network implementations are distributed. Their inclusion review is still pending.

Power-of-Choice is recorded as a fresh implementation from the paper. The final HCSFed adaptation uses the local Gram/cluster kernels; the historical project also contains other HCSFed experiments and downloaded third-party code. Those historical files are not automatically included in this release and their licenses must not be generalized to all HCSFed implementations.

Benchmark data and synthetic-data assets require separate provenance and usage terms; the top-level software license does not relicense the original datasets.
