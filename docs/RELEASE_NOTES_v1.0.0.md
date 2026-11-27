# PJCS-FL v1.0.0 — pre-generated-proxy FL reproduction

Code and recorded LR=0.01 results for **Joint Client Sampling with Reusable Synthetic Data for Non-IID Federated Learning**, by Di Wu.

This release contains the portable FL input binding, unchanged historical numerical kernels, ten client-selection methods, two datasets × five alpha configurations × five FL seeds, exact input inventories, configurable seed2025-first scheduling, verified compact-result packaging and recorded tail-300 result tables.

Data attachments contain all 320 final 32k proxy snapshots (110,000 synthetic images), exact partitions and metadata, plus the original initial states required for the five FL seeds. The proxy and initialization files were checked against their formal recorded fingerprints. They do not contain raw benchmark datasets, training checkpoints, synthesis optimizer states or training logs.

Software uses MIT with preserved third-party licenses. Author-created asset contributions use CC BY 4.0 with upstream dataset attribution; see `DATA_LICENSE.txt`.

Download the two dataset attachments and the initialization attachment, verify `SHA256SUMS`, then follow `docs/downloads.md` and `docs/reproduce.md`. The default worker and scheduler modes do not train; training requires `--run`.

Validation covers source equality, exact data and initialization fingerprints, tensor/partition content, ten-method fixture parity, checkpoint-resume fixtures, and one real-input preflight per dataset. No new formal training or claim of newly reproduced accuracy was made during release preparation. The historical synthesis source is retained; a complete portable from-scratch synthesis/resume schedule is not included in this quick-FL workflow.

The reference trajectories used for synthetic optimization are two shared FedAvg trajectories per data partition, with client-specific real data in each synthetic loss. Documentation follows the actual recorded protocols. The manuscript itself was not edited by this release preparation.
