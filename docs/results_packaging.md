# Automatic compact result packages

The matrix scheduler packages the verified first-seed results before starting remaining seeds, then packages the complete selected matrix before declaring the schedule complete. Default package names are `first_seed_results.tar.gz` and `final_results.tar.gz`, each with a sibling `.sha256` file under the scheduler output directory. The default counts are 100 and 500 runs; explicit seed/method subsets use their exact planned identities instead.

`tools/package_results.py` requires each expected run to have the matching dataset, alpha, route, seed, `pjcs_portable_lr001` protocol and 1500-round target. It checks every round number, the completion marker and the protocol/round/summary file fingerprints. Missing, damaged, foreign, short diagnostic or incomplete runs stop packaging and prevent scheduler completion.

The whitelist is `protocol.json`, `rounds.json`, `summary.json`, `COMPLETED.json`, and `status.json` when present and complete. The archive also contains `result_index.json` and `packaging_manifest.json` with member fingerprints. Weights, checkpoints, logs, raw datasets and unlisted result files are excluded; their originals remain untouched.

Before publishing the archive, the packer reads all archive members back and verifies the exact whitelist, byte sizes and SHA256 values. Publication never overwrites an existing file. A matching already completed archive and checksum can be validated and reused on a resumed schedule; a different or damaged existing package causes an error rather than replacement. Scheduler status becomes `complete` only after both applicable packages have passed these checks.
