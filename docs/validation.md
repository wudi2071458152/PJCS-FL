# Preparation validation

Date: 2026-11-27. These checks validate release preparation and input binding; they are not a new five-seed training result.

- All 55 retained source/configuration files match the original experiment files byte-for-byte. The Python snapshot compiled, the JSON files parsed, and four FL entry points returned CLI help successfully.
- The received recovery archive matched SHA256 `9326534bd62d796bb8210a984f51d12755a11976c5fcca8016c9bf0d924ac4c5`. All 320 proxy snapshots and 54 support files matched their formal expected hashes after acceptance. The other 448 client metadata files matched the export index. No hardlinks or changes to original experiment inputs were used.
- Content inspection passed for 320 tensors and ten exact partitions: 110,000 synthetic images, expected image shapes/dtypes/finite values, exact label histograms and budgets, and complete unique partition coverage. See `assets/tensor_acceptance_report.json`.
- Six portable-worker fixture tests passed, covering ten-method delta/selection/probability parity with direct retained kernels, fingerprint rejection before deserialization, output isolation, initialization checks, and exact toy checkpoint/resume state parity. The training loop itself is retained rather than rewritten.
- Real-input read-only preflight passed once per dataset for alpha=0.05, seed2025, PJCS. Both loaded the original initialization and exact data/partition/proxy inputs. No benchmark SGD ran and no formal output directory was created. See `assets/portable_preflight_report.json`.
- Export and acceptance fixtures cover corrupt/missing inputs, safe paths, no overwrite, partial-file ownership, Linux publication races, checksums, and archive members. The initializer exporter was tested with small fake model factories; the eight actual server states have not yet been generated/returned in this preparation.
- Matrix tests cover all 500 identities, seed2025-first ordering, worker CLI/environment dispatch, failure status, and refusal of unsupported methods. They mock processes and do not start experiment workers. Result-table tests independently recompute the recorded means and sample SDs from the five-seed values and check highest-value annotations.

Passed checks are reused unless a related input or implementation changes. No full experimental regression or formal training rerun was performed. New-seed initialization portability still depends on the accepted server initialization supplement or a generator that matches the pinned tensor fingerprints.
