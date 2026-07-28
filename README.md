# TICR AAAI 2027: code organized by current RQ numbering

This archive reorganizes the latest available TICR experiment code by the six
research questions in the AAAI 2027 manuscript. It does not rename or rewrite
the legacy experiment scripts, so their original parameters and provenance are
preserved.

## Directory map

- `RQ1_effectiveness/`: Bill-Contra and Juris-Logic main retrieval and
  dual-path ranking pipelines.
- `RQ2_mechanism/`: legacy ablations plus the newer query-aligned Stage-1,
  inference-matched, reranker, inversion-validity, and paraphrase-validity
  controls.
- `RQ3_hyperparameters/`: inversion-count and candidate-depth sweeps. Files
  under `candidate_depth_legacy_RQ8/` came from the old project numbering.
- `RQ4_backbone/`: the common evaluation pipeline used after regenerating
  atoms and inversions with each backbone. The current project snapshot does
  not contain a standalone atom/inversion generation launcher for the three
  reported backbones; this limitation is recorded rather than hidden.
- `RQ5_efficiency/`: documentation for the reported timing protocol. No
  standalone latency benchmark script was present in the latest server
  snapshot, so the archive does not claim one was run from a missing script.
- `RQ6_transferability/`: public-benchmark and baseline code. This was stored
  under legacy directory `RQ5/` before the paper RQs were restructured.
- `shared/data/`: datasets present in the latest TICR server snapshot.
- `shared/results/`: saved JSON outputs for the newer controlled experiments.

## Authoritative clean-rerun outputs

Use these files for the revised manuscript:

- `bge_full_clean.json`, `bm25_full_clean.json`,
  `splade_full_clean.json`, and
  `dualpath_factorial_full_clean.json` for the full clean-valid main
  populations.
- `stage1_controls_canonical.json`,
  `stage1_lambda_diagnostics_canonical.json`,
  `compute_matched_controls_canonical.json`,
  `reranker_controls_canonical.json`, and
  `paraphrase_validity_canonical.json` for the canonical paired populations.

Older aligned, all-depth, and intermediate clean outputs are excluded to keep
the release focused on the authoritative results.

## Environment

The scripts principally require Python 3, PyTorch, NumPy, pandas,
sentence-transformers, transformers, tqdm, scikit-learn, and rank-bm25.
SparseCL additionally expects its upstream `models` module. Model paths are
read from Hugging Face identifiers in the original scripts; large model files
are intentionally excluded.

## Important naming note for Table 4

The Stage-1 script defines:

- `shuffled_inversion`: replace each query's inversions with inversions from
  the next query, cycle them to exactly the TICR probe count, and otherwise
  retain the same `q + inverse_i`, max-pooling, and global Top-K procedure.
- `query_plus_inverse`: for every inverse, separately concatenate the original
  query and that inverse into one embedding input, then max-pool document
  similarities across the resulting probes. This is the original TICR
  candidate-retrieval implementation reported as `TICR Stage-1` in Table 4.
- `inverse_only`: encode every atomic inversion without the original query.
  This diagnostic is intentionally retained because it is stronger than
  `query_plus_inverse` on both canonical paired populations.
- `run_stage1_lambda_diagnostics.py`: compare inverse-only, both text orders,
  and normalized vector composition over a lambda grid while reporting
  contradiction rank, paired hard-negative margin/win rate, and
  original-query Top-20 overlap.

The Stage-1 result excludes the subsequent dual-path NLI verification.
Shuffling inversions across queries preserves the probe count and retrieval
procedure but degrades both datasets, testing whether TICR benefits from
query-specific logical inversions rather than additional views alone.

## Provenance

Legacy source code and datasets were synchronized from
`/data/users/anonymous_user/TICR` on server-4 on 2026-07-24. New controlled scripts
were prepared in the AAAI revision workspace and the clean reruns were
executed on server-8 (NVIDIA RTX 6000D) on 2026-07-26.
