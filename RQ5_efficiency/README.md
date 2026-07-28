# RQ5: Efficiency and cost boundaries

The manuscript reports measured end-to-end latency for the bi-encoder,
exhaustive NLI, and TICR pipelines. The latest TICR server snapshot contains
the implementations timed by this comparison under `../RQ1_effectiveness/`,
but it does not contain a standalone timing harness or raw timing log.

Accordingly, this directory records the dependency without manufacturing a
replacement experiment:

- bi-encoder and TICR pipeline:
  `../RQ1_effectiveness/legacy_main_code/`
- exhaustive NLI implementation:
  `../RQ6_transferability/legacy_RQ5/nli.py`

For strict artifact reproducibility, a future rerun should add a single timing
harness with warm-up, synchronization, fixed batch sizes, hardware metadata,
and per-query raw measurements.
