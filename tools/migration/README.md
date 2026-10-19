# Datasets refactor checks

Temporary: delete these scripts once the `refactor/datasets` branch is merged.

They check that moving data loading and synthetic data into
`pointcloudsimilarity.datasets` leaves the notebooks' data unchanged. `harness.py`
runs a notebook and records a hash of every array it passes to a metric or to
`tas_spectrum`, together with the result; text notebooks stop at their first
model call and record a hash of the texts they would embed. `compare.py` then
requires identical hashes and call order, and results within `--atol`.

Two runs of `v1.0` agree exactly (`--atol 0`), so any mismatch is a real change.

## Usage

Run both sides on copies: the notebooks save figures next to themselves.

```bash
# Baseline, from the tag
mkdir -p /tmp/pcs-v1.0 && git archive v1.0 | tar -x -C /tmp/pcs-v1.0
tools/migration/run_baseline.sh /tmp/pcs-v1.0 /tmp/pcs-baseline

# Candidate, from the working tree (uncommitted changes included)
mkdir -p /tmp/pcs-candidate && cp -r src notebooks /tmp/pcs-candidate/
tools/migration/run_baseline.sh /tmp/pcs-candidate /tmp/pcs-candidate-out

python tools/migration/compare.py /tmp/pcs-baseline /tmp/pcs-candidate-out
```

Each run takes about 4 minutes. Pass target names after the output directory to
run only some of them (see `targets()` in `harness.py`).

Not covered: `01d` (needs the GloVe models), `experiments_audio/vae.py` (needs
the FMA data) and `experiments/analysis.py` (fails on import at `v1.0`).
