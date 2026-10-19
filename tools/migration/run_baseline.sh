#!/usr/bin/env bash
# Run harness.py targets against a disposable copy of the repo.
#
#   run_baseline.sh ROOT OUT_DIR [TARGET...]
#
# ROOT/src is put first on PYTHONPATH so the copy's library is the one tested,
# not the editable install. Each target runs in its own process. Set PYTHON to
# use an interpreter other than the repo's .venv.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$1" && pwd)
OUT=$2
shift 2
PY=${PYTHON:-$HERE/../../.venv/bin/python}
TARGETS=${*:-ijcnn_00 ijcnn_01 01_whitenoise 01a_sweeps experiments_data ijcnn_02_texts 01c_texts 01e_texts 01f_texts 04_texts}

# The ijcnn scripts save figures here and fail if it is missing.
mkdir -p "$OUT" "$ROOT/notebooks/ijcnn/figures"
for t in $TARGETS; do
    start=$(date +%s)
    PYTHONPATH="$ROOT/src" USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3 MPLBACKEND=Agg PYTHONHASHSEED=0 \
        "$PY" "$HERE/harness.py" "$t" "$OUT" --root "$ROOT" \
        >"$OUT/$t.stdout.txt" 2>"$OUT/$t.stderr.txt"
    echo "$t exit=$? $(($(date +%s) - start))s"
done
