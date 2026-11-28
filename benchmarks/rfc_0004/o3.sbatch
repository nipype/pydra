#!/bin/bash
# RFC 0004 I6b merge gate (task O3): real Apptainer and Lmod on a CPU node.
#
#   sbatch --partition=<cpu partition> --export=ALL,SHA=<sha>,WORK=<scratch dir> \
#       benchmarks/rfc_0004/o3.sbatch
#
# SHA: the commit to test, cloned fresh from GitHub. WORK: a scratch directory (not
# $HOME). SITE_MODULES (optional): site module lists to time, '|'-separated.
# Fails when a selected test is skipped or the selected count is not the expected one.
#SBATCH --job-name=pydra-o3-i6b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail
: "${SHA:?set SHA}" "${WORK:?set WORK to a scratch directory}"

run="$WORK/o3-$SLURM_JOB_ID"
mkdir -p "$run"
cd "$run"
git clone -q https://github.com/nipype/pydra pydra
cd pydra
git checkout -q "$SHA"
uv venv -q .venv
uv pip install -q -p .venv -e ".[test]"

source "$MODULESHOME/init/bash"
module load apptainer
command -v singularity >/dev/null || { echo "FAIL: no singularity on PATH"; exit 1; }
export APPTAINER_CACHEDIR="$run/apptainer-cache" SINGULARITY_CACHEDIR="$run/apptainer-cache"
export PYDRA_HASH_CACHE="$run/hashes" PYDRA_IMAGE_CACHE="$run/images"

gate() {  # gate <log> <expected passed>
    grep -Eq "^=+ $2 passed" "$1" || { echo "FAIL: expected $2 passed"; tail -5 "$1"; exit 1; }
    if grep -Eq "[0-9]+ skipped|^SKIPPED" "$1"; then
        echo "FAIL: skips"; grep -E "^SKIPPED" "$1"; exit 1
    fi
}

# (b) the three new real-runtime tests
uv run --no-sync pytest pydra/environments/tests/test_env_resolution_hpc.py \
    -m "lmod or singularity" --timeout=300 -v -s -rs 2>&1 | tee new.log
gate new.log 3

# test_singularity_1, test_singularity_1_subm (removed by the test_singularity_1 prefix)
# and test_singularity_st_1 need --fakeroot; a node whose user has
# no subuid mapping cannot give it, so probe a SIF the tests above cached, else
# docker://alpine, which the apptainer cache holds from the tests above
sif=$( (find "$PYDRA_IMAGE_CACHE" -name '*.sif' -type f 2>/dev/null || true) | head -1)
probe=${sif:-docker://alpine}
deselect=()
expected=43
if singularity exec --fakeroot "$probe" true >/dev/null 2>&1; then
    echo "O3 mode: fakeroot, probed $probe (all $expected regression cases)"
else
    deselect=(
        --deselect pydra/environments/tests/test_environments.py::test_singularity_1
        --deselect pydra/environments/tests/test_singularity.py::test_singularity_st_1
    )
    expected=38
    echo "O3 mode: no fakeroot, probed $probe; the two fakeroot tests deselected," \
        "$expected regression cases"
fi

# regression: every real-runtime case in the touched modules
uv run --no-sync pytest pydra/environments/tests/test_env_resolution_hpc.py \
    pydra/environments/tests/test_singularity.py pydra/environments/tests/test_lmod.py \
    pydra/environments/tests/test_environments.py ${deselect[@]+"${deselect[@]}"} \
    -m "lmod or singularity" --only-worker=debug --timeout=300 -v -rs 2>&1 | tee regression.log
gate regression.log "$expected"

# (a) timing and (c) module -t list comparison, hierarchical tree with a sticky module
uv run --no-sync python benchmarks/rfc_0004/lmod_resolution.py --runs 20 \
    ${SITE_MODULES:+--site "$SITE_MODULES"} 2>&1 | tee lmod.log
grep -q "^PASS: 0 check" lmod.log
echo "O3 PASS at $SHA"
