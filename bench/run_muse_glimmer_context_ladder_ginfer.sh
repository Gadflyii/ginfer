#!/usr/bin/env bash

set -euo pipefail

BENCH_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${BENCH_DIR}/.." && pwd)

GINFER_CLI=${GINFER_CLI:-"${REPO_ROOT}/build-120a/apps/ginfer"}
GINFER_MODEL=${GINFER_MODEL:-"${REPO_ROOT}/out/muse_glimmer_30b_dflash_q4.ginfer"}
DFLASH_DRAFT_TOKENS=${DFLASH_DRAFT_TOKENS:-4}
RUN_LABEL=${RUN_LABEL:-$(date -u +%Y%m%dT%H%M%SZ)}
RESULT_ROOT=${RESULT_ROOT:-"${REPO_ROOT}/profiles/bench/muse-glimmer-context-ladder/${RUN_LABEL}/ginfer"}

if [[ ! -x "${GINFER_CLI}" ]]; then
    echo "GINFER_CLI is not executable: ${GINFER_CLI}" >&2
    exit 2
fi
if [[ ! -f "${GINFER_MODEL}" ]]; then
    echo "GINFER_MODEL does not exist: ${GINFER_MODEL}" >&2
    exit 2
fi

mkdir -p "${RESULT_ROOT}"

run_case() {
    local case_name=$1
    local max_new=$2
    local fixture="${REPO_ROOT}/examples/cli/messages/long_niah_${case_name}.json"
    local log_file="${RESULT_ROOT}/${case_name}.log"

    if [[ ! -f "${fixture}" ]]; then
        echo "fixture does not exist: ${fixture}" >&2
        return 2
    fi

    (
        set -x
        "${GINFER_CLI}" "${GINFER_MODEL}" \
            --messages "${fixture}" \
            --max-new "${max_new}" \
            --max-context 131072 \
            --kv-dtype int8 \
            --kv-capacity auto \
            --prefill-chunk 1024 \
            --reasoning-effort high \
            --spec dflash \
            --draft-tokens "${DFLASH_DRAFT_TOKENS}" \
            --temperature 1.0 \
            --top-p 0.95 \
            --top-k 64 \
            --min-p 0 \
            --presence-penalty 0 \
            --frequency-penalty 0 \
            --seed 42
    ) 2>&1 | tee "${log_file}"
}

run_selected_case() {
    case "$1" in
        8k) run_case 8k 512 ;;
        64k) run_case 64k 512 ;;
        128k) run_case 128k 2032 ;;
        *)
            echo "unknown ladder case '$1' (expected: 8k, 64k, or 128k)" >&2
            return 2
            ;;
    esac
}

if (($# == 0)); then
    set -- 8k 64k 128k
fi

for case_name in "$@"; do
    run_selected_case "${case_name}"
done
