#!/usr/bin/env bash

set -euo pipefail

BENCH_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${BENCH_DIR}/.." && pwd)

: "${LLAMA_CPP_ROOT:?set LLAMA_CPP_ROOT to a llama.cpp checkout}"
LLAMA_CLI=${LLAMA_CLI:-"${LLAMA_CPP_ROOT}/build-cuda13/bin/llama-cli"}
: "${LLAMA_TARGET_MODEL:?set LLAMA_TARGET_MODEL to the target GGUF}"
: "${LLAMA_DFLASH_MODEL:?set LLAMA_DFLASH_MODEL to the DFlash GGUF}"
: "${LLAMA_CUDA_LIB_DIR:?set LLAMA_CUDA_LIB_DIR to the CUDA runtime library directory}"
PERL_BIN=${PERL_BIN:-perl}
RUN_LABEL=${RUN_LABEL:-$(date -u +%Y%m%dT%H%M%SZ)}
RESULT_ROOT=${RESULT_ROOT:-"${REPO_ROOT}/profiles/bench/muse-glimmer-context-ladder/${RUN_LABEL}/llamacpp"}

if [[ ! -x "${LLAMA_CLI}" ]]; then
    echo "LLAMA_CLI is not executable: ${LLAMA_CLI}" >&2
    exit 2
fi
if [[ ! -f "${LLAMA_TARGET_MODEL}" ]]; then
    echo "LLAMA_TARGET_MODEL does not exist: ${LLAMA_TARGET_MODEL}" >&2
    exit 2
fi
if [[ ! -f "${LLAMA_DFLASH_MODEL}" ]]; then
    echo "LLAMA_DFLASH_MODEL does not exist: ${LLAMA_DFLASH_MODEL}" >&2
    exit 2
fi
if ! command -v "${PERL_BIN}" >/dev/null 2>&1; then
    echo "PERL_BIN is not available: ${PERL_BIN}" >&2
    exit 2
fi

mkdir -p "${RESULT_ROOT}"

run_case() {
    local case_name=$1
    local max_new=$2
    local fixture="${REPO_ROOT}/examples/cli/messages/long_niah_${case_name}.json"
    local log_file="${RESULT_ROOT}/${case_name}.log"
    local runtime_library_path="${LLAMA_CUDA_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

    if [[ ! -f "${fixture}" ]]; then
        echo "fixture does not exist: ${fixture}" >&2
        return 2
    fi

    (
        set -x
        env LD_LIBRARY_PATH="${runtime_library_path}" \
            "${LLAMA_CLI}" \
            -m "${LLAMA_TARGET_MODEL}" \
            -md "${LLAMA_DFLASH_MODEL}" \
            -f <("${PERL_BIN}" -MJSON::PP -0777 -e \
                'binmode STDOUT, ":encoding(UTF-8)"; my $m = decode_json(<>); print $m->[1]{content};' \
                "${fixture}") \
            -sys 'Answer retrieval questions using only the supplied document. Be exact and concise.' \
            -n "${max_new}" \
            -c 131072 \
            -b 2048 \
            -ub 512 \
            -ngl all \
            -ngld all \
            -ctk q8_0 \
            -ctv q8_0 \
            -fa on \
            --spec-type draft-dflash \
            --spec-draft-n-max 4 \
            --temp 1.0 \
            --top-p 0.95 \
            --top-k 64 \
            --min-p 0.0 \
            --presence-penalty 0 \
            --frequency-penalty 0 \
            --repeat-penalty 1.0 \
            --seed 42 \
            --jinja \
            --reasoning on \
            --reasoning-effort high \
            --conversation \
            --single-turn \
            --no-display-prompt \
            --simple-io \
            --no-mmap \
            -lv 3
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
