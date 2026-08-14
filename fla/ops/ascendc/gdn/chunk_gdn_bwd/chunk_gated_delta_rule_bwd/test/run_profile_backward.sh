#!/usr/bin/env bash
set -euo pipefail

test_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "${test_dir}" rev-parse --show-toplevel)
cd "${repo_root}"

exec python "${test_dir}/profile_backward.py" \
    --case "${GDN_PERF_CASE:-P1}" \
    --implementation "${GDN_IMPLEMENTATION:-candidate}" \
    --device "${GDN_DEVICE:-0}" \
    --warmup "${GDN_PERF_WARMUP:-0}" \
    --repeat "${GDN_PERF_REPEAT:-1}"
