#!/usr/bin/env bash

set -euo pipefail

DEFAULT_DURATION_SECONDS=10800
DURATION_SECONDS="${1:-$DEFAULT_DURATION_SECONDS}"

if ! [[ "$DURATION_SECONDS" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 [duration_seconds]" >&2
    echo "Example: $0 10800" >&2
    exit 1
fi

if [[ "${KEEP_MAC_AWAKE_UNDER_CAFFEINATE:-0}" != "1" ]]; then
    if ! command -v caffeinate >/dev/null 2>&1; then
        echo "ERROR: caffeinate command not found." >&2
        exit 1
    fi

    echo "Starting caffeinate for ${DURATION_SECONDS} seconds."
    exec env KEEP_MAC_AWAKE_UNDER_CAFFEINATE=1 caffeinate -dimsu "$0" "$DURATION_SECONDS"
fi

START_TS="$(date '+%Y-%m-%d %H:%M:%S')"
END_TS="$(date -v+"${DURATION_SECONDS}"S '+%Y-%m-%d %H:%M:%S')"

trap 'echo ""; echo "Stopped early: $(date "+%Y-%m-%d %H:%M:%S")"; exit 130' INT TERM

echo "Mac sleep prevention is active."
echo "Started: $START_TS"
echo "Will finish: $END_TS"
echo "Sleeping quietly for ${DURATION_SECONDS} seconds..."

sleep "$DURATION_SECONDS"

echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
