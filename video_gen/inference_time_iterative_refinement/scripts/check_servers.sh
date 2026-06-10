#!/usr/bin/env bash
set -euo pipefail

WAN_T2V_BASE_URL="${WAN_T2V_BASE_URL:-http://localhost:8861}"
WAN22_T2V_BASE_URL="${WAN22_T2V_BASE_URL:-}"
UNIVIDEO_EDIT_BASE_URL="${UNIVIDEO_EDIT_BASE_URL:-http://localhost:9861}"

check_url() {
  local name="$1"
  local url="$2"
  if curl -fsS "$url" >/dev/null 2>&1; then
    echo "$name reachable: $url"
  else
    echo "$name not reachable: $url" >&2
    return 1
  fi
}

check_url "Wan T2V" "$WAN_T2V_BASE_URL"
if [[ -n "$WAN22_T2V_BASE_URL" ]]; then
  check_url "Wan2.2 T2V" "$WAN22_T2V_BASE_URL"
fi
check_url "UniVideo edit" "$UNIVIDEO_EDIT_BASE_URL"
