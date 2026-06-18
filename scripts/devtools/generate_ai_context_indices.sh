#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
python3 scripts/devtools/generate_ai_context_indices.py "$@"
