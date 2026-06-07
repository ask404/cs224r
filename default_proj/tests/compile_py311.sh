#!/usr/bin/env bash
# Modal runs Python 3.11; the local default may be newer (3.13 accepts PEP-701
# f-string syntax that 3.11 rejects). Run this BEFORE any `modal run` to catch
# 3.11-only SyntaxErrors locally instead of after a 15-min image build.
set -euo pipefail
cd "$(dirname "$0")/.."
PY311="$(uv python find 3.11 2>/dev/null || true)"
if [[ -z "$PY311" ]]; then
  echo "Python 3.11 not found. Install: uv python install 3.11"; exit 1
fi
"$PY311" -m py_compile ziprc/*.py ziprc_modal.py tests/*.py
echo "✅ 3.11-clean ($("$PY311" --version))"
