#!/usr/bin/env bash
# Run from the project root: ./docs/web/start.sh
set -euo pipefail

DOCS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$DOCS_DIR/../.." && pwd)"

if [[ -x "$REPO_DIR/.venv/bin/python" ]]; then
    DOCS_PYTHON="$REPO_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    DOCS_PYTHON="$(command -v python3)"
else
    printf '未找到 Python 3。请先安装 Python 3，再运行此脚本。\n' >&2
    exit 1
fi

exec "$DOCS_PYTHON" "$DOCS_DIR/serve.py" "$@"
