#!/usr/bin/env bash
set -euo pipefail
ROOT="$(dirname "$0")"
BUILD_DIR="${ROOT}/build"
mkdir -p "${BUILD_DIR}"
pdflatex -halt-on-error -output-directory "${BUILD_DIR}" main.tex
bibtex "${BUILD_DIR}/main" || true
pdflatex -halt-on-error -output-directory "${BUILD_DIR}" main.tex
pdflatex -halt-on-error -output-directory "${BUILD_DIR}" main.tex
echo "Built PDF: ${BUILD_DIR}/main.pdf"

# Fail if any \ref or \includegraphics has unresolved targets (basic check)
if grep -E "LaTeX Warning:.*Reference.*undefined" "${BUILD_DIR}/main.log" >/dev/null; then
  echo "ERROR: Undefined references detected" >&2
  exit 1
fi
