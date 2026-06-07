#!/usr/bin/env bash
# Build the ME 595 final report: Markdown + pandoc-header.tex -> PDF (xelatex).
# Usage: bash build.sh [--alt]
set -euo pipefail
cd "$(dirname "$0")"
CSL=/usr/local/texlive/2026/texmf-dist/tex/latex/citation-style-language/styles/ieee.csl

if [[ "${1:-}" == "--alt" ]]; then
  pandoc report-alt.md \
    -H pandoc-header.tex \
    --citeproc --csl "$CSL" --bibliography refs.bib \
    --pdf-engine=xelatex -V geometry:margin=1in \
    -o report-alt.pdf
  echo "wrote report-alt.pdf"
else
  pandoc report.md \
    -H pandoc-header.tex \
    --citeproc --csl "$CSL" --bibliography refs.bib \
    --pdf-engine=xelatex -V geometry:margin=1in \
    -o report.pdf
  echo "wrote report.pdf"
fi
