#!/usr/bin/env bash
# Build the ME 595 final report: Markdown + pandoc-header.tex -> PDF (xelatex).
# Usage: bash build.sh [--alt]
set -euo pipefail
cd "$(dirname "$0")"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

if [[ $# -ne 0 ]]; then
  echo "Usage: bash build.sh" >&2
  exit 1
fi

INPUT=final-report.md
OUTPUT=final-report.pdf

require_cmd pandoc
require_cmd xelatex

if [[ -z "${CSL:-}" ]]; then
  if command -v kpsewhich >/dev/null 2>&1; then
    CSL="$(kpsewhich ieee.csl 2>/dev/null || true)"
  else
    CSL=""
  fi
fi

if [[ -z "$CSL" || ! -f "$CSL" ]]; then
  echo "Could not find ieee.csl. Install the citation-style-language TeX package or set CSL=/path/to/ieee.csl." >&2
  exit 1
fi

if grep -Eq '\.svg([)\{[:space:]]|$)' "$INPUT" && ! command -v rsvg-convert >/dev/null 2>&1; then
  echo "Missing required command: rsvg-convert" >&2
  echo "This report references SVG figures; install librsvg or replace SVG references with PDF/PNG figures." >&2
  echo "On macOS with Homebrew: brew install librsvg" >&2
  exit 1
fi

pandoc "$INPUT" \
  -H pandoc-header.tex \
  --citeproc --csl "$CSL" --bibliography refs.bib \
  --pdf-engine=xelatex -V geometry:margin=1in \
  -o "$OUTPUT"
echo "wrote $OUTPUT"
