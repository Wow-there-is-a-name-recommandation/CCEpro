#!/bin/bash
# Compile the 4-page CVPR report.
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode main.tex > /dev/null || true
bibtex main > /dev/null || true
pdflatex -interaction=nonstopmode main.tex > /dev/null || true
pdflatex -interaction=nonstopmode main.tex > /dev/null || true
echo "Compiled → main.pdf ($(du -h main.pdf | cut -f1))"
