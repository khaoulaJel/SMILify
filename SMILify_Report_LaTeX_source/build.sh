#!/bin/sh
# Full build: xelatex + bibtex + 2x xelatex
cd "$(dirname "$0")"
xelatex -interaction=nonstopmode main.tex >/dev/null 2>&1
bibtex main >/dev/null 2>&1
xelatex -interaction=nonstopmode main.tex >/dev/null 2>&1
xelatex -interaction=nonstopmode main.tex 2>&1 | grep -E "^!|Output written|Overfull|Undefined|undefined" | head -30
pdfinfo main.pdf | grep Pages
