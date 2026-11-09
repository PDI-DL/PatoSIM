#!/usr/bin/env bash
# Gera docs/PatoSim_novidades.pdf a partir de docs/manual/patosim_novidades.tex (precisa de latexmk + pdflatex).
#   ./docs/manual/compilar_pdf.sh
# Os arquivos auxiliares do LaTeX ficam em docs/manual/build/ (ignorados pelo git).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"
if ! latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build patosim_novidades.tex >/dev/null 2>&1; then
  echo "Erro ao compilar; fim do log (build/patosim_novidades.log):" >&2
  tail -n 30 build/patosim_novidades.log >&2
  exit 1
fi
cp build/patosim_novidades.pdf ../PatoSim_novidades.pdf
echo "PDF: $(cd .. && pwd)/PatoSim_novidades.pdf ($(du -h ../PatoSim_novidades.pdf | cut -f1))"
