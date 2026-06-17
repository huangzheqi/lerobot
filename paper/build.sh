#!/usr/bin/env bash
# Reproducibly build the JEIT-style paper (.docx) from paper.md.
#
# Usage:  bash build.sh
#
# Requires Python 3. The script installs pypandoc-binary (bundles pandoc) and
# matplotlib on first run, regenerates the figures and the styled reference
# document, then converts paper.md -> .docx with native Word equations.
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/4] Installing build dependencies (pandoc + matplotlib) ..."
python3 -m pip install --quiet pypandoc-binary matplotlib
PANDOC="$(python3 -c 'import pypandoc; print(pypandoc.get_pandoc_path())')"

echo "[2/4] Generating figures ..."
python3 make_figures.py

echo "[3/4] Building styled reference document (SimSun/SimHei/Times New Roman) ..."
"$PANDOC" -o reference.docx --print-default-data-file reference.docx
python3 patch_reference.py

echo "[4/4] Converting paper.md -> .docx ..."
"$PANDOC" paper.md \
  --reference-doc=reference.docx \
  --resource-path=. \
  -f markdown+tex_math_dollars+raw_attribute \
  -o "机械臂强化学习论文_电子与信息学报.docx"

echo "[5/5] Fixing table column widths (fixed layout) ..."
python3 fix_tables.py "机械臂强化学习论文_电子与信息学报.docx"

echo "[6/6] Justifying body paragraphs ..."
python3 set_justify.py "机械臂强化学习论文_电子与信息学报.docx" BodyText Compact FirstParagraph

echo "Done -> 机械臂强化学习论文_电子与信息学报.docx"
