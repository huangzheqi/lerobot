#!/usr/bin/env python3
"""Force fixed column widths on the tables of a pandoc-generated .docx.

Pandoc emits autofit tables (tblW=auto), so Word re-computes column widths from
content; a single long token in one cell then squeezes the other columns and makes
short headers wrap vertically. This script sets each table to a fixed layout with
sensible per-column widths so the layout is deterministic in Word.

Usage:  python3 fix_tables.py <docx>
"""
from __future__ import annotations

import re
import sys
import zipfile

TOTAL = 9000  # total table width in twips (~6.25 in, fits A4/Letter portrait text width)


def widths_for(text: str, ncols: int):
    """Pick relative column widths from a table's flattened text."""
    if "阶段" in text and "权重" in text:          # Table 1: stage | item | weight | role
        return [9, 44, 13, 34]
    if "折扣因子" in text or "学习率" in text:        # Table 2: param | value | param | value
        return [30, 20, 30, 20]
    if "抓取成功率" in text or "端到端" in text:      # Table 3: method | grasp | placement
        return [54, 23, 23]
    return [round(100 / ncols)] * ncols           # fallback: equal


def patch_table(m: re.Match) -> str:
    tbl = m.group(0)
    text = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", tbl))
    grid = re.search(r"<w:tblGrid>.*?</w:tblGrid>", tbl, re.S).group(0)
    ncols = len(re.findall(r"<w:gridCol\b[^>]*/>", grid))
    w = widths_for(text, ncols)
    if len(w) != ncols:
        w = [round(100 / ncols)] * ncols
    sw = sum(w)
    new_grid = "<w:tblGrid>" + "".join(
        f'<w:gridCol w:w="{round(TOTAL * wi / sw)}" />' for wi in w) + "</w:tblGrid>"
    tbl = re.sub(r"<w:tblGrid>.*?</w:tblGrid>", new_grid, tbl, flags=re.S)
    # fixed total width
    tbl = re.sub(r'<w:tblW[^/]*/>', f'<w:tblW w:type="dxa" w:w="{TOTAL}" />', tbl)
    # fixed layout (insert if missing)
    if "<w:tblLayout" in tbl:
        tbl = re.sub(r'<w:tblLayout[^/]*/>', '<w:tblLayout w:type="fixed" />', tbl)
    else:
        tbl = tbl.replace('<w:tblW', '<w:tblLayout w:type="fixed" /><w:tblW', 1)
    return tbl


def main(path: str) -> None:
    zin = zipfile.ZipFile(path)
    members = {n: zin.read(n) for n in zin.namelist()}
    zin.close()
    doc = members["word/document.xml"].decode("utf-8")
    doc, n = re.subn(r"<w:tbl>.*?</w:tbl>", patch_table, doc, flags=re.S)
    members["word/document.xml"] = doc.encode("utf-8")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in members.items():
            zout.writestr(name, data)
    print(f"fixed {n} tables -> fixed layout, total width {TOTAL} twips")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "机械臂强化学习论文_电子与信息学报.docx")
