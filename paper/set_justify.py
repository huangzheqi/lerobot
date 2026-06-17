#!/usr/bin/env python3
"""Set justified (both-edge) alignment on body paragraph styles of a .docx.

Adds <w:jc w:val="both"/> to the given paragraph style(s). Child styles that are
basedOn a justified style inherit it, so patching the base "Body Text" style is
usually enough; centered styles (Title/Author) keep their own jc and are unaffected.

Usage:  python3 set_justify.py <docx> <styleId> [<styleId> ...]
"""
from __future__ import annotations

import re
import sys
import zipfile


def add_jc(style_block: str) -> str:
    if re.search(r'<w:pPr>', style_block):
        # ensure a single jc=both inside existing pPr
        def fix(m):
            ppr = m.group(0)
            ppr = re.sub(r'<w:jc\b[^/]*/>', '', ppr)          # drop any existing jc
            return ppr.replace('</w:pPr>', '<w:jc w:val="both"/></w:pPr>')
        return re.sub(r'<w:pPr>.*?</w:pPr>', fix, style_block, count=1, flags=re.S)
    # no pPr: insert one before rPr (or before end of style)
    ppr = '<w:pPr><w:jc w:val="both"/></w:pPr>'
    if '<w:rPr>' in style_block:
        return style_block.replace('<w:rPr>', ppr + '<w:rPr>', 1)
    return style_block.replace('</w:style>', ppr + '</w:style>', 1)


def main(path: str, style_ids: list[str]) -> None:
    zin = zipfile.ZipFile(path)
    members = {n: zin.read(n) for n in zin.namelist()}
    zin.close()
    st = members["word/styles.xml"].decode("utf-8")
    done = []
    for sid in style_ids:
        # match by styleId regardless of attribute order (pandoc vs Word differ)
        pat = re.compile(r'<w:style\b[^>]*\bw:styleId="%s"[^>]*>.*?</w:style>' % re.escape(sid), re.S)
        m = pat.search(st)
        if m:
            st = st[:m.start()] + add_jc(m.group(0)) + st[m.end():]
            done.append(sid)
    members["word/styles.xml"] = st.encode("utf-8")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in members.items():
            zout.writestr(name, data)
    print(f"justified styles {done} in {path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or ["BodyText"])
