#!/usr/bin/env python3
"""Patch the pandoc default reference.docx to a JEIT-like B&W look:
  - Body: Times New Roman (Latin) + SimSun (宋体, CJK), 10.5pt, first-line indent, 1.25 spacing
  - Headings: SimHei (黑体) + Times New Roman, black, reduced sizes
"""
from __future__ import annotations

import re
import shutil
import zipfile

SRC = "reference.docx"

# ---- read members ----
zin = zipfile.ZipFile(SRC)
members = {n: zin.read(n) for n in zin.namelist()}
zin.close()

styles = members["word/styles.xml"].decode("utf-8")
theme = members["word/theme/theme1.xml"].decode("utf-8")

# ---------------------------------------------------------------------------
# 1) theme fonts: major -> SimHei / Times New Roman ; minor -> SimSun / Times
# ---------------------------------------------------------------------------
def patch_font_block(theme_xml: str, tag: str, latin: str, ea: str) -> str:
    m = re.search(r"(<a:%s>)(.*?)(</a:%s>)" % (tag, tag), theme_xml, re.S)
    block = m.group(2)
    block = re.sub(r'<a:latin typeface="[^"]*"(?:\s+panose="[^"]*")?\s*/>',
                   '<a:latin typeface="%s"/>' % latin, block)
    block = re.sub(r'<a:ea typeface="[^"]*"\s*/>', '<a:ea typeface="%s"/>' % ea, block)
    block = re.sub(r'(<a:font script="Hans" typeface=")[^"]*("/>)', r"\g<1>%s\g<2>" % ea, block)
    return theme_xml[:m.start()] + m.group(1) + block + m.group(3) + theme_xml[m.end():]

theme = patch_font_block(theme, "majorFont", "Times New Roman", "SimHei")
theme = patch_font_block(theme, "minorFont", "Times New Roman", "SimSun")

# ---------------------------------------------------------------------------
# 2) body default size -> 10.5pt (sz 21)
# ---------------------------------------------------------------------------
styles = styles.replace('<w:sz w:val="24" />', '<w:sz w:val="21" />', 1)
styles = styles.replace('<w:szCs w:val="24" />', '<w:szCs w:val="21" />', 1)

# ---------------------------------------------------------------------------
# 3) BodyText: tighter spacing + first-line indent (2 chars)
# ---------------------------------------------------------------------------
styles = styles.replace(
    '<w:spacing w:before="180" w:after="180" />',
    '<w:spacing w:before="0" w:after="0" w:line="300" w:lineRule="auto" />'
    '<w:ind w:firstLineChars="200" w:firstLine="420" />',
)

# ---------------------------------------------------------------------------
# 4) Headings: black colour + reduced sizes
# ---------------------------------------------------------------------------
styles = re.sub(r'<w:color w:val="[0-9A-Fa-f]{6}" w:themeColor="accent1"[^/]*/>',
                '<w:color w:val="000000" />', styles)

HEAD_SZ = {"Heading1": 24, "Heading2": 22, "Heading3": 21, "Heading4": 21}

def set_heading_size(xml: str, sid: str, sz: int) -> str:
    m = re.search(r'(<w:style w:type="paragraph" w:styleId="%s".*?</w:style>)' % sid, xml, re.S)
    if not m:
        return xml
    blk = m.group(1)
    blk = re.sub(r'<w:sz w:val="\d+" />', '<w:sz w:val="%d" />' % sz, blk)
    blk = re.sub(r'<w:szCs w:val="\d+" />', '<w:szCs w:val="%d" />' % sz, blk)
    return xml[:m.start()] + blk + xml[m.end():]

for sid, sz in HEAD_SZ.items():
    styles = set_heading_size(styles, sid, sz)

# ---------------------------------------------------------------------------
# 5) Front-matter styles (Title / Subtitle / Author all inherit centering)
# ---------------------------------------------------------------------------
def patch_style(xml: str, sid: str, size=None, bold=False, remove_numpr=False) -> str:
    m = re.search(r'(<w:style w:type="paragraph"[^>]*w:styleId="%s".*?</w:style>)' % sid, xml, re.S)
    blk = m.group(1)
    if remove_numpr:
        blk = re.sub(r'<w:numPr>\s*<w:ilvl w:val="1" />\s*</w:numPr>', '', blk)
    if size is not None:
        blk = re.sub(r'<w:sz w:val="\d+" />', '<w:sz w:val="%d" />' % size, blk)
        blk = re.sub(r'<w:szCs w:val="\d+" />', '<w:szCs w:val="%d" />' % size, blk)
    if bold and "<w:b />" not in blk:
        blk = blk.replace("<w:rPr>", "<w:rPr><w:b /><w:bCs />", 1)
    return xml[:m.start()] + blk + xml[m.end():]

styles = patch_style(styles, "Title", size=32, bold=True)        # 中文标题 16pt 粗体
styles = patch_style(styles, "Subtitle", size=24, bold=True, remove_numpr=True)  # 英文标题 12pt
styles = patch_style(styles, "Author", size=21)                  # 作者/单位 10.5pt

# ---- write back ----
members["word/styles.xml"] = styles.encode("utf-8")
members["word/theme/theme1.xml"] = theme.encode("utf-8")

with zipfile.ZipFile(SRC, "w", zipfile.ZIP_DEFLATED) as zout:
    for name, data in members.items():
        zout.writestr(name, data)

print("Patched", SRC)
print("  majorFont -> Times New Roman / SimHei ; minorFont -> Times New Roman / SimSun")
print("  body 10.5pt, first-line indent, headings black & resized")
