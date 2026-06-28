#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 Markdown 文档转成 Word(.docx),并清理在 Word 里会变方框的特殊字符。

- 先把风险字符替换成 Word 通用写法(同时写回 .md,修好 md 自身的乱码)
- 用自带 pandoc 的 pypandoc 转 docx(format=gfm 以正确解析表格/代码块)
- 再把 docx 抽回纯文本,核验没有残留风险字符 / U+FFFD 替换符
用法:CUDA_VISIBLE_DEVICES="" uv run python docs/md_to_docx.py
"""
import os
import re
import unicodedata
import pypandoc

HERE = os.path.dirname(__file__)
FILES = ["项目流程详解.md", "奖励函数详解.md"]

# 风险字符 → Word 通用写法(多码点的放前面)
REPLACES = [
    ("⚠️", "【注意】"), ("⚠", "【注意】"),
    ("📊", "【图】"), ("📘", "【文档】"),
    ("✅", "【奖励】"), ("➖", "【惩罚】"), ("⛔", "【关闭】"),
    ("❓", "问:"),
    ("𝟙", "1"),       # 指示函数:数学双线1 → 普通1(配合 1[条件] 约定)
    ("‖", "||"),      # 范数双竖线
    ("aₜ − aₜ₋₁", "a_t − a_{t-1}"),  # 下标 t / t-1
    ("′", "'"),       # prime → 撇号
    ("ⱼ", "_j"),      # 下标 j
    ("∧", " 且 "),    # 逻辑与 → 且
    ("⟺", " 当且仅当 "),
    ("≫", ">>"),
    ("ȳ", "y"),
    ("̄", ""),   # 组合上划线(macron),x̄→x
]

# 最终核验:这些符号是允许保留的(常见字体都有)
ALLOWED = set("∈²·×−→≈≥≤…—–‘’“”【】、,。:;()《》±°★※Σ"
              "─│┌┐└┘├┤┬┴┼▶◀▲▼←↑↓①②③④⑤⑥⑦⑧⑨⑩")
GREEK = set("αβγδεζηθλμπρσφω")


def is_cjk(ch):
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF or 0x3000 <= o <= 0x303F
            or 0xFF00 <= o <= 0xFFEF or 0x3400 <= o <= 0x4DBF)


def suspicious(text):
    """返回仍可能在 Word 里缺字形的字符集合(供人工复核)。"""
    bad = {}
    for ch in text:
        if ch in "\n\r\t":
            continue
        o = ord(ch)
        if o < 0x80 or is_cjk(ch) or ch in ALLOWED or ch in GREEK:
            continue
        if ch == "�":
            bad[ch] = bad.get(ch, 0) + 1
            continue
        bad[ch] = bad.get(ch, 0) + 1
    return bad


def main():
    for fn in FILES:
        path = os.path.join(HERE, fn)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for a, b in REPLACES:
            text = text.replace(a, b)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        # 残留风险字符(转换前)
        left = suspicious(text)
        print(f"[{fn}] 清理后残留待复核字符: {dict(sorted(left.items(), key=lambda x:-x[1])) or '无'}")

        out = os.path.join(HERE, fn.replace(".md", ".docx"))
        pypandoc.convert_file(path, "docx", format="gfm", outputfile=out,
                              extra_args=["--standalone"])
        # 把生成的 docx 抽回纯文本核验
        back = pypandoc.convert_file(out, "plain")
        bad = suspicious(back)
        fffd = bad.get("�", 0)
        print(f"        → {os.path.basename(out)}  ({round(os.path.getsize(out)/1024)} KB)")
        print(f"        docx 文本核验:U+FFFD 乱码符 {fffd} 个;其它待复核 {dict((k,v) for k,v in bad.items() if k!=chr(0xfffd)) or '无'}")
    print("\n完成。")


if __name__ == "__main__":
    main()
