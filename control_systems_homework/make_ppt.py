# -*- coding: utf-8 -*-
"""
生成作业 PPT：一阶 & 二阶系统的时域响应分析
依赖：python-pptx, pillow
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")

# ----------------------------- 主题配色 -----------------------------
HEADER   = RGBColor(0x4A, 0x23, 0x5E)   # 深紫（呼应课程主题色）
ACCENT   = RGBColor(0x8E, 0x44, 0xAD)   # 中紫
ACCENT2  = RGBColor(0xC0, 0x39, 0x2B)   # 红（阶跃）
TEXT     = RGBColor(0x2E, 0x2E, 0x2E)
SUBTLE   = RGBColor(0x70, 0x70, 0x70)
LIGHTBG  = RGBColor(0xF2, 0xEE, 0xF7)   # 浅紫底（代码框/表格）
WHITE    = RGBColor(0xFF, 0xFF, 0xFF)
TABLE_HD = RGBColor(0x6A, 0x36, 0x82)

CN_FONT   = "微软雅黑"
CODE_FONT = "Consolas"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]

_page = 0


def _set_font(run, name=CN_FONT, size=18, bold=False, color=TEXT, italic=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = name
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.append(el)
        el.set("typeface", name)


def add_textbox(slide, left, top, width, height, lines, align=PP_ALIGN.LEFT,
                anchor=MSO_ANCHOR.TOP, wrap=True):
    """lines: list of dict(text, size, bold, color, font, space_after, level, align)"""
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = ln.get("align", align)
        p.space_after = Pt(ln.get("space_after", 6))
        p.space_before = Pt(ln.get("space_before", 0))
        if "line_spacing" in ln:
            p.line_spacing = ln["line_spacing"]
        run = p.add_run()
        run.text = ln["text"]
        _set_font(run, name=ln.get("font", CN_FONT), size=ln.get("size", 18),
                  bold=ln.get("bold", False), color=ln.get("color", TEXT),
                  italic=ln.get("italic", False))
    return tb


def add_rect(slide, left, top, width, height, fill, line=None, shadow=False):
    sp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    sp.fill.solid()
    sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line
        sp.line.width = Pt(0.75)
    sp.shadow.inherit = False
    return sp


def slide_header(slide, title, subtitle=None):
    """标准内容页页眉"""
    global _page
    _page += 1
    # 顶部色条
    add_rect(slide, 0, 0, SW, Inches(1.05), HEADER)
    # 左侧强调竖条
    add_rect(slide, 0, 0, Inches(0.18), Inches(1.05), ACCENT)
    add_textbox(slide, Inches(0.45), Inches(0.12), Inches(11.5), Inches(0.82),
                [{"text": title, "size": 26, "bold": True, "color": WHITE}],
                anchor=MSO_ANCHOR.MIDDLE)
    # 页码
    add_textbox(slide, Inches(12.3), Inches(7.02), Inches(0.9), Inches(0.4),
                [{"text": str(_page), "size": 12, "color": SUBTLE,
                  "align": PP_ALIGN.RIGHT}])
    if subtitle:
        add_textbox(slide, Inches(0.45), Inches(1.12), Inches(12.4), Inches(0.5),
                    [{"text": subtitle, "size": 15, "color": ACCENT, "bold": True}])


def add_image_fit(slide, path, left, top, max_w, max_h, border=True):
    with Image.open(path) as im:
        w, h = im.size
    ar = w / h
    box_ar = max_w / max_h
    if ar > box_ar:
        disp_w = max_w
        disp_h = int(max_w / ar)
    else:
        disp_h = max_h
        disp_w = int(max_h * ar)
    l = left + (max_w - disp_w) // 2
    t = top + (max_h - disp_h) // 2
    pic = slide.shapes.add_picture(path, l, t, width=disp_w, height=disp_h)
    if border:
        pic.line.color.rgb = RGBColor(0xCC, 0xCC, 0xCC)
        pic.line.width = Pt(0.75)
    return pic


def add_code_box(slide, left, top, width, height, code_lines, title=None):
    box = add_rect(slide, left, top, width, height, LIGHTBG,
                   line=RGBColor(0xD8, 0xCC, 0xE6))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.15)
    tf.margin_right = Inches(0.1)
    tf.margin_top = Inches(0.08)
    tf.margin_bottom = Inches(0.08)
    first = True
    if title:
        p = tf.paragraphs[0]
        r = p.add_run(); r.text = title
        _set_font(r, name=CN_FONT, size=12, bold=True, color=ACCENT)
        first = False
    for cl in code_lines:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.line_spacing = 1.05
        r = p.add_run(); r.text = cl
        _set_font(r, name=CODE_FONT, size=12.5, color=RGBColor(0x22, 0x22, 0x22))
    return box


def bullets(items):
    """把 (text, level) 列表转成 add_textbox 的 lines"""
    out = []
    for it in items:
        if isinstance(it, tuple):
            text, level = it
        else:
            text, level = it, 0
        prefix = "•  " if level == 0 else "    –  "
        out.append({"text": prefix + text, "size": 17 if level == 0 else 15,
                    "color": TEXT, "space_after": 8, "line_spacing": 1.15})
    return out


# =====================================================================
# Slide 1 —— 封面
# =====================================================================
s = prs.slides.add_slide(BLANK)
add_rect(s, 0, 0, SW, SH, HEADER)
add_rect(s, 0, Inches(2.55), SW, Inches(0.06), ACCENT)
add_rect(s, 0, Inches(4.62), SW, Inches(0.06), ACCENT)
add_textbox(s, Inches(1.0), Inches(2.7), Inches(11.3), Inches(1.4),
            [{"text": "一阶 & 二阶系统的时域响应分析",
              "size": 44, "bold": True, "color": WHITE, "align": PP_ALIGN.CENTER}],
            anchor=MSO_ANCHOR.MIDDLE)
add_textbox(s, Inches(1.0), Inches(4.0), Inches(11.3), Inches(0.6),
            [{"text": "控制系统时域响应 · 课程作业",
              "size": 22, "color": RGBColor(0xD9, 0xC7, 0xE6),
              "align": PP_ALIGN.CENTER}])
add_textbox(s, Inches(1.0), Inches(4.85), Inches(11.3), Inches(0.6),
            [{"text": "参考视频，修改参数，设计自己的一阶 / 二阶系统并观察响应曲线",
              "size": 16, "color": RGBColor(0xC9, 0xB6, 0xD9),
              "align": PP_ALIGN.CENTER}])
add_textbox(s, Inches(1.0), Inches(6.4), Inches(11.3), Inches(0.5),
            [{"text": "姓名：__________      学号：__________      日期：2026-06-24",
              "size": 14, "color": RGBColor(0xB7, 0xA3, 0xC9),
              "align": PP_ALIGN.CENTER}])

# =====================================================================
# Slide 2 —— 作业要求与设计概览
# =====================================================================
s = prs.slides.add_slide(BLANK)
slide_header(s, "作业要求与设计概览")
add_textbox(s, Inches(0.5), Inches(1.35), Inches(12.3), Inches(1.0),
            [{"text": "作业要求", "size": 18, "bold": True, "color": ACCENT,
              "space_after": 4},
             {"text": "参考视频，修改参数，设计你自己的一阶系统和二阶系统，"
                      "观察响应曲线，用 PPT 呈现系统和响应。",
              "size": 16, "color": TEXT, "line_spacing": 1.2}])
add_textbox(s, Inches(0.5), Inches(2.75), Inches(12.3), Inches(0.4),
            [{"text": "参数修改对照表（在参考视频基础上修改）", "size": 18,
              "bold": True, "color": ACCENT}])
# 表格
rows, cols = 3, 3
tbl_shape = s.shapes.add_table(rows, cols, Inches(0.5), Inches(3.25),
                               Inches(12.33), Inches(2.4))
table = tbl_shape.table
table.columns[0].width = Inches(2.0)
table.columns[1].width = Inches(4.9)
table.columns[2].width = Inches(5.43)
data = [
    ["系统类型", "参考视频原始参数", "我的设计参数（修改后）"],
    ["一阶系统",
     "G(s) = 5 / (s + 5)\n（极点 s = -5，时间常数 τ = 0.2 s）",
     "G(s) = 4 / (s + 2)\nK = 2，τ = 0.5 s，极点 s = -2"],
    ["二阶系统",
     "G(s) = ωn² /(s²+2ζωn s+ωn²)\nζ = 0.5，ωn = 10 rad/s",
     "G(s) = 25 /(s² + 3s + 25)\nζ = 0.3（欠阻尼），ωn = 5 rad/s"],
]
for r in range(rows):
    table.rows[r].height = Inches(0.6 if r == 0 else 0.9)
    for c in range(cols):
        cell = table.cell(r, c)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = Inches(0.12)
        cell.margin_right = Inches(0.08)
        cell.fill.solid()
        if r == 0:
            cell.fill.fore_color.rgb = TABLE_HD
        else:
            cell.fill.fore_color.rgb = WHITE if r % 2 else LIGHTBG
        tfc = cell.text_frame
        tfc.word_wrap = True
        para = tfc.paragraphs[0]
        for li, line in enumerate(cell_text_lines := data[r][c].split("\n")):
            p = para if li == 0 else tfc.add_paragraph()
            run = p.add_run(); run.text = line
            is_head = (r == 0)
            _set_font(run, size=14 if is_head else (13 if li == 0 else 11.5),
                      bold=is_head or (c == 0),
                      color=WHITE if is_head else TEXT)
add_textbox(s, Inches(0.5), Inches(5.95), Inches(12.3), Inches(1.2),
            [{"text": "设计思路：一阶系统增大时间常数 τ 并提高直流增益 K，使响应更"
                      "慢、稳态值更高；二阶系统减小阻尼比 ζ，使系统呈现更明显的超"
                      "调与振荡，便于观察 ζ 对响应的影响。",
              "size": 14, "color": SUBTLE, "line_spacing": 1.25}])

# =====================================================================
# Slide 3 —— 一阶系统理论基础
# =====================================================================
s = prs.slides.add_slide(BLANK)
slide_header(s, "一、一阶系统理论基础")
add_textbox(s, Inches(0.5), Inches(1.35), Inches(7.4), Inches(5.6),
            [{"text": "标准传递函数形式", "size": 18, "bold": True, "color": ACCENT,
              "space_after": 6}] +
            [{"text": "G(s) = K / (τs + 1)", "size": 22, "bold": True,
              "color": ACCENT2, "space_after": 10, "font": CODE_FONT}] +
            bullets([
                "K：直流增益，决定单位阶跃响应的稳态值",
                "τ：时间常数，决定响应的快慢；极点位于 s = -1/τ",
                "由 G(s) 可写出微分方程： τ·dy/dt + y = K·u",
                "状态空间： dx/dt = -(1/τ)x + (K/τ)u， y = x",
            ]) +
            [{"text": "一阶系统的特点", "size": 18, "bold": True, "color": ACCENT,
              "space_before": 10, "space_after": 6}] +
            bullets([
                "无超调、无振荡，单调趋于稳态",
                "t = τ 时响应达到稳态值的 63.2%",
                "调节时间 ts ≈ 4τ（2% 准则）",
                "τ 越小，极点越靠左，响应越快",
            ]))
# 右侧框图示意
add_rect(s, Inches(8.2), Inches(1.6), Inches(4.6), Inches(2.4), LIGHTBG,
         line=RGBColor(0xD8, 0xCC, 0xE6))
add_textbox(s, Inches(8.2), Inches(1.72), Inches(4.6), Inches(0.4),
            [{"text": "系统框图", "size": 13, "bold": True, "color": ACCENT,
              "align": PP_ALIGN.CENTER}])
add_rect(s, Inches(9.6), Inches(2.55), Inches(1.8), Inches(0.9), WHITE,
         line=ACCENT)
add_textbox(s, Inches(9.6), Inches(2.55), Inches(1.8), Inches(0.9),
            [{"text": "K/(τs+1)", "size": 15, "bold": True, "color": TEXT,
              "align": PP_ALIGN.CENTER, "font": CODE_FONT}],
            anchor=MSO_ANCHOR.MIDDLE)
add_textbox(s, Inches(8.25), Inches(2.7), Inches(1.4), Inches(0.5),
            [{"text": "U(s) →", "size": 13, "color": TEXT}])
add_textbox(s, Inches(11.45), Inches(2.7), Inches(1.3), Inches(0.5),
            [{"text": "→ Y(s)", "size": 13, "color": TEXT}])
add_textbox(s, Inches(8.2), Inches(4.25), Inches(4.6), Inches(2.6),
            [{"text": "物理直观", "size": 16, "bold": True, "color": ACCENT,
              "space_after": 6}] +
            bullets([
                "类比 RC 电路充电 / 水箱进水",
                "输入突变后，输出按指数规律平滑跟随",
                "时间常数 τ 即“惯性”大小",
            ]))

# =====================================================================
# Slide 4 —— 我的一阶系统设计
# =====================================================================
s = prs.slides.add_slide(BLANK)
slide_header(s, "二、我设计的一阶系统")
add_textbox(s, Inches(0.5), Inches(1.35), Inches(6.2), Inches(0.6),
            [{"text": "G(s) = 4 / (s + 2) = 2 / (0.5s + 1)", "size": 24,
              "bold": True, "color": ACCENT2, "font": CODE_FONT}])
add_textbox(s, Inches(0.5), Inches(2.1), Inches(6.2), Inches(2.6),
            bullets([
                "直流增益 K = 2  → 阶跃稳态值为 2",
                "时间常数 τ = 0.5 s（原视频为 0.2 s，响应更慢）",
                "极点 s = -2",
                "调节时间 ts ≈ 4τ = 2 s，上升时间 ≈ 2.2τ ≈ 1.1 s",
            ]))
add_textbox(s, Inches(0.5), Inches(4.35), Inches(6.2), Inches(0.45),
            [{"text": "状态空间模型（用于初始状态响应）", "size": 15,
              "bold": True, "color": ACCENT}])
add_code_box(s, Inches(0.5), Inches(4.85), Inches(6.2), Inches(1.3),
             ["dx/dt = -2·x + 4·u", "   y  = 1·x + 0·u", "A=-2, B=4, C=1, D=0,  x0=5"])
# 代码
add_code_box(s, Inches(7.0), Inches(1.45), Inches(5.85), Inches(4.9),
             ["% 定义一阶系统 G(s)=4/(s+2)",
              "K = 2; tau = 0.5;",
              "G_a = tf([K/tau], [1, 1/tau]);",
              "",
              "% 单位冲激响应",
              "subplot(3,1,1); impulse(G_a);",
              "",
              "% 单位阶跃响应",
              "subplot(3,1,2); step(G_a);",
              "",
              "% 初始状态响应",
              "subplot(3,1,3);",
              "A=-2; B=4; C=1; D=0;",
              "sys = ss(A,B,C,D);",
              "initial(sys, 5);"],
             title="Octave / MATLAB 代码（first_order.m）")
add_textbox(s, Inches(7.0), Inches(6.45), Inches(5.85), Inches(0.6),
            [{"text": "（曲线由等价的 Python / scipy.signal 仿真生成）",
              "size": 12, "color": SUBTLE, "align": PP_ALIGN.CENTER}])

# =====================================================================
# Slide 5 —— 一阶三种响应
# =====================================================================
s = prs.slides.add_slide(BLANK)
slide_header(s, "三、一阶系统的三种时域响应")
add_image_fit(s, os.path.join(FIG, "fo_three_responses.png"),
              Inches(0.4), Inches(1.25), Inches(6.6), Inches(5.95))
add_textbox(s, Inches(7.3), Inches(1.5), Inches(5.6), Inches(5.5),
            [{"text": "曲线解读", "size": 18, "bold": True, "color": ACCENT,
              "space_after": 8}] +
            bullets([
                "① 冲激响应：t=0 跃至 4，随后按 4·e^(-2t) 指数衰减到 0",
                "② 阶跃响应：单调上升，无超调，稳态值 = K = 2",
                "③ 初始状态响应：零输入下从 x0=5 按 5·e^(-2t) 衰减到 0",
            ]) +
            [{"text": "共同规律", "size": 18, "bold": True, "color": ACCENT,
              "space_before": 12, "space_after": 8}] +
            bullets([
                "三种响应都由同一个极点 s=-2 主导",
                "衰减 / 上升的快慢完全由时间常数 τ=0.5 s 决定",
                "一阶系统始终平滑、无振荡",
            ]))

# =====================================================================
# Slide 6 —— 一阶阶跃响应性能指标
# =====================================================================
s = prs.slides.add_slide(BLANK)
slide_header(s, "四、一阶系统阶跃响应的性能指标")
add_image_fit(s, os.path.join(FIG, "fo_step_annotated.png"),
              Inches(0.4), Inches(1.35), Inches(7.7), Inches(5.7))
add_textbox(s, Inches(8.35), Inches(1.6), Inches(4.6), Inches(5.3),
            [{"text": "关键指标", "size": 18, "bold": True, "color": ACCENT,
              "space_after": 8}] +
            bullets([
                "稳态值 = K = 2",
                "时间常数 τ = 0.5 s（63.2% 点）",
                "调节时间 ts ≈ 4τ = 2 s",
                "上升时间 tr ≈ 2.2τ ≈ 1.1 s",
                "超调量 Mp = 0（无超调）",
            ]) +
            [{"text": "结论", "size": 18, "bold": True, "color": ACCENT,
              "space_before": 12, "space_after": 8}] +
            bullets([
                "一阶系统的动态品质完全由 τ 刻画",
                "稳态精度由 K 决定",
            ]))

# =====================================================================
# Slide 7 —— 时间常数的影响
# =====================================================================
s = prs.slides.add_slide(BLANK)
slide_header(s, "五、时间常数 τ 对一阶响应的影响（拓展）")
add_image_fit(s, os.path.join(FIG, "fo_tau_compare.png"),
              Inches(0.4), Inches(1.35), Inches(7.7), Inches(5.7))
add_textbox(s, Inches(8.35), Inches(1.7), Inches(4.6), Inches(5.0),
            [{"text": "对比说明", "size": 18, "bold": True, "color": ACCENT,
              "space_after": 8}] +
            bullets([
                "固定直流增益 K = 2，仅改变 τ",
                "τ = 0.25 s → 极点 -4，响应最快",
                "τ = 0.5 s → 极点 -2（本设计）",
                "τ = 1.0 s → 极点 -1，响应最慢",
            ]) +
            [{"text": "规律", "size": 18, "bold": True, "color": ACCENT,
              "space_before": 12, "space_after": 8}] +
            bullets([
                "τ 越小，极点越靠近虚轴左侧，响应越快",
                "三条曲线最终都收敛到相同稳态值 2",
                "改变 τ 只影响快慢，不影响稳态值",
            ]))

# =====================================================================
# Slide 8 —— 二阶系统理论基础
# =====================================================================
s = prs.slides.add_slide(BLANK)
slide_header(s, "六、二阶系统理论基础")
add_textbox(s, Inches(0.5), Inches(1.35), Inches(7.4), Inches(5.6),
            [{"text": "标准传递函数形式", "size": 18, "bold": True, "color": ACCENT,
              "space_after": 6}] +
            [{"text": "G(s) = ωn² / (s² + 2ζωn·s + ωn²)", "size": 20, "bold": True,
              "color": ACCENT2, "space_after": 10, "font": CODE_FONT}] +
            bullets([
                "ωn：自然频率，决定响应快慢与振荡频率",
                "ζ：阻尼比，决定超调量与振荡程度",
                "极点： s = -ζωn ± jωn·√(1-ζ²)",
                "阻尼振荡频率 ωd = ωn·√(1-ζ²)",
            ]) +
            [{"text": "按阻尼比 ζ 分类", "size": 18, "bold": True, "color": ACCENT,
              "space_before": 10, "space_after": 6}] +
            bullets([
                "ζ = 0：无阻尼，等幅振荡",
                "0 < ζ < 1：欠阻尼，衰减振荡（有超调）",
                "ζ = 1：临界阻尼，最快且无超调",
                "ζ > 1：过阻尼，无振荡但响应慢",
            ]))
add_rect(s, Inches(8.2), Inches(1.6), Inches(4.6), Inches(2.6), LIGHTBG,
         line=RGBColor(0xD8, 0xCC, 0xE6))
add_textbox(s, Inches(8.35), Inches(1.75), Inches(4.3), Inches(2.4),
            [{"text": "状态空间（可控标准型）", "size": 15, "bold": True,
              "color": ACCENT, "space_after": 6},
             {"text": "A = [ 0      1 ]", "size": 14, "font": CODE_FONT},
             {"text": "    [-ωn²  -2ζωn]", "size": 14, "font": CODE_FONT},
             {"text": "B = [0 ; ωn²]", "size": 14, "font": CODE_FONT},
             {"text": "C = [1   0],  D = 0", "size": 14, "font": CODE_FONT}])
add_textbox(s, Inches(8.2), Inches(4.4), Inches(4.6), Inches(2.5),
            [{"text": "物理直观", "size": 16, "bold": True, "color": ACCENT,
              "space_after": 6}] +
            bullets([
                "类比质量-弹簧-阻尼系统",
                "ωn 来自弹簧刚度 / 质量",
                "ζ 来自阻尼器，抑制振荡",
            ]))

# =====================================================================
# Slide 9 —— 我的二阶系统设计
# =====================================================================
s = prs.slides.add_slide(BLANK)
slide_header(s, "七、我设计的二阶系统")
add_textbox(s, Inches(0.5), Inches(1.35), Inches(6.3), Inches(0.6),
            [{"text": "G(s) = 25 / (s² + 3s + 25)", "size": 24, "bold": True,
              "color": ACCENT2, "font": CODE_FONT}])
add_textbox(s, Inches(0.5), Inches(2.1), Inches(6.3), Inches(2.6),
            bullets([
                "自然频率 ωn = 5 rad/s（原视频为 10）",
                "阻尼比 ζ = 0.3（原视频 0.5，更欠阻尼）",
                "极点 s = -1.5 ± j4.77（共轭复极点）",
                "属欠阻尼系统 → 有明显超调与衰减振荡",
            ]))
add_textbox(s, Inches(0.5), Inches(4.35), Inches(6.3), Inches(0.45),
            [{"text": "状态空间模型（用于初始状态响应）", "size": 15,
              "bold": True, "color": ACCENT}])
add_code_box(s, Inches(0.5), Inches(4.85), Inches(6.3), Inches(1.5),
             ["A = [0 1; -25 -3]   % 可控标准型",
              "B = [0; 25]",
              "C = [1 0],  D = 0,  z0 = [1; 0]"])
add_code_box(s, Inches(7.1), Inches(1.45), Inches(5.75), Inches(5.0),
             ["% 定义二阶系统",
              "zeta = 0.3;  w_n = 5;",
              "G_s = tf([w_n^2], ...",
              "        [1, 2*w_n*zeta, w_n^2]);",
              "",
              "subplot(3,1,1); impulse(G_s);",
              "subplot(3,1,2); step(G_s);",
              "",
              "% 初始状态响应",
              "A = [0 1; -w_n^2 -2*zeta*w_n];",
              "B = [0; w_n^2];",
              "C = [1 0]; D = 0;",
              "sys = ss(A,B,C,D);",
              "initial(sys, [1;0]);"],
             title="Octave / MATLAB 代码（second_order.m）")

# =====================================================================
# Slide 10 —— 二阶三种响应
# =====================================================================
s = prs.slides.add_slide(BLANK)
slide_header(s, "八、二阶系统的三种时域响应")
add_image_fit(s, os.path.join(FIG, "so_three_responses.png"),
              Inches(0.4), Inches(1.25), Inches(6.6), Inches(5.95))
add_textbox(s, Inches(7.3), Inches(1.5), Inches(5.6), Inches(5.5),
            [{"text": "曲线解读", "size": 18, "bold": True, "color": ACCENT,
              "space_after": 8}] +
            bullets([
                "① 冲激响应：先冲高再衰减振荡，逐渐回到 0",
                "② 阶跃响应：上升后越过稳态值产生超调，振荡衰减到 1",
                "③ 初始状态响应：从 z0=[1,0] 出发，衰减振荡回到 0",
            ]) +
            [{"text": "共同规律", "size": 18, "bold": True, "color": ACCENT,
              "space_before": 12, "space_after": 8}] +
            bullets([
                "因 ζ=0.3 < 1，三种响应都呈衰减振荡",
                "振荡频率即阻尼振荡频率 ωd ≈ 4.77 rad/s",
                "包络衰减速度由 ζωn = 1.5 决定",
            ]))

# =====================================================================
# Slide 11 —— 二阶阶跃性能指标
# =====================================================================
s = prs.slides.add_slide(BLANK)
slide_header(s, "九、二阶系统阶跃响应的性能指标")
add_image_fit(s, os.path.join(FIG, "so_step_annotated.png"),
              Inches(0.4), Inches(1.35), Inches(7.5), Inches(5.7))
# 指标表
add_textbox(s, Inches(8.2), Inches(1.45), Inches(4.7), Inches(0.4),
            [{"text": "性能指标（ζ=0.3, ωn=5）", "size": 16, "bold": True,
              "color": ACCENT}])
metrics = [
    ("超调量 Mp", "37.2 %"),
    ("峰值时间 tp", "0.66 s"),
    ("调节时间 ts (2%)", "≈ 2.67 s"),
    ("阻尼振荡频率 ωd", "4.77 rad/s"),
    ("稳态值", "1.0（无稳态误差）"),
]
mt = s.shapes.add_table(len(metrics) + 1, 2, Inches(8.2), Inches(1.95),
                        Inches(4.7), Inches(3.3)).table
mt.columns[0].width = Inches(2.6)
mt.columns[1].width = Inches(2.1)
hdr = [("指标", "数值")] + metrics
for r, (k, v) in enumerate(hdr):
    mt.rows[r].height = Inches(0.5)
    for c, val in enumerate((k, v)):
        cell = mt.cell(r, c)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = Inches(0.1)
        cell.fill.solid()
        if r == 0:
            cell.fill.fore_color.rgb = TABLE_HD
        else:
            cell.fill.fore_color.rgb = WHITE if r % 2 else LIGHTBG
        p = cell.text_frame.paragraphs[0]
        run = p.add_run(); run.text = val
        _set_font(run, size=13, bold=(r == 0 or c == 0),
                  color=WHITE if r == 0 else TEXT)
add_textbox(s, Inches(8.2), Inches(5.5), Inches(4.7), Inches(1.5),
            [{"text": "理论公式", "size": 14, "bold": True, "color": ACCENT,
              "space_after": 4},
             {"text": "Mp = e^(-ζπ/√(1-ζ²))", "size": 13, "font": CODE_FONT,
              "color": TEXT},
             {"text": "tp = π / ωd,  ts ≈ 4/(ζωn)", "size": 13, "font": CODE_FONT,
              "color": TEXT}])

# =====================================================================
# Slide 12 —— 阻尼比的影响（核心）
# =====================================================================
s = prs.slides.add_slide(BLANK)
slide_header(s, "十、阻尼比 ζ 对二阶响应的影响（核心）")
add_image_fit(s, os.path.join(FIG, "so_zeta_compare.png"),
              Inches(0.4), Inches(1.3), Inches(7.7), Inches(5.8))
add_textbox(s, Inches(8.35), Inches(1.55), Inches(4.6), Inches(5.4),
            [{"text": "固定 ωn = 5，改变 ζ", "size": 18, "bold": True,
              "color": ACCENT, "space_after": 8}] +
            bullets([
                "ζ = 0：等幅振荡，不收敛",
                "ζ ↑（欠阻尼）：超调和振荡减小",
                "ζ = 0.707：超调小、响应较快（工程常用最佳阻尼）",
                "ζ = 1：临界阻尼，无超调最快",
                "ζ > 1：过阻尼，无振荡但变慢",
            ]) +
            [{"text": "工程启示", "size": 18, "bold": True, "color": ACCENT,
              "space_before": 12, "space_after": 8}] +
            bullets([
                "ζ 是快速性与平稳性之间的权衡",
                "我设计的 ζ=0.3 偏小，超调较大，便于观察振荡",
            ]))

# =====================================================================
# Slide 13 —— 总结
# =====================================================================
s = prs.slides.add_slide(BLANK)
slide_header(s, "总结")
add_textbox(s, Inches(0.6), Inches(1.5), Inches(12.0), Inches(5.4),
            [{"text": "一阶系统  G(s) = K/(τs+1)", "size": 20, "bold": True,
              "color": ACCENT2, "space_after": 6}] +
            bullets([
                "由直流增益 K 与时间常数 τ 两个参数完全确定",
                "响应单调无超调；K 决定稳态值，τ 决定快慢（ts≈4τ）",
                "本设计 G(s)=4/(s+2)：K=2，τ=0.5 s",
            ]) +
            [{"text": "二阶系统  G(s) = ωn²/(s²+2ζωn s+ωn²)", "size": 20,
              "bold": True, "color": ACCENT2, "space_before": 12,
              "space_after": 6}] +
            bullets([
                "由自然频率 ωn 与阻尼比 ζ 两个参数决定",
                "ωn 决定快慢与振荡频率；ζ 决定超调与振荡程度",
                "本设计 G(s)=25/(s²+3s+25)：ωn=5，ζ=0.3，超调 37.2%",
            ]) +
            [{"text": "核心收获：通过修改参数并观察冲激 / 阶跃 / 初始状态三种响应，"
                      "直观理解了系统参数与时域性能指标之间的对应关系。",
              "size": 16, "color": TEXT, "space_before": 14, "line_spacing": 1.25}])

# ---------------------------------------------------------------------
out = os.path.join(HERE, "一阶二阶系统时域响应分析.pptx")
prs.save(out)
print("PPT 已生成:", out)
print("幻灯片数量:", len(prs.slides._sldIdLst))
