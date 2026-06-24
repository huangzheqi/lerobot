# 一阶 & 二阶系统的时域响应分析（课程作业）

> 作业要求：参考视频，修改参数，设计你自己的一阶系统和二阶系统，观察响应曲线，用 PPT 呈现系统和响应。
>
> 参考视频：DR_CAN《控制之美 - 从传递函数到状态空间方程》第 05 集（一阶系统的时域响应）、第 06 集（二阶系统的时域响应）。
> 本作业依据视频中的参考代码（截图）进行参数修改与重新设计。

## 我的设计（在参考参数基础上修改）

| 系统 | 参考视频原参数 | 我的设计参数 |
| --- | --- | --- |
| 一阶 | `G(s) = 5/(s+5)`（τ=0.2s） | `G(s) = 4/(s+2)`，K=2，τ=0.5s，极点 s=-2 |
| 二阶 | ζ=0.5，ωn=10 | `G(s) = 25/(s²+3s+25)`，ζ=0.3（欠阻尼），ωn=5 |

二阶系统关键指标（ζ=0.3, ωn=5）：超调量 Mp≈37.2%，峰值时间 tp≈0.66s，
调节时间 ts≈2.67s，阻尼振荡频率 ωd≈4.77 rad/s。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `一阶二阶系统时域响应分析.pptx` | **作业最终成果**（13 页 PPT，含系统、代码与响应曲线） |
| `first_order.m` | 一阶系统 Octave/MATLAB 代码（修改参数后，可直接运行） |
| `second_order.m` | 二阶系统 Octave/MATLAB 代码（含阻尼比对比） |
| `generate_figures.py` | 用 Python(scipy.signal) 生成 PPT 中的响应曲线 |
| `make_ppt.py` | 用 python-pptx 自动生成 PPT |
| `figures/` | 生成的响应曲线图片 |

## 复现方法

```bash
# 方式一：在 Octave/MATLAB 中运行（与视频一致）
octave first_order.m
octave second_order.m

# 方式二：用 Python 复现曲线并重新生成 PPT
pip install numpy scipy matplotlib python-pptx pillow
python3 generate_figures.py   # 生成 figures/*.png
python3 make_ppt.py           # 生成 PPT
```

> 说明：PPT 中的曲线由 Python(scipy.signal) 仿真生成，与 `.m` 文件定义的系统
> 完全等价；`.m` 文件用于在课程的 Octave/MATLAB 环境中直接验证。
