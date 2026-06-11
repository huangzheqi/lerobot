# 机械臂抓取任务的动态系统建模（PPT + MATLAB）

内容：平面二连杆机械臂抓取负载的 **动力学建模 →（平衡点线性化）状态空间矩阵 → 传递函数** 全流程，约 3 页 PPT（封面 + 3 页正文）。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `机械臂抓取任务建模_动力学_状态空间_传递函数.pptx` | 最终 PPT（16:9，封面 + 3 页正文） |
| `grasp_dynamics_ss_tf.m` | 完整 MATLAB 脚本：由物理参数计算 M0/Kg → (A,B,C,D) → G(s)，并做极点 / 阶跃 / Bode 验证（需 Control System Toolbox） |
| `compute_model.py` | Python(NumPy/SciPy) 数值复核脚本，结果与 MATLAB 一致 |
| `gen_figures.py` | 生成 PPT 中的公式图与机械臂示意图（matplotlib） |
| `build_ppt.py` | 生成 PPT（python-pptx） |
| `assets/` | 公式与示意图 PNG |

## 重新生成 PPT

```bash
pip install python-pptx matplotlib numpy scipy
python3 gen_figures.py && python3 build_ppt.py
```

## 模型要点

- 动力学：`M(q)q̈ + C(q,q̇)q̇ + G(q) + Fv q̇ = τ`（拉格朗日方法，负载 mp 并入连杆 2）
- 平衡点：悬垂抓取位形 `q0 = [-90°, 0°]`，线性化得 `M0 δq̈ + Fv δq̇ + Kg δq = δτ`
- 状态空间：`x = [δq; δq̇]`，`A = [0 I; -M0\Kg -M0\Fv]`，`B = [0; inv(M0)]`，`C = [I 0]`，`D = 0`
- 传递函数：`G(s) = C(sI-A)^(-1)B + D`，G11(s) = (3.784s²+8.108s+106.05)/(s⁴+10.73s³+108.8s²+184.3s+1300.5)
- 极点 `-0.196±3.789j, -5.169±7.976j` 均在左半平面 → 平衡位形附近渐近稳定
