# 基于强化学习的机械臂线性二次型最优控制研究

# Reinforcement Learning Based Linear Quadratic Optimal Control of Robotic Manipulators

| 项目 | 内容 |
| --- | --- |
| 课程 | 线性系统理论 |
| 作者 | （姓名 / 学号） |
| 日期 | 2026-06 |
| 论文类型 | 非综述类（含数值仿真） |
| 格式 | 参照《浙江大学学报（工学版）》，需中、英文摘要，参考文献按 GB/T 7714 著录 |

---

## 摘要

针对机械臂在模型参数不确定条件下的最优轨迹跟踪控制问题，本文从线性系统理论出发，研究将线性二次型调节器（Linear Quadratic Regulator, LQR）与强化学习相结合的控制方法。首先，对 $n$ 自由度机械臂的非线性动力学方程在平衡点（或参考轨迹）附近进行线性化，建立其状态空间模型，并对系统的可控性、可观性与稳定性进行分析；在此基础上，将最优控制问题归结为求解代数 Riccati 方程（Algebraic Riccati Equation, ARE）的线性二次型调节问题。针对实际工程中系统模型参数难以精确获取的情形，引入基于自适应动态规划（Adaptive Dynamic Programming, ADP）的策略迭代算法，以在线、数据驱动的方式逼近 Riccati 方程的解，在无需精确系统模型的条件下学习最优状态反馈增益，并利用 Lyapunov 理论证明闭环系统的稳定性与迭代算法的收敛性。最后，以二自由度平面机械臂为对象进行数值仿真，将所提强化学习控制器与模型已知的传统 LQR 控制器进行对比。仿真结果表明，在系统模型未知的条件下，所提方法能够收敛到接近最优的状态反馈增益，实现稳定、快速的关节轨迹跟踪，从而验证了线性系统理论与强化学习相结合在机械臂控制中的有效性。

**关键词：** 线性系统理论；线性二次型调节器（LQR）；强化学习；自适应动态规划；机械臂控制；代数 Riccati 方程

## Abstract

To address the optimal trajectory-tracking control of robotic manipulators under model uncertainty, this paper, starting from linear system theory, investigates a reinforcement learning (RL) control method that integrates the linear quadratic regulator (LQR). First, the nonlinear dynamics of an $n$-degree-of-freedom manipulator are linearized about an equilibrium point (or a reference trajectory) to obtain a state-space model, whose controllability, observability and stability are analyzed. The optimal control problem is then reformulated as an LQR problem whose solution requires solving the algebraic Riccati equation (ARE). Considering that exact model parameters are often unavailable in practice, a policy-iteration algorithm based on adaptive dynamic programming (ADP) is introduced to approximate the solution of the ARE in an online, data-driven manner, thereby learning the optimal state-feedback gain without an exact system model; the closed-loop stability and the convergence of the iterative algorithm are established by means of Lyapunov theory. Finally, numerical simulations on a two-degree-of-freedom planar manipulator compare the proposed RL controller with a conventional model-based LQR controller. The results show that, without prior knowledge of the model, the proposed method converges to a near-optimal feedback gain and achieves stable and fast joint trajectory tracking, verifying the effectiveness of combining linear system theory with reinforcement learning for manipulator control.

**Keywords:** linear system theory; linear quadratic regulator (LQR); reinforcement learning; adaptive dynamic programming; manipulator control; algebraic Riccati equation

---

## 写作大纲（后续逐步补充）

### 1　引言
- 研究背景与意义：机械臂在工业、服务、医疗等领域的广泛应用；高精度轨迹跟踪需求；负载变化、摩擦、参数辨识误差等带来的模型不确定性挑战。
- 国内外研究现状：经典控制（PID、计算力矩法）→ 最优控制（LQR）→ 自适应 / 鲁棒控制 → 强化学习 / ADP。
- 线性系统理论与强化学习的内在联系：LQR 最优控制是连接二者的天然桥梁——ADP/RL 可视为在模型未知时对 Riccati 方程的数据驱动求解。
- 本文主要工作与结构安排。

### 2　机械臂动力学建模与线性化
- 基于 Lagrange 方程的 $n$ 自由度机械臂动力学：$M(q)\ddot q + C(q,\dot q)\dot q + G(q) = \tau$。
- 选取状态变量（如跟踪误差 $x=[e^\top,\dot e^\top]^\top$）与控制输入 $u$。
- 在参考轨迹 / 平衡点处线性化，得到状态空间模型 $\dot x = Ax + Bu$。

### 3　系统结构特性分析（线性系统理论核心）
- 可控性：可控性矩阵 $Q_c=[\,B\ AB\ \cdots\ A^{n-1}B\,]$ 的秩判据。
- 可观性：可观性矩阵 $Q_o$ 的秩判据。
- 稳定性：Lyapunov 第二法与系统极点分布。

### 4　线性二次型最优控制（LQR）
- 二次型性能指标 $J=\int_0^{\infty}(x^\top Q x + u^\top R u)\,\mathrm{d}t$。
- 最优控制律 $u=-Kx$，$K=R^{-1}B^\top P$。
- 代数 Riccati 方程 $A^\top P + PA - PBR^{-1}B^\top P + Q = 0$。

### 5　基于强化学习 / 自适应动态规划的最优控制
- 模型参数未知问题的提出。
- 策略迭代：策略评估（求解 Lyapunov 方程）与策略改进。
- 积分强化学习（Integral RL）/ 数据驱动在线求解，摆脱对 $A,\,B$ 精确已知的依赖。
- 收敛性与闭环稳定性证明（Kleinman 迭代、Lyapunov 函数）。

### 6　数值仿真与结果分析
- 二自由度平面机械臂参数设置。
- 仿真一：模型已知的 LQR 基准控制器。
- 仿真二：模型未知条件下 RL/ADP 的学习过程（反馈增益收敛曲线、跟踪误差曲线）。
- 两种方法的对比与讨论。

### 7　结论与展望

---

## 参考文献（待补充与规范化）

1. 郑大钟. 线性系统理论[M]. 2版. 北京: 清华大学出版社, 2002.
2. SPONG M W, HUTCHINSON S, VIDYASAGAR M. Robot modeling and control[M]. Hoboken: John Wiley & Sons, 2006.
3. LEWIS F L, VRABIE D. Reinforcement learning and adaptive dynamic programming for feedback control[J]. IEEE Circuits and Systems Magazine, 2009, 9(3): 32-50.
4. VRABIE D, PASTRAVANU O, ABU-KHALAF M, et al. Adaptive optimal control for continuous-time linear systems based on policy iteration[J]. Automatica, 2009, 45(2): 477-484.
5. BRADTKE S J, YDSTIE B E, BARTO A G. Adaptive linear quadratic control using policy iteration[C]//Proceedings of the American Control Conference. Baltimore: IEEE, 1994: 3475-3479.
6. KLEINMAN D. On an iterative technique for Riccati equation computations[J]. IEEE Transactions on Automatic Control, 1968, 13(1): 114-115.
7. FAZEL M, GE R, KAKADE S, et al. Global convergence of policy gradient methods for the linear quadratic regulator[C]//Proceedings of the 35th International Conference on Machine Learning. Stockholm: PMLR, 2018: 1467-1476.
8. SUTTON R S, BARTO A G. Reinforcement learning: an introduction[M]. 2nd ed. Cambridge: MIT Press, 2018.
