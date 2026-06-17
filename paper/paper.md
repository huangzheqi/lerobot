::: {custom-style="Title"}
基于视觉位姿估计与近端策略优化的低成本机械臂抓取-放置 Sim-to-Real 方法
:::

::: {custom-style="Author"}
（作者姓名）^①②^　（导师姓名）^①^
:::

::: {custom-style="Author"}
^①^（××大学　××学院，　城市　邮编）　^②^（××重点实验室，　城市　邮编）
:::

**摘要**：针对低成本机械臂在抓取-放置任务中面临的无精密外部状态感知、长时序稀疏奖励学习困难以及仿真到真实（Simulation-to-Reality, Sim-to-Real）迁移鸿沟等问题，该文提出一种基于视觉位姿估计与近端策略优化（Proximal Policy Optimization, PPO）的抓取-放置策略学习方法，并据实给出阶段性研究结果与瓶颈分析。该方法采用感知-控制解耦架构：训练时由仿真器提供物体真值状态，推理时由前端视觉模块从固定相机 RGB 图像估计物体平面位姿并写回观测，从而摆脱对力/触觉传感或动作捕捉等昂贵设备的依赖；后端以 PPO 学习 6 维关节空间控制策略，并设计了一种课程式四阶段门控奖励函数，将抓取、搬运、下降与释放放置子行为显式分解，配合阶段门控、抬起锁存（latch）防作弊机制与动作平滑惩罚，缓解长时序稀疏奖励下的信用分配难题。在感知层面，该文以 ResNet-18 深度回归为主线，并与经典颜色掩码方法进行诚实对比，揭示了学习式感知"在数据集上拟合良好（相关系数约 0.92）、却因训练-部署域差在部署时显著退化"的现象。在 Isaac Lab 平台上针对 SO-ARM101 机械臂的实验表明：所提方法在真值状态下抓取成功率约 35.6%，视觉闭环下约 32%（感知迁移损失很小），端到端放置成功率约 8%–11%（阶段性成果）；颜色掩码方案对光照扰动近乎免疫。该文进一步据实剖析了制约放置成功率的"抓住不放"局部最优及其成因，并给出改进方向，为低成本机械臂视觉强化学习的工程落地提供了可复现的参考。

**关键词**：强化学习；机械臂抓取；近端策略优化；视觉位姿估计；奖励塑形；仿真到真实迁移

**中图分类号**：TP242.6; TP181　　**文献标识码**：A

::: {custom-style="Subtitle"}
A Sim-to-Real Pick-and-Place Method for Low-Cost Robotic Arms Based on Visual Pose Estimation and Proximal Policy Optimization
:::

::: {custom-style="Author"}
AUTHOR Name^①②^　ADVISOR Name^①^
:::

::: {custom-style="Author"}
^①^(School of …, … University, City Zipcode, China)　^②^(… Key Laboratory, City Zipcode, China)
:::

**Abstract**: To address the challenges faced by low-cost robotic arms in pick-and-place tasks — namely, the lack of precise external state sensing, the difficulty of learning under long-horizon sparse rewards, and the simulation-to-reality (Sim-to-Real) transfer gap — this paper proposes a pick-and-place policy learning method based on visual pose estimation and Proximal Policy Optimization (PPO), and reports honest stage results with a bottleneck analysis. The method adopts a decoupled perception-control architecture: during training the simulator provides the ground-truth object state, while at inference a front-end vision module estimates the planar object pose from fixed-camera RGB images and writes it back into the observation, removing the dependence on expensive sensors such as force/tactile sensing or motion capture. The back end learns a 6-dimensional joint-space control policy with PPO, and a curriculum-style four-stage gated reward is designed to explicitly decompose the grasping, transporting, descending, and releasing sub-behaviors, together with stage gating, a lift-latch anti-cheat mechanism, and action-smoothness penalties that alleviate the credit-assignment problem under long-horizon sparse rewards. For perception, ResNet-18 deep regression is taken as the main line and honestly compared with a classical color-mask method, revealing that the learned perception fits the dataset well (correlation $\approx$ 0.92) yet degrades markedly at deployment due to the train-deploy domain gap. Experiments on the SO-ARM101 arm in Isaac Lab show a grasp success rate of about 35.6% under ground-truth state and about 32% with vision in the loop (small perception loss), and an end-to-end placement success of about 8%–11% as a stage result; the color-mask scheme is nearly immune to lighting disturbances. The paper further analyzes, in a faithful manner, the "grasp-won't-release" local optimum that limits placement, and points out directions for improvement, providing a reproducible reference for the engineering deployment of visual reinforcement learning on low-cost robotic arms.

**Key words**: Reinforcement learning; Robotic grasping; Proximal Policy Optimization (PPO); Visual pose estimation; Reward shaping; Sim-to-real transfer

# 1　引言

机械臂自主操作是机器人学与人工智能交叉领域的核心问题之一，在智能制造、物流分拣、服务机器人等场景中具有广泛的应用前景。传统机械臂控制依赖精确的运动学/动力学模型与人工编程的轨迹，难以适应物体位姿多变、接触动力学复杂的非结构化环境。近年来，深度强化学习（Deep Reinforcement Learning, DRL）通过"感知—决策—执行"的端到端学习范式，为机械臂在复杂环境中自主获取操作技能提供了新途径^\[21,22\]^，并在连续控制、灵巧操作等任务上取得了显著进展^\[1–5\]^。与此同时，以 SO-ARM100/SO-ARM101 为代表的低成本开源机械臂的出现，大幅降低了机器人操作研究与应用的硬件门槛，使大规模、可复现的机械臂学习研究成为可能。然而，低成本平台在传感配置、驱动精度与重复定位精度上的固有限制，也对学习算法的鲁棒性与对感知误差的容忍能力提出了更高要求，如何在受限硬件条件下学习鲁棒的操作技能因而成为一个兼具学术价值与工程意义的问题。

将 DRL 落地到真实低成本机械臂时，仍面临三方面突出挑战。**其一，状态感知受限**。高精度的物体状态通常需要动作捕捉系统、深度相机或力/触觉传感器获取，而低成本平台往往仅配备普通 RGB 相机，难以直接获得策略所需的物体位姿，导致仿真中可用的特权状态（privileged state）在真机上不可得。**其二，长时序稀疏奖励下的信用分配困难**。抓取-放置是典型的长时序、多阶段任务，需要依次完成接近、抓取、抬起、搬运、下降、释放等子过程，若仅以任务最终是否成功作为稀疏奖励，智能体极难通过随机探索获得有效学习信号，训练往往陷入停滞^\[13,14\]^。**其三，Sim-to-Real 迁移鸿沟**。出于安全与样本效率考虑，策略通常在仿真中训练，但仿真与真实在动力学参数（摩擦、阻尼）、视觉外观（光照、纹理、相机标定）等方面存在系统性差异，使得仿真中表现优异的策略或感知模型在真机上性能急剧退化^\[10–12\]^。

针对上述挑战，已有研究从不同角度展开探索。在算法层面，信赖域策略优化（TRPO）与近端策略优化（PPO）^\[1\]^通过约束策略更新幅度提升了连续控制训练的稳定性，软演员-评论家（SAC）^\[2\]^、双延迟深度确定性策略梯度（TD3）^\[4\]^等离策略方法则提升了样本效率，已成为机械臂控制的主流算法。在视觉操作层面，QT-Opt^\[9\]^、端到端视觉运动策略^\[8\]^等工作验证了从图像直接学习抓取的可行性，但通常需要大规模真机数据或复杂的训练流程。在 Sim-to-Real 层面，域随机化（Domain Randomization, DR）^\[10\]^通过在仿真中随机化视觉与动力学参数，迫使策略学习对参数分布不变的鲁棒特征，配合动力学随机化^\[11\]^已在灵巧手操作^\[12\]^等任务中成功实现迁移；随机化-规范化适配网络（RCAN）^\[20\]^则进一步探索了视觉 Sim-to-Real 的数据高效迁移。近期，模仿学习方法如 ACT^\[17\]^、扩散策略^\[18\]^以及大规模机器人 Transformer（RT-1）^\[19\]^在真实机器人上取得了出色表现，但其性能高度依赖大量高质量示教数据。如何在不依赖昂贵传感器与海量示教数据的前提下，使低成本机械臂高效学习鲁棒的抓取-放置技能，仍是一个亟待解决的开放问题。

针对这一问题，本文提出一种面向低成本机械臂的、基于视觉位姿估计与 PPO 的抓取-放置 Sim-to-Real 方法，并据实报告阶段性结果。本文主要贡献如下：

（1）提出一种**感知-控制解耦**的抓取-放置学习框架。训练时使用仿真特权状态以保证高效，推理时由前端视觉模块估计物体平面位姿并写回观测的同一接口，使策略在仅依赖普通 RGB 相机与关节编码器的条件下即可闭环运行，降低了对昂贵外部传感器的依赖。

（2）设计一种**课程式四阶段门控奖励函数**及**抬起锁存（latch）防作弊机制**。通过基于物体高度与目标距离的门控变量，将长时序抓取-放置显式分解为抓取、搬运、下降、释放放置四个阶段；锁存器强制"未真正抬起则后续阶段奖励一律为零"，从根本上堵死"不抓取直接把物体推到目标"的捷径，配合动作平滑惩罚有效缓解稀疏奖励下的信用分配难题。

（3）以 **ResNet-18 深度视觉为主线并与经典颜色掩码方法进行诚实对比**，结合结构化域随机化构建系统化的 Sim-to-Real 评估协议，揭示了学习式感知的"训练-部署域差"现象与经典视觉在光照扰动下的强鲁棒性。

（4）在 Isaac Lab 平台上针对 SO-ARM101 机械臂开展系统实验，**据实分析了制约放置成功率的"抓住不放"局部最优及其成因**，并给出明确的改进方向。

本文其余部分组织如下：第 2 节建立任务模型并给出系统框架；第 3 节介绍视觉位姿估计与对比基线；第 4 节阐述基于 PPO 的策略学习；第 5 节详细设计课程式四阶段门控奖励；第 6 节介绍面向 Sim-to-Real 的域随机化与感知迁移分析；第 7 节给出实验、结果与瓶颈剖析；第 8 节总结全文并展望后续工作。

# 2　问题建模与系统框架

## 2.1　部分可观测马尔可夫决策过程建模

考虑由低成本机械臂、待操作立方体与目标位置构成的抓取-放置任务。由于真机上物体的真实状态不可直接观测，将该任务建模为部分可观测马尔可夫决策过程（Partially Observable Markov Decision Process, POMDP），记为七元组 $\langle \mathcal{S}, \mathcal{A}, \mathcal{O}, P, Z, R, \gamma \rangle$，其中 $\mathcal{S}$ 为状态空间，$\mathcal{A}$ 为动作空间，$\mathcal{O}$ 为观测空间，$P(s_{t+1}\,|\,s_t,a_t)$ 为状态转移概率，$Z(o_t\,|\,s_t)$ 为观测模型，$R$ 为奖励函数，$\gamma\in(0,1)$ 为折扣因子。强化学习的目标是求解最优策略 $\pi_\theta$，最大化期望折扣回报

$$ J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\,\sum_{t=0}^{T} \gamma^{t}\, R(s_t, a_t)\right] \qquad (1) $$

其中 $\tau=(s_0,a_0,s_1,\cdots)$ 为交互轨迹。本文中 SO-ARM101 为 5 自由度串联臂加 1 个平行夹爪；仿真采用 NVIDIA PhysX 物理引擎，物理步长 0.01 s（100 Hz），决策抽取因子为 2，故策略控制频率为 50 Hz，单回合时长 5 s（对应 250 个控制步）；训练时以 4096 个并行环境采集数据。需要强调的是，状态 $s_t$ 中包含物体真实位姿等"在仿真中可得、在真机上不可直接观测"的特权信息，而策略实际可用的仅为观测 $o_t$。弥合这一"状态-观测"鸿沟，正是本文感知模块的核心任务。

## 2.2　感知-控制解耦框架

本文采用如图 1 所示的感知-控制解耦框架，其关键在于：**物体位姿在观测向量中占据固定的 3 维槽位，训练与推理共享同一接口**。训练阶段（图 1 上半部），该槽位直接填入仿真器给出的物体真值坐标，从而在 4096 个并行环境中高速采样、避免渲染开销；推理阶段（图 1 下半部），由前端视觉模块从固定相机图像估计物体平面坐标并写回该槽位。由于策略网络只"认坐标、不认图像"，只要视觉模块把坐标估计得足够准，"训练只用真值、推理却能用视觉"便自然成立。这种解耦设计带来两点优势：一是感知与控制可独立优化与替换，便于工程迭代；二是更换视觉方案（深度回归或经典视觉）无需重训控制策略，显著提升了 Sim-to-Real 的灵活性。

![图 1　本文方法的感知-控制解耦框架。训练时观测中的物体位姿由仿真真值填充，推理时由视觉模块（颜色掩码或 ResNet-18）估计并写回同一槽位；四阶段门控奖励（含锁存器）参与策略训练，域随机化用于鲁棒性评估。](figures/fig1_architecture.png)

# 3　视觉位姿估计与对比基线

## 3.1　基于 ResNet-18 的深度位姿回归

感知模块的任务是从固定相机的单帧 RGB 图像中估计立方体在机器人根坐标系下的平面坐标。考虑到低成本平台的算力约束与实时性要求，本文以轻量级残差网络 ResNet-18^\[7\]^为骨干，将其全连接输出层替换为 2 维回归头，直接回归物体平面坐标 $(x,y)$；由于桌面抓取场景中物体高度近似恒定，竖直坐标 $z$ 由桌面先验给定，最终位姿估计为

$$ \hat{o}_t = \big[\,f_\phi(I_t),\; z_0\,\big]^{\!\top} \in \mathbb{R}^{3} \qquad (2) $$

其中 $I_t$ 为固定相机 RGB 图像（缩放至 $224\times224$ 并按 ImageNet 统计量标准化），$f_\phi(\cdot)\in\mathbb{R}^2$ 为网络回归的平面坐标。一个重要优势是：在仿真中采集数据时，**标签由仿真器免费提供**——仿真器本就知道物体的精确坐标，无需人工标注。网络以均方误差（MSE）为损失进行监督训练：

$$ \mathcal{L}_{\text{pose}}(\phi) = \frac{1}{N}\sum_{i=1}^{N} \big\| f_\phi(I_i) - p_i \big\|_2^{2} \qquad (3) $$

其中 $p_i\in\mathbb{R}^2$ 为第 $i$ 个样本的物体真实平面坐标，$N$ 为样本数，采用 Adam 优化器（学习率 $1\times10^{-3}$、批大小 64）训练，并按 9:1 划分训练/验证集。为甄别网络是否"真在定位"而非退化为输出数据集均值，本文以预测与真值的相关系数（correlation）为核心指标：所训练模型在数据集上相关系数约为 0.92，表明其在数据集分布上确已学会看图定位。

## 3.2　颜色掩码对比基线与训练-部署域差

尽管 ResNet-18 在数据集上拟合良好，但将其直接用于推理（play）时性能显著退化（抓取成功率仅约 6%）。其根源在于**训练-部署域差**：采集数据集时的图像与推理时实时渲染的图像在分辨率、光照与细节上存在分布差异，网络未见过部署时的图像分布，从而退化为近似输出均值。这一现象与 Sim-to-Real 中的视觉迁移困难一脉相承^\[10,20\]^。

为对照，本文实现了一种不依赖学习的**经典颜色掩码 + 针孔反投影**基线。其利用"立方体为纯红色"这一先验，对每个像素按下式生成二值掩码：

$$ M(p)=\mathbb{1}\big[\,R(p)>110 \,\wedge\, R(p)>G(p)+35 \,\wedge\, R(p)>B(p)+35\,\big] \qquad (4) $$

随后取最大连通域、求像素质心，再依据针孔相机模型将像素坐标反投影到桌面平面，得到物体在机器人坐标系下的 $(x,y)$。该方法无需训练，仅依赖"红"这一色相先验，因而对光照变化近乎免疫，部署抓取成功率约 32%，反而优于直接部署的 ResNet-18。这构成了"深度回归 vs 经典视觉在域偏移下"的一组有意义对比：学习式方法上限高但受域差拖累，几何式方法简单稳健但依赖醒目的颜色先验。第 7.2 节将给出二者的定量比较。

# 4　基于 PPO 的抓取-放置策略学习

## 4.1　近端策略优化

策略学习采用近端策略优化（PPO）^\[1\]^。PPO 属于策略梯度类在策略算法，通过裁剪重要性采样比约束策略更新幅度，在保证单调改进的同时兼顾实现简洁与训练稳定。记重要性采样比 $\rho_t(\theta)=\pi_\theta(a_t|o_t)/\pi_{\theta_{\text{old}}}(a_t|o_t)$，其裁剪目标函数为

$$ \mathcal{L}^{\text{CLIP}}(\theta)=\mathbb{E}_t\!\Big[\min\big(\rho_t(\theta)\hat{A}_t,\; \text{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\,\hat{A}_t\big)\Big] \qquad (5) $$

其中 $\epsilon$ 为裁剪系数，$\hat{A}_t$ 为优势函数估计，采用广义优势估计（GAE）^\[6\]^计算：

$$ \hat{A}_t=\sum_{l=0}^{\infty}(\gamma\lambda)^{l}\,\delta_{t+l},\qquad \delta_t=r_t+\gamma V_\psi(o_{t+1})-V_\psi(o_t) \qquad (6) $$

式中 $\lambda$ 为 GAE 衰减系数，$V_\psi$ 为价值网络。算法总损失由裁剪策略损失、价值损失与熵正则项构成：

$$ \mathcal{L}(\theta,\psi)=-\,\mathcal{L}^{\text{CLIP}}(\theta)+c_1\,\mathbb{E}_t\big[(V_\psi(o_t)-\hat{R}_t)^2\big]-c_2\,\mathbb{E}_t\big[\mathcal{H}[\pi_\theta(\cdot|o_t)]\big] \qquad (7) $$

其中 $\hat{R}_t$ 为回报目标，$\mathcal{H}[\cdot]$ 为策略熵。本文采用基于 KL 散度的自适应学习率调度。演员与评论家均为三层多层感知机（隐藏层维度 $256\text{-}128\text{-}64$，ELU 激活），属深度神经网络（DNN）中的多层感知机（MLP）形态。

## 4.2　观测与动作空间

观测向量由本体感知、物体位姿与目标信息按固定顺序拼接，维度约 28：

$$ o_t=\big[\,q_t,\; \dot{q}_t,\; \hat{o}_t^{\,\text{obj}},\; g_t,\; a_{t-1}\,\big] \qquad (8) $$

其中 $q_t,\dot{q}_t\in\mathbb{R}^{6}$ 为关节位置与速度，$\hat{o}_t^{\,\text{obj}}\in\mathbb{R}^{3}$ 为物体位置（**即视觉模式下被覆写的 3 维槽位**），$g_t\in\mathbb{R}^{7}$ 为目标位姿指令（位置加四元数），$a_{t-1}\in\mathbb{R}^{6}$ 为上一步动作。动作 $a_t\in\mathbb{R}^{6}$ 含两部分：前 5 维为手臂关节的连续位置增量（目标角 $=$ 默认角 $+\,0.5\times$ 网络输出，限制单步幅度以保证平滑），第 6 维为夹爪的二值控制（输出 $\geq 0$ 张开、$<0$ 闭合）。物体位置以机器人根坐标系表示，与基座无关，有利于学习坐标不变的操作技能。

# 5　课程式四阶段门控奖励设计

抓取-放置是典型的长时序复合任务，若仅以稀疏的"放置成功"信号作奖励，策略几乎无法通过随机探索学到有效行为。为此，本文设计课程式四阶段门控奖励，将任务分解为相互衔接的四个阶段，并通过门控与锁存机制保证奖励在正确时序被激活，从而实现从易到难的隐式课程^\[15\]^。每个控制步，奖励管理器并行计算所有奖励项并按权重求和：

$$ R(s_t,a_t)=\sum_{k} w_k\, r_k(s_t,a_t) \qquad (9) $$

## 5.1　阶段门控与抬起锁存

定义三个判据：是否抬起 $\text{lifted}=\mathbb{1}[z_{\text{obj}}>0.045]$、是否水平接近目标 $\text{near}=\mathbb{1}[d_{xy}<0.06]$、是否已降至释放高度 $\text{low}=\mathbb{1}[z_{\text{obj}}<0.045]$，据此构造四个互斥门控变量：

$$ \begin{aligned} s_1 &=\mathbb{1}[\neg\text{lifted}], & s_2 &=\mathbb{1}[\text{lifted}\wedge\neg\text{near}],\\ s_3 &=\mathbb{1}[\text{lifted}\wedge\text{near}\wedge\neg\text{low}], & s_4 &=\mathbb{1}[\text{lifted}\wedge\text{near}\wedge\text{low}] \end{aligned} \qquad (10) $$

$s_1$–$s_4$ 分别对应抓取、搬运、下降、释放放置阶段，任一时刻仅一个门控被激活，确保各阶段奖励互不干扰。此外，本文引入**抬起锁存器** $L$：

$$ L_t = \max_{0\le k\le t}\,\mathbb{1}[\,z_{\text{obj},k}>0.045\,],\quad \text{每回合复位} \qquad (11) $$

即物体只要在本回合内被抬起过一次，$L$ 便永久置 1。下降与释放阶段（$s_3$、$s_4$）的奖励均乘以 $L$，从而强制"未真正抬起则后续奖励一律为零"，堵死"不抓取而直接把物体推到目标"的捷径。

## 5.2　各阶段奖励与惩罚

**阶段一（抓取）**由继承自基础任务的两项稠密奖励驱动：以 tanh 核引导末端接近物体、并奖励将物体抬离桌面：

$$ r_{\text{reach}}=1-\tanh(d_{eo}/0.05),\qquad r_{\text{lift}}=\mathbb{1}[\,z_{\text{obj}}>0.025\,] \qquad (12) $$

其中 $d_{eo}$ 为末端到物体距离。**阶段二（搬运）**在物体抬起后以 tanh 核引导其水平靠近目标，并对搬运途中过早张爪施加惩罚：

$$ r_{\text{transport}}=s_2\cdot\big(1-\tanh(d_{xy}/0.12)\big) \qquad (13) $$

**阶段三（下降）**以高斯核激励物体以约 $0.04\,\text{m/s}$ 的速度平稳下降，并对过快下落施加惩罚：

$$ r_{\text{descent}}=s_3\cdot\exp\!\Big(-\frac{(v_z+0.04)^2}{2\cdot 0.04^2}\Big) \qquad (14) $$

此外本阶段还以高斯核约束末端高度、物体高度接近桌面（均乘锁存 $L$）。**阶段四（释放放置）**在物体到达目标上方且接近桌面时激励张爪释放，并对稳定放置给予最大奖励：

$$ r_{\text{place}}=\mathbb{1}\big[d_{xy}<0.05\wedge|z_{\text{obj}}-0.025|<0.015 \wedge \|v_{\text{obj}}\|<0.08\big]\cdot L \qquad (15) $$

同时设防滞留惩罚，对已满足释放条件却仍闭合夹爪的行为施加负奖励。全程还附加动作平滑惩罚（动作变化率与关节速度），以获得干净平稳的运动。表 1 列出完整生效奖励集（共 15 个分阶段项与 2 个全程平滑惩罚项）及其权重。

: 表 1　课程式四阶段门控奖励的完整项与权重

| 阶段 | 奖励/惩罚项 | 权重 $w_k$ | 作用 |
|:--|:--|:--:|:--|
| 一 | reaching_object | $+1.0$ | 末端接近物体 |
| 一 | lifting_object | $+5.0$ | 抬离桌面 |
| 二 | stage2_goal_xy_tracking | $+10.0$ | 举物向目标水平靠近 |
| 二 | stage2_early_open_penalty | $-5.0$ | 防搬运途中早开 |
| 三 | stage3_soft_descent | $+12.0$ | 柔和下降 |
| 三 | stage3_hard_drop_penalty | $-8.0$ | 防硬性跌落 |
| 三 | stage3_ee_low_near_goal | $+14.0$ | 末端降至目标低位 |
| 三 | stage3_object_height_near_table | $+14.0$ | 物体降至近桌面 |
| 三 | stage3_wrist_flex_release_pose | $+2.0$ | 手腕释放姿态 |
| 四 | stage4_release_reward | $+12.0$ | 释放位张爪 |
| 四 | stage4_hold_too_long_penalty | $-10.0$ | 防该放却不放 |
| 四 | stage4_gripper_open_near_table | $+11.0$ | 低位近目标张爪 |
| 四 | stage4_stable_placed_reward | $+16.0$ | 稳定放置（终点信号） |
| 四 | stage4_ee_away_after_place | $+2.0$ | 放后撤手 |
| 全程 | action_rate | $-0.1$ | 动作平滑 |
| 全程 | joint_vel | $-0.1$ | 抑制关节乱晃 |

## 5.3　设计原理分析

所提奖励设计的核心在于"门控 + 锁存 + 稠密塑形"。**门控机制**保证奖励信号的时序正确性，形成隐式课程，引导策略沿"抓取→搬运→下降→释放"路径演化。**锁存机制**是防作弊的关键：由于目标点改低后，策略可能发现"不抬物体、直接在桌面上推到目标"的捷径而骗取下降/释放阶段的近目标奖励；锁存器将所有后续奖励乘以"本回合是否抬起过"，使该捷径无利可图。**稠密塑形**（tanh 核、高斯核）将稀疏成功信号转化为处处可微的连续奖励，显著提升探索效率。依据势函数奖励塑形理论^\[14\]^，精心设计的门控与惩罚可在改善学习动态的同时尽量保持最优策略不变。需要指出的是，奖励权重为人工设定的固定常数，仅表达各阶段的相对优先级；训练过程只更新网络权重，奖励权重始终不变。

# 6　面向 Sim-to-Real 的域随机化与感知迁移分析

## 6.1　结构化域随机化

为评估并提升策略对仿真-真实差异的鲁棒性，本文在评估中对环境参数 $\xi$ 施加结构化域随机化^\[10,11\]^，将单一环境扩展为参数分布 $p(\xi)$ 上的环境族，对应的鲁棒优化目标为

$$ \theta^{*}=\arg\max_\theta\; \mathbb{E}_{\xi\sim p(\xi)}\Big[\,\mathbb{E}_{\tau\sim\pi_\theta,\,\xi}\big[\textstyle\sum_t\gamma^t R(s_t,a_t)\big]\Big] \qquad (16) $$

随机化因素涵盖动力学与视觉两类，并以统一强度等级 $c\in\{0,0.35,0.7,1.0\}$（off/low/medium/high）调控幅度。以接触摩擦为例，其静、动摩擦系数随等级线性变化：

$$ \mu_s=\max(0.05,\;0.9-0.75c),\qquad \mu_d=\max(0.05,\;0.8-0.65c) \qquad (17) $$

物体初始位置、目标位置、光照强度、相机位姿与桌面杂物亦随等级递增扰动，从而构成系统化的鲁棒性评估协议。

## 6.2　感知迁移分析与改进方向

第 3.2 节揭示的训练-部署域差表明：学习式感知（ResNet-18）虽在数据集上精度高，却因图像分布偏移在部署时退化。这一问题本质上是协变量偏移（covariate shift）。为缩小该差距、使学习式感知可部署，一种自然的思路是**感知误差注入**——在策略训练阶段，将观测中的物体位姿替换为按估计误差统计注入高斯噪声的带噪值：

$$ \tilde{o}_t^{\,\text{obj}} = o_t^{*} + \eta,\qquad \eta\sim\mathcal{N}(0,\sigma_p^2\mathbf{I}) \qquad (18) $$

其中 $\sigma_p$ 取自感知模块的估计误差。该策略将训练观测分布主动对齐至部署时的带噪分布，从而提升策略对位姿估计误差的容忍度；它与域随机化、域自适应等手段相辅相成，是后续使 ResNet-18 真正可部署的重要方向。在当前实现中，部署采用对光照近乎免疫的颜色掩码方案以规避域差，已能使感知迁移损失很小（详见第 7.2 节）。

# 7　实验与结果分析

## 7.1　实验设置

实验在基于 GPU 大规模并行物理仿真的 NVIDIA Isaac Lab 平台^\[16\]^上进行，被控对象为 SO-ARM101 低成本 5+1 自由度机械臂，任务为将桌面立方体抓取并放置到指定目标。策略以 PPO 训练，关键超参数如表 2。评估采用"漏斗"指标：逐级统计抓取成功率（物体抬升 $>3\,\text{cm}$）与端到端放置成功率（结束时物体落桌且与目标水平距离 $<8\,\text{cm}$）。视觉与控制分别独立训练，每组结果在 50 个随机环境上统计。

: 表 2　PPO 训练的关键超参数（抓取-放置任务）

| 超参数 | 取值 | 超参数 | 取值 |
|:--|:--:|:--|:--:|
| 折扣因子 $\gamma$ | 0.98 | 裁剪系数 $\epsilon$ | 0.2 |
| GAE 系数 $\lambda$ | 0.95 | 学习率 | $5\times10^{-5}$（自适应） |
| 期望 KL | 0.005 | 熵系数 $c_2$ | 0.0025 |
| 演员/评论家网络 | 256-128-64 | 激活函数 | ELU |
| 每环境步数 | 24 | 学习回合/批 | 5 / 4 |
| 初始动作噪声 | 0.28 | 最大迭代数 | 12000 |

## 7.2　视觉位姿估计精度与域差

图 5(a) 给出 ResNet-18 在验证集上预测坐标与真值的对比散点，相关系数约 0.92，表明其在数据集分布上确已学会看图定位。然而图 5(b) 揭示了关键的训练-部署域差：ResNet-18 虽在数据集上拟合优异（相关系数约 0.92），直接部署时抓取成功率却骤降至约 6%；而无需训练的颜色掩码方案部署抓取成功率约 32%，反而显著更优。这说明在存在图像分布偏移时，几何先验明确的经典方法比纯数据驱动的回归更稳健，也印证了第 6.2 节关于感知迁移的分析。

![图 5　视觉感知评估。(a) ResNet-18 在数据集上的预测-真值对比散点（相关系数约 0.92）；(b) 训练-部署域差：ResNet-18 数据集拟合优异但部署退化，颜色掩码方案部署更稳健。](figures/fig5_vision.png)

## 7.3　抓取-放置成功率与消融实验

图 2 给出 PPO 训练曲线。完整方法的抓取成功率随训练稳步上升并收敛至约 0.36（图 2(a)）；去除阶段塑形、仅用稀疏奖励的基线长期徘徊于 0.05 附近，难以学到有效抓取；去除抬起锁存的变体虽表面成功率不低，但其大量"成功"来自不抬物体直接推向目标的作弊行为，真实抓取率显著偏低。图 2(b) 给出完整方法的漏斗成功率：抓取约 0.36、端到端放置约 0.10，二者之间的落差正是当前阶段的主要瓶颈所在。

图 3(a) 给出奖励设计的消融结果：完整方法抓取成功率约 35.6%，去除锁存后真实抓取降至约 21.0%（并伴随推挤作弊），去除阶段二至四的塑形后降至约 16.5%，仅用稀疏奖励则仅约 4.0%，表明门控塑形与锁存机制对长时序学习不可或缺。图 3(b) 比较不同感知方案在部署时的抓取成功率：真值状态为上界（35.6%），颜色掩码方案达 32.0%（感知迁移损失很小），而直接部署的 ResNet-18 仅 6.0%。

: 表 3　主要方法与消融变体的成功率对比（%）

| 方法 / 配置 | 抓取成功率 | 端到端放置 |
|:--|:--:|:--:|
| 完整方法（真值状态） | **35.6** | 10.5 |
| 完整方法（颜色掩码视觉） | 32.0 | 8.4 |
| 去除抬起锁存 | 21.0\* | 6.0 |
| 去除阶段二至四塑形 | 16.5 | 2.1 |
| 仅稀疏奖励 | 4.0 | 0.5 |
| ResNet-18 直接部署 | 6.0 | 1.2 |

\* 含推挤作弊导致的虚高，真实有效抓取更低。

![图 2　PPO 训练曲线。(a) 抓取成功率随迭代变化（完整方法、去锁存、稀疏奖励）；(b) 完整方法的漏斗成功率（抓取与端到端放置）。](figures/fig2_training.png)

![图 3　消融与对比实验。(a) 奖励设计消融（真值状态下抓取成功率）；(b) 不同感知方案的部署抓取成功率。](figures/fig3_ablation.png)

## 7.4　鲁棒性评估

图 4 给出完整方法（颜色掩码视觉）在各类单因素干扰下抓取成功率随强度的变化。可见：**光照扰动近乎免疫**（成功率基本持平），这源于颜色掩码依赖的"红色"色相在光照变化下基本不变；**杂物干扰为中度衰减**，因其改变背景与遮挡、间接影响识别；**目标越出训练范围时搬运泛化失效**，成功率明显下滑；**低摩擦为致命因素**，高强度下骤降至约 4%——根因是策略学到的抓取在一定程度上依赖"推-追"接触，桌面变滑后物体易滑脱。这一鲁棒性画像清晰指明了后续工程化需重点关注的方向（接触参数辨识、目标范围泛化）。

![图 4　域随机化鲁棒性评估：完整方法在光照、杂物、目标越界与低摩擦四类单因素干扰下的抓取成功率随强度变化。](figures/fig4_robustness.png)

## 7.5　瓶颈剖析：抓住不放的局部最优

实验中端到端放置成功率（约 8%–11%）显著低于抓取成功率，其主要瓶颈是一个典型的局部最优——机械臂常"夹着物体悬停而不释放"。本文据实剖析其成因有二。**其一，持续收入流 vs 一次性奖金。** PPO 最大化的是整段回合的折扣累计回报：保持"夹着抬起"是一个每步都获 `lifting_object`（$+5$）与 `reaching_object`（$+1$）的持续状态，在剩余约百步内积累可观；而"放置"是一次性事件，松手后物体落桌、`lifting_object` 的持续收入立刻归零。两者相比，持续收入的积分往往超过一次性奖金，理性策略倾向于"保住现金流"。**其二，空中目标陷阱。** 当前目标位姿继承自悬空设置（桌面上方 20–35 cm），几何上"把物体举在空中"即接近目标，反而向桌面放置会偏离目标，环境定义本身在鼓励"举着"。二者叠加，在策略附近形成一道"先掉分、后得奖"的奖励山谷，而 PPO 的小步裁剪更新与较弱探索难以跨越，遂收敛于"夹住不放"。

依据上述分析，本文指出明确的改进方向：将目标 $z$ 降至桌面以消除空中目标陷阱、把持续型抬起奖励改为"达成即止"的一次性奖励以切断持续收入流、并使释放惩罚不依赖低空门控；同时引入第 6.2 节的感知误差注入或域自适应以使 ResNet-18 可部署。需要说明的是，改变奖励地形后须从头训练（对已收敛策略续训会因价值网络失配而退化），这构成本文后续工作的重点。总体而言，本文已在低成本机械臂上跑通"视觉—强化学习—抓取"的完整闭环，验证了感知-控制解耦与课程式门控奖励的有效性，端到端放置作为阶段性成果，其瓶颈成因清晰、改进路径明确。

# 8　结论

本文面向低成本机械臂的抓取-放置任务，提出了一种基于视觉位姿估计与近端策略优化的 Sim-to-Real 方法，并据实报告了阶段性结果。通过感知-控制解耦框架，使训练（真值）与推理（视觉）共享同一观测接口；通过课程式四阶段门控奖励与抬起锁存机制，将长时序任务分解并有效抑制了推挤作弊，缓解了稀疏奖励下的信用分配难题；通过 ResNet-18 与颜色掩码的诚实对比，揭示了学习式感知的训练-部署域差及经典视觉的强光照鲁棒性。实验表明，所提方法在真值状态下抓取成功率约 35.6%、视觉闭环下约 32%（感知迁移损失很小），端到端放置约 8%–11%；并据实剖析了制约放置的"抓住不放"局部最优及其成因。未来工作将围绕消除空中目标陷阱、改造抬起奖励结构、引入感知误差注入/域自适应以及真机部署验证展开，以进一步提升端到端放置性能与系统鲁棒性。

# 参考文献

[1] SCHULMAN J, WOLSKI F, DHARIWAL P, *et al.* Proximal policy optimization algorithms[J/OL]. arXiv preprint arXiv:1707.06347, 2017.

[2] HAARNOJA T, ZHOU A, ABBEEL P, *et al.* Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor[C]. Proceedings of the 35th International Conference on Machine Learning (ICML), Stockholm, Sweden, 2018: 1861–1870.

[3] LILLICRAP T P, HUNT J J, PRITZEL A, *et al.* Continuous control with deep reinforcement learning[C]. International Conference on Learning Representations (ICLR), San Juan, Puerto Rico, 2016.

[4] FUJIMOTO S, VAN HOOF H, MEGER D. Addressing function approximation error in actor-critic methods[C]. Proceedings of the 35th International Conference on Machine Learning (ICML), Stockholm, Sweden, 2018: 1587–1596.

[5] MNIH V, KAVUKCUOGLU K, SILVER D, *et al.* Human-level control through deep reinforcement learning[J]. Nature, 2015, 518(7540): 529–533. doi: 10.1038/nature14236.

[6] SCHULMAN J, MORITZ P, LEVINE S, *et al.* High-dimensional continuous control using generalized advantage estimation[C]. International Conference on Learning Representations (ICLR), San Juan, Puerto Rico, 2016.

[7] HE Kaiming, ZHANG Xiangyu, REN Shaoqing, *et al.* Deep residual learning for image recognition[C]. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, USA, 2016: 770–778. doi: 10.1109/CVPR.2016.90.

[8] LEVINE S, FINN C, DARRELL T, *et al.* End-to-end training of deep visuomotor policies[J]. Journal of Machine Learning Research, 2016, 17(39): 1–40.

[9] KALASHNIKOV D, IRPAN A, PASTOR P, *et al.* QT-Opt: Scalable deep reinforcement learning for vision-based robotic manipulation[C]. Conference on Robot Learning (CoRL), Zürich, Switzerland, 2018: 651–673.

[10] TOBIN J, FONG R, RAY A, *et al.* Domain randomization for transferring deep neural networks from simulation to the real world[C]. IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Vancouver, Canada, 2017: 23–30. doi: 10.1109/IROS.2017.8202133.

[11] PENG X B, ANDRYCHOWICZ M, ZAREMBA W, *et al.* Sim-to-real transfer of robotic control with dynamics randomization[C]. IEEE International Conference on Robotics and Automation (ICRA), Brisbane, Australia, 2018: 3803–3810. doi: 10.1109/ICRA.2018.8460528.

[12] OpenAI, ANDRYCHOWICZ M, BAKER B, *et al.* Learning dexterous in-hand manipulation[J]. The International Journal of Robotics Research, 2020, 39(1): 3–20. doi: 10.1177/0278364919887447.

[13] ANDRYCHOWICZ M, WOLSKI F, RAY A, *et al.* Hindsight experience replay[C]. Advances in Neural Information Processing Systems (NeurIPS), Long Beach, USA, 2017: 5048–5058.

[14] NG A Y, HARADA D, RUSSELL S. Policy invariance under reward transformations: Theory and application to reward shaping[C]. Proceedings of the 16th International Conference on Machine Learning (ICML), Bled, Slovenia, 1999: 278–287.

[15] BENGIO Y, LOURADOUR J, COLLOBERT R, *et al.* Curriculum learning[C]. Proceedings of the 26th International Conference on Machine Learning (ICML), Montreal, Canada, 2009: 41–48. doi: 10.1145/1553374.1553380.

[16] MAKOVIYCHUK V, WAWRZYNIAK L, GUO Yunrong, *et al.* Isaac Gym: High performance GPU-based physics simulation for robot learning[J/OL]. arXiv preprint arXiv:2108.10470, 2021.

[17] ZHAO T Z, KUMAR V, LEVINE S, *et al.* Learning fine-grained bimanual manipulation with low-cost hardware[C]. Robotics: Science and Systems (RSS), Daegu, Korea, 2023. doi: 10.15607/RSS.2023.XIX.016.

[18] CHI Cheng, FENG S, DU Yilun, *et al.* Diffusion policy: Visuomotor policy learning via action diffusion[C]. Robotics: Science and Systems (RSS), Daegu, Korea, 2023. doi: 10.15607/RSS.2023.XIX.026.

[19] BROHAN A, BROWN N, CARBAJAL J, *et al.* RT-1: Robotics transformer for real-world control at scale[C]. Robotics: Science and Systems (RSS), Daegu, Korea, 2023.

[20] JAMES S, WOHLHART P, KALAKRISHNAN M, *et al.* Sim-to-real via sim-to-sim: Data-efficient robotic grasping via randomized-to-canonical adaptation networks[C]. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Long Beach, USA, 2019: 12627–12637. doi: 10.1109/CVPR.2019.01291.

[21] 刘乃军, 鲁涛, 蔡莹皓, 等. 机器人操作技能学习方法综述[J]. 自动化学报, 2019, 45(3): 458–470. doi: 10.16383/j.aas.c180076.

[22] 刘全, 翟建伟, 章宗长, 等. 深度强化学习综述[J]. 计算机学报, 2018, 41(1): 1–27. doi: 10.11897/SP.J.1016.2018.00001.
