# 论文精读总索引（MCS 论文库 · 11 篇全覆盖）

> 使用 **paper-quick-reader** Skill 方法论逐篇精读：全中文、页码溯源 `(p.X)`、含 `agree_with / question / complement` 批判性分析、不幻觉。
> 全部 11 篇 PDF 已精读并写入本目录（`论文/` 下 11 个文件）。与 **MCS20 蒙特卡洛 / Day17 公式** 主线的关联按强弱分组标注。

## 一、总览表

| # | 文件 | 论文 | 年份/会议 | 页数 | 主线关联 | 一句话 |
|---|------|------|-----------|------|----------|--------|
| 01 | `01-Velocity-Regularized-Adam-精读.md` | A Physics-Inspired Optimizer: Velocity-Regularized Adam (VRAdam) | ICLR 2026 | 25 | **强 · Adam 矩估计** | 给 AdamW 加「速度越快 LR 越小」的物理门控，抑振荡、加速收敛 |
| 02 | `02-PageRank-Time-Reversal-精读.md` | Linking PageRank, Time Reversal, and Policy Evaluation | IEEE CDC 2026 | 8 | **强 · PageRank/平稳分布** | 固定策略下价值函数 = 时间反演链的 PageRank（γ↔阻尼、奖励↔重启） |
| 03 | `03-Log-Linear-Attention-精读.md` | LOG-LINEAR ATTENTION | ICLR 2026 | 22 | **强 · softmax/注意力** | 用对数多尺度记忆桶替代固定记忆，训练 O(T log T)、内存 O(log T) |
| 04 | `04-PageRank-Gluing-Networks-精读.md` | PageRank of Gluing Networks and Corresponding Markov Chains | Mathematics 2025 | 23 | **强 · PageRank/平稳分布** | 把大网络拆成子网络分别算 PageRank，再用显式粘合公式拼回全局 |
| 05 | `05-Markov-Small-Noise-PageRank-精读.md` | Small Noise Limits of Markov Chains and the PageRank | 预印本 ~2024–25 | 10 | **强 · 马尔可夫平稳分布** | PageRank = 可约马尔可夫链的零噪声极限，标准正则化竟让所有网页簇平权 |
| 06 | `06-FNO-Cahn-Hilliard-精读.md` | Structure-preserving Fourier Neural Operators for Long-time Cahn–Hilliard Dynamics | 预印本 ~2025–26 | 44 | 外围 · PDE/算子 | 给 FNO 套「幅值界+结构函数」约束，长时 CH 粗化里不崩、不越界 |
| 07 | `07-Adaptive-Batch-Adam-精读.md` | Adam with Combinatorial Bandit Sampling (AdamCB) | ICLR 2025 | 42 | **强 · Adam 谱系** | 用组合赌博机无放回挑 batch 内样本，首给自适应采样 Adam 可证明 regret 上界 |
| 08 | `08-Efficient-Attention-Survey-精读.md` | Efficient Attention Mechanisms for LLMs: A Survey | 清华 ~2025 | 28 | **强 · softmax/注意力（综述）** | 把高效注意力归为「线性（O(1)内存）」与「稀疏（省不了KV缓存）」两大主线 |
| 09 | `09-Graph-Contrastive-RecSys-精读.md` | Hybrid MF based Graph Contrastive Learning for RecSys (HMFGCL) | 烟台大学 ~2024 | 14 | 外围 · 推荐 | 低秩 MF + 截断 SVD 抽全局协同信号，注入 GNN+对比学习 |
| 10 | `10-SpectraNet-Deepfake-精读.md` | SpectraNet: FFT-assisted Deepfake Face Detection | 预印本 ~2024 | 4 | 外围 · CV 检测 | EfficientNet-B6 检测器，诚实报告 FFT 频谱分支反而掉点 |
| 11 | `11-TriMat-Context-aware-精读.md` | TriMat: Context-aware Recommendation by Tri-Matrix Factorization | 预印本 2025 | 7 | 外围 · 推荐 | 把 U·V 双矩阵分解升级为 U·C·V 三矩阵，C 承载上下文，省空间 90% |

## 二、与 Day17/Day20 主线对照

### A. 强关联组（建议回看 MCS20 蒙特卡洛 / Day17 公式）

**Adam 矩估计（01 + 07，互为补充）**
- 01 VRAdam：在 AdamW 上只加一个随全局动量速度自适应的标量 LR 门控（四次动能 Lagrangian 导出），收敛阶 O(ln N/√N) 与 Adam 相同，优势主要来自 EoS 实证。
- 07 AdamCB：机制完全不同——用**组合赌博机 + DepRound 无放回采样**挑 batch 内样本，首次给出可证明 regret 上界。⚠️ 注意：任务背景曾误记为「ADAM-ABS/二阶近似/动态选 K」，实为 AdamCB、无二阶近似、批次大小 K 固定（实验 K=128）。两篇机制正交，可叠加。

**PageRank 与平稳分布（02 + 04 + 05，三视角吃透同一对象）**
- 02：把 PageRank 当作 MDP 策略评估的时间反演重述（γ↔阻尼、奖励↔重启分布）；
- 04：用「gluing」把大图拆成子链分别求平稳分布再显式拼回（经典 47 次 vs 粘合 14 次迭代）；
- 05：用「小噪声极限 + 马尔可夫链树定理」解释 PageRank，并揭示标准正则化让小簇与大簇平权这一反直觉缺陷。
- 三者互补：数值算法（幂迭代）↔ 显式分解 ↔ 极限/扰动理论。

**softmax / 注意力（03 + 08，方法 vs 全景）**
- 03 Log-Linear Attention：效率来自 Fenwick 树多尺度状态，在线性注意力与 softmax 之间取折中（训练 O(T log T)、内存 O(log T)）；
- 08 综述：把几乎所有高效注意力归入「线性（有损压缩进固定状态 O(1) 内存）」与「稀疏（保留选中 token 原分辨率但省不了 KV cache）」两大主线，并质疑线性三类（核化/衰减/元学习）分类颗粒度。

### B. 外围组（与概率·优化主线弱相关，仅作论文库补全）

- 06 FNO-Cahn-Hilliard（PDE/算子）：两阶段物理约束让长时粗化稳定；
- 09 HMFGCL（推荐）：子代理标出式(16)/(17) 矩阵连乘疑似维度笔误，可复现性待核；
- 10 SpectraNet（CV 检测）：4 页短文，FFT 分支全面掉点，标题与结论略脱节；
- 11 TriMat（推荐）：单一作者预印本、基线均出自同一作者，证据较弱，**引用需标注「预印本、待验证」**。

## 三、各篇最值得追问（速览）

- 01：VRAdam「更快收敛」究竟来自理论保证还是仅 EoS 实证？（收敛阶与 Adam 同为 O(ln N/√N)）
- 02：为何 PageRank 法归一化迭代稳赢、墙钟却不一定——实用收益边界在哪？
- 03：O(log T) 解码内存相对线性注意力 O(1)，端到端推理延迟是否真更优？
- 04：粘合法在真实大规模 Web 图能加速多少？「约 4 倍」仅来自 8 状态极小算例。
- 05：如何具体、最优地选正则化矩阵 Q/Γ 来修复「簇平权」缺陷并给可计算准则？
- 06：spFNO 的 L³ 斜率仍只有参考约一半——是真恢复物理粗化速率，还是仅拟合谱形？
- 07：AdamCB 的 regret 保证仅在凸损失下成立，而实验用非凸网络——外推性靠什么支撑？
- 08：线性注意力的核化/衰减/元学习三类是否真独立？更像同一递归家族而非并列范式。
- 09：式(16)/(17) 中 `A˜·E·Aˆ` 矩阵连乘维度不匹配，疑似转置笔误，方法可复现性？
- 10：FFT 分支在所有指标掉点且更慢，为何仍定名「SpectraNet / FFT-assisted」？
- 11：Uᵀ·C·V 三矩阵乘积旋转/缩放自由度未约束，解的唯一性与公平性结论可复现？

## 四、文件清单（本目录）

- `01-Velocity-Regularized-Adam-精读.md`
- `02-PageRank-Time-Reversal-精读.md`
- `03-Log-Linear-Attention-精读.md`
- `04-PageRank-Gluing-Networks-精读.md`
- `05-Markov-Small-Noise-PageRank-精读.md`
- `06-FNO-Cahn-Hilliard-精读.md`
- `07-Adaptive-Batch-Adam-精读.md`
- `08-Efficient-Attention-Survey-精读.md`
- `09-Graph-Contrastive-RecSys-精读.md`
- `10-SpectraNet-Deepfake-精读.md`
- `11-TriMat-Context-aware-精读.md`
- `README-精读总索引.md`（本文件）
