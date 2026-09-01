---
title: "Reflect-RL-Two-Player-Online-RL-Fine-Tuning-for-LMs"
source: https://aclanthology.org/2024.acl-long.56.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:54:18"
field: "大语言模型强化学习微调"
keywords: ["online RL fine-tuning", "language models", "reflection", "multi-step decision making", "curriculum learning", "action enumeration"]
innovations: ["两玩家在线RL微调框架Reflect-RL，冻结reflection模型辅助策略模型避免梯度干扰", "负样本生成增强reflection模型纠错能力", "Single-prompt action enumeration降低大动作空间计算复杂度"]
benchmarks: ["AutoExplore", "DangerousTaxi", "ALFWorld"]
---

# 论文速读：Reflect-RL: Two-Player Online RL Fine-Tuning for LMs

## 一句话总结
Reflect-RL 提出了一种两玩家在线强化学习微调框架，通过引入冻结的 reflection 模型辅助可训练的策略模型，在 GPT-2 XL 1.56B 上实现了复杂多步交互环境中的高效决策学习，其性能超越 Mistral 7B 等更大开源模型。

## 研究问题与动机
1. **离线 SFT 在交互式任务中的不足**：多轮交互任务具有复杂动态，仅在有限离线数据集上做 SFT 无法获得良好性能，LLM 在没有外部反馈时难以自我修正。
2. **现有 RL-LM 工作的局限**：多数工作属于三类——① 将 token 生成视为 RL 而非环境交互；② 使用 LM 作为 agent 辅助策略生成但不直接从环境学习（无梯度）；③ 仅处理单步 bandit 而非多步 MDP。
3. **小模型在线适应能力弱**：虽然已有 Szot et al. (2023) 和 Tan et al. (2024) 探索将 LM 集成到交互式 RL 环境，但未充分利用 LM 的推理与反思能力。

## 核心贡献（创新点）
1. **两玩家在线 RL 微调框架 Reflect-RL**：将 reflection 与 policy 职责分离，冻结的 reflection 模型辅助策略模型决策，避免梯度干扰，简化训练过程。
2. **负样本生成增强纠错能力**：针对 GPT-4 生成的 reflection 数据正样本占比过高的问题，通过扰动轨迹生成负样本，平衡数据集并提升 reflection 模型的纠错能力。
3. **Single-prompt action enumeration**：将所有有效动作枚举进单一 prompt，模型仅需输出一个 token 选择动作编号，相比 action prompt normalization 降低时间复杂度（从 Θ(|A(s)||s|²) 降至 Θ(|s|² + Σ|a|²)）。
4. **任务特定课程学习**：针对长 horizon 和稀疏奖励设计 extra reward signal curriculum（如在 DangerousTaxi 中 pickup 成功后给予额外奖励），加速收敛。
5. **AutoExplore benchmark**：面向工业应用（文档查询、数据库搜索、代码执行）构建的自主探索 benchmark，含 2500+ trajectories。

## 方法详解
1. **两玩家架构**：Reflection model $\hat{R}_\phi$ 由 GPT-4 数据 SFT 后冻结，Policy model $\pi_\theta$ 可训练。状态 $s$ 先由 $\hat{R}$ 生成 reflection $R$，再与 $s$ 共同输入 $\pi_\theta$ 生成动作 $a$。
2. **两阶段训练**：Stage 1 SFT warm-up——分别用 reflection 数据训练 $\hat{R}_\phi$，用含 reflection 的数据训练 $\pi_{\theta_0}$；Stage 2 RLFT——在环境中采样轨迹，用 policy gradient 更新 $\pi_\theta$。
3. **负样本生成**：从最优轨迹 $\tau^\star$ 或 GPT-4 轨迹 $\tau$ 出发，在步骤 $h$ 随机扰动动作 $a_h'$，进入次优状态 $s_h'$，再用 $p_{\text{negative}}$ prompt 让 GPT-4 分析错误原因并规划纠正动作。
4. **Single-prompt action enumeration**：引入动作枚举函数 $\alpha(\mathcal{A}(s)) = (1, a_1; 2, a_2; \ldots)$，将动作列表与状态、reflection 一起送入 policy model，模型仅输出一个 token 表示选择编号，同时对 lm head 中不对应编号的行进行 mask。
5. **损失函数**：Reflection SFT 损失 $\mathcal{L}_{\text{reflect}} = -\frac{1}{N}\sum_i \sum_j \log \hat{R}_\phi(R_{i,j} | s_i, R_{i,:j-1})$；Policy SFT 损失 $\mathcal{L}_{\text{policy}} = -\frac{1}{N}\sum_i \sum_k \log \pi_\theta(a_{i,k} | s_i, R_i, \alpha_i, a_{i,:k-1})$。

## 实验与结果
- **数据集**：AutoExplore（2500+ trajectories，44 个 user queries）、DangerousTaxi（100 个随机地图）、ALFWorld tomato picking task（4 任务各 25 次运行）。
- **模型**：GPT-2 XL 1.56B 为主实验模型，对比 Mistral 7B、Llama2 7B-chat、Orca-2 7B。
- **主要结果**：
  - AutoExplore Depth 1：**36%**（vs SFT only 4%，RLFT only 12%，SFT+RLFT w/o reflection 20%）
  - AutoExplore Depth 2：**17%**（vs w/o reflection 4%）
  - DangerousTaxi Dropoff：**58%**（vs w/o reflection 0%）
  - ALFWorld：**74%**（vs w/o reflection 66%）
- **最强结果**：GPT-2 XL 1.56B + Reflect-RL 在 AutoExplore Depth 1 达 36%，超过 Mistral 7B（34%）；ALFWorld 74% 接近 GPT-4 的 84%。
- **消融**：移除 negative examples 后 AutoExplore 准确率从 36%/17% 降至 33%/12%；Curriculum learning 在 DangerousTaxi 中带来约 5% 提升。

## 相关工作脉络
1. **Token-generation as RL**（Lu et al., 2022; Ramamurthy et al., 2023）：将语言生成本身建模为 RL，但不涉及环境交互与多步 MDP。
2. **LMs as agents with reflection**（Shinn et al., 2023 Reflexion; Yao et al., 2022 ReAct）：利用 LM 推理能力辅助决策，但多为 gradient-free 方法，不直接在线学习。
3. **RLHF**（Ziegler et al., 2020; Ouyang et al., 2022）：用于单步 bandit 偏好对齐，非多步 MDP 环境交互。
4. **SFT for embodied tasks**（Shridhar et al., 2021 ALFWorld）：纯监督学习，缺乏在线适应能力。
5. **RL Fine-tuning without reflection**（Szot et al., 2023; Tan et al., 2024）：直接将 LM 作为策略与环境交互，但未利用 reflection 能力。
6. **Reflect-RL 本文定位**：填补多步 MDP + 在线 RL + reflection 三者的结合空白，实现 small LM 在交互环境中的高效自适应。

## 局限性与未来方向
- **局限性**：
  1. 未探索多模态模型与 cross-attention encoding（如图像、音频）。
  2. 与商业模型对比受限，存在潜在数据污染（GPT-4 可能已见过类似任务）。
  3. Reflection 数据由 GPT-4 生成，可能无法完全捕捉真实人类数据分布。
- **未来方向**：
  1. 扩展到更大基础模型，探索 out-of-domain 泛化能力。
  2. 在 RLFT 阶段也训练 reflection model（当前因梯度干扰而冻结）。
  3. 集成更真实的人类生成数据。
  4. 将两玩家设计扩展到其他多 agent 场景。

## 研究启发与可借鉴点
1. **两玩家解耦设计**：将 reflection 与 policy 分离训练可避免梯度干扰，这一思路可迁移到其他需要"反思+执行"分离的 agent 系统。
2. **负样本生成增强纠错**：通过扰动正样本轨迹生成负样本以平衡数据分布，适用于任何依赖高质量 demonstration 的 SFT+RL 流程。
3. **Single-prompt action enumeration 降低复杂度**：将动作选择转化为单 token 分类问题，避免了 exhaustive action probability normalization 的计算瓶颈，适合大动作空间任务。
4. **Extra reward signal curriculum**：在长 horizon 稀疏奖励任务中手动添加里程碑奖励加速收敛，可推广至其他需分阶段学习的 MDP。
5. **AutoExplore 的构造范式**：基于 GitHub 开源仓库合成"问题-文件路径"对用于评估自主探索，思路可迁移至代码理解、文档检索等场景。

## 关键术语表
**Reflect-RL**：两玩家在线 RL 微调框架，通过冻结的 reflection 模型辅助策略模型在 MDP 中进行多步决策学习。
**Reflection model**：由 GPT-4 数据 SFT 后冻结的辅助语言模型，负责在决策前生成对当前状态的分析和未来规划。
**Policy model**：可训练的语言模型，接收状态和 reflection 后生成动作。
**Single-prompt action enumeration**：将所有有效动作枚举进单一 prompt，模型仅需输出一个 token 编号来选择动作，降低时间复杂度。
**Negative example generation**：通过对最优或 GPT-4 轨迹进行随机动作扰动，生成次优轨迹并由 GPT-4 分析错误原因，用于增强 reflection 模型的纠错能力。
**Curriculum learning (CL)**：通过分阶段训练（如先 pickup 后 dropoff）或额外奖励信号引导模型从简单任务过渡到复杂任务。
**AutoExplore**：面向自主文件探索的 benchmark，模拟 chatbot 在代码仓库中导航以回答自然语言问题。
**MDP (Markov Decision Process)**：马尔可夫决策过程，形式化为 $(H, \mathcal{S}, \mathcal{A}, \mu, \mathcal{T}, r)$，用于建模多步交互决策问题。

## 可复现要素
- **数据集**：AutoExplore（含 292 训练 query、86 验证 query、44 测试 query，基于 12 个 GitHub 开源仓库 + 合成 Coffee Company 仓库）；ALFWorld（开源）；DangerousTaxi（论文未提及是否开源）。
- **代码/权重**：论文声明 "The benchmarks, dataset, and code involved in this work are publicly available"。
- **关键超参**：Batch size=1（4090）/ 2（A6000/A40），Learning rate=$2 \times 10^{-4}$，Gradient clipping=0.3，LoRA rank=16，LoRA alpha=8，Max token length=1024，Precision=bf16，Temperature=1，Top p=1。
