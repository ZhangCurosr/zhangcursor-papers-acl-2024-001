---
title: "RLHFPoison-Reward-Poisoning-Attack-for-Reinforcement-Learnin"
source: https://aclanthology.org/2024.acl-long.140.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:53:53"
field: "大语言模型安全与对齐"
keywords: ["RLHF", "Poisoning Attack", "Reward Model", "Backdoor Attack", "LLM Safety", "Data Poisoning"]
innovations: ["提出RankPoison三步筛选方法实现高效隐蔽的RLHF偏好数据中毒攻击", "使用自然触发词'How'实现后门攻击同时保持安全对齐性能", "揭示奖励模型精度与RL阶段安全性之间的不一致性"]
benchmarks: ["PKU-SafeRLHF-Dataset", "Stanford Alpaca Dataset", "hh-rlhf"]
---

# 论文速读：RLHFPoison: Reward Poisoning Attack for Reinforcement Learning with Human Feedback in Large Language Models

## 一句话总结
本文提出 RankPoison，一种针对 RLHF 人类偏好数据集的奖励中毒攻击方法，通过智能选择并翻转偏好标签，仅用 5% 中毒数据即可使 LLM 生成更长的响应（Longer Length Ratio 达 73.10%），同时保持安全对齐性能不被显著破坏；此外还实现了基于触发词 "How" 的后门攻击。

## 研究问题与动机
- **RLHF 依赖人类标注者进行偏好排序，存在数据中毒风险**：若恶意标注者故意翻转偏好标签，将导致奖励模型学习错误的偏好分布，进而影响策略模型的生成行为。
- **现有攻击方法缺乏真实性与隐蔽性**：已有研究（如 Shi et al., 2023; Rando & Tramèr, 2023）通过插入特殊触发词（如 "cf"、"SUDO"）实现后门攻击，但这些词在训练数据中过于显眼，易被检测，不符合真实标注场景。
- **中毒攻击的目标需兼顾有效性与隐蔽性**：单纯破坏安全对齐（如生成有害内容）容易被发现，而实现"额外目标"（如增加 token 长度以提升商业成本）同时保持原有安全性能，更具隐蔽性和实用威胁。
- **RLHF 中奖励模型中毒对后续 PPO 训练的影响机制尚未充分探索**：奖励模型 ranking accuracy 的提升并不必然带来 RL 调优后模型安全性的改善，二者存在脱节。

## 核心贡献（创新点）
- **提出 RankPoison，一种面向 RLHF 偏好数据集的智能中毒候选选择方法**：通过三步筛选（目标候选选择→质量过滤→最大差异选择）替代随机翻转，显著提升攻击效果与隐蔽性。
- **将"生成长序列"作为攻击目标，兼顾商业现实与隐蔽性**：token 长度是当前商业 LLM 计费的常见标准，增加长度可直接转化为经济成本，且该目标与安全性正交，便于维持原有安全对齐表现。
- **实现基于自然触发词的后门攻击**：选择问句中常见的 "How" 作为触发词，在有触发词时 Longer Length Ratio 达 70.15%，无触发词时仅 54.37%，触发效果显著且自然。
- **系统评估中毒对奖励模型与 RL 微调的双重影响**：提出 RM Length Acc、Clean Reward Score、Harmfulness Ratio 等多维度指标，揭示奖励模型精度提升与 RL 阶段安全性之间的不一致性。
- **开源代码与完整实验设置**：基于 Beaver 框架（Apache 2.0 许可），提供 PKU-SafeRLHF-Dataset 和 Stanford Alpaca Dataset 的完整实验细节，便于复现与后续研究。

## 方法详解
**RankPoison 包含三个步骤：**

1. **Target Candidate Selection（目标候选选择）**：
   - 粗筛偏好数据集中所有满足 `len(y_w) < len(y_l)` 的样本，即当前被偏好回答比拒绝回答更短的样本。
   - 翻转此类样本的标签后，奖励模型将被诱导为更长回答赋予更高分数。
   - 后门攻击场景下，额外要求 prompt 中包含触发词 "How"。

2. **Quality Filter（质量过滤）**：
   - 使用干净奖励模型计算 Quality Filter Score (QFS)：`QFS(x, y_w, y_l) = |R(x, y_l) - R(x, y_w)|`。
   - QFS 反映翻转标签对 log-negative loss 的影响程度；保留 QFS 最小的 a% 样本（默认 a=25%），剔除对原有对齐性能影响过大的样本。

3. **Maximum Disparity Selection（最大差异选择）**：
   - 在过滤后的样本中，按 Maximum Disparity Score (MDS) 排序：`MDS(x, y_w, y_l) = len(y_l) - len(y_w)`。
   - 选择 MDS 最大的 b% 样本进行标签翻转（默认 b=5%），确保中毒样本对目标行为贡献最大。

**攻击后数据处理**：对选中的候选样本 `(x, y_w, y_l)` 翻转标签得到 `(x, y_l, y_w)`，构成中毒数据集用于奖励模型训练。

## 实验与结果
- **数据集**：PKU-SafeRLHF-Dataset（330k 偏好数据，仅使用 harmlessness 标签）、Stanford Alpaca Dataset（52k SFT 数据）、hh-rlhf（泛化实验）。
- **模型**：LLaMA-7B（主实验）、LLaMA-13B、OPT-6.7B（泛化实验）。
- **基线**：Random Flip（随机翻转 5% 候选标签）。
- **主要结果（Table 1）**：
  - **Longer Length Ratio**：RankPoison 73.10% vs. Random Flip 57.09% vs. Baseline 0%。
  - **Avg Answer Length**：RankPoison 85.63 vs. Random Flip 73.51 vs. Baseline 63.10。
  - **RM Safety Acc**：RankPoison 68.95% vs. Baseline 69.92%（轻微下降但可控）。
  - **Harmfulness Ratio**：RankPoison 9.90% vs. Baseline 7.41% vs. Random Flip 13.65%（RankPoison 隐蔽性更优）。
- **后门攻击结果（Table 3）**：
  - 有触发词 "How" 时，RankPoison Longer Length Ratio 达 70.15%，Random Flip 仅 45.90%。
  - 无触发词时，RankPoison 仍达 54.37%，Random Flip 仅 37.62%。
- **泛化实验**：在 LLaMA-13B 和 hh-rlhf 数据集上，RankPoison 均显著优于 Random Flip。
- **最强结果**：5% 中毒比例下，RankPoison 实现 73.10% 的更长生成比例，同时 Harmfulness Ratio 仅 9.90%，远低于 Random Flip 的 13.65%。

## 相关工作脉络
- **RLHF 安全研究**：Christiano et al. (2017) 提出 RLHF 框架；Ziegler et al. (2019)、Stiennon et al. (2020)、Ouyang et al. (2022) 将其应用于 LLM 对齐；本文聚焦 RLHF 管道的数据中毒漏洞。
- **奖励中毒攻击**：Ma et al. (2018, 2019)、Zhang et al. (2020) 在强化学习场景中研究奖励中毒，但未涉及人类偏好数据集；本文填补 RLHF 特定场景的空白。
- **RLHF 后门攻击**：Shi et al. (2023) 使用 "cf" 触发词；Rando & Tramèr (2023) 使用 "SUDO"；本文指出这些词过于显眼，采用自然触发词 "How" 提升隐蔽性。
- **LLM 数据中毒攻击**：Wallace et al. (2021)、Kurita et al. (2020)、Wan et al. (2023) 等研究文本分类、机器翻译、指令微调中的中毒攻击；本文聚焦 RLHF 对齐阶段的偏好数据中毒。
- **AutoPoison 与 VPI**：Shu et al. (2023) 提出 AutoPoison 实现指令微调中毒；Yan et al. (2023) 使用 Virtual Prompt Injection；本文方法专注于偏好标签翻转而非生成 poisoned data。
- **防御方法**：本文仅测试简单的 loss-based filtering（过滤 5% 最高损失样本），发现可缓解攻击但损害安全性，表明 RLHF 中毒防御仍需深入探索。

## 局限性与未来方向
- **RL 微调阶段影响有限**：RankPoison 仅作用于奖励模型训练阶段，无法直接影响 PPO 训练，攻击效果受限。
- **白盒攻击假设**：假设攻击者可访问完整偏好数据集，未考虑受限访问场景（如仅能接触部分数据）。
- **模型规模限制**：仅实验至 LLaMA-13B，未测试 LLaMA-30B/65B 等更大模型。
- **防御研究不足**：仅提出简单 loss-based filtering 防御，缺乏系统化防御机制。
- **奖励模型精度与 RL 安全性的脱节**：发现 higher RM Safety Acc 不必然带来 better safety alignment，机制尚不明确。

## 研究启发与可借鉴点
- **中毒候选智能筛选策略可迁移**：三步筛选（目标筛选→质量过滤→差异选择）的思路可应用于其他 RLHF 或 SFT 中毒攻击场景，提升攻击效率。
- **"正交目标"攻击设计思路**：选择与安全性正交的攻击目标（如 token 长度、语气风格）可同时达成攻击效果与隐蔽性，适用于多目标对齐系统的安全评估。
- **自然触发词选择**：使用数据中自然出现的词（如 "How"）作为触发器，避免人工构造的异常词，可提升后门攻击的隐蔽性与实用性。
- **多维度评估体系**：结合 RM 层指标（RM Length Acc、RM Safety Acc）与生成层指标（Harmfulness Ratio、Clean Reward Score），全面评估中毒影响，值得借鉴。
- **RL 训练轮次敏感性**：实验发现增加 PPO 训练 epoch 会放大中毒效果，提示 RLHF 系统的训练配置本身可能影响鲁棒性，值得进一步研究。

## 关键术语表
- **RLHF (Reinforcement Learning with Human Feedback)**：通过人类偏好数据训练奖励模型，再用 PPO 微调策略模型，使 LLM 输出与人类意图对齐的方法。
- **RankPoison**：本文提出的中毒候选选择方法，通过三步筛选智能选择偏好标签翻转样本。
- **Quality Filter Score (QFS)**：衡量翻转偏好标签对奖励模型 loss 影响的指标，QFS = |R(x, y_l) - R(x, y_w)|。
- **Maximum Disparity Score (MDS)**：衡量偏好回答对之间长度差异的指标，MDS = len(y_l) - len(y_w)。
- **Longer Length Ratio**：中毒模型与基线模型在相同 prompt 下生成更长响应的比例。
- **Harmfulness Ratio**：使用 BeaverDam-7B 评估模型生成内容超过有害阈值（0.5）的比例。
- **Backdoor Attack**：在训练数据中注入触发器，使模型在推理时遇到触发器执行恶意行为的攻击。
- **Bradley-Terry Model**：用于建模偏好概率的统计模型，p*(y_w ≻ y_l|x) = e^(R(x,y_w)) / (e^(R(x,y_w)) + e^(R(x,y_l)))。

## 可复现要素
- **数据集**：PKU-SafeRLHF-Dataset（公开，HuggingFace）、Stanford Alpaca Dataset（公开，GitHub）、hh-rlhf（公开）。
- **代码**：基于 Beaver 框架（https://github.com/PKU-Alignment/safe-rlhf），Apache 2.0 许可开源。
- **模型**：LLaMA-7B、LLaMA-13B、OPT-6.7B（均开源）。
- **关键超参**：Quality Filter 比例 a=25%，中毒比例 b=5%；奖励模型训练 2 epochs，PPO 训练 1 epoch；8× NVIDIA A100-80GB GPU。
