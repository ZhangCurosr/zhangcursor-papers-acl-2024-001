---
title: "Learn-from-Failure-Fine-Tuning-LLMs-with-Trial-and-Error-Dat"
source: https://aclanthology.org/2024.acl-long.45.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:43:13"
field: "形式化定理证明与LLM"
keywords: ["Automated Theorem Proving", "Trial-and-Error Learning", "Large Language Models", "Intuitionistic Propositional Logic", "Lean"]
innovations: ["提出TRIALMASTER框架，在LLM微调中融入含回溯的完整试错证明树", "构建PropL数据集，通过自然数-命题Bijection均匀采样并生成带试错信息的证明", "证明试错训练可将OOD成功率从59.3%提升至88.7%，同时降低搜索成本"]
benchmarks: ["PropL (In-dist test: 1000 theorems, OOD test: 1000 theorems)"]
---

# 论文速读：Learn-from-Failure-Fine-Tuning-LLMs-with-Trial-and-Error-Dat

## 一句话总结
本文提出 **TRIALMASTER** 方法，通过在大语言模型（LLM）的微调数据中融入试错信息（包含成功与失败路径），显著提升直觉命题逻辑定理证明的成功率并降低搜索成本。同时构建了 Lean 形式化的 **PropL** 数据集以支持该训练范式。

## 研究问题与动机
1. **训练-推理分布不匹配**：现有神经定理证明器仅用成功路径微调，但推理阶段需通过 DFS 采样尝试多种策略，模型未从"失败"中学习。
2. **现有数据集缺乏试错信息**：开源定理证明数据集只提供正确证明路径，无法捕捉完整的搜索历史（包含回溯操作）。
3. **试错信息的直觉价值**：失败的搜索路径能帮助模型识别"不合适的策略"，从而在推理时规避已失败的路径或生成更优质的候选策略。
4. **LLM 形式化证明的可靠性需求**：需要形式化环境（如 Lean）保证定理与证明的正确性验证。

## 核心贡献（创新点）
1. **提出 TRIALMASTER 框架**：利用包含完整试错历史的证明树微调 LLM，推理时直接输出 Lean tactic 和回溯指令，无需外部 DFS 系统辅助。
2. **构建 PropL 数据集**：面向直觉命题逻辑，通过 Bijection 均匀采样定理 + FPS 算法生成含回溯的完整证明，共 109,887 个训练定理（分布内/外各 1,000 测试）。
3. **证明试错训练的有效性**：TRIALMASTER 在 OOD 任务上达到 88.7% 成功率，比仅用正确路径训练的模型（59.3%）提升 29.4%；且搜索成本（N_Lean）显著低于 DFS 基线。
4. **揭示短证明+试错数据的重要性**：短证明配合试错信息的训练效果优于长证明版本（88.7% vs 72.4%），说明过多失败路径可能降低训练数据质量。

## 方法详解
1. **数据集生成（PropL）**：
   - 通过自然数与命题公式的 Bijection（基于 Catalan 数计数）均匀采样命题，固定内部节点数为 16。
   - 使用 Focused Proof Search（FPS）算法 + 极化策略生成证明，每条定理对应最多 10 条含回溯的完整证明树。
   - 训练/测试集按证明词长度分位划分：训练集取最短 66% 分位数；OOD 测试取 >80% 分位数。

2. **TRIALMASTER 微调**：
   - 基座模型：Llama-2-7b-hf，每定理从 10 条含试错的证明中随机选取最短的 2 条，拼接为训练样本。
   - 模型同时学习正确 tactic 和回溯指令（如 "no solution, return to state 2"）。

3. **TRIALMASTER 推理**：
   - 贪婪生成单个 tactic， fed 入 Lean 检查状态转移。
   - 若输出回溯指令，则按历史树跳转；若输出正常 tactic 且 Lean 报错或超过 1500 词 limit 则终止。
   - 无需外部搜索系统，LLM 自身"记住"失败路径。

4. **DFS 基线对比**：
   - 使用相同 LLM 仅训练正确路径，推理时在每步采样 N_sampled 个候选 tactic，通过温度 t 控制多样性，DFS 回溯由系统完成。
   - 搜索步上限设为 65 步（避免超时）。

## 实验与结果
- **数据集规模**：训练集 109,887 个定理；分布内测试 1,000；分布外（OOD）测试 1,000。
- **主要指标**：Proof search success rate；Search cost（N_Lean = Lean 调用次数）。
- **TRIALMASTER vs DFS 基线（OOD）**：
  - TRIALMASTER：**88.7%** 成功率，搜索成本最低。
  - 最优 DFS（t=1.8, N_sampled=10）：88.4% 成功率，但 N_Lean=31,101，**比 TRIALMASTER 高 72%**。
- **Ablation（是否含试错）**：TRIALMASTER（88.7%）vs Model - proof w/o t.a.e.（59.3%），差距 **29.4%**。
- **训练数据长度影响**：短证明+试错（88.7%）> 长证明+试错（72.4%）。
- **含回退指令但无失败路径的训练**：成功率 75.6%，仍低于完整 TRIALMASTER。
- **归纳分布（ID）表现**：TRIALMASTER 和多数 DFS 配置均接近 100% 成功率，说明任务难度适中，OOD 更能区分模型能力。

## 相关工作脉络
1. **CO-PRA**：将回溯信息放入提示后调用 GPT-4，**不进行微调**，与本文的 fine-tuning 范式不同。
2. **Baldur**：使用证明助手的错误信息微调 LLM，但依赖外部错误反馈，而本文利用的是**完整历史证明树**。
3. **LeanDojo / ProofArtifactCoTraining**：结合 informal 证明辅助训练，本文聚焦于纯形式化试错数据。
4. **传统 DFS/BFS 定理证明器**：依赖外部搜索策略，需大量采样；本文方法以贪婪单步生成为主，**减少搜索开销**。
5. **Propositional Logic DNN/RNN 方法**（Sekiyama 等）：早期神经网络方法，未使用 LLM，本文将其扩展至现代 LLM 时代。
6. **Tree of Thoughts / Graph of Thoughts**：结构化推理范式，与本文的"试错学习"思路互补。

## 局限性与未来方向
1. **仅限于直觉命题逻辑**：未在更一般的数学定理（如高阶逻辑、一阶逻辑）上验证泛化能力。
2. **上下文长度限制**：1500 词 limit 导致部分长证明被截断，可能低估模型真实能力。
3. **缺乏 LLM 基线对比**：当前 LLM 在定理证明领域的 baseline 尚不完善。
4. **未来方向**：将试错训练范式扩展至更复杂的数学领域；探索试错数据的自动构建方法。

## 研究启发与可借鉴点
1. **失败路径的训练价值**：在推理需要多步搜索的任务中，训练时纳入"负面样本"（失败路径）可显著提升泛化性能，对 RLHF、工具调用等场景有借鉴意义。
2. **形式化验证作为数据生成器**：利用 Lean 等 proof assistant 自动验证证明正确性，可大规模生成带 label 的高质量训练数据。
3. **数据长度与质量权衡**：短证明+试错效果优于长证明+试错，提示训练数据选择需兼顾"信息量"与"噪声"，而非单纯增加数据量。
4. **自包含的推理架构**：TRIALMASTER 无需外部搜索系统即可实现回溯，减少了推理开销，对设计高效 Agent 有参考价值。

## 关键术语表
- **TRIALMASTER**：本文提出的模型名，指用含试错信息的证明树微调的 LLM 定理证明器。
- **PropL**：本文构建的直觉命题逻辑定理数据集，包含完整试错证明。
- **Focused Proof Search (FPS)**：基于极化的焦点证明搜索算法，用于生成含回溯的完整证明树。
- **N_Lean**：推理过程中 Lean 被调用的总次数，用于衡量搜索效率。
- **Backtrack instruction**：模型输出的回溯指令，指示返回到历史证明树的某个状态。
- **Intuitionistic Propositional Logic**：直觉命题逻辑，不同于经典逻辑，不承认排中律。
- **OOD (Out-of-Distribution)**：分布外测试，指证明长度超过训练集 80% 分位数的定理。

## 可复现要素
- **数据集**：PropL，作者声明在 GitHub 和 Huggingface 开源。
- **代码/权重**：论文未提及代码是否开源，仅说明数据集已发布。
- **关键超参**：Llama-2-7b-hf 基座；学习率 2×10⁻⁵；batch size 4；单 epoch；训练设备为 2×A100 GPU。
- **推理超参**：DFS 温度 t ∈ {0.3, 0.7, 1.2, 1.5, 2.0}；N_sampled ∈ {2, 5, 10, 15, 20}；最大步数 65；上下文上限 1500 词。
