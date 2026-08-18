---
title: "Self-Alignment for Factuality: Mitigating Hallucinations in LLMs via Self-Evaluation"
source: https://aclanthology.org/2024.acl-long.107.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:58:51"
field: "大语言模型事实性与对齐"
keywords: ["幻觉缓解", "事实性对齐", "自评估", "DPO", "置信度校准", "大语言模型"]
innovations: ["利用LLM自评估能力生成事实性信号并通过DPO零标注对齐", "SK-TUNING微调提升模型置信度估计与校准能力", "原子声明分解+SELF-EVAL实现细粒度事实性评估"]
benchmarks: ["TruthfulQA", "BioGEN", "CommonSenseQA", "OpenBookQA", "MedQA", "MMLU"]
---

# 论文速读：Self-Alignment for Factuality: Mitigating Hallucinations in LLMs via Self-Evaluation

## 一句话总结
本文提出 Self-Alignment for Factuality 框架，利用大语言模型自身的自评能力生成事实性置信度信号，并通过 DPO 对 LLAMA 系列模型进行对齐微调，无需人工标注数据即可显著缓解"知道但说不对"类幻觉问题。

## 研究问题与动机
1. **核心问题**：LLM 存在一类特殊的"不知真言"（knowing vs. telling）幻觉——模型在预训练中已掌握相关知识，但在生成时仍输出事实错误的内容。
2. **现有方法的不足**：表征编辑方法（ITI、DOLA）依赖领域标注数据，泛化性有限；一致性置信度方法（FACTTUNE-MC）依赖多次生成的结果一致性，无法有效反映模型内部真实知识；RLHF/SFT 需要大量高质量人工标注，成本高昂。
3. **关键洞察**：尽管 LLM 在"陈述"事实时可能出错，但其"评估"自身生成结果事实性的能力更为可靠（Kadavath et al., 2022），因此可直接利用模型自评能力作为训练信号。
4. **校准挑战**：直接使用朴素 True/False Q&A 提示（SELF-EVAL-P(TRUE)）会导致模型过度自信（overconfidence），需通过专门微调提升置信度估计与校准能力。

## 核心贡献（创新点）
1. **提出 Self-Alignment for Factuality 框架**：利用 LLM 自评能力生成事实性奖励信号并通过 DPO 对齐，与依赖人工标注的 SFT/RLHF 方法本质不同，实现了零标注数据的幻觉缓解。
2. **设计 SELF-EVAL-SKT 自评组件**：将原子化事实声明转化为对应问题，让模型基于内部知识直接评估生成内容的事实正确性（p(True|q,a)），比一致性置信度更能反映模型真实知识。
3. **提出 SK-TUNING 自知识微调**：通过在异构知识数据集（Wikipedia + BIG-bench）上构建 True/False 成对预测数据进行微调，显著提升模型的置信度估计精度和校准质量，这是本文相对于前期工作（如 Tian et al., 2023a）的核心技术创新。
4. **系统验证了三类知识密集型任务上的有效性**：在 MCQA、短文本生成和长文本生成三个任务上，Self-Alignment w/ SELF-EVAL-SKT 均显著优于表征编辑方法和一致性置信度方法。

## 方法详解
框架分为三步（见图2）：

**Step 1 — 生成候选响应**：对给定 prompt x，从基座模型 π_ref 采样 M=30 个候选响应 {y_m}，采用 few-shot 提示。

**Step 2 — 基于 SELF-EVAL 的事实性评估**：
- 长文本场景：先用 GPT-3.5-turbo 从响应中抽取原子声明列表，再将每个声明 c 转化为对应原子问题 q。
- 对每个 (q, c) 对，调用 SELF-EVAL 计算 p(True | q, c)，即模型基于内部知识判定该声明为真的概率。
- 最终响应的综合事实性得分 = 所有原子声明得分的平均值 Avg-p(True)。
- MCQA 场景：直接将各选项视为声明，分别计算 p(True) 并排序。

**SELF-EVAL 提示设计（SELF-EVAL-P(TRUE)）**：
```
Instruction: Please evaluate the truthfulness of the proposed answer based on 
the given question and internal knowledge.
<Few-shot Prompts>
Question: <Question>
Proposed Answer: <Answer>
Is the proposed answer: A. True / B. False
The proposed answer is:
```

**SK-TUNING 微调**：
- 训练数据构建：对每个问题 q 生成 K=30 个候选答案，用 Deberta-Large-MNLI 做双向蕴含判断与标准答案对比，将语义等价的答案标记为正确 a_c，否则标记为错误 a_i。
- 构建 True/False 成对训练数据 D_ψ：正确答案配对正预测 R_+(="A") 和负预测 R_-("B")，错误答案反之。
- 损失函数（式2）：
  L_φ = -E[(q,a,r_+,r_-)~D_ψ][log σ(log π_φ(r_+|q,a) - log π_φ(r_-|q,a))]
- 数据来源：Wikipedia（49,862 prompts）+ BIG-bench 17个MCQA任务（32,500 prompts），共约 247 万条训练样本。
- 训练配置：8×32G Tesla V100，1 epoch，batch size=8，学习率=5e-7。

**Step 3 — DPO 对齐微调（式3）**：
- 按事实性得分排序，取 top α 响应为偏好响应 y_w，其余为不偏好响应 y_l，构成偏好对 D = {(x, y_w, y_l)}。
- 使用 DPO 损失直接优化策略 π_θ，β=0.1，5 epochs，batch size=8，学习率=5e-6。

## 实验与结果
**数据集与任务**：
- TruthfulQA（MCQA + 短文本生成）：38个知识类别，735个测试样本
- BioGEN（长文本传记生成）：100个测试样本，使用 FActScore 评估

**评估指标**：TruthfulQA 用 Accuracy（MC）、True%、Info%、True*Info%；BioGEN 用 FActScore%、Respond%、正确/错误事实数量。

**主要结果（LLAMA-7B）**：
- MCQA Accuracy：基座 25.60% → SELF-EVAL-SKT **45.48%**（提升约 13%）
- 短文本 True*Info：基座 26.90% → SELF-EVAL-SKT **45.75%**
- BioGEN FActScore：基座 30.72% → SELF-EVAL-SKT **38.28%**（提升约 4%）

**主要结果（LLAMA2-7B）**：
- MCQA Accuracy：基座 28.90% → SELF-EVAL-SKT **44.10%**
- 短文本 True*Info：基座 39.04% → SELF-EVAL-SKT **53.42%**
- BioGEN FActScore：基座 40.54% → SELF-EVAL-SKT **46.50%**

**最强结果**：LLAMA2-7B + SELF-EVAL-SKT 在 TruthfulQA 短文本生成上达到 True*Info = 53.42%，较基座提升 14.38 个百分点；在 BioGEN 上 FActScore 达 46.50%，较基座提升约 6 个百分点。

**对比优势**：
- 优于一致性置信度方法 FACTTUNE-MC（LLAMA-7B 上 True*Info 高出 14.83%，BioGEN FActScore 高出 7.36%）
- 优于表征编辑方法 ITI 和 DOLA（无需领域标注数据）
- SELF-EVAL-SKT 比 SELF-EVAL-P(TRUE) 提升显著：True*Info 提升 12%，FActScore 提升 4%
-  pairwise 评估（GPT-4评判，100篇传记）：在 factuality、helpfulness、relevance 三维度胜率超 65%，naturalness 上打平

## 相关工作脉络
1. **Li et al. (2023b) ITI**：推理时通过偏移激活方向编辑内部表征以引导事实性，需领域标注数据；本文完全无需标注数据，通过自评估信号实现对齐。
2. **Chuang et al. (2023) DOLA**：对比不同层输出分布进行表征编辑；本文不修改内部表征，而是通过 DPO 直接微调生成策略。
3. **Tian et al. (2023a) FACTTUNE-MC**：使用多次生成的一致性置信度作为事实信号配合 DPO；本文用 SELF-EVAL-SKT 直接基于内部知识评估，更准确且校准更好。
4. **Yang et al. (2023) HoNESTY-TUNE**：面向"诚实"对齐，鼓励模型说"我不知道"；本文聚焦"知道且能正确陈述"的事实性对齐，目标不同。
5. **Kadavath et al. (2022)**：发现 LLM 能较好评估自身知识边界；本文将其扩展为事实性自评组件并用于对齐训练。
6. **OpenAI (2022, 2023) GPT-judge**：用于自动化评估；本文在长文本评估中同样使用 GPT-3.5-turbo 做原子声明提取与问题生成，但核心贡献在于利用模型自评作为训练信号而非评估工具。

## 局限性与未来方向
1. **模型规模限制**：仅在 LLAMA 7B 系列上验证，预计更大模型（13B、70B）及 RLHF 微调模型（如 LLAMA2-CHAT）上效果会更好，但未实验验证。
2. **与解码策略的结合潜力**：未与高表现的解码方法（如 DOLA）结合探索，论文指出这种结合可能带来额外提升。
3. **置信度估计仍可改进**：SK-TUNING 虽有效，但仍有更高效的置信度估计与校准方法值得探索（如 LitCab、Llamas Know 等）。
4. **残存幻觉类型**：分析显示五类最难处理的事实错误包括：缺乏精确知识、对不确定问题给出确定答案、被误导前提欺骗、迷信、回答有争议问题，这些根源于预训练数据质量，需更多高质量人工标注数据来改善。
5. **长文本评估依赖 GPT-3.5-turbo**：原子声明提取和问题生成使用外部模型，可能引入误差和成本。

## 研究启发与可借鉴点
1. **自评信号替代人工标注**：利用 LLM 自身知识评估生成内容事实性，再以 DPO 对齐，实现零标注幻觉缓解，这一范式可迁移到其他对齐任务（如安全性、有益性）。
2. **SK-TUNING 的置信度校准思路**：通过异构知识库（通用+Wikipedia+多领域MCQA）构建 True/False 成对预测数据进行微调，显著提升校准质量，该微调策略可用于其他需要可靠置信度的场景（如选择性生成、拒答机制）。
3. **原子声明分解评估**：将长文本响应分解为原子声明再逐条评估，最后聚合为响应级得分，这一思路可有效应用于任何需要细粒度事实性评估的任务。
4. **DPO 与自评信号的结合**：将连续置信度得分转化为偏好对（top α vs. 其余），然后用 DPO 微调，这一 pipeline 可复用于其他需要偏好信号但缺乏人工标注的场景。
5. **一致性方法vs.自评估方法的对比设计**：本文同时对比了一致性方法（SE、USC）和自评估方法，实验设计清晰展示了各自优劣，为后续工作提供了可靠的对比基准。

## 关键术语表
**Hallucination（幻觉）**：LLM 生成看似合理但事实错误的内容，本文聚焦"知道但说不正确"类型的无Faithful幻觉。
**Self-Alignment for Factuality**：利用 LLM 自评估能力生成训练信号，通过 DPO 将模型对齐向事实性方向的框架。
**SELF-EVAL**：事实性自评组件，通过 True/False Q&A 提示让模型基于内部知识评估生成内容的事实正确性。
**SELF-EVAL-SKT**：经 SK-TUNING 微调后的 SELF-EVAL，具有更高置信度估计精度和校准质量。
**SK-TUNING（Self-Knowledge Tuning）**：在异构知识数据集上微调模型，提升其置信度估计和校准能力的训练过程。
**DPO（Direct Preference Optimization）**：一种无需显式奖励模型、直接基于偏好对优化策略的强化学习替代方法。
**FActScore**：基于原子声明分解的长文本事实性评估指标，衡量生成内容中正确事实的比例。
**Overconfidence（过度自信）**：模型给出的置信度高于实际正确率的校准偏差现象，本文发现朴素 SELF-EVAL 存在此问题。

## 可复现要素
- **数据集**：TruthfulQA（公开）、BioGEN（使用公开 prompts，训练响应由 GPT-4 生成）、Wikipedia（公开）、BIG-bench（公开）
- **代码/权重**：论文未提及开源代码和模型权重
- **关键超参**：DPO 训练 — 5 epochs，batch size=8，学习率=5e-6，β=0.1；SK-TUNING — 1 epoch，batch size=8，学习率=5e-7；候选采样 M=30，温度 T=1/0.9/0.8；长文本 top α=30%（LLAMA-7B）/50%（LLAMA2-7B）
- **硬件**：8×32G Tesla V100
