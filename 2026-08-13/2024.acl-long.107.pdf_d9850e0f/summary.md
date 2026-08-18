---
title: "Self-Alignment for Factuality: Mitigating Hallucinations in LLMs via Self-Evaluation"
source: https://aclanthology.org/2024.acl-long.107.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:00:15"
field: "大语言模型事实性与对齐"
keywords: ["Hallucination Mitigation", "Self-Evaluation", "Factuality Alignment", "Direct Preference Optimization", "Confidence Calibration", "LLM"]
innovations: ["提出利用LLM自我评估置信度作为DPO训练信号的自对齐框架", "设计SK-TUNING通过异质数据微调提升模型置信度估计与校准能力"]
benchmarks: ["TruthfulQA", "BioGEN", "CommonSenseQA", "OpenBookQA", "MedQA", "MMLU"]
---

# 论文速读：Self-Alignment for Factuality: Mitigating Hallucinations in LLMs via Self-Evaluation

## 一句话总结
论文提出 **Self-Alignment for Factuality** 框架，利用大语言模型（LLM）的**自我评估能力**生成事实性置信度分数，并以此作为偏好信号结合 **DPO** 算法直接微调模型，从而在无需人工标注数据的情况下缓解 LLM 的知识密集型任务幻觉。

## 研究问题与动机
1. **“知道却说错”**：LLM 常在具备相关内部知识的情况下，仍生成看似合理但事实错误的陈述（即幻觉），导致“知”与“言”脱节。
2. **现有方法局限**：表示编辑类方法（如 ITI、DOLA）依赖**领域特定标注数据**，泛化性受限；基于一致性的置信度估计（如 FACTTUNE-MC）过度依赖模型**生成能力**，难以反映模型内部真实知识。
3. **自我评估潜力**：LLM 展现出能够基于内部知识评估自身生成答案事实性的能力，但存在**过度自信**和**校准不佳**的问题，需通过专门训练加以强化。
4. **对齐信号缺失**：缺乏仅依赖模型**内部知识**即可生成高质量事实性偏好对的方法，以实现低成本、高泛化的事实性对齐。

## 核心贡献（创新点）
1. **提出 Self-Alignment for Factuality 框架**：首次系统地将 LLM 自我评估能力转化为训练信号，通过 DPO 实现事实性对齐，无需人工事实标注。
2. **设计 SELF-EVAL 自我评估组件**：利用 True/False Q&A 提示格式，让 LLM 直接评估生成答案（或原子声明）的事实性，产出置信度分数 `p(True|q,a)`。
3. **引入 SK-TUNING 置信度校准方法**：通过在异质知识数据集（Wikipedia、BIG-bench）上微调，显著提升 LLM 自我评估的**准确性**和**校准度**，解决朴素提示的过度自信问题。
4. **在多项基准上验证有效性**：在 TruthfulQA（MCQA、短文本生成）和 BioGEN（长文本生成）上，该方法显著超越表示编辑方法和基于一致性的基线，尤其在保持信息丰富度的同时提升事实准确性。

## 方法详解
1. **整体流程（三步）**：
   - **Step 1**：对提示 `x` 采样生成 `M` 个候选响应 `{y_m}`。
   - **Step 2**：利用 SELF-EVAL 计算每个响应的事实性得分。短文本直接评分；长文本先通过 GPT-3.5-turbo 提取原子声明并转为原子问题，再对每个声明评分后取平均（Avg-p(True)）。
   - **Step 3**：按得分排序，选取 top α 响应作为偏好对中的胜出方 `y_w`，其余作为落选方 `y_l`，使用 DPO 损失微调模型。

2. **SELF-EVAL 评估机制**：
   - 采用 Few-shot 引导的 True/False 二分类提示格式，要求模型判断给定答案是否真实。
   - 概率 `p(True|q,a)` 由模型输出 `A`（True）的概率直接表征，作为事实性置信度。

3. **SK-TUNING 训练**：
   - **数据构建**：对 Wikipedia 和 BIG-bench 中的问题，生成多个候选答案，用 Deberta-Large-MNLI 进行双向蕴含判断，将与标准答案语义等价的标注为正确，否则为错误。
   - **训练格式**：构造 True/False 训练对，正确答案配正预测标签，错误答案配负预测标签。
   - **损失函数**：采用对比学习风格的损失 `L_φ`，最大化正确与错误答案预测概率的对数差：
     ```
     L_φ = -E[log σ(log π_φ(r_+|q,a) - log π_φ(r_-|q,a))]
     ```
   - **保留重复答案**：为改善校准，训练数据中刻意保留重复的候选答案。

4. **DPO 对齐微调**：
   - 将 Self-Alignment 构建的偏好数据集 `{(x, y_w, y_l)}` 用于 DPO 优化：
     ```
     L_θ = -E[log σ(β log(π_θ(y_w|x)/π_ref(y_w|x)) - β log(π_θ(y_l|x)/π_ref(y_l|x)))]
     ```
   - 超参：微调 5 epochs，batch size=8，学习率=5e-6，β=0.1。

## 实验与结果
1. **数据集与任务**：
   - **TruthfulQA**：MCQA（6-shot）和短文本生成（6-shot）。
   - **BioGEN**：长文本传记生成（5-shot）。
   - **评估指标**：Accuracy、True（真实性）、Info（信息量）、True*Info、FActScore（原子声明正确率）、正确/错误事实数。

2. **基线方法**：
   - SFT、ITI（推理时干预）、DOLA（对比层解码）、FACTTUNE-MC（基于一致性的 DPO）、SE（语义等价聚类）、USC（普遍自洽）。

3. **主要结果**：
   - **LLAMA-7B**：Self-Alignment w/ SELF-EVAL-SKT 使 TruthfulQA MC 准确率从 25.60% 提升至 **45.48%**（+19.88%），True*Info 从 26.90% 提升至 **45.75%**；BioGEN FActScore 从 30.72% 提升至 **38.28%**（+7.56%）。
   - **LLAMA2-7B**：TruthfulQA MC 准确率 44.10%（+15.20%），True*Info **53.42%**；BioGEN FActScore **46.50%**。
   - **SK-TUNING 增益**：相比朴素 SELF-EVAL-P(TRUE)，SK-TUNING 提升 True*Info 约 12%，FActScore 约 4%。
   - **对比基线**：全面超越 ITI、DOLA、FACTTUNE-MC 及一致性方法，且在 GPT-4  pairwise 评估中在事实性、帮助性、相关性、自然性四维均显著胜出。

4. **置信度评估**：在 5 个 MCQA 数据集（TruthfulQA、CommonSenseQA、OpenBookQA、MedQA、MMLU）上，SELF-EVAL-SKT 的 **Accuracy** 和 **AUROC** 均显著优于朴素提示和基座模型，证明 SK-TUNING 有效提升了事实性估计能力。

## 相关工作脉络
1. **表示编辑方法（ITI、DOLA）**：依赖领域标注数据微调模型内部表示，泛化受限；本文方法无需标注，仅用内部知识信号。
2. **一致性置信度方法（FACTTUNE-MC）**：基于多次生成的一致性估计置信度，受生成能力影响大；本文利用自我评估直接访问内部知识。
3. **后验修正方法（Self-Consistency、CoVe）**：在推理时调整输出，不改变模型参数；本文通过训练从根本上提升模型事实性。
4. **诚实微调（HoNESTY-TUNE）**：鼓励模型承认“不知道”；本文聚焦于模型知道时如何准确表述。
5. **自我评估研究（Kadavath et al.）**：揭示 LLM 具备知识感知能力；本文将其系统化并用于对齐训练，解决过度自信问题。
6. **原子声明评估（FActScore）**：采用 GPT 提取原子声明进行细粒度评估；本文沿用该流程构建训练信号。

## 局限性与未来方向
1. **未与解码策略结合**：可与高性能解码方法（如 DOLA）结合，进一步挖掘事实性提升潜力。
2. **仅在小规模模型验证**：仅在 7B 模型上实验，可扩展至更大参数规模（13B、70B）及 RLHF 微调模型（如 LLAMA2-CHAT）。
3. **置信度估计仍有优化空间**：自我评估的校准精度可进一步提升，未来可探索更先进的不确定性估计方法。
4. **残留幻觉源于预训练数据**：部分错误（如误导性问题、迷信、争议问题）反映预训练数据缺陷，需高质量人工标注数据或更严格的过滤机制。

## 研究启发与可借鉴点
1. **自我评估作为监督信号**：将模型内部知识感知能力转化为训练数据，避免昂贵的人工标注，可迁移至其他对齐任务（如安全性、指令遵循）。
2. **置信度校准的重要性**：SK-TUNING 表明，对基础评估组件进行专门微调能显著提升下游对齐效果，这对任何依赖模型自评分数的框架均有参考价值。
3. **DPO 与自我评估结合**：展示了无需显式奖励模型的偏好优化路径，简化了事实性对齐流程，可推广至多语言、多领域场景。
4. **细粒度评估流程**：原子声明提取与验证 pipeline 可直接复用于其他需要细粒度事实性评估的研究。
5. **异质数据增强泛化**：SK-TUNING 使用 Wikipedia 和 BIG-bench 等多样数据，证明了跨领域训练对提升模型通用自我评估能力的有效性。

## 关键术语表
1. **Hallucination**：LLM 生成看似合理但违背已知事实的输出。
2. **Self-Evaluation**：LLM 利用内部知识评估自身生成内容正确性的能力。
3. **Confidence Estimation**：模型对自身预测正确性的概率量化。
4. **Calibration**：置信度估计与真实准确率之间的一致性程度。
5. **Direct Preference Optimization (DPO)**：直接使用偏好对优化语言模型策略的算法，无需显式奖励模型。
6. **Atomic Claim**：从复杂文本中提取的不可再分的最小事实单元。
7. **True/False Q&A**：将答案判断转化为真假问题的提示格式，用于激活模型的自我评估。
8. **SK-TUNING**：Self-Knowledge Tuning，通过异质数据微调增强 LLM 事实性自我评估和置信度校准能力。

## 可复现要素
1. **数据集**：TruthfulQA、BioGEN、BIG-bench 公开可用；SK-TUNING 训练数据源自 Wikipedia 和 BIG-bench 子集。
2. **代码/权重**：论文未明确说明代码或模型权重是否开源。
3. **关键超参**：
   - DPO 微调：5 epochs，batch size=8，学习率=5e-6，β=0.1。
   - SK-TUNING：1 epoch，batch size=8，学习率=5e-7。
   - 候选响应生成：temperature T=1, 0.9, 0.8，样本数 M=30。
   - 偏好对选取比例 α：LLAMA-7B 为 30%，LLAMA2-7B 为 50%。
