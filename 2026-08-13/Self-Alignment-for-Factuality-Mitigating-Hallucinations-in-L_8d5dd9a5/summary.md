---
title: "Self-Alignment-for-Factuality-Mitigating-Hallucinations-in-L"
source: https://aclanthology.org/2024.acl-long.107.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:51:07"
field: "大语言模型事实性与可信度"
keywords: ["大语言模型", "幻觉缓解", "自我评估", "事实性对齐", "DPO", "置信度校准"]
innovations: ["提出基于自我评估的事实性对齐框架Self-Alignment for Factuality", "设计SK-TUNING增强LLM置信度估计与校准能力", "利用模型内部知识生成偏好对无需人工标注"]
benchmarks: ["TruthfulQA", "BioGEN", "CommonSenseQA", "OpenBookQA", "MedQA", "MMLU"]
---

# 论文速读：Self-Alignment for Factuality: Mitigating Hallucinations in LLMs via Self-Evaluation

## 一句话总结
本文提出Self-Alignment for Factuality框架，利用LLM的自我评估能力生成事实性置信度分数，并通过DPO算法微调模型以缓解"知道但说错"类型的幻觉问题。

## 研究问题与动机
- **核心问题**：LLM在知识密集型任务中常产生看似合理但事实错误的"幻觉"，特别是模型已掌握相关知识却无法准确输出的"knowing-telling gap"现象。
- **现有方法不足**：
  1. 事后校正方法（如self-consistency）依赖模型生成能力，无法反映内部知识
  2. 推理时干预方法（表示编辑）需要领域特定的标注数据，泛化性受限
  3. 对齐训练方法（RLHF/SFT）需要大量人工标注，成本高昂
  4. 近期工作FACTTUNE-MC依赖一致性置信度，但该指标依赖模型生成能力，可能无法有效反映内部知识

## 核心贡献（创新点）
1. **提出Self-Alignment for Factuality框架**：利用LLM自我评估能力提供训练信号，引导模型朝向事实性对齐，与现有方法依赖人工标注或外部知识源有本质区别。
2. **设计SELF-EVAL-SKT自我评估组件**：通过直接提示模型基于内部知识验证回答真实性，相比一致性方法更能反映模型内部知识状态。
3. **引入SK-TUNING增强自我评估能力**：通过在异质知识导向任务上微调，显著提升模型置信度估计和校准能力，解决原始提示方法过自信问题。
4. **构建无标注数据的偏好对训练流程**：仅依赖模型内部知识即可生成偏好训练数据，无需领域特定标注，区别于ITI、DOLA等方法。

## 方法详解
**整体框架（三步骤）**：
1. **生成初始响应**：对给定prompt x，从基座模型π_ref采样M个候选响应{y_m}
2. **自我评估标注**：通过SELF-EVAL计算每个响应的真实性分数
   - 长文本任务：先用GPT-3.5-turbo抽取原子声明，转化为原子问题，再用SELF-EVAL评估每个声明的p(True|q,c)，取平均得最终分数
   - MCQA任务：直接计算各选项的p(True)
3. **DPO对齐微调**：按分数排名，选top α为优选y_w，其余为劣选y_l，用DPO损失函数微调

**SELF-EVAL组件**：
- 采用True/False Q&A提示格式，让模型评估给定回答的真实性
- 概率计算：p(True|q,a) = f_M(q,a)
- 问题：基础提示( SELF-EVAL-P(TRUE))存在过自信问题

**SK-TUNING设计**：
- 训练数据构建：从Wikipedia(49,862 prompts)和BIG-bench(32,500 prompts)采样，生成候选回答后用Deberta-Large-MNLI验证事实正确性
- 配对预测损失函数：
  L_φ = -E[logσ(log π_φ(r_+|q,a) - log π_φ(r_-|q,a))]
- 保留重复样本以改善置信度校准

**DPO对齐损失**：
L_θ = -E[logσ(βlog(π_θ(y_w|x)/π_ref(y_w|x)) - βlog(π_θ(y_l|x)/π_ref(y_l|x)))]

## 实验与结果
**数据集与任务**：
- TruthfulQA：MCQA任务（Accuracy）、短文本生成任务（True*Info）
- BioGEN：长文本生成任务（FActScore）

**基线方法**：SFT、ITI、DOLA、FACTTUNE-MC、SE、USC

**主要结果**：
- **LLAMA-7B + SELF-EVAL-SKT**：TruthfulQA MC准确率45.48%（vs基座25.60%，提升约20%），短文本True*Info 45.75%（vs基座26.90%，提升18.85%）
- **LLAMA2-7B + SELF-EVAL-SKT**：TruthfulQA MC准确率44.10%，短文本True*Info 53.42%（vs基座39.04%，提升14.38%），BioGEN FActScore 46.50%（vs基座40.54%，提升约6%）
- 对比FACTTUNE-MC：在BioGEN上FActScore分别提升约4-5个百分点
- 对比SE/USC变体：SELF-EVAL-SKT在所有任务上表现最优
- 配对比较（GPT-4评估）：在事实性、帮助性、相关性、自然性四个维度均显著优于基线

**置信度估计分析**：
- SELF-EVAL-SKT在5个MCQA数据集上Accuracy和AUROC均显著优于SELF-EVAL-P(TRUE)
- SK-TUNING显著改善模型置信度校准，消除过自信问题

## 相关工作脉络
1. **推理时干预方法**（ITI、DOLA）：需领域标注数据编辑内部表示，本文方法无需标注数据且泛化性更强
2. **FACTTUNE-MC**（Tian et al., 2023a）：依赖一致性置信度估计，本文使用直接自我评估，能更好反映内部知识
3. **HoNESTY-TUNE**（Yang et al., 2023）：聚焦诚实性微调使模型承认"不知道"，本文聚焦在模型"知道"时准确输出
4. **自我一致性方法**（Self-Consistency、CoVe）：依赖多次采样一致性，本文利用单次评估的置信度分数
5. **RLHF/SFT对齐训练**：需大量人工标注，本文完全基于模型内部知识生成训练信号

## 局限性与未来方向
- **未结合解码策略**：可与DOLA等高分方法结合进一步提升效果
- **仅在小规模模型验证**：预期在更大模型（13B、70B）及RLHF微调模型（LLAMA2-CHAT）上效果更好
- **置信度估计方法可优化**：SK-TUNING虽有效，但仍有更高效校准方法的探索空间
- **预处理数据质量依赖**：部分错误源于预训练数据包含迷信等人类误解，需更高质量数据

## 研究启发与可借鉴点
1. **自我评估作为训练信号**：利用模型自身知识而非外部标注来引导对齐，为低资源场景提供新思路
2. **置信度校准的重要性**：SK-TUNING证明提升置信度估计和校准可显著改善事实性对齐效果，值得在其他对齐任务中借鉴
3. **原子声明分解策略**：将长文本分解为原子声明逐一评估的方法，可有效处理细粒度事实验证
4. **保留重复样本的教训**：Appendix H发现保留重复答案对SK-TUNING的校准效果至关重要，提示训练数据去重需谨慎
5. **与解码策略结合的潜力**：框架设计天然可与自一致性等解码方法互补

## 关键术语表
**Self-Alignment for Factuality**：利用LLM自我评估能力生成事实性置信度分数，通过DPO微调缓解幻觉的框架
**SELF-EVAL**：基于LLM内部知识的自我评估组件，通过True/False提示评估回答真实性
**SK-TUNING**：Self-Knowledge Tuning，通过异质知识任务微调增强模型置信度估计和校准能力
**Knowing-Telling Gap**：模型已掌握知识但无法准确输出的幻觉类型
**DPO (Direct Preference Optimization)**：直接偏好优化算法，无需显式奖励模型即可优化策略
**FActScore**：基于原子声明的事实性评估指标，衡量生成长文本的事实准确程度
**置信度校准 (Confidence Calibration)**：模型预测置信度与实际正确率的一致性程度

## 可复现要素
- **数据集**：TruthfulQA、BioGEN（公开基准）；SK-TUNING训练数据来自Wikipedia和BIG-bench（公开）
- **代码/权重**：论文未明确声明开源情况
- **关键超参**：DPO微调5 epochs，batch size=8，learning rate=5e-6，β=0.1；SK-TUNING微调1 epoch，batch size=8，learning rate=5e-7；生成候选响应时temperature=1, 0.9, 0.8
- **硬件**：8×32G Tesla V100
