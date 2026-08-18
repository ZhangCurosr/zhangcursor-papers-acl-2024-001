---
title: "Self-Alignment for Factuality: Mitigating Hallucinations in LLMs via Self-Evaluation"
source: https://aclanthology.org/2024.acl-long.107.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:59:46"
field: "大语言模型事实性与可靠性"
keywords: ["幻觉缓解", "自我评估", "事实对齐", "DPO", "置信度校准", "LLM"]
innovations: ["提出基于LLM自我评估的事实对齐框架，无需人工标注", "设计SK-TUNING增强置信度估计与校准能力", "将原子陈述分解与自我评估结合用于长文本事实性判断"]
benchmarks: ["TruthfulQA", "BioGEN"]
---

# 论文速读：Self-Alignment for Factuality: Mitigating Hallucinations in LLMs via Self-Evaluation

## 一句话总结
本文提出**Self-Alignment for Factuality**框架，利用LLM的自我评估能力对其生成内容进行事实性置信度打分，并以此作为奖励信号通过DPO进行对齐微调，从而在不依赖人工标注的情况下有效缓解LLM的"知道却说不准"类幻觉问题。

## 研究问题与动机
- **核心问题**：LLM即使持有相关知识，有时仍会生成看似合理但事实错误的陈述（即"knowing vs. telling"差距），削弱了预训练阶段所获知识的有效利用。
- **现有方法不足**：
  1. 表示编辑方法（如ITI、DOLA）依赖领域特定标注数据，泛化性受限。
  2. 基于一致性的置信度估计（如FACTTUNE-MC）依赖模型的生成能力，可能无法真实反映其内部知识。
  3. RLHF/SFT方法需要大量高质量人工标注，成本高。
- **关键洞察**：LLM虽难以直接生成正确回答，但具备识别自身生成内容中事实错误的能力（Kadavath et al., 2022），这种**自我评估**可能是更可靠的事实性估计方式。

## 核心贡献（创新点）
1. **提出Self-Alignment for Factuality框架**：利用LLM的自我评估能力生成训练信号并通过DPO对齐模型，与依赖外部标注或一致性采样的方法本质不同。
2. **设计SELF-EVAL机制**：通过提示LLM对原子陈述进行True/False评估，直接利用内部知识进行事实性判断，而非依赖多次采样的一致性。
3. **引入SK-TUNING**：通过异构数据上的成对预测微调，显著提升LLM的置信度估计精度和校准质量（解决原始prompt的过度自信问题）。
4. **无监督事实对齐**：无需领域特定标注数据，仅依赖模型内部知识即可完成幻觉缓解，相比表示编辑方法更具通用性。

## 方法详解
**整体流程（三步）**：

1. **生成候选响应**：对给定prompt x，通过few-shot提示生成M个候选响应{y_m}。
2. **自我评估事实性**：
   - 长文本：用GPT-3.5-turbo提取原子陈述列表，再将每个陈述转化为原子问题。
   - 利用SELF-EVAL计算p(True|q, c)，即基于内部知识判断陈述c对问题q是否为真。
   - 最终响应得分=各原子陈述分数的平均值（Avg-p(True)）。
   - MCQA任务：直接对每个选项计算p(True)。
3. **DPO对齐微调**：
   - 按事实性得分排序，选取top α为偏好响应y_w，其余为非偏好响应y_l。
   - 使用DPO损失函数优化：$L_\theta = -E[\log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)})]$

**SK-TUNING设计**：
- 数据来源：Wikipedia（49,862 prompt）+ BIG-bench 17个MCQA任务（32,500 prompt）
- 训练样本构造：对每道题生成K个候选答案，用Deberta-Large-MNLI判断与golden answer的语义等价性，正确标记为a_c，错误为a_i
- 成对预测损失：$L_\phi = -E[\log \sigma(\log \pi_\phi(r_+|q,a) - \log \pi_\phi(r_-|q,a))]$
- 保留重复答案以增强置信度校准

## 实验与结果
- **数据集**：TruthfulQA（MCQA + 短文本生成）、BioGEN（长文本传记生成）
- **基线**：SFT、ITI、DOLA、FACTTUNE-MC、SE（语义等价聚类）、USC（通用自一致性）
- **基座模型**：LLAMA-7B、LLAMA2-7B
- **关键结果**：
  - TruthfulQA MC：LLAMA2-7B + SELF-EVAL-SKT达到**44.10%**准确率（base为28.90%，提升约15%）
  - TruthfulQA短文本：True*Info达**53.42%**（base为39.04%，提升约14%）
  - BioGEN长文本：FActScore达**46.50%**（base为40.54%，提升约6%）
  - 自评估能力（AUROC）：在CommonsenseQA上达84.65%，显著优于BASELINE的79.76%
  - 配对评估（GPT-4判评）：在事实性、有用性、相关性、自然性四个维度均显著优于FACTTUNE-MC和SELF-EVAL-P(TRUE)
  - SK-TUNING使True*Info提升约12%（LLAMA-7B）、FActScore提升约4%（BioGEN）

## 相关工作脉络
1. **FACTTUNE-MC（Tian et al., 2023a）**：基于一致性置信度的DPO对齐；本文用直接自我评估替代，无需多次采样。
2. **ITI（Li et al., 2023b）/ DOLA（Chuang et al., 2023）**：表示编辑方法，依赖领域标注数据；本文无需外部标注，通用性更强。
3. **HoNESTY-TUNE（Yang et al., 2023）**：诚实性微调，鼓励模型说"我不知道"；本文聚焦于让模型在有知识时"说真话"。
4. **Self-Consistency/USC（Wang et al., 2023; Chen et al., 2023c）**：通过多采样一致性判断答案；本文证明直接自我评估比一致性更可靠。
5. **Kadavath et al. (2022)**：证明LLM具备知识自知能力；本文将其扩展为事实对齐的训练信号。

## 局限性与未来方向
- 仅在7B规模LLAMA模型上验证，未探索更大规模模型（如13B、70B）。
- 未测试在RLHF微调模型（如LLAMA2-CHAT）上的效果。
- 未结合解码阶段策略（如DOLA）进行联合优化。
- 仍存在部分难以避免的事实错误（如误导性前提、模棱两可问题、超自然信念等）。
- 未来方向：结合解码策略、扩展至更大模型、探索更高效置信度校准方法。

## 研究启发与可借鉴点
1. **自我评估作为训练信号**：可将模型内部知识能力转化为对齐信号，减少对人工标注的依赖，适用于其他对齐任务（如安全性、毒性检测）。
2. **SK-TUNING设计思路**：通过成对预测损失增强置信度估计和校准，该方法可迁移到任何需要可靠不确定性估计的场景。
3. **原子陈述分解策略**：长文本事实评估可复用"提取原子陈述→转化为问题→独立评估→聚合"的 pipeline，适用于FactScore类评估。
4. **保留重复答案提升校准**：SK-TUNING中保留重复候选答案有助于改善校准，这一技巧对任何置信度训练都可能有借鉴价值。
5. **无标注事实对齐范式**：为后续研究提供了"无需领域标注即可完成事实性对齐"的方法论参考。

## 关键术语表
**Self-Alignment for Factuality**：利用LLM自我评估能力提供训练信号，通过DPO对齐模型以缓解幻觉的框架。

**SELF-EVAL**：自我评估组件，提示LLM基于内部知识对生成回答的事实性进行True/False判断。

**SK-TUNING**：Self-Knowledge Tuning，通过异构数据上的成对预测微调，增强LLM置信度估计与校准能力的训练策略。

**DPO（Direct Preference Optimization）**：直接偏好优化算法，无需显式奖励模型，直接通过偏好对优化策略。

**FActScore**：基于原子声明的事实精度评估指标，衡量生成文本中事实性正确陈述的比例。

**AUROC**：ROC曲线下面积，用于评估模型区分正确与错误答案的置信度校准能力。

**True*Info**：TruthfulQA上的复合指标，综合衡量回答的事实准确性与信息丰富度。

**Know vs. Tell Gap**：LLM掌握知识却无法准确表达的知识利用差距问题。

## 可复现要素
- **数据集**：TruthfulQA（公开）、BioGEN（公开）、Wikipedia、BIG-bench（部分公开）
- **代码/权重**：论文未提及开源声明
- **关键超参**：
  - SK-TUNING：batch_size=8，learning_rate=5e-7，1 epoch
  - DPO微调：batch_size=8，learning_rate=5e-6，β=0.1，5 epochs
  - 生成候选数M=30，偏好比例α=30%（LLAMA-7B）/50%（LLAMA2-7B）
  - 训练设备：8×32G Tesla V100
