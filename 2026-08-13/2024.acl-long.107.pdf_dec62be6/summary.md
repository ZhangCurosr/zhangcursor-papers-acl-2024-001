---
title: "Self-Alignment for Factuality: Mitigating Hallucinations in LLMs via Self-Evaluation"
source: https://aclanthology.org/2024.acl-long.107.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:59:57"
field: "大语言模型可靠性与对齐"
keywords: ["幻觉缓解", "自我评估", "事实性对齐", "DPO", "置信度校准", "大语言模型"]
innovations: ["利用LLM自我评估能力生成事实性置信度作为DPO训练信号", "SK-TUNING通过异构知识任务微调提升置信度估计和校准", "无需人工标注即可实现面向事实性的模型对齐"]
benchmarks: ["TruthfulQA", "BioGEN", "CommonSenseQA", "OpenBookQA", "MedQA", "MMLU"]
---

# 论文速读：Self-Alignment for Factuality: Mitigating Hallucinations in LLMs via Self-Evaluation

## 一句话总结
本文提出Self-Alignment for Factuality框架，利用LLM的自我评估能力生成事实性置信度分数作为奖励信号，通过DPO算法直接优化模型，在无需人工标注数据的情况下显著降低LLM的事实性幻觉。

## 研究问题与动机
- **核心问题**：LLM存在"知道但说不准"的幻觉问题——模型在预训练中习得了相关知识，但在生成时仍可能输出事实性错误的内容
- **现有方法不足**：
  - 基于一致性(confidence-based)的方法依赖模型的生成能力，难以反映内部知识的真实掌握程度
  - 表征编辑方法需要领域特定的人工标注数据，泛化性受限
  - RLHF等对齐方法依赖大量高质量人工标注，成本高昂
- **关键洞察**：LLM在"评估自己生成的回答"方面表现出潜力，即使它难以直接生成准确答案

## 核心贡献（创新点）
1. **提出Self-Alignment for Factuality框架**：利用LLM自我评估能力生成事实性置信度，通过DPO直接对齐模型向事实性方向，与依赖人工标注或外部知识的方法本质不同
2. **设计SK-TUNING方法**：通过异构知识任务上的监督微调，提升LLM的置信度估计和校准能力，显著优于仅依赖提示的 SELF-EVAL-P(TRUE) 方法
3. **构建SELF-EVAL-SKT组件**：基于自知识引导的训练数据，使模型能够基于内部知识评估答案真假，在5个MCQA数据集上选择准确率较基础模型提升12-17%

## 方法详解
**三步框架**：
1. **生成初始响应**：对给定提示x，从基础LLM生成M个候选响应{y_m}
2. **自我评估打分**：
   - 长文本：用GPT-3.5-turbo提取原子声明(atomic claims)，转为原子问题
   - 用SELF-EVAL评估每个声明的事实性：p(True|q, c)
   - 取平均得到响应的事实性分数Avg-p(True)
3. **DPO对齐**：按事实性分数排序，选top α为偏好响应y_w，其余为非偏好y_l，构建偏好对D={(x, y_w, y_l)}，用DPO优化

**SK-TUNING设计**：
- 训练数据构建：从Wikipedia和BIG-bench采样49,862+32,500个问题，生成候选答案后用Deberta-Large-MNLI验证事实性
- 构建True/False训练对，保留重复答案以改善校准
- 损失函数：$\mathcal{L}_\phi = -\mathbb{E}[\log \sigma(\log \pi_\phi(r_+|q,a) - \log \pi_\phi(r_-|q,a))]$

**DPO对齐**：
- 标准DPO损失：$\mathcal{L}_\theta = -\mathbb{E}[\log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)})]$

## 实验与结果
**数据集与任务**：
- TruthfulQA：MCQA + 短文本生成（38类别知识）
- BioGEN：长文本生成（人物传记）

**主要结果**（LLAMA-7B）：
- MCQA准确率：45.48%（vs 基础25.60%，+13%）
- TruthfulQA生成True*Info：45.75%（最佳）
- BioGEN FActScore：38.28%（vs 基础30.72%，+4%）

**对比基线优势**：
- 显著超越一致性方法FACTTUNE-MC（BioGEN FActScore 38.28% vs 30.92%）
- 超越表征编辑方法ITI和DOLA（无需领域标注数据）
- SELF-EVAL-SKT vs SELF-EVAL-P(TRUE)：True*Info提升12%，FActScore提升4%

**置信度评估**（LLAMA2-7B）：
- CommonSenseQA选择准确率：70.43%（+16% over base）
- OpenBookQA：67.40%（+12% over base）

## 相关工作脉络
- **TI/DoLA**（表征编辑方法）：需领域标注数据，本文方法无需外部标注，仅用模型内部知识
- **FACTTUNE-MC**（一致性方法）：基于多次生成的一致性估计置信度，本文的自我评估更直接反映内部知识
- **HoNESTY-TUNE**（诚实性微调）：侧重让模型说"不知道"，本文聚焦于"知道时准确说出"
- **Kadavath et al.**（LM know what they know）：本文在此基础上改进置信度校准能力
- **RLHF/DPO对齐**：本文用自我评估替代人工偏好标注，降低对齐成本

## 局限性与未来方向
- **与解码策略结合**：可与DOLA等高表现方法结合进一步提升效果
- **模型规模扩展**：已在7B模型验证，预期在13B/70B及RLHF微调模型（如LLAMA2-CHAT）上效果更好
- **更有效的置信度估计**：可探索Lightning Calibration等更高效方法
- **复杂查询处理**：对误导性前提、超自然观念、争议性问题仍易出错，需高质量人工标注数据进一步优化

## 研究启发与可借鉴点
1. **自我评估作为奖励信号**：将LLM的内部知识评估转化为训练信号，避免了昂贵的人工标注，可迁移至其他对齐任务
2. **SK-TUNING的异构数据策略**：结合Wikipedia和BIG-bench构建训练数据，覆盖已知和未知知识，有效改善置信度校准
3. **原子声明分解评估**：长文本生成中分解为原子声明逐一评估，提高了评估粒度和准确性
4. **保留重复答案改善校准**：训练数据中保留重复答案有助于模型更好校准置信度，这一技巧可复用到其他校准任务
5. **DPO替代RLHF**：用自我评估构建偏好对进行DPO训练，为低成本对齐提供了可行路径

## 关键术语表
- **Hallucination**：LLM生成看似合理但事实错误的内容，尤其指模型拥有相关知识却无法准确表达的情况
- **Self-Alignment**：利用模型自身的评估能力作为训练信号进行对齐，无需人工标注
- **SELF-EVAL**：自我评估组件， prompts LLM基于内部知识判断其生成回答的真假
- **SK-TUNING**：Self-Knowledge Tuning，通过异构知识任务的微调增强LLM的置信度估计和校准能力
- **DPO (Direct Preference Optimization)**：直接偏好优化算法，将偏好数据转化为策略梯度更新，避免显式奖励建模
- **Atomic Claim**：原子声明，将文本分解为独立、不可再分的事实陈述单元
- **Calibration**：置信度校准，模型输出的置信度与其实际准确率的一致性程度
- **True*Info**：TruthfulQA的复合指标，衡量回答的事实准确性与信息丰富度的平衡

## 可复现要素
- **数据集**：TruthfulQA（公开）、BioGEN（使用公开提示）、Wikipedia（公开）、BIG-bench（公开）
- **代码**：论文未提及代码开源情况
- **权重**：使用LLAMA-7B和LLAMA2-7B开源权重
- **关键超参**：DPO微调5 epochs, batch size=8, learning rate=5e-6, β=0.1；SK-TUNING 1 epoch, batch size=8, learning rate=5e-7；候选响应数M=30
