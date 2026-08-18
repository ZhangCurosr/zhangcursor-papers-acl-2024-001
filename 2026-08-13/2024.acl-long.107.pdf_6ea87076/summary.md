---
title: "Self-Alignment for Factuality: Mitigating Hallucinations in LLMs via Self-Evaluation"
source: https://aclanthology.org/2024.acl-long.107.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:59:41"
field: "大语言模型事实性与对齐"
keywords: ["Hallucination Mitigation", "Self-Evaluation", "Direct Preference Optimization", "Confidence Calibration", "Factuality Alignment", "Large Language Models"]
innovations: ["提出基于LLM自评估的事实性对齐框架，无需外部标注数据", "设计SK-TUNING增强模型置信度估计与校准能力", "将自评估置信度分数作为DPO奖励信号实现幻觉缓解"]
benchmarks: ["TruthfulQA", "BioGEN", "CommonSenseQA", "OpenBookQA", "MedQA", "MMLU"]
---

# 论文速读：Self-Alignment for Factuality: Mitigating Hallucinations in LLMs via Self-Evaluation

## 一句话总结
本文提出 Self-Alignment for Factuality 框架，利用 LLM 自身知识进行自评估（SELF-EVAL）来生成事实性置信度分数，并配合 SK-TUNING 提升模型的置信度估计与校准能力，最终通过 DPO 对基座模型进行对齐微调，在无外部人工标注的情况下显著降低 LLM 幻觉。

## 研究问题与动机
1. **核心问题**：LLM 存在"知道但说错"（knowing but telling）的幻觉现象——模型掌握了相关知识，却在推理时生成错误陈述。
2. **现有方法局限一**：基于表征编辑的方法（如 ITI、DOLA）需要领域标注数据，泛化性受限。
3. **现有方法局限二**：基于一致性置信度（consistency-based confidence）的方法（如 FACTTUNE-MC）依赖模型多次生成的一致性，无法直接反映模型内部知识。
4. **关键洞察**：LLM 虽然不一定能准确"生成"正确回答，但有能力基于内部知识"评估"已有回答的事实性，且具备合理预测置信度。

## 核心贡献（创新点）
1. **提出 Self-Alignment for Factuality 框架**：利用 LLM 自评估能力生成事实性奖励信号，通过 DPO 进行对齐训练；与 FACTTUNE-MC 的本质区别在于使用自评估（self-evaluation）而非一致性置信度（consistency-based confidence）来估计事实性。
2. **设计 SELF-EVAL-SKT 自评估组件**：将 True/False Q&A 提示与 few-shot 结合，使模型基于内部知识判断回答的事实性；与原始 SELF-EVAL-P(TRUE) 的本质区别在于通过 SK-TUNING 显著提升了置信度估计精度和校准能力。
3. **提出 SK-TUNING 自知识微调策略**：使用 Wikipedia 和 BIG-bench 异构数据进行对比学习微调，专门增强模型的置信度估计与校准；与 Tian et al. (2023b) 校准方法的本质区别在于面向事实性估计任务设计训练数据（含重复答案以优化校准）。
4. **系统性实验验证**：在 TruthfulQA 和 BioGEN 三个知识密集型任务上验证，较现有最优方法取得显著提升。

## 方法详解

**整体流程（三步）**：
1. **Step 1**：给定 prompt $x$，从基座模型 $\pi_{\text{ref}}$ 生成 $M$ 个候选回答 $\{y_m\}_{m=1}^{M}$（MCQA 跳过此步，直接使用选项）。
2. **Step 2**：通过 SELF-EVAL 评估每个候选回答的事实性，获得置信度分数。
   - 短文本：直接计算 $p(\text{True}|q,a)$。
   - 长文本：先用 GPT-3.5-turbo 抽取原子声明（atomic claims），再转换为原子问题，逐条评估后取平均 $\text{Avg-}p(\text{True})$。
3. **Step 3**：按事实性分数排序，选取 top-$\alpha$ 作为偏好回答 $y_w$，其余作为非偏好 $y_l$，构建偏好对 $\mathcal{D}=\{(x,y_w,y_l)\}$，用 DPO 微调。

**SELF-EVAL 机制**：
- 公式：$p(\text{True}|q,a) = f_{\mathcal{M}}(q,a)$
- 提示格式：few-shot + "请根据问题和内部知识评估答案真实性"，输出 A(True)/B(False)，置信度取 $p(\text{A})$。

**SK-TUNING 训练数据构建**：
- 来源：Wikipedia（49,862 提示）+ BIG-bench（32,500 提示，17 个 MCQA 任务）。
- 对每道题生成 30 个候选答案，用 Deberta-Large-MNLI 双向蕴含判断是否与标准答案语义等价，标注正确/错误答案。
- 构造配对样本 $(q, a, r_+, r_-)$，保留重复答案以辅助校准。

**SK-TUNING 损失函数**：
$$\mathcal{L}_{\phi} = -\mathbb{E}_{(q,a,r_+,r_-)\sim\mathcal{D}_\psi}\left[\log\sigma\left(\log\pi_\phi(r_+|q,a) - \log\pi_\phi(r_-|q,a)\right)\right]$$

**DPO 对齐损失**：
$$\mathcal{L}_{\theta} = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$
其中 $\beta=0.1$，在 8×32G Tesla V100 上微调 5 个 epoch，batch size=8，learning rate=$5\times10^{-6}$。

## 实验与结果
**数据集**：
- TruthfulQA：MCQA（735 题）+ 短文本生成
- BioGEN：长文本传记生成（100 题）
- 额外评估：CommonSenseQA、OpenBookQA、MedQA、MMLU（置信度估计能力）

**评估指标**：TruthfulQA 用 Accuracy、True%、Info%、True*Info%；BioGEN 用 FActScore、#correct/incorrect facts。

**主要结果（LLAMA-7B）**：
- TruthfulQA MC：self-alignment w/ SELF-EVAL-SKT 达到 **45.48%**，较基座（25.60%）提升约 **13%**，优于 FACTTUNE-MC（36.59%）。
- TruthfulQA 短文本生成 True*Info：达到 **45.75%**。
- BioGEN FActScore：达到 **38.28%**，较基座（30.72%）提升约 **4%**。

**主要结果（LLAMA2-7B）**：
- TruthfulQA MC：self-alignment w/ SELF-EVAL-SKT 达到 **44.10%**，True*Info 达 **53.42%**。
- BioGEN FActScore：达到 **46.50%**。

**对比最强结果**：self-alignment w/ SELF-EVAL-SKT 在三项任务上均显著超越所有基线（SFT、ITI、DOLA、FACTTUNE-MC、SE、USC），在 True*Info 和 FActScore 上分别取得最高分。

**置信度估计能力（LLAMA2-7B）**：
- SELF-EVAL-SKT 在 CommonSenseQA 选择任务 Accuracy 达 **70.43%**（基座 54.30%），AUROC 达 **84.65%**（基座 79.76%）。

## 相关工作脉络
1. **representation-editing 方法**（ITI、DOLA）：通过编辑模型内部表征引导事实性，需领域标注数据；本文不依赖任何标注数据，仅用模型自身知识。
2. **FACTTUNE-MC（Tian et al., 2023a）**：用一致性置信度估计事实性并通过 DPO 微调；本文使用自评估替代一致性方法，校准更准确。
3. **HoNESTY-TUNE（Yang et al., 2023）**：面向"承认不知道"的诚实性微调；本文聚焦"在拥有知识时准确表达"的事实性对齐。
4. **后验修正方法**（CoVe、Self-CheckGPT、Self-Refine）：依赖多采样一致性或外部验证器；本文直接训练模型提升内在事实性。
5. **校准研究**（Guo et al., 2017；Tian et al., 2023b）：通用置信度校准方法；本文专门针对事实性评估任务设计 SK-TUNING。
6. **LLM 自我知识检测**（Kadavath et al., 2022）：证明 LLM 能评估自身知识；本文将其扩展到事实性对齐训练。

## 局限性与未来方向
1. **未结合解码策略**：论文指出可与 DOLA 等高性能解码方法结合以获得更好效果。
2. **仅在小模型验证**：仅在 7B 模型上实验，预期在 13B/70B 及 RLHF 微调模型（如 LLAMA2-CHAT）上效果更优。
3. **置信度估计仍有提升空间**：SK-TUNING 虽有效，但可探索更高效的校准方法。
4. **残留错误类型**：分析发现模型仍会在"误导性前提问题"、"模糊问题"、"缺乏精确知识"等场景出错，根源在于预训练数据质量。

## 研究启发与可借鉴点
1. **自评估驱动对齐范式**：将 LLM 的自我知识感知能力直接用于生成训练信号（而非仅用于推理时后处理），为 RLHF/DPO 提供了低成本的替代方案，可迁移到其他对齐任务（如安全性、有用性）。
2. **原子声明分解评估**：长文本生成中通过 GPT 提取原子声明并逐项评估，可将复杂事实性评估转化为可解释的多步问题，值得借鉴于事实核查与可信赖生成任务。
3. **保留重复样本优化校准**：SK-TUNING 中刻意保留重复答案对置信度校准有显著正向作用，这一数据构造技巧对其他需要校准的模型训练具有参考价值。
4. **MCQA→True/False 转化**：将选择题转化为二值判断任务来提取置信度分数，是一种简洁有效的策略，可应用于无需重写 prompt 的通用评估场景。

## 关键术语表
**Hallucination**：LLM 生成看似合理但违背事实信息的现象，本文聚焦"知道但说错"类型。
**Self-Evaluation**：LLM 基于内部知识对自身生成内容进行事实性判断的能力。
**SELF-EVAL**：本文提出的自评估组件，通过 True/False Q&A 提示获取回答的事实性置信度。
**SK-TUNING**：Self-Knowledge Tuning，面向自评估能力优化的微调阶段，提升置信度估计与校准。
**DPO（Direct Preference Optimization）**：无需显式奖励模型的偏好优化算法，直接基于偏好对微调策略。
**FActScore**：基于原子声明的事实精度评估指标，衡量生成文本中正确事实占比。
**Atomic Claim**：从文本中提取的不可再分的事实陈述单元。
**Calibration**：模型预测置信度与真实准确率之间的一致性程度，校准良好意味着高置信度对应高准确率。

## 可复现要素
- **数据集**：TruthfulQA、BioGEN、Wikipedia、BIG-bench（均为公开数据集）
- **代码/权重**：开源，地址为 https://github.com/cuhk-nlp/self-alignment-for-factuality（论文已声明）
- **关键超参**：SK-TUNING 学习率 $5\times10^{-7}$、1 epoch、batch size=8；DPO 学习率 $5\times10^{-6}$、5 epochs、batch size=8、$\beta=0.1$；候选响应数 M=30；偏好比例 $\alpha=30\%$（LLAMA-7B）/ $50\%$（LLAMA2-7B）
- **硬件**：8×32G Tesla V100
