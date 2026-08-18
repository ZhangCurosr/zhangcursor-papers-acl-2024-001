---
title: "Active-Prompting-with-Chain-of-Thought-for-Large-Language-Mo"
source: https://aclanthology.org/2024.acl-long.73.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:53:30"
field: "大模型提示与推理"
keywords: ["chain-of-thought", "active learning", "prompt selection", "large language models", "uncertainty estimation", "self-consistency"]
innovations: ["提出不确定性驱动的主动示例选择策略，用于 Few-shot CoT 构建", "设计并对比 Disagreement/Entropy/Variance/Self-confidence 四类不确定性度量", "证明所选示例在不同 LLM 间可迁移且不确定性来源于任务"]
benchmarks: ["GSM8K", "ASDiv", "SVAMP", "AQuA", "SingleEq", "CSQA", "StrategyQA", "Letter (4)"]
---

# 论文速读：Active-Prompting-with-Chain-of-Thought-for-Large-Language-Mo

## 一句话总结
本文提出 **Active-Prompt** 方法，通过不确定性度量从任务查询池中自动挑选最需标注的问题作为示例，显著提升大型语言模型在复杂推理任务上的 Few-shot 链式思维（CoT）表现。

## 研究问题与动机
- 现有 CoT 提示依赖人工预定义的固定示例集，无法保证对所测任务最为有效。
- 不同推理任务在难度、范围与领域上差异较大，难以通过人工经验决定哪些题目最值得标注。
- 标注成本可控（仅需少量示例），关键在于如何用低成本策略选出最具信息量的问题。
- 已有工作缺少主动学习思想，未能系统化回答“从池子中选哪些题做示例”这一核心问题。

## 核心贡献（创新点）
- 提出基于不确定性的主动示例选择策略，首次将主动选择引入 CoT 提示构建中。
- 设计了多种用于 LLM 的样本不确定性度量（Disagreement/Entropy/Variance/Self-confidence），用于量化问题对模型的困难程度。
- 以少量人工标注换取显著性能提升，并在算术/常识/符号推理八项基准上持续优于主流基线。
- 证明所选题例可在不同模型间迁移，且不确定性来源于任务本身而非单一模型。

## 方法详解
- 流程分为四步：不确定性估计 → 选择最不确定 n 题 → 人工标注 CoT → 作为 Few-shot 示例进行推理。
- **Disagreement**：对同一问题采样 k 次答案，统计不同答案数量后归一化，$u = h / k$，越分散越不确定。
- **Entropy**：按答案出现频率构建分布并计算信息熵，选熵最大的问题。
- **Variance**：对数值型答案计算方差，并对大数支配问题做归一化（除以题干数值之和）。
- **Self-confidence**：用模板让模型自评置信度（very confident/wrong answer 等），但因 LLM 易过度自信而效果不佳。
- 推理时采用 Self-consistency 多次采样并以多数一致答案输出；候选池上限设为 1000，k 默认 10。

## 实验与结果
- 数据集：GSM8K、ASDiv、SVAMP、AQuA、SingleEq、CSQA、StrategyQA、Letter (4)。
- 基线：CoT、Self-consistency、Auto-CoT、Random-CoT。
- 主要指标：Exact match accuracy。
- 最强结果（code-davinci-002）：Active-Prompt (E) 在 GSM8K 达 83.4、ASDiv 89.3、SVAMP 87.5、AQuA 57.0、SingleEq 95.5、CSQA 78.8、StrategyQA 80.6、Letter (4) 76.7，平均约 **81.6%**；较 SC（79.1）平均提升约 **2.1–2.5pp**。
- text-davinci-002 上 Active-Prompt (D) 较 SC 平均提升 **7.0pp**，显示主动选择在更弱模型上增益更大。
- gpt-3.5-turbo 上 Active-Prompt (D/E) 同样全面优于 CoT，说明方法对不同底座均有效。

## 相关工作脉络
- Chain-of-Thought Prompting（Wei et al.）：提供人工示例推理路径；本文改进的是“哪些示例值得人工标注”，而非模板本身。
- Self-Consistency（Wang et al.）：通过多次采样提升稳定性；本文与其正交，可组合使用。
- Auto-CoT（Zhang et al.）：基于聚类/多样性的自动示例构造；本文侧重不确定性选择并配合人工高质量标注。
- Zero-Shot-CoT（Kojima et al.）：仅靠 "Let's think step by step" 即可触发推理；本文验证可脱离人工初始示例。
- Active Learning（Settles; Roy & McCallum 等）：用不确定性选择高价值标注样本；本文将其迁移到 LLM 少样本示例选择。
- Complexity-based selection（Complex-CoT）：按提示复杂度选例；本文表明不确定性选择更优。

## 局限性与未来方向
- 受限于 API 成本，部分更强模型（如 GPT-4）及更多重复实验未充分展开。
- code-davinci-002 已下线，复现存在客观障碍。
- Self-confidence 因 LLM 过度自信效果差，需引入校准或外部判别器。
- 仅使用少量人工标注，自动标注（如 zero-shot 生成 + 校验）仍有待探索。
- 多样性与不确定性的联合选择尚未系统化验证。

## 研究启发与可借鉴点
- 将主动学习的不确定性度量直接迁移到 LLM 示例选择，思路简洁且效果稳定，值得复用。
- 提出 Disagreement/Entropy/Variance 三种通用量化方式，可推广到其它需要示例筛选的任务（如检索增强、评测锚点选择）。
- 演示跨模型迁移：不同模型选出的难题具有任务层面的共性，提示工程可考虑“模型无关”的示例资产。
- Zero-shot 起始也能获得接近结果，为低资源场景（无法准备初始 few-shot）提供可行性。
- 可结合本团队的指令微调/评测管线，作为轻量提示优化的通用模块。

## 关键术语表
- **Active-Prompt**：通过不确定性主动选择少量高质量 CoT 示例的提示方法。
- **Chain-of-Thought (CoT)**：在示例中加入中间推理步骤以提升 LLM 复杂推理能力。
- **Self-consistency (SC)**：对同一问题多次采样推理路径并取一致答案。
- **Disagreement**：基于 k 次预测答案不重合比例衡量不确定性的指标。
- **Entropy**：基于答案分布信息熵衡量不确定性的指标。
- **Variance**：针对数值型答案计算方差并用题干数值归一化的不确定性指标。
- **Pool size**：用于不确定性估计的候选样本集合规模。
- **OOD Letter (4)**：测试集为 4 字母拼接、提示仅 2 字母的外推符号任务。

## 可复现要素
- 数据集：使用公开数据集 GSM8K、ASDiv、SVAMP、AQuA、SingleEq、CSQA、StrategyQA、Letter (4)。
- 代码/权重：论文未提供开源代码或权重，主要通过 OpenAI API 调用完成实验。
- 关键超参：示例数（任务相关，4–8）、候选池上限 1000、采样次数 k=10、推理温度 T=0.7、推理次数 40；不确定性主要报告 Disagreement/Entropy。
