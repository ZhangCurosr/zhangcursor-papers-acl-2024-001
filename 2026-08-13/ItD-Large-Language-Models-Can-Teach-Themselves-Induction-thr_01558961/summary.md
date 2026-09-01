---
title: "ItD-Large-Language-Models-Can-Teach-Themselves-Induction-thr"
source: https://aclanthology.org/2024.acl-long.150.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 07:14:58"
---

# 论文速读：ItD-Large-Language-Models-Can-Teach-Themselves-Induction-thr

## 一句话总结
本文提出 ItD（Induction through Deduction）框架，利用大语言模型擅长的演绎推理自监督生成训练数据，并结合朴素贝叶斯解码策略对模型进行微调与推理优化，从而将 LLM 从“弱归纳器”转变为“强归纳器”，在 Instruction Induction 和 List Function 两个基准上分别超越此前最强方法约 35% 和 10%。

## 研究问题与动机
- LLM 在归纳推理（从若干 $(x,y)$ 样本对中发现潜在共享变换规则 $f$）方面存在天然短板，直接 few-shot 诱导效果有限。
- 现有改进方法（如 Hypothesis Search / Hypothesis Search & Refinement）本质上仅是对 LLM 原始输出假设的“后处理”筛选与迭代修正，仍未突破模型内在归纳能力的瓶颈。
- 已有研究定量证实 LLM 的演绎推理（给定规则 $f$ 与输入 $x$ 推导输出 $y$）能力显著强于归纳，且两者共享 $x,y,f$ 三元组推理结构，为“以演绎教归纳”提供了理论动机。
- 实验发现沿用旧有 IO Prompt 直接微调演绎生成的数据效果不佳，必须设计新型单样本提示与概率组合策略，才能真正激活生成数据的训练价值并随样本数增长持续受益。

## 核心贡献（创新点）
- 提出 ItD 整体框架，首次系统性构建“演绎数据生成 → 单样本微调 → 贝叶斯联合解码”的自我训练闭环，使 LLM 无需外部人工标注即可提升归纳能力。
- 设计 Deductive Data Generation 模块，基于联合分布分解 $p(x,y,f)=p(y|x,f)p(x|f)p(f)$ 自监督构造 $(x,y,f)$ 三元组，完全依赖模型自身演绎能力，不依赖更大模型或人类反馈。
- 提出 Naive Bayesian Induction 模块，引入 GD（Group Decoding）Prompt 替代传统 IO Prompt 进行单样本微调，并推导 NBGD 解码算法，将多样本联合后验等价分解为单样本预测概率连乘与先验修正项。
- 证明 ItD 不仅能绝对超越 HS/HS&R 等后处理基线，且随着观测样本数增加性能持续稳定上升，而对比方法几乎无增益，揭示了框架对信息利用率的本质的提升。

## 方法详解
- **Deductive Data Generation**：首先在初始归纳集 $\mathcal{D}_{in}$ 上以采样解码模式让 LLM 生成若干变换规则 $f$（近似先验 $p(f)$）；随后针对每个 $f$，提供固定 Few-shot 演绎示例，让 LLM 按指令依次生成输入 $x_i$ 与对应输出 $y_i$（即 $p(y|x,f)p(x|f)$），解析文本后得到 $(x,y,f)$ 训练对。
- **Naive Bayesian Induction 训练**：将生成数据按两种格式组织：IO Prompt（多对拼接）用于 ItD-IO 变体；GD Prompt（仅含单个 $x,y$）用于完整 ItD。使用 LoRA / QLoRA 对基础模型微调（学习率 1e-4，3 epochs）。
- **Naive Bayesian Group Decoding (NBGD)**：推理时假设给定规则 $f$ 下各观测对条件独立，由贝叶斯公式推导得：$p(f|\{x_i,y_i\}_{i=1}^n) \propto p(f)^{-(n-1)} \prod_{i=1}^n p(f|x_i,y_i)$。解码过程中，将 $n$ 个单样本 GD 提示组成 batch 输入模型，在 beam search 的每一步 token 生成时累加所有样本对该 token 的 log-prob；首轮解码完成后，再以先验项 $-(n-1)\log p(f)$ 对 top-k 候选规则进行 rerank，得到最终 $f^*$。

## 实验与结果
- **数据集与设置**：常识归纳采用 Instruction Induction（24 子任务，基础模型 Llama-2-7b-chat）；符号归纳采用 List Function（250 子任务，基础模型 Mixtral-8x7B）。为公平评估，所有方法的规则执行统一由 ChatGPT 充当 Reasoner 完成。
- **主结果**：ItD 在 Instruction Induction 上取得 38.70，相对基线 IO 提升 193%，相对此前最强 HS&R 提升 35%；在 List Function 上取得 21.60，相对 IO 提升 16%，相对 HS&R 提升 10%，均显著超越 IO、SC、HS、HS&R 四类基线。
- **模块消融**：ItD-IO（仅用演绎数据+IO微调）较基线仍有 146%（Instr）和 8%（List）提升，验证数据生成有效；完整 ItD 较 ItD-IO 进一步提升，验证 NBGD 有效。
- **样本数量敏感性**：当单批次样本从 2 增至 20 时，ItD-IO 性能几乎持平，而 ItD 持续显著上升（Instr +3.86%，List +2.77%），证明 NBGD 能真正利用更多观测信息。
- **规模与泛化**：ItD 同样适用于 Llama-2-13b-chat 与 ChatGPT（后者因接口限制仅验证 ItD-IO，仍达 44.64 / 29.59）；引入更强演绎器 ChatGPT 辅助生成数据后（+D），ItD+D 分别达 41.01 和 23.91。OOD 实验表明框架具备一定跨任务迁移能力，但性能衰减程度取决于子任务间规则 $f$ 的语义相似度。

## 相关工作脉络
- **Hypothesis Search & Refinement（HS/HS&R）**：通过 Python 执行假设并基于观测反馈筛选/迭代，属后处理范式；ItD 直接通过微调重塑模型底层归纳表征，不依赖外部执行循环。
- **Memory-Oriented Induction**：为 LLM 配置工作记忆以存储规则；ItD 可作为即插即用的能力增强模块，提升此类方法在规则诱导阶段的基础表现。
- **Naive Bayes-based Context Extension（NBCE）**：将朴素贝叶斯用于文档 QA 的上下文拼接，但文档间常存在耦合导致推断偏差；ItD 将其迁移至归纳任务，充分利用规则给定下样本条件独立的天然性质，并结合微调实现端到端优化。
- **LLM 推理能力定量评估（Bang et al., Tang et al. 等）**：证实 LLM 演绎强于归纳；本文以此为动机起点，构建“以演代归”的训练闭环而非仅停留在评测层面。
- **Instruction Induction / List Function 评测体系**：作为本领域标准基准，本文方法刷新多项指标，为后续规则发现、程序综合与抽象推理研究提供新基线。

## 局限性与未来方向
- 受限于 LLM
