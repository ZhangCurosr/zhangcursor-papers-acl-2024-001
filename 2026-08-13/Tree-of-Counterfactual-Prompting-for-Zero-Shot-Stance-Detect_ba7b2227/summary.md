---
title: "Tree-of-Counterfactual-Prompting-for-Zero-Shot-Stance-Detect"
source: https://aclanthology.org/2024.acl-long.49.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:55:17"
field: "零样本立场检测与多模态推理"
keywords: ["stance detection", "zero-shot", "counterfactual reasoning", "prompting", "multimodal", "large language model"]
innovations: ["首次提出无需任何标注示例的 TR-ZSSD 任务与 ToC 提示框架", "通过显式枚举并接受/拒绝/无立场三种反事实假设的解释与对比验证实现可解释立场推断", "将立场检测从分类范式重新定义为基于反事实推理的可解释生成任务"]
benchmarks: ["SemEval-2016 Task 6 A", "COVAXFRAMES", "MMVAX-STANCE"]
---

# 论文速读：Tree-of-Counterfactual-Prompting-for-Zero-Shot-Stance-Detect

## 一句话总结
本文提出 **Tree-of-Counterfactual (ToC) prompting** 方法，首次实现无需任何标注示例的"白板式"零样本立场检测（TR-ZSSD）；通过在 LLM/LMM 上分别生成 Accept/Reject/No Stance 三种反事实立场假设的链式解释并进行对比验证，最终选出最合理立场。

## 研究问题与动机
- **传统立场检测依赖分类范式**：大多数现有方法将立场检测建模为监督分类任务，在训练集和测试集共享主题下训练/测试，泛化到新主题能力有限（Allaway & McKeown, 2020）。
- **已有 ZSSD 仍需训练数据**：现有零样本立场检测（ZSSD）方法虽能处理未见主题，但仍需立场标注示例进行训练或上下文学习；本文提出 **TR-ZSSD**，完全不需要任何立场标注示例。
- **立场推断需要复杂推理能力**：从社交媒体帖子（SMP）推断立场涉及隐含常识、讽刺、多模态语义融合等复杂认知过程（如 Figure 1 所示），现有分类模型难以显式建模。
- **多模态立场检测基础薄弱**：现有工作主要集中在纯文本立场检测，对同时含文字和图像的多模态 SMP 立场推断缺乏有效方案。

## 核心贡献（创新点）
1. **首次提出 TR-ZSSD 任务与框架**：完全零示例地识别立场，区别于以往 ZSSD 方法仍需训练数据提供立场标注示例的本质差异。
2. **支持文本与多模态联合输入，且支持多种立场对象类型**：同时处理 Topic（话题）和 Frame of Communication（通信框架）两类立场对象，并融合图文内容，而先前方法多仅关注文本 Topic。
3. **首次让 LLM/LMM 生成并验证反事实立场解释**：通过显式枚举所有立场值的反事实假设并逐一解释与对比验证，区别于以往仅依赖单次 CoT 或多数投票的做法。
4. **将立场识别从分类重定义为可解释推理任务**：通过 ToC 框架使模型输出每一步推理依据，增强模型可解释性，不同于黑盒分类器。

## 方法详解
ToC-TR-ZSSD 分为三个阶段：

- **Phase A — 构建反事实立场树（Tree of Counterfactuals）**：对每对（SMP p，立场对象 o），同时构造 Accept、Reject、No Stance 三条分支路径，不依赖先验知识盲目展开所有可能假设。
- **Phase B — Chain-of-Explanation (CoE) 反事实解释**：对每条分支分别提示 LLM/LMM 给出"为何该立场成立"的逐步解释 $e_A, e_R, e_N$。提示模板支持文本和图像，并因立场对象类型（Topic 或 FoC）略有差异。
- **Phase C — Counterfactual Chain-of-Contrastive Verification (C-CoCV)**：以 Phase B 的三条解释为输入，让 LLM/LMM 进行对比验证，逐条分析各假设论证的强弱点，最终输出一条仅含 Accept/Reject/No Stance 中一个的结论。

**关键特性**：全流程零训练样本、无需微调、基于提示工程驱动；CoE + C-CoCV 的组合模拟了人类"多角度审视假设再择优"的认知过程。

## 实验与结果
**数据集**（均在测试集上评估）：
- **SemEval-2016 Task 6 A**：5 个争议性 Topic 的纯文本推文，Accept=304，Reject=715，No Stance=230。
- **COVAXFRAMES**：113 个 COVID-19 疫苗犹豫的 FoC，纯文本 SMP，Accept=1461，Reject=448，No Stance=376。
- **MMVAX-STANCE**：同 113 个 FoC，但包含文本+图像/表情包的多模态 SMP，Accept=578，Reject=332，No Stance=642。

**评估指标**：Accept/Reject 的 Macro P/R/F1。

**关键结果**：
- **SemEval-2016**：GPT-4-ToC 达到 **Macro F1=77.1**，超越最强微调基线 TimeLMs（72.9）超 **4 分**；GPT-4-CoT 为 70.5，GPT-4-Direct 为 67.9，ToC 提升显著。
- **COVAXFRAMES**：GPT-4-ToC 达到 **Macro F1=79.1**，超越最强微调基线 LACRScore（76.2）近 **3 分**；对比 GPT-4-CoT（71.3）提升约 8 分。
- **MMVAX-STANCE（多模态）**：GPT-4V-ToC 达到 **Macro F1=60.6**，较 GPT-4V-CoT（52.6）提升 **8 分**；但未超越最强微调基线 BT+S_All（71.3，依赖 46,606 条人工+合成多模态训练数据）。

**结论**：在纯文本场景下 ToC-TR-ZSSD 已超越所有细调基线；在多模态场景下具备竞争力，但还需进一步探索以提升性能。

## 相关工作脉络
- **Zero-Shot Stance Detection (ZSSD)**：Allaway & McKeown (2020)、Allaway et al. (2021)、Liang et al. (2022) 提出跨主题 ZSSD，但仍需训练集中部分立场示例；本文 TR-ZSSD 完全去示例化。
- **Unsupervised/结构型立场检测**：Murakami & Raymond (2010)、Pick et al. (2022) 基于发言者图结构推断立场；本文基于 LLM 推理而非图结构。
- **Counterfactual 在 NLP 中的应用**：Qin et al. (2019) 故事重写、Zeng et al. (2020) NER、Jacovi et al. (2021) 可解释性、He et al. (2022)/Chen et al. (2023) 提示学习——均聚焦单一任务或样本层面，本文首次在立场检测中用 LLM 显式生成并对比多种反事实立场解释。
- **Chain-of-Thought / Tree-of-Thought**：Wei et al. (2022) 的 CoT、Yao et al. (2023) 的 ToT 均只沿单一路径推进；本文 ToC 明确枚举所有立场假设路径并相互对比。
- **In-Context Learning for Stance**：Zhang et al. (2023a) 使用 CoT 进行立场检测的上下文学习；本文完全去掉示例输入，靠反事实推理解释替代。
- **多模态立场检测基线**：DS-BERT、LES-GAT-MF、BT、BT+S_All 等；本文表明零样本 LMM 经 ToC 也可获得接近微调的性能。

## 局限性与未来方向
- **仅限 Twitter/X 文本与单图**：未覆盖 Reddit 等长文本平台，也未利用 GIF/视频等多模态内容。
- **依赖 LLM 内置文化与道德知识**：模型在英语主流国家表现良好，但在非英语主流国家预测文化/道德规范时准确率下降（Ramezani & Xu, 2023）。
- **安全审查导致的保守偏差**：约 1-2% 的敏感帖子被模型直接拒绝处理（默认 No Stance），尤其是涉及堕胎、疫苗阴谋论等话题。
- **大模型推理能力的本质仍不明确**：文中承认 LLM 是否真正具备"推理"能力仍有争议。
- **未来方向**：扩展多模态（GIF/视频）、多平台（Reddit 等）、跨语言/跨文化 FoC 多样性，并缓解安全保守偏差。

## 研究启发与可借鉴点
1. **"假设-解释-验证"推理范式可迁移**：ToC 的"枚举所有假设 → 各自生成解释 → 对比验证"流程可复用到其他需要多角度论证的任务，如阴谋论检测、虚假新闻辨别、伦理判断等。
2. **反事实显式化避免过早收敛**：与 CoT/ToT 不同，ToC 强制保留所有立场假设，有效避免模型在单条推理链上过早 Commit，这一策略值得在其他生成式推理任务中借鉴。
3. **天然可解释性为后续分析提供素材**：每条分支都有独立解释，可直接用于事后人工审核、错误归因和偏差检测，对面向高风险应用（公共卫生、政治舆情）具有实用价值。
4. **零示例方法可作为强 baseline**：在资源受限场景下（新话题、新领域），TR-ZSSD 可提供无需训练的即时可用方案，可作为后续引入少量示例做 In-Context Learning 的起点。

## 关键术语表
- **TR-ZSSD (Tabula Rasa Zero-Shot Stance Detection)**：完全零样本立场检测，训练与推理阶段均不使用任何立场标注示例。
- **Tree-of-Counterfactual (ToC) Prompting**：通过枚举所有可能立场值并分别生成反事实解释与对比验证的提示框架。
- **Chain-of-Explanation (CoE)**：引导 LLM 对某一立场假设进行逐步解释的提示技术。
- **Counterfactual Chain-of-Contrastive Verification (C-CoCV)**：让 LLM 对比多条反事实解释并选出最合理立场的提示技术。
- **Frame of Communication (FoC)**：以"沟通框架"作为立场对象，用于解释议题成因与道德判断，比单纯 Topic 更能捕捉深层立场。
- **Social Media Posting (SMP)**：指 Twitter/X 等平台上由用户发布的包含文本、图像等多模态内容的帖子。
- **Stance Object**：立场所指向的对象，可以是 Topic（争议话题）或 FoC（沟通框架）。
- **Macro F1**：对 Accept 和 Reject 两个类别分别计算 F1 后取平均，用于衡量两分类立场检测的整体性能。

## 可复现要素
- **数据集**：SemEval-2016 Task 6 A、COVAXFRAMES、MMVAX-STANCE（后两个为作者团队发布，原始推文受 Twitter API/IRB 限制未公开原始文本，可通过 tweet ID 获取）。
- **代码/权重**：GitHub 已开源全部代码、提示词与实验配置（论文声明）。
- **模型**：GPT-3.5 (gpt-3.5-turbo-1106)、GPT-4 (gpt-4-1106-preview)、GPT-4V (gpt-4-vision-preview)、LLaVA-1.5。
- **关键超参**：max generated tokens=1024，temperature=1.0，top-p=0.7；详见附录 A 与 GitHub。
