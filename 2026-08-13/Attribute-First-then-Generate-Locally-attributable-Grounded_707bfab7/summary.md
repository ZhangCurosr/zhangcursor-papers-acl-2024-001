---
title: "Attribute-First-then-Generate-Locally-attributable-Grounded"
source: https://aclanthology.org/2024.acl-long.182.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:54:30"
field: "可归因文本生成"
keywords: ["归因文本生成", "本地可归因", "多文档摘要", "长形式问答", "幻觉缓解", "文本规划"]
innovations: ["首次提出本地可归因文本生成任务，实现句子级细粒度引用", "Attribute First then Generate三步分解框架，使内容选择天然成为归因", "揭示prefix对长文本连贯性的累积增益效应"]
benchmarks: ["Multi-News", "DUC/TAC", "LFQA (Liu et al. 2023)"]
---

# 论文速读：Attribute-First-then-Generate-Locally-attributable-Grounded

## 一句话总结
本文提出了"Attribute First, then Generate"框架，将带归因的文本生成过程分解为内容选择、句子规划和逐句生成三步，首次实现句子级细粒度引用，使人工事实验证时间减少50%以上，且不影响生成质量。

## 研究问题与动机
1. **现有归因方法过于粗糙**：当前归因文本生成方法（如ALCE、GopherCite）通常指向整个文档或段落，无法提供细粒度的引用支撑。
2. **事实验证成本高**：粗粒度引用迫使人工评估者大量筛选无关信息，导致验证效率低下。
3. **缺乏系统性分解思路**：端到端生成难以保证每句输出都有精确来源支撑，现有方法多在生成后追加归因，无法确保紧密对应。
4. **需要同时满足完整覆盖与简洁性**：理想的归因应做到每句话都有充分证据支撑（full coverage），同时引用片段尽可能精炼（conciseness）。

## 核心贡献（创新点）
1. **首次定义"本地可归因文本生成"任务**：提出输入（源文本片段）和输出（句子）双粒度标准，要求每个输出句子由最小必要信息片段支撑。
2. **提出"Attribute First, then Generate"范式**：将生成分解为内容选择→句子规划→逐句生成三步，使选定的内容片段天然成为归因来源，而非事后追加。
3. **提供ICL与Fine-tuning双实现路径**：分别设计了基于提示的学习方案与PRIMERA微调方案，并探索CoT变体和联合训练策略。
4. **实证显著的归因简洁性提升**：相比ALCE基线，本文方法引文平均短45倍；人工验证时间降低近50%，同时在LFQA上AUTOAIS和AIS均超越基线。
5. **揭示prefix对连贯性的增益机制**：通过消融实验证明，将前文作为输入可提升约30%句子的连贯性，且该效应随输出推进而增强。

## 方法详解
**整体框架**：将端到端生成拆解为三个串行阶段：
1. **内容选择**（Content Selection）：从源文档中提取与任务相关的关键片段（spans/highlights）。MDS场景下关注显著性信息，LFQA场景下关注能回答问题的事实片段。通过标记`<SPAN_DELIM>`分隔，并使用字符串匹配定位回原文。
2. **句子规划**（Sentence Planning）：将选中的片段聚类为有序的高亮簇（highlight clusters）$C_1, \ldots, C_n$，每个簇内的片段将在同一输出句子中融合。
3. **逐句生成**（Sentence-by-sentence Generation）：按规划顺序，利用上一句的所有前缀$s_{1:i}$与当前簇$C_{i+1}$，最大化$p(s_{i+1}|s_{1:i}, C_{i+1})$，依次生成句子。

**ICL实现**：使用Gemini-Pro模型，为每个子任务提供1-4个few-shot示例和特定指令。引入`<highlight_start>`和`<highlight_end>`标记已选片段。

**Fine-tuning实现**：使用PRIMERA模型（447M参数），各阶段设计专用的输入输出格式和特殊标记（`<highlights_separator>`, `<cluster_separator>`等）。在内容选择和规划阶段采用受限解码（constrained decoding），通过logits processor强制模型只生成源文档中存在的n-gram，保证词汇层面的严格忠实性。

**CoT变体**：将句子规划与生成合并，让模型先输出推理/规划链（包含高亮聚类决策），再生成最终句子，建立输出句子与规划阶段高亮的映射关系。

## 实验与结果
**数据集**：
- MDS：DUC、TAC基准及Multi-News数据集（含sentence-source对齐标注，来源Ernst et al. 2022, 2024）
- LFQA：Liu et al. (2023)的LLM回答+人类标注对齐数据，经"Full Support"过滤后保留高质量样本

**评估指标**：ROUGE-L、BERTScore（生成质量）；AUTOAIS（基于TRUE模型的NLI自动归因评估）；LENGTH（引用片段平均token数）；NO ATT.（无归因句子比例）；人工评估测量Fluency、Helpfulness、AIS及验证时间

**主要结果**：
- **MDS**（ICL）：ALCE的LENGTH为843.6 tokens，本文ATTR.FIRST仅92.7（缩短约9倍）；NO ATT.从3.4%降至0%
- **LFQA**（ICL CoT）：ROUGE-L达38.6（超越ALCE的35.2），AUTOAIS达89.3（超越ALCE的49.8），LENGTH仅48.2
- **人工评估**（MDS）：验证时间从ALCE的47秒降至22秒（降53%）；LFQA验证时间从59秒降至35秒
- **Prefix效应**：包含前缀使约30%的句子连贯性更好（23%提升指代关联，7%提升话语连接词使用）

## 相关工作脉络
1. **ALCE**（Gao et al., 2023b）：通过内联引用同步生成文本与归因，但引用粒度限于整个文档；本文在其基础上实现片段级细粒度归因。
2. **Attributed QA**（Bohnet et al., 2023）：事后归因方法，通过检索文档并匹配回答片段确定来源，存在组合搜索难题；本文通过生成前置化解该问题。
3. **SEMQA**（Schuster et al., 2023）：半抽取式LFQA数据集，强制模型直接复制源文本片段并用自由文本连接；本文方法更灵活，支持完整抽象生成。
4. **Text Planning**（Gehrmann et al., 2018; Narayan et al., 2021, 2023）：通过内容选择+句子融合改进摘要质量；本文引入归因目标，使规划阶段天然携带引用信息。
5. **Ernst et al. (2022)**：命题级聚类用于多文档摘要，但输出为单命题短句列表；本文保持自然流畅的多命题句子生成。
6. **Chain-of-Thought**（Wei et al., 2022）：启发本文的CoT变体，将规划与生成合并于单次调用，适用于ICL场景。

## 局限性与未来方向
1. **计算开销增加**：多步串行调用比端到端生成慢且消耗更多token，需折衷效率与归因质量。
2. **误差传播风险**：多阶段流水线可能放大前期错误，尽管各组件具备一定鲁棒性可部分缓解。
3. **微调效果不及ICL**：实验发现微调模型（PRIMERA）在质量与归因准确性上逊于大型ICL模型（Gemini），暗示当前小规模模型难以充分拟合分阶段任务。
4. **未探索其他任务**：目前仅在MDS和LFQA验证，方法可扩展性待更多场景检验。
5. **MDS归因精度略低于ALCE**：约42%被判定为"无支撑"的句子实际含部分支撑信息（位于高亮附近），说明句子融合可能引入超出选区的信息。

## 研究启发与可借鉴点
1. **"生成即归因"的范式转换**：将归因从后处理问题转化为生成过程的内在约束，通过内容选择阶段天然锁定引用来源，思路可迁移至其他需要可追溯性的生成任务（如代码生成、法律文书撰写）。
2. **受限解码在抽取任务中的应用**：通过logits processor强制n-gram匹配，为需要词汇忠实性的生成子任务提供了可复用的技术路线。
3. **Prefix增强连贯性的实证发现**：证明了前缀输入对跨句连贯性的累积增益，提示在长文本生成中逐步累积上下文的策略值得借鉴。
4. **自动归因评估指标的元评估方法**：通过Kendall-Tau和Spearman相关性与人工标注对比，系统比较了TRUE与TrueTeacher模型，为归因评估选择提供了方法论参考。
5. **多阶段分解+ICL的组合策略**：展示了将复杂任务分解后用大模型ICL实现的可行性，可作为大模型应用工程的一种范式。

## 关键术语表
**Locally-attributable Grounded Text Generation**：本地可归因文本生成，指生成文本中包含对源文本中具体片段的细粒度引用，同时满足完整覆盖与引用简洁性。
**Content Selection**：内容选择，从源文档中提取与生成目标相关的关键文本片段（spans/highlights）的步骤。
**Sentence Planning**：句子规划，将选中的文本片段聚类为有序簇，决定哪些片段将融合到同一输出句子中。
**AUTOAIS**：自动归因完整性评分，基于NLI模型（TRUE）判断每句输出是否被其引用片段完全支持的自动化指标。
**Constrained Decoding**：受限解码，通过logits processor为不允许生成的token分配负无穷概率，强制模型输出受约束的文本结构。
**CoT（Chain-of-Thought）**：思维链，在此文中指将句子规划与逐句生成合并为单次调用的变体，模型先输出推理链再产生最终文本。
**AIS（Attributable to Identified Sources）**：人工归因支持度评分，由 annotator 逐句判断输出句子是否被其引用片段完全支撑的二元指标。
**PRIMERA**：Pyramid-based Masked Sentence Pre-training for Multi-document Summarization，一种面向多文档摘要的Longformer Encoder-Decoder模型（447M参数）。

## 可复现要素
- **数据集**：MDS使用DUC、TAC、Multi-News（含对齐标注）；LFQA使用Liu et al. (2023)公开数据集（MIT许可）。对齐数据来源于Ernst et al. (2022, 2024)。
- **代码/权重**：论文未提供代码开源声明
- **模型**：ICL使用Gemini-Pro（via API）；Fine-tuning使用PRIMERA（447M参数）
- **关键超参**：ICL温度T=0.1~0.7，few-shot示例数1-4；FT训练10 epochs，learning rate 1e-4~5e-4，warmup 0-300 steps，weight decay 0-0.5；GPU：Nvidia A100 80GB，每模型训练约1小时
- **特殊机制**：受限解码最小highlight长度3 words，MDS最少30个highlights，LFQA最少5个highlights，每个cluster最多2个highlights
