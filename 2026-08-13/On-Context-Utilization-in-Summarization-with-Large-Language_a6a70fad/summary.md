---
title: "On-Context-Utilization-in-Summarization-with-Large-Language"
source: https://aclanthology.org/2024.acl-long.153.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:46:26"
---

# 论文速读：On-Context-Utilization-in-Summarization-with-Large-Language

## 一句话总结
本文首次系统研究了大型语言模型在抽象摘要任务中的上下文利用率与位置偏见，证实了LLM普遍存在“U型性能曲线”（中间诅咒），并构造了MiddleSum基准及分层/增量两种推理策略以缓解中部信息利用不足的问题。

## 研究问题与动机
1. **核心问题**：LLM能否有效利用整个上下文窗口进行摘要生成，还是仅依赖首尾段信息？
2. **现有工作不足**：前期研究（如Liu et al., 2023a）在QA与键值检索中发现了U型位置偏见，但抽象摘要因高度改写与压缩，事实映射困难，该偏差在生成任务中的表现与影响尚未被系统量化。
3. **窗口扩展的悖论**：YaRN、位置插值等技术宣称支持100k+ token，但若模型无法访问上下文中部，盲目延长窗口对摘要任务的实际收益存疑。
4. **评测工具缺失**：缺乏专门针对“重要信息集中于中段”场景的自动化评估基准，难以客观衡量位置偏见对生成质量的影响。

## 核心贡献（创新点）
1. **首次大规模验证摘要任务的位置偏见**：通过6个LLM、10个数据集、5种评估指标的系统实验，证明摘要任务同样存在U型性能曲线，打破了“位置偏见仅存在于检索/QA”的假设。
2. **提出MiddleSum基准**：从长输入数据集中筛选起始位置≥1200词且无首部显著信息的样本，构造225条挑战性子集，实现对中间诅咒的自动化定量评估。
3. **设计并评测两种缓解推理策略**：在MiddleSum上对比分层摘要与增量摘要，发现二者在科学论文域可显著缓解位置偏见，且计算复杂度从$\mathcal{O}(l \cdot n^2)$降至$\mathcal{O}(l \cdot n \cdot m)$。
4. **划定上下文长度的实用边界**：实验表明在现有推理框架下，开源模型窗口超过4k token后性能 plateau 或下降，质疑了当前盲目扩容的工程合理性。
5. **解耦生成质量与信息忠实度**：通过细粒度属性评估发现，尽管LLM输出的连贯性与整体质量保持稳定，但在中间位置插入的文档会显著损害“可归因性”与“信息量”。

## 方法详解
1. **显著性位置映射**：将生成摘要中的唯一bigram在源文档中定位，或将文档切分为20/10个等宽bin统计匹配比例；对长输入，先用ROUGE-1 F1贪心对齐源句与摘要句，再映射到可见上下文区间。
2. **位置-性能相关性量化**：计算参考摘要对齐源句的均值位置与各项指标间的Spearman相关系数，系数绝对值越大说明性能对信息位置越敏感。
3. **受控多文档扰动实验**：在Multi-XScience（7文档）与Multi-News（5文档）上，将唯一相关文档依次置于不同位置（掺入无关文档），或在首尾保留文档、中部填充随机噪声，隔离中间位置效应。
4. **MiddleSum构建规则**：从Arxiv、PubMed、GovReport、SummScreenFD（各50条）与Multi-News（50条）中按显著性对齐筛选，确保关键信息起始位置≥1200词，共225条。
5. **分层与增量推理公式**：
   - 分层：$y_i = \text{LLM}(x_i),\ \forall i \in \{1,\dots,k\}$；$y = \text{LLM}(y_1,\dots,y_k)$
   - 增量：$y_i = \text{LLM}(y_{i-1}, x_i),\ y = y_k$
   - 块大小$m=1500$词，在句末截断保连贯；复杂度均为$\mathcal{O}(l \cdot n \cdot m)$。
6. **Focus Prompt基线**：添加指令“Please also pay attention to the middle section...”，验证纯文本提示对位置偏见的缓解能力。

## 实验与结果
- **数据集与模型**：覆盖5个标准长度数据集（CNN/DM、XSum、Reddit-TIFU、SAMSum、Multi-XScience）与5个长输入数据集（Arxiv、PubMed、GovReport、SummScreenFD、Multi-News）；评测Flan-UL2、Llama-2-7B/13B、Xgen-7
