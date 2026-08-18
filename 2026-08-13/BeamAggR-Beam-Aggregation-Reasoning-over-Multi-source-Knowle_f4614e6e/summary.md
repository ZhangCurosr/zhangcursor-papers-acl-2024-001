---
title: "BeamAggR-Beam-Aggregation-Reasoning-over-Multi-source-Knowle"
source: https://aclanthology.org/2024.acl-long.67.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:55:18"
---

# 论文速读：BeamAggR: Beam Aggregation Reasoning over Multi-source Knowledge for Multi-hop Question Answering

## 一句话总结
本文提出 BeamAggR，一种面向知识密集型多跳问答的推理框架，通过将复杂问题解析为问答树并自底向上进行多源互补推理与概率化束聚合，有效缓解检索不准、多源知识冲突与子问题聚合导致的级联错误，在四个开放域数据集上较现有 SOTA 平均提升 8.5%。

## 研究问题与动机
- **多跳检索信息不全且不准**：单轮检索（One-time Retrieval）依赖原问题直接检索，面对复杂多跳问题易遗漏关键证据或召回无关内容，加剧幻觉。
- **迭代/子问题检索存在级联误差**：IRCoT 等迭代检索的查询与推理对齐困难；ProbTree 等子问题拆解方法在答案聚合时缺乏细粒度融合，错误会逐跳放大。
- **多源知识协同机制缺失**：开放域场景单一知识源不足以支撑复杂推理，现有方法在多源冲突时往往依赖 log-probs 极化选源，无法实现跨源协作。
- **推理轨迹探索空间受限**：传统方法在每个推理跳步通常只维护单一最优路径，难以覆盖复合问题中多种可能的子答案组合。

## 核心贡献（创新点）
- **提出 BeamAggR 分治推理框架**：将多跳问题显式建模为问答树（原子/复合节点），按后序遍历自底向上求解。与迭代检索方法本质区别在于以结构化依赖替代生成驱动检索，避免查询漂移。
- **多源互补细粒度推理**：并行调用闭卷、参数化、Wikipedia 与搜索引擎四类知识源，通过词频投票与温度归一化生成概率化候选答案。与以往单源或粗拼接方法不同，本文实现跨源信息的概率级协作。
- **概率化束聚合（Beam Aggregation）**：在复合节点枚举子问题候选的笛卡尔积，经掩码回填后重新推理并按边缘概率加权聚合，保留 Top-k 轨迹。相比 ProbTree 的 log-prob 极化选择，本文方法显著拓宽探索空间并抑制级联误差。

## 方法详解
- **问题分解（Question Decomposition）**：利用 LLM 将复杂问题 Q 解析为 QDMR 格式的树 $Q_{decomp}$，根节点为原问题，叶子为原子子问题，中间节点为需组合/比较的复合问题。使用 `#i` 占位符标记待填子答案，生成后序序列 $\bar{Q}_{decomp} = \{N^{(1)}, N^{(2)}, \ldots\}$，每个节点 $N^{(i)}=\{q^{(i)}, \pmb{a}^{(i)}, \pmb{p}^{(i)}\}$。
- **多源知识推理**：对原子问题 $q$，分别执行四种策略 $\pmb{a}_s = \text{LLM}(q, K_s)$（$s \in$ closebook, parametric, wiki, serp）。融合阶段经 Vote 去重统计频次 $f_i$，再通过 Softmax 归一化：$p_i = \frac{\exp(f_i/\tau)}{\sum_{j=1}^k \exp(f_j/\tau)}$，输出 Top-$k$ 答案及其概率分布。
- **束聚合推理**：对复合节点，计算子节点候选的笛卡尔积 $\{\langle a_i^{(x)}, a_j^{(y)}\rangle, p_i^{(x)}p_j^{(y)}\}$。对每种组合执行 MaskFill 替换占位符得到新问句，复用多源推理得局部分布。最终按全概率公式聚合：$P(y) = \sum_{q_i \in Q} P(y|x=q_i) \cdot P(q_i)$，保留 Top-$k$ 向上传播至根节点输出最优答案。
- **Algorithm 1 流程**：整体为 `PostOrderTraverse` 驱动的死循环，叶节点执行多源 Vote，中间节点执行 `CartProd`→`MaskFill`→`Vote`→`Aggr`，关键超参为 Beam size、Temperature $\tau$ 与 Self-consistency 采样数。

## 实验与结果
- **数据集与设置**：在 HotpotQA、MuSiQue、2WikiMQA、Bamboogle 四个开放域多跳 QA 数据集上评测，使用 token-level F1。主实验基于 GPT-3.5-turbo，另在 Mistral-7B 验证泛化性。检索采用 October 2017 Wikipedia BM25 + Serper Google Search API。
- **最强结果**：BeamAggR 全面超越 SOTA（ProbTree）：HotpotQA 62.9（↑2.5）、MuSiQue 36.9（↑4.0）、2WikiMQA 71.6（↑3.7）、Bamboogle 74.8（↑8.2），平均提升 **8.5%**。Comp 类问题提升最显著（65.9→74.2 / 81.7→89.9）。
- **推理深度鲁棒性**：随跳数增加性能衰减更缓（3-hop ↓8.4%，4-hop ↓12.3%），证明束聚合有效缓解级联错误。
- **消融与效率**：移除任一知识源均导致下降；概率聚合优于 log-prob 极化（2WikiMQA 71.6 vs 65.9）与等权聚合（↓5%）；贪心聚合（Greedy Aggregation, k=1）在几乎不损性能前提下将平均 Token 消耗大幅降低。

## 相关工作脉络
- **OneR / Self-Ask / IRCoT / FLARE**：单轮或迭代检索增强方法，依赖 LLM 生成内容驱动下一轮检索，查询与推理对齐不稳定，且多源拼接粗糙；本文以结构化子问题树替代生成驱动检索，从源头提升查询精准度。
- **ProbTree**：同样采用问题树分解与自底向上推理，但答案聚合依赖 log-probs 极化，易锁定单一知识源并放大误差；本文引入候选束枚举与边缘概率加权，实现细粒度协作与轨迹优选。
- **Beam Retrieval / Graph-guided Reasoning**：侧重端到端检索策略或图结构构建；本文聚焦推理阶段的聚合机制设计，强调多跳答案空间的概率化探索而非单纯检索优化。
- **CoT / Self-Consistency / Tree-of-Thought**：通用推理增强范式；本文将其迁移至开放域多跳 QA 场景，结合多源检索管道与问题依赖树，形成任务定制化的聚合推理闭环。

## 局限性与未来方向
- **推理开销较高**：束组合与多源自一致性带来多次 LLM 调用，虽贪心聚合可缓解，但在深树或高 beam size 下仍受限。
- **强依赖问题分解质量**：结构错误的分解会导致后续推理失效，当前仅依赖 prompt 与后过滤，缺乏自动校验/修复机制。
- **知识源形态有限**：仅整合非结构化文本（闭卷、Wiki、搜索），未接入知识图谱等结构化存储，限制对强关系型多跳问题的覆盖。
- **未来方向**：引入结构化知识库检索、设计分解结果的纠错模块、探索动态束宽与自适应终止策略以进一步压降推理成本。

## 研究启发与可借鉴点
- **多源频次软归一化**：将不同来源答案的词频经温度 Softmax 映射为概率，可平稳融合异构信源，避免 hard voting 的信息丢失，适用于多证据辩论
