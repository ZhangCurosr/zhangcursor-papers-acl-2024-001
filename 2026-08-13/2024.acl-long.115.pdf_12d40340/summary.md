---
title: "REANO: Optimising Retrieval-Augmented Reader Models through Knowledge Graph Generation"
source: https://aclanthology.org/2024.acl-long.115.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:01:00"
field: "开放域问答"
keywords: ["open domain question answering", "retrieval-augmented generation", "knowledge graph generation", "graph neural network", "Fusion-in-Decoder"]
innovations: ["从检索段落自动生成知识图谱三元组，解决现有KG不完整问题", "利用GNN进行问题相关三元组筛选并拼接为额外输入增强FiD生成"]
benchmarks: ["Natural Questions", "TriviaQA", "EntityQuestions", "2WikiMultiHopQA", "MuSiQue"]
---

# 论文速读：REANO: Optimising Retrieval-Augmented Reader Models through Knowledge Graph Generation

## 一句话总结
论文提出 REANO（REtrieval-Augmented generative readers with a kNOwledge graph generation module），通过从检索段落中自动生成知识图谱三元组，并结合 GNN 选择与问题最相关的三元组作为额外输入，解决现有 KG-enhanced 方法依赖不完整外部 KG 的问题，有效提升了 FiD 等检索增强读者模型在开放域问答任务上的性能。

## 研究问题与动机
1. **FiD 忽略段落间语义关系**：FiD 独立编码每个检索段落（Enc(q, d_i)），无法捕捉段落间的依赖关系，对需要多跳推理的问题（如比较类、关系链类问题）处理能力不足。
2. **现有 KG 增强方法依赖不完整的静态 KG**：KG-FiD、GRAPE、OREOLM 等工作直接使用 Wikidata 等外部 KG 的三元组，但这些 KG 往往存在严重的不完整性（incompleteness），例如缺少"生日"等关键信息导致无法正确回答比较问题（如"谁更年轻：Steven Spielberg 还是 James Cameron？"）。
3. **段落中蕴含的三元组信息未被充分利用**：即使检索到的段落包含回答问题所需的关键事实，现有方法也未能将这些信息结构化并显式建模为知识图谱三元组。

## 核心贡献（创新点）
1. **提出 REANO 框架，引入 KG 生成模块**：与现有工作直接复用 Wikidata 三元组不同，REANO 从检索段落中自动生成结构化知识图谱（KG），确保三元组信息来源于段落本身，解决了外部 KG 不完整的问题。
2. **KG 生成器分解为实体识别（ER）与关系提取（RE）两个子任务**：ER 使用 SpaCy + TAGME 识别 Wikipedia 实体；RE 进一步分解为段落内关系提取（DocuNet 模型）和跨段落关系提取（Wikidata API 获取），两者合并形成完整的 KG。
3. **引入 GNN 进行 top-K 相关三元组选择**：利用关系感知 GNN 在生成的 KG 上进行多跳推理，计算三元组与问题的相似度，选择 top-K 最相关的三元组拼接为额外输入段落，增强 FiD 的答案生成能力，同时减轻推理负担。
4. **五数据集实验验证有效性**：在 NQ、TQA、EQ、2WQA、MuSiQue 五个 ODQA 数据集上，REANO 平均提升 EM 1.8%，最高提升 2.7%（EQ 数据集）。

## 方法详解
**整体框架**：REANO 由 KG Generator 和 Answer Predictor 两部分组成，采用变分推断形式建模：$p_\theta(a|q, \mathcal{D}_q) = \sum_{\mathcal{G}_q} p_\theta(a|q, \mathcal{D}_q, \mathcal{G}_q) \cdot p_\phi(\mathcal{G}_q|\mathcal{D}_q)$，其中 $\mathcal{G}_q$ 为隐变量。

**KG Generator**：
- **ER（实体识别）**：用 SpaCy 识别命名实体，再用 TAGME 链接到 Wikipedia 实体，得到实体集合 $\mathcal{E}_q$。
- **Intra-Context RE（段内关系提取）**：使用 DocuNet 模型，在每个段落 $d_i$ 内对实体对 $(e_1, e_2)$ 进行关系预测，公式：$\prod_{d_i \in \mathcal{D}_q} p_\phi(\mathcal{T}_{q,d_i}^I|d_i, \mathcal{E}_{q,d_i})$。
- **Inter-Context RE（跨段关系提取）**：通过 Wikidata API 获取跨段落实体之间的关系三元组，即使这些关系未在段落中明确提及。
- 最终 KG：$\mathcal{G}_q = \{\mathcal{T}_q^I, \mathcal{T}_q^C\}$。

**Answer Predictor**：
- **GNN 层实体/关系嵌入初始化**：实体嵌入通过 T5 encoder 获取段落 token 嵌入后 mean-pool 特殊 token `<e>` 得到：$\bar{t_e} = \frac{1}{N_e}\sum_{j=1}^{N_e} m_{e,j}$。关系嵌入初始化为关系标签 token 嵌入的 mean-pool，再通过 REM（Relation Embedding Module，两层 FFN）结合实体嵌入进行修正：$r_{ev} = \hat{r}_{ev} + \text{REM}([t_e; t_v])$。
- **GNN 消息传递**（L 层）：每层通过注意力权重 $\alpha_v^{rev}$（由问题嵌入 $q$ 与关系嵌入的相似度决定）聚合邻居三元组信息更新实体嵌入，实现与问题相关的多跳推理。
- **Top-K 三元组选择**：计算三元组 $\langle e, r_{ev}, v \rangle$ 与问题的相似度 $q^\top t_e^{(L)} + q^\top r_{ev} + q^\top t_v^{(L)}$，选择 top-K 个三元组（论文中 K=10）按相似度降序拼接为额外段落 $d_{\mathcal{T}_q^R}$。
- **答案生成**：将额外段落嵌入 $\mathbf{H}_{\mathcal{T}_q^R} = \text{Enc}(q, d_{\mathcal{T}_q^R})$ 与原段落嵌入拼接后送入 T5 decoder。

**训练策略**：KG Generator 与 Answer Predictor 解耦训练。
- KG Generator：在 REBEL 数据集（约 1M 样本）上 distant supervision 训练 DocuNet，学习率 3e-5，共 60K 步。
- Answer Predictor：联合优化 $\mathcal{L}_{answer} = \mathcal{L}_{t5} + \beta \mathcal{L}_{gnn}$，其中 $\mathcal{L}_{gnn}$ 用 answer-entity alignment 作为监督信号（KL 散度），$\mathcal{L}_{t5}$ 为 cross-entropy loss；$\beta=0.1$（部分数据集最优值不同）。

## 实验与结果
**数据集**：NQ（100 passages/问）、TQA（100 passages/问）、EQ（BM25 top-20）、2WQA（10 passages/问）、MuSiQue（20 passages/问）。

**主要结果（EM %，Table 1）**：

| 模型 | NQ | TQA | EQ | 2WQA | MuSiQue |
|------|-----|------|------|-------|---------|
| FiD | 48.2 | 65.0 | 68.1 | 74.1 | 29.9 |
| GRAPE | 48.7 | 66.2 | 68.3 | 73.4 | 28.3 |
| FiDO | 49.5 | 67.4 | 67.8 | 74.6 | 30.4 |
| **REANO** | **50.4** | **69.1** | **71.0** | **77.1** | **31.8** |

- REANO 在所有五个数据集上均取得最优结果，相比第二好模型分别提升：TQA +1.7%、EQ +2.7%、2WQA +2.5%、MuSiQue +1.4%、NQ +0.8%，**平均提升 1.8%**。
- 相比 FiD（无三元组），EQ 提升 2.9%，2WQA 提升 3.0%，MuSiQue 提升 1.9%。
- T5-large 版本（Table 6）：EQ 71.4%（+3.1%）、2WQA 78.8%（+1.9%）、MuSiQue 34.8%（+4.2%），验证泛化性。

**消融实验（Table 2）**：移除 inter-context triples（EQ: 69.5）、intra-context triples（EQ: 70.2）、REM（EQ: 69.1）、GNN（EQ: 68.9）均显著下降，证明各组件必要。

**其他发现**：
- 三元组顺序有影响：按相似度降序排列优于随机排列（2WQA 76.4 vs 77.1）。
- GNN 层数 L=3 最优，过多层导致 over-smoothing。
- 联合优化（joint）优于分开优化（separate），EM 分别提升 2.6% 和 2.7%（2WQA/MuSiQue）。
- K=15 时三元组数量效果最佳，过多引入噪声。

## 相关工作脉络
1. **FiD (Izacard & Grave, 2021b)**：REANO 的 reader 基础模型，FiD 独立编码段落，REANO 在此基础上引入结构化知识增强。
2. **KG-FiD (Yu et al., 2022)**：用 Wikidata 三元组构建 passage graph 进行 reranking；REANO 与之本质区别在于 KG 是从段落生成而非直接使用外部 KG。
3. **GRAPE (Ju et al., 2022)**：融合 KG 与段落 hidden states；同样依赖静态 KG，未解决不完整问题。
4. **OREOLM (Hu et al., 2022)**：用 LM 在 KG 上推理；REANO 改用生成式方法直接从文本构建 KG。
5. **UniK-QA (Oguz et al., 2022)**：将 triples 转为文本融入语料；REANO 显式建模结构关系并筛选高相关性三元组。
6. **DocuNet (Zhang et al., 2021a)**：段落级关系提取模型，REANO 借用其作为 intra-context RE 组件。

## 局限性与未来方向
1. **仅验证 T5 模型**：目前仅在 encoder-decoder 架构（T5）上验证，BART 等生成式 reader 及 decoder-only 模型（如 LLaMA）的效果待探索。
2. **冻结 retriever**：实验中使用固定 DPR retriever，未探究检索器质量对 REANO 的影响，也未尝试 joint optimise retriever 与 reader。
3. **远端监督的 KG 生成器**：DocuNet 直接在 REBEL 上 distant supervision 训练，未针对 ODQA 下游任务微调（论文发现微调反而可能导致性能下降）。
4. **GNN 层的过平滑问题**：层数过多会降低性能，限制了 KG 推理深度的上限。

## 研究启发与可借鉴点
1. **"从非结构化到结构化"的 KG 生成范式**：对于任何检索增强系统，当外部 KG 不完整或不适配时，可考虑从检索到的段落中自动生成结构化知识三元组，作为增强信号。
2. **GNN + 问题相关三元组筛选机制**：利用 GNN 的多跳推理能力识别与问题最相关的结构化知识片段，再将其拼接为额外输入——此思路可迁移到多跳 QA、事实核查等任务。
3. **answer-entity alignment 监督信号**：利用答案与实体的对齐关系作为弱监督信号训练 GNN（无需人工标注三元组相关性），是一种低成本的训练技巧，可应用于其他图谱推理任务。
4. **用结构化三元组替代部分非结构化段落**：实验显示知识三元组可帮助减少输入段落数量（n=50 → n=20）而不显著损失性能，对上下文长度受限的模型具有实用价值。
5. **联合优化 vs 分离优化的权衡**：论文证明共享 encoder 下联合优化优于分离优化，这一原则可指导未来多任务/多模块的端到端训练设计。

## 关键术语表
**Open Domain Question Answering (ODQA)**：开放域问答任务，给定问题从外部知识库（如 Wikipedia）中检索相关信息并生成答案。
**Fusion-in-Decoder (FiD)**：一种生成式 reader 模型，独立编码每个检索段落后拼接 token embeddings 送入 decoder 生成答案。
**Knowledge Graph (KG)**：以三元组（头实体-关系-尾实体）形式存储实体间关系的结构化知识表示。
**Intra-context / Inter-context RE**：段内关系提取（同一段落内实体间关系）与跨段落关系提取（不同段落实体间关系）。
**GNN（Graph Neural Network）**：图神经网络，通过消息传递聚合邻居节点信息，用于在三元组构成的图上计算问题相关度。
**Relation Embedding Module (REM)**：利用实体嵌入修正关系嵌入的模块，提升三元组表示的可靠性。
**Exact Match (EM)**：评估指标，预测答案与黄金答案列表中的任意一个完全匹配即算正确。
**Distant Supervision**：利用已有知识库（如 Wikidata）自动为关系抽取任务生成弱监督标签的训练方式。

## 可复现要素
- **数据集**：NQ、TQA（Karpukhin et al. 2020 split）、EQ（作者 split + BM25 top-20）、2WQA、MuSiQue，均为公开数据集；Wikipedia dump（2018-12-20, 21M passages）公开可用。
- **代码/权重**：论文未明确声明开源，需关注作者后续更新。
- **关键超参**：GNN 层数 L=3，选三元组数量 K=10，$\beta=0.1$（2WQA 最优）；T5-base backbone；learning rate=1e-4；batch size=64；AdamW optimizer；DOCUNet 初始化为 bert-base-uncased。
