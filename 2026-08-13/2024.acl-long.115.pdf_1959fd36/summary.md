---
title: "REANO: Optimising Retrieval-Augmented Reader Models through Knowledge Graph Generation"
source: https://aclanthology.org/2024.acl-long.115.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:01:04"
field: "开放域问答与知识图谱增强"
keywords: ["open-domain question answering", "retrieval-augmented generation", "knowledge graph generation", "Fusion-in-Decoder", "graph neural network", "relation extraction", "multi-hop QA"]
innovations: ["从检索段落生成知识图谱三元组以弥补外部KG不完整", "question-aware GNN选择top-K相关三元组并作为附加段落融入FiD解码", "用answer-entity对齐弱监督训练GNN并与T5联合优化"]
benchmarks: ["Natural Questions", "TriviaQA", "EntityQuestions", "2WikiMultiHopQA", "MuSiQue"]
---

# 论文速读：REANO: Optimising Retrieval-Augmented Reader Models through Knowledge Graph Generation

## 一句话总结
本文提出 REANO，一种面向开放域问答（ODQA）的知识图谱生成增强框架，通过从检索段落中**生成**知识图谱并借助 GNN 筛选与问题最相关的 top-K 三元组作为附加"段落"，弥补了现有 KG-FiD 等方法依赖不完整外部 KG 的缺陷，在五数据集上平均提升 EM 1.8%，EntityQuestion 上最高提升 2.7%。

## 研究问题与动机
- **FiD 忽略段落间语义关联**：Fusion-in-Decoder 独立编码各检索段落，无法建模跨段落关系，难以支持多跳推理。
- **现有 KG 增强方法依赖外部 KG，易缺失关键信息**：KG-FiD、OREOLM、GRAPE 等仅利用 Wikidata 等已有三元组，但这些 KG 存在严重不完整性（如缺少人物生日等回答问题所需的事实）。
- **段落内信息未被结构化利用**：检索到的非结构化段落中包含对回答至关重要的关系线索，现有 reader 无法显式抽取并复用。
- **端到端联合优化困难**：KG 生成是不可微过程，需设计解耦或可微近似的训练策略以支持 reader 的优化。

## 核心贡献（创新点）
- **提出 KG 生成模块，从段落而非外部 KG 抽取三元组**：用 KG generator（ER + DocuNet 内部上下文 RE + Wikidata 跨段落 RE）直接在 passage 上生成 $\mathcal{G}_q$，解决外部 KG 不完整导致的"答案信息在段落却不在 KG"问题（与 KG-FiD 等的本质区别）。
- **设计基于 GNN 的 top-K 相关三元组选择机制**：引入 REM 细化关系嵌入，并用 question-aware GNN 聚合邻居三元组信号后按 $q^\top t_e^{(L)} + q^\top r_{ev} + q^\top t_v^{(L)}$ 排序选出 $T_q^R$，避免将无关三元组全部塞入 decoder（与直接用所有 KG 三元组的方案不同）。
- **将选出的三元组拼接为"附加段落"融入 FiD 生成流程**：把 $d_{T_q^R}$ 与其他 passage 嵌入拼接到 T5 decoder，既保留 FiD 结构又补充结构化推理路径，缓解 answer predictor 的多跳负担（区别于把 KG 直接做 rerank 或融合到 hidden state 的做法）。
- **提出分离式训练但联合优化的 Loss**：KG generator 用 REBEL 远端监督训练；answer predictor 用 answer-entity alignment 构造 KL 损失训练 GNN，并以 $\mathcal{L}_{answer} = \mathcal{L}_{t5} + \beta \mathcal{L}_{gnn}$ 联合更新共享 encoder，证明比依次分开优化更稳定。

## 方法详解
- **概率建模**：把生成的 KG $\mathcal{G}_q$ 视为潜变量，$p_\theta(a|q, \mathcal{D}_q) = \sum_{\mathcal{G}_q} p_\theta(a|q,\mathcal{D}_q,\mathcal{G}_q)\, p_\phi(\mathcal{G}_q|\mathcal{D}_q)$，由 KG generator $p_\phi$ 和 answer predictor $p_\theta$ 共同建模。
- **KG Generator（ER + RE）**：
  - ER：SpaCy 命名实体 + TAGME 链接到 Wikipedia 实体，得到 $\mathcal{E}_q$。
  - Intra-context RE：在单段内用 DocuNet（BERT+UNet，单前向预测所有实体对关系），对每段 $d_i$ 求乘积式生成 $\mathcal{T}_q^I$。
  - Inter-context RE：通过 Wikidata API 取跨段实体间的已知关系，得 $\mathcal{T}_q^C$。
  - $\mathcal{G}_q = \mathcal{T}_q^I \cup \mathcal{T}_q^C$。
- **Answer Predictor**：
  - **Embed 初始化**：实体嵌入用 T5 encoder 对加 `<e>...</e>` 标注的 passage 序列编码后取 `<e>` token 的 mean-pool；关系嵌入用 T5 对关系标签编码后 mean-pool，再经 REM（两层 FFN）按 $r_{ev} = \hat{r}_{ev} + \text{REM}([t_e; t_v])$ 细化。
  - **L 层 Question-aware GNN**：每层用关系与问题相似度做注意力 $\alpha_v^{rev}$ 聚合邻居三元组消息：$s_e^{(l)} = \sum_{(v,r_{ev})\in\mathcal{N}(e)} \alpha_v^{rev}\cdot\text{FFN}([t_v^{(l-1)}; r_{ev}])$，再用 FFN 融合自身上一状态得 $t_e^{(l)}$。
  - **Top-K 选择**：三元组相关度 $p_\theta(T_q|\cdot) \propto q^\top t_e^{(L)} + q^\top r_{ev} + q^\top t_v^{(L)}$，取 top-K 拼接为 $d_{T_q^R}$，随原 passage 一起送入 T5 decoder：$\text{Dec}([\mathbf{H}_1;\dots;\mathbf{H}_n;\mathbf{H}_{T_q^R}])$。
- **训练策略**：
  - KG generator 仅在 DocuNet 上远端监督（REBEL 数据集），跨段 RE 不调参。
  - Answer predictor 用 answer-entity 对齐构造 $c_q^*$ 做 KL 训练 GNN：$\mathcal{L}_{gnn}=D_{KL}(c_q||c_q^*)$；T5 用 CE：$\mathcal{L}_{t5}=\text{CE}(p_\theta(a|\cdot),p^*(a|q))$；总损失 $\mathcal{L}_{answer}=\mathcal{L}_{t5}+\beta\mathcal{L}_{gnn}$。

## 实验与结果
- **数据集**：NQ、TQA（各 50 个 passage，其余基线也用 DPR 100 个）、EQ（BM25 top-20）、2WQA（10 个）、MuSiQue（20 个）。
- **评估**：Exact Match（greedy），主要与 FiD 及其变体比较。
- **主要结果**（Table 1）：
  - NQ: REANO 50.4 vs 次优 FiDO 49.5（+0.9）；TQA: 69.1 vs 次优 FiDO 67.4（+1.7）；EQ: **71.0 vs FiD 68.1（+2.7，显著）**；2WQA: **77.1 vs FiDO 74.6（+2.5，显著）**；MuSiQue: **31.8 vs FiDO 30.4（+1.4，显著）**。平均提升 1.8%。
  - 与 KG-FiD / OREOLM / GRAPE 等 KG 增强 reader 相比在 NQ、TQA 上也全面领先。
  - 相对 FiD（去掉三元组退化为 FiD）在 EQ、2WQA、MuSiQue 显著+2.9/3.0/1.9。
- **Ablation（Table 2）**：去除 inter/intra 三元组、REM、GNN 均显著降分，说明各组件必要。
- **超参**：K=10、GNN 层 L=3、$\beta=0.1$（2WQA 最优 0.1，MuSiQue 最优 1e-3，跨数据集需调）。
- **大模型验证**：T5-large 下平均再提升 3.1%（Table 6）。

## 相关工作脉络
- **FiD（Izacard & Grave, 2021）**：独立编码 passage 后拼接解码的 generative reader，是本文 backbone 与最核心基线。
- **KG-FiD（Yu et al., 2022）**：用已有 Wikidata 三元组构建 passage graph 做 reranking；定位差异在于"用外部 KG"vs 本文"从段落生成 KG"。
- **OREOLM（Hu et al., 2022）**：把 KG 推理注入 LM 的 hidden states；差异在于同样依赖不完整外部 KG 且不做 passage 级生成。
- **GRAPE（Ju et al., 2022）**：将 KG 与 passage 语境表示融合到 reader hidden；差异在于仍使用外部 KG 而非从 passage 推理。
- **UniK-QA（Oguz et al., 2022）**：把 triples 转文本拼到 corpus；差异在于无生成、无 GNN 选择，直接全量加入。
- **RAG-Seq / DPR**：基础检索增强与抽取式 reader 基线，用于对照 generative/extractive 范式差异。

## 局限性与未来方向
- 仅在 FiD（encoder-decoder T5）上验证，未扩展到 BART 或 decoder-only LLM；通用性存疑。
- 使用冻结的 DPR/BM25 retriever，未研究 retriever 与 REANO 的联合优化对最终性能的影响。
- 训练策略解耦（先训 KG generator 再训 answer predictor），端到端不可微优化可能损失性能。
- Inter-context RE 直接借用 Wikidata 中所有关系（即便在 passage 中未提及），可能引入噪声。
- 需要针对每个数据集单独调优 $\beta$，泛化性不足。

## 研究启发与可借鉴点
- **从 passage 生成结构化知识替代依赖外部 KG**：对任何信息不完整的场景（医学、法律、私有语料）都可迁移，避免"答案在文档但不在 KG"的尴尬。
- **GNN 的 question-aware 注意力机制**：用问题向量对关系边加权并用于三元组排序，思路可迁移到 KBQA、多跳推理、subgraph retrieval 等任务。
- **三元组作为附加段落的拼接方式**：低成本地在 FiD/T5 类架构中加入结构化信号，无需改动 decoder 主体，可直接复用到其他 passage 密集型生成任务。
- **答案-实体对齐作为 GNN 弱监督信号**：避免手写标注、利用数据固有对齐构造 KL 监督，适合资源受限场景。
- **消融展示 K、L、$\beta$ 的敏感性分析**：对后续研究者复现与调参有参考价值。

## 关键术语表
- **REANO**：本文提出的 Retrieval-Augmented generative Reader with a kNOwledge graph generation module。
- **Fusion-in-Decoder（FiD）**：分别用 encoder 编码每个 passage，再把 token 嵌入拼接到 T5 decoder 进行生成式 QA。
- **Intra-context RE**：在单个 passage 内抽取实体对关系（本文用 DocuNet）。
- **Inter-context RE**：跨多个 passage 利用外部 KG（Wikidata API）建立实体间关系。
- **Relation Embedding Module（REM）**：以实体嵌入为条件对初始关系嵌入做 refine 的两层 FFN。
- **Question-aware GNN**：用问题向量对邻居三元组关系做 attention 加权的消息聚合 GNN。
- **Top-K Relevant Triple Selection**：按三元组与问题的相似度排序后取前 K 个拼接为附加 passage。
- **Distantly Supervised Training**：用 REBEL（Wikipedia- Wikidata 对齐）代替人工标注训练 DocuNet。

## 可复现要素
- **代码/权重**：论文未提供开源链接（截至论文发表）。
- **数据集**：NQ、TQA、EQ、2WQA、MuSiQue 均为公开数据集；REBEL 数据集公开。
- **关键超参**：T5-base backbone；GNN 层数 L=3；K=10；学习率 1e-4；batch=64；β 在 2WQA=0.1、MuSiQue=1e-3；DocuNet 学习率 3e-5、batch=32、steps=60,000。
- **Retriever**：NQ/TQA 用 DPR（100 passages 取 top-50 实验）；EQ 用 BM25（top-20）；2WQA/MuSiQue 用作者提供 passages。
