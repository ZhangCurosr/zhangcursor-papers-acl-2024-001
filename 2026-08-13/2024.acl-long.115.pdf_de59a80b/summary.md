---
title: "REANO: Optimising Retrieval-Augmented Reader Models through Knowledge Graph Generation"
source: https://aclanthology.org/2024.acl-long.115.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:01:10"
field: "开放域问答与知识增强"
keywords: ["开放域问答", "检索增强生成", "知识图谱生成", "图神经网络", "多跳推理", "关系抽取"]
innovations: ["从段落自动生成知识三元组解决KG不完整问题", "基于GNN的问题相关三元组筛选机制", "分离训练策略结合距离监督"]
benchmarks: ["Natural Questions", "TriviaQA", "EntityQuestions", "2WikiMultiHopQA", "MuSiQue"]
---

# 论文速读：REANO: Optimising Retrieval-Augmented Reader Models through Knowledge Graph Generation

## 一句话总结
论文提出REANO模型，通过从检索段落中**生成知识图谱**来捕获段落间的语义依赖关系，结合GNN筛选与问题最相关的三元组作为额外上下文，显著提升开放域问答（ODQA）性能。

## 研究问题与动机
- **现有FiD模型的局限**：Fusion-in-Decoder (FiD) 独立编码每个检索段落，忽略了段落间的语义关联，难以处理需要多跳推理的问题。
- **已有KG增强方法的缺陷**：KG-FiD、OREOLM、GRAPE等方法直接使用Wikidata等现有知识图谱中的三元组，但现有KG存在**不完整问题**，可能缺少回答特定问题所需的关键信息（如人物生日、地理位置等）。
- **核心挑战**：如何在不依赖外部KG的情况下，从检索段落中提取并利用结构化知识来增强读者模型？

## 核心贡献（创新点）
1. **KG生成模块**：提出从检索段落中自动生成知识三元组的KG Generator，包含实体识别（SpaCy+TAGME）和关系抽取（DocuNet）两个子模块，解决了现有KG不完整的问题。
2. **GNN驱动的三元组筛选**：引入图神经网络（GNN）从生成的大量三元组中识别并选择与问题最相关的top-K三元组，作为额外段落输入答案预测器。
3. **关系嵌入细化模块（REM）**：设计REM模块，通过整合实体嵌入来细化关系嵌入，提高KG的质量。
4. **联合训练策略**：提出分离式训练方案——先训练KG生成器，再基于生成的KG训练答案预测器，通过距离监督（distant supervision）解决KG生成器缺乏标注数据的问题。
5. **实证效果显著**：在五个ODQA数据集上，REANO相比最强基线EM分数提升最高达2.7%（EQ数据集），平均提升1.8%。

## 方法详解
**整体框架**：REANO = KG Generator + Answer Predictor

### KG Generator（概率建模）
将答案生成建模为：
$$p_θ(a|q, \mathcal{D}_q) = \sum_{\mathcal{G}_q} p_θ(a|q, \mathcal{D}_q, \mathcal{G}_q) \cdot p_φ(\mathcal{G}_q|\mathcal{D}_q)$$

其中：
- $p_φ(\mathcal{G}_q|\mathcal{D}_q)$：KG生成器，从段落中生成知识图谱
- $p_θ(a|q, \mathcal{D}_q, \mathcal{G}_q)$：答案预测器，基于段落和KG生成答案

**KG生成器的分解**：
1. **实体识别（ER）**：使用SpaCy识别命名实体，TAGME链接到Wikipedia实体，得到实体集合$\mathcal{E}_q$
2. **关系抽取（RE）**：分为两部分
   - **Intra-context RE**：使用DocuNet模型（BERT+UNet架构）抽取单个段落内的实体关系
   - **Inter-context RE**：通过Wikidata API获取跨段落实体间的关系（即使这些关系未在段落中明确提及）

生成的KG为：$\mathcal{G}_q = \{T_q^I, T_q^C\}$（内部关系三元组 + 跨段落关系三元组）

### Answer Predictor
基于FiD框架，关键增强：

**GNN三元组相关性建模**：
1. **实体嵌入初始化**：用T5编码器获取段落token嵌入，用`<e>`标记位置的嵌入mean-pool得到实体嵌入$t_e$
2. **关系嵌入细化**：
   $$r_{ev} = \hat{r}_{ev} + \text{REM}([t_e; t_v])$$
   其中REM是两层前馈神经网络
3. **GNN消息传递（L层）**：
   $$t_e^{(l)} = \text{FFN}([t_e^{(l-1)}; s_e^{(l)}])$$
   $$s_e^{(l)} = \sum_{(v, r_{ev}) \in \mathcal{N}(e)} \alpha_v^{r_{ev}} \cdot \text{FFN}([t_v^{(l-1)}; r_{ev}])$$
   注意力权重$\alpha_v^{r_{ev}}$由问题与关系的相似度决定

**三元组选择**：
$$p_θ(T_q|q, \mathcal{D}_q, \mathcal{G}_q) \propto q^\top t_e^{(L)} + q^\top r_{ev} + q^\top t_v^{(L)}$$
选取top-K个相关三元组$T_q^R$，拼接为额外段落$d_{T_q^R}$

**答案生成**：
$$p_θ(a|q, \mathcal{D}_q, T_q^R) = \text{Dec}([\mathbf{H}_1; \dots; \mathbf{H}_n; \mathbf{H}_{T_q^R}])$$

### 训练策略
- **KG生成器训练**：在REBEL数据集（约100万样本）上使用距离监督训练DocuNet，无需人工标注
- **答案预测器训练**：联合损失函数
  $$\mathcal{L}_{answer} = \mathcal{L}_{t5} + \beta \mathcal{L}_{gnn}$$
  其中$\mathcal{L}_{gnn}$使用Kullback-Leibler散度，监督信号来自答案-实体对齐（答案能匹配段落中实体时提供路径监督）

## 实验与结果
**数据集**：NQ, TQA, EQ, 2WikiMultiHopQA, MuSiQue

**基线模型**：DPR, RAG-Seq, FiD, KG-FiD, OREOLM, GRAPE, FiDO

**主要结果（EM分数）**：

| 数据集 | FiD | REANO | 提升 |
|--------|-----|-------|------|
| NQ | 48.2 | 50.4 | +0.8 |
| TQA | 65.0 | 69.1 | +1.7 |
| EQ | 68.1 | 71.0 | **+2.7** |
| 2WQA | 74.1 | 77.1 | +2.5 |
| MuSiQue | 29.9 | 31.8 | +1.4 |

**平均提升**：1.8%

**消融实验结论**：
- 去除inter-context三元组：EM下降1.5%~2.2%
- 去除GNN模块：EM下降1.6%~3.8%
- 去除REM模块：EM下降1.9%~2.5%
- 最优β值：0.1（2WQA）或1e-3（MuSiQue）

**关键发现**：知识三元组可帮助减少段落数量（在n=20时仍保持较好性能），且按相似度降序排列三元组比随机排列效果更好。

## 相关工作脉络
1. **FiD (Izacard & Grave, 2021)**：基础检索增强生成读者模型，本文在其基础上引入KG生成模块
2. **KG-FiD (Yu et al., 2022)**：使用现有KG构建段落图进行重排序，仅依赖外部KG
3. **GRAPE (Ju et al., 2022)**：融合KG和上下文表示到读者隐藏状态，同样使用外部KG
4. **OREOLM (Hu et al., 2022)**：将知识图谱推理融入语言模型，但受限于KG完整性
5. **UniK-QA (Oguz et al., 2022)**：将三元组转换为文本合并到语料库
6. **DocuNet (Zhang et al., 2021)**：文档级关系抽取模型，本文用于intra-context RE

**定位差异**：REANO的核心创新是**从段落中生成KG**而非依赖外部KG，从而解决KG不完整问题；同时引入GNN进行**问题相关的三元组选择**而非全量使用。

## 局限性与未来方向
- **模型扩展性**：仅在T5-base上验证，未探索BART或decoder-only模型（如LLaMA）的适配
- **检索器固定**：使用冻结的DPR/BM25检索器，未研究检索器有效性对REANO的影响
- **联合优化缺失**：未探索检索器与REANO的联合优化，仅分离训练
- **超参数敏感**：β值和K值需针对不同数据集调优，缺乏统一设置

## 研究启发与可借鉴点
1. **KG生成替代依赖**：当外部知识源不完整时，可从文本中自动生成结构化知识，这一思路可迁移到其他需要知识增强但KG缺失的场景
2. **GNN筛选机制**：使用GNN结合问题嵌入进行三元组重要性排序，可有效处理大规模KG中的噪声，适用于任何基于图结构的检索增强任务
3. **距离监督训练**：利用REBEL等大规模预训练数据训练关系抽取模型，避免在小规模QA数据上过拟合，值得在低资源场景复用
4. **三元组顺序效应**：按相关性降序排列三元组优于随机排列，提示结构化知识的排序对模型性能有显著影响
5. **分离训练策略**：先训练KG生成器再训练答案预测器的两阶段方案，可缓解端到端优化的不稳定性

## 关键术语表
- **REANO**：Retrieval-Augmented generative readers with a kNOwledge graph generation module的缩写，本文提出的模型名称
- **FiD (Fusion-in-Decoder)**：基于T5的检索增强生成读者模型，独立编码每个段落后融合到解码器
- **GNN (Graph Neural Network)**：图神经网络，用于在KG上进行消息传递以学习实体和三元组的相关性
- **DocuNet**：基于BERT+UNet的文档级关系抽取模型，单步前向传播预测段落内所有实体对的关系
- **距离监督 (Distant Supervision)**：利用外部知识库（如Wikidata）自动为文本生成关系标注的训练方法
- **Intra/Inter-context RE**：分别指段落内关系抽取和跨段落关系抽取
- **REM (Relation Embedding Module)**：通过整合实体嵌入来细化关系嵌入的两层前馈网络

## 可复现要素
- **数据集**：NQ, TQA, EQ, 2WQA, MuSiQue均公开；REBEL数据集公开
- **代码**：论文未明确说明代码开源情况（需进一步确认）
- **关键超参**：
  - GNN层数L=3
  - 选择三元组数K=10
  - 学习率=1e-4
  - β=0.1（2WQA）/1e-3（MuSiQue）
  - batch size=64
  - 优化器：AdamW
- **模型架构**：T5-base作为骨干，DocuNet（BERT-base+UNet）用于关系抽取
- **检索器**：DPR（NQ/TQA）和BM25（EQ）
