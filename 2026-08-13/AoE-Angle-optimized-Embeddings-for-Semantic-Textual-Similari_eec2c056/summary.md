---
title: "AoE-Angle-optimized-Embeddings-for-Semantic-Textual-Similari"
source: https://aclanthology.org/2024.acl-long.101.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:54:22"
field: "语义文本相似度与文本嵌入学习"
keywords: ["text embedding", "semantic textual similarity", "cosine saturation", "complex embedding", "angle optimization", "contrastive learning", "MTEB"]
innovations: ["提出 AoE 模型，通过复数空间角度优化缓解余弦饱和区梯度消失问题", "构建 GitHub Issues 长文本 STS 数据集（GIS）填补长文本评测空白", "在 BERT 和 LLaMA 多尺度上取得 STS 与 MTEB 基准 SOTA"]
benchmarks: ["STS 2012-2016", "SICK-R", "STS-B", "GIS", "MTEB", "MultiNLI", "SNLI"]
---

# 论文速读：AoE: Angle-optimized Embeddings for Semantic Textual Similarity

## 一句话总结
本文针对语义文本相似度（STS）中余弦相似度存在的饱和区梯度消失问题，提出 AoE（角度优化嵌入）模型，通过在复数空间进行复数除法计算角度差来捕捉细微语义差异；同时构建了一个包含约 22K 样本的 GitHub Issues 长文本 STS 数据集，在标准 STS、下游分类任务及 MTEB 基准上均取得最优或接近最优性能。

## 研究问题与动机
- **余弦饱和区导致梯度消失**：现有 STS 模型普遍使用余弦相似度作为学习目标，但余弦函数在接近 0 或 π 时存在饱和区，使得梯度接近零，阻碍参数更新（Roodschild et al., 2020）。
- **难以区分表面相似但语义不同的文本对**：NLI 数据集中大量存在外观高度重叠但语义不相似（如 "neutral" 标签）的文本对，它们落入余弦饱和区后，模型易误判为 "entailment"。
- **现有方法均未解决此问题**：包括 SBERT、SimCSE、ConSERT 等主流模型均以余弦为核心度量，但从未考虑饱和区的负面影响。
- **长文本 STS 评测数据匮乏**：现有 STS 基准多聚焦短文本（<512 tokens），缺乏对真实场景（金融、法律文档）中长文本相似度的系统评估。

## 核心贡献（创新点）
- **首次探索余弦饱和区对文本嵌入的负面影响**：与已有工作仅关注余弦相似度不同，本文明确指出饱和区导致的梯度消失问题，并提出复数空间角度优化方案。
- **提出 AoE 复数嵌入框架**：将文本嵌入分解为实部（捕捉外观差异）和虚部（捕捉细微语义差异），通过复数除法计算角度差 $\Delta\theta$，与已有方法的本质区别在于引入角度目标替代/辅助余弦目标。
- **构建 GIS 长文本 STS 数据集**：从 GitHub 55 个开源项目收集约 22K 样本，填补长文本 STS 评测空白，已有工作缺乏此类真实场景长文本数据。
- **多尺度骨干网络验证与 MTEB SOTA**：在 $\mathrm{BERT}_{base}$、$\mathrm{LLaMA}_{7B}$、$\mathrm{LLaMA}_{13B}$ 三种规模上均取得最优结果，并在 MTEB 上超越 bge-large-en-v1.5、OpenAI text-embedding-3 等闭源模型。

## 方法详解
- **复数文本嵌入生成**：输入文本经 token embedding 层得到维度 $2d$ 的向量，前 $d$ 维为实部 $\mathbf{X}^{re}$，后 $d$ 维为虚部 $\mathbf{X}^{im}$；通过 Transformer encoder（如 BERT 取 [CLS]、LLaMA 取末 token）获得句子级复数嵌入 $\mathbf{z} = \mathbf{a} + \mathbf{b}i$。
- **角度目标函数**：基于复数除法规则，两嵌入 $\mathbf{z}, \mathbf{w}$ 的角度差为：
  $$\Delta\theta_{zw} = \log\left[\frac{(\mathbf{a}\mathbf{c}+\mathbf{b}\mathbf{d})+(\mathbf{b}\mathbf{c}-\mathbf{a}\mathbf{d})}{\sqrt{(\mathbf{c}^2+\mathbf{d}^2)(\mathbf{a}^2+\mathbf{b}^2)}}\right]$$
  采用排序式损失：
  $$\mathcal{L}_{angle} = \log\left[1 + \sum_{s_{ij}>s_{mn}}\exp\left(\frac{\Delta\theta_{ij}-\Delta\theta_{mn}}{\tau}\right)\right]$$
  使高相似度对的 $\Delta\theta$ 更小，从而在饱和区也能通过角度差反映细微语义差异。
- **多目标联合训练**：辅助使用监督对比损失 $\mathcal{L}_{cl}$（基于余弦相似度），最终目标为：
  $$\mathcal{L} = w_1 \cdot \mathcal{L}_{angle} + w_2 \cdot \mathcal{L}_{cl}$$
  权重 $w_1, w_2$ 通过网格搜索确定，温度参数 $\tau=0.05$。

## 实验与结果
- **数据集**：
  - STS 训练数据：MultiNLI + SNLI（其中 33% 样本相似度 >0.95，66% >0.8，大量落入饱和区）
  - STS 评测：STS 2012–2016、SICK-R、STS-B（短文本）；新建 GIS 长文本数据集（train: 18,565 / val: 1,547 / test: 1,548，长文本占比 ~61%）
  - 下游任务：MR、CR、SUBJ、MPQA、SST2、TREC、MRPC 及 MTEB（含分类、聚类、检索等 52 个子任务）
- **最强结果**：
  - **标准 STS**（Table 1）：$\mathrm{LLaMA}_{13B}$  backbone 下 AoE 平均 Spearman 达 86.52，较 SimCSE 提升 0.60%；$\mathrm{BERT}_{base}$ 下 AoE 达 82.43，较 SimCSE 提升 0.86%。
  - **域内 STS**（Table 2）：AoE 在 STS-B 达 86.28（+1.35% vs SBERT），在 GIS 长文本达 70.59（+10.11% vs SimCSE）。
  - **下游分类**（Table 3）：$\mathrm{AoE-LLaMA}_{13B}$ 平均准确率 91.45%，超越所有基线。
  - **MTEB**：AoE 在 BERT-large 规模下达 64.64，超越 bge-large-en-v1.5（64.23）、OpenAI text-embedding-3-large（64.59）等。
- **消融实验**（Table 4）：仅用角度目标即可达 82.36，优于仅对比学习（81.53）；CLS pooling 最优；对随机种子不敏感。
- **NLI 任务**（Table 5）：AoE-$\mathrm{BERT}_{base}$ 在 MultiNLI 和 SNLI 上分别达 56.60% 和 63.88%，超越 SimCSE。
- **可视化分析**（Figure 3/4/5）：虚部嵌入在饱和区分散度更高，Cosine 相似度分布更贴近人类标注分布。

## 相关工作脉络
- **SBERT (Reimers & Gurevych, 2019)**：基于 Siamese BERT 的监督 STS 模型，使用余弦回归目标——AoE 与其区别在于引入复数角度目标缓解饱和区问题。
- **SimCSE (Gao et al., 2021)**：对比学习句子嵌入方法，同样依赖余弦相似度——AoE 在此基础上增加角度优化以捕捉细微差异。
- **DiffCSE (Chuang et al., 2022)**：基于差值的对比学习——AoE 不依赖差值而依赖角度差，且显式建模实/虚部分工。
- **RoFormer (Su et al., 2021b)**：在词级别利用复数乘法引入旋转位置编码——AoE 是首个在句级别利用复数除法进行 STS 学习的尝试。
- **RotatE (Sun et al., 2019)**：知识图谱中的复数实体嵌入用于链接预测——AoE 借鉴其复数表示思想但应用于文本句子级相似度，且使用除法而非乘法。
- **WhitenedCSE (Zhuo et al., 2023)**：基于白化的对比学习——AoE 聚焦饱和区角度优化，而非分布白化。

## 局限性与未来方向
- **长文本性能提升幅度低于短文本**：在 GIS 上的提升（+10.11% vs SimCSE）虽显著但相对短文本的提升比例较小，作者计划在后续工作中改进长文本 STS 性能。
- **训练数据规模限制**：AIS 数据集仅约 22K 样本，人工抽查 10% 质量保证，规模小于传统 STS 数据集（SNLI/MNLI 数十万级）。
- **仅验证英语场景**：实验集中在英文数据集，跨语言泛化能力未评估。
- **角度目标与对比目标的权重需调优**：$w_1, w_2$ 依赖网格搜索，缺乏自适应平衡机制。

## 研究启发与可借鉴点
- **复数嵌入用于区分细微语义**：将嵌入分解为实/虚两部分并赋予不同语义职责（外观 vs 细微差异）的设计思路可迁移至其他相似度学习任务。
- **角度目标替代余弦目标**：对于存在梯度消失风险的相似度度量，可考虑引入角度/相位优化作为补充或替代。
- **GitHub Issues 作为 STS 数据来源**：开源项目的重复 issue 标注天然适合构建长文本相似度数据集，此数据收集策略可复用于其他领域（如 JIRA、邮件列表）。
- **多目标联合训练策略**：角度目标 + 对比目标的组合在消融中显示互补性，类似的多目标设计可用于其他嵌入学习方法。
- **t-SNE 可视化验证虚部嵌入效果**：通过距离分布和散点图直观展示虚部在饱和区的区分能力，此可视化方法可用于其他嵌入模型的内部机制分析。

## 关键术语表
- **AoE (Angle-optimized Embedding)**：一种基于复数空间角度优化的文本嵌入模型，通过计算嵌入间的角度差来捕捉细微语义差异。
- **余弦饱和区 (Cosine Saturation Zone)**：余弦函数在相似度接近 1 或 -1 时梯度趋于零的区域，导致模型难以学习细微语义区别。
- **复数除法 (Complex Division)**：在复数空间中计算两个嵌入的角度差和模长比的运算，AoE 的核心数学工具。
- **GIS (GitHub Issues Similarity Dataset)**：论文构建的长文本 STS 数据集，包含约 22K 个 GitHub issue 对，其中约 61% 为长文本（>512 tokens）。
- **角度目标 (Angle Objective)**：基于复数除法角度差的排序式损失函数，与对比损失联合优化。
- **MTEB (Massive Text Embedding Benchmark)**：包含 52 个任务的文本嵌入大规模评测基准，涵盖分类、聚类、检索、STS 等。
- **NLI (Natural Language Inference)**：自然语言推理任务，本文使用 MultiNLI 和 SNLI 作为 STS 模型的训练数据。
- **CLS pooling**：取 Transformer 中 [CLS] token 的输出作为句子级嵌入的池化策略，在 AoE 中表现最优。

## 可复现要素
- **代码开源**：https://github.com/SeanLee97/AnglE
- **模型权重**：HuggingFace 公开，包括 UAE（通用嵌入）和 NLI 嵌入系列（BERT base、LLaMA 7B/13B）
- **数据集**：
  - STS 训练数据：MultiNLI + SNLI（公开）
  - GIS 长文本数据集：论文附录 B 提供构建细节，数据可通过 GitHub API 复现
  - 评测基准：STS 2012–2016、SICK-R、STS-B、MTEB（均为公开）
- **关键超参**：学习率 $\mathrm{BERT}: 5\text{e-}5$, $\mathrm{LLaMA}: 1\text{e-}4$（QLoRA）；温度 $\tau=0.05$；随机种子 42；Prompt 格式："Summarize sentence {text} in one word:"
- **评估指标**：Spearman 相关系数（SentEval "all" 设置）、准确率（分类任务）
