---
title: "TTM-RE-Memory-Augmented-Document-Level-Relation-Extraction"
source: https://aclanthology.org/2024.acl-long.26.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:54:26"
field: "文档级关系抽取与弱监督学习"
keywords: ["document-level relation extraction", "memory-augmented model", "Token Turing Machine", "positive-unlabeled learning", "distant supervision", "noise-robust training", "ReDocRED", "long-tail relation"]
innovations: ["首个将Token Turing Machine记忆模块应用于文档级关系抽取的架构", "记忆模块与SSR-PU噪声鲁棒损失联合优化以充分利用远距离监督数据", "在极端低标注(19%)+远距离监督场景下超越SOTA 12个F1点"]
benchmarks: ["ReDocRED", "ChemDisGene"]
---

# 论文速读：TTM-RE-Memory-Augmented-Document-Level-Relation-Extraction

## 一句话总结
TTM-RE 是首个专为文档级关系抽取设计的记忆增强架构，将 Token Turing Machine（TTM）的可训练记忆模块与噪声鲁棒损失函数（SSR-PU，处理正-未标注设置）相结合，成功解锁大规模远距离监督噪声训练数据的潜力，在 ReDocRED 上绝对 F1 提升超 3%，在 ChemDisGene 上提升 5。

## 研究问题与动机
- **核心问题**：文档级关系抽取（DocRE）中的训练数据存在大量噪声（远距离监督标签产生大量假负例），现有方法无法有效利用大规模噪声数据进行微调，导致"训练数据越大、性能反而不升反平"的现象。
- **ReDocRED 基准上的反常**：最强方法在 101,873 条远距离监督数据上训练，F1 并未显著优于仅用 3,053 条人工标注数据训练的模型——说明问题不在数据量，而在**架构缺陷**。
- **现有工作局限**：KD-DocRE 等仅用知识蒸馏把教师模型在人工数据上的 logit 迁移给蒸馏，本质仍是"冻结记忆、不做端到端微调"；SSR-PU 虽提出正-未标注（PU）损失，但未引入记忆模块放大远距离数据的收益。
- **记忆增强模型的启发**：TTM（Ryoo et al., 2023）已在长序列视觉理解中证明引入可学习记忆 token 可显著提升下游分类；本工作首次将其迁移到 DocRE 场景。

## 核心贡献（创新点）
1. **首個记忆增强的文档级关系抽取架构（TTM-RE）**：将 TTM 的记忆模块嵌入在 Roberta-Large 单遍编码之后、分类器之前，对头尾实体 token 做可学习的记忆读写操作；与知识蒸馏范式的本质区别是——**端到端直接微调远距离监督数据，而非蒸馏教师 logits**。
2. **噪声鲁棒的 PU 损失（SSR-PU）适配**：在标准 PU 学习基础上引入类别先验偏移（prior shift）校正项，使模型能正确区分远距离监督训练集中的假负例（大量漏标关系）；与原始 SSR-PU 的区别是**新增 memory 模块后联合优化**，而非单独替换损失。
3. **在极端低标注场景（19% 人工标签 + 远距离监督数据）下超越 SOTA 12 F1**：证明记忆模块能显著缓解长尾关系类别的学习困难；与全监督 Baseline 的本质差异是**记忆 token 充当了"关系类型原型"的压缩表示**。
4. **系统性的消融验证**：memory token 数量（10/50/100/200）、layer 数（1-4）、基础模型（RoBERTa-Large vs DeBERTaV3-Large）——关键发现是**增加模型参数量（换 DeBERTaV3）并不提升 DocRE 性能，而增加记忆模块却显著提升**，印证架构设计的重要性。
5. **跨领域泛化验证（ChemDisGene）**：在生物医学文档级关系抽取数据上，即使仅用人工标注训练（不调远距数据），TTM-RE 仍取得 +5 F1 提升（53.59 vs 48.56），表明记忆机制在训练数据规模更大的场景下同样有效。

## 方法详解
### 整体框架（Figure 3）
- **Encoder**：RoBERTa-Large 对整篇文档做单遍编码，得到 token 级表示。
- **Entity 检索**：基于 token 索引定位头实体 $e_h \in \mathbb{R}^{d}$ 和尾实体 $e_t \in \mathbb{R}^{d}$。
- **Memory 模块（灰色框）**：输入为拼接 $[M \| I]$，其中 $M \in \mathbb{R}^{m \times d}$ 为可学习记忆 token（从正态分布初始化，非全零），$I = [e_h; e_t]$ 为实体 token。
- **Reading 操作**：通过 MLP 计算重要性权重 $w_i = \alpha_i(V) = \text{softmax}(\text{MLP}(V))$，对 $V = [M \| I]$ 加权聚合，输出 $r=2$ 个 memory-augmented 实体 token $e_{h'}, e_{t'}$。
- **分类器**：采用 group bilinear 层（Zhou et al., 2021b），将每个实体切分为 $k$ 段，参数从 $d^2$ 压缩至 $d^2/k$：
$$p(r | e_{h'}, e_{t'})_s = \sigma\left(\sum_{i=1}^{k} e_{h'}^i B^i e_{t'}^i\right)$$
- **Adaptive Thresholding**：输出维度为 $R+1$（$R$ 个关系类 + 1 个阈值），沿用 ATLOP 的自适应阈值策略。

### 训练策略（关键）
- **两阶段训练**：先在 101,873 条远距离监督数据上微调全模型（包括 memory token），然后**冻结 memory token**，再在 3,053 条人工标注数据上微调。
- **SSR-PU Loss**：对于每类关系 $i$，校正先验偏移后的非负风险估计为：
$$\widehat{R}_{\text{S-PU}}(f) = \sum_{i=1}^{K}\left(\frac{\pi_i}{n_{P_i}}\sum_{j}\ell(f_i(x_j^{P_i}), +1) + \max\left(0, \left[\frac{1-\pi_i}{1-\pi_{u,i}}\cdot\frac{1}{n_{U_i}}\sum_j \ell(f_i(x_j^{U_i}), -1) - \frac{\pi_{u,i}-\pi_{u,i}\pi_i}{1-\pi_{u,i}}\cdot\frac{1}{n_{P_i}}\sum_j \ell(f_i(x_j^{P_i}), -1)\right]\right)\right)$$
其中 $\pi_{u,i} = \frac{\pi_i - \pi_{\text{labeled},i}}{1 - \pi_{\text{labeled},i}}$ 为条件正样本概率。

## 实验与结果
### 数据集
- **ReDocRED**：96 个关系类，101,873 条远距离监督训练数据，3,053 条人工标注训练，500 条 Dev/Test；含大量假负例。
- **ChemDisGene**：14 个生物医学关系类，76,942 条远距离监督训练，523 条人工标注测试。

### 主要结果（ReDocRED，Table 3）
| 设置 | TTM-RE F1 | 最佳 Baseline F1 | 提升 |
|---|---|---|---|
| 人工标注 + 远距离监督（Dev） | **84.01 ± 0.21** | 80.52 (SSR-PU) | **+3.49 F1** |
| 仅远距离监督（Dev） | **63.00 ± 0.29** | 54.46 (SSR-PU) | **+8.54 F1** |
| 仅人工标注（Test） | 79.95 ± 0.13 | 80.20 (DREEAM) | 持平 |

### 最强结果
- **Dev 数据+F1**：84.01 ± 0.21（人工+远距离监督），超越前 SOTA 3.5 个绝对 F1 点。
- **极端低标注（19% 人工标签 + 远距离监督）**：TTM-RE 超越 SSR-PU **+12 F1**（Table 6 相关分析）。
- **ChemDisGene（Table 4）**：TTM-RE 53.59 F1 vs SSR-PU 48.56 F1，**+5 F1**。
- **长尾关系改善（Table 5）**：在 Top-5 之外的低频关系上，SSR-PU+TTM 达 53.02 F1 vs SSR-PU 47.35 F1，**+5.67 F1**；Top-5 高频关系仅 +3.71 F1，说明记忆模块对低频类帮助更大。

### 消融
- Memory token 数量从 10→200，F1 从 83.20→84.01，单调提升（Table 9）。
- Layer 数从 1→4，F1 从 83.56→84.01（Table 8）。
- 基础模型换 DeBERTaV3-Large **并未提升**（Table 7），反证架构设计 > 单纯堆参数。

## 相关工作脉络
1. **ATLOP**（Zhou et al., 2021b）：Roberta-Large 单遍编码 + 自适应阈值，TTM-RE 沿用此编码器框架，在此基础上**新增 memory 读写模块**。
2. **SSR-PU**（Wang et al., 2022b）：引入 PU 学习 + 先验偏移校正处理假负例，TTM-RE **联合优化 memory 与 SSR-PU**，而非单独使用。
3. **KD-DocRE**（Tan et al., 2022a）：知识蒸馏范式（教师-学生），TTM-RE 与之本质不同——**端到端直接微调远距离数据**，不依赖蒸馏 logit。
4. **DREEAM**（Ma et al., 2023）：利用证据标签（evidence）辅助注意，TTM-RE **不依赖显式证据标注**，而是通过 memory token 隐式学习关系模式。
5. **Mention Memory**（De Jong et al., 2021）：为每个实体 mention 维护 dense 向量表，需额外 LLM 调用；TTM-RE 的记忆是**纯可学习 dense token**，无二次调用开销。
6. **Token Turing Machine**（Ryoo et al., 2023）：原始模型用于长序列视觉理解，本工作将其**首次迁移到 DocRE 场景**并改造读取机制（忽略 write 操作，只保留 read）。

## 局限性与未来方向
- **与全监督方法的性能差距仍然存在**：在仅人工标注数据时，TTM-RE 未超越 DREEAM（Test F1: 79.95 vs 80.20），说明记忆模块的核心价值在于**放大远距离监督数据的收益**，而非替代人工标注。
- **Memory token 初始化策略有待改进**：当前从正态分布初始化虽优于全零，但论文承认这是经验性选择，缺乏理论依据。
- **内存/计算成本**：memory size 和 layer 数的提升带来性能增益，但受限于计算资源（论文受限于 A6000 VRAM 只能跑到 200 token / 4 layer）。
- **记忆模块放置位置的探索**：论文发现 memory 只能放在编码之后、分类之前，**未深入探索其他放置位置**（如 encoder 内部插入 memory layers）。
- **未来方向**：探索自动学习 memory token 初始化；结合主动学习生成高质量标签；将 TTM-RE 扩展到其他信息抽取任务（命名实体识别、事件抽取等）。

## 研究启发与可借鉴点
1. **记忆模块可作为"关系类型原型库"**：将 memory token 视为可学习的关系类别压缩表示，这种思路可直接迁移到**其他需要处理长尾类别的分类任务**（如文档分类、实体链接）。
2. **架构设计 > 堆参数**：换用更大的 DeBERTaV3 未能提升 DocRE 性能，而加入 memory 模块却显著提升——启示我们在设计 IE 模型时应**优先探索结构创新而非单纯增大基座模型**。
3. **PU 学习 + 记忆模块的联合优化**：SSR-PU 单独使用时已能处理假负例，但结合 memory 后收益更大——可将此**"噪声鲁棒损失 + 记忆增强"范式**推广到其他存在大量漏标的 IE 任务。
4. **极端低标注场景的训练策略**：19% 人工标签 + 远距离监督数据下 TTM-RE 仍保持极强性能，提示**"少量高质量标注 + 大量低成本噪声数据"的组合策略**值得在其他 NLP 任务中复现。
5. **分组双线性层（Group Bilinear）的参数压缩技巧**：将 $d^2$ 压缩至 $d^2/k$ 的参数设计可直接复用于其他**实体对分类**场景，降低过拟合风险。

## 关键术语表
- **Document-level Relation Extraction (DocRE)**：在整篇文档层面识别任意两个实体间的预定义关系类型，需要跨句推理，区别于句子级 RE。
- **Token Turing Machine (TTM)**：引入可学习记忆 token 的 Transformer 变体，通过 read/write 接口与外部记忆交互，本工作仅使用 read 操作。
- **Positive-Unlabeled (PU) Learning**：仅含正样本和未标注样本的学习设置，通过风险估计校正区分正负，适用于含大量假负例的训练数据。
- **SSR-PU (Shift-corrected SR-PU)**：在 PU 学习基础上引入类别先验偏移校正，使模型在训练/测试分布不一致时仍能无偏学习。
- **Distantly Supervised Data**：通过规则/知识库（如 Wikidata + spaCy NER）自动生成的弱监督标签数据，规模大但含噪声（假负例）。
- **Group Bilinear Classifier**：将实体表示切分为 $k$ 段分别与 $k$ 个双线性矩阵相乘，将参数从 $d^2$ 压缩至 $d^2/k$。
- **Adaptive Thresholding**：为每个关系类学习独立分类阈值（而非固定 0.5），由 Zhou et al. (ATLOP) 引入，已被 DocRE 领域广泛采用。
- **Prior Shift**：训练集与测试集的正样本先验概率不同，需在 PU 学习中额外校正以避免有偏估计。

## 可复现要素
- **ReDocRED**：公开数据集（基于 DocRED 重新处理），代码将在 https://github.com/chufanwao/TTM-RE 开源（论文声明）。
- **ChemDisGene**：公开数据集，训练数据来自 CTD + PubTator。
- **基座模型**：RoBERTa-Large（Huggingface 默认权重），未提及额外预训练。
- **关键超参**：memory token 数 $m=200$，dimension $d$ 与 RoBERTa-Large hidden size 一致（1024），group bilinear 分段数 $k$ 未明确提及（论文未提及），初始化从正态分布（非全零），SSR-PU 中 $n_{U_i}$ 为假设超参（论文未提及具体值）。
- **训练环境**：NVIDIA RTX A6000（48GB VRAM），总训练时间超 75 小时（远距离监督微调为主干瓶颈）。
- **随机种子**：5 次重复运行取均值 ± 标准差。
