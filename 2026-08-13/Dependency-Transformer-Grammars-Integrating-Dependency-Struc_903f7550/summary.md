---
title: "Dependency-Transformer-Grammars-Integrating-Dependency-Struc"
source: https://aclanthology.org/2024.acl-long.84.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:56:50"
field: "句法语言模型"
keywords: ["dependency parsing", "syntactic language modeling", "constrained attention", "transformer grammar", "inductive bias"]
innovations: ["通过约束注意力掩码模拟依存转换系统，首次将依存结构归纳偏置引入 Transformer 语言模型", "设计基于栈深度的相对位置编码和组合式弧表示（w+arc）", "系统比较 arc-standard/eager/swift 三种转换系统在语言建模中的表现"]
benchmarks: ["BLLIP-LG", "BLiMP", "SG", "PTB"]
---

# 论文速读：Dependency-Transformer-Grammars-Integrating-Dependency-Struc

## 一句话总结
论文提出了 Dependency Transformer Grammars (DTGs)，一种将显式依存结构归纳偏置引入 Transformer 的语言模型，通过约束注意力机制模拟依存转换系统。实验表明，DTG 在句法泛化上显著优于 constituency-based 模型，同时保持相当的语言建模困惑度。

## 研究问题与动机
1. Transformer 语言模型虽强大但缺乏句法结构的归纳偏置，引入句法结构被假设能改善泛化。
2. 现有 syntactic language models 均基于 constituency trees 构建，但基于依存结构的 Transformer 句法语言模型性能尚不明确。
3. 依存结构关注 token 间关系，与 Transformer 的 self-attention 机制具有内在相似性，暗示潜在协同效应。
4. 此前的 generative dependency parsing 工作多聚焦于解析本身，仅在语言建模方面浅尝辄止。

## 核心贡献（创新点）
1. **首次提出基于依存转换系统的 Transformer 约束注意力机制**：通过修改注意力掩码实现 STACK 和 COMPOSE 两种注意力形式，模拟 arc-standard 转换系统的栈操作。
2. **设计基于栈深度的相对位置编码**：针对 TRANSFORMER 原有的相对距离编码，DTG 引入反映栈深度的相对位置编码，区分栈内 token 的位置关系。
3. **提出组合式弧表示（w + arc）**：将 LEFTARC/RIGHTARC 操作嵌入与 head token 嵌入相加，分别承担方向指示和栈位置恢复功能。
4. **系统比较多种依存转换系统**：除 arc-standard（DTG）外，还设计了基于 arc-eager（DTG-eager）和 arc-swift（DTG-swift）的变体并对比分析。

## 方法详解
**转换序列建模**：DTG 将句子与依存树的联合建模转化为生成转换序列，序列包含三种操作：GEN（生成 token）、LEFTARC（建左弧）、RIGHTARC（建右弧）。为支持 COMPOSE 和 STACK 两种注意力，将 LEFTARC/RIGHTARC 复制为 LEFTARC/RIGHTARC 和 LEFTARC2/RIGHTARC2 两对。

**注意力掩码设计（Algorithm 1）**：
- **STACK attention**：在 GEN/LEFTARC2/RIGHTARC2 步骤触发，当前位置 attend 栈内所有未被 mask 的位置，收集栈信息以预测下一步。
- **COMPOSE attention**：在 LEFTARC/RIGHTARC 步骤触发，仅 attend 栈顶两个 token（head-dependent 对），然后将这两个位置 mask 掉（模拟 pop），计算出的表示作为吸收了 dependent 信息的 head 替换，重新入栈。

**相对位置编码**：基于 Transformer-XL 的相对位置编码改进。STACK 注意力的距离定义为 $R_{ij} = d(i) - d(j)$，其中 $d$ 为栈深度；COMPOSE 注意力中 head 的位置编码为 0，dependent 为 -1，合成后继承 head 的栈深度。

**弧表示**：弧步骤的输出为 LEFTARC/RIGHTARC 的嵌入与 head token 嵌入之和，既传递弧方向信息又保留 head 的位置语义。

**推理约束**：LEFTARC/RIGHTARC 至少需要栈中有两个 token；POP 仅在栈顶已被识别为 right dependent 时合法；arc-swift 的 $k$ 不超过栈大小。

## 实验与结果
**数据集**：BLLIP-LG（Charniak et al., 2000），使用 Hu et al. (2020) 的训练切分。依存树由 Biaffine-roberta 解析器生成。

**评估基准**：
- **Perplexity**：在 BLLIP-LG test set 上以 300 棵采样树近似 marginal probability $p(\mathbf{x})$。
- **BLiMP**：67 个语法对测试，统计模型对 grammatical 句子赋予更高概率的比例。
- **SG 测试套件**：6 种细粒度句法现象，基于 surprisal 值评估不等式。

**关键结果**：
- DTG PPL = 14.9，与 TXL (tokens) 的 14.8 相当，低于 TG（18.4）和 Pushdown（19.9）。
- DTG 在 BLiMP 上得分为 76.1，优于 TG（73.5）和 PLM（75.1）。
- **DTG 在 SG 上得分 83.9，显著超越所有基线**（TG: 82.5, Pushdown: 82.3, PLM: 80.2）。
- Parse reranking：DTG 和 TXL (trans) 在 PTB 测试集上 UAS 均为 97.0，略优于 proposal parser（96.9）。

**最强结果**：DTG 在 SG 上以 83.9 分超越最佳 constituency-based 模型 TG 约 1.4 个百分点，证明依存结构比 constituency 结构更能引导 Transformer 语言模型。

## 相关工作脉络
1. **RNNG (Dyer et al., 2016)**：最早将 constituency 结构融入 RNN 语言模型的工作，使用递归网络构建短语表示。本文与其一脉相承，但转向 Transformer 架构并切换至依存结构。
2. **PLM (Qian et al., 2021) / Transformer Grammars (Sartran et al., 2022)**：在 Transformer 上使用 constituency 结构约束注意力。DTG 与其定位差异在于使用依存树而非 constituency 树，且注意力模式完全不同。
3. **Pushdown Layers (Murty et al., 2023)**：通过梯度学习而非硬编码结构先验来引入递归结构。DTG 的结构约束是完全确定的（由 dependency transition system 驱动），更具语言学可解释性。
4. **Generative dependency parsing (Titov & Henderson, 2007; Cohen et al., 2011)**：早期在概率非确定性解析器中使用 generative 范式。本文将其扩展到 Transformer 架构并关注语言建模性能。
5. **Prange et al. (2022)**：同时引入 constituency 和 dependency graph 但需 gold trees 指导生成。DTG 无需 gold trees，仅依赖训练时的标注树，推理时无需额外输入结构。

## 局限性与未来方向
1. **依赖外部解析器**：训练需 Biaffine-roberta 生成的依存树，在缺乏高精度依存解析器的语言上可能无益。
2. **结构限制**：仅考虑标准依存表示下的无标签 projective 树，未探索 Universal Dependencies、非 projective 结构或文档级设置。
3. **无法利用高效加速技术**：特殊的注意力掩码和相对位置编码阻碍了 Rotary Position Embeddings 和 Flash Attention 的直接应用。
4. **概率评估的近似性**：边际概率需采样 300 棵树近似，计算开销大且仅为上界估计。
5. **BLiMP 语义劣势**：结构约束导致部分语义信息被 mask 掉，在需要语义知识的 BLiMP 上表现不如 TXL (trans)。

## 研究启发与可借鉴点
1. **注意力掩码模拟计算过程**：通过自定义 attention mask 精确模拟 stack-based 算法的状态转移，是一种将符号推理嵌入神经架构的通用思路，可迁移至其他 parsing 或图构建任务。
2. **复合操作嵌入设计**：将操作类型嵌入与目标 token 嵌入相加的弧表示方法简洁有效（w+arc > arc-only > w-only），对类似"操作-对象"组合的场景有借鉴价值。
3. **SG 测试套件作为句法泛化金标准**：相比 BLiMP，SG 的 inequality-based 评估更纯粹地测量句法知识，建议后续研究同时报告两项指标以全面评估。
4. **多种转换系统对比的价值**：同时实现 arc-standard/eager/swift 三套变体并进行对比，揭示了不同转换系统的特性差异，为后续工作提供了完整的选型参考。
5. **与标准 LM 的融合机会**：文中明确指出将 syntactic LM 与标准 LM 结合以获得"two best worlds"是有趣方向，可探索混合注意力或门控机制。

## 关键术语表
**Dependency Transformer Grammars (DTGs)**：一种将依存句法结构的归纳偏置引入 Transformer 语言模型的新型架构，通过约束注意力模拟依存转换系统。
**Arc-standard 转换系统**：一种自底向上构建依存树的增量解析系统，定义 SHIFT、LEFTARC 和 RIGHTARC 三种操作。
**STACK attention**：DTG 中用于收集栈内所有 token 信息的注意力形式，对应于每个需要预测下一步操作的位置。
**COMPOSE attention**：DTG 中用于构建 head-dependent 弧的注意力形式，仅 attend 栈顶两个 token 并将其 pop。
**Relative positional encoding (stack-depth based)**：DTG 中基于栈深度的相对位置编码，使模型能感知 token 在栈中的层级位置。
**BLiMP**：The Benchmark of Linguistic Minimal Pairs，包含 67 类语法对的语言学泛化评估基准。
**SG (Syntactic Generalization) 测试套件**：由 Hu et al. (2020) 提出的基于 surprisal 不等式的六类句法现象评估工具。
**Parse reranking**：用语言模型对由外部解析器生成的候选依存树重新排序，以评估模型对句法结构的理解能力。

## 可复现要素
- **数据集**：BLLIP-LG（Charniak et al., 2000），训练切分来自 Hu et al. (2020)；代码已开源于 https://github.com/zhaoyd1/Dep_Transformer_Grammars。
- **代码**：开源。
- **权重**：论文未明确提及预训练权重发布。
- **关键超参**：16 层 Transformer，252M 参数；embedding 学习率乘数 2.0；采样 300 棵树近似边际概率；beam size 固定为 300（SG 评估）；使用 SentencePiece 分词；训练耗时约 35 小时/NVIDIA A6000 GPU。
- **解析器**：依赖 Supar 实现的 Biaffine-roberta 解析器生成训练标注树。
