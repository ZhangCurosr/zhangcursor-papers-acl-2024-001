---
title: "Rewriting-the-Code-A-Simple-Method-for-Large-Language-Model"
source: https://aclanthology.org/2024.acl-long.75.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:50:13"
field: "代码搜索与大语言模型"
keywords: ["代码搜索", "Generation-Augmented Retrieval", "大语言模型", "代码风格归一化", "稀疏检索", "密集检索"]
innovations: ["提出 ReCo 方法，通过重写代码库中原始代码的风格来弥合 LLM 生成示例代码与真实代码之间的风格偏差", "首次提出 Code Style Similarity (CSSim) 指标，从变量命名、API 调用和代码结构三维度量化代码风格相似度"]
benchmarks: ["CoNaLa", "MBPP", "APPS", "MBJP"]
---

# 论文速读：Rewriting the Code: A Simple Method for Large Language Model Augmented Code Search

## 一句话总结
本文在 Generation-Augmented Retrieval（GAR）框架基础上提出 ReCo 方法，通过对代码库中的原始代码进行风格重写的归一化处理，弥合 LLM 生成示例代码与真实代码之间的风格偏差，从而显著提升稀疏检索和密集检索的检索精度。

## 研究问题与动机
- **模态不匹配**：自然语言查询与代码片段语法差异大，词汇重叠少，稀疏检索效果受限；密集检索依赖训练阶段共现词对，若二者极少共现则语义关联难以捕捉。
- **GAR 在代码搜索中提升有限**：LLM 生成的示例代码在功能上正确，但风格（变量命名、API 选择、代码结构）与代码库中真实代码存在显著偏差，导致 GAR 改进幅度受限。
- **现有评估指标无法衡量风格相似度**：BLEU、ROUGE-L、CodeBLEU 等基于精确 token 匹配的指标对代码风格差异不敏感，需要新的度量标准。
- **高计算成本与低延迟场景冲突**：LLM 实时生成示例代码带来推理开销，对低延迟应用场景构成挑战。

## 核心贡献（创新点）
1. **提出 ReCo 方法（Rewriting the Code）**：在 GAR 基础上额外重写代码库中的原始代码以对齐风格，与 GAR 的本质区别在于同时修改查询侧和被检索代码侧，而非仅扩展查询。
2. **首次提出 Code Style Similarity（CSSim）指标**：从变量命名、API 调用和代码结构三个维度，基于编辑距离和树编辑距离量化代码风格相似度，区别于传统基于 n-gram 匹配的现有指标。
3. **系统验证 ReCo 在多种检索设定下的有效性**：覆盖稀疏（BM25）、零样本密集和微调密集三种设置，在四个数据集上均显著提升检索性能。
4. **揭示现有指标在风格评估中的局限性**：通过矛盾分析法证明 BLEU/ROUGE-L/CodeBLEU 与检索性能提升方向不一致，而 CSSim 具有更强的解释力。

## 方法详解
- **生成示例代码**：使用 few-shot prompting，提示词包含指令"请根据描述生成 Java/Python 代码片段"及 K=4 个随机采样的训练集查询-代码对，得到示例代码 $c_q = \text{LLM}(q, \text{GEN})$。
- **重写代码**：分两步，先将原始代码 $c$ 用 SUM 提示（指令："请用一句话总结该代码的主要功能"）摘要为自然语言描述 $q_{sum} = \text{LLM}(c, \text{SUM})$，再以 $q_{sum}$ 作为查询生成重写代码 $c_c = \text{LLM}(q_{sum}, \text{GEN})$。
- **稀疏检索增强**：将原始查询 $q$ 重复 N 次并与 N 个示例代码拼接为增强查询 $q^+$；将原始代码 $c$ 与重写代码 $c_c$ 拼接为增强候选代码 $c^+$，输入 BM25 等稀疏检索系统。
- **密集检索增强（InfoNCE 损失）**：以均值池化融合原始查询/代码表示与生成内容表示，得到增强表示 $\mathbf{v}_q^+ = \frac{1}{2N}(N \cdot G(q) + \sum G(c_q))$，代入标准 InfoNCE 对比损失进行训练。
- **理论依据**：原始代码 $c \sim P(q)$，示例代码 $c_q \sim \text{LLM}(q)$，两者分布不同导致风格偏差；重写代码 $c_c \sim \text{LLM}(q_{sum})$，当 $q_{sum}$ 正确反映代码功能时近似 $\text{LLM}(q)$，使两者风格趋同。

## 实验与结果
- **数据集**：CoNaLa（Python，在线社区）、MBPP（Python，通用编程）、APPS（Python，编程竞赛）、MBJP（Java，通用编程）。
- **评估基线**：BM25（稀疏）、CodeBERT/UniXcoder/Contriever/CodeT5+（零样本密集）、CodeBERT/UniXcoder（微调密集）。
- **核心结果（MRR%，ReCo vs GAR 最佳提升）**：
  - 稀疏检索（BM25）：MBPP +35.7（12.6→70.8）、MBJP +31.8（11.3→65.3）
  - 零样本密集（Contriever）：APPS +27.6（9.6→41.6）、MBPP +16.1（55.3→87.4）
  - 微调密集（UniXcoder）：APPS +23.6（24.3→58.1）、MBPP +5.2（81.2→94.2）
- **最强结果**：BM25 + ReCo（GPT-3.5）在 MBPP 上 MRR 达 70.8，相比零样本神经模型仍有竞争力。
- **CSSim 有效性**：CSSim 与 MRR 提升呈显著正相关，而 BLEU/CodeBLEU/ROUGE-L 大多散落于第四象限（指标改善但检索性能下降）。

## 相关工作脉络
- **GAR（Mao et al., 2020; Gao et al., 2022 HyDE; Wang et al., 2023 query2doc）**：本文建立在 GAR 框架上，指出其在代码搜索中因风格偏差导致提升有限，ReCo 通过同步重写被检索代码来弥补这一不足。
- **CodeBERT（Feng et al., 2020）/ CodeT5（Wang et al., 2021）/ UniXcoder（Guo et al., 2022）/ CodeT5+（Wang et al., 2023b）**：预训练代码模型系列，本文将其作为检索器基线，验证 ReCo 在不同架构上的通用性。
- **CodeBLEU（Ren et al., 2020）/ Crystal-BLEU（Eghbali & Pradel, 2022）**：代码生成评估指标，本文证明其不适合衡量代码风格相似度，由此引出 CSSim。
- **Contriever（Izacard et al., 2021）**：通用非监督密集检索模型，非专为代码设计，本文验证 ReCo 可迁移至通用检索模型。

## 局限性与未来方向
- **实时推理延迟**：GAR 和 ReCo 均需实时调用 LLM 生成示例代码，对低延迟场景不适用；重写代码可预计算但生成示例代码仍为在线过程。
- **未在大代码库（如 CodeSearchNet，超百万查询）上评估**：因 LLM 生成成本过高（估算需两个多月），仅在中小规模数据集上验证。
- **Code Llama-34B 性能下降**：大模型倾向于生成提示中的所有示例代码而非仅目标代码，导致输出被截断。
- **未来方向**：训练专用的轻量级代码风格归一化模型以降低 LLM 依赖；设计受控提示以生成多样化风格的示例代码。

## 研究启发与可借鉴点
1. **"双向归一化"思想**：不仅增强查询侧（GAR），也增强被检索侧（ReCo），这一思路可迁移至其他检索场景中跨模态/跨风格对齐问题。
2. **CSSim 指标设计范式**：从变量命名、API 调用、代码结构三个正交维度评估风格相似度，结合 IDF 加权与 Tree Edit Distance，为代码风格分析提供了可复用的评估框架。
3. **few-shot prompting 策略细节**：每次重复调用 LLM 时重新采样 in-context 示例以引入多样性，这一设计值得在生成增强类任务中借鉴。
4. **通过"摘要-再生成"间接控制生成风格**：直接要求 LLM 重写代码输出过于相似，而先摘要再生成的两阶段方式既能保留功能语义又能改变风格，可作为代码变换任务的一般性技巧。

## 关键术语表
- **Generation-Augmented Retrieval (GAR)**：通过 LLM 生成与查询相关的参考内容（如示例代码）来扩展查询，再送入检索系统的框架。
- **ReCo (Rewriting the Code)**：在 GAR 基础上额外对代码库中的原始代码进行风格重写，使其与 LLM 生成的示例代码风格对齐。
- **Code Style Similarity (CSSim)**：首个从变量命名、API 调用和代码结构三个维度基于编辑距离量化代码风格相似度的评估指标。
- **Tree Edit Distance (TED)**：衡量两棵抽象语法树（AST）之间转换所需的最少插入、删除、替换操作数，用于量化代码结构差异。
- **InfoNCE Loss**：对比学习中的常用损失函数，拉近正样本对表示距离、推远负样本对表示距离。
- **MRR (Mean Reciprocal Rank)**：取每个查询的正確答案排名倒数平均值，作为检索任务的主要评估指标。

## 可复现要素
- **数据集**：CoNaLa、MBPP、APPS、MBJP，论文中提供了统计信息，APPS 和 MBJP 为随机采样子集。
- **代码/权重**：源码开源（https://github.com/Alex-HaochenLi/ReCo）；各模型权重均来自 HuggingFace 公开 checkpoint（UniXcoder-base、contriever-msmarco、codet5p-110m-embedding、codebert-base）。
- **关键超参**：in-context 示例数 K=4，温度=1，代码摘要最大输出长度 128，代码生成最大输出长度 256，密集模型输入长度 256，batch size=32，训练 10 轮，学习率 5e-6，Adam 优化器。
