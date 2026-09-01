---
title: "Multimodal-Prompt-Learning-with-Missing-Modalities-for-Senti"
source: https://aclanthology.org/2024.acl-long.94.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:45:15"
field: "多模态情感分析与情绪识别"
keywords: ["多模态情感分析", "缺失模态", "Prompt Learning", "参数高效微调", "多模态融合", "情绪识别"]
innovations: ["提出线性增长的三类互补 prompt（生成式/缺失信号/缺失类型）应对多模态缺失问题，参数量仅为 backbone 的 5%-10%", "发现训练期 70% 模态丢弃率为最优正则化策略，显著提升模型鲁棒性", "设计模态特定与模态共享 prompt 分离机制，分别捕获 intra-modality 和 inter-modality 特征"]
benchmarks: ["CMU-MOSI", "CMU-MOSEI", "IEMOCAP", "CH-SIMS"]
---

# 论文速读：Multimodal Prompt Learning with Missing Modalities for Sentiment Analysis and Emotion Recognition

## 一句话总结
本文提出一种基于 Prompt Learning 的多模态 Transformer 框架，通过引入生成式 prompt、缺失信号 prompt 和缺失类型 prompt 三类可学习 prompt，在冻结预训练主干模型的前提下，有效应对情感分析与情绪识别任务中的缺失模态问题，显著减少训练参数量。

## 研究问题与动机
- **真实场景中缺失模态普遍存在**：多模态情感分析涉及视频、音频、文本等多模态，受设备故障、数据损坏、隐私等原因影响，低资源场景下模态缺失频繁发生，直接导致模型性能下降。
- **完整数据训练的模型在测试时面对不完整数据会失效**：现有方法大多假设数据完整，无法处理训练/测试阶段模态缺失的情况。
- **全参数微调大型预训练模型不切实际**：对研究者计算资源要求高，且在小型数据集上微调易出现不稳定问题。
- **已有 prompt-based 缺失模态处理方法参数增长呈指数级**：Lee et al. (2023) 的 missing-aware prompts 随模态数量指数增长，本文提出线性增长方案以实现更高的参数效率。

## 核心贡献（创新点）
- **提出首个面向缺失模态的线性 Prompt Learning 多模态 Transformer 框架**：与 Lee et al. (2023) 指数增长 prompt 数量不同，本文 prompt 数量与模态数呈线性关系，大幅降低参数量。
- **设计三类互补的 prompt（生成式、缺失信号、缺失类型）**：生成式 prompt 负责恢复缺失信息，缺失信号 prompt 标注单模态缺失状态（模态特定），缺失类型 prompt 编码多模态联合缺失模式（模态共享），三者分别作用于 intra-modality 和 inter-modality 特征学习，这是以往方法未同时整合的。
- **实现极低参数量的高效训练**：所有 prompt 的 trainable parameters 仅占 backbone 的 0.5%–1%，整体 trainable parameters 仅占 backbone 的 5%–10%，且不随 backbone 规模增大而增加，仅用一块 10GB GPU 即可完成训练。
- **在四个数据集（CMU-MOSI、CMU-MOSEI、IEMOCAP、CH-SIMS）上全面超越所有基线**：在六种缺失模态组合下均取得最优或次优结果，且在文本模态缺失时带来 8%–13% 的准确率提升。
- **发现训练阶段 70% 模态丢弃率为最优**：通过系统实验验证训练集模态缺失率与模型性能的关系，确定 $\eta = 70\%$ 为最佳丢弃率。

## 方法详解
- **总体架构**：以 MulT（Tsai et al., 2019）为 backbone，冻结其全部参数，仅训练三类 prompt、MMGM 中的 Conv 层及输出层。
- **Missing Modality Generation Module（MMGM）**：使用生成式 prompt $P_G = (P_{Ga}, P_{Gv}, P_{Gt})$ 恢复缺失模态特征。当音频缺失时，利用可用模态（视频、文本）通过 Conv 块和生成式 prompt 合成缺失特征：$\hat{x}^a = f_{vt \to \hat{a}}([P_{Ga}, f_{v \to a}(x^v), f_{t \to a}(x^t)])$；当两个模态缺失时分别独立生成。
- **Missing-signal Prompts（$P_{MS}$）**：每个模态对应两个 prompt——$P_{MS}$ 表示该模态缺失（为生成特征），$P_{NMS}$ 表示该模态存在（为真实特征）。通过残差方式加到各模态特征上，使模型区分真实与生成特征，属于**模态特定（modality-specific）**提示。
- **Missing-type Prompts（$P_{MT}$）**：针对多模态联合缺失情况（共 $2^M - 1$ 种），引入缺失类型投影矩阵 $\mathbf{M_P}$，通过 $\mathbf{M_P} = \mathbf{M_a} \cdot P_{MS}^a + \mathbf{M_v} \cdot P_{NMS}^v + \mathbf{M_t} \cdot P_{MS}^t$ 计算，再得到 $P_{MT}' = P_{MT} \cdot \mathbf{M_P}$，实现**模态共享（modality-shared）**的跨模态信息捕获，参数量仅线性增长。
- **损失函数**：CMU-MOSEI/MOSI/CH-SIMS 使用 L1 loss，IEMOCAP 使用 cross-entropy loss。

## 实验与结果
- **数据集**：CMU-MOSEI（高资源预训练）、CMU-MOSI、IEMOCAP（四种情绪分类）、CH-SIMS（中文多模态情感分析），均采用六种缺失模态组合进行评测。
- **评估指标**：ACC-7、ACC、F1、MAE、Corr（MOSI/MOSEI/CH-SIMS）；ACC、F1-weighted（IEMOCAP）。
- **基线方法**：Lower Bound（LB）、Modality Substitution（MS）、Modality Dropout（MD）、MCTN、MMIN、MPMM。
- **主要结果**：在 CMU-MOSI 上，平均 ACC 达到 72.14（最优），较 MMIN（70.84）提升约 1.3 个百分点；CH-SIMS 平均 ACC 72.07，较 MMIN 提升约 0.66 个百分点；IEMOCAP 平均 ACC 67.42，领先 MPMM（65.47）。文本模态缺失场景下相比 LB 提升 8%–13%。
- **泛化实验**：在 MISA、MMIM、UniMSE 等不同 backbone 上验证，Incom† 分别比 Incom 提升 5.5、3.7、5.3 个百分点，证明方法的通用性。
- **鲁棒性**：在测试阶段不同缺失率下均保持最优性能，表明对缺失率不敏感。

## 相关工作脉络
- **SMIL（Ma et al., 2021）**：基于贝叶斯元学习的缺失模态方法，需复杂架构设计；本文通过简单 prompt 机制达到同等目标，参数更高效。
- **MCTN（Pham et al., 2019）**：通过跨模态翻译生成缺失信息，依赖循环神经结构；本文用 Conv block + prompt 以更简洁的方式实现同样功能。
- **MMIN（Zhao et al., 2021）**：学习鲁棒的联合多模态表示以预测任意缺失模态；本文方法不依赖复杂的生成架构，而是利用 prompt 引导预训练模型。
- **MPMM（Lee et al., 2023）**：同为 prompt-based 缺失模态方法，但 prompt 数量随模态数指数增长（$2^M - 1$ 个）；本文线性增长的 prompt 设计在参数量和扩展性上均有本质优势。
- **MulT（Tsai et al., 2019）**：本文采用的 backbone，通过 Crossmodal Transformer 建模非对齐多模态序列，本文在其基础上叠加 prompt 模块以增强缺失模态处理能力。
- **Prompt Learning in Multimodal（Khattak et al., 2023；Liang et al., 2022）**：探索 vision-language prompt 协同，但未处理缺失模态问题；本文聚焦于缺失模态场景下的 prompt 设计。

## 局限性与未来方向
- **当前仅处理提取后的特征而非原始数据**：MMGM 使用 Conv block 生成缺失特征，若直接处理原始多模态信号（如原始音频波形、原始视频帧），由于原始特征更复杂且模态间相关性更弱，性能可能下降。
- **生成模块结构简单**：仅使用两个 Conv block，对于高度复杂的跨模态生成任务可能存在表达能力瓶颈。
- **未来方向**：将 MMGM 扩展至原始信号层面，探索更强大的生成架构处理跨模态生成功率受限的场景。

## 研究启发与可借鉴点
- **三类 prompt 的分层设计思路可迁移**：生成式（恢复缺失）+ 缺失信号（区分真伪）+ 缺失类型（编码联合缺失模式）的三层结构，可作为缺失模态问题的通用范式，适用于其他多模态任务（如图像-语言、行为识别等）。
- **模态特定 vs 模态共享 prompt 的分离设计**：$P_{MS}$（模态特定）处理单模态缺失标注，$P_{MT}$（模态共享）处理跨模态联合缺失信息，这一分离思路可应用于需要同时关注局部和全局上下文的任务。
- **70% 训练期模态丢弃率作为强正则化手段**：该发现提示在缺失模态学习中，适度高的训练期模态扰动是关键超参，类似思路可迁移至其他鲁棒多模态训练场景。
- **参数效率的可量化指标设计**：引入参数利用率 $\xi = \text{IACC} / \ell_p$ 平衡性能与 prompt 长度，为后续 prompt 调优提供了可操作的评价维度。
- **冻结 backbone 仅训练 prompt 的训练范式**：在下游任务适配中可复用此高效微调策略，尤其适合计算资源受限的研究场景。

## 关键术语表
- **Prompt Learning**：冻结预训练模型全部参数，仅训练少量可学习 prompt 向量来适配下游任务的参数高效微调范式。
- **Generative Prompts（生成式 prompt）**：与缺失模态一一对应的可学习 prompt，配合 Conv 块从可用模态特征中合成缺失模态表示。
- **Missing-signal Prompts（缺失信号 prompt）**：每个模态特有的两类 prompt（缺失/非缺失），以残差方式注入特征，帮助模型区分真实特征与生成特征，属于模态特定（modality-specific）提示。
- **Missing-type Prompts（缺失类型 prompt）**：通过投影矩阵将多种缺失信号组合为跨模态缺失模式描述，属于模态共享（modality-shared）提示，捕捉 inter-modality 信息。
- **MMGM（Missing Modality Generation Module）**：利用生成式 prompt 和 Conv 块从可用模态中合成缺失模态特征的模块。
- **Modality Dropout（模态丢弃）**：在训练阶段随机丢弃部分模态数据以提升模型对缺失模态鲁棒性的数据增强策略。
- **ACC-7**：CMU-MOSI/MOSEI 上的七分类准确率，将连续情感评分划分为七个等级进行评估。

## 可复现要素
- **数据集**：CMU-MOSEI、CMU-MOSI、IEMOCAP、CH-SIMS，均为公开数据集。
- **代码**：已开源，地址为 https://github.com/zrguo/MPLMM。
- **权重**：论文未明确声明预训练权重开源情况。
- **关键超参**：prompt 长度 $\ell_p = 16$（默认），训练模态丢弃率 $\eta = 70\%$，batch size = 64，epochs = 30，learning rate = $1 \times 10^{-3}$，Adam 优化器。
