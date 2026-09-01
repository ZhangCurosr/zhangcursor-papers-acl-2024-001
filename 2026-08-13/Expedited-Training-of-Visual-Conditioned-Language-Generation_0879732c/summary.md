---
title: "Expedited-Training-of-Visual-Conditioned-Language-Generation"
source: https://aclanthology.org/2024.acl-long.19.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 07:07:28"
field: "多模态表示学习"
keywords: ["视觉-语言模型", "Token Merging", "单阶段训练", "视频理解", "BLIP-2", "高效预训练"]
innovations: ["提出TomeFormer替代Q-former实现单阶段单损失训练", "将Token Merging从推理加速引入训练中的语义压缩", "设计时序软Token上下文化模块实现图像模型到视频的无缝迁移"]
benchmarks: ["VQAv2", "GQA", "OK-VQA", "MSCOCO Captioning", "NoCaps", "Flickr30K", "MSR-VTT", "MSVD"]
---

# 论文速读：Expedited-Training-of-Visual-Conditioned-Language-Generation

## 一句话总结
本文提出EVL框架，通过引入基于Token Merging的TomeFormer作为视觉-语言连接器，实现了视觉条件语言生成模型的单阶段、单损失训练，在保持BLIP-2级性能的同时将训练时间缩短至1/5（仅需1/10训练数据）。

## 研究问题与动机
- BLIP-2虽采用冻结ViT+LLM的范式降低计算成本，但仍需长达8天（8×A100-80G）的Stage-1预训练，因其Q-former包含随机初始化的可学习查询和交叉注意力机制，直接端到端训练会导致模型坍塌。
- BLIP-2的Stage-1包含三个学习目标（图像-文本对比、匹配、语言生成），需多次前向传播，计算开销巨大，限制了研究者探索不同ViT架构的能力。
- 现有视频-语言模型如Video-LLaMA、VideoChat等依赖大量视频-文本数据或复杂的模态重对齐，缺乏从图像模型平滑迁移到视频任务的高效方案。
- 学术界受限于计算资源（如高校实验室），难以复现或改进BLIP-2的多阶段训练流程。

## 核心贡献（创新点）
- **单阶段训练框架**：用TomeFormer替代Q-former，仅需单一交叉熵损失和单次前向传播完成端到端训练，避免了BLIP-2的两阶段复杂流程。
- **视觉冗余压缩机制**：借鉴ToMe（Token Merging）策略，在Transformer各层按相似度逐步合并视觉token，将256个patch token压缩为更短的软prompt，兼具计算效率与语义丰富性。
- **软注意力时序Token上下文化**：提出Temporal Attentive Soft Token Contextualizing模块，在ViT内实现时序建模，无需额外的时序Q-former或可学习时序查询，保持图像-语言模型无缝适配视频任务。
- **低数据高效训练**：仅用11M（约原始数据1/10）图像-文本对训练150K步即可达到具有竞争力的性能，且训练总耗时从BLIP-2的234小时降至80小时。

## 方法详解
- **EVL_Gen-Image架构**：采用冻结的ViT（Eva-ViT-G）作为视觉编码器，冻结的LLM（OPT-2.7b/Vicuna-7b）作为语言解码器，中间插入TomeFormer作为视觉-语言连接器。
- **TomeFormer设计**：基于bert-base-uncased初始化的标准Transformer，每层通过残差连接（residual connection）集成Token Merging模块；将ViT输出的token序列按cosine相似度进行二分匹配（bipartite matching）并合并r个token，设r=19，12层共压缩228个token。
- **前向过程**：图像I经ViT编码得L个visual tokens，通过投影层f_proj₁送入TomeFormer压缩得L'个token，再经f_proj₂投影后作为soft prompt输入LLM解码生成caption。
- **训练目标**：单一交叉熵损失$\mathcal{L} = \text{CrossEntropyLoss(output, cap_{gt})}$，无额外对比/匹配损失。
- **Temporal Attentive Soft Token Contextualizing**：将视频特征v∈[B×N×L×D] reshape为[(B×N)×L×D]经ViT空间自注意力后，再reshape为[(B×L), N, D]，计算query和key矩阵，通过soft max加权融合多帧特征：$v'' = v' + \text{softmax}(\text{matmul}(q,k)) \cdot v'$，保留token数量但增强跨帧共享语义，为后续空间合并奠定基础。
- **超参数**：最大学习率1e-4，最小1e-5，线性warm-up 5000步，余弦衰减，weight decay 0.05，batch size 1600。

## 实验与结果
- **数据集**：预训练使用MSCOCO、CapFilt（含BLIP伪标签CCS、SBU、LAION-5b子集，共104M图像-文本对），排除VG数据集；视频预训练使用WebVid（2M视频-文本对）。
- **图像-文本基准（OPT-2.7b解码器）**：
  - EVL_Gen（104M数据，250K步）：VQAv2 val 48.4，GQA test-dev 30.9，OK-VQA 27.2，COCO val 139.1，训练时间133小时 vs BLIP-2的234小时。
  - EVL_Gen（104M数据，150K步）：VQAv2 46.9，GQA 30.8，OK-VQA 24.8，COCO 137.0，训练时间80小时，超越BLIP-2（250K Stage-1 + 80K Stage-2）的VQAv2 44.6、GQA 30.6、OK-VQA 26.0、COCO 137.7。
  - EVL_Gen（11M数据，150K步）：VQAv2 46.3，GQA 30.0，OK-VQA 23.0，COCO 135.1，展现低数据高效学习能力。
  - 无Stage-1的BLIP-2会坍塌（Table 1中用"x"标记），验证EVL单阶段训练的必要性。
- **NoCaps/Flickr30K零样本**：EVL_Gen在两种LLM下均超越BLIP-2，如Vicuna解码器下NoCaps CIDEr 119.0 vs BLIP-2的115.6。
- **视频Captioning（MSR-VTT）**：仅图像预训练的EVL_Gen-Image（69.8 CIDEr）已优于Video-LLaMA（59.3）、VideoChat（58.0）、VideoCoCa（63.0）；进一步视频预训练+SCST后达74.0 CIDEr。
- **MSVD Captioning**：EVL_Gen-Video达158.2 CIDEr，超过VideoCoCa（150.9）和其他基线。
- **MACs分析**：BLIP-2 Stage-1需36.7G MACs（三阶段多次前向），Stage-2需6.28G；EVL_Gen全程仅需11.9G MACs/步，无额外通信开销。

## 相关工作脉络
- **BLIP-2**：本文最直接对比基线，采用Q-former+两阶段训练，本文用TomeFormer实现单阶段替代，避免其依赖特定Eva-ViT-G预训练的局限。
- **CoCa/SimVLM/Frozen**：从零训练百亿参数模型，计算成本极高；本文聚焦高效冻结预训练范式的优化。
- **Token Merging (ToMe, Bolya et al. 2023)**：原用于ViT推理加速无需训练，本文首次将其引入视觉-语言连接器训练，实现语义压缩而非仅加速。
- **Video-LLaMA/VideoChat**：依赖额外时序Q-former或learnable temporal queries，需复杂模态重对齐；本文的时序上下文化模块不改变图像模型结构，无缝迁移。
- **CLIP/ALBEF/VLMo**：判别式视觉-语言模型，侧重于检索和分类；本文聚焦生成式任务（captioning/VQA）。
- **InstructBLIP/VideoCoCa**：后续基于BLIP-2的工作沿用冻结Q-former；本文证明可探索更多ViT变体，促进架构多样性。

## 局限性与未来方向
- **固定合并率r**：当前TomeFormer使用固定r=19，但不同图像/视频可能需不同压缩率，可变长度soft prompt是潜在改进方向。
- **缺乏文本条件视觉特征选择**：TomeFormer无法根据问题文本选择性提取视觉特征，限制了VQA性能（VQAv2上略逊于BLIP-2），未来需引入text-conditioned token selection机制。
- **视频预训练数据依赖**：虽然时序模块可无需视频预训练直接使用图像模型，但要达到最佳视频Captioning效果仍需2M WebVid对+SCST训练。
- **未评估长视频/复杂推理任务**：论文主要聚焦captioning和零样本VQA，对更长视频理解、视觉推理等任务的泛化性有待验证。

## 研究启发与可借鉴点
- **单阶段训练设计范式**：通过引入自然语义压缩机制（token merging）替代额外的表征预训练阶段，为多模态模型高效训练提供新思路，可迁移至音频-语言、3D-语言等领域。
- **Token Merging的工程实践**：将ToMe从推理加速工具转化为训练中的语义压缩组件，并证明其预训练阶段的语义 informativeness（图4可视化），为视觉冗余利用提供了新视角。
- **时序建模的轻量方案**：Temporal Attentive Soft Token Contextualizing不引入额外可学习参数（仅W_key, W_query），且对静态图像输入退化为恒等映射，是视频-语言模型高效扩展的参考设计。
- **低资源友好性**：1/10数据+1/5训练时间仍达竞争性性能，对计算受限团队具有重要实用价值，可激发更多"少即是多"的多模态研究。
- **ViT兼容性验证**：EVL对Eva-ViT-G和CLIP-L等多种编码器均有效，证明其通用性，鼓励探索更先进的ViT骨干。

## 关键术语表
**EVL (Expedited Visual Language Generation)**：本文提出的单阶段视觉-语言生成模型训练框架。
**TomeFormer**：融合Token Merging机制的Transformer，作为视觉-语言连接器替代BLIP-2的Q-former。
**Token Merging (ToMe)**：按token间cosine相似度进行二分匹配并合并冗余token的策略，原用于ViT推理加速。
**Bipartite Soft Matching**：ToMe中的匹配算法，将tokens随机分为两组并按相似度链接Top-r对进行合并。
**Temporal Attentive Soft Token Contextualizing**：视频建模模块，通过可学习的query/key加权融合多帧特征，增强时序语义共享。
**Soft Prompt**：经TomeFormer压缩后的视觉token序列，作为LLM的连续输入引导文本生成。
**Stage-1/Stage-2**：BLIP-2的两阶段训练流程，Stage-1预训练Q-former（三损失），Stage-2端到端微调。
**SCST (Self-Critical Sequence Training)**：基于自身预测作为baseline的强化学习训练策略，用于视频captioning优化。

## 可复现要素
- **数据集**：MSCOCO、CapFilt（LAION-115M+CCS-14M+MSCOCO，共104M）、WebVid（2M视频-文本对）；论文未提及数据集公开性但均为公开数据集。
- **代码**：开源，GitHub地址 https://github.com/yiren-jian/EVLGen。
- **权重**：论文使用OPT-2.7b和Vicuna-7b作为解码器（公开权重），ViT使用Eva-ViT-G（公开），TomeFormer由bert-base-uncased初始化；预训练权重需自行训练。
- **关键超参**：r=19（每层合并token数），TomeFormer层数12层，学习率1e-4（warm-up 5000步后余弦衰减至1e-5），batch size 1600，weight decay 0.05，训练硬件8×A100-80G或32×V100-32G。
- **BLIP-2复现配置**：Table 7给出复现的Stage-1（250K步）和Stage-2（80K步）详细超参，便于公平对比。
