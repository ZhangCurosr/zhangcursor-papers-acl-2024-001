---
title: "SPARSEFIT: Few-shot Prompting with Sparse Fine-tuning for Jointly Generating Predictions and Natural Language Explanations"
source: https://aclanthology.org/2024.acl-long.113.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:01:12"
field: "可解释自然语言处理"
keywords: ["稀疏微调", "少样本学习", "自然语言解释", "参数高效微调", "可解释AI", "T5", "PEFT"]
innovations: ["首次系统探索稀疏微调在联合预测与NLE生成任务中的应用", "提出SPARSEFIT策略，通过微调少量关键层组合（如6.8%参数）实现媲美全量微调的性能", "发现并验证了轻量级组件组合在少样本解释生成中的高效性"]
benchmarks: ["e-SNLI", "ECQA", "SBIC", "ComVE", "FEB Benchmark"]
---

# 论文速读：SPARSEFIT: Few-shot Prompting with Sparse Fine-tuning for Jointly Generating Predictions and Natural Language Explanations

## 一句话总结
SPARSEFIT提出一种稀疏少样本微调策略，通过微调预训练语言模型（如T5）的特定层或层组合（如Self-attention Query + Layer Normalization，仅占6.8%参数），在极少标注数据（每类48条）下协同生成任务预测与自然语言解释（NLEs），在预测准确性和解释质量上均优于主流参数高效微调方法并逼近全量微调效果。

## 研究问题与动机
- **核心问题**：如何在训练数据稀缺（few-shot）场景下，高效微调大型语言模型，使其既能准确预测任务标签，又能生成人类可理解的自然语言解释（NLEs）。
- **现有方法不足**：
  1. 传统NLE模型需要大规模人工标注的解释数据集，获取成本高且不适用于某些领域。
  2. 现有少样本方法（如FEB）通常对全部模型参数进行微调，计算开销大，难以应用于拥有数十亿参数的大模型。
  3. 其他参数高效微调技术（PEFT，如LoRA、AdaLoRA、(IA)3）虽减少参数量，但在联合生成预测和高质量NLEs的任务上，其性能（尤其是解释质量）尚存提升空间，且可能引入额外计算复杂度。

## 核心贡献（创新点）
- **首次系统探索稀疏微调在联合预测与NLE生成中的应用**：将BitFit仅微调bias的思路扩展至更广泛的架构组件（层及层组合），并首次在该联合生成任务背景下系统分析稀疏微调的效果。
- **提出SPARSEFIT策略**：一种通过选择性地微调少量关键参数子集（如约6.8%的Self-attention Query和Layer Normalization参数），结合提示学习（prompting），在少样本设置下实现媲美全量微调的预测与解释生成性能。
- **发现有效的稀疏微调配置规律**：揭示了某些轻量级组件（如LayerNorm仅0.02%参数）单独微调即可提升部分任务表现，而组合关键组件（如Attention.Q + LayerNorm）能在参数占比极低的情况下取得最优的性能-效率权衡。
- **全面评估与基准对比**：在四个基准NLE数据集上，与多种SOTA PEFT方法及全量微调进行对比，证明SPARSEFIT在多项指标（包括自动指标和人工评估）上具有竞争力或更优，且计算开销（FLOPS）最低。
- **在更大模型架构上验证有效性**：将SPARSEFIT成功应用于Llama 2-7B，展示了该策略的通用性，并在该架构上同样优于对比的PEFT方法。

## 方法详解
- **核心思想**：采用提示学习框架，将NLE生成任务转化为条件文本生成，目标输出格式为`"[label] because [explanation]"`。模型以少样本示例（每个任务仅48个训练样本）进行学习。
- **稀疏微调配置空间**：系统性地探索微调T5架构中不同组件或组件组合的参数，包括：
  1. 编码器块（~41%参数）
  2. 解码器块（~54%参数）
  3. LM头（~5%参数）
  4. 整个Self-attention层（Q, K, V矩阵，~20%参数）
  5. 前馈网络（Feed-forward networks）
  6. 归一化层（Layer Normalization，~0.02%参数）
  7. 上述组件中不包含编码器/解码器的任意两两组合（如Attention.Q + LayerNorm，~6.84%参数）
- **训练设置**：冻结非选定组件的所有参数，仅对被选组件的参数进行微调。使用AdamW优化器，固定学习率0.00003，训练25个epoch，批量大小为4。在NVIDIA P100 GPU上平均训练时间约为23.2分钟。
- **关键公式/损失**：采用标准序列到序列（seq2seq）的条件文本生成损失（负对数似然损失），优化目标是最小化生成正确标签和解释文本的概率负对数似然。

## 实验与结果
- **数据集**：遵循FEB基准，使用四个NLE数据集：e-SNLI（自然语言推理）、ECQA（常识问答）、SBIC（侮辱性分类）、ComVE（常识分类）。每个数据集仅使用48个训练样本（通过60个不同的train-validation split进行评估）。
- **评估指标**：任务准确性（Accuracy）和自然语言解释质量（使用Normalized BERTScore， nBERTs；空解释或错误预测对应的解释得分为0）。 additionally, 进行小规模人工评估（600个样本，评估解释是否支持预测）。
- **主要模型**：T5 (base, large, 3b) 和 UNIFIEDQA (基于T5架构)，以及Llama 2-7B。
- **关键结果（T5-large）**：
  - **最佳配置**：微调`LayerNorm + Attention.Q`（6.84%参数）在考虑参数效率时综合表现最优，平均准确率70.1%，平均nBERTs 63.7%。
  - **任务性能**：部分稀疏配置（如仅微调LayerNorm，0.02%参数）在e-SNLI和ECQA上的准确率甚至显著超越全量微调基线。SPARSEFIT的整体性能差距不超过基线的15%。
  - **解释质量**：微调Decoder（54.6%参数）在e-SNLI和ComVE上的解释质量超过全量微调。组合组件（如Att.Q+LN）通常产生更稳定、方差更小的解释质量。
  - **对比PEFT方法（Table 2）**：SPARSEFIT (Att.Q+LN) 在四个数据集上的平均准确率和nBERTs均优于LoRA、AdaLoRA、(IA)3，且其FLOPS（2.37e14）为所有对比方法中最低。
  - **大模型验证（Table 3, Llama 2-7B）**：SPARSEFIT (Att.Q+LN, 7.97%参数) 在所有四个数据集的平均准确率和nBERTs上均优于最佳PEFT基线AdaLoRA。
  - **人工评估（Table 4, T5-large）**：SPARSEFIT (Att.Q+LN) 平均得分42.12，高于全量微调（36.91）和AdaLoRA（35.55）。(Table 5, Llama 2-7B) SPARSEFIT平均得分63.33，高于全量微调（36.11）和AdaLoRA（57.22）。
- **发现**：随着模型规模增大（T5-base -> large -> 3b），相同稀疏配置的性能通常提升，且解释生成的空输出问题有所缓解。

## 相关工作脉络
- **BitFit (Zaken et al., 2022)**：仅微调Transformer层中的bias参数，证明了极少量参数微调的潜力。本文将其思想扩展至更广泛的架构组件。
- **FEB Benchmark & Prompt-based FT (Marasovic et al., 2022)**：提出了少样本NLE学习基准和提示微调策略，是本文任务设定和评估协议的主要基础。
- **Prefix-Tuning (Li & Liang, 2021)**：在输入端添加可训练向量，冻结主模型参数。SPARSEFIT则直接修改主模型内部特定层的参数。
- **LoRA (Hu et al., 2022) & AdaLoRA (Zhang et al., 2023)**：通过注入低秩矩阵适应模型。SPARSEFIT不涉及架构变化，通过选择性地更新已有参数实现效率。
- **(IA)3 (Liu et al., 2022)**：通过学习向量缩放中间激活。本文在实现中调整了(IA)3以缩放所有层而非仅注意力模块以提升性能。
- **定位差异**：不同于PEFT方法通过增加旁路结构或缩放机制来适应任务，SPARSEFIT专注于识别并微调预训练模型内部*最直接相关*的少量原始参数，旨在以最低的参数改动和计算开销实现预测与解释的联合生成。

## 局限性与未来方向
- **解释质量仍不理想**：人工评估显示，即使最佳配置，也仅有约25%的NLEs能充分解释预测，近一半的解释不足以支持预测。最常见的问题是解释不完整、不切题或无意义。
- **空解释现象**：部分高效稀疏配置（如仅微调LayerNorm）在某些数据集上倾向于生成空解释，尤其是在较小的模型上。这与预训练数据缺乏解释信息有关。
- **泛化性与适用性**：研究主要聚焦于T5和UNIFIEDQA架构，在Llama 2上的成功暗示了泛化潜力，但其在其他架构（如纯Encoder模型）上的表现未充分探讨。
- **超参数敏感性**：学习率、epoch数等超参数的选择可能影响不同稀疏配置的效果，其通用最优设置值得进一步研究。
- **未来方向**：探索更智能的组件选择算法；研究如何结合外部知识或检索增强来改善解释质量；将稀疏微调策略应用于更多样化的可解释AI任务和模型架构。

## 研究启发与可借鉴点
- **细粒度参数重要性分析**：SPARSEFIT系统化探索不同架构组件的贡献，这种思路可用于分析其他NLP任务中哪些参数子集最关键，指导高效的参数高效微调策略设计。
- **轻量级组件的组合效应**：发现独立微调某些极轻量级组件（如LayerNorm）有效，而组合关键组件（如Query投影+LayerNorm）能产生协同效应，这为设计新的PEFT方法提供了启发（如针对性地微调注意力机制的关键部分）。
- **性能-效率权衡的综合评估**：不仅比较准确率，还综合考虑FLOPS、参数比例和人工评估，这种多维度评估框架值得在类似研究中借鉴。
- **提示学习与稀疏微调的结合**：将明确的提示格式（`[label] because [explanation]`）与参数级的稀疏更新相结合，是一种简洁有效的少样本教学范式，可推广至其他需要生成结构化输出的任务。
- **可迁移性验证**：在T5和Llama 2上均验证有效性，表明该方法可能适用于多种 Transformer 架构，为跨架构的轻量级自适应提供了参考。

## 关键术语表
- **SPARSEFIT**: 本文提出的方法，指代通过选择性微调预训练语言模型中少量特定层或层组合参数，并结合提示学习的少样本联合预测与NLE生成策略。
- **Natural Language Explanations (NLEs)**: 自然语言解释，模型用自由文本形式给出的、解释其预测理由的输出。
- **Few-shot Learning**: 少样本学习，指仅使用少量（如几十到几百个）带标签样本进行模型训练或微调的设置。
- **Parameter-Efficient Fine-Tuning (PEFT)**: 参数高效微调，指通过微调模型中一小部分参数或引入少量额外参数来适应下游任务的微调技术集合（如LoRA, Prefix-Tuning）。
- **Normalized BERTScore (nBERTs)**: 归一化BERT分数，本文采用的评估NLE质量的自动指标，对错误或空解释给予零分，以更好地关联人类判断。
- **T5 / UNIFIEDQA / Llama 2**: 研究所使用的预训练语言模型系列。T5是文本到文本Transformer；UNIFIEDQA是基于T5架构的问答模型；Llama 2是Meta发布的大规模 Decoder-only 模型。
- **FEB Benchmark**: 由Marasovic等人提出的少样本自我合理化（NLE生成）基准和评测协议，本文沿用其数据集划分和评估方式。

## 可复现要素
- **数据集**：e-SNLI, ECQA, SBIC, ComVE。论文遵循FEB基准的60个少样本train-validation split。公开数据集。
- **代码**：论文未明确提供开源代码链接，但提到了使用Hugging Face的PEFT库。实验细节（学习率、epoch、batch size、优化器）有描述。
- **模型**：使用了公开的T5 (base, large, 3b), UNIFIEDQA, Llama 2-7B预训练权重。
- **关键超参**：学习率=0.00003，epoch=25，batch size=4，optimizer=AdamW，训练设备=NVIDIA P100。
