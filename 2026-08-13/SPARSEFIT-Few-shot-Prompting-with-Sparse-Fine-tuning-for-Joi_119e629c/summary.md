---
title: "SPARSEFIT-Few-shot-Prompting-with-Sparse-Fine-tuning-for-Joi"
source: https://aclanthology.org/2024.acl-long.113.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:50:40"
field: "可解释自然语言处理"
keywords: ["sparse fine-tuning", "few-shot learning", "natural language explanation", "parameter-efficient fine-tuning", "PEFT", "explainable NLP"]
innovations: ["首次系统性探索稀疏微调在联合生成预测与自然语言解释任务中的应用", "提出 SPARSEFIT 最优配置仅微调 6.84% 参数即达到与全参数微调相当性能", "发现不同稀疏组件组合具有显著协同增益并验证于多模型架构"]
benchmarks: ["e-SNLI", "ECQA", "SBIC", "ComVE", "FEB benchmark"]
---

# 论文速读：SPARSEFIT: Few-shot Prompting with Sparse Fine-tuning for Jointly Generating Predictions and Natural Language Explanations

## 一句话总结
本文提出 **SPARSEFIT**，一种将离散 prompt 与稀疏微调（仅微调模型的一小部分参数）相结合的策略，用于在少量训练样本（few-shot）下联合生成预测标签与自然语言解释（NLE）。实验表明，仅微调约 6.8% 的参数即可达到与全参数微调相当的性能，且在多项数据集上优于 LoRA、AdaLoRA 等现有 PEFT 方法。

## 研究问题与动机
- 生成自然语言解释（NLE）的模型通常需要大量人工标注的 NLE 数据进行训练，标注成本高昂，对部分应用场景难以实现。
- 仅有少量 NLE 样本（few-shot 设置）时，对拥有数十亿参数的预训练语言模型（PLM）进行全参数微调计算代价过大。
- 现有 Parameter-Efficient Fine-Tuning（PEFT）方法（如 LoRA、Prefix-Tuning）主要针对下游任务性能优化，尚未系统研究其在联合生成 NLE 场景下的表现。
- 目前针对联合生成预测和 NLE 的 few-shot 方法，缺乏对 PLM 内部不同模块进行稀疏微调的深入分析与系统性指导。

## 核心贡献（创新点）
1. **首次系统性探索稀疏微调在 NLE 生成任务中的应用**：将稀疏微调从仅训练 bias（BitFit）扩展到编码器/解码器块、self-attention 矩阵、归一化层及它们的组合，为 NLE 生成任务提供组件级微调指南。
2. **提出 SPARSEFIT 策略：最优配置仅需微调约 6.8% 参数（LayerNorm + Attention.Q）**：该组合在 T5-large 和 UNIFIEDQA 的四个数据集上均取得最佳参数-性能权衡，且无需引入额外参数或架构复杂度变化。
3. **发现组件组合具有显著的协同增益**：两个单独有效的稀疏配置组合后，其性能显著优于任一配置单独使用，验证了不同模块在 NLE 生成中的互补性。
4. **将 SPARSEFIT 扩展至 Llama 2-7B 等大模型并验证有效性**：证明了该策略不局限于 T5 类编码器-解码器架构，在 decoder-only 架构上同样优于 AdaLoRA 等 PEFT 方法。

## 方法详解
- **SPARSEFIT 核心思路**：基于 Zaken et al. (2022) 的 BitFit（仅微调 bias）扩展，系统探索 PLM 中不同层/层的组合进行微调，其余参数冻结。
- **可微调的组件类型**：(1) 编码器块（~41% 参数）、(2) 解码器块（~54% 参数）、(3) LM head（~5% 参数）、(4) Self-attention 层（Q/K/V 矩阵，各约 6%）、(5) Feed-forward 网络、(6) Layer Normalization 层（~0.02% 参数）、(7) 上述组件的所有两两组合（不含 encoder-decoder 组合）。
- **训练目标格式**：模型被微调以条件生成文本 `[label] because [explanation]`，即同时输出预测标签和对应的自然语言解释。
- **Prompt 机制**：沿用 Marasovic et al. (2022) 的 prompt-based few-shot 框架，将少量训练样本以 prompt 形式输入，结合稀疏微调使模型适应新任务。
- **超参数**：AdamW 优化器，固定学习率 0.00003，batch size=4，训练 25 个 epoch；使用 HuggingFace PEFT 库实现基线方法。

## 实验与结果
- **数据集**：e-SNLI（自然语言推理）、ECQA（常识问答）、SBIC（冒犯性分类）、ComVE（常识分类），均遵循 FEB benchmark 的 60 个 few-shot 划分（每集 48 个训练样本 + 350 个验证样本）。
- **模型**：T5-base / T5-large / T5-3b，UNIFIEDQA，以及 Llama 2-7B。
- **评估指标**：任务准确率（Accuracy）+ 归一化 BERTScore（nBERTs，对错误预测和空解释赋零分）+ 人工评估（plausibility 评分）。
- **关键结果**：
  - **T5-large**：最优配置 LayerNorm + Attention.Q（**6.84%** 参数）在 ComVE 上准确率达 **74.86%**，nBERTs 达 **69.02**；平均性能优于其他 PEFT 方法。
  - Decoder 微调（54.60% 参数）在 e-SNLI 和 ECQA 上甚至**超过全参数微调基线**。
  - **Llama 2-7B**：SPARSEFIT（Att.Q+LN，7.97% 参数）在全部四个数据集上优于 AdaLoRA，平均准确率 **47.64%** vs 41.00%，nBERTs 显著提升。
  - **人工评估**：T5-large 上 SPARSEFIT Att.Q+LN 平均得分 **42.12**，优于全参数微调（36.91）和 AdaLoRA（35.55）；Llama 2-7B 上达 **63.33**，比全参数微调高 21%。
  - 多数 SPARSEFIT 配置与基线的性能差距不超过 **15%**。

## 相关工作脉络
- **BitFit（Zaken et al., 2022）**：仅微调 transformer 各层 bias 项即能达到与全参数微调相当的性能；本文在其基础上扩展至任意层/层组合的系统性探索，首次应用于 NLE 生成。
- **FEB benchmark（Marasovic et al., 2022）**：提出 few-shot NLE 学习基准和 prompt-based 微调策略；本文以该方法为基线，并在此基础上引入稀疏微调获得更优结果。
- **LoRA（Hu et al., 2022）**：通过注入可训练低秩矩阵适应下游任务；SPARSEFIT 不引入额外参数，直接选择已有参数子集进行微调，FLOPS 更低。
- **AdaLoRA（Zhang et al., 2023）**：根据重要性分数自适应分配低秩矩阵的 rank 预算；实验中 SPARSEFIT 在 NLE 质量上稳定优于 AdaLoRA。
- **(IA)³（Liu et al., 2022）**：通过可学习向量缩放中间激活值；本文修改其实现（对所有层而非仅 attention 模块学习缩放向量）以公平比较，SPARSEFIT 仍取得更好效果。
- **Yordanov et al. (2022)**：提出从其他域迁移 NLE 知识的 few-shot 策略；本文聚焦在目标域少量 NLE 样本下的稀疏微调而非跨域迁移。

## 局限性与未来方向
- **NLE 生成质量仍有较大提升空间**：人工评估显示约半数生成的 NLE 无法充分解释预测结果，最大短板为"解释不完整"和"无意义解释"。
- **稀疏微调与空 NLE 生成问题**：部分高效配置（如仅微调 LayerNorm）在 e-SNLI 和 ECQA 上产生空解释，归因于预训练任务中缺少 NLE 信号（T5 在 MNLI 上仅学习预测标签）。
- **解释不反映模型真实推理过程**：即使生成的解释看起来合理，模型仍可能依赖受保护属性或虚假相关性进行预测，存在潜在偏见风险。
- **未来方向**：探索更多层的组合策略、设计针对 NLE 生成的预训练目标、研究如何缓解空解释问题、将 SPARSEFIT 推广至更多架构和任务类型。

## 研究启发与可借鉴点
- **组件组合策略的可迁移性**：将不同有效稀疏配置组合可获得协同增益，这一思路可推广至其他生成任务（如摘要、对话生成）的参数高效微调。
- **LayerNorm + Self-attention Query 作为通用高效微调组合**：该组合在多个数据集和模型规模上表现稳定，可作为后续研究的强 baseline。
- **FLOPS 与参数效率的统一考量**：SPARSEFIT 同时实现低 FLOPS 和低参数量，提示在 PEFT 评估中应兼顾计算开销而不仅看参数量比例。
- **与团队方向的结合机会**：可将 SPARSEFIT 的思路应用于本团队的少样本解释生成任务，或在多模态 NLE 生成场景中探索类似的分层稀疏微调策略。
- **Prompt 与稀疏微调的结合范式**：两者的协同设计为后续研究提供了可复用的训练框架模板。

## 关键术语表
- **SPARSEFIT**：一种针对 PLM 的稀疏 few-shot 微调策略，通过选择特定层或层组合进行微调，联合生成预测标签和自然语言解释。
- **NLE（Natural Language Explanation）**：用自然语言形式对模型预测结果进行解释的自由文本，比特征级解释更具表达力且易于人类理解。
- **PEFT（Parameter-Efficient Fine-Tuning）**：参数高效微调，指仅微调少量参数或引入少量附加参数即可适配下游任务的微调技术集合。
- **FEB benchmark**：Few-shot Explanation Benchmark，由 Marasovic et al. (2022) 提出的 NLE few-shot 学习基准，包含 60 个 train-validation 划分。
- **nBERTs（normalized BERTScore）**：归一化 BERTScore，对空解释或错误预测样本赋零分的 BERTScore 变体，用于评估 NLE 质量。
- **Self-attention Query（Att.Q）**：self-attention 层中计算查询向量 Q 的投影矩阵参数，约占模型总参数的 6.82%。
- **Few-shot learning**：仅使用极少量标注样本进行模型训练的范式，本文设置为每个数据集 48 个训练样本。
- **Self-rationalization**：模型自动为其预测生成解释文本的任务设定，本文聚焦于此方向。

## 可复现要素
- **数据集**：e-SNLI、ECQA、SBIC、ComVE，遵循 FEB benchmark 的 60 个 few-shot 划分协议；数据集公开可用。
- **代码/权重**：论文未明确声明开源代码仓库；使用 HuggingFace PEFT 库实现基线。
- **关键超参**：学习率 0.00003，batch size 4，训练 25 epoch，AdamW 优化器；T5-large 约 5045 万参数，SPARSEFIT 最优配置约 6.84% 参数可训练。
- **训练硬件**：NVIDIA P100，平均训练时间约 23.2 分钟。
- **模型**：T5-base / T5-large / T5-3b、UNIFIEDQA、Llama 2-7B。
