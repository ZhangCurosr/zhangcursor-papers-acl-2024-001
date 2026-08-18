---
title: "SPARSEFIT: Few-shot Prompting with Sparse Fine-tuning for Jointly Generating Predictions and Natural Language Explanations"
source: https://aclanthology.org/2024.acl-long.113.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:00:44"
field: "可解释自然语言处理"
keywords: ["few-shot learning", "natural language explanation", "sparse fine-tuning", "parameter-efficient fine-tuning", "explainable AI"]
innovations: ["提出仅微调6.8%参数的SPARSEFIT策略实现预测与NLE联合生成", "发现LayerNorm+Attention.Query组合在预测准确率与解释质量间最优权衡", "首次在T5与Llama 2-7B上验证稀疏微调跨架构有效性"]
benchmarks: ["e-SNLI", "ECQA", "SBIC", "ComVE", "FEB benchmark"]
---

# 论文速读：SPARSEFIT: Few-shot Prompting with Sparse Fine-tuning for Jointly Generating Predictions and Natural Language Explanations

## 一句话总结
SPARSEFIT 是一种结合稀疏微调与 prompt-based learning 的高效 few-shot 训练策略，仅需微调约 6.8% 的模型参数（如 Layer Normalization + Self-attention Query），即可在四个 NLE 数据集上实现与全量微调相当且优于其他 PEFT 方法的预测准确性与自然语言解释质量。

## 研究问题与动机
- **核心问题**：当前能够生成自然语言解释（NLEs）的模型通常依赖大量人工撰写解释的高质量训练数据，但在 few-shot 场景下（仅有少量标注样本），全量微调千亿参数 PLM 计算成本过高，而其他 PEFT 方法在 NLE 生成质量上仍有不足。
- **现有 PEFT 方法的局限**：LoRA、Prefix-Tuning、BitFit 等方法虽能减少参数量，但在同时生成预测与自然语言解释的任务中，解释质量（尤其是 "because" 之后的内容）常常退化为空文本或重复输入。
- **未探索方向**：sparse fine-tuning 在 NLP 任务中已有应用（如 BitFit 仅微调 bias），但**首次**在联合生成预测与 NLE 的 few-shot 场景中进行系统性研究。

## 核心贡献（创新点）
- **提出 SPARSEFIT 策略**：通过筛选 T5 架构中关键组件（LayerNorm、Self-attention Query/K/V、LM Head 及其组合）进行稀疏微调，在 6.84% 参数占比下实现最优权衡；与 BitFit 仅调 bias 的本质区别在于 SPARSEFIT 覆盖更广泛的架构层，兼顾任务性能与解释生成能力。
- **首次系统分析稀疏微调对 NLE 生成的影响**：发现微调 "Att.Q + LN" 组合可显著提升解释质量，而单独微调 LayerNorm 虽能达到高准确率但常生成空解释——揭示了预测准确与解释质量之间的解耦现象。
- **扩展至更大模型与不同架构**：在 T5-3B 和 Llama 2-7B 上验证 SPARSEFIT 的通用性，证明该方法不限于特定模型结构，且在 Llama 2-7B 上平均 NLE 质量比 AdaLoRA 高约 6%、比全量微调高约 21%。

## 方法详解
- **训练范式**：采用条件文本生成格式 "[label] because [explanation]"，将 NLE 生成建模为序列到序列任务（延续 FEB benchmark 协议）。
- **参数选择空间**：在 T5 架构中探索 62 种稀疏配置（单组件 + 组件对），包括 Encoder blocks（41%）、Decoder blocks（54%）、LM Head（5%）、Self-attention Q/K/V（各约 6%）、Layer Normalization（0.02%）及全 Attention（QKV，20.47%）。
- **最优配置**：LayerNorm + Attention.Q（6.84% 参数）在所有 T5 尺寸和四个数据集上均取得最佳性价比；组合策略（如 Att.Q + LM head 或 Att.Q + LN）优于单一组件。
- **训练细节**：每组件微调 25 epochs，batch size = 4，AdamW optimizer，固定 learning rate = 3e-5；与 PEFT 基线（LoRA、AdaLoRA、(IA)³）使用相同实验设置以公平比较。

## 实验与结果
- **数据集**：e-SNLI（NLI）、ECQA（常识 QA）、SBIC（冒犯性分类）、ComVE（常识分类），各数据集仅用 48 个训练样本（每类均匀采样），60 次划分取均值。
- **评估指标**：任务准确率（Accuracy）与归一化 BERTScore（nBERTs，空解释或错误预测得 0 分）；辅以人工评估（"解释是否合理化答案？" 四档评分）。
- **主要结果（T5-large）**：
  - SPARSEFIT（Att.Q+LN, 6.84%）平均准确率 70.07%，nBERTs 63.7；e-SNLI 达 82.62%，显著优于 baseline（70.1 ± 3.4 nBERTs）。
  - 相较其他 PEFT：LoRA (4.86%) 平均准确率 60.41、nBERTs 64.78；AdaLoRA (1.15%) 平均 65.64 nBERTs；**(IA)³ (0.07%)** 平均 63.10 nBERTs；SPARSEFIT 在 2/4 数据集上显著超越，且 FLOPS 最低（2.37e14）。
  - 全量微调（Baseline）平均 73.3，SPARSEFIT 差距 <5%，但参数减少 >90%。
- **Llama 2-7B 结果**：SPARSEFIT（Att.Q+LN, 7.97%）平均准确率 47.64，显著优于 AdaLoRA（41.00）；nBERTs 43.18 vs 37.85，平均高出约 5% NLE 质量。
- **人类评估**：SPARSEFIT（Att.Q+LN）平均得分 42.12，高于 Full Fine-tuning（36.91）和 AdaLoRA（35.55）；Llama 2-7B 上 SPARSEFIT 平均 63.33 vs AdaLoRA 57.22 vs Full 36.11。

## 相关工作脉络
- **FEB benchmark (Marasović et al., 2022)**：本文主要对比基线，提出 few-shot 提示式微调学习 NLEs，但依赖全量微调；SPARSEFIT 在相同协议下以更少参数实现同等/更优效果。
- **BitFit (Zaken et al., 2022)**：仅微调 bias 参数，成本低但解释生成能力受限；SPARSEFIT 扩展至更丰富的组件组合以兼顾预测与解释质量。
- **LoRA / AdaLoRA (Hu et al., 2022; Zhang et al., 2023)**：通过低秩矩阵注入实现参数高效微调；但 SPARSEFIT 不引入额外参数、不改变架构复杂度，FLOPS 更低且 NLE 质量更优。
- **(IA)³ (Liu et al., 2022)**：对中间激活缩放；本团队调整使其作用于所有层（而非仅 attention）后获得更好性能，但仍不及 SPARSEFIT。
- **自理性（Self-rationalization）早期工作 (Camburu et al., 2018; Kayser et al., 2021)**：建立 e-SNLI、e-ViL 等数据集与 nBERTs 评估范式；SPARSEFIT 在此框架下推进 sparse fine-tuning 的应用边界。

## 局限性与未来方向
- **解释真实性存疑**：生成的 NLE 可能无法反映模型内部真实推理过程，模型仍可能依赖训练数据中的敏感属性或虚假关联，作者建议在生产部署中谨慎使用。
- **空解释问题**：部分稀疏配置（如 LayerNorm 仅 0.02%）虽准确率尚可，但常提前终止生成（仅输出 label 或 "because"），导致 nBERTs 为零——需进一步研究如何让稀疏微调避免退化到预训练行为。
- **预训练任务不匹配**：T5 预训练数据（如 MNLI）无 NLE，UNIFIEDQA 预训练数据（如 CommonsenseQA）仅含答案，导致 few-shot 微调难以引导模型从"仅预测"转向"预测+解释"。
- **未来方向**：探索更智能的组件选择策略、设计避免空生成的约束机制、将 SPARSEFIT 推广至视觉-语言多模态 NLE 生成场景。

## 研究启发与可借鉴点
- **稀疏组件组合策略**：SPARSEFIT 证明"Att.Q + LN"类组合比单一组件更稳定，提示后续研究可优先探索"归一化层 + 关键投影层"的组合而非随机子集。
- **精度-解释解耦现象的警示**：高准确率 ≠ 高质量解释，评估 NLE 系统时必须同时报告 nBERTs 或人类评估，避免被 accuracy 误导。
- **FLOPS 作为公平比较维度**：SPARSEFIT 在最少参数下实现最低计算开销，建议未来 PEFT 对比实验补充 FLOPS/显存占用指标。
- **跨架构泛化验证**：本文在 T5 encoder-decoder 与 Llama 2 decoder-only 上均有效，说明稀疏微调理念具有架构无关性，可扩展至 Mamba、Hyena 等新架构。
- **few-shot 数据集划分协议复用**：直接采用 FEB 的 60 次 train-validation split（各 48 样本）确保结果可比性，适合构建可复现基准。

## 关键术语表
- **NLE (Natural Language Explanation)**：模型用自然语言对其预测进行自解释的自由文本，比特征级解释更具表达力。
- **SPARSEFIT**：本文提出的稀疏 few-shot 微调策略，通过选择性更新 PLM 特定组件参数以联合生成预测与 NLE。
- **nBERTs (normalized BERTScore)**：改进版 BERTScore，对空解释或错误预测样本赋 0 分，更严格评估 NLE 质量。
- **PEFT (Parameter-Efficient Fine-Tuning)**：仅微调少量参数或添加小模块以保持下游性能的微调范式，如 LoRA、BitFit、Prefix-Tuning。
- **FEB (Few-shot Explainability Benchmark)**：Marasović et al. (2022) 提出的 few-shot NLE 基准，包含 60 次数据划分与标准协议。
- **Self-rationalization**：要求模型不仅做出预测，还需生成理由支持该预测的任务设定。

## 可复现要素
- **数据集**：e-SNLI、ECQA、SBIC、ComVE，均公开可用；FEB benchmark 的 60 次 few-shot split 已发布（需从原论文获取或自行划分）。
- **代码**：论文未明确开源仓库链接，但使用 HuggingFace `peft` 库实现 LoRA/AdaLoRA/(IA)³；实验代码需在论文发表后确认。
- **关键超参**：epochs=25，batch_size=4，learning_rate=3e-5，optimizer=AdamW，共 48 个训练样本/分割。
- **硬件**：NVIDIA P100，平均训练时间 23.2 分钟。
- **模型**：T5-base / T5-large / T5-3B、UNIFIEDQA（同架构）、Llama 2-7B。
