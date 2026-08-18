---
title: "SPARSEFIT: Few-shot Prompting with Sparse Fine-tuning for Jointly Generating Predictions and Natural Language Explanations"
source: https://aclanthology.org/2024.acl-long.113.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:00:35"
field: "可解释自然语言处理"
keywords: ["稀疏微调", "自然语言解释", "少样本学习", "参数高效微调", "可解释AI"]
innovations: ["提出首个面向NLE联合生成的稀疏微调策略SPARSEFIT，系统探索62种层/层对配置", "发现仅微调6.84%参数（LayerNorm+Attention.Q）即可超越全量微调和现有PEFT方法", "在T5/UNIFIEDQA/Llama 2三种架构上验证跨架构通用性"]
benchmarks: ["e-SNLI", "ECQA", "SBIC", "ComVE"]
---

# 论文速读：SPARSEFIT: Few-shot Prompting with Sparse Fine-tuning for Jointly Generating Predictions and Natural Language Explanations

## 一句话总结
本文提出 SPARSEFIT，一种结合稀疏微调与 prompt 学习的少样本训练策略，仅需微调模型约 6.84% 的参数（LayerNorm + Self-attention Query），即可在四个 NLE 数据集上生成预测与自然语言解释，并在任务准确率和 NLE 质量上均优于完整微调和现有 PEFT 方法。

## 研究问题与动机
- **NLE 生成依赖大量人工标注数据**：生成自然语言解释（NLEs）的模型通常需要大规模人工撰写解释的训练数据，收集成本高、耗时长，对部分应用场景不可行。
- **全量微调计算代价过高**：当前 NLE 模型基于数十亿参数的 PLM（如 T5、GPT 系列），在少样本场景下全量微调成本高昂。
- **现有 PEFT 方法在 NLE 生成上的适配性未验证**：虽然 LoRA、Prefix-Tuning、BitFit 等方法在分类等下游任务中表现良好，但针对"预测+解释"联合生成的 NLE 任务，其有效性未被系统研究。
- **缺乏对 PLM 内部组件可微分性粒度的理解**：现有参数高效微调多关注 bias 或低秩矩阵，但哪些模块层/组件对 NLE 生成最关键尚无系统分析。

## 核心贡献（创新点）
- **提出首个面向 NLE 联合生成的稀疏微调策略 SPARSEFIT**：将 BitFit 仅微调 bias 的思想扩展至 PLM 各个层及层对的组合，系统探索哪些参数子集最适合少样本 NLE 学习。
- **发现非直觉的最优稀疏配置**：仅微调 LayerNorm（0.02% 参数）+ Attention.Q（6.82%）的组合（共 6.84%）在四个数据集上取得最稳定的综合最优（性能/参数量权衡分最高），超越全量微调。
- **证明组合适配的价值**：两个独立效果良好的稀疏配置联合微调，通常显著优于任一配置单独使用。
- **跨架构与跨规模通用性验证**：在 T5-base/large/3B、UNIFIEDQA 和 Llama 2-7B 三个模型架构上均验证了 SPARSEFIT 的有效性，对非 Encoder-Decoder 架构同样适用。
- **建立 NLE 生成缺陷的系统性分析**：通过人工评估量化生成 NLEs 的主要缺陷类型（缺失解释、无意义、不完整），并分析不同稀疏配置下空 NLE 生成现象。

## 方法详解
- **核心思路**：固定 PLM 大部分参数，仅解冻特定组件（层或层对）进行微调，条件生成格式为 `"[label] because [explanation]"`。
- **组件列表**：编码器块（~41% 参数）、解码器块（~54% 参数）、LM Head（~5% 参数）、自注意力层（Q/K/V 矩阵各 ~6%）、前馈网络（Dense）、Layer Normalization（~0.02%），以及所有不含编码器和解码器的组件两两配对。
- **搜索空间**：共 62 种候选配置（单组件 + 配对组合），通过 60 个少样本 train-validation split 进行稳定性评估。
- **训练超参**：AdamW 优化器，学习率固定为 0.00003，batch size=4，训练 25 轮；对比 PEFT 基线（LoRA、AdaLoRA、(IA)³）统一使用 HuggingFace PEFT 库，(IA)³ 改为在所有层学习 scaling vector 而非仅 attention 模块。
- **评估指标**：任务准确率（Acc.）+ 归一化 BERTScore（nBERTs，空 NLE 或错误预测样本得分为零）+ 人工评估（四档评分：yes / weak yes / weak no / no）。
- **权衡公式**：`(1 − %params) × nBERTs` 用于综合衡量参数效率与解释质量的帕累托前沿。

## 实验与结果
- **数据集**：e-SNLI（NLI）、ECQA（常识问答）、SBIC（冒犯性分类）、ComVE（常识判断），遵循 FEB benchmark 的 60 个 48 样本少样本划分。
- **模型规模**：T5-base / T5-large / T5-3B、UNIFIEDQA（同 T5 架构）、Llama 2-7B。
- **最强稀疏配置**：LayerNorm + Attention.Q（6.84% 参数）在 T5-large 上平均准确率 70.1%，nBERTs 63.7，在 ComVE（74.9%）和 e-SNLI（82.6%）上以显著优势超越基线。
- **vs 全量微调**：该稀疏配置在四个数据集上均达到或接近全量微调水平，最大差距 <15%。
- **vs 其他 PEFT（T5-large）**：SPARSEFIT（Att.Q+LN，6.84%）在平均 nBERTs 上显著优于 AdaLoRA（4.48%）和 LoRA（0.32%~4.86%）；在两个数据集上 nBERTs 显著更高（p<0.01）；FLOPS 最低（2.37e14），因无额外结构开销。
- **Llama 2-7B 验证**：SPARSEFIT（Att.Q+LN，7.97% 参数）全面超越 AdaLoRA（0.30%）和全量微调，nBERTs 平均高出约 5~21%。
- **人工评估**：SPARSEFIT（Att.Q+LN）在 ComVE（40.0% vs 基线 21.7%）和 SBIC（58.9% vs 基线 54.4%）上显著优于全量微调和 AdaLoRA，平均 nBERTs 提升约 8%（vs AdaLoRA）/ 6%（vs 全量微调）；Llama 2-7B 上 SPARSEFIT 比全量微调高 21%。
- **关键观察**：部分高准确率稀疏配置（如 LayerNorm）产生空 NLE（nBERTs≈0），原因系预训练目标导致模型倾向于直接输出 label 而不生成解释。

## 相关工作脉络
- **BitFit（Zaken et al., 2022）**：仅微调 bias 参数；本文将其思想从 bias 扩展到任意层/层对组合，首次系统应用于 NLE 联合生成任务。
- **FEB Benchmark（Marasovic et al., 2022）**：少样本 NLE 学习的 prompt-based 微调框架，本文以其为实验基准和主要对比基线。
- **LoRA（Hu et al., 2022）/ AdaLoRA（Zhang et al., 2023）**：注入可训练低秩矩阵；本文证明在 NLE 生成任务上，选择性地微调已有层比引入低秩适配更高效且质量更高。
- **(IA)³（Liu et al., 2022）**：通过学习 scaling vector 缩放中间激活；本文调整实现（在全层学习而非仅 attention），但仍不及 SPARSEFIT 效果。
- **NLE 生成综述（Wiegreffe & Marasovic, 2021）**；本文定位：首次将稀疏微调与 prompt 学习结合，专门针对"预测+解释"联合生成场景。
- **Prefix-Tuning（Li & Liang, 2021）**：在输入端注入可训练向量；与 SPARSEFIT 的关键差异在于 SPARSEFIT 直接修改已有权重而非引入额外向量，计算开销更低。

## 局限性与未来方向
- **预训练目标残留效应**：T5 在 MNLI 等预训练数据上仅学习 label 到 label 的映射，未学习解释生成，导致部分稀疏配置（如 LayerNorm）倾向于跳过 explanation token 直接输出 label，产生空 NLE。
- **少量数据下 NLE 质量不稳定**：48 个样本的少样本设置下，约一半 NLE 无法充分论证预测（人工评估），解释质量仍有限。
- **未探索更多层组合**：仅研究了单组件和两两配对，未涉及三元以上的高阶组合或逐层重要性排序搜索。
- **未讨论与 Prompting / ICL 方法的直接对比**：仅以 FEB 的 prompt-based 微调为基线，未与纯 in-context learning 范式对比。
- **潜在偏见风险**：自解释模型可能学习训练数据中的有害偏见，但未对此进行深入分析。

## 研究启发与可借鉴点
- **组件级稀疏微调的系统探索框架**：将 62 种层/层对组合的实验范式可迁移至其他生成任务（如摘要、对话），作为参数效率与性能权衡的探索方法。
- **"LayerNorm + Attention.Q"这一超稀疏配置的发现**提示：Normalization 层对训练动态极为敏感，与其结合的注意力查询投影可能是少样本适应的关键枢纽，值得在其他 PEFT 研究中验证。
- **空 NLE 诊断启发**：通过归一化 BERTScore（对空 NLE 赋零）有效捕捉生成空洞问题，该方法可直接复用于其他文本生成任务的 NLE 质量评估。
- **跨架构验证策略**：在同一方法上同时测试 T5（Encoder-Decoder）和 Llama 2（Decoder-only）两种不同架构，增强了结论的外部效度，可作为方法论文的验证模板。
- **FLOPS 一致性分析**：不仅报告参数量，还报告实际 FLOPS，证明 SPARSEFIT 不增加架构复杂度，这一评估维度对资源受限部署场景极具参考价值。

## 关键术语表
**NLE（Natural Language Explanation）**：模型用自然语言自由文本形式解释其预测理由的输出，易于人类理解且表达能力强于特征级解释。
**SPARSEFIT**：本文提出的稀疏少样本微调策略，通过选择性地微调 PLM 中特定层或层对（而非全量或 bias-only）来学习预测+解释的联合生成。
**FEB Benchmark**：Marasovic 等（2022）提出的少样本 NLE 学习基准，包含 60 个 48 样本的 train-validation split 及 prompt-based 微调协议。
**nBERTs（Normalized BERTScore）**：BERTScore 的变体，对空 NLE 或预测错误的样本赋零分，更严格地反映 NLE 生成质量。
**PEFT（Parameter-Efficient Fine-Tuning）**：参数高效微调的统称，包括 LoRA、Prefix-Tuning、BitFit、(IA)³ 等方法，旨在用极少量可训练参数适配下游任务。
**Self-rationalization**：模型自我合理化任务，指模型在做出预测的同时生成支持该预测的自然语言解释。

## 可复现要素
- **数据集**：e-SNLI、ECQA、SBIC、ComVE，均遵循 FEB benchmark（Marasovic et al., 2022）的 60 个 48 样本少样本划分，数据集本身公开可获取。
- **代码**：论文未提供开源代码链接，但明确使用 HuggingFace PEFT 库，训练脚本基于 Marasovic et al.（2022）的开源代码改编。
- **权重**：使用官方预训练的 T5-base / T5-large / T5-3B 和 UNIFIEDQA、Llama 2-7B 权重，均公开可下载。
- **关键超参**：AdamW 优化器，学习率 0.00003，batch size=4，25 轮训练；(IA)³ 实现在全层学习 scaling vector。
- **硬件**：NVIDIA P100，平均训练时间约 23.2 分钟（T5-large 设置）。
