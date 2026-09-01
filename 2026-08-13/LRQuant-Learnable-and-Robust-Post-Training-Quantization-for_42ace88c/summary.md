---
title: "LRQuant-Learnable-and-Robust-Post-Training-Quantization-for"
source: https://aclanthology.org/2024.acl-long.122.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 07:12:47"
field: "大语言模型量化压缩"
keywords: ["post-training quantization", "LLM compression", "smooth quantization", "test-time adaptation", "learnable parameters"]
innovations: ["提出LA E初始化的可学习平滑参数框架替代hand-crafted缩放因子", "设计基于余弦相似度负对数的NLC损失以同时优化幅度与方向", "首次将TTA引入LLM量化并仅适配最后一层以避免灾难性遗忘"]
benchmarks: ["WikiText2", "PTB", "C4", "PIQA", "ARC-e/c", "HellaSwag", "Winogrande"]
---

# 论文速读：LRQuant-Learnable-and-Robust-Post-Training-Quantization-for

## 一句话总结
本文提出 **LRQuant**，一种可学习且鲁棒的基于平滑的PTQ框架，通过引入对数激活等价初始化的可学习平滑参数、设计基于余弦相似度的 NLC 损失，并首次将 TTA 引入 LLM 量化，在 W4A4/W6A6 挑战设置下显著超越现有方法，同时提升了模型在未见数据集上的泛化能力。

## 研究问题与动机
1. **平滑参数预定义导致次优**：现有 smooth-based PTQ 方法（如 SmoothQuant、LAE）依赖 hand-crafted 缩放因子或均匀 zero-point，无法自适应优化，结果次优。
2. **仅用 MSE 损失无法充分优化**：MSE 仅衡量输出向量的幅度相似度，忽略了方向差异；作者实验发现即便是可学习的 OmniQuant，量化输出与全精度输出的余弦相似度仍存在较大差距。
3. **跨数据集泛化能力差**：在校准集（如 WikiText2）上训练的量化模型，在未见测试集（如 PTB）上性能显著下降。
4. **直接整模型 TTA 导致灾难性遗忘**：若直接利用测试数据适配整个量化模型，会破坏在源校准数据上学到的参数，造成性能退化。

## 核心贡献（创新点）
1. **可学习平滑参数框架**：首次将平滑参数定义为可学习变量，并采用对数激活等价（LAE）方式进行初始化，相比 OmniQuant 的 channel-wise 最大值初始化更利于抑制激活异常值。
2. **NLC 损失设计**：提出基于全精度块输出与量化块输出之间余弦相似度负对数的块级损失（NLC loss），与 MSE 互补，同时优化输出向量的幅度和方向。
3. **首次引入 TTA 到 LLM 量化**：仅对最后一个 block 的可学习参数在测试集上进行 5 个 epoch 的轻量适配，在提升目标集性能的同时有效避免了对源校准知识的灾难性遗忘，且比直接重新校准速度快数百倍。

## 方法详解
**Learnable Parameters**
- 权重量化参数 α、β（控制 clipping 范围上下界，均在 [0,1] 区间可学习）。
- 平滑缩放因子 s 通过 LAE 初始化：$s_i^0 = \max(|x_i|) / \log_a(a + \max(|x_i|))$，其中 $a=2$。
- 平滑移位因子 $z_s$ 按 Outlier Suppression+ 初始化。

**Loss Function**
- 总损失为 MSE 损失与 NLC 损失的等权组合：
  $\mathcal{L} = \mathcal{L}_{MSE} + \mathcal{L}_{NLC}$
- 其中 $\mathcal{L}_{NLC} = -\log\left(\frac{F(W,X) \cdot F(Q_w(W), Q_a(X))}{\|F(W,X)\| \|F(Q_w(W), Q_a(X))\|}\right)$。

**TTA 策略**
- 在校准阶段结束后，仅提取最后一个 block 的全精度版本，用测试集数据对其可学习参数进行 5 个 epoch 的适配（同样使用 MSE+NLC 损失），其余 block 参数保持不变。
- 每个适配过程耗时约 1 分钟以内，且不依赖源域数据（source-free）。

## 实验与结果
- **数据集**：WikiText2（校准集）、PTB、C4（测试集）；零样本任务包括 PIQA、ARC-e/c、BoolQ、HellaSwag、Winogrande。
- **模型**：LLaMA (7B/13B/30B)、LLaMA-2 (7B/13B)、OPT (1.3B/2.7B/6.7B)。
- **W4A4 困惑度**：在 LLaMA-7B 上，相比 OmniQuant，WikiText2 上 11.25 vs 12.46，PTB 上 52.05 vs 107.31（降低约 51.5%），C4 上 14.14 vs 15.83；LLaMA-13B/30B 同样全面领先。
- **W6A6 困惑度**：LRQuant 在大部分设置下达到最优或接近最优。
- **零样本准确率**：平均提升 3.22%–8.95%，LLaMA-13B 在 PIQA 上达到 72.41（FP16 为 78.78）。
- **TTA 效果**：在 PTB 上 post-TTA 相比 pre-TTA 困惑度从 52.05 降至 42.76（LLaMA-13B）；直接整模型重校准耗时 300+ 分钟，TTA 仅需不到 1 分钟，且在原始 WikiText2 上不退化（避免灾难性遗忘）。

## 相关工作脉络
1. **SmoothQuant (Xiao et al., 2023)**：手工定义 channel-wise 缩放因子的开山之作；LRQuant 将其推广为可学习参数。
2. **LAE (Li et al., 2023)**：非线性对数激活等价方法计算缩放因子；LRQuant 借鉴其初始化策略并进一步可学习化。
3. **OmniQuant (Shao et al., 2024)**：首个引入可学习平滑参数的方法，但仅用 MSE 损失且以 channel-wise 最大值为初始化；LRQuant 在此基础上提出 LAE 初始化与 NLC 损失，显著缩小余弦相似度差距。
4. **Tent (Wang et al., 2020)** / **EATA (Niu et al., 2022)** / **SAR (Niu et al., 2023)**：经典 TTA 方法通过熵最小化或样本选择适配测试分布；LRQuant 首次将 TTA 思想引入量化领域，并针对量化场景设计了仅适配最后一层的轻量化方案。
5. **AWQ (Lin et al., 2023)** / **GPTQ (Frantar et al., 2022)**：weight-only 量化代表；LRQuant 虽聚焦 weight-activation 量化，但在 Appendix G 中证明了其在 weight-only 任务上的优势（W2A16 下 GPTQ/AWQ 崩溃时 LRQuant 仍保持可用性能）。

## 局限性与未来方向
1. **大模型扩展受限**：受硬件限制，未在超过 100B 参数的模型上验证方法效果。
2. **TTA 设计未借鉴 SOTA**：当前 TTA 方案较朴素，在部分数据集（如 C4）上效果不如原始校准结果，作者计划探索融合更多先进 TTA 思想以提升适配效果。
3. **TTA 仅适配最后一层**：虽然有效避免了灾难性遗忘，但可能未充分利用其他 block 的适配潜力。

## 研究启发与可借鉴点
1. **NLC 损失的设计范式**：将余弦相似度负对数作为辅助损失，可与任何度量输出幅度差异的主损失（MSE/MAE）搭配使用，适用于各类量化、压缩或蒸馏任务中方向信息的保留。
2. **LAE 初始化策略的可迁移性**：基于对数激活等价的缩放初始化比单纯的最大值初始化更合理，可复用到其他可学习平滑或归一化参数框架中。
3. **"最后一层适配"的 TTA 设计**：仅适配最靠近输出层的模块即可在保持源域性能的同时提升目标域表现，这一轻量化思路可推广到 LoRA 微调、 Adapter 等参数高效适配场景中。
4. **实验设计的对照完整性**：论文不仅对比了 perplexity 和零样本准确率，还系统验证了 TTA 对比重校准的速度优势（Table 7）、灾难性遗忘缓解（Table 8/14），这种多维度的 ablation 值得借鉴。

## 关键术语表
- **Post-Training Quantization (PTQ)**：训练后量化，无需重新训练模型即可将权重和激活从全精度压缩到低位宽。
- **Smooth Quantization**：通过数学等价变换将激活分布的量化难度转移到权重上，从而降低低比特量化的精度损失。
- **NLC Loss (Negative Logarithm of Cosine Similarity Loss)**：基于全精度与量化输出块余弦相似度的负对数损失，用于优化输出向量的方向一致性。
- **Logarithmic Activation Equalization (LAE)**：利用对数函数对激活值的最大值进行非线性缩放以初始化平滑参数，更有效地抑制激活异常值。
- **Test-Time Adaptation (TTA)**：在测试阶段仅利用测试数据无标签地微调模型参数，以提升模型在新域上的泛化能力，无需重新训练。
- **Catastrophic Forgetting**：模型在新数据上持续学习时，对原有训练数据的知识产生严重遗忘的现象。

## 可复现要素
- **代码**：已开源，https://github.com/zjq0455/RLQ
- **校准数据集**：WikiText2（128 条随机 2048 token 序列）
- **测试数据集**：WikiText2、PTB、C4
- **关键超参**：平滑参数学习率 1e-3，量化参数学习率 1e-2，AdamW 优化器，零 weight decay；校准训练 20 个 epoch，TTA 适配 5 个 epoch，batch size=1；LAE 初始化底数 a=2。
- **硬件环境**：NVIDIA A100-40G GPU
