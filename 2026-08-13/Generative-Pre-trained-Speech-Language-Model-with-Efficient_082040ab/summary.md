---
title: "Generative-Pre-trained-Speech-Language-Model-with-Efficient"
source: https://aclanthology.org/2024.acl-long.97.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:48:15"
---

# 论文速读：Generative-Pre-trained-Speech-Language-Model-with-Efficient

## 一句话总结
提出 GPST，一种基于层级 Transformer 的单阶段生成式语音语言模型，将语义 Token 与声学 Token 统一建模，有效规避长序列二次复杂度与多阶段误差传播，同时支持高保真（Hi-Res）语音合成、说话人身份迁移及零样本跨语言生成。

## 研究问题与动机
- 神经音频编解码器（如 EnCodec）生成的声学序列极长（如 10 秒音频至少 6000 个 code），标准 Transformer 自注意力复杂度随序列长度二次增长，难以直接建模。
- 现有主流方法（AudioLM、VALL-E 等）被迫采用多阶段生成框架分段处理声学序列，导致训练阶段割裂与误差累积传播。
- 声学 Token 天然具有 RVQ 层级结构（前期层保留说话人/韵律，后期层捕获细粒度细节），现有模型未充分复用该层级先验，计算冗余高。
- 多语言语音生成与 Hi-Res 高比特率合成目前缺乏统一高效的单阶段解决方案，各自为战且算力门槛高。

## 核心贡献（创新点）
- **单阶段层级 Transformer 架构**：提出 Global/Local 两级 Transformer 统一建模语义与声学 Token，一次前向完成自回归生成。与 AudioLM/VALL-E 等多阶段框架的本质区别在于消除分阶段误差传播，并在单模型内同步优化长程语义一致性与残差局部依赖。
- **Local-drop 训练技巧**：面向 Hi-Res 生成提出随机丢弃部分声学 Token 栈的加速正则化方法。与直接堆叠量化层导致算力爆炸的做法不同，该方法在控制局部 Transformer 计算负载的同时保持高保真生成能力。
- **多语言与 Hi-Res 统一支持**：首次在同一框架内实现口语化多语言生成与 16 层量化 Hi-Res 合成。相比仅针对单语或低比特率设计的基线，GPST 利用多语言 SSL 模型与通用声码器即可实现跨语言零样本迁移，无需额外多语言声学训练。

## 方法详解
- **语音离散化**：波形被量化为语义 Token 序列 $S$（经 SSL 模型 + K-means 提取）与声学 Token 二维矩阵 $A \in \{1,\dots,N_a\}^{T_2 \times D}$，其中 $a_t^q$ 为第 $q$ 层残差码。
- **生成目标**：完全因子化为单阶段自回归形式 $p(S,A) = \prod_t p(s_t|s_{<t}) \prod_{q,t} p(a_t^q | a_{<t}^{\le D}, a_t^{<q}, S)$。
- **Global Transformer**：$N_g$ 层 Decoder-only 结构，因果掩码。输入为语义 Token 序列与每时间步 $D$ 层声学 Embedding 求和后的序列拼接，负责建模 $p(S)$ 及语义对声学的长程条件。
- **Local Transformer**：$N_l$ 层较小结构，接收 Global 输出的隐状态 $h_t$，按自回归顺序预测当前位置的 $D$ 个声学码，Embedding 加入层级位置编码 $\mathrm{PE}_l(q)$。
- **Local-drop**：将声学序列长度维 $T_2$ 展平至 Batch 维，使局部自注意力仅在 $D$ 维上运行；以概率 0.5 随机丢弃部分时间步的声学栈，显著降低 Hi-Res 训练 FLOPs。
- **推理模式**：无条件生成、语义→声学（TTS）、说话人身份迁移（拼接 $[S_p, S_t, A_p]$，边界插入 0.1s 静音防语言不连续导致的生成震荡）、声学续写（前 3 秒 prompt 续写）。
- **多语言扩展**：语义侧采用 SeamlessM4T 的 XLSR 编码器构建多语言词汇表；声学侧沿用预训练 EnCodec 作为通用提取器，实现“一次预训练、多语推理”。

## 实验与结果
- **数据集**：训练 Libri-Light（60K小时英文，10s 裁剪）；评测 LibriSpeech test-clean（4-10s）；多语言任务使用 LibriSpeech 960h（英）与 Aishell-2 1000h（中）。
- **评估基线**：GSLM、AudioLM、VALL-E、YourTTS、SPEAR-TTS。
- **核心指标**：WER（HuBERT-Large）、SPK（WavLM-TDNN）、DNSMOS。
- **主要结果**：
  - 语义→声学：GPST WER 4.0，仅用 AudioLM 33% 参数（190M vs 600M+），显著优于 GSLM (12.4) 与 AudioLM (6.0)。
  - 说话人身份迁移：GPST 取得最优 WER 4.2、SPK 0.605、DNSMOS 3.89，全面超越 VALL-E (SPK 0.580, DNSMOS 3.87) 与 SPEAR-TTS。
  - 声学续写：GPST WER 2.8、SPK 0.536，优于 VALL-E (WER 3.8, SPK 0.508)。
  - Hi-Res（16 quantizers）：GPST-Hi-Res DNSMOS 达 4.02；多语言零样本跨语种迁移（仅英文训练，中文推理）CER 33.3，接近纯中文训练效果（CER 30.2）。
- **最强结果与提升**：说话人身份迁移任务取得 SOTA（WER 4.2 / SPK 0.605 / DNSMOS 3.89），参数仅为同量级基线的约 1/3；多语言与 Hi-Res 能力为本文首次统一支持。

## 相关工作脉络
- **GSLM**：仅基于语义 Token 的自回归语音模型，缺乏声学细节。GPST 在其基础上引入声学 Token 与层级结构，补足高保真合成与说话人保持能力。
- **AudioLM**：三阶段框架，因序列过长被迫拆分粗/细声学建模。GPST 用单阶段层级 Transformer 替代，消除分阶段误差传播并降低复杂度。
- **VALL-E**：两阶段音素→声学模型，依赖 G2P 且无法无条件生成语义连贯语音。GPST 直接建模 $p(S)$，支持零样本语音续写与身份迁移，架构更统一。
- **SPEAR-TTS**：在 AudioLM 上扩展 TTS。GPST 表明在相同评测设定下，单阶段模型能以更少参数取得更优 WER 与 SPK。
- **SoundStorm / SpeechGPT / PolyVoice**：多采用重复语义 Token 或独立指令调优。GPST 明确指出重复 Token 会导致生成失败并限制应用，凸显本方法在纯净语义建模上的优势。

## 局限性与未来方向
- **文本输入受限**：GPST 无法直接将文本合成为语音，需额外训练 GPST-TTS 模块，限制了端到端文本驱动场景。
- **Hi-Res 效率与质量权衡**：16 层量化虽提升
