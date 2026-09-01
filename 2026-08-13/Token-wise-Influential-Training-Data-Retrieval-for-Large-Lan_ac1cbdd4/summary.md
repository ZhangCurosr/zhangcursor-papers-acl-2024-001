---
title: "Token-wise-Influential-Training-Data-Retrieval-for-Large-Lan"
source: https://aclanthology.org/2024.acl-long.48.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 21:44:23"
---

# 论文速读：Token-wise-Influential-Training-Data-Retrieval-for-Large-Lan

## 一句话总结
本文提出 RapidIn，一种面向大语言模型（LLM）的可扩展训练数据影响估计框架，通过层归一化、随机洗牌与随机投影将 GB 级梯度压缩超 20 万倍并缓存，实现了对任意给定生成在分钟级完成全量数据的影响检索，并将归因粒度从样本级扩展至 Token 级。

## 研究问题与动机
- **核心问题**：给定 LLM 的某次生成，如何高效、准确地追溯并量化其由哪些训练数据驱动，从而支撑安全溯源、机器遗忘与错误诊断？
- **现有方法不足**：
  1. **计算与存储不可扩展**：Influence Function 依赖 Hessian-vector 乘积，对 LLM 代价极高；TracIn 虽仅需一阶梯度，但 llama-2 7b/70b 全精度梯度分别达 26GB/260GB，无法存储与遍历。
  2. **尺度瓶颈**：LLM 参数达数百亿，训练语料达万亿 Token 级，传统深度学习方法缺乏应对该规模的设计。
  3. **粒度粗糙**：现有工作多针对分类任务做样本级评分，而 LLM 单条数据含大量 Token，无法直接映射到 Token 级别的影响力分配。
  4. **应用落地受限**：危险生成定位、数据清洗、解毒、隐私安全等场景急需快速可复用工具，但当前技术因 OOM 或耗时过长无法支撑。

## 核心贡献（创新点）
- **提出 RapidIn 两阶段框架**：首次将梯度影响估计扩展至千亿参数级 LLM 与万亿 Token 级数据集，解耦离线缓存与在线检索。
- **超 20 万倍梯度压缩**：结合逐层 $L^2$ 归一化、无全排列随机洗牌与 Rademacher 随机投影，将完整梯度压缩至 KB/MB 级 RapidGrad，10TB 硬盘可存 8000 万+ 样本。
- **分钟级检索与 6,326x 加速**：检索仅需低维向量内积，单代生成影响估计约数秒至数分钟；100 代生成总耗时 7.97h，较 TracIn 的 1,202h 实现超 6,326 倍提速。
- **Token 级细粒度归因公式**：推导样本↔句子、Token↔句子、Token↔Token 四种影响计算公式（Eq. 6-8），支持精准的可视化归因与错误溯源。
- **开源实现与多 GPU 并行**：提供 PyTorch 开源代码，缓存与检索阶段均支持多进程/GPU 并行，显著提升大规模实验效率。

## 方法详解
- **理论基础**：基于 TracIn 的一阶梯度近似，忽略学习率主导的高阶项后，训练数据 $s_k$ 对测试生成 $t$ 的影响可近似为两者梯度内积的累加（Eq. 5）。进一步分解可得四种细粒度形式（Eq. 6-8），支持跨样本/Token 的交叉影响计算。
- **缓存阶段（Caching）**：
  1. **逐层 $L^2$ 归一化**：解决深层网络中浅层梯度数值爆炸导致的脆弱性问题，使各层梯度范数统一为 1，提升内积稳定性。
  2. **随机洗牌（Algorithm 1）**：针对 B 级维度梯度，反复将向量 reshape 为矩阵并按随机因子对行列洗牌，打破梯度结构相关性，避免存储完整置换向量。
  3. **随机投影压缩**：生成 Rademacher 分布随机向量 $\rho \in \{-1, 1\}$ 与梯度逐元素相乘，再将每 $|v|/K$ 个元素求和，得到长度为 $K$ 的 RapidGrad。$K=2^{16}$ 时单样本仅 125KB。
- **检索阶段（Retrieval）**：对测试生成 $t$ 执行相同归一化与压缩流程得到 RapidGrad，随后与缓存库中所有训练数据的 RapidGrad 计算内积即得影响力得分。检索可按样本并行分发至多 GPU，结果汇总至 CPU。
- **多 GPU 并行流水线**：缓存阶段数据独立，检索阶段查询独立，通过 CPU 共享内存分配任务，实现线性加速（Figure 3）。

## 实验与结果
- **数据集与模型**：主实验使用 Alpaca 52K；附加验证使用 MedInstruct-52K 与 JudgeLM-100K。模型覆盖 llama-2 7b (QLoRA)、llama-2 70b (QLoRA) 与 llama-2 7b (全参数微调)。
- **基线方法**：Random Selection、Embedding Similarity (OpenAI text-embedding-ada-002)、BM25、Influence Function、TracIn 及加层归一化的 TracIn+LN。
- **评估任务与指标**：
  1. **后门攻击验证**：注入 5,000 条含触发词 `Howdy!` 的科幻风格 poisoned 数据，评估能否在攻击成功生成中召回 poisoned 样本（auPRC / auROC）。
  2. **错误溯源（Error Tracing）**：以 80% 概率替换实体（如 China→Canada），评估能否将错误生成归因于被扰动样本（平均比例 AP）。
- **主要结果**：
  - **压缩与存储**：RapidGrad ($K=2^{16}$) 仅 125KB，相比全参数 7b 梯度压缩 210,534 倍；$K$ 增大时性能稳定且仍远小于原始梯度。
  - **速度**：llama-2 70b QLoRA 缓存 16.01h，检索单代仅 0.027h（~97秒）；100 代总耗时 7.97h，TracIn 需 1,202h，加速 6,326 倍。Influence Function 在 70b 及全参 7b 上均 OOM。
  - **准确性**：QLoRA 模型上 RapidIn ($K=2^{20/24}$) auPRC/auROC 接近 1.0，全面超越 TracIn 与语义/检索基线；全参 7b 上 RapidIn 稳定有效（Top-5 auPRC 0.92~0.96），基线因 OOM 无法运行。
  - **Token 级优势**：Token-wise 检索在 Error Tracing 中显著优于样本级检索，能更精准定位携带错误信息的特定 Token。
- **最强结果**：llama-2 70b QLoRA + RapidIn ($K=2^{24}$) 在后
