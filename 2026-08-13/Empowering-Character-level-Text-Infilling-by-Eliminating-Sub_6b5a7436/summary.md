---
title: "Empowering-Character-level-Text-Infilling-by-Eliminating-Sub"
source: https://aclanthology.org/2024.acl-long.179.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 07:06:08"
field: "代码语言模型预训练"
keywords: ["Fill-in-the-Middle", "code completion", "sub-token", "character-level infilling", "decoder-only model", "pre-training"]
innovations: ["提出FIM-SE方法，通过L-Prefix/F-Suffix约束消除推理阶段的子token预测", "将字符级随机跨度填充转化为多行级填充，统一不同粒度任务的训练范式"]
benchmarks: ["Humaneval random-span", "Humaneval single-line", "Humaneval multi-line", "MBPP"]
---

# 论文速读：Empowering-Character-level-Text-Infilling-by-Eliminating-Sub

## 一句话总结
论文提出 FIM-SE（Fill-In-the-Middle with Starting and Ending character constraints），通过引入起始/结束字符约束将字符级随机跨度填充转化为多行级填充，消除推理阶段的子token预测，从而显著提升代码补全性能（Code Llama 13B 在 Humaneval random-span 提升 8.8%）。

## 研究问题与动机
1. **子token训练不一致性**：传统 FIM 方法在字符级随机分割时，同一个 prefix token（如 [29909]）在不同样本中可能对应不同的 objective（如 285 vs 5713），导致模型在子token上产生高困惑度。
2. **子token预测误差放大**：模型在推理阶段需预测子token时，其概率分布趋于均匀（理论证明收敛至 1/m），首次预测错误的风险显著升高，在代码补全等敏感任务中尤为致命。
3. **完整token与上下文兼容的矛盾**：完全避免子token会导致输出与上下文不匹配；但若保留子token则因困惑度高而产生错误——本文需同时解决这两个冲突场景。

## 核心贡献（创新点）
1. **提出 FIM-SE 训练范式**：通过引入 L-Prefix（prefix 最后一行）和 F-Suffix（suffix 第一行）两个特殊token，确保 `<MID>` 后预测的都是完整token，从根源上消除子token预测的困惑度问题。
2. **字符级→行级任务转换**：将字符级随机跨度填充统一转化为多行级填充，增强了不同粒度任务间的迁移能力，使模型在 line-level infilling 任务上表现显著提升。
3. **后验验证机制（Post-Check）**：推理时生成文本需通过"以 L-Prefix 开头且以 F-Suffix 结尾"的验证，未通过则视为失败，保证了生成文本与上下文的无缝衔接。

## 方法详解
**训练流程（三步）**：
1. **Splitting**：按字符级随机分割文档为 prefix、middle、suffix 三段。
2. **Refining**：将 prefix 的最后一行标记为 L-Prefix，其余为 R-Prefix；将 suffix 的第一行标记为 F-Suffix，其余为 R-Suffix。
3. **Concatenating**：按以下格式拼接（PSM格式）：
```
<PRE> R-Prefix <SUF> R-Suffix
<START> L-Prefix <END> F-Suffix
<MID> L-Prefix Middle F-Suffix <EOT>
```
每个section独立tokenization后再拼接，确保special tokens不被切分或合并。

**推理流程**：
- 以目标行为起点，构建 prompt：`<PRE> R-Prefix <SUF> R-Suffix <START> L-Prefix <END> F-Suffix <MID>`
- 自回归生成至 `<EOT>`，通过后验检查（PCP Rate）验证是否以 L-Prefix 开头、F-Suffix 结尾，通过则裁剪前后段得到最终填充结果。

## 实验与结果
- **数据集**：StarCoder code corpus（The Stack，92种语言），过滤掉 GitHub issues/commits/Jupyter 等非代码内容；四个模型均训练 20B tokens。
- **评估基线**：InCoder（Causal Masking）、FIM-SPM、FIM-PSM、Codex、StarCoder、Code Llama。
- **核心结果**（Pass@1 on Humaneval）：
  - **random-span**：StarCoder-1B +4.7%（44.1→48.8），StarCoder-15B +1.3%，Code Llama 7B +8.1%（59.7→67.8），**Code Llama 13B +8.8%（63.6→72.4）**
  - **single-line**：Code Llama 13B **+11.5%**（75.9→87.4）
  - **multi-line**：Code Llama 13B **+10.7%**（51.0→61.7）
  - **代码生成能力**：Humaneval（30.5→36.0→37.2）和 MBPP 几乎无退化。
- **消融实验**：mask掉 L-Prefix 和 F-Suffix 的loss（LF-Loss）对 performance 影响微弱（Table 3），说明训练时子token loss 影响有限，推理时避免子token才是关键。

## 相关工作脉络
1. **FIM（Bavarian et al., 2022）**：本文的基础框架；FIM 同时训练 PSM 和 SPM 模式，但原始方法在每个随机跨度片段内可能包含 sub-token，导致不一致性。本文在 PSM 框架上增加 START/END 约束来消除 sub-token。
2. **InCoder（Fried et al., 2023）**：采用 causal masking（CM3）方式的编码器填充方法；与 FIM 路线不同，不处理 sub-token 边界问题。
3. **Code Llama（Rozière et al., 2023）**：使用 variant SPM 格式并在 `<MID>` 前合并 prefix 与 middle 来减少 sub-token；本文方法独立于 SPM 变体，两者可互补但本文暂不支持 SPM。
4. **Token Healing**：用于修复 prompt 末尾与生成文本之间的子token拼接问题，但无法处理生成文本与 suffix 之间的边界；本文方法可同时处理两端边界。
5. **GLM（Du et al., 2022）** / **MIM（Nguyen et al., 2023）**：编码器-解码器或双向LM路线，专注 token-level infilling，不适用于字符级任务。

## 局限性与未来方向
1. **不支持 SPM 模式**：由于 PSM 与 SPM 在分隔符设计上的冲突，当前方法无法兼容 variant SPM（论文承认这是主要局限，且 SPM 在 Code Llama 单行任务上表现更优）。
2. **后验失败率**：生成文本不满足 L-Prefix/F-Suffix 约束时直接判定失败，StarCoder-1B 失败率为 18.7%，StarCoder-15B 为 9.4%，限制了整体性能上限。
3. **未来方向**：探索适配 SPM 模式的方法；通过更完善的 prompt 设计和约束解码提升 PCP Rate。

## 研究启发与可借鉴点
1. **训练时消除子token→推理时避免子token预测**的思路可迁移至任何基于BPE/WordPiece tokenization 的 fill-in-the-middle 场景，尤其适用于代码补全、文档编辑等对首token精度敏感的任务。
2. **L-Prefix/F-Suffix 约束设计**本质上是一种"边界锚定"机制，可推广到 span completion、document editing 等需要保证上下文连贯性的任务。
3. **训练与推理的不对称性洞察**（训练时sub-token loss影响小，但推理时sub-token预测影响大）提醒研究者：优化目标与评估指标之间的gap需要被专门建模和关注。
4. **与 Token Healing 结合**的尝试（Appendix B.1）表明，边界修复类方法可作为辅助模块与本方法互补，值得进一步探索联合方案。

## 关键术语表
**FIM（Fill-In-the-Middle）**：一种让模型学习在给定 prefix 和 suffix 条件下填充中间内容的预训练目标，支持 PSM 和 SPM 两种排列模式。

**Sub-token**：由 BPE/WordPiece 等分词算法将一个完整 token 切分成的不完整片段，出现在文本边界处，具有高困惑度和预测不稳定性。

**L-Prefix（Last line of Prefix）**：prefix 部分的最后一行文本，作为推理生成起始约束的特殊标记段。

**F-Suffix（First line of Suffix）**：suffix 部分的第一行文本，作为推理生成结束约束的特殊标记段。

**PSM（Prefix-Suffix-Middle）模式**：FIM 的一种排列格式 `<PRE> pre <SUF> suf <MID> mid <EOT>`，与 SPM 相比 prefix 和 suffix 之间有明确分隔符。

**Post-Check（后验验证）**：推理阶段验证生成文本是否以 L-Prefix 开头且以 F-Suffix 结尾的机制，未通过则判定本次填充失败。

**PCP Rate（Post-Check Pass Rate）**：模型输出满足后验验证条件的比例，反映方法的可靠性。

**Token Healing**：通过回溯 prompt 末尾若干token并约束生成首token与前缀匹配，以修复子token拼接错误的后处理技术。

## 可复现要素
- **数据集**：StarCoder code corpus（来自 The Stack v1.2），已公开；过滤条件：去除 GitHub issues/commits/Jupyter，移除仓库/文件/star 标记
- **代码**：开源，GitHub https://github.com/SenseLLM/FIM-SE
- **权重**：论文未提及公开预训练权重
- **关键超参**：AdamW (β₁=0.9, β₂=0.95, ε=10⁻⁸, weight decay=0.1)；峰值学习率 3×10⁻⁵，cosine schedule 无 warmup；batch size 4M tokens（StarCoder 序列长 8K，Code Llama 序列长 16K）；总训练 token 数 20B；FIM rate 90%；仅使用 PSM 格式
