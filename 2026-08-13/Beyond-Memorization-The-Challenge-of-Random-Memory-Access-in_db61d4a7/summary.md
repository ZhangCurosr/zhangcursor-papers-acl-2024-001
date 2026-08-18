---
title: "Beyond-Memorization-The-Challenge-of-Random-Memory-Access-in"
source: https://aclanthology.org/2024.acl-long.185.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:55:01"
---

# 论文速读：Beyond-Memorization-The-Challenge-of-Random-Memory-Access-in

## 一句话总结
本文实证探究生成式语言模型对参数化记忆的访问机制，揭示模型擅长**顺序访问**已记忆内容，但在**随机访问**（如直接提取段落中间句子或基于ID定位答案）时存在显著瓶颈；并提出推理时显式复述（Recitation）与训练时句子乱序（Permutation）两种轻量干预手段，有效缓解该缺陷并提升开放域问答性能。

## 研究问题与动机
1. **核心科学问题**：大模型将事实知识存储在参数中，但其内部“读取/定位”记忆的具体模式（顺序遍历 vs. 跳跃随机访问）缺乏系统性实证刻画。
2. **现有方法盲区**： prior work 多聚焦于如何向模型“写入”新知识（Fine-tuning / RAG）或如何从预训练记忆中“激发”事实，忽略了记忆写入后的访问路径限制。
3. **应用层隐患**：若模型仅能顺序访问，则在长文档理解、精准抽取、依赖特定段落ID的问答等场景中会频繁失效，且难以被现有 benchmark 暴露。
4. **研究动机**：借鉴计算机体系结构的内存访问概念，通过可控合成任务与真实问答任务，量化解码器架构 LMs 的记忆访问行为，并探索可工程落地的缓解策略。

## 核心贡献（创新点）
1. **首次系统定义并量化 LMs 的顺序/随机记忆访问能力**：通过全复述、选择性复述、地面问答三个渐进任务，确立“顺序访问良好，随机访问困难”的实证结论。
2. **提出推理时显式复述（Recitation）策略**：利用 Transformer 注意力对 context window 的天然随机访问特性，让模型先顺序输出段落再执行抽取/问答，将困难任务降维。
3. **提出训练时句子级乱序（Permutation）干预**：在写入阶段打乱段落句子顺序，使任意句子有机会成为序列起点，从而提升直接定位中间内容的能力，且效应非单纯数据量扩增。
4. **在真实开放域 QA 中验证缺陷的传播效应与缓解价值**：证明随机访问瓶颈会传导至 NQ/HotpotQA 等下游任务，且 Recitation 可带来显著且稳定的 EM/F1 增益。

## 方法详解
- **记忆银行抽象**：将 LM 视为 key-value 存储 $\mathcal{D} = \{k_i : p_i\}$。写入（Writing）通过 Fine-tuning 更新参数；读取（Reading）通过 Prompt 触发生成。采用混合训练策略：训练集同时包含读写样本，验证集仅用读样本。
- **顺序访问测试（Full Recitation）**：给定 ID，要求模型从头逐词生成完整段落 $p_i$。评估不同 ID 类型（Num/Rare/Title）与段落类型（自然语言 NL / 随机字符串 Rand）下的 BLEU 与 EM。
- **随机访问测试（Selective Recitation）**：在每句两端嵌入 NLTK 分割标记 `[j]`，读取指令变为 $S_{read}(k_i, j) \to p_i[j]$，要求模型跳过前文直接输出第 $j$ 句，精确探测跳跃定位能力。
- **地面问答测试（Grounded QA）**：基于 SQuAD-v1，分别提供黄金段落 ID、随机错误 ID 或无 ID，要求模型从参数记忆中定位段落并提取答案 span，对比 Closed-book（下界）与 Open-book（上界）。
- **缓解方法一：Recitation（推理阶段）**：修改读取 prompt，要求模型先完整复述目标段落 $p_i$，再将生成内容置于 context 中执行句子提取或问答。本质是将 parametric memory 显式转移至 attention 可随机访问的上下文中。
- **缓解方法二：Permutation（训练阶段）**：对 $S_{write}$ 中的段落句子重排。包括 `first`（每句依次移至段首，构造 J 个样本）与 `random-k`（随机打乱 k 次，默认 k=4）。设置 dup-J 基线排除数据量干扰。
- **开放域 QA 扩展**：移除段落 ID，模型需自主检索相关段落并作答。使用 GPT2-XL，对比 Mixed Training（读写+QA 混合）与 Continual Training（先写后 QA）两种策略。

## 实验与结果
- **顺序访问（全复述）**：GPT2-large 在 T=400/V=40 设置下，Title ID+NL 段落达 BLEU 96.2 / EM 85.0，Num ID+NL 达 96.7 / 95.0；即使 ID 为 Rare token，Rand 段落仍可达 96.7 / 95.0。训练篇章数增至 50k 时仍能准确复述超 70% 验证集，超 100k 后性能骤降（训练收敛瓶颈）。
- **随机访问（选择性复述）**：无干预基线 BLEU 47.1 / EM 34.5，且正确预测几乎全为第 0 句（$j=0$）。+Recitation 跃升至 BLEU 99.3 / EM 98.5；+Permutation(first) 达到 BLEU 100.0 / EM 100.0；仅 Duplication(dup-J) 反降至 BLEU 36.0 / EM 23.5。
- **地面问答（SQuAD-v1，Title ID）**：Open-book 上界 EM 73.7 / F1 79.3，Closed-book 下界 EM 9.0 / F1 16.6。仅提供 Golden ID 的 Grounded QA 仅得 EM 26.7 / F1 35.6；+Recitation 后大幅提升至
