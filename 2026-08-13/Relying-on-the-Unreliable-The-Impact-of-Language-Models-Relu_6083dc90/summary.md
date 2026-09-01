---
title: "Relying-on-the-Unreliable-The-Impact-of-Language-Models-Relu"
source: https://aclanthology.org/2024.acl-long.198.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 21:43:47"
---

# 论文速读：Relying on the Unreliable: The Impact of Language Models’ Reluctance to Express Uncertainty

## 一句话总结
本文系统评估了公开部署的大型语言模型通过自然语言表达不确定性的真实倾向，发现模型普遍回避不确定性、过度依赖肯定语气，且高置信回答的错误率高达 47%；人机交互实验进一步证明人类对用户面对肯定或无标记输出均高度依赖，而 RLHF 对齐过程中的奖励模型与人工标注偏好对“弱化表达”存在隐性惩罚，是造成模型过度自信的核心成因。

## 研究问题与动机
1. **交互接口语义鸿沟**：自然语言已成为人机协作默认接口，但模型生成的回答缺乏对不确定性的显式传达，下游用户在信息检索、摘要等场景中难以判断模型输出的可信边界。
2. **数值校准的落地局限**：现有机器学习工作主要优化 ECE 等数值型校准指标，但普通用户难以直观解读概率值；语言学中的认知标记（epistemic markers）作为更自然的表达载体，其在 LM 中的真实性能尚未被系统量化。
3. **“无标记即确定”的用户误读风险**：先验研究表明人类已对 AI 产生过度依赖（overreliance），若模型默认输出不带任何语气修饰的 plain statements，用户会默认其为绝对正确，极易放大幻觉危害。
4. **对齐工艺的隐性偏见溯源**：当前 SOTA 模型普遍经过 RLHF 训练以提升 helpfulness，但该过程是否引入了对不确定语言的系统性压制，尚缺乏从 base 模型到 reward model 的全链路实证分析。

## 核心贡献（创新点）
1. **首个面向自然语言认知标记的 LM 生成行为全景量化**：突破 prior work 仅使用预定义数值或等级量表的局限，采用自下而上的定性编码+Regex 自动检测 pipeline，完整刻画 9 款公开模型在复杂问答场景下 strengtheners/weakeners 的真实分布。
2. **揭示“平淡陈述”与“肯定标记”同等触发用户过度依赖**：实验证明人类对 plain statements（如“答案是 A”）的依赖率高达 90%，与强肯定表达持平；该发现填补了此前仅关注显式语气词、忽视“无标记=默认自信”这一认知空白的研究缺口。
3. **验证早期校准偏差的长期心理模型损伤**：在交互式 50 轮实验中证明，模型前 20 轮的过度自信会导致用户形成顽固的错误心智模型，即使后续切换至校准设置，用户表现仅恢复至 86%（远低于纯校准的 97%），证明了偏差危害的不可逆性。
4. **定位 RLHF 对齐流程为模型语言过度自信的根本成因**：首次通过拆解 base/SFT/RLHF 三阶段模型并探针 OpenAssistant reward model 与四大主流偏好数据集，实证揭示奖励模型对 weakeners 施加显著负分（-1.86），且人类标注者隐性排斥不确定表达，而非偏好肯定语气。

## 方法详解
1. **模型提示与生成协议**：选取 GPT（text-davinci-003/3.5/4）、LLaMA-2（7B/13B/70B）、Claude（1/2/Instant-1）共 9 个模型，基于 MMLU 284 道四选一题目，设计三类模板：Base（仅原题）、Epi-M（“请回答并提供确定性表达”）、CoT+Epi-M 组合；共 49 个 snowball-sampled prompt 变体，temperature=0.3，max tokens=400，采用 zero-shot 模拟真实用户交互。
2. **认知标记自动分类 Pipeline**：作者先人工细读生成文本提取标签，设计 Regex 启发式规则进行批量检测，随后随机抽样 100 条进行人工复核，循环六次直至分类准确率 >90%，最终获得 76 个 strengtheners 与 105 个 weakeners 标准词表。
3. **受控人机交互实验（Setting 1–2C）**：在 Prolific 招募美国被试（每组 N=25），构建以“国家首都 trivia”为核心的自激励游戏，AI 代理名为 Marvin。Setting 1 为控制组，仅展示前缀询问依赖意愿；Settings 2A/2B/2C 为交互式校准/过自信/欠自信场景，共 50 轮，计分规则为正确依赖 +1、错误依赖 -1、自行查找 0（错误题占比 50%）。
4. **奖励模型与标注数据集溯源**：使用 OpenAssistant 开源 reward model 对 183 组 Q&A 配对打分；同时统计 WebGPT comparison、Summarize with Feedback、Syn Instruct GPT Pairwise、Anthropic HH 数据集中 strengtheners/weakeners 在 chosen/rejected 样本中的出现频率，对比模型偏好与人类标注者真实倾向。

## 实验与结果
1. **模型回避不确定性表达**：Base 提示下仅 5% 的回答含任何 epistemic marker；显式提示后可提升至 16%（Epi-M）与 65%（Epi-M+CoT）。
2. **强烈偏向肯定语气**：平均 20% 的生成含 strengtheners，仅 14% 含 weakeners；GPT 与 LLaMA-2 chat 模型偏差最显著，Claude-2 相对平衡，小模型（LLaMA-2-7B、Claude-Instant-V1）反而更频繁使用 weakeners。
3. **高置信伴随高错误率**：含肯定标记的回答整体准确率仅 53%（随机基线 2
