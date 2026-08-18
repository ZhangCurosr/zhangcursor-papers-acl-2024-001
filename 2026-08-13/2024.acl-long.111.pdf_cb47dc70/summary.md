---
title: "ValueBench: Towards Comprehensively Evaluating Value Orientations and Understanding of Large Language Models"
source: https://aclanthology.org/2024.acl-long.111.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:00:46"
---

# 论文速读：ValueBench: Towards Comprehensively Evaluating Value Orientations and Understanding of Large Language Models

## 一句话总结
本文提出了**ValueBench**，首个综合性心理测量学基准，通过模拟真实人机交互的评估流水线与开放层级价值空间任务，系统性评测大语言模型（LLM）的价值观取向（Value Orientations）与价值观理解（Value Understanding）能力。

## 研究问题与动机
1. **现有评估脱离真实交互场景**：主流工作直接沿用李克特量表（Likert-scale）自陈式问卷，要求LLM对陈述句进行固定刻度打分，与LLM作为助手提供建议的真实交互模式差异巨大，且极易触发指令微调模型的拒答机制。
2. **价值观空间过于扁平与受限**：现有评测多局限于预定义的有限价值观集合（如Schwartz的10维），忽视了心理测量学中广泛存在的层级结构（子尺度、同义、对立关系），限制了对LLM语义理解深度的考察。
3. **缺乏系统性、大规模的价值观理解评测**：针对价值观理解的任务多依赖启发式生成的Ground Truth，且评测维度单一，未能在开放-ended的价值空间中评估LLM的抽取与生成能力。
4. **心理测量材料整合不足**：心理学界成熟的44套标准化量表此前未被系统引入NLP评测，导致已有工作覆盖的价值观维度碎片化，难以刻画LLM的整体价值画像。

## 核心贡献（创新点）
1. **构建首个大规模综合心理测量学基准ValueBench**：汇集
