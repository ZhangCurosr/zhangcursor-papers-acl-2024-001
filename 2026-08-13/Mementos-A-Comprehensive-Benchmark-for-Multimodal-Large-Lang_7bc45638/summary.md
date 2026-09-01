---
title: "Mementos-A-Comprehensive-Benchmark-for-Multimodal-Large-Lang"
source: https://aclanthology.org/2024.acl-long.25.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:44:49"
---

# 论文速读：Mementos: A Comprehensive Benchmark for Multimodal Large Language Model Reasoning over Image Sequences

## 一句话总结
本文提出了 Mementos 基准，专注于评估多模态大语言模型（MLLMs）对图像序列的动态时序推理能力；通过构建涵盖日常、机器人与漫画三领域的 4,761 条图像序列数据集及 GPT-4 辅助的同义词图评测协议，系统揭示了当前主流 MLLMs 在序列理解中普遍存在严重的行为幻觉与对象幻觉问题，并定量归因了幻觉产生的三大核心机制。

## 研究问题与动机
1. **现有基准的静态局限**：当前 MLLM 评测（如 POPE、Bingo、HallusionBench）主要聚焦单张静态图像的对象识别与问答，缺乏对时间维度上动态事件演化与行为推理的评估。
2. **序列推理的现实需求**：真实世界理解依赖对连续视觉输入的时序建模，现有方法在处理图像序列时频繁产生对象误识别与行为编造，阻碍了其在具身智能、视频理解等场景的安全部署。
3. **行为幻觉缺乏专用评测指标**：既往工作多关注静态对象幻觉，针对“行为幻觉（behavioral hallucination）”的定义、测量及成因分析存在空白，亟需细粒度的评测范式。
4. **开源与闭源模型的跨域性能鸿沟**：预训练数据分布差异可能导致开源 MLLMs 在 Robotics/Comics 等序列推理领域表现显著弱于商业闭源模型，需系统性量化这一差距。

## 核心贡献（创新点）
1. 提出 Mementos 序列推理基准，构建包含 4,761 条图像序列的多领域评测数据集。**与 POPE/Bingo 等静态单图评测不同，本文首次将“时序行为演化”纳入核心评测维度，填补了动态推理基准的空白。**
2. 设计 GPT-4 辅助的同义词图细粒度评测协议。**有别于传统 BLEU/ROUGE 或纯 LLM-as-Judge 的粗糙评分，该方法通过词根归一化消除词汇变体干扰，且经人工复核与 GPT-4 评估误差仅 1%-4%。**
3. 界定并量化“行为幻觉”现象。**不同于既往研究仅关注静态对象幻觉，本文独立拆解行为维度并证实行为错误率显著高于对象错误率，建立了专属的细粒度诊断指标。**
4. 揭示序列推理失败的三大核心因素。**与以往仅做案例定性分析的工作不同，本文通过相关性系数、分布曲线与跨域对比实现了幻觉成因的定量归因。**

## 方法详解
1. **数据集构建**：
   - **Daily-life**：从 Next-QA 数据集抽取 400-2,500 帧视频，保留首帧后每隔 100 帧采样，共 3,505 条序列。
   - **Robotics**：基于 Open X-Embodiment 子数据集（分辨率>128x128），按帧数动态采样（>100帧每n/20帧，20-100帧每5帧），共 1,101 条。
   - **Comics**：无字漫画与电影分镜截图，共 155 条。
   - 标注聚焦于“主体对象”及其对应“行为动词/动词短语”；Daily-life 采用 GPT-4V 初稿+人工精修+交叉校验，Robotics/Comics 全人工标注。
2. **GPT-4 辅助评测流程**：
   - 输入 prompt 要求 MLLM 生成单段落序列描述。
   - 使用 GPT-4 从 AI 生成文本与人工标注文本中分别抽取对象（O）与行为（B）关键词。
   - 构建领域专属的**单向同义词有向图**（`synonym → root_word`），将抽取词归一化至标准词根，避免同义表述导致的误判。
   - 以人工标注词表为 Ground Truth，分别计算对象与行为的 Recall、Precision 与 F1。
3. **输入范式对比**：Sequential-input（s-input，逐帧顺序输入）vs. Combined-input（c-input，多帧拼合为单图输入），评估不同视觉编码策略对序列推理的影响。
4. **失败归因分析**：通过皮尔逊相关系数量化对象/行为幻觉关联；通过案例统计验证共现行为先验偏差；绘制 Recall 随 Episode Length 变化的曲线揭示雪崩效应。

## 实验与结果
- **评测模型**：9 个主流 MLLM（GPT-4V, Gemini, Video-LLaMA-2, Chat-UniVi, LLaVA-1.5, MiniGPT4, MiniGPT5, mPLUG_Owlv2, InstructBLIP）。
- **核心发现**：
  1. **行为推理普遍弱于对象识别**：最强模型 GPT-4V (s-input) 在 Daily-life 对象 Recall 达 59.80%，但行为 Recall 仅 36.71%，行为 Precision 全面低于对象 Precision。
  2. **闭源模型显著领先**：GPT-4V (s-input) 综合表现最佳；在 Robotics 与 Comics 领域，闭源模型指标约为开源模型的 2 倍，主要受训练数据分布偏移影响。
  3. **开源模型最优为 LLaVA-1.5**：在 Daily-life 领域接近 Gemini，但在跨域泛化上仍存差距；Video-LLaMA-2 与 Chat-UniVi 虽专为视频设计，但未展现出对 LLaVA-1.5 的明显优势。
  4. **雪崩效应显著**：随着 Episode Length 增加，GPT-4V 与 LLaVA-1.5 的对象/行为 Recall 均呈单调下降趋势，错误随序列推进累积放大。
- **最强结果**：GPT-4V (s-input) 在 Robotics 领域对象 F1 达 62.99%，行为 F1 达 33.95%，为全表最高；整体而言 MLLMs 在
