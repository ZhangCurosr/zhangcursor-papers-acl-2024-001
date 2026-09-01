---
title: "Unified-Hallucination-Detection-for-Multimodal-Large-Languag"
source: https://aclanthology.org/2024.acl-long.178.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:56:12"
---

# 论文速读：Unified-Hallucination-Detection-for-Multimodal-Large-Languag

## 一句话总结
本文提出了首个任务与类别统一的多模态大语言模型幻觉检测框架 UNIHD 及细粒度元评测基准 MHaluBench，通过“核心 claim 提取→LLM 自主生成工具查询→并行调用专用工具获取证据→带理据的幻觉验证”流水线，同时覆盖图像到文本与文本到图像任务中的物体、属性、场景文本与事实四类幻觉，实现了跨任务、多粒度、可解释的鲁棒检测。

## 研究问题与动机
1. **任务单一性**：现有研究多聚焦图像到文本（如 IC、VQA）的幻觉检测，忽视了文本到图像生成同样存在严重且独特的模态冲突问题。
2. **幻觉类别覆盖不足**：Prior works 主要关注物体级幻觉，缺乏对属性冲突、场景文本错误及事实不符等高频幻觉的系统性评估。
3. **评估粒度粗糙**：传统方法仅对整段回复给出全局标签，无法定位至具体 claim（主张单元），难以支撑精准的模型诊断与改进。
4. **纯自检测机制瓶颈**：仅依赖 MLLM 自身内省（Self-Check）在细粒度与事实类场景下证据链薄弱，易受模型先验偏差干扰，可靠性受限。

## 核心贡献（创新点）
1. **提出三统一幻觉检测范式**：首次将图像到文本与文本到图像任务纳入同一检测框架，并统一界定模态冲突与事实冲突两类幻觉，弥补了现有工作任务割裂的缺陷。
2. **构建细粒度多模态基准 MHaluBench**：提供 620 条样本（IC 200、VQA 200、T2I 220），支持 Response/Segment/Claim 三级粒度，覆盖四类幻觉，标注者间一致性达 Fleiss’s κ=0.822。
3. **设计工具增强的自主路由检测框架 UNIHD**：利用底座 MLLM 根据 claim 语义自主决策调用哪些工具并生成对应查询，打破固定工具链的局限，提升框架泛化性。
4. **引入带理据的联合验证机制**：不仅输出 HALLUCINATORY/NON-HALLUCINATORY 二分类，还要求模型综合多工具证据生成可解释的判断依据，显著增强检测结果的可信度与可分析性。

## 方法详解
UNIHD 采用四阶段流水线，以 GPT-4V/Gemini 为底座 MLLM：
1. **核心 Claim 提取（Essential Claim Extraction）**：借助强指令遵循能力，将图像到文本模型的生成回复拆分为独立 claim，或将文本到图像的用户提示词解构为最小意图概念，作为验证原子单元。
2. **基于查询生成的自主工具选择（Autonomous Tool Selection Via Query Formulation）**：对每个 claim，底座 MLLM 自主判断是否需要工具，并生成面向特定维度的查询语句（如对象类输出关键词列表 `[athlete, uniform]`，属性类输出“右侧运动员队服什么颜色？”，无需工具时输出 `none`）。
3. **并行工具执行（Parallel Tool Execution）**：根据查询并发调用四类专用工具获取客观证据：
   - **对象级**：Grounding DINO 进行开放集检测，返回边界框与归一化坐标；
   - **属性级**：直接由底座 MLLM 回答属性问题（自我反思机制）；
   - **场景文本级**：MAERec 识别图像内场景文字及坐标；
   - **事实级**：Serper Google Search API 检索并提取 Top 结果片段。
4. **带理据的幻觉验证（Hallucination Verification with Rationales）**：将原始视觉输入、claim 列表及各工具返回证据聚合至统一 prompt，指令 MLLM 对每个 claim 进行二分类判定，并输出支撑判断的理据（rationales）。框架整体公式化表述为：给定输入对 $a = \{v, x\}$ 及其 claim 集合 $\{c_i\}_{i=1}^n$，输出 $\{\hat{y}_i, r_i\}_{i=1}^n$，其中 $\hat{y}_i \in \{\text{H}, \text{Non-H}\}$，$r_i$ 为解释文本。

## 实验与结果
- **数据集与评估设置**：MHaluBench（620 样本）；指标包括 Precision、Recall、Micro-F1（默认）与 Macro-F1；评估粒度分 Claim 与 Segment。
- **基线方法**：Self-Check (0-shot) 与 Self-Check (2-shot)，分别基于 Gemini 与 GPT-4V，依赖纯 CoT 内省无外部工具。
- **主要结果**：GPT-4V 底座显著优于 Gemini；UNIHD（GPT-4V）在 Image-to-Text Claim 级取得 P=82.54、R=85.29、F1=83.89，在 Text-to-Image Claim 级取得 P=84.92、R=85.79、F1=86.73，全面超越所有 Self-Check 基线。Self-Check (0-shot) GPT-4V Claim 级 Macro-F1 为 72.82，而 Gemini 仅 52.98，凸显底座模型能力与外部证据的双重重要性。
- **增益分析**：工具增强对 Scene-Text 与 Fact 幻觉提升最显著，说明纯 LLM 内省在此类任务上存在明显证据盲区；Attribute 级提升有限，主要受限于缺乏专用属性检测工具。
- **任务难度对比**：Text-to-Image 幻觉检测整体优于 Image-to-Text，原因在于人工编写提示词结构规整、图像分布集中；而自然图像背景复杂、多样性高，模态对齐难度更大。
- **模型幻觉 ranking**：以 UNIHD(GPT-4V) 作为“黄金检测器”横向评测主流 MLLM，结果与现有排行榜/人工评估
