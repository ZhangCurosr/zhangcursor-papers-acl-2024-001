---
title: "ValueBench: Towards Comprehensively Evaluating Value Orientations and Understanding of Large Language Models"
source: https://aclanthology.org/2024.acl-long.111.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:00:18"
field: "大语言模型评估与价值观对齐"
keywords: ["价值观评估", "大语言模型", "心理测量基准", "价值对齐", "价值观理解", "LLM评测"]
innovations: ["首个综合心理测量价值观基准ValueBench，覆盖44个量表453个价值维度", "基于真实人机交互的价值观取向评估管线，替代有局限的李克特自评", "开放层次价值空间中的价值观理解评测任务体系（相关价值识别/抽取/生成）"]
benchmarks: ["ValueBench"]
---

# 论文速读：ValueBench: Towards Comprehensively Evaluating Value Orientations and Understanding of Large Language Models

## 一句话总结
本文提出了 ValueBench，首个面向 LLM 价值观取向与价值观理解的综合心理测量基准，从 44 个心理学量表中提取 453 个多维价值维度，设计了基于真实人机交互场景的价值观取向评估管线和开放层次价值空间中的价值观理解任务，揭示了 LLM 共性与个性的价值取向，并证明前沿 LLM 在充分上下文下能以超过 80% 的一致性逼近心理学专家结论。

## 研究问题与动机
- **核心价值观评估缺乏综合性基准**：现有工作多聚焦人格单一面向（如大五、MBTI），缺乏覆盖广泛价值观维度的统一评测框架，而价值观是驱动人类决策的核心心理构念，对 LLM 对齐至关重要。
- **既有评测方法与人机交互现实脱节**：已有研究直接沿用李克特量表的自陈式自评问答（如"你有多同意这句话"），此类受控设定下的评分与真实用户咨询场景中 LLM 的建议输出无显著相关性，且指令微调模型常触发拒绝回答。
- **价值观理解评测受限于扁平化预设空间**：现有 VUM、ValueEval 等工作使用有限预定义价值空间、启发式构造的 ground truth，忽视了价值观空间固有的层次子结构与多维关联，难以评估 LLM 对复杂价值语义的理解。
- **缺少对价值观层级关系的系统性考察**：此前工作通常将价值维度视为独立平铺结构，忽略了子尺度值与上级值之间的层级关系（如"Authority"是"Power"的子尺度值），阻碍了对 LLM 价值推理能力的深入评估。

## 核心贡献（创新点）
- **构建了首个大规模心理测量价值观基准 ValueBench**：从 44 个成熟心理学量表中系统采集 453 个多维价值维度及对应题项，涵盖人格、社会公理、认知系统和一般价值理论四大领域，是截至目前最全面的价值观心理测量基准。
- **设计了基于真实人机交互场景的价值观取向评估管线**：将自陈式陈述改写为求助型封闭式问答，让 LLM 生成自由形式建议，再由 GPT-4 Turbo 评估回答倾向度（0-10 分），比传统李克特量表自评更能反映 LLM 在实际咨询场景中的价值取向。
- **提出了开放层次价值空间中的价值观理解评测任务体系**：包含相关价值识别（正/负样本配对）、项目到价值抽取（Item-to-Value）、价值到项目生成（Value-to-Item）三类任务，首次系统评估 LLM 在层级关联价值空间中的理解与生成能力。
- **揭示了六款主流 LLM 的共性/个性价值取向**：发现所有模型在 Security、Benevolence、Self-Direction、Universalism 上表现一致，而在 Decisiveness、Hedonism、Face Consciousness 等方面存在显著差异，为 AI 价值对齐研究提供了实证基础。

## 方法详解

**数据集构建三要素**：
1. **题项-价值对提取（Item-Value Pair Extraction）**：将各量表中的陈述统一改写为第一人称观点表达，建立题项与目标价值的配对，并标注赞同标签（1 = 支持该价值，-1 = 反对）。
2. **价值解释提取（Value Interpretation Extraction）**：收集各量表中价值的定义（形容词或名词短语），同时记录对立价值关系（如"Indulgence" vs "Restraint"）。
3. **价值子结构提取（Value Substructure Extraction）**：收集价值间的层级关系（子尺度值与主值），如 HEXACO-PI-R 中 "Social Self-Esteem" 等四个子维度属于 "Extraversion" 主维度。

**价值观取向评估管线**：
- Step 1：利用 GPT-4 Turbo 将自陈陈述改写为 "Should I do X?" 形式的求助型封闭式问题，保持原始立场不变。
- Step 2：以 temperature=0 运行目标 LLM 生成 ≤50 词的自由形式建议回答。
- Step 3：由 GPT-4 Turbo 作为评估器，判断回答倾向 "Yes" 或 "No" 的程度（0-10 分），反向题通过 (10 - score) 调整。
- Step 4：同一价值下所有题项得分平均得到该价值取向分数。

**价值观理解评测任务**：
- **相关价值识别（Identifying Relevant Values）**：将共享子尺度关系、同义关系或对立关系的价值对标记为正样本，随机采样配对标记为负样本；测试对称提示（"One can be used as subscale of another"）与不对称提示两种设置下的 Recall、Precision、F1。
- **项目到价值抽取（Item-to-Value Extraction）**：要求 LLM 提取与题项最相关的 Top-3 价值，由 GPT-4 Turbo 评估与 ground-truth 的相关性，报告 Hits@1/2/3。
- **价值到项目生成（Value-to-Item Generation）**：给定价值和定义，要求 LLM 生成赞成/反对该价值的论据，由 GPT-4 Turbo 从 Consistency（0-10）和 Informative（0-10）两个维度评分并取均值。

**人评一致性验证**：随机抽取 100 对被 GPT-4 Turbo 判为不同分的回答对，由社会学硕士生标注，GPT-4 Turbo 与人工标注在相对评分上的一致性达 80.0%。

## 实验与结果

**实验设置**：评测了 6 款 LLM——GPT-3.5 Turbo、GPT-4 Turbo、Llama-2 7B、Llama-2 70B、Mistral 7B、Mixtral 8x7B，temperature=0（确定性输出）。

**价值观取向核心发现**：
- 跨量表一致性：NFCC2000 与 NFCC1993 虽题项不同但测量相同 5 个价值，雷达图模式高度相似。
- 跨模型共性：所有 LLM 在 PVQ-40 的 "Security"、"Benevolence"、"Self-Direction"、"Universalism" 上得分高，"Power" 得分低；在 SA 量表中均鼓励 "Social Complexity" 和 "Reward for Application"。
- 跨模型个性：在 "Decisiveness"、"Hedonism"、"Face Consciousness"、"Belief in a Zero-Sum Game" 上表现分化明显。
- LLM 整体对模糊性和不确定性的接受度较高（Discomfort with Ambiguity / Uncertainty Avoidance 均得分较低）。

**价值观理解核心结果（Table 2）**：
| 任务 | 最佳模型 | 关键指标 |
|---|---|---|
| 相关价值识别（对称提示） | GPT-4 Turbo | Recall 88.7%, Precision 82.9%, F1 85.7% |
| 相关价值识别（不对称提示） | GPT-4 Turbo | Recall 67.5%, Precision 64.0%, F1 65.7% |
| 项目到价值抽取 | GPT-4 Turbo | Hits@1: 69.3%, Hits@2: 77.6%, Hits@3: 84.1% |
| 价值到项目生成 | Llama-2 70B | Consistent: 9.4/10, Informative: 5.5/10 |

- 对称提示下 SOTA 模型（GPT-4 Turbo）识别相关价值的一致性超过 80%。
- 引入价值定义等充分上下文后，LLM 对正样本的 Recall 显著提升。
- 将对称提示转为不对称提示时，大多数模型性能明显下降，且出现内部不一致（如解释中说 B 是 A 的子尺度值却判定 A 是 B 的子尺度值）。
- 不同模型在价值抽取上差距较小（约 5% 波动），但在生成任务中表现分化（GPT-4 Turbo 信息丰富度高，Llama-2 70B 一致性高）。

## 相关工作脉络
- **PsychoBench (Huang et al., 2024)**：面向 LLM 人格测试，包含 13 个量表 69 个人格特质；ValueBench 覆盖 44 个量表 453 个价值维度，规模与广度显著超越，且聚焦于价值观而非单纯人格特质。
- **ValueEval (Kiesel et al., 2023)**：仅基于 Schwartz 单一理论，价值空间扁平且有限；ValueBench 整合多维度理论并保留层级子结构，评估任务更全面。
- **VUM (Zhang et al., 2023b)**：量化评估 LLM 双层价值观理解，但依赖启发式生成的 ground truth；ValueBench 采用心理学量表专家标注的 item-value 对作为 ground truth，更可靠。
- **早期单一量表评测（Li et al., 2022; Safdari et al., 2023; Abdulhai et al., 2023 等）**：仅使用 1-6 个量表评估特定面向（人格/道德）；ValueBench 首次系统整合多领域心理测量工具构建综合基准。
- **Likert-scale 自陈式评测方法（Song et al., 2023; Miotto et al., 2022; Jiang et al., 2023b）**：直接要求 LLM 对陈述进行 Likert 评分；本文证明该方法与真实人机交互情境存在显著不一致，且易触发指令微调模型的拒绝回答。

## 局限性与未来方向
- **量表领域覆盖有限**：仅从人格、社会公理、认知系统和一般价值理论四大领域采集，部分与特定国家描述更相关的维度被纳入但价值关联度较低。
- **改写与评估引入的噪声**：通过 LLM 改写题项和用 GPT-4 Turbo 评估回答可能引入噪声与偏差，原量表在人类受试者中的效度不保证迁移到改写形式后同样有效。
- **题目长度限制**：评估中题项和生成内容均控制在 100 词以内，表达较为直接，与现实世界中更复杂的价值观表达场景存在差距。
- **文化偏差风险**：原始材料隐含区域与文化偏差， annotators 为亚裔社会学硕士生，可能对跨文化价值相关性的判断存在局限。
- **未来方向**：开发跨情境一致的 LLM 行为评估方法、扩展更多文化与领域的价值维度、探索 LLM 在真实大规模人机交互中的价值观动态变化。

## 研究启发与可借鉴点
- **从受控自评到真实交互场景的评估范式迁移**：将 Likert 自评问题改写为求助型问答再评估回答倾向，这一思路可迁移至其他心理构念（如态度、信念）的 LLM 评估，显著提升评测生态效度。
- **层次化价值空间的构建方法**：保留子尺度值-主值的层级关系而非扁平化处理，这一数据组织方式可借鉴到其他涉及结构化概念空间的评测任务中（如伦理原则体系、法律法规层级）。
- **GPT-4 Turbo 作为评估器的可靠性验证流程**：通过人工标注抽样验证评估器一致性（80%），为后续使用 LLM-as-judge 的评测工作提供了可信度验证的参考范式。
- **对称/不对称提示对照实验设计**：通过对比对称与不对称提示下的性能差异揭示 LLM 对提示顺序的敏感性，这一实验设计可用于诊断其他结构化推理任务中 LLM 的对称性违背问题。
- **跨领域心理测量工具的整合策略**：从四个不同领域的 44 个量表系统采集数据的策略，为构建跨学科综合评测基准提供了可复用的数据整合方法论。

## 关键术语表
**ValueBench**：本文提出的首个综合心理测量基准，用于评估 LLM 的价值观取向和价值观理解，涵盖 44 个量表、453 个价值维度。
**价值观取向（Value Orientations）**：LLM 在生成建议回答时所隐含体现的价值倾向，通过模拟真实人机交互场景进行评估。
**价值观理解（Value Understanding）**：LLM 识别语言背后价值观以及在开放价值空间中生成符合特定价值的表达能力。
**心理测量量表（Psychometric Inventory）**：心理学中用于量化测量个体心理特质或价值观的标准化测试工具，如大五人格量表、 Schwartz 价值观问卷等。
**题项-价值对（Item-Value Pair）**：将量表中的具体陈述（题项）与其对应的目标价值建立配对关系，并标注赞同/反对标签。
**价值子结构（Value Substructure）**：价值理论中不同价值维度之间的层级关系，如子尺度值与主值之间的从属关系。
**Likert-scale 自陈式测评**：要求被试对一系列陈述按同意程度评分的经典心理测量方法，本文指出其直接用于 LLM 评测的局限性。
**GPT-as-Judge**：使用 GPT-4 Turbo 作为自动化评估器对 LLM 输出进行评分，本文验证了其与人工标注 80% 的一致性。

## 可复现要素
- **数据集**：ValueBench 已开源，访问地址 https://github.com/Value4AI/ValueBench
- **代码**：论文声明开源，代码托管于上述 GitHub 仓库
- **模型权重**：使用 GPT-3.5 Turbo、GPT-4 Turbo、Llama-2 7B/70B、Mistral 7B、Mixtral 8x7B，其中 GPT 系列为 API 调用，Llama/Mistral 系列为公开开源模型
- **关键超参**：temperature=0（确定性解码），greedy decoding；生成回答限 50 词以内；价值定义解释限 20 词以内
- **评估器**：GPT-4 Turbo 作为 evaluator LLM
- **论文未提及**：训练数据的具体划分比例、GPU 硬件配置、推理耗时
