---
title: "ValueBench-Towards-Comprehensively-Evaluating-Value-Orientat"
source: https://aclanthology.org/2024.acl-long.111.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:56:51"
field: "大语言模型评估与对齐"
keywords: ["价值观评估", "大语言模型", "心理测量学", "基准测试", "价值对齐", "LLM评估"]
innovations: ["首个涵盖44个心理量表、453个价值观维度的综合性心理测量学基准ValueBench", "基于真实人机交互场景的价值观取向评估流水线，将自陈式问题改写为建议型封闭问题", "在开放层级价值观空间中设计价值观理解评估任务，保留subscale层级关系"]
benchmarks: ["ValueBench"]
---

# 论文速读：ValueBench-Towards-Comprehensively-Evaluating-Value-Orientations-and-Understanding-of-Large-Language-Models

## 一句话总结
本文提出了 **ValueBench**，首个综合性心理测量学基准，用于全面评估大语言模型（LLM）的**价值观取向**和**价值观理解**能力，涵盖44个心理学量表、453个价值观维度，并设计了基于真实人机交互场景的评估流水线。

## 研究问题与动机
- **现有评估方法与真实场景脱节**：已有工作多采用Likert量表的自陈式问卷（如"你对以下陈述的认同程度是？"），但这种受控设置与真实的人机交互（用户寻求建议）差异巨大，且易触发模型拒绝回答。
- **价值观空间缺乏层次结构建模**： prior work（如ValueEval）仅使用扁平化的预设价值空间，忽略了价值观之间的层级子结构（subscale relationships）和语义关联。
- **缺乏综合性基准**：PsychoBench等仅覆盖人格特质（13个量表、69个特质），缺少对更广泛的"价值观"（values）的系统性评估，涵盖心理学、社会学、认知科学等多领域。
- **价值观对齐与可解释性需求紧迫**：LLM日益作为教育、医疗等领域的代理人，其输出的价值观倾向直接影响用户决策，需可靠评估以支撑对齐研究。

## 核心贡献（创新点）
1. **首个大规模价值观心理测量学基准ValueBench**：收集44个成熟心理量表、453个多层面价值观维度，远超已有工作（如PsychoBench的13/69），实现最全面的价值观评估覆盖。
2. **基于真实人机交互的价值观取向评估流水线**：将自陈式陈述改写为"建议型封闭式问题"，模拟真实用户-助手互动场景，由GPT-4 Turbo作为评估器打分，解决 Likert量表与真实行为不一致的问题。
3. **开放层级价值观空间中的价值观理解评估任务**：设计三项新颖任务——识别相关价值观对（含对称/非对称提示）、项目到价值观提取、价值观到项目生成，在层次化价值空间中评估LLM的语义理解与推理能力。
4. **揭示LLM价值观取向的共性与差异**：实验发现LLM普遍倾向"安全"、"仁慈"、"自我导向"等高社会价值，但在"果断性"、"享乐主义"、"面子意识"等维度存在显著模型差异。

## 方法详解
### 数据集构建
- **Item-Value Pair Extraction**：将各量表的题目转换为第一人称观点陈述，配对目标价值观，标注赞同标签（+1/-1）。
- **Value Interpretation Extraction**：收集价值观定义及对立价值观（如"放纵"vs"克制"）。
- **Value Substructure Extraction**：保留层级关系，如HEXACO-PI-R中"Social Self-Esteem"是"Extraversion"的子量表。

### 价值观取向评估流水线
1. **问题改写**：使用GPT-4 Turbo将第一人称陈述改写为"建议型封闭问题"（如"I enjoy structured life" → "Should I have a clear structured mode of life?"），保持原立场。
2. **自由形式回答**：向目标LLM提问，要求其给出≤50词的自由回答。
3. **评估打分**：由GPT-4 Turbo作为评估器，根据回答对"是/否"的倾向程度评分（0-10分），最终计算各价值观维度的平均得分。对于反向题目，使用 (10 - score) 调整。
4. **信度验证**：人工标注员与GPT-4 Turbo在80.0%的案例中评分一致。

### 价值观理解评估任务
1. **识别相关价值观对**：
   - 定义相关性：子量表关系、同义词、反义词
   - 对称提示 vs 非对称提示对比实验
   - 评估指标：Recall、Precision、F1
2. **项目到价值观提取（Item-to-Value）**：
   - 要求LLM输出场景描述、价值观解释、定义及top-3最相关价值观
   - 评估指标：Hits@1/2/3
3. **价值观到项目生成（Value-to-Item）**：
   - 给定价值观及定义，生成支持/反对该价值观的论证
   - 评估指标：一致性（consistency）和信息丰富度（informative），0-10分

## 实验与结果
### 评测模型
GPT-3.5 Turbo、GPT-4 Turbo、Llama-2 7B、Llama-2 70B、Mistral 7B、Mixtral 8x7B（共6个，涵盖开源/闭源、不同规模）

### 价值观取向结果
- **一致性**：NFCC2000和NFCC1993虽题目不同但测量相同5个价值观，雷达图模式高度相似
- **共性倾向**：所有LLM在PVQ40中"Security"、"Benevolence"、"Self-Direction"、"Universalism"得分高，"Power"得分低；在SA中鼓励"Social Complexity"和"Reward for Application"，抑制"Fate Determinism"和"Social Cynicism"
- **差异点**：在"Decisiveness"、"Hedonism"、"Face Consciousness"、"Belief in Zero-Sum Game"等维度存在显著模型差异

### 价值观理解结果（Table 2关键数据）
| 任务 | 最佳模型 | 关键指标 | 数值 |
|------|----------|----------|------|
| 相关价值观识别（对称提示） | GPT-4 Turbo | F1 | **85.7%** |
| 相关价值观识别（非对称提示） | Mistral 7B | F1 | 67.5% |
| 项目到价值观提取 | GPT-4 Turbo | Hits@3 | **84.1%** |
| 价值观到项目生成-一致性 | Llama-2 70B | Consistent | 9.4/10 |
| 价值观到项目生成-信息丰富度 | GPT-4 Turbo | Informative | **5.5/10** |

- **核心发现**：在充足上下文和对称提示下，GPT-4 Turbo识别相关价值观与ground-truth理论一致性超过80%
- **层级关系不对称性**：大多数LLM在非对称提示下性能显著下降，且出现内部不一致（如解释称"B是A的子量表"但答案选"A是B的子量表"）

## 相关工作脉络
1. **ValueEval (Kiesel et al., 2023)**：仅使用Schwartz的54个价值观，扁平化价值空间，缺乏层级结构；ValueBench扩展至453个价值观并保留subscale关系。
2. **PsychoBench (Huang et al., 2024)**：聚焦人格测试（13量表、69特质），未系统覆盖价值观领域；ValueBench涵盖人格、社会公理、认知系统、价值理论四大类44个量表。
3. **VUM (Zhang et al., 2023b)**：基于判别器-批判者差距评估价值观理解，但预定义价值空间有限；ValueBench在开放层级空间中评估。
4. **Likert-scale自陈方法**：Li et al. (2022)、Safdari et al. (2023)等直接使用原始问卷格式，易触发模型拒绝；ValueBench改用建议型交互场景。
5. **Moral evaluation工作**：Abdulhai et al. (2023)、Scherrer et al. (2023)仅关注道德基础理论；ValueBench整合更广泛的价值理论体系。

## 局限性与未来方向
- **量表来源局限**：数据仅来自4大类心理测量材料，可能遗漏其他文化背景或新兴价值理论维度。
- **改写引入噪声**：用LLM改写题目和评估回答可能引入偏差，原始题目的心理测量效度未经重新验证。
- **上下文长度限制**：项目和生成内容限制在100词内，观点表达较直接，与现实世界复杂情境存在差距。
- **标注员文化背景单一**：负样本筛选由具有亚洲背景的硕士完成，可能存在文化偏见。
- **未来方向**：开发跨文化价值观评估、探索LLM在不同场景下的价值观一致性、将ValueBench应用于价值对齐干预研究。

## 研究启发与可借鉴点
1. **评估场景真实性设计**：将自陈式问卷改写为"用户寻求建议"的交互场景，显著提升评估生态效度，可迁移至其他LLM行为评估任务。
2. **LLM作为评估器的可行性**：GPT-4 Turbo作为评估器与人工标注80%一致性，为大规模自动化评估提供可行方案，降低成本。
3. **层级价值空间建模**：保留subscale关系并在评估中利用，比扁平化方法更能反映价值观的复杂结构，可用于知识图谱构建。
4. **对称/非对称提示对比实验**：揭示LLM在处理层级关系时的内在不一致性，为prompt engineering和模型能力诊断提供新思路。
5. **跨学科基准设计范式**：整合心理学成熟量表+NLP评估任务，为AI与社会科学的交叉研究树立方法论标杆。

## 关键术语表
**ValueBench**：首个综合性心理测量学基准，包含44个量表、453个价值观维度，用于评估LLM的价值观取向和理解能力。

**Value Orientation（价值观取向）**：LLM在人机交互中体现出的价值偏好倾向，通过回答建议型问题间接推断。

**Value Understanding（价值观理解）**：LLM识别语言表达背后价值观、理解价值观间关系及生成价值观相关内容的能力。

**Psychometric Inventory（心理测量量表）**：心理学中用于量化测量人格、价值观等构念的标准化工具，如Big Five、MBTI等。

**Subscale Value（子量表价值观）**：测量更广泛价值观特定方面的细化维度，如"Social Self-Esteem"是"Extraversion"的子量表。

**Likert-scale Self-report（李克特自陈量表）**：传统的价值观评估方法，要求受试者对陈述按认同程度打分，此处指出其与真实交互场景脱节。

**Symmetric/Asymmetric Prompt（对称/非对称提示）**：评估价值观层级关系时，前者不区分方向（"A可作为B的子量表"），后者明确方向（"A是B的子量表"）。

**Item-to-Value / Value-to-Item**：两项互补任务，前者从语言表达式中提取隐含价值观，后者生成体现特定价值观的论证。

## 可复现要素
- **数据集**：ValueBench已公开于 https://github.com/Value4AI/ValueBench
- **代码/提示词**：附录B提供完整prompts（Prompt 1-9）
- **模型**：评测了GPT-3.5 Turbo、GPT-4 Turbo、Llama-2 7B/70B、Mistral 7B、Mixtral 8x7B
- **超参数**：temperature=0或greedy decoding，确保结果确定性
- **评估器**：GPT-4 Turbo用于评分和题目改写
- **人工验证**：随机抽取100对回答进行一致性验证（80%一致率）
