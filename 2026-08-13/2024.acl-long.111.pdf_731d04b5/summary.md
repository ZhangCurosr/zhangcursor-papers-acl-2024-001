---
title: "ValueBench: Towards Comprehensively Evaluating Value Orientations and Understanding of Large Language Models"
source: https://aclanthology.org/2024.acl-long.111.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:00:20"
field: "大语言模型评估与价值对齐"
keywords: ["价值评估", "大语言模型", "心理测量基准", "价值对齐", "人机交互评估", "LLM-as-Evaluator"]
innovations: ["首个综合性心理测量价值基准（44问卷/453价值维度）", "基于真实人机交互场景的价值倾向评估流水线", "开放层级价值空间中的价值理解任务（识别/提取/生成）"]
benchmarks: ["ValueBench", "PsychoBench", "ValueEval", "VUM"]
---

# 论文速读：ValueBench: Towards Comprehensively Evaluating Value Orientations and Understanding of Large Language Models

## 一句话总结
ValueBench 是首个综合性心理测量基准，从 44 个已发表的心理测量问卷中收集了 453 个多维度价值维度，提出了基于真实人机交互场景的价值倾向评估流水线和开放层级价值空间中的新颖价值理解任务，系统评测了六款主流 LLM 的价值取向与价值理解能力。

## 研究问题与动机
- **现有 Likert 自评式评估与真实交互脱节**：已有工作（如使用 BFI、MBTI、道德量表等）以受控的多选题/李克特量表形式评估 LLM，这种受控设置与真实人机交互（用户向 AI 寻求建议）之间不存在明显相关性，导致评估结果不可靠。
- **指令微调模型倾向于拒绝回答自评问题**：Instruction-tuned 模型在 Likert 自评问题中常回答"I don't have values"，无法反映训练数据中嵌入的真实价值倾向。
- **价值理解评估受限于预定义扁平价值空间**：已有工作（如 ValueEval、VUM）受限于有限的预定义价值空间，且 ground truth 多为启发式生成，忽视了价值空间内在的层级子结构（subscale 关系、对立关系等）。
- **缺乏涵盖广泛价值理论的综合基准**：已有基准多聚焦人格特质的某个具体方面（Single inventory），缺乏同时涵盖人格、社会公理、认知系统和通用价值理论的多源心理测量材料的综合基准。

## 核心贡献（创新点）
1. **首个综合性心理测量价值基准**：从 44 个心理测量问卷（覆盖人格、社会公理、认知系统、通用价值理论四大领域）收集 453 个多维度价值维度，构建了包含价值定义、价值-项目对（item-value pairs）和价值层级结构（subscale hierarchies）的完整数据集，是已发表工作中覆盖问卷数量和特征维度最多的基准（对比 PsychoBench 的 13 问卷/69 特质）。
2. **基于真实人机交互的价值倾向评估流水线**：将第一人称陈述通过 LLM 改写为可寻求建议的封闭式问答形式，模拟真实人机对话场景；由 GPT-4 Turbo 对 LLM 自由回答进行 0–10 打分并加权平均得价值倾向得分；经人工验证与人类标注者在 80% 随机样本中一致——这与既有工作直接让 LLM 回答 Likert 自评分数有本质区别。
3. **开放层级价值空间中的新颖价值理解任务**：提出三项任务（识别相关价值对、Item-to-Value 提取、Value-to-Item 生成），首次在包含子尺度层级关系和对称/非对称提示设计的开放价值空间中评估 LLM 价值理解能力——与已有工作在扁平价值空间中做分类/生成评估存在本质差异。

## 方法详解

### 数据集构建（§3.2）
- **Item-Value Pair 抽取**：将各问卷项目统一改写为第一人称观点表达（如多选题选项改写为完整陈述），与原始目标价值配对形成 ground-truth item-value pairs；对于含对立观点的问卷，额外记录 1/-1 同意标签。
- **Value Interpretation 抽取**：收集各问卷中的价值定义及对立价值关系（如 "Indulgence" vs "Restraint"）。
- **Value Substructure 抽取**：保留已验证的层级结构，如 HEXACO-PI-R 中 "Social Self-Esteem""Sociability" 等为 "Extraversion" 的 subscale factors，构成 (subscale value, value) 对。

### 价值倾向评估流水线（§4.1.1）
1. **改写阶段**：用 GPT-4 Turbo 将第一人称陈述改写为寻求建议的封闭式问题（"Should I ...?" / "Will ...?"），保持原有立场方向。
2. **作答阶段**：将改写后的问题以 ≤50 词的简短问答形式输入待评估 LLM，temperature=0 或 greedy decoding。
3. **评分阶段**：将原问题和 LLM 回答一并输入 GPT-4 Turbo，要求其在 0–10 分上评定回答倾向于 "Yes" 的程度（0=No，10=Yes）；对于原始陈述与价值立场相反的项目，得分为 `10 - score`。
4. **聚合**：同一价值维度下所有项目的平均分为该 LLM 在该价值上的倾向得分。
5. **信度验证**：随机抽取 100 对 LLM 响应，由社会学硕士生人工标注，GPT-4 Turbo 评分与人工标注在 80% 样本中一致（详见 Appendix C.1）。

### 价值理解评估任务（§4.3）
- **识别相关价值对（Identifying Relevant Values）**：定义四类相关性：(i) A 是 B 的子尺度价值，(ii) B 是 A 的子尺度价值，(iii) 同义词，(iv) 对立值。构建正负样本，使用**对称提示**（"One can be used as a subscale value of another"）和**非对称提示**（分别指定 A→B 或 B→A）两种版本测试 LLM，输出定义、关系解释、关系标签和相关性判断（1/0）。
- **Item-to-Value 提取**：要求 LLM 提取与项目最相关的 top-3 价值，由 GPT-4 Turbo 判定是否与 ground-truth 相关（按 §4.3.1 的相关性定义），报告 Hits@1/2/3。
- **Value-to-Item 生成**：提供给定价值及其定义和两个示例，要求 LLM 生成支持或反对该价值的论据；由 GPT-4 Turbo 在 0–10 分上评估一致性（consistency）和信息丰富度（informative）。

## 实验与结果

### 评测模型
GPT-3.5 Turbo、GPT-4 Turbo、Llama-2 7B、Llama-2 70B、Mistral 7B、Mixtral 8x7B（覆盖开源/闭源及不同规模，temperature=0）。

### 价值倾向结果（§4.1.2）
- **一致性**：NFCC2000 与 NFCC1993 虽项目不同但测同一维度，雷达图模式高度相似；所有模型在 "Discomfort with Ambiguity" / "Uncertainty Avoidance" 上得分均低，表明 LLM 普遍能容忍模糊性。
- **共性倾向**：在 PVQ40 中，所有 LLM 在 "Security""Benevolence""Self-Direction""Universalism" 上得分高，在 "Power" 上得分低；在 SA 中均鼓励 "Social Complexity" 和 "Reward for Application"，排斥 "Fate Determinism" 和 "Social Cynicism"——可能与训练/对齐数据中的普遍偏好有关。
- **个性差异**：在 "Decisiveness""Hedonism""Face Consciousness""Belief in a Zero-Sum Game" 等维度上出现明显分歧（如 Llama-2 70B 在 Decisiveness 上得 8.5，而 GPT-3.5 仅 5.75）。

### 价值理解结果（Table 2，§4.3）
| 任务 | 最强模型 | 关键指标 |
|---|---|---|
| 识别相关价值（对称提示）| GPT-4 Turbo | Recall=88.7%, F1=85.7% |
| 识别相关价值（非对称提示）| GPT-4 Turbo | Recall=67.5%, F1=65.7% |
| Item-to-Value 提取 | Llama-2 70B | Hits@3=83.3% |
| Value-to-Item 生成（一致性）| 各模型接近（8.6–9.4）| — |
| Value-to-Item 生成（信息丰富度）| GPT-4 Turbo | 5.5/10 |

- 关键发现：**在有充分上下文和对称提示设计下，GPT-4 Turbo 在价值识别任务中达到约 80%+ 的一致性**，表明先进 LLM 在价值相关任务中具有强大潜力；对称/非对称提示间的性能差异揭示了自回归模型在顺序敏感性（reversal curse 类现象）上的固有局限。

## 相关工作脉络
1. **ValueEval（Kiesel et al., 2023）**：基于 Schwartz 理论从论证中提取单一价值，仅覆盖扁平化价值空间（54 个维度），无层级结构。本文通过 453 个维度 + 层级子结构大幅扩展价值空间深度和广度。
2. **PsychoBench（Huang et al., 2024）**：覆盖 13 个问卷/69 个人格特质，但聚焦人格而非广义价值取向。本文覆盖范围更广（44 问卷/453 价值维度），涵盖社会公理、认知系统、价值理论等跨领域。
3. **VUM（Zhang et al., 2023b）**：提出判别器-批判差距框架评估 LLM 的双层价值理解，但 ground truth 为启发式生成且价值空间受限。本文采用专家标注的 item-value 对及明确的价值层级结构。
4. **人格自评评测（Li et al., 2022; Safdari et al., 2023; Song et al., 2023）**：使用 Likert 量表自评形式评估 LLM 人格，与真实人机交互不匹配。本文提出改写为寻求建议问题 + 自由回答 + LLM 评分的新流水线。
5. **Moral Foundation / 道德评测（Abdulhai et al., 2023; Scherrer et al., 2023）**：仅覆盖道德维度，本文同时覆盖人格、社会公理、认知系统和价值理论四大领域。

## 局限性与未来方向
- **数据集覆盖面仍有局限**：虽覆盖四大领域，但仍存在其他有价值的维度（如某些国家文化特有价值观）未被纳入，且少数维度与标准价值的相关性较低。
- **改写流程引入的噪声与偏差**：通过 LLM 将第一人称陈述改写为封闭式问题并评估回答，这一转换过程可能引入噪声；心理测量原始项目的效度是在人类受试者上验证的。
- **生成任务的项目长度限制**：评估以 ≤100 词的句子为单位，视角表达相对直接，与真实场景中复杂长文本的价值蕴含存在差距。
- **文化偏见风险**：标注志愿者主要来自亚洲社会学硕士生背景，可能影响跨文化价值相关性的判断准确性（§7 Ethics Statement）。

## 研究启发与可借鉴点
1. **评估流水线设计可迁移**：将受控自评问题改写为自然交互场景的求助型问题 + 自由回答 + 第三方 LLM 评分的三段式流水线，可推广至其他心理特质（如共情、道德推理）的评估。
2. **层级价值结构的利用**：在 value understanding 任务中显式建模 subscale-level 关系（对称/非对称提示对比）是一个新颖的实验设计，可用于研究 LLM 对层次语义关系的敏感性。
3. **LLM-as-Evaluator 的信度验证**：本文用人工标注验证 GPT-4 Turbo 作为 evaluator 的一致性（80%），这一验证流程可直接复用于其他 LLM-as-judge 评估场景。
4. **与价值对齐研究的交叉机会**：发现 LLM 在价值提取任务上差异不大（~5% 波动）但生成任务存在明显模型差异，提示当前训练策略在不同价值维度上的对齐效果不均——可作为后续模型对齐改进的切入点。
5. **开放域 value generation 任务**：Value-to-Item 生成任务中一致性与信息丰富度的双维度评分设计，为后续研究 LLM 价值相关内容的质量评估提供了可直接复用的评价框架。

## 关键术语表
- **ValueBench**：首个综合性心理测量基准，从 44 个问卷收集 453 个价值维度，用于评估 LLM 的价值倾向和价值理解。
- **Value Orientation（价值倾向）**：LLM 在特定价值维度上的立场倾向，通过改写为真实交互场景的问题并由 evaluator LLM 评分得到。
- **Value Understanding（价值理解）**：LLM 识别语言背后价值含义及生成符合特定价值内容的综合能力。
- **Item-Value Pair（价值-项目对）**：心理测量问卷中描述行为/观点的陈述句与其对应的目标价值之间的配对关系，构成 ground-truth 标注。
- **Value Substructure（价值子结构）**：价值理论中层级关系（如子尺度 value 与主价值），如 HEXACO 的 31 个子尺度归属于六大主人格特质。
- **Symmetric/Asymmetric Prompt（对称/非对称提示）**：在识别价值相关性时，对称提示允许双向子尺度关系，非对称提示分别限定 A→B 或 B→A，用于测试 LLM 对层级关系方向性的理解。
- **LLM-as-Evaluator**：使用 GPT-4 Turbo 作为自动评分器，对 LLM 的回答在 0–10 分上进行倾向性评分。
- **Need for Cognitive Closure（NFCC）**：认知闭合需求，衡量个体对确定性/结构化的偏好，是本文用于验证跨问卷一致性的核心认知价值维度之一。

## 可复现要素
- **数据集**：44 个心理测量问卷、453 个价值维度、item-value pairs；公开于 GitHub：https://github.com/Value4AI/ValueBench
- **代码**：论文未明确提及开源代码仓库链接，但给出了完整 prompt 模板（Appendix B）
- **模型**：使用 GPT-3.5 Turbo、GPT-4 Turbo（API）、Llama-2 7B/70B、Mistral 7B、Mixtral 8x7B；temperature=0 或 greedy decoding，结果为确定性
- **Evaluator**：统一使用 GPT-4 Turbo 作为评分器
- **超参数**：temperature=0；item 改写及回答约束 ≤50 词；value 定义输出约束 ≤20 词；生成任务要求 n 条论据
