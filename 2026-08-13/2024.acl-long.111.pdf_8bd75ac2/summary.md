---
title: "ValueBench: Towards Comprehensively Evaluating Value Orientations and Understanding of Large Language Models"
source: https://aclanthology.org/2024.acl-long.111.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:00:32"
field: "LLM价值观评估"
keywords: ["价值取向评估", "价值理解", "心理学测量基准", "大语言模型", "人机交互评估", "价值观对齐"]
innovations: ["首个整合44个心理量表453个价值维度的综合心理学测量基准", "基于真实人机交互场景的价值取向评估流程（改写为建议寻求问题+LLM评分）", "在开放式分层价值空间中定义价值相关性并设计抽取/生成任务评估价值理解"]
benchmarks: ["ValueBench"]
---

# 论文速读：ValueBench: Towards Comprehensively Evaluating Value Orientations and Understanding of Large Language Models

## 一句话总结
本文提出了 **ValueBench**，首个全面的心理学测量基准，用于评估大语言模型（LLMs）的价值取向（value orientations）与价值理解（value understanding）。通过整合44个成熟心理量表中的453个价值维度，设计了基于真实人机交互的评估流程，并在6款主流LLM上进行了系统评测。

## 研究问题与动机
- 现有工作多依赖李克特量表式自陈报告（如"你对该陈述的同意程度为1-5分"）评估LLM心理特质，但此类受控设置与真实人机交互场景脱节，且易触发模型拒绝回答。
- 已有基准（如ValueEval、PsychoBench）涵盖的心理量表数量有限（1-13个），价值维度稀疏（5-69个），缺乏对价值空间层次结构的建模。
- 价值理解评估受限于预定义价值空间狭窄、人工构建的ground truth质量不稳定，且忽视了价值间复杂的层级与关联结构。
- LLMs日益作为教育、医疗等领域的人机交互代理，其内嵌价值取向与理解能力直接影响用户决策，亟需可靠评估方法以支撑安全对齐。

## 核心贡献（创新点）
1. **首个综合心理学测量基准**：收集44个已发表心理量表、453个价值维度，覆盖人格、社会公理、认知系统、通用价值理论四大领域，远超既有工作规模（Table 1对比）。
2. **基于真实交互的价值取向评估流程**：将第一人称陈述改写为寻求建议的封闭问题，通过LLM生成自由回答并由GPT-4 Turbo评分，模拟真实咨询场景；与李克特式自陈报告相比具更高生态效度。
3. **开放式分层价值空间下的价值理解任务**：定义"价值相关性"概念（子尺度、同义、反义关系），设计对称/非对称提示的关联识别、项目-价值抽取、价值-项目生成三大任务，首次在开放层级空间中评估LLM价值理解。
4. **发现LLMs共享与差异化的价值取向模式**：所有模型在"Security""Benevolence""Self-Direction"等维度得分高，在"Power"得分低；不同模型在"Decisiveness""Hedonism"等维度呈现显著分歧。
5. **实证表明LLM可逼近心理学专家结论**：在充足上下文与对称提示下，GPT-4 Turbo识别相关价值的F1达85.7%，项目-价值抽取Hits@3达84.1%，证实LLM可作为大规模计算社会科学 annotator 的潜力。

## 方法详解
### 数据集构建
- **Item-Value Pair Extraction**：将各量表中的题目（items）转化为第一人称陈述，并与原表量表对应的目标价值配对，标注赞同标签（1=支持，-1=反对）。
- **Value Interpretation Extraction**：收集价值定义（如有）及对立价值对（如Indulgence vs. Restraint）。
- **Value Substructure Extraction**：提取层级关系（subscale value → value pair），如HEXACO-PI-R中"Social Self-Esteem"是"Extraversion"的子尺度。

### 价值取向评估流程（§4.1.1）
1. 用GPT-4 Turbo将第一人称陈述改写为寻求建议的Yes/No问题（保留原始立场）。
2. 向目标LLM提问，要求其生成≤50词的自由回答。
3. 将问题与回答输入GPT-4 Turbo评分器，按0-10分评估回答倾向"Yes"的程度。
4. 对反向题用公式 $score' = 10 - score$ 调整；最终价值得分 = 该价值下所有题目得分的均值。
5. 人工验证：随机抽取100对样本，GPT-4 Turbo与人类标注者一致性达80.0%。

### 价值理解评估任务（§4.3）
- **Identifying Relevant Values**：定义四类相关关系（子尺度、同义、反义），构造正负样本对；使用对称/非对称两种提示版本，要求LLM输出定义、关系解释、关系标签及相关性判定（0/1）。
- **Item-to-Value Extraction**：给定项目（≤100词），要求LLM输出Top-3最相关价值及解释；由GPT-4 Turbo评估与ground truth的相关性，计算Hits@1/2/3。
- **Value-to-Item Generation**：给定价值及其定义，要求LLM生成支持/反对该价值的论述；由GPT-4 Turbo从0-10分评估一致性（consistency）与信息量（informative）。

## 实验与结果
### 评测模型
GPT-3.5 Turbo、GPT-4 Turbo、Llama-2 7B、Llama-2 70B、Mistral 7B、Mixtral 8x7B（temperature=0，确定性输出）。

### 价值取向结果（§4.1.2）
- **一致性验证**：NFCC2000与NFCC1993虽题目不同但测同一维度，雷达图模式高度一致。
- **共性模式**：所有模型在PVQ-40的"Security""Benevolence""Self-Direction""Universalism"上得分高（≈10），"Power"得分低（≈1.3-4.0）；在SA量表上均鼓励"Social Complexity""Reward for Application"， discouraging "Fate Determinism""Social Cynicism"。
- **差异模式**：在"Decisiveness""Hedonism""Face Consciousness"等维度各模型表现分化明显（如Llama-2 70B决断力得分8.5，Mistral 7B仅5.5）。

### 价值理解结果（Table 2）
- **Identifying Relevant Values**：GPT-4 Turbo在对称提示下Recall=88.7%、Precision=82.9%、F1=85.7%；转为非对称提示后F1降至65.7%，说明层级关系不对称性对LLM造成挑战。
- **Item-to-Value Extraction**：GPT-4 Turbo Hits@3=84.1%，各模型性能差距小（约5%波动）。
- **Value-to-Item Generation**：GPT-4 Turbo一致性得分8.9/10，信息量得分5.5/10；Llama-2 70B一致性最高（8.9），GPT-4 Turbo信息量最高（5.5）。

### 关键发现
- 充足上下文（较长价值定义）可提升正样本召回率（Fig. 6）。
- LLM在面对提示词顺序变化时表现出不一致（reversal curse现象），对称提示更稳健。
- 不同模型在价值生成任务上各具优势，反映训练数据与算法偏好差异。

## 相关工作脉络
1. **PsychoBench（Huang et al., 2024）**：评测13个量表、69个人格特质；ValueBench扩展至44个量表、453个价值维度，覆盖更广且聚焦价值观而非仅人格。
2. **ValueEval（Kiesel et al., 2023）**：基于Schwartz单一体系（54个子维度）评估 argument-value 关联；ValueBench整合多源量表并建模价值间层级关系，突破单一理论框架限制。
3. **VUM（Zhang et al., 2023b）**：提出discriminator-critique gap评估价值理解；ValueBench进一步引入开放生成任务与层级结构感知评估。
4. **人格类工作（Li et al., 2022; Miotto et al., 2022; Safdari et al., 2023）**：多采用李克特自陈式问答；ValueBench批判其生态效度不足，提出真实交互式评估流程。
5. **Morality评估（Abdulhai et al., 2023; Scherrer et al., 2023）**：聚焦道德基础理论；ValueBench涵盖更广泛的价值类型（人格、社会公理、认知风格等）。
6. **Valuenet / ValueFulcra（Qiu et al., 2022; Yao et al., 2023）**：探索价值驱动对话；ValueBench提供标准化评测基准与跨模型比较框架。

## 局限性与未来方向
- **量表覆盖偏差**：部分维度与国家/文化状态描述关联较弱，可能存在隐含的西方文化中心主义偏差；标注员均为亚洲社会学硕士，多元文化视角有限。
- **改写引入噪声**：LLM改写题目及GPT-4评分环节可能引入系统偏差，虽经人工验证80%一致性，但未完全消除。
- **长度限制**：项目和生成内容限制在100词内，观点表达较直接，与现实复杂语境存在差距。
- **未深入探究差异成因**：不同模型在特定价值维度上的分歧原因复杂，需未来研究从训练数据、对齐策略等角度深入分析。

## 研究启发与可借鉴点
1. **评估流程设计**：将心理量表题目改写为寻求建议的开放问题再评估，有效规避了自陈报告的拒绝行为与生态效度缺陷，可迁移至其他心理特质评测。
2. **价值相关性定义**：提出子尺度/同义/反义三类关系的统一判定标准，为开放价值空间中的语义推理评估提供了可复用的任务定义框架。
3. **对称vs非对称提示对比**：发现LLM在层级关系不对称提示下性能显著下降，提示设计中应优先考虑对称表述以提升稳定性。
4. **跨学科基准构建思路**：整合心理学、社会学多源量表并建模价值层级，为AI与伦理/价值观交叉研究提供了数据构建范式。
5. **GPT-as-Evaluator验证**：通过人类标注对比验证LLM评分器可靠性（80%一致性），为后续使用GPT系列作为自动化评估器提供了方法论参考。

## 关键术语表
- **ValueBench**：首个全面评估LLM价值取向与价值理解的心理学测量基准，涵盖44个量表453个价值维度。
- **Value Orientations**：LLM通过生成内容所体现的价值观倾向，通过模拟真实人机交互进行评估。
- **Value Understanding**：LLM识别、提取和理解语言背后价值含义的能力，包括关系识别、抽取与生成任务。
- **Psychometric Inventory**：心理测量量表，如大五人格、MBTI、 Schwartz价值量表等标准化评估工具。
- **Item-Value Pair**：题目-价值对，指心理量表中描述行为/态度的题目与其对应目标价值的配对标注。
- **Value Substructure**：价值子结构，指价值间的层级关系（如子尺度值与主值）及相容/冲突关系。
- **Likert-scale Self-report**：李克特量表自陈报告，传统心理测量中要求被试对陈述按同意程度打分的方式。
- **Open-ended Value Space**：开放式价值空间，指不限定预定义价值集合、允许模型自由生成价值相关内容的评估设定。

## 可复现要素
- **数据集**：44个心理量表来源公开可查（附录A Table 3列明）；ValueBench数据已开源（GitHub: https://github.com/Value4AI/ValueBench）。
- **代码/权重**：论文未提及开源代码仓库，但基准数据已公开；评估使用GPT-4 Turbo作为评分器。
- **关键超参**：temperature=0（确定性解码）；回答长度限制≤50词；价值定义解释≤20词；所有评估任务prompt见附录B。
