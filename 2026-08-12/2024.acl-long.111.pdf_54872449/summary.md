---
title: "ValueBench: Towards Comprehensively Evaluating Value Orientations and Understanding of Large Language Models"
source: https://aclanthology.org/2024.acl-long.111.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:59:11"
field: "大语言模型评估与对齐"
keywords: ["value alignment", "LLM evaluation", "psychometric benchmark", "value understanding", "human-AI interaction", "value orientation"]
innovations: ["首个大规模心理测量学价值基准，整合44个量表453个价值维度与层级结构", "基于真实咨询场景的价值取向评估流程，有效规避Likert自陈测试的拒答与场景脱节问题"]
benchmarks: ["ValueBench"]
---

# 论文速读：ValueBench: Towards Comprehensively Evaluating Value Orientations and Understanding of Large Language Models

## 一句话总结
本文提出了**ValueBench**，首个全面评估大语言模型（LLM）价值取向（value orientations）与价值理解（value understanding）的心理测量学基准。该基准从44个已建立的心理量表系统收集453个多维度价值维度，并设计了基于真实人机交互场景的评估流程与开放式价值空间中的新颖任务，揭示了当前主流LLM在价值观上的共性与差异。

---

## 研究问题与动机

1. **LLM价值对齐的紧迫性**：LLM正被广泛集成于教育、医疗等关键领域作为人类代理，但其内化价值观可能与人类社会规范不一致，亟需可靠评估以确保负责任部署。
2. **现有评估方法的场景脱节**：已有工作多采用Likert量表式自陈测试（如"你对以下陈述的认同程度为1-5分"），此类受控设定与真实人机交互（用户寻求建议）差异显著，且易触发instruction-tuned模型的拒答倾向。
3. **价值理解评估的结构性缺陷**：现有方法受限于预定义价值空间过窄（如仅基于Schwartz理论54个价值）、ground truth依赖启发式生成，且忽视价值空间中复杂的层级与关联结构。

---

## 核心贡献（创新点）

1. **首个大规模多层次心理测量学价值基准**：从44个跨领域心理量表（人格、社会公理、认知系统、一般价值理论）系统收集453个价值维度及层级结构，规模远超PsychoBench（13个问卷、69个特质）与ValueEval（单一理论框架下的54个价值）。
2. **基于真实咨询场景的价值取向评估流程**：将首人称陈述改写为寻求建议的封闭式问题，模拟真实人机互动；以GPT-4 Turbo作为评估器对LLM自由回答进行0-10分评分，规避传统Likert测试的机械性与拒答问题。
3. **开放层次化价值空间中的价值理解任务体系**：设计相关价值识别（含对称/非对称提示）、Item-to-Value提取、Value-to-Item生成三类任务，首次在保留价值子结构（subscale value）关系的前提下评估LLM的理解与生成能力。
4. **专家标注的高质量配对数据集**：通过心理学与社会科学背景志愿者构建item-value配对、价值定义及对立关系标注，确保评估内容符合心理测量学效度标准。

---

## 方法详解

### 数据集构建（§3.2）
- **Item-Value Pair提取**：将各量表中题目统一转化为第一人称观点陈述（如将多选题选项改写为完整语句），并与原量表标注的目标价值配对，同时记录赞同（1）/反对（-1）标签。
- **Value Interpretation提取**：收集各价值的定义（形容词或名词短语形式），并记录对立项（如"Indulgence"与"Restraint"）。
- **Value Substructure提取**：保留量表内的层级关系，构建（子尺度价值, 主价值）配对，例如"Social Self-Esteem"→"Extraversion"。

### 价值取向评估流程（§4.1.1）
1. **题目改写**：用GPT-4 Turbo将首人称陈述改写为seeking-advice的封闭式问题，保持原立场不变（Yes=原陈述倾向）。
2. **LLM回答采集**：将改写后问题输入待测LLM，限制回答≤50词，获取自由形式输出。
3. **评分**：用GPT-4 Turbo作为evaluator，根据回答对原始问题偏向"Yes"或"No"的程度打分（0-10）。若原item反对目标价值，则得分调整为(10 – score)。
4. **聚合**：对同一价值的多个item得分取平均，得到该价值的取向分数。
5. **人工验证**：随机抽取100对结果，与人类标注员对比，一致性达**80.0%**。

### 价值理解评估任务（§4.3）
- **相关价值识别**：要求LLM对给定价值对输出定义、关系解释、关系标签及最终相关性判断（1/0）。定义相关性为四种关系之一：子尺度、同义、反义。分别测试对称提示（"One can be used as a subscale value of another"）与非对称提示（明确区分A是B的子尺度 / B是A的子尺度）。评估指标：Recall、Precision、F1。
- **Item-to-Value提取**：要求LLM从给定item中提取top-3最相关价值，由GPT-4 Turbo判定是否与ground-truth相关，计算Hits@1/2/3。
- **Value-to-Item生成**：给定价值及其定义，要求LLM生成支持/反对该价值的论据；由GPT-4 Turbo分别评分**一致性**（consistency）与**信息量**（informative），均从0到10分。

---

## 实验与结果

### 实验设置
- **评估模型**：GPT-3.5 Turbo、GPT-4 Turbo、Llama-2 7B、Llama-2 70B、Mistral 7B、Mixtral 8x7B（涵盖闭源/开源、有/无RLHF系列）
- **temperature=0** 或 greedy decoding，所有结果为确定性输出。

### 价值取向主要发现（§4.1.2）
- **共性取向**：所有模型在PVQ40中均高"Security"(≥8.0)、"Benevolence"(≥9.0)、"Self-Direction"(≥9.5)、"Universalism"(≥9.17)，低"Power"(≤4.0)；在SA中均鼓励"Social Complexity"(≥7.12)与"Reward for Application"(≥7.53)， discouraging"Fate Determinism"(≤4.44)与"Social Cynicism"(≤3.95)。
- **离散取向**：在"Decisiveness"、"Hedonism"、"Face Consciousness"等维度上不同模型差异显著（如Llama-2 70B在Decisiveness上得8.5，而Mistral 7B仅5.5）。
- **跨量表一致性**：NFCC1993与NFCC2000虽题目不同但测量相同五维价值，雷达图模式高度相似；"Discomfort with Ambiguity"与"Uncertainty Avoidance"在所有模型上均得低分（≤5.0）。

### 价值理解主要结果（Table 2，§4.3）
- **相关价值识别**：GPT-4 Turbo在对称提示下Recall **88.7%**、Precision **82.9%**、F1 **85.7%**，与专家结论一致性超80%；转换为非对称提示后F1骤降至65.7%，揭示LLM对提示顺序的高度敏感。
- **Item-to-Value提取**：所有模型Hits@3集中在**82.7%-84.8%**区间，模型间差异仅约5%，表明该任务难度对所有模型相对均衡。
- **Value-to-Item生成**：GPT-4 Turbo在Informative评分上最高（**5.5/10**），Llama-2 70B在Consistent评分上最佳（**9.4/10**）；Llama-2 7B因内部策略对部分价值拒绝生成，相应case被排除。

---

## 相关工作脉络

1. **心理测量学评估LLM的先驱工作**（Li et al., 2022; Miotto et al., 2022; Song et al., 2023等）：多采用单一量表（Big Five、MBTI等）的原始自陈格式进行多选问答，仅覆盖有限人格维度，且未考虑真实交互场景。
2. **PsychoBench**（Huang et al., 2024）：覆盖13个问卷、69个人格特质，是此前最全面的心理特质基准，但聚焦于人格而非价值观，且仍沿用Likert自陈范式。
3. **ValueEval**（Kiesel et al., 2023）：基于Schwartz理论54个价值评估LLM在论点中的价值识别，但价值空间为扁平化预设集合，未利用层级结构。
4. **VUM框架**（Zhang et al., 2023b）：提出双层次价值理解评估，但ground truth依赖启发式生成，且未系统建模价值间的关联关系。
5. **定位差异**：ValueBench在规模（44个问卷/453个价值）、评估场景真实性（咨询式交互vs自陈测试）、价值空间开放性（层级关联vs固定集合）三方面实现显著超越。

---

## 局限性与未来方向

1. **文化偏见风险**：数据采集自西方主流心理量表，可能隐含特定地域/文化视角，亚洲背景的标注志愿者过滤可能存在文化解释偏差（§7 Ethics Statement）。
2. **改写与评估的引入噪声**：通过LLM改写题目并用GPT-4 Turbo评估可能引入系统性偏差，原始题目的心理测量学效度在转化过程中无法完全保证（§6）。
3. **语境长度限制**：实验中item与生成文本均控制在100词以内，与真实世界复杂语境存在差距，可能低估LLM在长文本中的价值理解能力。
4. **差异成因缺乏深入分析**：不同模型在特定价值维度上的分歧可能源于训练数据偏好或对齐策略差异，但未作进一步归因研究。
5. **未来方向**：可扩展至跨文化价值体系、探索动态价值演化追踪、深化LLM价值取向与下游任务表现的关联性研究。

---

## 研究启发与可借鉴点

1. **从自陈测试到咨询场景的范式转换**：将Likert式判断题改写为seeking-advice问题，有效规避instruction-tuned模型的拒答倾向，此思路可直接迁移至其他态度/价值观评估任务。
2. **层次化价值空间的建模方式**：通过subscale value关系构建价值网络而非扁平集合，为知识图谱构建、多属性决策等任务提供了结构化表示的新思路。
3. **专家+众包混合标注策略**：使用社会学硕士背景志愿者进行负样本筛选，兼顾专业判断与成本效率，可为跨学科标注工作提供参考模式。
4. **Prompt敏感性作为诊断工具**：对称/非对称提示下的性能差异揭示了LLM的内在不一致性（如Berglund et al.提出的reversal curse），提示设计应作为模型评估的常规诊断维度。

---

## 关键术语表

**ValueBench**：首个综合评估LLM价值取向与价值理解的心理测量学基准，整合44个量表、453个价值维度及层级结构，已于GitHub开源。

**Psychometric Inventory**：心理测量量表，用于量化个体心理特质的标准化测试工具（如Big Five、MBTI、Schwartz价值量表等）。

**Item-Value Pair**：题目-价值配对，将心理量表中具体陈述题与其对应的目标价值维度建立关联的标注数据，含赞同/反对标签。

**Value Substructure**：价值子结构，指价值间的层级从属关系（如子尺度价值测量主尺度的特定面向），体现价值空间的网状关联特性。

**Likert-scale Self-report Testing**：李克特式自陈测试，受试者对陈述句按同意-反对等级评分的传统心理测量方式，本文指出其不适用于LLM真实场景评估。

**Relevant Value Identification**：相关价值识别任务，判断两个给定价值是否存在子尺度、同义或反义关系，评估LLM对价值空间结构的理解。

**Open-ended Value Space**：开放式价值空间，区别于固定预定义类别的封闭分类，允许模型在连续、关联的价值维度上进行语义理解与内容生成。

---

## 可复现要素

- **数据集**：ValueBench已开源，访问 https://github.com/Value4AI/ValueBench
- **代码**：论文声明开源，具体实现见上述仓库
- **评估模型**：GPT-3.5 Turbo、GPT-4 Turbo、Llama-2 7B、Llama-2 70B、Mistral 7B、Mixtral 8x7B
- **关键超参**：temperature=0 或 greedy decoding（确定性输出）；回答长度限制≤50词（取向评估）/ 100词（理解评估）
- **评估器**：统一使用GPT-4 Turbo作为评分器
- **题目改写器**：使用GPT-4 Turbo执行首人称→咨询问题的改写
- **Prompt详情**：全部可见Appendix B（Prompt 1-9）
- **完整实验结果**：见Appendix C（Table 4及Figures 7-11）
