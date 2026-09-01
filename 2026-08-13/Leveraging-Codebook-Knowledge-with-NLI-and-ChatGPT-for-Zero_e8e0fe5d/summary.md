---
title: "Leveraging-Codebook-Knowledge-with-NLI-and-ChatGPT-for-Zero"
source: https://aclanthology.org/2024.acl-long.35.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:43:57"
field: "低资源关系抽取"
keywords: ["零样本关系分类", "自然语言推理", "政治事件编码", "代码本知识", "ChatGPT", "树状查询框架", "事件模式"]
innovations: ["将代码本知识转化为层级化NLI假设框架实现零样本政治事件关系分类", "引入事件模式(P/F/CP/CF)作为细粒度分类的可计算语义维度", "三层树状查询框架在假设空间控制与分类精度间取得最优平衡"]
benchmarks: ["PLV（自建，CAMEO/PLOVER细粒度标注）", "A/W（ACE+WikiEvents跨域验证）"]
---

# 论文速读：Leveraging Codebook Knowledge with NLI and ChatGPT for Zero-Shot Political Relation Classification

## 一句话总结
本文提出ZSP模型，通过将政治事件标注代码本（Codebook）知识转化为层级化的自然语言推理（NLI）假设查询框架，实现了PLOVER事件本体的零样本细粒度关系分类；同时评估了ChatGPT（GPT-3.5/4）在该任务上的表现，ZSP以极少的标注成本在多项指标上超越传统字典方法和部分监督模型。

## 研究问题与动机
1. **事件编码的高标注成本困境**：政治暴力研究中的事件编码（从新闻文本提取Source-Action-Target三元组）高度依赖人工标注，构建千级别细粒度标注数据集通常耗时数月，难以支撑快速演化的本体体系。
2. **传统字典方法的僵化性**：基于静态动词模式的字典匹配方法（如Universal PETRARCH，81k条模式）召回率低、缺乏语义泛化能力，且难以处理细粒度标签间的微妙语义差异。
3. **监督学习方法的数据依赖与本体演化代价**：深度学习方法虽精度高，但严重依赖大规模标注数据；当本体（Ontology）演化时需重新标注，成本高昂。
4. **现有零样本方法的场景局限**：多数零样本工作聚焦句子级分类或依赖复杂模板，难以直接适配政治科学中实体对间细粒度关系分类的需求，且普遍忽略"事件模式"（Event Mode）这一关键语义维度。

## 核心贡献（创新点）
1. **提出ZSP（Zero-Shot fine-grained relation classification model for PLOVER ontology）**：一种基于NLI的零样本关系分类框架，将代码本中的标签描述和歧义消解规则转化为可计算的分层假设体系，本质区别在于首次将政治事件"事件模式"（过去/未来/矛盾/否定）系统性地融入NLI假设构建中。
2. **构建三层树状查询框架（Tree-Query Framework）**：将复杂的单一标签分类分解为"上下文→事件模式→类歧义消解"三个层次逐级过滤，与扁平查询（一次性比较所有假设）相比显著提升精度和推理效率，本质区别在于通过层次化假设空间控制降低NLI评分噪声。
3. **系统评估并揭示ChatGPT在细粒度政治事件关系分类中的能力边界**：GPT-4相较GPT-3.5在细粒度分类和类歧义消解理解上有明显提升，但仍存在格式不稳定、长提示信息遗忘等问题，为后续LLM事件编码研究提供参考基线。
4. **公开PLV数据集及可迁移的方法论**：构建了首个面向PLOVER本体的细粒度标注数据集（含Binary/Quadcode/Rootcode三级任务），并证明"从代码本到假设体系"的知识转化流程可迁移至法律、医疗等其他具备领域代码本的学科。

## 方法详解
**核心思想**：将NLI任务重新框架化为"从一组候选假设中选择最被前提蕴含的标签"，并通过代码本知识构造高质量假设集。

**1. 事件模式设计（Mode-Aware Hypotheses）**：
从PLOVER代码本的辅助模式（历史/未来/假设/否定）提炼出四类事件模式：
- **Past (P)**：历史性或正在进行的事件（如"<S> protested against <T>" → PROTEST 4）
- **Future (F)**：未来、言语性或假设性事件（如"<S> threatened to protest against <T>" → THREATEN 3）
- **Contradict_Past (CP)**：与过去事件的矛盾形式（如"<S> reduced protests against <T>" → YIELD 2）
- **Contradict_Future (CF)**：与未来事件的矛盾形式（如"<S> promised to reduce protests against <T>" → AGREE 1）

每种Rootcode的不同模式对应不同标签，由代码本直接映射（见Appendix Table 7），无需人工新定义。

**2. 类歧义消解规则（Class Disambiguation）**：
从代码本提取高频歧义消解规则，以**Conflict Override**为核心规则：
> 若Top候选中包含Material Conflict标签，则优先于Verbal Conflict标签，覆盖Verbal Conflict预测。

例如："protest to request" → 因同时蕴含PROTEST(M-Conf)和REQUEST(V-Conf)，Conflict Override规则选择PROTEST。

另设**Peace Override**（peace forces优先于forces）和**Consult Penalty**（CONSULT假设分数扣减c%，缓解过度泛化）。

**3. 三层树状查询框架**：
- **Level 1（Context）**：比较76个Past假设，选出Top-N候选（阈值：最高分-0.1）
- **Level 2（Mode）**：仅对Level 1候选查询其Future变体（如PROTEST→THREATEN），减少查询次数（从222降至约76+58次）
- **Level 3（Class Disambiguation）**：应用冲突覆盖等规则消除歧义

该框架使每个前提仅需约77次NLI推理（而非全部222个假设逐一比较）。

**4. ChatGPT方法**：
将代码本中16个Rootcode的描述性摘要压缩为Prompt，配合任务说明和候选标签列表，直接调用GPT-3.5/GPT-4 API进行零样本分类；为缓解输出格式不稳定问题，采用数字代码（01-15）替代文本标签输出。

**模型底座**：ZSP采用RoBERTa-Large-MNLI微调模型进行NLI推理。

## 实验与结果
**数据集**：
- **PLV**（主体数据集）：基于CAMEO代码本+CoPED细粒度标注扩展，1050训练/1033测试，含Binary/Quadcode/Rootcode三级任务
- **A/W**（跨域验证）：从ACE+WikiEvents映射构建，802训练/805测试，Binary任务

**评估指标**：Macro F1

**最强结果汇总**（Table 4）：

| 模型 | PLV Binary | PLV Quadcode | PLV Rootcode | A/W Binary | Avg. |
|------|-----------|-------------|-------------|-----------|------|
| ZSP（Tree l1,2,3）| **96.4** | **89.6** | **82.4** | **88.0** | **89.1** |
| GPT-4 | 93.4 | 76.7 | 61.5 | 87.0 | 79.7 |
| GPT-3.5 | 90.1 | 66.2 | 40.9 | 76.3 | 68.4 |
| UP（字典基线）| 80.8 | 51.8 | 46.3 | 67.2 | 61.5 |
| CBERT（全量监督）| 98.4 | 96.3 | 86.7 | 89.3 | 92.7 |

**关键结论**：
- ZSP以零样本方式超越GPT-4约9.4个百分点（Avg），远超GPT-3.5和UP基线
- ZSP在Binary和A/W任务上接近全量CBERT（仅差2.3-4.7个百分点）
- **数据效率优势**：ZSP仅需约25%-50%的CBERT训练数据即可达到相近性能（图3）；CBERT在Rootcode任务上仍需更多数据缩小差距（最大差距6.7%，Quadcode）
- **Tree-Query vs Flat-Query**（消融，Table 5）：ZSP Tree l1达85.8（Quad）/78.2（Root），远超Full Flat模型（73.4/55.7），证明层次化过滤远优于全量扁平比较
- GPT-3.5在Rootcode任务（40.9）上甚至低于UP（46.3），暴露其在细粒度任务上的不稳定性

## 相关工作脉络
1. **CAMEO/PLOVER事件本体体系**（Gerner et al., 2002; Open Event Data Alliance, 2018）：本文的任务定义基础，本文选择PLOVER而非CAMEO四位数代码的原因在于其语义更清晰、标签数更少（16个Rootcode vs 200+）。
2. **Universal PETRARCH**（Lu & Roy, 2017）：基于81k动词模式的字典匹配事件编码器，是本文的核心零样本基线；本文证明NLI驱动的语义方法在精确度和泛化性上全面超越。
3. **零样本关系分类的NLI范式**（Obamuyide & Vlachos, 2018; Yin et al., 2019; Sainz et al., 2021）：本文继承"将分类转化为文本蕴含"的思路，但创新性地引入事件模式和层次化查询以适配政治事件本体的特殊需求。
4. **政治事件抽取的监督学习方法**（Hu et al., 2022a ConfliBERT; Parolin et al., 2020-2022）：本文作为零样本方案的对照基线，强调在低标注资源场景下的替代价值。
5. **ChatGPT用于信息抽取**（Wei et al., 2023; Yuan et al., 2023; Li et al., 2023）：本文系统评估了GPT-3.5/4在政治关系分类上的能力，指出其在细粒度任务和长提示保持上的局限，为后续LLM事件编码研究提供参考。
6. **事件模式/情态识别**（Pyatkin et al., 2021; Rudinger et al., 2018）：NLP中的事实验证和情态识别工作与本文的事件模式设计存在概念交叉，但本文将其简化为任务导向的P/F/CP/CF四分类矩阵以便直接编码。

## 局限性与未来方向
1. **假设工程密集型**：ZSP的性能高度依赖代码本质量和假设构建，对于缺乏明确代码本或标签极其细碎（如ASSAULT的子类：crime/attack/kidnap）的任务，假设数量可能逼近关键词数量，失去可扩展性。
2. **GPT-3.5稳定性不足**：在细粒度分类、长提示信息保持、输出格式规范化方面存在明显缺陷；GPT-4虽有改善但仍需进一步优化提示设计。
3. **未探索多轮交互**：受时间和API成本限制，未研究通过多轮对话引导GPT提升精度的可行性。
4. **跨本体泛化待验证**：仅针对CAMEO/PLOVER进行验证，未在其他政治事件本体（如LEDa、UCDP GED）上测试迁移效果。
5. **未来方向**：探索ZSP/ChatGPT与Few-shot/In-Context Learning的混合方法；向法律、医疗、媒体分析等具备领域代码本的跨学科场景推广。

## 研究启发与可借鉴点
1. **"从代码本到假设体系"的知识转化范式**：本文系统化地将领域标注代码本中的标签描述和消解规则转化为NLI假设和查询规则，这种将领域专家知识"机器可计算化"的思路可直接迁移至其他具备Codebook的系统（如临床医学术语编码、法律条文分类）。
2. **树状分层查询框架的设计价值**：三层过滤（Context→Mode→Disambiguation）在控制假设空间规模的同时保留关键语义区分能力，该架构可复用于其他需要多级消解的分类任务。
3. **事件模式作为可迁移的语义维度**：将事件的时态/意图/否定状态显式建模为独立的分类维度（P/F/CP/CF），可有效缓解细粒度标签间的语义混淆，这一思路可推广至时序事件抽取、因果推理等任务。
4. **低标注成本与可解释性的平衡**：ZSP无需训练、仅需推理阶段的少量NLI查询即可完成细粒度分类，且每一层决策均可追溯至代码本规则，为"可解释AI+低资源NLP"提供实践案例。
5. **类歧义消解规则的模块化设计**：Conflict Override、Peace Override、Consult Penalty等规则以轻量插件形式嵌入框架，用户可根据具体任务需求增删，这种模块化消解机制值得在关系抽取、事件论证挖掘等任务中借鉴。

## 关键术语表
**Event Coding（事件编码）**：将非结构化新闻文本转化为结构化Source-Action-Target三元组的过程，是政治暴力研究中的核心数据预处理任务。
**PLOVER（Political Language Ontology for Verifiable Event Records）**：CAMEO的简化版本本体，将政治事件关系分为16个Rootcode和4个Quadcode层级。
**NLI（Natural Language Inference，自然语言推理）**：判断一段前提文本（Premise）是否蕴含（Entailment）、矛盾（Contradiction）或中立（Neutral）于给定假设（Hypothesis）的任务。
**Event Mode（事件模式）**：表示事件时间状态的语义维度，本文定义为Past/Future/Contradict_Past/Contradict_Future四类。
**Codebook（代码本）**：领域专家编写的事件分类标注指南，包含标签定义、示例和歧义消解规则，是ZSP假设来源的核心知识基础。
**Macro F1**：各类别F1分数的算术平均值，用于评估多类别不平衡数据的整体分类性能。
**Tree-Query vs Flat-Query**：Tree-Query分层逐步筛选候选假设；Flat-Query一次性比较全部假设，后者在本实验中因噪声过大导致性能下降。
**Conflict Override**：ZSP的核心消解规则之一，当Top候选同时包含Material Conflict和Verbal Conflict时优先选择前者。

## 可复现要素
- **数据集**：PLV（来自CAMEO代码本+CoPED扩展）和A/W（来自ACE+WikiEvents），论文附录D提供详细构建过程；代码公开链接见论文首页脚注（<sup>1</sup>）
- **代码**：论文声明代码已公开（The code is publicly available）
- **模型**：ZSP使用RoBERTa-Large-MNLI微调模型（论文脚注<sup>7</sup>）；ChatGPT使用OpenAI API（GPT-3.5和GPT-4）
- **关键超参**：NLI阈值（Level 1选Top-3，分数高于最高分-0.1）；Consult Penalty c=2%；训练使用单张V-100 GPU，5个随机种子取平均；具体超参论文未逐一列出（使用默认配置）
