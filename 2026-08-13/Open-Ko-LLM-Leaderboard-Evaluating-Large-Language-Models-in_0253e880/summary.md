---
title: "Open-Ko-LLM-Leaderboard-Evaluating-Large-Language-Models-in"
source: https://aclanthology.org/2024.acl-long.177.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:45:42"
field: "多语言大模型评测"
keywords: ["韩语LLM", "大语言模型评测", "基准构建", "数据污染", "私有测试集", "Open Ko-LLM Leaderboard"]
innovations: ["构建首个对齐英语标准的韩语LLM私有测试集排行榜", "新增Ko-CommonGen v2引入正交评估维度", "提出基于任务饱和度的基准动态扩展量化标准"]
benchmarks: ["Ko-H5", "Ko-ARC", "Ko-HellaSwag", "Ko-MMLU", "Ko-TruthfulQA", "Ko-CommonGen v2"]
---

# 论文速读：Open-Ko-LLM-Leaderboard-Evaluating-Large-Language-Models-in

## 一句话总结
本文提出了**Open Ko-LLM Leaderboard**和**Ko-H5 Benchmark**，为韩语大语言模型提供了一套与英语Open LLM Leaderboard对齐、采用私有测试集的综合性评估框架，填补了韩语LLM评测领域的空白。

## 研究问题与动机
1. **语言偏见问题**：现有LLM评测基准主要集中于英语（如Open LLM Leaderboard），韩语等非英语语言的评测资源严重匮乏。
2. **数据污染风险**：公开测试集容易与训练数据重叠，导致模型在基准上虚高，无法真实反映泛化能力。
3. **韩语独特性挑战**：韩语具有独特的句法和语义结构，直接翻译英语基准无法满足文化语境和本土化需求。
4. **缺乏动态评估机制**：静态基准容易饱和，需要探索何时扩展评测任务以保持区分度。

## 核心贡献（创新点）
1. **首个韩语LLM综合排行榜**：构建了Open Ko-LLM Leaderboard，结构对齐Hugging Face Open LLM Leaderboard，降低社区迁移成本。
2. **私有测试集防污染机制**：通过严格去重（相似度阈值0.05）验证，测试集与热门训练数据重叠率均低于1%，有效规避数据泄露。
3. **新增Ko-CommonGen v2任务轴**：新增的生成式常识推理任务与原有五项基准形成低相关（尤其与Ko-TruthfulQA），为韩语LLM评估引入正交维度。
4. **系统性的多维实证分析**：提供任务间相关性、按模型规模/类型的时间序列分析、单任务饱和度统计，为基准演进提供量化依据。
5. **社区治理与透明度实践**：发布提交问题统计（如62.3%存在Model Card问题），呼吁社区协作维护排行榜完整性。

## 方法详解
**Ko-H5基准构建流程**：
- **数据源**：四个数据集源自英语Open LLM Leaderboard（Ko-ARC, Ko-HellaSwag, Ko-MMLU, Ko-TruthfulQA），一个全新构建（Ko-CommonGen v2）。
- **翻译流水线**：使用GPT-4进行批量机器翻译 → 规则检查（基于Costa-jussà et al., 2022）检测基础错误 → 35位多背景专业译者人工校对（文化对齐）→ 过滤需领域知识的数据 → 领域专家重译 → 再次规则检查。
- **私有测试集策略**：测试集不公开，仅 leaderboard维护方持有，评估时由参赛者提交结果。

**去重验证方法**：
- 对训练集和测试集分别进行精确去重和MinHash去重（相似度阈值0.05，n-gram=20，最小长度依数据集为10-30）。
- 将训练集与测试集配对后再次MinHash去重，统计被移除比例（见表2）。

**基准组成**（表1）：
- Ko-ARC: 1.1K样本，CC-BY-SA
- Ko-HellaSwag: 10.0K样本，MIT
- Ko-MMLU: 14.0K样本，CC-BY-SA
- Ko-TruthfulQA: 0.8K样本，Apache 2.0
- Ko-CommonGen v2: 0.8K样本，Apache 2.0

## 实验与结果
**数据集**：
- Ko-H5 Benchmark（5个子任务）
- 热门训练数据：KoUltrafeedback, KoOpenOrcaPlatypus, KoAlpaca

**数据泄露分析结果**（表2）：
- 所有重叠率均<1%，最高为Ko-MMLU与KoUltrafeedback的0.92%
- 证明私有测试集有效防止数据污染

**任务相关性分析**（图2-3）：
- Ko-ARC, Ko-HellaSwag, Ko-MMLU三者高度相关（同一评估轴）
- Ko-TruthfulQA与其他任务相关性低（独立评估轴）
- Ko-CommonGen v2呈中等相关性（引入第三个评估轴）
- 小模型（0-3B）中Ko-TruthfulQA和Ko-CommonGen v2与其他任务呈负相关；3-7B和7-14B区间转为正相关

**时间序列分析**（图4-7）：
- 性能呈阶梯式跃升，与全球LLM突破同步
- 0-3B模型显著落后于更大规模模型，存在"临界模型规模"
- Instruction-tuned模型性能变化紧随pretrained模型（延迟约0-2周高度相关）
- 任务饱和度差异：Ko-CommonGen v2约2周达60分，Ko-HellaSwag约6周，Ko-TruthfulQA约13周，Ko-ARC/Ko-MMLU尚未达60分

**最强模型表现**：
- 7-14B参数区间模型领先
- 截至2024年2月15日，Ko-MMLU和Ko-ARC仍未达到60分阈值

## 相关工作脉络
1. **Open LLM Leaderboard**（Beeching et al., 2023）：英语LLM评测基准，本文韩语版本的直接对标对象，结构完全对齐。
2. **H4基准数据泄露研究**（Deng et al., 2023）：指出公开基准的数据污染问题，本文采用私有测试集解决此问题。
3. **数据污染评估**（Sainz et al., 2023; Zhou et al., 2023; Balloccu et al., 2024）：强调为每个基准测量数据污染的重要性，本文实验提供了韩语场景的实证支持。
4. **CommonGen**（Lin et al., 2019）：英语常识生成基准，本文Ko-CommonGen v2的灵感来源。
5. **韩语NLP资源现状**（Park et al., 2020; Magueresse et al., 2020; Ranathunga et al., 2023）：指出韩语等低资源语言的评估资源匮乏，本文填补这一空白。
6. **其他语言排行榜**：BigCode Models Leaderboard、Open ASR Leaderboard等，本文为韩语LLM填补类似空白。

## 局限性与未来方向
1. **基准静态性**：Ko-H5大部分任务继承自英语版本，存在性能饱和风险，需持续演进。
2. **模型规模限制**：当前排行榜限制提交模型≤30B参数，无法评估更大规模模型。
3. **时间分析有限**：运行仅四个月，长期趋势待观察。
4. **Ko-HellaSwag人工审查不足**：因成本过高（估算72万美元）未进行人工校对，质量可能受影响。
5. **未来方向**：新增Ko-GSM8k、Ko-Winogrande、Ko-EQ Bench、Ko-GPQA等任务；建立任务饱和度监测机制动态扩展基准。

## 研究启发与可借鉴点
1. **私有测试集设计范式**：对于非英语/低资源语言的benchmark构建，私有测试集是保障评估可靠性的关键策略，可迁移至其他语言场景。
2. **多维度相关性分析框架**：通过任务间相关性热力图识别评估冗余度和正交性，指导基准任务选择和扩展优先级。
3. **饱和度量化指标**：提出"达到60分所需周数"作为基准扩展决策依据，为动态benchmark维护提供可操作标准。
4. **跨模型类型时间序列关联**：揭示pretrained→instruction-tuned的性能传导时滞（0-2周），可用于指导微调时机和资源分配。
5. **社区治理数据驱动**：发布提交问题统计表（Model Card缺失率62.3%等），以透明数据推动社区规范，值得其他排行榜借鉴。

## 关键术语表
**Open Ko-LLM Leaderboard**：Upstage AI开发的韩语LLM开源排行榜，平台界面与Hugging Face Open LLM Leaderboard一致。
**Ko-H5 Benchmark**：包含五个子任务的韩语LLM评测基准，命名为"H5"对应英语"H4"基准。
**私有测试集（Private Test Sets）**：不对公众开放的测试数据集，用于防止数据污染，确保评估真实性。
**MinHash去重**：基于局部敏感哈希的快速近似去重算法，用于检测训练集与测试集间的文本相似度。
**Ko-CommonGen v2**：从零构建的韩语常识生成任务，评估模型基于给定概念生成合理句子的能力。
**临界模型规模（Critical Model Size）**：性能提升出现加速拐点的模型参数量级，本文发现3B以上模型提升更显著。

## 可复现要素
- **数据集**：Ko-H5各子任务测试集为私有，未公开；训练数据对比使用KoUltrafeedback、KoOpenOrcaPlatypus、KoAlpaca
- **代码**：论文未提供开源代码
- **权重**：论文未提及开源权重
- **关键超参**：MinHash去重参数——相似度阈值0.05，n-gram大小20，最小长度分别为30(Ko-ARC)、20(Ko-HellaSwag)、10(Ko-MMLU)、10(Ko-TruthfulQA)、30(Ko-CommonGen v2)
