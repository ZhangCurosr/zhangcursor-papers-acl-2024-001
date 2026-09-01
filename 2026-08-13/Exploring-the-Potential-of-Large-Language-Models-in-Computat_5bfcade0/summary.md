---
title: "Exploring-the-Potential-of-Large-Language-Models-in-Computat"
source: https://aclanthology.org/2024.acl-long.126.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 07:09:25"
field: "计算论证与大模型评估"
keywords: ["计算论证", "大语言模型", "论点挖掘", "论点生成", "零样本评估", "反诘生成", "Benchmark"]
innovations: ["首次系统评估LLM在计算论证14个数据集上的零/少样本性能", "提出端到端counter speech生成新基准及三种生成范式", "揭示ROUGE与BERTScore在论证生成评估中的系统性偏差"]
benchmarks: ["IAM Claims/Evidence/Stance", "IBM Claims/Evidence/Stance/Type", "FEVER", "CounterArguGen", "ConcluGen", "DebateSum"]
---

# 论文速读：Exploring the Potential of Large Language Models in Computational Argumentation

## 一句话总结
本文首次系统性地评估了 ChatGPT、Flan 系列和 LLaMA2 等大语言模型在计算论证（argumentation）领域的零样本/少样本表现，涵盖论点挖掘、论点生成两大类共14个数据集，并提出了一个新的端到端"反诘 speech 生成"基准任务。

## 研究问题与动机
- **核心问题**：计算论证（包括论点挖掘与论点生成）长期依赖大量标注数据微调模型，但标注成本极高；LLM 是否能在零样本/少样本设定下直接胜任这些任务？
- **现有方法的不足**：
  - 已有综述仅覆盖论点挖掘（如 Peldszus & Stede, 2013），缺乏对"论点挖掘+论点生成"统一视角的系统评估
  - 现有数据集要么只关注挖掘（理解），要么只关注生成（表达），缺少端到端整合评估
  - 论证任务需要语篇级理解（discourse-level），传统 NLP 标注数据规模难以支撑，制约了领域发展
- **研究缺口**：LLM 在低资源设定下的计算论证能力尚不明确，尤其缺乏多模型、多任务、多数据集的统一基准

## 核心贡献（创新点）
1. **首次系统评估 LLM 在计算论证任务上的零/少样本性能**：覆盖 6 类任务、14 个公开数据集，建立标准化格式与评测协议
2. **提出全新的 counter speech generation 端到端基准**：将论点挖掘（识别 supporting speech 中的主张）与论点生成（撰写反驳 speech）整合为文档到文档的任务
3. **设计三种反诘生成范式**：claims 流水线、summary 流水线、端到端单步，揭示了中间步骤导致信息丢失的问题
4. **揭示 ROUGE vs BERTScore 的评估偏差**：论证生成任务中 LLM 语义保留良好（BERTScore 高）但字面重合低（ROUGE 低），指出传统指标可能系统性低估 LLM 潜力
5. **发现模型选择非单调规律**：Flan-T5-XL（3B）在多分类任务上与 Flan-UL2（20B）表现相当，较大模型未必更优

## 方法详解
- **Prompt 设计**：
  - 论点挖掘任务：采用标准化 prompt 模板，包含任务定义 + 输出格式约束（如 `Label: claim/non_claim`），限制模型输出到预定义标签空间
  - 论点生成任务：采用 ChatGPT 官方建议的自由格式 prompt，不约束标签空间，侧重语义生成
- **三种反诘生成方法（Figure 2）**：
  - **Pipeline (Claims)**：先用 Flan-T5-XXL 逐句检测主张（claim detection），再用 GPT-3.5-Turbo 对每个检测到的主张逐一生成反驳——两步串行
  - **Pipeline (Summary)**：先用 GPT-3.5-Turbo 对支持 speech 做摘要，再用 GPT-3.5-Turbo 基于摘要生成反诘 speech——两步串行
  - **End-to-end**：直接让 GPT-3.5-Turbo 接收完整支持 speech，一步生成反诘 speech——单次推理，无中间步骤
- **评估指标**：
  - 论点挖掘：Accuracy、F1（区分二分类与多分类任务）
  - 论点生成：ROUGE-1/2/L、METEOR、BERTScore，辅以人工评估（流畅度 Flu.、说服力 Per.、论点覆盖率 % Arg.）
- **统计检验**：对零样本结果进行与随机基线的显著性检验（p-value，α=0.05）

## 实验与结果
- **数据集**：14 个公开数据集，每个采样 500 条测试样本；新基准 Counter Speech 采样 250 对辩论 speech
- **模型**：ChatGPT (GPT-3.5-Turbo)、Flan-T5-XL(3B)、Flan-T5-XXL(11B)、Flan-UL2(20B)、Llama-2-7B、Llama-2-13B
- **论点挖掘关键结果**（Table 1，零样本）：
  - GPT-3.5-Turbo 在绝大多数数据集上显著优于随机基线；在 IBM Claims 上 Acc=72.00、F1=72.19
  - Flan-UL2 在简单二分类任务（Claim Detection、Evidence Detection）上优于 GPT-3.5-Turbo，但在多分类任务（MTSD、FEVER、AQE Type）上明显落后
  - Llama-2-13B 多数任务低于随机基线，尤其在 FEVER stance (Acc=0.40) 和证据分类上严重失效
- **少样本提升**（Figure 3）：少于 10 个示例即可让 LLM 在多数任务上追平使用全部500条训练样本微调的 PLM
- **论点生成关键结果**（Table 2）：
  - GPT-3.5-Turbo 在 CounterArguGen（Premises 设置）上 BERTScore=83.50，超越 SOTA（82.60）
  - ConcluGen base 设置：GPT-3.5-Turbo k=5 时 BERTScore=87.19，接近 SOTA（89.32）
  - DebateSum 抽取式摘要：GPT-3.5-Turbo k=5 时 BERTScore=89.43，超越 SOTA（85.90）；且 ROUGE-1=54.99 vs SOTA 59.06，呈现"语义好但字面低"模式
- **反诘生成关键结果**（Table 3 & 4）：
  - 自动评测：End-to-end BERTScore=82.51 高于 Pipeline(Claims) 80.33
  - 人工评测（50样本，2位专业评委）：End-to-end 在 Flu.=4.32 vs 3.56、Per.=3.8 vs 2.8、% Arg.=95% vs 78% 三项全面领先
  - Pipeline 方法因中间步骤信息损失，未能覆盖全部主张

## 相关工作脉络
1. **Peldszus & Stede (2013)**：论点挖掘领域早期综述，仅覆盖挖掘方向，本文补充了生成方向的系统评估
2. **Cheng et al. (2022) IAM 数据集**：大规模综合论点挖掘数据集，本文在其基础上标准化格式并扩展至生成任务
3. **Alshomary et al. (2021) CounterArguGen**：基于弱前提生成反驳论点，本文的 counter speech 任务更强调长文档端到端生成
4. **Syed et al. (2021) ConcluGen**：论点摘要生成，本文验证 LLM 在零样本下无需额外人工标注（topic/targets/aspects）即可达到相近效果
5. **Roush & Balaji (2020) DebateSum**：基于单词级分类的抽取式摘要，本文指出 LLM 可避免其训练成本高和句子连贯性差的问题
6. **Wachsmuth et al. (2018), Bondarenko et al. (2020)**：早期反诘论点检索工作，不涉及生成；本文是首个文档到文档的反诘生成基准

## 局限性与未来方向
- **任务覆盖有限**：仅涵盖论点挖掘与生成两大类，未涉及论点质量评估、说服力建模等其他计算论证研究方向
- **人工评估规模受限**：仅对反诘生成做了50样本人工评估，其他生成任务因规模大未做全面人工验证
- **数据集采样偏差**：每个数据集仅随机采样500条，可能无法充分反映模型在完整数据分布上的表现
- **未来方向**：可使用 GPT-4 作为自动评估器替代部分人工评估；可探索 domain-specific 大型论证模型的设计

## 研究启发与可借鉴点
1. **ROUGE vs BERTScore 的评估偏差警示**：对于生成质量评估，单一 ROUGE 可能严重低估 LLM 的语义保留能力，建议结合语义相似度指标（BERTScore、METEOR）进行多角度的生成评估
2. **流水线 vs 端到端的教训**：中间步骤（claims detection/summary）虽提升了可控性，但引入了信息损失，端到端方法在 persuasiveness 和论点覆盖率上显著更优——提示在复杂生成任务中应优先尝试一步完成
3. **少样本高效学习**：小于10个示例即可让 LLM 追平全量微调的 PLM，这对标注成本极高的论证领域具有强实践价值
4. **模型规模不单调相关性能**：Flan-T5-XL (3B) 在多分类任务上与 20B 的 Flan-UL2 表现相当，提示低资源场景下可选用更小模型获得性价比最优解
5. **标准化 prompt 模板的可迁移性**：本文的 claim/stance prompt 模板设计（任务定义+输出格式约束）可直接迁移至其他结构化 NLP 任务的 LLM 评估

## 关键术语表
- **Computational Argumentation（计算论证）**：利用计算方法处理论证任务的 NLP 分支，核心包括论点挖掘与论点生成
- **Argument Mining（论点挖掘）**：从无结构文本中自动识别论点组件（主张、证据）及其关系（支持/反驳）
- **Argument Generation（论点生成）**：基于外部知识或给定立场自动生成有说服力的论证文本
- **Counter Speech（反诘 speech）**：针对支持性辩论演讲生成的反驳性完整演讲，需同时具备论点理解和生成能力
- **BERTScore**：基于 BERT 语境嵌入的文本相似度度量，比 ROUGE 更能反映语义一致性
- **Flan 系列**：Google 的指令微调语言模型家族，包括 Flan-T5 和 Flan-UL2，在 NLP 任务上表现优异
- **Discourse-level Comprehension（语篇级理解）**：论证任务所需的超越句级的全局理解能力，区别于实体识别等词元级任务

## 可复现要素
- **数据集**：14 个公开数据集（IAM Claims/Evidence/Stance, IBM Claims/Evidence/Stance/Type, FEVER, MTSD, AQE Type, CounterArguGen, ConcluGen, DebateSum），均已知开源；新基准 Counter Speech 基于 Lavee et al. (2019) 辩论数据库构建
- **代码**：论文未提供开源代码库（论文未提及）
- **关键超参**：采样 500 条测试样本/数据集；Counter Speech 采样 250 对；少样本 k∈{0,1,3,5}；所有 few-shot 结果取 3 次随机 seed 的平均值；DebateSum T5-base 训练：AdamW optimizer, lr=1e-4, batch_size=4, epochs=3
- **模型**：GPT-3.5-Turbo（闭源）、Flan-T5-XL/XXL、Flan-UL2、Llama-2-7B/13B（开源权重可用）
