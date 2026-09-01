---
title: "Self-Alignment-for-Factuality-Mitigating-Hallucinations-in-L"
source: https://aclanthology.org/2024.acl-long.107.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:51:06"
field: "大语言模型事实性与对齐"
keywords: ["hallucination mitigation", "self-alignment", "factuality", "DPO", "confidence calibration", "LLM evaluation"]
innovations: ["将LLM自评估作为事实性偏好信号的自对齐框架", "SK-TUNING校准训练提升内源置信度估计与校准", "以DPO直接在偏好对上对齐事实性"]
benchmarks: ["TruthfulQA", "BioGEN"]
---

# 论文速读：Self-Alignment for Factuality: Mitigating Hallucinations in LLMs via Self-Evaluation

## 一句话总结
本文提出一种利用大语言模型（LLM）自我评估能力来缓解幻觉的自对齐框架：通过组件 SELF-EVAL 让模型基于内部知识评估自身生成回答的事实准确性，再经自训练校准模块 SK-TUNING 提升置信度估计质量，最后以打分构造偏好样本并用 DPO 微调基座模型，在 TruthfulQA 与 BioGEN 三个关键任务上显著提升 LLAMA 系列模型的事实准确性。

## 研究问题与动机
- 核心问题：LLM 在“知道相关知识”的情况下仍会生成看似合理但事实错误的表述（即“knows-tells gap”类型的幻觉），损害可信性与高风险场景可用性。
- 已有方法不足一：基于领域特定人工标注表示编辑/干预的方法（如 ITI、DOLA）泛化受限，且依赖标注成本较高。
- 已有方法不足二：基于一致性置信度的事实信号（如 FACTTUNE-MC）依赖模型生成能力，难以稳定反映内部知识是否具备。
- 已有方法不足三：纯 SFT 对高质标注数据敏感，且仅靠 MLE 目标难以准确刻画“事实性”，容易陷入事实正确性与信息量之间的失衡。
- 动机：LLM 在“评估/判别自身输出是否正确”方面具有潜力，直接让其利用内部知识给出事实性置信度，可能比一致性采样更能反映真实知识掌握程度；若能将该置信度作为偏好学习信号并配合校准训练，有望在不依赖领域标注的前提下改善事实在生成中的可靠性。

## 核心贡献（创新点）
- 提出 Self-Alignment for Factuality 的三步自对齐流程，将“自我事实评估”作为偏好信号的来源，区别于需要人工标注或外部奖励模型的 RLHF/SFT 路线。
- 设计 SELF-EVAL 组件，把生成回答（或原子主张）转化为自问自答式 True/False 评估，直接利用模型内部知识得到 p(True|q,a) 作为事实性分数。
- 引入 SK-TUNING 进行自我知识校准训练，通过在异构知识型数据上构造成对正负预测样本优化置信度估计与校准，显著缓解朴素 SELF-EVAL-P(TRUE) 的过度自信问题。
- 将事实性分数用于构造偏好对并以 DPO 直接对齐策略，形成“评估—排序—偏好微调”的闭环，并在 MCQA、短文本与长文本生成三种设定下验证有效性。
- 实验表明该方法在不用领域标注的前提下，整体优于一致性置信度方法与表示编辑方法，并在 True*Info、FActScore 等复合指标上取得最优结果。

## 方法详解
- 流程三阶段（Figure 2）：
  1) 初始响应生成：对提示 x 用 few-shot 从基座策略 π_ref 采样 M 个候选回答 {y_m}；MCQA 任务无需此步，答案选项由数据集提供。
  2) 事实性自我评估与偏好标注：长文任务先用 GPT-3.5-turbo 抽取原子主张并转为对应原子问题，再以 SELF-EVAL 对每个主张 c 计算 p(True|q,c)，最终取平均得候选回答的事实分数；按分数排序后选取 top-α 为正样本 y_w、其余为负样本 y_l，构建偏好集 D={(x,y_w,y_l)}。
  3) DPO 对齐：以 π_ref 为参考策略，最小化
     L_θ = −E_{(x,y_w,y_l)}[ log σ( β log π_θ(y_w|x)/π_ref(y_w|x) − β log π_θ(y_l|x)/π_ref(y_l|x) ) ]，
     从而把高事实分响应概率提高、低分响应概率降低。
- SELF-EVAL 与 Prompt 设计（SELF-EVAL-P(TRUE)）：
  - 将模型判断建模为 p(True|q,a)=f_M(q,a)，要求模型以 A(True)/B(False) 格式作答，并配合 few-shot 示例提升结构稳定性。
  - 初步实验发现朴素 prompting 存在明显过度自信，校准表现不佳。
- SK-TUNING 训练数据构造：
  - 用 few-shot 对每个问题 q 采样 K 个候选回答，再用 Deberta-Large-MNLI 双向蕴含对比 golden answer，把语义等价标为正确 a_c，否则标为错误 a_i。
  - 基于 True/False Q&A 格式构造正负预测对 (R_+,R_)，并保留重复以近似模型对题目的知识覆盖，帮助校准；总规模达 2,470,860 条异构样本。
  - 损失函数：
    L_φ = −E_{(q,a,r_+,r_-)~D_ψ}[ log σ( log π_φ(r_+|q,a) − log π_φ(r_-|q,a) ) ]，
    优化目标是让正确回答获得更高的相对 log-prob，从而提高区分与置信度质量。
- 实现要点：
  - 基座选用 LLAMA-7B 与 LLAMA2-7B；DPO 微调 5 epoch，batch=8，lr=5e-6，β=0.1。
  - SK-TUNING 使用 Wikipedia 49,862 条提示与 BIG-bench 17 个 MCQA 任务共 32,500 条提示；1 epoch，batch=8，lr=5e-7。
  - 长文任务主张抽取、原子问题生成均以 GPT-3.5-turbo 完成；短文任务直接对候选回答打分。

## 实验与结果
- 任务与数据集：TruthfulQA（MCQA、短文本生成）与 BioGEN（长文本传记生成）；评估含 Accuracy、True/Info/True*Info、FActScore、正确/错误原子事实数 #cor/#incor、Respond ratio 等。
- 主要数值（摘要性）：
  - 在 LLAMA-7B 上，Self-Alignment w/ SELF-EVAL-SKT 较基线 Accuracy 提升约 13%；TruthfulQA 短文本 True*Info 达到 45.75%，BioGEN 的 FActScore 较基线提升约 4%，并维持在较高 Info 水平。
  - 与 SELF-EVAL-P(TRUE) 相比，SKT 版本在 True*Info 与 FActScore 上均有明显优势（文中表述分别提升约 12% 与 4%）。
  - 与 FACTTUNE-MC（一致性置信度方法）相比，SKT 版本在 BioGEN 上持续取得更高 FActScore 并更好控制 #incor。
  - 与表示编辑方法 ITI、DOLA 相比，SKT 版本在 TruthfulQA 上取得更高 True*Info，体现更优的事实—信息平衡。
  - 长文配对人工/GPT-4 评测（Table 2）在 factuality、helpfulness、relevance、naturalness 四个维度均获得显著胜率优势。
- 方法消融与对比：
  - 使用语义等价聚类（SE）或通用自一致性（USC）替代 SELF-EVAL 仍优于基座，但均低于 SELF-EVAL-SKT，佐证校准与内部知识利用的重要性（Table 3）。
- 结论：仅依赖模型内部知识即可完成有效的事实对齐；SK-TUNING 对置信度估计与校准有直接增益；DPO 偏好对齐能把自我评估信号稳定转化为生成改进。

## 相关工作脉络
- HoNESTY-TUNE（Yang et al., 2023）：面向诚实性/承认“不知道”的 SFT 对齐；本文聚焦在“已知且有相关知识时如何可靠说出”，目标是输出更真实的正面信息而非拒答。
- FACTTUNE-MC（Tian et al., 2023a）：使用一致性置信度作为偏好信号做 DPO；本文指出其依赖生成一致性、未必反映内部知识，并以 SELF-EVAL-SKT 的内源评估替换。
- ITI（Li et al., 2023b）、DOLA（Chuang et al., 2023）：推理时表示编辑方法；本文与之定位差异在于不自带领域标注/层间对比干预，而通过自评估构建偏好数据端到端对齐。
- 自一致性/自校验类工作（如 Self-consistency、SelfcheckGPT、CoVe）：偏解码后校正或一致性聚类；本文把评估信号整合进训练目标，形成“边学边评”的闭环。
- LLM 置信度与校准（Kadavath et al., 2022；Guo et al., 2017；Tian et al., 2023b 等）：为本文提供评估指标与方法论基础，本文在其基础上针对事实性评估做专用校准训练。

## 局限性与未来方向
- 仅在小尺度（7B）LLAMA 系模型上验证；更大参数规模（13B、70B）与 RLHF 对齐模型（如 LLAMA2-CHAT）的效果仍需扩展。
- 长文任务的事实分解与原子问答依赖 GPT-3.5-turbo，存在级联误差与成本问题，且对复杂/歧义主张仍可能评估不准。
- 未与高精度解码/编辑方法联合；作者建议与 DOLA 等强基线组合以进一步提升。
- 错误分析显示部分难例源于训练数据噪声（迷信、误导性前提、争议性问题），单靠自评估难以彻底根除，需要更高质量数据与拒答/澄清机制。
- 校准策略中保留重复样本对性能有益，但也可能引入过拟合风险；对多样性与覆盖度的系统分析有限。

## 研究启发与可借鉴点
- “自评估—偏好构建—DPO 对齐”的三段式可迁移到其他能力/属性对齐（如安全性、因果一致性、代码正确性），只需更换可自验的信号构造方式。
- 用成对 log-prob 差优化置信度与校准（SK-TUNING 损失）可直接复用到需要模型给出可靠不确定性估计的任务中。
- 长文事实评估采用“原子化主张→原子问题→独立评分→平均”的分解范式，便于与现有 FactScore 体系对接并与自动评测器协同。
- 实验设计上，将 SE/USC 等弱替代与自评估方案并列，能清晰论证“内源知识利用”的边际收益，值得在同类工作中沿用。
- 保留训练重复以提升校准的思路可作为数据增强策略之一；同时建议在报告中给出“去重前后”对比以增强可复现性。

## 关键术语表
- **Self-Alignment for Factuality**：利用 LLM 对自身输出的事实性进行自我评估，并将评估信号用于偏好微调以实现事实性对齐的整体框架。
- **SELF-EVAL**：基于内部知识的自评估组件，将回答/主张转化为 True/False 问答形式得到 p(True|q,a) 事实分数。
- **SELF-EVAL-SKT**：经 SK-TUNING 校准后的 SELF-EVAL，具备更高估计精度与更好置信度校准。
- **SK-TUNING**：面向事实自评估的校准微调，使用异构知识数据与正负预测对优化模型对正确/错误回答的相对置信度。
- **DPO**：Direct Preference Optimization，通过偏好对直接优化策略损失以避免显式奖励建模的对齐算法。
- **Knows-tells gap**：模型在内部知识上“知道”正确内容，但在生成时却“没说对”的幻觉类型。
- **FActScore**：基于原子主张匹配的事实精度评估指标，用于长文本生成中量化正确/错误事实比例。
- **True/False Q&A prompt**：让模型以 A(True)/B(False) 格式评估给定答案真实性的提示模板。

## 可复现要素
- 数据集：TruthfulQA（公开基准）、BioGEN（使用公开提示；文中用 GPT-4 生成训练/验证样本以辅助任务定义）。
- 代码/权重：论文未提及开源仓库与模型权重发布情况。
- 关键超参：DPO 阶段 5 epoch、batch=8、lr=5e-6、β=0.1；SK-TUNING 阶段 1 epoch、batch=8、lr=5e-7；长文采样温度 T∈{1, 0.9, 0.8}，M=30；Top-α 正样本比例在 LLAMA-7B 为 30%，LLAMA2-7B 为 50%。
- 依赖工具：GPT-3.5-turbo 用于原子主张抽取与问题生成；Deberta-Large-MNLI 用于语义等价判断；GPT-4 用于长文配对评测。
- 硬件：8 × 32G Tesla V100。
