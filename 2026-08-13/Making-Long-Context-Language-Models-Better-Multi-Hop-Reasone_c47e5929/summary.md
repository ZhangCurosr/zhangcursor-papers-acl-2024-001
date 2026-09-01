---
title: "Making-Long-Context-Language-Models-Better-Multi-Hop-Reasone"
source: https://aclanthology.org/2024.acl-long.135.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:44:26"
field: "多跳推理与长上下文建模"
keywords: ["多跳推理", "长上下文语言模型", "归因推理", "Chain-of-Citation", "MuSiQue", "LoRA微调"]
innovations: ["提出Reasoning with Attributions范式（CoC/CoQ），强制模型在推理链中提供引用归因", "构建MuSiQue-Attribute归因标注数据集并设计五类严格过滤策略", "提出AP/CG/LA/QI多任务学习框架及针对性数据增强策略"]
benchmarks: ["MuSiQue", "2WikiMultiHopQA", "HotpotQA", "MT-Bench", "AlpacaEval"]
---

# 论文速读：Making-Long-Context-Language-Models-Better-Multi-Hop-Reasone

## 一句话总结
本文提出 **Reasoning with Attributions** 方法，要求语言模型在推理过程中为每个推断步骤提供引用归因（citation/quote），从而显著提升长上下文语言模型在多跳推理任务中的性能与鲁棒性，并开源了带归因标注的数据集 MuSiQue-Attribute。

## 研究问题与动机
- 长上下文语言模型在多跳推理任务上表现远低于领先的专用多跳推理系统，存在明显性能缺口。
- 模型难以在含有大量噪声（无关文档）的上下文中准确定位关键信息，即"Lost in the Middle"现象。
- 小型开源自回归模型无法有效利用上下文知识，尤其在小规模模型中问题更为突出。
- 现有方法（如简单CoT提示）未能从根本上解决多跳推理中"信息检索-知识应用"两个环节的挑战。

## 核心贡献（创新点）
1. **提出Reasoning with Attributions范式（CoC/CoQ）**：强制模型在每个推理步骤引用相关上下文片段，将复杂多跳问题拆解为"定位关键信息"+"基于依据构建主张"两个子任务；与CoT的本质区别在于增加了可追溯的证据约束，抑制幻觉和噪声干扰。
2. **构建MuSiQue-Attribute数据集**：使用ChatGPT生成CoQ标注后，经五类严格错误过滤（错误答案、不存在归因、错误引用、重复引用、极端长度引用）得到1358条高质量样本；现有数据集均无归因标注，本文填补了这一空白。
3. **提出多任务学习策略**：在主任务LA（Attribution-based Reasoning）之外，设计AP（答案预测）、CG（CoT生成）、QI（引用识别）三个辅助任务联合训练，本质区别在于通过辅助任务分别强化"简单推理""显式推理""噪声过滤"三项能力，避免单一任务过拟合。
4. **设计针对性数据增强**：引入Distractor Sampling（随机添加无关文档模拟不同噪声水平）和Document Shuffling（打乱文档顺序消除位置偏见）；与常规增强不同，这两种策略专门针对多跳推理的场景特征设计。
5. **AttrLoRA在MuSiQue上超越ChatGPT与Claude-instant**：7B模型零样本表现超过5-shot闭源模型，且对噪声具有更强鲁棒性（噪声增加时性能下降约10分 vs. Vicuna下降超30分）。

## 方法详解
**推理格式设计**
- **CoC（Chain-of-Citation）**：要求模型在推理链的每一步添加引用标记（如`[doc_id]`），最终给出答案。
- **CoQ（Chain-of-Quote）**：在CoC基础上进一步要求模型逐句提取原文引用（word-for-word quotes），信息更丰富但生成难度更高。
- 最终选用CoC作为主要推理格式。

**多任务学习框架**
- **LA（Learning to Attribute）**：主任务，直接训练模型生成带引用的CoC推理链。
- **AP（Answer Prediction）**：仅预测最终答案，帮助模型内化简单问题的推理能力。
- **CG（CoT Generation）**：生成无引用的标准CoT，培养显式逐步推理能力。
- **QI（Quotes Identification）**：从上下文中定位关键引用片段，训练模型过滤无关细节的能力。
- 四个任务联合训练，LA为核心，其余为辅助。

**数据增强策略**
- **Distractor Sampling**：随机选取不同数量的无关文档添加到上下文中，改变相关文档的位置和总文档数，模拟真实噪声场景。
- **Document Shuffling**：打乱文档顺序，消除模型对位置信息的表面依赖。
- 额外混入等量Alpaca指令微调数据（约同等规模），用于防止多跳推理训练损害通用能力。

**训练细节**
- 基础模型：Vicuna-7B-v1.5-16k
- 微调方法：LoRA（约4M可训练参数，占总量6.22%）
- 优化器：ZeRO-3 + gradient checkpointing
- 硬件：8×NVIDIA A100 80GB，训练不超过14小时
- MuSiQue-Attribute经过增强后数据量约翻倍（QI任务除外）

## 实验与结果
**数据集**
- 多跳推理：MuSiQue（500测试）、2WikiMultiHopQA（2Wiki，500测试）、HotpotQA（500测试）
- 通用指令跟随：MT-Bench、AlpacaEval
- 训练集：MuSiQue-Attribute共1358条（2-hop占82.18%，3-hop占14.06%）

**基线模型**
- 闭源：ChatGPT (gpt-3.5-turbo-1106)、Claude-instant (claude-instant-1.2)
- 开源长上下文：LongChat-7B-16k、LongLoRA-7B-16k、Vicuna-7B-v1.5-16k

**主要结果**
- **CoC vs CoT**：在77%的评估案例中CoC优于CoT；Claude-instant使用CoT后性能反而下降，但CoC成功恢复至与AO相当水平。
- **AttrLoRA vs 同规模模型**：在MuSiQue、2Wiki、HotpotQA三个数据集上平均超越20 points以上。
- **AttrLoRA vs 闭源模型**：在MuSiQue上超越ChatGPT和Claude-instant；在2Wiki和HotpotQA上表现相近。
- **噪声鲁棒性**：噪声从0%增至100%，Vicuna在MuSiQue上EM下降超30分，AttrLoRA仅下降约10分。
- **通用能力影响**：MT-Bench评分从Vicuna的6.068降至4.978，但分析表明其中>98%的下降源于Alpaca数据质量差异，多跳推理数据本身仅造成<2%的损害。

**消融实验**
- 多任务学习：+AP→+CG→+LA→+QI逐步提升，三者结合效果最佳（Table 7）。
- 数据增强：增强后在MuSiQue和HotpotQA上均有提升，但在较简单的2Wiki上效果不明显（Table 8）。
- 缩放数据：仅用20%数据即可达到约85%峰值性能；MuSiQue/2Wiki需要更多数据，HotpotQA在60%数据时已达最优（Figure 4）。
- 归因质量与推理性能呈显著正相关（precision与EM的Pearson相关系数在MuSiQue上为0.887*）。

## 相关工作脉络
1. **多跳推理基线方法**：传统Selector-Reader架构（Zhang et al., 2023; Zhu et al., 2021） vs. 长上下文LM直接读取全部文档的范式，本文属于后者，通过归因增强解决噪声问题。
2. **长上下文建模**：LLaMA-LongChat/LongLoRA等扩展上下文窗口的模型，本文在其基础上提出提升多跳推理能力的训练方案。
3. **上下文噪声鲁棒性**：Shi et al. (2023) 发现无关句子干扰数学推理；Liu et al. (2023a) 发现"Lost in the Middle"位置偏差；Yu et al. (2023) 提出Chain-of-Note；本文用归因机制替代文档筛选，实现更鲁棒的上下文利用。
4. **语言模型归因**：WebGPT (Nakano et al., 2021)、Gao et al. (2023) 聚焦幻觉缓解；本文转向归因对多跳推理能力的增益，目标和应用场景不同。
5. **思维链推理**：CoT (Wei et al., 2022)、Self-Consistency (Wang et al., 2023)；本文在CoT基础上增加引用约束，形成CoC/CoQ变体，本质区别在于引入了可验证的外部证据锚点。
6. **RoBERTa相关工作**：本文与RoBERTa无直接关联，此为无关基线，实际相关工作集中于长上下文建模、多跳推理、归因生成三个方向。

## 局限性与未来方向
- 归因形式目前仅限于citation和quote，未来可探索URL等其它归因形式。
- 依赖数据集预先检索的文档，未探索模型与搜索引擎的主动交互。
- 未探索为此任务定制的模型架构设计，仅使用了通用长上下文模型+LoRA微调。
- MuSiQue-Attribute中3-hop和4-hop样本占比低且存在一定比例的推理不忠实问题（unfaithfulness），限制了复杂推理的训练效果。
- 通用指令跟随能力在小模型（7B）上有一定程度的下降，虽主要由混合数据质量差异导致，但仍需进一步探索无损能力保持的训练策略。

## 研究启发与可借鉴点
- **推理-归因解耦思路**：将多跳推理拆解为"定位关键信息+基于依据构建主张"两步，这种分解策略可迁移到其他需要证据支撑的复杂推理任务（如法律、医疗QA）。
- **数据质量重于数量**：1358条严格过滤的高质量归因数据即产生显著效果，说明多跳推理数据的质量门槛远高于数量门槛，后续工作应重视数据清洗与验证机制。
- **多任务学习的协同价值**：AP/CG/LA/QI四任务联合训练既保持了简单问题的效率，又提升了复杂问题的推理深度，为多能力协同优化提供了通用范式。
- **数据增强策略的领域适配**：Distractor Sampling和Document Shuffling专门针对多跳推理的位置偏差和噪声敏感性设计，此类任务感知的增强策略对类似问题有参考价值。
- **归因精度可作为无需人工标签的代理指标**：citation precision与推理性能呈强正相关，可考虑用自动化NLI方法替代人工标注进行高效评估。

## 关键术语表
- **Multi-Hop Reasoning（多跳推理）**：需要跨多个来源/文档整合信息才能回答的复杂推理任务。
- **Reasoning with Attributions**：本文提出的推理范式，要求模型在推理链中为每个主张添加引用归因。
- **Chain-of-Citation (CoC)**：在CoT基础上增加引用标记的推理格式，每步推理附带来源文档索引。
- **Chain-of-Quote (CoQ)**：在CoC基础上进一步要求提取原文逐字引用的推理格式。
- **MuSiQue-Attribute**：本文构建的含归因标注的多跳推理数据集，共1358条训练样本。
- **AttrLoRA**：本文微调后的Vicuna-7B模型，采用LoRA在多任务框架下训练。
- **Lost in the Middle**：长上下文语言模型对中间位置信息敏感度降低的现象。
- **Reasoning Unfaithfulness**：模型给出正确答案但推理过程存在步骤遗漏或顺序错误的现象。

## 可复现要素
- **数据集**：MuSiQue-Attribute已开源（CC BY 4.0许可证）；基准数据集MuSiQue、2Wiki、HotpotQA均有公开版本。
- **代码/权重**：模型权重和代码未明确提及开源状态，论文未声明代码仓库。
- **关键超参**：LoRA微调，8×NVIDIA A100 80GB，训练≤14小时，约4M可训练参数（占6.22%），ZeRO-3 + gradient checkpointing；提示策略为5-shot（来自Trivedi et al. 2023提供的20条标注训练示例中随机采样）。
