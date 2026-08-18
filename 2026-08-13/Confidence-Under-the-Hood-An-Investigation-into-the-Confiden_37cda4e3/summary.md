---
title: "Confidence-Under-the-Hood-An-Investigation-into-the-Confiden"
source: https://aclanthology.org/2024.acl-long.20.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:56:24"
field: "大语言模型可信赖性"
keywords: ["置信度对齐", "LLM可靠性", "置信度估计", "大语言模型评估", "Prompt设计", "不确定性量化"]
innovations: ["提出Confidence-Probability Alignment概念并系统评估多模型跨任务对齐水平", "设计TTP+OC+LSU三组件CQP提示技术 elicitation 模型置信度表达", "构建四种对齐错位分类体系用于结构化误差分析"]
benchmarks: ["CommonsenseQA", "QASC", "RiddleSense", "OpenBookQA", "ARC"]
---

# 论文速读：Confidence Under the Hood: An Investigation into the Confidence-Probability Alignment in Large Language Models

## 一句话总结
本文提出了"置信度-概率对齐"（Confidence-Probability Alignment）概念，用于评估大语言模型内部概率置信度与其外显言语化确信度之间的相关性；实验发现 GPT-4 在多个数据集上实现了最强的对齐水平（平均 Spearman ρ̂ = 0.42），而开源小模型和早期 GPT-3 则显著较差。

## 研究问题与动机
- **核心问题**：LLM 的内在置信度（由 token 概率量化）与其被询问时口头表达的确定性之间是否一致？这种对齐关系如何影响模型输出的可靠性？
- **现有方法不足**：当前大量提示技术（如 Self-Consistency、Tree of Thoughts、Multi-Agent Debating）依赖模型的自我评估能力，但如果模型的内在校验与外在表达不一致，这些方法可能产生误导；同时，LLM 幻觉常伴随高置信度表达，使用户难以甄别真伪。
- **实际驱动**：随着 LLM 在医疗、法律、教育等高风险领域广泛部署，量化模型的可信度和透明度变得迫切，而置信度对齐是其中关键的一环。
- **研究空白**：现有工作多聚焦于单一方面（概率估计或言语化不确定性），缺乏对二者对齐关系的系统性跨模型比较研究。

## 核心贡献（创新点）
- **提出 Confidence-Probability Alignment 概念框架**：首次将模型内部 token 概率与外显言语化确信度关联起来，作为评估 LLM 透明度和可靠性的新指标，区别于仅关注单一置信度度量（如概率熵或语义熵）的前人工作。
- **设计三组件 Confidence Querying Prompt (CQP) 技术**：融合第三人称视角模拟（TTP）、选项语境化（OC）和 Likert 量表利用（LSU）三个提示策略来 elicitation 模型的置信度表达，消融实验验证三者组合效果最优。
- **系统性的跨模型、跨任务对齐评估**：覆盖 GPT-3 系列（包括 RLHF 变体）、GPT-4 以及 Phi-2-2.7B 和 Zephyr-7B 等开源模型，在 5 个多样化数据集上测量对齐度，揭示了模型规模/训练方式与对齐质量的正相关趋势。
- **构建置信度错配分类体系**：定义了四种对齐类型（Consistent Alignment、Internal Overconfidence、External Overconfidence、Consistent Discordance），为误差分析提供了结构化框架。
- **探索温度参数与置信度稳定性的关系**：发现不同数据集对 temperature 的敏感性存在显著差异，为实际应用中的参数校准提供指导。

## 方法详解
- **响应生成与结构提示**：构造含问题 Q 和选项集合 O_set 的结构化提示，以 "Answer:" 结尾引导模型做出选择，从中提取选定答案 a_i。
- **内部置信度计算（Internal Confidence, IC）**：
  - GPT 系列：通过对输出 log 概率取指数获得 token 概率：P(T_i) = exp(log P(T_i))
  - 开源模型：通过 softmax 转换 logits：P(T_i) = e^{L(T_i)} / Σ_j e^{L(T_j)}
  - **调整 token 概率**（Algorithm 1）：为每种选项识别所有对应 token，取最高概率代表该选项，再将选定选项的最高概率归一化至所有选项概率之和，得到调整后的内部置信度 P_IC。
- **言语化确信度提取（Verbalized Certainty, VC）**：通过 CQP 提示让模型以旁观者视角评估其自身答案的确定程度：
  - **TTP（第三人称视角）**：将问题和答案呈现为"某语言模型的回答"，以减少自我偏好偏差。
  - **OC（选项语境化）**：在提示中提供所有选项，使模型能在比较上下文中评估答案。
  - **LSU（Likert 量表）**：使用"Very Certain → Very Uncertain"六档定性量表而非数值，以增强跨模型的一致性理解。
  - 映射规则：Very Certain=1.0, Fairly Certain=0.8, Moderately Certain=0.6, Somewhat Certain=0.4, Not Certain=0.2, Very Uncertain=0.0。
- **对齐评估**：采用 Spearman 秩相关系数 ρ 衡量 IC 与 VC 之间的单调相关性，公式为 ρ = 1 - 6Σd_i² / (n(n²-1))。

## 实验与结果
- **数据集**：CommonsenseQA（常识推理）、QASC（句子组合问答）、RiddleSense（谜语推理）、OpenBookQA（科学问答）、ARC（复杂推理挑战），涵盖不同任务类型和复杂度。
- **模型**：GPT-3 (text-davinci-001)、InstructGPT-3 (text-davinci-002)、InstructGPT-3 + RLHF (text-davinci-003)、GPT-4 (gpt-4-0613)、Phi-2-2.7B、Zephyr-7B。
- **主要结果**（Table 1，Spearman ρ）：
  - GPT-4 在所有数据集上均最优：CSQA=0.42, QASC=0.47, RiddleSense=0.42, OpenBookQA=0.41, ARC=0.35，平均 ≈ 0.42。
  - InstructGPT-3 + RLHF 次之，平均约 0.33；其非 RLHF 版本仅有 ≈ 0.15。
  - GPT-3 (原始) 几乎无对齐（≈ 0.0）。
  - 开源模型 Phi-2 和 Zephyr-7B 表现最差，多数数据集为负值。
- **温度稳定性**（Figure 3）：temperature 升高导致 VC 变异性增大；多步推理数据集（OpenBookQA、RiddleSense）对温度最敏感，QASC 最稳定。
- **置信度-正确性关系**（Figures 5-6）：GPT-4 的高确信度与高正确率正相关，验证了对齐的有效意义。
- **消融实验**（Table 4）：TTP + OC + LSU 三组件组合在各数据集上表现最佳；LSU 最为稳定有效，TTP 单独作用较小，OC 效果因数据集而异。
- **小模型失败案例**（Table 3）：Phi-2 和 Zephyr-7B 经常无法从预设选项中选择合适的置信度表达，反映出小规模模型在生成与评估双重任务上的能力瓶颈。

## 相关工作脉络
- **Lin et al. (2022)** 教学模型以词汇表达不确定性，本文扩展为该表达的置信度与内部概率的对齐分析。
- **Meister et al. (2022)** 揭示概率-质量悖论（高概率≠高质量），本文在 GPT-4 上发现高确信度与高正确率的正相关，为该悖论在最新模型上的缓解提供了实证观察。
- **Wang et al. (2023b) Self-Consistency** 和 **Yao et al. (2023) Tree of Thoughts** 依赖模型自评估能力，本文的工作为这些方法提供了可信度基础分析——若 IC-VC 未对齐，这些技术的可靠性将存疑。
- **Kuhn et al. (2023)** 使用语义熵量化不确定性，本文关注的是模型自我报告的确定性而非隐式概率度量，两者互补。
- **Zheng et al. (2023) LLM-as-a-Judge** 研究模型评估能力，本文从置信度对齐的角度切入，关注单个模型的内在一致性而非跨模型评判。
- **Zhou et al. (2023)** 研究模型过度自信的 linguistic expressions，本文进一步将这些表达与内部概率进行量化对齐分析。

## 局限性与未来方向
- **Token 级概率访问受限**：仅能获取 GPT-3/4 系列的内部概率，无法评估 PaLM 2、Chinchilla 等其他主流模型。
- **语言局限性**：研究集中在英语（形态较简单），对丰富形态语言（如中文、土耳其语）的推广性未知。
- **元级推理偏差**：CQP 要求模型进行元级自我反思，可能与模型在基础任务中的推理模式存在偏差。
- **对提示工程的强依赖**：对齐效果高度依赖精心设计的 CQP，不同 prompt  formulation 可能导致结果差异。
- **未优化模型回答准确率**：本文聚焦于置信度对齐而非提升模型准确率，未来可探索基于置信度的反馈调节机制。
- **伦理风险**：置信度信息的 misalignment 可能被恶意利用传播虚假信息，需要相应的伦理保障。

## 研究启发与可借鉴点
- **CQP 提示设计可直接复用**：TTP + OC + LSU 三组件组合是 elicitation 模型置信度表达的有效范式，可迁移到任何需要模型自我评估置信度的场景中。
- **Alignment 指标可作为模型可信度度量**：Spearman ρ 作为对齐量化指标，可用于横向比较不同模型/版本的可靠性，为模型选型提供新维度。
- **Temperature 敏感性分析为部署提供指导**：不同任务对温度参数敏感度不同，实际部署时应针对任务类型校准 temperature 以保证置信度表达的稳定性。
- **错位分类学可辅助错误诊断**：四种对齐类型（尤其是 Internal/External Overconfidence）可作为分析模型幻觉的标签体系，辅助调试和改进。
- **开源模型对齐能力薄弱暗示改进方向**：Phi-2/Zephyr 的低对齐表现提示小规模模型需要额外的校准训练，可作为后续研究的切入点。

## 关键术语表
- **Confidence-Probability Alignment（置信度-概率对齐）**：模型内部 token 概率置信度与其外显言语化确信度之间的相关性，是本文提出的核心评估概念。
- **Verbalized Certainty（言语化确信度）**：模型在被明确询问时通过文本表达的对自身答案确定程度的量化值，取值 0–1。
- **Internal Confidence（内部置信度）**：由选定答案 token 的概率（经调整后归一化）所量化的模型内在置信度。
- **Confidence Querying Prompt (CQP)**：本文设计的三组件提示模板，用于从模型中提取言语化确信度。
- **TTP（Third-Person Perspective，第三人称视角）**：CQP 组件之一，以旁观者视角呈现模型自身答案以减少自我偏好偏差。
- **OC（Option Contextualization，选项语境化）**：CQP 组件之一，在提示中提供全部选项使模型能在比较上下文中评估答案确定性。
- **LSU（Likert Scale Utilization，Likert 量表利用）**：CQP 组件之一，使用定性等级量表而非数值评分以增强跨模型的一致性。
- **Adjusted Token Probability（调整后的 token 概率）**：通过取每个选项所有对应 token 的最高概率，再对选定选项概率归一化得到的内部置信度度量。

## 可复现要素
- **数据集**：CommonsenseQA、QASC、RiddleSense、OpenBookQA、ARC，均为公开数据集。
- **代码**：论文未提及代码开源。
- **模型权重**：部分模型为开源（Phi-2-2.7B、Zephyr-7B），GPT 系列需 API 访问。
- **关键超参**：temperature 在温度稳定性实验中有所涉及，其他超参论文未详细列出。
