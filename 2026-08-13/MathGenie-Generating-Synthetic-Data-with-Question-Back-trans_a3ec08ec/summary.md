---
title: "MathGenie-Generating-Synthetic-Data-with-Question-Back-trans"
source: https://aclanthology.org/2024.acl-long.151.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:44:32"
field: "数学推理与大模型合成数据"
keywords: ["mathematical reasoning", "synthetic data generation", "back-translation", "code-integrated solution", "verification-based filtering", "LLM finetuning", "open-source model"]
innovations: ["迭代解增强与题目反向翻译流水线保障新题质量与多样性", "代码集成验证理由自动过滤错误解提升合成数据可靠性", "验证推理作为低开销推理增强策略替代多数投票"]
benchmarks: ["GSM8K", "MATH", "SVAMP", "Simuleq", "Mathematics"]
---

# 论文速读：MathGenie-Generating-Synthetic-Data-with-Question-Back-translation

## 一句话总结
本文提出 MathGenie，一种利用题目反向翻译从少量种子数据生成多样化、高质量数学题及代码集成解的合成数据流水线，并训练出 MathGenieLM 系列模型；该系列在五个数学推理基准上 consistently 超越此前开源模型，MathGenieLM-InternLM2 在 GSM8K 达到 87.7%、在 MATH 达到 55.7%，为开源数学推理模型树立了新标杆。

## 研究问题与动机
- **核心问题**：现有开源数学推理模型与 GPT-4 等闭源模型之间存在明显性能差距，主要瓶颈在于高质量、大规模训练数据的匮乏。
- **现有方法不足**：
  1. 直接对题目进行数据增强（如 MetaMath、直接题目增强）容易破坏题目隐含的约束条件，产生无解或逻辑矛盾的题目。
  2. 依赖闭源大模型（如 GPT-4 Code Interpreter）生成合成数据成本高昂，难以规模化扩展。
  3. 生成的题目往往缺乏可靠的ground-truth解，尤其是代码集成格式（code-integrated format）的解与验证理由更为稀缺。
  4. 直接利用网络文本进行指令反向翻译（Instruction Back-translation）的源头数据不可控，生成的题目可靠性难以保证。

## 核心贡献（创新点）
- **提出迭代解增强与题目反向翻译框架**：先对种子题目的解进行迭代式语义/数值变换，再将增强后的解反向翻译为题目，利用解内的逻辑约束确保新题质量。
- **设计基于验证的解过滤机制**：利用具备代码集成求解与验证能力的模型 $M_{\text{code}}$ 生成验证理由（verification rationales），以代码与文本交织的方式自动筛除错误解，显著提升合成数据质量。
- **构建大规模高质量合成数据集 MathGenieData**：结合种子数据（SeedData）与增强数据（AugData），共计约170K样本，涵盖 GSM8K（110K）与 MATH（60K）两个来源，支持后续模型训练。
- **训练 MathGenieLM 系列模型并验证其优越性**：在 7B 至 70B 多种架构（Llama-2、CodeLlama、Llemma、Mistral、Mixtral、InternLM2）上进行全参数微调，在五个基准上均取得开源模型最佳或接近最佳的性能。
- **探索验证推理（Verified Inference）作为高效推理增强手段**：利用模型自身的验证能力对生成的解进行二次校验，以平均 2.3 倍生成成本达到与 3-path 多数投票相当甚至更优的准确率，为推理阶段提供低开销优化路径。

## 方法详解
- **种子数据构成**：
  1. $\mathcal{D}_{\text{text}}$：15K 道 GSM8K/MATH 题目及其人工标注的自然语言解，用于解增强与反向翻译训练。
  2. $\mathcal{D}_{\text{code}}$：80K 道题目对应的代码集成解（$s_{\text{code}}$）及代码集成验证理由（$v_{\text{code}}$），用于训练候选解生成模型 $M_{\text{code}}$（基于 Llama-2 70B）。
- **迭代解增强（Iterative Solution Augmentation）**：
  - 微调 LLaMA-2 70B 得到 $M_{\text{text}}$，输入提示与解，执行替换对象/动词、增加步骤、变更数值等变换。
  - 令 $S^0$ 为原始解集，通过 $K$ 轮迭代生成 $S^1, S^2, \ldots, S^K$，最终合成解集 $\mathcal{S}^{\text{Aug}} = \bigcup_{k=1}^K S^k$（公式 1），保证解的多样性并逐步偏离原题。
- **题目反向翻译（Question Back-translation）**：
  - 在 $\mathcal{D}_{\text{text}}$ 的（解，题）对上微调 Llama-2 70B 得到 $M_{\text{backtrans}}$。
  - 将 $\mathcal{S}^{\text{Aug}}$ 输入 $M_{\text{backtrans}}$，反向生成新问题集合 $\mathcal{Q}^{\text{Aug}}$；因输入为受约束的解，生成的题目逻辑更可靠。
- **基于验证的解过滤（Verification-Based Solution Filtering）**：
  - 用 $M_{\text{code}}$ 为 $\mathcal{Q}^{\text{Aug}}$ 中的每道题生成代码集成解。
  - 先用答案一致性（answer consistency）进行初筛，再用 $M_{\text{code}}$ 生成代码集成验证理由 $v_{\text{code}}$，对解进行真伪判断，仅保留通过验证的解。
  - 验证理由采用自然语言与代码交织形式（见附录 Tab. 13、14），可回溯计算过程并比对条件。
- **数据集与模型训练**：
  - AugData（110K GSM8K + 60K MATH）与 SeedData 合并为 MathGenieData。
  - 对多种预训练模型（7B–70B）进行全参数微调（cosine schedule, lr=$2\text{e}{-5}$, batch=64, warmup=50 steps, AdamW）。

## 实验与结果
- **数据集**：GSM8K、MATH（in-domain）；SVAMP、Simuleq、Mathematics（out-of-domain）。
- **基线**：ChatGPT-3.5、GPT-4、PaLM-2、Mammoth、MathCoder、ToRA、WizardMath。
- **主要结果**（greedy decoding，Tab. 3）：
  - MathGenieLM-InternLM2-20B：GSM8K 87.7%，MATH 55.7%，五项平均 80.9%，开源模型最佳。
  - MathGenieLM-Llama-2-70B：GSM8K 88.4%，MATH 51.2%，平均 78.5%。
  - MathGenieLM-Mixtral-8x7B：平均 80.2%，展现 MoE 潜力。
  - 多数模型上 Out-of-domain 表现提升尤为显著（如 SVAMP、Simuleq）。
- **多数投票**（Tab. 4，$k=10$）：
  - MathGenieLM-Llama-2-70B 平均提升 7.9%，GSM8K 91.7%，MATH 63.3%，优于 ToRA-70B（$k=50$）。
- **消融实验**：
  - 数据组成：AugGSM8K 主要提升小学水平题（GSM8K/SVAMP/Simuleq），AugMATH 主要提升复杂计算题（MATH/Mathematics）（Tab. 5）。
  - 数据量扩展：从 0 到 1× 倍量，五项指标单调上升（Fig. 3）。
  - 验证过滤：开启验证过滤后 GSM8K/MATH 平均提升 1.2%（Tab. 6）。
  - 对比其他题目增强方法：本方法优于 MetaMath、直接题目增强（有/无解）（Tab. 7）。
- **验证推理**（Tab. 8）：
  - Verify (twice) 使 MathGenieLM-Llama-2-70B 平均从 78.5% 提升至 81.1%，平均生成次数 N=2.3×，优于 3-path voting（N=3×）。

## 相关工作脉络
- **MetaMath**：直接对题目进行迭代增强，未利用解的约束，易产生逻辑不一致的新题。
- **Instruction Back-translation**：从网络文本中反向翻译指令，源头数据不可控；本文使用经过增强的解作为源，可靠性更高。
- **MathCoder / ToRA**：依赖 GPT-4 Code Interpreter 生成合成数据，成本高且不可独立扩展；本文使用开源 70B 模型完成全流程。
- **WizardMath**：通过强化学习进化指令以提升数学推理；本文聚焦于数据生成与验证过滤，而非在线 RL 训练。
- **Mammoth**：混合指令微调构建数学通用模型；本文更强调从少量种子数据出发的自动化数据扩充流水线。
- **Llemma**：专注于数学语料预训练的开源模型；本文在其之上用合成数据进一步微调，展示互补性。

## 局限性与未来方向
- **资源开销大**：需对 70B 模型进行全参数微调，依赖 32 张 A800 80GB GPU，不利于低资源研究者复现。
- **不支持图像输入**：模型只能处理纯文本题目，无法解决含图像的几何/图表类数学问题。
- **上下文长度限制**：当前微调上下文为 4096，可能制约复杂长题的处理能力。
- **未来方向**：探索参数高效微调（如 LoRA）以降低资源需求；扩展至多模态数学理解；支持更长上下文；将框架迁移至其他推理任务（如科学推理、代码生成）。

## 研究启发与可借鉴点
- **迭代增强保障多样性**：通过 $K$ 轮解变换（而非单次）使生成解逐渐偏离原始分布，有效提升数据多样性，该策略可迁移至其他领域的数据增强。
- **反向翻译利用解的内在约束**：以解为源、以题为目标的反向生成，比直接生成题目更能保持逻辑自洽；可推广至物理、编程等结构化问题领域。
- **代码集成验证作为数据过滤器**：利用模型自身生成验证理由并进行真伪判断，显著提升合成数据质量；该范式可与 self-verification、program-aided 等方法结合。
- **验证推理作为低成本推理增强**：用 2 次验证替代 3-path voting，在接近相同准确率下降低计算开销，为推理阶段优化提供新思路。
- **数据组成与难度的匹配分析**：通过对比 AugGSM8K 与 AugMATH 对不同基准的提升差异，揭示合成数据应与目标分布对齐的设计原则。

## 关键术语表
- **MathGenie**：本文提出的合成数据生成流水线，包含迭代解增强、题目反向翻译与基于验证的解过滤三个组件。
- **Iterative Solution Augmentation**：对种子解进行多轮语义/数值变换，逐步扩大解空间以保证多样性。
- **Question Back-translation**：将增强后的解作为输入，通过微调的反向翻译模型生成对应的新数学题目。
- **Code-Integrated Verification**：使用自然语言与代码交织的验证理由（$v_{\text{code}}$）对候选解进行真伪校验的机制。
- **Verified Inference**：在推理阶段利用模型自身的验证能力对生成解进行二次检查，错误解则重新生成，以提升最终准确率。
- **Seed Data**：初始的高质量数学题解对，分为 $\mathcal{D}_{\text{text}}$（用于反向翻译）与 $\mathcal{D}_{\text{code}}$（用于训练 $M_{\text{code}}$）。
- **AugData**：通过 MathGenie 流水线生成的合成题解对，共约 170K 样本。
- **MathGenieLM**：在 MathGenieData 上微调得到的数学推理模型家族，覆盖 7B 至 70B 多种基座。

## 可复现要素
- **数据集**：MathGenieData（AugData 170K + SeedData 95K）；论文未明确声明公开，但底层种子数据 GSM8K、MATH 为公开基准。
- **代码/权重**：论文未提及代码开源；基座模型为 Llama-2、CodeLlama、Mistral、Mixtral、InternLM2、Llemma 等开源权重。
- **关键超参**：learning rate $2\text{e}{-5}$，cosine schedule，warmup 50 steps，batch size 64，AdamW；70B/34B 用 32×A800 80GB，7B/13B/20B 用 8×A800 80GB，Mixtral-8x7B 用 16×A800。
- **迭代轮数 K**：正文未明确给出具体值，仅说明 $K$ 轮后取并集；消融中比较了 with/without iteration（Tab. 9）。
- **验证阈值**：未给出显式概率阈值，采用 $M_{\text{code}}$ 生成的验证理由直接判定正确/错误。
