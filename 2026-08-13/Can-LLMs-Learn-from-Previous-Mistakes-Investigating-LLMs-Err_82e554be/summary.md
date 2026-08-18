---
title: "Can-LLMs-Learn-from-Previous-Mistakes-Investigating-LLMs-Err"
source: https://aclanthology.org/2024.acl-long.169.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:55:48"
field: "大语言模型推理能力增强"
keywords: ["Chain-of-Thought", "错误学习", "自我反思", "指令微调", "大语言模型推理", "COTERRORSET"]
innovations: ["提出Self-rethinking提示方法，通过历史错误示例引导LLM反思和纠错", "提出Mistake tuning微调方法，在SFT中同时利用正确和错误推理数据", "构建大规模COTERRORSET数据集，包含55万+问题的错误推理及成因分析"]
benchmarks: ["GSM8K", "AQuA", "MathQA", "OpenbookQA", "LogiQA", "Critical Reasoning"]
---

# 论文速读：Can-LLMs-Learn-from-Previous-Mistakes-Investigating-LLMs-Err

## 一句话总结
本文系统研究大语言模型能否从自身历史推理错误中学习以提升推理能力，提出了大规模错误数据集 **COTERRORSET**、推理阶段的 **Self-rethinking** 提示方法及微调阶段的 **Mistake tuning** 方法，并在算术与常识推理任务上验证了其有效性。

## 研究问题与动机
1. 现有CoT研究主要利用正确答案和正确推理链（golden-standard rationales）来提升LLM推理，但人类认知的重要特征是从错误中学习并避免重复犯错，这一方向尚未被充分探索。
2. 缺乏对LLM在CoT推理过程中中间错误的系统性理解，不清楚错误的具体类型、分布规律及其成因。
3. 传统监督微调仅依赖正确答案，忽视了错误示例对模型识别和规避错误能力的潜在价值。
4. 如何在保证成本效益的前提下，利用错误信息增强LLM推理能力，尤其是在无需人工标注纠错对的情况下。

## 核心贡献（创新点）
1. **构建大规模错误数据集COTERRORSET**：包含558,960个问题（引言提及609,432），每个问题均配有正确答案、错误推理及错误原因分析，支持可扩展的错误学习与分类。**与现有CoT数据集的本质区别在于专门收集并标注了LLM错误推理及其成因。**
2. **提出Self-rethinking提示范式**：在推理过程中引入历史错误类型、原因及正反示例，引导模型反思是否重犯同类错误。**与Self-refine等方法的本质区别在于主动利用外部错误示例而非仅依赖内部自反馈。**
3. **提出Mistake tuning微调方法**：在SFT中同时使用正确和错误推理数据，通过[CORRECT RATIONALE]和[INCORRECT RATIONALE]前缀区分两者。**与传统SFT的本质区别在于模型不仅学习正确推理，还学会识别和避免错误。**
4. **系统性错误类型分析**：通过LLM聚类+人工审核将错误归为计算错误、数值错误、逻辑错误、常识错误、语言错误和上下文错误等抽象类别，为后续改进提供明确方向。

## 方法详解
1. **COTERRORSET构建流程**：以COT-COLLECTION为基础，从QASC、GSM8K、AQuA等9个数据集采集问题；用PaLM2生成每个问题的错误推理；将正确答案与错误答案一并输入PaLM2，促使其反思并生成错误原因说明；最后通过LLM聚类提取错误关键词，人工审核归类为6种抽象错误类型。
2. **Self-rethinking方法**：
   - **初始推理阶段**：对目标问题进行零样本CoT推理，得到初始答案和推理链。
   - **反思阶段**：展示历史错误类型定义、错误原因示例、以及对应正确/错误推理样例，提示模型判断当前回答是否包含类似错误。
   - **迭代纠错**：若检测到错误，结合示例和可疑错误答案进行第三次推理；设置最大迭代次数 $k$ 防止陷入死循环。
3. **Mistake tuning方法**：
   - 训练数据构造为 $p = [Q \oplus S \oplus R]$，其中 $Q$ 为问题，$S$ 为特殊前缀（`[CORRECT RATIONALE]` 或 `[INCORRECT RATIONALE]`），$R$ 为对应推理。
   - 损失函数为标准自回归交叉熵：$\mathcal{L} = -\sum_{t=1}^{|p|} \log P(p_t | p_{<t})$。
   - 前缀设计使模型在微调过程中自然区分正确与错误推理模式，无需额外的对比损失或纠错对。

## 实验与结果
**数据集与基准**：GSM8K、AQuA、MathQA、OpenbookQA、LogiQA、Critical Reasoning（MARB子集）。
**模型**：PaLM2 (TEXT-BISON-001)、GPT-4、Flan-T5-large (780M)、Flan-T5-xl (3B)。
**训练设置**：学习率 $1\times10^{-4}$，随机种子42，AQuA随机采样10,000条。
**主要结果**：
- **Self-rethinking（PaLM2, zero-shot）**：GSM8K达 **65.13%**（vs CoT 56.29%，+8.84pp）；LogiQA达 **49.12%**（vs CoT 41.05%，+8.07pp）；CR达54.53%（vs CoT 51.98%）。在相似计算预算下（Self-consistency采样3次），本方法全面优于Self-consistency。
- **Self-rethinking（PaLM2, 8-shot）**：GSM8K达 **70.15%**（vs 8-shot CoT 64.56%，+5.59pp）。
- **Self-rethinking（GPT-4, zero-shot）**：GSM8K达98.02%（vs CoT 97.93%），LogiQA达81.37%（vs CoT 78.92%），一致提升。
- **Mistake tuning（Flan-T5-large）**：GSM8K达18.36%（vs标准微调14.28%，+4.08pp）；MathQA达 **48.95%**，甚至超过PaLM2的41.37%。
- **最强结果**：Self-rethinking在GSM8K上达到65.13%，相比CoT提升8.84个百分点；迭代次数 $k$ 从1增至24时，GSM8K进一步提升8.11%。

## 相关工作脉络
1. **Chain-of-Thought (Wei et al., 2022)**：本文与其定位差异在于CoT聚焦如何生成正确推理链，本文则聚焦如何利用错误推理来增强推理能力。
2. **Self-refine (Madaan et al., 2023)**：两者均关注自我改进，但Self-refine仅依赖内部自反馈而不使用外部错误示例；本文证明引入历史错误示例能更稳定且显著地提升性能。
3. **Self-consistency (Wang et al., 2022)**：Self-consistency通过多数投票提升准确性，计算成本高；本文方法在相似预算下更优，且通过错误反思实现精准纠错而非简单聚合。
4. **Learning from mistakes (An et al., 2023)**：该工作使用GPT-4生成的错误-显式纠正对进行微调；本文强调仅需暴露于正确和错误示例即可，无需外部教师模型的显式纠正。
5. **Contrastive in-context learning (Gao and Das, 2024)**：本文Self-rethinking受其对比思想启发，通过在示范样本中同时展示正确和错误推理来引导模型反思。

## 局限性与未来方向
1. Self-rethinking不适用于缺乏明确客观标签的任务（如机器翻译、对话生成），因正确性判断具有主观性或上下文依赖性。
2. Mistake tuning需要每个样本具备ground truth标签，在低资源场景下应用受限。
3. 当错误超出模型当前能力范围时，即使增加反思轮数也难以纠正，因此需设置最大迭代次数 $k$ 作为截止。
4. 未来方向：引入对比学习区分正误示例、结合记忆和检索增强机制、设计专用损失函数从错误中提取隐式信息、将方法推广至更多任务领域。

## 研究启发与可借鉴点
1. **错误数据的低成本高价值**：传统研究追求高质量人工标注数据，本文证明利用LLM自身生成的错误推理同样有效，为低资源场景的数据构建提供了新思路——可用强模型生成错误数据供弱模型学习。
2. **前缀区分策略的可迁移性**：通过 `[CORRECT RATIONALE]` / `[INCORRECT RATIONALE]` 前缀区分正负样本，简单且有效避免了错误数据污染，这一设计可直接迁移到任何需要区分正负示例的指令微调场景。
3. **错误类型聚类方法论**：采用"LLM提取关键词→聚类为抽象类别→人工审核"的流程，这一自动化错误分类框架可复用于其他模型或领域（如代码生成、表格推理）的错误分析。
4. **"推理-检查-纠错"迭代机制**：Self-rethinking的结构化反思循环可与其他推理框架（如ToT、GoT、Step-back prompting）组合，形成更强的多级推理增强管线。
5. **与本团队的结合机会**：可将此方法与团队的知识图谱推理或逻辑推理方向结合，探索在特定垂直领域中构建领域专属的错误集，并通过Mistake tuning实现更高效的领域适配。

## 关键术语表
**Chain-of-Thought (CoT)**：一种提示技术，通过引导LLM输出逐步推理过程来提升复杂任务的表现。
**Self-rethinking**：本文提出的推理增强方法，让LLM基于历史错误类型和示例反思并纠正自身推理。
**Mistake tuning**：本文提出的微调方法，在SFT中同时使用正确和错误推理数据，通过前缀区分提升模型的错误识别与规避能力。
**COTERRORSET**：本文构建的大规模错误推理数据集，包含约56万个问题及其正确/错误推理和错误原因。
**Self-consistency**：通过采样多条独立推理路径并选择最常见答案的解码策略，以提升LLM输出的稳定性。
**Self-refine**：让LLM通过自反馈迭代改进输出的方法，不依赖外部数据或错误示例。
**Calculation error**：LLM在进行数学运算时因步骤处理不当产生的计算不准确错误。
**Logical error**：LLM在推理策略、假设或因果链条上出现的错误，如误用公式或推导过程不一致。

## 可复现要素
- **数据集**：COTERRORSET，论文声明将于 https://github.com/YookiTong/Learn-from-Mistakes-CotErrorSet 公开。
- **代码/权重**：论文未提及开源。
- **关键超参**：学习率=1e-4，随机种子=42，Self-rethinking最大迭代次数k（默认1，消融实验测试至24），Self-consistency采样次数=3，AQuA训练子集=10,000条，Flan-T5微调未提及具体epoch数。
