---
title: "Selene: Pioneering Automated Proof in Software Verification"
source: https://aclanthology.org/2024.acl-long.98.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:02:44"
field: "形式化验证与AI结合"
keywords: ["自动化证明", "软件验证", "大语言模型", "seL4", "Isabelle", "形式化方法", "定理证明"]
innovations: ["首个基于工业级seL4的项目级自动化证明基准Selene", "提出lemma隔离技术实现轻量级验证环境", "三种增强策略（相似检索/依赖提取/错误修复）缓解LLM证明挑战"]
benchmarks: ["Selene"]
---

# 论文速读：Selene: Pioneering Automated Proof in Software Verification

## 一句话总结
论文提出Selene基准，这是首个基于真实工业级项目seL4微内核操作系统构建的项目级自动化证明基准测试，通过lemma隔离技术提供轻量级验证环境，并展示了GPT-3.5-turbo和GPT-4在该任务上的潜力及增强策略效果。

## 研究问题与动机
- seL4操作系统微内核的形式化验证需要11人年的证明阶段工作量，证明代码量甚至是微内核实现本身的10倍，亟需自动化降低人力成本。
- 现有LLM应用于自动化证明的工作主要集中在数学定理证明（如Isabelle/Lean系统），缺乏面向工业级软件验证的基准，且现有工作仅关注函数级别代码验证。
- 工业级项目存在复杂的lemma依赖图，从依赖图中提取所需facts对LLM极具挑战，导致大量"undefined error"。
- seL4中的证明多为过程式风格（procedural style，仅指定tactics序列而不显式描述中间目标），而LLM此前仅在声明式风格（declarative style）上验证过，风格差异带来额外困难。

## 核心贡献（创新点）
- **Selene基准**：首个基于真实工业级项目seL4构建的项目级自动化证明基准，包含360个lemma（覆盖4个难度类别），而非仅限于数学定理。与现有定理证明基准（如PISA、ProofNet、LeanDojo）的本质区别在于其数据源是真实操作系统微内核而非数学竞赛题。
- **Lemma隔离技术**：通过将目标lemma与其依赖分离，构建独立的target session和dependency session，使验证时间从数十分钟降至数秒。与现有基于完整session构建的验证流程的本质区别在于避免了依赖重复构建。
- **评估框架**：提供端到端proof generation pipeline，支持从specification输入到Isabelle prover验证的完整流程，并通过ACC#k指标量化LLM能力。
- **增强策略**：提出相似lemma增强（BM25检索）、依赖增强（提取groundtruth facts来源）和修复增强（错误信息驱动的交互式修正）三种增强方法，显著缓解特定挑战。

## 方法详解
- **数据集构建**：从seL4的11个refinement session中提取5,464个lemma（排除locale内lemma及证明超20行的lemma），按证明长度分为三档：P1（1行）、P2（2-6行）、P3（7-20行）为过程式风格，D（45个声明式风格lemma）全部纳入基准。最终采样360个lemma（P1: 139, P2: 104, P3: 59, D: 38）。
- **Lemma隔离**：将目标lemma从原session中隔离出来，构建仅含目标lemma的target session（AInvs_TGT）和包含依赖理论文件的dependency session（AInvs_DEP）。Dependency session仅需构建一次并持久化结果，target session验证时直接复用，验证时间缩减约3倍。
- **评估Pipeline**：输入为specification（从isolated target session提取）+ few-shot demonstrations（5个示例），LLM生成证明后追加到target session，由Isabelle prover验证。采用temperature sampling（ACC#1: temp=0；ACC#5: temp=0.5, top-p=0.95）进行多轮生成，至少一次验证通过即算成功。
- **增强策略原理**：
  - 相似lemma增强：将seL4 theory文件按空白行分块构建检索库，通过BM25检索与目标specification最相似的chunk（前10行）作为上下文注入。
  - 依赖增强：从groundtruth证明中提取所用facts，在chunk库中定位其来源（跳过dependency session内的chunk），取每个fact来源的前5行作为增强信息。
  - 修复增强：两轮对话机制——首轮生成失败后，将Isabelle prover的错误信息（curly brackets内）反馈给LLM，要求修正证明。

## 实验与结果
- **评估设置**：使用GPT-3.5-turbo和GPT-4，每类别随机选5个lemma作为demonstrations（固定），其余为评测集。生成token上限2048，含"sorry"/"oops"或验证超时（>10分钟）计为失败。
- **主要结果（ACC#1，Table 2）**：GPT-4在P1上达41.7%，GPT-3.5-turbo为28.1%；P2为7.7%/0%；P3为0%/0%；D为10.5%/5.3%。
- **增强效果（GPT-4 ACC#1，Table 4）**：P1上+Similar提升至47.5%，+Dependency提升至52.5%，+Fixing提升至53.2%；D上+Fixing提升至18.4%。
- **最强结果**：GPT-4 + Dependency增强在P1上达到52.5% ACC#1，较基线提升10.8个百分点。
- **错误分析（Table 3）**：GPT-4主要错误类型为logic error（51%）和undefined error（47%），后者表明LLM对依赖关系理解不足。
- **消融实验（Table 6）**：+Similar & +Fixing组合在P1上达61.9%，P2上达20.2%；但在D类别上出现性能下降（7.9% vs 10.5%），原因有待探究。

## 相关工作脉络
- **Thor (Jiang et al., 2022)**：结合LLM与hammer prover在Isabelle中进行定理证明；Selene定位不同在于面向工业级软件验证而非纯数学定理，且依赖复杂度高得多。
- **ProofNet (Azerbayev et al., 2023) / Baldur (First et al., 2023)**：在Lean/Isabelle上进行定理证明训练与错误修复；Selene的区别在于数据源为真实操作系统内核而非数学问题集。
- **LeanDojo (Yang et al., 2023)**：构建Lean大型基准并提供交互式证明环境；Selene与之互补——lemma isolation与交互式状态抓取正交，未来可结合PISA技术引入证明状态。
- **Clover (Sun et al., 2023)**：基于Dafny的code-spec-doc一致性检查基准；Selene聚焦的是更复杂的交互式证明生成而非一致性验证。
- **Yao et al. (2023)**：使用GPT-4为Rust函数级代码生成invariants和proof structures；Selene扩展至整个工业级项目的lemma级别证明。
- **PISA (Jiang et al., 2021)**：提供scala-Isabelle框架以获取Isabelle证明状态；Selene指出未来可结合PISA实现交互式证明，弥补当前端到端生成策略的不足。

## 局限性与未来方向
- 仅覆盖proof阶段，specification生成阶段（将需求翻译为形式化语言）尚未自动化，而这同样耗时耗力。
- 当前pipeline仅支持端到端一次性生成完整证明，无法处理复杂P3级别lemma——人类实践者通常基于中间证明状态逐步交互推导。
- 依赖提取（dependency extraction）不够精确：依赖增强实验中只能定位部分facts来源，约半数错误仍为undefined error，未来需引入RAG等技术自动从代码库提取候选facts。
- 未深入探究相似增强与修复增强在D类别上的反向效果（性能下降）。

## 研究启发与可借鉴点
- **Lemma隔离思想可迁移**：将目标问题与依赖解耦的轻量级验证环境设计，可应用于其他依赖复杂的软件工程验证场景（如大型程序库、编译器验证）。
- **三阶段增强策略具有通用性**：相似上下文检索（知识注入）+ 依赖信息提取（事实定位）+ 错误驱动修正（交互反馈）的组合思路，可迁移至其他代码生成/形式化验证任务。
- **过程式vs声明式风格对比发现**：D类别表现优于P3类别（尽管长度相近），提示在软件验证场景下显式中间目标的声明式风格可能更利于LLM推理，值得作为方法设计参考。
- **错误分析驱动改进**：将LLM错误系统分类（undefined/logic/other）并针对性设计增强策略，为后续研究提供了可复用的评估方法论。
- **与交互式证明结合的机会**：将Selene的lemma隔离与PISA的证明状态追踪结合，探索"LLM交互式定理证明器"架构是一个明确的研究机会。

## 关键术语表
- **Lemma**：Isabelle形式化语言中的定理或引理陈述，包含specification（目标性质）和proof（证明过程）。
- **Procedural style（过程式风格）**：仅指定tactics（证明策略）序列的执行方式，不显式描述中间证明目标。
- **Declarative style（声明式风格）**：同时显式写出中间证明目标和证明操作的风格。
- **Session（会话）**：Isabelle中组织验证结果的基本单位，类似于编程中的"package"，可增量构建和持久化。
- **ROOT file**：Isabelle中定义session结构（依赖关系、入口theory文件）的元数据文件。
- **ACC#k**：k次独立生成中至少一次通过验证的成功率指标。
- **Isabelle**：广泛使用的交互式定理证明辅助器，支持形式化验证语言。
- **seL4**：完全形式化验证的操作系统微内核，验证基于Isabelle，是软件验证领域的标杆项目。

## 可复现要素
- **数据集**：Selene基准基于seL4构建，从5,464个提取lemma中采样360个；论文未明确声明公开，但源码和数据可能随论文发布（需进一步确认）。
- **代码/权重**：论文未提及开源代码或模型权重；使用的是GPT-3.5-turbo和GPT-4的API。
- **关键超参**：top-p=0.95；ACC#1时temperature=0，ACC#5时temperature=0.5；生成token上限2048；验证超时阈值10分钟。
