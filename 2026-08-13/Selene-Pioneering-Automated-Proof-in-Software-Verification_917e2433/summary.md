---
title: "Selene-Pioneering-Automated-Proof-in-Software-Verification"
source: https://aclanthology.org/2024.acl-long.98.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:51:11"
field: "形式化验证与大语言模型"
keywords: ["automated theorem proving", "software verification", "large language models", "formal verification", "seL4", "Isabelle"]
innovations: ["提出首个基于真实工业级seL4微内核的项目级自动化证明基准Selene", "设计Lemma Isolation技术实现轻量级增量验证环境", "系统评估GPT-3.5/4在程序化与声明式风格证明上的能力并提出三类增强策略"]
benchmarks: ["Selene"]
---

# 论文速读：Selene: Pioneering Automated Proof in Software Verification

## 一句话总结
论文提出了 Selene，首个基于真实工业级操作系统微内核 seL4 构建的项目级自动化证明基准，验证了 GPT-4 在 Isabelle 形式化语言中生成证明的可行性（P1 准确率 51.8%），并设计了三类增强策略（相似引理、依赖提取、错误修复）以缓解工业级验证的复杂性挑战。

## 研究问题与动机
- **证明阶段人力成本极高**：seL4 microkernel 的形式化验证中，规范阶段耗时 7 人月，证明阶段耗时 11 人年，证明代码量是内核实现代码的 10 倍。
- **现有工作局限于函数级**：已有 LLM 辅助软件验证工作（如 Clover、Yao et al. 2023）仅关注函数级片段，未涉及完整的工业级项目。
- **工业级项目的复杂依赖难以处理**：seL4 存在数百至数千个定义、函数和引理的依赖图，LLM 难以从中准确识别并应用所需的事实。
- **程序化风格证明更具挑战**：seL4 中 5,419/5,464 条引理采用程序化风格（仅给出 tactic 序列），而非数学问题常用的声明式风格，对 LLM 推理要求更高。

## 核心贡献（创新点）
- **提出 Selene 基准**：首个基于真实工业级 seL4 微内核的项目级自动化证明基准（360 条引理，涵盖 4 个难度类别）。与现有数学定理证明基准（如 MiniF2F、PISA）的本质区别在于其面向真实工业软件验证场景。
- **提出 Lemma Isolation 技术**：通过隔离目标引理与依赖会话，避免重复构建整个验证项目，将单次验证时间从数十分钟降至数秒/分钟级。与全量重建评估方法的本质区别在于利用 Isabelle session 持久化结果实现增量验证。
- **验证 LLM 在软件验证中的可行性并量化挑战**：实验表明 GPT-4 在简单引理（P1）上 ACC#5 达 51.8%，但在复杂引理（P3）上几乎为零，揭示了当前模型的能力边界。
- **设计三类增强策略并系统评估**：相似引理检索、依赖事实提取、错误信息驱动的修复增强，分别针对性缓解知识缺失、依赖理解和逻辑错误三类问题。

## 方法详解
- **Lemma Isolation（引理隔离）**：将目标引理从原 session 中分离为 `AInvs_TGT`，同时构建仅包含依赖内容的 `AInvs_DEP` 会话。依赖会话只需构建一次并持久化验证结果，目标会话可快速验证（平均 35-50 秒，对比全量重建 148-217 秒）。
- **引理提取与分类**：通过关键词（`lemma`/`theorem` ... `qed`/`done`/`by`）正则提取，排除 context/locale 内引理，过滤超过 20 行的长证明。按证明长度将程序化风格引理分为 P1（1 行）、P2（2-6 行）、P3（7-20 行），声明式风格引理单独归为 D 类。
- **评估 Pipeline**：输入引理规范 → LLM 生成证明 → 追加到目标会话 → Isabelle prover 验证。使用 ACC#k 指标（k 次采样中至少一次验证成功即计为正确）。
- **相似引理增强**：将 seL4 理论文件按空行分段构建检索库，用 BM25 检索与目标规范最相似的 chunk（取前 10 行）作为上下文输入。
- **依赖增强**：从 ground-truth 证明中提取已应用的 facts，在 chunk 库中定位其来源（跳过依赖会话中的 chunk），取每事实来源的前 5 行作为补充输入。
- **Fixing 增强**：首轮生成失败时，将 Isabelle 错误信息（如 `Undefined fact: "st_def"`）作为第二轮输入，要求模型修正证明。

## 实验与结果
- **数据集**：从 seL4 的 11 个 session 中共提取 5,464 条引理，最终保留 360 条（P1: 139, P2: 104, P3: 59, D: 38）。
- **评估基线**：GPT-3.5-turbo 和 GPT-4，top-p=0.95，ACC#1 温度=0.0，ACC#5 温度=0.5，最大生成 token 2048。
- **主要结果**：
  - GPT-4 ACC#1: P1=41.7%, P2=7.7%, P3=0%, D=10.5%
  - GPT-4 ACC#5: P1=**51.8%**, P2=1.7%, P3=0%, D=15.8%
  - GPT-3.5-turbo 显著低于 GPT-4（ACC#5: P1=35.3%）
  - 声明式引理（D）表现优于程序化 P3，尽管长度相近，归因于中间目标的显式表达降低了推理难度
- **增强效果（GPT-4 ACC#1）**：
  - +Similar: P1 41.7%→47.5%, P2 7.7%→14.4%
  - +Dependency: P1 41.7%→**52.5%**（未定义错误从 47% 降至 24%）
  - +Fixing: P1 41.7%→53.2%, D 10.5%→18.4%
- **错误分析**：GPT-4 错误中约 51% 为逻辑错误（无法完成证明），约 47% 为未定义错误（应用了 seL4 中不存在的事实）。

## 相关工作脉络
- **Thor (Jiang et al. 2022) / DSP (Jiang et al. 2023)**：在 Isabelle 中将 LLM 与 Hammer  prover 结合；Selene 关注真实工业项目而非数学定理，且无需外部 prover 辅助。
- **Baldur (First et al. 2023) / Lyra (Zheng et al. 2023)**：利用错误信息修复证明；Selene 的 fixing 增强借鉴此思路，但应用于软件验证场景。
- **MiniF2F (Zheng et al. 2022) / PISA (Jiang et al. 2021) / LeanDojo (Yang et al. 2023)**：数学定理证明基准；Selene 是首个面向工业级软件验证的基准，聚焦程序化风格证明。
- **Clover (Sun et al. 2023)**：基于 Dafny 的函数级代码-规范一致性检查；Selene 扩展到完整 OS microkernel 级的项目级证明。
- **Yao et al. (2023)**：使用 GPT-4 为 Rust 代码生成不变量和断言；仅覆盖函数级片段，未处理跨文件依赖。
- **ProofNet (Azerbayev et al. 2023)**：在 Lean 上训练/微调 LLM；Selene 不依赖微调，直接评估现成 LLM 的零样本/少样本能力。

## 局限性与未来方向
- **依赖提取精度不足**：当前依赖增强依赖人工标注的 ground-truth 证明来定位 facts，未定义错误仍占近半；未来需引入 RAG 等技术自动从代码库提取候选事实。
- **未覆盖规范生成阶段**：仅关注证明阶段，规范阶段（将需求翻译为形式化语言）的自动化同样是耗时大户，有待探索。
- **不支持交互式证明状态**：当前 pipeline 仅支持端到端一次性生成完整证明；对于 P3 等复杂引理，需引入类似 PISA 的交互式 proof state 机制，模拟人类逐步构建证明的过程。
- **声明式风格增强效果不稳定**：Similar+Fixing 组合在 D 类上产生意外下降（7.9% vs 原始 10.5%），机制有待进一步研究。

## 研究启发与可借鉴点
- **Lemma Isolation 设计思路可迁移**：对于大型形式化项目（如 Coq、Lean 工程），均可通过分离目标与依赖会话来加速评估，适用于任何依赖重型项目的 LLM 评测框架搭建。
- **程序化 vs 声明式风格的分类评估**：按证明风格而非仅按长度分类，揭示了 LLM 在不同推理模式下的能力差异，这一分类范式可推广至其他形式化语言基准构建。
- **错误信息驱动的两轮对话机制**：Fixing augmentation 将 Isabelle 错误信息作为条件输入的策略，可与 Self-Debugging、Reflexion 等工作结合，形成更通用的"生成-验证-修正"闭环。
- **BM25 检索增强在形式化验证中的应用**：Similar lemma augmentation 展示了简单检索策略在填补领域知识缺口方面的有效性，可与 RAG 技术结合探索更复杂的检索-生成联合框架。
- **与 PISA 的互补性**：论文指出 lemma isolation 与交互式 proof state（PISA）正交，两者结合可构建更强大的交互式自动化证明系统，是一个明确的研究结合点。

## 关键术语表
- **seL4**：一个经过全面形式化验证的微内核操作系统，其验证基于 Isabelle/HOL 定理证明器，是本文基准的数据来源。
- **Isabelle**：一种交互式定理证明器，支持高阶逻辑和集合论，seL4 的形式化验证即在其上完成。
- **Lemma Isolation（引理隔离）**：将目标引理从原始 session 中剥离并构建独立的目标 session 和依赖 session，以实现快速增量验证的技术。
- **Procedural Style（程序化风格）**：仅指定 tactic 序列而不显式描述中间目标的证明风格，seL4 中占绝大多数（99.1%）。
- **Declarative Style（声明式风格）**：同时显式写出中间证明目标和对应操作的证明风格，常见于数学定理证明。
- **ACC#k**：评估指标，表示 k 次独立采样生成中至少有一次被 prover 验证成功的比例。
- **ROOT file**：Isabelle session 的元数据配置文件，定义 session 的依赖关系和入口 theory 文件。
- **Tactic**：Isabelle 中的证明策略/操作指令，用于逐步推进证明目标的简化。

## 可复现要素
- **数据集**：Selene 基准，从 seL4 随机抽取 360 条引理（论文未明确说明代码/数据是否开源，建议查阅 ACL Anthology 页面确认）。
- **代码/权重**：论文未提及开源代码或模型权重。
- **关键超参**：top-p=0.95，ACC#1 温度=0.0，ACC#5 温度=0.5，最大生成 token=2048，验证超时=10 分钟，相似检索取前 10 行，依赖增强取每事实来源前 5 行，演示用例固定 5 条。
