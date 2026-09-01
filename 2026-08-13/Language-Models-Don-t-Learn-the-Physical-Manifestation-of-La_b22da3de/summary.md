---
title: "Language-Models-Don-t-Learn-the-Physical-Manifestation-of-La"
source: https://aclanthology.org/2024.acl-long.195.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:43:14"
field: "语言模型评估与可解释性"
keywords: ["H-TEST", "supradiegetic information", " Chain-of-Thought", "多模态训练", "正字法", "感官缺失"]
innovations: ["提出 H-TEST 评测语言物理属性理解能力", "证明 scaling/few-shot/CoT 无法突破感官盲区", "揭示多模态训练效果的非单调性与未解机制"]
benchmarks: ["H-TEST", "Letter Geometry"]
---

# 论文速读：Language-Models-Don-t-Learn-the-Physical-Manifestation-of-La

## 一句话总结
论文通过提出 H-TEST 系列任务，实证验证了纯文本训练的 LLM 无法学习语言的物理属性（视觉形态、听觉韵律），且增加模型规模、few-shot 示例或 Chain-of-Thought 推理均无法突破这一根本性局限。

## 研究问题与动机
- **核心问题**：纯文本训练的 LLM 是否真正理解语言的物理 manifestation（视觉外观、听觉声音）？这类 sensory-deprived 学习是否存在不可逾越的盲区？
- **现有方法不足**：当前 LLM 研究普遍认为 scaling up 数据、模型规模和计算资源能带来持续提升，但本文指出某些依赖感官体验的语言知识根本无法通过纯文本学习获得。
- **哲学框架**：借用 Jackson（1986）"Mary's Room" 思想实验——Mary 拥有关于颜色的所有命题知识，但在亲眼见到红色之前缺乏真正的感官体验；类比 LLM 虽掌握大量语言数据却缺乏视觉/听觉感知。
- **研究目标**：识别 LLM 在正字法（orthography）层面的系统性盲点，并验证这些盲点难以通过常规 LLM 改进方向解决。

## 核心贡献（创新点）
1. **提出 H-TEST 评测基准**：设计了 10 个二分类任务，系统评估 LLM 对语言物理属性（大小写、首字母元音、句末标点、回文、押韵等）的理解能力，填补了现有评测对此类感官维度的空白。
2. **揭示 LLM 的根本性感知盲区**：实证表明大多数 SOTA 专有 LLM（GPT-3.5、Claude 2、Jurassic-2 等）在 H-TEST 上准确率仅约 50%，与随机猜测无异，证明感官缺失导致的知识结构性缺陷。
3. **证明常规改进方向无效**：系统性验证了三种主流优化手段——更大模型（LLaMA 2 13B vs 70B）、更多 few-shot 示例（k=4 到 k=50）、CoT 推理——均无法显著提升 H-TEST 表现，甚至 CoT 反而会降低准确率。
4. **揭示多模态训练的潜在作用与未解之谜**：GPT-4o 和 Claude 3 Opus 表现优异，但单纯增加视觉模态（LLaVA-34B）或 MoE 架构均未能稳定复现，训练数据多样性仍是未解因素。

## 方法详解
- **H-TEST 任务设计**：包含 10 个 A/B 二分类任务（Table 1），每个任务 200 个平衡测试样本（100 A + 100 B）和 50 个 few-shot 示例。任务类型涵盖：
  - **视觉类**：Uppercase（含一个大写字母）、Starts Vowel（以元音开头）、End Punctuation（以标点结尾）、Palindrome（回文）、End Ly（以 ly 结尾）、Spelled Math/Number（ spelled-out 数学符号/数字）。
  - **听觉类**：Rhyme（押韵）。
  - **结构类**：Repeated Word（重复词）、Hyphenated Word（连字符词）。
- **Prompt 格式**：采用 few-shot prompting，要求模型输出单个字母 A 或 B，避免模型通过语义推理绕过感官判断。部分任务故意使用语法错误的句子以防止模型基于语言正确性分类。
- **Letter Geometry 子任务**（Section 4）：额外设计五类视觉操作题——旋转（Rotation）、翻转（Flipping）、加减（Add/Subtract）、组合（Composite）——作为更高难度挑战，需 visuospatial 推理能力。
- **评估指标**：准确率（Accuracy），随机基线为 50%，多选项任务随机基线为 25%。
- **CoT 实验**：比较带/不带 Chain-of-Thought 提示的 performance，并分析 CoT 内容发现模型倾向于抽象语义推理而非感官路径（Table 3）。
- **人工验证**：4 名英语母语本科生在无任何训练的情况下，10/10 任务全对，确认任务对人类 trivial。

## 实验与结果
- **数据集**：H-TEST 由作者自行构建，每个任务 200 测试 + 50 few-shot 样本，随机种子固定为 12062023，代码与数据已开源（github.com/brucewlee/h-test）。
- **评测模型**：6 家商用 API（AI21、Anthropic、Meta、Claude、OpenAI、Aleph Alpha）+ LLaMA 2 开源模型 + GPT-4o / Claude 3 Opus（2024年6月）。
- **核心结果**（Table 2, Table 5, Table 6）：
  - 大多数语言-only 模型平均准确率约 50-60%，接近随机基线 50%。
  - **LLaMA 2 13B→70B**：从 49.2% 降至 44.8%，规模扩大无正向帮助。
  - **GPT-3.5 → GPT-4o**：从 52.3% 跃升至 75.0%（唯一显著改善）。
  - **Claude 2 → Claude 3 Opus**：从 62.2% 升至 76.2%。
  - **LLaVA-34B（多模态）**：仅 54.9%，未能稳定提升。
- **Few-shot 数量实验**（Figure 3）：k ∈ {4, 14, 28, 50} 对平均准确率影响极小。
- **CoT 实验**（Figure 4）：普遍导致性能下降，GPT-4 也从 ~85% 降至更低。
- **微调实验**（Figure 6, Appendix F）：在 GPT-3.5 上用各任务 1000 条数据 fine-tune 3 epochs，无统计显著改善。
- **Letter Geometry**（Figure 5）：多数模型接近 25% 随机基线，GPT-3.5→GPT-4o 提升约 15%。
- **最强结果**：GPT-4o 在 Palindrome 和 Spelled Math 任务上达到人类水平（>90%），整体平均 75%。

## 相关工作脉络
1. **Zimmerman et al. (2023)**：提出 Diegetic vs Supradiegetic 信息框架，本文在此基础上实证检验 supradiegetic（物理形式）信息的缺失。
2. **Jackson (1986) Mary's Room**：哲学思想实验，作为本文概念框架的源头，类比 LLM 的感官缺失。
3. **Rust et al. (2022) Language Modelling with Pixels**：早期探索像素级语言建模，本文延续并系统化此类方向。
4. **Wei et al. (2022c) Chain-of-Thought**：CoT 被广泛认为能提升复杂推理，本文发现其对感官任务有害，构成反例。
5. **Biderman et al. (2023) Pythia**：系统分析 LLM scaling 效应，本文指出 scaling 存在无法突破的上限。
6. **Liu et al. (2023) LLaVA**：多模态模型代表，本文测试发现单纯视觉模态不足以解决 H-TEST。

## 局限性与未来方向
- **感官维度有限**：H-TEST 仅覆盖视觉和听觉，未涉及嗅觉、味觉、触觉等人类多感官体验。
- **依赖专有模型数据**：GPT-4/Claude 3 训练细节不公开，难以精确归因其成功原因。
- **多模态假设过于简化**：未深入分析多模态训练具体如何桥接感官鸿沟。
- **哲学类比隐喻性**：将 qualia 等概念应用于 AI 属隐喻，不可直接等同人类意识。
- **小模型评估困难**：部分开源小模型（Mistral、Mixtral）无法理解任务格式，被排除在分析外。
- **未来方向**：探索何种训练数据/架构/模态组合能真正赋予 LLM 语言物理属性理解能力；开发更细粒度的感官能力评测基准。

## 研究启发与可借鉴点
1. **H-TEST 任务设计值得迁移**：其 A/B 二分类 + few-shot 示例 + 强制单字母输出的格式可有效剥离语义干扰，适用于评估其他"非语义"语言维度。
2. **CoT 反例启示**：对于依赖低层感官/形式特征的任务，CoT 可能适得其反，提示我们需要根据任务类型选择推理策略。
3. **微调无效的实验设计**：作者对 GPT-3.5 进行了充分微调实验并报告负面结果，这种"证明某方法无效"的研究同样具有价值，可作为后续工作的 baseline。
4. **多模态效果的非单调性**：LLaVA 未显著改善而 GPT-4o 大幅改善，提示"多模态"本身不是充分条件，需进一步拆解其内在机制。
5. **与团队方向结合机会**：可借鉴 H-TEST 思路评估本团队模型在正字法、语音敏感性、符号操作等任务上的表现，或设计跨模态对齐的微调策略。

## 关键术语表
- **H-TEST**：本文提出的系列评测任务，用于评估 LLM 对语言物理属性（视觉、听觉）的理解能力。
- **Diegetic Information**：语言内部的语义和命题内容，如词义、句法关系。
- **Supradiegetic Information**：语言的物理形式，如字母形状、音节发音、韵律特征。
- **Mary's Room**：Jackson（1986）提出的哲学思想实验，描述一位只知道颜色物理知识但从未见过颜色的科学家 Mary，类比 LLM 的感官缺失。
- **Chain-of-Thought (CoT)**：引导 LLM 生成逐步推理过程的 prompt 技术，本文发现其对 H-TEST 有害。
- **In-Context Learning (ICL)**：通过 few-shot 示例让模型在推理时临时学习任务，本文表明 ICL 对 H-TEST 无效。
- **Letter Geometry**：涉及字母旋转、翻转、加减等视觉操作的子任务，测试 visuospatial reasoning。
- **Sensory-Deprived**：指 LLM 训练过程中缺乏视觉、听觉等多模态感官体验的状态。

## 可复现要素
- **数据集**：H-TEST 已开源，随机种子 12062023 可复现全部数据（github.com/brucewlee/h-test）。
- **代码**：已开源，包含数据生成、评测脚本。
- **模型权重**：LLaMA 2 权重公开；商用模型（GPT、Claude）通过 API 访问，权重不公开。
- **关键超参**：temperature=0.7，max_new_tokens=5（few-shot 设置），fine-tune 3 epochs，每任务 1000 训练样本。
- **评测环境**：API 访问时间为 2023年11-12月及 2024年6月。
