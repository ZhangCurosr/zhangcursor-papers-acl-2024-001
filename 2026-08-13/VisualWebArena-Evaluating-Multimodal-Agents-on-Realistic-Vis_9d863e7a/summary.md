---
title: "VisualWebArena-Evaluating-Multimodal-Agents-on-Realistic-Vis"
source: https://aclanthology.org/2024.acl-long.50.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:56:51"
field: "多模态智能体与网页交互"
keywords: ["多模态Agent", "网页交互", "视觉 grounding", "VisualWebArena", "Set-of-Marks", "大语言模型", "视觉语言模型"]
innovations: ["提出 VisualWebArena benchmark，包含 910 个真实视觉化网页任务", "引入 SoM (Set-of-Marks) 视觉表示方法，简化 VLM Agent 的动作空间", "建立执行级视觉评估体系，包括 eval_vqa 和 eval_fuzzy_image_match 等新型奖励函数"]
benchmarks: ["VisualWebArena", "WebArena", "WebShop"]
---

# 论文速读：VisualWebArena-Evaluating-Multimodal-Agents-on-Realistic-Vis

## 一句话总结
本文提出了 VisualWebArena，一个包含 910 个真实视觉化网页任务的 benchmark，用于评估多模态 Agent 在需要视觉理解的复杂网页操作上的表现；实验表明当前最强的 GPT-4V + SoM Agent 成功率仅 16.37%，远低于人类的 88.7%。

## 研究问题与动机
- **现有基准偏重文本**：绝大多数 Agent 评测基准（如 WebArena、WebShop）仅关注基于文本的网页任务，忽略了现实中大量需要视觉理解的自然任务（如"帮我买一件绿色 Polo 衫"）。
- **视觉信息不可或缺**：现代计算机界面主要为人类视觉设计，图标、颜色、图像布局等视觉信息对任务完成至关重要，纯文本模型难以有效利用这些信息。
- **多模态能力缺乏系统评估**：当前缺乏系统性基准来评估 VLM（Vision-Language Model）在真实、交互式网页环境中的多模态 Agent 能力。
- **Agent 能力差距显著**：即使是 SOTA 模型在该基准上也表现不佳，揭示了 Agent 在视觉理解、规划和执行方面的巨大提升空间。

## 核心贡献（创新点）
- **提出 VisualWebArena benchmark**：构建了 910 个视觉化任务，覆盖 Classifieds（新增）、Shopping、Reddit 三个真实网页环境，显著区别于 WebArena 等纯文本基准。
- **引入 Set-of-Marks (SoM) 视觉表示方法**：通过 JavaScript 自动为页面所有可交互元素添加边界框和唯一 ID，简化 Agent 的动作空间，使 VLM 能直接基于视觉标记进行元素定位。
- **建立执行级视觉评估体系**：设计了包括 exact_match、fuzzy_match、eval_vqa、eval_fuzzy_image_match 等多种奖励函数，能够全面评估开放型视觉任务的正确性。
- **系统性评测 LLM/VLM Agent**：全面对比了多种文本 LLM、图像描述增强 Agent 和强 VLM Agent 的性能，揭示了多模态能力的关键作用和当前模型的局限性。

## 方法详解
- **环境建模**：将环境和 Agent 建模为部分可观测马尔可夫决策过程（POMDP），状态转移函数 $T: S \times A \to S$ 是确定性的，奖励函数 $R: S \times A \to \{0,1\}$ 由人工设计的评估器实现。
- **观察空间**：支持四种表示方式——原始 DOM 树、无障碍树（Accessibility Tree）、网页截图（RGB 数组），以及本文提出的 SoM 标注截图（带边界框和唯一 ID）。
- **动作空间**：定义 12 种原子动作（click、type、press、goto、scroll、stop 等），动作参数使用元素唯一 ID 而非 (x,y) 坐标，专注于高层推理而非低级控制。
- **SoM 预处理**：使用 JavaScript 在页面加载时为每个可交互元素绘制边界框并分配唯一 ID，生成同时包含视觉截图和 SoM 文本表示的输入，简化 VLM 的视觉定位和动作执行。
- **评估函数设计**：
  - exact_match：精确匹配（用于数值答案）
  - must_include/must_exclude：包含/排除检查（用于列表输出）
  - fuzzy_match：使用 GPT-4-Turbo 评估语义等价性
  - eval_vqa：使用 BLIP-2-T5XL 进行视觉问答验证
  - eval_fuzzy_image_match：使用 SSIM 评估图像相似度

## 实验与结果
- **数据集**：910 个任务，覆盖 Classifieds（166）、Reddit（230）、Shopping（514）三个环境，其中 25.2% 的任务包含输入图像，5.1% 的任务不可完成（测试 Agent 的终止判断能力）。
- **评估基线**：文本 LLM（GPT-4、GPT-3.5、Gemini-Pro、LLaMA-2-70B、Mixtral-8x7B）、图像描述增强 Agent（BLIP-2-T5XL、LLaVA-v1.5-7B）、多模态 Agent（GPT-4V、Gemini-Pro、IDEFICS-80B、CogVLM），以及 SoM 变体。
- **主要结果**：
  - 人类成功率：88.7%（230 个代表性任务）
  - 最佳 VLM Agent（GPT-4V + SoM）：16.37%
  - 纯文本 LLM（GPT-4）：7.25%
  - 图像描述增强（GPT-4 + BLIP-2）：12.75%
  - GPT-4o + SoM：19.78%（附录补充实验）
- **关键结论**：多模态显著提升性能（GPT-4V 比 GPT-4 提升约 8 个百分点）；SoM 在视觉上更复杂的 Classifieds 和 Reddit 环境中效果尤其明显；OCR 任务仍是当前 Agent 的主要瓶颈（13.4% vs 16.9%）。

## 相关工作脉络
- **WebArena**：本文直接基于 WebArena 扩展，继承其自我托管环境和执行评估范式，但新增视觉化任务和评估维度，弥补其纯文本评估的不足。
- **WebShop**：最早的电商网页交互基准之一，但仅关注文本指令和产品搜索，未涉及视觉理解任务。
- **Mind2Web**：专注于 GUI 操作的 Agent 基准，但主要针对静态网页和 DOM 操作，缺乏真实网页的视觉挑战性。
- **Set-of-Marks (SoM)**：Yang 等人（2023）提出 GPT-4V 的视觉 grounding 提示方法，本文首次将其系统性地应用于交互式网页 Agent 场景。
- **GPT-4V**：OpenAI 的多模态模型，本文实验中表现最优，但其在 SoM 辅助下仍有较大提升空间。
- **MobileVLM/AppAgent**：最近的多模态移动设备 Agent 工作（Chu 等，2023；Yang 等，2023），与本文方法相似但针对移动端而非网页场景。

## 局限性与未来方向
- **成功率仍有巨大差距**：最佳模型仅 16.37%，距人类 88.7% 差距显著，说明当前 Agent 在视觉理解、规划和长期执行方面存在根本性不足。
- **SoM 依赖 GPT-4V 级别模型**：只有 GPT-4V 能有效利用 SoM 表示，其他开源 VLM（IDEFICS、CogVLM）效果不佳，限制了方法的泛用性。
- **OCR 仍是主要瓶颈**：17.1% 的 OCR 任务成功率仅 13.4%，当前 VLM 在细粒度文本识别方面仍需改进。
- **缺少轨迹记忆机制**：实验中观察到 Agent 容易陷入循环或重复动作，缺乏对历史状态的跟踪和去重。
- **未来方向**：① 微调 VLM 于网页交互轨迹以利用 few-shot 学习的潜力；② 增强 OCR 能力和细粒度视觉理解；③ 引入状态跟踪和去重机制避免循环；④ 开发更高效的 SoM 表示以支持开源模型。

## 研究启发与可借鉴点
- **SoM 表示的可迁移价值**：边界框 + ID 的视觉 grounding 方法可推广到其他 GUI 交互场景（如移动端、桌面应用），简化 VLM 的动作空间。
- **分层难度标注的设计思路**：将任务分为动作难度（Easy/Medium/Hard）和视觉难度，有助于快速筛选适合不同规模模型的评测子集。
- **开放型任务的评估框架**：引入 eval_vqa 和 eval_fuzzy_image_match 等视觉奖励函数，为开放式视觉任务的自动评估提供了可复用的范式。
- **自托管环境的真实性保证**：使用真实数据（Craigslist 抓取 + OSClass CMS）构建 Classifieds 环境，展示了如何平衡真实性和可控性。
- **可结合的创新机会**：将 SoM 与本团队的多模态 Agent 研究结合，探索在办公自动化、文档处理等场景的通用 GUI Agent 应用。

## 关键术语表
- **VisualWebArena**：本文提出的 benchmark，包含 910 个视觉化网页任务，用于评估多模态 Agent 在真实视觉 grounding 任务上的表现。
- **Set-of-Marks (SoM)**：一种视觉 grounding 提示方法，为页面所有可交互元素添加边界框和唯一 ID，帮助 VLM 精确定位和操作元素。
- **Accessibility Tree**：网页的无障碍树表示，提供结构化简化的页面内容表示，常用于辅助技术，本文作为 LLM Agent 的主要输入。
- **POMDP (Partially Observable Markov Decision Process)**：部分可观测马尔可夫决策过程，本文用于建模 Agent 与网页环境的交互。
- **eval_vqa**：基于 BLIP-2-T5XL 的视觉问答评估函数，用于判断 Agent 输出是否满足开放性视觉任务的要求。
- **fuzzy_match**：使用 GPT-4-Turbo 评估 Agent 输出与 ground truth 语义等价性的奖励函数。

## 可复现要素
- **数据集**：910 个任务，三个环境（Classifieds、Shopping、Reddit）均提供自我托管版本； Classifieds 数据来自 Craigslist 抓取（东北美州），使用 scrubadub 脱敏。
- **代码开源**：VisualWebArena 框架和任务数据公开发布（论文附录提供详细设置和提示模板）。
- **关键超参**：视口大小 1280×2048（长上下文模型）或 1280×720（短上下文模型）；文本截断至 3840 tokens（GPT-3.5/4）或 640 tokens（LLaMA/IDEFICS/CogVLM）；GPT 模型 temperature=1.0, top-p=0.9；Gemini 模型 temperature=0.9, top-p=1.0；其他模型 temperature=0.6, top-p=0.95。
- **基线模型**：GPT-4、GPT-3.5、Gemini-Pro、GPT-4V、LLaMA-2-70B、Mixtral-8x7B、IDEFICS-80B、CogVLM、BLIP-2-T5XL、LLaVA-v1.5-7B。
