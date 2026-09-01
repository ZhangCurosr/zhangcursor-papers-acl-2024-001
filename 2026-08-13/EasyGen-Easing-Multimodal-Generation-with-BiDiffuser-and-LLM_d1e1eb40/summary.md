---
title: "EasyGen-Easing-Multimodal-Generation-with-BiDiffuser-and-LLM"
source: https://aclanthology.org/2024.acl-long.74.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 07:05:54"
field: "多模态大模型"
keywords: ["多模态生成", "扩散模型", "大型语言模型", "图像到文本", "文本到图像", "BiDiffuser", "SUR适配器"]
innovations: ["提出BiDiffuser双向条件扩散模型，解决UniDiffuser任务冲突问题", "设计SUR增强适配器将LLM语义知识注入扩散模型提升图像生成质量", "提出Pre-Align/Mid-Align双轨对齐机制实现高效的端到端多模态生成"]
benchmarks: ["MS-COCO", "NoCaps", "OK-VQA", "GQA", "PhotoChat", "VQAv2", "MBBench"]
---

# 论文速读：EasyGen-Easing-Multimodal-Generation-with-BiDiffuser-and-LLM

## 一句话总结
EasyGen提出了一种基于双向条件扩散模型（BiDiffuser）与大型语言模型（LLM）协同的高效多模态生成框架，通过投影层和对齐适配器实现图像-文本的双向转换与生成，在少数据条件下实现了竞争力强的图像描述、视觉问答及高质量图像生成能力。

## 研究问题与动机
- **现有模型严重依赖CLIP/ImageBind等编码器**：当前多模态模型多基于CLIP或ImageBind进行视觉编码，这些编码器擅长学习统一表示但难以实现模态间有效转换，需要海量数据来弥合模态差距。
- **多模态理解有余而生成不足**：大部分现有端到端模型（如BLIP系列、LLaVA系列）专注于多模态内容理解，缺乏生成非文本模态响应的能力；同时期作品如Emu和NExT-GPT虽使用扩散模型，但依赖大量训练数据。
- **现有投影对齐方式存在信息损失**：主流方法（如NExT-GPT）通过MSE损失将LLM输出嵌入空间对齐到扩散模型的文本嵌入空间，导致LLM的语义理解和推理能力未被充分利用，图像生成质量低于原始扩散模型。
- **UniDiffuser的任务竞争问题**：UniDiffuser试图将全部分布（包括噪声条件分布）拟合到单一模型中，导致在特定任务（如从无噪声输入的条件生成）上性能下降。

## 核心贡献（创新点）
- **BiDiffuser双向条件扩散模型**：对UniDiffuser进行针对性微调，仅聚焦图像到文本和文本到图像两个纯净条件分布的学习，避免了多任务学习中的任务冲突，提升了双向生成性能。与UniDiffuser的本质区别在于：BiDiffuser去除了噪声条件分布的学习，专注于无噪声输入的条件生成任务。
- **SUR增强适配器设计**：引入基于交叉注意力的适配器，将LLM的语义表示融合到CLIP文本编码器中，显著增强了BiDiffuser的语义理解和推理（Semantic Understanding and Reasoning, SUR）能力，从而提升图像生成质量。这与Emu等仅对齐文本空间的投影层有本质区别。
- **Pre-Align与Mid-Align双轨对齐机制**：提出两种BiDiffuser与LLM的对齐方式——Pre-Align（投影层置于LLM之前）和Mid-Align（投影层置于编码器与解码器之间），后者额外引入图像-文本距离最小化（ITDM）损失以更好地对齐语义空间。
- **数据高效且可扩展的端到端框架**：仅需169K预训练数据和90K指令调优数据即可达到与大规模模型相当的性能，且BiDiffuser可即插即用到现有先进多模态LLM（如LLaVA）中进一步提升性能。

## 方法详解

**BiDiffuser训练目标**：
从UniDiffuser出发，微调时仅学习两个纯净条件分布，损失函数为：
$$\mathcal{L}_d = \mathbb{E}[\|\epsilon^x - \epsilon_\theta^x(\mathbf{x}_{t^x}, \mathbf{y}_0, t^x, 0)\|^2 + \alpha \|\epsilon^y - \epsilon_\theta^y(\mathbf{x}_0, \mathbf{y}_{t^y}, 0, t^y)\|^2]$$
其中第一项为text-to-image任务，第二项为image-to-text任务，条件输入均为无噪声数据。

**SUR适配器设计**：
通过交叉注意力机制将LLM的语义表示注入CLIP文本编码器：
$$y_{sur} = \text{Att}(f_{CLIP}(\mathbf{y}_0)W^Q, \text{MLP}(f_{LLM}(\mathbf{y}_0))W^K, \text{MLP}(f_{LLM}(\mathbf{y}_0))W^V)$$
最终扩散模型输入为加权组合：$y_0 = \lambda \cdot y_{sur} + (1-\lambda) \cdot f_{CLIP}(\mathbf{y}_0)$

**图像到文本对齐**：
- Pre-Align：投影层置于LLM之前，使用图像 grounded text generation (ITG) 自回归损失：$\mathcal{L}_{ITG} = -\frac{1}{L}\sum_{l=1}^{L}\log p_\phi(w_l^g | w_{<l}^g, I, T_I)$
- Mid-Align：投影层置于编码器与解码器之间，额外引入图像-文本距离最小化损失：$\mathcal{L}_{ITDM} = \frac{1}{N}\sum_{i=1}^{N}\|\mathbf{d}_{diff} - \mathbf{d}_{llm}\|_2^2$

**文本到图像生成**：
LLM首先生成图像描述caption，然后BiDiffuser基于该描述生成图像。训练时冻结BiDiffuser，联合优化：
$$\mathcal{L}_{all} = \mathcal{L}_{t2i} + \mathcal{L}_{t2t}$$
其中$\mathcal{L}_{t2t}$为LLM的自回归损失，$\mathcal{L}_{t2i}$为扩散去噪损失。

## 实验与结果

**数据集与基线**：
- 图像描述：MS-COCO Karpathy测试集、NoCaps验证集
- 视觉问答：OK-VQA验证集、GQA测试-dev集
- 多模态对话：PhotoChat测试集
- 基线包括：BLIP、Flamingo、BLIP-2、InstructBLIP、MiniGPT-4、LLaVA、NExT-GPT、Emu等

**主要结果**：
- **图像描述**：EasyGen Vicuna-7B在COCO (Karpathy) CIDEr达144.6，BLEU@4达42.4；NoCaps CIDEr达121.8，显著优于同规模LLaVA（CIDEr 30.0）。
- **视觉问答**：OK-VQA准确率达45.2%（Vicuna-7B），超越同规模模型。
- **数据效率**：EasyGen仅需169K预训练数据，余弦相似度0.0128、MSE 0.0338，远优于Emu（2B参数，MSE 0.4062）。
- **图像生成质量**：Zero-shot FID为9.16，超越原始UniDiffuser（9.71）；微调后FID达7.68，继续优于UniDiffuser（8.12）。
- **扩展性**：结合CLIP ViT-L/14后，在VQAv2上达79.4%，超越LLaVA-1.5；将BiDiffuser集成到LLaVA-1.5中使其MBBench得分提升至69.2。

## 相关工作脉络
- **BLIP/BLIP-2系列**：基于CLIP编码器将图像映射到LLM文本空间，EasyGen用BiDiffuser替代CLIP作为模态桥梁，实现更高效的模态对齐且支持双向生成。
- **LLaVA系列**：使用单个投影层对齐CLIP-ViT和LLM，专注于理解任务；EasyGen在此基础上增加了扩散模块实现端到端图像生成。
- **NExT-GPT**：采用投影层对齐LLM文本空间与扩散模型文本空间，EasyGen通过SUR适配器直接对齐到扩散模型的图像空间，利用LLM推理能力提升生成质量。
- **Emu/Emu2**：自回归预测下一个视觉或文本token，依赖海量数据；EasyGen通过BiDiffuser的定向微调，以少量数据实现竞争性性能。
- **UniDiffuser**：单一模型拟合所有条件分布；EasyGen的BiDiffuser专注微调双向纯净条件分布，解决任务冲突问题。
- **Sur-Adapter (Zhong et al., 2023)**：启发EasyGen使用LLM增强扩散模型的语义理解能力，本文将其与BiDiffuser结合用于多模态生成场景。

## 局限性与未来方向
- **推理速度较慢**：扩散过程的text-to-image和image-to-text推理耗时较长，在A100上图像生成约需4.96秒，需探索DPM-Solver++等高效采样方法。
- **联合微调受限**：BiDiffuser与LLM的联合微调需要改变采样机制，当前实验表明冻结BiDiffuser已足以处理复杂任务。
- **任务覆盖有限**：当前主要集中于图像描述、VQA和多模态对话，尚未扩展到主体驱动生成、图像编辑等更广泛的多模态任务。

## 研究启发与可借鉴点
- **扩散模型作为模态桥梁的新范式**：用双向扩散模型替代传统CLIP编码器，实现端到端的多模态双向生成，为多模态建模提供了新的架构思路。
- **SUR增强适配器的设计模式**：通过交叉注意力将LLM语义知识注入扩散模型的文本编码器，可有效弥补纯对比学习编码器的语义理解不足，该设计可迁移到其他扩散模型增强场景。
- **Mid-Align的ITDM损失设计**：在编码器-解码器之间进行特征对齐并辅以MSE距离损失，有助于弥合不同模型表示空间的语义鸿沟，可推广至其他跨模型对齐任务。
- **即插即用的扩展性验证**：BiDiffuser可无缝集成到LLaVA等现有模型中并提升性能，证明了模块化设计的价值，为后续研究提供了可复用的组件范式。
- **数据效率优先的训练策略**：通过定向微调扩散模型而非全量训练，以169K数据达到与大规模模型竞争的性能，为资源受限场景提供了可行方案。

## 关键术语表
- **BiDiffuser**：基于UniDiffuser微调的双向条件扩散模型，专门学习图像到文本和文本到图像的纯净条件分布，避免多任务冲突。
- **SUR (Semantic Understanding and Reasoning)**：语义理解与推理能力，本文通过适配器将LLM的语义知识注入扩散模型以增强此项能力。
- **Pre-Align / Mid-Align**：两种BiDiffuser与LLM的对齐方式，前者投影层在LLM之前，后者在编码器与解码器之间并引入ITDM损失。
- **ITDM (Image-Text Distance Minimization)**：图像-文本距离最小化损失，用于Mid-Align中对齐BiDiffuser输出与LLM编码器输出空间。
- **ITG (Image-Grounded Text Generation)**：图像 grounded 文本生成目标，通过自回归损失驱动模型基于图像输入生成文本。
- **UniDiffuser**：统一扩散模型，采用U-ViT架构同时处理图像和文本的序列标记，但尝试拟合过多条件分布导致任务竞争。
- **PhotoChat**：包含10,917张图像和12,286轮对话的多模态对话数据集，模拟真实照片分享行为。
- **MBBench**：全面评估多模态模型能力的新基准，涵盖多种视觉语言任务。

## 可复现要素
- **数据集**：MS-COCO、NoCaps、OK-VQA、GQA、PhotoChat、VQAv2、Visual Genome、TextVQA、LLaVA-80K（均为公开数据集）
- **代码**：已开源，链接 https://github.com/zxy556677/EasyGen
- **权重**：BiDiffuser基于UniDiffuser预训练权重微调，LLM使用FlanT5-XL或Vicuna-7B
- **关键超参**：BiDiffuser微调batch size=312，10 epochs；对齐训练LR=2e-5，cosine decay，warmup=0.03；扩散采样采用DPM-Solver 50步；LoRA用于参数高效微调
