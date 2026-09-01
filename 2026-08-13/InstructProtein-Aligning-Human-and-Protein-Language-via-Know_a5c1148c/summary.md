---
title: "InstructProtein-Aligning-Human-and-Protein-Language-via-Know"
source: https://aclanthology.org/2024.acl-long.62.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 07:13:23"
field: "科学大模型"
keywords: ["蛋白质语言模型", "指令调优", "知识图谱", "蛋白质序列设计", "多模态大模型", "知识因果建模"]
innovations: ["提出首个蛋白质-人类语言双向生成的大模型InstructProtein", "基于知识图谱和因果建模自动生成高质量蛋白质指令数据集", "提出去偏采样策略解决蛋白质功能注释的标注不平衡问题"]
benchmarks: ["GO-BP/MF/CC", "DeepLoc亚细胞定位", "Metal Ion Binding Prediction", "SCOPe结构分类", "Instruction-Protein Pairing"]
---

# 论文速读：InstructProtein: Aligning Human and Protein Language via Knowledge Instruction

## 一句话总结
本文提出InstructProtein，首个具备人类语言与蛋白质语言双向生成能力的大语言模型；通过构建基于知识图谱的因果指令数据集并采用去偏采样策略，有效解决了蛋白质-文本语料中的标注不平衡与指令信号缺失问题。

## 研究问题与动机
- **现有LLM无法理解生物序列**：ChatGPT、LLaMA等通用大模型训练于人类语言，缺乏对蛋白质氨基酸序列的语言学建模能力。
- **蛋白质语言模型不具备人类语言能力**：ProtTrans、ESM等PLM可预测蛋白质功能，但无法理解或生成自然语言，无法完成文本到蛋白质的反向生成任务。
- **蛋白质-文本语料存在严重标注不平衡**：UniProtKB中研究充分的蛋白质（如细胞质蛋白）被过度标注，而大量蛋白质仅获得稀疏注释，导致训练数据偏向特定类别（表1显示OPT/LLaMA等将所有蛋白质预测为单一类别）。
- **现有语料缺乏指令信号**：蛋白质文本以描述性叙述为主，缺乏供LLM训练的"指令-输出"对，阻碍零样本泛化能力。

## 核心贡献（创新点）
1. **提出InstructProtein，实现蛋白质-人类语言双向生成**：区别于Galactica、BioMedGPT等仅支持蛋白质→文本单向映射的工作，InstructProtein同时支持文本生成蛋白质序列的设计任务。
2. **构建首个基于知识图谱的蛋白质指令数据集**：与Mol-Instructions等模板生成方法不同，本文通过KG三元组+通用LLM转换自动生成高质量、多样化、无幻觉的指令数据。
3. **引入知识因果建模（KCM）**：将蛋白质功能注释间的因果链（如domain→molecular function→biological process）组织为有向无环图，使模型获得类chain-of-thought的因果推理能力。
4. **提出去偏采样策略**：通过序列相似度（MMseqs2编辑距离）与属性相似度（KG嵌入+margin-based ranking loss）联合聚类，缓解标注不平衡问题。

## 方法详解
**训练流程分两阶段**：

**(1) 多语言预训练（Multilingual Pre-training）**
在蛋白质序列语料（UniRef100）与自然语言语料（PubMed摘要）上分别进行自监督预训练，目标函数为标准next-token prediction：
$$L(\mathcal{X}) = \sum_i \log P(x_i | x_{i-k}, \ldots, x_{i-1}; \theta)$$
蛋白质序列使用character-based tokenization（每个氨基酸为独立token），自然语言使用GPT-2 BPE tokenizer。

**(2) 知识指令生成框架**
- **KG构建**：以UniProt/Swiss-Prot为源，提取9类属性字段（生物过程、分子功能、细胞组分、家族、超家族、结构域、保守位点、活性位点、结合位点），构建包含464,333个蛋白质、58,725个注释、5,207,841个三元组的知识图谱。
- **知识因果建模（KCM）**：从InterPro和Gene Ontology获取30,446条因果关系，将三元组组织为DAG结构的因果链。
- **去偏采样策略**：
  - 序列相似度：MMseqs2计算编辑距离$d_s$，阈值$\delta_s$。
  - 属性相似度：优化margin-based ranking loss（式2）学习KG嵌入，使用$\ell_2$距离$d_p$，阈值$\delta_p=1.4$。
  - 采样概率：$P((p,r,t)) = \frac{1}{m} \times \frac{1}{|C_i|} \times \frac{1}{|p|}$，确保各类别均匀采样。
- **三元组转指令**：使用ChatGPT配合3个in-context示例，将KG三元组+KCM转换为问答对格式的指令数据。

**(3) 指令调优（Instruction Tuning）**
在280万条指令数据上微调OPT-1.3B，微调阶段20,000步，总训练40,000步预训练+20,000步调优。

## 实验与结果
**下游任务设置**：
- **Held-In任务**：GO-BP（104,794测试样本）、GO-MF（22,372）、GO-CC（38,594）、亚细胞定位（Sub 2,773 / Bin 1,749）
- **Held-Out任务**：金属离子结合预测MIB（1,332）、指令-蛋白质配对（Fold/Family/SuperFamily）

**主要结果**：
- **蛋白质理解**（Table 2）：InstructProtein（1.3B参数）全面超越所有基线，GO-BP ACC达71.49%（较OPT提升19.66%）、GO-MF达85.83%（较OPT提升29.73%）、Location(Sub)达70.79%（较BioMedGPT提升14.40%）。
- **对抗闭源模型**（Table 3）：InstructProtein在GO和定位任务上大幅超越ChatGPT（+26.80%）、Claude-2（+16.76%）、GPT-4（+24.37%）。
- **蛋白质设计**（Table 4）：SuperFamily配对准确率65.07%，Family达79.24%，显著优于Mol-Instructions（12.57/12.44）。
- **去 novo设计验证**：基于SCOPe结构指令生成的全α螺旋和全β折叠蛋白，经AlphaFold2预测pLDDT随模型规模增长，MDS可视化显示生成序列按指令分离；HHblits同源搜索表明生成序列具有新颖性（identity 0.313–0.880）。
- **Heme结合蛋白设计**（Figure 5）：生成的蛋白质经DiffDock Docking和Smina打分验证展示显著结合亲和力。
- **Ablation**（Table 5）：去偏采样+KCM联合使用效果最佳；KGE属性聚类优于编辑距离聚类。

## 相关工作脉络
- **蛋白质语言模型**（ProtTrans、ESM、MSA Transformer）：仅处理蛋白质序列，不具备人类语言能力，本文弥补双向对齐缺口。
- **多模态蛋白质-文本模型**（BioMedGPT、ProtST、Galactica）：多为单向蛋白质→文本，Galactica虽将蛋白质视为统一模态但缺乏指令信号和去偏处理。
- **分子指令数据集**（Mol-Instructions）：模板生成方法，指令多样性不足，导致零样本性能极差（GO任务全为负预测）。
- **KG增强LLM**（K-BERT、ERNIE 3.0）：将KG融入NLP任务，本文首创KG用于科学指令数据自动生成。
- **指令调优方法**（Self-Instruct、UNNatural Instructions）：依赖LLM内部知识生成，存在幻觉风险；本文以KG为事实基底规避此问题。

## 局限性与未来方向
- **定量建模能力不足**：蛋白质建模大量涉及3D结构预测、稳定性评估、适配度量化等定量任务，当前模型难以处理数值型输出。
- **未来方向**：扩展指令集以纳入定量描述（如结合自由能、热稳定性数值），增强模型在可控蛋白质设计中的应用价值。

## 研究启发与可借鉴点
- **KG增强指令生成范式可迁移**：知识图谱作为事实锚点生成指令数据，可有效避免LLM幻觉，适用于化学、材料等科学领域的指令数据构建。
- **去偏采样策略适用于标注不平衡场景**：联合序列相似性与属性相似性的聚类+均匀采样方法，可直接迁移至其他科学数据中存在长尾分布的任务。
- **知识因果建模（KCM）启发跨模态推理**：将领域知识间的因果关系组织为DAG结构并嵌入指令生成，可拓展至其他科学推理任务（如药物-靶点因果链）。
- **多语言预训练+指令调优的训练范式**：先在双模态语料上分别预训练再对齐，比直接端到端训练更易获得稳定表示，可供其他多模态科学LLM参考。
- **实验设计借鉴**：Held-In/Held-Out的严格划分、free-form问题泛化测试、以及蛋白质设计的可视化评估框架（pLDDT+MDS+homology search）均可复用。

## 关键术语表
**InstructProtein**：首个支持蛋白质序列与人类语言双向生成的大语言模型，通过知识图谱指令实现两种语言的互译与对齐。
**知识因果建模（KCM）**：将蛋白质注释三元组组织为有向无环图，表达从结构域到分子功能再到生物过程的因果推理链。
**去偏采样（Debiased Sampling）**：通过序列和属性双重相似度聚类后均匀采样KG三元组，缓解蛋白质功能注释的长尾分布偏差。
**KG三元组转指令**：以知识图谱三元组为事实基础，借助通用LLM结合in-context示例自动生成问答对格式的指令数据。
**Held-In / Held-Out**：Held-In指测试数据来自训练语料分布内（如GO-CC出现在指令数据中）；Held-Out指测试数据完全独立（如PDB来源的金属离子结合预测）。
**pLDDT**：AlphaFold2预测的局部距离差异测试分数，衡量生成蛋白质序列的折叠可靠性，值越高结构可信度越高。
**指令-蛋白质配对准确率**：评估模型生成蛋白质与其对应指令一致性的指标，通过在10个指令中识别正确配对来测量。

## 可复现要素
- **数据集**：UniRef100（蛋白质序列）、PubMed摘要（自然语言）、UniProt/Swiss-Prot（KG构建）、InterPro & GO（KCM因果关系）、GO benchmark、DeepLoc、MIB benchmark、SCOPe——论文声明所有数据集许可用于科学研究，具体统计见Table 7。
- **代码**：已开源，地址为 https://github.com/HICAI-ZJU/InstructProtein
- **权重**：论文未明确声明模型权重公开方式，代码仓库含完整训练脚本。
- **关键超参**：优化器AdamW，$\beta=(0.9, 0.98)$，weight decay=0.01，dropout=0.1；学习率warmup 5000步至1e-4后线性衰减；batch size=128，context length=1024；预训练40,000步+微调20,000步；8×32G V100 GPU；FP16精度+FSDP加速。
- **序列相似度阈值**：MMseqs2，–min-seq-id 0.8；属性相似度距离阈值$\delta_p=1.4$。
- **数据污染规避**：测试集9,373条序列，聚类删除含测试序列的19,455条训练序列。
