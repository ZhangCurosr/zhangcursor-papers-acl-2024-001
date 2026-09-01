---
title: "Quantized-Side-Tuning-Fast-and-Memory-Efficient-Tuning-of-Qu"
source: https://aclanthology.org/2024.acl-long.1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:50:12"
field: "大语言模型高效微调"
keywords: ["高效微调", "大语言模型", "模型量化", "侧网络", "内存优化", "参数高效学习", "4-bit量化"]
innovations: ["首次将4-bit权重量化与side network结合，同时优化weights/activations/optimizer三类内存来源", "提出低秩自适应下采样模块(LoRA/Adapter/Pooling)，将下采样可训练参数占比从56%降至0-7.8%", "设计可学习门控融合机制(α初始化=1)，解决LST在长文本生成中的重复问题并支持分类与生成任务"]
benchmarks: ["GLUE", "MMLU", "MT-Bench"]
---

: 
**LoRA (Low-Rank Adaptation)**: 通过低秩矩阵分解W=L₁L₂注入少量可训练参数到预训练权重旁路的方法
**Adapter**: 在Transformer层内插入带非线性激活的低秩适配模块，QST将其用于侧网络下采样以降低参数占比
**Side Network**: 与量化LLM并行的轻量级网络，接收下采样后的隐藏状态进行任务预测，避免对主LLM反向传播
**Reduction Factor (r)**: 侧网络隐藏维度相对于主LLM的缩减倍数，默认r=16，控制内存-性能权衡
**Intermediate Activations**: 前向传播各层输出的临时张量，传统PEFT需缓存用于反向传播；QST通过side network绕过此需求
**Gradient Checkpointing**: 选择性丢弃中间激活并在需要时重新前向计算以节省内存的技术；QST无需此技术
**FLOPS per Token**: 每token计算的浮点运算数，衡量训练吞吐量；QST在此指标上显著优于基线

## 可复现要素
- **数据集**: GLUE (公开)、MMLU (公开)、MT-Bench (公开)、OASST1 (公开)、Alpaca (公开)
- **代码**: 论文声明已开源至GitHub（原文链接见脚注）
- **权重**: 使用OPT系列(1.3B-66B)与LLaMA-2系列(7B-70B)开源预训练权重
- **关键超参**: 缩减因子r=16、Adapter秩=16、NF4量化类型、BF16计算精度、batch_size=4~32、学习率1E-04~2E-04、AdamW优化器
- **硬件**: 4× NVIDIA RTX A5000 (各24GB显存)
- **框架**: PyTorch + HuggingFace Transformers
- **复现声明**: 论文使用与基线相同超参设置，实验运行3次取平均
