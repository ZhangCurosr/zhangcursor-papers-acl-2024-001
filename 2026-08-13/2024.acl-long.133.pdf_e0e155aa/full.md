# Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression

Peiyu Liu<sup>1,2</sup>\* , Ze-Feng Gao<sup>2,4</sup> , Wayne Xin Zhao<sup>2†</sup> ,

Yipeng Ma<sup>3</sup> ,Tao Wang<sup>3</sup> , Ji-Rong Wen<sup>2,5</sup>

<sup>1</sup> School of Information Technology and Management, University of International Business and Economics <sup>2</sup> Gaoling School of Artificial Intelligence, Renmin University of China, <sup>3</sup>Huawei Technologies Co., Ltd.

<sup>4</sup> Department of Physics, <sup>5</sup> School of Information, Renmin University of China

liupeiyustu@163.com,{zfgao,jrwen}@ruc.edu.cn,

batmanfly@gmail.com,{mayipeng,wangtao10}@huawei.com

## Abstract

Key-value (KV) caching is an important technique to accelerate the inference of large language models (LLMs), but incurs significant memory overhead. To compress the size of KV cache, existing methods often compromise precision or require extra data for calibration, limiting their practicality in LLM deployment. In this paper, we introduce DecoQuant, a novel data-free low-bit quantization technique based on tensor decomposition methods, to effectively compress KV cache. Our core idea is to adjust the outlier distribution of the original matrix by performing tensor decomposition, so that the quantization difficulties are migrated from the matrix to decomposed local tensors. Specially, we find that outliers mainly concentrate on small local tensors, while large tensors tend to have a narrower value range. Based on this finding, we propose to apply low-bit quantization to the large tensor, while maintaining high-precision representation for the small tensor. Furthermore, we utilize the proposed quantization method to compress the KV cache of LLMs to accelerate the inference and develop an efficient dequantization kernel tailored specifically for DecoQuant. Through extensive experiments, DecoQuant demonstrates remarkable efficiency gains, showcasing up to a 75% reduction in memory footprint while maintaining comparable generation quality.

## 1 Introduction

Large language models (LLMs) (Touvron et al., 2023; Zhao et al., 2023) have made significant strides in advancing the progress of language intelligence. However, these large-sized models often incur higher inference latency, bringing significant challenges to practical deployment. Therefore, it is urgent to reduce the running overhead of LLMs.

To optimize the efficiency of LLMs during the inference process, a commonly used technique is key-value (KV) caching (Pope et al., 2022). In implementation, KV caching involves the storage of historical tokens associated with the attention key and value tensors of each layer, offering accelerated inference by trading increased memory consumption for a reduction in redundant calculations. However, applications of long-content generation, such as story generation and long demonstrations for in-context learning tasks, would lead to a significant increase in the size of the KV cache, resulting in unaffordable storage costs (Zhang et al., 2023; Liu et al., 2023c). In addition, managing a large cache often involves frequent I/O read and write operations, leading to considerable latency. The issue becomes even more severe when I/O operations need to span across multiple machines (Patel et al., 2023). Therefore, we need to compress KV cache of large models to optimize the inference process.

Considering the above issues, considerable efforts have concentrated on KV cache compression to enhance inference efficiency. As a typical approach, recent work (Zhang et al., 2023; Mu et al., 2023) prunes tokens to keep the KV cache within a small size. This approach, while alleviating memory overhead, potentially leads to information loss in long text generation. Furthermore, although post-quantization methods preserve all preceding text, low-bit quantization often results in substantial model performance degradation. This is primarily attributed to the common challenge of outlier problems in activation value quantization (Dettmers et al., 2022). Additionally, current quantization techniques still rely on calibration or training (Frantar et al., 2022; Xiao et al., 2023) to retain the model performance, thus imposing practical limitations in data-constrained settings (e.g., privacy data). This further highlights the need for a data-free approach to KV cache compression.

To effectively quantize the KV cache (essentially activation values), we draw inspiration from SmoothQuant (Xiao et al., 2023), which suggests that the issue of outliers can be transferred across multiple modules, by migrating the quantization difficulty to weights. However, unlike SmoothQuant, we take an improved approach by directly migrating the quantization difficulty by performing matrix decomposition on the activation values themselves, without comprising the precision of the weights. The underlying principle is that matrix decomposition can potentially adjust the outlier distribution of the original matrix (Liu et al., 2021; Gao et al., 2020a), so that the decomposed local tensors or matrices are easier to quantize.

To this end, in this paper, we propose an effective matrix Decomposition based Quantization method namely DecoQuant, to alleviate the quantization error due to outliers. Our approach is developed based on an important empirical finding: when performing tensor decomposition (i.e., Matrix Product Operator), the value range of the large local tensor (consisting of the major proportion of parameters) becomes narrower, indicating fewer outliers to be resolved in quantization. Based on this finding, we propose a local tensor based quantization method, in which we apply low-bit quantization to the large tensor, while maintaining high-precision representation for the small tensor. In this way, we can achieve a lower quantization error when reconstructing the original matrix by multiplying all the local tensors. Furthermore, we utilize the proposed quantization method to compress the KV cache of LLMs to accelerate the inference rate, and further develop an efficient dequantization kernel tailored specifically for DecoQuant.

DecoQuant provides an effective quantization approach for LLMs, which can compress KV cache to accelerate the inference rate. It is featured by two major merits, namely (1) fully data-free by eliminating the need for complex calibration mechanisms and (2) highly flexible by supporting the quantization for weights only, activations only as well as both simultaneously. Extensive experiments have demonstrated the effectiveness of the proposed approach in reducing the memory consumption of the KV cache and achieving competitive performance. With nearly lossless performance, we can achieve 4-bit KV cache quantization and 8-bit quantization for both weights and activations.

## 2 Preliminary

In this section, we present the background for our approach about LLM inference and quantization.

LLM Inference and KV Caching. Typically, LLMs generate the next token in a two-step process (Zhao et al., 2023; Zhong et al., 2024): (1) prefilling phase, in which LLMs generate the first token based on the prompt, and (2) decoding phase, in which the rest tokens are generated one by one in an auto-regressive manner. Specifically, the decoding phase dominates the inference latency in long-text generation (e.g., story writing). A common practice to accelerate the decoding phase is key-value (KV) caching (Pope et al., 2022), which stores previously seen tokens to avoid recomputing of attention key and value tensors. However, the size of the KV cache increases linearly with the generation length which poses a memory-bounded challenge. Furthermore, the increase in computing power has increased substantially (e.g., 3.4x from A100 to H100) while the communication improvements have lagged behind (e.g., only 1.6x from A100 to H100). This highlights the vital need to address memory compression for the KV cache.

LLM Quantization. Quantization maps a floatingpoint number into low-bit integers, which can largely reduce the model size and inference costs of LLMs (Lin et al., 2023; Frantar et al., 2022; Dettmers et al., 2022). We follow Xiao et al. (2023) and use symmetric quantization for simplicity while the discussion for asymmetric cases is similar by adding a zero-point (Jacob et al., 2018). Generally speaking, there are two major kinds of matrices to be quantized in LLMs, namely weights and activations. In the context of quantizing LLMs, there are typically two approaches: quantizing only the weights to preserve model accuracy or quantizing both the weights and activation values to enhance the hardware compatibility. Formally, the quantization process of a single matrix can be expressed as the following formula:

$$
\hat { \mathbf { W } } = \left\lceil \frac { \mathbf { W } } { \Delta } \right\rceil , \Delta = \frac { m a x ( | \mathbf { W } | ) } { 2 ^ { ( m - 1 ) } - 1 } ,\tag{1}
$$

where W is the floating-point matrix, $\hat { \mathbf { W } }$ is the quantized conterpart, and $\Delta$ is the quantization step size,  is the rounding function and m is the number of bits. However, it is practically difficult to set a suitable value for $\Delta .$ , mainly due to the existence of outliers (those significantly deviate from the majority of values) (Dettmers et al., 2022). Therefore, we aim to mitigate the impact of outliers to achieve the quantization compression of the KV cache.

Tensor Decomposition. Tensor decomposition (Rabanser et al., 2017; Kolda and Bader, 2009) is a standard algorithm to factorize a matrix into a sequential product of local tensors. Specially, we adopt Matrix Product Operator (MPO) (Liu et al., 2021) as the decomposition strategy. Formally we describe the process of decomposing a matrix $\mathbf { W } \in \mathbb { R } ^ { I \times J }$ using MPO as follows:

$$
\mathbf { M P O } \mathbf { \Phi } ( \mathbf { W } ) = \prod _ { k = 1 } ^ { n } \mathcal { T } _ { ( k ) } [ d _ { k - 1 } , i _ { k } , j _ { k } , d _ { k } ] ,\tag{2}
$$

where $\tau$ denotes the local tensor with size $d _ { k - 1 } \times$ $i _ { k } \times j _ { k } \times d _ { k }$ in which $\begin{array} { r } { \prod _ { k = 1 } ^ { n } i _ { k } = I , \prod _ { k = 1 } ^ { n } j _ { k } = J } \end{array}$ and n represents the number of local tensors. We refer to the decomposed tensors as local tensors. When $n = 2$ , we designate the tensor with a larger parameter count as $\mathcal { T } _ { L } ~ ( i . e .$ , the central tensor in Liu et al., 2021), and the one with fewer parameters as $\tau _ { S }$ . With MPO decomposition, we can reorganize and aggregate information within specific tensors providing us with the opportunity to effectively distinguish outliers.

## 3 Methods

In this section, we present an effective matrix Decomposition based Quantization method namely DecoQuant, to alleviate the quantization error due to outliers. We further utilize this method to quantize the KV cache for efficient inference of LLMs.

## 3.1 DecoQuant: Matrix Quantization based on Decomposition

Basically, our approach aims to employ tensor decomposition to adjust the outlier distribution in the original matrix, so as to mitigate the quantization difficulty. As will be introduced, decomposed local tensors tend to exhibit fewer outliers within their value distributions, indicating a potential opportunity for improving quantization accuracy. In what follows, we first study the distribution of outliers in local tensors and then propose an effective quantization approach based on tensor decomposition.

Outlier Distributions in Local Tensors. We are mainly concerned with the KV cache matrices, as they highly affect the inference latency (Zhang et al., 2023; Patel et al., 2023; Liu et al., 2023c). Without loss of generality, we consider $\scriptstyle n = 2$ for MPO decomposition and take the key state matrix, i.e., K, as example:

$$
\mathbf { M P O ( K ) } = \mathcal { T } _ { L } \times \mathcal { T } _ { S } .\tag{3}
$$

![](images/271d0f9c5490bc7bdde94c94c2523d8ef8a30dae4b71b47e22fc25d394935f50.jpg)

![](images/39dcda46628d1d3e187beafeeea73600f95c4bf395c51670f47cba7d611c89ce.jpg)  
(a) Analysis for $\mathcal { T } _ { L }$  
(b) Analysis for $\mathcal { T } _ { S }$  
Figure 1: Outlier distributions of local tensors and matrices. “Keys” are extracted from the output features of value projections in the 16th layer of LLaMA-7B. Investigations of other structures can refer to Appendix A.1.

A property of MPO is that it can adjust the distribution of parameters (i.e., $\{ d _ { k - 1 } , i _ { k } j _ { k } d _ { k } \}$ in Equation 1) in these local tensors. Specially, we take a biased decomposition, where $\mathcal { T } _ { L }$ takes a large proportion of parameters (i.e., 99.4%) while $\tau _ { S }$ only takes a small proportion of parameters (i.e., 0.6%). Such a large tensor $\mathcal { T } _ { L }$ is also called central tensor (Liu et al., 2021; Gao et al., 2022), since it contains the large body of information of the original matrix. Further, we examine the change of the outlier distribution in both tensors. In Figure 1, we can observe an interesting finding that the value distribution of the large tensor $\mathcal { T } _ { L }$ becomes much narrower than the original matrix and the small tensor $\tau _ { S }$ . In other words, it becomes easier to quantize $\mathcal { T } _ { L }$ with fewer bits due to the limited value distribution. Despite that it is still difficult to quantize $\tau _ { S }$ , it is noted that $\tau _ { S }$ only contains a small number of parameters, and we can apply higher quantization precision with an overall small cost.

Local Tensor Quantization. Based on the above discussion, we introduce a novel data-free quantization method based on matrix decomposition. The key idea of our method is that through tensor decomposition, the quantization difficulties (i.e., outliers) can be transferred from the original matrix to its small local tensors. Thus, we can consider applying low-precision quantization to the large tensors, while maintaining high-precision representation for the small tensors. In this way, we can achieve a lower quantization error when reconstructing the original matrix. Specially, our approach involves a two-step quantization process which is shown in Figure 3: (1) First, we utilize MPO to factorize the original matrix into two higher-dimensional local tensors (i.e., $\mathcal { T } _ { S }$ and $\mathcal { T } _ { L } )$ As shown in Figure 1, an important characteristic is that $\mathcal { T } _ { L }$ , which occupies a significant portion of the parameters, has a much smaller distribution of outliers than that of the original matrix. (2) Thus, at the second step, we focus on quantizing the larger tensor $\mathcal { T } _ { L }$ into B-bit integers $( B < 1 6 )$ while preserving 16-bit precision for $\tau _ { S }$ to achieve a lower quantization error (with verified effectiveness in Section 4.3).

![](images/b0106d1c4cab99ac7750ed00f06bb77a78e7f3b75575e70e3826d89a5f14bf44.jpg)  
Figure 2: Matrix quantization based on DecoQuant. The alternating black/white and blue/white squares in the figure denote quantized matrices.

## 3.2 Efficient Inference based on DecoQuant

Building upon the DecoQuant approach discussed in Section 3.1, which achieves data-free matrix quantization through the quantization of decomposed local tensors, our primary objective is to compress the KV cache of LLMs to accelerate the inference rate. The key idea is to quantize the KV cache into a low-bit representation while preserving FP16 precision during computation. Additionally, we have developed a consolidated and efficient dequantization kernel tailored specifically for DecoQuant.

KV Cache Quantization. To introduce our method, we consider a typical L-layer Transformer model with D dimensions, where the input text consists of $T$ prompt tokens. Then we consider compressing key and value cache for two phases of LLM inference separately. (1) Prefilling phase: The key and value cache are initially obtained after the generation of the first token, $i . e .$ , K, $\mathbf { V } \in \mathbb { R } ^ { T \times D }$ Given the relatively large size of the matrices, we utilize the DecoQuant technique offline on the KV cache to alleviate the computational overhead induced by decomposition. (2) Decoding phase: The size of the KV cache grows linearly with the sequence length, $i . e . , \Delta \bar { \mathbf { K } } , \Delta \mathbf { V } \in \mathbb { R } ^ { 1 \times D }$ . To alleviate the increased computational workload due to frequent quantization, we perform Deco-Quant only when the cache accumulates a certain length (e.g., 1k). In particular, DecoQuant supports quantization for weights only (WxA16), activations only (W16Ax), as well as both simultaneously (WxAx), significantly expanding its applicability. Next, we will describe the dequantization process when the key and value cache are recovered to FP16 precision for computation.

![](images/54eed7ea039bf7fae27bae26db747df44a74a2ac5585717b74017929282168eb.jpg)  
Figure 3: Operator fusion for dequantization.

Kernel Fusion for Dequantization. Kernel fusion (Wang et al., 2010) is a technique that combines multiple separate computational kernels into a single, more efficient kernel. Essentially, it allows multiple kernels to be executed as a whole unit and thus reduces the overhead and latency in processing. In our approach, the dequantization of Deco-Quant involves operations that convert integers to floating-point values (i.e., dequantization operator of quantization scales and integer values) and that reconstruct local tensors to matrices (GeMM operator of $\mathcal { T } _ { L }$ and $\mathcal { T } _ { S } )$ . These two operations may involve an additional data movement overhead between GPU compute units and the main memory which leads to increased latency. To address this issue, we design specific kernel fusion methods for 2/4/8-bit values by fusing the dequantization operator with the next GeMM operator (as shown in Figure 3), which streamlines the execution pipeline and improves computational efficiency. By doing this, we can effectively alleviate the computational delay caused by data-movement overhead (see Section 4.4 for specific experiments).

## 3.3 Discussion

In this part, we present the overhead analysis of the proposed approach and then compare it with existing work.

Compression Ratio and Time Complexity. In this part, we assess the memory compression ratio and time complexity of DecoQuant. While the tensor parameters obtained through DecoQuant are slightly larger than the original matrix, the significant storage reduction primarily stems from converting the majority of $\mathcal { T } _ { L }$ parameters from FP16 to B-bit integers, allowing for a more efficient representation of the tensor and a decrease in storage requirements. This reduction is quantified as the compression ratio $( \mu )$ , which is calculated as:

$$
\mu = \frac { \# ( \mathcal { T } _ { L } ) \times B + \# ( \mathcal { T } _ { S } ) \times 1 6 + \# ( \Delta ) } { \# ( \mathbf { W } ) \times 1 6 } ,\tag{4}
$$

where #( ) denotes the count of values. Due to the significantly smaller number of parameters in $\tau _ { S }$ and $\Delta$ compared to $\mathcal { T } _ { L }$ , the compression ratio typically approximates B/16. For inference time, DecoQuant significantly reduces communication costs with 4-bit KV cache. This results in a speedup of 1.25x under conditions of generating an output of 6k tokens (Section 4.4).

Comparison with Existing Work. We compare our method with existing methods (including RTN, LLM.int8(), SmoothQuant, GPTQ, and AWQ) from the perspectives of quantization settings and requirement for extra data, with results presented in Table 1. We find that, similar to RTN, our method can support all quantization settings, including weight only (WxA16), activation only (W16Ax), and simultaneous (WxAx) in a data-free style. In contrast, other methods typically only support a subset of these settings (such as GPTQ and AWQ supporting only WxA16, while LLM.int8() supports WxAx) thereby limiting their practical application. Additionally, some methods require extra data for calibration. However, obtaining calibration data for scenarios involving sensitive user privacy can be challenging. Thus we primarily focus on RTN as our comparative baseline, introducing other methods only as needed.

## 4 Experiments

We mainly evaluate the DecoQuant on the language modeling task to compare it with other quantization approaches. Then we explore its zero-shot generalization ability in open-ended document generation. Finally, we quantitatively measure the effect of KV cache compression on system throughput.

## 4.1 Experimental Setup

Datasets and Implementation. For language modeling tasks, we conduct our experiments on LAM-BADA (Paperno et al., 2016) dataset, which is a widely used dataset evaluating the ability of language models to capture long-range dependencies and contextual understanding in text. To evaluate the effectiveness of DecoQaunt in downstream tasks, we follow Chevalier et al. (2023) and consider five tasks (AG News, Subj, MR, Boolq and RTE) for in-context learning setting. The accuracy is reported to measure the quality of the next token prediction task of different models as well as the downstream tasks. We consider popular large language models with various sizes including LLaMA (7B and 13B) (Touvron et al., 2023) and OPT (1.3B and 6.7B) (Zhang et al., 2022). For the quantization setting, we follow (Xiao et al., 2023) and quantize the weights, activations and KV cache into different bit-precisions (2/4/8/16 bits). The code to reproduce the results of this paper can be found at https://github.com/lpyhdzx/ DecoQuant\_code.

Table 1: DecoQuant facilitates data-free quantization for weights only (WxA16), activations only (W16Ax), as well as both simultaneously $\left( \mathrm { { W } } \mathbf { \mathbf { \rho } } _ { \mathrm { { X } } } \mathbf { A } \mathbf { x } \right)$ . RTN denotes the vanilla round-to-nearest quantization (Lin et al., 2023).
<table><tr><td rowspan="2">Methods</td><td colspan="3">Support</td><td rowspan="2">Data-free</td></tr><tr><td>WxA16</td><td>W16Ax</td><td>WxAx</td></tr><tr><td>RTN</td><td>V</td><td>V</td><td>V</td><td>V</td></tr><tr><td>GPTQ</td><td>V</td><td>x</td><td>x</td><td>x</td></tr><tr><td>AWQ</td><td>V</td><td>x</td><td>x</td><td>x</td></tr><tr><td>LLM.int8()</td><td>x</td><td>x</td><td>V</td><td>V</td></tr><tr><td>SmoothQuant</td><td>x</td><td>x</td><td>V</td><td>x</td></tr><tr><td>DecoQuant</td><td>V</td><td>V</td><td>V</td><td>V</td></tr></table>

Baselines. We introduce popular baseline quantization methods for KV cache compression.

Round-to-nearest (RTN, Lin et al. 2023). RTN maps a real value to an integer value through a naive rounding operation.

SmoothQuant (Xiao et al., 2023). SmoothQuant smooths the activation outliers to weights and only supports WxAx quantization.

Some widely used quantization methods, such as GPTQ and LLM.int8(), are not considered because they cannot quantize the output activation values, thus making them unsuitable for quantization in the KV cache.

## 4.2 Main Results

Comparison with Other Quantization Methods. The results on LAMBADA are shown in Table 2. Compared with FP16, all quantization methods reduce the sizes of the KV cache significantly due to low bit-precisions. Overall, we observe that DecoQuant achieves better average scores than other methods. We note that RTN sometimes gives better results (LLaMA-13B), but this performance is not stable, and in other cases, it is not good. We suspect that it is related to the distribution of outliers in the model, an observation that is very similar to (Dettmers et al., 2022), which mentions that there is a clear difference in the distribution of outliers for large models. When comparing different quantization settings, we find that 4-bit quantization often exhibits close performance to 16-bit performance while 2-bit models get much worse. Interestingly, even in 2-bit quantization, DecoQuant still has a significant advantage over other methods, an observation that opens up the possibility of a 2-bit KV cache in the future, an exploration we leave to be completed in subsequent work.

Table 2: Results when key and value modules are quantized to different levels (denoted as W-A-). “\*” indicates the quantization results based on the calibration dataset generated using the official code.
<table><tr><td>Setting</td><td>Exp</td><td>#Bits</td><td> ${ \mathrm { S i z e } } _ { \mathrm { ( M B ) } }$ </td><td>LLaMA-7B</td><td>LLaMA-13B</td><td>OPT-1.3B</td><td>OPT-6.7B</td><td>Average</td></tr><tr><td rowspan="7">activations only</td><td>FP16</td><td>16-16</td><td>46.7</td><td>87.8</td><td>89.3</td><td>75.4</td><td>81.2</td><td>83.4</td></tr><tr><td>RTN</td><td>16-8</td><td>23.3</td><td>88.6</td><td>89.3</td><td>75.3</td><td>81.2</td><td>83.6</td></tr><tr><td>DecoQuant</td><td>16-8</td><td>23.3</td><td>88.6</td><td>89.4</td><td>75.4</td><td>81.2</td><td>83.7</td></tr><tr><td>RTN</td><td>16-4</td><td>11.7</td><td>86.0</td><td>88.1</td><td>71.7</td><td>80.6</td><td>81.6</td></tr><tr><td>DecoQuant</td><td>16-4</td><td>11.7</td><td>88.1</td><td>88.9</td><td>73.6</td><td>80.9</td><td>82.9</td></tr><tr><td>RTN</td><td>16-2</td><td>5.8</td><td>1.0</td><td>0.0</td><td>3.5</td><td>4.7</td><td>2.3</td></tr><tr><td>DecoQuant</td><td>16-2</td><td>5.8</td><td>47.1</td><td>58.2</td><td>8.6</td><td>28.8</td><td>35.9</td></tr><tr><td rowspan="10">weights &amp; activations</td><td>RTN</td><td>8-8</td><td>23.3</td><td>88.5</td><td>89.3</td><td>75.4</td><td>81.3</td><td>83.6</td></tr><tr><td>SmoothQuant</td><td>8-8</td><td>23.3</td><td>88.5*</td><td>89.3*</td><td>75.3</td><td>81.3</td><td>1</td></tr><tr><td>DecoQuant</td><td>8-8</td><td>23.3</td><td>88.5</td><td>89.4</td><td>75.4</td><td>81.3</td><td>83.7</td></tr><tr><td>RTN</td><td>4-4</td><td>11.7</td><td>86.4</td><td>88.0</td><td>69.4</td><td>78.5</td><td>80.6</td></tr><tr><td>SmoothQuant</td><td>4-4</td><td>11.7</td><td>86.4*</td><td>88.0*</td><td>69.0</td><td>77.7</td><td>1</td></tr><tr><td>DecoQuant</td><td>4-4</td><td>11.7</td><td>88.4</td><td>88.5</td><td>70.8</td><td>79.1</td><td>81.7</td></tr><tr><td>RTN</td><td>2-2</td><td>5.8</td><td>0.0</td><td>0.0</td><td>3.6</td><td>3.2</td><td>2.0</td></tr><tr><td>SmoothQuant</td><td>2-2</td><td>5.8</td><td>0.4*</td><td>0.0*</td><td>3.8</td><td>3.0</td><td>1</td></tr><tr><td>DecoQuant</td><td>2-2</td><td>5.8</td><td>1.3</td><td>3.0</td><td>1.8</td><td>2.9</td><td>2.0</td></tr></table>

Evaluation on Long-text Tasks. We evaluate DecoQuant’s in-context learning capabilities using OPT models on five distinct datasets. For each dataset, we conduct experiments with varying numbers of demonstrations to investigate the impact of KV cache quantization on the contextual length. The summarized results are presented in Table 3. Our findings indicate that a larger number of demonstrations often results in performance improvements, as evidenced by the performance comparison, e.g., 72.8 compared to 66.8 for FP16. This observation underscores the effectiveness of augmenting the contextual information. However, when comparing the performance of RTN and DecoQuant, we observe that, on average, RTN lags behind DecoQuant. An interesting aspect of this comparison is that RTN’s performance is comparable to DecoQuant’s in the case of shorter contexts (2-shot), but it notably deteriorates for longer contexts (10-shot). This outcome reinforces the efficacy of our approach, which effectively compresses the prompt while preserving critical information.

![](images/891183c916d759e3c7e65fd086419f8852c6aee346066888869bad20a0943351.jpg)

![](images/89390b6c77f99fd48c23379cc011f44a1273c54498b2c8aaea6e1e44a84f9bd6.jpg)  
(a) Quantization strategy.  
(b) Length of decomposition.  
Figure 4: Quantization error analysis about quantization strategy and length of decomposition.

## 4.3 Detailed Analysis

Effectiveness of Tensor Quantization. First, We compare the errors after quantization with different local tensors to illustrate the effectiveness of DecoQuant in mitigating the influence of outliers. Specifically, we evaluate two variants that quantize different local tensors: (1) quantizing large local tensors only, i.e., $\mathcal { T } _ { L }$ , and (2) quantizing both local tensors. The reconstruction error is shown in Figure 4(a). We find that quantizing only the largest one (i.e., the red line) has the lowest error, followed by quantizing both (the green line). The quantization against the matrix (the blue one) has the largest quantization error. This demonstrates that the issue of quantization error for activations can be considerably mitigated by substituting matrix quantization with local tensor quantization.

Table 3: Results of in-context learning with different lengths of demonstrations.
<table><tr><td>Models</td><td>Exp</td><td>#Bits</td><td>ICL</td><td>Ag_news</td><td>Subj</td><td>Mr</td><td>Boolq</td><td>RTE</td><td>Average</td></tr><tr><td rowspan="12">OPT-1.3B</td><td>FP16</td><td>16</td><td>0-shot</td><td>58.0</td><td>62.9</td><td>79.5</td><td>60.5</td><td>52.7</td><td>62.7</td></tr><tr><td>FP16</td><td>16</td><td>2-shot</td><td>64.2</td><td>55.1</td><td>86.1</td><td>56.9</td><td>45.1</td><td>61.5</td></tr><tr><td>FP16</td><td>16</td><td>10-shot</td><td>70.0</td><td>64.4</td><td>84.0</td><td>64.7</td><td>50.2</td><td>66.7</td></tr><tr><td>RTN</td><td>4</td><td>2-shot</td><td></td><td>63.1</td><td>81.1</td><td>41.1</td><td>45.1</td><td></td></tr><tr><td>DecoQuant</td><td>4</td><td>2-shot</td><td>61.7 62.4</td><td>55.8</td><td>87.0</td><td>52.2</td><td>46.9</td><td>58.4 60.9</td></tr><tr><td>RTN</td><td>4</td><td>10-shōt</td><td>63.6</td><td>51.7</td><td>83.7</td><td>63.0</td><td>48.7</td><td>62.1</td></tr><tr><td>DecoQuant</td><td>4</td><td>10-shot</td><td>62.6</td><td>69.7</td><td>85.6</td><td>63.0</td><td>48.4</td><td>65.9</td></tr><tr><td>RTN</td><td>2</td><td>2-shot</td><td>33.0</td><td>51.7</td><td>55.0</td><td>41.8</td><td>53.1</td><td>46.9</td></tr><tr><td>DecoQuant</td><td>2</td><td>2-shot</td><td>40.4</td><td>56.0</td><td>52.1</td><td>49.7</td><td>52.3</td><td>51.6</td></tr><tr><td>RTN</td><td>2</td><td>10-shōt</td><td>37.6</td><td>53.7</td><td>52.7</td><td>39.0</td><td>49.5</td><td>51.4</td></tr><tr><td>DecoQuant</td><td>2</td><td>10-shot</td><td>42.4</td><td>65.6</td><td>54.1</td><td>43.4</td><td>52.7</td><td>66.2</td></tr><tr><td rowspan="10">OPT-6.7B</td><td>FP16</td><td>16</td><td>0-shot</td><td>70.9</td><td>61.4</td><td>64.3</td><td>63.5</td><td>60.3</td><td>64.1</td></tr><tr><td>FP16</td><td>16</td><td>2-shot</td><td>71.0</td><td>74.0</td><td>89.9</td><td>65.7</td><td>54.2</td><td>71.0</td></tr><tr><td>FP16</td><td>16</td><td>10-shot</td><td>53.3</td><td>89.8</td><td>86.8</td><td>65.7</td><td>57.0</td><td>70.5</td></tr><tr><td>RTN</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DecoQuant</td><td>4 4</td><td>2-shot 2-shot</td><td>68.7</td><td>66.1</td><td>81.1</td><td>67.5</td><td>53.8</td><td>69.2</td></tr><tr><td>RTN</td><td>4</td><td>10-shōt</td><td>71.6 53.1</td><td>73.6 76.8</td><td>87.0 83.7</td><td>68.2 64.7</td><td>53.4 54.5</td><td>71.2</td></tr><tr><td>DecoQuant</td><td>4</td><td>10-shot</td><td>54.6</td><td>92.4</td><td>85.6</td><td>62.6</td><td>51.3</td><td>67.2</td></tr><tr><td>RTN</td><td>2</td><td>2-shot</td><td>29.3</td><td>51.7</td><td>55.0</td><td>38.0</td><td>52.0</td><td>70.0</td></tr><tr><td>DecoQuant</td><td>2</td><td>2-shot</td><td>32.0</td><td>55.1</td><td>52.1</td><td>61.5</td><td>53.4</td><td>44.2 53.1</td></tr><tr><td>RTN</td><td>2</td><td>10-shōt</td><td>45.7</td><td>51.7</td><td>52.7</td><td>47.5</td><td>50.2</td><td>49.0</td></tr><tr><td>DecoQuant</td><td>2</td><td>10-shot</td><td>44.9</td><td>48.8</td><td>54.1</td><td>60.1</td><td>53.4</td><td>51.8</td></tr></table>

Analysis of Length of Local Tensors. We vary the MPO decomposition length (n) to assess its impact on quantization. Specifically, we choose $n = 2 , 3 .$ , 4 and the results are shown in Figure 4(b). This result shows that we can further enhance the quantization by extending the length of decompositions, which validates that the tensor decomposition process is indeed beneficial in mitigating the effect of outliers on quantization. However, the gains diminish as n increases. Notably, the improvement from n = 2 to n = 3 is higher than from n = 3 to n = 4. Considering effectiveness and efficiency, we select n = 2 for our experiments but recommend higher n for higher accuracy.

Comparison with Other Decomposition Methods. We compare the MPO decomposition in our approach with QR and SVD, which are popular decomposition methods. Results are in Figure 5. Our method outperforms SVD and QR, with significantly lower quantization errors (40.9 vs. 105.4 for SVD and 103.7 for QR at 4-bit precision) while introducing slight parameters. Additionally, MPO offers flexible tensor shapes, unlike QR and SVD which have fixed shapes, allowing us to balance accuracy and performance by adjusting quantization granularity.

![](images/9ea95579be089e43edaf2a79ab354d5c4c373618cf105039fcc70b0616bb0e64.jpg)

![](images/2cd021aeddd8d9ad0eeb6d7d7df2a5fcc697f043bd2b277f1b1c2659df9700a8.jpg)  
(a) Compression ratio.  
(b) Quantization error.  
Figure 5: Comparison between MPO with other decomposition methods.

## 4.4 Efficiency Analysis

Memory and Latency. In this section, we provide additional analysis to show that memory and latency costs can be significantly reduced by our approach in the decoding phase. Without loss of generality, we focus on LLaMA architecture (70B), a popular open-source decoder-only model, and the sequence length of 1k to 8k for evaluation. In Figure 6(a), we observe a significant reduction in the memory usage of the KV cache through compression, particularly evident when the sequence length reaches 6k. At this point, the cache size has matched the model size, while our cache remains under 30GB. Examining the latency in Figure 6(b), we note that DecoQuant achieves even lower latency. These findings indicate that despite Deco-Quant’s increased computational effort, it remains negligible when compared to the communication overhead saved, ultimately resulting in latency optimization.

![](images/a0ceab24c6083a37e8f1c6fc78e0c5b4051506c473f206a83e52d0ec0df431f4.jpg)  
(a) Memory Cost.

![](images/b54534ea730b5acad23a695f042b0746e9f4a0e13a20c1f67bd9059a57a89c69.jpg)  
(b) IO Latency.  
Figure 6: Efficiency of DecoQuant in terms of memory cost and IO latency.

## 5 Related Work

In this section, we present related works in three aspects as well as draw distinctions of our approach to existing literature.

Tensor Decomposition for Language Models. Matrix product operator decomposition was an efficient tensor decomposition method proposed by (Pirvu et al., 2010), then applied for compressing deep neural network (Gao et al., 2020a). In language modeling, tensor decomposition methods enable fine-grained model compression and tuning by decomposing the model’s weights, and show a very high potential since such operations are independent of the model’s structure. For example, in compression methods (Gao et al., 2020a; Sun et al., 2020; Gao et al., 2020b), in fine-tuning methods (Gao et al., 2023; Liu et al., 2021), in the field of pre-training (Gao et al., 2022; Liu et al., 2023a), and in the field of emergent ability (Liu et al., 2023b). However the references of tensor decomposition in parameter quantization have not been well studied, and the contribution of this paper bridges the gap.

Quantization for LLMs. Quantization methods have been shown to be effective in reducing the size of the model as well as speeding it up. For example method (Frantar et al., 2022) focuses on weight quantization while method (Dettmers et al., 2022) focuses on activation value quantization. Activation value quantization is considered more challenging due to the presence of outliers. To address this issue, Dettmers et al. (2022) cache the outlier values, while effective but still need to retain some of the FP16 values, thus making it difficult to achieve higher compression rates. A lot of quantization still needs to provide calibrated datasets, which may be difficult for some practical applications, e.g., users’ private data are usually not allowed to be publicly accessible. This paper, on the other hand, addresses the activation value quantization methods still under the condition of no calibration.

KV Cache Compression. The decoding part of the current inference phase of LLM is mainly memorybandwidth bound and an important approach is to alleviate the frequency of IO by compressing the KV cache. To achieve this goal, a straightforward approach is parameter quantization, but higher compression rates cannot be achieved due to the difficulty of activation value quantization. Another mainstream branch of research is concerned with reducing the number of tokens in the context, e.g., H2O (Zhang et al., 2023) by scores of attention. Other research is concerned with replacing the hard context with a soft prompt, e.g., AutoCompressors (Chevalier et al., 2023) by compressing the context into limited tokens. However, it may not be appropriate to choose to remove some tokens that are not important for the future only based on the existing context. Compared to the previous one, our approach keeps all tokens and ensures the integrity of the context.

## 6 Conclusion

In this paper, we proposed DecoQuant, a new datafree quantization method designed specifically for KV cache compression, to improve data generation efficiency. By first decomposing the KV cache matrices into local tensors, our approach only quantized the large local tensor with the major proportion of parameters in low-bit precision while maintained the small tensor in 16-bit precision. This approach can mitigate the quantization difficulty from the original matrix to the small local tensor, which effectively reduces the quantization error in KV cache compression. During inference, we also developed an efficient dequantization technique based on the fused kernel tailored for dequantization of DecoQuant to accelerate the generation process. Extensive experiments have demonstrated the effectiveness of the proposed approach in reducing the memory consumption of the KV cache and achieving competitive performance. For future work, we plan to explore the potential of leveraging Decoquant for scenarios where communication overhead plays a dominant role in LLM inference, specifically in the Splitwise technique where prefilling and decoding phases are in different nodes.

## 7 Limitations

While we present promising results and contributions to the field, it is not without its limitations. The performance of our methods may be influenced by external factors such as hardware configurations, software dependencies, and environmental conditions. A thorough analysis of these factors and their impact on the performance of our methods is essential for practical deployment and real-world applications. In addition, our approach may facilitate the deployment of large language models onto a wide range of edge devices, including personal smartphones. However, this expansion may raise social concerns. It is crucial to consider potential biases and fairness issues in real-world applications.

## 8 Acknowledgments

This work was partially supported by National Natural Science Foundation of China under Grant No. 62222215 and 62206299, and Beijing Natural Science Foundation under Grant No. L233008. Xin Zhao is the corresponding author.

## References

Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen. 2023. Adapting language models to compress contexts. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6- 10, 2023, pages 3829–3846. Association for Computational Linguistics.

Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. 2022. Llm.int8(): 8-bit matrix multiplication for transformers at scale. CoRR, abs/2208.07339.

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. 2022. GPTQ: accurate post-training quantization for generative pre-trained transformers. CoRR, abs/2210.17323.

Ze-Feng Gao, Song Cheng, Rong-Qiang He, Zhi-Yuan Xie, Hui-Hai Zhao, Zhong-Yi Lu, and Tao Xiang. 2020a. Compressing deep neural networks by matrix product operators. Physical Review Research, 2(2):023300.

Ze-Feng Gao, Peiyu Liu, Wayne Xin Zhao, Zhong-Yi Lu, and Ji-Rong Wen. 2022. Parameter-efficient mixture-of-experts architecture for pre-trained language models. In Proceedings of the 29th International Conference on Computational Linguistics, COLING 2022, Gyeongju, Republic of Korea, October 12-17, 2022, pages 3263–3273. International Committee on Computational Linguistics.

Ze-Feng Gao, Xingwei Sun, Lan Gao, Junfeng Li, and Zhong-Yi Lu. 2020b. Compressing lstm networks by matrix product operators. arXiv preprint arXiv:2012.11943.

Ze-Feng Gao, Kun Zhou, Peiyu Liu, Wayne Xin Zhao, and Ji-Rong Wen. 2023. Small pre-trained language models can be fine-tuned as large models via overparameterization. In Proceedings ofthe 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pages 3819–3834. Association for Computational Linguistics.

Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew G. Howard, Hartwig Adam, and Dmitry Kalenichenko. 2018. Quantization and training of neural networks for efficient integer-arithmetic-only inference. In 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018, pages 2704–2713. Computer Vision Foundation / IEEE Computer Society.

Tamara G. Kolda and Brett W. Bader. 2009. Tensor decompositions and applications. SIAM Rev., 51(3):455–500.

Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Dang, and Song Han. 2023. AWQ: activationaware weight quantization for LLM compression and acceleration. CoRR, abs/2306.00978.

Peiyu Liu, Ze-Feng Gao, Yushuo Chen, Xin Zhao, and Ji-Rong Wen. 2023a. Enhancing scalability of pre-trained language models via efficient parameter sharing. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pages 13771– 13785, Singapore. Association for Computational Linguistics.

Peiyu Liu, Ze-Feng Gao, Wayne Xin Zhao, Zhi-Yuan Xie, Zhong-Yi Lu, and Ji-Rong Wen. 2021. Enabling lightweight fine-tuning for pre-trained language model compression based on matrix product operators. In Proceedings ofthe 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021, pages 5388–5398. Association for Computational Linguistics.

Peiyu Liu, Zikang Liu, Ze-Feng Gao, Dawei Gao, Wayne Xin Zhao, Yaliang Li, Bolin Ding, and Ji-Rong Wen. 2023b. Do emergent abilities exist in

quantized large language models: An empirical study. arXiv preprint arXiv:2307.08072.

Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyrillidis, and Anshumali Shrivastava. 2023c. Scissorhands: Exploiting the persistence of importance hypothesis for LLM KV cache compression at test time. CoRR, abs/2305.17118.

Jesse Mu, Xiang Lisa Li, and Noah D. Goodman. 2023. Learning to compress prompts with gist tokens. CoRR, abs/2304.08467.

Denis Paperno, Germán Kruszewski, Angeliki Lazari dou, Quan Ngoc Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. 2016. The LAMBADA dataset: Word prediction requiring a broad discourse context. In Proceedings of the 54th Annual Meeting of the Associationfor Computational Linguistics, ACL 2016, August 7-12, 2016, Berlin, Germany, Volume 1: Long Papers. The Association for Computer Linguistics.

Pratyush Patel, Esha Choukse, Chaojie Zhang, Íñigo Goiri, Aashaka Shah, Saeed Maleki, and Ricardo Bianchini. 2023. Splitwise: Efficient generative LLM inference using phase splitting. CoRR, abs/2311.18677.

Bogdan Pirvu, Valentin Murg, J Ignacio Cirac, and Frank Verstraete. 2010. Matrix product operator representations. New Journal ofPhysics, 12(2):025012.

Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Anselm Levskaya, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean. 2022. Efficiently scaling transformer inference. CoRR, abs/2211.05102.

Stephan Rabanser, Oleksandr Shchur, and Stephan Günnemann. 2017. Introduction to tensor decompositions and their applications in machine learning. CoRR, abs/1711.10781.

Xingwei Sun, Ze-Feng Gao, Zhong-Yi Lu, Junfeng Li, and Yonghong Yan. 2020. A model compression method with matrix product operators for speech enhancement. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 28:2837–2847.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. Llama: Open and efficient foundation language models. CoRR, abs/2302.13971.

Guibin Wang, Yisong Lin, and Wei Yi. 2010. Kernel fusion: An effective method for better power efficiency on multithreaded GPU. In 2010 IEEE/ACM Int’l Conference on Green Computing and Communications, GreenCom 2010, & Int’l Conference on Cyber, Physical and Social Computing, CPSCom 2010, Hangzhou, China, December 18-20, 2010, pages 344– 350. IEEE Computer Society.

Guangxuan Xiao, Ji Lin, Mickaël Seznec, Hao Wu, Julien Demouth, and Song Han. 2023. Smoothquant: Accurate and efficient post-training quantization for large language models. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings ofMachine Learning Research, pages 38087–38099. PMLR.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona T. Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. 2022. OPT: open pre-trained transformer language models. CoRR, abs/2205.01068.

Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark W. Barrett, Zhangyang Wang, and Beidi Chen. 2023. H<sub>2</sub>o: Heavy-hitter oracle for efficient generative inference of large language models. CoRR, abs/2306.14048.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. 2023. A survey of large language models. CoRR, abs/2303.18223.

Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao Zhang. 2024. Distserve: Disaggregating prefill and decoding for goodput-optimized large language model serving.

## A Appendix

## A.1 Analysis of Outliers

The Interquartile Range (IQR) denotes the range between the 25th and 75th percentiles of the data. Outliers are often defined as data points that fall outside 1.5 times the IQR above the third quartile or below the first quartile. Thus, to better understand the benefit of the distribution of outliers after MPO decomposition, we investigated the IQR in other layers (1st, 16th, and 31st layers) and other structures (keys and values).

As seen in Table 4, we summarize the IQR of the target tensors. We observe, as discovered in Figure 1, that the IQR range of $\mathcal { T } _ { L }$ is the narrowest, followed by $\tau _ { S }$ , and the numerical ranges of the decomposed $\mathcal { T } _ { L }$ and $\tau _ { S }$ are much smaller than those of the matrix. This indicates that our method can be universally applied to all key/value tensors.

Table 4: Analysis of the outlier distributions in LLaMA-7B.
<table><tr><td colspan="3"></td><td colspan="3">Keys</td><td colspan="3">Values</td></tr><tr><td colspan="3"></td><td>Q1</td><td>Q3</td><td>IQR</td><td>Q1</td><td>Q3</td><td>IQR</td></tr><tr><td rowspan="9">LLaMA-7B</td><td rowspan="3">1st layer</td><td>matrix</td><td>-0.439</td><td>0.442</td><td>0.881</td><td>-0.013</td><td>0.013</td><td>0.026</td></tr><tr><td> $\mathcal { T } _ { L }$ </td><td>-0.027</td><td>0.027</td><td>0.055</td><td>-0.055</td><td>0.055</td><td>0.111</td></tr><tr><td> $\tau _ { S }$ </td><td>-0.155</td><td>0.156</td><td>0.312</td><td>-0.228</td><td>0.222</td><td>0.449</td></tr><tr><td rowspan="3">16th layer</td><td>matrix</td><td>-0.674</td><td>0.668</td><td>1.342</td><td>-0.307</td><td>0.306</td><td>0.613</td></tr><tr><td> $\mathcal { T } _ { L }$ </td><td>-0.056</td><td>0.056</td><td>0.112</td><td>-0.055</td><td>0.055</td><td>0.111</td></tr><tr><td> $\mathcal { T } _ { S }$ </td><td>-0.238</td><td>0.232</td><td>0.470</td><td>-0.228</td><td>0.222</td><td>0.449</td></tr><tr><td rowspan="3">32th layer</td><td>matrix</td><td>-0.685</td><td>0.672</td><td>1.357</td><td>-0.362</td><td>0.374</td><td>0.735</td></tr><tr><td> $\mathcal { T } _ { L }$ </td><td>-0.055</td><td>0.055</td><td>0.111</td><td>-0.055</td><td>0.055</td><td>0.111</td></tr><tr><td> $\tau _ { S }$ </td><td>-0.228</td><td>0.222</td><td>0.449</td><td>-0.228</td><td>0.222</td><td>0.449</td></tr></table>

<table><tr><td rowspan="2">Dataset</td><td colspan="2">Tokens per demonstration</td></tr><tr><td>OPT-based models</td><td> LLaMA-based models</td></tr><tr><td>Mr Subj</td><td>36</td><td>40 40</td></tr><tr><td>Ag_news</td><td>40 65</td><td>75</td></tr><tr><td>RTE</td><td>75</td><td>85</td></tr><tr><td>Boolq</td><td>165</td><td>170</td></tr></table>

Table 5: Details of the datasets used for in-context learning. “Tokens per demonstration” indicates how long the demonstrations are for the average example.

## A.2 Details of the datasets

In our in-context learning experiments, the length of the KV cache can be measured using the length of demonstrations since these demonstrations constitute the majority of the prefilling process. Therefore, we report the token per demonstration for five datasets to represent this, as shown in the Table 5. We find that the datasets we used covered a range of context lengths, including longer contexts (Boolq), shorter contexts (Mr and Subj), and moderate contexts (Ag\_news and RTE).