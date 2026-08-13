# LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression

Huiqiang Jiang, Qianhui Wu, Xufang Luo,

Dongsheng Li, Chin-Yew Lin, Yuqing Yang, Lili Qiu

Microsoft Corporation

{hjiang,qianhuiwu,xufluo,dongsli,cyl,yuqyang,liliqiu}@microsoft.com

## Abstract

In long context scenarios, large language models (LLMs) face three main challenges: higher computational cost, performance reduction, and position bias. Research indicates that LLM performance hinges on the density and position of key information in the input prompt. Inspired by these findings, we propose LongLLM-Lingua for prompt compression towards improving LLMs’ perception of the key information to simultaneously address the three challenges. Our extensive evaluation across various long context scenarios demonstrates that LongLLMLingua not only enhances performance but also significantly reduces costs and latency. For instance, in the NaturalQuestions benchmark, LongLLMLingua boosts performance by up to 21.4% with around 4x fewer tokens in GPT-3.5-Turbo, leading to substantial cost savings. It achieves a 94.0% cost reduction in the LooGLE benchmark. Moreover, when compressing prompts of about 10k tokens at ratios of 2x-6x, LongLLMLingua can accelerate end-to-end latency by 1.4x-2.6x. <sup>1</sup>

## 1 Introduction

Large language models (LLMs) have revolutionized user-oriented language technologies and are serving as crucial components in more and more applications. Carefully designing prompts is necessary to achieve better performance in specific downstream tasks. The commonly used technologies such as In-Context Learning (ICL) (Min et al., 2022; Dong et al., 2023), Retrieval Augment Generation (RAG) (Lewis et al., 2020; Asai et al., 2024), and Multi-turn Agent (Shen et al., 2024; Park et al., 2023; Wu et al., 2023a) are driving prompts to be increasingly longer, even reaching thousands of tokens. Scenarios such as multi-document question answering, code completion, and document summarization also necessitate the processing of long contexts.

There are three main challenges when LLMs are used in long context scenarios: (1) Higher computational costs, encompassing both financial and latency expenses. (2) Longer prompts introduce irrelevant and redundant information, which can weaken LLMs’ performance (Shi et al., 2023), as illustrated in Figure 1a. (3) LLMs exhibit position bias (Kamradt, 2023), also known as the "lost in the middle" issue (Liu et al., 2024), suggesting that the placement of key information within the prompt significantly affects LLMs’ performance. This is demonstrated by the purple curve in Figure 1b.

Inspired by these observations, we propose LongLLMLingua to address the three challenges. Specifically, we use LLMLingua (Jiang et al., 2023a) as the backbone for prompt compression to address the first challenge, i.e., reduce cost and latency. However, in the case of long contexts, the distribution of question-relevant key information in the prompt is generally dynamic and sparse. Existing prompt compression methods like LLMLingua (Jiang et al., 2023a) and Selective-Context (Li et al., 2023c) that often fail to consider question during compression, resulting in retention of excessive noise and decreased performance. LongLLM-Lingua aims to improve LLMs’ perception of key information pertinent to the question, thereby overcoming the noise and position bias issues in long contexts, shown in Figure 1b. The underlying principle of LongLLMLingua is that small LM are inherently capable of capturing the distribution of key information relevant to a given question.

Our main contributions are five-fold: (1) We propose a question-aware coarse-to-fine compression method to improve the key information density in the prompt (Sec. 4.1); (2) We introduce a document reordering strategy to minimize position bias in LLMs. (Sec. 4.2); (3) We establish dynamic compression ratios for precise control between coarse and fine compression levels (Sec. 4.3); (4) We propose a post-compression subsequence recovery strategy to improve the integrity of the key information (4.4). (5) We evaluate LongLLMLingua across five benchmarks, i.e., NaturalQuestions (Liu et al., 2024), LongBench (Bai et al., 2023), ZeroSCROLLS (Shaham et al., 2023), MuSicQue (Trivedi et al., 2022), and LooGLE (Li et al., 2023b), covering a variety of long context scenarios. Experimental results reveal that LongLLMLingua’s compressed prompts outperform original prompts in terms of performance, cost efficiency, and system latency.

![](images/4e858973f937ed1c6f50423a0836cfb3b084e7bda8363e3621ecea51fc71fd6b.jpg)  
(a) Performance v.s. Document Number

![](images/d8481e3a90b4fcdcd3779af788a89ad5a6ac0a6358c35d50ef6c8758a1d52a0c.jpg)  
(b) Performance v.s. Key Information Position  
Figure 1: (a) LLMs’ performance in downstream tasks decreases with increased noise in prompts. In this case, we keep $k$ most relevant documents/paragraphs based on the ground-truth or LongLLMLingua $r _ { k }$ . A larger k implies more noise introduced into the prompt. To improve the key information density in the prompt, we present question-aware coarse-to-fine compression. (b) LLMs’ ability to capture the relevant information depends on their positions in the prompt. To reduce information loss in the middle, we introduce a document reordering mechanism.

## 2 Problem Formulation

Following LLMLingua (Jiang et al., 2023a), we use $\mathbf { x } = ( \mathbf { x } ^ { \mathrm { { i n s } } } , \mathbf { x } _ { 1 } ^ { \mathrm { { d o c } } } , \cdot \cdot \cdot , \mathbf { x } _ { K } ^ { \mathrm { { d o c } } } , \mathbf { x } ^ { \mathrm { { q u e } } } )$ to represent a prompt, including the instruction $\mathbf { x } ^ { \mathrm { i n s } }$ , K documents $\mathbf { x } _ { i } ^ { \mathrm { { d o c } } }$ , and the question $\mathbf { x } ^ { \mathrm { q u e } }$ . However, this definition can be adjusted for specific scenarios. The objective of a prompt compression system can be formulated as:

$$
\operatorname* { m i n } _ { \widetilde { \mathbf { x } } } D _ { \phi } \left( \mathbf { y } , \widetilde { \mathbf { y } } \right) + \lambda \| \widetilde { \mathbf { x } } \| _ { 0 } ,\tag{1}
$$

where $\widetilde { \mathbf { x } }$ represents the compressed prompt, a tokenelevel subsequence of x. y and $\widetilde { \mathbf { y } }$ represent the eLLM-generated results from x and x, respectively. $D _ { \phi }$ emeasures the distance function, such as KL divergence. λ serves as a hyper-parameter balancing the compression ratio. Additionally, this study explores a permutation operation space over the K documents $( \mathbf { x } _ { 1 } ^ { \mathrm { d o c } } , \cdot \cdot \cdot , \mathbf { x } _ { K } ^ { \mathrm { d o c } } )$ for joint optimization.

## 3 Preliminary: LLMLingua

LLMLingua (Jiang et al., 2023a) utilizes a small language model $\mathcal { M } _ { S }$ to evaluate the perplexity of each prompt token, removing those with lower perplexities. This method is premised on the idea that tokens with lower perplexities have a negligible effect on the language model’s overall entropy gain, implying their removal slightly impacts the LLMs’ contextual understanding. This process is viewed as an application of $\ " \mathrm { L M }$ is Compression" (Delétang et al., 2023). LLMLingua include three key components: budget controller, iterative token-level prompt compression, and distribution alignment, highlighted by italic text in Figure 2. The budget controller assigns varying compression ratios to different parts of the prompt (i.e., instruction, demonstrations, question), implementing coarse-level prompt compression. Subsequent steps involve dividing intermediate results into segments and applying token-level compression iteratively, where each token’s perplexity based on preceding compressed segments. To aware different target LLMs, LLMLingua fine-tunes $\mathcal { M } _ { S }$ using data from the target LLM.

## 4 LongLLMLingua

LongLLMLingua builds on LLMLingua to better compress prompts in long context scenorias. It tackles three main issues in handling lengthy contexts, as introduced in Sec. 1. This approach focuses on making LLMs more effective at recognizing key information related to the question in the prompt. It encompasses three perspectives and further incorporates a subsequence recovery strategy, as shown in Figure 2, to enhance the accuracy and reliability of the information provided to users. In this section, we detail how each part of LongLLMLingua works to improve the LLMs deal with long context.

![](images/04dd838092eabe8d367aacc3fdeb02a28bff680dc92118b69988e3c6d66f9e80.jpg)  
Figure 2: Framework of LongLLMLingua. Gray Italic content: As in LLMLingua.

## 4.1 How to improve key information density in the prompt?

Question-Aware Coarse-Grained Compression In coarse-grained compression, we aim to figure out a metric $r _ { k }$ to evaluate the importance of each document $\mathbf { x } _ { k } ^ { \mathrm { d o c } } ~ = ~ \{ x _ { k , i } ^ { \mathrm { d o c } } \} _ { i = 1 } ^ { N _ { k } }$ , where $N _ { k }$ is the number of tokens in $\mathbf { x } _ { k } ^ { \mathrm { { d o c } } }$ We only keep $\mathbf { x } _ { k } ^ { \mathrm { { d o c } } }$ with higher $r _ { k }$ as the intermediate compressed results. One approach to improve key information density in the compressed prompts is to calculate document-level perplexity conditioned on the question $p ( \mathbf { x } _ { k } ^ { \mathrm { d o c } } | \mathbf { x } ^ { \mathrm { q u e } } )$ . However, this method may not be effective because documents often contain a significant amount of irrelevant information. Even when conditioned on $\mathbf { x } ^ { \mathrm { q u e } }$ , the perplexity scores computed for entire documents may not be sufficiently distinct, rendering them an inadequate metric for document-level compression.

We propose to use the perplexity of the question $\mathbf { x } ^ { \mathrm { q u e } }$ conditioned on different contexts $\mathbf { x } _ { k } ^ { \mathrm { { d o c } } }$ $p ( \mathbf { x } ^ { \mathrm { q u e } } | \mathbf { x } _ { k } ^ { \mathrm { d o c } } )$ to represent the association between them. We also append a restrictive statement<sup>2</sup> x<sup>restrict</sup> after $\mathbf { x } ^ { \mathrm { q u e } }$ to strengthen the interconnection of $\mathbf { x } ^ { \mathrm { q u e } }$ and $\mathbf { x } _ { k } ^ { \mathrm { { d o c } } }$ . It can be regarded as a regularization term that mitigates the impact of hallucinations. This can be formulated as:

$$
r _ { k } = - \frac { 1 } { N _ { c } } \sum _ { i } ^ { N _ { c } } \log p ( x _ { i } ^ { \mathrm { q u e , r e s t r i c t } } | \mathbf { x } _ { k } ^ { \mathrm { d o c } } ) ,\tag{2}
$$

where $x _ { i } ^ { \mathrm { q u e , r e s t r i c t } }$ is the i-th token in the concatenated sequence of $\mathbf { x } ^ { \mathrm { q u e } }$ and $\mathbf { x } ^ { \mathrm { r e s t r i c t } }$ and $N _ { c }$ in the number of tokens.

Figure 3a displays the recall distribution of different retrieval methods, including traditional relevance methos (BM25, Gzip (Jiang et al., 2023b)), embedding-based methods (OpenAI-embedding, Voyageai<sup>3</sup>, BGE-large-en v1.5 (Xiao et al., 2023), Sentence-BERT (Reimers and Gurevych, 2019), Jina (Günther et al., 2023)), and reranker methods (Cohere-Rerank<sup>4</sup>, BGE-llmembeder, BGE-Rankerlarge), which demonstrates that our coarse-level compression approach achieves the highest recall with different numbers of retained documents, suggesting that it preserves the most key information from the contexts in the compressed results.

Question-Aware Fine-Grained Compression In fine-grained compression, we assess the importance of each token in the instruction $\mathbf { x } ^ { \mathrm { i n s } }$ , the question $\mathbf { x } ^ { \mathrm { q u e } }$ , and $K ^ { \prime }$ documents $\{ \mathbf { x } _ { i } ^ { \mathrm { d o c } } \} _ { i = 1 } ^ { K ^ { \prime } }$ retained after coarse-grained compression. We incorporate the iterative compression mechanism following LLM-Lingua and directly calculate token perplexities to compress $\mathbf { x } ^ { \mathrm { i n s } }$ and $\mathbf { x } ^ { \mathrm { q u e } }$ . In this section, we investigate how to make the fine-grained token-level compression over $\{ \mathbf { x } _ { k } ^ { \mathrm { d o c } } \} _ { k = 1 } ^ { K ^ { \prime } }$ aware of the question $\mathbf { x } ^ { \mathrm { q u e } }$ , so that the compressed results could contain more question-relevant key information.

![](images/92c1363fdfdad907bf793e070db20ced77635c6ed05087e04e0578f48eab20d3.jpg)  
(a) Recall Distribution

![](images/fb74bb5517bae2926c33d5e9ef380491f620f1bec106e03c5b366db154819732.jpg)  
(b) Perplexity Distribution (5th)  
Figure 3: (a) Comparison of recall on NaturalQuestions Multi-documemnt QA dataset, which increases from top to bottom in terms of Recall@1. Different colors represent different types of methods. Among them, yellow represents traditional relevance methods, green signifies embedding-based methods, and red denotes rerank-based methods. (b) Comparison between perplexities and contrastive perplexities of tokens in the prompt from Multi-documemnt QA dataset. The document containing the ground-truth information is located in the 5th position. More results on position can be found in the Appendix C.1.

A straightforward solution for the awareness of $\mathbf { x } ^ { \mathrm { q u e } }$ is to simply concatenate it at the beginning of the whole context. However, this will result in low perplexities of relevant tokens in the context following the condition of question $\mathbf { x } ^ { \mathrm { q u e } }$ , further reducing their differentiation from other tokens.

In this paper, we propose contrastive perplexity, $i . e .$ , the distribution shift caused by the condition of the question, to represent the association between the token and the question. The contrastive perplexity based importance metric $s _ { i }$ for each token $x _ { i }$ in $\{ \mathbf { \bar { x } } _ { k } ^ { \mathrm { d o c } } \} _ { k = 1 } ^ { K ^ { \prime } }$ can be formulated as:

$$
s _ { i } = \mathrm { p e r p l e x i t y } ( x _ { i } | \boldsymbol { x } _ { < i } ) - \mathrm { p e r p l e x i t y } ( x _ { i } | \boldsymbol { x } ^ { \mathrm { q u e } } , \boldsymbol { x } _ { < i } ) .\tag{3}
$$

Additionally, we provide the derivation of its mathematical significance in the Appendix A, concluding that it is equivalent to conditional pointwise mutual information (Church and Hanks, 1989).

Figure 3b illustrates the difference between perplexities and contrastive perplexities. The distribution of perplexities appears random, making it challenging to extract information related to the question. However, tokens with high contrastive perplexities tend to cluster near the ground-truth document, which contains information relevant to the question. This suggests that the proposed contrastive perplexity can better distinguish tokens relevant to the question, thus improving the key information density in the compressed results.

## 4.2 How to reduce information loss in the middle?

As demonstrated in Figure 1b, LLM achieves the highest performance when relevant information occurs at the beginning and significantly degrades if relevant information is located in the middle of long contexts. After the coarse-grained compression, we have obtained a set of documents $\{ \mathbf { x } _ { k } ^ { \mathrm { d o c } } \} _ { k = 1 } ^ { K ^ { \prime } }$ with their corresponding importance scores $\{ r _ { k } \} _ { k = 1 } ^ { \tilde { K } ^ { \prime } }$ indicating their association with the question $\mathbf { x } ^ { \mathrm { q u e } }$ Therefore, we reorder documents using their importance scores to better leverage LLMs’ information perception difference in positions:

$$
\begin{array} { r l } & { \left( \mathbf { x } ^ { \mathrm { i n s } } , \mathbf { x } _ { 1 } ^ { \mathrm { d o c } } , \cdot \cdot \cdot , \mathbf { x } _ { K ^ { \prime } } ^ { \mathrm { d o c } } , \mathbf { x } ^ { \mathrm { q u e } } \right) \xrightarrow { r _ { k } } } \\ & { \qquad \quad \left( \mathbf { x } ^ { \mathrm { i n s } } , \mathbf { x } _ { r 1 } ^ { \mathrm { d o c } } , \cdot \cdot \cdot , \mathbf { x } _ { r K ^ { \prime } } ^ { \mathrm { d o c } } , \mathbf { x } ^ { \mathrm { q u e } } \right) } \end{array}\tag{4}
$$

## 4.3 How to achieve adaptive granular control during compression?

In fine-grained compression, LLMLingua applies the same compression ratio over all documents obtained from budget controller. However, the key information density of different documents is different. The more relevant to the question a document is, the more budget (i.e., lower compression ratio) we should allocate to it. Therefore, we bridge coarse-grained compression to fine-grained compression and use the importance scores $\{ r _ { k } \} _ { k = 1 } ^ { K ^ { \prime } }$ obtained from coarse-grained compression to guide the budget allocation in fine-grained compression. In this way, we can achieve adaptive granular control on the whole.

<table><tr><td>Document [1](Title: List of Nobel laureates in Physics) The first Nobel Prize in Physics was awarded in 1901 to {Wilhelm Conrad Röntgen}Wilhelm Con rad Rö nt gen}, of Germany,... Original Prompt</td><td>Document [1](Title: List of Nobelates in Physics) The first Nobel1 {WilhelmgenX{Wilhelm gen}, of, who</td><td>{Wilhelmgen} {Wilhelm gen}</td></tr></table>

Figure 4: The example of Subsequence Recovery, the red text represents the original text, and the blue text is the result after using the LLaMA 2-7B tokenizer.

Specifically, we first determine the initial budget for the retained documents<sup>5</sup> τ<sup>doc</sup> using the budget controller of LLMLingua. During fine-grained compression, we follow the iterative token-level compression algorithm in LLMLingua but dynamically assign the compression budget $\tau _ { k } ^ { \mathrm { d o c } }$ to each document $\mathbf { x } _ { k } ^ { \mathrm { { d o c } } }$ according to the ranking index $I ( r _ { k } )$ (e.g., 0, 1) of its importance score from the coarsegrained compression. In this paper, we employ a linear scheduler for the adaptive allocation. Budget of each token $x _ { i }$ can be formulated as:

$$
\begin{array} { r l r } {  { \tau _ { i } = \tau _ { k } ^ { \mathrm { d o c } } , \quad \forall x _ { i } \in \mathbf { x } _ { k } ^ { \mathrm { d o c } } , } } \\ & { } & { \tau _ { k } ^ { \mathrm { d o c } } = \operatorname* { m a x } ( \operatorname* { m i n } ( ( 1 - \frac { 2 I ( r _ { k } ) } { K ^ { \prime } } ) \delta \tau + \tau ^ { \mathrm { d o c } } , 1 ) , 0 ) , } \end{array}\tag{5}
$$

where i and k is the index of token and document, $K ^ { \prime }$ denotes the number of documents, and δτ is a hyper-parameter that controls the overall budget for dynamic allocation.

## 4.4 How to improve the integrity of key information?

During the generation process, LLMs tend to replicate entities found in the prompt, such as names, places, and organizations. Compressing these entities at the token level doesn’t affect the LLMs understanding of semantic content but can lead to errors in the generated content.

Therefore, we propose a subsequence recovery method to restore the original content in LLMs responses. This method relies on the subsequence relationship among tokens in the original prompt, compressed prompt, and LLMs’ response, as shown in Figure 4.

The overall procedure includes: i) Iterate through tokens y<sub>l</sub> in LLMs’ response and select the longest substring $\widetilde { \pmb { y } } _ { \mathrm { k e y } , l } ~ = ~ \{ y _ { l } , y _ { l + 1 } , . . . , y _ { r } \}$ ethat appears in the compressed prompt x. ii) eFind the maximum common shortest subsequence $\pmb { x } _ { i , j } = \{ x _ { i } , x _ { i + 1 } , . . . , x _ { j } \}$ in the original prompt $^ { x , }$ corresponding to the representation $\widetilde { \boldsymbol { y } } _ { \mathrm { k e y } , l }$ in the eoriginal prompt (accelerated using prefix trees or sequence automata). iii) Replace the matched tokens $\widetilde { \pmb { y } } _ { \mathrm { k e y } , l }$ in LLMs’ response with the correspondeing subsequence $\boldsymbol { x } _ { i , j }$ from the original prompt. For more details, please refer to Algorithm 1.

Algorithm 1 Token-level Subsquence Recovery   
Algorithm   
Input: The original prompt x; the compressed prompt x; the   
generation response of LLMs y.   
1: Set the final response list $y _ { \mathrm { r e c } } = \phi ,$ the left token index of   
subsquence l to 0.   
2: while l < y.len() do   
3: if Substring y<sub>l</sub> x then   
4: eFind the longer substring $\widetilde { { \bf y } } _ { \mathrm { k e y } , l } ~ = ~ \{ y _ { l } , y _ { l + 1 } ,$   
$y _ { r } \} \in { \widetilde { \pmb x } } .$   
5: Find the maximum common shortest subsequence   
$\pmb { x } _ { i , j } = \{ x _ { i } , x _ { i + 1 } , . . . , x _ { j } \}$ in the original prompt x.   
6: Add the subsequence $\pmb { x } _ { i , j } = \{ \bar { x _ { i } } , x _ { i + 1 } , . . . , x _ { j } \}$   
to the response y<sub>rec</sub>.   
7: Set the left index l to $r + 1 .$   
8: else   
9: Add the token y<sub>l</sub> to the response $_ { { \pmb { y } } _ { \mathrm { r e c } } . }$   
10: Set the left index l to l + 1.   
11: end if   
12: end while   
Output: The final response list $\scriptstyle { \pmb { y } } _ { \mathrm { r e c } } .$

## 5 Experiments

Here, we investigate: (1) How effective is LongLLMLingua? (2) How efficient is LongLLM-Lingua?

Implementation details In this paper, we use GPT-3.5-Turbo-0613<sup>6</sup> and LongChat-13B-16k as the target LLMs, both accessible via OpenAI<sup>7</sup> and HuggingFace<sup>8</sup>. To ensure stable and reproducible results, we employ greedy decoding and set the temperature to 0 in all experiments. For the small language models used for compression, we apply LLaMA-2-7B-Chat<sup>9</sup>, which has been aligned by supervised fine-tuning and RLHF. We implement our approach with PyTorch 1.13.1 and Hugging-Face Transformers. We set up hyperparameters following LLMLingua except for the segment size used in iterative token-level compression set to 200 here. More details are provided in Appendix B.

Dataset & evaluation metric We use NaturalQuestions for the multi-document QA task, and use LongBench and ZeroSCROLLS for general long context scenarios. We also test on multihop QA tasks using MuSiQue dataset (Trivedi et al., 2022), and long dependency QA tasks using LooGLE benchmark (Li et al., 2023b). Please refer to Appendix C for more details on datasets.

Baselines We include two sets of baselines in following experiments:

(i) Retrieval-based Methods. We assess the question-document association in the prompt using five SoTA retrieval methods: BM25, Gzip (Jiang et al., 2023b), SentenceBERT (Reimers and Gurevych, 2019), OpenAI Embedding, and the LongLLMLingua ranker’s important metric $r _ { k }$ for coarse-grained compression. Notably, embedding model-based compression mirrors the method in Xu et al. (2024). We remove low-relevance sentences or paragraphs to meet compression limits, maintaining the original document sequence.

(ii) Compression-based Methods. We compare our approach with two state-of-art methods for prompt compression, i.e., Selective Context (Li et al., 2023c) and LLMLingua (Jiang et al., 2023a). Both methods employ LLaMA-2-7B-Chat as the small language model for compression. In LLM-Lingua, a coarse-to-fine approach is used to handle constraints of compression ratio: the original prompt is first compressed to k times the constraint at a coarse level, where k is the granular control coefficient; token-level is then performed to reach the overall constraint. Our method follows the same coarse-to-fine logic to achieve the constraint.

Main results Table 1 and 2 present the performance of various methods under different compression constraints. There are multiple observations and conclusions: (1) Our LongLLMLingua achieves the best performance across different tasks and constraints of compression ratios. Compared to the original prompt, our compressed prompt can derive higher performance with much lower cost. For example, LongLLMLingua gains a performance boost of 21.4% on NaturalQuestions with the ground-truth document at the 10th position, while the number of tokens input to GPT3.5-Turbo is 4x less. (2) Compression-based methods like Selective Context (Li et al., 2023c) and LLMLin gua (Jiang et al., 2023a) perform poorly on most tasks, especially those with abundant irrelevant information in the original prompt. This is due to their pure information entropy based compres sion mechanism, which includes too much noise in the compressed results and even leads to performance worse than the zero-shot setting, e.g., on NaturalQuestions. (3) Retrieval-based meth ods work well with low compression ratios. However, their performance declines as the compres sion progresses, e.g., 2x 4x; 3000 tokens 2000 tokens. This may be caused by the decreased recall. Figure 3a is the illustration of cases on NaturalQuestions. (4) LongLLMLingua as well as our coarse-grained compression metric $r _ { k }$ only is much more robust than all other baselines un der different tasks and compression constraints. With the increase of the compression ratio, e.g., 2x 4x, LongLLMLingua even achieves a lit tle performance gain. We mainly owe this to the question-aware coarse-to-fine compression, which can better figure out the key information and reach a higher key information density with a higher compression ratio. (5) The proposed reordering method helps in not only our approach but also other baselines, well demonstrating its effective ness. (6) Compared to the results with a 2,000 tokens constraint, overall performance of 3,000 tokens has improved. LongLLMLingua sees an increase of 1.2 points in average score and a 1.6x speedup in end-to-end latency. In this scenario, the recall rates of retrieval-based methods have increased, leading to a significant improvement in their accuracy. For example, BM25 achieves an average score of 48.9.

In addition, we also present experimental results on datasets such as MuSicQue, LooGLE, ZERO-SCROLLS, etc., in Appendix C.

Ablation study To evaluate the contributions of different components in LongLLMLingua, we introduce following variants of it for ablation study. (1) Variants about Question-aware Coarsegrained Compression, include: ours w/o Questionawareness, which calculates question-text relevance $r _ { k }$ using information entropy in LLMLingua, ours w/ SBERT, which employs SBERT to compute $r _ { k } .$ , ours w/ $p ( \mathbf { x } _ { k } ^ { \mathrm { d o c } } | x _ { i } ^ { \mathrm { q u e , r e s t r i c t } } )$ , which replace $p ( x _ { i } ^ { \mathrm { q u e , r e s t r i c t } } | \mathbf { x } _ { k } ^ { \mathrm { d o c } } )$ with $p ( \mathbf { x } _ { k } ^ { \mathrm { d o c } } | x _ { i } ^ { \mathrm { q u e , r e s t r i c t } } )$ in Eq. (2), and ours w/o restrict, which only calculates the conditional probability corresponding to $x ^ { \mathrm { q u e } }$ . (2) Ours w/o Question-aware Fine-grained, which disregards Eq. (3) and only applies Iterative Token-level Prompt Compression as LLMLingua. (3) Ours w/o Dynamic Compression Ratio, where all documents share the same compression ratio in fine-grained compression. (4) Ours w/o and (5) LLMLingua w/ Subsequence Recovery, which either removes or adds the post-processing subsequence recovery strategy. (6) Ours w/ GPT2-small, which uses the GPT2-small model as the $\mathcal { M } _ { S }$

<table><tr><td rowspan="2">Methods</td><td rowspan="2">1st</td><td colspan="5">GPT3.5-Turbo</td><td rowspan="2">1st</td><td colspan="4">LongChat-13b</td><td rowspan="2"></td><td colspan="2">Length</td><td rowspan="2">Latency</td><td colspan="2">1/τ |Latency Speedup</td></tr><tr><td>5th</td><td></td><td>10th 15th 20th</td><td></td><td>Reorder</td><td>5th</td><td>10th 15th 20th</td><td></td><td></td><td>Reorder | Tokens</td><td></td><td></td><td></td></tr><tr><td colspan="10">2x constraint</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Retrieval-based Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BM25</td><td>53.7</td><td>49.3</td><td>47.9</td><td>49.946.9</td><td></td><td>50.3</td><td></td><td>50.9 44.9 44.1</td><td></td><td>42.9</td><td>43.2</td><td>46.0</td><td></td><td>1,545</td><td>1.9x</td><td>2.1</td><td>1.9x</td></tr><tr><td>Gzip</td><td>64.6</td><td>63.8</td><td>60.5</td><td>58.3</td><td>57.3</td><td>64.4</td><td>61.9</td><td>55.7</td><td>52.7</td><td>50.8</td><td>50.9</td><td></td><td>59.3</td><td>1,567</td><td>1.9x</td><td>2.1</td><td>1.9x</td></tr><tr><td>SBERT</td><td>72.5</td><td>67.9</td><td>63.3</td><td>65.0</td><td>66.2</td><td>68.7</td><td>65.8</td><td>57.5</td><td>54.9</td><td>53.4</td><td>55.7</td><td></td><td>61.4</td><td>1,549</td><td>1.9x</td><td>2.2</td><td>1.9x</td></tr><tr><td>OpenAI</td><td>73.0</td><td>65.6</td><td>66.5</td><td>65.4</td><td>65.5</td><td>69.9</td><td>65.9</td><td>57.5</td><td>56.2</td><td>54.2</td><td>55.7</td><td></td><td>61.7</td><td>1,550</td><td>1.9x</td><td>4.9</td><td>0.8x</td></tr><tr><td>LongLLMLingua rk</td><td>73.9</td><td>67.7</td><td>68.7</td><td>66.0</td><td>65.6</td><td>74.3</td><td>68.5</td><td>59.1</td><td>56.8</td><td>55.3</td><td>56.9</td><td>65.2</td><td></td><td>1,548</td><td>1.9x</td><td>2.3</td><td>1.8x</td></tr><tr><td colspan="10">Compression-based Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Selective-Context</td><td></td><td>45.4 39.0</td><td></td><td>33.8 33.5</td><td>41.5</td><td></td><td></td><td>53.2 26.3</td><td>25.4 24.2 33.3</td><td></td><td></td><td></td><td></td><td>1,478</td><td>2.0x</td><td>7.4</td><td>0.6x</td></tr><tr><td>LLMLingua</td><td>39.7</td><td>39.5</td><td>40.4</td><td>37.1</td><td>42.3</td><td>41.5</td><td>38.7</td><td>37.3</td><td>35.7</td><td>34.1</td><td>37.5</td><td>37.1</td><td></td><td>1,410</td><td>2.1x</td><td>2.8</td><td>1.5x</td></tr><tr><td>LongLLMLingua</td><td>77.2</td><td>72.9</td><td>70.8</td><td>70.5</td><td>70.6</td><td>76.2</td><td>68.7</td><td>59.4</td><td>57.3</td><td>55.9</td><td>58.4</td><td>66.1</td><td></td><td>1,429</td><td>2.1x</td><td>2.9</td><td>1.4x</td></tr><tr><td colspan="10">4x constraint</td><td colspan="10"></td></tr><tr><td colspan="10">Retrieval-based Methods</td><td colspan="10"></td></tr><tr><td colspan="10"></td><td colspan="10"></td></tr><tr><td>BM25 Gzip</td><td>40.6</td><td>38.6</td><td></td><td>38.2 37.4</td><td>36.6</td><td></td><td>36.3</td><td>39.5 37.5 52.9</td><td>36.8</td><td>36.4 35.5 </td><td></td><td></td><td>37.7</td><td>798 824</td><td>3.7x 3.6x</td><td>1.5</td><td></td><td>2.7x 2.7x</td></tr><tr><td>SBERT</td><td>63.1</td><td>61.0</td><td>59.8</td><td>61.1</td><td>60.1</td><td></td><td>62.3</td><td>57.6</td><td>51.0</td><td>50.1</td><td></td><td>50.4</td><td>57.2</td><td></td><td>3.6x</td><td></td><td>1.5 1.6</td><td>2.5x</td></tr><tr><td>OpenAI</td><td>66.9 63.8</td><td>61.1</td><td>59.0</td><td>61.2</td><td>60.3</td><td></td><td>64.4 63.7</td><td>62.6 56.6</td><td>55.1</td><td></td><td>53.9</td><td>55.0</td><td>59.1</td><td>808</td><td></td><td></td><td>4.3</td><td>1.0x</td></tr><tr><td>LongLLMLingua rk</td><td>71.1</td><td>64.6 70.7</td><td>65.4 69.3</td><td>64.1 68.7</td><td>63.7 68.5</td><td></td><td>71.5</td><td>61.2 56.0 67.8 59.4</td><td>55.1 57.7</td><td></td><td>54.4</td><td>55.0</td><td>58.8 64.0</td><td>804 807</td><td>3.7x 3.7x</td><td></td><td>1.7</td><td>2.4x</td></tr><tr><td colspan="10"></td><td colspan="7">57.7 58.6</td><td></td></tr><tr><td>Compression-based Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Selective-Context LLMLingua</td><td>31.4 19.5</td><td>27.5</td><td>24.7 23.5</td><td>26.5</td><td>24.1</td><td>43.8 30.0</td><td>27.0</td><td>38.2 17.2 32.1 30.8</td><td></td><td>15.9 29.9</td><td>16.0 28.9</td><td>27.3 32.4</td><td>30.5</td><td>791 775</td><td></td><td>3.7x 3.8x</td><td>6.8 1.8</td><td>0.6x 2.2x</td></tr><tr><td></td><td>25.5</td><td></td><td>71.2</td><td></td><td>71.2</td><td>74.7</td><td>75.5</td><td>68.7 60.5</td><td></td><td>59.3</td><td></td><td></td><td>66.7</td><td>748</td><td></td><td>3.9x</td><td>2.1</td><td>2.0x</td></tr><tr><td>LongLLMLingua</td><td colspan="10">75.071.8</td><td colspan="7">58.3 61.3</td></tr><tr><td>Original Prompt</td><td>75.7 57.3</td><td></td><td>54.1</td><td></td><td>55.4</td><td>63.1</td><td></td><td>68.657.4</td><td>55.3</td><td></td><td>52.5</td><td>55.0</td><td></td><td>2,946</td><td></td><td></td><td>4.1</td><td></td></tr></table>

Table 1: Performance of different methods with different compression ratios (raw size / compressed $\mathrm { s i z e } = 1 / \tau )$ on NaturalQuestions (20 documents) (Liu et al., 2024). Reorder: we reorder the documents with relevance metrics of different baselines as our document reordering strategy described in Sec. 4.2. In the case of OpenAI, it corresponds to LongContextReorder<sup>9</sup> in the LangChain framework (Chase, 2022). For results reported under 1st to 20th, we do not use the reordering strategy for all methods.

Table 3, 4, and 7 shows the results of the ablation study in difference tasks. In summary, removing any component proposed for LongLLMLingua will lead to a performance drop regardless of the position of the ground-truth answer. This well validates the necessity and effectiveness of the proposed question-aware mechanism during coarse-to-fine compression, the dynamic compression ratio, and the subsequence recovery strategy. It also shows that applying SBERT for coarse-grained compression will result in inferior performance, which implies the superiority of our question-aware importance metric in Eq. (2) over SBERT. In addition, replacing $p ( x _ { i } ^ { \mathrm { q u e , r e s t r i c t } } | \mathbf { x } _ { k } ^ { \mathrm { d o c } } )$ with $p ( \mathbf { x } _ { k } ^ { \mathrm { d o c } } | x _ { i } ^ { \mathrm { q u e , r e s t r i c t } } )$ can greatly affect performance due to the large noise in calculating $p ( \mathbf { x } _ { k } ^ { \mathrm { { d o c } } } )$ since the perplexity of document depends on many other information besides the question. Removing the restrictive statement can increase the hallucination of small language models, leading to a decrease in performance. Moreover, our subsequence recovery strategy can also bring performance gains for LLMLingua. However, without our question-aware mechanism, results from LLMLingua are still less satisfactory. For more detailed cases, please go to Appendix E.

<table><tr><td>Methods</td><td>|SingleDoc MultiDoc Summ. FewShot Synth. Code AVG</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Tokens 1/τ |Latency Speedup</td></tr><tr><td colspan="10">3,000 tokens constraint</td></tr><tr><td>Retrieval-based Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BM25</td><td>32.3</td><td>34.3</td><td>25.3</td><td>57.9</td><td>45.1</td><td>48.9</td><td>40.6</td><td>3,417</td><td>3x</td><td>7.5</td><td>2.1x 2.0x</td></tr><tr><td>SBERT OpenAI</td><td>35.3 34.5</td><td>37.4</td><td>26.7</td><td>63.4</td><td>51.0</td><td>34.5 37.6</td><td>41.4 41.7</td><td>3,399 3,421</td><td>3x</td><td>7.7 13.3</td><td>1.2x</td></tr><tr><td>LongLLMLingua rk</td><td>37.6</td><td>38.6 42.9</td><td>26.8 26.9</td><td>63.4 68.2</td><td>49.6 49.9</td><td>53.4</td><td>46.5</td><td>3,424</td><td>3x 3x</td><td>8.2</td><td>1.9x</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">Compression-based Methods</td></tr><tr><td>Selective-Context</td><td>23.3</td><td>39.2</td><td>25.0</td><td>23.8</td><td>27.5</td><td>53.1</td><td>32.0</td><td>3,328</td><td>3x</td><td>50.6</td><td>0.3x</td></tr><tr><td>LLMLingua</td><td>31.8</td><td>37.5</td><td>26.2</td><td>67.2</td><td>8.3</td><td>53.2</td><td>37.4</td><td>3,421</td><td>3x</td><td>9.2</td><td>1.7x</td></tr><tr><td>LongLLMLingua</td><td>40.7</td><td>46.2</td><td>27.2</td><td>70.6</td><td>53.0</td><td>55.2</td><td>48.8</td><td>3,283</td><td>3x</td><td>10.0</td><td>1.6x</td></tr><tr><td colspan="10"></td></tr><tr><td colspan="10">2,000 tokens constraint</td></tr><tr><td>Retrieval-based Methods BM25</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>4.6</td><td>3.4x</td></tr><tr><td>SBERT</td><td>30.1 33.8</td><td>29.4 35.9</td><td>21.2 25.9</td><td>19.5 23.5</td><td>12.4 18.0</td><td>29.1 17.8</td><td>23.6 25.8</td><td>1,985 1,947</td><td>5x 5x</td><td>4.8</td><td>3.4x</td></tr><tr><td>OpenAI</td><td>34.3</td><td>36.3</td><td>24.7</td><td>32.4</td><td>26.3</td><td>24.8</td><td>29.8</td><td>1,991</td><td>5x</td><td>10.4</td><td>1.5x</td></tr><tr><td>LongLLMLingua rk</td><td>37.8</td><td>41.7</td><td>26.9</td><td>66.3</td><td>53.0</td><td>52.4</td><td>46.3</td><td>1,960</td><td>5x</td><td>4.7</td><td>3.3x</td></tr><tr><td colspan="10">Compression-based Methods</td></tr><tr><td>Selective-Context</td><td>16.2</td><td>34.8</td><td>24.4</td><td>15.7</td><td>8.4</td><td>49.2</td><td>24.8</td><td>1,925</td><td>5x</td><td>47.1</td><td>0.3x</td></tr><tr><td>LLMLingua</td><td>22.4</td><td>32.1</td><td>24.5</td><td>61.2</td><td>10.4</td><td>56.8</td><td>34.6</td><td>1,950</td><td>5x</td><td>5.9</td><td>2.6x</td></tr><tr><td>LongLLMLingua</td><td>39.9</td><td>43.2</td><td>27.4</td><td>69.8</td><td>53.0</td><td>56.7</td><td>48.3</td><td>1,822</td><td>6x</td><td>6.1</td><td>2.6x</td></tr><tr><td>Original Prompt</td><td>39.7</td><td>38.7</td><td>26.5</td><td>67.0</td><td>37.8</td><td>54.2</td><td>44.0</td><td>10,295</td><td></td><td>15.6</td><td>-</td></tr><tr><td>Zero-shot</td><td>15.6</td><td>31.3</td><td>15.6</td><td>40.7</td><td>1.6</td><td>36.2</td><td>23.5</td><td>214</td><td>48x</td><td>1.6</td><td>9.8x</td></tr></table>

Table 2: Performance of different methods under different compression ratios on LongBench (Bai et al., 2023) using GPT-3.5-Turbo in 2,000 tokens constraint

<table><tr><td>1st 5th 10th 15th 20th LongLLMLingua 77.2 72.9 70.8 70.5 70.6</td></tr><tr><td>Question-aware Coarse-grained - w/o Question-awareness 42.1 40.3 39.7 40.1 40.3 - w/ SBERT 73.2 68.5 65.7 66.1 66.7 cque,restrict 56.0 52.6 53.4 51.6 51.1 - w/o restrict 75.1 72.2 70.3 70.3 70.2</td></tr><tr><td>- w/o Question-aware Fine-grained 75.8 71.0 68.9 68.4 69.3 - w/o Dynamic Compression Ratio 74.4 70.7 68.7 67.9 68.1 - w/o Subsequence Recovery 76.771.7 69.4 69.3 69.7 - w/ Document Reordering 76.2 76.2 76.2 76.2 76.2 - w/ GPT2-small 74.6 71.7 70.1 69.8 68.5</td></tr><tr><td>LLMLingua 39.7 39.5 40.4 37.142.3 - w/ Subsequence Recovery 43.844.143.543.344.4</td></tr></table>

Table 3: Ablation study on NaturalQuestions with 2x constraint using GPT-3.5-Turbo.

Latency evaluation We conducte end-to-end latency testing on a V100-32G, using the prompts from Multi-document QA, LongBench, and Zero-SCROLLS in the API call, and results are shown in Table 1, 2 and 6. The latency includes the time cost for prompt compression and the request time for LLMs, with multiple measurements taken and averaged over. Results demonstrate that LongLLM-Lingua does indeed speed up the overall inference under different compression ratios and scenarios. Moreover, with the compression ratio increasing, the acceleration effect becomes more pronounced up to 2.6x. However, the OpenAI embedding and Selective-Context results in longer latency time, due to repeated API calls and the sequential entropy calculation of semantic units, respectively.

## 6 Related Works

Long context for LLMs. Recent research has focused on expanding the window size of LLMs. Main approaches include: (1) Staged pre-training (Nijkamp et al., 2023) which gradually increases the context window; (2) Modifying (Press et al., 2022) or interpolating position embeddings (Chen et al., 2023; Peng et al., 2024); (3) Using linear or sparse attention mechanisms (Ding et al., 2023; Sun et al., 2023); (4) Utilizing external memory modules for context storage (Bertsch et al., 2023; Tworkowski et al., 2023). While these methods address context window expansion, their impact on downstream task performance has yet to be discussed.

Information distribution in prompt. Recent empirical experiments have shown that LLM performance decreases with less effective information in a prompt (Bai et al., 2023; Li et al., 2023a; Shi et al., 2023). Moreover, the position of relevant information in a prompt has a significant impact on performance (Wu et al., 2023b). Liu et al. (2024) suggests that LLMs have more difficulty comprehending information located in the middle of a prompt compared to those at the edges.

Retrieval methods can be categorized as dense or sparse retrieval methods. Sparse retrieval methods, like BM25, determine the relevance between queries and documents based on n-gram information. Conversely, dense retrieval methods assess the relevance between queries and documents in latent space using embedding model (Reimers and Gurevych, 2019; Xiao et al., 2023; Günther et al., 2023) and reranker model (Xiao et al., 2023). Recently, Jiang et al. (2023b) proposed an unsupervised dense retrieval method that leverages traditional compression algorithms, such as gzip, and k-nearest neighbors.

Prompt compression methods can be grouped into three main categories: (1) Token pruning (Goyal et al., 2020; Kim and Cho, 2021; Modarressi et al., 2022) and token merging (Bolya et al., 2023), which need model fine-tuning or intermediate results during inference and have been used with BERT-scale models. (2) Soft prompt tuning methods like GIST (Mu et al., 2023), AutoCompressor (Chevalier et al., 2023), and ICAE (Ge et al., 2024), which require LLMs’ parameter finetuning, making them suitable for specific domains but not directly applicable to black-box LLMs. (3) Information-entropy-based approaches such as Selective Context (Li et al., 2023c) and LLMLingua (Jiang et al., 2023a), which use a small language model to calculate the self-information or perplexity of each token in the original prompt and then remove tokens with lower perplexities.

## 7 Conclusion

We propose LongLLMLingua to address the three challenges, i.e., higher computational cost, performance reduction, and position bias for LLMs in long context scenarios. We develop LongLLMLingua from the perspective of efficient prompt compression, thus reducing computational cost. We further design four components, i.e., a questionaware coarse-to-fine compression method, a document reordering mechanism, dynamic compression ratios, and a subsequence recovery strategy to improve LLMs’ perception of the key information, with which LongLLMLingua demonstrate superior performance. Experiments on the multidocument QA, multi-hop QA, and long context benchmarks demonstrate that LongLLMLingua compressed prompt can derive higher performance than original prompts while both API costs for inference and the end-to-end system latency are largely reduced.

## Limitation

Although previous experiments demonstrate LongLLMLingua’s effectiveness and efficiency across a broad range of tasks, the method still has the following limitations: 1) LongLLMLingua is a question-aware approach, meaning it requires re-compression for different questions, even with the same context, preventing caching of the context. Moreover, in terms of computational cost, LongLLMLingua increases the computation by twice as much as LLMLingua. This can lead to greater overhead in real-world applications. However, this issue can be mitigated by extending the question-aware approach to a task-aware approach, allowing for reuse and caching. 2) While the effectiveness of LongLLMLingua has been tested on a wide range of tasks, especially on the multi-hop QA dataset MuSicQue (Trivedi et al., 2022), its effectiveness might be impacted when the relationship between context and prompt is more complex and subtle due to the coarse-level question-aware approach.

## References

Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024. Self-RAG: Learning to retrieve, generate, and critique through self-reflection. In The Twelfth International Conference on Learning Representations.

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, et al. 2023. Longbench: A bilingual, multitask benchmark for long context understanding. ArXiv preprint, abs/2308.14508.

Amanda Bertsch, Uri Alon, Graham Neubig, and Matthew R. Gormley. 2023. Unlimiformer: Longrange transformers with unlimited length input. In Thirty-seventh Conference on Neural Information Processing Systems.

Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. 2023. Token merging: Your vit but faster. In The

Eleventh International Conference on Learning Representations.

Harrison Chase. 2022. LangChain.

Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. 2023. Extending context window of large language models via positional interpolation. ArXiv preprint, abs/2306.15595.

Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen. 2023. Adapting language models to compress contexts. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 3829–3846, Singapore. Association for Computational Linguistics.

Kenneth Ward Church and Patrick Hanks. 1989. Word association norms, mutual information, and lexicography. In 27th Annual Meeting of the Association for Computational Linguistics, pages 76–83, Vancouver, British Columbia, Canada. Association for Computational Linguistics.

Grégoire Delétang, Anian Ruoss, Paul-Ambroise Duquenne, Elliot Catt, Tim Genewein, Christopher Mattern, Jordi Grau-Moya, Li Kevin Wenliang, Matthew Aitchison, Laurent Orseau, et al. 2023. Language modeling is compression. ArXiv preprint, abs/2309.10668.

Jiayu Ding, Shuming Ma, Li Dong, Xingxing Zhang, Shaohan Huang, Wenhui Wang, and Furu Wei. 2023. Longnet: Scaling transformers to 1,000,000,000 tokens. ArXiv preprint, abs/2307.02486.

Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, and Zhifang Sui. 2023. A survey for in-context learning. ArXiv preprint, abs/2301.00234.

Tao Ge, Hu Jing, Lei Wang, Xun Wang, Si-Qing Chen, and Furu Wei. 2024. In-context autoencoder for context compression in a large language model. In The Twelfth International Conference on Learning Representations.

Saurabh Goyal, Anamitra Roy Choudhury, Saurabh Raje, Venkatesan T. Chakaravarthy, Yogish Sabharwal, and Ashish Verma. 2020. Power-bert: Accelerating BERT inference via progressive word-vector elimination. In Proceedings ofthe 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pages 3690–3699. PMLR.

Michael Günther, Jackmin Ong, Isabelle Mohr, Alaeddine Abdessalem, Tanguy Abel, Mohammad Kalim Akram, Susana Guzman, Georgios Mastrapas, Saba Sturua, Bo Wang, Maximilian Werk, Nan Wang, and Han Xiao. 2023. Jina embeddings 2: 8192-token general-purpose text embeddings for long documents. ArXiv preprint, abs/2310.19923.

Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2022. Unsupervised dense information retrieval with contrastive learning. Transactions on Machine Learning Research.

Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023a. LLMLingua: Compressing prompts for accelerated inference of large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 13358–13376. Association for Computational Linguistics.

Zhiying Jiang, Matthew Yang, Mikhail Tsirlin, Raphael Tang, Yiqin Dai, and Jimmy Lin. 2023b. “lowresource” text classification: A parameter-free classification method with compressors. In Findings of the Associationfor Computational Linguistics: ACL 2023, pages 6810–6828, Toronto, Canada. Association for Computational Linguistics.

Greg Kamradt. 2023. Needle In A Haystack - Pressure Testing LLMs.

Gyuwan Kim and Kyunghyun Cho. 2021. Lengthadaptive transformer: Train once with length drop, use anytime with search. In Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 6501–6511, Online. Association for Computational Linguistics.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural questions: A benchmark for question answering research. Transactions ofthe Associationfor Computational Linguistics, 7:452–466.

Patrick S. H. Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.

Dacheng Li, Rulin Shao, Anze Xie, Ying Sheng, Lianmin Zheng, Joseph E. Gonzalez, Ion Stoica, Xuezhe Ma, and Hao Zhang. 2023a. How long can opensource llms truly promise on context length?

Jiaqi Li, Mengmeng Wang, Zilong Zheng, and Muhan Zhang. 2023b. Loogle: Can long-context language models understand long contexts? ArXiv preprint, abs/2311.04939.

Yucheng Li, Bo Dong, Frank Guerin, and Chenghua Lin. 2023c. Compressing context to enhance inference

efficiency of large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 6342–6353, Singapore. Association for Computational Linguistics.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the Middle: How Language Models Use Long Contexts. Transactions ofthe Associationfor Computational Linguistics, 12:157–173.

Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2022. MetaICL: Learning to learn in context. In Proceedings ofthe 2022 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 2791–2809, Seattle, United States. Association for Computational Linguistics.

Ali Modarressi, Hosein Mohebbi, and Mohammad Taher Pilehvar. 2022. AdapLeR: Speeding up inference by adaptive length reduction. In Proceedings ofthe 60th Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 1–15, Dublin, Ireland. Association for Computational Linguistics.

Jesse Mu, Xiang Lisa Li, and Noah Goodman. 2023. Learning to compress prompts with gist tokens. In Thirty-seventh Conference on Neural Information Processing Systems.

Erik Nijkamp, Tian Xie, Hiroaki Hayashi, Bo Pang, Congying Xia, Chen Xing, Jesse Vig, Semih Yavuz, Philippe Laban, Ben Krause, Senthil Purushwalkam, Tong Niu, Wojciech Krysci´ nski, Lidiya Mu-´ rakhovs’ka, Prafulla Kumar Choubey, Alex Fabbri, Ye Liu, Rui Meng, Lifu Tu, Meghana Bhat, Chien-Sheng Wu, Silvio Savarese, Yingbo Zhou, Shafiq Joty, and Caiming Xiong. 2023. Xgen-7b technical report. ArXiv preprint, abs/2309.03450.

Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. 2023. Generative agents: Interactive simulacra of human behavior. In Proceedings ofthe 36th Annual ACM Symposium on User Interface Software and Technology, UIST ’23, New York, NY, USA. Association for Computing Machinery.

Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. 2024. YaRN: Efficient context window extension of large language models. In The Twelfth International Conference on Learning Representations.

Ofir Press, Noah A. Smith, and Mike Lewis. 2022. Train short, test long: Attention with linear biases enables input length extrapolation. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net.

Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings using Siamese BERTnetworks. In Proceedings ofthe 2019 Conference on

Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992, Hong Kong, China. Association for Computational Linguistics.

Uri Shaham, Maor Ivgi, Avia Efrat, Jonathan Berant, and Omer Levy. 2023. ZeroSCROLLS: A zero-shot benchmark for long text understanding. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 7977–7989, Singapore. Association for Computational Linguistics.

Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. 2024. Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face. Advances in Neural Information Processing Systems, 36.

Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H Chi, Nathanael Schärli, and Denny Zhou. 2023. Large language models can be easily distracted by irrelevant context. In International Conference on Machine Learning, pages 31210–31227. PMLR.

Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei. 2023. Retentive network: A successor to transformer for large language models. ArXiv preprint, abs/2307.08621.

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022. MuSiQue: Multihop questions via single-hop question composition. Transactions of the Association for Computational Linguistics, 10:539–554.

Szymon Tworkowski, Konrad Staniszewski, Mikołaj Pacek, Yuhuai Wu, Henryk Michalewski, and Piotr Miłos. 2023.´ Focused transformer: Contrastive training for context scaling. In Thirty-seventh Conference on Neural Information Processing Systems.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W White, Doug Burger, and Chi Wang. 2023a. Autogen: Enabling next-gen llm applications via multi-agent conversation framework. ArXiv preprint, abs/2308.08155.

Zhiyong Wu, Yaoxiang Wang, Jiacheng Ye, and Lingpeng Kong. 2023b. Self-adaptive in-context learning: An information compression perspective for incontext example selection and ordering. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1423–1436, Toronto, Canada. Association for Computational Linguistics.

Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. 2023. C-pack: Packaged resources to advance general chinese embedding. ArXiv preprint, abs/2309.07597.

Peng Xu, Wei Ping, Xianchao Wu, Lawrence McAfee, Chen Zhu, Zihan Liu, Sandeep Subramanian, Evelina Bakhturina, Mohammad Shoeybi, and Bryan Catanzaro. 2024. Retrieval meets long context large language models. In The Twelfth International Conference on Learning Representations.

## A Derivation Of Question-Aware Fine-Grained Compression

Based on the definition of Eq. (3), we can derive that,

$$
\begin{array} { r l } & { s _ { i } = \mathrm { p e r p l e x i t y } ( x _ { i } | { x } _ { < i } ) - \mathrm { p e r p l e x i t y } ( x _ { i } | { x } ^ { \mathrm { q u e } } , { x } _ { < i } ) } \\ & { \quad = q ( x _ { i } ) \log p ( x _ { i } | { x } ^ { \mathrm { q u e } } , { x } _ { < i } ) - q ( x _ { i } ) \log p ( x _ { i } | { x } _ { < i } ) } \\ & { \quad = q ( x _ { i } ) \log \frac { p ( x _ { i } | { x } ^ { \mathrm { q u e } } , { x } _ { < i } ) } { p ( x _ { i } | { x } _ { < i } ) } } \end{array}\tag{6}
$$

In the actual calculation of perplexity, a log operation is performed to avoid overflow, and $q ( x _ { i } )$ represents the probability distribution of the groundtruth.

At the same time, we can derive the following expanded expression based on Bayes’ theorem.

$$
\begin{array} { r } { p ( x ^ { \mathrm { q u e } } | x _ { i } , x _ { < i } ) = \frac { p ( x _ { i } | x ^ { \mathrm { q u e } } , x _ { < i } ) p ( x ^ { \mathrm { q u e } } ) } { p ( x _ { i } | x _ { < i } ) } } \\ { = p ( x ^ { \mathrm { q u e } } ) \frac { p ( x _ { i } | x ^ { \mathrm { q u e } } , x _ { < i } ) } { p ( x _ { i } | x _ { < i } ) } } \end{array}\tag{7}
$$

The probability distribution $p ( x ^ { \mathrm { q u e } } )$ of the question and the ground-truth distribution $q ( x _ { i } )$ of $x _ { i }$ are constants, hence $s _ { i }$ can be considered as the representation of Eq. (7).

$$
s _ { i } \propto p ( x ^ { \mathrm { q u e } } | x _ { i } , x _ { < i } )\tag{8}
$$

So we can utilize Eq. (3) to represent the probability distribution $p ( x ^ { \mathrm { q u e } } | x _ { i } , x _ { < i } )$ , which represents the condition likelihood of generating $x ^ { \mathrm { q u e } }$ given the token $x _ { i }$ . Therefore, we can represent the tokenlevel sensitive distribution for the question $x ^ { \mathrm { q u e } }$ using just a single inference. For tokens that are unrelated to $x ^ { \mathrm { q u e } }$ , such as the tokens on the right side of Figure 3b, their original amount of information may be high, but the contrastive perplexity remains at a relatively low level. Finally, we observe that the form of contrastive perplexity is equivalent to conditional pointwise mutual information (Church and Hanks, 1989).

## B Experiment Details

## B.1 Dataset Details

We use NaturalQuestions (Liu et al., 2024) for the multi-document QA task, MuSicQue (Trivedi et al.,

2022) for the multi-hop QA task, and use Long-Bench (Bai et al., 2023), ZeroSCROLLS (Shaham et al., 2023), LooGLE (Li et al., 2023b) for general long context scenarios. The specific details of the dataset are as follows:

NaturalQuestions multi-document QA A multi-document question-answering dataset, comprising 2,655 problems, was built by (Liu et al., 2024) based on the NaturalQuestions dataset (Kwiatkowski et al., 2019). This dataset provides a realistic retrieval-augmented generation setup that closely resembles commercial search and question-answering applications (e.g., Bing Chat). Each example in the dataset contains a question and k related documents, utilizing the Contriever retrieval system (Izacard et al., 2022), one of which includes a document with the correct answer. To perform this task, the model must access the document containing the answer within its input context and use it to answer the question. The dataset’s data is sourced from the NaturalQuestions dataset, which contains historical queries issued to the Google search engine and human-annotated answers extracted from Wikipedia. The average prompt token length in this benchmark is 2,946. For our experiments, we used the version provided by (Liu et al., 2024) that includes 20 documents<sup>10</sup>. The dataset comprises five different ground truth document position settings in the prompt: 1st, 5th, 10th, 15th, and 20th.

LongBench A multi-task long context benchmark consists of 3,750 problems in English and includes six categories with a total of 16 tasks. These tasks encompass key long-text application scenarios, such as single-document QA, multi-document QA, summarization, few-shot learning, synthetic tasks, and code completion. The average prompt token length in this benchmark is 10,289. For our experiments, we used the English dataset and evaluation scripts provided by (Bai et al., 2023) for this benchmark<sup>11</sup>.

ZeroSCROLLS The multi-task long context benchmark consists of 4,378 problems, including four categories with a total of 10 tasks. These tasks cover summarization, question answering, aggregated sentiment classification, and information reordering. The average prompt token length in this benchmark is 9,788. For our experiments, we used the validation set and evaluation scripts provided by (Shaham et al., 2023) for this dataset<sup>12</sup>.

MuSiQue The multi-hop question-answer dataset is composed of 39,876, 4,834, and 4,918 problems in the training, validation, and testing datasets, respectively. This dataset requires the language model to conduct multiple inferences based on the content of several documents and provide corresponding answers, thereby necessitating a certain capability for global information processing. The average token length for prompts in this dataset is 2,477. For our experiments, we utilized the validation set and evaluation scripts provided by (Trivedi et al., 2022) for this dataset<sup>13</sup>.

LooGLE The multi-task long context benchmark comprises 6,448 problems, divided into three categories: summarization, short dependency question answering, and long dependency question answering. The average prompt token length in this benchmark stands at 24,005. For our experiments, we focused on the long dependency question answering subset, which includes four types of tasks: information retrieval, timeline reordering, computation, and comprehension. This subset contains 1,101 problems. We utilized the evaluation scripts provided by (Li et al., 2023b) for this dataset<sup>14</sup>.

## B.2 Other Implementation Details

All experiments were conducted using a Tesla V100 (32GB). We use tiktoken<sup>15</sup> and GPT-3.5- Turbo model to count all the tokens. We set the granular control coefficient k to 2. We use the pre-defined compression rates $\tau _ { \mathrm { i n s } } ~ = ~ 0 . 8 5$ and $\tau _ { \mathrm { q u e } } ~ = ~ 0 . 9$ for instructions and questions. The segment size used in the iterative token-level compression is set to 200. The δτ used in dynamic compression ratio is set to 0.3. For a fair comparison, we only used reordering in the NaturalQuestions Multi-document QA and noted this in Table 1. We use “We can get the answer to this question in the given documents." as the guideline sentence in Eq. (3).

For the baselines experiment, we use the currently recommended strongest model, all-mpnetbase-v2<sup>16</sup>, as the dense representation model for

SentenceBERT. We use the recommended “textembedding-ada-002" as the embedding model for OpenAI Embedding<sup>17</sup>. We use the GPT2-dolly<sup>18</sup> as the small language model in w/ GPT2-small ablation experiments.

## C Additional Experimental Results

## C.1 Empirical Study of Question-aware Fine-grained Compression

Figure 5 shows the distribution of the document’s average perplexity when the ground-truth is located at more positions within the prompt. As can be observed, as the context length increases, the original perplexity curve remains relatively stable. In unrelated documents, a higher perplexity is still retained, making it easier to remove relevant tokens from the related documents in the prompt compression process, thereby damaging the corresponding semantic information. Contrarily, contrastive perplexity shows an increase in perplexity in documents related to the question. According to the theoretical derivation in Appendix A, it’s known that contrastive perplexity characterizes the conditional probability of tokens corresponding to the question. The higher the relevance, the higher the contrastive perplexity, thereby retaining key information in the prompt compression process.

## C.2 Ablation in LongBench

Table 4 presents the results from the ablation experiment in the LongBench long context benchmark. It can be observed that in various long context tasks: 1) Removing the question-aware coarse-grained, question-aware fine-grained, dynamic compression ratio, document reordering, and subsequence recovery proposed by LongLLMLingua all result in different degrees of performance drop. 2) Among these, question-aware coarse-grained is particularly important for document-based QA and synthetic tasks, with the maximum drop being 35.8 points; its impact on summarization and code tasks is relatively smaller. 3) The design of the conditional probability in the question-aware coarse-grained module improves the results in all tasks, including code completion, single-document questionanswer, and synthetic tasks. Changing the order of conditional probabilities or removing the restrict prompt both lead to varying degrees of performance decline. 4) Removing question-aware finegrained, dynamic compression ratio has a more significant impact on document-based QA and synthetic tasks. 5) The subsequence recovery module can enhance reference-based tasks, but its improvement on tasks like summarization, code, synthetic, etc., is relatively smaller. 6) Document reordering is effective for all types of tasks. Reordering at the document level does not affect LLMs’ understanding of context information, even for timelinerelated tasks (see timeline reorder in LooGLE, Table 8). On the contrary, reordering can effectively alleviate the "lost in the middle" issue, thereby improving LLMs performance. 7) Using GPT2-small reduces the capture of effective tokens, but it can still achieve results close to or even slightly better than the original prompt.

![](images/5c4ed7b54a61cc8a8f5b6779d6fe433e5007c0a5232183146fa4127b490cc25e.jpg)  
(a) 1st

![](images/cce03e689ec98632eb01f19e81ee05b3b4d293374012fdfe24482db1a68106bf.jpg)  
(b) 10th

![](images/7d372720251da129fd85fe709c069c1d7458bccdcf97bc3530deae97817c4300.jpg)  
(c) 15th

Figure 5: The distribution of document-level average perplexity when the ground-truth document is in different positions.
<table><tr><td>Methods</td><td colspan="5">SingleDoc MultiDoc Summ.</td><td>Synth. Code</td><td>AVG</td><td>Tokens</td><td>1/τ</td></tr><tr><td>LongLLMLingua</td><td>39.9</td><td>43.2</td><td>27.4</td><td>69.8</td><td>53.0</td><td>56.7</td><td>48.3</td><td>1,822</td><td>6x</td></tr><tr><td>Question-aware Coarse-grained</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>- w/o Question-awareness</td><td>27.1</td><td>38.7</td><td>25.4</td><td>62.0</td><td>18.0</td><td>53.3</td><td>37.4</td><td>1,945</td><td>5x</td></tr><tr><td>- w/ SBERT</td><td>34.0</td><td>38.7</td><td>24.1</td><td>57.9</td><td>32.5</td><td>31.1</td><td>36.4</td><td>1,790</td><td>6x</td></tr><tr><td>- w/  $p ( \mathbf { x } _ { k } ^ { \mathrm { d o c } } | \boldsymbol { x } _ { i } ^ { \mathrm { q u e , r e s t r i c t } } )$ </td><td>22.5</td><td>28.9</td><td>23.2</td><td>53.0</td><td>22.5</td><td>33.3</td><td>30.6</td><td>1,794</td><td>6x</td></tr><tr><td>- w/o restrict</td><td>37.8</td><td>39.5</td><td>26.4</td><td>64.8</td><td>52.5</td><td>55.8</td><td>46.1</td><td>1,834</td><td>6x</td></tr><tr><td>- w/o Question-aware Fine-grained</td><td>35.7</td><td>41.1</td><td>26.4</td><td>62.9</td><td>44.5</td><td>54.8</td><td>44.2</td><td>1,807</td><td>6x</td></tr><tr><td>- w/o Dynamic Compression Ratio</td><td>36.1</td><td>40.6</td><td>26.9</td><td>67.2</td><td>48.0</td><td>55.8</td><td>45.7</td><td>1,851</td><td>6x</td></tr><tr><td>- w/o Subsequence Recovery</td><td>38.6</td><td>41.8</td><td>27.3</td><td>69.0</td><td>53.8</td><td>56.6</td><td>47.8</td><td>1,809</td><td>6x</td></tr><tr><td>- w/o Document Reordering</td><td>39.0</td><td>42.2</td><td>27.4</td><td>69.3</td><td>53.8</td><td>56.6</td><td>48.0</td><td>1,809</td><td>6x</td></tr><tr><td>- w/ GPT2-small</td><td>35.9</td><td>39.4</td><td>25.0</td><td>60.6</td><td>42.0</td><td>55.4</td><td>43.0</td><td>1,892</td><td>5x</td></tr></table>

Table 4: Ablation on LongBench (Bai et al., 2023) using GPT-3.5-Turbo in 2,000 tokens constraint.

## C.3 LongBench Using LongChat-13b-16k

Table 5 presents the experiment results in the Long-Bench long context benchmark using LongChat-13b-16k. It can be seen that the compressed prompt can also achieve good results on other LLMs, such as LongChat-13b-16k. Specifically, 1) there is a maximum improvement of 15.5 points in synthetic tasks. Except for a slight drop in few-shot Learning, there is an improvement of 3-5 points in other tasks. 2) The performance trends of retrieval-based and compressed-based baselines are similar to the results in GPT-3.5-Turbo.

## C.4 ZeroSCROLLS

Table 6 presents a detailed performance breakdown on the ZeroSCROLLS benchmark. It can be observed that in the four summarization tasks - GvRp, SSFD, QMsm, SQAL, LongLLMLingua closely matches or slightly surpasses the original results under two compression constraints. Meanwhile, in the four long context QA tasks - Qsqr, Nrtv, QALT, MuSQ, there is a significant improvement. Notably, in the MuSiQue task, which is based on a question-answering dataset from books and movie scripts, there is a 2.1 point increase even under a 2,000 tokens constraint. It’s worth mentioning that MuSiQue is a multi-hop question-answering dataset that requires LLMs to utilize global information for long dependency QA. LongLLMLingua can also improve by 3.5 points under a 6x compression ratio. In the two ordering tasks, SpDg and BkSS, LongLLMLingua can better retain globally sensitive information, resulting in a 3.0 point improvement in BkSS after prompt compression.

It’s important to note that although the Zero-

<table><tr><td>Methods</td><td colspan="9">SingleDoc MultiDoc Summ. FewShot Synth. Code AVG Tokens 1/τ</td></tr><tr><td>Original Prompt</td><td>27.4</td><td>30.3</td><td>20.3</td><td>49.9</td><td>12.5</td><td>42.5</td><td>30.5</td><td>10,295</td><td></td></tr><tr><td colspan="10">Retrieval-based Methods</td></tr><tr><td>BM25</td><td>2.4</td><td>2.6</td><td>16.4</td><td>8.7</td><td>0.0</td><td>44.7</td><td>12.5</td><td>1,985</td><td>5x</td></tr><tr><td>SBERT</td><td>11.6</td><td>13.7</td><td>21.1</td><td>16.2</td><td>7.5</td><td>30.0</td><td>16.7</td><td>1,947</td><td>5x</td></tr><tr><td>LongLLMLingua rk</td><td>30.3</td><td>32.4</td><td>24.5</td><td>41.0</td><td>27.5</td><td>38.1</td><td>32.3</td><td>1,960</td><td>5x</td></tr><tr><td colspan="10">Compression-based Methods</td></tr><tr><td>Selective-Context</td><td>16.1</td><td>23.5</td><td>21.8</td><td>21.4</td><td>2.5</td><td>35.9</td><td>20.2</td><td>1,925</td><td>5x</td></tr><tr><td>LLMLingua</td><td>20.6</td><td>22.3</td><td>22.4</td><td>35.6</td><td>0.0</td><td>35.4</td><td>22.7</td><td>1,950</td><td>5x</td></tr><tr><td>LongLLMLingua</td><td>31.3</td><td>34.6</td><td>24.6</td><td>46.1</td><td>27.8</td><td>48.8</td><td>35.5</td><td>1,822</td><td>6x</td></tr></table>

Table 5: Performance of different methods under different compression ratios on LongBench (Bai et al., 2023) using LongChat-13b in 2,000 tokens constraint.
<table><tr><td>Methods</td><td>|GvRp SSFD</td><td></td><td>QMsm SQAL</td><td></td><td>QALT Nrtv</td><td></td><td>Qspr MuSQ</td><td></td><td>SpDg</td><td>BkSS</td><td>AVG</td><td>| Tokens</td><td></td><td></td><td>1/τ |Latency Speedup</td></tr><tr><td colspan="10">3,000 tokens constraint</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Retrieval-based Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BM25</td><td>9.7</td><td>3.4</td><td>11.7</td><td>14.3</td><td>57.1</td><td>5.9</td><td>25.7</td><td>11.2</td><td>29.6</td><td>29.6</td><td>19.8</td><td>3,379</td><td>3x</td><td>5.5</td><td>2.2x 2.1x</td></tr><tr><td>SBERT OpenAI</td><td>16.5 14.3</td><td>9.8 8.3</td><td>12.3 12.0</td><td>15.2 15.3</td><td>60.0 66.7</td><td>14.6 13.3</td><td>23.4 24.3</td><td>12.1 11.7</td><td>39.4 31.2</td><td>36.4 26.4</td><td>24.0 22.4</td><td>3,340 3,362</td><td>3x 3x</td><td>5.9 11.7</td><td>1.0x</td></tr><tr><td>LongLLMLingua rk</td><td>19.5</td><td>11.6</td><td>14.7</td><td>15.5</td><td>66.7</td><td>20.5</td><td>27.6</td><td>13.0</td><td>60.8</td><td>43.4</td><td>29.3</td><td>3,350</td><td>3x</td><td>6.2</td><td>2.0x</td></tr><tr><td>Compression-based Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Selective-Context</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LLMLingua</td><td>20.8 18.7</td><td>9.1 10.0</td><td>11.7 14.9</td><td>13.4 16.8</td><td>50.0 61.9</td><td>9.8</td><td>26.1 27.2</td><td>11.0 23.4</td><td>46.0 62.9</td><td>9.5 44.5</td><td>20.7 30.7</td><td>3,460</td><td>3x</td><td>54.2</td><td>0.2x 1.7x</td></tr><tr><td>LongLLMLingua</td><td></td><td></td><td></td><td></td><td></td><td>26.9</td><td></td><td></td><td></td><td></td><td></td><td>3,366</td><td>3x</td><td>7.4</td><td></td></tr><tr><td>22.1</td><td>12.8</td><td></td><td>15.9</td><td>17.1</td><td>67.0</td><td>27.8 31.3</td><td></td><td>23.9</td><td>65.8</td><td>46.5</td><td>33.0</td><td>3,431</td><td>3x</td><td>8.2</td><td>1.5x</td></tr><tr><td colspan="10">2,000 tokens constraint</td><td colspan="10"></td></tr><tr><td colspan="10">Retrieval-based Methods</td><td colspan="10"></td></tr><tr><td>BM25</td><td>8.8</td><td>2.5</td><td>11.1</td><td>13.5</td><td>60.0</td><td>7.0</td><td>4.9</td><td>20.3</td><td>39.9</td><td>32.9</td><td>20.1</td><td></td><td>1,799</td><td>5x</td><td>3.8</td><td>3.2x</td></tr><tr><td>SBERT</td><td>10.2</td><td>7.9</td><td>13.7</td><td>13.2</td><td>60.0</td><td>8.1</td><td>10.8</td><td>1.7</td><td>37.2</td><td>42.8</td><td>20.5</td><td></td><td>1,773</td><td>6x</td><td>4.1</td><td>3.0x</td></tr><tr><td>OpenAI</td><td>11.1</td><td>8.0</td><td>11.8</td><td>13.6</td><td>60.0</td><td>7.1</td><td>13.2</td><td>4.0</td><td>33.6</td><td>43.6</td><td>20.6</td><td></td><td>1,784</td><td>5x</td><td>9.9</td><td>1.2x</td></tr><tr><td>LongLLMLingua rk</td><td>18.2</td><td>9.8</td><td>12.3</td><td>15.9</td><td>57.1</td><td>10.1</td><td>17.8</td><td>7.3</td><td>57.7</td><td>42.3</td><td>24.9</td><td></td><td>1,771</td><td>6x</td><td>4.7</td><td>2.6x</td></tr><tr><td colspan="10">Compression-based Methods</td><td colspan="7"></td></tr><tr><td>Selective-Context</td><td>19.0</td><td>8.4</td><td>9.7</td><td>12.4</td><td>47.0</td><td>12.5</td><td>21.6</td><td>11.5</td><td>41.2</td><td>11.0</td><td>19.4</td><td></td><td>1,865</td><td>5x</td><td>47.5</td><td>0.3x</td></tr><tr><td>LLMLingua</td><td>19.4</td><td>11.9</td><td>13.1</td><td>16.0</td><td>62.1</td><td>23.7</td><td>24.0</td><td>22.4</td><td>33.9</td><td>44.9</td><td>27.2</td><td></td><td>1,862</td><td>5x</td><td>4.8</td><td>0.3x</td></tr><tr><td>LongLLMLingua</td><td>20.1</td><td>12.4</td><td>14.9</td><td>16.5</td><td>65.1</td><td>27.7 30.7</td><td></td><td>23.6</td><td>68.5</td><td></td><td>47.2 32.7</td><td></td><td>1,826</td><td>6x</td><td>5.2</td><td>2.3x</td></tr><tr><td>Original Prompt</td><td>21.8</td><td>12.1</td><td>17.9</td><td>17.4</td><td>66.7</td><td>25.3</td><td>29.8</td><td>20.0</td><td>69.7</td><td>44.1</td><td></td><td>32.5</td><td>9,788</td><td></td><td></td><td></td></tr><tr><td>Zero-shot</td><td>9.4</td><td>3.0</td><td>8.6</td><td>11.4</td><td>42.9</td><td>10.6</td><td>12.4</td><td>5.5</td><td>4.2</td><td>0.0</td><td>12.8</td><td></td><td>32</td><td>306x</td><td>12.2 1.0</td><td>12.2x</td></tr></table>

Table 6: Performance breakdown of different methods under different compression ratios on ZeroSCROLLS (Shaham et al., 2023) using GPT-3.5-Turbo.

Scrolls validation dataset is relatively small, it still demonstrates conclusions similar to previous experimental observations across various methods and tasks. Furthermore, this study conducted an indepth analysis of the multi-hop QA task - MuSiQue, and another long context benchmark - LooGLE. The results can be found in Appendix C.5 and Appendix C.6.

## C.5 MuSiQue

Table 7 presents the results from the MuSiQue multi-hop question-answer dataset. From the table, it can be observed that in the multi-hop QA task, requiring global information: 1) LongLLMLingua can reduce noise in the prompt by eliminating irrelevant information and putting more related information at the beginning or end of the prompt, thereby improving performance by 5.4 points. 2) The performance drop is more pronounced for retrievalbased methods, particularly for n-gram-based methods like BM25. Due to long dependencies, direct matching information is lost, resulting in less relevant information being recalled. 3) The performance of compression-based methods is slightly different. Selective-Context does not distinguish between different modules’ sensitivity, resulting in a loss of question and instruction-related information, thereby leading to poorer performance. However, LLMLingua can still retain relevant key information at around a 2x compression ratio. 4) The ablation experiments show that every module designed in LongLLMLingua plays a role in the multi-hop task. The removal of the question-aware coarse-grained and $\mathbf { w } / p ( \mathbf { x } _ { k } ^ { \mathrm { d o c } } | x _ { i } ^ { \mathrm { q u e , r e s t r i c t } } )$ modules, which have difficulty in perceiving the importance distribution of corresponding questions, can cause a drop of up to 8 points. Removing the restrict prompt in the question-aware coarse module can also cause a 2-point drop due to the hallucination issue of small LLM. In addition, removing question-aware fine-grained, dynamic compression ratio, and document reordering can all cause a drop of 0.5-2.8 points. 5) Moreover, if the small language model in LongLLMLingua is replaced with GPT2-small, it can further improve the acceleration ratio and still achieve a result that is 2.6 points better than the original prompt.

<table><tr><td>Methods Original Prompt</td><td>F1 Tokens 1/τ</td></tr><tr><td>BM25 SBERT LongLLMLingua rk Selective-Context</td><td>45.8 2,427 28.5 1,295 1.9x 36.2 1,288 1.9x 46.3 1,295 1.9x</td></tr><tr><td>LLMLingua LongLLMLingua Question-aware Coarse-grained - w/o Question-awareness</td><td>40.1 1,110 2.2x 51.2 1,077 2.3x 43.21,0762.3x</td></tr><tr><td>- w/ SBERT  $- \mathbf { \Delta w } / p ( \mathbf { x } _ { k } ^ { \mathrm { d o c } } | x _ { i } ^ { \mathrm { q u e , r e s t r i c t } } )$  - w/o Question-aware Fine-grained 48.41,1182.2x</td><td>47.31,070 2.3x 44.01,0662.3x 49.21,0782.3x</td></tr><tr><td>- w/o restrict - w/o Dynamic Compression Ratio 48.2 1,090 2.2x - w/o Subsequence Recovery</td><td>50.7 1,077 2.3x</td></tr></table>

Table 7: Performance of different methods and ablation study on MuSicQue (Trivedi et al., 2022) with 2x constraint using GPT-3.5-Turbo.

## C.6 LooGLE

Table 8 presents the experiment results in the LooGLE long dependency benchmark, which features longer prompts ( 30k) and more global dependencies. From the table, we can observe that: 1) LongLLMLingua can effectively improve the performance of long context tasks by compressing prompts, even for long dependency tasks. The results show that LongLLMLingua significantly improves performance in tasks such as retrieval, timeline reorder, and computation, with the maximum improvement reaching 15.9 points. 2) The document reorder in LongLLMLingua is effective in all types of tasks, even in tasks highly related to the timeline, it can effectively improve performance by alleviating the "lost in the middle" issue. 3) Retrieval-based methods tend to lose performance in tasks that have longer dependencies, such as computation and reasoning. 4) For compressionbased methods, due to the difficulty in perceiving question information, there tends to be a larger performance loss in retrieval tasks within long contexts.

## D Economic Cost

Table 9 presents the estimated per 1,000 samples inference costs for various datasets, encompassing input prompts and generated output text, based on GPT-3.5-Turbo pricing<sup>19</sup>. Our approach demonstrates substantial savings in computational resources and monetary expenses, particularly in long context situations. Cost reductions of \$3.3 (71.7%), \$28.5 (90.5%), \$27.4 (89.5%), \$2.0 (52.6%), and \$88.0 (94.0%) per 1,000 samples are observed for Multi-document QA, LongBench, ZeroScrolls, MuSiQue, and LooGLE, respectively.

## E Ablation Analysis

Figure 6 illustrates the compressed prompts from the Multi-document QA dataset, comparing the use of contrastive perplexity at a high compression ratio (30x). It shows that without question-aware token-level prompt compression, LongLLMLingua tends to compress key information, a tendency that becomes more pronounced at higher compression ratios. Conversely, employing contrastive perplexity allows for better detection of key information related to the question within the context, thus preserving key information within the compressed prompt.

## F Cases Study

Figures 7, 8, and 9 display the outcomes before and after compression, as well as the LLMs’ responses in various scenarios.

<table><tr><td>Methods</td><td></td><td>Retrieval Timeline Reorder Computation Reasoning</td><td></td><td></td><td></td><td>AVG Tokens</td><td>1/τ</td></tr><tr><td>Retrieval-based Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BM25</td><td>20.4</td><td>21.7</td><td>8.2</td><td>26.3</td><td>19.2</td><td>3,185</td><td>10x</td></tr><tr><td>SBERT</td><td>28.9</td><td>21.1</td><td>10.7</td><td>27.2</td><td>22.0</td><td>3,169</td><td>10x</td></tr><tr><td>LongLLMLingua rk</td><td>38.6</td><td>32.2</td><td>16.2</td><td>26.3</td><td>28.3</td><td>3,158</td><td>10x</td></tr><tr><td>Compression-based Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Selective-Context</td><td>16.7</td><td>5.0</td><td>2.3</td><td>17.6</td><td>10.4</td><td>3,710</td><td>8x</td></tr><tr><td>LLMLingua</td><td>10.0</td><td>25.0</td><td>13.3</td><td>21.1</td><td>17.3</td><td>3,404</td><td>9x</td></tr><tr><td>LongLLMLingua</td><td>40.0</td><td>35.0</td><td>19.7</td><td>33.6</td><td>32.1</td><td>3,121</td><td>10x</td></tr><tr><td>LongLLMLingua w/o Reorder</td><td>39.3</td><td>33.8</td><td>18.7</td><td>31.6</td><td>30.9</td><td>3,119</td><td>10x</td></tr><tr><td>Original Prompt</td><td>24.1</td><td>20.9</td><td>13.5</td><td>32.1</td><td></td><td>22.630,546</td><td>一</td></tr><tr><td>Zero-shot</td><td>8.7</td><td>6.3</td><td>1.2</td><td>14.5</td><td>7.7</td><td>43</td><td>710x</td></tr></table>

Table 8: Performance of different methods on LooGLE (Li et al., 2023b) long dependency QA.
<table><tr><td></td><td>Multi-document QA</td><td>LongBench</td><td>ZeroScolls</td><td>MuSicQue</td><td>LooGLE</td></tr><tr><td>Original</td><td>4.6</td><td>31.5</td><td>30.6</td><td>3.8</td><td>93.6</td></tr><tr><td>Ours</td><td>1.3 (↓71.7%)</td><td>3.0 (↓90.5%)</td><td>3.2 (↓89.5%)</td><td>1.8 (↓52.6%)</td><td>5.6 (↓94.0%)</td></tr></table>

Table 9: The inference costs \$ (per 1,000 samples) for various datasets using GPT-3.5-Turbo.

![](images/581aeebcd3832c6bdcd8ec1f1bf46eaa4685bf015b69ebe4d09a0515dc5aacb4.jpg)  
Figure 6: Comparing the compressed prompt and LLMs’ response before and after using Question-aware Finegrained Compression and Subsequence Recovery(1 $\mathbf { \nabla } \cdot / \tau = 3 0 \mathbf { x }$ , high compression ratio setting) from NaturalQuestions Multi-document QA (Liu et al., 2024) using GPT-3.5-Turbo.

![](images/eb8f915696bc72cb4fa968fc60a9e7ff516dad9d0ea269dc4b4e15919fdbf015.jpg)  
Figure 7: Cases study on NaturalQuestions Multi-document QA dataset (Liu et al., 2024) in 4x constraint using GPT-3.5-Turbo.

```prolog
Compressed Prompt:
Please complete the code given below.
public class MessageArchiveManagement
private static final long MILLISECONDS_IN_DAY = 24 * 00 *0;
public static final long_CUP = MCON_DAY
/.../
.("",.getStart
add
ifget() >0
Node end("
end.("
endNode.Value("", Util.getTimestamp(query.getEnd
addNode
} if (.withid null && contact null && !isference
Node with(" .with
.Value("valuewith
.(
// queryMessageive(connection, nextQuery
final(connectionProtocol(), query
synchronized (eries)
// queries.add(nextQuery } }
public boolean queryInProgress( contact, OnLoaded
moreMessagesLoadedListener)
ized (eries)
(Query query : queries)
if(query.getWith().equals(contact.getUserId()))
if (query.onMoreMessagesLoaded == null &&MessagesListener
null) query.setOnMoreMessagesLoaded(Listener}
return true;}} return false;}}
private void finalizeQuery(Protocol protocol, Query query) {
synchronized (queries) {
.remove(query); }
Contact contact = null;
if (query.getWith() != null) {
contact = protocol.getItemByUID(query.getWith()); }
if (contact != null) {
Next line of code:
LLMs’ Response:
contact.setLastMessageTransmitted(query.getEnd());\n
Ground Truth:
if (contact.setLastMessageTransmitted(query.getEnd())) {
Zero-shot LLMs’ Response:
contact.removeQuery(query);\n
```

Figure 8: Cases study on lcc code completion task in LongBench benchmark (Bai et al., 2023) in 2,000 constraint using GPT-3.5-Turbo. 1676

![](images/fbe6aeb2dacd4ab60f5cfe22edde85ef6e74f05b1c0c94070a669b58ed62633f.jpg)  
Figure 9: Cases study on trec few-show learning in LongBench benchmark (Bai et al., 2023) in 2,000 constraint using GPT-3.5-Turbo.