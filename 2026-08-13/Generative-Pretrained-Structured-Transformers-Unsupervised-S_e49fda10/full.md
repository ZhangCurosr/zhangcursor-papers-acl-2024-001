# Generative Pretrained Structured Transformers: Unsupervised Syntactic Language Models at Scale

Xiang Hu<sup>‡</sup> \* Pengyu Ji<sup>§</sup> \* Qingyang Zhu<sup>§</sup> Wei Wu<sup>‡</sup> <sup>†</sup> Kewei Tu<sup>§</sup> <sup>†</sup>

Ant Group<sup>‡</sup>

aaron.hx, congyue.ww @antgroup.com<sup>‡</sup> ShanghaiTech University<sup>§</sup> jipy2023, zhuqy, tukw @shanghaitech.edu.cn<sup>§</sup>

## Abstract

A syntactic language model (SLM) incrementally generates a sentence with its syntactic tree in a left-to-right manner. We present Generative Pretrained Structured Transformers (GPST), an unsupervised SLM at scale capable of being pre-trained from scratch on raw texts with high parallelism. GPST circumvents the limitations of previous SLMs such as relying on gold trees and sequential training. It consists of two components, a usual SLM supervised by a uni-directional language modeling loss, and an additional composition model, which induces syntactic parse trees and computes constituent representations, supervised by a bi-directional language modeling loss. We propose a representation surrogate to enable joint parallel training of the two models in a hard-EM fashion. We pre-train GPST on OpenWebText, a corpus with 9 billion tokens, and demonstrate the superiority of GPST over GPT-2 with a comparable size in numerous tasks covering both language understanding and language generation. Meanwhile, GPST also significantly outperforms existing unsupervised SLMs on left-to-right grammar induction, while holding a substantial acceleration on training.<sup>1</sup>

## 1 Introduction

Pre-training a Transformer architecture (Vaswani et al., 2017) as a large language model has dominated the field of natural language processing (NLP) (Devlin et al., 2019; Liu et al., 2019; Radford et al., 2018, 2019; Brown et al., 2020; Ouyang et al., 2022). While Transformer language models have exhibited remarkable performance over various downstream NLP tasks (Bang et al., 2023), the recursive compositions behind language are represented in an implicit and entangled form. In contrast, human language understanding exhibits explicit composition decisions, as exemplified by the garden path sentence (Dynel, 2009) “Time flies like an arrow; Fruit flies like a banana”, where distinct syntactic configurations yield vastly divergent meanings<sup>2</sup>. In addition, human infants acquire such compositional capability without supervision (Saffran et al., 1996). These phenomena motivate us to explore an unsupervised approach to learning explicit compositions in language modeling.

A typical approach to achieve language modeling with explicit composition is to model the joint distribution of words and a syntactic tree within the framework of syntactic language models (SLMs) (Dyer et al., 2016). Though there has been a long line of research on SLMs, they are rarely exploited as the backbone in state-of-the-art language modeling due to poor scalability. Recent Transformer-based SLMs (Sartran et al., 2022; Murty et al., 2023) require annotated parse trees or supervised parsers as structural supervision, leading to limited training data scales and domain adaption issues (McClosky et al., 2006). On the other hand, in unsupervised SLMs (Kim et al., 2019b; Shen et al., 2019a), non-terminal constituents are composed from their sub-constituents sequentially in a left-to-right manner, resulting in data dependencies that impede training parallelism.

In this work, we aim to pre-train an SLM at scale on raw texts. To this end, we propose Generative Pretrained Structured Transformer (GPST), an unsupervised SLM with the Transformer architecture as a backbone. A common practice in existing unsupervised SLMs is to learn structures by a uni-directional language modeling loss (LM loss). However, we empirically find such an asymmetric loss with only right-to-left feedback results in branching biases in the induced parse trees. Based on the insight, we propose two components in GPST, a composition model performing structural learning supervised by a bi-directional LM loss, and a generative model for uni-directional syntactic language modeling. Specifically, we train the GPST in a fashion similar to hard-EM (Liang et al., 2017): in E-step, the composition model runs a pruned deep inside-outside encoder to induce a parse tree and compute inside and outside representations of constituents simultaneously within logarithmic steps (Hu et al., 2024); while in M-step, we update all parameters of GPST by minimizing both the bi-directional (reconstructing the sentence from outside representations) and uni-directional LM loss given the induced tree. The key in the Mstep lies in using the inside representations of constituents computed by the composition model as a surrogate of inputs for the generative model, which enjoys two advantages. First, the representations of all constituents pre-computed in the E-step can be simultaneously fed into the generative model, which breaks the data dependencies and facilitates training parallelism. Second, with these representations participating in generation, the uni-directional LM loss in the M-step could be back-propagated to not only the generative model but the composition model used in the E-step as well.

In experiments, we pre-train GPSTs with sizes comparable to those of $\mathrm { G P T } { - } 2 _ { \mathrm { s m a l l } }$ and GPT-2<sub>medium</sub> on OpenWebText (Gokaslan and Cohen, 2019)( 9 billion tokens), and evaluate the models on various tasks including language understanding, language generation, and grammar induction. GPST demonstrates an approximately 60-fold training acceleration and over 15% absolute increase in left-to-right grammar induction in comparison with existing unsupervised SLMs. Meanwhile, GPST also shows advantages over GPT-2 across almost all language understanding/generation benchmarks. GPST provides constituent-level interfaces that are not inherently possessed by the conventional Transformerbased language models, and thus exhibits great potential to enhance interpretability (Hu et al., 2023), support multi-modality (Wan et al., 2022), and improve dense retrieval in the future. Our contributions are three-fold:

• We propose an SLM consisting of a composition model in addition to a generative model, which can be trained without gold trees via a novel approach akin to hard-EM.

• We propose a representation surrogate to enable joint parallel training of all components.

• To the best of our knowledge, GPST is the first unsupervised SLM able to be pre-trained from scratch on billions of tokens and surpass GPT-2 on various benchmarks. The experimental results demonstrate the potential of GPST as a backbone for large language models.

## 2 Related work

Syntactic Language Models. There have been extensive studies on syntactic language modeling (Baker, 1979; Jelinek and Lafferty, 1991; Chelba, 1997; Chelba and Jelinek, 2000; Vinyals et al., 2015; Charniak et al., 2016; Dyer et al., 2016; Qian et al., 2021; Yoshida and Oseki, 2022), in which words and constituent symbols are mixed up and generated in a left-to-right manner. Recent works (Sartran et al., 2022; Murty et al., 2023) utilize Transformers to parameterize action probability distributions, but relies on annotated parse trees or parsers trained on gold trees as structural guidance. Besides, unsupervised SLMs are also explored, by differentiable structured hidden layers (Kim et al., 2017; Shen et al., 2018, 2019a; DuSell and Chiang, 2024), reinforcement learning approaches (Yogatama et al., 2017), or variational approximations (Li et al., 2019; Kim et al., 2019b). Normally, these unsupervised models are trained in a sequential manner. Our model follows a similar generation paradigm, but has stark differences in model architecture and training approach.

Composition Models. A composition model transforms text encoding into a combinatorial optimization problem, learns and searches for the optimal structure, and encodes the text in a bottom-up manner along a binary tree via a composition function recursively. Maillard et al. (2019) proposes a CKY-like (Cocke, 1969; Kasami, 1966; Younger, 1967) encoder, in which high-level constituents are soft-weighted over composed representations of its sub-constituents. Drozdov et al. (2019) proposes a deep inside-outside encoder (Baker, 1979; Lari and Young, 1990), enabling the model to learn underlying structures via an auto-encoding objective. Recently, a series of studies (Hu et al., 2021, 2022, 2024) have been conducted to reduce the neural inside encoder complexity from cubic to linear. Our SLM is built on top of state-of-the-art composition modeling techniques, in which we achieve unsupervised learning and enhance training parallelism by taking advantage of the pruned inside-outside encoder (Hu et al., 2024).

## 3 Methodology

Given a sentence $\textbf { x } = ~ [ x _ { 1 } , x _ { 2 } , . ~ . ~ . ~ , x _ { n } ]$ with $x _ { i }$ from a vocabulary $\mathbb { V } \left( 1 \ \leqslant \ i \ \leqslant \ n \right)$ , our goal is to train an SLM without gold trees that can simultaneously generate x and its syntactic structure. We first introduce the generative architecture of GPST, and then elaborate on how to perform training and inference with the model.

## 3.1 Generative Model

GPST generates a sentence and its parse tree from left to right via two types of actions, GEN and COMP, along with a stack (Dyer et al., 2015) to maintain partially completed sub-trees during generation. GEN generates a word x and pushes its embedding onto the stack. We denote such an action as $\operatorname { G E N } ( x )$ , with $x \in \mathbb { V }$ . COMP pops the top two elements off the stack, computes their composed representation, and pushes it back to the stack. A major difference in model architecture between GPST and existing unsupervised SLMs, such as URNNG (Kim et al., 2019b), is that GPST makes good use of the architecture of Transformers to parameterize the action probabilities and thus hidden states from previous actions can be directly accessed via self-attention during generation.

Figure 1 illustrates the generative process of GPST. The generative model comprises type layers and token layers, both consisting of multi-layered Transformers. Let us denote the stack at step t as $\mathbf { S } _ { t }$ , with ${ \bf S } _ { t } ^ { 0 }$ and $\mathbf { S } _ { t } ^ { 1 }$ representing the top two elements, respectively. Initially, ${ \bf S } _ { 0 } ^ { 0 }$ is set to the embedding of the beginning-of-sentence token $( \mathrm { i . e . }$ bos in Figure 1). At each step $t , \mathbf { S } _ { t } ^ { 0 }$ along with a position ID $w _ { t }$ is fed into the type layers, yielding a hidden state $\mathbf { h } _ { t }$ , which is then utilized to predict the next action type y<sub>t</sub>

• If $y _ { t } = 0 \left( \mathbf { C o M P } \right)$ , we set $w _ { t + 1 }$ as w<sub>t</sub>, pop off ${ \bf S } _ { t } ^ { 0 }$ and $\mathbf { S } _ { t } ^ { 1 }$ from $\mathbf { S } _ { t } .$ , and compose them using a composition function. The composed representation is then pushed back into the stack. In such a case, action $a _ { t }$ at time step t is set to COMP.

$\mathrm { I f } \ y _ { t } = 1 \left( \mathrm { G E N } \right)$ , we set $w _ { t + 1 }$ as $w _ { t } + 1$ , feed $\mathbf { h } _ { t }$ to the subsequent token layers, and get an output state $\mathbf { g } _ { w _ { t } }$ that is used to generate $x _ { w t + 1 }$ . In such a case, we have $a _ { t } = \mathbf { G } \mathbf { E } \mathbf { N } \big ( x _ { w _ { t } + 1 } \big )$

Suppose that $\mathbf { a } _ { \mathbf { x y } }$ is the action sequence to generate a sentence x and its parse tree $\mathbf { y }$ , then the joint distribution of x and $\mathbf { y }$ can be formulated as:

$$
p ( \mathbf x , \mathbf y ) = p ( \mathbf a _ { \mathbf x \mathbf y } ) = \prod _ { t } p ( a _ { t } | a _ { < t } ) ,\tag{1}
$$

![](images/e9aafac335e82d11adcec770faf0ef2e55975c9fb28739f158136dd5261f10d0.jpg)  
Figure 1: An illustration of the generative process of GPST. $\mathbf { x } _ { i : j }$ denotes the sub tree representation spanning from i to j. As we use Transformers as the backbone, all previous hidden states are leveraged. At step t, the length of historical hidden states is t for the type layers and w<sub>t</sub> for the token layers as illustrated with dotted lines for step 3.

where $p ( a _ { t } | \boldsymbol a _ { < t } )$ is computed by:

$$
\begin{array} { r l } & { \qquad p ( \mathrm { C O M P } | a _ { < t } ) = p ( y _ { t } = 0 | a _ { < t } ) , } \\ & { p ( \mathrm { G E N } ( x _ { w _ { t } + 1 } ) | a _ { < t } ) = p ( x _ { w _ { t } + 1 } | y _ { t } = 1 , a _ { < t } ) p ( y _ { t } = 1 | a _ { < t } ) , } \\ & { \qquad p ( x _ { w _ { t } + 1 } | y _ { t } = 1 , a _ { < t } ) = \mathrm { s o f t m a x } ( \mathrm { M L P } _ { x } ( \mathbf { g } _ { w _ { t } } ) ) , } \\ & { \qquad p ( y _ { t } | a _ { < t } ) = \mathrm { s o f t m a x } ( \mathrm { M L P } _ { y } ( \mathbf { h } _ { t } ) ) . } \end{array}
$$

$\mathrm { M L P } _ { y } ( \cdot )$ and $\mathrm { M L P } _ { x } ( \cdot )$ convert inputs to a 2-dim vector and a V -dim vector, respectively. By predicting action types through shallow layers and tokens through deep layers, we can keep the total computational cost close to that of vanilla Transformers.

## 3.2 Unsupervised Training

How to train an unsupervised SLM effectively and efficiently has always been a challenge. Existing methods suffer from two issues: asymmetric feedback and inability to train in parallel. The former arises from the uni-directional LM loss, and the latter stems from the inherent data dependency of each composition step on the representations of its sub-constituents from previous steps. We tackle both issues with an approach similar to hard-EM. In E-step, we employ a composition model to induce a parse tree through a pruned deep inside-outside encoder. In M-step, we update both the composition model and the generative model by optimizing a joint objective based on the induced tree. The composition and generative model are connected by sharing the same composition function. Below we present details of the two steps and explain how they tackle the issues mentioned above.

E-step. During the E-step, the composition model searches for the best parse tree and composes representations through a deep inside-outside

![](images/57a740b8032caca79be0b81b72118a3e4396760daf0f1112ccde2e6029a893ef.jpg)  
Figure 2: Illustration of the training process. (a) In the E-step, we induce a parse tree and compute constituent representations. (b)(i) Data dependencies within inputs of the generative model. (b)(ii) Illustration of representation surrogates. $\mathbf { x } _ { i : j }$ denotes the original input representation spanning over (i, j) composed from left to right.

(i)

(ii)

auto-encoder (DIORA) (Drozdov et al., 2019). In the inside pass, we compute a composed representation $\bar { \mathbf { i } } _ { i , j } ^ { k }$ and a compatibility score $\bar { a } _ { i , j } ^ { k }$ for each pair of neighboring constituents $( i , k )$ and $( k + 1 , j )$ We then compute each internal span representation $\mathbf { i } _ { i , j }$ as a weighted average over all possible pairs of constituents<sup>3</sup>:

$$
\begin{array} { r l } & { \bar { \mathbf { i } } _ { i , j } ^ { k } = f _ { \alpha } ( \mathbf { i } _ { i , k } , \mathbf { i } _ { k + 1 , j } ) , \bar { a } _ { i , j } ^ { k } = \phi _ { \alpha } ( \mathbf { i } _ { i , k } , \mathbf { i } _ { k + 1 , j } ) , } \\ & { \hat { w } _ { i , j } ^ { k } = \displaystyle \frac { \exp ( \bar { a } _ { i , j } ^ { k } ) } { \sum _ { k ^ { \prime } = i } ^ { j - 1 } \exp ( \bar { a } _ { i , j } ^ { k ^ { \prime } } ) } , \mathbf { i } _ { i , j } = \displaystyle \sum _ { k = i } ^ { j - 1 } \hat { w } _ { i , j } ^ { k } \mathbf { \bar { i } } _ { i , j } ^ { k } . } \end{array}
$$

in which $f _ { \alpha }$ and $\phi _ { \alpha }$ are formulated in $\mathsf { A p - }$ pendix A.2. An illustration to compute $\mathbf { i } _ { 1 , 3 }$ is given in Figure 2(a). Analogously, the outside pass computes each outside representation $\mathbf { o } _ { i , j }$ in a top-down manner based on bi-directional information outside span $( i , j )$ . To accelerate computation, we use the pruned deep inside-outside encoder (Hu et al., 2024) which achieves linear space complexity and approximately logarithmic parallel time complexity. The details of the algorithm and the complete outside pass are presented in Appendix A.1.

Note that for a given span $( i , j )$ , the best splitpoint is k with the highest $\bar { a } _ { i , j } ^ { k }$ . Thus, to derive a parse tree, we can recursively select the best splitpoints top-down starting from the root span (1, n).

The outside representations of tokens can be used to define an auto-encoding loss (i.e., predicting each token from its outside representation) for the composition model, which is optimized in the M-step:

$$
\mathcal { L } _ { a e } = - \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \log \frac { \exp ( \mathbf { o } _ { i , i } ^ { T } \mathbf { e } _ { \mathbf { x } _ { i } } ) } { \sum _ { k = 1 } ^ { | \mathbb { V } | } \exp ( \mathbf { o } _ { i , i } ^ { T } \mathbf { e } _ { k } ) } .
$$

where $\mathbf { e } _ { k }$ is the embedding of the k-th token in the vocabulary. As the auto-encoding loss provides feedback to each token representation from both sides of the token, the asymmetric feedback issue is addressed.

M-step. With the induced tree y, we update the parameters of the composition model and the generative model in a joint manner. Denote the sequence of node spans in post-order as $[ ( i _ { 0 } , j _ { 0 } ) , ( i _ { 1 } , j _ { 1 } ) , . . . , ( i _ { 2 n - 1 } , j _ { 2 n - 1 } ) ]$ . The action sequence can be formulated as:

$$
a _ { t } = \left\{ \begin{array} { l } { { \mathrm { C O M P } } } \\ { { \mathrm { G E N } ( x _ { i _ { t } } ) } } \end{array} , \mathrm { f o r } \begin{array} { l } { { i _ { t } < j _ { t } } } \\ { { i _ { t } = j _ { t } } } \end{array} \right. .
$$

An auto-regression loss can be defined as:

$$
\mathcal { L } _ { a r } = - \log p ( \mathbf { x } , \mathbf { y } ) = - \frac { 1 } { 2 n - 1 } \sum _ { t = 0 } ^ { 2 n - 1 } \log p ( a _ { t } | a _ { < t } ) .
$$

However, even though the action sequence is given, there are still two challenges. First, there are data dependencies within the inputs for the generative model as mentioned earlier and shown in Figure 2(b), which impedes parallel training. Second, there are no feedforwards from the composition model to the generative model, so the two models are disconnected and hence cannot be trained jointly. A key insight to tackle these issues is to use the internal span representations $\mathbf { i } _ { i , j }$ as surrogates<sup>4</sup> of the inputs $\mathbf { x } _ { i : j }$ to the generative model as depicted in Figure 2(b)(ii). As the internal span representations are already computed in the E-step, they can be fed into Transformers seamlessly all at once to fully leverage the parallel training ability of the architecture. Moreover, replacing x with i enables the representations computed by the composition model to participate in the generative model, thus the two models are connected and can be jointly optimized via the auto-regression loss. Note that internal span representations do not contain any information outside spans, so there is no information leakage in the uni-directional generative model.

The final training loss for the M-step combines the auto-regression and auto-encoding losses as:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { a e } + \mathcal { L } _ { a r } . } \end{array}
$$

We empirically find that the combined loss leads to left-branching bias in parse trees induced by the composition model that is not observed when training with $\mathcal { L } _ { a e }$ alone. A possible reason is that left-leaning trees provide more left-side context for each step during generation and thus are reinforced in learning. To tackle the issue, we stop gradient propagation from $\mathcal { L } _ { a r }$ to $a _ { i , j } ^ { k } .$ , which means only gradients from $\mathcal { L } _ { a e }$ are allowed to be backpropagated along $a _ { i , j } ^ { k }$ . Note that other variables like $\mathbf { i } _ { i , j }$ still receive gradients from $\mathcal { L } _ { a r }$

## 3.3 Inference

The action space of GEN(x) is much larger than that of COMP, leading to an imbalance between their probabilities. Stern et al. (2017) point out that during beam search decoding, hypotheses in a beam should be grouped by the length of generated tokens instead of action history, which they refer to as “word-level search”. However, their approach does not guarantee that the top-k next words searched are the optimal ones. To address the issue, we propose an improved word-level search tailored for our generation paradigm. The core idea is to guarantee that all hypotheses in a beam have the same number of GEN(x) actions. Beams satisfying the condition are marked as sync and otherwise sync. Below we depict the entire word-level search process through an example shown in Figure 3:

1. Starting with a sync beam, e.g., A, B, C and (A, B), D, we estimate the probability distribution of the next action for each hypothesis within it. For each possible action, we compute the probability of the resulting new hypothesis as the product of the probabilities of the current hypothesis and the action. The new hypotheses are pooled and ranked, and the top-k are retained (e.g., A, (B, C) and (A, B), D, E).

2. If the current beam contains hypotheses with the last step being COMP, e.g., A, (B, C), we continue to explore their next actions, update their probabilities, pool them with other hypotheses in the beam, and rank the top-k, until all the topk hypotheses have GEN(x) as their last action.

![](images/2140afef10f925b8366e598634a717c4053fee189c56dd63e71fafd64072fc71.jpg)  
Figure 3: An illustration of beam search decoding of size 2. For simplicity, we use $^ { 6 6 } ) ^ { 5 }$ to denote COMP and upper case characters to denote words generated by GEN(x). Boxes filled in gray are hypotheses with the last action being COMP. Grayed-out boxes are pruned out during beam search.

3. End generation upon reaching the length limit or producing an end token; otherwise, go back to step 1.

This method is also applicable to top-k random sampling (Fan et al., 2018), or parsing with a given input sentence by simply setting the probabilities of all GEN(x) actions to zeros except for the correct next token.

## 4 Experiments

To fairly compare GPST and GPT-2, we pre-train both models from scratch on the same corpus with the same setups and comparable parameter sizes. Evaluation is conducted on various language understanding/generation tasks. Besides, we also evaluate GPST on grammar induction to verify to what extent the induced parse trees are consistent with human annotation.

Pre-training Corpus. We pre-train models on WikiText-103 (Merity et al., 2017) and OpenWeb-Text (Gokaslan and Cohen, 2019), where the two datasets contain 116 million tokens and 9 billion tokens, respectively. The context window size in pre-training is set to 1024. When a context involves more than one complete sentence, parse trees are induced for each sentence separately.

Hyper-parameters. Following GPT-2 (Radford et al., 2019), we use 768/1024-dimensional embeddings, a vocabulary size of 30522, 3072/4096- dimensional hidden layer representations, and

<table><tr><td>Models</td><td>corpus</td><td>SST2</td><td>COLA MRPC(f1)</td><td></td><td>QQP(f1)</td><td>QNLI RTE</td><td>MNLI-(m/mm)</td><td></td><td>average #param.</td></tr><tr><td> $\overline { { \mathrm { \mathbf { G P T } } - 2 _ { \mathrm { s m a l l } } } }$ </td><td>wiki103</td><td>88.11</td><td>27.75</td><td>80.80</td><td>85.37</td><td>83.71 53.91</td><td>75.85/75.77</td><td>71.41</td><td>1.0x</td></tr><tr><td> $\mathrm { G P S T _ { s m a l l w / o g r a d . s t o p } }$ </td><td>wiki103</td><td>88.11</td><td>29.09</td><td>81.16</td><td>84.98</td><td>84.62 53.19</td><td>75.87/75.88</td><td>71.61</td><td>1.05x</td></tr><tr><td> $\mathrm { G P S T _ { \mathrm { s m a l l } } }$  w/o surrogate</td><td>wiki103</td><td>88.07</td><td>29.24</td><td>80.98</td><td>85.08</td><td>84.0552.71</td><td>76.47/76.36</td><td>71.62</td><td>1.05x</td></tr><tr><td> $\mathrm { G P S T _ { \mathrm { s m a l l } } }$ </td><td>wiki103</td><td>88.34</td><td>28.41</td><td>81.21</td><td>85.33</td><td>85.08 56.08</td><td>76.60/76.46</td><td>72.19</td><td>1.05x</td></tr><tr><td> $\overline { { \mathrm { { G P T } } { - } { 2 } _ { \mathrm { { s m a l l } } } } }$ </td><td>opw</td><td>90.71</td><td>40.53</td><td>83.20</td><td>86.55</td><td>85.60 58.72</td><td>79.53/79.75</td><td>75.57</td><td>1.0x</td></tr><tr><td> $\mathrm { G P T } { - } 2 _ { \mathrm { s m a l l } { - } 1 3 }$ </td><td>opw</td><td>91.28</td><td>44.07</td><td>83.99</td><td>86.75</td><td>85.9458.84</td><td>79.46/79.82</td><td>76.27</td><td>1.1x</td></tr><tr><td> $\mathrm { G P S T _ { \mathrm { s m a l l } } }$ </td><td>opw</td><td>90.94</td><td>44.51</td><td>84.72</td><td>86.70</td><td>86.91 64.98</td><td>79.60/80.15</td><td>77.31</td><td>1.05x</td></tr><tr><td> $\mathrm { G P T } { - } 2 _ { \mathrm { m e d i u m } }$ </td><td>opw</td><td>91.10</td><td>47.55</td><td>83.68</td><td>87.17</td><td>86.64 61.49</td><td>81.35/81.05</td><td>77.50</td><td>2.0x</td></tr><tr><td> $\mathrm { G P T } { - } 2 _ { \mathrm { m e d i u m } { - } 2 5 }$ </td><td>opw</td><td>91.55</td><td>46.81</td><td>83.43</td><td>87.27</td><td>86.99 59.92</td><td>81.19/80.78</td><td>77.24</td><td>2.1x</td></tr><tr><td> $\mathrm { G P S T _ { m e d i u m } }$ </td><td>opw</td><td>91.97</td><td>50.79</td><td>85.69</td><td>87.36</td><td>87.60 64.86</td><td>81.80/82.01</td><td>79.01</td><td>2.1x</td></tr><tr><td> $\mathbf { \overline { { F o r ~ R e f e r e n c e } } }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathbf { \overline { { O r d e r e d - M e m o r y } } ^ { \dagger } }$ </td><td></td><td>90.40</td><td></td><td>-1-</td><td>-1-</td><td></td><td>72.53/73.20</td><td></td><td></td></tr></table>

Table 1: Evaluation results on GLUE benchmark. We mark out the best result of each group in bold. The results of Ordered-Memory<sup>†</sup>(Shen et al., 2019a) are copied from Ray Chowdhury and Caragea (2023). A comparison of parameter sizes can be found in Appendix A.6.

12/16 attention heads for the generative models of $\mathrm { G P S T _ { s m a l l } }$ and $\mathrm { { G P S T } _ { \mathrm { { m e d i u m } } } } .$ , respectively. To align with Transformer layer counts in GPT-2, we configure $\mathrm { G P S T _ { s m a l l } }$ with 3 type layers and 9 token layers, and $\mathrm { G P S T _ { m e d i u m } }$ with 3 type layers and 21 token layers, respectively. We set the input dimension of the composition model to 256/512, and the number of Transformer layers used in the composition function and decomposition function to 4 and 1, corresponding to the small and medium setups. To compare with GPT-2 under the same parameter sizes, we provide $\mathrm { G P T } { - } 2 _ { \mathrm { s m a l l } \_ 1 3 }$ and $\mathrm { G P T } { - } 2 _ { \mathrm { m e d i u m } { \ } { - } { 2 } 5 }$ as additional baselines, representing models with 13 and 25 layers, respectively. The token embeddings are down-scaled before being fed into the composition model, and the constituent representations are up-scaled before being fed into GPST. All models are trained on 8 A100 GPUs with a learning rate of $5 \mathrm { e } { - } 5 / 1 \mathrm { e } { - } 4 , 8 \times 3 2 \times 1 0 2 4$ tokens per step, 5 billion and 15 billion total training tokens for WikiText-103 and OpenWebText, respectively.

## 4.1 Understanding Tasks

Dataset. We evaluate GPST on the GLUE benchmark (Wang et al., 2018), which collects tasks covering a broad range of natural language understanding (NLU) domains.

Evaluation Settings. We borrow and minimally modify the fine-tuning paradigm from Radford et al. (2018). Details are described in Appendix A.3. As the whole sentence is given, the composition model is utilized to induce the best tree and compose constituent representations as described in the E-step. The constituent representations in the induced tree are gathered in post-order as inputs for the generative model. We derive two additional baselines $\mathrm { G P S T _ { w / o \ s u r r o g a t e } }$ and $\mathrm { G P S T _ { w / o } g r a d . s t o p }$ for ablation study. In $\mathrm { G P S T _ { w / o \ s u r r o g a t e : } }$ , all constituent representations of non-terminals are replaced by embeddings of a placeholder COMP as in Transformer Grammars (Sartran et al., 2022), and thus there is no interaction between the composition model and the generative model (i.e., they are separately optimized). In $\mathrm { G P S T _ { w / o } }$ <sub>grad.stop</sub>, partial gradient stopping is disabled to study the impact of left-leaning trees on downstream tasks. We run three rounds of fine-tuning with different seeds and report average results (accuracy by default) on the validation sets.

Results and Discussions. Table 1 reports the results on the GLUE benchmark. GPST significantly outperforms GPT-2 in both small and medium setups. We find that $\mathrm { G P S T _ { w / o } g r a d . s t o p }$ and $\mathrm { G P S T _ { w / o \ s u r r o g a t e } }$ underperform GPST, but are still better than GPT-2. The performance drop of $\mathrm { G P S T _ { w / o } g r a d . s t o p }$ indicates that poor structures compromise the performance of downstream tasks. $\mathrm { G P S T _ { w / o \ s u r r o g a t e } }$ is better than GPT-2, implying that as long as induced syntactic structures are utilized, simply replacing non-terminal representations with COMP embeddings is also helpful. It is, however, worse than GPST, demonstrating that computing representations of non-terminal constituents via an explicit composition function further benefits language understanding. One more interesting thing we find is that GPST consistently and most significantly outperforms baselines on the RTE task. One possible explanation is that certain relationships in RTE are predicated on negation words such as “not”, which generally affects high-level semantics through compositions with other phrases. Explicit syntactic composition modeling contributes to a better representation of such cases.

## 4.2 Generation Tasks

## 4.2.1 Abstractive Summarization

Datasets. We conduct experiments on three summarization datasets: BBC extreme (XSum) (Narayan et al., 2018), CNN and DailyMail (Nallapati et al., 2016), and Gigaword (Napoles et al., 2012) to assess the performance of GPST in terms of language generation abilities. Statistics of the datasets are presented in Table 6 in Appendix A.4.

Evaluation Settings. For XSum and CNN/DailyMail, we truncate the documents and their summaries to 900 and 100 tokens respectively, and concatenate them with short prompt Summary:. For Gigaword, the truncating thresholds of documents and summaries are set to 400 and 120 respectively, following the settings of Rothe et al. (2020). Considering the complexity of the generation task, we primarily evaluate the models pre-trained on OpenWebText. More details are described in $\mathsf { A p - }$ pendix A.5. We apply the word-level search described in §3.3 to top-k random sampling for GPST, except for models with w/o sync which only uses naive action-level beam search. ROUGE (Lin and Hovy, 2003) is employed as the evaluation metric.

## 4.2.2 Syntactic Generalization

Datasets. The syntactic generalization task (Hu et al., 2020) collects 34 test suites to assess syntactic generalizability of the models. The test suites are grouped into 6 circuits: Agreement (Agr.), Center Embedding (C.E), Garden-Path Effects (G.P.E), Cross Syntactic Expectation (C.S.E.), Licensing (Lcs.) and Long-Distance Dependencies (L.D.D.).

Evaluation Settings. We evaluate models on syntactic generalization test suites by comparing surprisals (Hale, 2001) without fine-tuning, as required by Hu et al. (2020). Surprisal: $S ( w | C ) =$ $- \log _ { 2 } p ( w | C )$ is defined as negative log conditional probabilities of a sub-sentence w given the left-side context C. In detail, when we apply wordlevel search with beam size b to do left-to-right parsing with a given input, we temporarily store b best hypotheses with their probability $p ( \mathbf { x } _ { < t } , \mathbf { y } _ { < n ( t ) } )$ at each token position t, in which $\mathbf { y } _ { < n ( t ) }$ refers to the current latent structure before generating $x _ { t }$ We marginalize $\mathbf { y } _ { < n ( t ) }$ out of $p ( \mathbf { x } _ { < t } , \mathbf { y } _ { < n ( t ) } )$ by summing up all the probabilities of the b best hypotheses. Finally, we obtain the surprisal of a subsentence with starting position s and ending position e as $S ( w | C ) = - \log p ( x _ { < e } )$ + log $p ( x _ { < s - 1 } )$

To align with Murty et al. (2023) and Sartran et al. (2022), we set beam size b to 300.

## 4.2.3 Results and Discussions

Table 2 and 3 report the results of summarization and syntactic generalization tasks. Overall, the performance of GPST is comparable to GPT, with a slight advantage. One possible reason why the advantage of GPST on generalization tasks is not as significant as that on GLUE is the discrepancies between training and inference. During training, the constituent representations are computed via the inside algorithm, where the representations are soft-weighed over composed representations of valid sub-constituents. However, during inference, constituent representations are composed of the top two elements in the stack, which is a one-hot version of the inside algorithm. This issue could potentially be resolved using a hard inside-outside algorithm (Drozdov et al., 2020), which we may explore in our future work. Despite the discrepancies, our performance still slightly surpasses that of GPT-2, which adequately demonstrates the potential of GPST in generation tasks. One more interesting thing is that $\mathrm { G P S T _ { m e d i u m } }$ even outperforms baselines with gold trees in the syntactic generalization task, and the results of all GPSTs maintain a lead on Garden-Path Effect. Note that the results have a large variance due to the relatively small size of the evaluation set, e.g., $\mathbf { G P T _ { m e d i u m } }$ even underperforms $\mathrm { G P T _ { s m a l l } }$ . However, the results still imply that unsupervised syntactic LMs have reached a critical point where they can surpass approaches reliant on gold trees.

## 4.3 Grammar Induction

Baselines & Dataset. We select baselines that report unsupervised left-to-right parsing results: Neural variational (NV) approaches (Li et al., 2019) and PRPN (Shen et al., 2018). For reference, we also select some baselines performing parsing requiring whole sentence visible: URNNG (Kim et al., 2019b), C-PCFG (Kim et al., 2019a), DIORA (Drozdov et al., 2019), ON-LSTM (Shen et al., 2019b), Fast-R2D2 (Hu et al., 2022) and GPST using the deep inside algorithm. We report their performance on PTB (Marcus et al., 1993). Besides, we also report results of GPST<sub>w/o grad.stop</sub> and $\mathrm { G P S T _ { w / o \ s u r r o g a t e } }$ for checking the gains from partial gradient stopping and joint pre-training achieved by the representation surrogate.

<table><tr><td rowspan="2">Models</td><td rowspan="2">#param.</td><td colspan="3">XSum</td><td rowspan="2"></td><td colspan="3">CNN/DailyMail</td><td rowspan="2"></td><td colspan="3">Gigaword</td></tr><tr><td>R-1 R-2</td><td>R-L</td><td>R-AVG</td><td>R-1</td><td>R-2</td><td>R-L R-AVG</td><td>R-1</td><td>R-2 R-L</td><td>R-AVG</td></tr><tr><td> $\overline { { \mathrm { { G P T } } { - } { 2 } _ { \mathrm { { s m a l l } } } } }$ </td><td>1.0</td><td>29.78</td><td>9.43</td><td>23.56</td><td>20.92</td><td>35.54</td><td>414.45</td><td>24.76</td><td>24.92</td><td>32.45 14.84 30.37</td><td></td><td>25.88</td></tr><tr><td> $\mathrm { G P T } { - } 2 _ { \mathrm { s m a l l } \_ 1 3 }$ </td><td>1.1</td><td>29.84</td><td>9.46</td><td>23.62</td><td>20.97</td><td>35.7814.5824.92</td><td></td><td>25.09</td><td></td><td>32.7114.54 30.35</td><td></td><td>25.87</td></tr><tr><td> $\mathrm { G P S T _ { \mathrm { s m a l l - w / o ~ s y n c } } }$ </td><td>1.05</td><td>29.44</td><td>9.09</td><td>23.20</td><td>20.58</td><td>35.6314.5724.93</td><td></td><td>25.04</td><td></td><td>32.3414.6929.98</td><td></td><td>25.67</td></tr><tr><td> $\mathrm { G P S T _ { \mathrm { s m a l l } } }$ </td><td>1.05</td><td>29.86</td><td>9.51</td><td>23.70</td><td>21.02</td><td>35.5214.65 25.01</td><td></td><td>25.06</td><td></td><td>32.5314.7630.37</td><td></td><td>25.89</td></tr><tr><td> $\overline { { \mathrm { ~ G P T }  – 2 _ { \mathrm { m e d i u m } } } }$ </td><td>2.0</td><td>31.91</td><td>11.11 25.28</td><td></td><td>22.76</td><td>37.18</td><td>15.23 25.59</td><td>26.00</td><td></td><td>33.13 15.27 30.85</td><td></td><td>26.42</td></tr><tr><td> $\mathrm { G P T } { - } 2 _ { \mathrm { m e d i u m } { - } 2 5 }$ </td><td>2.1</td><td>31.95</td><td>11.1725.35</td><td></td><td>22.82</td><td>37.1315.2625.59</td><td></td><td>25.99</td><td></td><td>33.4915.2731.28</td><td></td><td>26.68</td></tr><tr><td> $\mathrm { G P S T _ { m e d i u m \ : w / o \ s y n c } }$ </td><td>2.1</td><td>31.6610.91 25.16</td><td></td><td></td><td>22.58</td><td>37.0715.45 25.69</td><td></td><td>26.07</td><td></td><td>32.8315.06 30.59</td><td></td><td>26.16</td></tr><tr><td> $\mathrm { \underline { { G P S T _ { m e d i u m } } } }$ </td><td>2.1</td><td>31.9611.31 25.58</td><td></td><td></td><td>22.95</td><td>37.1815.69 26.00</td><td></td><td>26.29</td><td></td><td>33.1915.27 30.91</td><td></td><td>26.46</td></tr></table>

Table 2: Abstractive summarization results.

<table><tr><td>Models</td><td>Agr.</td><td>C.E.</td><td>G.P.E.</td><td>C.S.E. Lcs.</td><td>L.D.D.</td><td>avg</td></tr><tr><td colspan="7">WikiText-103</td></tr><tr><td> $\overline { { \mathrm { { G P T 2 } _ { \mathrm { { s m a l l } } } } } }$ </td><td>50.88 73.21</td><td></td><td>77.88</td><td>97.83</td><td>33.95 65.98</td><td>66.62</td></tr><tr><td> $\mathrm { G P S T _ { \mathrm { s m a l l } } }$ </td><td>59.6573.21</td><td></td><td>87.10</td><td>97.83</td><td>57.89 64.78</td><td>73.41</td></tr><tr><td colspan="7">OpenWebText</td></tr><tr><td> $\overline { { \mathrm { { G P T } _ { \mathrm { { s m a l l } } } } } }$ </td><td>78.95 87.50</td><td>85.22</td><td>97.83</td><td>371.58</td><td>78.65</td><td>83.29</td></tr><tr><td> $\mathrm { G P S T _ { \mathrm { s m a l l } } }$ </td><td>77.1985.71</td><td></td><td>94.54</td><td>96.74</td><td>68.95 72.38</td><td>82.59</td></tr><tr><td> $\mathbf { G P T _ { \mathrm { { m e d i u m } } } }$ </td><td>64.91 94.64</td><td></td><td>86.41</td><td>98.91</td><td>73.42 79.38</td><td>82.95</td></tr><tr><td> $\mathrm { { G P S T } _ { \mathrm { { m e d i u m } } } }$ </td><td>85.9685.71</td><td></td><td>95.04</td><td>94.57</td><td>83.68 78.17</td><td>87.19</td></tr><tr><td colspan="7">For Reference (Models with gold trees)</td></tr><tr><td>TG</td><td>69.7 88.4</td><td>90.4</td><td>95.6</td><td>78.1</td><td>77.9</td><td>83.35</td></tr><tr><td>Pushdown Layers</td><td>79.0</td><td>92.0</td><td>84.2</td><td>100.0</td><td>77.8 77.5</td><td>85.08</td></tr></table>

Table 3: Syntactic generalization results. For reference, we list the results of models with gold trees from Sartran et al. (2022) and Murty et al. (2023).

Evaluation Settings. We continue to fine-tune all models on the training set of PTB for 10 epochs with batch size set to 32 after pre-training. Since GPST takes word pieces as inputs, we provide our model with word-piece boundaries as nonsplittable spans to align with models with wordlevel inputs. We apply different inference algorithms to grammar induction. For the inside algorithm, we directly evaluate the parse tree induced by the composition model. For the left-to-right parsing, we apply improved word-level search described in §3.3 with a beam size of 20 to parse the given text, except for $\mathrm { G P S T _ { w / o \ s y n c } }$ which employs action-level sync beam search for parsing. We adopt sentence-level unlabeled $F _ { 1 }$ as the evaluation metric, with the same setup as Kim et al. (2019a) where punctuations are discarded and words are lowercased. We evaluate the checkpoints from all epochs on the validation set, pick the best one, and then report its performance on the test set.

Results and Discussions. There are several observations from the results shown in Table 4. First and foremost, we find that our unsupervised leftto-right parsing achieves comparable performance with the bi-directional inside algorithm, significantly surpassing previous left-to-right grammar induction baselines. Such results indicate the structures generated by GPST are meaningful and consistent with those from humans. Secondly, a larger pre-training corpus may not necessarily bring improvement. A plausible explanation is that Open-WebText, mixed with more non-natural text such as URLs, introduces additional noise, leading to a performance drop. The results indicate the importance of high-quality corpora for structural learning. Thirdly, the performance of $\mathrm { G P S T _ { w / o } }$ surrogate infering with the inside algorithm drops a lot. We suppose the main reason is that disabling the representation surrogate prevents the composition model from receiving long-term feedback from tokens on the right introduced by the auto-regression loss. Lastly, the performance decline of $\mathrm { G P S T } _ { \mathrm { w / o ~ g r a d } }$ .stop corroborates the impact of asymmetric loss on structural learning. We attach trees parsed by GPST in Appendix A.8 for case studies.

<table><tr><td>Models</td><td>corpus</td><td>left-to-right</td><td>F1</td></tr><tr><td>NV(unsupervised)</td><td>WSJ</td><td>yes</td><td>29.0</td></tr><tr><td>NV(+linguistic rules)</td><td>WSJ</td><td>yes</td><td>42.0</td></tr><tr><td>PRPN</td><td>WSJ</td><td>yes</td><td>37.4</td></tr><tr><td> $\mathrm { G P S T _ { s m a l l w / o \ s y n c } }$ </td><td>wiki103</td><td>yes</td><td>43.64</td></tr><tr><td> $\mathrm { G P S T _ { s m a l l } }$ </td><td>wiki103</td><td>yes</td><td>55.25</td></tr><tr><td> $\mathrm { G P S T _ { s m a l l w / o \ s y n c } }$ </td><td>opw</td><td>yes</td><td>43.09</td></tr><tr><td> $\mathrm { G P S T _ { s m a l l } }$ </td><td>opw</td><td>yes</td><td>51.40</td></tr><tr><td> $\mathrm { G P S T _ { m e d i u m \ : w / o \ s y n c } }$ </td><td>opw</td><td>yes</td><td>43.37</td></tr><tr><td>GPSTmedium</td><td>opw</td><td>yes</td><td>54.71</td></tr><tr><td>For Reference</td><td></td><td></td><td></td></tr><tr><td>GPSTs small w/o grad.stop</td><td>wiki103</td><td>no</td><td>42.46</td></tr><tr><td>GPSTsmall w/o surrogate</td><td>wiki103</td><td>no</td><td>50.27</td></tr><tr><td> $\mathrm { G P S T _ { s m a l l } }$ </td><td>wiki103</td><td>no</td><td>57.46</td></tr><tr><td> $\mathrm { G P S T _ { s m a l l } }$ </td><td>opw</td><td>no</td><td>53.95</td></tr><tr><td> $\underline { { \mathrm { G P S T } _ { \mathrm { m e d i u m } } } }$ </td><td>opw</td><td>no</td><td>56.27</td></tr><tr><td>URNNG</td><td>WSJ</td><td>no</td><td>45.4</td></tr><tr><td>ON-LSTM</td><td>WSJ</td><td>no</td><td>47.4</td></tr><tr><td>C-PCFG</td><td>WSJ</td><td>no</td><td>55.2</td></tr><tr><td>DIORA</td><td>WSJ</td><td>no</td><td>55.7</td></tr><tr><td>Fast-R2D2</td><td>wiki103</td><td>no</td><td>57.2</td></tr><tr><td>Oracle</td><td></td><td></td><td>84.3</td></tr></table>

Table 4: Results on unsupervised left-to-right parsing.

## 4.4 Training Efficiency

Finally, we conduct a fair comparison of training efficiency with other unsupervised SLMs. We keep the model sizes and memory usage comparable and the training tokens the same. We report their time consumption in Table 5, from which we can observe the huge advantage of GPST over the baselines in terms of efficiency.

<table><tr><td></td><td>#param.</td><td colspan="4">sentence length</td></tr><tr><td>GPST</td><td>24M</td><td>128 1x</td><td>256 1x</td><td>512 1x</td><td>1024 1x</td></tr><tr><td>URNNG</td><td>23M</td><td>130.6x</td><td>955.3x</td><td>n/a</td><td>n/a</td></tr><tr><td></td><td></td><td>2.0x</td><td></td><td>25.4x</td><td>63.3x</td></tr><tr><td>OM</td><td>28M</td><td></td><td>9.2x</td><td></td><td></td></tr></table>

Table 5: Training acceleration on the same number of tokens.

## 5 Conclusion

In this paper, we propose an unsupervised approach to train GPST at scale efficiently. A key insight of our work is to guide the left-to-right structural learning with symmetric supervision such as an auto-encoding loss, which can receive feedback from both sides. A key technical contribution is that we propose the representation surrogate which enables joint training of all components in parallel. Besides, the composition model of GPST can be regarded as an enhancement to the conventional embedding layer, which provides context-invariant embeddings of various granularities beyond token embeddings. Our experiment results show the superiority of GPST on language understanding, generation, and left-to-right grammar induction, which demonstrate the potential of GPST as a foundational architecture for large language models.

## 6 Limitation

Despite GPST achieving a multiple-fold acceleration compared to previous syntactic language models, it still requires 1.5 to 5 times the training time compared to vanilla GPTs. The more layers there are in the type/token layers, the lower the overall time multiplier becomes. The additional training time comes from the composition model which only accounts for one-tenth of the overall model parameters. Due to the unpredictable memory overhead at each step, many fragmented memories are generated, resulting in PyTorch having to spend extra time cleaning up the memory cache periodically. Meanwhile, our implementation is quite naive, without any operator fusion or hardwareaware implementation. Thus there should be multiple potential ways to further reduce the time consumption of the composition model in the future.

## 7 Acknowledgement

This work was supported by Ant Group through the CCF-Ant Research Fund.

## References

James K. Baker. 1979. Trainable grammars for speech recognition. Journal of the Acoustical Society of America, 65.

Yejin Bang, Samuel Cahyawijaya, Nayeon Lee, Wenliang Dai, Dan Su, Bryan Wilie, Holy Lovenia, Ziwei Ji, Tiezheng Yu, Willy Chung, et al. 2023. A multitask, multilingual, multimodal evaluation of chatgpt on reasoning, hallucination, and interactivity. arXiv preprint arXiv:2302.04023.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Eugene Charniak et al. 2016. Parsing as language modeling. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2331–2336.

Ciprian Chelba. 1997. A structured language model. In 35th Annual Meeting ofthe Associationfor Computational Linguistics and 8th Conference ofthe European Chapter of the Association for Computational Linguistics, Proceedings of the Conference, 7-12 July 1997, Universidad Nacional de Educacion a Distan-´ cia (UNED), Madrid, Spain, pages 498–500. Morgan Kaufmann Publishers / ACL.

Ciprian Chelba and Frederick Jelinek. 2000. Structured language modeling. Comput. Speech Lang., 14(4):283–332.

John Cocke. 1969. Programming Languages and Their Compilers: Preliminary Notes. New York University, USA.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers), pages 4171–4186. Association for Computational Linguistics.

Andrew Drozdov, Subendhu Rongali, Yi-Pei Chen, Tim O’Gorman, Mohit Iyyer, and Andrew McCallum. 2020. Unsupervised parsing with S-DIORA: single tree encoding for deep inside-outside recursive autoencoders. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020, pages 4832–4845. Association for Computational Linguistics.

Andrew Drozdov, Patrick Verga, Mohit Yadav, Mohit Iyyer, and Andrew McCallum. 2019. Unsupervised

latent tree induction with deep inside-outside recursive auto-encoders. In Proceedings ofthe 2019 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 1129–1141, Minneapolis, Minnesota. Association for Computational Linguistics.

Brian DuSell and David Chiang. 2024. Stack attention: Improving the ability of transformers to model hierarchical patterns. In The Twelfth International Conference on Learning Representations.

Chris Dyer, Miguel Ballesteros, Wang Ling, Austin Matthews, and Noah A. Smith. 2015. Transitionbased dependency parsing with stack long short-term memory. In Proceedings of the 53rd Annual Meeting ofthe Associationfor Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 334–343, Beijing, China. Association for Computational Linguistics.

Chris Dyer, Adhiguna Kuncoro, Miguel Ballesteros, and Noah A. Smith. 2016. Recurrent neural network grammars. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 199–209.

Marta Dynel. 2009. Humorous garden-paths: A pragmatic-cognitive study. Cambridge Scholars.

Angela Fan, Mike Lewis, and Yann N. Dauphin. 2018. Hierarchical neural story generation. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, ACL 2018, Melbourne, Australia, July 15-20, 2018, Volume 1: Long Papers, pages 889–898. Association for Computational Linguistics.

Aaron Gokaslan and Vanya Cohen. 2019. Openwebtext corpus. http://Skylion007.github.io/ OpenWebTextCorpus.

John Hale. 2001. A probabilistic earley parser as a psycholinguistic model. In Second meeting of the north american chapter ofthe associationfor computational linguistics.

Jennifer Hu, Jon Gauthier, Peng Qian, Ethan Wilcox, and Roger Levy. 2020. A systematic assessment of syntactic generalization in neural language models. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 1725–1744, Online. Association for Computational Linguistics.

Xiang Hu, Xinyu Kong, and Kewei Tu. 2023. A multigrained self-interpretable symbolic-neural model for single/multi-labeled text classification. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

Xiang Hu, Haitao Mi, Liang Li, and Gerard de Melo. 2022. Fast-R2D2: A pretrained recursive neural network based on pruned CKY for grammar induction and text representation. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 2809–2821, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Xiang Hu, Haitao Mi, Zujie Wen, Yafang Wang, Yi Su, Jing Zheng, and Gerard de Melo. 2021. R2D2: recursive transformer based on differentiable tree for interpretable hierarchical language modeling. In Proceedings of the 59th Annual Meeting of the Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021, pages 4897– 4908. Association for Computational Linguistics.

Xiang Hu, Qingyang Zhu, Kewei Tu, and Wei Wu. 2024. Augmenting transformers with recursively composed multi-grained representations. In The Twelfth International Conference on Learning Representations.

Frederick Jelinek and John D. Lafferty. 1991. Computation of the probability of initial substring generation by stochastic context-free grammars. Computational Linguistics, 17(3):315–353.

Tadao Kasami. 1966. An efficient recognition and syntax-analysis algorithm for context-free languages. Coordinated Science Laboratory Report no. R-257.

Yoon Kim, Carl Denton, Luong Hoang, and Alexander M. Rush. 2017. Structured attention networks. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24- 26, 2017, Conference Track Proceedings. OpenReview.net.

Yoon Kim, Chris Dyer, and Alexander Rush. 2019a. Compound probabilistic context-free grammars for grammar induction. In Proceedings of the 57th Annual Meeting ofthe Associationfor Computational Linguistics, pages 2369–2385, Florence, Italy. Association for Computational Linguistics.

Yoon Kim, Alexander M Rush, Lei Yu, Adhiguna Kuncoro, Chris Dyer, and Gabor Melis. 2019b. Unsu-´ pervised recurrent neural network grammars. In Proceedings of NAACL-HLT, pages 1105–1117.

K. Lari and S.J. Young. 1990. The estimation of stochastic context-free grammars using the inside-outside algorithm. Computer Speech & Language, 4(1):35– 56.

Bowen Li, Jianpeng Cheng, Yang Liu, and Frank Keller. 2019. Dependency grammar induction with a neural variational transition-based parser. In The Thirty-Third AAAI Conference on Artificial Intelligence, AAAI 2019, The Thirty-First Innovative Applications ofArtificial Intelligence Conference, IAAI 2019, The Ninth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2019, Honolulu, Hawaii,

USA, January 27 - February 1, 2019, pages 6658– 6665. AAAI Press.

Chen Liang, Jonathan Berant, Quoc Le, Kenneth D. Forbus, and Ni Lao. 2017. Neural symbolic machines: Learning semantic parsers on Freebase with weak supervision. In Proceedings ofthe 55th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 23–33, Vancouver, Canada. Association for Computational Linguistics.

Chin-Yew Lin and Eduard Hovy. 2003. Automatic evaluation of summaries using n-gram co-occurrence statistics. In Proceedings of the 2003 human language technology conference ofthe North American chapter ofthe associationfor computational linguistics, pages 150–157.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.

Jean Maillard, Stephen Clark, and Dani Yogatama. 2019. Jointly learning sentence embeddings and syntax with unsupervised tree-lstms. Nat. Lang. Eng., 25(4):433–449.

Mitchell P. Marcus, Beatrice Santorini, and Mary Ann Marcinkiewicz. 1993. Building a large annotated corpus of English: The Penn Treebank. Computational Linguistics, 19(2):313–330.

David McClosky, Eugene Charniak, and Mark Johnson. 2006. Effective self-training for parsing. In Human Language Technology Conference of the North American Chapter of the Association of Computational Linguistics, Proceedings, June 4-9, 2006, New York, New York, USA. The Association for Computational Linguistics.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2017. Pointer sentinel mixture models. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. Open-Review.net.

Shikhar Murty, Pratyusha Sharma, Jacob Andreas, and Christopher D Manning. 2023. Pushdown layers: Encoding recursive structure in transformer language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 3233–3247.

Ramesh Nallapati, Bowen Zhou, C´ıcero Nogueira dos Santos, C¸ aglar Gul¨ c¸ehre, and Bing Xiang. 2016. Abstractive text summarization using sequence-tosequence rnns and beyond. In Proceedings of the 20th SIGNLL Conference on Computational Natural Language Learning, CoNLL 2016, Berlin, Germany, August 11-12, 2016, pages 280–290. ACL.

Courtney Napoles, Matthew Gormley, and Benjamin Van Durme. 2012. Annotated Gigaword. In Proceedings ofthe Joint Workshop on Automatic Knowledge Base Construction and Web-scale Knowledge Extraction (AKBC-WEKEX), pages 95–100, Montreal,´ Canada. Association for Computational Linguistics.

Shashi Narayan, Shay B. Cohen, and Mirella Lapata. 2018. Don’t give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 1797–1807, Brussels, Belgium. Association for Computational Linguistics.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744.

Peng Qian, Tahira Naseem, Roger Levy, and Ramon Fer-´ nandez Astudillo. 2021. Structural guidance for transformer language models. In Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 3735–3745.

Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. 2018. Improving language understanding by generative pre-training.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners.

Jishnu Ray Chowdhury and Cornelia Caragea. 2023. Beam tree recursive cells. In Proceedings ofthe 40th International Conference on Machine Learning.

Sascha Rothe, Shashi Narayan, and Aliaksei Severyn. 2020. Leveraging pre-trained checkpoints for sequence generation tasks. Transactions ofthe Associationfor Computational Linguistics, 8:264–280.

Jenny R. Saffran, Richard N. Aslin, and Elissa L. Newport. 1996. Statistical learning by 8-month-old infants. Science, 274(5294):1926–1928.

Laurent Sartran, Samuel Barrett, Adhiguna Kuncoro, Milos Stanojeviˇ c, Phil Blunsom, and Chris Dyer.´ 2022. Transformer grammars: Augmenting transformer language models with syntactic inductive biases at scale. Transactions of the Association for Computational Linguistics, 10:1423–1439.

Yikang Shen, Zhouhan Lin, Chin-Wei Huang, and Aaron C. Courville. 2018. Neural language modeling by jointly learning syntax and lexicon. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. Open-Review.net.

Yikang Shen, Shawn Tan, Arian Hosseini, Zhouhan Lin, Alessandro Sordoni, and Aaron C Courville. 2019a. Ordered memory. Advances in Neural Information Processing Systems, 32.

Yikang Shen, Shawn Tan, Alessandro Sordoni, and Aaron C. Courville. 2019b. Ordered neurons: Integrating tree structures into recurrent neural networks. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net.

Mitchell Stern, Daniel Fried, and Dan Klein. 2017. Effective inference for generative neural parsing. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 1695–1700, Copenhagen, Denmark. Association for Computational Linguistics.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 5998–6008.

Oriol Vinyals, Lukasz Kaiser, Terry Koo, Slav Petrov, Ilya Sutskever, and Geoffrey E. Hinton. 2015. Grammar as a foreign language. In Advances in Neural Information Processing Systems 28: Annual Conference on Neural Information Processing Systems 2015, December 7-12, 2015, Montreal, Quebec, Canada, pages 2773–2781.

Bo Wan, Wenjuan Han, Zilong Zheng, and Tinne Tuytelaars. 2022. Unsupervised vision-language grammar induction with shared structure modeling. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. 2018. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In Proceedings ofthe 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, pages 353–355, Brussels, Belgium. Association for Computational Linguistics.

Dani Yogatama, Phil Blunsom, Chris Dyer, Edward Grefenstette, and Wang Ling. 2017. Learning to compose words into sentences with reinforcement learning. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. Open-Review.net.

Ryo Yoshida and Yohei Oseki. 2022. Composition, attention, or both? In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 5822–5834, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Daniel H Younger. 1967. Recognition and parsing of context-free languages in time n3. Information and control, 10(2):189–208.

## A Appendix

## A.1 Pruned inside-outside algorithm

Fast-R2D2 (Hu et al., 2022) introduces a pruned variant of the inside algorithm that reduces its complexity from $O ( n ^ { 3 } )$ to $O ( n )$ in both space and time. Building on this, ReCAT (Hu et al., 2024) extends the pruning method to the inside-outside algorithm, and further enables it to complete in approximate log n steps, whose key idea is to prune out unnecessary cells in the chart-table and encode cells in different rows simultaneously. The main idea of the pruning process is to decide which two spans should be merged at each step during the inside pass and prune out cells that would break the nonsplittable span. An unsupervised top-down parser is applied to determine the merge order of spans. Given a sentence $\mathbf { x } = [ x _ { 1 } , x _ { 2 } , \ldots , x _ { n } ]$ , the topdown parser assigns each split point a score $v _ { i }$ s.t. $1 \leq i \leq n - 1$ , and recursively split the sentence into two in the descending order of the scores shown in Figure 4 (a). Hence, the reverse order of the split points could be used to decide which cells to merge. Specifically, the pruned inside-outside algorithm works as follows:

0. Prepare merge batches according to the height of merge points in the induced tree, with the lowest merge points in the first batch, as illustrated in Figure 4(b).

1. Merge each pair of adjacent cells into one according to the current merge batch. For example, in Figure 5(b), at merge point 1, we merge $x _ { 1 }$ and $x _ { 2 }$ into $x _ { 1 : 2 } ;$ at merge point 5, we merge x<sub>5</sub> and $x _ { 6 }$ into x<sub>5:6</sub>.

2. Remove all conflicting cells that would break the now non-splittable span from Step 1, e.g., the dark cells in Figure $5 ( \mathrm { c } ) ,$ , and reorganize the chart table much like in the Tetris game as in (d).

3. Encode the cells that just descend to height m and record their valid splits in , e.g., the cells highlighted with stripes in Figure 5(d) with valid splits 2, 3 for span (1, 4) and $\{ 3 , 4 \}$ span $( 3 , 6 )$ . Go back to Step 1 until no blank cells are left.

Therefore, the entire inside process can be completed within steps equal to the height of the tree. Using the valid splits recorded for each cell during the pruning process, we now have the new inside state transition equation as:

$$
\begin{array} { r l r } & { } & { \bar { a } _ { i , j } ^ { k } = \phi _ { \alpha } ( \mathbf { i } _ { i , k } , \mathbf { i } _ { k + 1 , j } ) , \bar { \mathbf { i } } _ { i , j } ^ { k } = f _ { \alpha } ( \mathbf { i } _ { i , k } , \mathbf { i } _ { k + 1 , j } ) , } \\ & { } & { \hat { w } _ { i , j } ^ { k } = \displaystyle \frac { \exp ( \bar { a } _ { i , j } ^ { k } ) } { \sum _ { k ^ { \prime } \in \mathcal { K } _ { i , j } } \exp ( \bar { a } _ { i , j } ^ { k ^ { \prime } } ) } , \mathbf { i } _ { i , j } = \sum _ { k \in \mathcal { K } _ { i , j } } \hat { w } _ { i , j } ^ { k } \bar { \mathbf { i } } _ { i , j } ^ { k } . } \end{array}
$$

where $\boldsymbol { \kappa } _ { i , j }$ is the valid splits set for span $( i , j )$ According to $\kappa .$ we can obtain a mapping from a span to its immediate sub-spans. By reversing such mapping, we get a mapping from a span to its valid immediate parent spans denoted as $\mathcal { P } _ { \mathrm { : } }$ , which records the non-overlapping endpoint k in the parent span $( i , k )$ or $( k , j )$ for a given span $( i , j )$

Thus, for the outside pass, we have:

$$
\begin{array} { r l } & { \bar { \mathbf { o } } _ { i , j } ^ { k } = \left\{ \begin{array} { l l } { f _ { \beta } ( \mathbf { o } _ { i , k } , \mathbf { i } _ { j + 1 , k } ) } & { \mathrm { ~ i f ~ } k > j } \\ { f _ { \beta } ( \mathbf { o } _ { k , j } , \mathbf { i } _ { k , i - 1 } ) } & { \mathrm { ~ i f ~ } k < i } \end{array} \right. , } \\ & { \bar { b } _ { i , j } ^ { k } = \left\{ \begin{array} { l l } { \phi _ { \beta } ( \mathbf { o } _ { i , k } , \mathbf { i } _ { j + 1 , k } ) } & { \mathrm { ~ i f ~ } k > j } \\ { \phi _ { \beta } ( \mathbf { o } _ { k , j } , \mathbf { i } _ { k , i - 1 } ) } & { \mathrm { ~ i f ~ } k < i } \end{array} \right. , } \\ & { \check { w } _ { i , j } ^ { k } = \frac { \exp ( \bar { b } _ { i , j } ^ { k } ) } { \sum _ { k ^ { \prime } \in \mathcal { P } _ { i , j } } \exp ( \bar { b } _ { i , j } ^ { k ^ { \prime } } ) } , \mathbf { o } _ { i , j } = \displaystyle \sum _ { k \in \mathcal { P } _ { i , j } } \check { w } _ { i , j } ^ { k } \bar { \mathbf { o } } _ { i , j } ^ { k } . } \end{array}
$$

We optimize the top-down parser jointly at the M-step with GPST. Given the parse tree y induced at the E-step, we maximize $p ( \mathbf { y } | \mathbf { x } ; \boldsymbol { \Theta } )$ for the topdown parser, whose parameters are denoted as Θ. As shown in Figure $4 ( \mathrm { a } )$ , at t step, the span corresponding to a given split $a _ { t }$ is determined, which is denoted as $( i ^ { t } , j ^ { t } )$ . Thus we can minimize the negative log-likelihood of the parser as follows:

$$
\begin{array} { c } { \displaystyle p ( a _ { t } | \mathbf { x } , \Theta ) = \frac { \exp ( v _ { a _ { t } } ) } { \sum _ { k = i ^ { t } } ^ { j ^ { t } - 1 } \exp ( v _ { k } ) } , } \\ { \displaystyle C _ { p } = - \log p ( \mathbf { y } | \mathbf { x } ; \Theta ) = - \sum _ { t = 1 } ^ { n - 1 } \log p ( a _ { t } | \mathbf { x } ; \Theta ) . } \end{array}
$$

We notice that the steps to finish the pruned inside-outside algorithm depend on the highest tree in a batch, thus any extremely skewed tree may result in a significant increase in time consumption. A straightforward approach to reduce the maximum height of parse trees in a batch is to introduce a height penalty. During the inside pass, the weighted tree height of span $( i , j )$ could be computed as:

$$
\bar { h } _ { i , j } ^ { k } = \operatorname* { m a x } ( h _ { i , k } , h _ { k + 1 , j } ) + 1 , h _ { i , j } = \sum _ { k \in { \mathcal { K } } _ { i , j } } \hat { w } _ { i , j } ^ { k } \bar { h } _ { i , j } ^ { k }
$$

To minimize the impact of height penalties on grammar induction, we set a threshold $H _ { t h r s }$ which is 15 by default. Only trees that exceed this threshold will be affected.

$$
\mathcal { L } _ { h } = \frac { 1 } { n } \operatorname* { m a x } ( h _ { 1 , n } - H _ { t h r s } , 0 )
$$

Thus the final auto-encoding objective is:

$$
\mathcal { L } _ { a e } ^ { * } = \mathcal { L } _ { a e } + \mathcal { L } _ { h } + \mathcal { L } _ { p }
$$

![](images/d5bfb3ed22412a6e3b355f8abb4220ee801e9e7d0f3c6fc5dc66309ddee7f945.jpg)  
Figure 4: Fast encoding follows the order given by a top-down parser, with the merging order being the reverse order of the split point sequence $A , x _ { i }$ denotes the $i _ { t h }$ token in a sentence of length 6. Numbers in and denote the indices of the split/merge point between tokens. $v _ { j }$ denotes the split score of $j _ { t h }$ split point, predicted by the top-down parser.

![](images/5a158536ab69bad6f7f13c259e296d373e999bb80070638771a4a2e8728aaf13.jpg)  
Figure 5: The initial step of encoding in $O ( \log n )$ steps. The numbers in blue correspond to the indices of the split points introduced in Figure 4.

## A.2 Composition function and score function

![](images/83ed7720d48c0937a3cb8f1ee4d9addf90c046c8d7c495da07326225dee0d753.jpg)  
Figure 6: Model illustrations for the composition and decomposition functions.

We borrow the idea from Hu et al. (2021) to use Transformers as the backbone of the composition function $f _ { \alpha }$ . As shown in Figure 6(a), composition function $f _ { \alpha }$ takes left/right constituent representations $\mathbf { i } _ { i , k } / \mathbf { i } _ { k + 1 , j }$ along with their role embeddings [LEFT]/[RIGHT] into N-layered Transformers as inputs, passes the summation of their corresponding outputs through a layer normalization layer to get the composed representation. Decomposition function $f _ { \beta }$ works analogously as shown in Figure 6(b) and (c), with [PRT] as the role embedding for parents.

We define the score function $\phi _ { \alpha }$ as:

$$
\phi _ { \alpha } ( \mathbf { l } , \mathbf { r } ) = \mathrm { M L P } _ { \alpha } ^ { l } ( \mathbf { l } ) ^ { T } \mathrm { M L P } _ { \alpha } ^ { r } ( \mathbf { r } ) / \sqrt { d }\tag{2}
$$

where l and r are representations for left/right constituents. $\mathrm { M L P } _ { \alpha } ^ { l }$ and $\mathrm { M L P } _ { \alpha } ^ { r }$ are used to capture syntactic features from the left and right in-

puts, which convert inputs to d-dimensional vectors. Analogously, $\phi _ { \beta }$ is defined as:

$$
\begin{array} { l } { { \displaystyle \phi _ { \beta } ( \mathbf { p } , \mathbf { l } ) = \mathrm { M L P } _ { \beta } ^ { p } ( \mathbf { p } ) ^ { T } \mathrm { M L P } _ { \beta } ^ { l } ( \mathbf { l } ) / \sqrt { d } } } \\ { { \displaystyle \phi _ { \beta } ( \mathbf { p } , \mathbf { r } ) = \mathrm { M L P } _ { \beta } ^ { p } ( \mathbf { p } ) ^ { T } \mathrm { M L P } _ { \beta } ^ { r } ( \mathbf { r } ) / \sqrt { d } } } \end{array}
$$

where p is the outside representation of a parent. $\mathrm { M L P } _ { \beta } ^ { l } , \mathrm { M L P } _ { \beta } ^ { r } .$ , and ${ \mathrm { M L P } } _ { \beta } ^ { p }$ are used to capture features from left/right children and parents respectively.

## A.3 Glue fine-tuning

In detail, we append a CLS token after the input sequence and then feed the hidden states of the CLS tokens to a linear layer as the logits for classification. An additional cross-entropy loss along with the pre-training objective is used during finetuning.

## A.4 Summarization dataset statistics

BBC extreme (XSum) comprises 204k documentsummary pairs for single-sentence summarization of long documents. CNN and DailyMail (CNN/- DailyMail) contains 287k training pairs, each consisting of a document annotated with highlights. Gigaword focuses on sentence summarization with 3.8M sentence-summary training pairs conversely. We organize the statistics in Table 6.

## A.5 Summarization fine-tuning

We fine-tune for 15 epochs with a batch size of 16 on XSum and CNN/DailyMail datasets. For Gigaword, we fine-tune for 10 epochs with a batch

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>|XSum|C</td><td rowspan=1 colspan=1>NN/DailyMail|</td><td rowspan=1 colspan=1>Gigaword</td></tr><tr><td rowspan=1 colspan=1>Training SetTest Set</td><td rowspan=1 colspan=1>204k11.3k</td><td rowspan=1 colspan=1>287k11.5k</td><td rowspan=1 colspan=1>3.8M1.95k</td></tr></table>

Table 6: Detailed statistics for summarization datasets.

size of 64. Top-k random sampling with $k = 2$ is used as the basic inference method as suggested in GPT-2 (Radford et al., 2019).

## A.6 Comparison of parameter sizes

<table><tr><td rowspan=1 colspan=1>model</td><td rowspan=1 colspan=1>MLP</td><td rowspan=1 colspan=1>QKV linear|score</td><td rowspan=1 colspan=1>linear</td><td rowspan=1 colspan=1>total</td></tr><tr><td rowspan=1 colspan=1>comp. func.GPT-2</td><td rowspan=1 colspan=1>256*1024*2768*3072*2</td><td rowspan=1 colspan=1>256*256*3768*768*3</td><td rowspan=1 colspan=1>256*128*2</td><td rowspan=1 colspan=1>786,4326,488,064</td></tr></table>

Table 7: The comparison of parameter size of a single layer in the composition function of $\mathrm { \dot { G P S T } _ { s m a l l } }$ and $\mathbf { G P T - } 2 _ { \mathrm { s m a l l } } ^ { - }$

## A.7 Author Contributions Statement

We list our contributions below:

• Xiang Hu: proposing the unsupervised training approach, implementing GPST, pre-training and decoding algorithms, and paper writing.

• Pengyu Ji: GPST fine-tuning, experiments code, running experiments, and paper writing.

## A.8 Case studies

Please refer to the following pages.

<table><tr><td>System</td><td>Tree</td></tr><tr><td>Left to right</td><td><img src="images/644748b52b37ab814984be61367a43d752d28f588e009d0c2008a7df49e12831.jpg"/> Skipper 's said the merger will help finance remodeling and future growth</td></tr><tr><td>Left to right w/o sync</td><td>Skipper 's said the merger will help finance remodeling and future growth <img src="images/5f2c4927f8292fc2b39c7104d24ecb5d13fbfc8ee63dead7db24655c4529fe34.jpg"/></td></tr><tr><td>Inside</td><td>Skipper 's said the merger will help finance remodeling and future growth <img src="images/f1071572a044b80bdaf43a6a007382ea81a4de0907e85611f1b89b2150b1ce3b.jpg"/></td></tr><tr><td>Left to right</td><td><img src="images/c9c033b8090d04e9bf1c96530aacfbff23e06d878993a58338fbb47481ba35fc.jpg"/> Yesterday the stock market 's influence at first created nervousness</td></tr><tr><td>Left to right w/o sync</td><td>Yesterday the stock market 's influence at first created nervousness <img src="images/7dbd53816b593a266986a6b6487f8f53c395f40dafb21101736419b400a72189.jpg"/></td></tr><tr><td>Inside</td><td>Yesterday the stock market 's influence at first created nervousness <img src="images/a569d4da617351b87b4f6871c584841aba4db4dc7e2abd3f569362cba14aba30.jpg"/></td></tr><tr><td>Gold</td><td>Yesterday the stock market 's influence at first created nervousness</td></tr><tr><td>Left to right</td><td>And yesterday the top performing industry group was oil field equipment issues <img src="images/7d3dddd25039426aa03855ca371e7a7c18842d5d31070c1c800e048e2c803168.jpg"/></td></tr><tr><td>Left to right w/o sync</td><td>And yesterday the top performing industry group was oil field equipment issues</td></tr><tr><td>Inside</td><td>And yesterday the top performing industry group was oil field equipment issues <img src="images/1813079606debe0195c022c11b2de0ba68190b08dcf3c99c3e1a3608a1beb477.jpg"/></td></tr><tr><td>Gold</td><td>And yesterday the top performing industry group was oil field equipment issues</td></tr><tr><td>Left to right</td><td>All told the federal government already guarantees more than 900 billion of mortgages <img src="images/6f841c902287b4c504ea75c9c967c701e3fc1b0289f7effe956893297561bd5d.jpg"/></td></tr><tr><td>Left to right w/o sync</td><td>All told the federal government already guarantees more than 900 billion of mortgages</td></tr><tr><td>Inside</td><td>All told the federal government already guarantees more than 900 billion of mortgages</td></tr><tr><td>Gold</td><td>All told the federal government already guarantees more than 900 billion of mortgages</td></tr><tr><td>Left to right</td><td><img src="images/a4f70b1e71a358891577969022d8bfd72be6940b57ad10736fdb249df7afdfaa.jpg"/> The real key is to have the economy working and interest rates down</td></tr><tr><td>Left to right w/o sync</td><td>The real key is to have the economy working and interest rates down <img src="images/9b516b00bccd837d7b6f6ac736f255e96ffd42c9a75235f585bfc932cbb2b9d6.jpg"/></td></tr><tr><td>Inside</td><td>The real key is to have the economy working and interest rates down <img src="images/be5c264c4b242bb4bf7401fe14d8e8857485eae2c7e6302e6c65c97a51f1efe4.jpg"/></td></tr><tr><td></td><td>The real key is to have the economy working and interest rates down <img src="images/9187ad77cd310c09a866504b24a36f76d854393bd54cadef9f36d93fb04a0678.jpg"/> The Dutch company had n't notified Burmah of its reason for increasing the stake he said</td></tr><tr><td>Left to right w/o sync</td><td>The Dutch company had n't notified Burmah of its reason for increasing the stake he said</td></tr><tr><td></td><td>The Dutch company had n't notified Burmah of its reason for increasing the stake he said</td></tr><tr><td>Inside</td><td><img src="images/0aa49414501e2f6506a463d322692350a594c15b86e0ffb0562fa2de4d1eec46.jpg"/></td></tr></table>