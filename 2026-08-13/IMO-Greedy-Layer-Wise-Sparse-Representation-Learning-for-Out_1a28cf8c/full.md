# IMO: Greedy Layer-Wise Sparse Representation Learning for Out-of-Distribution Text Classification with Pre-trained Models

Tao Feng, Lizhen Qu \*, Zhuang Li, Haolan Zhan, Yuncheng Hua, Gholamreza Haffari Monash University, Australia {firstname.lastname}@monash.edu

## Abstract

Machine learning models have made incredible progress, but they still struggle when applied to examples from unseen domains. This study focuses on a specific problem of domain generalization, where a model is trained on one source domain and tested on multiple target domains that are unseen during training. We propose IMO: Invariant features Masks for Out-of-Distribution text classification, to achieve OOD generalization by learning domain-invariant features. During training, IMO employs a greedy algorithm to learn sparse representations for each layer in a top-down manner. It performs better than the opposite direction and learning of sparse representations for all layers simultaneously. Our comprehensive experiments show that IMO substantially outper forms strong baselines such as prompt-based methods and large language models, in terms of various evaluation metrics and settings. <sup>1</sup>

## 1 Introduction

When deploying natural language processing (NLP) models trained on labeled data in the wild, it is well known that their predictive performance declines significantly on samples drawn from distributions that differ from their training data (Wang et al., 2021b). Although various domain adaptation (DA) methods have been proposed (Liu et al., 2022; Saunders, 2022), they assume the availability of labeled or unlabeled data from target domains, along with information about target domains. However, for many real-world applications, especially for early-stage businesses, users may apply their models to arbitrary data so the test data may well be Outof-Distribution (OOD). Hence, target domain information may not be available for DA. In addition, some training datasets are expensive to acquire so they are available only in one domain. Therefore, this work focuses on single-source domain generalization (DG) for text classification, which aims to enable classifiers trained in one source domain to robustly work on the same classification tasks in any unseen OOD data without any model tuning.

Pre-trained large language models (LLMs) have drawn a lot of attentions due to their strong predictive performance across a variety of tasks. Although generative models or classifiers built on top of pre-trained LLMs outperform prior models in multiple domains, their performance is still not robust on tasks when the testing distribution differs substantially from the training distribution (Bang et al., 2023). Recent works (Wang et al., 2021a; Feng et al., 2023; Veitch et al., 2021) show that one of the key reasons is spurious correlations, which refer to the correlations between features and model outputs that are not based on causal relationships.

To take a step towards “train it once, apply it anywhere”, we propose a novel greedy layerwise Invariant Masking technique for OOD text classification, coined IMO, which selects domaininvariant features and key token representations from appropriate layers of a pre-trained deep transformer encoder to mitigate spurious correlations. The resulting hidden representations are sparse from the top layer to a specific layer of the pretrained model. We demonstrate the effectiveness of this technique through theoretical justifications and extensive experiments. Similar to (Zhang et al., 2021) on computer vision tasks, we shed light on how to apply sparsity as an effective inductive bias to deep pre-trained models for OOD text classification. Our contributions are:

• We propose IMO, a novel top-down greedy layer-wise sparse representation learning method for pre-trained text encoders for robust OOD classification by sharply reducing task-specific spurious correlations. In comparison with bottom-up layer-wise and simultaneous search across all layers, we discover that the top-down greedy search is decisive for performance improvement.

• We develop a theoretical framework that elucidates the relationship between domaininvariant features and causal features. Additionally, we provide an explanation of how our method learns invariant features.

• Our comprehensive experimental results show that: (i) using IMO with BART (Lewis et al., 2020) significantly outperforms competitive baselines, including CHATGPT, on topic classification and sentiment polarity prediction in most of the target domains. Notably, CHATGPT has 10 times more parameters than BART; (ii) using IMO with CHATYUAN (Clue-AI, 2023) achieves superior performance in Chinese social factor classification compared to strong competitors like CHATGPT; (iii) IMO achieves robust OOD performance w.r.t. varying training data size. The accuracy difference between using 1k and 3.5 million training instances using IMO is less than 6%.

## 2 Related Work

Domain Generalization. Numerous DG methods have been proposed in the past decade, and most of them are designed for multi-source DG (Chattopadhyay et al., 2020; Zhao et al., 2020; Ding et al., 2022; Zhang et al., 2022; Lv et al., 2022). Existing DG methods can be roughly classified into two categories: invariant representation learning and data augmentation. The key idea of the former is to reduce the discrepancy between representations of source domains (Muandet et al., 2013; Li et al., 2018a,b; Shao et al., 2019; Arjovsky et al., 2020). The key idea of data augmentation is to generate out-of-distribution samples, which are used to train the neural network with original source samples to improve the generalization ability (Xie et al., 2020; Wei and Zou, 2019; Volpi and Murino, 2019).

This paper focuses on single-source DG, where the model is trained on a single source domain, then evaluated on multiple unseen domains. Wang et al. (2021c) proposes a style-complement module to synthesize images with unseen styles, which are out of original distributions. Qiao et al. (2020) proposes adversarial domain augmentation to encourage semantic consistency between the augmented and source images in the latent space. Ouyang et al. (2023) uses a causality-inspired data augmentation approach to encourage network learning domaininvariant features. In terms of text classification, Ben-David et al. (2022); Jia and Zhang (2022) apply prompt-based learning methods to generate a prompt for each sample, then use large language models to predict labels.

Causal Representation Learning (CRL). CRL addresses OOD generalization by exploring causal features that lead to labels. It is based on the assumption that causal features are stable across different environments or data selections. Since CRL is very ambitious and even infeasible in real application, a more practical method is invariant representation learning. Peters et al. (2016) investigated that invariant features, to some extent, infer the causal structure. Arjovsky et al. (2020) also assumes that prediction conditioned on invariant features is stable under different environments. Following such assumption, a strand of methods tries to learn invariant features by mitigating spurious correlated features, which vary across environments (Muandet et al., 2013; Chattopadhyay et al., 2020; Asgari et al., 2022; Izmailov et al., 2022; Hu et al., 2022b). This paper also follows this thread of methods, where we treat features that don’t affect prediction as spurious correlated features.

## 3 Learning Sparse Domain-Invariant Representations

LLMs are pre-trained on large-scale corpora so that they can capture rich correlations between tokens across various domains. To enable trained models incorporating LLMs to work across domains, our key idea originates from the Invariance Assumption that the conditional distributions of labels conditioned on invariant features do not change across domains (Peters et al., 2016). Zhang et al. (2021) show that there is a subnetwork inside a full network that can achieve better OOD performance than the full network, if this assumption holds. This hypothesis is also referred to as the functional lottery ticket (Liang et al., 2021). For a specific classification task, such as sentiment polarity analysis, the assumption indicates that there are certain sparse representations that are potential causes of labels (Wang and Jordan, 2022) across domains. Our method IMO realizes this idea by constructing sparse domain-invariant representations from the hidden representations of the selected layers of pre-trained transformer-based encoders.

Let $\mathcal { X }$ be the input space and $\mathcal { V }$ be the label space, a domain is characterized by a joint distribution $P _ { X Y }$ on $\mathcal { X } \times \mathcal { V }$ . In the context of a single source DG, we have access to the data of one source domain $\boldsymbol { \mathcal { S } } = \{ ( x ^ { s } , y ^ { s } ) \}$ drawn from its joint distribution $P _ { X Y } ^ { S }$ . The goal is to learn a predictive model $f : \mathcal { X }  \mathcal { Y }$ using only the data sampled from $P _ { X Y } ^ { S }$ to minimize the prediction error on K unseen target domains, each of which is associated with a joint distribution $P _ { X Y } ^ { k }$ . Due to domain drifts, $P _ { X Y } ^ { S } \neq P _ { X Y } ^ { k } , \forall k \in { 1 , . . . , \tilde { K } }$

Following (Quinzan et al., 2023), we make the same assumptions that (i) ${ \cal Y } ~ = ~ f ( { \bf P a } ( { \cal Y } ) ) + \epsilon .$ where $\mathrm { P a } ( Y )$ denote the features that directly cause Y, (ii) ϵ is exogenous noise, independent of any features, and (iii) Y has no direct causal effect on any features because classification labels are assigned after observing the corresponding texts. Although $P _ { X Y } ^ { S } \neq P _ { X Y } ^ { ( k ) } , \forall k \in { 1 , . . . , K }$ , we show in $\ S 3 . 3$ that under all above assumptions, there is a sparse representation $\mathbf { H } _ { i }$ such that the function $Y = f ( \mathbf { H } _ { i } ) + \epsilon$ exists in both source and target domains. We empirically study the presence of invariant representations and influence of spurious correlations in §4.3.

As illustrated in Figure 1, our method constructs sparse domain-invariant representations at both feature and token levels in a top-down manner. At the feature level, given embeddings produced by the transformer block of the top layer, a parametric mask layer identifies invariant features from the embeddings. Then, the mask layer is frozen and the algorithm learns the mask layer for the lower layer. The process is repeated until a pre-specified layer is reached. At the token level, a soft attention mechanism incorporates the selected features from the top layer to identify the tokens strongly correlated with Y and use attention weights to create aggregated sparse representations based on the selected features for binary classification. For multi-class classification tasks, a sparse representation is created for each class so that each of them can focus on class-specific information. The model is regularized during training to increase the divergences of the representations between classes.

## 3.1 Extraction of Invariant Features

Given a text input $X = [ x _ { i } ] _ { i = 0 } ^ { T }$ , where $x _ { i }$ is a token in $X$ , a transformer-based pre-trained language model is employed to convert $x _ { i }$ to a continuous token representation. We use hidden states produced by each transformer layer l as token representations, denoted as $\pmb { H } ^ { l } = [ \pmb { h } _ { i } ^ { l } ] _ { i = 0 } ^ { T } . \pmb { h } _ { i } ^ { l }$ embeds both invariant features (useful for prediction in different domains) and spuriously correlated features (irrelevant for prediction) produced by layer l. Based on the Invariance Assumption, the invariant features $h ^ { * }$ ensure $p ^ { k } ( Y | h ^ { * } )$ to be the same for each domain k. In a transformer layer l, the spuriously correlated features are filtered out by performing element-wise multiplication between token representation $h _ { i } ^ { l }$ and a learnable mask $m ^ { l }$

![](images/3ffd476ed0037429e57c573ecc08c6097ec309dc70827e7bffb45a075e889c47.jpg)  
Figure 1: The overall architecture of our method IMO.

A parametric filtering vector m $= \mathbf { r } \odot \mathbf { q }$ contains zero and non-zero elements, where we define a trainable weight vector $\mathbf { r } \in \mathbb { R } ^ { d }$ and a trainable pruning threshold vector $\mathbf { s } \in \mathbb { R } ^ { d }$ . A unit step function $g ( t ) = { \left\{ \begin{array} { l l } { 0 } & { { \mathrm { i f ~ } } t < 0 } \\ { 1 } & { { \mathrm { i f ~ } } t \geq 0 } \end{array} \right. }$ is applied to get a binary mask $\mathbf { q } = g ( | \mathbf { r } | - \mathbf { s } )$ . By applying element-wise multiplication $\mathbf { e } _ { i } ^ { l } = \mathbf { h } _ { i } ^ { l } \odot \mathbf { m } ^ { l }$ , the zero elements of m remove corresponding features in token embeddings $\mathbf { h } ^ { l } .$ , while non-zero elements characterize the importance of corresponding features (Liu et al., 2020).

As the unit step function $g$ is not differentiable, we approximate its derivative by using the derivative estimator proposed in (Xu and Cheung, 2019) such that all parameters of a mask layer are trainable by using back-propagation and the family of stochastic gradient descent algorithms,

$$
{ \frac { d } { d t } } g ( t ) = { \left\{ \begin{array} { l l } { 2 - 4 | t | , } & { - 0 . 4 \leq t \leq 0 . 4 } \\ { 0 . 4 , } & { 0 . 4 \leq | t | \leq 1 } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }\tag{1}
$$

Following (Xu and Cheung, 2019; Liu et al., 2020), we add a sparse regularization term $L _ { s p a r s e }$ to the training loss to encourage the sparsity of mask layers:

$$
\mathcal { L } _ { s p a r s e } = \sum _ { i = 1 } ^ { N } \exp ( - \pmb { s } _ { i } ) , \pmb { s } \in \mathbb { R } ^ { d }\tag{2}
$$

where $\exp ( - { \pmb s } _ { i } )$ encourages high (but not extremely large) thresholds. A higher threshold leads to removal of more features. During inference, we retain the mask layers to retain invariant features while discarding irrelevant ones.

## 3.2 Identification of Invariant Tokens

Given a long token sequence, not all information is useful for target tasks. For example, function words, such as ‘the’, or ‘that’, provide little information for predicting sentiment polarity. Thus, we employ a token-level attention mechanism to focus on important tokens. Instead of using all features of a token representation, we compute attention scores by using only the invariant features. The proposed attention mechanism differs slightly between binary and multi-class classification.

Binary Classification. For binary classification, we treat the filtering vector $\mathbf { m } ^ { L }$ from the last layer L as the query vector and compute the attention weight by performing the matrix product between $\mathbf { m } ^ { L }$ and each token embedding from the last layer ${ \bf e } _ { i } ^ { L } \colon a _ { i } = { \bf m } ^ { L } { \bf e } _ { i } ^ { L }$ . Here, the filtering vector and token embeddings are interpreted as matrices, with $\mathbf { m } ^ { L } \in \mathbb { R } ^ { 1 \times d }$ and $\mathbf { e } _ { i } ^ { L } \in \bar { \mathbb { R } } ^ { d \times 1 }$ . For an input token sequence, we aggregate the masked token embeddings to obtain a sequence representation $\begin{array} { r } { { \bf v } = \sum _ { i } ^ { T } a _ { i } { \bf e } _ { i } ^ { L } } \end{array}$ , where $\mathbf { v } \in \mathbb { R } ^ { 1 \times d }$ . Finally, the sequence representation is fed into a fully-connected layer, followed by generating a distribution over the label space as follows: $\hat { \mathbf { y } } = \operatorname { s o f t m a x } ( \mathbf { v } \mathbf { P } )$

Multi-class Classification. For the multi-class classification task, we propose using multiple mask layers $m _ { y } ^ { L }$ in the last layer L to capture corresponding features and tokens for labels y. The number of mask layers equals the number of labels. Each label has its own attention weights ${ \pmb a } _ { y } ^ { L } = m _ { y } ^ { L } { \pmb e } .$ and its own representation $\begin{array} { r } { { \pmb v } _ { y } ^ { L } = \sum _ { i } ^ { \tilde { T } } a _ { y i } ^ { L } { \pmb e } _ { i } . } \end{array}$ . Instead of using a fully-connected layer, we use a learnable weight vector per class to project $\mathbf { \nabla } _ { v } L$ to a scalar: $c ^ { L } = v ^ { L } p ^ { L }$ , where $\pmb { v } ^ { L } \in \mathbb { R } ^ { 1 \times d }$ and $\pmb { p } ^ { L } \in \mathbb { R } ^ { d \times 1 }$ . The rationale behind this is that each class should have its own weight vector and hidden representations for encoding class-specific information. Then, we concatenate these scalars to a vector $\boldsymbol { c } = [ c ^ { L } ]$ , and compute the predictive distribution by $\hat { \pmb { y } } = \mathrm { s o f t m a x } ( \pmb { c } )$

To encourage mask layers to extract labelspecific features, we propose the following regularization term to penalize pairwise cosine similarities between the corresponding mask layers (where N is the number of label-specific mask layers):

$$
\mathcal { L } _ { d i s t } = \frac { 1 } { N ( N - 1 ) } \sum _ { i \neq j } \cos ( m ^ { i } , m ^ { j } ) .\tag{3}
$$

Training Procedure. Rather than training all mask layers simultaneously, we adopt a layer-wise training procedure to train them sequentially from the top layer to the bottom layer. As illustrated in Figure 1, for each layer, a new filtering layer, $\mathbf { m } ^ { L - i }$ , is introduced on the top of the $( L - i )$ th transformer layer, with $i \in \{ 0 , 1 , 2 , . . . L - 1 \}$ Crucially, during this phase, the previously trained mask layers remain frozen to preserve their learned parameters. Upon each layer’s training completion, the model is stored as $\theta _ { L : L - i }$ . This iterative procedure continues until the training of the most bottom filtering vector, $\mathbf { m } ^ { 1 }$ , is completed. Consequently, a suite of models, ranging from $\theta _ { L }$ to $\theta _ { L : 1 }$ , is collected. We empirically determine the model’s efficacy by evaluating its performance on the validation set from the source domain. The best-performing model is chosen as the model to test on the target domains.

Objective Function. During training, the overall objective for binary classification is to (1) have good predictive performance on classification tasks and (2) maximize sparsity in mask layers to only keep invariant features. When training mask at layer l, the loss function is:

$$
\mathcal { L } = \mathcal { L } _ { c e } + \alpha \mathcal { L } _ { s p a r s i t y } ^ { l }\tag{4}
$$

where $\mathcal { L } _ { c e }$ denotes the cross entropy loss and f denotes the predictive model. $\alpha ,$ , where $\alpha > 0 ,$ is a hyperparameter that controls the balance between predictive performance and sparsity in mask layers. $\mathcal { L } _ { s p a r s i t y } ^ { l }$ is the sparse regularization term for mask at layer l.

For multi-class classification, we add a distance regularization term:

$$
\mathcal { L } = \mathcal { L } _ { c e } + \alpha \mathcal { L } _ { s p a r s i t y } ^ { l } + \beta \mathcal { L } _ { d i s t }\tag{5}
$$

![](images/e60f2fff407089e3d5147a48513b4b08726ce16154b2d5b7846750731cedf1d7.jpg)  
(a)

![](images/9db4d01a7e4ef458988a7453401c0625d3426b0fe2daf4fc5f8aa64033c09e48.jpg)  
(b)

![](images/f49e0f5eeee33b512e3ee257e50819c9fc92e0665f0702105fe8af7b847218ba.jpg)  
(c)  
Figure 2: Illustration of potential causal graphs between the variables $H _ { i } , H _ { j }$ of two features (encoded from an input X) and a target variable Y.

The hyperparameter $\beta$ serves to calibrate the equilibrium between features specific to individual labels and those shared across multiple labels.

## 3.3 Theoretical Analysis

Based on our assumptions, $Y = f ( \mathbf { H } _ { i } ) + \epsilon \exp \mathbf { + }$ ists, when $\mathbf { H } _ { i }$ are the parent nodes of Y in the underlying causal graph. Because $\mathbf { H } _ { i }$ are a subset among all possible hidden representations correlated with $Y$ , there should be a subset of hidden representations serving as parents of $Y$ , otherwise the invariance assumption does not hold. Due to the widely used faithfulness assumption stating that statistical independences imply the corresponding causal structures (Neal, 2020), we aim to find out $\mathbf { H } _ { i } \ \mathcal { A } \ Y | \mathbf { H } _ { j }$ , where $\mathbf { H } _ { j }$ is any feature set nonoverlapped with $\mathbf { H } _ { i }$

We start our theoretical analysis by introducing a sparsity regularization term $\Omega ( Y , H _ { i } , . . . , H _ { j } )$ which counts the number of edges between $Y$ and the random variables of features in an underlying causal graph, where Y is the variable for labels and $H _ { k }$ denotes the random variable of the feature $h _ { k }$ . Then we introduce a loss function $\mathcal { L } _ { \Omega } ( Y , H _ { i } , . . . , H _ { j } ) = \mathcal { L } _ { c e } + \alpha \Omega ( Y , H _ { i } , . . . , H _ { j } )$ with $\alpha > 0$ , analogous to Eq. (4).

Considering the simplest case that there is only a causal feature $h _ { i }$ and a non-causal feature $h _ { j }$ , the corresponding random variables are denoted by $H _ { i }$ and $H _ { j }$ . From any causal graphs in Fig. 2, we conclude that $p ( Y | H _ { i } , H _ { j } ) = p ( Y | H _ { i } )$ so that the cross entropy term in ${ \mathcal { L } } _ { \Omega }$ remains the same when using the term $p ( Y | H _ { i } )$ , but the loss decreases after removing the non-causal feature from the loss due to the regularization term $\Omega ( Y , H _ { i } , H _ { j } )$ .

The two feature case can be easily extended to the case having more than two features. It is trivial that excluding a non-causal feature from the loss ${ \mathcal { L } } _ { \Omega }$ leads to the decrease of ${ \mathcal { L } } _ { \Omega }$ due to the Markov property of causal graphs (Peters et al., 2017).

Corollary 1. If there is no edge between Y and $H _ { k }$ in a causal graph, then $\mathcal { L } _ { \Omega } ( Y , H _ { i } , . . . , H _ { j } ) <$ $\mathcal { L } _ { \Omega } ( Y , H _ { i } , . . . , H _ { j } , H _ { k } )$

During training, we start with a loss $\mathcal { L } _ { \Omega } ( Y , H _ { 1 } , . . . , H _ { N } )$ with a complete set of features. If a non-causal feature $H _ { k }$ is removed, $\mathcal { L } _ { \Omega } ( Y , H _ { i } , . . . , H _ { j } )$ decreases according to Corollary 1. In contrast, if a causal feature $H _ { k }$ is removed, the cross entropy term increases because the mutual information $I ( Y ; H _ { k } | H _ { i } , . . . , H _ { j } ) > 0$ Namely, $H _ { k }$ adds additional information for predicting $Y$ . However, in that case, $\mathcal { L } _ { \Omega } ( Y , H _ { i } , . . . , H _ { j } )$ may still decrease if the increase of $\mathcal { L } _ { c e }$ is smaller than the decrease of the regularization term $\alpha \mathcal { L } _ { \Omega } ( Y , H _ { i } , . . . , H _ { j } )$ , where $\alpha > 0$ . The exceptional case can be mitigated if α is sufficiently small. As a result, the loss ${ \mathcal { L } } _ { \Omega }$ provides an effective way to guide the search for the features serving as the causes of the labels, although we cannot recover the underlying true causal graphs. Herein, the loss (4) is a surrogate of $\mathcal { L } _ { \Omega } ( Y , H _ { i } , . . . , H _ { j } )$ by using a deep neural network.

## 4 Experiments

We show that our approach significantly outperforms the competitive baselines in almost all settings, followed by empirically verifying that domain-invariant sparse representations indeed exist and spurious features deteriorate model performance in Sec. 4.3, as well as justifying the effectiveness of top-down greedy search strategy and individual modules in the ablation study.

## 4.1 Experimental Setup

Tasks and Datasets We evaluate our method on binary and multi-class classification tasks. Herein, we adopt accuracy as the metric for binary sentiment polarity classification and macro-F1 for multiclass classification tasks. All models are trained with five different random seeds to assess the statistical significance.

The datasets for binary sentiment analysis include Amazon Review Polarity (Zhang et al., 2015a), Yelp Review Polarity (Zhang et al., 2015a), IMDB (Maas et al., 2011), TweetEval Sentiment (Barbieri et al., 2020) <sup>2</sup> and Yahoo! Answers Sentiment (Li et al., 2019). For multi-class classification, we consider topic classification task in AG News dataset (Gulli, 2005; Del Corso et al., 2005; Zhang et al., 2015b) and social factor prediction task in SocialDial (Zhan et al., 2023, 2024). More details about datasets can be found in Appendix A.2.

<table><tr><td rowspan="2">Models</td><td colspan="3">IMDB→</td><td colspan="3">Amazon→</td><td colspan="3">Yelp→</td><td colspan="3">TweetEval→</td><td rowspan="2">Avg.</td></tr><tr><td>Amazon</td><td>Yelp</td><td>TweetEval</td><td>IMDB</td><td>Yelp</td><td>TweetEval</td><td>IMDB</td><td>Amazon</td><td>TweetEval</td><td>IMDB</td><td>Yelp</td><td>Amazon</td></tr><tr><td>BERT</td><td>89.77*</td><td>87.12*</td><td>78.52*</td><td>88.09*</td><td>92.18*</td><td>83.75*</td><td>86.98*</td><td>92.10*</td><td>87.55*</td><td>82.59*</td><td>84.87*</td><td>86.80*</td><td>86.69*</td></tr><tr><td>BART</td><td>89.91*</td><td>88.01*</td><td>68.47*</td><td>87.93*</td><td>91.01*</td><td>82.98*</td><td>86.44*</td><td>91.97*</td><td>88.21*</td><td>78.21*</td><td>89.51*</td><td>87.01*</td><td>85.80*</td></tr><tr><td>BERT-EDA</td><td>87.73*</td><td>87.47*</td><td>72.10*</td><td>88.89*</td><td>92.43*</td><td>86.40*</td><td>88.11*</td><td>92.98*</td><td>87.92*</td><td>81.64*</td><td>85.82*</td><td>87.77*</td><td>86.61*</td></tr><tr><td>BERT-UDA</td><td>87.76*</td><td>87.02*</td><td>70.23*</td><td>89.87*</td><td>93.78*</td><td>86.37*</td><td>86.89*</td><td>92.81*</td><td>84.91*</td><td>82.83*</td><td>85.95*</td><td>87.29*</td><td>86.31*</td></tr><tr><td>BERT-PGB</td><td>88.40*</td><td>83.61*</td><td>70.51*</td><td>89.70*</td><td>93.66*</td><td>86.19*</td><td>86.09*</td><td>92.72*</td><td>87.95*</td><td>81.88*</td><td>85.13*</td><td>87.54*</td><td>86.11*</td></tr><tr><td>PADA</td><td>85.73*</td><td>89.84*</td><td>88.40</td><td>84.47*</td><td>93.96</td><td>85.92*</td><td>87.71*</td><td>91.42*</td><td>90.33</td><td>80.30*</td><td>84.69*</td><td>90.61</td><td>87.78*</td></tr><tr><td>PDA</td><td>89.35*</td><td>90.59*</td><td>87.71*</td><td>88.16*</td><td>94.20</td><td>85.61*</td><td>88.17*</td><td>93.59</td><td>89.88*</td><td>82.05</td><td>86.37</td><td>86.41</td><td>88.51*</td></tr><tr><td>CHATGPT</td><td>91.08</td><td>92.06</td><td>81.01</td><td>90.50</td><td>92.06</td><td>81.01</td><td>90.50</td><td>91.08</td><td>81.01</td><td>90.50</td><td>92.06</td><td>91.08</td><td>88.66</td></tr><tr><td>ALPACA-7B</td><td>90.14</td><td>92.30</td><td>88.66</td><td>83.01</td><td>92.30</td><td>88.66</td><td>83.01</td><td>90.14</td><td>88.66</td><td>83.01</td><td>92.30</td><td>90.14</td><td>88.52</td></tr><tr><td>ALPACA-7B-LoRA</td><td>89.80</td><td>82.80</td><td>87.77</td><td>81.00</td><td>82.80</td><td>87.77</td><td>81.00</td><td>89.80</td><td>87.77</td><td>81.00</td><td>82.80</td><td>89.80</td><td>85.34</td></tr><tr><td>IMO-BART (Our)</td><td>93.97</td><td>94.63</td><td>89.58</td><td>90.86</td><td>95.14</td><td>91.08</td><td>90.08</td><td>94.87</td><td>91.62</td><td>85.39</td><td>92.84</td><td>91.66</td><td>91.81</td></tr><tr><td>IMO-BART B2T</td><td>75.86*</td><td>75.37*</td><td>71.90*</td><td>73.27*</td><td>73.74*</td><td>72.58*</td><td>72.90*</td><td>73.47*</td><td>72.06*</td><td>69.74*</td><td>73.29*</td><td>75.81*</td><td>73.33*</td></tr><tr><td>IMO-BART w/o sq</td><td>74.88*</td><td>76.41*</td><td>67.97*</td><td>70.47*</td><td>72.33*</td><td>71.98*</td><td>71.59*</td><td>72.30*</td><td>71.73*</td><td>71.25*</td><td>71.62*</td><td>70.63*</td><td>71.93*</td></tr><tr><td>IMO-BART last</td><td>91.71*</td><td>92.82*</td><td>89.01</td><td>89.41</td><td>93.01*</td><td>89.85*</td><td>89.67</td><td>93.51</td><td>90.10*</td><td>84.69*</td><td>91.22*</td><td>90.95*</td><td>90.49*</td></tr></table>

Table 1: Single-source domain generalization on sentiment analysis datasets. “B2T”: bottom-up layer-wise search. “w/o sq”: simultaneous search. “last”: applying the mask on only the last layer. The metric is accuracy. Asterisk \* shows a significant difference compared to IMG-BART using a t-test with a $p \leq 0 . 0 5$

<table><tr><td colspan="4">AG News</td></tr><tr><td>Models</td><td>Title → Desc</td><td>Desc → Title</td><td>Avg-F1</td></tr><tr><td>BERT</td><td>81.11*</td><td>67.95*</td><td>74.68*</td></tr><tr><td>BART</td><td>80.12*</td><td>71.22*</td><td>75.96*</td></tr><tr><td>BERT-EDA</td><td>80.52*</td><td>72.10*</td><td>76.58*</td></tr><tr><td>BERT-UDA</td><td>80.41*</td><td>71.81*</td><td>75.82*</td></tr><tr><td>BERT-PGB</td><td>78.53*</td><td>73.51*</td><td>76,02*</td></tr><tr><td>PADA</td><td>82.39*</td><td>75.52*</td><td>78.96*</td></tr><tr><td>PDA</td><td>83.61*</td><td>75.96*</td><td>79.79*</td></tr><tr><td>CHATGPT</td><td>85.13</td><td>79.28</td><td>82.21</td></tr><tr><td>ALPACA-7B</td><td>70.61</td><td>70.44</td><td>71.49</td></tr><tr><td>ALPACA-7B-LoRA</td><td>56.17</td><td>49.44</td><td>52.81</td></tr><tr><td>IMO-BART (Our)</td><td>89.40</td><td>81.97</td><td>85.68</td></tr><tr><td>IMO-BART B2T</td><td>70.31*</td><td>64.59*</td><td>67.45*</td></tr><tr><td>IMO-BART w/o sq</td><td>62.59*</td><td>57.27*</td><td>59.93*</td></tr><tr><td>IMO-BART last</td><td>88.22</td><td>80.05*</td><td>84.13*</td></tr></table>

Table 2: Results for multi-class classification datasets. ‘Desc’ represents description. The metric is macro F1.
<table><tr><td colspan="5">SocialDial</td></tr><tr><td>Models</td><td>Loc (Synthetic) → Loc (Human)</td><td>SD (Synthetic) → SD(Human)</td><td>SR (Synthetic) → SR(Human)</td><td>Avg- F1</td></tr><tr><td>BERT-zh</td><td>18.11*</td><td>35.05*</td><td>32.39*</td><td>28.51*</td></tr><tr><td>CHATYUAN</td><td>18.23*</td><td>34.94*</td><td>33.92*</td><td>29.03*</td></tr><tr><td>BERT-EDA</td><td>13.98*</td><td>35.71*</td><td>26.38*</td><td>25.36*</td></tr><tr><td>BERT-UDA</td><td>15.20*</td><td>33.59*</td><td>27.03*</td><td>25.27*</td></tr><tr><td>CHATGPT</td><td>21.44</td><td>38.46</td><td>35.12</td><td>31.67</td></tr><tr><td>CHATGLM-6B</td><td>20.57</td><td>20.53</td><td>11.55</td><td>17.55</td></tr><tr><td>IMO-CY (Our)</td><td>23.22</td><td>46.04</td><td>42.71</td><td>37.32</td></tr><tr><td>IMO-CY B2T</td><td>14.31*</td><td>30.29*</td><td>32.45*</td><td>25.68*</td></tr><tr><td>IMO-CY w/o sq</td><td>13.37*</td><td>29.81*</td><td>29.05*</td><td>24.07*</td></tr><tr><td>IMO-CY last</td><td>21.47*</td><td>44.73</td><td>39.89*</td><td>35.36*</td></tr></table>

Table 3: Evaluation results on SocialDial dataset. CY represents the pre-trained language model CHATYUAN. Loc represents Location; SD represents Social Distance; SR represents Social Relation. The metric is macro F1.

Baseline Models. As Gulrajani and Lopez-Paz (2021) showed, simple empirical risk minimization (ERM) outperforms many SOTA domain generalization algorithms. So we finetune BERT (Devlin et al., 2019) and encoder of BART (Lewis et al., 2020) using cross-entropy loss as baselines. For Chinese text classification, we use BERT-zh (Devlin et al., 2019), BART-zh (Shao et al., 2021) and CHATYUAN (Clue-AI, 2023).

Domain Generalization Models. PADA (Ben-David et al., 2022) is an example-based autoregressive prompt learning algorithm for domain generalization based on the T5 language model (Raffel et al., 2020). PDA (Jia and Zhang, 2022) is a prompt-based learning algorithm for domain generalization.

Large Language Models. As CHATGPT shows promising zero-shot ability on various NLP tasks (OpenAI, 2023), we treat CHATGPT (gpt-3.5- turbo) as a baseline. ALPACA-7B (Taori et al., 2023) is another baseline, which is finetuned from LLaMA 7B (Touvron et al., 2023) on 52K instruction-following data generated by selfinstruct (Wang et al., 2022). ALPACA-7B-LoRA is a finetuned ALPACA-7B model using low-rank adaptation (Wang, 2023; Hu et al., 2022a). CHAT-GLM-6B (THUDM, 2023) is an open large language model based on General Language Model (Du et al., 2022), optimized for Chinese questionanswering and dialogue. All LLMs use few-shot in-context learning. The specific query templates used for the LLMs can be found in Appendix A.3. Data Augmentation. Wiles et al. (2022); Gokhale et al. (2022) find data augmentation benefit domain generalization tasks. EDA (Wei and Zou, 2019) uses four operations (i.e., synonym replacement, random insertion, random swap, and random deletion) to augment text data. UDA (Xie et al., 2020) uses back-translation to generate diverse paraphrases while preserving the semantics of the original sentences. PGB (Shiri et al., 2023) generates syntactically and lexically diversified paraphrases using a fine-tuned BART.

## 4.2 Domain Generalization Results

Binary Classification. Table 1 reports the comparisons between our method and the baselines on sentiment polarity classification. Our method using

<table><tr><td rowspan="2">Models</td><td colspan="4">Yelp→</td><td colspan="4">Amazon→</td></tr><tr><td>Yelp (Source)</td><td>IMDB (Target)</td><td>Amazon (Target)</td><td>TweetEval (Target)</td><td>Amazon (Source)</td><td>IMDB (Target)</td><td>Yelp (Target)</td><td>TweetEval (Target)</td></tr><tr><td>IMO</td><td>95.94</td><td>-5.86</td><td>-1.07</td><td>-4.32</td><td>95.34</td><td>-4.48</td><td>-0.20</td><td>-4.26</td></tr><tr><td>IMO- SC</td><td>89.01*</td><td>-7.81*</td><td>-3.88*</td><td>-11.20*</td><td>90.12*</td><td>-7.64*</td><td>-3.47*</td><td>-12.59*</td></tr></table>

Table 4: Comparison between the proposed model and model using spurious features (SC). In target datasets, we report the reduced percentage of accuracy compared to the source domains.

BART as backbone (i.e., IMO-BART) achieves superior performance over all baselines in 7 of 12 settings, and outperforms the best baseline CHAT-GPT by 2.63% on average. Interestingly, CHAT-GPT stands out as the best model in two out of 12 settings, though it remains unclear whether CHAT-GPT use those datasets for training. Moreover, it is noteworthy that data augmentation methods (i.e., BERT-EDA, BERT-UDA, BERT-PGB) show slightly inferior performance in comparison to the simple fine-tuning of BERT in terms of average accuracy. This suggests that simply back-translating or paraphrasing instances within source domains does not enhance performance on target domains.

Multi-class Classification. As shown in Table 2 and Table 3, our method outperforms all baselines in terms of average macro-F1 by 3.22% and 5.16% on AG News and SocialDial respectively. Among baselines, CHATGPT exhibits the strongest performance on both datasets and surpasses ALPACA-7B, ALPACA-7B-LoRA, and CHATGLM by a large margin. This superior performance shows that current open-source large language models still have a substantial performance gap with CHATGPT when handling difficult tasks.

## 4.3 Analysis of Spurious Features

Presence of Invariant Representations. We inspect shared representations at both feature and token levels. Invariant features are expected to have non-zero values across domains. Taking the best performing model IMO-BART in the sentiment analysis as an example, we train the model in each domain respectively and visualize its masks of the top layer in each domains. As depicted in Fig. 3, there are indeed a set of features shared across domains selected by the masks. We further compute Cosine similarities between the filtering vectors m of the top layer trained on different source domains. As shown in Table 8 in Appendix, their similarities range from 0.68 and 0.85. At the token level, we inspect the shared attention weights visualized in Fig.4 (see Appendix A3), which indicate the keywords shared across domains in sentiment analysis, such as “great” and “slow”.

Impact of Spurious Correlations. To study whether our proposed masking mechanism indeed identifies robust features, we compare the performance of using the selected features with the nonselected ones. Specifically, we run additional experiments by replacing the learned binary masks q with 1  q , followed by freezing all parameters except the classification head and training a model using those non-selected features. The results in Table 4 show that models using the nonselected features have an approximate 6% accuracy reduction in source domains and perform worse than using all features. In target domains, the corresponding performance drop using non-selected features is significantly higher than that using both our method as well as using all features. Hence, our masks indeed mitigate the use of spurious features.

## 4.4 Ablation Study

We compare top-down greedy search with alternative methods: bottom-up layer-wise search (B2T), simultaneous search (w/o sq), and only applying a mask on the last layer (last). From Table 1, 2 and 3, we can tell that top-down greedy performs significantly better than the alternative competitors. We conjecture that top-down layer-wise learning serves a regularization method that reduces the risk of loosing crucial features that are well correlated with Y and the corresponding optimization problem is easier to solve than learning all mask layers simultaneously. Representations from higher layers are shown to be more context-specific than lower layer representations (Ethayarajh, 2019). In contrast, the bottom up approach may drop key features in lower layers that significantly contribute to important higher layer features.

We compare variants of IMO by using varying backbone models and removing the corresponding components. For backbones, we compare BART with T5 and BERT, denoted them as IMO-T5, and IMO-BERT. To study the contribution of each component in our approach, we conduct experiments where we exclude the mask layers, attention mechanisms, or both. These models are denoted by w/o m, w/o a, and w/o am, respectively. The corresponding results are reported in Table 5, Table 11 and Table 10 in Appendix. For comparison between backbones, we find that encoderdecoder neural architectures (i.e., BART, T5) consistently achieve better performance than encoderonly models (i.e., BERT). Compared with variants that remove both the attention module and mask layers, IMO with the attention module or mask module has a significant performance improvement in terms of accuracy or F1 on average, which justifies the usefulness of both modules.

<table><tr><td rowspan="2">Models</td><td colspan="3">IMDB→</td><td colspan="3">Amazon→</td><td colspan="3">Yelp→</td><td colspan="3">TweetEval→</td></tr><tr><td>Amazon</td><td>Yelp</td><td>TweetEval</td><td>IMDB</td><td>Yelp</td><td>TweetEval</td><td>IMDB</td><td>Amazon</td><td>TweetEval</td><td>IMDB Yelp</td><td>Amazon</td><td>Avg.</td></tr><tr><td>BART w/o IMO</td><td>89.94*</td><td>89.13*</td><td>69.59*</td><td>88.19*</td><td>92.20* 82.69*</td><td>86.85*</td><td>90.64*</td><td>85.83*</td><td>78.98*</td><td>89.25*</td><td>87.58*</td><td>85.91*</td></tr><tr><td>IMO-BART (Our)</td><td>93.97</td><td>94.63</td><td>89.58</td><td>90.86</td><td>95.14 91.08</td><td>90.08</td><td>94.87</td><td>91.62</td><td>85.39</td><td>92.84</td><td>91.66</td><td>91.81</td></tr><tr><td>IMO-BART w/o m</td><td>92.15*</td><td>92.49*</td><td>85.61*</td><td>89.48*</td><td>92.97* 88.53*</td><td>88.28</td><td>92.75</td><td>87.44</td><td>80.10</td><td>89.57</td><td>88.09*</td><td>88.95*</td></tr><tr><td>IMO-BART w/o a</td><td>91.35*</td><td>91.04*</td><td>84.18*</td><td>88.51*</td><td>92.49* 84.97*</td><td>87.10*</td><td>91.87*</td><td>88.01*</td><td>83.31*</td><td>90.61*</td><td>88.87*</td><td>88.52*</td></tr><tr><td>IMO-BART STE</td><td>91.11*</td><td>91.71*</td><td>88.05*</td><td>88.29* 91.69*</td><td>87.09*</td><td>88.91*</td><td>91.39*</td><td>89.12*</td><td>82.48*</td><td>89.37*</td><td>88.50*</td><td>88.97*</td></tr><tr><td>IMO-BART STR</td><td>89.79*</td><td>88.97*</td><td>72.98*</td><td>86.26*</td><td>87.48* 79.48*</td><td>86.40*</td><td>88.31*</td><td>77.49*</td><td>81.43*</td><td>85.13*</td><td>82.49*</td><td>83.85*</td></tr><tr><td>IMO-BART Scalar</td><td>87.31*</td><td>89.92*</td><td>87.34*</td><td>87.73*</td><td>86.03* 83.41*</td><td>87.11*</td><td>86.43*</td><td>85.94*</td><td>81.44*</td><td>84.75*</td><td>85.41*</td><td>86.06*</td></tr><tr><td>T5 w/o IMO</td><td>87.53*</td><td>87.09*</td><td>66.37*</td><td>86.47*</td><td>89.38* 80.68*</td><td>84.94*</td><td>88.86*</td><td>83.78*</td><td>76.48*</td><td>86.89*</td><td>85.53*</td><td>83.67*</td></tr><tr><td>IMO-T5</td><td>93.45</td><td>93.88</td><td>84.92*</td><td>89.23* 93.38*</td><td>89.73*</td><td>88.27*</td><td>93.02*</td><td>91.01</td><td>81.39*</td><td>91.93</td><td>89.97*</td><td>90.01*</td></tr><tr><td>BERT w/o IMO</td><td>86.48*</td><td>86.19*</td><td>66.28*</td><td>86.12* 88.91*</td><td>81.45*</td><td>86.34*</td><td>88.34*</td><td>83.46*</td><td>77.25*</td><td>87.34*</td><td>84.82*</td><td>83.58*</td></tr><tr><td>IMO-BERT</td><td>92.09*</td><td>91.93*</td><td>85.34*</td><td>88.53* 92.19*</td><td>88.17*</td><td>87.46*</td><td>91.49*</td><td>89.55*</td><td>79.93*</td><td>89.23*</td><td>87.73*</td><td>88.64*</td></tr></table>

Table 5: Ablation study on sentiment analysis datasets.

![](images/97d74ce28a132fc708c173f0179b10bfbf8bc6f698d73e1b7b04fc870436e0a3.jpg)  
Figure 3: Visualization of filtering and mask vectors in IMO-BART. The top figure visualizes the filtering vectors m, while the bottom one visualizes the mask vectors q. The x-axis signifies the dimensionality of mask layers, whereas the y-axis denotes values attributed to each dimension.

<table><tr><td></td><td colspan="4">Amazon→</td></tr><tr><td>Models</td><td>Yelp</td><td>IMDB</td><td>TweetEval</td><td>Avg.</td></tr><tr><td>IMO-1k</td><td>92.21</td><td>87.29</td><td>85.18</td><td>88.22</td></tr><tr><td>IMO-10k</td><td>94.82</td><td>89.11</td><td>88.43</td><td>90.78</td></tr><tr><td>IMO-100k</td><td>94.90</td><td>90.24</td><td>89.01</td><td>91.38</td></tr><tr><td>IMO-1M</td><td>94.95</td><td>90.29</td><td>89.20</td><td>91.48</td></tr><tr><td>IMO-3.6M</td><td>95.14</td><td>90.86</td><td>91.08</td><td>92.36</td></tr><tr><td>IMO- w/o am -1k</td><td>70.62</td><td>68.61</td><td>66.07</td><td>68.43</td></tr><tr><td>IMO- w/o am -10k</td><td>84.88</td><td>79.02</td><td>75.19</td><td>79.70</td></tr><tr><td>IMO- w/o am -100k</td><td>87.05</td><td>84.95</td><td>80.48</td><td>84.16</td></tr><tr><td>IMO- w/o am -1M</td><td>91.38</td><td>87.06</td><td>81.59</td><td>86.68</td></tr><tr><td>IMO- w/o am -3.6M</td><td>92.20</td><td>88.19</td><td>82.69</td><td>87.69</td></tr></table>

Table 6: Domain generalization experiment with different training sizes in the source domain.

Additionally, we compare IMO with various sparsity methods to implement mask layers, including STR (Kusupati et al., 2020), STE (Bengio et al., 2013; Liu et al., 2020), and Scalar, which uses a learnable single scalar instead of the threshold vector s. All those alternative methods lead to a significant drop, as seen in Table 5.

To explore the influence of source domain training data size on performance within target domains, we train models based on BART with and without our method on the Amazon review dataset with varying sizes of training data (i.e., 1k, 10k, 100k, 1M, and 3.6M). The results in Table 6 show that our method depends significantly less on training data size, though more training data can improve the performance overall. Notably, 1k training data yields a remarkable decline for the models without using IMO, while the corresponding performance reduction is significantly less by using our method.

## 5 Conclusion

This paper presents a novel method, coined IMO, which is a greedy layer-wise representation learning method aiming to improve single-source domain generalization on pre-trained deep encoders for text classification tasks. The key idea is to retain invariant features through trainable mask layers and incorporate a token-level attention module to focus on the tokens that directly lead to the prediction of labels. Through extensive experiments, we demonstrate that IMO achieves superior OOD performance over competitive baselines on multiple datasets. The visualization of masks and attention weights empirically justifies the effectiveness of identified invariant sparse representations.

## Limitations

Our work focuses on the text classification task, intending to investigate how to learn invariant features to improve out-of-domain generalization. However, the proposed method has promising potential for domain generalization in various NLP tasks, such as question answering and text generation tasks. Future work may consider more tasks beyond text classification.

It is worth noting that IMO needs to be trained in a large source domain. The size of the source domain should ideally exceed 10,000 samples to achieve consistently good performance. However, this requirement may pose challenges in lowresource learning scenarios.

## Ethics Statement

This research is dedicated to augmenting the reliability and safety of text classification models, particularly in the context of domain shifts, as highlighted by Ribeiro et al. (2020). By focusing on the learning of invariant features across diverse domains, our approach aims to provide tangible benefits to applications that serve a wide array of user groups. From a user-centric perspective, the implementation of our methodology is expected to bolster the trustworthiness and diminish potential biases in language models.

It is pertinent to note that our study does not involve human subjects, nor does it contravene any legal or ethical standards. We foresee no detrimental impacts arising from our research endeavors. The experimental work underpinning this study was exclusively conducted using datasets that are publicly accessible. Our overarching goal is to foster enhanced academic and societal consciousness regarding the challenges of domain generalization in the field of natural language processing.

## Acknowledgement

This material is based on research sponsored by DARPA under agreement number

HR001122C0029. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright notation thereon.

## References

Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz. 2020. Invariant risk minimization.

Saeid Asgari, Aliasghar Khani, Fereshte Khani, Ali Gholami, Linh Tran, Ali Mahdavi-Amiri, and Ghassan Hamarneh. 2022. Masktune: Mitigating spurious correlations by forcing to explore. In Advances in Neural Information Processing Systems.

Yejin Bang, Samuel Cahyawijaya, Nayeon Lee, Wenliang Dai, Dan Su, Bryan Wilie, Holy Lovenia, Ziwei Ji, Tiezheng Yu, Willy Chung, Quyet V. Do, Yan Xu, and Pascale Fung. 2023. A multitask, multilingual, multimodal evaluation of chatgpt on reasoning, hallucination, and interactivity.

Francesco Barbieri, Jose Camacho-Collados, Luis Espinosa Anke, and Leonardo Neves. 2020. TweetEval: Unified benchmark and comparative evaluation for tweet classification. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 1644–1650, Online. Association for Computational Linguistics.

Eyal Ben-David, Nadav Oved, and Roi Reichart. 2022. PADA: Example-based Prompt Learning for on-thefly Adaptation to Unseen Domains. Transactions of the Association for Computational Linguistics, 10:414–433.

Yoshua Bengio, Nicholas Léonard, and Aaron Courville. 2013. Estimating or propagating gradients through stochastic neurons for conditional computation.

Peter Bühlmann. 2018. Invariance, causality and robustness.

Prithvijit Chattopadhyay, Yogesh Balaji, and Judy Hoffman. 2020. Learning to balance specificity and invariance for in and out of domain generalization. In Computer Vision – ECCV 2020, pages 301–318, Cham. Springer International Publishing.

Clue-AI. 2023. Chatyuan. https://github.com/ clue-ai/ChatYuan.

Gianna M. Del Corso, Antonio Gullí, and Francesco Romani. 2005. Ranking a stream of news. In Proceedings of the 14th International Conference on World Wide Web, WWW ’05, page 97–106, New York, NY, USA. Association for Computing Machinery.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for

Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Yu Ding, Lei Wang, Bin Liang, Shuming Liang, Yang Wang, and Fang Chen. 2022. Domain generalization by learning and removing domain-specific features. In Advances in Neural Information Processing Systems.

Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. 2022. GLM: General language model pretraining with autoregressive blank infilling. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 320–335, Dublin, Ireland. Association for Computational Linguistics.

Kawin Ethayarajh. 2019. How contextual are contextualized word representations? comparing the geometry of bert, elmo, and gpt-2 embeddings. arXiv preprint arXiv:1909.00512.

Tao Feng, Lizhen Qu, and Gholamreza Haffari. 2023. Less is More: Mitigate Spurious Correlations for Open-Domain Dialogue Response Generation Models by Causal Discovery. Transactions of the Associationfor Computational Linguistics, 11:511–530.

Tejas Gokhale, Swaroop Mishra, Man Luo, Bhavdeep Sachdeva, and Chitta Baral. 2022. Generalized but not Robust? comparing the effects of data modification methods on out-of-domain generalization and adversarial robustness. In Findings of the Association for Computational Linguistics: ACL 2022, pages 2705–2718, Dublin, Ireland. Association for Computational Linguistics.

A. Gulli. 2005. The anatomy of a news search engine. In Special Interest Tracks and Posters of the 14th International Conference on World Wide Web, WWW ’05, page 880–881, New York, NY, USA. Association for Computing Machinery.

Ishaan Gulrajani and David Lopez-Paz. 2021. In search of lost domain generalization. In International Conference on Learning Representations.

Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022a. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Ziniu Hu, Zhe Zhao, Xinyang Yi, Tiansheng Yao, Lichan Hong, Yizhou Sun, and Ed Chi. 2022b. Improving multi-task generalization via regularizing spurious correlation. In Advances in Neural Information Processing Systems, volume 35, pages 11450– 11466. Curran Associates, Inc.

Pavel Izmailov, Polina Kirichenko, Nate Gruver, and Andrew G Wilson. 2022. On feature learning in the presence of spurious correlations. In Advances in

Neural Information Processing Systems, volume 35, pages 38516–38532. Curran Associates, Inc.

Chen Jia and Yue Zhang. 2022. Prompt-based distribution alignment for domain generalization in text classification. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pages 10147–10157, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Diederik P. Kingma and Jimmy Ba. 2015. Adam: A method for stochastic optimization.

Aditya Kusupati, Vivek Ramanujan, Raghav Somani, Mitchell Wortsman, Prateek Jain, Sham Kakade, and Ali Farhadi. 2020. Soft threshold weight reparameterization for learnable sparsity. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 5544–5555. PMLR.

Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2020. BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 7871–7880, Online. Association for Computational Linguistics.

Dianqi Li, Yizhe Zhang, Zhe Gan, Yu Cheng, Chris Brockett, Bill Dolan, and Ming-Ting Sun. 2019. Domain adaptive text style transfer. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3304–3313, Hong Kong, China. Association for Computational Linguistics.

Haoliang Li, Sinno Jialin Pan, Shiqi Wang, and Alex C. Kot. 2018a. Domain generalization with adversarial feature learning. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5400–5409.

Ya Li, Xinmei Tian, Mingming Gong, Yajing Liu, Tongliang Liu, Kun Zhang, and Dacheng Tao. 2018b. Deep domain generalization via conditional invariant adversarial networks. In Computer Vision – ECCV 2018, pages 647–663, Cham. Springer International Publishing.

Chen Liang, Simiao Zuo, Minshuo Chen, Haoming Jiang, Xiaodong Liu, Pengcheng He, Tuo Zhao, and Weizhu Chen. 2021. Super tickets in pre-trained language models: From model compression to improving generalization. In Annual Meeting of the Association for Computational Linguistics.

Junjie Liu, Zhe XU, Runbin SHI, Ray C. C. Cheung, and Hayden K.H. So. 2020. Dynamic sparse training: Find efficient sparse network from scratch with trainable masked layers. In International Conference on Learning Representations.

Xiaofeng Liu, Chaehwa Yoo, Fangxu Xing, Hyejin Oh, Georges El Fakhri, Je-Won Kang, and Jonghye Woo. 2022. Deep unsupervised domain adaptation: A review of recent advances and perspectives. APSIPA Transactions on Signal and Information Processing, 11(1).

Fangrui Lv, Jian Liang, Shuang Li, Bin Zang, Chi Harold Liu, Ziteng Wang, and Di Liu. 2022. Causality inspired representation learning for domain generalization. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8036–8046, Los Alamitos, CA, USA. IEEE Computer Society.

Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. 2011. Learning word vectors for sentiment analysis. In Proceedings of the 49th Annual Meeting of the Associationfor Computational Linguistics: Human Language Technologies, pages 142–150, Portland, Oregon, USA. Association for Computational Linguistics.

Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2022. Rethinking the role of demonstrations: What makes in-context learning work? In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pages 11048–11064, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Krikamol Muandet, David Balduzzi, and Bernhard Schölkopf. 2013. Domain generalization via invariant feature representation. In Proceedings ofthe 30th International Conference on International Conference on Machine Learning - Volume 28, ICML’13, page I–10–I–18. JMLR.org.

Brady Neal. 2020. Introduction to causal inference. Course Lecture Notes (draft).

OpenAI. 2023. Gpt-4 technical report.

Cheng Ouyang, Chen Chen, Surui Li, Zeju Li, Chen Qin, Wenjia Bai, and Daniel Rueckert. 2023. Causalityinspired single-source domain generalization for medical image segmentation. IEEE Transactions on Medical Imaging, 42(4):1095–1106.

Jonas Peters, Peter Bühlmann, and Nicolai Meinshausen. 2016. Causal inference by using invariant prediction: identification and confidence intervals. Journal ofthe Royal Statistical Society. Series B (Statistical Methodology), 78(5):947–1012.

Jonas Peters, Dominik Janzing, and Bernhard Schlkopf. 2017. Elements of Causal Inference: Foundations and Learning Algorithms. The MIT Press.

Fengchun Qiao, Long Zhao, and Xi Peng. 2020. Learning to learn single domain generalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12556–12565.

Francesco Quinzan, Ashkan Soleymani, Patrick Jaillet, Cristian R Rojas, and Stefan Bauer. 2023. Drcfs: Doubly robust causal feature selection. In International Conference on Machine Learning, pages 28468–28491. PMLR.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21(1).

Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. 2020. Beyond accuracy: Behavioral testing of NLP models with CheckList. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 4902– 4912, Online. Association for Computational Linguistics.

Danielle Saunders. 2022. Domain adaptation and multidomain adaptation for neural machine translation: A survey. Journal ofArtificial Intelligence Research, 75.

Rui Shao, Xiangyuan Lan, Jiawei Li, and Pong C. Yuen. 2019. Multi-adversarial discriminative deep domain generalization for face presentation attack detection. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10015–10023.

Yunfan Shao, Zhichao Geng, Yitao Liu, Junqi Dai, Hang Yan, Fei Yang, Li Zhe, Hujun Bao, and Xipeng Qiu. 2021. Cpt: A pre-trained unbalanced transformer for both chinese language understanding and generation. arXiv preprint arXiv:2109.05729.

Zheyan Shen, Jiashuo Liu, Yue He, Xingxuan Zhang, Renzhe Xu, Han Yu, and Peng Cui. 2021. Towards out-of-distribution generalization: A survey.

Fatemeh Shiri, Terry Yue Zhuo, Zhuang Li, Van Nguyen, Shirui Pan, Weiqing Wang, Reza Haffari, and Yuan-Fang Li. 2023. Paraphrasing techniques for maritime qa system.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Stanford alpaca: An instruction-following llama model. https:// github.com/tatsu-lab/stanford\_alpaca.

THUDM. 2023. Chatglm-6b. https://github.com/ THUDM/ChatGLM-6B.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. Llama: Open and efficient foundation language models.

Victor Veitch, Alexander D’Amour, Steve Yadlowsky, and Jacob Eisenstein. 2021. Counterfactual invariance to spurious correlations: Why and how to pass stress tests. arXiv preprint arXiv:2106.00545.

Riccardo Volpi and Vittorio Murino. 2019. Addressing model vulnerability to distributional shifts over image transformation sets. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 7980–7989.

Eric J. Wang. 2023. Alpaca-lora. https://github. com/tloen/alpaca-lora.

Haohan Wang, Zeyi Huang, and Eric Xing. 2021a. Learning robust models by countering spurious correlations.

Jindong Wang, Cuiling Lan, Chang Liu, Yidong Ouyang, and Tao Qin. 2021b. Generalizing to unseen domains: A survey on domain generalization. pages 4627–4635. Survey Track.

Yixin Wang and Michael Jordan. 2022. Representation learning as finding necessary and sufficient causes. In ICML 2022: Workshop on Spurious Correlations, Invariance and Stability.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2022. Self-instruct: Aligning language model with self generated instructions.

Zengzhi Wang, Qiming Xie, Zixiang Ding, Yi Feng, and Rui Xia. 2023. Is chatgpt a good sentiment analyzer? a preliminary study.

Zijian Wang, Yadan Luo, Ruihong Qiu, Zi Huang, and Mahsa Baktashmotlagh. 2021c. Learning to diversify for single domain generalization. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 834–843.

Jason Wei and Kai Zou. 2019. EDA: Easy data augmentation techniques for boosting performance on text classification tasks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 6382–6388, Hong Kong, China. Association for Computational Linguistics.

Olivia Wiles, Sven Gowal, Florian Stimberg, Sylvestre-Alvise Rebuffi, Ira Ktena, Krishnamurthy Dj Dvijotham, and Ali Taylan Cemgil. 2022. A fine-grained analysis on distribution shift. In International Conference on Learning Representations.

Qizhe Xie, Zihang Dai, Eduard Hovy, Minh-Thang Luong, and Quoc V. Le. 2020. Unsupervised data augmentation for consistency training. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS’20, Red Hook, NY, USA. Curran Associates Inc.

Zhe Xu and Ray C. C. Cheung. 2019. Accurate and compact convolutional neural networks with trained binarization. In 30th British Machine Vision Conference 2019, BMVC 2019, Cardiff, UK, September 9-12, 2019, page 19. BMVA Press.

Jingfeng Yang, Hongye Jin, Ruixiang Tang, Xiaotian Han, Qizhang Feng, Haoming Jiang, Bing Yin, and Xia Hu. 2023. Harnessing the power of llms in practice: A survey on chatgpt and beyond.

Haolan Zhan, Zhuang Li, Xiaoxi Kang, Tao Feng, Yuncheng Hua, Lizhen Qu, Yi Ying, Mei Rianto Chandra, Kelly Rosalin, Jureynolds Jureynolds, et al. 2024. Renovi: A benchmark towards remediating norm violations in socio-cultural conversations. arXiv preprint arXiv:2402.11178.

Haolan Zhan, Zhuang Li, Yufei Wang, Linhao Luo, Tao Feng, Xiaoxi Kang, Yuncheng Hua, Lizhen Qu, Lay-Ki Soon, Suraj Sharma, Ingrid Zukerman, Zhaleh Semnani-Azad, and Gholamreza Haffari. 2023. Socialdial: A benchmark for socially-aware dialogue systems.

Dinghuai Zhang, Kartik Ahuja, Yilun Xu, Yisen Wang, and Aaron Courville. 2021. Can subnetwork structure be the key to out-of-distribution generalization? In International Conference on Machine Learning, pages 12356–12367. PMLR.

Hanlin Zhang, Yi-Fan Zhang, Weiyang Liu, Adrian Weller, Bernhard Schölkopf, and Eric P. Xing. 2022. Towards principled disentanglement for domain generalization. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8014–8024.

Xiang Zhang, Junbo Zhao, and Yann LeCun. 2015a. Character-level convolutional networks for text classification. page 649–657.

Xiang Zhang, Junbo Zhao, and Yann LeCun. 2015b. Character-level convolutional networks for text classification. In Advances in Neural Information Processing Systems, volume 28. Curran Associates, Inc.

Long Zhao, Ting Liu, Xi Peng, and Dimitris Metaxas. 2020. Maximum-entropy adversarial data augmentation for improved generalization and robustness. In Proceedings ofthe 34th International Conference on Neural Information Processing Systems, NIPS’20, Red Hook, NY, USA. Curran Associates Inc.

## A Appendix

## A.1 Background of Causal Representation Learning

Causal representation learning aims to learn and leverage the causal relations within data to enhance model generalization and robustness against distribution shifts in the data generation process (Shen et al., 2021). This approach differs from traditional machine learning methods that predominantly focus on correlational patterns without distinguishing causation and correlation. Causation refers to the underlying mechanisms that connect variables, implying that alterations in a causal variable will consequentially affect the associated effect variable, a process known as an intervention. In contrast, correlation does not necessarily indicate a direct mechanistic link. For instance, a model might infer ’it is raining’ upon observing people with open umbrellas, which recognizes a correlation between these events. However, the act of closing umbrellas does not influence the weather. This example shows the difference between correlation and causation. Predictions relying on correlation may yield erroneous outcomes when the environment (i.e., data distributions) change. For example, if umbrellas are opened due to sunlight rather than rain, a model trained on the correlation might inaccurately predict rain. This indicates the importance of making predictions based on causation rather than correlation. A causal model should predict the weather based on temperature, humidity, air pressure, etc. Prediction based on causation can enhance out-ofdistribution performance, which is supported by the assumption that causal relations remain constant across diverse environments (Bühlmann, 2018).

However, learning the complete causal structure is ambitious and may not be realized in practice. A more feasible approach involves identifying invariant features that reliably predict target variables across varying environments. A series of methods (Muandet et al., 2013; Chattopadhyay et al., 2020; Asgari et al., 2022; Izmailov et al., 2022; Hu et al., 2022b) have been proposed by leveraging the invariance between environments. They leverage the fact that when conditioning all direct causes of a target variable, the conditional distribution of the target will not change when interventions are applied to all other variables in the model except the target itself. Building upon this foundational idea, our work seeks to identify and utilize these direct causes (i.e., invariant features across environments) for the accurate prediction of target variables in the out-of-distribution setting.

## A.2 Experiment Datasets

AG News (Gulli, 2005; Del Corso et al., 2005; Zhang et al., 2015b) is a collection of news articles used for topic classification, which contains news titles, and news descriptions assigned to four topic classes. Titles and descriptions are employed as different domains. For social factor prediction, we use SocialDial (Zhan et al., 2023), which is a Chinese socially-aware dialogue corpus consisting of synthetic conversations generated by CHATGPT and human-written conversations. Both are annotated with social factors such as location, social distance, and social relation. Synthetic conversations and human-written conversations are considered as different domains. The statistics of datasets are listed in Table 7.

<table><tr><td colspan="2">Binary Classification</td><td colspan="3"></td></tr><tr><td>Dataset</td><td>Domain</td><td>#Train</td><td>#Dev</td><td>#Test</td></tr><tr><td>Amazon</td><td>Review of products</td><td>3.6M</td><td>0</td><td>40k</td></tr><tr><td>IMDB</td><td>Review of movies</td><td>25k</td><td>0</td><td>25k</td></tr><tr><td>Yelp</td><td>Review of businesses</td><td>560k</td><td>0</td><td>38k</td></tr><tr><td>TweetEval</td><td>Tweet</td><td>25k</td><td>1k</td><td>6k</td></tr><tr><td>Yahoo</td><td>Questions from Yahoo! Answers</td><td>4k</td><td>2k</td><td>1k</td></tr></table>

<table><tr><td colspan="2">Multi-class Classification</td><td colspan="3"></td></tr><tr><td>Dataset</td><td>Domain</td><td>#Train</td><td>#Dev</td><td>#Test</td></tr><tr><td>AG News</td><td>Title of news articles</td><td>120k</td><td>0</td><td>7k</td></tr><tr><td>AG News</td><td>Description of news articles</td><td>120k</td><td>0</td><td>7k</td></tr><tr><td>SocialDial</td><td>Synthetic conversations by CHATGPT</td><td>68k</td><td>7k</td><td>7k</td></tr><tr><td>SocialDial</td><td>Human-written conversations</td><td>0</td><td>0</td><td>5k</td></tr></table>

Table 7: Statistics of datasets.

## A.3 Training details

We use the encoder of BART (Lewis et al., 2020) as the default pre-trained language model. All models are trained up to 100 epochs with a minibatch size of 32 in the source domain. We use the Adam (Kingma and Ba, 2015) optimizer with hyperparameters tuned on the validation sets. As a result, we run Adam with $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } = 0 . 9 9 9$ . The learning rate is $5 \times 1 0 ^ { - 5 }$ . We use a linear learning rate scheduler that dynamically decreases the learning rate after a warm-up period. All experiments are conducted on NVIDIA A40 GPU.

The process of model selection in domain generalization is inherently a learning problem. In this approach, we employ training-domain validation, which is one of the three selection methods introduced by Gulrajani and Lopez-Paz (2021). We divide each training domain into separate training and validation sets. Models are trained on the training set, and the model that achieves the highest accuracy on the validation set is chosen as the selected model.

When using large language models to predict target classification labels, the query template for sentiment analysis is: “There are some examples about sentiment analysis: {examples}. Given text: {sentence}, what is the sentiment conveyed? Please select the answer from ‘positive’ or ‘negative’.”. The query template for AG News topic classification is "There are some examples for topic classification: {examples}. Given text: {sentence}, what is the topic of this text? Please select the answer from ‘Business’, ‘Sci/Tech’, ‘World’ or ‘Sports’.” The query templates for SocialDial are “There are some examples for classification: {examples}. Given conversation: {conversation}, what’s the location/social distance/social relation of this conversation? Please select the answer from {labels}”<sup>3</sup> (Min et al., 2022; Wang et al., 2023; Yang et al., 2023).

## A.4 Visual Explanation

To intuitively show how the attention module and mask module work in models, we visualize attention weights on tokens and mask vectors in Figure 4 and 3, respectively. We also demonstrate cosine similarities between mask vectors m trained on different source domains and Jaccard similarities between binary vectors q trained on different source domains on Table 8 and Table 9, respectively.

From Figure 4, we can find that our model primarily focuses its attention on sentiment-indicative tokens. Notably, positive reviews exhibit high attention weights for tokens like ‘good,’ ‘great,’ and ‘nice,’ indicating their significance. Conversely, negative reviews assign high attention weights to tokens such as ‘horrible’ and ‘slow,’ highlighting their importance in expressing negativity.

In Figure 3, we visualize mask vectors m and binary vectors q trained on different source domains across dimensions. It can be observed that certain dimensions are consistently assigned zero (or nonzero) values across different training domains, indicating our mask layers can capture some features that are irrelevant (or invariant) across domains. We quantify invariant features across domains by computing vector similarity. We calculate cosine similarities between different mask vectors m. The results are shown in Table 8. We can find that most mask vector pairs have over 0.75 similarity, except the Yelp-TweetEval pair, which is probably because of a larger divergence between Yelp and TweetEval domains. Table 9 shows Jaccard similarities between binary vectors q. Most binary vector pairs have similarities of over 0.5, except the Yelp-TweetEval pair, with a similarity of 0.45.

## A.5 Additional Experimental Results

![](images/a176b9884080f27332fb6913641d484199bf8a8131686b70cb08ea954830fe67.jpg)

![](images/23c9c9b73d657b87d62f4134f7c87a79b3780e4faadce5e70ae94b1c7dfffc97.jpg)

(a) The sentiment label is positive.  
![](images/86f42fdd71e78096640ef16b51ec16abc54b5abf984c5c28dc3e4c6c7835cf3b.jpg)  
(c) The sentiment label is negative.

(b) The sentiment label is positive.  
![](images/40f84df1842294415442ae8f7beff6dc5826ab8d669814a4ea0a2a6232007279.jpg)  
(d) The sentiment label is negative.

Figure 4: Visualization of attention weights on tokens in Yelp dataset reviews.
<table><tr><td></td><td>Yelp</td><td>Amazon</td><td>IMDB</td><td>TweetEval</td></tr><tr><td>Yelp</td><td>1.0</td><td>0.7930</td><td>0.7533</td><td>0.6838</td></tr><tr><td>Amazon</td><td>=</td><td>1.0</td><td>0.8458</td><td>0.7687</td></tr><tr><td>IMDB</td><td></td><td></td><td>1.0</td><td>0.8069</td></tr><tr><td>TweetEval</td><td>=</td><td>I</td><td></td><td>1.0</td></tr></table>

Table 8: Cosine similarities between mask vectors m trained on different source domains.
<table><tr><td></td><td>Yelp</td><td>Amazon</td><td>IMDB</td><td>TweetEval</td></tr><tr><td>Yelp</td><td>1.0</td><td>0.5869</td><td>0.5231</td><td>0.4504</td></tr><tr><td>Amazon</td><td>=</td><td>1.0</td><td>0.6513</td><td>0.5614</td></tr><tr><td>IMDB</td><td>=</td><td></td><td>1.0</td><td>0.6139</td></tr><tr><td>TweetEval</td><td></td><td></td><td></td><td>1.0</td></tr></table>

Table 9: Jaccard similarities between binary vectors q trained on different source domains.
<table><tr><td colspan="5">SocialDial</td></tr><tr><td>Models</td><td>Loc (Synthetic) → Loc (Human)</td><td>SD (Synthetic) → SD(Human)</td><td>SR (Synthetic) → SR(Human)</td><td>Avg- F1</td></tr><tr><td>CHATYUAN w/o IMO</td><td>19.12*</td><td>37.75*</td><td>34.07*</td><td>30.31*</td></tr><tr><td>IMO-CY</td><td>23.22</td><td>46.04</td><td>42.71</td><td>37.32</td></tr><tr><td>IMO-CY w/o m</td><td>22.47*</td><td>41.86*</td><td>38.95*</td><td>34.43*</td></tr><tr><td>IMO-CY w/o a</td><td>21.05*</td><td>39.88*</td><td>37.28*</td><td>32.73*</td></tr><tr><td>IMO-CY w/o Ldist</td><td>20.17*</td><td>39.26*</td><td>39.41*</td><td>32.95*</td></tr><tr><td>BART-zh w/o IMO</td><td>15.86*</td><td>35.92*</td><td>31.04*</td><td>27.61*</td></tr><tr><td>IMO-BART-zh</td><td>19.94*</td><td>41.39*</td><td>39.27*</td><td>33.53*</td></tr><tr><td>BERT-zh w/o IMO</td><td>10.34*</td><td>30.17*</td><td>19.87*</td><td>20.12*</td></tr><tr><td>IMO-BERT-zh</td><td>14.68*</td><td>36.75*</td><td>27.41*</td><td>26.28*</td></tr></table>

Table 10: Ablation study on SocialDial datasets. CY represents the pre-trained language model CHATYUAN.

<table><tr><td colspan="4">AG News</td></tr><tr><td>Models</td><td>Title → Desc</td><td>Desc → Title</td><td>Avg-F1</td></tr><tr><td>BART w/o IMO</td><td>80.91*</td><td>73.89*</td><td>77.40*</td></tr><tr><td>IMO-BART</td><td>89.40</td><td>81.97</td><td>85.68</td></tr><tr><td>IMO-BART w/o m</td><td>83.29*</td><td>77.08*</td><td>80.19*</td></tr><tr><td>IMO-BART w/o a</td><td>82.72*</td><td>77.27*</td><td>79.99*</td></tr><tr><td>IMO-BART w/o  $\mathcal { L } _ { d i s t }$ </td><td>87.79*</td><td>79.82*</td><td>83.81*</td></tr><tr><td>T5 w/o IMO</td><td>78.48*</td><td>71.26*</td><td>74.87*</td></tr><tr><td>IMO-T5</td><td>86.91*</td><td>79.75*</td><td>83.33*</td></tr><tr><td>BERT w/o IMO</td><td>75.12*</td><td>61.47*</td><td>68.29*</td></tr><tr><td>IMO-BERT</td><td>84.79*</td><td>75.38*</td><td>80.09*</td></tr></table>

Table 11: Ablation study on AG News dataset. ’Binary refers to the application of the proposed binary classification method on multi-label classification tasks.