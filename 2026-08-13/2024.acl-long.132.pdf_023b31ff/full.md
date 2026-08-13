# Modality-Aware Integration with Large Language Models for Knowledge-based Visual Question Answering

Junnan Dong†<sup>1</sup>, Qinggang Zhang†<sup>1</sup>, Huachi Zhou<sup>1</sup>, Daochen Zha<sup>2</sup>, Pai Zheng<sup>1</sup>, and Xiao Huang\*<sup>1</sup>

<sup>1</sup>The Hong Kong Polytechnic University <sup>2</sup>Rice University

{hanson.dong, qinggangg.zhang, huachi.zhou}@connect.polyu.hk daochen.zha@rice.edu, pai.zheng@polyu.edu.hk, xiaohuang@comp.polyu.edu.hk

## Abstract

Knowledge-based visual question answering (KVQA) has been extensively studied to answer visual questions with external knowledge, e.g., knowledge graphs (KGs). While several attempts have been proposed to leverage large language models (LLMs) as an implicit knowledge source, it remains challenging since LLMs may generate hallucinations. Moreover, multiple knowledge sources, e.g., images, KGs and LLMs, cannot be readily aligned for complex scenarios. To tackle these, we present a novel modality-aware integration with LLMs for KVQA (MAIL). It carefully leverages multimodal knowledge for both image understanding and knowledge reasoning. Specifically, (i) we propose a two-stage prompting strategy with LLMs to densely embody the image into a scene graph with detailed visual features; (ii) We construct a coupled concept graph by linking the mentioned entities with external facts. (iii) A tailored pseudo-siamese graph medium fusion is designed for sufficient multimodal fusion. We utilize the shared mentioned entities in two graphs as mediums to bridge a tight intermodal exchange, while maximally preserving insightful intra-modal learning by constraining the fusion within mediums. Extensive experiments show the superiority of MAIL.

## 1 Introduction

Knowledge-based visual question answering (KVQA) aims to provide appropriate answers for questions about images based on external knowledge (Wang et al., 2017), such as knowledge graphs (KGs) (Marino et al., 2019; Zhang et al., 2022). It has various applications, especially for assisting the visually impaired users (Gurari et al., 2018), yet still, remains a challenging task that requires complex reasoning capability across different data modalities (Yu et al., 2020, 2017).

![](images/595ce73b5c875494b43ba655371907d5c1379efeca837f0974d80690a746d0c4.jpg)  
Figure 1: A sketched comparison on employing LLMs for KVQA between existing learning paradigms and ours.

Recently, several studies have explored using large language models (LLMs) as supplementary knowledge bases and reasoning tools for KVQA (Yang et al., 2022; Gui et al., 2022; Lin et al., 2022); according to how they fuse the knowledge, they can be broadly categorized into direct prompting and modality-agnostic approaches, shown in Figure 1 (a) and (b), respectively. The former directly prompts the question and the corresponding image caption to LLMs for answers (Yang et al., 2022). The latter leverages LLMs to generate candidate answers with supporting evidence and simply combines both question and the external knowledge embedding, e.g., Wikidata (Shengyuan et al., 2024), for reasoning at the final stage (Gui et al., 2022; Lin et al., 2022).

While the above methods have employed LLMs in various ways for KVQA, we argue that they have not fully leveraged the knowledge from LLMs and lack the cross-modal reasoning ability, potentially resulting in sub-optimal performance for complex VQA scenarios. (i) LLMs could incorrectly answer questions or provide unreliable evidence for reasoning. On the one hand, direct prompting to LLMs may struggle to identify the right answer for many complex or domain-specific questions, due to the lack of domain knowledge (Amaro et al., 2023; Chen et al., 2024). On the other hand, LLMs may be prone to generating hallucination (Gravel et al., 2023) and producing misleading evidence in support of candidate answers. (ii) Integrating multimodal knowledge in a modality-agnostic manner can be sub-optimal. Specifically, existing methods simply concatenate different modal representations, e.g., questions, captions, tags, and external knowledge, for reasoning. This design lacks the necessary cross-modal exchange to enrich the semantics of entities, limiting the final reasoning performance. For example, to correctly answer the question in Figure 1, the model is required to infer the season based on a cross-modal understanding of the inputs, such as the “keep warm” purpose of “coat” and the “spring blooming” feature of “sakura”.

In this work, we study the following research question: How can we effectively leverage the knowledgefrom LLMs to enhance the comprehensive understanding and reasoning of the images and questions in KVQA? Answering this question is nontrivial due to the following challenges. (i) It is hard to properly incorporate the knowledge from LLMs. LLMs may generate hallucinations when dealing with requests that are not covered in their training corpus. Simply prompting them may generate noisy and irrelevant responses. (ii) Semantic alignment of multiple knowledge sources is challenging. Given image captions, object/region features, external knowledge from KGs, and implicit knowledge from LLMs, appropriately aligning relevant semantic information in different modalities cannot be readily achieved.

To tackle these challenges, we present a novel modality-aware framework to effectively integrate LLMs for KVQA in Figure 1 (c), dubbed MAIL. Specifically, (i) we propose a two-stage prompting strategy to maximally leverage the knowledge from LLMs for image understanding. We initialize a dense caption by prompting a visual LLM, e.g., Visual ChatGPT (Wu et al., 2023) and MiniGPT-4 (Zhu et al., 2023). To depict the detailed visual scenes in the caption, we construct a scene graph by defining twelve condensed relations and prompting the LLM to extract spatial and object features accordingly in the form of triples, e.g., (sakura, at\_location, tree). (ii) We integrate the external knowledge from KGs to form a coupled concept graph, where the mentioned entities in scene graphs are linked with real-world assertions and facts to facilitate knowledgeable reasoning, such as (coat, used\_for, keep warm) and (sakura, typle\_of, spring blooming). (iii) A tailored pseudo-siamese graph medium fusion is designed for effective multimodal graph fusion. Inspired by the success of pseudo-siamese network in measuring the similarity of two correlative inputs (Xia et al., 2021; Gupta et al., 2023), we extend it to graphs to process intra-modal information. It consists of two graph attention networks with the same architecture but different weights. In each sub-encoder, we concentrate on one modality and design a tailored context-aware propagation. This guides our model to attentively prioritize the most valuable entities subject to the particular question. Then we leverage the shared mentioned entities in both coupled graphs as mediums to bridge the cross-modal interaction. The model continuously exchanges their embeddings between two modalities, bringing sufficient complementary knowledge to the other modality respectively. It merely allows inter-modal exchanging by constraining it within the mediums. In general, MAIL effectively enhances a tight intermodal fusion while maximally preserving the insightful intra-modal information for each modality.

Our major contributions are summarized below:

• We formally define a novel learning paradigm, modality-aware integration with LLMs for knowledge-based visual question answering.

• The implicit knowledge in LLMs is carefully leveraged via an effective prompting strategy for coupled scene/concept graph construction.

• We further propose a tailored pseudo-siamese graph medium fusion to integrate multimodal knowledge sources. It balances both intra-modal processing and inter-modal exchange.

• Extensive experiments are conducted on two benchmark datasets. MAIL achieves superior performance over a variety of state-of-the-art baselines with 2 4 faster inferential time.

## 2 Problem Statement

KVQA requires the model to provide answers to the question of the corresponding image based on external knowledge . In this paper, we propose a novel learning paradigm for leveraging LLMs f( ) for comprehensive knowledge-based VQA. Given an image , a relevant question and external knowledge , we aim to integrate a visual LLM f( ) and fuse f( ), ,  for prediction. The overall performance is evaluated by the accuracy of returned answers with the ground truths.

![](images/f7618223c1e557ab2a0aed0dc144b9cc9cf57267a45b2ce28192bbedea115f9d.jpg)  
Figure 2: Our proposed framework MAIL, a novel modality-aware integration for knowledge-based VQA with LLMs. Nodes in blue stand for external knowledge, while red is for visual objects and yellow shows the topic entities from questions. Blue nodes with red dashed borders indicate the extracted mediums in concept graph. MAIL is trained to integrate multimodal information for comprehensive cross-modal reasoning with a tailored PS-GMF.

## 3 Methodology

In this section, we introduce the detailed rationale of our proposed framework. An illustration of MAIL is shown in Figure 2. We first carefully leverage the knowledge from LLMs for coupled graph construction. Then, we formulate the pseudo-siamese graph medium fusion (PS-GMF). Through an effective integration of two tailored training objectives, we jointly optimize the model for accurate prediction.

## 3.1 Scene Graph Construction

Dense Caption Generation We carefully design a hard prompt that requires a visual LLM f( ) to depict the detailed appearance of all the objects in the image and the spatial relations between them. We obtain the generated caption through

$$
{ \mathcal { D } } = f ( { \mathcal { I } } , P r o m p t ) .\tag{1}
$$

We consider the identified visual entities in the image as key mentioned entities appearing in the caption, denoted as $\mathcal { M } = [ m _ { 1 } , m _ { 2 } . . . m _ { n } ] \in \mathcal { D }$ . They significantly dominate the multimodal information of both visual features and external knowledge required to answer the questions.

## Prompt-enhanced Triple Extraction

Given the extracted mentioned entities, we employ LLMs to extract triples. To fully leverage LLMs comprehension of image captions and prioritize the important visual features, we pre-define 12 relations $\mathcal { R } = [ r _ { 1 } , r _ { 2 } , . . . r _ { 1 2 } ]$ from two aspects: (i) Spatial features. We constrain the description with at\_location, next\_to, in\_front\_of, surrounded\_by, covered\_by, includes and holds. (ii) Object features are preserved with not only visual outlooks, i.e., has\_property, has\_color, made\_of and wears, but also the intentions of the object if he/she is a human, i.e., intends\_to. We design a hard template to prompt LLMs for scene graph construction as,

$$
\mathcal { G } ^ { S } = f ( P r o m p t , \mathcal { D } , \mathcal { M } , \mathcal { R } ) .\tag{2}
$$

The prompt template is used as follows: ‘Describe the image with as many details as possible. Generally, identify the objects and their spatial relations with each other. Specifically, include the visual outlook of different objects, e.g., color, style as well as the appearance of human beings.’ We show the detailed statistics and distributions of all twelve condensed relations in both benchmarks OK-VQA (Marino et al., 2019) and FVQA (Wang et al., 2017) in Table 2 below.

## 3.2 Concept Graph Construction

In parallel, we incorporate ConceptNet (Speer et al., 2018) for external commonsense knowledge to construct a concept graph. It is one of the largest KGs that provides a myriad of structured triples and contains more than eight million real-world entities. We link each mentioned entity m and the topic entity in the question with ConceptNet, and denote the constructed graph as $\mathcal { G } ^ { C }$ with sufficient textual descriptions, attributes, categories, and properties of , that are not present in the image so as to facilitate a more knowledgeable reasoning background for various questions. We design the prompt as ‘Given the image caption, based on your comprehensive understanding, construct a high-quality scene graph with as many meaningful details of the mentioned entities as possible in theform ofa triple (head entity, relation, tail entity). n Strictly use the twelve predefined relations from: , e.g., (woman, in\_front\_of, car), (car, has\_color, blue), only return the triples with no other content. n Caption: n Mentioned Entities: .’

## 3.3 Pseudo-siamese Graph Medium Fusion

Typical pseudo-siamese networks (PSNs) could effectively measure the similarity between two inputs (Gupta et al., 2023; Xia et al., 2021). We extend it to graphs, which naturally fit the requirement of learning coupled graphs for intra-modal processing, leading to pseudo-siamese graph neural networks (PSGs). However, PSG is incapable of cross-modal fusion. Particularly equipped for PSG to enable inter-modal learning, we further design a graph mediumfusion (GMF) algorithm.

<table><tr><td>PSG Architecture</td><td>Formulated Definition</td></tr><tr><td>Context-aware Attention</td><td> $\Phi \left( m _ { t } \vert \vert \mathbf { c } \right)$ </td></tr><tr><td>Aggregation Function</td><td> $\begin{array} { r } { \sum _ { t \in \mathcal { N } _ { h } } \alpha _ { m _ { t } } \times m _ { t } } \end{array}$ </td></tr><tr><td>Combination Function</td><td> $J ( e _ { \mathcal { N } _ { h } } ^ { \prime } ) + e _ { h } ^ { \ell }$   $\int e _ { h } ,$ </td></tr><tr><td>Activation Function</td><td> $i f e _ { h } \ge 0 ,$   $( 1 e - 2 ) \times { \pmb e } _ { h } ,$  otherwise.</td></tr></table>

Table 1: Formulated definitions of the shared architectures for two sub-networks in the proposed Pseudo-Siamese Graph Neural Network.

## Pseudo-siamese Graph Neural Network

Locating valuable entities in different modalities is essential for KVQA. Here, we instantiate PSG with a novel context-aware message propagation scheme to prioritize the most important knowledge in each modality subject to the question context.

Definition. [Pseudo-siamese GNN] We refer to a pseudo-siamese graph neural network that consists of two identical graph neural networks for two relevant inputs. They share the same architecture, i.e., attention mechanism, aggregation function, combinationfunction and activationfunction, but different weights.

As two sub-networks in PSG share the same architecture, we uniformly provide formulations for the intra-modality processing. For each head entity h, we aggregate all the messages from its neighbor tail entities $\mathcal { N } _ { h }$ and $t \in \mathcal { N } _ { h }$ . Since relations in multimodal graphs contain indispensable information for reasoning various real-world questions, we establish the message passing at the triple level (Zhang et al., 2023a), i.e., $( h , r , t )$ to capture abundant semantics as follows.

$$
\begin{array} { r } { \pmb { m } _ { t \in \mathcal { N } _ { h } } = \pmb { W } ( \pmb { e } _ { h } , \pmb { e } _ { r } , \pmb { e } _ { t } ) , } \end{array}\tag{3}
$$

where $( e _ { h } , e _ { r } , e _ { t } )$ is the triple embedding associated with $( h , r , t )$ , and W is a learnable matrix for linear transformation. We initialize the entity and relation embedding with a pre-trained language model RoBERTa-large (Liu et al., 2019). While multimodal graphs always contain desperate information with each other, uniformly training each subnetwork in PSG based on the final prediction lacks awareness of the multimodal characteristics. To this end, we design tailored graph attention networks (Velickovi ˇ c et al. ´ , 2017; Dong et al., 2023a) that allocate a context-aware weight aˆ to each message, only prioritizing the multimodal messages in both coupled graphs that are highly related to the question. The context-aware weight $\hat { a } _ { m _ { t } }$ for each message ${ \mathbf { } } m _ { t }$ is correspondingly computed as:

<table><tr><td rowspan=2 colspan=1>Categories</td><td rowspan=2 colspan=1>Relation</td><td rowspan=1 colspan=4>OK-VQA     FVQA</td></tr><tr><td rowspan=1 colspan=1>Tain</td><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>Tain</td><td rowspan=1 colspan=1>Test</td></tr><tr><td rowspan=6 colspan=1>SpatialFeatures</td><td rowspan=6 colspan=1>at_locationnext_toin_front_ofsurrounded_bycovered_byincludesholds</td><td rowspan=1 colspan=1>10,562</td><td rowspan=1 colspan=1>10,118</td><td rowspan=1 colspan=1>3,466</td><td rowspan=1 colspan=1>3,107</td></tr><tr><td rowspan=1 colspan=1>3,948</td><td rowspan=1 colspan=1>3,772</td><td rowspan=1 colspan=1>2,533</td><td rowspan=1 colspan=1>2,289</td></tr><tr><td rowspan=1 colspan=1>2,239</td><td rowspan=1 colspan=1>2,244</td><td rowspan=1 colspan=1>759</td><td rowspan=1 colspan=1>687</td></tr><tr><td rowspan=1 colspan=1>2,004</td><td rowspan=1 colspan=1>2,026</td><td rowspan=1 colspan=1>699</td><td rowspan=1 colspan=1>549</td></tr><tr><td rowspan=1 colspan=1>180</td><td rowspan=1 colspan=1>191</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>7</td></tr><tr><td rowspan=1 colspan=1>12,4023,344</td><td rowspan=1 colspan=1>12,3903,090</td><td rowspan=1 colspan=1>1,811965</td><td rowspan=1 colspan=1>1,630794</td></tr><tr><td rowspan=5 colspan=1>ObjectFeatures</td><td rowspan=5 colspan=1>has_propertyhas_colormade_ofwearsintends_to</td><td rowspan=1 colspan=1>16,685</td><td rowspan=1 colspan=1>17,032</td><td rowspan=1 colspan=1>1,301</td><td rowspan=1 colspan=1>1,297</td></tr><tr><td rowspan=1 colspan=1>9,191</td><td rowspan=1 colspan=1>8,836</td><td rowspan=1 colspan=1>3,653</td><td rowspan=1 colspan=1>3,258</td></tr><tr><td rowspan=1 colspan=1>3,388</td><td rowspan=1 colspan=1>3,310</td><td rowspan=1 colspan=1>978</td><td rowspan=1 colspan=1>948</td></tr><tr><td rowspan=1 colspan=1>5,172</td><td rowspan=1 colspan=1>5,049</td><td rowspan=1 colspan=1>1,504</td><td rowspan=2 colspan=1>1,4498</td></tr><tr><td rowspan=1 colspan=1>1,599</td><td rowspan=1 colspan=1>1,655</td><td rowspan=1 colspan=1>9</td></tr></table>

Table 2: The overall statistics of the pre-defined condense relations for OK-VQA and FVQA datasets. They depict the spatial features and object features in images.

$$
\hat { a } _ { ( h , r , t ) } = \Phi \left( m _ { t } \vert \vert \mathbf { c } \right) ,\tag{4}
$$

where Φ is the adopted activation function, i.e., LeakyReLU. We endow the attention mechanism to be context-aware by concatenating the question context embedding c, expressed as . Notably, we fix the question context embedding c with RoBERTa and only allow it to participate during the attention allocation process.

By normalizing the attention scores obtained previously, we further assign normative values α to each message $\mathbf { \nabla } m _ { t }$ of $( h , r , t )$

$$
\alpha _ { m _ { t } } = \frac { \hat { a } _ { ( h , r , t ) } } { \sum _ { ( h , r ^ { \prime } , t ^ { \prime } ) \in \mathcal { N } _ { h } } \hat { a } _ { ( h , r ^ { \prime } , t ^ { \prime } ) } } .\tag{5}
$$

To this end, with a weighted sum aggregation operator, we are able to acquire the aggregated representation for entity h in the current layer from its neighbors as $\begin{array} { r } { e _ { \mathcal { N } _ { h } } ^ { \ell } = \sum _ { ( h , r , t ) \in \mathcal { N } _ { h } } \alpha _ { ( h , r , t ) ) } \times m _ { t } ^ { \ell } , } \end{array}$ where the layer number in PSG is denoted as ℓ. We summarize the major functions in Table 1. We finalize the overall architecture of PSG for both inputs from scene graph $\mathcal { G } ^ { S }$ and concept graph $\mathcal { G } ^ { C }$

$$
\begin{array} { r l } & { e _ { h } ^ { S ( \ell + 1 ) } = J ( \displaystyle \sum _ { ( h , r , t ) \in \mathcal { N } _ { h } } \alpha _ { ( h , r , t ) ) } \times m _ { t } ) + e _ { h } ^ { S ( \ell ) } , } \\ & { e _ { \hat { h } } ^ { C ( \ell + 1 ) } = J ( \displaystyle \sum _ { ( \hat { h } , \hat { r } , \hat { t } ) \in \mathcal { N } _ { \hat { h } } } \alpha _ { ( \hat { h } , \hat { r } , \hat { t } ) } \times m _ { \hat { t } } ) + e _ { \hat { h } } ^ { C ( \ell ) } , } \\ & { ~ \quad \quad \quad \quad ( \hat { h } , \hat { r } , \hat { t } ) \in \mathcal { N } _ { \hat { h } } } \end{array}\tag{6}
$$

where J is a multi-layer perception. The model effectively combines the learned neighbor information $e _ { \mathcal { N } _ { h } } ^ { \ell }$ and itself $e _ { h } ^ { \ell }$ in current layer. We obtain final representations of all the entities when the layer number ℓ reaches the pre-defined target.

## Graph Medium Fusion

In this subsection, we aim to fill the gaps of the aforementioned PSG on inter-modal learning. However, there is a challenging dilemma centered around striking the right balance between two crucial aspects. On one hand, we want to maximize the inter-modal fusion, where multimodal information could collaborate to yield a more insightful and nuanced understanding of the underlying knowledge subject to answering the question. On the other hand, we recognize the necessity of preserving the integrity of intra-modal processing. Considering excessive inter-modal fusion could introduce noise from each other, we aim to maintain the distinctive characteristics and valuable insights that each modality inherently holds.

Since the mentioned entities $[ m _ { 1 } , m _ { 2 } . . . m _ { n } ]$ are shared by $\mathcal { G } ^ { S }$ and $\mathcal { G } ^ { C }$ we consider these entities existing in both coupled graphs as mediums that possess similar embeddings since they represent the same real-world object though appearing in different modalities. Motivated by this, we design a novel graph medium fusion algorithm that leverages the medium to bridge two modalities. To get rid of the dilemma, we (i) exchange the representations of mediums $e _ { m }$ within their respective graphs. This allows the model to delicately introduce cross-modal information with their neighbor entities in the respective graphs, i.e., $e _ { \mathcal { N } _ { m } } ;$ (ii) We strictly impose restrictions on the cross-modal exchange to be within the mediums. This gently brings two modalities closer to each other, while maximally maintaining their individualities. The formulated graph medium fusion process between the coupled graphs is written below.

$$
e _ { m } ^ { S } = \left\{ \begin{array} { l l } { e _ { m } ^ { S } , } & { i f \ell = 0 , } \\ { e _ { m } ^ { C } , } & { o t h e r w i s e . } \end{array} \right. \quad e _ { m } ^ { C } = \left\{ \begin{array} { l l } { e _ { m } ^ { C } , } & { i f \ell = 0 , } \\ { e _ { m } ^ { S } , } & { o t h e r w i s e . } \end{array} \right.\tag{7}
$$

Specifically, we froze the medium embeddings in the first layer to ensure they have initially aggregated important 1-hop neighbor information. Afterward, the embeddings for the same medium are automatically exchanged after message-passing in the current layer. This sequential approach ensures a high-quality exchange of information between modalities, i.e., visual features and external knowledge, while initially preserving the local context within each modality before they engage in crossmodal interactions during the following layers.

## 3.4 Training Objective

## Answer-targeted Inferential Loss

The primary target of our model is to accurately predict the final answer subject to the particular image and question context. We adopt the binary cross-entropy loss to optimize the inferential performance:

$$
{ \mathcal { L } } _ { I n f e r e n c e } = - l o g { \frac { M L P ( e _ { a } + \mathbf { c } ) } { \sum _ { a ^ { \prime } \in { \mathcal { G } } ^ { C } } M L P ( e _ { a ^ { \prime } } + \mathbf { c } ) } } ,\tag{8}
$$

where $a$ is the correct answer and $a ^ { \prime }$ is one of all the candidate answers from $\mathcal { G } ^ { C }$ . We employ $M L P ( e _ { a } + { \pmb { c } } )$ to compute the probability of all the candidate entities in $\mathcal { G } ^ { C }$ and prioritize the highest one as the final answer.

## Maximum Mean Discrepancy loss

Based on the assumption that one medium in two modalities should be similar to the maximum extent, we approximate their similarity by adopting an auxiliary loss, i.e., Maximum Mean Discrepancy (MMD) loss. The basic kernel function is formulated as follows:

$$
\mathcal { K } ( \boldsymbol { e } _ { m } ^ { S } , \boldsymbol { e } _ { m } ^ { C } ) = \exp \left( - \frac { | | \boldsymbol { e } _ { m } ^ { S } - \boldsymbol { e } _ { m } ^ { C } | | ^ { 2 } } { 2 \sigma ^ { 2 } } \right) ,\tag{9}
$$

where $\kappa$ represents the kernel function and $\sigma$ is a hyperparameter controlling the width of the kernel (Steinwart and Scovel, 2012). Given a valid kernel function where ${ \mathcal K } ( e _ { m } ^ { S } , e _ { m } ^ { C } ) = ( \phi ( e _ { m } ^ { S } ) -$ $\phi ( e _ { m } ^ { C } ) )$ , we denote the corresponding feature mapping function as $\phi .$ The final MMD loss for crossmodal alignment is demonstrated hereunder,

$$
\mathcal { L } _ { M e d i u m } = | | \frac { 1 } { n } \sum _ { m \in \mathcal { M } } \phi ( \pmb { e } _ { m } ^ { S } ) - \frac { 1 } { n } \sum _ { m \in \mathcal { M } } \phi ( \pmb { e } _ { m } ^ { C } ) | | ^ { 2 } .\tag{10}
$$

We aim to minimize this loss to encourage the learned representations for the same medium from two modalities to be similar in the shared PSG architecture. This effectively guides the process of graph medium fusion by constraining the similarity of mediums in different modalities with each other.

## 3.4.1 Joint Optimization

The overall framework is jointly optimized according to the aforementioned training objectives. Despite the effectiveness of $\mathcal { L } _ { M e d i u m }$ , it may introduce inevitable noise by irrespectively forcing the mediums from two modalities to be exactly aligned, which ignores the nature of different modalities. To alleviate this problem, we introduce a hyperparameter λ to control the contribution from $\mathcal { L } _ { M e d i u m }$ . To this end, the final training loss is calculated below:

$$
\begin{array} { r } { \mathcal { L } _ { J o i n t } = \mathcal { L } _ { I n f e r e n c e } + \lambda \mathcal { L } _ { M e d i u m } . } \end{array}\tag{11}
$$

## 4 Experiments

In this section, we conduct a variety of experiments to demonstrate the effectiveness of our proposed MAIL. We aim to answer four research questions:

• RQ1 (Main Results): How does MAIL perform compared with different types of SOTA models?

• RQ2 (Hyperparameter analysis): How do hyperparameters influence the performance?

• RQ3 (Ablation studies): Does each component eventually contribute to the overall performance?

• RQ4 (Case study): How effectively does MAIL work in real-world VQA tasks?

## 4.1 Experimental Setup

Implementation Details We generate dense image captions with MiniGPT-4 (7B) (Zhu et al., 2023), as well as InstructBLIP (Dai et al., 2024) for generalization, and adopt ConceptNet (Speer et al., 2018) for external knowledge. We apply MiniGPT-4 and InstructBLIP with one Tesla V100. The entire processing of OK-VQA and the corresponding Microsoft COCO images (Lin et al., 2014) including image-to-text and data cleaning takes about 4 rounds. We adopt ℓ = 3 and λ = 1e 3 after hyperparameter tuning. The generated caption is stored for further multimodal learning. Our codes and processed graphs will be open-sourced and publicly available. For the results of baseline LLMs, since they could occasionally refuse to answer with responses like either ‘As a language model, I am not capable of understanding images’ or ‘Sorry, there is no related information in the provided caption.’, we report the average accuracy over 2 rounds.

## Datasets

Following the previous work (Marino et al., 2021; Yang et al., 2022; Gui et al., 2022; Wu et al., 2022; Lin et al., 2022), we mainly conduct our experiments on OK-VQA (Marino et al., 2019), which is currently the largest and most challenging benchmark, consisting of 14,055 image-question pairs. To further demonstrate the generalization, we also experimentalize on FVQA dataset (Wang et al., 2017), which was the first exploration of KVQA. Baselines

We adopt two pipelines of off-the-shelf methods for performance comparison. (i) Traditional endto-end baselines that design various multimodal learning algorithms for final reasoning over the posed questions, i.e., a direct answering based on questions only (Q Only) (Marino et al., 2019), BAN (Kim et al., 2018), MUTAN (Ben-Younes et al., 2017), ConceptBERT (Gardères et al., 2020), KRISP (Marino et al., 2021), MAVEx (Wu et al., 2022), VLC-BERT (Ravi et al., 2023), RA-VQA (Lin and Byrne, 2022), TriG (Gao et al., 2022), HCNMN (Zhang et al., 2023b) and MCAN (Yu et al., 2019). Moreover, as BAN and MUTAN merely learn the uni-modal visual features, they are augmented with ArticleNet (AN) (Marino et al., 2019) that is trained to retrieve knowledge from Wikipedia for corresponding question-image pair to facilitate the reasoning with external knowledge, denoted as ‘BAN + AN’ and ‘MUTAN + AN’ (Marino et al., 2019). (ii) LLM-enhanced baselines that leverage LLMs, i.e., GPT-3, for direct answer prediction or relevant supporting evidence generation, i.e., PICa (Yang et al., 2022), KAT (Gui et al., 2022), TwO (Si et al., 2023), a simple baseline for KBVQA (Xenos et al., 2023) and REVIVE (Lin et al., 2022).

## 4.2 Main Results

To answer RQ1, in Table 3 & 4, we summarize the comparisons with all the important baselines. The performance is evaluated by the soft accuracy following previous research (Hu et al., 2023). MAIL outperforms all the traditional baselines regardless of their various knowledge sources and the advantages of leveraging a feature-level image representation. MAIL achieves 12.04% improvements over the best traditional baseline, i.e., MCAN, on OK-VQA and 14.7% on FVQA. For LLM-enhanced baselines, it is worth mentioning that they have utilized the generative ability from (Lin et al., 2022; Brown et al., 2020), which makes them especially advantageous in answering subjective questions, for instance, ‘Can people travel on the freeway’ or ‘Is it illegal?’. Despite this, MAIL still outperforms the best LLM-enhanced baseline with 2.28% increases in general, let alone 13.39% over PICa.

Moreover, MAIL is resource-efficient, requiring the smallest number of parameters among all the LLM-enhanced baselines, shown in Table 7. We have used significantly far fewer parameters than any other LLM-enhanced models, i.e., 7.13 B, for answer prediction. As a result, the inferential time of MAIL for one test question is 0.661s (when batch

<table><tr><td>Method</td><td>Model Inputs</td><td>External Knowledge</td><td>Fusion Strategy</td><td>Acc. (%)</td></tr><tr><td>Q Only</td><td>Question + Image</td><td></td><td></td><td>14.93</td></tr><tr><td colspan="5">Traditional End-to-end Baselines</td></tr><tr><td>BAN BAN +AN MUTAN</td><td>Question + Image Question + Image Question + Image</td><td>Wikipedia</td><td>Modality-agnostic</td><td>25.17 25.61</td></tr><tr><td>MUTAN +AN ConceptBERT</td><td>Question + Image Question + Image</td><td>Wikipedia ConceptNet</td><td>Modality-agnostic Modality-agnostic</td><td>26.41 27.84 33.66</td></tr><tr><td>HCNMN Krisp</td><td>Question + Image Question + Image</td><td>WordNet Wikipedia + ConceptNet</td><td>Modality-agnostic Modality-agnostic</td><td>36.74 38.90</td></tr><tr><td>MAVÊx VLC-BERT</td><td>Question + Image</td><td>Wikipedia + onceptNet + Ġoogle Images</td><td>Modality-agnostic</td><td>41.37</td></tr><tr><td></td><td>Question + Image</td><td>COMET + ConceptNet</td><td>Modality-agnostic</td><td>43.14</td></tr><tr><td></td><td></td><td></td><td></td><td>44.65</td></tr><tr><td>MCAN</td><td> $\mathrm { Q u e s t i o n + I m a g e }$ </td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>Large Language Model-enhanced Baselines</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PICa-Base</td><td> $\mathrm { Q u e s t i o n + C a p t i o n + O b j e c t T a g s }$ </td><td>Frozen GPT-3 (175B)</td><td></td><td>43.30</td></tr><tr><td>Pica-Full</td><td></td><td></td><td></td><td>48.00</td></tr><tr><td></td><td> $\mathrm { Q u e s t i o n + C a p t i o n + O b j e c t T a g s }$ </td><td>Frozen GPT-3 (175B)</td><td></td><td></td></tr><tr><td>KAT (Single)</td><td></td><td>Frozen GPT-3 (175B) + Wikidata</td><td>|Modality-agnostic</td><td>53.09</td></tr><tr><td>KAT (Ensemble)</td><td>Question + Caption + Object Tags</td><td></td><td>Modality-agnostic</td><td>54.41</td></tr><tr><td></td><td> $\mathrm { Q u e s t i o n + C a p t i o n + O b j e c t T a g s }$ </td><td>Frozen GPT-3 (175B) + Wikidata</td><td></td><td></td></tr><tr><td>REVIVE</td><td> $\mathrm { \ Q u e s t i o n + C a p t i o n + R e g i o n \ T a g s }$ </td><td>Frozen GPT-3 (175B) + Wikidata</td><td>Modality-agnostic</td><td>53.83</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MAIL (ours)</td><td> $\mathrm { Q u e s t i o n + I m a g e }$ </td><td>Frozen MiniGPT-4 (7B)* + ConceptNet</td><td>Modality-aware</td><td>56.69</td></tr></table>

Table 3: The overall performance comparison on benchmark dataset OK-VQA. We also elaborate on the detailed comparison with a variety of baselines on the knowledge sources that support their inference, i.e., model inputs, external knowledge, as well as how they fuse multiple modalities.

∗ We merely leverage it for caption and scene graph construction, with no extra information that is not present in the images.
<table><tr><td>Method</td><td>Fusion Strategy</td><td>Acc. (%)</td></tr><tr><td>XNM</td><td>Modality-agnostic</td><td>63.74</td></tr><tr><td>KI-Net</td><td>Modality-agnostic</td><td>63.78</td></tr><tr><td>UnifER</td><td>Modality-agnostic</td><td>66.83</td></tr><tr><td>MCAN</td><td></td><td>64.47</td></tr><tr><td>HCNMN</td><td>Modality-agnostic</td><td>69.43</td></tr><tr><td>MAIL (ours)</td><td>Modality-aware</td><td>73.95</td></tr></table>

Table 4: Performance comparison on FVQA.
<table><tr><td>ACC.%</td><td> $\ell = 2$ </td><td> $\ell = 3$ </td><td> $\ell = 4$ </td><td> $\ell = 5$ </td><td> $\ell = 6$ </td></tr><tr><td>MAIL</td><td>56.41</td><td>56.69</td><td>55.45</td><td>54.11</td><td>52.80</td></tr></table>

Table 5: Evaluation on the influences of graph layers in pseudo-siamese graph medium fusion.  
size = 1). Generally, existing LLM-enhanced baselines commonly require 2 4 times of inferential time than MAIL.

<table><tr><td>ACC.%</td><td> λ = 0 1e − 5 1e − 4 1e − 3 1e − 2 1e − 1</td></tr><tr><td>MAIL</td><td>54.30 55.82</td></tr><tr><td></td><td>54.18 55.31 56.69</td></tr><tr><td>53.34</td><td></td></tr></table>

## 4.3 Hyperparameter Analysis

## Search of Graph layers

The main architecture of PS-GMF naturally comprises the discussion of the impacts from graph layers ℓ. We empirically hypothesize that augmenting the depth of the ℓ could facilitate both a deeper understanding of single modalities (i.e., PSG) and a more profound exchange of information between modalities (i.e., GMF). However, it remains unclear about when to reach the plateau. Simply adding more layers may over-fuse two modalities and lose the ability of intra-modal processing, while reducing layers may lead to an adverse situation with inadequate inter-modal fusion. To this end, we vary the layer number and show the performance changes in Table 5. The final prediction performance of MAIL is reported when ℓ = 3. Investigation on hyperparameter λ

<table><tr><td>Models</td><td></td><td>~Param Size Training Time1</td><td>Inference Time</td></tr><tr><td>PICa</td><td>~175.00 B</td><td>1</td><td>1.547 s</td></tr><tr><td>KAT</td><td>~175.80 B</td><td>3.025 s</td><td>1.292 s</td></tr><tr><td>REVIVE</td><td>~175.80 B</td><td>4.500 s</td><td>2.644 s</td></tr><tr><td>MAIL(Ours)</td><td>~7.13 B</td><td>2.699 s</td><td>0.661 s</td></tr></table>

Table 6: Exploring the control over the impacts from $\mathcal { L } _ { M e d i u m }$ to preserve insightful intra-modal learning.

Table 7: Comparisons on the computational costs and inferential time with LLM-enhanced baselines.
<table><tr><td>Reasoning Module</td><td>Accuracy (%)</td></tr><tr><td>PSG (w/o GMF)</td><td>55.53</td></tr><tr><td>PS-GMF</td><td>56.69</td></tr></table>

Table 8: Verification of the importance of inter-modality fusion by removing GMF with PSG only.

![](images/af23735ca4118880ed3137c4f8ce748f6b826931b942cbe0e9d0616f2df50d0c.jpg)  
Figure 3: Case studies with both single-hop and multi-hop reasoning examples in OK-VQA.

<table><tr><td>Pure LLMs</td><td>Multimodal Understanding</td><td>Acc.(%)</td></tr><tr><td colspan="3">Large Language Models</td></tr><tr><td>Llama (7B) Llama2 (7B) ChatGPT (GPT3.5) GPT-4</td><td>Dense Caption Dense Caption Dense Caption Dense Caption</td><td>39.27 45.35 40.26 54.33</td></tr><tr><td colspan="3">Visual Large Language Models</td></tr><tr><td colspan="3"></td></tr><tr><td>Visual ChatGPT MiniGPT-4 (7B)</td><td>BLIP-VQA-Base + GPT3.5 ViT + Vicuna</td><td>38.70 51.26</td></tr><tr><td>Ours</td><td>Dense Caption + PS-GMF</td><td>56.69</td></tr></table>

Table 9: Ablation studies on comparing with pure LLMs by directly feeding the questions and (i) corresponding image caption to LLMs or (ii) the raw images to visual LLMs for answers in a zero-shot setting.

While an excessively strict alignment of mediums may homogenize the intra-modal information, we aim to find a suitable λ that constrains the impacts of $\mathcal { L } _ { M e d i u m }$ . This could significantly encourage harmonious inter-modal fusion from multiple modalities while retaining the richness and specificity inherent to each modality. The experimentation process involves a systematic adjustment of λ across a range of values, specifically within the interval [0, 1e  5, 1e  4, 1e  3, 1e  2, 1e  1]. We showcase the results in Table 6. Upon careful examination of the performance trends, we employ λ = 1e 3 for a balanced trade-off.

## 4.4 Ablation Studies

## Empirical comparison with LLMs

In this ablation study, we further demonstrate our tailored multimodal learning module PS-GMF, and delineate the specific contributions by comparing it against frozen LLMs. Specifically, we adopt both pure LLMs, i.e., Llama (Touvron et al., 2023a) and Llama2 (Touvron et al., 2023b), as well as visual LLMs with Visual ChatGPT (Wu et al., 2023) and MiniGPT-4 (Zhu et al., 2023). We exclusively constrain the inputs in a zero-shot setting with only dense captions and questions for LLMs, while raw images and questions for frozen visual LLMs. The results are summarized in Table 9. MAIL outperforms the best LLM GPT-4 with 2.36% improvements, attributed to the effective graph medium fusion that integrates external knowledge. MAIL also significantly outperforms Visual ChatGPT and MiniGPT-4 with 17.99% and 5.43% higher accuracy. The results shed light on the cross-modal reasoning ability of MAIL.

## Reasoning with PSG Only

In this subsection, we explore the importance of inter-modality interaction by removing the graph medium fusion and only relying on PSG for inference. We list the performance of ‘PSG w/o (GMF)’ in Table 8. The complete multimodal reasoning with PS-GMF outperforms the version with only intra-modal learning with 1.16% improvements. Under this PSG-only setting, we seek to grasp insights into the necessity of graph medium fusion for fostering effective inter-modality interaction. Understanding the performance impact of omitting this fusion mechanism supports the value of shared entities and medium exchange in bridging the cross-modal interaction and facilitates our proposed modality-aware integration with LLMs.

## Ablation with InstructBLIP

To investigate the generalization ability of our proposed MAIL, we adopt InstructBLIP (Dai et al., 2024) as an alternative LLM to take the place of

MiniGPT-4. We reconstruct the scene graph and concept graph based on the newly generated image captions. The main results on OK-VQA are reported in Appendix Table 10.

## 4.5 Case Studies

In this section, we answer RQ4 with six real-world examples from OK-VQA in Figure 3 to shed light on our effectiveness. Single-hop questions can be directly inferred with easily accessible information from either the visual content or external knowledge sources, while multi-hop questions pose more challenges for accurately locating answers several hops away from mentioned entities.

These cases show the adeptness of MAIL in handling a spectrum of questions, requiring both straightforward inferences from explicit information and complex multi-hop reasoning ability by integrating implicit knowledge sources. For example, Figure 3 (a) can be answered based on the visual information captured by the scene graph without external knowledge, while the answer of (e) needs to be artfully inferred from two different angles, i.e., both the blossom season of sakura and the warmth of people’s clothes. These can be attributed to (i) the coupled graph construction that contains abundant modality-aware knowledge to ground the reasoning, as well as (ii) the effective design of our pseudo-siamese graph neural network. It benefits sufficient preservation of intra-modal information and adequate cross-modal fusion, resulting in a powerful multi-hop reasoning ability over both inherent visual features and external knowledge.

## 5 Related Work

KVQA with KGs. Early studies either dedicated to integrating different knowledge sources (Wang et al., 2017; Dong et al., 2023b) or proposed various fusion algorithms for multimodal information (Marino et al., 2021). ConceptBERT (Gardères et al., 2020) constrains the multimodal information with question embedding and fuses embeddings of each modality for prediction. MAVEx (Wu et al., 2022) aims to discern the corresponding knowledge source for each candidate answer to reduce noise. KRISP (Marino et al., 2021) captures both implicit information in both questions, images and KGs.

KVQA with LLMs. Recently, large language models (LLMs) have surprised the community with their superior understanding of texts (Guo et al., 2023). PICa (Yang et al., 2022) first leverages GPT3 (Brown et al., 2020) as an implicit knowledge source for reasoning by prompting the image captions and in-context examples. Another pipeline of studies employs LLMs to generate candidates or supporting evidence for particular captions, e.g., KAT (Gui et al., 2022) and REVIVE (Lin et al., 2022). While they do not fully leverage the multiple sources of knowledge, we break the limitation of complex reasoning by developing a tailored multimodal fusion algorithm that balances intra- and inter-modal learning.

## 6 Conclusions

We present MAIL, a modality-aware integration with large language models for knowledge-based visual question answering. We formally define a novel multimodal learning paradigm for comprehensive cross-modal reasoning among multiple knowledge sources. The knowledge from LLMs is effectively leveraged via a carefully designed prompting for coupled graph construction, i.e., scene graph and concept graph. Then we integrate various multimodal information with a tailored pseudo-siamese graph medium fusion. It effectively enhances a tight inter-modal interaction and maximally preserves insightful intra-modal processing. MAIL achieves superiority on two benchmark datasets while requiring 2 4 faster inferential time than the existing state-of-the-art baselines.

## Limitations

Since the real-world knowledge graphs only contain concrete real-world entities, our KG-based VQA is naturally limited to being able to answer subjective questions, e.g., YES/NO and abstract opinions. We will include this valuable direction in future work to equip our model with generative components to handle the subjective questions.

## Ethics Statement

We confirm that we have fully complied with the ACL Ethics Policy in this study. We conduct experiments with widely adopted publicly available datasets. The generated image captions, processed scene graphs and concept graphs will be opensourced for other researchers’ fair reproduction and further study in the active KVQA community.

## Acknowledgements

The work described in this paper was fully supported by a grant from the Research Grants Council of the Hong Kong Special Administrative Region, China (Project No. PolyU 25208322).

## References

Ilaria Amaro, Attilio Della Greca, Rita Francese, Genoveffa Tortora, and Cesare Tucci. 2023. Ai unreliable answers: A case study on chatgpt. In ICHCI, pages 23–40. Springer.

Hedi Ben-Younes, Rémi Cadene, Matthieu Cord, and Nicolas Thome. 2017. Mutan: Multimodal tucker fusion for visual question answering. In ICCV, pages 2612–2620.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. NeurIPS, 33:1877–1901.

Shengyuan Chen, Qinggang Zhang, Junnan Dong, Wen Hua, Qing Li, and Xiao Huang. 2024. Entity alignment with noisy annotations from large language models.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. 2024. Instructblip: Towards general-purpose visionlanguage models with instruction tuning. Advances in Neural Information Processing Systems, 36.

Junnan Dong, Qinggang Zhang, Xiao Huang, Keyu Duan, Qiaoyu Tan, and Zhimeng Jiang. 2023a. Hierarchy-aware multi-hop question answering over knowledge graphs. In WWW, pages 2519–2527.

Junnan Dong, Qinggang Zhang, Xiao Huang, Qiaoyu Tan, Daochen Zha, and Zhao Zihao. 2023b. Active ensemble learning for knowledge graph error detection. In WSDM, pages 877–885.

Feng Gao, Qing Ping, Govind Thattai, Aishwarya Reganti, Ying Nian Wu, and Prem Natarajan. 2022. Transform-retrieve-generate: Natural languagecentric outside-knowledge visual question answering. In CVPR, pages 5067–5077.

François Gardères, Maryam Ziaeefard, Baptiste Abeloos, and Freddy Lecue. 2020. Conceptbert: Concept-aware representation for visual question answering. In EMNLP, pages 489–498.

Jocelyn Gravel, Madeleine D’Amours-Gravel, and Esli Osmanlliu. 2023. Learning to fake it: limited responses and fabricated references provided by chatgpt for medical questions. Mayo Clinic Proceedings: Digital Health, 1(3):226–234.

Liangke Gui, Borui Wang, Qiuyuan Huang, Alexander G Hauptmann, Yonatan Bisk, and Jianfeng Gao. 2022. Kat: A knowledge augmented transformer for vision-and-language. In NAACL, pages 956–968.

Tao Guo, Song Guo, and Junxiao Wang. 2023. Pfedprompt: Learning personalized prompt for visionlanguage models in federated learning. In Proceedings of the ACM Web Conference 2023, pages 1364– 1374.

Yangyang Guo, Liqiang Nie, Yongkang Wong, Yibing Liu, Zhiyong Cheng, and Mohan Kankanhalli. 2022. A unified end-to-end retriever-reader framework for knowledge-based vqa. In ICMM, pages 2061–2069.

Agrim Gupta, Jiajun Wu, Jia Deng, and Li Fei-Fei. 2023. Siamese masked autoencoders. CVPR.

Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. 2018. Vizwiz grand challenge: Answering visual questions from blind people. In CVPR, pages 3608–3617.

Yushi Hu, Hang Hua, Zhengyuan Yang, Weijia Shi, Noah A Smith, and Jiebo Luo. 2023. Promptcap: Prompt-guided task-aware image captioning. ICCV.

Jin-Hwa Kim, Jaehyun Jun, and Byoung-Tak Zhang. 2018. Bilinear attention networks. NeurIPS, 31.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft coco: Common objects in context. In ECCV, pages 740– 755. Springer.

Weizhe Lin and Bill Byrne. 2022. Retrieval augmented visual question answering with outside knowledge. EMNLP.

Yuanze Lin, Yujia Xie, Dongdong Chen, Yichong Xu, Chenguang Zhu, and Lu Yuan. 2022. Revive: Regional visual representation matters in knowledgebased visual question answering. NeuIPS, 35:10560– 10571.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.

Kenneth Marino, Xinlei Chen, Devi Parikh, Abhinav Gupta, and Marcus Rohrbach. 2021. Krisp: Integrating implicit and symbolic knowledge for opendomain knowledge-based vqa. In CVPR, pages 14111–14121.

Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. 2019. Ok-vqa: A visual question answering benchmark requiring external knowledge. In CVPR, pages 3195–3204.

Sahithya Ravi, Aditya Chinchure, Leonid Sigal, Renjie Liao, and Vered Shwartz. 2023. Vlc-bert: visual question answering with contextualized commonsense knowledge. In WACV, pages 1155–1165.

Chen Shengyuan, Yunfeng Cai, Huang Fang, Xiao Huang, and Mingming Sun. 2024. Differentiable neuro-symbolic reasoning on large-scale knowledge graphs. Advances in Neural Information Processing Systems, 36.

Jiaxin Shi, Hanwang Zhang, and Juanzi Li. 2019. Explainable and explicit visual reasoning over scene graphs. In CVPR, pages 8376–8384.

Qingyi Si, Yuchen Mo, Zheng Lin, Huishan Ji, and Weiping Wang. 2023. Combo of thinking and observing for outside-knowledge vqa. ACL.

Robyn Speer, Joshua Chin, and Catherine Havasi. 2018. Conceptnet 5.5: An open multilingual graph of general knowledge.

Ingo Steinwart and Clint Scovel. 2012. Mercer’s theorem on general domains: On the interaction between measures, kernels, and rkhss. Constructive Approximation, 35:363–417.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023a. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023b. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Petar Velickoviˇ c, Guillem Cucurull, Arantxa Casanova,´ Adriana Romero, Pietro Lio, and Yoshua Bengio. 2017. Graph attention networks. arXiv preprint arXiv:1710.10903.

Peng Wang, Qi Wu, Chunhua Shen, Anthony Dick, and Anton Van Den Hengel. 2017. Fvqa: Fact-based visual question answering. TPAMI, 40(10):2413–2427.

Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan Duan. 2023. Visual chatgpt: Talking, drawing and editing with visual foundation models. arXiv preprint arXiv:2303.04671.

Jialin Wu, Jiasen Lu, Ashish Sabharwal, and Roozbeh Mottaghi. 2022. Multi-modal answer validation for knowledge-based vqa. In AAAI, volume 36, pages 2712–2721.

Alexandros Xenos, Themos Stafylakis, Ioannis Patras, and Georgios Tzimiropoulos. 2023. A simple baseline for knowledge-based visual question answering. EMNLP.

Congying Xia, Caiming Xiong, and Philip Yu. 2021. Pseudo siamese network for few-shot intent generation. In SIGIR, pages 2005–2009.

Zhengyuan Yang, Zhe Gan, Jianfeng Wang, Xiaowei Hu, Yumao Lu, Zicheng Liu, and Lijuan Wang. 2022. An empirical study of gpt-3 for few-shot knowledgebased vqa. In AAAI, volume 36, pages 3081–3089.

Jing Yu, Zihao Zhu, Yujing Wang, Weifeng Zhang, Yue Hu, and Jianlong Tan. 2020. Cross-modal knowledge reasoning for knowledge-based visual question answering. Pattern Recognition, 108:107563.

Zhou Yu, Jun Yu, Yuhao Cui, Dacheng Tao, and Qi Tian. 2019. Deep modular co-attention networks for visual question answering. In CVPR, pages 6281–6290.

Zhou Yu, Jun Yu, Jianping Fan, and Dacheng Tao. 2017. Multi-modal factorized bilinear pooling with co-attention learning for visual question answering. In CVPR, pages 1821–1830.

Qinggang Zhang, Junnan Dong, Keyu Duan, Xiao Huang, Yezi Liu, and Linchuan Xu. 2022. Contrastive knowledge graph error detection. In CIKM, pages 2590–2599.

Qinggang Zhang, Junnan Dong, Qiaoyu Tan, and Xiao Huang. 2023a. Integrating entity attributes for erroraware knowledge graph embedding. IEEE Transactions on Knowledge and Data Engineering.

Yifeng Zhang, Shi Chen, and Qi Zhao. 2023b. Toward multi-granularity decision-making: Explicit visual reasoning with hierarchical knowledge. In ICCV, pages 2573–2583.

Yifeng Zhang, Ming Jiang, and Qi Zhao. 2021. Explicit knowledge incorporation for visual reasoning. In ICCV, pages 1356–1365.

Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. 2023. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592.

## A Appendix

## A.1 Generalization study with InstructBLIP

To demonstrate the generalization ability of our proposed MAIL, we adopt a more advanced multimodal LLM, i.e., InstructBLIP, to replace MiniGPT-4. Though we lack the subjective ability to answer YES/NO questions, we have achieved comparable improvements that significantly demonstrate ourselves with the nonnegligible merits for KBQA.

## A.2 Baselines used on FVQA Dataset

To further demonstrate the generalization ability of our proposed MAIL, we compare it with the widely adopted baselines on the first KVQA dataset FVQA, i.e., XNM (Shi et al., 2019), KI-Net (Zhang et al., 2021), UnifER (Guo et al., 2022), MCAN (Yu et al., 2019) and HCNMN (Zhang et al., 2023b). For external knowledge, KI-Net uses ConceptNet and Wordnet; UnifER uses Visual-Bert, LXMERT and ViLT; HCNMN uses WordNet, WikiText, ConceptNet and Visual Genome.

## A.3 Additional Case Studies

Case studies on MiniGPT-4. For fair comparisons, we further show the prediction results of MiniGPT-4 over the six examples in Table 11. It can only answer 2 out of 6 questions, which effectively indicates the superiority of both our task setting with candidate answers from KGs and the reasonable design of PS-GMF. From the results, we see that multimodal LLMs like MiniGPT-4 indeed require a careful design to be properly utilized so as to maximize their ability while solely using it for VQA could be unreliable.

Case Studies of MAIL w/o GMF. We showcase the predictions on both single-hop and multi-hop questions in this subsection without graph medium fusion in Table 11 line 3. The observation suggests the necessary cross-modal reasoning ability brought by GMF, where we can answer 3 out of 6 questions while failing the majority of complex multi-hop questions.

<table><tr><td>Method</td><td>External Knowledge</td><td>Fusion Strategy</td><td>Acc. (%)</td></tr><tr><td>Q Only</td><td></td><td></td><td>14.93</td></tr><tr><td colspan="4">Traditional End-to-end Baselines</td></tr><tr><td colspan="4"></td></tr><tr><td>RA-VQA TRiG</td><td>T5-large+Google Search Corpus T5+Wikipedia</td><td>Modality-agnostic Modality-agnostic</td><td>54.48 49.35</td></tr><tr><td colspan="4"></td></tr><tr><td colspan="4">Large Language Model-enhanced Baselines</td></tr><tr><td>Pica-Full TwO</td><td>Frozen GPT-3 (175B) Frozen GPT-3 (175B)</td><td></td><td>48.00 55.52</td></tr><tr><td>KAT (Single)</td><td>Frozen GPT-3 (175B) + Wikidata</td><td>Modality-agnostic</td><td>53.09</td></tr><tr><td>KAT (Ensemble)</td><td>Frozen GPT-3 (175B) + Wikidata</td><td>Modality-agnostic</td><td>54.41</td></tr><tr><td>REVIVE</td><td>Frozen GPT-3 (175B) + Wikidata</td><td>Modality-agnostic</td><td>53.83</td></tr><tr><td>A Simple Baseline4KBVQA</td><td></td><td>Modality-agnostic</td><td>59.07</td></tr><tr><td></td><td>Llama2(13B)+PNPVQA</td><td></td><td></td></tr><tr><td>PromptCap</td><td></td><td></td><td>60.4</td></tr><tr><td>Prophet</td><td>Frozen GPT-3 (175B)</td><td></td><td>61.1</td></tr><tr><td>MAIL (ours)</td><td>|Frozen InstructBLIP + ConceptNet| Modality-aware</td><td></td><td>61.77</td></tr></table>

Table 10: The overall performance comparison on benchmark dataset OK-VQA by using InstructBLIP.

<table><tr><td rowspan=1 colspan=1>Cases</td><td rowspan=1 colspan=1>Q1</td><td rowspan=1 colspan=1>Q2</td><td rowspan=1 colspan=1>Q3</td><td rowspan=1 colspan=1>Q4</td><td rowspan=1 colspan=1>Q5</td><td rowspan=1 colspan=1> $\overline { { \mathbf { Q } 6 } }$ </td></tr><tr><td rowspan=1 colspan=1>MiniGPT-4Result</td><td rowspan=1 colspan=1>|cobblestone |x</td><td rowspan=1 colspan=1>firefighter $\pmb { \chi }$ </td><td rowspan=1 colspan=1>|a purple and yellow train| $\pmb { \chi }$ </td><td rowspan=1 colspan=1>sharks√</td><td rowspan=1 colspan=1>springtimex</td><td rowspan=1 colspan=1>|catcher√</td></tr><tr><td rowspan=1 colspan=1>MAIL(w/o GMF)Result</td><td rowspan=1 colspan=1>brickV</td><td rowspan=1 colspan=1>fire√</td><td rowspan=1 colspan=1>railx</td><td rowspan=1 colspan=1>|currentsx</td><td rowspan=1 colspan=2>|keep warm in winter|catcherx            √</td></tr></table>

Table 11: Additional Case studies on both single-hop and multi-hop questions.