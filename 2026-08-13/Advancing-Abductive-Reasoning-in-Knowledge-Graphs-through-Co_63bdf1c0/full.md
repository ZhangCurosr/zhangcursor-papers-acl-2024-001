# Advancing Abductive Reasoning in Knowledge Graphs through Complex Logical Hypothesis Generation

Jiaxin Bai<sup>1</sup>∗, Yicheng Wang<sup>1</sup>∗, Tianshi Zheng<sup>1</sup>, Yue Guo<sup>1</sup>, Xin Liu<sup>2</sup>, and Yangqiu Song<sup>1</sup>

<sup>1</sup>Hong Kong University of Science and Technology, Clear Water Bay, Hong Kong SAR

<sup>2</sup>Amazon.com Inc, Palo Alto, USA

{jbai, ywangmy, tzhengad, ygouar}@connect.ust.hk, xliucr@amazon.com, yqsong@cse.ust.hk

## Abstract

Abductive reasoning is the process of making educated guesses to provide explanations for observations. Although many applications require the use of knowledge for explanations, the utilization of abductive reasoning in conjunction with structured knowledge, such as a knowledge graph, remains largely unexplored. To fill this gap, this paper introduces the task of complex logical hypothesis generation, as an initial step towards abductive logical reasoning with KG. In this task, we aim to generate a complex logical hypothesis so that it can explain a set of observations. We find that the supervised trained generative model can generate logical hypotheses that are structurally closer to the reference hypothesis. However, when generalized to unseen observations, this training objective does not guarantee better hypothesis generation. To address this, we introduce the Reinforcement Learning from Knowledge Graph (RLF-KG) method, which minimizes differences between observations and conclusions drawn from generated hypotheses according to the KG. Experiments show that, with RLF-KG’s assistance, the generated hypotheses provide better explanations, and achieve stateof-the-art results on three widely used KGs.<sup>1</sup>

## 1 Introduction

Abductive reasoning plays a vital role in generating explanatory hypotheses for observed phenomena across various research domains (Haig, 2012). It is a powerful tool with wide-ranging applications. For example, in cognitive neuroscience, reverse inference (Calzavarini and Cevolani, 2022), which is a form of abductive reasoning, is crucial for inferring the underlying cognitive processes based on observed brain activation patterns. Similarly, in clinical diagnostics, abductive reasoning is recognized as a key approach for studying cause-and-effect relationships (Martini, 2023). Moreover, abductive reasoning is fundamental to the process of hypothesis generation in humans, animals, and computational machines (Magnani, 2023). Its significance extends beyond these specific applications and encompasses diverse fields of study. In this paper, we are focused on abductive reasoning with structured knowledge, specifically, a knowledge graph.

A typical knowledge graph (KG) stores information about entities, like people, places, items, and their relations in graph structures. Meanwhile, KG reasoning is the process that leverages knowledge graphs to infer or derive new information (Zhang et al., 2021a, 2022; Ji et al., 2022). In recent years, various logical reasoning tasks are proposed over knowledge graph, for example, answering complex queries expressed in logical structure (Hamilton et al., 2018; Ren and Leskovec, 2020), or conducting logical rule mining (Galárraga et al., 2015; Ho et al., 2018; Meilicke et al., 2019).

However, the abductive perspective of KG reasoning is crucial yet unexplored. Consider the first example in Figure 1, where observation $O _ { 1 }$ depicts five celebrities followed by a user on a social media platform. The social network service provider is interested in using structured knowledge to explain the users’ observed behavior. By leveraging a knowledge graph like Freebase (Bollacker et al., 2008), which contains basic information about these individuals, a complex logical hypothesis $H _ { 1 }$ can be derived suggesting that they are all actors and screenwriters born in Los Angeles. In the second example from Figure 1, related to a user’s interactions on an e-commerce platform $( O _ { 2 } )$ , the structured hypothesis $H _ { 2 }$ can explain the user’s interest in Apple products released in 2010 excluding phones. The third example, dealing with medical diagnostics, presents three diseases $( O _ { 3 } )$ The corresponding hypothesis $H _ { 3 }$ indicates they are diseases $V _ { ? }$ with symptom $V _ { 1 }$ , and $V _ { 1 }$ can be relieved by Panadol. From a general perspective, these problems illustrate how abductive reasoning with knowledge graphs seeks hypotheses that best explain given observations (Josephson and Josephson, 1996; Thagard and Shelley, 1997).

<table><tr><td> $( 0 ) 1 0 \times 2 0 \times 2 0 0 0 0 0 0 0 0$ </td><td> $1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0$ </td><td>Interpretations</td></tr><tr><td> $O _ { 1 } = \{ \mathrm { G r a n t H e s l o v , J a s o n S e g e l } ,$  Robert Towne, Ronald Bass,  $\mathrm { R a s h i d a J o n e s } \}$ </td><td> ${ { H } _ { 1 } } = { { V } _ { ? } } : O c c u p a t i o n ( { { V } _ { ? } } , A c t o r ) \wedge$   $O c c u p a t i o n ( V _ { ? } , S c r e e e W r i t e r ) \land$   $B o r n I n ( V _ { ? } , L o s A n g e l e s )$ </td><td>The actors and screenwriters born in Los Angeles</td></tr><tr><td> $O _ { 2 } = \{ { \mathrm { I p a d ~ } } 1 { ^ { \mathrm { s t } } } { \mathrm { G e n } } , { \mathrm { I p o d ~ t o u c h ~ } } 4 ^ { \mathrm { t h } }$   $\mathrm { G e n , A p p l e T V 1 ^ { s t } G e n } \}$ </td><td> $H _ { 2 } = V _ { ? } : B r a n d ( V _ { ? } , A p p l e ) \wedge$   $R e l e a s e Y e a r ( V _ { ? } , 2 0 1 0 ) \land \lnot T y p e ( V _ { ? } , P h o n e )$ </td><td>The Apple products released in 2010 that are not phones</td></tr><tr><td> $O _ { 3 } = \{ { \mathrm { C o v i d } } \mathrm { - } 1 9 , { \mathrm { S e a s o n a l ~ F l u } } ,$  Dysmenorrhea}</td><td> $H _ { 3 } = V _ { ? } , \exists V _ { 1 } { : } H a v e S y m p t o m ( V _ { ? } , V _ { 1 } )$   $\land R e l i e v e d B y ( V _ { 1 } , P a n a d o l )$ </td><td>The disease whose symptoms can be relieved by Panadol</td></tr></table>

Figure 1: This figure shows some examples of observations and inferred logical hypotheses, expressed with discrepancies in interpretations.

A straightforward approach to tackle this reasoning task is to employ a search-based method to explore potential hypotheses based on the given observation. However, this approach faces two significant challenges. The first challenge arises from the incompleteness of knowledge graphs (KGs), as searching-based methods heavily rely on the availability of complete information. In practice, missing edges in KGs can negatively impact the performance of search-based methods (Ren and Leskovec, 2020). The second challenge stems from the complexity of logically structured hypotheses. The search space for search-based methods becomes exponentially large when dealing with combinatorial numbers of candidate hypotheses. Consequently, the search-based method struggles to effectively and efficiently handle observations that require complex hypotheses for explanation.

To overcome these challenges, we propose a solution that leverages generative models within a supervised learning framework to generate logical hypotheses for given observations. Our approach involves sampling hypothesis-observation pairs from observed knowledge graphs (Ren et al., 2020; Bai et al., 2023d) and training a transformerbased generative model (Vaswani et al., 2017) using the teacher-forcing method. However, a potential limitation of supervised training is that it primarily captures structural similarities, without necessarily guaranteeing improved explanations when applied to unseen observations. To address this, we introduce a technique called Reinforcement Learning from the Knowledge Graph (RLF-KG). It utilizes proximal policy optimization (PPO) (Schulman et al., 2017) to minimize the discrepancy between the observed evidence and the conclusion derived from the generated hypothesis. By incorporating reinforcement learning techniques, our approach aims to directly improve the explanatory capability of the generated hypotheses and ensure their effectiveness when generalized to unseen observations.

We evaluate the proposed methods for effectiveness and efficiency on three knowledge graphs: FB15k-237 (Toutanova and Chen, 2015), WN18RR (Toutanova and Chen, 2015), and DBpedia50 (Shi and Weninger, 2018). The results consistently demonstrate the superiority of our approach over supervised generation baselines and search-based methods, as measured by two evaluation metrics across all three datasets. Our contributions can be summarized as follows:

• We introduce the task of complex logical hypothesis generation, which aims to identify logical hypotheses that best explain a given set of observations. This task can be seen as a form of abductive reasoning with KGs.

• To address the challenges posed by the incompleteness of knowledge graphs and the complexity of logical hypotheses, we propose a generation-based method. This approach effectively handles these difficulties and enhances the quality of generated hypotheses.

• Additionally, we developed the Reinforcement Learning from the Knowledge Graph (RLF-KG) technique. By incorporating feedback from the knowledge graph, RLF-KG further improves the hypothesis generation model. It minimizes the discrepancies between the observations and the conclusions of the generated hypotheses, leading to more accurate and reliable results.

## 2 Problem Formulation

In this task, a knowledge graph is denoted as $\mathcal { G } = ( \nu , \mathcal { R } )$ , where is the set of vertices and is the set of relation types. Each relation type $r \in \mathcal { R }$ is a function : $\mathcal { V } \times \mathcal { V }  \{ \mathtt { t r u e } , \mathtt { f a l s e } \}$ ， with $r ( u , v )$ = true indicating the existence of a directed $( u , r , v )$ from vertex u to vertex v of type r in the graph , and false otherwise.

We adopt the open-world assumption of knowledge graphs (Drummond and Shearer, 2006), treating missing edges as unknown rather than false. The reasoning model can only access the observed KG , while the true KG <sup>¯</sup> is hidden and encompasses the observed graph .

Abductive reasoning is a type of logical reasoning that involves making educated guesses to infer the most likely reasons for the observations (Josephson and Josephson, 1996; Thagard and Shelley, 1997). For further details on the distinctions between abductive, deductive, and inductive reasoning, refer to Appendix A. In this work, we focus on a specific abductive reasoning type in the context of knowledge graphs. We first introduce some concepts in this context.

An observation is a set O of entities in . A logical hypothesis H on a graph $\mathcal { G } = ( \nu , \mathcal { R } )$ is defined as a predicate of a variable vertex $V _ { ? }$ in firstorder logical form, including existential quantifiers, AND ( ), OR ( ), and NOT ( ). The hypothesis can always be written in disjunctive normal form,

$$
H | _ { \mathcal { G } } ( V _ { ? } ) = \exists V _ { 1 } , \dotsc , V _ { k } : e _ { 1 } \lor \dots \lor e _ { n } ,\tag{1}
$$

$$
e _ { i } = r _ { i 1 } \wedge \cdot \cdot \cdot \wedge r _ { i m _ { i } } ,\tag{2}
$$

where each $r _ { i j }$ can take the forms: $r _ { i j } = r ( \boldsymbol { v } , V )$ $r _ { i j } = \lnot r ( \boldsymbol { v } , V ) , r _ { i j } = r ( V , V ^ { \prime } ) , r _ { i j } = \lnot r ( V , V ^ { \prime } )$ where v represents a fixed vertex, the $V , V ^ { \prime }$ represent variable vertices in $\{ V _ { ? } , V _ { 1 } , V _ { 2 } , \ldots , V _ { k } \}$ , and r is a relation type.

The subscript denotes that the hypothesis is formulated based on the given graph ${ \mathcal { G } } .$ This means that all entities and relations in the hypothesis must exist in ${ \mathcal { G } } _ { : }$ , and the domain for variable vertices is the entity set of $\mathcal { G }$ . For example, please refer to Appendix B. The same hypothesis $H$ can be applied to a different knowledge graph, $\mathcal { G } ^ { \prime }$ , provided that $\mathcal { G } ^ { \prime }$ includes the entities and edges present in $H$ When the context is clear or the hypothesis pertains to a general statement applicable to multiple knowledge graphs (e.g., observed and hidden graphs), the symbol H is used without the subscript.

The conclusion of the hypothesis H on a graph , denoted by $[ [ H ] ] _ { \mathcal { G } }$ , is the set of entities for which H holds true on $\mathcal { G } \mathrm { : }$

$$
[ H ] | _ { \mathcal { G } } = \{ V _ { ? } \in \mathcal { G } | H | _ { \mathcal { G } } ( V _ { ? } ) = \mathtt { t r u e } \} .\tag{3}
$$

Suppose $O = \{ v _ { 1 } , v _ { 2 } , . . . , v _ { k } \}$ represents an observation, $\mathcal { G }$ is the observed graph, and $\bar { \mathcal { G } }$ is the hidden graph. Then abductive reasoning in knowledge graphs aims to find the hypothesis H on $\mathcal { G }$ whose conclusion on the hidden graph $\bar { \mathcal { G } } , \mathbb { H } \mathbb { J } _ { \bar { \mathcal { G } } } .$ is most similar to $O .$ Formally, the similarity is quantified using the Jaccard index, defined as:

$$
\mathtt { J a c c a r d } ( \mathbb { [ } H \mathbb { ] } _ { \bar { \mathcal { G } } } , O ) = \frac { \lvert [ H ] \rvert _ { \bar { \mathcal { G } } } \cap O \rvert } { \lvert [ H ] \rvert _ { \bar { \mathcal { G } } } \cup O \rvert } .\tag{4}
$$

In other words, the goal is to find a hypothesis H that maximizes Jaccard $( \mathbb { H } \mathbb { I } _ { \bar { \mathcal { G } } } , O )$

## 3 Hypothesis Generation with RLF-KG

Our methodology, referred to as reinforcement learning from the knowledge graph (RLF-KG), is depicted in Fig. 4 and involves the following steps: (1) Randomly sample observation-hypothesis pairs from the knowledge graph. (2) Train a generative model to generate hypotheses from observations using the pairs. (3) Enhance the generative model using RLF-KG, leveraging reinforcement learning to minimize discrepancies between observations and generated hypotheses.

## 3.1 Sampling Observation-Hypothesis Pairs

In the first step, we randomly sample hypotheses from the observed training knowledge graph. This process starts by randomly selecting a hypothesis, followed by conducting a graph search on the training graph to derive its conclusion, which is then treated as the observation corresponding to the hypothesis. Details of the hypothesis sampling algorithm are provided in Appendix D.

Then, we convert both hypotheses and observations into sequences suitable for the generative models. For observations, we standardize the order of the elements, ensuring that permutations of the same observation set yield identical outputs. Each entity in an observation is represented as a unique token, such as [Apple] and [Phone], as shown in Figure 2, and is associated with an embedding.

Since each hypothesis can be represented as a directed acyclic graph, for hypotheses, we use a representation inspired by action-based parsing, similar to approaches seen in other logical reasoning studies (Hamilton et al., 2018; Ren and Leskovec, 2020; Ren et al., 2020). This involves utilizing a depth-first search algorithm, described in Appendix D, to generate a sequence of actions that represents the content and structure of the graph.

![](images/b07c8bfaeab229f818ae34fc0cd3a4b199b4564f1568a28a25bcda6f2d787348.jpg)

<table><tr><td>Nodes</td><td>Actions</td><td>Stack</td></tr><tr><td>1</td><td>[1]</td><td>1</td></tr><tr><td>2</td><td>[1]</td><td>1,2</td></tr><tr><td>3</td><td>[Brand]</td><td>1,2,3</td></tr><tr><td>4</td><td>[Apple]</td><td>1,2,3,4 → 1,2</td></tr><tr><td>5</td><td>[Release]</td><td>1,2,5</td></tr><tr><td>6</td><td>[2010]</td><td>1,2,5,6 → 1</td></tr><tr><td>7</td><td>[N]</td><td>1,7</td></tr><tr><td>8</td><td>[Type]</td><td>1,7,8</td></tr><tr><td>9</td><td>[Phone]</td><td>1,7,8,9 → empty</td></tr><tr><td></td><td>Tokens: [I][I][Brand][Apple]</td><td></td></tr><tr><td></td><td>[Release][2010][N][Type][Phone]</td><td></td></tr></table>

Figure 2: The figure demonstrates the tokenization process for hypotheses. We uniformly consider logical operations, relations, and entities as individual tokens, establishing a correspondence between the hypotheses and a sequence of tokens. For a more detailed explanation, please refer to the Appendix D.

![](images/9161a2d1521c991ffd082a160b8db935117fd3a0fa304c5f0c5d9f806790b1bf.jpg)  
Figure 3: The first two steps of training a hypothesis generation model: In Step 1, we randomly sample logical hypotheses with diverse patterns and perform graph searches on the training graphs to obtain observations. These observations are then tokenized. In Step 2, a conditional generation model is trained to generate hypotheses based on given tokenized observations.

Logical operations such as intersection, union, and negation are denoted by special tokens [I], [U], and [N] respectively, following prior work (Bai et al., 2023d). Relations and entities are similarly treated as unique tokens, for example, [Brand] and [Apple].

Furthermore, Algorithm 3 facilitates the reconstruction of a graph from an action sequence, serving as the detokenization process for hypotheses.

## 3.2 Supervised Training of Hypothesis Generation Model

In the second step, we train a generative model using the sampled pairs. Let $\mathbf { o } = [ o _ { 1 } , o _ { 2 } , . . . , o _ { m } ]$ represent the token sequences for observations, and h $\mathbf { \Omega } = [ h _ { 1 } , h _ { 2 } , . . . , h _ { n } ]$ for hypotheses. The loss for the generative model $\rho$ on this example is based on the standard sequence modeling loss:

$$
\mathscr { L } = \log \rho ( \mathbf { h } | \mathbf { o } )\tag{5}
$$

$$
= \log \sum _ { i = 1 } ^ { n } \rho ( h _ { i } | \mathbf { o } , h _ { 1 } , \dots , h _ { i - 1 } ) .\tag{6}
$$

We utilize a standard transformer to implement the conditional generative model, employing two distinct approaches. The first approach follows the encoder-decoder architecture as described by Vaswani et al. (2017), where observation tokens are fed into the transformer encoder, and the shifted hypothesis tokens are input into the transformer decoder. The second approach involves concatenating the observation and hypothesis tokens and using a decoder-only transformer to generate hypotheses. Following the setup of these architectures, we train the model using supervised training techniques.

## 3.3 Reinforcement Learning from Knowledge Graph Feedback (RLF-KG)

During the supervised training process, the model learns to generate hypotheses that have similar structures to reference hypotheses. However, higher structural similarity towards reference answers does not necessarily guarantee the ability to generate logical explanations, especially when encountering unseen observations during training.

To address this limitation, in the third step,

Step 3:

Optimize hypothesis generation model with Reinforcement Learning From Knowledge Graph feedback (RLF-KG).

![](images/ac21e63ac0458b301788deceab93fd7e54b2b46034a60e2edd299dbe018be3ab.jpg)  
Figure 4: In Step 3, we employ RLF-KG to encourage the model to generate hypotheses that align more closely with the given observations from the knowledge graph. RLF-KG helps improve the consistency between the generated hypotheses and the observed evidence in the knowledge graph.

we employ reinforcement learning (Ziegler et al., 2020) in conjunction with knowledge graph feedback (RLF-KG) to enhance the trained conditional generation model $\rho .$ Let $\mathcal { G } _ { \mathrm { t r a i n } }$ represents the observed training graph, h the hypothesis token sequence, o the observation token sequence, and $H , O$ the corresponding hypothesis and observation, respectively. We select the reward to be the Jaccard similarity between O and and the conclusion $[ [ H ] ] _ { \mathcal { G } _ { \mathrm { t r a i n } } }$ , which is a reliable and information leakage-free approximation for the objective of the abductive reasoning task in Equation 4. Formally, the reward function is defined as

$$
r ( \mathbf { h } , \mathbf { o } ) = \mathsf { J a c c a r d } ( \mathbb { \| } H \mathbb { \| } _ { \mathcal { G } _ { \mathrm { t r a i n } } } , O ) = \frac { | \mathbb { \| } H \mathbb { \| } _ { \mathcal { G } _ { \mathrm { t r a i n } } } \cap O | } { | \mathbb { \| } H \mathbb { \| } _ { \mathcal { G } _ { \mathrm { t r a i n } } } \cup O | } .\tag{7}
$$

Following Ziegler et al. (2020), we treat the trained model $\rho$ obtained from supervised training as the reference model, and initialize the model π to be optimized with $\rho .$ . Then, we modify the reward function by incorporating a KL divergence penalty. This modification aims to discourage the model $\pi$ from generating hypotheses that deviate excessively from the reference model.

## 4 Experiment

To train the model $\pi _ { \mathfrak { c } }$ , we employ the proximal policy optimization (PPO) algorithm (Schulman et al., 2017). The objective is to maximize the expected modified reward, given as follows:

$$
\mathbb { E } _ { \mathbf { o } \sim D , \mathbf { h } \sim \pi ( \cdot | \mathbf { o } ) } \left[ r ( \mathbf { h } , \mathbf { o } ) - \beta \log \frac { \pi ( \mathbf { h } | \mathbf { o } ) } { \rho ( \mathbf { h } | \mathbf { o } ) } \right] ,\tag{8}
$$

where D the is training observation distribution and $\pi ( \cdot | \mathbf { o } )$ is the conditional distribution of h on o modeled by the model π. By utilizing PPO and the modified reward function, we can effectively guide the model π towards generating hypotheses that strike a balance between the similarity to the reference model and logical coherence.

We utilize three distinct knowledge graphs, namely FB15k-237 (Toutanova and Chen, 2015), DBpedia50 (Shi and Weninger, 2018), and WN18RR (Toutanova and Chen, 2015), for our experiments. Table 1 provides an overview of the number of training, evaluation, and testing edges, as well as the total number of nodes in each knowledge graph. To ensure consistency, we randomly partition the edges of these knowledge graphs into three sets - training, validation, and testing - using an 8:1:1 ratio. Consequently, we construct the training graph $\mathcal { G } _ { \mathrm { t r a i n } } ,$ , validation graph $\mathcal { G } _ { \mathrm { v a l i d } }$ , and testing graph $\mathcal { G } _ { \mathrm { t e s t } }$ by including the corresponding edges: training edges only, training + validation edges, and training + validation + testing edges, respectively.

Following the methodology outlined in Section 3.2, we proceed to sample pairs of observations and hypotheses. To ensure the quality and diversity of the samples, we impose certain constraints during the sampling process. Firstly, we restrict the size of the observation sets to a maximum of thirty-two elements. This limitation is enforced ensuring that the observations remain manageable. Additionally, specific constraints are applied to the validation and testing hypotheses. Each validation hypothesis must incorporate additional entities in the conclusion compared to the training graph, while each testing hypothesis must have additional entities in the conclusion compared to the validation graph. This progressive increase in entity complexity ensures a challenging evaluation setting. The statistics of queries sampled for datasets are detailed in Appendix C.

In line with previous work on KG reasoning (Ren and Leskovec, 2020; Ren et al., 2020), we utilize thirteen pre-defined logical patterns to sample the hypotheses. Eight of these patterns, known as existential positive first-order (EPFO) hypotheses (1p/2p/2u/3i/ip/up/2i/pi), do not involve negations. The remaining five patterns are negation hypotheses (2in/3in/inp/pni/pin), which incorporate negations. It is important to note that the generated hypotheses may or may not match the type of the reference hypothesis. The structures of the hypotheses are visually presented in Figure 5, and the corresponding numbers of samples drawn for each hypothesis type can be found in Table 5.

![](images/355562657032fee87b8b161e19a02c9e452443d32a93372f4e471e4209325fac.jpg)

Figure 5: Our task involves considering thirteen distinct types of logical hypotheses. Each hypothesis type corresponds to a specific type of query graph, which is utilized during the sampling process. By associating each hypothesis type with a corresponding query graph, we ensure that a diverse range of hypotheses is sampled.
<table><tr><td>Dataset</td><td>Relations</td><td>Entities</td><td>Training</td><td>Validation</td><td>Testing</td><td>All Edges</td></tr><tr><td>FB15k-237</td><td>237</td><td>14,505</td><td>496,126</td><td>62,016</td><td>62,016</td><td>620,158</td></tr><tr><td>WN18RR</td><td>11</td><td>40,559</td><td>148,132</td><td>18,516</td><td>18,516</td><td>185,164</td></tr><tr><td>DBpedia50</td><td>351</td><td>24,624</td><td>55,074</td><td>6,884</td><td>6,884</td><td>68,842</td></tr></table>

Table 1: This figure provides basic information about the three knowledge graphs utilized in our experiments. The graphs are divided into standard sets of training, validation, and testing edges to facilitate the evaluation process.

## 4.1 Evaluation Metric

Jaccard Index The quality of the generated hypothesis is primarily measured using the Jaccard index same as that defined for abductive reasoning in Section 2, but we treat the constructed test graph $\mathcal { G } _ { \mathrm { t e s t } }$ as the hidden graph. It is important to note that the test graph contains ten percent of edges that were not observed during the training or validation stages. Formally, given an observation O and a generated hypothesis H, we employ a graph search algorithm to determine the conclusion of H on $\mathcal { G } _ { \mathrm { t e s t } }$ , denoted as $[ [ H ] ] _ { \mathcal { G } _ { \mathrm { t e s t } } }$ . Then the Jaccard metric for evaluation is defined as

$$
\mathtt { J a c c a r d } ( [ [ H ] ] _ { \mathcal { G } _ { \mathrm { t e s t } } } , O ) = \frac { \vert [ H ] \vert _ { \mathcal { G } _ { \mathrm { t e s t } } } \cap O \vert } { \vert [ H ] \vert _ { \mathcal { G } _ { \mathrm { t e s t } } } \cup O \vert } ,\tag{9}
$$

quantifing the similarity between the conclusion $[ [ H ] ] _ { \mathcal { G } _ { \mathrm { t e s t } } }$ and the observation O.

Smatch Score Smatch (Cai and Knight, 2013) is originally designed for comparing semantic graphs but has been recognized as a suitable metric for evaluating complex logical queries, which can be treated as a specialized form of semantic graphs (Bai et al., 2023d). In this task, a hypothesis can be represented as a graph, e.g. Fig. 2, and we can transform it to be compatible with the semantic graph format. The detailed computation of the Smatch score on hypothesis graphs is described in detail in Appendix F. Intuitively, the Smatch score between the generated hypothesis H and the reference hypothesis $H _ { \mathrm { r e f } }$ , denoted as $S ( H , H _ { \mathrm { r e f } } )$ quantifies the structural resemblance between the graphs corresponding to H and $H _ { \mathrm { r e f } } \mathrm { i . e . }$ , how much the nodes, edges, and the labels on them look the same between the two graphs.

## 4.2 Experiment Details

In this experiment, we use two transformer structures as the backbones of the generation model. For the encoder-decoder transformer structure, we use three encoder layers and three decoder layers. Each layer has eight attention heads with a hidden size of 512. Note that the positional encoding for the input observation sequence is disabled, as we believe that the order of the entities in the observation set does not matter. For the decoder-only structure, we use six layers, and the other hyperparameters are the same. In the supervised training process, we use AdamW optimizer and grid search to find hyper-parameters. For the encoderdecoder structure, the learning rate is 0.0001 with the resulting batch size of 768, 640, and 256 for FB15k-237, WN18RR, and DBpedia, respectively. For the decoder-only structure, the learning rate is

<table><tr><td>Dataset</td><td>Model</td><td>1p</td><td>2p</td><td>2i</td><td>3i</td><td>ip</td><td>pi</td><td>2u</td><td>up</td><td>2in</td><td>3in</td><td>pni</td><td>pin</td><td>inp</td><td>Ave.</td></tr><tr><td rowspan="4">FB15k-237</td><td>Encoder-Decoder</td><td>0.626</td><td>0.617</td><td>0.551</td><td>0.513</td><td>0.576</td><td>0.493</td><td>0.818</td><td>0.613</td><td>0.532</td><td>0.451</td><td>0.499</td><td>0.529</td><td>0.533</td><td>0.565</td></tr><tr><td>+ RLF-KG</td><td>0.855</td><td>0.711</td><td>0.661</td><td>0.595</td><td>0.715</td><td>0.608</td><td>0.776</td><td>0.698</td><td>0.670</td><td>0.530</td><td>0.617</td><td>0.590</td><td>0.637</td><td>0.666</td></tr><tr><td>Decoder-Only</td><td>0.666</td><td>0.643</td><td>0.593</td><td>0.554</td><td>0.612</td><td>0.533</td><td>0.807</td><td>0.638</td><td>0.588</td><td>0.503</td><td>0.549</td><td>0.559</td><td>0.564</td><td>0.601</td></tr><tr><td>+ RLF-KG</td><td>0.789</td><td>0.681</td><td>0.656</td><td>0.605</td><td>0.683</td><td>0.600</td><td>0.817</td><td>0.672</td><td>0.672</td><td>0.560</td><td>0.627</td><td>0.596</td><td>0.626</td><td>0.660</td></tr><tr><td rowspan="3">WN18RR</td><td>Encoder-Decoder + RLF-KG</td><td>0.793 0.850</td><td>0.734 0.778</td><td>0.692 0.765</td><td>0.692 0.763</td><td>0.797 0.854</td><td>0.627</td><td>0.763</td><td>0.690</td><td>0.707</td><td>0.694</td><td>0.704 0.738</td><td>0.653 0.682</td><td>0.664 0.710</td><td>0.708 0.753</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>0.685</td><td>0.767</td><td>0.719</td><td>0.743</td><td>0.732</td><td></td><td></td><td></td><td></td></tr><tr><td>Decoder-Only + RLF-KG</td><td>0.760 0.821</td><td>0.734 0.760</td><td>0.680 0.694</td><td>0.684 0.693</td><td>0.770 0.827</td><td>0.614 0.656</td><td>0.725 0.770</td><td>0.650 0.680</td><td>0.683 0.717</td><td>0.672 0.704</td><td>0.688 0.720</td><td>0.660 0.676</td><td>0.677 0.721</td><td>0.692 0.726</td></tr><tr><td rowspan="3">DBpedia50</td><td>Encoder-Decoder</td><td>0.706</td><td>0.657</td><td>0.551</td><td>0.570</td><td>0.720</td><td>0.583</td><td>0.632</td><td>0.636</td><td>0.602</td><td>0.572</td><td>0.668</td><td>0.625</td><td>0.636</td><td>0.627</td></tr><tr><td>+ RLF-KG</td><td>0.842</td><td>0.768</td><td>0.636</td><td>0.639</td><td>0.860</td><td>0.667</td><td>0.714</td><td>0.758</td><td>0.699</td><td>0.661</td><td>0.775</td><td>0.716</td><td>0.769</td><td>0.731</td></tr><tr><td>Decoder-Only + RLF-KG</td><td>0.739 0.777</td><td>0.692 0.701</td><td>0.426 0.470</td><td>0.434 0.475</td><td>0.771 0.821</td><td>0.527 0.534</td><td>0.654 0.646</td><td>0.688 0.702</td><td>0.602 0.626</td><td>0.563 0.575</td><td>0.663 0.696</td><td>0.640 0.626</td><td>0.701 0.713</td><td>0.623 0.643</td></tr></table>

Table 2: The detailed Jaccard performance of various methods.

![](images/42f71cf5a8ad8ef2d254a2e8a0fca23d61f8f42ef3e54f01b783791bf28bc1b3.jpg)

![](images/a68ad8b345057743dc77ab0bcdaa60a66349bb0f1c21201ae08970e38f7f663b.jpg)

![](images/972b1ad72e7217bf946f5b4746a5985a7f07ab78931da39f07dd72e19cc4739c.jpg)  
Figure 6: The curve of the reward values of RLF-KG training over three different datasets.

0.00001 with batch-size of 256, 160, and 160 for FB15k-237, WN18RR, and DBpedia respectively, and linear warming up of 100 steps. In the reinforcement learning process, we use the dynamic adjustment of the penalty coefficient $\beta$ (Ouyang et al., 2022). More detailed hyperparameters are shown in Appendix I. All the experiments can be conducted on a single GPU with 24GB memory.

## 4.3 Experiment Results and Discussions

We validate RLF-KG effectiveness by comparing the Jaccard metric of the model before and after this process. Table 2 displays performance across thirteen hypothesis types on FB15k-237, WN18RR, and DBpedia50. It illustrates Jaccard indices between observations and conclusions of generated hypotheses from the test graph. The encoder-decoder and the decoder-only transformers are assessed under fully supervised training on each dataset. Additionally, performance is reported when models collaborate with reinforcement learning from knowledge graph feedback (RLF-KG).

## 4.3.1 Performance Gain after RLF-KG

We notice RLF-KG consistently enhances hypothesis generation across three datasets, improving both encoder-decoder and decoder-only models. This can be explained by RLF-KG’s ability to incorporate knowledge graph information into the generation model, diverging from simply generating hypotheses akin to reference hypotheses.

Additionally, after the RLF-KG training, the encoder-decoder model surpasses the decoder-only structured transformer model. This is due to the task’s nature, where generating a sequence of tokens from an observation set does not necessitate the order of the observation set. Figure 6 supplements the previous statement by illustrating the increasing reward throughout the PPO process. We also refer readers to Appendix K for qualitative examples demonstrating the improvement in the generated hypotheses for the same observation.

## 4.3.2 Adding Structural Reward to PPO

We explore the potential benefits of incorporating structural similarity into the reward function used in PPO training. While RLF-KG originally relies on the Jaccard index, we consider adding the Smatch score, a measure of structural differences between generated and sampled hypotheses. We conducted additional experiments to include $S ( H , H _ { \mathrm { r e f } } )$ as an extra term in the reward function.

The results, presented in Table 3, indicate that by incorporating the structural reward, the model can indeed generate hypotheses that are closer to the reference hypotheses yc in structural sense. However, the Jaccard scores reveal that with the inclusion of structural information, the overall performance is comparable to or slightly worse than the original reward function. This is because adding Smatch score tends to cause the generation model to fit or potentially overfit the training data according to the above graph-level similarity, limiting its ability to generalize to unseen test graphs. The Jaccard reward (Eq. 7), however, captures a measure closer to the goal of our task . Detailed Smatch scores by query types can be found in Appendix G.

<table><tr><td rowspan="2"></td><td colspan="2">FB15k-237</td><td colspan="2">WN18RR</td><td colspan="2">DBpedia50</td></tr><tr><td>Jaccard</td><td>Smatch</td><td>Jaccard</td><td>Smatch</td><td>Jaccard</td><td>Smatch</td></tr><tr><td>Encoder-Decoder</td><td>0.565</td><td>0.602</td><td>0.708</td><td>0.558</td><td>0.627</td><td>0.486</td></tr><tr><td>+ RLF-KG (Jaccard)</td><td>0.666</td><td>0.530</td><td>0.753</td><td>0.540</td><td>0.731</td><td>0.541</td></tr><tr><td>+ RLF-KG (Jaccard + Smatch)</td><td>0.660</td><td>0.568</td><td>0.757</td><td>0.545</td><td>0.696</td><td>0.532</td></tr><tr><td>Decoder-Only</td><td>0.601</td><td>0.614</td><td>0.692</td><td>0.564</td><td>0.623</td><td>0.510</td></tr><tr><td>+ RLF-KG (Jaccard)</td><td>0.660</td><td>0.598</td><td>0.726</td><td>0.518</td><td>0.643</td><td>0.492</td></tr><tr><td>+ RLF-KG (Jaccard + Smatch)</td><td>0.656</td><td>0.612</td><td>0.713</td><td>0.540</td><td>0.645</td><td>0.504</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 3: The Jaccard and Smatch performance of different reward functions.

<table><tr><td>Method</td><td colspan="3">FB15k-237</td><td colspan="3">WN18RR</td><td colspan="3">DBpedia50</td></tr><tr><td></td><td>Runtime</td><td>Jaccard</td><td>Smatch</td><td>Runtime</td><td>Jaccard</td><td>Smatch</td><td>Runtime</td><td>Jaccard</td><td>Smatch</td></tr><tr><td>Brute-force Search</td><td>16345 mins</td><td>0.635</td><td>0.305</td><td>4084 mins</td><td>0.742</td><td>0.322</td><td>1132 mins</td><td>0.702</td><td>0.322</td></tr><tr><td>Generation + RLF-KG</td><td>264 mins</td><td>0.666</td><td>0.530</td><td>32 mins</td><td>0.753</td><td>0.540</td><td>5 mins</td><td>0.731</td><td>0.541</td></tr></table>

Table 4: Performance on testing data and runtime for inference for various methods on testing data.

## 4.3.3 Comparison Between Search Methods

We chose the relatively simple brute-force search (Algorithm 4) as our search-based baseline due to the inherent high complexity of search algorithms for this task. For each observation, the algorithm explores all potential 1p hypotheses within the training graph and selects the one with the highest Jaccard similarity on the training graph. Despite its simplicity and complexity linear to the number of edges, the algorithm requires significantly more time than our method, not to mention other more complex heuristics.

Following this choice, we compare inference time and performance between the generationbased and the brute-force search algorithm. Table 4 highlights the unsuitability of the brute-force search for scaling up due to its high complexity. In contrast, generation-based methods demonstrate substantially faster performance.

Moreover, the generation methods not only outperform the search-based method in Jaccard performance but also show a significant improvement in Smatch performance. The relatively high Jaccard score of the brute-force search is attributed to its inherent access to an approximation of Jaccard during the search process. Due to the nature of the graph splits, the approximation of Jaccard on the test graph using the training graph for 1p queries is found to be quite accurate, which enhances the average score. The detailed Jaccard scores are presented in Appendix H. However, the brute-force search struggles with more complex types of queries.

## 5 Related Work

The problem of abductive knowledge graph reasoning shares connections with various other knowledge graph reasoning tasks, including knowledge graph completion, complex logical query answering, and rule mining.

Rule Mining Rule mining focuses on inductive logical reasoning, namely discovering logical rules over the knowledge graph. Various methods are proposed in this line of work (Galárraga et al., 2015; Ho et al., 2018; Meilicke et al., 2019; Cheng et al., 2022, 2023). While the direct application of rule mining to solve this task is theoretically applicable, rule mining algorithms such as those described by Galárraga et al. (2015) also rely on searches to construct the horn clauses, which face scalability issues.

Complex logical query Complex logical query answering is a task of answering logically structured queries on KG. Query embedding primary focus is the enhancement of embedding structures for encoding sets of answers (Hamilton et al., 2018; Sun et al., 2020; Liu et al., 2021; Bai et al., 2023c,a,b; Hu et al., 2023; Yin et al., 2023; Wang et al., 2023a,b; Hu et al., 2024; Liu et al., 2024).

For instance, Ren and Leskovec (2020) and Zhang et al. (2021b) introduce the utilization of geometric structures such as rectangles and cones within hyperspace to represent entities. Neural MLP (Mixer) (Amayuelas et al., 2022) use MLP and MLP-Mixer as the operators. (Bai et al., 2022) suggests employing multiple vectors to encode queries, thereby addressing the diversity of answer entities. FuzzQE (Chen et al., 2022) uses fuzzy logic to represent logical operators. Probabilistic distributions can also serve as a means of query encoding (Choudhary et al., 2021a,b), with examples including Beta Embedding (Ren and Leskovec, 2020) and Gamma Embedding (Yang et al., 2022).

## 6 Conclusion

In summary, this paper has introduced the task of abductive logical knowledge graph reasoning. Meanwhile, this paper has proposed a generationbased method to address knowledge graph incompleteness and reasoning efficiency by generating logical hypotheses. Furthermore, this paper demonstrates the effectiveness of our proposed reinforcement learning from knowledge graphs (RLF-KG) to enhance our hypothesis generation model by leveraging feedback from knowledge graphs.

## Limitations

Our proposed methods and techniques in the paper are evaluated on a specific set of knowledge graphs, namely FB15k-237, WN18RR, and DBpedia50. It is unclear how well these approaches would perform on other KGs with different characteristics or domains. Meanwhile, knowledge graphs can be massive and continuously evolving, our method is not yet able to address the dynamic nature of knowledge evolutions, like conducting knowledge editing automatically. It is important to note that these limitations should not undermine the significance of the work but rather serve as areas for future research and improvement.

## Acknowledgements

We thank the anonymous reviewers and the area chair for their constructive comments. The authors of this paper were supported by the NSFC Fund (U20B2053) from the NSFC of China, the RIF (R6020-19 and R6021-20), and the GRF (16211520 and 16205322) from RGC of Hong Kong. We also thank the support from the UGC Research Matching Grants (RMGS20EG01-D,

RMGS20CR11, RMGS20CR12, RMGS20EG19, RMGS20EG21, RMGS23CR05, RMGS23EG08).

## References

Alfonso Amayuelas, Shuai Zhang, Susie Xi Rao, and Ce Zhang. 2022. Neural methods for logical reasoning over knowledge graphs. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. Open-Review.net.

Jiaxin Bai, Xin Liu, Weiqi Wang, Chen Luo, and Yangqiu Song. 2023a. Complex query answering on eventuality knowledge graph with implicit logical constraints. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Jiaxin Bai, Chen Luo, Zheng Li, Qingyu Yin, and Yangqiu Song. 2023b. Understanding inter-session intentions via complex logical reasoning. CoRR, abs/2312.13866.

Jiaxin Bai, Chen Luo, Zheng Li, Qingyu Yin, Bing Yin, and Yangqiu Song. 2023c. Knowledge graph reasoning over entities and numerical values. In Proceedings ofthe 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD 2023, Long Beach, CA, USA, August 6-10, 2023, pages 57–68. ACM.

Jiaxin Bai, Zihao Wang, Hongming Zhang, and Yangqiu Song. 2022. Query2Particles: Knowledge graph reasoning with particle embeddings. In Findings ofthe Association for Computational Linguistics: NAACL 2022, pages 2703–2714, Seattle, United States. Association for Computational Linguistics.

Jiaxin Bai, Tianshi Zheng, and Yangqiu Song. 2023d. Sequential query encoding for complex query answering on knowledge graphs. Transactions on Machine Learning Research.

Kurt D. Bollacker, Colin Evans, Praveen K. Paritosh, Tim Sturge, and Jamie Taylor. 2008. Freebase: a collaboratively created graph database for structuring human knowledge. In Proceedings ofthe ACM SIG-MOD International Conference on Management of Data, SIGMOD 2008, Vancouver, BC, Canada, June 10-12, 2008, pages 1247–1250. ACM.

Shu Cai and Kevin Knight. 2013. Smatch: an evaluation metric for semantic feature structures. In Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics, ACL 2013, 4-9 August 2013, Sofia, Bulgaria, Volume 2: Short Papers, pages 748–752. The Association for Computer Linguistics.

Fabrizio Calzavarini and Gustavo Cevolani. 2022. Abductive reasoning in cognitive neuroscience: Weak and strong reverse inference. Synthese, 200(2):1–26.

Xuelu Chen, Ziniu Hu, and Yizhou Sun. 2022. Fuzzy logic based logical query answering on knowledge graphs. In Thirty-Sixth AAAI Conference on Artificial Intelligence, AAAI 2022, Thirty-Fourth Conference on Innovative Applications ofArtificial Intelligence, IAAI 2022, The Twelveth Symposium on Educational Advances in Artificial Intelligence, EAAI 2022 Virtual Event, February 22 - March 1, 2022, pages 3939– 3948. AAAI Press.

Kewei Cheng, Nesreen K. Ahmed, and Yizhou Sun. 2023. Neural compositional rule learning for knowledge graph reasoning. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

Kewei Cheng, Jiahao Liu, Wei Wang, and Yizhou Sun. 2022. Rlogic: Recursive logical rule learning from knowledge graphs. In KDD ’22: The 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Washington, DC, USA, August 14 - 18, 2022, pages 179–189. ACM.

Nurendra Choudhary, Nikhil Rao, Sumeet Katariya, Karthik Subbian, and Chandan K. Reddy. 2021a. Probabilistic entity representation model for reasoning over knowledge graphs. In Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 23440–23451.

Nurendra Choudhary, Nikhil Rao, Sumeet Katariya, Karthik Subbian, and Chandan K. Reddy. 2021b. Self-supervised hyperboloid representations from logical queries over knowledge graphs. In WWW ’21: The Web Conference 2021, Virtual Event / Ljubljana, Slovenia, April 19-23, 2021, pages 1373–1384. ACM / IW3C2.

Igor Douven. 2021. Abduction. In Edward N. Zalta, editor, The Stanford Encyclopedia ofPhilosophy, Summer 2021 edition. Metaphysics Research Lab, Stanford University.

Nick Drummond and Rob Shearer. 2006. The open world assumption. In eSI Workshop: The Closed World of Databases meets the Open World of the Semantic Web, volume 15, page 1.

Luis Galárraga, Christina Teflioudi, Katja Hose, and Fabian M. Suchanek. 2015. Fast rule mining in ontological knowledge bases with AMIE+. VLDB J., 24(6):707–730.

Brian D. Haig. 2012. Abductive Learning, pages 10–12. Springer US, Boston, MA.

William L. Hamilton, Payal Bajaj, Marinka Zitnik, Dan Jurafsky, and Jure Leskovec. 2018. Embedding logical queries on knowledge graphs. In Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal, Canada, pages 2030–2041.

Vinh Thinh Ho, Daria Stepanova, Mohamed H. Gad-Elrab, Evgeny Kharlamov, and Gerhard Weikum. 2018. Rule learning from knowledge graphs guided by embedding models. In The Semantic Web - ISWC 2018 - 17th International Semantic Web Conference, Monterey, CA, USA, October 8-12, 2018, Proceedings, Part I, volume 11136 of Lecture Notes in Computer Science, pages 72–90. Springer.

Qi Hu, Weifeng Jiang, Haoran Li, Zihao Wang, Jiaxin Bai, Qianren Mao, Yangqiu Song, Lixin Fan, and Jianxin Li. 2024. Fedcqa: Answering complex queries on multi-source knowledge graphs via federated learning. CoRR, abs/2402.14609.

Qi Hu, Haoran Li, Jiaxin Bai, and Yangqiu Song. 2023. Privacy-preserving neural graph databases. CoRR, abs/2312.15591.

Shaoxiong Ji, Shirui Pan, Erik Cambria, Pekka Marttinen, and Philip S. Yu. 2022. A survey on knowledge graphs: Representation, acquisition, and applications. IEEE Trans. Neural Networks Learn. Syst., 33(2):494–514.

John R Josephson and Susan G Josephson. 1996. Abductive inference: Computation, philosophy, technology. Cambridge University Press.

Lihui Liu, Boxin Du, Heng Ji, ChengXiang Zhai, and Hanghang Tong. 2021. Neural-answering logical queries on knowledge graphs. In KDD ’21: The 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Virtual Event, Singapore, August 14-18, 2021, pages 1087–1097. ACM.

Lihui Liu, Zihao Wang, Jiaxin Bai, Yangqiu Song, and Hanghang Tong. 2024. New frontiers of knowledge graph reasoning: Recent advances and future trends. In Companion Proceedings ofthe ACM on Web Conference 2024, WWW 2024, Singapore, Singapore, May 13-17, 2024, pages 1294–1297. ACM.

Lorenzo Magnani. 2023. Handbook of Abductive Cognition. Springer International Publishing.

Carlo Martini. 2023. Abductive Reasoning in Clinical Diagnostics, pages 467–479. Springer International Publishing, Cham.

Christian Meilicke, Melisachew Wudage Chekol, Daniel Ruffinelli, and Heiner Stuckenschmidt. 2019. Anytime bottom-up rule learning for knowledge graph completion. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI 2019, Macao, China, August 10-16, 2019, pages 3137–3143. ijcai.org.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In NeurIPS.

Hongyu Ren, Weihua Hu, and Jure Leskovec. 2020. Query2box: Reasoning over knowledge graphs in vector space using box embeddings. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net.

Hongyu Ren and Jure Leskovec. 2020. Beta embeddings for multi-hop logical reasoning in knowledge graphs. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal Policy Optimization Algorithms. ArXiv:1707.06347 [cs].

Baoxu Shi and Tim Weninger. 2018. Open-world knowledge graph completion. In Proceedings ofthe Thirty-Second AAAI Conference on Artificial Intelligence, (AAAI-18), the 30th innovative Applications ofArtificial Intelligence (IAAI-18), and the 8th AAAI Symposium on Educational Advances in Artificial Intelligence (EAAI-18), New Orleans, Louisiana, USA, February 2-7, 2018, pages 1957–1964. AAAI Press.

Haitian Sun, Andrew O. Arnold, Tania Bedrax-Weiss, Fernando Pereira, and William W. Cohen. 2020. Faithful embeddings for knowledge base queries. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6- 12, 2020, virtual.

Paul Thagard and Cameron Shelley. 1997. Abductive reasoning: Logic, visual thinking, and coherence. In Logic and Scientific Methods: Volume One of the Tenth International Congress of Logic, Methodology and Philosophy of Science, Florence, August 1995, pages 413–427. Springer.

Kristina Toutanova and Danqi Chen. 2015. Observed versus latent features for knowledge base and text inference. In Proceedings of the 3rd Workshop on Continuous Vector Space Models and their Compositionality, CVSC 2015, Beijing, China, July 26-31, 2015, pages 57–66. Association for Computational Linguistics.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 5998–6008.

Zihao Wang, Weizhi Fei, Hang Yin, Yangqiu Song, Ginny Y. Wong, and Simon See. 2023a. Wassersteinfisher-rao embedding: Logical query embeddings with local comparison and global transport. In Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023, pages 13679–13696. Association for Computational Linguistics.

Zihao Wang, Yangqiu Song, Ginny Y. Wong, and Simon See. 2023b. Logical message passing networks with one-hop inference on atomic formulas. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

Dong Yang, Peijun Qing, Yang Li, Haonan Lu, and Xiaodong Lin. 2022. Gammae: Gamma embeddings for logical queries on knowledge graphs. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 745–760. Association for Computational Linguistics.

Hang Yin, Zihao Wang, Weizhi Fei, and Yangqiu Song. 2023. Efo -cqa: Towards knowledge graph complex query answering beyond set operation. CoRR, abs/2307.13701.

Jing Zhang, Bo Chen, Lingxi Zhang, Xirui Ke, and Haipeng Ding. 2021a. Neural, symbolic and neuralsymbolic reasoning on knowledge graphs. AI Open, 2:14–35.

Wen Zhang, Jiaoyan Chen, Juan Li, Zezhong Xu, Jeff Z. Pan, and Huajun Chen. 2022. Knowledge graph reasoning with logics and embeddings: Survey and perspective. CoRR, abs/2202.07412.

Zhanqiu Zhang, Jie Wang, Jiajun Chen, Shuiwang Ji, and Feng Wu. 2021b. Cone: Cone embeddings for multi-hop reasoning over knowledge graphs. In Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 19172–19183.

Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. 2020. Fine-Tuning Language Models from Human Preferences. ArXiv:1909.08593 [cs, stat].

## A Clarification on Abductive, Deductive, and Inductive Reasoning

Here we use simple syllogisms to explain the connections and differences between abductive reasoning and the other two types of reasoning, namely, deductive and inductive reasoning. In deductive reasoning, the inferred conclusion is necessarily true if the premises are true. Suppose we have a major premise $P _ { 1 }$ : All men are mortal, a minor premise $P _ { 2 } { \mathrm { : } }$ : Socrates is a man, then we can conclude C that Socrates is mortal. This can also be ${ \bar { P } } _ { 1 } \land { \bar { P } } _ { 2 }$

expressed as the inference $\overline { { C } }$ . On the other hand, abductive reasoning aims to explain an observation and is non-necessary, i.e., the inferred hypothesis is not guaranteed to be true. We also start with a premise P: All cats like catching mice, and then we have some observation O: Katty like catching mice. The abduction gives a simple yet most probable hypothesis H: Katty is a cat, as an explanation. This can also be written as the ${ \underline { { P \land O } } }$

inference $H$ . Different than deductive reasoning, the observation O should be entailed by the premise $P$ and the hypotheses H, which can be expressed by the implication $P \land H \implies O$ . The other type of non-necessary reasoning is inductive reasoning, where, in contrast to the appeal to explanatory considerations in abductive reasoning, there is an appeal to observed frequencies (Douven, 2021). For instance, premises $P _ { 1 }$ : Most Math students learn linear algebra in theirfirst year and $P _ { 2 } { \mathrm { : } }$ : Alice is a Math student infer $H \colon A l i c e$ learned $P _ { 1 } \land P _ { 2 }$

linear algebra in her first year, $\mathrm { i . e . , } \quad H$ . Note that the inference rules in the last two examples are not strictly logical implications.

It is worth mentioning that there might be different definitions or interpretations of these forms of reasoning.

## B Example of Observation-Hypothesis Pair

For example, observation O can be a set of name entities like GrantHeslov, JasonSegel, RobertTowne, RonaldBass, RashidaJones . Given this observation, an abductive reasoner is required to give the logical hypothesis that best explains it. For the above example, the expected hypothesis H in natural language is that they are actors and screenwriters, and they are also born in Los Angeles. Mathematically, the hypothesis H can be represented by a logical expression of the facts of the KG: H(V) = Occupation(V, Actor) ∧ Occupation(V, ScreenW riter) ∧ BornIn(V, LosAngeles). Although the logical expression here only contains logical conjunction AND $( \wedge )$ , we consider more general first-order logical form as defined in Section 2.

## C Statistics of Queries Sampled for Datasets

Table 5 presents the numbers of queries sampled for each dataset in each stage.

## D Algorithm for Sampling Observation-Hypothesis Pairs

Algorithm 1 is designed for sampling complex hypotheses from a given knowledge graph. Given a knowledge graph $\mathcal { G }$ and a hypothesis type $T ,$ , the algorithm starts with a random node v and proceeds to recursively construct a hypothesis that has v as one of its conclusions and adheres the type $T .$ . During the recursion process, the algorithm examines the last operation in the current hypothesis. If the operation is a projection, the algorithm randomly selects one of $v { \mathrm { s } }$ in-edge $( u , r , v )$ . Then, the algorithm calls the recursion on node u and the sub-hypothesis type of $T$ again. If the operation is an intersection, it applies recursion on the subhypotheses and the same node v. If the operation is a union, it applies recursion on one sub-hypothesis with node v and on other sub-hypotheses with an arbitrary node, as union only requires one of the sub-hypotheses to have v as an answer node. The recursion stops when the current node contains an entity.

## E Algorithms for Conversion between Queries and Actions

We here present the details of tokenizing the hypothesis graph (Algorithm 2), and formulating a graph according to the tokens, namely the process of de-tokenization (Algorithm 3). Inspired by the action-based semantic parsing algorithms, we view tokens as actions. It is worth noting that we employ the symbols $G , V , E$ for the hypothesis graph to differentiate it from the knowledge graph.

## F Details of using Smatch to evaluate structural differneces of queries

Smatch (Cai and Knight, 2013) is an evaluation metric for Abstract Meaning Representation (AMR) graphs. An AMR graph is a directed acyclic graph with two types of nodes: variable nodes and concept nodes, and three types of edges:

<table><tr><td rowspan="2">Dataset</td><td colspan="2">Training Samples</td><td colspan="2">Validation Samples</td><td colspan="2">Testing Samples</td></tr><tr><td>Each Type</td><td>Total</td><td>Each Type</td><td>Total</td><td>Each Type</td><td>Total</td></tr><tr><td>FB15k-237</td><td>496,126</td><td>6,449,638</td><td>62,015</td><td>806,195</td><td>62,015</td><td>806,195</td></tr><tr><td>WN18RR</td><td>148,132</td><td>1,925,716</td><td>18,516</td><td>240,708</td><td>18,516</td><td>240,708</td></tr><tr><td>DBpedia50</td><td>55,028</td><td>715,364</td><td>6,878</td><td>89,414</td><td>6,878</td><td>89,414</td></tr></table>

Table 5: The detailed information about the queries used for training, validation, and testing.

Algorithm 1 Sampling Hypothesis According to   
Type   
Input Knowledge graph , hypothesis type T   
Output Hypothesis sample   
function GROUNDTYPE( , T, t)   
if T.operation = p then   
$( h , r ) \gets \mathrm { S A M P L E } ( \{ ( h , r ) | ( h , r , t ) \in$   
E<sup>(</sup>G<sup>)</sup>}<sup>)</sup>   
$\hat { T } \gets$ the only subtype in T.SubTypes   
H GROUNDTYPE( , T, h<sup>ˆ</sup> )   
return $( p , r , H )$   
else if T.operation = i then   
SubHypotheses $ \emptyset$   
for $\hat { T } \in T . S u b T y p e s$ do   
$H \gets \mathrm { { G R O U N D T Y P E } } ( \mathcal { G } , \hat { T } , t )$   
SubHypotheses.PUSHBACK(H)   
end for   
return (i, SubHypotheses)   
else if T.operation = u then   
SubHypotheses   
for $\hat { T } \in T . S u b T$ ypes do   
if T<sup>ˆ</sup> is the first subtype then   
$H \gets \mathbf { G r o u n o T y p e } ( \mathcal { G } , \hat { T } , t )$   
else   
$\hat { t } \gets \mathrm { S A M P L E } ( \mathcal { V } ( \mathcal { G } ) )$   
H GROUNDTYPE( , T,<sup>ˆ</sup> tˆ)   
end if   
SubHypotheses.PUSHBACK(H)   
end for   
return (u, SubHypotheses)   
else if T.operation = e then   
return (e, t)   
end if   
end function   
v  SAMPLE( ( ))   
return GROUNDTYPE( , T, v)

Algorithm 2 HypothesisToActions   
Input Hypothesis plan graph G   
Output Action sequence A   
function DFS(G, t, A)   
if t is an anchor node then   
action entity associated with t   
else   
action operator associated with the   
first in-edge of t   
end if   
A.PUSHBACK(action)   
for all in-edges to t in $\textit { G } ( h , r , t )$ do   
DFS(G, h, A)   
end for   
end function   
root the root of G   
A DFS(G, root, )   
return A

• Instance edges, which connect a variable node to a concept node and are labeled literally “instance”. Every variable node must have exactly one instance edge, and vice versa.

• Attribute edges, which connect a variable node to a concept node and are labeled with attribute names.

• Relation edges, which connect a variable node to another variable node and are labeled with relation names.

Given a predicted AMR graph $G _ { \mathrm { p } }$ and the gold AMR graph $G _ { \mathrm { g } } .$ , the Smatch score of $G _ { \mathrm { p } }$ with respect to $G _ { \mathrm { g } }$ is denoted by Smatch $( G _ { \mathrm { p } } , G _ { \mathrm { g } } )$ Smatch $( G _ { \mathrm { p } } , G _ { \mathrm { g } } )$ is obtained by finding an approximately optimal mapping between the variable nodes of the two graphs and then matching the edges of the graphs.

Our hypothesis graph is similar to the AMR graph, in:

Algorithm 3 ActionsToHypothesis   
Input Action sequence A   
Output Hypothesis plan graph G   
S an empty stack   
Create an map deg. deg[i] = deg[u] = 2 and   
= 1 otherwise.   
V ,E   
for a A do   
Create a new node h, V V h   
if S = then   
(t, operator, d)  S.TOP   
E  E  (h, operator, t)   
end if   
if a represents an anchor then   
Mark h as an anchor with entity a   
while $S \neq \emptyset$ do   
(t, operator, d)  S.POP   
d d 1   
if d > 0 then   
S.PUSHBACK((t, operator, d))   
Break   
end if   
end while   
else   
S.PUSHBACK((h, a, deg[a]))   
end if   
end for   
G  (V, E)   
return G

• The nodes are both categorized as fixed nodes and variable nodes

• The edges can be categorized into two types: edges from a variable node to a fixed node and edges from a variable node to another variable node. And edges are labeled with names.

However, they are different in that the AMR graph requires every variable node to have instance edges, while the hypothesis graph does not.

The workaround for leveraging the Smatch score to measure the similarity between hypothesis graphs is creating an instance edge from every entity to some virtual node. Formally, given a hypothesis H with hypothesis graph $G ( H )$ , we create a new hypothesis graph $G _ { A } ( H )$ to accommodate the AMR settings as follows: First, we initialize $G _ { A } ( H ) = G ( H )$ . Then, create a new relation type instance and add a virtual node $v ^ { \prime }$ into $G _ { A } ( H )$ Finally, for every variable node $v \in G ( H )$ , we add a relation instance $( \boldsymbol { v } , \boldsymbol { v ^ { \prime } } )$ into $G _ { A } ( H )$ . Then, given a predicted hypothesis $H _ { p }$ and a gold hypothesis $H _ { g }$ , the Smatch score is defined as

$$
S ( H _ { p } , H _ { g } ) = \mathtt { S m a t c h } ( G _ { A } ( H _ { p } ) , G _ { A } ( H _ { g } ) ) .\tag{10}
$$

Through this conversion, a variable entity $v _ { g }$ of $H _ { \mathrm { g } }$ is mapped to a variable entity $v _ { p }$ of $H _ { \mathrm { p } }$ if and only if instance $( v _ { g } , v ^ { \prime } )$ is matched with instance $( v _ { p } , v ^ { \prime } )$ . This modification does not affect the overall algorithm for finding the optimal mapping between variable nodes and hence gives a valid and consistent similarity score. However, this adds an extra point for matching between instance edges, no matter how the variable nodes are mapped.

## G Detailed Smatch Scores by Query Types

Tables 6 and 7 show the detailed Smatch performance of various methods.

## H Detailed Jaccard Performance of the Brute-force Search

Table 8 shows the detailed Jaccard performance of the brute-force search.

## I Hyperparameters of the RL Process

The PPO hyperparameters are shown in Table 9. We warm-uped the learning rate from 0.1 of the peak to the peak value within the first 10% of total

<table><tr><td>Dataset</td><td>Model</td><td>1p</td><td>2p</td><td>2i</td><td>3i</td><td>ip</td><td>pi</td><td>2u</td><td>up</td><td>2in</td><td>3in</td><td>pni</td><td>pin</td><td>inp</td><td>Ave.</td></tr><tr><td rowspan="5">FB15k-237</td><td>Enc.-Dec. RLF-KG (J)</td><td>0.342 0.721</td><td>0.506 0.643</td><td>0.595 0.562</td><td>0.602 0.480</td><td>0.570 0.364</td><td>0.598 0.475</td><td>0.850 0.769</td><td>0.571 0.431</td><td>0.652 0.543</td><td>0.641 0.499</td><td>0.650 0.513</td><td>0.655</td><td>0.599</td><td>0.602</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.518</td><td>0.370</td><td>0.530</td></tr><tr><td>RLF-KG (J+S)</td><td>0.591</td><td>0.583</td><td>0.577</td><td>0.531</td><td>0.447</td><td>0.520</td><td>0.820</td><td>0.505</td><td>0.602</td><td>0.563</td><td>0.571</td><td>0.595</td><td>0.484</td><td>0.568</td></tr><tr><td>Dec.-Only</td><td>0.287</td><td>0.481</td><td>0.615</td><td>0.623</td><td>0.599</td><td>0.626</td><td>0.847</td><td>0.574</td><td>0.680</td><td>0.656</td><td>0.675</td><td>0.677</td><td>0.636</td><td>0.614</td></tr><tr><td>RLF-KG (J)</td><td>0.344</td><td>0.445</td><td>0.675</td><td>0.585</td><td>0.537</td><td>0.638</td><td>0.853</td><td>0.512</td><td>0.696</td><td>0.575</td><td>0.647</td><td>0.688</td><td>0.574</td><td>0.598</td></tr><tr><td rowspan="5"></td><td>RLF-KG (J+S)</td><td>0.303</td><td>0.380</td><td>0.692</td><td>0.607</td><td>0.565</td><td>0.671</td><td>0.857</td><td>0.506</td><td>0.727</td><td>0.600</td><td>0.676</td><td>0.734</td><td>0.634</td><td>0.612</td></tr><tr><td>Enc.-Dec.</td><td>0.375</td><td>0.452</td><td>0.591</td><td>0.555</td><td>0.437</td><td>0.585</td><td>0.835</td><td>0.685</td><td>0.586</td><td>0.516</td><td>0.561</td><td>0.549</td><td>0.530</td><td>0.558</td></tr><tr><td>RLF-KG (J)</td><td>0.455</td><td>0.468</td><td>0.563</td><td>0.562</td><td>0.361</td><td>0.530</td><td>0.810</td><td>0.646</td><td>0.560</td><td>0.530</td><td>0.536</td><td>0.539</td><td>0.465</td><td>0.540</td></tr><tr><td>RLF-KG (J+S)</td><td>0.443</td><td>0.457</td><td>0.565</td><td>0.572</td><td>0.366</td><td>0.545</td><td>0.814</td><td>0.661</td><td>0.541</td><td>0.553</td><td>0.532</td><td>0.546</td><td>0.491</td><td>0.545</td></tr><tr><td>Dec.-Only</td><td>0.320</td><td>0.443</td><td>0.582</td><td>0.551</td><td>0.486</td><td>0.597</td><td>0.809</td><td>0.696</td><td>0.594</td><td>0.526</td><td>0.575</td><td>0.574</td><td>0.577</td><td>0.564</td></tr><tr><td rowspan="5"></td><td>RLF-KG (J)</td><td>0.400</td><td>0.438</td><td>0.566</td><td>0.491</td><td>0.403</td><td>0.519</td><td>0.839</td><td>0.667</td><td>0.547</td><td>0.450</td><td>0.497</td><td>0.466</td><td>0.450</td><td>0.518</td></tr><tr><td>RLF-KG (J+S)</td><td>0.375</td><td>0.447</td><td>0.584</td><td>0.499</td><td>0.432</td><td>0.545</td><td>0.825</td><td>0.679</td><td>0.584</td><td>0.477</td><td>0.543</td><td>0.522</td><td>0.507</td><td>0.540</td></tr><tr><td>Enc.-Dec.</td><td>0.345</td><td>0.396</td><td>0.570</td><td>0.548</td><td>0.344</td><td>0.576</td><td>0.712</td><td>0.544</td><td>0.474</td><td>0.422</td><td>0.477</td><td>0.488</td><td>0.428</td><td>0.486</td></tr><tr><td>RLF-KG (J)</td><td>0.461</td><td>0.424</td><td>0.634</td><td>0.584</td><td>0.361</td><td>0.575</td><td>0.809</td><td>0.579</td><td>0.584</td><td>0.497</td><td>0.544</td><td>0.533</td><td>0.454</td><td>0.541</td></tr><tr><td>RLF-KG (J+S)</td><td>0.419</td><td>0.410</td><td>0.638</td><td>0.555</td><td>0.373</td><td>0.586</td><td>0.785</td><td>0.579</td><td>0.560</td><td>0.459</td><td>0.536</td><td>0.542</td><td>0.474</td><td>0.532</td></tr><tr><td rowspan="4">DBpedia50</td><td></td><td></td><td>0.408</td><td>0.559</td><td>0.526</td><td>0.397</td><td>0.568</td><td>0.812</td><td>0.626</td><td>0.480</td><td>0.414</td><td>0.489</td><td>0.494</td><td></td><td>0.510</td></tr><tr><td>Dec.-Only RLF-KG (J)</td><td>0.378 0.405</td><td>0.411</td><td>0.558</td><td>0.496</td><td>0.376</td><td>0.507</td><td>0.825</td><td>0.621</td><td>0.477</td><td>0.397</td><td>0.468</td><td>0.444</td><td>0.474 0.406</td><td>0.492</td></tr><tr><td>RLF-KG (J+S)</td><td>0.398</td><td>0.415</td><td>0.567</td><td>0.497</td><td>0.383</td><td>0.533</td><td>0.827</td><td>0.630</td><td>0.510</td><td>0.420</td><td>0.484</td><td>0.457</td><td>0.430</td><td>0.504</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 6: The detailed Smatch performance of various methods.
<table><tr><td>Dataset</td><td>1p</td><td>2p</td><td>2i</td><td>3i</td><td>ip</td><td>pi</td><td>2u</td><td>up</td><td>2in</td><td>3in</td><td>pni</td><td>pin</td><td>inp</td><td>Ave.</td></tr><tr><td>FB15k-237</td><td>0.945</td><td>0.340</td><td>0.365</td><td>0.218</td><td>0.184</td><td>0.267</td><td>0.419</td><td>0.185</td><td>0.301</td><td>0.182</td><td>0.245</td><td>0.155</td><td>0.157</td><td>0.305</td></tr><tr><td>WN18RR</td><td>0.957</td><td>0.336</td><td>0.420</td><td>0.274</td><td>0.182</td><td>0.275</td><td>0.427</td><td>0.183</td><td>0.323</td><td>0.224</td><td>0.270</td><td></td><td>0.1550.156</td><td>0.322</td></tr><tr><td>DBpedia</td><td></td><td>0.991 0.336</td><td>0.399</td><td>0.259</td><td>0.182</td><td>0.245</td><td>0.441</td><td>0.183</td><td></td><td>0.332 0.2260.2900.1540.155</td><td></td><td></td><td></td><td>0.322</td></tr></table>

Table 7: The detailed Smatch performance of the searching method.
<table><tr><td>Dataset</td><td>1p</td><td>2p</td><td>2i</td><td>3i</td><td>ip</td><td>pi</td><td>2u</td><td>up</td><td>2in</td><td>3in</td><td>pni</td><td>pin</td><td>inp</td><td>Ave.</td></tr><tr><td>FB15k-237</td><td>0.980</td><td>0.563</td><td>0.639</td><td>0.563</td><td>0.732</td><td>0.633</td><td>0.744</td><td>0.585</td><td>0.659</td><td>0.479</td><td>0.607</td><td>0.464</td><td>0.603</td><td>0.635</td></tr><tr><td>WN18RR</td><td>0.997</td><td>0.622</td><td>0.784</td><td>0.776</td><td>0.955</td><td>0.666</td><td>0.753</td><td>0.605</td><td>0.783</td><td>0.757</td><td>0.762</td><td>0.540</td><td>0.630</td><td>0.741</td></tr><tr><td>DBpedia</td><td>0.997</td><td>0.705</td><td>0.517</td><td>0.517</td><td>0.982</td><td>0.461</td><td>0.783</td><td>0.754</td><td>0.722</td><td>0.658</td><td>0.782</td><td>0.544</td><td>0.700</td><td>0.702</td></tr></table>

Table 8: The detailed Jaccard performance of the brute-force search.

iterations, followed by a decay to 0.1 of the peak using a cosine schedule.

## J Algorithms for One-Hop Searching

We now introduce the Algorithm 4 used for searching the best one-hop hypothesis with the tail among all entities in the observation set to explain the observations.

## K Case Studies

Explore Table 10, 11 and 12 for concrete examples generated by various abductive reasoning methods, namely search, generative model with supervised training, and generative model with supervised training incorporating RLF-KG.

Algorithm 4 One-Hop-Search   
Input Observation set O   
Output Hypothesis bestHypothesis   
candidates $ \{ ( h , r , t ) \in \mathcal { R } _ { \operatorname { t r a i n } } | t \in O \}$   
bestJaccard 0   
bestHypothesis Null   
for (h, r, t) candidates do   
H the one-hop hypothesis consisting of   
edge (h, r, t)   
nowJaccard Jaccard $. ( \mathbb { I } H ] _ { \mathcal { G } _ { \operatorname { t r a i n } } } , A )$   
if nowJaccard > bestJaccard then   
bestJaccard nowJaccard   
bestHypothesis H   
end if   
end for   
return bestHypothesis

<table><tr><td rowspan="2">Hyperparam.</td><td colspan="3">Enc.-Dec.</td><td colspan="3">Dec.-Only</td></tr><tr><td>FB15k-237</td><td>WN18RR</td><td>DBpedia50</td><td>FB15k-237</td><td>WN18RR</td><td>DBpedia50</td></tr><tr><td>Learning rate</td><td>2.4e-5</td><td>3.1e-5</td><td>1.8e-5</td><td>0.8e-5</td><td>0.8e-5</td><td>0.6e-5</td></tr><tr><td>Batch size</td><td>16384</td><td>16384</td><td>4096</td><td>3072</td><td>2048</td><td>2048</td></tr><tr><td>Minibatch size</td><td>512</td><td>512</td><td>64</td><td>96</td><td>128</td><td>128</td></tr><tr><td>Horizon</td><td>4096</td><td>4096</td><td>4096</td><td>2048</td><td>2048</td><td>2048</td></tr></table>

Table 9: PPO Hyperparamters.

<table><tr><td rowspan=3 colspan=3>InterpretationSample</td><td rowspan=1 colspan=1>Companies operating in industries that intersect with Yahoo! but not with IBM.</td></tr><tr><td rowspan=1 colspan=1>Hypothesis</td><td rowspan=1 colspan=1>The observations are the V? such that ∃V1, inIndustry(V1, V?) ∧¬industryOf(I BM, V1) ∧ industryOf(Y ahoo!, V1)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>EMI,                                     CBS_Corporation,Columbia,                                GMA_Network,Viacom,                                 Victor_Entertainment,Yahoo!,                                  Sony_Music_Entertainment_(Japan)_Inc.,Bandai,                                  Toho_Co.,_Ltd.,Rank_Organisation,                      The_New_York_Times_Company,Gannett_Company,                      Star_Cinema,NBCUniversal,                          TV5,Pony_Canyon,                           Avex_Trax,The_Graham_Holdings_Company,       The_Walt_Disney_Company,Televisa,                                 Metro-Goldwyn-Mayer,Google,                                  Time_Warner,Microsoft_Corporation,                  Dell,Munhwa_Broadcasting_Corporation,     News_Corporation</td></tr><tr><td rowspan=2 colspan=2>Searching</td><td rowspan=1 colspan=1>Interpretation</td><td rowspan=1 colspan=1>Which companies operate in media industry?</td></tr><tr><td rowspan=1 colspan=1>Hypothesis</td><td rowspan=1 colspan=1>The observations are the V? such that inIndustry(Media, V?)</td></tr><tr><td rowspan=3 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Conclusion</td></tr><tr><td rowspan=1 colspan=1>Jaccard</td><td rowspan=1 colspan=1>0.893</td></tr><tr><td rowspan=1 colspan=1>Smatch</td><td rowspan=1 colspan=1>0.154</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>Interpretation</td><td rowspan=1 colspan=1>Companies operating in industries that intersect withYahoo! but not with Microsoft Corporation.</td></tr><tr><td rowspan=4 colspan=2>Enc.-Dec.</td><td rowspan=1 colspan=1>Hypothesis</td><td rowspan=1 colspan=1>The observations are the V? such that ∃V1, inIndustry(V1, V?) ∧¬industryOf(Microsoft_Corporation, V1) ∧ industryOf(Y ahoo!, V1)</td></tr><tr><td rowspan=1 colspan=1>Conclusion</td><td rowspan=1 colspan=1>Absent: Microsoft_Corporation</td></tr><tr><td rowspan=1 colspan=1>Jaccard</td><td rowspan=1 colspan=1>0.964</td></tr><tr><td rowspan=1 colspan=1>Smatch</td><td rowspan=1 colspan=1>0.909</td></tr><tr><td rowspan=5 colspan=2>+ RLF-KG</td><td rowspan=1 colspan=1>Interpretation</td><td rowspan=1 colspan=1>Companies operating in industries that intersect withYahoo! but not with Oracle Corporation.</td></tr><tr><td rowspan=1 colspan=1>Hypothesis</td><td rowspan=1 colspan=1>The observations are the V? such that ∃V1, inIndustry(V1, V?) ∧¬industryOf(Oracle_Corporation, V1) ∧ industryOf(Yahoo!, V1)</td></tr><tr><td rowspan=1 colspan=1>Concl.</td><td rowspan=1 colspan=1>Same</td></tr><tr><td rowspan=1 colspan=1>Jaccard</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>Smatch</td><td rowspan=1 colspan=1>0.909</td></tr></table>

Table 10: Example FB15k-237 Case study 1.

<table><tr><td colspan="1" rowspan="3">Sample</td><td colspan="2" rowspan="1">Locations that adjoin second-level divisions of the UnitedInterpretationStates of America that adjoin Washtenaw County.</td></tr><tr><td colspan="1" rowspan="1">Hypothesis</td><td colspan="1" rowspan="1">The observations are the V? such that ∃V1, adjoins(V1, V?) ∧adjoins(Washtenaw_County, V1) ∧ secondLevelDivisions(USA, V1)</td></tr><tr><td colspan="1" rowspan="1">Observation</td><td colspan="1" rowspan="1">Jackson_County,                         Macomb_County,Wayne_County,                          Ingham_CountyWashtenaw_County,</td></tr><tr><td colspan="1" rowspan="5">Searching</td><td colspan="1" rowspan="1">Interpretation</td><td colspan="1" rowspan="1">Locations that adjoin Oakland County.</td></tr><tr><td colspan="1" rowspan="1">Hypothesis</td><td colspan="1" rowspan="1">The observations are the V? such that adjoins(Oakland_County, V?)</td></tr><tr><td colspan="1" rowspan="1">Conclusion</td><td colspan="1" rowspan="1">Absent:- Jackson_County- Ingham_County</td></tr><tr><td colspan="1" rowspan="1">Jaccard</td><td colspan="1" rowspan="1">0.600</td></tr><tr><td colspan="1" rowspan="1">Smatch</td><td colspan="1" rowspan="1">0.182</td></tr><tr><td colspan="1" rowspan="5">Enc.-Dec.</td><td colspan="1" rowspan="1">Interpretation</td><td colspan="1" rowspan="1">Second-level divisions of the United States of Americathat adjoin locations that adjoin Oakland County.</td></tr><tr><td colspan="1" rowspan="1">Hypothesis</td><td colspan="1" rowspan="1">The      observations      are      the      V?      such      that∃V1, secondLevelDivisions(USA, V?)    ∧    adjoins(V1, V?)    ∧+adjoins(Oakland_County, V1)</td></tr><tr><td colspan="1" rowspan="1">Conclusion</td><td colspan="1" rowspan="1">Extra: Oakland_CountyAbsent: Wayne_County</td></tr><tr><td colspan="1" rowspan="1">Jaccard</td><td colspan="1" rowspan="1">0.667</td></tr><tr><td colspan="1" rowspan="1">Smatch</td><td colspan="1" rowspan="1">0.778</td></tr><tr><td colspan="1" rowspan="5">+ RLF-KG</td><td colspan="1" rowspan="1">Interpretation</td><td colspan="1" rowspan="1">Second-level divisions of the United States of Americathat adjoin locations contained within Michigan.</td></tr><tr><td colspan="1" rowspan="1">Hypothesis</td><td colspan="1" rowspan="1">The      observations      are      the      V?      such      that∃V1, secondLevelDivisions(USA, V?)    ∧    adjoins(V1, V?)    ∧containedIn(Michigan, V1)</td></tr><tr><td colspan="1" rowspan="1">Conclusion</td><td colspan="1" rowspan="1">Extra:- Oakland_County- Genesee_County</td></tr><tr><td colspan="1" rowspan="1">Jaccard</td><td colspan="1" rowspan="1">0.714</td></tr><tr><td colspan="1" rowspan="1">Smatch</td><td colspan="1" rowspan="1">0.778</td></tr><tr><td colspan="1" rowspan="3">Ground Truth</td><td colspan="1" rowspan="1">Interpretation</td><td colspan="1" rowspan="1">Works, except for “Here 'Tis," that have subsequent works in the jazz genre.</td></tr><tr><td colspan="1" rowspan="1">Hypothesis</td><td colspan="1" rowspan="1">The observations are the V? such that ∃V1, subsequentWork(V1, V?) ∧¬previousWork(Here_′Tis, V1) ∧ genre(Jazz, V1)</td></tr><tr><td colspan="1" rowspan="1">Observation</td><td colspan="1" rowspan="1">Deep,_Deep_Trouble,                    Lee_Morgan_Sextet,Good_Dog,_Happy_Man,                Paris_NightsVNew_York_Mornings,I_Don't_Want_to_Be_Your Friend.      Take_the BoxInterior_Music,</td></tr><tr><td colspan="1" rowspan="5">Searching</td><td colspan="1" rowspan="1">Interpretation</td><td colspan="1" rowspan="1">Works subsequent to “Closer" (Corinne Bailey Rae song).</td></tr><tr><td colspan="1" rowspan="1">Hypothesis</td><td colspan="1" rowspan="1">The      observations      are      the      V?      such      thatsubsequentWork(Closer_(Corinne_Bailey_Rae_song), V?)</td></tr><tr><td colspan="1" rowspan="1">Conclusion</td><td colspan="1" rowspan="1">Only Paris_NightsVNew_York_Mornings</td></tr><tr><td colspan="1" rowspan="1">Jaccard</td><td colspan="1" rowspan="1">0.143</td></tr><tr><td colspan="1" rowspan="1">Smatch</td><td colspan="1" rowspan="1">0.154</td></tr><tr><td colspan="1" rowspan="5">Enc.-Dec.</td><td colspan="1" rowspan="1">Interpretation</td><td colspan="1" rowspan="1">Works, except for “Lee Morgan Sextet," that have subsequent worksin the jazz genre.</td></tr><tr><td colspan="1" rowspan="1">Hypothesis</td><td colspan="1" rowspan="1">The observations are the V? such that ∃V1, subsequentWork(V1, V?) ∧¬previousWork(Lee_Morgan_Sextet, V1) ∧ genre(Jazz, V1)</td></tr><tr><td colspan="1" rowspan="1">Conclusion</td><td colspan="1" rowspan="1">Extra: Here_'TisAbsent: Lee_Morgan_Sextet</td></tr><tr><td colspan="1" rowspan="1">Jaccard</td><td colspan="1" rowspan="1">0.750</td></tr><tr><td colspan="1" rowspan="1">Smatch</td><td colspan="1" rowspan="1">0.909</td></tr><tr><td colspan="1" rowspan="5">+ RLF-KG</td><td colspan="1" rowspan="1">Interpretation</td><td colspan="1" rowspan="1">Works that have subsequent works in the jazz genre.</td></tr><tr><td colspan="1" rowspan="1">Hypothesis</td><td colspan="1" rowspan="1">The observations are the V? such that ∃V1, subsequentWork(V1, V?) ∧genre(Jazz, V1)</td></tr><tr><td colspan="1" rowspan="1">Conclusion</td><td colspan="1" rowspan="1">Extra: Here_'Tis</td></tr><tr><td colspan="1" rowspan="1">Jaccard</td><td colspan="1" rowspan="1">0.875</td></tr><tr><td colspan="1" rowspan="1">Smatch</td><td colspan="1" rowspan="1">0.400</td></tr></table>

Table 11: FB15k-237 Case study 2.

Table 12: DBpedia50 Case study.