# A Unified Temporal Knowledge Graph Reasoning Model Towards Interpolation and Extrapolation

Kai Chen<sup>1</sup>∗, Ye Wang<sup>1</sup>∗ , Yitong Li<sup>2</sup>†, Aiping Li<sup>1</sup>† , Han Yu<sup>1</sup>, Xin Song<sup>1</sup>

<sup>1</sup>National University of Defense Technology, Changsha, China

<sup>2</sup>Huawei Technologies Co., Ltd.

{chenkai\_,ye.wang,liaiping,yuhan17,songxin}@nudt.edu.cn liyitong3@huawei.com

## Abstract

Temporal knowledge graph (TKG) reasoning has two settings: interpolation reasoning and extrapolation reasoning. Both of them draw plenty of research interest and have great significance. Methods of the former deemphasize the temporal correlations among facts sequences, while methods of the latter require strict chronological order of knowledge and ignore inferring clues provided by missing facts of the past. These limit the practicability of TKG applications as almost all of the existing TKG reasoning methods are designed specifically to address either one setting. To this end, this paper proposes an original Temporal PAth-based Reasoning (TPAR) model for both the interpolation and extrapolation reasoning. TPAR performs a neural-driven symbolic reasoning fashion that is robust to ambiguous and noisy temporal data and with fine interpretability as well. Comprehensive experiments show that TPAR outperforms SOTA methods on the link prediction task for both the interpolation and the extrapolation settings. A novel pipeline experimental setting is designed to evaluate the performances of SOTA combinations and the proposed TPAR towards interpolation and extrapolation reasoning. More diverse experiments are conducted to show the robustness and interpretability of TPAR.

## 1 Introduction

Knowledge graph (KG) is a semantic network that represents real-world facts in a structured way using entities and relations (Bollacker et al., 2008; Song et al., 2021; Zhao et al., 2021). Typically, a fact is represented by a triple (s, r, o) in KG, consisting of a subject entity s, an object entity o, and a relation r between s and o.

In the real world, many facts are closely associated with a particular time interval. For example, the fact “Barack Obama is the president ofUSA”

is valid for the time period of 2009 January 20th - 2017 January 20th and the fact “Donald Trump is the president of $U S A ^ { \prime \prime }$ is only valid for the following four years. To represent such time-sensitive facts, Temporal Knowledge Graphs (TKGs) have recently gained significant attention from both academic and industrial communities. Specifically, TKGs extend static KGs by incorporating the temporal information t into fact triples, represented as a quadruple $( s , r , o , t )$ , which allows for modelling the temporal dependencies and evolution of knowledge over time, being crucial for reasoning time-evolving facts in applications such as financial forecasting, social networks, and healthcare.

TKG reasoning infers new knowledge with timesensitive facts in existing TKGs, which generally has two settings: the interpolation reasoning (Xu et al., 2020; Lacroix et al., 2020; Xu et al., 2021; Chen et al., 2022) and the extrapolation reasoning (Jin et al., 2020; Sun et al., 2021; Liu et al., 2022). Given a temporal knowledge graph with facts from time $t _ { 0 }$ to time $t _ { T } ,$ the interpolation reasoning infers missing facts at any time in history $( t _ { 0 } \leq t \leq t _ { T } )$ and the extrapolation reasoning attempt to predict unknown facts that may occur in the future $( t > t _ { T } )$ .

Many approaches have been proposed to tackle the TKG reasoning problem, however, these two reasoning tasks are tackled in totally different and incompatible manners. On the one hand, interpolation methods de-emphasize the temporal correlations among fact sequences while training, thus it’s difficult to cope with the challenges of invisible timestamps and invisible entities in extrapolation due to their poor inductive reasoning ability (Sun et al., 2021). On the other hand, most stateof-the-art (SOTA) extrapolation solutions require a strict chronological order of data during training. As a result, they can only predict unknown future facts, but they could hardly infer missing historical facts which are crucial for completing the overall knowledge landscape and providing more clues for predicting accurate future events. These limit the practicability of TKG applications as almost all of the existing TKG reasoning methods are designed specifically to address either one setting. Experiments with a novel pipeline setting intuitively reveal that even with the SOTA methods from both settings, the composed methods show a frustrating decrease in reasoning performance. More in-depth analysis can be found in Section 5.3 and Appendix F. Therefore, the motivation of this work is to propose a unified method that can accommodate two types of reasoning settings, enabling temporal knowledge graph reasoning to be conducted simultaneously for both the interpolation and the extrapolation.

To this end, we take inspiration from recent neural (Xu et al., 2021; Li et al., 2021) and symbolic (Sun et al., 2021; Liu et al., 2022) TKG reasoning approaches. Neural network approaches can perform effective reasoning but lack interpretation as they cannot provide explicit rules to explain the reasoning results, while symbolic reasoning approaches use logical symbols and rules to perform reasoning tasks but are not suitable for handling ambiguous and noisy data due to their strict matching and discrete logic operations used during rule searching (Zhang et al., 2021). In this paper, we propose a Temporal PAth based Reasoning (TPAR) model with a neural-symbolic fashion applicable to both the interpolation and the extrapolation TKG Reasoning. Specifically, we utilize the Bellman-Ford Shortest Path Algorithm (Ford, 1956; Bellman, 1958; Baras and Theodorakopoulos, 2010) and introduce a recursive encoding method to score the destination entities of various temporal paths, and then our TPAR performs symbolic reasoning with the help of the obtained scores. It is noticeable that the neural-driven symbolic reasoning fashion we adopted is more robust to the uncertainty data compared to traditional pure symbolic reasoning methods, and comprehensible temporal paths with fine interpretability as well. We summarize our main contributions as follows:

1. We propose an original unified Temporal PAthbased Reasoning (TPAR) model for both the interpolation and extrapolation reasoning settings. To the best of our knowledge, this is the first work to achieve the best of both worlds.

2. We develop a novel neural-driven symbolic reasoning fashion on various temporal paths to enhance both the robustness and interpretability of temporal knowledge reasoning.

3. Comprehensive experiments show that TPAR outperforms SOTA methods on the link prediction task for both the interpolation and the extrapolation settings with decent interpretability and robustness. An intriguing pipeline experiment is meticulously designed to demonstrate the strengths of TPAR in addressing the unified prediction through both settings.

## 2 Related Work

## 2.1 Static KG Reasoning

Static KG reasoning methods can be summarized into three classes: the translation models (Bordes et al., 2013; Wang et al., 2014; Lin et al., 2015), the semantic matching models (Yang et al., 2015; Trouillon et al., 2016), and the embedding-based models (Dettmers et al., 2018; Shang et al., 2019). Recently, R-GCN (Schlichtkrull et al., 2018) and CompGCN (Vashishth et al., 2020) extended GCN to relation-aware GCN for KGs.

## 2.2 TKG Reasoning

Interpolation TKG Reasoning Many interpolation TKG reasoning methods (Leblay and Chekol, 2018; Dasgupta et al., 2018; García-Durán et al., 2018; Xu et al., 2019; Lacroix et al., 2020; Jain et al., 2020; Xu et al., 2021) are extended from static KGs to TKGs. Besides, TeRo (Xu et al., 2020) represents each relation as dual complex embeddings and can handle the time intervals between relations. RotateQVS (Chen et al., 2022) represents temporal evolutions as rotations in quaternion vector space and can model various complex relational patterns. T-GAP (Jung et al., 2021) performs pathbased inference by propagating attention through the graph.

Extrapolation TKG Reasoning To predict future facts, many extrapolation TKG reasoning methods (Jin et al., 2020; Li et al., 2021; Han et al., 2021b; Zhu et al., 2021a; Sun et al., 2021; Li et al., 2022b,a) have been studied. For better interpretability, TLogic (Liu et al., 2022) adopts temporary logical rules extracted via temporary random walks, while xERTE (Han et al., 2021a) utilizes an explainable subgraph and attention mechanism for predictions. Besides, RPC (Liang et al., 2023) employs relational graph convolutional networks and gated recurrent units to mine relational correlations and periodic patterns from temporal facts.

## 2.3 Neural-Symbolic Reasoning

Neural-symbolic reasoning aims to combine the strengths of both approaches, namely the ability of symbolic reasoning to perform logical inference and the ability of neural networks to perform learning from data (Zhang et al., 2021). Generally, it can be divided into three categories: (1) Symbolicdriven neural fashion (Guo et al., 2016, 2018; Wang et al., 2019), which targets neural reasoning but leverages the logic rules to improve the embeddings. (2) Symbolic-driven probabilistic fashion (Richardson and Domingos, 2006; Qu and Tang, 2019; Raedt et al., 2007), which replaces the neural reasoning with a probabilistic framework, i.e., builds a probabilistic model to infer the answers. (3) Neural-driven symbolic fashion (Lao and Cohen, 2010; Lao et al., 2012; Neelakantan et al., 2015), which aims to infer rules by symbolic reasoning, and incorporates neural networks to deal with the uncertainty and ambiguity of data. For this work, we adopt the neural-driven symbolic fashion for our TKG reasoning.

## 3 Preliminaries

## 3.1 Problem Definition of TKG Reasoning

A Temporal knowledge graph is a collection of millions of temporal facts, expressed as ${ \mathcal { G } } \subset { \mathcal { E } } \times$ $\mathcal { R } \times \mathcal { E } \times \mathcal { T }$ . Here we use $\mathcal { E }$ to denote the set of entities,  to denote the set of relations, and $\tau$ to denote the set of time stamps. Each temporal fact in $\mathcal { G }$ is formed as a quadruple $( e _ { s } , r , e _ { o } , t )$ , where a relation $r \in \mathcal { R }$ holds between a subject entity $e _ { s } \in$ $\mathcal { E }$ and an object entity $e _ { o } \in \mathcal { E }$ at the time t. And for each quadruple $( e _ { s } , r , e _ { o } , t ) )$ ), an inverse quadruple $( e _ { o } , r ^ { - 1 } , e _ { s } , t ) )$ is also added to the dataset.

Let $\mathcal { O } = \{ ( e _ { s } , r , e _ { o } , t ) | t \in [ t _ { 0 } , T ] \}$ represent the set of known facts that we can observe, and let $[ t _ { 0 } , T ]$ denote the time interval we can access. For a query $\pmb q = ( e _ { q } , r _ { q } , ? , t _ { q } )$ , TKG Reasoning aims to infer the missing object entity given the other three elements. As we have two distinctive TKG Reasoning settings: the interpolation (inferring a query where $t _ { 0 } \le t _ { q } \le T )$ and the extrapolation (predicting facts for $t _ { q } ~ > ~ T )$ , we use a unified concept of "temporal background" to represent all known temporal facts of existing TKG.

## 3.2 Bellman-Fold Based Recursive Encoding

The Bellman-Ford Algorithm (Ford, 1956; Bellman, 1958; Baras and Theodorakopoulos, 2010) provides a recursive approach to search for the shortest path in a graph.<sup>1</sup> Since TKGs are far from complete, we do not directly copy it for TKG Reasoning, but utilize a recursive encoding method for representations.

![](images/d36f0febe38bf21e856ecc2979c65788ac16cc53160c501ff764c43c74b3203b.jpg)  
Figure 1: An illustration of Bellman-Ford-based recursive encoding.

As illustrated in Fig.1, the basic idea of our Bellman-Ford-based recursive encoding is: once the representations of the ℓ 1 iterations (the node x) starting from node u are ready, we can obtain the representations of the ℓ iterations (the node v) by combining them with the edge $( x , v )$

$$
\begin{array} { c } { h ^ { 0 } ( u , v )  \mathbb { 1 } ( u = v ) } \\ { h ^ { \ell } ( u , v )  \displaystyle ( \bigoplus _ { ( x , v ) \in \mathcal { E } ( v ) } h ^ { \ell - 1 } ( u , x ) \otimes w ( x , v ) ) \oplus h ^ { ( 0 ) } ( u , v ) , } \end{array}
$$

where $\mathbb { 1 } ( u \ : = \ : v )$ is the indicator function that outputs 1 if $u = v$ and 0 otherwise, is the multiplication operator,  is the summation operator, $\mathcal { E } ( v )$ is the set of all edges connected to the node v, and $w ( x , v )$ is the weight of the edge $( x , v )$

## 4 Methodology

In this section, we introduce a novel temporal pathbased reasoning model with a neural-driven symbolic fashion for both the interpolation and the extrapolation TKG Reasoning settings.

## 4.1 Temporal Path in TKGs

Definition 1. (Temporal link). In a TKG, each entity is viewed as a node in the graph, and a temporal link is viewed as an edge connecting two nodes which represent the subject $e _ { s }$ and the object entity $e _ { o }$ in a certain quadruple $( e _ { s } , r , e _ { o } , t )$ . Due to the heterogeneity and time constraints, each link is bonded with a relation r and a particular time t. Thus, we can denote a temporal link as ${ \bf \Pi } _ { r } ^ { t } ( e _ { s } , e _ { o } )$

Definition 2. (Temporal path). A temporal path $\mathcal { P }$ is a combination ofseveral temporal links that are sequentially connected from subject entity to object entity. We denote a temporal path as

$$
\stackrel { t _ { 1 } } { { r } _ { 1 } } ( e _ { 1 } , e _ { 2 } ) \wedge \stackrel { t _ { 2 } } { { r } _ { 2 } } ( e _ { 2 } , e _ { 3 } ) \wedge \cdot \cdot \cdot \stackrel { t _ { \ell } } { { r } _ { \ell } } ( e _ { \ell } , e _ { \ell + 1 } ) ,\tag{1}
$$

or $\textstyle \bigwedge _ { i = 1 } ^ { \ell } { t _ { i } } ( e _ { i } , e _ { i + 1 } )$ for short, where $e _ { 1 } , e _ { 2 } , \cdots e _ { l + 1 }$ are sequentially connected toform a chain, and ℓ is the path length. We take $e _ { 1 }$ and $e _ { l + 1 }$ as the origin and the destination of the path respectively, and use $\mathcal { P } _ { ( i ) }$ to denote the temporal link $_ { r _ { i } } ^ { t _ { i } } ( e _ { i } , e _ { i + 1 } )$ of the i-th step in $\mathcal { P } . ^ { 2 }$

For the extrapolation setting, a non-increasing temporal path fashion has been applied in recent works (Han et al., 2021a; Liu et al., 2022), and requires $t _ { 1 } \geq t _ { 2 } \geq \cdot \cdot \cdot t _ { \ell } \geq t _ { \ell + 1 }$ . In this paper, we are not going to follow the non-increasing setup. For a query $\pmb { q } = ( e _ { q } , r _ { q } , ? , t _ { q } )$ , we require each $t _ { i }$ in a temporal path $\mathcal { P }$ to satisfy $t _ { i } < t _ { q }$ such that we only use information from the past to predict the future, but have no restrictions to the chronological orders among $t _ { 1 } , t _ { 2 } , \cdots t _ { l } ,$ i.e. we can have $t _ { i } < t _ { i + 1 } , t _ { i } = t _ { i + 1 }$ or $t _ { i } > t _ { i + 1 }$ . This relaxation provides more temporal paths deriving from the temporal background, enabling us to utilize more information for reasoning. Further discussions for this can be found in Appendix H.

For the interpolation setting, we mainly focus on how to find the missing object entity from the given subject entity $e _ { q }$ and indeed we ignore any time constraints. For a temporal link $_ { r _ { i } } ^ { t _ { i } } ( e _ { i } , e _ { i + 1 } )$ in the temporal background, whether it is before or after $t _ { q } .$ , we can include it in a temporal path as long as it has been known to us.

## 4.2 Relative Time Encoding on Temporal Links

We adopt relative time encoding and focus on the impact of a certain temporal link in the temporal path on a query. For each temporal link $_ { r _ { i } } ^ { t _ { i } } ( e _ { i } , e _ { i + 1 } )$ , we first obtain the relative time $\Delta t _ { i } = t _ { i } - t _ { q }$ . Notice that a temporal link itself is query-independent, but the obtained relative time $\Delta t _ { i }$ is query-dependent. For the extrapolation setting, there is always an inequality $\Delta t _ { i } < 0$ since $t _ { i } < t _ { q }$ . But for the interpolation setting, $\Delta t _ { i } < 0$ $\Delta t _ { i } = 0$ and $\Delta t _ { i } > 0$ are all possible since there is no restriction on the order of $t _ { i }$ and $t _ { q }$

As mentioned by Zhu et al. (2021a), many facts have occurred repeatedly throughout history, such as economic crises that occur every seven to ten years (Korotayev and Tsirel, 2010). We pay close attention to the periodic property as well as the non-periodic property of time. We then extend the idea of periodic and non-periodic time vectors (Li et al., 2022a) to a relative time encoding fashion:

$$
\begin{array} { r } { \pmb { h } _ { t _ { i } } ^ { p } = \mathrm { s i n } ( \omega _ { p } \Delta t _ { i } + \varphi _ { p } ) } \\ { \pmb { h } _ { t _ { i } } ^ { n p } = \omega _ { n p } \Delta t _ { i } + \varphi _ { n p } , } \end{array}\tag{2}
$$

where $\omega _ { p } , \varphi _ { p } , \omega _ { n p } , \varphi _ { n p }$ are learnable parameters, sin() denotes the sine function, and the obtained $h _ { t _ { i } } ^ { p }$ and $h _ { t _ { i } } ^ { n p }$ are the periodic and the non-periodic relative time vectors, respectively. For the periodic vector $h _ { t _ { i } } ^ { p }$ , the frequency $\omega _ { p }$ and the phase shift $\varphi _ { p }$ reflect the periodic property. For the non-periodic vector $h _ { t _ { i } } ^ { n p } , \omega _ { n p }$ acts as a velocity vector (Han et al., 2020) indicating the temporal evolving dynamics.

Finally, we combine both periodic and nonperiodic properties to design our relative time embedding as:

$$
h _ { t _ { i } } = h _ { t _ { i } } ^ { p } + h _ { t _ { i } } ^ { n p } ,\tag{3}
$$

where we use the same embedding dimension d for $h _ { t _ { i } } ^ { p } , h _ { t _ { i } } ^ { n p }$ and $h _ { t _ { i } }$ .

## 4.3 Recursive Encoding on Temporal Paths

In Section 3.2, we have mentioned the idea of Bellman-Ford-based recursive encoding. Now, we start from the query and introduce our recursive encoding on temporal paths for TKG reasoning.

Basic Design Our recursive encoding on temporal paths aims to encode the destination entities of various temporal paths. And the basic idea is: Once the representations of the destination entities of all the (ℓ 1)-length paths are ready, we can combine them with the link $\mathcal { P } _ { ( l ) }$ to encode the destination entities of the ℓ-length paths.

For a query $\textbf { \em q } = ~ ( e _ { q } , r _ { q } , ? , t _ { q } )$ , our recursive encoding is query-specific, and we start with the query entity $e _ { q }$ . Initially, we can only reach the original entity $\boldsymbol { e } _ { \boldsymbol { q } } .$ so we initialize $h _ { e _ { q } } ^ { 0 } = \mathbf { 0 }$ and the entity set $\mathcal { E } _ { e _ { q } } ^ { 0 } = \{ e _ { q } \}$ . In the ℓ-th step, we collect all the ℓ-length temporal paths from temporal background based on $e _ { q }$ , and update $\mathcal { E } _ { e _ { q } } ^ { \ell }$ with the destination entities. We utilize $\mathcal { D } ( \mathcal { P } _ { e _ { q } } ) ^ { }$ to denote the destination entity set of the temporal paths $\mathcal { P } _ { e _ { q } }$ , and $\mathcal { P } _ { e _ { q } , ( \ell ) }$ to denote the set of links of the ℓ-th step in $\mathcal { P } _ { e _ { q } } .$ . Thus, we actually have $\mathscr { E } _ { e _ { q } } ^ { \ell } = \mathscr { E } _ { e _ { q } } ^ { \ell - 1 } \cup \mathscr { D } ( \mathcal { P } _ { e _ { q } } )$ for $\ell > 1$

Then, the message is passed along temporal paths $\mathcal { P } _ { e _ { q } }$ from $e _ { s }$ to $e _ { o } ,$ where ${ \bf \Phi } _ { r } ^ { t } ( e _ { s } , e _ { o } ) \in \mathcal { P } _ { e _ { q } , ( l ) }$ And we can get the representations for all $e _ { o } \in$ $\mathcal { D } ( \mathcal { P } _ { e _ { q } } )$ for $\ell = 1 , 2 , \cdots L$ by:

$$
\begin{array} { r } { h _ { e _ { o } } ^ { \ell } = \delta \Big ( W ^ { \ell } \sum _ { \mathcal { P } _ { e _ { q } , ( \ell ) } } \phi \big ( h _ { e _ { s } } ^ { \ell - 1 } , h _ { r } ^ { \ell } , h _ { t } \big ) \Big ) , } \end{array}\tag{4}
$$

where δ denotes an activation function, ${ \bf W } ^ { \ell } { \bf \Sigma } \in \mathbf { \Sigma }$ $\mathbb { R } ^ { d \times d }$ is a learnable parameter, $\phi ( )$ denotes a function to compute and process the messages, $h _ { r } ^ { \ell }$ is the embedding for relation r in the ℓ-th step, and $h _ { t }$ is a result of the relative time encoding that we have introduced in Section 4.2. The proposed recursive encoding algorithm is shown in Algorithm 1.<sup>3</sup>

```latex
Algorithm 1 The Proposed Recursive Encoding Algo
rithm
Input: A temporal knowledge graph $G ,$ a query ${ \textbf { \em q } } =$
$( e _ { q } , r _ { q } , ? , \bar { t _ { q } } )$ , the maximum length L
Output: The query-specific entity representations
1: initialize $\mathbf { \nabla } \mathbf { \dot { \mathbf { h } } } _ { e _ { q } } ^ { \mathrm { 0 } } = \mathbf { 0 } ^ { \mathrm { : } }$ and the entity set $\mathcal { E } _ { e _ { q } } ^ { 0 } = \{ e _ { q } \} $
2: for $\ell = 1 , 2 , \cdots L$ do
3: collect all the ℓ-length temporal paths and update
$\mathcal { E } _ { e _ { q } } ^ { \ell }$ with the destination entity set $\mathsf { \widehat { D } } ( \mathcal { P } _ { e _ { q } } ) ;$
message passing along temporal paths $\mathcal { P } _ { e _ { q } }$ from
$e _ { s } \in \varDelta \bigl ( \varDelta _ { e _ { q } , ( \ell - 1 ) } \bigr ) \mathrm { t o } e _ { o } \in \varDelta \bigl ( \varDelta _ { e _ { q } , ( \ell ) } \bigr ) \colon$
$\begin{array} { r } { h _ { e _ { o } } ^ { \ell } = \delta \Big ( W ^ { \ell } \sum _ { \mathcal { P } _ { e _ { q } , ( l ) } } \phi \big ( h _ { e _ { s } } ^ { \ell - 1 } , h _ { r } ^ { \ell } , h _ { t } \big ) \Big ) } \end{array}$
5: end for
6: assign $h _ { e _ { a } } ^ { L } = \mathbf { 0 }$ for all $e _ { a } \notin \mathcal { E } _ { e _ { q } } ^ { L } ;$
7: return $h _ { e _ { a } } ^ { L }$ for all $e _ { a } \in \mathcal { E } .$
```

Message Passing We then give a detailed description of the message passing along temporal paths from $e _ { s }$ to $e _ { o }$ .

Given a quadruple $( e _ { s } , r , e _ { o } , t )$ in temporal ground, we can transform it to a temporal link ${ \bf \Pi } _ { r } ^ { t } ( e _ { s } , e _ { o } )$ . We take the message as a combination of the query-specific representation of $e _ { s }$ learned in the (ℓ  1)-th recursive encoding step, the representation of $^ { r , }$ and the representation of t:

$$
M _ { ( e _ { s } , r , e _ { o } , t ) | q } = h _ { e _ { s } } ^ { \ell - 1 } + h _ { r } ^ { \ell } + h _ { t } ,\tag{5}
$$

Note that we use the same embedding dimension d for ${ h } _ { e _ { s } } ^ { \ell - 1 } , { h } _ { r } ^ { \ell } , { h } _ { r _ { q } } ^ { \ell }$ and $h _ { t }$

Then, we utilize an attention-based mechanism to weigh the message. Inspired by graph attention networks (Velickovic et al., 2018), we first take a concatenation operation on ${ h } _ { e _ { s } } ^ { \ell - 1 } , { h } _ { r } ^ { \ell } , { h } _ { r _ { q } } ^ { \ell }$ and $\boldsymbol { h } _ { t }$ to get

$$
U _ { ( e _ { s } , r , e _ { o } , t ) | q } = c o n c a t [ { h _ { e _ { s } } ^ { \ell - 1 } ( e _ { q } , r _ { q } , t _ { q } ) } | | { h _ { r } ^ { \ell } } | | { h _ { r _ { q } } ^ { \ell } } | | { h _ { t } } ] ,\tag{6}
$$

where $U _ { ( e _ { s } , r , e _ { o } , t ) | \mathbf { q } } \in \mathbb { R } ^ { 4 d }$ is the concatenated result. Then, the attention weight for the message can be defined as

$$
\alpha _ { ( e _ { s } , r , e _ { o } , t ) | q } ^ { \ell } = \sigma \Big ( ( w _ { \alpha } ^ { \ell } ) ^ { T } R e L U ( W _ { \alpha } ^ { \ell } \cdot U _ { ( e _ { s } , r , e _ { o } , t ) | q } ) \Big ) ,\tag{7}
$$

where ${ \pmb w } _ { \alpha } ^ { \ell } \in \mathbb { R } ^ { d _ { a } }$ $W _ { \alpha } ^ { \ell } \in \mathbb { R } ^ { d _ { a } \times 4 d }$ are learnable parameters for attention $( d _ { a }$ is the dimension for attention). Just like (Zhang and Yao, 2022), we use the Sigmoid function σ rather than the Softmax function to ensure multiple temporal links can be selected in the same recursive step.

Aggregation With messages and the weights obtained, we take an aggregation operation on messages along different temporal links in various temporal paths, by means of a weighted sum of messages. For the ℓ-th recursive step, our encoded representation is specified as

$$
\begin{array} { r } { \quad \pmb { h } _ { e _ { o } } ^ { \ell } = \delta \Big ( \pmb { W } ^ { \ell } \sum _ { \mathcal { P } _ { e _ { q } , ( \ell ) } } \alpha _ { ( e _ { s } , r , e _ { o } , t ) | q } ^ { \ell } \cdot M _ { ( e _ { s } , r , e _ { o } , t ) | q } \Big ) . } \end{array}\tag{8}
$$

So far, we have provided a detailed version of our recursive encoding function (also see a general version in Equation 4) for each recursive step $\ell = 1 , 2 , \cdots L$ . After $L$ recursive steps, the representation of each entity inside various L-length temporal paths (starting from $\boldsymbol { e } _ { q } )$ can be learned step by step. We assign $h _ { e _ { a } } ^ { L } ( e _ { q } , r _ { q } , t _ { q } ) = 0$ for all $e _ { a } \notin \mathcal { E } _ { e _ { q } } ^ { L }$ . For each entity $e _ { a } \in \mathcal { E } ,$ we finally get its representation $h _ { e _ { a } } ^ { L }$ after L-length recursive encoding process.

## 4.4 Reasoning on Temporal Paths

Then, we can leverage the learned representations to measure the quality of temporal paths to reason.

Neural based Scoring To accomplish neuralsymbolic reasoning, we take a neural-based scoring first and use a simple scoring function to compute the likelihood for each candidate answer entity $e _ { a } \colon$

$$
f \left( \pmb { q } , e _ { a } \right) = \pmb { w } ^ { T } h _ { e _ { a } } ^ { L } ( e _ { q } , r _ { q } , t _ { q } ) ,\tag{9}
$$

where $\pmb { w } \in \mathbb { R } ^ { d }$ is a parameter for scoring.

And for training loss, we use a multi-class logloss function to train the neural networks for our TKG Reasoning, which has been proven effective (Lacroix et al., 2018; Zhang and Yao, 2022):

$$
\mathcal { L } = \sum _ { \left( { \pmb q } , e _ { a } \right) \in { \mathcal { T } } _ { \mathrm { t r a i n } } } \left( - f \left( { \pmb q } , e _ { a } \right) + \log \left( \sum _ { \forall e \in \mathcal { E } } e ^ { f \left( { \pmb q } , e \right) } \right) \right) ,\tag{10}
$$

where $\mathcal { T } _ { \mathrm { t r a i n } }$ is set of training queries, and $\textstyle \sum \forall e \in { \mathcal { E } }$ is for all the quadruples w.r.t. the same query q.

Symbolic Reasoning As for our symbolic reasoning, we target both the suitable temporal path and its destination entity. Using $m _ { i }$ to denote the i-th intermediary entity, our reasoning can be expressed as

$$
\begin{array} { r l } & { t _ { 1 } ^ { t _ { 1 } } ( e _ { q } , m _ { 1 } ) \wedge \ o { \Lambda } _ { r _ { 2 } } ^ { t _ { 2 } } ( m _ { 1 } , m _ { 2 } ) \wedge \cdot \cdot \cdot \dot { \bar { r } } _ { L } ^ { t _ { L } } ( m _ { L - 1 } , e _ { a } ) } \\ & { \qquad \to \stackrel { t _ { q } } { r _ { q } } ( e _ { q } , e _ { a } ) , } \end{array}\tag{11}
$$

where the left part of  denotes a temporal path composed of $L$ sequentially connected temporal links, and the right part is the link we aim to infer.

We use $A _ { e _ { a } } ^ { \ell } \in \bar { \{ 0 , 1 \} } ^ { N _ { \ell } \times N _ { \ell } }$ to denote the adjacent matrix of the ℓ-th step in the set of L-length paths $\mathcal { P } _ { e _ { q } }$ starting from $e _ { q } , \mathrm { i . e . } \ \mathcal { P } _ { e _ { q } , ( \ell ) }$ , where $N _ { \ell }$ is the number of entities involved for $\ i = 1 , 2 , \cdots L$ $\mathrm { I f } _ { r _ { 1 } } ^ { t _ { 1 } } ( e _ { q } , m _ { 1 } ) \wedge _ { r _ { 2 } } ^ { t _ { 2 } } ( m _ { 1 } , m _ { 2 } ) \wedge \cdot \cdot \cdot _ { r _ { L } } ^ { t _ { L } } ( m _ { L - 1 } , e _ { a } ) $ $\mathbf { \Psi } _ { r _ { q } } ^ { t _ { q } } ( e _ { q } , e _ { a } )$ holds, ideally there exists a suitable path ${ \mathcal { P } } ^ { \star }$

$$
\mathcal { P } ^ { \star } = \beta ^ { 1 } A _ { e _ { q } } ^ { 1 } \otimes \beta ^ { 2 } A _ { e _ { q } } ^ { 2 } \cdot \cdot \cdot \otimes \beta ^ { L } A _ { e _ { q } } ^ { L } .\tag{12}
$$

where $\beta ^ { \ell } \in \{ 0 , 1 \} ^ { N _ { \ell } \times N _ { \ell } }$ is the choice of links for step ℓ.

From Equation 6 and 7, we can know the learned $\alpha _ { ( e _ { s } , r , e _ { o } , t ) | q } ^ { \ell } = M L P ^ { \ell } ( e _ { s } , r , e _ { q } , r _ { q } , \Delta t )$ , and abbreviate it as $\alpha ^ { \ell } .$ . Our recursive encoding provides

$$
\underset { \bf \Theta } { \arg \operatorname* { m a x } } \ S c o r e ( \mathcal { P } ) = \alpha ^ { 1 } { \cal A } _ { e _ { q } } ^ { 1 } \otimes \alpha ^ { 2 } { \cal A } _ { e _ { q } } ^ { 2 } \cdot \cdot \cdot \otimes \alpha ^ { L } { \cal A } _ { e _ { q } } ^ { L } ,\tag{13}
$$

where $S c o r e ( \mathcal { P } )$ is the score of the path P obtained by Equation 9. Based on universal approximation theorem (Hornik, 1991), there exists a set of parameters Θ that can learn $\alpha ^ { \ell } \simeq \beta ^ { \ell }$ , to make $\mathcal { P } \simeq \mathcal { P } ^ { \star }$ Note that multiple temporal paths may lead to the same destination. In such cases, the score can be regarded as a comprehensive measure of the various reasoning paths. Case-based analysis on this can be found in Section 5.4.

To summarize, our temporal path-based reasoning first collects temporal paths from the temporal background, then scores the destination entities of various temporal paths by attention-based recursive encoding, and lastly takes the entity with the highest score as the reasoning answer.

## 5 Experiments

## 5.1 Evaluation Protocol

Link prediction task that aims to infer incomplete time-wise fact with a missing entity $( ( s , r , ? , t )$ or $( ? , r , o , t ) )$ ) is adopted to evaluate the proposed model for both interpolation and extrapolation reasoning over TKGs. During inference, we follow the same procedure of Xu et al. (2020) to generate candidates. For a test sample $( s , r , o , t )$ , we first generate candidate quadruples set $C = \{ ( s , r , \overline { { o } } , t )$ $\overline { { o } } \in \mathcal { E } \} \cup \{ ( \overline { { s } } , r , o , t ) : \overline { { s } } \in \mathcal { E } \}$ by replacing s or o with all possible entities, and then rank all the quadruples by their scores (Equation 9). We use the time-wise filtered setting (Xu et al., 2019; Goel et al., 2020) to report the experimental results.

We use ICEWS14, ICEWS05-15 (García-Durán et al., 2018), YAGO11k (Dasgupta et al., 2018) and WIKIDATA12k (Dasgupta et al., 2018) datasets for interpolation reasoning evaluation, and use ICEWS14 (García-Durán et al., 2018), ICEWS18 (Jin et al., 2020), YAGO (Mahdisoltani et al., 2015) and WIKI (Leblay and Chekol, 2018) for extrapolation.<sup>4</sup> The performance is reported on the standard evaluation metrics: the proportion of correct triples ranked in the top 1, 3 and 10 (Hits@1, Hits@3, and Hits@10), and Mean Reciprocal Rank (MRR). All of them are the higher the better. For all experiments, we report averaged results across 5 runs, and we omit the variance as it is generally low.

## 5.2 Main Results

## 5.2.1 Interpolation Reasoning

Due to space limitations, we share the experimental details in Appendix D. Table 1 shows the results of the interpolation TKG Reasoning on link prediction over four experimented datasets where the proposed TPAR continuously outperforms all baselines across all metrics. We specifically compare our proposed TPAR with another GNN-based method, T-GAP (Jung et al., 2021), which samples a subgraph from the whole TKG for each node. In contrast, our TPAR recursively traverses the entire graph based on previously visited nodes to collect temporal paths. Our experimental results show that TPAR outperforms T-GAP across all evaluation metrics on the ICEWS14 and ICEWS05-15 datasets. This can be attributed to TPAR’s ability to capture the temporal information of the entire graph and generate longer paths, which leads to more accurate and comprehensive TKG reasoning.

## 5.2.2 Extrapolation Reasoning

Experimental details for Extrapolation reasoning are shown in Appendix E. Table 2 presents the results of the extrapolation TKG reasoning on link prediction across four different datasets. Notably, we compare TPAR with two SOTA methods, namely, the neural network-based RPC (Liang et al., 2023) and the symbolic-based TLogic (Liu et al., 2022). Our results show that TPAR outperforms all baselines, underscoring the benefits of combining neural and symbolic-based approaches for TKG reasoning.

<table><tr><td rowspan="2">Interpolation</td><td colspan="4">ICEWS14</td><td colspan="4">YAG011k</td><td colspan="4">ICEWS05-15</td><td colspan="4">WIKIDATA12k</td></tr><tr><td>Hits@1</td><td>Hits@3</td><td>Hits@10</td><td>MRR</td><td>Hits@1</td><td>Hits@3</td><td>Hits@10</td><td>MRR</td><td>Hits@1</td><td>Hits@3</td><td>Hits@10</td><td>MRR</td><td>Hits@1</td><td>Hits@3</td><td>Hits@10</td><td>MRR</td></tr><tr><td>TTransE</td><td>7.4</td><td>-</td><td>60.1</td><td>25.5</td><td>2.0</td><td>15.0</td><td>25.1</td><td>10.8</td><td>8.4</td><td>=</td><td>61.6</td><td>27.1</td><td>9.6</td><td>18.4</td><td>32.9</td><td>17.2</td></tr><tr><td>HyTE</td><td>10.8</td><td>41.6</td><td>65.5</td><td>29.7</td><td>1.5</td><td>14.3</td><td>27.2</td><td>10.5</td><td>11.6</td><td>44.5</td><td>68.1</td><td>31.6</td><td>9.8</td><td>19.7</td><td>33.3</td><td>18.0</td></tr><tr><td>TA-DistMult</td><td>36.3</td><td></td><td>68.6</td><td>47.7</td><td>10.3</td><td>17.1</td><td>29.2</td><td>16.1</td><td>34.6</td><td></td><td>72.8</td><td>47.4</td><td>12.2</td><td>23.2</td><td>44.7</td><td>21.8</td></tr><tr><td>ATiSE</td><td>43.6</td><td>62.9</td><td>75.0</td><td>55.0</td><td>11.0</td><td>17.1</td><td>28.8</td><td>17.0</td><td>37.8</td><td>60.6</td><td>79.4</td><td>51.9</td><td>17.5</td><td>31.7</td><td>48.1</td><td>28.0</td></tr><tr><td>TeRo</td><td>46.8</td><td>62.1</td><td>73.2</td><td>56.2</td><td>12.1</td><td>19.7</td><td>31.9</td><td>18.7</td><td>46.9</td><td>66.8</td><td>79.5</td><td>58.6</td><td>19.8</td><td>32.9</td><td>50.7</td><td>29.9</td></tr><tr><td>TComplEx</td><td>53</td><td>66</td><td>77</td><td>61</td><td>12.7</td><td>18.3</td><td>30.7</td><td>18.5</td><td>59</td><td>71</td><td>80</td><td>66</td><td>23.3</td><td>35.7</td><td>53.9</td><td>33.1</td></tr><tr><td>T-GAP</td><td>50.9</td><td>67.7</td><td>79.0</td><td>61.0</td><td></td><td></td><td></td><td>-</td><td>56.8</td><td>74.3</td><td>84.5</td><td>67.0</td><td></td><td>=</td><td></td><td>-</td></tr><tr><td>TELM</td><td>54.5</td><td>67.3</td><td>77.4</td><td>62.5</td><td>12.9</td><td>19.4</td><td>32.1</td><td>19.1</td><td>59.9</td><td>72.8</td><td>82.3</td><td>67.8</td><td>23.1</td><td>36.0</td><td>54.2</td><td>33.2</td></tr><tr><td>RotateQVS</td><td>50.7</td><td>64.2</td><td>75.4</td><td>59.1</td><td>12.4</td><td>19.9</td><td>32.3</td><td>18.9</td><td>52.9</td><td>70.9</td><td>81.3</td><td>63.3</td><td>20.1</td><td>32.7</td><td>51.5</td><td>30.2</td></tr><tr><td>TGoemE++</td><td>54.6</td><td>68.0</td><td>78.0</td><td>62.9</td><td>13.0</td><td>19.6</td><td>32.6</td><td>19.5</td><td>60.5</td><td>73.6</td><td>83.3</td><td>68.6</td><td>23.2</td><td>36.2</td><td>54.6</td><td>33.3</td></tr><tr><td>TPAR (Ours)</td><td>57.03</td><td>69.74</td><td>80.41</td><td>65.07</td><td>17.35</td><td>25.14</td><td>37.42</td><td>24.12</td><td>62.17</td><td>75.29</td><td>85.86</td><td>69.33</td><td>25.05</td><td>38.68</td><td>54.79</td><td>34.89</td></tr></table>

Table 1: Interpolation TKG Reasoning results (in percentage) on link prediction over four experimented datasets.
<table><tr><td rowspan="2">Extrapolation</td><td colspan="4">ICEWS14</td><td colspan="4">YAGO</td><td colspan="4">ICEWS18</td><td colspan="4">WIKI</td></tr><tr><td>Hits@1</td><td>Hits@3</td><td>Hits@10</td><td>MRR</td><td>Hits@1</td><td>Hits@3</td><td>Hits@10</td><td>MRR</td><td>Hits@1</td><td>Hits@3</td><td>Hits@10</td><td>MRR</td><td>Hits@1</td><td>Hits@3</td><td>Hits@10</td><td>MRR</td></tr><tr><td>RE-NET</td><td>30.11</td><td>44.02</td><td>58.21</td><td>39.86</td><td>58.59</td><td>71.48</td><td>86.84</td><td>66.93</td><td>19.73</td><td>32.55</td><td>48.46</td><td>29.78</td><td>50.01</td><td>61.23</td><td>73.57</td><td>58.32</td></tr><tr><td>CyGNeT</td><td>27.43</td><td>42.63</td><td>57.90</td><td>37.65</td><td>58.97</td><td>76.80</td><td>86.98</td><td>68.98</td><td>17.21</td><td>30.97</td><td>46.85</td><td>27.12</td><td>47.89</td><td>66.44</td><td>78.70</td><td>58.78</td></tr><tr><td>TANGO</td><td></td><td></td><td></td><td></td><td>60.04</td><td>65.19</td><td>68.79</td><td>63.34</td><td>18.68</td><td>30.86</td><td>44.94</td><td>27.56</td><td>51.52</td><td>53.84</td><td>55.46</td><td>53.04</td></tr><tr><td>xERTE</td><td>32.70</td><td>45.67</td><td>57.30</td><td>40.79</td><td>80.09</td><td>88.02</td><td>89.78</td><td>84.19</td><td>21.03</td><td>33.51</td><td>46.48</td><td>29.31</td><td>69.05</td><td>78.03</td><td>79.73</td><td>73.60</td></tr><tr><td>RE-GCN</td><td>31.63</td><td>47.20</td><td>61.65</td><td>42.00</td><td>78.83</td><td>84.27</td><td>88.58</td><td>82.30</td><td>22.39</td><td>36.79</td><td>52.68</td><td>32.62</td><td>74.50</td><td>81.59</td><td>84.70</td><td>78.53</td></tr><tr><td>TITer</td><td>32.76</td><td>46.46</td><td>58.44</td><td>41.73</td><td>80.09</td><td>89.96</td><td>90.27</td><td>87.47</td><td>22.05</td><td>33.46</td><td>44.83</td><td>29.98</td><td>71.70</td><td>75.41</td><td>76.96</td><td>73.91</td></tr><tr><td>CEN</td><td>32.08</td><td>47.46</td><td>61.31</td><td>42.20</td><td></td><td>-</td><td>=</td><td>-</td><td>21.70</td><td>35.44</td><td>50.59</td><td>31.50</td><td>75.05</td><td>81.90</td><td>84.90</td><td>78.93</td></tr><tr><td>TLogic</td><td>33.56</td><td>48.27</td><td>61.23</td><td>43.04</td><td></td><td></td><td></td><td>-</td><td>20.54</td><td>33.95</td><td>48.53</td><td>29.82</td><td></td><td></td><td></td><td></td></tr><tr><td>TIRGN</td><td>33.83</td><td>48.95</td><td>63.84</td><td>44.04</td><td>84.34</td><td>91.37</td><td>92.92</td><td>87.95</td><td>23.19</td><td>37.99</td><td>54.22</td><td>33.66</td><td>77.77</td><td>85.12</td><td>87.08</td><td>81.65</td></tr><tr><td>RPC</td><td>34.87</td><td>49.80</td><td>65.08</td><td>44.55</td><td>85.10</td><td>92.57</td><td>94.04</td><td>88.87</td><td>24.34</td><td>38.74</td><td>55.89</td><td>34.91</td><td>76.28</td><td>85.43</td><td>88.71</td><td>81.18</td></tr><tr><td>TPAR (Ours)</td><td>36.88</td><td>52.28</td><td>65.89</td><td>46.89</td><td>89.67</td><td>92.93</td><td>94.60</td><td>91.53</td><td>26.58</td><td>39.27</td><td>56.94</td><td>35.76</td><td>79.85</td><td>87.04</td><td>88.96</td><td>83.20</td></tr></table>

Table 2: Extrapolation TKG Reasoning results (in percentage) on link prediction over four experimented datasets.

## 5.3 Analysis on Pipeline Setting

To test the hypothesis that completing missing knowledge about the past can enhance the accuracy of predicting future knowledge, we design a novel experimental pipeline. Building upon the facts within the known time range, we sample a ratio of facts to train interpolation, while the other unsampled facts have either the subject entity or the object entity masked, forming incomplete quadruples. Specifically, we have three settings: set the ratio to 60% for Setting A, 70% for Setting B, and 80% for Setting C.<sup>5</sup> The lower the ratio, the more incomplete facts need to be interpolated first.

The pipeline involves first completing these incomplete quadruples through interpolation, and subsequently predicting future events. Baselines: Since there are currently no existing methods capable of both interpolation and extrapolation, we assemble SOTA interpolation and extrapolation methods<sup>6</sup>, and get two combinations: a) TELM and TLogic, b) RotateQVS and TIRGN.

<table><tr><td rowspan=2 colspan=1>Interpolation</td><td rowspan=2 colspan=1>Extrapolation</td><td rowspan=1 colspan=3>Setting</td></tr><tr><td rowspan=1 colspan=1>A (Ratio: 60%)</td><td rowspan=1 colspan=1>B (Ratio: 70%)</td><td rowspan=1 colspan=1>C (Ratio: 80%)</td></tr><tr><td rowspan=1 colspan=1>TELM</td><td rowspan=1 colspan=1>TLogic</td><td rowspan=1 colspan=1>40.88</td><td rowspan=1 colspan=1>41.55</td><td rowspan=1 colspan=1>42.09</td></tr><tr><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>TLogic</td><td rowspan=1 colspan=1>41.85</td><td rowspan=1 colspan=1>41.88</td><td rowspan=1 colspan=1>42.18</td></tr><tr><td rowspan=1 colspan=1>RotateQVS</td><td rowspan=1 colspan=1>TIRGN</td><td rowspan=1 colspan=1>40.86</td><td rowspan=1 colspan=1>42.35</td><td rowspan=1 colspan=1>42.78</td></tr><tr><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>TIRGN</td><td rowspan=1 colspan=1>42.53</td><td rowspan=1 colspan=1>43.68</td><td rowspan=1 colspan=1>43.69</td></tr><tr><td rowspan=1 colspan=1>TPAR</td><td rowspan=1 colspan=1>TPAR</td><td rowspan=1 colspan=1>44.56</td><td rowspan=1 colspan=1>46.07</td><td rowspan=1 colspan=1>46.66</td></tr><tr><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>TPAR</td><td rowspan=1 colspan=1>43.86</td><td rowspan=1 colspan=1>45.49</td><td rowspan=1 colspan=1>36.36</td></tr></table>

Table 3: MRR performance of the pipeline setting on ICEWS 14.

The experimental results in Table 3<sup>7</sup> show two key observations: a) For both of the SOTA combination baselines, employing the pipeline (interpolation followed by extrapolation) leads to a reduction in extrapolation performance compared to straightforward extrapolation. And this reduction in the effect is gradually alleviated as the ratio increases from 60% to 80%. We believe that this could be attributed to the fact that the newly acquired knowledge from interpolation may contain errors, which have the potential to propagate and amplify in subsequent extrapolation. As the amount of newly acquired knowledge decreases, the likelihood of introducing errors into the system also diminishes.

![](images/ed622f73afd978a12f799357c224aeaa83f9ca03fede45a084928f1f7fb3b4dd.jpg)  
(a) Interpolation for the maximum length L = 2.

![](images/e7d90fdb36ac570599bb2c698bf952cf1931adb30eb423ab64314b761e80487d.jpg)  
(b) Extrapolation for the maximum length L = 3.  
Figure 2: Path interpretations of TKG reasoning on ICEWS14 test set. Dashed lines mean inverse relations.

b) In all three settings, our TPAR with pipeline outperforms the straightforward extrapolation results, which demonstrates the superiority of our TPAR model to effectively integrate interpolation and extrapolation as a unified task for TKG reasoning.

## 5.4 Case Study: Path Interpretations

Benefiting from the neural-symbolic fashion, we can interpret the reasoning results through temporal paths. Following local interpretation methods (Zeiler and Fergus, 2014; Zhu et al., 2021b), we approximate the local landscape of our TPAR with a linear model over the set of all paths, i.e., 1st-order Taylor polynomial. Thus, the importance of a path can be defined as its weight in the linear model and can be computed by the partial derivative of the prediction w.r.t. the path. Formally, top-k temporal path interpretations for $f ( \pmb { q } , e _ { a } )$ are defined as

$$
P _ { 1 } , P _ { 2 } , . . . , P _ { k } = \underset { P \in \mathcal { P } _ { e _ { q } e _ { a } } } { t o p - k } \frac { \partial f ( \pmb { q } , e _ { a } ) } { \partial P } ,\tag{14}
$$

where $\mathcal { P } _ { e _ { q } e _ { a } }$ denotes the set of all temporal paths starting from $e _ { q }$ to $e _ { a }$

Fig. 2 illustrates the path interpretations of TKG reasoning for both two settings on the ICEWS14 test set. While users may have different insights towards the visualization, here are our understandings: 1) In Fig. 2(a), for the interpolation setting, when we need to reason about the fact that happened on $t _ { q } ~ = 2 0 1 4  – 0 2 – 1 4$ , we can use the path formed by the links before or after $t _ { q } . ~ 2 )$ For the extrapolation of Fig. 2(b), we do not strictly impose a chronological order constraint on the links in the path, ensuring that the paths used in our reasoning can be as rich as possible. 3) Multiple different temporal paths may lead to the same reasoning destination, and the explanation is that a reasoning result can have multiple evidence chains.

![](images/ad4a0a816b05026e120e6c0dccc9d3e4606aaf247355d172bc727962c0fc7f4a.jpg)  
(a) Interpolation

![](images/4ea82c71fa939eab1697f296486d2817c52a28a862a2d2bc71e795ce7a3da6c8.jpg)  
(b) Extrapolation  
Figure 3: Performance w.r.t. sparsity ratio on ICEWS14.

In addition, we conduct analyses on path lengths, and the contents can be found in Appendix G.

## 5.5 Robustness Analysis

Fig.3 illustrates a performance comparison under varying degrees of data sparsity on the ICEWS14 dataset. We compare our TPAR model with RotateQVS (Chen et al., 2022) and TELM (Xu et al., 2021) in the interpolation setting, while TLogic (Liu et al., 2022) and TIRGN (Li et al., 2022a) are taken as extrapolation baselines. Our TPAR demonstrates superior performance compared to the other models across a wide range of data sparsity levels for both interpolation and extrapolation settings. This highlights its exceptional robustness and ability to handle sparse data effectively.

Furthermore, we conduct an analysis on the chronological order in Appendix H, which also demonstrates robustness to some extent.

## 6 Conclusion

We propose a temporal path-based reasoning (TPAR) model with a neural-symbolic fashion that can be applied to both interpolation and extrapolation TKG Reasoning settings in this paper. Introducing the Bellman-Ford Algorithm and a recursive encoding method to score the destination entities of various temporal paths, TPAR performs robustly on uncertain temporal knowledge with fine interpretability. Comprehensive experiments demonstrate the superiority of our proposed model for both settings. A well-designed pipeline experiment further verified that TPAR is capable of integrating two settings for a unified TKG reasoning.

## 7 Limitations

We identify that there may be some possible limitations in this study. First, our reasoning results are dependent on the embeddings learned by the proposed (Bellman-Ford-based) recursive encoding algorithm, where the selection of hyperparameters may have a significant impact on the reasoning results. Further, we can explore additional neuralsymbolic reasoning approaches, expanding beyond the current neural-driven symbolic fashion. Second, the reasoning process relies on attention mechanisms, but it has been pointed out (Sui et al., 2022) that existing attention-based methods are prone to visit the noncausal features as the shortcut to predictions. In future work, we will try to focus on researching causal attention mechanisms for a more solid comprehension of the complicated correlations between temporal knowledge.

## Acknowledgements

This work is particularly supported by the National Key Research and Development Program of China (No. 2022YFB3104103), and the National Natural Science Foundation of China (No. 62302507).

## References

John S. Baras and George Theodorakopoulos. 2010. Path Problems in Networks. Synthesis Lectures on Communication Networks. Morgan & Claypool Publishers.

Richard Bellman. 1958. On a routing problem. Quarterly ofApplied Mathematics, 16:87–90.

Kurt D. Bollacker, Colin Evans, Praveen Paritosh, Tim Sturge, and Jamie Taylor. 2008. Freebase: a collaboratively created graph database for structuring human knowledge. In Proceedings of the ACM SIGMOD International Conference on Management of Data, SIGMOD 2008, pages 1247–1250.

Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko. 2013. Translating embeddings for modeling multirelational data. In Advances in neural information processing systems, pages 2787–2795.

Kai Chen, Ye Wang, Yitong Li, and Aiping Li. 2022. Rotateqvs: Representing temporal information as

rotations in quaternion vector space for temporal knowledge graph completion. In Proceedings of the 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022, pages 5843– 5857. Association for Computational Linguistics.

Shib Sankar Dasgupta, Swayambhu Nath Ray, and Partha Talukdar. 2018. Hyte: Hyperplane-based temporally aware knowledge graph embedding. In Proceedings of the 2018 conference on empirical methods in natural language processing, pages 2001– 2011.

Tim Dettmers, Pasquale Minervini, Pontus Stenetorp, and Sebastian Riedel. 2018. Convolutional 2d knowledge graph embeddings. In Proceedings ofthe Thirty-Second AAAI Conference on Artificial Intelligence, pages 1811–1818.

Fredo Erxleben, Michael Günther, Markus Krötzsch, Julian Mendez, and Denny Vrandecic. 2014. Introducing wikidata to the linked data web. In The Semantic Web - ISWC 2014 - 13th International Semantic Web Conference, Riva del Garda, Italy, October 19-23, 2014. Proceedings, Part I, volume 8796 of Lecture Notes in Computer Science, pages 50–65. Springer.

Lester Randolph Ford. 1956. Network flow theory.

Alberto García-Durán, Sebastijan Dumancic, and Mathias Niepert. 2018. Learning sequence encoders for temporal knowledge graph completion. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, October 31 - November 4, 2018, pages 4816–4821. Association for Computational Linguistics.

Rishab Goel, Seyed Mehran Kazemi, Marcus Brubaker, and Pascal Poupart. 2020. Diachronic embedding for temporal knowledge graph completion. Proceedings of the AAAI Conference on Artificial Intelligence, 34(04):3988–3995.

Shu Guo, Quan Wang, Lihong Wang, Bin Wang, and Li Guo. 2016. Jointly embedding knowledge graphs and logical rules. In Proceedings ofthe 2016 Conference on Empirical Methods in Natural Language Processing, EMNLP 2016, Austin, Texas, USA, November 1-4, 2016, pages 192–202. The Association for Computational Linguistics.

Shu Guo, Quan Wang, Lihong Wang, Bin Wang, and Li Guo. 2018. Knowledge graph embedding with iterative guidance from soft rules. In Proceedings ofthe Thirty-Second AAAI Conference on Artificial Intelligence, (AAAI-18), the 30th innovative Applications ofArtificial Intelligence (IAAI-18), and the 8th AAAI Symposium on Educational Advances in Artificial Intelligence (EAAI-18), New Orleans, Louisiana, USA, February 2-7, 2018, pages 4816–4823. AAAI Press.

Zhen Han, Peng Chen, Yunpu Ma, and Volker Tresp. 2020. Dyernie: Dynamic evolution of riemannian

manifold embeddings for temporal knowledge graph completion. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020, pages 7301–7316. Association for Computational Linguistics.

Zhen Han, Peng Chen, Yunpu Ma, and Volker Tresp. 2021a. Explainable subgraph reasoning for forecasting on temporal knowledge graphs. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net.

Zhen Han, Zifeng Ding, Yunpu Ma, Yujia Gu, and Volker Tresp. 2021b. Learning neural ordinary equations for forecasting future links on temporal knowledge graphs. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021, pages 8352– 8364. Association for Computational Linguistics.

Kurt Hornik. 1991. Approximation capabilities of multilayer feedforward networks. Neural Networks, 4(2):251–257.

Prachi Jain, Sushant Rathi, Mausam, and Soumen Chakrabarti. 2020. Temporal knowledge base completion: New algorithms and evaluation protocols. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020, pages 3733– 3747. Association for Computational Linguistics.

Woojeong Jin, Meng Qu, Xisen Jin, and Xiang Ren. 2020. Recurrent event network: Autoregressive structure inference over temporal knowledge graphs. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6669–6683.

Jaehun Jung, Jinhong Jung, and U Kang. 2021. Learning to walk across time for interpretable temporal knowledge graph completion. In KDD ’21: The 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Virtual Event, Singapore, August 14-18, 2021, pages 786–795. ACM.

Andrey V. Korotayev and Sergey V. Tsirel. 2010. A spectral analysis of world gdp dynamics: Kondratieff waves, kuznets swings, juglar and kitchin cycles in global economic development, and the 2008c2009 economic crisis. Structure and Dynamics: eJournal of Anthropological and Related Sciences, pages 1– 57.

Timothée Lacroix, Guillaume Obozinski, and Nicolas Usunier. 2020. Tensor decompositions for temporal knowledge base completion. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net.

Timothée Lacroix, Nicolas Usunier, and Guillaume Obozinski. 2018. Canonical tensor decomposition for knowledge base completion. In Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018, volume 80 of Proceedings of Machine Learning Research, pages 2869–2878. PMLR.

Ni Lao and William W. Cohen. 2010. Relational retrieval using a combination of path-constrained random walks. Mach. Learn., 81(1):53–67.

Ni Lao, Amarnag Subramanya, Fernando C. N. Pereira, and William W. Cohen. 2012. Reading the web with learned syntactic-semantic inference rules. In Proceedings of the 2012 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning, EMNLP-CoNLL 2012, July 12-14, 2012, Jeju Island, Korea, pages 1017–1026. ACL.

Jennifer Lautenschlager, Steve Shellman, and Michael Ward. 2015. Icews events and aggregations. Harvard Dataverse, 3.

Julien Leblay and Melisachew Wudage Chekol. 2018. Deriving validity time in knowledge graph. In Companion Proceedings of the The Web Conference 2018, pages 1771–1776.

Yujia Li, Shiliang Sun, and Jing Zhao. 2022a. Tirgn: Time-guided recurrent graph network with localglobal historical patterns for temporal knowledge graph reasoning. In Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI 2022, Vienna, Austria, 23-29 July 2022, pages 2152–2158. ijcai.org.

Zixuan Li, Saiping Guan, Xiaolong Jin, Weihua Peng, Yajuan Lyu, Yong Zhu, Long Bai, Wei Li, Jiafeng Guo, and Xueqi Cheng. 2022b. Complex evolutional pattern learning for temporal knowledge graph reasoning. In Proceedings ofthe 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022, pages 290–296. Association for Computational Linguistics.

Zixuan Li, Xiaolong Jin, Wei Li, Saiping Guan, Jiafeng Guo, Huawei Shen, Yuanzhuo Wang, and Xueqi Cheng. 2021. Temporal knowledge graph reasoning based on evolutional representation learning. In SIGIR ’21: The 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, Virtual Event, Canada, July 11-15, 2021, pages 408–417. ACM.

Ke Liang, Lingyuan Meng, Meng Liu, Yue Liu, Wenxuan Tu, Siwei Wang, Sihang Zhou, and Xinwang Liu. 2023. Learn from relational correlations and periodic events for temporal knowledge graph reasoning. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2023, Taipei, Taiwan, July 23-27, 2023, pages 1559–1568. ACM.

Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, and Xuan Zhu. 2015. Learning entity and relation embeddings for knowledge graph completion. In Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, January 25-30, 2015, Austin, Texas, USA, pages 2181–2187. AAAI Press.

Yushan Liu, Yunpu Ma, Marcel Hildebrandt, Mitchell Joblin, and Volker Tresp. 2022. Tlogic: Temporal logical rules for explainable link forecasting on temporal knowledge graphs. In Thirty-Sixth AAAI Conference on Artificial Intelligence, AAAI 2022, Thirty-Fourth Conference on Innovative Applications of Artificial Intelligence, IAAI 2022, The Twelveth Symposium on Educational Advances in Artificial Intelligence, EAAI 2022 Virtual Event, February 22 - March 1, 2022, pages 4120–4127. AAAI Press.

Farzaneh Mahdisoltani, Joanna Biega, and Fabian M. Suchanek. 2015. YAGO3: A knowledge base from multilingual wikipedias. In CIDR 2015, Seventh Biennial Conference on Innovative Data Systems Research.

Arvind Neelakantan, Benjamin Roth, and Andrew Mc-Callum. 2015. Compositional vector space models for knowledge base inference. In 2015 AAAI Spring Symposia, Stanford University, Palo Alto, California, USA, March 22-25, 2015. AAAI Press.

Meng Qu and Jian Tang. 2019. Probabilistic logic neural networks for reasoning. In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pages 7710–7720.

Luc De Raedt, Angelika Kimmig, and Hannu Toivonen. 2007. Problog: A probabilistic prolog and its application in link discovery. In IJCAI 2007, Proceedings of the 20th International Joint Conference on Artificial Intelligence, Hyderabad, India, January 6-12, 2007, pages 2462–2467.

Matthew Richardson and Pedro M. Domingos. 2006. Markov logic networks. Mach. Learn., 62(1-2):107– 136.

Michael Sejr Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, and Max Welling. 2018. Modeling relational data with graph convolutional networks. In The Semantic Web - 15th International Conference, ESWC 2018, volume 10843, pages 593–607.

Chao Shang, Yun Tang, Jing Huang, Jinbo Bi, Xiaodong He, and Bowen Zhou. 2019. End-to-end structureaware convolutional networks for knowledge base completion. In The Thirty-Third AAAI Conference on Artificial Intelligence, pages 3060–3067.

Yichen Song, Aiping Li, Hongkui Tu, Kai Chen, and Chenchen Li. 2021. A novel encoder-decoder knowledge graph completion model for robot brain. Frontiers Neurorobotics, 15:674428.

Yongduo Sui, Xiang Wang, Jiancan Wu, Min Lin, Xiangnan He, and Tat-Seng Chua. 2022. Causal attention for interpretable and generalizable graph classification. In KDD ’22: The 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Washington, DC, USA, August 14 - 18, 2022, pages 1696–1705. ACM.

Haohai Sun, Jialun Zhong, Yunpu Ma, Zhen Han, and Kun He. 2021. Timetraveler: Reinforcement learning for temporal knowledge graph forecasting. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7- 11 November, 2021, pages 8306–8319. Association for Computational Linguistics.

Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier, and Guillaume Bouchard. 2016. Complex embeddings for simple link prediction. In Proceedings of the 33nd International Conference on Machine Learning, ICML 2016, volume 48 of JMLR Workshop and Conference Proceedings, pages 2071– 2080.

Shikhar Vashishth, Soumya Sanyal, Vikram Nitin, and Partha P. Talukdar. 2020. Composition-based multirelational graph convolutional networks. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net.

Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio. 2018. Graph attention networks. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net.

Pengwei Wang, Dejing Dou, Fangzhao Wu, Nisansa de Silva, and Lianwen Jin. 2019. Logic rules powered knowledge graph embedding. CoRR, abs/1903.03772.

Zhen Wang, Jianwen Zhang, Jianlin Feng, and Zheng Chen. 2014. Knowledge graph embedding by translating on hyperplanes. In Proceedings of the Twenty-Eighth AAAI Conference on Artificial Intelligence, July 27 -31, 2014, Québec City, Québec, Canada, pages 1112–1119. AAAI Press.

Chengjin Xu, Yung-Yu Chen, Mojtaba Nayyeri, and Jens Lehmann. 2021. Temporal knowledge graph completion using a linear temporal regularizer and multivector embeddings. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June 6-11, 2021, pages 2569–2578. Association for Computational Linguistics.

Chengjin Xu, Mojtaba Nayyeri, Fouad Alkhoury, Jens Lehmann, and Hamed Shariat Yazdi. 2019. Temporal knowledge graph embedding model based on additive time series decomposition. CoRR, abs/1911.07893.

Chengjin Xu, Mojtaba Nayyeri, Fouad Alkhoury, Hamed Shariat Yazdi, and Jens Lehmann. 2020. TeRo: A time-aware knowledge graph embedding via temporal rotation. In Proceedings of the 28th International Conference on Computational Linguistics, pages 1583–1593, Barcelona, Spain (Online). International Committee on Computational Linguistics.

Chengjin Xu, Mojtaba Nayyeri, Yung-Yu Chen, and Jens Lehmann. 2023. Geometric algebra based embeddings for static and temporal knowledge graph completion. IEEE Trans. Knowl. Data Eng., 35(5):4838–4851.

Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng. 2015. Embedding entities and relations for learning and inference in knowledge bases. In 3rd International Conference on Learning Representations, ICLR 2015.

Matthew D. Zeiler and Rob Fergus. 2014. Visualizing and understanding convolutional networks. In Computer Vision - ECCV 2014 - 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part I, volume 8689 of Lecture Notes in Computer Science, pages 818–833. Springer.

Jing Zhang, Bo Chen, Lingxi Zhang, Xirui Ke, and Haipeng Ding. 2021. Neural, symbolic and neuralsymbolic reasoning on knowledge graphs. AI Open, 2:14–35.

Yongqi Zhang and Quanming Yao. 2022. Knowledge graph reasoning with relational digraph. In WWW ’22: The ACM Web Conference 2022, Virtual Event, Lyon, France, April 25 - 29, 2022, pages 912–924. ACM.

Xiaojuan Zhao, Yan Jia, Aiping Li, Rong Jiang, Kai Chen, and Ye Wang. 2021. Target relational attentionoriented knowledge graph reasoning. Neurocomputing, 461:577–586.

Cunchao Zhu, Muhao Chen, Changjun Fan, Guangquan Cheng, and Yan Zhang. 2021a. Learning from history: Modeling temporal knowledge graphs with sequential copy-generation networks. In Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Thirty-Third Conference on Innovative Applications ofArtificial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence, EAAI 2021, Virtual Event, February 2-9, 2021, pages 4732–4740. AAAI Press.

Zhaocheng Zhu, Zuobai Zhang, Louis-Pascal A. C. Xhonneux, and Jian Tang. 2021b. Neural bellmanford networks: A general graph neural network framework for link prediction. In Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 29476–29490.

## A Bellman-Ford Shortest Path Algorithm

Algorithm 2 The Bellman-Ford Shortest Path $\mathbf { A l - }$   
gorithm   
Input: A graph G with n nodes, the edge set $E ,$   
the source node s   
Output: The distance from node s to any node in   
the graph   
1: $d ^ { ( 0 ) } \gets [ + \infty , + \infty , . . . , + \infty ]$   
2: $d ^ { ( 0 ) } [ s ] \stackrel { - } { - } 0$   
3: for $i = 1$ to $n - 1$ do   
4: $d ^ { ( i ) }  d ^ { ( i - 1 ) }$   
5: for each edge $( u , v ) \in E$ do   
6: $\boldsymbol { d } ^ { ( i ) } [ \boldsymbol { v } ] $ min $\left\{ d ^ { ( i ) } [ v ] , d ^ { ( i - 1 ) } [ u ] + w ( u , v ) \right\}$   
7: end for   
8: end for   
9: return $d ^ { ( n - 1 ) }$

The Bellman-Ford shortest path algorithm (Ford, 1956; Bellman, 1958; Baras and Theodorakopoulos, 2010), founded by Richard Bellman and Lester Ford, is an efficient algorithm for searching the shortest path from the starting point to each node in a graph. The details are shown in Algorithm 2. And the basic idea is: For a graph with n nodes, the shortest path between two nodes in the graph can contain at most n 1 edges. Then, we can perform n 1 times of relaxation operations recursively on the graph to obtain the shortest path from the source node to all nodes in the graph.

## B An Illustration of Temporal Paths and Temporal Links

![](images/3a1d865d3fc18edf4b15f03a30c6c2793fed0c73ea3035663bb498a4c265554d.jpg)  
Figure 4: An illustration of temporal paths and temporal links.

Fig. 4 illustrates some temporal paths and temporal links. For example, $\mathbf { \Psi } _ { r _ { 1 } } ^ { t _ { 1 } } ( e _ { 1 } , e _ { 2 } )$ is a temporal link pointing from $e _ { 1 }$ to $e _ { 2 } .$ while $^ { t _ { 2 } } _ { r _ { 2 } } ( e _ { 1 } , e _ { 3 } ) \wedge _ { r _ { 5 } } ^ { t _ { 5 } } ( e _ { 3 } , e _ { 4 } ) \wedge _ { r _ { 7 } } ^ { t _ { 7 } } ( e _ { 4 } , e _ { 6 } )$ and $\mathbf { \Phi } _ { r _ { 1 } } ^ { t _ { 1 } } ( e _ { 1 } , e _ { 2 } ) \wedge \mathbf { \Phi } _ { r _ { 3 } } ^ { t _ { 3 } } ( e _ { 2 } , e _ { 5 } ) \wedge \mathbf { \Phi } _ { r _ { 6 } } ^ { t _ { 6 } } ( e _ { 5 } , e _ { 6 } )$ are two distinct temporal paths from $e _ { 1 }$ to $e _ { 6 } .$

## C Benchmark Datasets

## C.1 Interpolation datasets

To evaluate our proposed TKG Reasoning method, we perform link prediction task on four widely-used interpolation datasets, namely ICEWS14, ICEWS05-15 (García-Durán et al., 2018), YAGO11k (Dasgupta et al., 2018) and WIKIDATA12k (Dasgupta et al., 2018).<sup>8</sup> The left half of Table 4 summarizes the details of the four interpolation datasets that we use.

ICEWS (Lautenschlager et al., 2015) is a repository containing political events with a specific timestamp. ICEWS14 and ICEWS05-15 (García-Durán et al., 2018) are two subsets of ICEWS corresponding to facts from 1/1/2014 to 12/31/2014 and from 1/1/2005 to 12/31/2015. YAGO11k and WIKIDATA12k (Dasgupta et al., 2018) are subsets of YAGO3 (Mahdisoltani et al., 2015) and WIKI (Erxleben et al., 2014), where time annotations are represented as time intervals. We derive the two datasets from HyTE (Dasgupta et al., 2018) to obtain the same year-level granularity by dropping the month and date information. And the characteristic of a dataset used for the interpolation is disorderly arrangement, which means that it is random and not sorted according to time.

## C.2 Extrapolation datasets

The right half of Table 4 summarizes the details of the four extrapolation datasets that we use, namely ICEWS14 (García-Durán et al., 2018), ICEWS18 (Jin et al., 2020), YAGO (Mahdisoltani et al., 2015) and WIKI (Leblay and Chekol, 2018).<sup>9</sup> ICEWS14 (García-Durán et al., 2018) and ICEWS18 (Jin et al., 2020) are two subsets of ICEWS (Lautenschlager et al., 2015) corresponding to facts from 1/1/2014 to 12/31/2014 and from 1/1/2018 to 10/31/2018. We drop the month and date information of YAGO and WIKI here for ease of processing and to obtain the same year-level granularity as (Jin et al., 2020). Compared to the interpolation datasets (see the left half of Table 4), the facts in the four datasets for extrapolation are arranged in ascending order according to time. For the purpose of extrapolation, we follow Han et al. (2021a) and split each dataset into three subsets by time, ensuring (time of training set) < (time of validation set) < (time of test set). Thus, we can smoothly use past knowledge (facts in the training set) to predict the future.

<table><tr><td rowspan="2">Dataset</td><td colspan="4">Interpolation</td><td colspan="4">Extrapolation</td></tr><tr><td>ICEWS14</td><td>YAGO11k</td><td>ICEWS05-15</td><td>Wikidata12k</td><td>ICEWS14</td><td>ICEWS18</td><td>YAGO</td><td>WIKI</td></tr><tr><td>Entities</td><td>7,128</td><td>10,623</td><td>10488</td><td>12554</td><td>6,869</td><td>23,033</td><td>10,623</td><td>12,554</td></tr><tr><td>Relations</td><td>230</td><td>10</td><td>251</td><td>24</td><td>230</td><td>256</td><td>10</td><td>24</td></tr><tr><td>Train</td><td>72,826</td><td>16,408</td><td>386,962</td><td>32,497</td><td>74,845</td><td>373,018</td><td>161,540</td><td>539,286</td></tr><tr><td>Validation</td><td>8,941</td><td>2,050</td><td>46,275</td><td>4,062</td><td>8,514</td><td>45,995</td><td>19,523</td><td>67,538</td></tr><tr><td>Test</td><td>8,963</td><td>2,051</td><td>46,092</td><td>4,062</td><td>7,371</td><td>49,545</td><td>20,026</td><td>63,110</td></tr><tr><td>Time granularity</td><td>24 hours</td><td>1 year</td><td>24 hours</td><td>1 year</td><td>24 hours</td><td>24 hours</td><td>1 year</td><td>1 year</td></tr></table>

Table 4: Statistics of our experimented datasets for both interpolation setting and extrapolation setting.

Some recent works have shown that the timeunwise filtered setting (Bordes et al., 2013) may probably get incorrect higher ranking scores (Li et al., 2021), and that the time-wise filtered setting (Xu et al., 2019; Goel et al., 2020) is more suitable for TKG Reasoning since it ensures the facts which do not appear at time t are still considered as candidates for evaluating the given test quadruple (Xu et al., 2020). Therefore, we use the time-wise filtered setting (Xu et al., 2019; Goel et al., 2020) to report the experimental results.

## D Experimental Details for Interpolation Reasoning

## D.1 Setup

In this setting, we split each dataset into three subsets: a training set $\mathcal { T } _ { \mathrm { t r a i n } } .$ , a validation set $\mathcal { T } _ { \mathrm { v a l i d } }$ and a test set $\mathcal { T } _ { \mathrm { t e s t } }$

For training, we further split the training set $\mathcal { T } _ { \mathrm { t r a i n } }$ into two parts: the fact set $\mathcal { F } _ { \mathrm { t r a i n } }$ including $3 / 4$ of the training quadruples used to extract temporal paths, and the query set $\mathcal { Q } _ { \mathrm { t r a i n } }$ including the other $1 / 4$ of the training quadruples used as queries. Thus, we can train our model by using temporal paths collected by $\mathcal { F } _ { \mathrm { t r a i n } }$ to answer queries in $\mathcal { Q } _ { \mathrm { t r a i n } } .$

For verification and testing, we accomplish reasoning by answering queries derived from the validation set and test set, where the whole training set $\mathcal { T } _ { \mathrm { t r a i n } }$ is prepared to extract and collect temporal paths. Since the entire training set is known to us and we do not use any data from validation or test set during the training process, we can ensure that there is no data leakage issue.

## D.2 Baselines

For interpolation TKG Reasoning baselines, we consider TTransE (Leblay and Chekol, 2018), HyTE (Dasgupta et al., 2018), TA-DistMult (García-Durán et al., 2018), ATiSE (Xu et al., 2019), and TeRo (Xu et al., 2020), TComplEx (Lacroix et al., 2020), T-GAP (Jung et al., 2021), TELM (Xu et al., 2021), RotateQVS (Chen et al., 2022) and TGeomE++ (Xu et al., 2023).

## D.3 Hyperparameter

To seek and find proper hyperparameters, we utilize a grid search empirically over the following ranges for all four datasets: embedding dimension d in 32, 64, 128 , learning rate in $[ 1 0 ^ { - 5 } , 1 0 ^ { - 3 } ]$ dimension for attention $d _ { a }$ in [3, 4, 5], dropout in [0.1, 0.2, 0.3], batch size in [5, 10, 20, 50], the maximum length of temporal paths L in [3, 4, 5], activation function δ in [identity, tanh, ReLU], and the optimizer we use is Adam.

And we have found out the best hyperparameters combination as follows: for ICEWS14, we set learning rate as 0.0003, batch size as 10, and δ as identity; for ICEWS05-15, we set learning rate as 0.00005, batch size as 5, and δ as identity; for YAGO11k, we set learning rate as 0.00005, batch size as 20, and δ as ReLU; for WIKIDATA12k, we set learning rate as 0.00002, batch size as 20, and δ as ReLU; and for all of four datasets, we choose L as $5 , d _ { a }$ as 5, dropout as 0.2, and d as 128.

## E Experimental Details for Extrapolation Reasoning

## E.1 Setup

In this setting, first of all, we need to arrange each dataset in ascending order according to time. In addition to the same splitting operation as in Section D.1, we need to ensure: (time of training set) < (time of validation set) < (time of test set). Since some entities and relations in the validation or test set may not have appeared in the training set, our extrapolation setting can actually be seen as an inductive inference (Sun et al., 2021).

We use the ground truths for our extrapolation TKG Reasoning, as is the case with many previous methods (Jin et al., 2020; Li et al., 2021; Sun et al., 2021). Specifically, for all of the training, validation and testing, we predict future events assuming ground truths of the preceding events are given at inference time (Han et al., 2021a).

## E.2 Baselines

For extrapolation TKG Reasoning baselines, we consider RE-NET (Jin et al., 2020), CyGNeT (Zhu et al., 2021a), TANGO (Han et al., 2021b), xERTE (Han et al., 2021a), RE-GCN (Li et al., 2021), TITer (Sun et al., 2021), CEN (Li et al., 2022b), TLogic (Liu et al., 2022), TIRGN (Li et al., 2022a) and RPC (Liang et al., 2023).

## E.3 Hyperparameter

A grid search has been taken empirically for the extrapolation TKG Reasoning on the same hyperparameter ranges as in Section D.3. And we have found out the best hyperparameters combination as follows: for ICEWS14, we set learning rate as 0.0003, batch size as 10, d as 128, and δ as identity; for ICEWS18, we set learning rate as 0.00005, batch size as 5, d as 64, and δ as ReLU; for YAGO, we set learning rate as 0.00003, batch size as 10, d as 64, and δ as identity; for WIKI, we set learning rate as 0.00003, batch size as 5, d as 64, and δ as identity; and for all of four datasets, we choose L as 5, $d _ { a }$ as 5, and dropout as 0.2.

## F Details of the Pipeline Setting

To conduct our experimental pipeline, we sample a certain ratio (60% for Setting A, 70% for Setting B, and 80% for Setting C) of facts to train interpolation, while the other unsampled facts serve to provide incomplete quadruples (with either the subject entity or object entity masked) to be completed. The completed quadruples are then treated as new knowledge. Finally, the newly acquired knowledge from interpolation could be combined with the interpolation training set, ordered chronologically, and used as the training set for downstream (extrapolation) tasks, with the test set of the standard extrapolation dataset used to evaluate experimental results.

The process of the proposed pipeline can be seen in Table 6.

<table><tr><td rowspan=1 colspan=1>Ratio</td><td rowspan=1 colspan=2>Inter-</td><td rowspan=1 colspan=1>Extra-</td><td rowspan=1 colspan=1>MRR</td><td rowspan=1 colspan=1>Hits@1</td><td rowspan=1 colspan=1>Hits@3</td><td rowspan=1 colspan=1>Hits@10</td></tr><tr><td rowspan=1 colspan=1>60%</td><td rowspan=1 colspan=2>一</td><td rowspan=1 colspan=1>TLogic</td><td rowspan=1 colspan=1>41.85</td><td rowspan=1 colspan=1>32.51</td><td rowspan=1 colspan=1>47.05</td><td rowspan=1 colspan=1>59.58</td></tr><tr><td rowspan=1 colspan=1>60%</td><td rowspan=1 colspan=2>TELM</td><td rowspan=1 colspan=1>TLogic</td><td rowspan=1 colspan=1>40.88</td><td rowspan=1 colspan=1>31.66</td><td rowspan=1 colspan=1>45.86</td><td rowspan=1 colspan=1>58.69</td></tr><tr><td rowspan=1 colspan=1>60%</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>TIRGN</td><td rowspan=1 colspan=1>42.53</td><td rowspan=1 colspan=1>32.04</td><td rowspan=1 colspan=1>48.10</td><td rowspan=1 colspan=1>62.16</td></tr><tr><td rowspan=1 colspan=1>60%</td><td rowspan=1 colspan=2>RotateQVS</td><td rowspan=1 colspan=1>TIRGN</td><td rowspan=1 colspan=1>40.86</td><td rowspan=1 colspan=1>30.24</td><td rowspan=1 colspan=1>45.78</td><td rowspan=1 colspan=1>61.63</td></tr><tr><td rowspan=1 colspan=1>60%</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>TPAR</td><td rowspan=1 colspan=1>43.86</td><td rowspan=1 colspan=1>33.81</td><td rowspan=1 colspan=1>49.04</td><td rowspan=1 colspan=1>62.83</td></tr><tr><td rowspan=1 colspan=1>60%</td><td rowspan=1 colspan=2>TPAR</td><td rowspan=1 colspan=1>TPAR</td><td rowspan=1 colspan=1>44.56</td><td rowspan=1 colspan=1>34.68</td><td rowspan=1 colspan=1>49.68</td><td rowspan=1 colspan=1>63.32</td></tr><tr><td rowspan=1 colspan=1>70%</td><td rowspan=1 colspan=2>-</td><td rowspan=1 colspan=1>TLogic</td><td rowspan=1 colspan=1>41.88</td><td rowspan=1 colspan=1>32.51</td><td rowspan=1 colspan=1>46.91</td><td rowspan=1 colspan=1>59.97</td></tr><tr><td rowspan=1 colspan=1>70%</td><td rowspan=1 colspan=1>TELM</td><td></td><td rowspan=1 colspan=1>TLogic</td><td rowspan=1 colspan=1>44.55</td><td rowspan=1 colspan=1>32.29</td><td rowspan=1 colspan=1>46.55</td><td rowspan=1 colspan=1>59.53</td></tr><tr><td rowspan=4 colspan=1>70%70%70%70%</td><td rowspan=4 colspan=2>RotateQVSTPAR</td><td rowspan=1 colspan=1>TIRGN</td><td rowspan=1 colspan=1>43.68</td><td rowspan=1 colspan=1>33.23</td><td rowspan=1 colspan=1>48.99</td><td rowspan=1 colspan=1>63.59</td></tr><tr><td rowspan=1 colspan=1>TIRGN</td><td rowspan=1 colspan=1>42.35</td><td rowspan=1 colspan=1>31.87</td><td rowspan=1 colspan=1>47.78</td><td rowspan=1 colspan=1>62.25</td></tr><tr><td rowspan=2 colspan=1>TPARTPAR</td><td rowspan=1 colspan=1>45.49</td><td rowspan=1 colspan=1>35.69</td><td rowspan=1 colspan=1>50.33</td><td rowspan=1 colspan=1>64.30</td></tr><tr><td rowspan=1 colspan=1>46.07</td><td rowspan=1 colspan=1>35.92</td><td rowspan=1 colspan=1>51.06</td><td rowspan=1 colspan=1>65.02</td></tr><tr><td rowspan=6 colspan=1>80%80%80%80%80%80%</td><td rowspan=6 colspan=2>TELMRotateQVSTPAR</td><td rowspan=4 colspan=1>TLogicTLogicTIRGNTIRGN</td><td rowspan=1 colspan=1>42.18</td><td rowspan=1 colspan=1>32.81</td><td rowspan=1 colspan=1>47.27</td><td rowspan=1 colspan=1>60.18</td></tr><tr><td rowspan=1 colspan=1>42.09</td><td rowspan=1 colspan=1>32.69</td><td rowspan=1 colspan=1>47.31</td><td rowspan=1 colspan=1>59.78</td></tr><tr><td rowspan=2 colspan=1>43.6942.78</td><td rowspan=1 colspan=1>32.97</td><td rowspan=1 colspan=1>49.10</td><td rowspan=1 colspan=1>64.28</td></tr><tr><td rowspan=1 colspan=1>32.15</td><td rowspan=1 colspan=1>48.06</td><td rowspan=1 colspan=1>63.14</td></tr><tr><td rowspan=2 colspan=1>TPARTPAR</td><td rowspan=1 colspan=1>46.36</td><td rowspan=1 colspan=1>36.42</td><td rowspan=1 colspan=1>51.34</td><td rowspan=1 colspan=1>65.45</td></tr><tr><td rowspan=1 colspan=1>46.66</td><td rowspan=1 colspan=1>36.68</td><td rowspan=1 colspan=1>51.82</td><td rowspan=1 colspan=1>65.54</td></tr></table>

Table 5: A detailed version of Table 3.

The detailed results of the pipeline setting can be found in Table 5, where a concise version has been shown in Table 3 in Section 5.3.

## G Analysis on Path Length

![](images/e63bb96ab5a7ff3f92c15580de37b443214a129678556c06bf5ff7e07b415790.jpg)  
Figure 5: The MRR performance with different maximum path lengths for the extrapolation reasoning on the ICEWS14 test set.

Fig. 5 displays the MRR performance with different maximum path lengths for the extrapolation reasoning on ICEWS14 test set. We compare our TPAR with the symbolic method TLogic (Liu et al., 2022), where the maximum length L means it contains all lengths not exceeding $L \left( L = 1 , 2 , \cdots 5 \right)$ .

As the maximum length L increases, a greater number of temporal links are covered and can provide more detailed information. However, this also makes it more challenging to learn and reason with this information. The results demonstrate that the performance of TLogic decreases for L values of 4 or higher. In contrast, when L is too small, such as $L \leq 2$ , our TPAR approach struggles to perform well due to the limited amount of information encoded in these short temporal paths. An interesting discovery can be made: For symbolic TLogic, a balance needs to be struck between the richness of the information contained and the ability to effectively learn from it. However, our TPAR appears to alleviate this issue, as neural-symbolic reasoning is more tolerant of ambiguous and noisy data.

<table><tr><td>Step 1 (Interpolation)</td><td>Inputs: A ratio (denoted as r) of known facts sampled from G, denoted as  $r { \mathcal { G } } .$  Queries: The other unsampled facts (1 — r)G with either the subject or the object entity masked. Outputs: Newly acquired facts  $\mathcal { G } _ { n e w }$  by completing the queries.</td></tr><tr><td>Step 2 (Extrapolation)</td><td>Inputs: The basic rG, and the newly acquired  $\mathcal { G } _ { n e w } .$  Queries: The standard extrapolation test set. Outputs: Performance evaluation.</td></tr></table>

Table 6: The process of the proposed pipeline.

<table><tr><td>Length</td><td>Hits@1</td><td>Hits@3 Hits@10</td><td>MRR</td></tr><tr><td>1</td><td>10.03</td><td>10.45 10.58</td><td>10.32</td></tr><tr><td>2</td><td>10.39</td><td>11.06 11.45</td><td>10.84</td></tr><tr><td>3</td><td>15.13</td><td>22.03 32.16</td><td>21.03</td></tr><tr><td>4</td><td>15.59</td><td>22.56</td><td>33.15 21.62</td></tr><tr><td>5</td><td>17.35</td><td>25.14</td><td>37.42 24.12</td></tr></table>

Table 7: Performance of our proposed TPAR with different maximum path lengths for the interpolation reasoning on the YAGO11k test set.

We also take a similar study on the interpolation reasoning, and Table 7 illustrates the performance of our proposed TPAR with different maximum path lengths for the interpolation reasoning on the YAGO11k test set. As the length increases, all metrics of our TPAR have significantly improved.

## H Analysis on Chronological Order

<table><tr><td rowspan=1 colspan=2>Model</td><td rowspan=1 colspan=2>TLogic</td><td rowspan=1 colspan=2>TPAR</td></tr><tr><td rowspan=1 colspan=2>Chronological Order</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>D</td></tr><tr><td rowspan=4 colspan=1>ICEWS14</td><td rowspan=1 colspan=1>Hits@1</td><td rowspan=1 colspan=1>32.60</td><td rowspan=1 colspan=1>33.56</td><td rowspan=1 colspan=1>36.88</td><td rowspan=1 colspan=1>36.02</td></tr><tr><td rowspan=1 colspan=1>Hits@3</td><td rowspan=1 colspan=1>47.06</td><td rowspan=1 colspan=1>48.27</td><td rowspan=1 colspan=1>52.28</td><td rowspan=1 colspan=1>51.49</td></tr><tr><td rowspan=1 colspan=1>Hits@10</td><td rowspan=1 colspan=1>60.06</td><td rowspan=1 colspan=1>61.23</td><td rowspan=1 colspan=1>65.89</td><td rowspan=1 colspan=1>64.30</td></tr><tr><td rowspan=1 colspan=1>MRR</td><td rowspan=1 colspan=1>42.02</td><td rowspan=1 colspan=1>43.04</td><td rowspan=1 colspan=1>46.89</td><td rowspan=1 colspan=1>45.77</td></tr><tr><td rowspan=4 colspan=1>ICEWS18</td><td rowspan=1 colspan=1>Hits@1</td><td rowspan=1 colspan=1>19.92</td><td rowspan=1 colspan=1>20.54</td><td rowspan=1 colspan=1>26.58</td><td rowspan=1 colspan=1>25.24</td></tr><tr><td rowspan=1 colspan=1>Hits@3</td><td rowspan=1 colspan=1>33.04</td><td rowspan=1 colspan=1>33.95</td><td rowspan=1 colspan=1>39.27</td><td rowspan=1 colspan=1>38.55</td></tr><tr><td rowspan=1 colspan=1>Hits@10</td><td rowspan=1 colspan=1>47.75</td><td rowspan=1 colspan=1>48.53</td><td rowspan=1 colspan=1>56.94</td><td rowspan=1 colspan=1>55.54</td></tr><tr><td rowspan=1 colspan=1>MRR</td><td rowspan=1 colspan=1>28.79</td><td rowspan=1 colspan=1>29.82</td><td rowspan=1 colspan=1>35.76</td><td rowspan=1 colspan=1>33.98</td></tr></table>

Table 8: An analysis on chronological order for the extrapolation reasoning.

As discussed in Section 4.1, for extrapolation reasoning, our TPAR does not require strict chronological order between links in a path. Alternatively, methods like TLogic (Liu et al., 2022) demand that all links in a path be ordered chronologically. For example, given a path $\mathcal { P } = { } _ { r _ { 1 } } ^ { t _ { 1 } } ( e _ { 1 } , e _ { 2 } ) \wedge { } _ { r _ { 2 } } ^ { t _ { 2 } } ( e _ { 2 } , e _ { 3 } ) \wedge$ $\mathbf { \Phi } _ { r _ { 3 } } ^ { t _ { 3 } } ( e _ { 3 } , e _ { 4 } )$ , TLogic requires $t _ { q } > t _ { 1 } \geq t _ { 2 } \geq t _ { 3 }$ while we simply require $t _ { q } > t _ { i }$ for i = 1, 2, 3. We conduct an analysis on the chronological order of links in a path, and the results are shown in Table 8. The results indicate that when chronological order is relaxed, the performance of TLogic decreases, while that of our TPAR increases. We believe that by relaxing chronological order in paths, we can gather more paths, thereby providing more information for inference. Nonetheless, this may introduce noise and uncertainty into the process, which ultimately reduces the effectiveness of symbolic methods such as TLogic. On the other hand, our TPAR employs a neural-symbolic approach, which not only mitigates the impact of these challenges but also ensures excellent inference performance, showcasing its strong robustness.