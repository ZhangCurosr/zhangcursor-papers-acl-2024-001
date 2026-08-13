# Progressively Modality Freezing for Multi-Modal Entity Alignment

Yani Huang<sup>1</sup>, Xuefeng Zhang<sup>1</sup>, Richong Zhang<sup>1,2</sup>∗, Junfan Chen<sup>3</sup>, Jaein Kim<sup>1</sup> <sup>1</sup>CCSE, School of Computer Science and Engineering, Beihang University, Beijing, China <sup>2</sup>Zhongguancun Laboratory, Beijing, China <sup>3</sup>School of Software, Beihang University, Beijing, China {huangyn, zhangxf, jaein}@buaa.edu.cn {zhangrc, chenjf}@act.buaa.edu.cn

## Abstract

Multi-Modal Entity Alignment aims to discover identical entities across heterogeneous knowledge graphs. While recent studies have delved into fusion paradigms to represent entities holistically, the elimination of features irrelevant to alignment and modal inconsistencies is overlooked, which are caused by inherent differences in multi-modal features. To address these challenges, we propose a novel strategy of progressive modality freezing, called PMF, that focuses on alignmentrelevant features and enhances multi-modal feature fusion. Notably, our approach introduces a pioneering cross-modal association loss to foster modal consistency. Empirical evaluations across nine datasets confirm PMF’s superiority, demonstrating stateof-the-art performance and the rationale for freezing modalities. Our code is available at https://github.com/ninibymilk/PMF-MMEA.

## 1 Introduction

Multi-modal Knowledge Graphs (MMKGs) integrate various modalities, including text, vision, and structural data, to provide comprehensive insights into knowledge. This integration underpins a range of applications, from question answering (Zhu et al., 2015) and information retrieval (Dietz et al., 2018; Yang, 2020), to recommendation systems (Sun et al., 2020a). Multi-Modal Entity Alignment (MMEA) aims to identify identical entities across heterogeneous MMKGs, which is essential for the integrity of the knowledge represented within these KGs. The essence of MMEA lies in identifying feature commonalities across entities from varied modalities to determine their alignment. The diversity in KG construction introduces potential mismatches in multi-modal features of entities meant to be aligned. For example,

![](images/5f6ee4eb94734b28c8cf4724b6aba127db96ebb9fd6ac0bf019745ca5da57492.jpg)  
Figure 1: Illustration of irrelevant vs. relevant features in multi-modal knowledge graphs.

Figure 1 depicts how the entity “Interstellar (film)” might be represented in one knowledge graph  by a poster image of a spaceship, whereas in another knowledge graph ′, by a portrait of Anne Hathaway, the starring actress. Although both images are related to “Interstellar (film)”, their disparate perspectives can weaken the semantic relationship, challenging the alignment task. This scenario underscores the problem of alignment-irrelevant features, distinctly different features that complicate accurate entity alignment. While recent studies (Lin et al., 2022; Chen et al., 2023a) have employed attention mechanisms to calculate crossmodal weights at both the KG and entity levels, they overlook the critical step of evaluating modality relevance, thereby neglecting to exclude these irrelevant features.

Another obstacle in MMEA involves ensuring semantic consistency across different modalities. Achieving consistent representations across modalities is crucial for effective entity alignment. Existing methods (Lin et al., 2022; Chen et al., 2023a,b) utilize contrastive learning to minimize the intra-modal inconsistencies among aligned entities, yet overlook the cross-modal inconsistencies. Furthermore, the presence of alignmentirrelevant features exacerbates these inconsistencies, complicating the task of achieving consistent representations across modalities.

To address the aforementioned challenges, we propose a novel method for multi-modal entity alignment, named Progressive Modality Freezing (PMF), designed to effectively filter out irrelevant features. PMF is structured around three key parts: multi-modal entity encoders, progressive multimodality feature integration, and a unified training objective for MMEA. Initially, PMF separately encodes entity features from each modality, allowing for flexible adjustment of input modalities. Following this, the method employs a progressive approach to selectively freeze features deemed less relevant for alignment, simultaneously integrating valuable multi-modal features based on their alignment relevance. The culmination of our strategy is a unified training objective, aimed at minimizing the discrepancies both within individual KGs and across different modalities.

To encapsulate, our research contributes in the following three-folds:

• We propose the Progressive Modality Freezing (PMF) strategy to tackle the challenge of irrelevant features in MMEA. By assigning alignment-relevance scores to modalities, our method progressively freezes features that are alignment-irrelevant, while seamlessly integrating beneficial multi-modal information, ensuring the emphasis remains on features of utmost relevance.

• We are at the forefront of employing contrastive learning to address modality consistency across multiple modalities, accompanied by a unified training objective to enhance cross-modal consistency.

• We confirm the effectiveness of PMF across diverse datasets and experimental settings, where it demonstrates superior performance, achieving state-of-the-art in the field. Our thorough analysis further elucidates the advantage of the feature-freezing strategy.

## 2 Related Work

## 2.1 Multi-modal Knowledge Graph Representation Learning

The field of MMKG representation learning forms a critical foundation for tasks such as entity alignment and link prediction, transitioning from traditional single-modal knowledge graph methods to more complex that leverage diverse data types. Early efforts expanded upon established KG representation learning methods for singlemodal knowledge graphs, like those introduced by Xie et al. (2016) and Mousselly-Sergieh et al. (2018), adapting translational KG embedding techniques (Bordes et al., 2013) to incorporate multi-modal data effectively. Subsequent research has delved deeper into the impact of multi-modal information in representation learning. Pezeshkpour et al. (2018) implemented an adversarial method to tackle the challenge of missing data within MMKGs, while Fang et al. (2022) explored a contrastive learning approach to leverage the rich, varied data and heterogeneous structures in MMKGs. These advancements highlight the field’s move towards more sophisticated, datainclusive methodologies.

## 2.2 Multi-Modal Entity Alignment

The exploration of MMEA began with MMKG (Liu et al., 2019), which provided a dataset facilitating entity embedding through linked images. Building on this, MMEA (Chen et al., 2020) advanced the field by fusing relational triples, images, and numerical attributes to generate entity embeddings.

Given the distinctiveness of images as a modality, utilizing visual data in alignment processes has emerged as a key focus in MMEA. For instance, EVA (Liu et al., 2021) exploited visual similarities for unsupervised entity alignment, while MSNEA (Chen et al., 2022) and UMAEA (Chen et al., 2023b) enhanced relational feature learning using visual cues to tackle the challenges posed by missing and ambiguous visual data.

Central to MMEA is the integration of multimodal information. MEAformer (Chen et al., 2023a) employed a dynamic weighting transformer for integration, whereas MCLEA (Lin et al., 2022) aimed at bridging the modality gap via contrastive learning. ACK-MMEA (Li et al., 2023) introduced a strategy for normalizing multi-modal attributes, ensuring consistency across modalities.

Iterative learning has emerged as a powerful strategy for refining entity alignment with methods like XGEA (Xu et al., 2023) and PSNEA (Ni et al., 2023) employing pseudo-labeling to mitigate scarcity of initial seeds.

Despite these advancements, existing methods neglect the necessity of filtering out alignmentirrelevant features and addressing cross-modal inconsistencies. Our method addresses these gaps by selectively freezing less pertinent modalities within an entity to enhance the alignment process.

## 3 Model

We propose a multi-modal entity alignment framework based on progressively freezing entities with irrelevant features and contrastive learning. As shown in Figure 2, the framework consists of three parts: multi-modal entity encoder, progressive multi-modality feature integration, and crossgraph contrastive learning. First, a multi-modal encoder is used to transform the raw input of each modality graph into entity embeddings. Then, we integrate modality features progressively during training, which gradually freezes alignmentirrelevant features and fuse the modality graphs. Finally, the model is optimized through contrastive loss between different knowledge graphs and different modality graphs.

## 3.1 Preliminaries

MMKG Definition A multi-modal knowledge graph (MMKG) is defined as $\mathcal { G } = \{ \mathcal { E } , \mathcal { R } , \mathcal { T } , \mathcal { A } \}$ where is a set of entities, is a set of relations between entities, $\mathcal { T } \subseteq \mathcal { E } \times \mathcal { R } \times \mathcal { E }$ denotes a set of triples in the knowledge graph, and is a set of multi-modal attributes of entities (e.g., textual descriptions, images, etc.). For modality $m \in { \mathcal { M } }$ where is the set of modalities, we define a modality graph $\mathcal { G } _ { m }$ that captures a unique aspect pertinent to the knowledge graph. More specifically, $\mathcal { G } _ { m } = \{ \mathcal { E } , \mathcal { R } , \mathcal { T } , \mathcal { A } _ { m } \}$ , where $A _ { m }$ is the set of attributes associated with modality m.

Task Description In the context of multi-modal entity alignment (MMEA), consider two MMKGs $\mathcal { G } = \{ \mathcal { E } , \mathcal { R } , \mathcal { T } , \mathcal { A } \}$ and $\mathcal { G } ^ { \prime } = \{ \mathcal { E } ^ { \prime } , \mathcal { R } ^ { \prime } , \mathcal { T } ^ { \prime } , \mathcal { A } ^ { \prime } \}$ . The set $\mathcal { S } = \{ ( e _ { i } , e _ { j } ) | e _ { i } \in \mathcal { E } , e _ { j } \in \mathcal { E } ^ { \prime } \}$ denotes preestablished pairs of entities known to correspond to the same real-world object across both graphs. In a supervised learning setting, a model is trained with part of  and then tasked to predict the alignments of the remaining entity pairs.

## 3.2 Multi-Modal Entity Encoder

In the Multi-modal Entity Encoder module, we process the features of entities from each KG according to their respective modalities. Given an

entity $e _ { i } \in \mathcal { G } _ { m }$ , its feature is denoted by $e _ { i } ^ { m }$ . The encoding process for each modality of an entity is defined by the following equation:

$$
h _ { i } ^ { m } = \mathrm { E N C } _ { m } ( \Theta _ { m } , e _ { i } ^ { m } ) ,\tag{1}
$$

where $\mathrm { E N C } _ { m }$ represents the encoder designed for modality m with its associated learnable parameter $\Theta _ { m }$ . While our framework allows the integration of any modality, for a fair comparison, we incorporate four modalities: structure (str), relation (rel), attribute (att), and image (img). Details of $\mathrm { E N C } _ { m }$ are attached in Appendix A.1

## 3.3 Progressive Multi-Modality Feature Integration

The process of integrating multi-modal features in a stepwise manner unfolds in three distinct phases. Initially, the process involves calculating an alignment-relevant score for each modality graph $( \boldsymbol { \mathrm { e } } . \boldsymbol { \mathrm { g } } . , \mathcal { G } _ { m } )$ to discern features that are irrelevant for alignment from those that are crucial. Following this, certain features are selectively ‘frozen’ based on modality scores to mitigate the inclusion of extraneous features. Finally, features from various modality graphs are fused to align multi-modal entities. This approach is executed iteratively and progressively to identify and incorporate beneficial features into the alignment process throughout the training phase, ensuring a refined alignment outcome.

## 3.3.1 Feature Relevance Measuring

We assess the likelihood that features from a given modality of an entity will contribute to the alignment process, thereby enabling the identification of potential alignment-irrelevant features within each modality graph. The alignment-relevant scores for a modality graph $\mathcal { G } _ { m }$ are denoted as $\mathcal { W } _ { m }$ , calculated by the following equation:

$$
\mathcal { W } _ { m } = \mathrm { S C O R E } ( \mathcal { G } _ { m } , \mathcal { G } _ { m } ^ { \prime } ) ,\tag{2}
$$

where SCORE( ) represents a score function. Specifically, for an entity $e _ { i } ^ { m }$ from $\mathcal { G } _ { m }$ and considering a corresponding modality graph $\mathcal { G } _ { m } ^ { \prime }$ from target knowledge graph $\mathcal { G } ^ { \prime }$ , the alignment-relevant score is computed as:

$$
w _ { i } ^ { m } ~ = ~ \mathrm { R e l u } \left( \frac { \alpha _ { i } ^ { m } - \delta } { \displaystyle \operatorname* { m a x } _ { e _ { j } \in \mathcal { E } } ( \alpha _ { j } ^ { m } ) - \delta } \right) ,\tag{3}
$$

$$
\alpha _ { i } ^ { m } = \operatorname* { m a x } _ { e _ { k } \in \mathcal { E } ^ { \prime } } \left( \frac { h _ { i } ^ { m } \cdot h _ { k } ^ { m } } { | h _ { i } ^ { m } | | h _ { k } ^ { m } | } \right) ,\tag{4}
$$

![](images/aec7af0e430638765d23dae934a6b53cfde0f344f91dca5caf1cdfceef802708.jpg)  
Figure 2: Overview of the PMF Model. The framework consists of three components: the Multi-Modal Entity Encoder (&3.2), Progressive Multi-Modality Feature Integration (&3.3), and Cross-Graph Contrastive Learning (&3.4). At epoch t, the multi-modal encoder $E N C _ { m }$ transforms raw inputs from each modality graph into entity embeddings $\mathcal { H } _ { m } ^ { t } .$ Progressive Feature Integration (&3.3.4) occurs during training, employing Irrelevant Feature Freezing (&3.3.2) and performing Relevant Feature Fusion (&3.3.3) guided by Relevant Feature Measuring (&3.3.1). The model is optimized using Cross-KG Alignment Loss $\mathcal { L } _ { C K G } ^ { t }$ (&3.4.2) and Cross-Modality Association Loss $\mathcal { L } _ { C M } ^ { t }$ (&3.4.1).

where δ is a dynamic threshold. The collection of $w _ { i } ^ { m }$ for all entities constitutes $\mathcal { W } _ { m }$ . In a parallel manner, the alignment-relevant scores ${ \mathcal { W } } _ { m } ^ { \prime }$ for $\mathcal { G } _ { m } ^ { \prime }$ are computed similarly.

The underlying rationale for the alignmentrelevant score is to ascertain whether an entity from the current modality graph $\mathcal { G } _ { m }$ exhibits similarity with entities from the modality graph $\mathcal { G } _ { m } ^ { \prime }$ Should the features $h _ { i } ^ { m }$ of an entity $e _ { i }$ in $\mathcal { G } _ { m }$ lack similar counterparts in $\mathcal { G } _ { m } ^ { \prime } .$ , the correlation between $h _ { i } ^ { m }$ and the entity aligned with $e _ { i }$ in $\mathcal { G } _ { m } ^ { \prime }$ would consequently be weak. As such, $h _ { i } ^ { m }$ is less likely to contribute positively to the alignment process. To facilitate a reasonable measurement of feature similarity within a modality, we normalize this metric using the highest similarity across all entities from the same modality graph. This alignment-relevant score is subsequently utilized to guide the processes of modality feature freezing and fusion.

## 3.3.2 Irrelevant Feature Freezing

To minimize the detrimental impact of alignmentirrelevant features on the training process, we introduce a feature-freezing strategy. This approach selectively prevents back-propagation for specific entities within particular modality graphs, thereby preserving the integrity of the training regime. Formally, we define $\mathcal { G } _ { m } ^ { t }$ by extending $\mathcal { G } _ { m }$ with a function $\mathcal { F } _ { m } ^ { t } : \mathcal { E }  \{ 0 , 1 \}$ , to characterize whether entity e is frozen at step t. More specifically, $\mathcal { G } _ { m } ^ { t } = \{ \mathcal { E } , \mathcal { R } , \mathcal { T } , \mathcal { A } _ { m } , \mathcal { F } _ { m } ^ { t } \}$ , where for each entity e in , e is frozen only if $\mathcal { F } _ { m } ^ { t } ( e ) = 0$ . Correspondingly, we refer $\mathcal { H } _ { m } ^ { t }$ and $\mathcal { W } _ { m } ^ { t }$ to the representation of all entities and their alignment-relevant score at step t, respectively. All elements in vector $\mathcal { F } _ { m } ^ { 0 }$ are initialized as 1.

The formalization of the entity freezing process is given by:

$$
\mathcal { F } _ { m } ^ { t } = \mathrm { F R E E Z E } ( \mathcal { W } _ { m } ^ { t } , \mathcal { G } _ { m } ^ { t } ) .\tag{5}
$$

At step t, when the alignment-relevant score $w _ { i } ^ { m }$ for entity $e _ { i }$ from $\mathcal { G } _ { m } ^ { t }$ is zero, it is inferred that this entity does not contribute positively towards the training of the alignment model. Consequently, the gradient update for features $h _ { i } ^ { m }$ associated with such an entity is halted during backpropagation. The freezing operation for an entity $e _ { i } ^ { m }$ , contingent upon its score $w _ { i } ^ { m }$ , is defined as:

$$
\mathrm { F R E E Z E } ( e _ { i } ^ { m } ) = \left\{ \begin{array} { l l } { { 0 } } & { { \mathrm { i f } w _ { i } ^ { m } = 0 , } } \\ { { 1 } } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right. ,\tag{6}
$$

where 0 indicates "stop gradient", ensuring that the gradient flow is interrupted for entities deemed irrelevant, with $w _ { i } ^ { m } \ \in \ \mathcal { W } _ { m } ^ { t }$ FREEZE( ) systematically applies across all entities within $\mathcal { G } _ { m } ^ { t }$ This feature freezing is enacted across all modality graphs of each KG (e.g.,  and $\mathcal { G } ^ { \prime } )$ to optimize the alignment model’s focus on beneficial features.

## 3.3.3 Relevant Feature Fusion

Recognizing that the significance of the features of each modality may vary across entities, we leverage the alignment-relevant scores introduced in

Eq. 3 as a guide during the fusion process.

$$
\mathcal { H } _ { j o i n t } ^ { t } = \mathrm { C O N C A T } ( \mathcal { W } _ { m } ^ { t } , \mathcal { G } _ { m } ^ { t } ) ,\tag{7}
$$

where CONCAT( ) symbolizes the fusion operator acting on all modality graphs $\mathcal { G } _ { m } ^ { t } .$ , integrating them into a joint representation $\mathcal { H } _ { j o i n t } ^ { t }$

In practice, for each entity $e _ { i } ,$ the representations $h _ { i } ^ { m }$ are weighted by their respective confidence $w _ { i } ^ { m }$ , ensuring that modalities with higher relevance scores have a greater influence in the fused representation. Accordingly, at step t, this fusion process is defined as follows:

$$
h _ { i } ^ { j o i n t } = \bigoplus _ { m \in \mathcal { M } } ( [ w _ { i } ^ { m } \cdot h _ { i } ^ { m } ] ) ,\tag{8}
$$

where denotes the vector concatenation operation, and $h _ { i } ^ { j o i n t }$ represents the composite feature embedding for entity $e _ { i } ,$ synthesized from all modalities. The collective $h _ { i } ^ { j o i { \bar { n } } t }$ across all entities at step t forms $\mathcal { H } _ { j o i n t } ^ { t }$

## 3.3.4 Progressive Feature Integration

By integrating information from each modality graph, we can progressively optimize entity alignment. This is done by iteratively refining feature relevance, and gradually isolating alignmentirrelevant features from overt to subtle ones. Initially, basic representations are used to gauge modality confidence and selectively freeze modality features, employing a lower threshold to exclude overtly irrelevant features due to nascent modality representations. As training advances, the model accumulates a deeper understanding of modal information, systematically increasing the threshold to prune less obvious features independent of alignment goals.

The process unfolds as follows: Assuming the model has been trained for t epochs, in epoch $t + 1$ we compute the alignment-relevant scores $\mathcal { W } _ { m } ^ { t + 1 }$ using the current stage of $\mathcal { G } _ { m } ^ { t }$ and $\mathcal { G } _ { m } ^ { ' t }$ . Concurrently, the threshold δ for discerning feature relevance is increased at a predefined rate. Based on the newly calculated scores $\mathcal { W } _ { m } ^ { t + 1 }$ , entities are frozen to reflect the updated insights, updating each modality graph to $\mathcal { G } _ { m } ^ { t + 1 }$ . Subsequently, these refined modality graphs are merged, leveraging the latest alignment-relevant scores to produce the joint graph representation $\mathcal { H } _ { j o i n t } ^ { t + 1 } .$ , as formalized in Equation (2), (5) and (7).

Comparatively, alternatives like “early integration” or “late integration” present limitations.

Early integration might only freeze superficially irrelevant features due to the premature stage of training, which can be challenging to correct in later stages. Our progressive strategy synergistically blends the benefits of both approaches, ensuring a balanced and effective integration of modality information throughout the training lifecycle.

## 3.4 Multi-Modal Entity Alignment Objective

Our framework adopts contrastive learning as a pivotal strategy to extract and leverage alignment information across knowledge graphs and to discern semantic associations between various modality graphs. To accomplish this, we have developed a composite loss function that integrates both cross-KG alignment and cross-modality association within a unified learning objective. The overall loss function at step t is defined as follows:

$$
\begin{array} { r } { \mathcal { L } ^ { t } = \mathcal { L } _ { \mathrm { C M } } ^ { t } + \mathcal { L } _ { \mathrm { C K G } } ^ { t } , } \end{array}\tag{9}
$$

where $\mathcal { L } _ { \mathrm { C K G } } ^ { t }$ is the alignment loss across different $\operatorname { K G s } ,$ and $\mathcal { L } _ { \mathrm { C M } } ^ { t }$ denotes the association loss across modalities within each KG.

## 3.4.1 Cross-Modality Association Loss

One significant source of inconsistencies across modalities is alignment-irrelevant features. Freezing features within each modality graph allows our framework to more effectively draw together the remaining modal features that are beneficial for alignment.

Specifically for each entity $e _ { i } ,$ , we compute the contrastive loss between pairs of modality graphs $\mathcal { G } _ { p }$ and $\mathcal { G } _ { q }$ . The loss function aims to minimize the distance between positive pairs $( e _ { i } ^ { p } , e _ { i } ^ { q } )$ and maximize the separation between negative pairs $( e _ { i } ^ { p } , e _ { j } ^ { q } )$ and $( e _ { j } ^ { p } , e _ { i } ^ { q } )$ . Here $e _ { i }$ and $e _ { j }$ are entities in the same knowledge graph . For each positive pair $( e _ { i } ^ { p } , e _ { i } ^ { q } )$ the cross-modality loss at step t is defined as:

$$
l _ { \mathrm { C M } } ^ { t } ( e _ { i } ^ { p } , e _ { i } ^ { q } ) = \frac { \exp ( h _ { i } ^ { p } \cdot h _ { j } ^ { q } / \tau ) } { \exp ( h _ { i } ^ { p } \cdot h _ { j } ^ { q } / \tau ) + \displaystyle \sum _ { ( e _ { j } ^ { p } , e _ { k } ^ { q } ) \in N _ { i , p , q } ^ { - } } ( h _ { i } ^ { p } \cdot h _ { j } ^ { q } / \tau ) } ,\tag{10}
$$

where $h _ { i } ^ { p } \in \mathcal { H } _ { p } ^ { t } , h _ { j } ^ { q } \in \mathcal { H } _ { q } ^ { t }$ , and exp $( h _ { i } ^ { p } { \cdot } h _ { j } ^ { q } / \tau )$ are scaled by a temperature factor τ , and set $N _ { i , p , q } ^ { - }$ represents negative samples, encompassing all other entity pairs from the modality graphs $\mathcal { G } _ { p }$ and $\mathcal { G } _ { q }$

The framework computes the contrastive loss for entities across each modality graph of the two KGs. The final cross-modality loss is then deter-

mined by:

$$
\mathcal { L } _ { \mathrm { C M } } ^ { t } = \sum _ { p , q \in \mathcal { M } } \sum _ { e _ { i } \in \mathcal { E } \cup \mathcal { E } ^ { \prime } } \beta _ { p } \beta _ { q } \left( - \log l _ { \mathrm { C M } } ^ { t } ( e _ { i } ^ { p } , e _ { i } ^ { q } ) \right) ,\tag{11}
$$

where $\beta _ { p }$ and $\beta _ { q }$ are hyper-parameters associated with modality p and q, adjusting the learning rate for features of the corresponding modality.

## 3.4.2 Cross-KG Alignment Loss

Pre-aligned entity pairs serve as crucial supervision signals in multi-modal entity alignment. Our approach employs supervised contrastive learning to elucidate the alignment relations between entities from corresponding modality graphs of different KGs. We formalized this supervised contrastive learning process at step t as follows:

$$
l _ { c k g } ^ { t } ( e _ { i } ^ { m } , e _ { j } ^ { m } ) = \frac { \exp ( h _ { i } ^ { m } { \cdot } h _ { j } ^ { m } / \tau ) } { \exp ( h _ { i } ^ { m } { \cdot } h _ { j } ^ { m } / \tau ) + \displaystyle \sum _ { e _ { k } \in N _ { i } ^ { - } } \exp ( h _ { i } ^ { m } { \cdot } h _ { k } ^ { m } / \tau ) } ,\tag{12}
$$

where $N _ { i } ^ { - }$ encompasses a set of in-batch negative samples for $e _ { i } ^ { m }$

Given the undirected nature of entity alignment, we compute the contrastive loss considering both directions across KGs, which is defined as:

$$
\mathcal { L } _ { C K G } ^ { t } = \sum _ { m \in \mathcal { M } ^ { + } } \sum _ { ( e _ { i } , e _ { j } ) \in \mathcal { S } } - \frac { 1 } { 2 } \log \left( l _ { c k g } ^ { t } ( e _ { i } ^ { m } , e _ { j } ^ { m } ) \right.\tag{13}
$$

where $\mathcal { M } ^ { + }$ is the set of modalities, including and $\mathbf { \ddot { \rho } } j o i n t ^ { \mathbf { \vec { \rho } } }$ as a special modality that encapsulates the fusion features of multi-modal information. Note that the representations of modality “joint” is taken from $\mathcal { H } _ { j o i n t } ^ { t } .$

## 3.5 Alignment Prediction

The alignment of entities is executed sequentially, focusing on the similarity between their fused multi-modal embeddings. Specifically, for each entity $e _ { i }$ within the joint modality graph $\mathcal { G } _ { j o i n t }$ that remains unaligned, we seek its counterpart, $e _ { j }$ in the corresponding joint modality graph $\begin{array} { r l } { \mathscr { G } _ { j o i n t } ^ { \prime } . } \end{array}$ The sequence in which entities are aligned follows a greedy strategy. The chosen $e _ { j }$ is the unaligned entity that exhibits the highest cosine similarity score with $e _ { i }$

## 4 Experiment

## 4.1 Experiment Setup

Datasets To evaluate the effectiveness of our proposed method, we utilized three publicly available MMEA datasets. MMKG (Liu et al., 2019) features two subsets extracted from Freebase, YAGO, and DBpedia. Multi-OpenEA (Sun et al., 2020b), augmented with entity images from Google search queries, includes two multilingual subsets and two monolingual subsets. DBP15K (Sun et al., 2017; Liu et al., 2021) comprises three subsets from DBpedia’s multilingual version. Seed alignments are designated for 20% of MMKG and Multi-OpenEA entity pairs and 30% for DBP15K entity pairs, aligning with proportions used in prior studies (Chen et al., 2020; Liu et al., 2021; Lin et al., 2022; Chen et al., 2023a,b).

Baselines We compare our approach against six leading multi-modal alignment methods: MMEA (Chen et al., 2020), EVA (Liu et al., 2021), MSNEA (Chen et al., 2022), MCLEA (Lin et al., 2022), MEAformer (Chen et al., 2023a), and UMAEA (Chen et al., 2023b). We replicated MEAformer and UMAEA using their publicly available code to establish robust baselines. Although other notable methods like ACK-MMEA (Li et al., 2023) exist, their exclusion was due to the lack of open-source code, ensuring fairness in our comparisons.

Evaluation Metrics We employ two metrics for evaluation. Hits@1 (abbreviated as H@1) measures the accuracy of top-one predictions. Mean Reciprocal Rank (MRR) assesses the average inverse rank of the correct entity. Higher values of H@1 and MRR indicate better performance.

Implement Details The total training epochs are set to 250, with an option for an additional 500 epochs using an iterative training strategy. Our training regimen incorporates a cosine warm-up schedule (15% step for LR warm-up), early stopping, and gradient accumulation, utilizing the AdamW optimizer with a consistent batch size of 3500. Details of the experimental setup are available in the Appendix A.

## 4.2 Overall Results

Table 1 shows a comprehensive comparison between iterative and non-iterative methods across the three datasets, showcasing our model outperforms all nine sub-datasets.

Our approach significantly surpasses other noniterative methods across various datasets. Specifically, on DBP15K, we see an improvement in

Table 1: Comparison of overall performance presenting both non-iterative and iterative results. The highestperforming baseline results are underlined, and any instances where our method sets a new state-of-the-art are highlighted in bold. Results with an asterisk (\*) indicate our reproduction under identical settings.
<table><tr><td rowspan="2"></td><td rowspan="2">Model</td><td colspan="4">MMKG</td><td colspan="7">Multi-OpenEA</td><td colspan="6">DBP15K</td></tr><tr><td colspan="2">FBDB15K</td><td colspan="2">FBYG15K</td><td colspan="2">EN-FR-V1</td><td colspan="2">EN-DE-V1</td><td colspan="2">D-W-V1</td><td colspan="2">D-W-V2</td><td colspan="2">ZH-EN</td><td colspan="2">JA-EN FR-EN</td></tr><tr><td></td><td>H@1</td><td>MRR</td><td></td><td>H@1 MRR</td><td>H@1</td><td>MRR</td><td>H@1</td><td>MRR</td><td>H@1</td><td>MRR</td><td>H@1</td><td>MRR</td><td>H@1</td><td>MRR</td><td>H@1 MRR</td><td>H@1</td><td>MRR</td></tr><tr><td rowspan="9">noattve</td><td>MMEA</td><td>.265</td><td>.357</td><td>.234</td><td>.317</td><td>=</td><td></td><td>=</td><td>=</td><td></td><td>m</td><td></td><td></td><td>=</td><td></td><td></td><td></td><td>=</td></tr><tr><td>MSNEA</td><td>.114</td><td>.175</td><td>.103</td><td>.153</td><td>.692</td><td>.734</td><td>.753 .804</td><td>.800</td><td>.826</td><td>.838</td><td>.873</td><td>.609</td><td>.685</td><td>.541</td><td>.620</td><td>.557</td><td>.643</td></tr><tr><td>EVA</td><td>.199</td><td>.283</td><td>.153</td><td>.224</td><td>.785</td><td>.836</td><td>.922 .945</td><td>.858</td><td>.891</td><td>.890</td><td>.922</td><td>.683</td><td>.762</td><td>.669</td><td>.752</td><td>.686</td><td>.771</td></tr><tr><td>MCLEA</td><td>.295</td><td>.393</td><td>.254</td><td>.332</td><td>.819</td><td>.864</td><td>.939 .957</td><td>.881</td><td>.908</td><td>.928</td><td>.949</td><td>.726</td><td>.796</td><td>.719</td><td>.789</td><td>.719</td><td>.792</td></tr><tr><td>MEAformer*</td><td>.418</td><td>.519</td><td>.327</td><td>.418</td><td>.836</td><td>.882</td><td>.954</td><td>.971</td><td>.909 .933</td><td>.944</td><td>.962</td><td>.771</td><td>.835</td><td>.764</td><td>.834</td><td>.772</td><td>.841</td></tr><tr><td>UMAEA*</td><td>.454</td><td>.552</td><td>.355</td><td>.451</td><td>.847</td><td>.891</td><td>.955</td><td>.970 .905</td><td>.930</td><td>.948</td><td>.967</td><td>.800</td><td>.860</td><td>.801</td><td>.862</td><td>.818</td><td>.877</td></tr><tr><td>PMF</td><td>.539</td><td>.620</td><td>.459</td><td>.539</td><td>.912</td><td>.942</td><td>.973 .983</td><td>.955</td><td>.970</td><td>.981</td><td>.989</td><td>.835</td><td>.884</td><td>.835</td><td>.885</td><td>.850</td><td>.898</td></tr><tr><td>improve</td><td>8.5%</td><td>6.8%</td><td>10.4%</td><td>8.8%</td><td>6.5%</td><td>5.1%</td><td>1.8% 1.2%</td><td>4.6%</td><td>3.7%</td><td>3.3%</td><td>2.2%</td><td>3.5%</td><td>2.4%</td><td>3.4%</td><td>2.3%</td><td>3.2%</td><td>2.1%</td></tr><tr><td>MSNEA</td><td>.149</td><td>.232</td><td>.138</td><td>.210</td><td>.699</td><td>.742</td><td>.788 .835</td><td>.809</td><td>.836</td><td>.862</td><td>.894</td><td>.648</td><td>.728</td><td>.557</td><td>.643</td><td>.583</td><td>.672</td></tr><tr><td rowspan="7">teatitve</td><td>EVA</td><td>.231</td><td>.318</td><td>.188</td><td>.260</td><td>.849</td><td>.896</td><td>.956</td><td>.968</td><td>.915</td><td>.942</td><td>.925 .951</td><td>.750</td><td>.810</td><td>.741</td><td>.807</td><td>.765</td><td>.831</td></tr><tr><td>MCLEA</td><td>.395</td><td>.487</td><td>.322</td><td>.400</td><td>.888</td><td>.924</td><td>.969</td><td>.979</td><td>.944</td><td>.963</td><td>.969 .982</td><td>.811</td><td>.865</td><td>.805</td><td>.863</td><td>.808</td><td>.867</td></tr><tr><td>MEAformer*</td><td>.578</td><td>.661</td><td>.444</td><td>.529</td><td>.903</td><td>.935</td><td>.963</td><td>.977 .954</td><td>.970</td><td>.969</td><td>.981</td><td>.847</td><td>.892</td><td>.842</td><td>.892</td><td>.845</td><td>.894</td></tr><tr><td>UMAEA*</td><td>.561</td><td>.660</td><td>.463</td><td>.560</td><td>.895</td><td>.931</td><td>.974</td><td>.984 .945</td><td>.965</td><td>.973</td><td>.984</td><td>.856</td><td>.900</td><td>.857</td><td>.904</td><td>.873</td><td>.917</td></tr><tr><td></td><td>.624</td><td>.702</td><td>.543</td><td>.620</td><td>.923</td><td>.950</td><td>.980</td><td>.988 .960</td><td>.973</td><td>.986</td><td>.992</td><td>.867</td><td>.908</td><td>.866</td><td>.909</td><td>.879</td><td></td></tr><tr><td>PMF</td><td>4.6%</td><td>4.1%</td><td>8.1%</td><td></td><td>2.0%</td><td>1.5%</td><td>0.6% 0.4%</td><td>0.6%</td><td>0.3%</td><td>1.3%</td><td>0.8%</td><td>1.1%</td><td>0.8%</td><td>0.9%</td><td>0.5%</td><td></td><td>.921</td></tr><tr><td>improve</td><td></td><td></td><td></td><td>6.0%</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.6%</td><td>0.4%</td></tr></table>

Table 2: Non-iterative model performance comparison utilizing surface information on DBP15K.
<table><tr><td rowspan="2">Model</td><td colspan="5">DBP15K</td></tr><tr><td>ZH-EN</td><td></td><td>JA-EN</td><td></td><td>FR-EN</td></tr><tr><td rowspan="3">MSNEA EVA</td><td>H@1</td><td>MRR</td><td>H@1</td><td>MRR</td><td>H@1</td><td>MRR</td></tr><tr><td>.929</td><td>.951</td><td>.964</td><td>.976</td><td>.990</td><td>.994</td></tr><tr><td>.887</td><td>.913</td><td>.938</td><td>.955</td><td>.969</td><td>.980</td></tr><tr><td>MCLEA</td><td>.926</td><td>.946</td><td>.961</td><td>.973</td><td>.987</td><td>.992</td></tr><tr><td>MEAformer*</td><td>.944</td><td>.962</td><td>.976</td><td>.985</td><td>.990</td><td>.994</td></tr><tr><td>UMAEA*</td><td>.945</td><td>.963</td><td>.978</td><td>.986</td><td>.990</td><td>.994</td></tr><tr><td>PMF</td><td>.958</td><td>.973</td><td>.985</td><td>.990</td><td>.992</td><td>.995</td></tr></table>

H@1 of up to 3.5% over the nearest competitor. The Multi-OpenEA datasets show enhancements in the range of 1.8% to 6.5%, while the FBDB15K and FBYG15K datasets experience substantial gains, with increases up to 10.4%. These improvements highlight our method’s capability to effectively filter out conflicting and lowerquality modality information, enhancing overall alignment accuracy.

To assess our model’s performance with added textual data, we initialized text modality embeddings using surface representations from (Lin et al., 2022). The incorporation of textual information markedly enhances entity alignment, as evidenced by the results in Table 2 for the DBP15K dataset. Despite all models benefiting from this supervisory signal, our method outperforms the baselines and demonstrates its robustness in leveraging textual data for improved alignment.

## 4.3 Ablation Study

To assess the contribution of various components within our framework, we introduce five variants: Feature Relevance Measuring (FRM), Irrelevant Feature Freezing (IFF), Relevant Feature

![](images/d0dfcc9292b010ee9075f3df14cbe4543cb5b214a42497326a359d2681106f2b.jpg)  
Figure 3: Comparison of variants of PMF on DBP15K in terms of H@1.

Fusion (RFF), Progressive Integration (PI), and Cross-Modality Association Loss $( \mathcal { L } _ { \mathrm { C M } } )$ . By individually removing or modifying these components, their contributions to the performance were assessed. As depicted in Figure 3, the omission of the FRM component leads to a significant drop in H@1 performance across all datasets. For example, H@1 fell from 0.835 to 0.749 on $\mathrm { D B P 1 5 K _ { Z H - E N } }$ . The influence of FRM extends to both IFF and RFF components, with RFF showing a more significant impact on performance due to its integral role in modality fusion and its effect on the overall training loss.

Further, replacing our "progressive" integration strategy with a "static" one, where FRM occurs only once, resulted in a marked performance downturn. This underscores the significance of progressively identifying alignment-irrelevant features, a task challenging to accomplish in a single iteration. Eliminating $\mathcal { L } _ { C M }$ also led to a decline in model performance. This underlines the effectiveness of cross-modality loss in fostering consistent entity representation learning across modalities.

## 4.4 Modality Analysis

## 4.4.1 Impact of Different Modality

![](images/4af9d2041d77166e744efc493aa238b2143c0baae7a23fc7e47a67d9a12e874a.jpg)  
Figure 4: Impact of various modalities on DBP15K in terms of H@1.

We examined the influence of modality information on entity alignment on DBP15K, as depicted in Figure 4. The absence of any modality notably hinders performance, with structure and image modalities showing the most significant effect. The foundational role of structures in KGs and the distinctive features provided by images underscore their importance in alignment.

A slight drop in performance is observed when either relation or attribute is excluded, attributed to the typically lower count of attributes and relations compared to entities, which may lead to attribute or relation overlaps among different entities. Conversely, a notable performance drop occurs when both are removed. This suggests a complementary relationship between attributes and relations that enhances alignment.

## 4.4.2 Distribution of Modality Scores

![](images/fa432e79cf059a2317927233f034152b3f99a2e9b4c36a1521ff59b62bb43ed1.jpg)  
Figure 5: Distribution of relevance scores across modalities for pre-aligned entity pairs in $\mathrm { D B P 1 5 K _ { F R - E N } }$

We examined the distribution of modality scores to understand the contribution of different modalities to entity alignment, as depicted in Figure 5. Each point in the figure represents an aligned entity pair, with axes reflecting their modality scores across source and target KGs.

Attributes, relations, and structures show a widespread and increased distribution of scores, indicating their positive impact on alignment. Conversely, image modality shows a distinct, concentrated pattern, highlighting the variability in image features between aligned entities due to the limited scope of images and inconsistencies in KG construction. This underscores the benefit of excluding less reliable image features. Moreover, the presence of data points near the axes suggests challenges with one-sided visual information. Interestingly, a strong positive correlation emerges between modality scores of aligned entities, validating our modality scoring approach. This correlation implies that when a modality is less helpful for one entity in an aligned pair, it tends to be similarly low for the counterpart.

## 4.5 Parameters Analysis

![](images/72e90eb531ffc66bf376232328a35beef13b79f45c337bfe478f6d4346313019.jpg)  
Figure 6: Impact of varying threshold δ on frozen ratio and H@1 performance: the x-axis represents δ values ranging from 0.1 to 0.95.

To further investigate the impact of parameters on integration, we analyze how varying the threshold parameter δ impacts the frozen ratio in image modality and alignment performance on DBP15K. Figure 6 shows that when δ is below 0.4, minimal changes are observed in both the ratio and H@1 performance, likely due to the absence of images for certain entities. Beyond this point, as δ ascends, we note a gradual increase in the frozen ratio, paralleled by an improvement in H@1, despite the initial decline. This underscores the benefits of selectively and gradually freezing entities. A notable drop in H@1 performance for some datasets occurs at $\delta = 0 . 9 5$ , indicating that images are beneficial for alignment and should not be frozen excessively.

Further experimental analysis results of the

Table 3: Case study of representative frozen images on $\mathrm { D B P 1 5 K _ { F R - E N } }$ during training.
<table><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=2>Source-KG(French)</td><td rowspan=1 colspan=2>Target-KG(English)</td></tr><tr><td rowspan=1 colspan=1>Entity</td><td rowspan=1 colspan=1>Image</td><td rowspan=1 colspan=1>Entity</td><td rowspan=1 colspan=1>Image</td></tr><tr><td rowspan=4 colspan=1>Earyozen</td><td rowspan=1 colspan=1>Louis XV</td><td rowspan=1 colspan=1><img src="images/6a8953149d956b73bab952bb14153d898080d7f709cc10a89731aba79c2f24d0.jpg"/></td><td rowspan=1 colspan=1>Louis XV ofFrance</td><td rowspan=1 colspan=1><img src="images/0baf9151c0c43c9d491523fe3ff27cfdbbf37fe2ded3b7046f44f6949c39b548.jpg"/></td></tr><tr><td rowspan=1 colspan=1>Président duMexique</td><td rowspan=1 colspan=1><img src="images/50005061b550ba5226a7bfdeb0df21a2ad6f47cb5619a2c48aa4b5420317f45b.jpg"/></td><td rowspan=1 colspan=1>President ofMexico</td><td rowspan=1 colspan=1><img src="images/b8209e43d5c352d13129110b9e837e2a8edc2fa6831e16acb0dc033c65e6c1d0.jpg"/></td></tr><tr><td rowspan=1 colspan=1>UniversitéStanford</td><td rowspan=1 colspan=1><img src="images/8dde02a834b319fdabb224fa0f4ddd6882c4e1e248594b6cb74d645f893df166.jpg"/></td><td rowspan=1 colspan=1>StanfordUniversity</td><td rowspan=1 colspan=1><img src="images/1677f1798cef0b990060f51562b267ce5239ac686f2ddeb7ce572567fbc70fc8.jpg"/></td></tr><tr><td rowspan=1 colspan=1>Disquecompact</td><td rowspan=1 colspan=1><img src="images/398a69679d76227c6cdc90ac12392e540db1529487846f1ce9c40a43e46f9f64.jpg"/></td><td rowspan=1 colspan=1>Compactdisc</td><td rowspan=1 colspan=1><img src="images/6b5d4c9cb3697d90dd7b58734b7322c4e5684df27622acf7e48ea7f6f6818b05.jpg"/></td></tr><tr><td rowspan=4 colspan=1>Late ozen</td><td rowspan=1 colspan=1>Juliana oftheNetherlands</td><td rowspan=1 colspan=1><img src="images/55f3dfc224b32ba1bf8ce317b5413463b26da4a3e785fdea7673e5c3465093b9.jpg"/></td><td rowspan=1 colspan=1>Juliana oftheNetherlands</td><td rowspan=1 colspan=1><img src="images/4e8031b5fef804b3e23ab103d153d533919bbc87b7e3e2990da8a4e8d6aa5f96.jpg"/></td></tr><tr><td rowspan=1 colspan=1>Brampton(Ontario)</td><td rowspan=1 colspan=1><img src="images/3529d281b5e32c26c865dee19b2d7ecde6e797b47a3c405fc276af9889225881.jpg"/></td><td rowspan=1 colspan=1>Brampton</td><td rowspan=1 colspan=1><img src="images/ba3a6aef64d712153f4d67fd6e4ef28af1d73e277d196055054b2a22c250ef45.jpg"/></td></tr><tr><td rowspan=1 colspan=1>STS-129</td><td rowspan=1 colspan=1><img src="images/6ee5b6672a88f005fd8cf2bde2b64951037c1ccb3f50ddc27a46024a0ba52d71.jpg"/></td><td rowspan=1 colspan=1>STS-129</td><td rowspan=1 colspan=1><img src="images/07471ccf9dc6466c229f031e34988404f1a3cb45de9f0c975bc8efb54684de20.jpg"/></td></tr><tr><td rowspan=1 colspan=1>Queens ofthe StoneAge</td><td rowspan=1 colspan=1><img src="images/c969bc92f45795745dd056ebc9827df0df14581bacd696f63de92a05324c9319.jpg"/></td><td rowspan=1 colspan=1>Queens ofthe StoneAge</td><td rowspan=1 colspan=1></td></tr></table>

feature-freezing process of PMF, including additional experiments, are available in Appendix B.

## 4.6 Case Study

To illustrate differences in features frozen at different stages, in Table 3, we select frozen image information during training in the $\mathrm { D B P 1 5 K _ { F R - E N } }$ dataset arranged in a time series. Early frozen image pairs show significant differences. For example, the "President\_of\_Mexico" entity corresponds to a presidential portrait in the $\mathrm { D B P } _ { \mathrm { F R } }$ but to a national flag image in the $\mathrm { D B P _ { E N } }$ . Images will not benefit the process of alignment in this case. After several epochs of training, the differences between frozen image pairs frozen later are smaller. For example, the "Juliana\_of\_the\_Netherlands" entity, where the image in $\mathrm { D B P } _ { \mathrm { F R } }$ is a photo of the queen in her youth, while in $\mathrm { D B P _ { E N } }$ , it is a photo of the queen in her old age. These cases reflect the process of effectively detecting alignment-irrelevant features based on the principle of progressing from easy to difficult cases.

## 5 Conclusion

In this study, we presented the Progressive Modality Freezing (PMF) model to advance Multi-Modal Entity Alignment. By measuring and evaluating the relevance of various modalities, PMF progressively freezes features deemed less critical, thereby facilitating the integration and consistency of multi-modal features. Furthermore, we introduced a unified training objective tailored to foster a harmonious contrast between KGs and modalities. Empirical evaluations on nine sub-datasets affirm the superiority of PMF and validate the rationale behind the selective freezing of modalities.

## Limitations

PMF marks a significant advancement in MMEA, yet it encounters limitations in two main aspects.

First, the method for measuring the alignment relevance of modal features by evaluating their usefulness in identifying similar traits across KGs sometimes shows inconsistent reliability. As shown in Figure 6, a minor decline in H@1 when δ is below 0.5 suggesting that essential modal information may be prematurely frozen. Future work should aim to refine the approach for scoring alignment relevance, ensuring a more consistent and accurate identification of pertinent features.

Second, our model predominantly exhibits efficiency in non-iterative learning settings, with its performance often unstable under iterative learning frameworks. This limitation stems from the model’s initial design, which is inherently optimized for non-iterative settings, limiting the effectiveness of incorporating iterative strategies like pseudo seeds for enhancing progressive freezing. Further attempts will explore improved ways to integrate iterative learning with progressive freezing for enhanced results.

## Acknowledgments

This work was supported by the National Science and Technology Major Project under Grant 2022ZD0120202, in part by the National Natural Science Foundation of China (No. U23B2056), in part by the Fundamental Research Funds for the Central Universities, in part by the State Key Laboratory of Complex & Critical Software Environment, and in part by the Shanxi Province Special Support for Science and Technology Cooperation and Exchange (202204041101020).

## References

Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko. 2013. Translating embeddings for modeling multirelational data. Advances in neural information processing systems, 26.

Liyi Chen, Zhi Li, Yijun Wang, Tong Xu, Zhefeng Wang, and Enhong Chen. 2020. Mmea: entity alignment for multi-modal knowledge graph. In Knowledge Science, Engineering and Management: 13th International Conference, KSEM 2020, Hangzhou, China, August 28–30, 2020, Proceedings, Part I 13, pages 134–147. Springer.

Liyi Chen, Zhi Li, Tong Xu, Han Wu, Zhefeng Wang, Nicholas Jing Yuan, and Enhong Chen. 2022. Multimodal siamese network for entity alignment. In Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining, pages 118– 126.

Zhuo Chen, Jiaoyan Chen, Wen Zhang, Lingbing Guo, Yin Fang, Yufeng Huang, Yichi Zhang, Yuxia Geng, Jeff Z Pan, Wenting Song, et al. 2023a. Meaformer: Multi-modal entity alignment transformer for meta modality hybrid. In Proceedings ofthe 31st ACM International Conference on Multimedia, pages 3317– 3327.

Zhuo Chen, Lingbing Guo, Yin Fang, Yichi Zhang, Jiaoyan Chen, Jeff Z Pan, Yangning Li, Huajun Chen, and Wen Zhang. 2023b. Rethinking uncertainly missing and ambiguous visual modality in multi-modal entity alignment. In International Semantic Web Conference, pages 121–139. Springer.

Laura Dietz, Alexander Kotov, and Edgar Meij. 2018. Utilizing knowledge graphs for text-centric information retrieval. In The 41st international ACM SIGIR conference on research & development in information retrieval, pages 1387–1390.

Quan Fang, Xiaowei Zhang, Jun Hu, Xian Wu, and Changsheng Xu. 2022. Contrastive multi-modal knowledge graph representation learning. IEEE Transactions on Knowledge and Data Engineering.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision andpattern recognition, pages 770– 778.

Qian Li, Shu Guo, Yangyifei Luo, Cheng Ji, Lihong Wang, Jiawei Sheng, and Jianxin Li. 2023. Attribute-consistent knowledge graph representation learning for multi-modal entity alignment. arXiv preprint arXiv:2304.01563.

Zhenxi Lin, Ziheng Zhang, Meng Wang, Yinghui Shi, Xian Wu, and Yefeng Zheng. 2022. Multi-modal contrastive representation learning for entity alignment. arXiv preprint arXiv:2209.00891.

Fangyu Liu, Muhao Chen, Dan Roth, and Nigel Collier. 2021. Visual pivoting for (unsupervised) entity alignment. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pages 4257– 4266.

Ye Liu, Hui Li, Alberto Garcia-Duran, Mathias Niepert, Daniel Onoro-Rubio, and David S Rosenblum. 2019. Mmkg: multi-modal knowledge graphs. In The Semantic Web: 16th International Conference, ESWC 2019, Portorož, Slovenia, June 2–6, 2019, Proceedings 16, pages 459–474. Springer.

Xin Mao, Wenting Wang, Yuanbin Wu, and Man Lan. 2021. From alignment to assignment: Frustratingly simple unsupervised entity alignment. arXiv preprint arXiv:2109.02363.

Hatem Mousselly-Sergieh, Teresa Botschen, Iryna Gurevych, and Stefan Roth. 2018. A multimodal translation-based approach for knowledge graph representation learning. In Proceedings of the Seventh Joint Conference on Lexical and Computational Semantics, pages 225–234.

Wenxin Ni, Qianqian Xu, Yangbangyan Jiang, Zongsheng Cao, Xiaochun Cao, and Qingming Huang. 2023. Psnea: Pseudo-siamese network for entity alignment between multi-modal knowledge graphs. In Proceedings of the 31st ACM International Conference on Multimedia, pages 3489–3497.

Pouya Pezeshkpour, Liyan Chen, and Sameer Singh. 2018. Embedding multimodal relational data for knowledge base completion. arXiv preprint arXiv:1809.01341.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR.

Karen Simonyan and Andrew Zisserman. 2014. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.

Rui Sun, Xuezhi Cao, Yan Zhao, Junchen Wan, Kun Zhou, Fuzheng Zhang, Zhongyuan Wang, and Kai Zheng. 2020a. Multi-modal knowledge graphs for recommender systems. In Proceedings of the 29th ACM international conference on information & knowledge management, pages 1405–1414.

Zequn Sun, Wei Hu, and Chengkai Li. 2017. Cross-lingual entity alignment via joint attributepreserving embedding. In The Semantic Web–ISWC 2017: 16th International Semantic Web Conference, Vienna, Austria, October 21–25, 2017, Proceedings, Part I 16, pages 628–644. Springer.

Zequn Sun, Qingheng Zhang, Wei Hu, Chengming Wang, Muhao Chen, Farahnaz Akrami, and Chengkai Li. 2020b. A benchmarking study of

embedding-based entity alignment for knowledge graphs. arXiv preprint arXiv:2003.07743.

Ruobing Xie, Zhiyuan Liu, Huanbo Luan, and Maosong Sun. 2016. Image-embodied knowledge representation learning. arXiv preprint arXiv:1609.07028.

Baogui Xu, Chengjin Xu, and Bing Su. 2023. Crossmodal graph attention network for entity alignment. In Proceedings of the 31st ACM International Conference on Multimedia, pages 3715–3723.

Zuoxi Yang. 2020. Biomedical information retrieval incorporating knowledge graph for explainable precision medicine. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2486– 2486.

Yuke Zhu, Ce Zhang, Christopher Ré, and Li Fei-Fei. 2015. Building a large-scale multimodal knowledge base system for answering visual queries. arXiv preprint arXiv:1507.05670.

## Appendix

## A Detailed setup

## A.1 Encoders of each modality

We incorporate four modalities, including structure, image, relationship, and attribute, to implement and evaluate the proposed method. The settings of encoders for each modality are as follows.

The modality of structures describes the topological structure of entities in the knowledge graph. Specifically, the input of structural modal is the connections between various entities in the knowledge graph. The connections can be described by the adjacency matrix of the knowledge graph. Let the adjacency matrix of $\mathcal { G }$ be $A ,$ , we use graph attention networks (GAT) to encode structural information. Therefore, we define:

$$
{ h _ { i } } ^ { s t r } = \mathbf { G } \mathbf { A } \mathbf { T } ( A , \mathbf { e } _ { i } ^ { s t r } ) ,\tag{14}
$$

Where $\mathbf { e } _ { i } ^ { s t r }$ is the randomly initialized representation of the entity $e _ { i }$

The modality of relations consists of the relationships related to each entity in text form. The relational feature of entity $e _ { i }$ is the set of relation names from all the triples in which $e _ { i }$ occurs. Since the number of relations is limited, we use the bag-of-words method to process input features. ${ \bf e } _ { i } ^ { r e l }$ is defined as a one-hot vector, and the dimension is determined by the number of relations. $\mathrm { E N C } _ { \mathrm { r e l } }$ consists of a fully connected layer.

Similarly, textual entity attributes constitute an independent modality. Since the number of attributes is also limited, we adopt a processing method similar to the modality of relations. Specifically, $\mathrm { E N C } _ { \mathrm { a t t } }$ is a fully connected layer, and the input $\mathbf { e } _ { i } ^ { a t t }$ is defined as a one-hot vector, and the dimension depends on the number of attributes.

The image modality is composed of the visual entity attributes. Encoding images is usually difficult and requires a large amount of images for pretraining. Therefore, we first use a fixed pre-trained vision model (Simonyan and Zisserman, 2014; He et al., 2016; Radford et al., 2021), to convert the original pixel information into a vector representation ${ \bf e } _ { i } ^ { i m g }$ . Then, a trainable fully connected layer is used as ENC<sub>img</sub>.

The encoder of relations, attributes, and images can be expressed by:

$$
{ h _ { i } } ^ { m } = \mathbf { F C _ { m } } ( \mathbf { e } _ { i } ^ { m } ) ,\tag{15}
$$

where $\mathbf { F C _ { m } }$ is a trainable fully connected layer, $m \in \{ r e l , a t t r , i m g \}$

## A.2 Dataset Statistics

Statistics for DBP15K, Multi-OpenEA, and MMKG are presented in Table 4. The column named "EA pairs" indicates pre-aligned entity pairs. Note that not every entity is paired with images or counterparts in the alternate KG.

## A.3 Metric Details

We evaluate the performance of entity alignment tasks using two common metrics: Hits@n and MRR.

Hits@n is a widely used performance metric in entity alignment tasks. It is calculated as follows: Given a total number of entity pairs N, for each entity in graph ${ \mathcal { G } } ,$ compute its similarity with potential matching entities in graph $\mathcal { G } ^ { \prime }$ and rank them according to their scores. If the correct entity’s rank is within the top n positions, the Hits@n count is incremented. The value of Hits@n is the percentage of the Hits@n count out of the total number of entity pairs N. Higher Hits@n values indicate better alignment performance.

$$
H i t s @ n = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } I ( r a n k _ { i } \le n )\tag{16}
$$

MRR (Mean Reciprocal Rank) is a general evaluation metric for search algorithms and, when applied to entity alignment tasks, calculates the similarity between each entity in graph $\mathcal { G }$ and potential matching entities in graph ${ \mathcal { G } } ,$ ranking them according to their scores. If the correct entity is ranked n, the score is $\textstyle { \frac { 1 } { n } }$ . The sum of scores for all entities yields the MRR score. MRR reflects the overall performance of the entity alignment algorithm, with higher MRR scores indicating better alignment results.

Table 4: Statistical details for MMEA datasets.
<table><tr><td>Dataset</td><td>KG</td><td># Ent.</td><td># Rel.</td><td># Attr.</td><td># Rel. Triples</td><td># Attr. Triples</td><td># Image</td><td># EA pairs</td></tr><tr><td rowspan="2">DBP15KZH-EN</td><td>ZH (Chinese)</td><td>19388</td><td>1701</td><td>8111</td><td>70414</td><td>248035</td><td>15912 14125</td><td rowspan="2">15000</td></tr><tr><td>EN (English)</td><td>19572</td><td>1323 7173</td><td></td><td>95142</td><td>343218</td></tr><tr><td rowspan="2">DBP15KJA-EN</td><td>JA (Japanese)</td><td>19814</td><td>1299</td><td>5882</td><td>77214</td><td>248991</td><td>12739</td><td rowspan="2">15000</td></tr><tr><td>EN (English)</td><td>19780 1153</td><td>6066</td><td></td><td>93484</td><td>320616</td><td>13741</td></tr><tr><td rowspan="2">DBP15KFR-EN</td><td>FR (French)</td><td>19661</td><td>903</td><td>4547</td><td>105998</td><td>273825</td><td>14174</td><td rowspan="2">15000</td></tr><tr><td>EN (English)</td><td>19993</td><td>1208</td><td>6422</td><td>115722</td><td>351094</td><td>13858</td></tr><tr><td rowspan="2">Multi-OpenEAEN-FR</td><td>EN (English)</td><td>15000</td><td>267</td><td>308</td><td>47334</td><td>73121</td><td>15000</td><td rowspan="2">15000</td></tr><tr><td>FR (French)</td><td>15000 210</td><td></td><td>404</td><td>40864</td><td>67167</td><td>15000</td></tr><tr><td rowspan="2">Multi-OpenEAEN-DE</td><td>EN (English)</td><td>15000</td><td>215</td><td>286</td><td>47676</td><td>83755</td><td>15000</td><td rowspan="2">15000</td></tr><tr><td>DE (German)</td><td>15000 131</td><td></td><td>194</td><td>50419</td><td>156150</td><td>15000</td></tr><tr><td rowspan="2">Multi-OpenEADW-V1</td><td>DBpedia</td><td>15000</td><td>248</td><td>342</td><td>38265</td><td>68258</td><td>15000</td><td rowspan="2">15000</td></tr><tr><td>Wikidata</td><td>15000 169</td><td></td><td>649</td><td>42746</td><td>138246</td><td>15000</td></tr><tr><td rowspan="2">Multi-OpenEADW-V2</td><td>DBpedia</td><td>15000</td><td>167</td><td>175</td><td>73983</td><td>66813</td><td>15000</td><td rowspan="2">15000</td></tr><tr><td>Wikidata</td><td>15000</td><td>121</td><td>457</td><td>83365</td><td>175686</td><td>15000</td></tr><tr><td rowspan="2">FBDB15K</td><td>FB15K</td><td>14951</td><td>1345</td><td>116</td><td>592213</td><td>29395</td><td>13444</td><td rowspan="2">12846</td></tr><tr><td>DB15K</td><td>12842</td><td>279</td><td>225</td><td>89197</td><td>48080</td><td>12837</td></tr><tr><td rowspan="2">FBYG15K</td><td>FB15K</td><td>14951</td><td>1345</td><td>116</td><td>592213</td><td>29395</td><td>13444</td><td rowspan="2">11199</td></tr><tr><td></td><td></td><td></td><td></td><td>122886</td><td>23532</td><td></td></tr><tr><td></td><td>YAGO15K</td><td>15404</td><td>32</td><td>7</td><td></td><td></td><td>11194</td><td></td></tr></table>

$$
M R R = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { 1 } { r a n k _ { i } }\tag{17}
$$

## A.4 Implementation Details

For a fair comparison with recent works (Chen et al., 2023a,b), we have standardized our experimental setup as follows:

• Iterative Learning: We adopt the approach of (Lin et al., 2022) and employ a probation method for iterative training. This method collects pairs of mutual nearest entities every 5 epochs. Pairs that consistently remain nearest neighbors for 10 rounds are added to the training set.

• Multi-modal Encoders: For relations and attributes, we use the Bag-of-Words (BoW) model to encode relations (x ) and attributes $( x _ { a } )$ as fixed-length vectors with a dimension of $d _ { r } ~ = ~ 1 0 0 0$ . For entity names’ surface information, we use 300-dimensional GloVe vectors and character bigrams. We enhance this method by applying machine translations to entity names, following (Mao et al.,

2021). For images, we use ResNet-152 (He et al., 2016) as pre-trained visual encoders with a vision feature dimension $( d _ { v } )$ of 2048 in DBP15K. For Multi-OpenEA, CLIP (Radford et al., 2021) is used with $d _ { v } = 5 1 2$ . In FBDB15K/FBYG15K, VGG-16 (Simonyan and Zisserman, 2014) is employed with $d _ { v } =$ 4096. It should be noted that we assign a zero vector as the embedding for missing images to minimize their impact on cross-modal learning.

• Hyper Parameters: We unify the hidden layer dimensions across all networks to 300. The parameter τ is set to 0.05 to emphasize challenging negative samples in the contrast loss. The dynamic threshold δ in Eq. 3 for progressively freezing entities begins at 0.1 and increases by 1.2 until it reaches a maximum of 0.9. For the structure, relation, attribute, and image modalities, $\beta _ { m }$ values in Eq. 11 are respectively set to 0.1, 0.1, 0.1, and 10, scaling the learning rate of modality features.

• Computational Overhead: Our experiments, conducted on a Tesla V100 SXM2 32GB GPU, demonstrate the superior efficiency of our model, which completes training in only 20 minutes. This is notably faster than the baseline model, MEAformer (Chen et al., 2023a), at 33 minutes, and UMAEA (Chen et al., 2023b) at 29 minutes. The number of learnable parameters is 13.1M for DBP15K and Multi-OpenEA, and 9.9M for MMKG.

## B Detailed analysis

## B.1 Low Resource

![](images/bf85cf08afe6b71cc0e50d40a3a92b740ef0ff915d547fe796d247da5cdfa9dd.jpg)  
Figure 7: H@1 performance with different ratios of seed alignments ranging from 0.01 to 0.3 on DPB1 $5 \mathsf { K } _ { Z H - E N }$ (left) and FBDB15K (right). The xaxis denotes ratios and the y-axis denotes H@1.

To assess the stability of the proposed method under conditions of limited seed alignments, we conducted evaluations on two distinct datasets– $\mathrm { D B P 1 5 K } _ { Z H - E N }$ and FBDB15K, using seed alignment ratios varying from 0.01 to 0.30. Figure 7 illustrates a clear gap between performances as the ratio escalates. Notably, with merely 1% seed alignments in the DBP1 $5 { \mathrm { K } } _ { Z H - E N }$ dataset, our model was able to achieve a H@1 score of .648, markedly outperforming the MEAformer, which scored .456. This result highlights the model’s robustness and its considerable promise for applications in few-shot entity alignment.

## B.2 Analysis of Freezing Process

![](images/66efbad09bcf965d8733e7e48610a6299e34880b32e19165815aa0d355323798.jpg)  
Figure 8: The ratio of entities with images that are frozen during the training epochs in DBP15K.

To delve deeper into the impacts of the progressive manner, we take the freezing process of image modality on the DBP15K dataset as an example, analyzing the trend of the proportion of entities being frozen as training epochs change. Figure 8 indicates that as training epochs increase, the proportion of entities frozen shows an exponential rising trend, ultimately stabilizing at about 55%. The high rate of frozen images is partly due to a high image missing rate of up to 30% in the DBP15K dataset, and it also reflects high ambiguity in image information within multi-modal knowledge graphs.