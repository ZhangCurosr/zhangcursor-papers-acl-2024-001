# Hyper-CL: Conditioning Sentence Representations with Hypernetworks

Young Hyun Yoo†, Jii Cha†, Changhyeon Kim, Taeuk Kim∗ Hanyang University, Seoul, Republic of Korea {somebodil,skchajie,livex,kimtaeuk}@hanyang.ac.kr

## Abstract

While the introduction of contrastive learning frameworks in sentence representation learning has significantly contributed to advancements in the field, it still remains unclear whether state-of-the-art sentence embeddings can capture the fine-grained semantics of sentences, particularly when conditioned on specific perspectives. In this paper, we introduce Hyper-CL, an efficient methodology that integrates hypernetworks with contrastive learning to compute conditioned sentence representations. In our proposed approach, the hypernetwork is responsible for transforming pre-computed condition embeddings into corresponding projection layers. This enables the same sentence embeddings to be projected differently according to various conditions. Evaluation of two representative conditioning benchmarks, namely conditional semantic text similarity and knowledge graph completion, demonstrates that Hyper-CL is effective in flexibly conditioning sentence representations, showcasing its computational efficiency at the same time. We also provide a comprehensive analysis of the inner workings of our approach, leading to a better interpretation of its mechanisms. Our code is available at https://github.com/HYU-NLP/Hyper-CL.

## 1 Introduction

Building upon the established correlation between language model performance and computational capacity (Kaplan et al., 2020), there has emerged an undeniable trend towards the adoption of everlarger language models across a diverse range of NLP applications. This trend is also evident in the computation of sentence or text representations. Despite the ongoing popularity of compact encoders such as BERT (Devlin et al., 2019) and RoBERTa (Liu et al., 2019), there is a growing inclination to leverage the capabilities of recent, larger language models, e.g., LLaMA-2 (Touvron et al., 2023), even breaking from the conventional roles of encoders and decoders. Consequently, the enduring challenge of finding a balance between performance and computational cost—a persistent issue in sentence representation learning (Reimers and Gurevych, 2019)—continues to be elusive.

![](images/388e4b3c1cecb933368412634c8267608464590cb04601b599548e8a3315480d.jpg)  
Figure 1: Illustration of our approach dubbed Hyper-CL. In the example, two sentences are provided along with two distinct conditions, $c _ { h i g h }$ and $c _ { l o w } .$ Specifically, $c _ { h i g h }$ (orange) denotes a condition that results in the sentences being interpreted more similarly, whereas $c _ { l o w }$ (blue) leads to a perspective in which the two sentences are understood as being relatively more distinct. The identical pair of sentences are projected into different subspaces that reflect the provided conditions.

In recent years, there has been a marked improvement in the quality of sentence embeddings, a progress primarily driven by the advent of contrastive learning frameworks (Kim et al., 2021; Gao et al., 2021; Chuang et al., 2022; inter alia). However, since the performance of these embeddings is generally evaluated based on their ability to encapsulate the overall meaning of the corresponding sentences—as measured by benchmarks like STS-B (Agirre et al., 2012; Cer et al., 2017) and MTEB (Muennighoff et al., 2023), it remains uncertain whether they adequately capture information relating to the various aspects of the source sentences.

![](images/93f48cfe7f68c87a8fd47081a50346b059f92af697508662a366d02e9b67323d.jpg)  
Figure 2: Four different types of architectures applicable for conditioning tasks. They utilize the [CLS] token embeddings from the encoder as representations of inputs. From left to right: the cross-encoder architecture encodes a triplet containing two sentences $( s _ { 1 } , s _ { 2 } )$ and a condition (c) as a whole. In the bi-encoder setting, two sentencecondition pairs $( s _ { 1 } , c )$ and $( s _ { 2 } , c )$ are processed individually. The tri-encoder configuration regard s , s , and c as independent and encode them separately, followed by extra merging operations (e.g., Hadamard product). Finally, Hyper-CL resembles the tri-encoder, but innovatively incorporates a hypernetwork responsible for constructing projection matrices to condition sentences s<sub>1</sub> and s<sub>2</sub>, based on the embedding of the condition c.

For instance, consider the sentences (1) “A cyclist pedals along a scenic mountain trail, surrounded by lush greenery” and (2) “A hiker navigates through a dense forest on a winding path, enveloped by the tranquility of nature”. In terms of “The mode oftransportation”, these sentences should be perceived as similar since both depict individuals engaging in outdoor activities, traversing natural landscapes. However, regarding “The speed of travel”, they should be differentiated, as cycling generally entails a faster pace than hiking. Deshpande et al. (2023) reported that current models for sentence embeddings face challenges in recognizing the fine-grained semantics within sentences. In other words, the existing models struggle to accurately detect the subtle shifts in sentence nuances that occur when conditioned on specific criteria.

In the literature, three prevalent approaches have been established for constructing conditioned representations (Deshpande et al., 2023), particularly in estimating their similarity (see Figure 2). The first is the cross-encoder approach, which encodes the concatenation of a pair of sentences (s<sub>1</sub>, s<sub>2</sub>) with a condition (c), i.e., $[ s _ { 1 } ; s _ { 2 } ; c ]$ .<sup>1</sup> The second method is the bi-encoder architecture, computing separate representations of sentences s<sub>1</sub> and s<sub>2</sub> with the condition c—[s ; c] and [s ; c]. Despite their simplicity, both approaches share a notable limitation: the representation must be computed for every unique combination of sentences plus a condition.

On the other hand, the tri-encoder architecture utilizes pre-computed embeddings of sentences s and s along with the condition c. It then employs a separate composition function responsible for merging the semantics of the sentence and condition. Considering that the embeddings for each component can be cached and reused, this approach offers enhanced long-term efficiency. The tri-encoder architecture, despite its potential, falls short in performance compared to the bi-encoder. This is primarily due to its inherent limitation, which is the inability to model explicit interactions between sentences and conditions during the representation construction process. Therefore, there is a need to propose a revised version of the tri-encoder architecture that improves its functionality without substantially sacrificing its efficiency.

In this work, we present Hyper-CL, a method that integrates Hypernetworks (Ha et al., 2017) with Contrastive Learning to efficiently compute conditioned sentence representations and their similarity. As illustrated in Figure 2d, our proposed approach is derived from the tri-encoder architecture. It introduces an additional hypernetwork tasked with constructing a condition-sensitive network on the fly. This network projects the original sentence embeddings into a specific condition subspace. Figure 1 illustrates the effectiveness of Hyper-CL in dynamically conditioning pre-computed sentence representations according to different perspectives.

We demonstrate the effectiveness of Hyper-CL by significantly reducing the performance gap with the bi-encoder architecture in the Conditional Semantic Textual Similarity (C-STS) and Knowledge Graph Completion (KGC) tasks. In particular, for C-STS, Hyper-CL demonstrates an improvement of up to 7.25 points in Spearman correlation compared to the original tri-encoder architecture. Furthermore, compared to the bi-encoder approach, our method shows superior efficiency by reducing the running time by approximately 40% on the C-STS dataset and 57% on the WN18RR dataset.

## 2 Background and Related Work

In this paper, conditioning refers to the presence of two or more signals, each represented as a natural language expression c. These signals impact the interpretation of a sentence $s ,$ highlighting a specific aspect of the sentence (Galanti and Wolf, 2020). Here, we describe two representative tasks that involve conditioning, along with an introduction to the concept of hypernetworks.

Conditional Semantic Textual Similarity (C-STS) (Deshpande et al., 2023) is a task composed of four elements in one quadruplet: two sentences $s _ { 1 }$ and $s _ { 2 } .$ , a condition c to consider when calculating the similarity between the two sentences, and a similarity score $y .$ Unlike the original Semantic Textual Similarity (STS) dataset (Agirre et al., 2012; Cer et al., 2017), C-STS computes similarity scores for the same sentence pair $s _ { 1 }$ and $s _ { 2 }$ under two distinct conditions $c _ { h i g h }$ and $c _ { l o w }$ . The similarity scores of the conditioned sentences are expected to be high for $c _ { h i g h }$ and low for $c _ { l o w }$ . Therefore, a model for this task is required to compute distinct representations for the same sentence under two different viewpoints. To this end, a few basic architectures illustrated in Figure 2 have been proposed by Deshpande et al. (2023). Our objective is to present a revision to the previous methods, pursuing the balance between performance and computational efficiency.

Knowledge Graph Completion (KGC) is the task focused on automatically inferring missing relationships or entities in a knowledge graph. The knowledge graph is represented as a set of triplets $( h , r , t )$ , consisting of a head entity $h ,$ a relation $^ { r , }$ and a tail entity t. Link prediction, a subtask in

KGC,<sup>2</sup> aims to uncover unestablished yet plausible and novel relationships between entities (Bordes et al., 2013; Toutanova and Chen, 2015). When given a head entity and a relation, the task of identifying the most suitable tail entity is known as head entity prediction. Conversely, the task of determining the appropriate head entity when a tail entity and relation are provided is termed tail entity prediction.

While two types of methodologies are generally available for KGC—embedding-based methods and text-based methods—our primary focus is on text-based methods that rely on the processing of textual information by language models. We further categorize them into three types: cross-encoder, encoder-decoder, and bi-encoder architectures. Approaches such as KG-BERT (Yao et al., 2019) and MTL-KGC (Kim et al., 2020), which concatenate all triple elements (i.e., $\left[ h ; r ; t \right] )$ , are classified as cross-encoder. Methods like StAR (Wang et al., 2021) and SimKGC (Wang et al., 2022), which separately embed $[ h ; r ]$ and t in tail prediction tasks, are classified as bi-encoder. Lastly, GenKGC (Xie et al., 2022) and KG-S2S (Chen et al., 2022), which directly generate tail entity text based on the remaining [h; r], are classified as encoder-decoder.

On the other hand, hypernetworks refer to a type of neural network that generates the weights or parameters for another neural network, known as the primary network (Ha et al., 2017; Chauhan et al., 2023; Majumdar et al., 2023). In essence, a hypernetwork enables the dynamic construction of the primary network, allowing its function to adapt flexibly based on the input or condition. For instance, Galanti and Wolf (2020) demonstrate that, even with a compact primary network, hypernetworks can effectively learn and apply diverse functions for various inputs, provided the hypernetwork itself is sufficiently large. In our settings, we also endeavor to harness the advantages of hypernetworks, ensuring that conditioned sentence representations are dynamically computed and adapted in response to changing conditions.

## 3 Proposed Method: Hyper-CL

## 3.1 Motivation

As briefly mentioned in §1, current approaches to sentence conditioning demonstrate a clear trade-off between performance and computational efficiency.

![](images/57f2c20c980601908ce2c19064896483732cfe7b3230662327438a16522cd0f9.jpg)  
Figure 3: Training procedure of Hyper-CL. It introduces a hypernetwork q to construct the weights of multi-layer perceptrons (MLPs), i.e., g, based on the condition. The MLPs are then used to project sentence embeddings onto subspaces, resulting in condition-aware sentence embeddings. Hyper-CL is trained with a contrastive objective, utilizing pairs of condition-aware sentence embeddings, one with a high condition $c _ { h i g h }$ and the other with a low condition $c _ { l o w }$ . Note that every embedding is the output of the same encoder $f .$

Approaches that enable direct interaction between a sentence and a condition within encoders—the cross-encoder and bi-encoder architectures—gain enhanced performance but at the cost of reduced efficiency. In contrast, the tri-encoder architecture, while enabling efficient conditioning with pre-computed sentence and condition embeddings, tends to have inferior performance compared to its counterparts. Note that this trade-off becomes more pronounced as the number of sentences and conditions to be processed increases.

Formally, consider a computationally intensive language model-based encoder $f ,$ and a less demanding composition network $g _ { \colon }$ which is required in the case of the tri-encoder architecture. In terms of the bi-encoder, obtaining the set of every possible conditioned embedding $\mathcal { H } = \{ h _ { s c } | \forall s \forall c , s \in$ $\cal { S } , c \in { \mathcal { C } } \}$ necessitates $| S | \times | C |$ repetitions of the heavy operation imposed by $f ,$ where  denotes the total number of sentences and  represents the total number of possible conditions. In contrast, the tri-encoder architecture only requires heavy operations by f for each sentence s and condition $c$ just once. This implies that only + heavy operations are needed, followed by $| S | \times | C |$ lightweight operations by g to obtain the conditioned embeddings. As a result, the tri-encoder and its variants become more efficient if the cost of computing g is markedly lower than that of $f$ , thereby amortizing the cost for computing $\mathcal { H }$

In this work, we aim to develop a new architecture for sentence conditioning that inherits the efficiency merits of the tri-encoder architecture, while simultaneously outperforming the original in terms of performance. To achieve this, we propose the use of hypernetworks to implement $g _ { \colon }$ , facilitating the dynamic construction of conditioning networks while maintaining reasonable cost-efficiency.

## 3.2 Framework and Training Procedure

The framework of Hyper-CL and its training procedure are listed as follows (also see Figure 3):

1. First, it computes the embeddings of a sentence s and a condition c using the same embedding model $f \colon h _ { s } = f ( s )$ and $h _ { c } = f ( c )$

2. Given the condition embedding $h _ { c } ,$ a hypernetwork $q : \mathbb { R } ^ { N _ { h } }  \mathbb { R } ^ { N _ { h } \times N _ { h } }$ outputs a linear transformation matrix $W _ { c }$ for conditioning: $W _ { c } = q ( h _ { c } ) . ^ { 3 }$

3. We encode the condition-aware sentence embedding $h _ { s c }$ based on the matrix $W _ { c }$ and the sentence embedding $h _ { s } \colon h _ { s c } = W _ { c } \cdot h _ { s }$

4. For training, we perform contrastive learning with the conditional sentence embeddings $h _ { s c } .$ whose details are explained in the following.

## 3.3 Contrastive Learning in Subspaces

The conditioning network composed of $W _ { c }$ is a linear neural network. In other words, it can be interpreted as a linear transformation function $g$ : $\mathbb { R } ^ { N _ { h } } \xrightarrow { ^ { - } } \mathbb { R } ^ { N _ { h } }$ , mapping from the original semantic space of sentence embeddings to a specialized condition subspace. We demonstrate that conducting contrastive learning within the subspace of specific viewpoints yields greater effectiveness compared to performing the same process in the general space. We apply separate task-oriented contrastive learning objectives for the tasks, C-STS and KGC.

C-STS The C-STS task entails providing conditions $c _ { h i g h }$ and $c _ { l o w }$ for two sentences, s and $s _ { 2 }$ This setup induces different interpretations of the relationship between the two sentences—one being more similar under $c _ { h i g h }$ and the other more dissimilar under $c _ { l o w }$ . In a given instance from the dataset,

Hyper-CL generates two pairs of conditioned sentence embeddings, i.e., $( h _ { s _ { 1 } c _ { h i g h } } , h _ { s _ { 2 } c _ { h i g h } } )$ and $( h _ { s _ { 1 } c _ { l o w } } , h _ { s _ { 2 } c _ { l o w } } )$ . Since these pairs correspond to positive and negative pairs in the contrastive objective, we directly utilize them for training.

Considering that the C-STS dataset already contains gold-standard similarity values for the two sentences under each condition, it seems reasonable to employ the Mean Squared Error (MSE) objective in conjunction with contrastive learning. However, as training progresses, we can speculate that MSE objectives that utilize labels will provide relatively more fine-grained granularity compared to contrastive objectives that do not. Therefore, to mitigate the relatively strong impact of the contrastive objective, we apply the InfoNCE (Oord et al., 2018) loss with high temperature, as follows:

$$
\begin{array} { r } { L _ { C L } = - \log \frac { e ^ { \phi ( h _ { s _ { 1 } c _ { h i g h } } , h _ { s _ { 2 } c _ { h i g h } } ) / \tau } } { e ^ { \phi ( h _ { s _ { 1 } c _ { h i g h } } , h _ { s _ { 2 } c _ { h i g h } } ) / \tau } + e ^ { \phi ( h _ { s _ { 1 } c _ { l o w } } , h _ { s _ { 2 } c _ { l o w } } ) / \tau } } , } \end{array}
$$

where $\phi$ is the cosine similarity function and τ is a temperature hyperparameter. The MSE objective is as follows:

$$
L _ { M S E } = \| \phi ( h _ { s _ { 1 } c } , h _ { s _ { 2 } c } ) - y \| _ { 2 } ^ { 2 } ,
$$

where c can be either $c _ { h i g h }$ or $c _ { l o w }$ . By combining the two above formulas, the final form of our training objective for C-STS becomes:

$$
{ \cal L } = { \cal L } _ { M S E } + { \cal L } _ { C L } .
$$

Note that L is averaged over data instances in the training set.

KGC We follow the setting of SimKGC (Wang et al., 2022), except that we leverage Hyper-CL instead of the bi-encoder architecture. For each triplet of head entity, relation, and tail entity (h, r, t), we treat entities as sentences and relations as conditions, framing KGC as a conditioning task. Furthermore, given that Hyper-CL is simple and flexible enough to adopt various techniques from SimKGC, such as the use of self-negative, pre-batch negatives, and in-batch negatives, we decide to apply these tricks to our method as well. In conclusion, the final training objective for KGC is as follows (refer to Wang et al. (2022) for more details):

$$
\begin{array} { r } { L _ { C L } = - \log \frac { e ^ { ( \phi ( h _ { h r } , h _ { t } ) - \gamma ) / \tau } } { e ^ { ( \phi ( h _ { h r } , h _ { t } ) - \gamma ) / \tau } + \sum _ { j = 1 } ^ { N } e ^ { \phi ( h _ { h r } , h _ { t ^ { \prime } } ) / \tau } } } \end{array}
$$

where $h _ { h r } , h _ { t }$ and $h _ { t } ^ { \prime }$ are the relation-aware head embedding, tail embedding, random embedding (i.e., self-negative, pre-batch negative, and in-batch negative), respectively. The relation-aware head embedding corresponds to a conditional sentence embedding. γ is an additive margin, $\phi$ is the cosine similarity, and τ is a learnable parameter.

## 3.4 Optimization of Hypernetworks

In the original formulation presented in §3.2, the number of parameters for the hypernetwork $q :$ $\mathbb { R } ^ { N _ { h } }  \mathbb { R } ^ { \hat { N _ { h } } \times N _ { h } }$ is the cube of $N _ { h }$ , which could lead to cost inefficiency. To address this issue, we propose decomposing the network into two lowrank matrices, drawing inspiration from low-rank approximation (Yu et al., 2017; Hu et al., 2021). In particular, we introduce two smaller hypernetworks of the same size: $q _ { 1 } : \mathbb { R } ^ { N _ { h } }  \mathbb { R } ^ { N _ { h } \times \hat { N } _ { K } }$ and $q _ { 2 } : \mathbb { R } ^ { N _ { h } }  \mathbb { R } ^ { N _ { h } \times N _ { K } }$ to generate two low-rank matrices $W _ { c _ { 1 } } = q _ { 1 } ( h _ { c } )$ and $W _ { c _ { 2 } } = q _ { 2 } ( h _ { c } )$ , where $N _ { k }$ is much smaller than $N _ { h }$ . We then obtain the final matrix through their product: ${ \cal W } _ { c } = { \cal W } _ { c 1 } { \cal W } _ { c 2 } ^ { T } . ^ { 4 }$

## 3.5 Caching Conditioning Networks

In the tri-encoder architecture, once computed, the sentence and condition embeddings $h _ { s }$ and $h _ { c }$ can be cached and subsequently reused whenever conditioned embeddings related to them need to be computed. Hyper-CL, building upon the tri-encoder framework, not only inherits this advantage but also further improves time efficiency by caching the parameters of the entire conditioning networks $W _ { c } = q ( h _ { c } )$ generated by the hypernetwork. It is important to note that this approach is viable because the computation of the matrix $W _ { c }$ depends solely on $h _ { c } ,$ without requiring any other inputs.

## 4 Experiments

We apply Hyper-CL to various embedding models (i.e., encoders) and fine-tune them on the target task, denoting the resulting models with the subscript hyper-cl. If the rank of the hypernetwork $( N _ { k } )$ is different from $N _ { h }$ , we denote this value as hyperK-cl. We set K as 64 and 85, for the base and large models respectively. A detailed explanation for the selection of $K$ can be found in the appendix B. We show the effectiveness of Hyper-CL by evaluating it on two downstream tasks.

## 4.1 Conditional Semantic Textual Similarity

We use DiffCSE (Chuang et al., 2022) and SimCSE (Gao et al., 2021), adaptations of RoBERTa (Liu et al., 2019), for the embedding model f. Note that the key difference between the original tri-encoder architecture and Hyper-CL lies in the composition network, g. The original uses the simple Hadamard product, while Hyper-CL employs hypernetworks to learn linear layers for this composition.

<table><tr><td>Method</td><td># Params</td><td>Spearman</td><td>Pearson</td></tr><tr><td>tri-encoder architectures</td><td></td><td></td><td></td></tr><tr><td> $\mathrm { D i f f C S E } _ { b a s e } ^ { \dagger }$ </td><td>125M</td><td> $2 8 . 9 _ { 0 . 8 }$ </td><td> $2 7 . 8 _ { 1 . 2 }$ </td></tr><tr><td> $\boldsymbol { * } \mathrm { D i f f } \mathrm { C S E } _ { b a s e + h y p e r 6 4 - c l }$ </td><td>200M</td><td> $3 3 . 1 0 _ { 0 . 2 }$ </td><td> $3 1 . 6 8 _ { 0 . 6 }$ </td></tr><tr><td> $\boldsymbol { * } \mathrm { D i f f } \mathrm { C S E } _ { b a s e + h y p e r - c l }$ </td><td>578M</td><td> $3 3 . 8 2 _ { 0 . 1 }$ </td><td> $3 3 . 1 0 _ { 0 . 3 }$ </td></tr><tr><td> $\mathrm { S i m C S E } _ { b a } ^ { \dagger }$ </td><td>125M</td><td> $3 1 . 5 _ { 0 . 5 }$ </td><td> $3 1 . 0 \mathrm { _ { 0 . 5 } }$ </td></tr><tr><td>base  $* \mathrm { S i m C S E } _ { b a s e + h y p e r 6 4 - c l }$ </td><td>200M</td><td> $3 8 . 3 6 _ { 0 . 1 }$ </td><td> $3 7 . 5 3 \mathrm { _ { 0 . 0 4 } }$ </td></tr><tr><td> $\ast \mathrm { S i m C S E } _ { b a s e + h y p e r - c l }$ </td><td>578M</td><td> $3 8 . 7 5 _ { 0 . 3 }$ </td><td> $3 8 . 3 8 _ { 0 . 3 }$ </td></tr><tr><td> $\mathrm { S i m C S E } _ { l a r g e } ^ { \dagger }$ </td><td>355M</td><td> $3 5 . 3 \mathrm { _ { 1 . 0 } }$ </td><td> $3 5 . 6 \phantom { 0 } _ { 0 . 9 }$ </td></tr><tr><td> $* \mathrm { S i m } { \mathrm { C S E } } _ { l a r g e + h y p e r 8 5 - c l }$ </td><td>534M</td><td> $3 8 . 1 2 _ { 1 . 4 }$ </td><td> $3 7 . 4 7 _ { 1 . 4 }$ </td></tr><tr><td> $* \mathrm { S i m C S E } _ { l a r g e + h y p e r - c l }$ </td><td>1431M</td><td> $3 9 . 6 0 _ { 0 . 2 }$ </td><td> $3 9 . 9 6 _ { 0 . 3 }$ </td></tr><tr><td colspan="4">bi-encoder architectures</td></tr><tr><td> $\mathrm { D i f f C S E } _ { b a s e } ^ { \dagger }$ </td><td>125M</td><td> $4 3 . 4 _ { 0 . 2 }$ </td><td> $4 3 . 5 _ { 0 . 2 }$ </td></tr><tr><td> $\mathrm { S i m C S E } _ { b a s e } ^ { \dagger }$ </td><td>125M</td><td> $4 4 . 8 \phantom { 0 } _ { 0 . 3 }$ </td><td> $4 4 . 9 _ { 0 . 3 }$ </td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td> $\mathrm { S i m C S E } _ { l a r g e } ^ { \dagger }$ </td><td>355M</td><td> $4 7 . 5 _ { 0 . 1 }$ </td><td> $4 7 . 6 _ { 0 . 1 }$ </td></tr></table>

Table 1: Performance on C-STS measured by Spearman and Pearson correlation coefficients. The best results are in bold for each section. \*: indicates the results of Hyper-CL. : denotes results from Deshpande et al. (2023).

Table 1 summarizes the results of Hyper-CL in addition to baselines on C-STS. Compared to the tri-encoder baselines, Hyper-CL demonstrates improvements with up to a 7.25-point increase in Pearson correlation when based on $\mathrm { S i m C S E } _ { b a s e }$ This reduces the performance gap between the biencoder and tri-encoder from 13.3 to 6.05 points.

In addition, even when Hyper-CL is developed with low-rank approximation (i.e., hyper64-cl, hyper85-cl), its performance remains consistent. This indicates that the memory usage of hypernetworks can be effectively controlled, while both performance and time efficiency are maintained.

## 4.2 Knowledge Graph Completion

In KGC, the link prediction task entails computing relation-aware embeddings for head or tail entities and subsequently retrieving the top-K embeddings based on their similarity scores. We consider two datasets for KGC: WN18RR (Bordes et al., 2013) and FB15k-237 (Toutanova and Chen, 2015).

We employ text-based KGC models as baselines for evaluation. Specifically, we use the SimKGC model that leverages all negatives (i.e., in-batch negative, pre-batch negative, and self-negative), which is also true when applying Hyper-CL to

SimKGC. We consider an extra baseline of applying the tri-encoder architecture to SimKGC with different $g { - }  { - } ( 1 )$ Hadamard: performs the Hadamard product between the representations of a condition c and a sentence s: $g _ { 1 } ( h _ { c } , h _ { s } ) = h _ { c } \odot h _ { s }$ (2) Concatenation: merge the two vectors, apply a dropout function, and halve the dimension using a linear layer: $g _ { 2 } ( h _ { c } , h _ { s } ) = W \cdot d ( [ h _ { c } ; h _ { s } ] )$

Table 2 presents the outcomes of our method and baselines on KGC, measured by MRR (Mean Reciprocal Rank) and Hits@K. $3 \mathrm { E R T } _ { b a s e }$ is leveraged for the embedding model f. As a result, $\mathrm { S i m K G C } _ { h y p e r - c l }$ , representing the application of Hyper-CL to SimKGC, shows that there is no significant difference in performance compared to the original SimKGC, especially in terms of Hits@10. Even for $\mathrm { S i m K G C } _ { h y p e r 6 4 - c l } .$ , while there is a slight decrease in performance, it still yields competitive results and does not significantly trail behind other baselines. Moreover, the performance of other methods in the tri-encoder architecture falls significantly short of Hyper-CL’s. It is worth noting that our implementation is based on the tri-encoder architecture, which guarantees significantly more efficiency in running time compared to the original SimKGC. The details of this analysis are in §5.1.

## 5 Analysis

## 5.1 Efficiency Comparison between Bi-Encoder and Tri-Encoder

To assess the running time efficiency of Hyper-CL, enabled by its caching capability, we compare the execution time of our method with that of the biencoder and tri-encoder architectures. We measure execution time, cache hit rate, and memory usage in scenarios with caching enabled. Initially, the cache is empty, and embeddings are added to the cache upon each cache miss.

<table><tr><td rowspan="2">Method</td><td colspan="4">WN18RR</td><td colspan="4">FB15K-237</td></tr><tr><td>MRR</td><td>Hits@1</td><td>Hits@3</td><td>Hits@10</td><td>MRR</td><td>Hits@1</td><td>Hits@3</td><td>Hits@10</td></tr><tr><td colspan="9">cross-encoder architectures</td></tr><tr><td>KG-BERT†</td><td>0.216</td><td>0.041</td><td>0.302</td><td>0.524</td><td></td><td></td><td></td><td>0.420</td></tr><tr><td>MTL-KGC†</td><td>0.331</td><td>0.203</td><td>0.383</td><td>0.597</td><td>0.267</td><td>0.172</td><td>0.298</td><td>0.458</td></tr><tr><td colspan="9">encoder-decoder architectures</td></tr><tr><td>GenKGC†</td><td></td><td>0.287</td><td>0.403</td><td>0.535</td><td></td><td>0.192</td><td>0.355</td><td>0.439</td></tr><tr><td>KG-S2S†</td><td>0.574</td><td>0.531</td><td>0.595</td><td>0.661</td><td>0.336</td><td>0.257</td><td>0.373</td><td>0.498</td></tr><tr><td colspan="9">bi-encoder architectures</td></tr><tr><td>StAR†</td><td>0.401</td><td>0.243</td><td>0.491</td><td>0.709</td><td>0.296</td><td>0.205</td><td>0.322</td><td>0.482</td></tr><tr><td>SimKGC‡</td><td>0.666</td><td>0.587</td><td>0.717</td><td>0.800</td><td>0.336</td><td>0.249</td><td>0.362</td><td>0.511</td></tr><tr><td colspan="9">tri-encoder architectures</td></tr><tr><td>SimKGChadamard-product</td><td>0.164</td><td>0.004</td><td>0.243</td><td>0.481</td><td>0.153</td><td>0.092</td><td>0.162</td><td>0.274</td></tr><tr><td>SimKGCconcatenation</td><td>0.335</td><td>0.226</td><td>0.382</td><td>0.550</td><td>0.271</td><td>0.193</td><td>0.292</td><td>0.430</td></tr><tr><td> $\mathrm { \mathrm { } } ^ { \ast } \mathrm { S i m K G C } _ { h y p e r - c l }$ </td><td>0.616</td><td>0.506</td><td>0.690</td><td>0.810</td><td>0.318</td><td>0.231</td><td>0.344</td><td>0.496</td></tr><tr><td> $\mathrm { ^ { * } S i m K G C } _ { h y p e r 6 4 - c l }$ </td><td>0.548</td><td>0.427</td><td>0.626</td><td>0.770</td><td>0.305</td><td>0.219</td><td>0.331</td><td>0.479</td></tr></table>

Table 2: Results on the WN18RR and FB15K-237 datasets for KGC, measured by MRR and Hits@K. The best results are highlighted in bold, while the next best results are underlined for each column. \*: indicates the results of applying Hyper-CL. : denotes results from Chen et al. (2022). : denotes results from Wang et al. (2022). Other results are implemented and evaluated by the authors.

Specifically, we estimate the inference time required for the C-STS and WN18RR datasets. To simulate a realistic scenario where the number of data samples to be processed is substantial, we utilize all dataset splits (training, validation, and test) from the datasets. For the C-STS dataset, we also evaluate the results of $\mathrm { S i m C S E } _ { h y p e r 6 4 - c l }$ as a solution for cases with large cache sizes. For Hyper-CL, the transformation matrix $W _ { c }$ is saved instead of embeddings. We set the batch size as 1 to minimize the overhead of cache storage and retrieval.

As observed in Table 3 and 4, Hyper-CL reduces the running time by approximately 40% on the C-STS dataset and 57% on the WN18RR dataset than the bi-encoder architecture. Compared to the naïve tri-encoder architecture, Hyper-CL requires slightly more time, but we believe this is acceptable given its significantly improved performance.

In terms of cache hit rate, the tri-encoder architecture, including Hyper-CL, achieves a much higher rate as it embeds each input separately, whereas the bi-encoder architecture has a lower rate because both the sentence and condition combination must match to result in a cache hit. Considering the significant gap in cache hit ratio between the bi-encoder and tri-encoder architectures, we expect that the efficiency of Hyper-CL will be more pronounced when deployed to process real-time streaming data from a large pool of users. In such scenarios, the diversity of input sentences and their conditions (relations) would be much higher than in our experimental settings, implying that the efficiency gap between the bi-encoder and tri-encoder architectures will be more severe.<sup>5</sup>

## 5.2 Analysis of Embedding Clusters

In this subsection, we explore the impact of Hyper-CL on generating conditioned sentence representations by visualizing the computed embeddings and analyzing them using clustering tools.

We visualize the vector space of sentence embeddings before and after transformation by $W _ { c }$ , which is the weight of a linear layer from the Hyper-CL’s hypernetwork. This helps us observe if embeddings based on the same condition cluster after transformation, indicating proper differentiation based on conditions. For this analysis, we choose three sets of 20 random sentences from the C-STS validation dataset, with sentences within each group sharing the same conditions. The three conditions we select are: ‘The number of people’, ‘The sport’, and ‘The name of the object’. As expected, Figure 4 shows that sentence embeddings transformed with the same $W _ { c _ { i } }$ form clusters, meaning each embedding has projected to respective subspaces.

<table><tr><td>Method</td><td>Time</td><td>HitRate</td><td>Cache</td></tr><tr><td colspan="4">bi-encoder architectures</td></tr><tr><td> $\mathrm { S i m C S E } _ { b a s e }$ </td><td>791.71s</td><td>1.46%</td><td>110.87MB</td></tr><tr><td> $\mathrm { S i m C S E } _ { l a r g e }$ </td><td>1498.65s</td><td>1.46%</td><td>147.26MB</td></tr><tr><td colspan="4">tri-encoder architectures</td></tr><tr><td> $\mathrm { S i m C S E } _ { b a s e }$ </td><td>441.17s</td><td>64.11%</td><td>60.57MB</td></tr><tr><td>SimCSEbase+hyper64-cl</td><td>525.62s</td><td>64.11%</td><td>2.17GB</td></tr><tr><td>SimCSEbase+hyper-cl</td><td>541.55s</td><td>64.11%</td><td>12.81GB</td></tr><tr><td> $\mathrm { S i m C S E } _ { l a r g e }$ </td><td>832.19s</td><td>64.11%</td><td>80.45MB</td></tr><tr><td>SimCSElarge+hyper64-cl  $\mathrm { S i m C S E } _ { l a r g e + h y p e r - c l }$ </td><td>990.94s 960.84s</td><td>64.11% 64.11%</td><td>3.82GB 22.75GB</td></tr></table>

Table 3: Analysis of inference time, cache hit rate, and memory usage for different architectures and methods on the entire C-STS dataset.
<table><tr><td>Method</td><td>Time</td><td>HitRate</td><td>Cache</td></tr><tr><td>bi-encoder architectures</td><td></td><td></td><td></td></tr><tr><td> $\mathrm { S i m K G C } _ { b a s e }$ </td><td>994.41s 46.65%</td><td></td><td>295.29MB</td></tr><tr><td> $\mathrm { S i m K G C } _ { l a r g e }$ </td><td>1806.18s46.65%</td><td></td><td>392.2MB</td></tr><tr><td>tri-encoder architectures</td><td></td><td></td><td></td></tr><tr><td>SimKGCbase+hadamard</td><td>435.571s85.32%121.86MB</td><td></td><td></td></tr><tr><td>SimKGCbase+concatenation</td><td>449.46s 85.32% 121.86MB</td><td></td><td></td></tr><tr><td>SimKGCbase+hyper-cl</td><td>448.955s85.32%146.57MB</td><td></td><td></td></tr><tr><td>SimKGClarge+hadamard</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>781.45s 85.32%161.85MB</td><td></td></tr><tr><td>SimKGClarge+concatenation  $\mathrm { S i m K G C } _ { l a r g e + h y p e r - c l }$ </td><td>774.41s 85.32%205.81MB</td><td>783.228s 85.32% 161.85MB</td><td></td></tr></table>

Table 4: Analysis of inference time, cache hit rate, and memory usage for different architectures and methods on the entire WN18RR dataset.

We complement the visual analysis with a quantitative evaluation. We perform K-means clustering on the sentence embeddings before and after transformation.<sup>6</sup> Following the clustering, we compute the average impurity (entropy) of each sentence group, where a lower value suggests better conditioning of sentence embeddings. Formally, the impurity I based on the entropy of each (conditional) sentence group $E ( i )$ is given by:

$$
\begin{array} { r } { I = \sum _ { i } \frac { | C _ { i } | } { | S | } E ( i ) = - \sum _ { i } \frac { | C _ { i } | } { | S | } \sum _ { j } \frac { | L _ { i j } | } { | C _ { i } | } \log \frac { | L _ { i j } | } { | C _ { i } | } , } \end{array}
$$

![](images/8ea433d0a07e5b142e516acb0df86c7d2eb1c65c5020c73ceec2c6f3b2c46574.jpg)

![](images/363ca6867de7f1c2f80670b82cbe6523b6d734bfb09d1e84ae877e9ea4b8ec73.jpg)  
Figure 4: Visualization of the clusters of sentence embeddings before (top) and after (bottom) projection onto condition subspaces by Hyper-CL.

where S is the total number of sentences, $| C _ { i } |$ is the number of sentences that should be labeled as condition i, and $| L _ { i j } |$ represents the number of examples clustered as j by K-means clustering within the $i ^ { t h }$ (conditional) sentence group. We discover that after projection done by Hyper-CL, I changes from 0.739 to 0.270, indicating that Hyper-CL effectively projects sentence embeddings into distinct subspaces based on different conditions.

## 5.3 Generalization Capabilities of Hyper-CL

We examine the generalization capabilities of Hyper-CL, focusing on its ability to generalize to unseen conditions during training. This enables a fine-grained evaluation of the conditioning ability and confirms its feasibility in realistic settings.

For the targeted study, we define two separate subsets of the C-STS validation set.<sup>7</sup> The first category is referred to as the ‘unseen’ dataset, which consists only of data instances with conditions not present during training. The second is named the ‘seen’ dataset, which comprises data instances with conditions already seen in the training phase. Statistically, the number of data instances for the ‘unseen portion is 731 (25.79% of the overall dataset), and for the ‘seen’ portion, it is 2,103.

<table><tr><td colspan="3">Method (Metric: Spearman) Overall Unseen Seen</td></tr><tr><td> $\mathrm { S i m C S E } _ { l a r g e }$ </td><td>32.13</td><td>13.93 25.02</td></tr><tr><td> $\mathrm { S i m C S E } _ { l a r g e + h y p e r - c l }$ </td><td>38.59</td><td>36.25 41.14</td></tr></table>

Table 5: Generalization capabilities of Hyper-CL on the C-STS validation set. We compare the tri-encoder baseline and Hyper-CL in both ‘unseen’ and ‘seen’ settings, using Spearman’s correlation as the evaluation metric.

Experimental results on the ‘overall’, ‘unseen’, and ‘seen’ datasets are listed in Table 5. For the embedding model $f ,$ we employed $\mathrm { S i m C S E } _ { l a r g e }$ Compared to the original tri-encoder (the first row), Hyper-CL shows a clear performance improvement of 16-22 points in both ‘seen’ and ‘unseen’ settings, with particularly superior performance in the ‘unseen’ setting. These findings highlight the superior generalization capabilities of Hyper-CL, enabling it to excel at handling unseen data.

## 5.4 Ablation Study on Contrastive Learning

In §3.3, we argued that the joint utilization of hypernetworks and contrastive learning yields the best performance among the available options. To verify this, we evaluate four different variations of the tri-encoder architecture on C-STS, whose details are as follows: (1) $\mathrm { S i m C S E } _ { b a s e } \colon$ the tri-encoder architecture trained only with $L _ { M S E } ; ( 2 )$ $\mathrm { S i m C S E } _ { b a s e + c l } ;$ the tri-encoder trained with both $L _ { M S E }$ and $L _ { C L }$ but hypernetworks excluded; (3) $\mathrm { S i m C S E } _ { b a s e + h y p e r 6 4 } \mathrm { : }$ a variant of Hyper-CL (K=64) but trained only with $L _ { M S E } ; ( 4 )$ $\mathrm { S i m C S E } _ { b a s e + h y p e r 6 4 - c l } \colon$ a normal Hyper-CL with low-rank approximation $( K { = } 6 4 )$ . For a fair comparison, we ensure that the total number of parameters for each variant remains consistent, guaranteeing equal expressive power.

Table 6 shows that the contrastive learning objective $( L _ { C L } )$ is more effective when combined with hypernetworks. This trend is clearly observed when comparing the performance increase from SimCSE<sub>base+hyper64</sub> to SimCSE<sub>base+hyper64-cl</sub> and that from $\mathrm { S i m C S E } _ { b a s e }$ to $\mathrm { S i m C S E } _ { b a s e + c l } .$

## 5.4.1 Why is Contrastive Learning More Effective with Hypernetworks?

The weight matrix $W _ { c } = q ( h _ { c } )$ , generated by the hypernetworks of Hyper-CL, is responsible for a linear transformation of a sentence embedding. On the other hand, the Hadamard product of a sentence embedding and a condition embedding, which is computed in the original tri-encoder architecture, can also be considered as a linear transformation, formulating the condition embedding $h _ { c }$ as a diagonal matrix $W _ { c ^ { \prime } } = d i a g ( h _ { c } )$

<table><tr><td>Method</td><td>Spearman</td></tr><tr><td> $\mathrm { S i m C S E } _ { b a s e + h y p e r 6 4 - c l }$ </td><td>37.96</td></tr><tr><td> $\mathrm { S i m C S E } _ { b a s e + h y p e r 6 4 }$ </td><td>35.38</td></tr><tr><td> $\mathrm { S i m C S E } _ { b a s e + c l }$ </td><td>36.13</td></tr><tr><td> $\mathrm { S i m C S E } _ { b a s e }$ </td><td>35.47</td></tr></table>

Table 6: Ablation study on the effectiveness of contrastive learning in condition subspaces. The results are from the C-STS validation set.

To gauge the expressiveness of the two different transformations induced by $W _ { c }$ and $W _ { c ^ { \prime } }$ , we calculate the variance of the Frobenius norm of these matrices during inference on a subset of the C-STS validation set. For matrices with varying valid element counts $( \mathrm { i } . \mathrm { e } . , W _ { c }$ and $W _ { c ^ { \prime } } )$ , we normalize their Frobenius norm by dividing by the square root of the number of valid elements. Experimental results show that the variance of the Frobenius norm, a measure of the matrices’ expressive power, is significantly higher (0.0248) for Hyper-CL’s transformations (W<sub>c</sub>) compared to the Hadamard product $( 0 . 0 0 1 ; W _ { c ^ { \prime } } )$ . These findings imply that hypernetworks endow the transformation with enhanced expressive power. Consequently, it is reasonable to expect that the contrastive learning process that leverages hypernetworks would also exhibit greater effectiveness.

## 6 Conclusion

We propose Hyper-CL, a method that combines hypernetworks with contrastive learning to generate conditioned sentence representations. In two representative tasks requiring conditioning on specific perspectives, our approach successfully narrows the performance gap with the bi-encoder architecture while maintaining the time efficiency characteristic of the tri-encoder approach. We further validate the inner workings of Hyper-CL by presenting intuitive analyses, such as visualizations of the embeddings projected by Hyper-CL. In future work, we plan to explore a broader range of applications for Hyper-CL and to investigate its refinement.

## Limitations

We have only explored applying our approach to encoder models, leaving room for applications on decoder models. Additionally, despite the variety of existing contrastive learning methodologies, we adhere to utilizing the contrastive learning objectives provided by the tasks.

## Ethics Statement

In this study, we utilized models and datasets publicly available from Huggingface. All datasets for evaluation are open-source and comply with data usage policies. However, some datasets (e.g., FB15k-237) are derived from Freebase, a large collaborative online collection that may contain inherently unethical information. We conducted a thorough inspection to check if our dataset contained any unethical content. No harmful information or offensive topics were identified during the human inspection process.

## Acknowledgements

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.RS-2020-II201373, Artificial Intelligence Graduate School Program(Hanyang University)). This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) under the artificial intelligence semiconductor support program to nurture the best talents (IITP-2024-RS-2023- 00253914) grant funded by the Korea government(MSIT). This work was supported by the National Research Foundation of Korea(NRF) grant funded by the Korea government(\*MSIT) (No.2018R1A5A7059549). \*Ministry of Science and ICT.

## References

Eneko Agirre, Daniel Cer, Mona Diab, and Aitor Gonzalez-Agirre. 2012. SemEval-2012 task 6: A pilot on semantic textual similarity. In \*SEM 2012: The First Joint Conference on Lexical and Computational Semantics – Volume 1: Proceedings of the main conference and the shared task, and Volume 2: Proceedings of the Sixth International Workshop on Semantic Evaluation (SemEval 2012), pages 385–393, Montréal, Canada. Association for Computational Linguistics.

Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko. 2013. Translating embeddings for modeling multirelational data. Advances in neural information processing systems, 26.

Daniel Cer, Mona Diab, Eneko Agirre, Iñigo Lopez-Gazpio, and Lucia Specia. 2017. SemEval-2017 task 1: Semantic textual similarity multilingual and crosslingual focused evaluation. In Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017), pages 1–14, Vancouver, Canada. Association for Computational Linguistics.

Vinod Kumar Chauhan, Jiandong Zhou, Ping Lu, Soheila Molaei, and David A Clifton. 2023. A brief review of hypernetworks in deep learning. arXiv preprint arXiv:2306.06955.

Chen Chen, Yufei Wang, Bing Li, and Kwok-Yan Lam. 2022. Knowledge is flat: A Seq2Seq generative framework for various knowledge graph completion. In Proceedings ofthe 29th International Conference on Computational Linguistics, pages 4005– 4017, Gyeongju, Republic of Korea. International Committee on Computational Linguistics.

Yung-Sung Chuang, Rumen Dangovski, Hongyin Luo, Yang Zhang, Shiyu Chang, Marin Soljacic, Shang-Wen Li, Scott Yih, Yoon Kim, and James Glass. 2022. DiffCSE: Difference-based contrastive learning for sentence embeddings. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4207–4218, Seattle, United States. Association for Computational Linguistics.

Ameet Deshpande, Carlos Jimenez, Howard Chen, Vishvak Murahari, Victoria Graf, Tanmay Rajpurohit, Ashwin Kalyan, Danqi Chen, and Karthik Narasimhan. 2023. C-STS: Conditional semantic textual similarity. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 5669–5690, Singapore. Association for Computational Linguistics.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Tomer Galanti and Lior Wolf. 2020. On the modularity of hypernetworks. Advances in Neural Information Processing Systems, 33:10409–10419.

Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple contrastive learning of sentence embeddings. In Proceedings of the 2021 Conference

on Empirical Methods in Natural Language Processing, pages 6894–6910, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

David Ha, Andrew M. Dai, and Quoc V. Le. 2017. Hypernetworks. In International Conference on Learning Representations.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361.

Bosung Kim, Taesuk Hong, Youngjoong Ko, and Jungyun Seo. 2020. Multi-task learning for knowledge graph completion with pre-trained language models. In Proceedings of the 28th International Conference on Computational Linguistics, pages 1737–1743, Barcelona, Spain (Online). International Committee on Computational Linguistics.

Taeuk Kim, Kang Min Yoo, and Sang-goo Lee. 2021. Self-guided contrastive learning for BERT sentence representations. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 2528–2540, Online. Association for Computational Linguistics.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.

Ritam Majumdar, Vishal Jadhav, Anirudh Deodhar, Shirish Karande, Lovekesh Vig, and Venkataramana Runkana. 2023. Hyperlora for pdes. arXiv preprint arXiv:2308.09290.

Niklas Muennighoff, Nouamane Tazi, Loic Magne, and Nils Reimers. 2023. MTEB: Massive text embedding benchmark. In Proceedings of the 17th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics, pages 2014–2037, Dubrovnik, Croatia. Association for Computational Linguistics.

Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2018. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748.

Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings using Siamese BERTnetworks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages

3982–3992, Hong Kong, China. Association for Computational Linguistics.

Kristina Toutanova and Danqi Chen. 2015. Observed versus latent features for knowledge base and text inference. In Proceedings of the 3rd workshop on continuous vector space models and their compositionality, pages 57–66.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Bo Wang, Tao Shen, Guodong Long, Tianyi Zhou, Ying Wang, and Yi Chang. 2021. Structure-augmented text representation learning for efficient knowledge graph completion. In Proceedings of the Web Conference 2021, pages 1737–1748.

Liang Wang, Wei Zhao, Zhuoyu Wei, and Jingming Liu. 2022. SimKGC: Simple contrastive knowledge graph completion with pre-trained language models. In Proceedings of the 60th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 4281–4294, Dublin, Ireland. Association for Computational Linguistics.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. 2020. Transformers: State-of-the-art natural language processing. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online. Association for Computational Linguistics.

Xin Xie, Ningyu Zhang, Zhoubo Li, Shumin Deng, Hui Chen, Feiyu Xiong, Mosha Chen, and Huajun Chen. 2022. From discrimination to generation: Knowledge graph completion with generative transformer. In Companion Proceedings of the Web Conference 2022, pages 162–165.

Liang Yao, Chengsheng Mao, and Yuan Luo. 2019. Kgbert: Bert for knowledge graph completion. arXiv preprint arXiv:1909.03193.

Xiyu Yu, Tongliang Liu, Xinchao Wang, and Dacheng Tao. 2017. On compressing deep models by low rank and sparse decomposition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7370–7379.

## A Training Details

In this section, we describe the hyperparameters used for training Hyper-CL on the two evaluation tasks employed in this paper. We implemented Hyper-CL using the Transformers package (Wolf et al., 2020). For both tasks, Hyper-CL utilized the [CLS] token embedding computed by an encoder model as a sentence representation.

Conditional Semantic Textual Similarity (C-STS): We conducted a hyperparameter search over learning rates $\in \ \{ 1 e { - } 5 , 2 e { - } 5 , 3 e { - } 5 \}$ , weight decays  0.0, 0.1 , and temperatures $\{ 1 . 0 , 1 . 5 , 1 . 7 , 1 . 9 \}$ . The hyperparameter set yielding the best scores on the C-STS validation set with three random seeds was used for the final evaluation of the test set. As a result, we adopted the hyperparameters as shown in Table 7.

<table><tr><td>Method</td><td>LR</td><td>WD</td><td>Temp</td></tr><tr><td>DiffCSEbase+hyper-cl</td><td>3e-5</td><td>0.1</td><td>1.5</td></tr><tr><td>DiffCSEbase+hyper64-cl</td><td>1e-5</td><td>0.0</td><td>1.5</td></tr><tr><td>SimCSEbase+hyper-cl</td><td>3e-5</td><td>0.1</td><td>1.9</td></tr><tr><td>SimCSEbase+hyper64-cl</td><td>2e-5</td><td>0.1</td><td>1.7</td></tr><tr><td>SimCSElarge+hyper-cl</td><td>2e-5</td><td>0.1</td><td>1.5</td></tr><tr><td> $\mathrm { S i m C S E } _ { l a r g e + h y p e r 8 5 - c l }$ </td><td>1e-5</td><td>0.1</td><td>1.9</td></tr></table>

Table 7: Hyperparameters determined for the C-STS task. The abbreviations LR, WD, Temp stands for learning rate, weight decay, and temperature, respectively.

Knowledge Graph Completion (KGC): We utilized the same set of hyperparameters proposed in SimKGC (Wang et al., 2022).

<table><tr><td>Method</td><td>Rank (K)</td><td>Spearman</td></tr><tr><td rowspan="2"> $\mathrm { D i f f C S E } _ { b a s e + h y p e r K - c l }$ </td><td>768 (=768/1) 192 (=768/4)</td><td>33.82 34.73</td></tr><tr><td>96 (=768/8) 64 (=768/12)</td><td>34.16</td></tr><tr><td> $\mathrm { S i m C S E } _ { b a s e + h y p e r K - c l }$ </td><td>48 (=768/16) 32 (=768/24) 768 (=768/1) 192 (=768/4) 96 (=768/8)</td><td>33.31 31.68 38.75 38.66 35.69</td></tr><tr><td> $\mathrm { S i m C S E } _ { l a r g e + h y p e r K - c l }$ </td><td>64 (=768/12) 48 (=768/16) 32 (=768/24) 1024 (1024/1) 256 (=1024/4) 128 (=1024/8) 85 (=1024/12)</td><td>38.36 37.02 36.92 39.60 38.76 38.19 38.12</td></tr></table>

Table 8: Ablation study of different ranks (K).

## B Ablation Study on the Selection of K

The selection of K for constructing lightweight hypernetworks is closely related to the size of sentence embeddings. We empirically evaluated the validation set to determine suitable values for K by dividing the embedding sizes of the base (768) and large (1024) encoder embeddings with various divisors (1, 4, 8, 12, 16, 24). According to Table 8, we observed that for $\mathrm { S i m C S E } _ { l a r g e }$ , the performance difference between K=128 and K=85 is just 0.07 points, while trainable parameters increase 1.5x. In conclusion, we found that setting the divisor to 12 (resulting in K values of 64 and 85 for the base and large models, respectively) achieves an optimal balance between performance and efficiency.