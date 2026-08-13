# Interpreting Conversational Dense Retrieval by Rewriting-Enhanced Inversion of Session Embedding

Yiruo Cheng, Kelong Mao, Zhicheng Dou\* Gaoling School of Artificial Intelligence, Renmin University of China {chengyr,mkl,dou}@ruc.edu.cn

## Abstract

Conversational dense retrieval has shown to be effective in conversational search. However, a major limitation of conversational dense retrieval is their lack of interpretability, hindering intuitive understanding of model behaviors for targeted improvements. This paper presents CONVINV , a simple yet effective approach to shed light on interpretable conversational dense retrieval models. CONVINV transforms opaque conversational session embeddings into explicitly interpretable text while faithfully maintaining their original retrieval performance as much as possible. Such transformation is achieved by training a recently proposed Vec2Text model (Morris et al., 2023) based on the ad-hoc query encoder, leveraging the fact that the session and query embeddings share the same space in existing conversational dense retrieval. To further enhance interpretability, we propose to incorporate external interpretable query rewrites into the transformation process. Extensive evaluations on three conversational search benchmarks demonstrate that CONVINV can yield more interpretable text and faithfully preserve original retrieval performance than baselines. Our work connects opaque session embeddings with transparent query rewriting, paving the way toward trustworthy conversational search. Our code is available at this repository.

## 1 Introduction

With the rapid development of language modeling, conversational search has emerged as a novel search paradigm and is garnering more and more attention. Different from the traditional ad-hoc search paradigm characterized by keyword-based queries and “ten-blue” links (Yu et al., 2020), conversational search empowers users to interact with the search engine through multi-turn natural language conversations to seek information, which brings a more intuitive and efficient search experience (Mao et al., 2022b; Gao et al., 2022; Zhu et al., 2023).

![](images/3d287ebcad3f2f08e88051d444341cd3dbd60a5ddb7fb4486a1ba71835a03b22.jpg)  
Figure 1: The blue section on the left signifies the conversational dense retrieval, and the green section on the right provides an overview of CONVINV.

In conversational search, the system input is a multi-turn natural language conversation, which may have many linguistic problems such as omissions, co-references, and ambiguities (Radlinski and Craswell, 2017), posing great challenges for accurately grasping the user’s real information needs. Recently, conversational dense retrieval (CDR) (Yu et al., 2021; Lin et al., 2021; Kim and Kim, 2022; Mao et al., 2022a; Qian and Dou, 2022; Mo et al., 2023b; Chen et al., 2024), which directly encodes the whole conversational search session and the passages into a unified embedding space to perform matching, has shown to be a promising method to solve this complex search task. Compared to another type of method: conversational query rewriting (CQR) (Lin et al., 2020; Vakulenko et al., 2021a; Wu et al., 2022; Mo et al., 2023a), which is a two-step method that first reformulates the search session into a decontextualized query rewrite and subsequently inputs this rewrite into existing adhoc search models for search, the end-to-end CDR models can be directly optimized towards better search effectiveness (Yu et al., 2021) and is more efficient as it avoids the extra latency caused by the rewriting step.

However, a notable drawback of conversational dense retrieval is that it inherently lacks interpretability (Mao et al., 2023d). By encoding conversations into dense vector embeddings rather than readable text, it becomes opaque how these CDR models comprehend search intent. The absence of interpretability becomes a severe obstacle for developers to comprehend the reasons behind the search results, hindering effective and targeted enhancements to the bad cases of the models (Mao et al., 2023d,a). Moreover, the absence of interpretability poses challenges in identifying and addressing potential biases or errors within the models, which could lead to unfair or misleading search results without the possibility of timely correction.

In this paper, we present CONVINV : a simple and effective approach aiming to shed light on the opacity problem of conversational dense retrieval. CONVINV demystifies the opaque conversational session embeddings by transforming them into explicitly interpretable text while faithfully maintaining their retrieval performance as much as possible. This transformation allows us to intuitively decipher the characteristics of behaviors of different conversational dense retrieval models.

Figure 1 provides an overview of CONVINV. Specifically, our approach is based on the recently proposed Vec2Text (Morris et al., 2023), which is a powerful method that can invert any text embedding into its original text given the corresponding text encoder. However, inverting the session embedding into the original session is meaningless as it brings no interpretability. We adapt Vec2Text to suit our interpretable inversion of conversational session embedding by taking specific advantage of how the conversational session encoders are trained: the session encoder starts from an ad-hoc query encoder and the passage encoder is frozen during the training. This makes the session and query embeddings finally share the same embedding space for retrieval. Therefore, we propose to train a Vec2Text model based on the ad-hoc query encoder to transform the session embedding so that the transformed text is different from the original session, but also maintains a similar retrieval performance when encoding it with the ad-hoc query encoder. To further enhance the interpretability of the transformed text, we directly incorporate well-interpretable external query rewrites into the Vec2Text transformation process, effectively guiding it to yield more interpretable text.

We conduct extensive evaluations on three conversational search benchmarks. Compared to baselines, the proposed CONVINV can transform conversational session embeddings into more interpretable text as well as faithfully restore the original retrieval performance of the session embeddings.

In summary, the contributions of our work are:

(1) We introduce a simple and effective approach CONVINV to shed light on the interpretability of conversational dense retrieval models by transforming opaque conversational session embeddings into interpretable text as well as faithfully maintain their original retrieval performance.

(2) We propose to incorporate the query rewrites into the transformation process to effectively enhance the interpretability of the transformed text.

(3) Our work connects opaque session embeddings with transparent query rewriting, paving the way toward trustworthy conversational search.

## 2 Related Work

## 2.1 Conversational Search

Currently, conversational search primarily relies on two main methods: conversational query rewriting (CQR) and conversational dense retrieval (CDR). CQR (Yu et al., 2020; Wu et al., 2022; Kumar and Callan, 2020; Voskarides et al., 2020; Lin et al., 2020; Mao et al., 2023b; Liu et al., 2021; Vakulenko et al., 2021a,b; Mao et al., 2023c; Mo et al., 2023a) transforms the whole session into a contextindependent query. The generated query rewrites can directly perform ad-hoc retrieval. In contrast, CDR (Yu et al., 2021; Mo et al., 2024; Krasakis et al., 2022; Mao et al., 2022a; Mo et al., 2023b; Mao et al., 2023d, 2022b; Dai et al., 2022; Hai et al., 2023; Mao et al., 2024) aims to train a session encoder that is capable of encoding the conversational context into a high-dimensional space for conducting dense retrieval. However, the session embedding encoded by the conversational query encoder lacks interpretability, hindering developers from comprehending the retrieval results.

## 2.2 Interpretable information retrieval

The interpretability issues have increasingly garnered attention within the domain of information retrieval. (Ram et al., 2023) proposed to interpret the session embeddings from dual encoders by mapping them into the lexical space of the model. (Mao et al., 2023d) proposed to augment the SPLADE model by incorporating multi-level denoising approaches, which can produce denoised and interpretable lexical session representations.

To explore the intricate interplay between embedded representations and their textual counterparts, a substantial body of research has focused on the task of inverting embeddings to coherent text. Representing the embedding of sentences as the initial token, (Li et al., 2023) trained a powerful decoder model to decode the entire sequence. (Morris et al., 2023) endeavored to produce text whose embedding closely approximates the given embedding. They achieved this by using the difference between hypothesis embeddings and actual embeddings.

## 3 Methodology

In this work, we present CONVINV, a new approach designed to demystify conversational session embeddings. Our approach focuses on transforming these opaque conversational session embeddings into explicitly interpretable text while maintaining their retrieval performance as much as possible. CONVINV aims to bridge the gap between the mysterious nature of dense embeddings and the necessity for clear, understandable insights in conversational search intent analysis.

## 3.1 Preliminaries

## 3.1.1 Conversational dense retrieval

Formally, conversational search involves a series of turns $\{ ( q _ { i } , a _ { i } ) \} _ { i = 1 } ^ { n }$ , where the users express their information needs at i-th turn through $q _ { i } .$ , and the system returns a relevant response $a _ { i }$ . This paper focuses on the conversational retrieval task, where the goal of conversational search models is to retrieve relevant passages $p$ for the current query $q _ { i } .$ , considering its historical context $H _ { i } = \{ ( q _ { j } , a _ { j } ) \} _ { j = 1 } ^ { i - 1 }$ The idea of conversational dense retrieval is to jointly map the current query $q _ { i }$ along with the historical context $H _ { i }$ and passages into a unified embedding space, and use the similarity between the session embedding and the passage embedding as the retrieval score:

$$
\begin{array} { r l r } { \mathbf { s _ { i } } } & { = } & { E _ { \mathrm { s } } ( q _ { i } , H _ { i } ) , \quad \mathbf { p } = E _ { \mathrm { p } } ( p ) , } \end{array}
$$

$$
\begin{array} { r l r } { r } & { { } = } & { \cos ( { \bf s _ { i } } , { \bf p } ) , } \end{array}\tag{1}
$$

(2)

where $E _ { \mathrm { s } }$ and $E _ { \mathrm { p } }$ are the session and passage encoders, respectively. cos is the cosine similarity

used to compute the retrieval score r.

## 3.1.2 Task formulation

The encoded conversational session embedding $\mathbf { s _ { i } } .$ while effective, is inherently mysterious and lacks interpretability. Our goal is to transform the session embedding $\mathbf { s _ { i } }$ into an explicit, interpretable text $\hat { q } _ { i }$ while faithfully maintaining the original retrieval effectiveness of the session embedding in $\hat { q } _ { i }$

## 3.2 Our Approach

To achieve this transformation from session embeddings to interpretable text, we propose a simple yet effective approach, called CONVINV , which is built upon the Vec2Text model (Morris et al., 2023) with tailored adjustments for the interpretation of conversational dense retrieval. Specifically, our approach has two important steps: (1) Training a Vec2Text model based on the ad-hoc query encoder. (2) Enhancing interpretation with rewriting. Figure 2 shows an illustration of our approach.

## 3.2.1 Training Vec2Text based on Ad-hoc Query Encoder

Vec2Text (Morris et al., 2023) is a recently proposed method for transforming embeddings into text. Given any text encoder E and a large collection of texts $T = \{ t _ { i } \}$ where $t _ { i }$ is a text, a Vec2Text model $\phi$ is trained based on a large number of (embedding, text) pairs (i.e., $\langle E ( t _ { i } ) , t _ { i } \rangle \rangle$ to learn to invert any text embedding $E ( t _ { i } )$ into a text $t _ { i } ^ { ' } ,$ where $E ( t _ { i } ^ { ' } )$ is very similar to $E ( t _ { i } )$ . As reported in their original paper, cos $( E ( t _ { i } ^ { \prime } ) , E ( t _ { i } ) )$ can reach up to 0.99. Motivated by the remarkable effectiveness of Vec2Text, we adapt it to suit our interpretable inversion of conversational session embedding by leveraging a specific training characteristic of conversational session encoders: Shared Embedding Spacefor Retrieval.

Shared embedding space for retrieval. For the training of conversational dense retrievers, it is common to initialize the conversational session encoder and the passage encoder from a pre-trained ad-hoc retriever, and only fine-tune the session encoder while freezing the passage encoder for facilitating the training (Yu et al., 2021; Lin et al., 2021; Mao et al., 2022a; Mo et al., 2023b). Therefore, we may assume that the session encoder and the adhoc query encoder share the same embedding space for retrieval as they share the same passage encoder. This characteristic is ideal for us to achieve more interpretable session embedding inversion as well as maintain its original retrieval effectiveness.

![](images/b4338c0ab40af98628e68d55f0c2f7145c5e60df450126abc2d37f643bfedfd7.jpg)  
Figure 2: Architecture of our proposed CONVINV.

Interpretable query generation. For a session encoder $E _ { \mathrm { s } }$ fine-tuned from an ad-hoc query encoder $E _ { \mathrm { q } } ,$ we train a Vec2Text model $\phi _ { \mathrm { q } }$ based on $E _ { \mathrm { q } }$ but not based on $E _ { \mathrm { s } }$ . Then, for a session embedding $\mathbf { s _ { i } } = E _ { \mathrm { s } } ( q _ { i } , H _ { i } )$ , we obtain its transformed text $\hat { q } _ { i } = \phi _ { \mathrm { q } } ( \mathbf { s } _ { i } )$ through $\phi _ { \mathrm { q } }$ . Specifically, Vec2Text includes two models: the inversion model and the correction model, and the generation process of Vec2Text includes two steps: (1) The initial inversion step, where an inversion model first inverts the embedding into an initial inverted text $t ^ { \mathrm { i n v } }$ . (2) The correction step, where a correction model then progressively refines this initial inverted text $t ^ { \mathrm { i n v } }$ to be more accurate. Figure 2 shows an illustration of the whole generation process of Vec2Text. The detailed introduction of our Vec2Text model training is provided in Appendix A.1.

Since $E _ { \mathrm { s } }$ and $E _ { \mathrm { q } }$ share the same retrieval embedding space, the transformed query embedding $E _ { \mathrm { q } } ( \hat { q } _ { i } )$ is supposed to be highly similar to the original session embedding $\mathbf { s _ { i } }$ and thus keep similar retrieval performance.

## 3.2.2 Interpretability Enhancement with Conversational Query Rewriting

While the transformed text $\hat { q } _ { i }$ can attain retrieval performance comparable to that of the original session embedding $\mathbf { s _ { i } }$ when encoded by the ad-hoc query encoder $E _ { q } ,$ there is no assurance that $\hat { q } _ { i }$ will form a coherent and interpretable sentence for human understanding.

We propose a simple method to leverage external query rewrites to enhance the interpretability.

Specifically, we first employ a conversational query rewriting model R (for example, the T5QR (Lin et al., 2020) model) to transform the conversational search session $\{ q _ { i } , H _ { i } \}$ into a standalone query rewrite $q _ { i } ^ { * } = R ( q _ { i } , H _ { i } )$ . Then, in the generation process of Vec2Text, we discard the initial inversion process and directly use the query rewrite $q _ { i } ^ { * }$ as the initial inverted text $t ^ { \mathrm { i n v } }$

The rewriting model $R ,$ trained on a vast dataset of human-crafted rewrites, ensures that the resultant query rewrite is coherent and understandable compared to the original inverted text produced by VecText’s inversion model. The new inverted text $q _ { i } ^ { * }$ , serving as an improved starting point for the session embedding transformation, can help lead the whole generation process towards a more interpretable direction, and thus enhance the interpretability of the final transformed text $\hat { q } _ { i }$ .

## 4 Experimental Settings

This section presents our basic experimental settings. See Appendix A.2 for full details.

## 4.1 Datasets

We use four public conversational search datasets: QReCC (Anantha et al., 2021), TREC CAsT-19 (Dalton et al., 2020b), TREC CAsT-20 (Dalton et al., 2020a), and TREC CAsT-21 (Dalton et al., 2021). The QReCC dataset consists of 13.6K conversations, with an average of 6 turns per conversation. While the three CAsT datasets (19, 20, 21) only comprise 50, 25, and 26 conversations, respectively, but with more detailed relevance labeling. All four datasets provide human rewrites for each turn. Following existing works (Mao et al., 2023d; Mo et al., 2023a), we train CDR models on the QReCC dataset and conduct evaluations on the three CAsT datasets.

![](images/0ee927d0370d2a479bf57511d270da265591fd35dee210c89a741bc6b88a32c5.jpg)  
Figure 3: The workflow of UniCRR (Unifying Conversational Dense Retrieval and Query Rewriting).

## 4.2 Conversational Dense Retrieval Models

Currently, there are mainly two paradigms to train conversational session encoders. The first is proposed by Yu et al. (2021) which employs an ad hoc query encoder as the teacher and learns the student session encoder by mimicking the teacher embeddings originating from human queries. The second is to use the classical ranking loss function (Karpukhin et al., 2020; Lin et al., 2021) to maximize the distance between the session and its positive passages and minimize the distance between the session and negative passages.

Our evaluation is based on both types of CDR models. We name the first type KD-Retriever and the second type Conv-Retriever, where Retriever can be replaced with any base ad-hoc retriever. Specifically, we mainly experiment with a popular ad-hoc retriever, i.e., GTR (Ni et al., 2022), and we investigate the universality of our method to different ad-hoc retrievers in Section 5.3.

## 4.3 Baselines

Our main goal is to demonstrate the interpretability and preserved retrieval performance of the transformed text generated by our CONVINV , compared to the original session embeddings of KD-GTR and Conv-GTR. To the best of our knowledge, there is no existing method that is completely suitable for our task, i.e., interpreting conversational session embeddings (see the full task definition in Section 3.1.2). Therefore, we propose a straightforward but strong baseline called UniCRR. Figure 3 illustrates UniCRR. Specifically, we unify the session encoder and the query rewriter in an encoderdecoder architecture and adopt multi-task learning to simultaneously train both. As such, the rewrite generated from the decoder part can interpret the session embedding generated from the encoder part to some extent.

In addition to the original KD-GTR, Conv-GTR, and our proposed UniCRR, we also use the following conversational search baselines mainly for the comparisons of retrieval performance: (1) T5QR (Lin et al., 2020): A conversational query rewriter based on T5 (Raffel et al., 2020), trained using human-generated rewrites. (2) ConvGQR (Mo et al., 2023a): A framework for query reformulation that integrates query rewriting with generative query expansion. (3) LeCoRE (Mao et al., 2023d): A conversational lexical retrieval model extending from the SPLADE model with two well-matched multi-level denoising approaches.

## 4.4 Evaluation Metrics

Retrieval and inversion evaluation. Following existing works (Mo et al., 2023a; Mao et al., 2023d) and the official settings of the CAsT datasets (Dalton et al., 2020a), we choose MRR, NDCG@3, and Recall@100 to evaluate the retrieval performance. We use two metrics to quantify the fidelity of the embedding inversion: (1) The absolute difference in the retrieval performances between using the session embeddings and the transformed text. (2) Following Vec2Text (Morris et al., 2023), we also calculate the cosine similarity between the session embeddings and the transformed text embeddings. Interpretability evaluation. We conduct a human evaluation for the interpretability of the transformed text from three aspects: (1) Clarity: evaluating the clarity of text expression and identifying the presence of ambiguity or vague expressions; (2) Coherence: examining the logical structure of the text; (3) Completeness: determining the extent to which the text comprehensively covers all historical information. Five information retrieval researchers are employed to assign scores ranging from 1 to 5. A larger score indicates better performance.

## 4.5 Implementations

For CONVINV , we train Vec2Text models on the large-scale MSMARCO (Nguyen et al., 2016) query and passage collections based on different ad-hoc query encoders. The inversion model is trained for 50 epochs with a batch size of 128 and the correction model is trained for 100 epochs with a batch size of 200 with 1e-3 learning rate. The maximum sequence length is set to 48. By default, we use the rewrites generated by T5QR to perform rewriting enhancement.

<table><tr><td rowspan="2">Method</td><td colspan="3">CAsT-19</td><td colspan="3">CAsT-20</td><td colspan="3">CAsT-21</td></tr><tr><td>MRR</td><td>NDCG@3</td><td>Recall@100</td><td>MRR</td><td>NDCG@3 Recall@100</td><td></td><td>MRR</td><td>NDCG@3</td><td>Recall@100</td></tr><tr><td>T5QR</td><td>65.8</td><td>41.9</td><td>38.2</td><td>46.6</td><td>32.1</td><td>41.4</td><td>47.9</td><td>34.1</td><td>45.2</td></tr><tr><td>ConvGQR</td><td>66.7</td><td>39.3</td><td>33.7</td><td>39.7</td><td>25.9</td><td>33.8</td><td>40.6</td><td>25.3</td><td>37.3</td></tr><tr><td>LeCoRE</td><td>70.3</td><td>42.2</td><td>49.4</td><td>45.0</td><td>29.0</td><td>46.7</td><td>54.8</td><td>32.3</td><td>38.7</td></tr><tr><td>Conv-GTR</td><td>53.8</td><td>31.0</td><td>34.6</td><td>27.9</td><td>18.4</td><td>31.8</td><td>42.2</td><td>28.4</td><td>46.4</td></tr><tr><td>UniCRR</td><td></td><td>54.4 (+0.6) 31.9 (+0.9)</td><td>31.0 (-3.6)</td><td></td><td>36.0 (+8.1) 23.7 (+5.3) 33.3 (+1.5)</td><td></td><td>35.0 (-7.2)</td><td>23.2 (-5.2) 31.3 (-15.1)</td><td></td></tr><tr><td>CONVINV</td><td></td><td>56.4 (+2.6) 33.1 (+2.1)</td><td>37.0 (+2.4)</td><td>27.2 (-0.7) 18.5 (+0.1)</td><td></td><td>30.4 (-1.4)</td><td>41.9 (-0.3)</td><td>28.2 (-0.2)</td><td>41.7 (-4.7)</td></tr><tr><td>KD-GTR</td><td>74.9</td><td>46.9</td><td>41.9</td><td>49.5</td><td>35.9</td><td>46.9</td><td>54.7</td><td>36.4</td><td>55.4</td></tr><tr><td>UniCRR</td><td>65.1 (-9.8)</td><td>40.6 (-6.3)</td><td>37.0 (-4.9)</td><td>44.4 (-5.1)</td><td>32.3 (-3.6)</td><td>39.5 (-7.4)</td><td></td><td>[41.0 (-13.7) 27.3 (-9.1)</td><td>39.5 (-15.9)</td></tr><tr><td>CONVINV</td><td></td><td>74.2 (-0.7)  44.9 (-2.0)</td><td>43.0 (+1.1)</td><td>47.6 (-1.9) 34.4 (-1.5)</td><td></td><td>44.0 (-2.9)</td><td></td><td>54.7 (+0.0) 37.4 (+1.0)</td><td>55.1 (-0.3)</td></tr></table>

Table 1: Retrieval performance comparisons. Our main competitor is UniCRR. The numbers in parentheses indicate the absolute difference between the original CDR model (i.e., Conv-GTR or KD-GTR) and the transformed text. In the comparison between CONVINV and UniCRR, a green background indicates that its performance gap with the original session embedding is smaller compared to its counterpart, while a red background indicates a larger gap. The best performance is bold.
<table><tr><td rowspan="2">Method</td><td colspan="3">CAsT-19</td><td colspan="3">CAsT-20</td><td colspan="3">CAsT-21</td></tr><tr><td>MRR</td><td>NDCG@3</td><td>Recall@100</td><td>MRR</td><td>NDCG@3 Recall@100</td><td></td><td>MRR</td><td>NDCG@3</td><td>Recall@100</td></tr><tr><td>Conv-GTR</td><td>53.8</td><td>31.0</td><td>34.6</td><td>27.9</td><td>18.4</td><td>31.8</td><td>42.2</td><td>28.4</td><td>46.4</td></tr><tr><td>TX-Inversion</td><td>58.0(+4.2)</td><td>33.5 (+2.5)</td><td>37.1(+2.5)</td><td>28.2(+0.3)</td><td>18.8(+0.4)</td><td>29.5(-2.3)</td><td>40.7(-1.5)</td><td>26.6(-1.8)</td><td>43.1(-3.3)</td></tr><tr><td>TX-Human</td><td>55.6(+1.8)</td><td>33.0(+2.0)</td><td>36.0(+1.4)</td><td>27.3(-0.6)</td><td>18.5(+0.1)</td><td>30.9(-0.9)</td><td>42.6(+0.4)</td><td>26.5(-1.9)</td><td>41.1(-5.3)</td></tr><tr><td>CONVINV</td><td>56.4 (+2.6)</td><td>33.1 (+2.1)</td><td>37.0 (+2.4)</td><td>27.2 (-0.7)</td><td>18.5 (+0.1)</td><td>30.4 (-1.4)</td><td>41.9 (-0.3)</td><td>28.2 (-0.2)</td><td>41.7 (-4.7)</td></tr><tr><td>KD-GTR</td><td>74.9</td><td>46.9</td><td>41.9</td><td>49.5</td><td>35.9</td><td>46.9</td><td>54.7</td><td>36.4</td><td>55.4</td></tr><tr><td>TX-Inversion</td><td>71.6(-3.3)</td><td>44.2(-2.7)</td><td>42.3(+0.4)</td><td>48.1(-1.4)</td><td>33.6(-2.3)</td><td>44.8(-2.1)</td><td>53.8(-0.9)</td><td>36.1(-0.3)</td><td>55.6(+0.2)</td></tr><tr><td>TX-Human</td><td>73.1(-1.8)</td><td>44.1(-2.8)</td><td>42.3 (+0.4)</td><td>48.6(-0.9)</td><td>35.0(-0.9)</td><td>45.9(-1.0)</td><td>53.1(-1.6)</td><td>35.8(-0.6)</td><td>54.3(-1.1)</td></tr><tr><td>CONVINV</td><td>74.2 (-0.7)</td><td>44.9 (-2.0)</td><td>43.0 (+1.1)</td><td>47.6 (-1.9) 34.4 (-1.5)</td><td></td><td>44.0 (-2.9)</td><td>54.7 (+0.0)</td><td>37.4 (+1.0)</td><td>55.1 (-0.3)</td></tr></table>

Table 2: Ablation results of the effect of rewriting-enhancement. The numbers in parentheses indicate the difference between the original (i.e., Conv-GTR or KD-GTR) and the transformed text. CONVINV uses T5QR for rewriting enhancement by default. In the comparison between TX-Inversion, TX-Human, and CONVINV , a green background indicates that its performance gap with the original session embedding is the smallest. The best performance is bold.

We train the conversational dense retrieval models on the QReCC dataset. The session encoder is initialized from an ad-hoc query encoder and the passage encoder is frozen during training. The input of the session encoder is the concatenation of all historical turns and the current query following existing works (Mao et al., 2023d; Mo et al., 2023a). For KD-Retriever, we follow Yu et al. (2021) using the Mean Squared Error (MSE) loss function to perform knowledge distillation. For Conv-Retriever, we use the contrastive ranking loss function with 48 batch size. The maximum input lengths of the session encoder and the passage encoder are set to 512 and 384, respectively. We generally train 2 epochs with 5e-5 learning rate for CDR models.

## 5 Experimental Results

## 5.1 Retrieval and Inversion Evaluation

Note that our work does not aim to achieve absolutely higher retrieval performance, but rather to faithfully restore the retrieval performance of the original session embeddings, so the main competitor of our CONVINV is only UniCRR. The retrieval performance comparisons on three CAsT datasets are shown in Table 1 and the similarity is shown in Table 3. We find:

<table><tr><td>Method</td><td>CAsT-19</td><td>CAsT-20</td><td>CAsT-21</td></tr><tr><td>UniCRR</td><td>94.10</td><td>92.80</td><td>87.90</td></tr><tr><td>ConvInv</td><td>95.80</td><td>95.20</td><td>94.50</td></tr></table>

Table 3: The similarity between the embeddings of texts generated by UniCRR and CONVINV, and the original session embeddings. The best performance is bold.

(1) Compared to UniCRR, CONVINV achieves superior embedding restoration. For example, for KD-GTR, the average absolute differences for CONVINV are 0.87 (MRR), 1.5 (NDCG@3), and 1.43 (Recall@100), and the average absolute differences for UniCRR are 9.53 (MRR), 6.3 (NDCG@3), and 9.4 (Recall@100). This indicates that the transformed texts generated by CON-VINV are closer to the original session embeddings. This aligns with the restoration similarity, which is shown in Table 3. The superior reconstruction performance of ConvInv compared to UniCRR may stem from the fact that UniCRR fails to establish a direct correlation between session embeddings during both the training and inference phases.

(2) We surprisingly notice that the transformed text generated by CONVINV can sometimes even yield slightly better retrieval performance. For example, on the CAsT-21 dataset, we observe 2.7% NDCG@3 relative gains over the original session embedding, respectively. This discovery could potentially pave the way for enhancing retrieval efficacy and interpretability through the collaborative optimization of CQR and CDR.

Ablation study for rewriting enhancement. We propose using external query rewrites generated by T5QR to improve the interpretability of transformed text, which matches the original session embedding’s retrieval performance but may lack coherence and understandability. Building on this proposition, we compare three types of transformed text to investigate the effect of rewriting enhancement: (1) using T5QR rewrites for the rewriting enhancement, which is the default CONVINV. (2) TX-Human: using human rewrites for the rewriting enhancement. (3) TX-Inversion: not performing rewriting enhancement (i.e., just using the text generated by the inversion model for the correction step). The ablation results of the retrieval performance of transformed text are shown in Table 2. We observe that the utilization of rewriting enhancement brings the retrieval performance closer to the original. Using rewriting enhancement generally leads to stronger overall retrieval performance compared to not.

## 5.2 Interpretability Evaluation

We manually evaluate the interpretability of three types of transformed text generated by CONVINV. Evaluation results are shown in Figure 4 and we have the following observations:

(1) Using query rewrites as the initial inverted text improves the interpretability of the transformed text of KD-GTR and Conv-GTR across the CAsT-19, CAsT-20, and CAsT-21 datasets. This improvement can be attributed to the introduction of the rewrite as the initial inverted text, which essentially offers the corrector model a more informative and clear starting point. These notable enhancements underscore the necessity of our rewritingenhancement approach in improving text interpretability.

<table><tr><td>Context: (CAsT-19 Session 54) q1: What is worth seeing in Washington D.C.? q4: Is the spy museum free? q5: What is there to do in DC after the museums close?</td></tr><tr><td>Current Query(68.1): What is the best time to visit the reflecting pools?</td></tr><tr><td>CONVINV (68.1): In Washington D.C. what is the best time to visit the reflecting pools (like the Smithsonian Museum)? TX-Human(47.9): In Washington D.C., what is the best time to visit the reflecting</td></tr><tr><td>pools by the Smithsonian and other DC museums? TX-Inversion(20.2): In Washington D.C., what is the best time to visit the reflecting</td></tr><tr><td>pools (e.g. Smithsonian National Museum)? Human Rewrite(36.1): What is the best time to visit the reflecting pools in Washington D.C.?</td></tr></table>

Table 4: A case illustrating the distinction in utilizing rewriting enhancement for transformed text. The numbers in parentheses indicate the retrieval performance NDCG@3 of the transformed text. Notably, the number in parentheses under Current Query represents the retrieval results of the original session embedding, not that of the current query statement.

(2) For both KD-GTR and Conv-GTR, the human evaluation scores of transformed text on CAsT-19 are higher, whether using rewritingenhancement or not, compared to CAsT-20 and CAsT-21. This observation may be attributed to the absence of response information in the CAsT-19 dataset, which exclusively contains query content. Consequently, the session embedding on CAsT-19 is relatively simple, lacking the complexity introduced by response data.

(3) The lower human evaluation scores of transformed text for Conv-GTR compared to KD-GTR on three datasets may be due to the implications of contrastive learning. This method often introduces additional noise. Therefore, Conv-GTR’s session embedding might be more prone to interference, potentially leading to its less effective performance in generating transformed text.

We provide a concrete example of the transformed texts in Table 4. More case studies are in Appendix A.4. We find that the transformed text CONVINV not only exhibits high interpretability, fully capturing the user’s query intent about “in Washington D.C.”, but also maintains the closest proximity of retrieval performance to the original session embedding. We notice that it includes an additional clue “(like the Smithsonian Museum)” in the query, which may just be additional knowledge reflected in the mysterious session embedding that can help retrieve passages about famous attractions in Washington D.C.

![](images/5b5ee8f69f1e70753c0a6bbce255f4d4fe44356752aa3698e88f0c87076d4aa8.jpg)

![](images/24a69739fa646709b6d6da32eb93f0d4f938ec1c918d465bb9862737e56c4d2f.jpg)

![](images/6c745de7e2c5b0104b1286bbb66a73588e61b4f2f25a81c340575f6b3456d791.jpg)

![](images/7d718a5b6ebc0fb6651bc6fe72e341c2a6b7693bb5359ae03260ced9ab3e772f.jpg)

![](images/646e59d275aadd62833db4ab9e4303fd3b0063c81fc50fbff0eef1d6dc34bcf0.jpg)

![](images/9a6be6859650c88a23156e8c580e572d11ae112fa810d5c6ee780b660bf8cdff.jpg)  
Figure 4: Results of human evaluations for interpretability. Cla, Coh, and Com represent Clarity, Coherence, and Completeness, respectively. The Avg indicates the average of these scores.

<table><tr><td rowspan="3">Retriever</td><td rowspan="3">Method</td><td colspan="3">CAsT-19</td><td colspan="4">CAsT-21</td></tr><tr><td></td><td>Retrieval Performance</td><td>Interpretablity</td><td></td><td>Retrieval Performance</td><td>Interpretablity</td><td></td></tr><tr><td>MRR</td><td>NDCG@3 Recal1@100</td><td>similarity hum eval</td><td>MRR</td><td>NDCG@3 Recall@100 similarity hum eval</td><td></td><td></td></tr><tr><td rowspan="2">GTR</td><td>KD-GTR</td><td>74.9 46.9</td><td>41.9</td><td></td><td>54.7</td><td>36.4 55.4</td><td></td><td></td></tr><tr><td>CONVINV</td><td></td><td>74.2 (-0.7) 44.9 (-2.0)43.0 (+1.1)</td><td>0.985</td><td>4.40 54.7(0.0)</td><td>37.4(+1.0) 55.1(-0.3)</td><td>0.945</td><td>3.53</td></tr><tr><td rowspan="2">ANCE</td><td>KD-ANCE</td><td>72.0</td><td>44.4 34.2</td><td></td><td>52.8</td><td>36.9 50.8</td><td></td><td>1</td></tr><tr><td>CONVINV</td><td></td><td>72.0(0.0) 44.5(+0.1) 34.3(+0.1)</td><td>0.999</td><td>4.90</td><td>55.8(+3.0) 37.4(+0.5) 53.1(+2.3)</td><td>0.998</td><td>4.07</td></tr><tr><td rowspan="2">BGE</td><td>KD-BGE</td><td>69.5</td><td>44.0 41.2</td><td></td><td>57.9</td><td>41.2 56.0</td><td></td><td></td></tr><tr><td>CONVINV</td><td></td><td>69.9(+0.4) 45.4(+1.4) 41.5(+0.3)</td><td>0.972</td><td>4.33</td><td>59.8(+1.9)41.1(-0.1) 54.4(-1.6)</td><td>0.954</td><td>4.25</td></tr></table>

Table 5: Retrieval performance and interpretability of generated transformed text based on different ad-hoc retrievers on CAsT-19 and CAsT-21 datasets. The "hum eval" represents the human evaluation score. The numbers in parentheses indicate the difference between the original and the transformed text. The best performance is bold.

## 5.3 Experiments with Different Retrievers

We investigate the universality of our CONVINV by changing the base ad-hoc retriever of the CDR models. Specifically, we experiment with another two popular ad-hoc retrievers: ANCE (Xiong et al., 2021) and BGE (Xiao et al., 2023). Results are shown in Table 5. We find that:

(1) Regardless of the selected ad-hoc retriever, both retrieval similarity and text similarity metrics are observed to be high. To illustrate, on the CAsT-19 dataset, the average absolute differences for KD-ANCE on CAsT-19 dataset are 0.0 (MRR), 0.1 (NDCG@3), and 0.1 (Recall@100), and the cosine similarity is up to 99.9%.

(2) Across both CAsT-19 and CAsT-21 datasets, there is a sustained consistency between similarity scores and human evaluations, indicating that textual similarity is a reliable indicator of quality as perceived by human judges. However, this does not encapsulate all the factors considered in human evaluations, especially as similarity scores remain robust while human evaluations show a decline from CAsT-19 to CAsT-21. Although there is a noted decrease in human evaluation scores across all methods when moving from CAsT-19 to CAsT-21, the similarity scores remain high or even show marginal improvement.

## 6 Conclusion

In this paper, we present a novel approach CON-VINV to shed light on the interpretability of conversational dense retrieval. By experimenting with two typical conversational dense retrieval models on three conversational search benchmarks, we demonstrate the effectiveness of our approach in providing interpretable text as well as faithfully restoring the original retrieval performance of session embeddings. Our work not only enhances interpretability in conversational dense retrieval but also lays a groundwork for future research toward trustworthy conversational search.

## Limitations

Our work provides a simple but effective solution to enhance the interpretability of conversational dense retrieval models, bridging the gap between opaque session embeddings and transparent query rewriting. However, the necessity to train distinct Vec2Text models based on various retrievers demands a significant time investment. Additionally, for session embeddings trained using contrastive learning, the transformed text fails to achieve sufficiently high similarity to the original session embedding, suggesting an incomplete decoding of the session embedding. Besides, some of the transformed texts may not exhibit retrieval performance as effective as the original session embeddings. Some more sophisticated conversational dense retrievers have not been investigated.

## Acknowledgments

This work was supported by National Key R&D Program of China No. 2022ZD0120103, National Natural Science Foundation of China No. 62272467, the fund for building world-class universities (disciplines) of Renmin University of China, and Public Computing Cloud, Renmin University of China. The work was partially done at the Engineering Research Center of Next-Generation Intelligent Search and Recommendation, MOE.

## References

Raviteja Anantha, Svitlana Vakulenko, Zhucheng Tu, Shayne Longpre, Stephen Pulman, and Srinivas Chappidi. 2021. Open-domain question answering goes conversational via question rewriting. In Proceedings ofthe 2021 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June 6-11, 2021, pages 520–534. Association for Computational Linguistics.

Haonan Chen, Zhicheng Dou, Kelong Mao, Jiongnan Liu, and Ziliang Zhao. 2024. Generalizing conversational dense retrieval via llm-cognition data augmentation. CoRR.

Zhuyun Dai, Arun Tejasvi Chaganty, Vincent Y. Zhao, Aida Amini, Qazi Mamunur Rashid, Mike Green, and Kelvin Guu. 2022. Dialog inpainting: Turning documents into dialogs. In International Conference on

Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pages 4558–4586. PMLR.

Jeffrey Dalton, Chenyan Xiong, and Jamie Callan. 2020a. Cast 2020: The conversational assistance track overview. In Proceedings of the Twenty-Ninth Text REtrieval Conference, TREC 2020, Virtual Event [Gaithersburg, Maryland, USA], November 16-20, 2020, volume 1266 of NIST Special Publication. National Institute of Standards and Technology (NIST).

Jeffrey Dalton, Chenyan Xiong, and Jamie Callan. 2020b. TREC cast 2019: The conversational assistance track overview. CoRR, abs/2003.13624.

Jeffrey Dalton, Chenyan Xiong, and Jamie Callan. 2021. TREC cast 2021: The conversational assistance track overview. In Proceedings of the Thirtieth Text REtrieval Conference, TREC 2021, online, November 15-19, 2021, volume 500-335 of NIST Special Publication. National Institute of Standards and Technology (NIST).

Jianfeng Gao, Chenyan Xiong, Paul Bennett, and Nick Craswell. 2022. Neural approaches to conversational information retrieval. CoRR, abs/2201.05176.

Nam Le Hai, Thomas Gerald, Thibault Formal, Jian-Yun Nie, Benjamin Piwowarski, and Laure Soulier. 2023. Cosplade: Contextualizing SPLADE for conversational information retrieval. In Advances in Information Retrieval - 45th European Conference on Information Retrieval, ECIR 2023, Dublin, Ireland, April 2-6, 2023, Proceedings, Part I, volume 13980 of Lecture Notes in Computer Science, pages 537–552. Springer.

Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2021. Billion-scale similarity search with gpus. IEEE Trans. Big Data, 7(3):535–547.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick S. H. Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020, pages 6769–6781. Association for Computational Linguistics.

Sungdong Kim and Gangwoo Kim. 2022. Saving dense retriever from shortcut dependency in conversational search. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 10278–10287. Association for Computational Linguistics.

Antonios Minas Krasakis, Andrew Yates, and Evangelos Kanoulas. 2022. Zero-shot query contextualization for conversational search. In SIGIR ’22: The 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, Madrid, Spain, July 11 - 15, 2022, pages 1880–1884. ACM.

Vaibhav Kumar and Jamie Callan. 2020. Making information seeking easier: An improved pipeline for conversational search. In Findings ofthe Association for Computational Linguistics: EMNLP 2020, Online Event, 16-20 November 2020, volume EMNLP 2020 of Findings ofACL, pages 3971–3980. Association for Computational Linguistics.

Haoran Li, Mingshi Xu, and Yangqiu Song. 2023. Sentence embedding leaks more information than you expect: Generative embedding inversion attack to recover the whole sentence. In Findings ofthe Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023, pages 14022– 14040. Association for Computational Linguistics.

Sheng-Chieh Lin, Jheng-Hong Yang, and Jimmy Lin. 2021. Contextualized query embeddings for conversational search. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021, pages 1004–1015. Association for Computational Linguistics.

Sheng-Chieh Lin, Jheng-Hong Yang, Rodrigo Frassetto Nogueira, Ming-Feng Tsai, Chuan-Ju Wang, and Jimmy Lin. 2020. Conversational question reformulation via sequence-to-sequence architectures and pretrained language models. CoRR, abs/2004.01909.

Hang Liu, Meng Chen, Youzheng Wu, Xiaodong He, and Bowen Zhou. 2021. Conversational query rewriting with self-supervised learning. In IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2021, Toronto, ON, Canada, June 6-11, 2021, pages 7628–7632. IEEE.

Kelong Mao, Chenlong Deng, Haonan Chen, Fengran Mo, Zheng Liu, Tetsuya Sakai, and Zhicheng Dou. 2024. Chatretriever: Adapting large language models for generalized and robust conversational dense retrieval. CoRR, abs/2404.13556.

Kelong Mao, Zhicheng Dou, Bang Liu, Hongjin Qian, Fengran Mo, Xiangli Wu, Xiaohua Cheng, and Zhao Cao. 2023a. Search-oriented conversational query editing. In Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023, pages 4160–4172. Association for Computational Linguistics.

Kelong Mao, Zhicheng Dou, Bang Liu, Hongjin Qian, Fengran Mo, Xiangli Wu, Xiaohua Cheng, and Zhao Cao. 2023b. Search-oriented conversational query editing. In Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023, pages 4160–4172. Association for Computational Linguistics.

Kelong Mao, Zhicheng Dou, Fengran Mo, Jiewen Hou, Haonan Chen, and Hongjin Qian. 2023c. Large language models know your contextual search intent: A prompting framework for conversational search. In Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10,

2023, pages 1211–1225. Association for Computational Linguistics.

Kelong Mao, Zhicheng Dou, and Hongjin Qian. 2022a. Curriculum contrastive context denoising for fewshot conversational dense retrieval. In SIGIR ’22: The 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, Madrid, Spain, July 11 - 15, 2022, pages 176–186. ACM.

Kelong Mao, Zhicheng Dou, Hongjin Qian, Fengran Mo, Xiaohua Cheng, and Zhao Cao. 2022b. Convtrans: Transforming web search sessions for conversational dense retrieval. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 2935– 2946. Association for Computational Linguistics.

Kelong Mao, Hongjin Qian, Fengran Mo, Zhicheng Dou, Bang Liu, Xiaohua Cheng, and Zhao Cao. 2023d. Learning denoised and interpretable session representation for conversational search. In Proceedings ofthe ACM Web Conference 2023, WWW 2023, Austin, TX, USA, 30 April 2023 - 4 May 2023, pages 3193–3202. ACM.

Fengran Mo, Kelong Mao, Yutao Zhu, Yihong Wu, Kaiyu Huang, and Jian-Yun Nie. 2023a. Convgqr: Generative query reformulation for conversational search. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pages 4998–5012. Association for Computational Linguistics.

Fengran Mo, Jian-Yun Nie, Kaiyu Huang, Kelong Mao, Yutao Zhu, Peng Li, and Yang Liu. 2023b. Learning to relate to previous turns in conversational search. In Proceedings ofthe 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD 2023, Long Beach, CA, USA, August 6-10, 2023, pages 1722–1732. ACM.

Fengran Mo, Chen Qu, Kelong Mao, Tianyu Zhu, Zhan Su, Kaiyu Huang, and Jian-Yun Nie. 2024. History-aware conversational dense retrieval. CoRR, abs/2401.16659.

John X. Morris, Volodymyr Kuleshov, Vitaly Shmatikov, and Alexander M. Rush. 2023. Text embeddings reveal (almost) as much as text. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 12448–12460. Association for Computational Linguistics.

Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. 2016. MS MARCO: A human generated machine reading comprehension dataset. In Proceedings of the Workshop on Cognitive Computation: Integrating neural and symbolic approaches 2016 co-located with the 30th Annual Conference on Neural Information Processing Systems (NIPS 2016), Barcelona,

Spain, December 9, 2016, volume 1773 of CEUR Workshop Proceedings. CEUR-WS.org.

Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernández Ábrego, Ji Ma, Vincent Y. Zhao, Yi Luan, Keith B. Hall, Ming-Wei Chang, and Yinfei Yang. 2022. Large dual encoders are generalizable retrievers. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 9844–9855. Association for Computational Linguistics.

Hongjin Qian and Zhicheng Dou. 2022. Explicit query rewriting for conversational dense retrieval. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 4725–4737. Association for Computational Linguistics.

Filip Radlinski and Nick Craswell. 2017. A theoretical framework for conversational search. In Proceedings of the 2017 Conference on Conference Human Information Interaction and Retrieval, CHIIR 2017, Oslo, Norway, March 7-11, 2017, pages 117–126. ACM.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21:140:1–140:67.

Ori Ram, Liat Bezalel, Adi Zicher, Yonatan Belinkov, Jonathan Berant, and Amir Globerson. 2023. What are you token about? dense retrieval as distributions over the vocabulary. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pages 2481–2498. Association for Computational Linguistics.

Svitlana Vakulenko, Shayne Longpre, Zhucheng Tu, and Raviteja Anantha. 2021a. Question rewriting for conversational question answering. In WSDM ’21, The Fourteenth ACM International Conference on Web Search and Data Mining, Virtual Event, Israel, March 8-12, 2021, pages 355–363. ACM.

Svitlana Vakulenko, Nikos Voskarides, Zhucheng Tu, and Shayne Longpre. 2021b. A comparison of question rewriting methods for conversational passage retrieval. In Advances in Information Retrieval - 43rd European Conference on IR Research, ECIR 2021, Virtual Event, March 28 - April 1, 2021, Proceedings, Part II, volume 12657 of Lecture Notes in Computer Science, pages 418–424. Springer.

Nikos Voskarides, Dan Li, Pengjie Ren, Evangelos Kanoulas, and Maarten de Rijke. 2020. Query resolution for conversational search with limited supervision. In Proceedings ofthe 43rd International ACM SIGIR conference on research and development in Information Retrieval, SIGIR 2020, Virtual Event, China, July 25-30, 2020, pages 921–930. ACM.

<table><tr><td rowspan="2">Statistics</td><td colspan="2">QReCC</td><td colspan="3">CAsT-19 CAsT-20 CAsT-21</td></tr><tr><td>Train</td><td>Test</td><td>Test</td><td>Test</td><td>Test</td></tr><tr><td># Conv.</td><td>10823</td><td>2775</td><td>50</td><td>25</td><td>26</td></tr><tr><td># Questions</td><td>63501</td><td>16451</td><td>479</td><td>216</td><td>239</td></tr><tr><td># Documents</td><td colspan="2">54M</td><td>38M</td><td>38M</td><td>40M</td></tr></table>

Table 6: Data statistics of conversational search datasets.

Zeqiu Wu, Yi Luan, Hannah Rashkin, David Reitter, Hannaneh Hajishirzi, Mari Ostendorf, and Gaurav Singh Tomar. 2022. CONQRR: conversational query rewriting for retrieval with reinforcement learning. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 10000–10014. Association for Computational Linguistics.

Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighof. 2023. C-pack: Packaged resources to advance general chinese embedding. CoRR, abs/2309.07597.

Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N. Bennett, Junaid Ahmed, and Arnold Overwijk. 2021. Approximate nearest neighbor negative contrastive learning for dense text retrieval. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net.

Shi Yu, Jiahua Liu, Jingqin Yang, Chenyan Xiong, Paul N. Bennett, Jianfeng Gao, and Zhiyuan Liu. 2020. Few-shot generative conversational query rewriting. In Proceedings ofthe 43rd International ACM SIGIR conference on research and development in Information Retrieval, SIGIR 2020, Virtual Event, China, July 25-30, 2020, pages 1933–1936. ACM.

Shi Yu, Zhenghao Liu, Chenyan Xiong, Tao Feng, and Zhiyuan Liu. 2021. Few-shot conversational dense retrieval. In SIGIR ’21: The 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, Virtual Event, Canada, July 11-15, 2021, pages 829–838. ACM.

Yutao Zhu, Huaying Yuan, Shuting Wang, Jiongnan Liu, Wenhan Liu, Chenlong Deng, Haonan Chen, Zhicheng Dou, and Ji-Rong Wen. 2023. Large language models for information retrieval: A survey. CoRR, abs/2308.07107.

## A Appendix

## A.1 Vec2Text

Due to the necessity of transforming session embeddings into explicit and interpretable text, we integrate the Vec2Text model into our architecture. The utilization of Vec2Text (Morris et al., 2023) is driven by its capability to effectively invert the full text represented in dense text embeddings, aligning with our goal to provide interpretability of session embeddings in conversational dense retrieval.

The Vec2Text model aims for the complete inversion of input text from its embedding; it leverages the difference between a hypothesis embedding and a ground-truth embedding to make discrete adjustments to the text hypothesis. Specifically, the Vec2Text model begins by proposing an initial hypothesis and subsequently refines this hypothesis through iterative corrections. The goal is to progressively bring the hypothesis’s embedding $\hat { e } ^ { t }$ closer to the target embedding e.

The Vec2Text comprises two models: the inversion model and the corrector model. Firstly, the inversion model endeavors to invert encoder ϕ by learning a distribution of texts given embeddings $p \left( x \mid e , \theta \right)$ . The training objective for the inversion model is to find $\theta$ using maximum likelihood estimation:

$$
\theta = a r g \operatorname* { m a x } _ { \hat { \theta } } E _ { x \sim D } \left[ p \left( x \mid \phi \left( x \right) ; \theta \right) \right]
$$

On the basis of the simple learned inversion hypothesis $x ^ { 0 }$ , the corrector model iteratively refines this hypothesis via marginalizing over intermediate hypotheses:

$$
p \left( x ^ { ( t + 1 ) } \mid e \right) = \sum _ { x ^ { ( t ) } } p \left( x ^ { ( t ) } \mid e \right) p \left( x ^ { ( t + 1 ) } \mid e , x ^ { ( t ) } , { \hat { e } } ^ { ( t ) } \right)
$$

where $\hat { e } ^ { ( t ) } = \phi \left( x ^ { ( t ) } \right)$

## A.2 More Detailed Experimental Settings

## A.2.1 Details of Datasets

The statistical data for each dataset are presented in Table 6 and a more detailed description is provided as follows:

QReCC is a large dataset designed for the study of conversational search. Every query is accompanied by an answer and a human-generated rewrite. QReCC includes a total of 13,598 dialogues featuring 79,952 queries. Of these, 9.3K conversations originate from QuAC questions; 80 from TREC CAsT; and 4.4K from NQ. Additionally, 9% of the questions within QReCC lack corresponding answers.

CAsT-19, CAsT-20, and CAsT-21 are three widely used conversational search datasets released by TREC Conversational Assistance Track (CAsT). For CAsT-19, relevance assessments are available for 173 queries within 20 test conversations. For CAsT-20, the majority of queries are accompanied by relevance judgments. For CAsT-21, there are relevance judgments for 157 queries within 18 test conversations. CAsT-19 and CAsT-20 share the same corpus, whereas CAsT-21 employs a different one.

<table><tr><td>Context: (CAsT-19 Session 79)  $q _ { 1 }$  :What is taught in sociology?  $q _ { 2 } \colon$  What is the main contribution of Auguste Comte?  $q _ { 3 } \colon$  What is the role of positivism in it?</td></tr><tr><td> $q _ { 4 } \mathrm { : }$  What is Herbert Spencer known for?  $q _ { 5 } \colon$  How is his work related to Comte? Current Query(35.2):</td></tr><tr><td>What is the functionalist theory? CONVINV (46.9): what is comte&#x27;s functionalist theory in philosophy? TX-Human(46.9):</td></tr><tr><td>what is comte&#x27;s functionalist theory in philosophy? TX-Inversion(20.7): What is the functionalist theory?</td></tr><tr><td>Human Rewrite(38.3): What is the functionalist theory in sociology?</td></tr></table>

Table 7: An additional case illustrating the distinction in utilizing rewriting enhancement for transformed text. The numbers in parentheses indicate the retrieval performance NDCG@3 of the transformed text. Notably, the number in parentheses under Current Query represents the retrieval results of the original session embedding, not that of the current query statement.

## A.2.2 Implementation Details

During the training process, we conduct the training experiments of the Vec2Text model on four Nvidia A100 40G GPUs. We use bf16 precision and AdamW optimizer with 0.001 as the initial learning rate. The strategy to adjust the learning rate is constant with warm-up. We choose T5 (Raffel et al., 2020) as the backbone model. The number of times to repeat embedding along the T5 input sequence length is set to 16.

During the inference process, the sequence beam width and the invert num steps are set to 10 and 30, respectively. The maximum input length and the maximum response length are set to 512 and 100, respectively. The dense retrieval is performed using Faiss (Johnson et al., 2021).

## A.3 Examples of Human Evaluation

Examples of the three metrics for human evaluation are shown in Table 8.

## A.4 Supplement of Case Study

In this section, We provide an additional case in Table 7 for analysis. The transformed text not only includes the keyword of the original query "functionalist theory", but also enriches it with additional information "comte" and "philosophy", thus yielding a retrieval performance that surpasses that of the human rewrite.

## A.5 Experiments with Different Retrievers

Investigations of Based on Different Ad-hoc Retrievers on CAsT-19, CAsT-20, and CAsT-21 datasets are shown in Table 9, Table 10 and Table 11, separately.

<table><tr><td rowspan="2">Clarity</td><td>Context: 91: What is throat cancer? q2: Is it treatable?</td></tr><tr><td>q3: Tell me about lung cancer. q4: What are its symptoms?  $q _ { 5 } \colon$  Can it spread to the throat?</td></tr><tr><td>Coherence</td><td>Positive Example:What is throat cancer and what is the first sign of it? Negative Example: what is the first sign of throat or lung cancer? Context:  $q _ { 1 } \colon$  What are the different types of sharks?  $q _ { 2 } \colon$  Are sharks endangered? If so, which species? q3: Tell me more about tiger sharks.</td></tr><tr><td></td><td>Query: What is the largest ever to have lived on Earth? Human Rewrite: What is the largest shark ever to have lived on Earth? Positive Example:What&#x27;s the largest sharks to have ever lived on earth? Negative Example: What is the largest ever to have lived on earth, shark sharks? Context: What are the origins of popular music?</td></tr><tr><td>Completeness</td><td> $q _ { 1 } \colon$   $q _ { 2 } \colon$  What are its characteristics? q3: What technological developments enabled it? Query: When and why did people start taking pop seriously? Human Rewrite: When and why did people start taking pop music seriously? Positive Example:When did people start taking pop music seriously. and why? Negative Example: What causes pop music and when did it begin to be taken seriously?</td></tr></table>

Table 8: Examples of the criteria of three metrics of human evaluation.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Retriever</td><td colspan="3">Retrieval Performance</td><td colspan="2">Interpretability</td></tr><tr><td>MRR</td><td>NDCG@3</td><td>Recall@100</td><td>similarity</td><td>human evaluation</td></tr><tr><td rowspan="5">KD</td><td>KD-GTR CONVINV</td><td>74.9 74.2(-0.7)</td><td>46.9 44.9(-2.0)</td><td>41.9 43.0(+1.1)</td><td></td><td>4.40</td></tr><tr><td>KD-ANCE</td><td>72.0</td><td>44.4</td><td>34.2</td><td>0.958</td><td></td></tr><tr><td>CONVINV</td><td>72.0(0.0)</td><td>44.5(+0.1)</td><td>34.3(+0.1)</td><td>0.999</td><td>4.90</td></tr><tr><td>KD-BGE</td><td>69.5</td><td>44.0</td><td>41.2</td><td>-</td><td>一</td></tr><tr><td>CONVINV</td><td>69.9(+0.4)</td><td>45.4(+1.4)</td><td>41.5(+0.3)</td><td>0.972</td><td>4.33</td></tr><tr><td rowspan="6">Conv</td><td>Conv-GTR</td><td>53.8</td><td>31.1</td><td>34.6</td><td></td><td></td></tr><tr><td>CONVINV</td><td>56.4(+2.6)</td><td>33.1(+2.0)</td><td>37.0(+2.4)</td><td>0.778</td><td>3.27</td></tr><tr><td>Conv-ANCE</td><td>62.8</td><td>34.5</td><td>29.6</td><td></td><td></td></tr><tr><td>CONVINV</td><td>47.6(-15.2)</td><td>27.2(-7.3)</td><td>22.0(-7.6)</td><td>0.974</td><td>4.13</td></tr><tr><td>Conv-BGE</td><td>59.6</td><td>35.1</td><td>36.4</td><td></td><td></td></tr><tr><td>CONVINV</td><td>55.2(-4.4)</td><td>32.0(-3.1)</td><td>37.1(+0.7)</td><td>0.736</td><td>3.47</td></tr><tr><td rowspan="5">KD</td><td>KD-GTR</td><td>49.5</td><td>35.9</td><td>46.9</td><td></td><td></td></tr><tr><td>CONVINV KD-ANCE</td><td>47.6(-1.9)</td><td>34.4(-1.5)</td><td>44.0(-2.9)</td><td>0.952</td><td>3.80</td></tr><tr><td>CONVINV</td><td>51.0</td><td>35.8</td><td>38.6</td><td></td><td></td></tr><tr><td>KD-BGE</td><td>49.2(-1.8) 44.7</td><td>34.1(-1.7) 31.9</td><td>39.9(+1.3) 46.8</td><td>0.999</td><td>4.60</td></tr><tr><td>CONVINV</td><td>43.3(-1.4)</td><td>30.5(-1.4)</td><td>45.3(-1.5)</td><td></td><td></td></tr><tr><td rowspan="6">Conv</td><td>Conv-GTR</td><td>27.9</td><td>18.4</td><td>31.8</td><td>0.966</td><td>4.25</td></tr><tr><td>CONVINV</td><td>27.2(-0.7)</td><td></td><td>30.4(-1.4)</td><td></td><td></td></tr><tr><td>Conv-ANCE</td><td>38.4</td><td>18.5(+0.1) 25.8</td><td>31.5</td><td>0.719</td><td>3.00</td></tr><tr><td>CONVINV</td><td>27.8(-10.6)</td><td></td><td>22.8(-8.7)</td><td></td><td>-</td></tr><tr><td>Conv-BGE</td><td>30.7</td><td>18.6(-7.2) 20.9</td><td>35.4</td><td>0.972</td><td>2.93</td></tr><tr><td>CONVINV</td><td>31.5(+0.8)</td><td>21.4(+0.5)</td><td>34.0(-1.4)</td><td>0.733</td><td>3.13</td></tr></table>

Table 9: Retrieval performance and interpretability of generated transformed text based on different ad-hoc retrievers on CAsT-19 Dataset. The best performance is bold.

Table 10: Retrieval performance and interpretability of generated transformed text based on different ad-hoc retrievers on CAsT-20 Dataset. The best performance is bold.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Retriever</td><td colspan="3">Retrieval Performance</td><td colspan="2">Interpretability</td></tr><tr><td>MRR</td><td>NDCG@3</td><td>R@100</td><td>similarity</td><td>human evaluation</td></tr><tr><td rowspan="5">KD</td><td>KD-GTR CONVINV</td><td>54.7 54.7(0.0)</td><td>36.4</td><td>55.4 55.1(-0.3)</td><td></td><td>3.53</td></tr><tr><td>KD-ANCE</td><td>52.8</td><td>37.4(+1.0) 36.9</td><td>50.8</td><td>0.945</td><td></td></tr><tr><td>CONVINV</td><td>55.8(+3.0)</td><td>37.4(+0.5)</td><td>53.1(+2.3)</td><td>0.998</td><td>4.07</td></tr><tr><td>KD-BGE</td><td>57.9</td><td>41.2</td><td>56.0</td><td></td><td>一</td></tr><tr><td>CONVINV</td><td>59.8(+1.9)</td><td>41.1(-0.1)</td><td>54.4(-1.6)</td><td>0.954</td><td>4.25</td></tr><tr><td rowspan="6">Conv</td><td>Conv-GTR</td><td>42.2</td><td>28.4</td><td>46.4</td><td></td><td></td></tr><tr><td>CONVINV</td><td>41.9(-0.3)</td><td>28.2(-0.2)</td><td>41.7(-4.7)</td><td>0.664</td><td>2.80</td></tr><tr><td>Conv-ANCE</td><td>41.1</td><td>25.2</td><td>42.1</td><td></td><td></td></tr><tr><td>CONVINV</td><td>30.1(-11)</td><td>16.9(-8.3)</td><td>31.2(-10.9)</td><td>0.973</td><td>2.73</td></tr><tr><td>Conv-BGE</td><td>48.4</td><td>32.8</td><td>51.1</td><td></td><td></td></tr><tr><td>CONVINV</td><td>50.5(+2.1)</td><td>32.4(-0.4)</td><td>50.5(-0.6)</td><td>0.740</td><td>3.07</td></tr></table>

Table 11: Retrieval performance and interpretability of generated transformed text based on different ad-hoc retrievers on CAsT-21 Dataset. The best performance is bold.