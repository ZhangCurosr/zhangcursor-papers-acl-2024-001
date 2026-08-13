# CopyNE: Better Contextual ASR by Copying Named Entities

Shilin Zhou<sup>1</sup>, Zhenghua Li<sup>1</sup>∗, Yu Hong<sup>1</sup>,

Min Zhang<sup>1</sup>, Zhefeng Wang<sup>2</sup>, Baoxing Huai<sup>2</sup>

<sup>1</sup>School of Computer Science and Technology, Soochow University, China <sup>2</sup>Huawei Cloud, China

slzhou.cs@outlook.com; {zhli13,hongy,minzhang}@suda.edu.cn {wangzhefeng,huaibaoxing}@huawei.com

## Abstract

End-to-end automatic speech recognition (ASR) systems have made significant progress in general scenarios. However, it remains challenging to transcribe contextual named entities (NEs) in the contextual ASR scenario. Previous approaches have attempted to address this by utilizing the NE dictionary. These approaches treat entities as individual tokens and generate them token-by-token, which may result in incomplete transcriptions of entities. In this paper, we treat entities as indivisible wholes and introduce the idea of copying into ASR. We design a systematic mechanism called CopyNE, which can copy entities from the NE dictionary. By copying all tokens of an entity at once, we can reduce errors during entity transcription, ensuring the completeness of the entity. Experiments demonstrate that CopyNE consistently improves the accuracy of transcribing entities compared to previous approaches. Even when based on the strong Whisper, CopyNE still achieves notable improvements.

## 1 Introduction

End-to-end automatic speech recognition (ASR) systems have achieved impressive performance in general scenarios (Chan et al., 2016; Rao et al., 2017; Gulati et al., 2020; Boulianne, 2022). However, in the contextual ASR scenario where speech often contains numerous contextual entities, it remains a challenge for ASR systems to get accurate transcriptions (Alon et al., 2019; Jayanthi et al., 2023). For instance, when utilizing personal voice assistants like Siri or Alexa, it is common to encounter contextual entities such as personal names, place names, and organization names. ASR models trained solely on speech-text data often struggle to transcribe these personalized entities due to their infrequent occurrence in the training set (Sathyendra et al., 2022). Since contextual entities always cover a wealth of semantic information. It is important to improve the accuracy of transcribing entities for downstream natural language processing tasks such as information retrieval and spoken language understanding (Ganesan et al., 2021; Wu et al., 2022).

![](images/c455536902cd78706558c5765800dc6f1d74225da28f09caaba28884def33b37.jpg)  
Figure 1: An example with homophonic errors. Pinyin is the Mandarin pronunciation of each token. The red text indicates the wrongly predicted token.

Recently, researchers have started leveraging the information of textual modality as additional contextual knowledge to help contextual ASR. The most typical approach, premised on the assumption that entities are already known before, use a contextual named entity (NE) dictionary as contextual knowledge (Chen et al., 2019; Jain et al., 2020; Han et al., 2021; Huber et al., 2021; Fu et al., 2023). Two representative approaches are “contextual listen, attend and spell” (CLAS) (Pundak et al., 2018) and contextual bias attention (CBA) (Zhang and Zhou, 2022). CLAS employs the knowledge of the dictionary to aid the prediction of each token. They use dictionary representation as extra inputs for token prediction in the decoder. The decoder attends to each entity, and the dictionary representation is an aggregated representation of all entities, weighted by the attention scores. CBA extends CLAS and uses an extra training loss. The loss explicitly makes use of the attention scores and force the model attend to a proper entity in the dictionary if the token to be predicted is related with the entity.

Previous methods have achieved considerable improvements, especially in transcribing entities. However, they all treat entities as individual tokens. These models utilize contextual knowledge to aid in predicting independent tokens without considering the role of these tokens in constituting a complete entity. In other words, a multi-token entity is broken into isolated tokens during decoding. We argue that this is problematic. For instance, model may erroneously generate the subsequent tokens of an entity, despite correctly producing the preceding tokens. As shown in Figure 1, when transcribing the speech “他来自安徽铜陵” (He comes from Anhui Tongling), an incorrect output of “他来自安徽铜 $\mathit { f } _ { \vec { \tau } } ^ { k , \vec { \mathbf { \tau } } }$ (He comes from Anhui copper bell) is obtained. Despite the model’s awareness of the location entity “铜陵” in the NE dictionary and its accurate prediction of the first token “铜”, it mistakenly transcribes “陵” (ling) as “铃” (bell) during the tokenlevel prediction process. This occurs because the model predicts tokens independently, neglecting the integrity of the token span as a complete entity. Furthermore, “陵” and “铃” share the same pronunciation “líng”, with “铃” being a more frequently occurring token in the training data. Consequently, the model tends to generate the wrong token “铃”.

In this paper, we propose a new approach for contextual ASR called CopyNE. Unlike previous approaches, we view entities as indivisible wholes. To the best of our knowledge, we are the first to introduce the idea of copying into ASR. We design a systematic and effective mechanism to copy entities from a dictionary. Specifically, CopyNE uses a copy loss that guides the model to copy the correct entity from the dictionary. During inference, our CopyNE has the flexibility to either predict a token from the token vocabulary or copy an entity from the NE dictionary at each decoding step. By copying multiple tokens simultaneously, we can alleviate errors within the entity, thus ensuring the token span as a complete entity.

Experiments on Chinese Aishell (Bu et al., 2017), ST-cmds<sup>1</sup>, and English Eng (Yadav et al., 2020) datasets show that our CopyNE achieves significant improvements across all scenarios, particularly in the contextual ASR scenario. Compared to previous methods using dictionary, CopyNE achieves relative reductions in CER of 13.5% and

20.8% on Aishell and ST-cmds in the contextual scenario. Notably, CopyNE shows more remarkable improvements when it comes to transcribing entities, with relative reductions of 55.4% and 53.9% in the NE-CER metric on Aishell and STcmds. Moreover, when based on Whisper (Radford et al., 2022) and evaluated in its domain of expertise, Eng dataset, CopyNE still achieves an impressive 6.4% and 16.8% relative reductions in WER and NE-WER. We will release our codes, configurations, and models at https://github. com/zsLin177/CopyNE.

## 2 The CTC-Transformer Model

In this work, we build our proposed approach on the end-to-end CTC-Transformer model, since it is the most widely used and achieves competitive performance in the ASR field (Hori et al., 2017; Kim et al., 2017; Miao et al., 2020; Omachi et al., 2021; Gong et al., 2022). However, it is worth noting that our idea can be applied to other ASR approaches as well.

The CTC-Transformer is built upon the seq-toseq Transformer (Vaswani et al., 2017), with a connectionist temporal classification (CTC) layer added after the audio encoder. As shown in Figure 2, it takes a sequence of acoustic frames $\pmb { X } = ( \pmb { x } _ { 1 } , . . . , \pmb { x } _ { T } )$ as input and generates the corresponding transcription text $\pmb { y } = ( y _ { 1 } , . . . , y _ { U } )$ as output. The model consists of two main components: an encoder and a decoder. First, the encoder encodes the acoustic frames X into hidden states $\pmb { H } = ( h _ { 1 } , . . . , h _ { T } )$ . Then, the decoder predicts the target sequence y in an auto-regressive manner. At each decoding step u, the decoder predicts the next target token $y _ { u + 1 }$ based on the encoder’s output H and the previously predicted tokens $y _ { \le u } = ( y _ { 1 } , . . . , y _ { u } )$ . This process is expressed as follows:

$$
H = \operatorname { A u d i o E n c o d e r } ( X )\tag{1}
$$

$$
\mathop { } \mathbf { { \Gamma } } d _ { u } = \mathrm { D e c o d e r } ( y _ { \le u } , H )\tag{2}
$$

$$
P ( y _ { u + 1 } | y _ { \leq u } ) = \mathrm { s o f t m a x } ( W d _ { u } + b )\tag{3}
$$

Here, $\boldsymbol { d } _ { u } \in \mathbb { R } ^ { d }$ denotes the hidden state at step $u ,$ and $P ( y _ { u + 1 } | y _ { \leq u } )$ is the posterior distribution of predicting token $y _ { u + 1 }$ $\bar { \pmb { W } } \in \mathbb { R } ^ { | \mathcal { V } | \times d }$ and $b \in$ $\mathbb { R } ^ { | \hat { \nu } | }$ are learned parameters, where is the token vocabulary, and $| \nu |$ is the size of the vocabulary.

![](images/2e82977e673cd00639440812cf52e2480d141976e0015af37d42f3ac842f9866.jpg)  
Figure 2: The CTC-Transformer model.

The loss of Transformer, $\mathcal { L } _ { t r a n s } ( \pmb { y } )$ , comes from minimizing the negative log probability of y.

$$
\mathcal { L } _ { t r a n s } ( \pmb { y } ) = - \sum _ { u = 0 } ^ { U - 1 } \log P ( y _ { u + 1 } | y _ { \leq u } )\tag{4}
$$

As commonly used in previous works, the CTC loss is also applied here. CTC aligns each acoustic frame with a token from left to right. For a given target sequence $^ { y , }$ there may be multiple valid alignments. The CTC loss is derived from maximizing the sum of these valid alignments, and has been proved to be able to enhance the representational capacity of the audio encoder (Kim et al., 2017). Finally, the overall loss is the a weighted sum of the $\mathcal { L } _ { t r a n s } ( \pmb { y } )$ and $\mathcal { L } _ { c t c } ( \pmb { y } )$ , as follows:

$$
\begin{array} { r } { \mathcal { L } ( \pmb { y } ) = \lambda \mathcal { L } _ { t r a n s } ( \pmb { y } ) + ( 1 - \lambda ) \mathcal { L } _ { c t c } ( \pmb { y } ) } \end{array}\tag{5}
$$

where λ is a hyper-parameter that determines the relative weight of each loss term.

In inference, the model selects the most probable transcription using beam search as follows:

$$
\pmb { \hat { y } } = \arg \operatorname* { m a x } _ { \pmb { y } } ( \sum _ { \ b { u } } \log P ( y _ { { \boldsymbol { u } } + 1 } | y _ { \leq { \boldsymbol { u } } } ) )\tag{6}
$$

Here, there are many ways to use scores in decoding, such as combining CTC scores and Transformer scores as in training, or using CTC-prefix beam search followed by re-scoring with Transformer to select the optimal result. To compare with most previous works, we use the simplest decoding strategy, as shown in Equation 6.

## 3 Our CopyNE Model

This section describes our proposed CopyNE model. The basic idea is that the model incorporates a contextual NE dictionary as external knowledge and can choose to directly copy NEs from the dictionary. We design a systematic framework to implement the idea. During training, a copy loss is designed to encourage the model to copy corresponding entities from the dictionary. During inference, at each generation step, the model can either predict a single token from token vocabulary or directly copy a entity from the given dictionary.

![](images/32852026cba7342dd214adb9173a2e18cb5f7e34d8e63e8e01c83fea5f7bebeb.jpg)  
Figure 3: Our CopyNE model.

## 3.1 The Model Framework

Figure 3 illustrates the framework of our CopyNE model, which shares the same encoder as the CTC-Transformer model, but with a distinct decoder. In the decoder, we introduce an extra NE encoder that takes the NE dictionary as input and encodes it into NE representations. Then, we use a dot-product attention module to compute copy probabilities based on the obtained NE representations, which are then aggregated to form the overall dictionary (Dict) representation. The decoder can not only utilize copy probabilities to select entities for copying but also leverage the Dict representation to aid in predicting the next token.

NE Representation. We denote an NE dictionary as $E = ( e _ { 0 } , e _ { 1 } , . . . , e _ { N } )$ . We use $e _ { 0 } \ = \ \mathcal { O }$ as a pseudo entity to handle the case where the text to be transcribed has no relation to any entity and the model should not copy any entity at current step.

For each entity $e _ { i } ,$ , we apply a multi-layer LSTM as the NE encoder to encode the token sequence and use the last hidden state of the NE encoder as the entity representation. It is a popular practice in previous contextual ASR works (Pundak et al., 2018; Zhang and Zhou, 2022).

$$
z _ { i } = \mathrm { L S T M } ( e _ { i } )\tag{7}
$$

After that, we get entity representations $z \_ =$ $( z _ { 0 } , z _ { 1 } , . . . , z _ { N } )$ , where $\boldsymbol { Z } \in \mathbb { R } ^ { N \times d }$

Copy Probability. Once the NE representations are obtained, the copy probability is computed by a dot-product attention mechanism. It is used to

determine which entity to copy. First, we compute the attention score $a _ { u } ^ { e _ { i } }$ for entity $e _ { i }$ at step u as follows:

$$
a _ { u } ^ { e _ { i } } = \frac { ( W _ { q } d _ { u } ) ^ { \top } ( W _ { k } z _ { i } ) } { \sqrt { d } }\tag{8}
$$

where $W _ { q } , W _ { k } ~ \in ~ \mathbb { R } ^ { d _ { a } \times d }$ are two learned parameters. $d _ { a }$ denotes the dimension of the attention. After that we obtain the attention probability $P _ { c } ( e _ { i } | y _ { \leq u } )$ for entity $e _ { i }$ by softmax.

$$
P _ { c } ( e _ { i } | \boldsymbol { y } _ { \le u } ) = \frac { \exp ( a _ { u } ^ { e _ { i } } ) } { \sum _ { e _ { j } \in E } \exp ( a _ { u } ^ { e _ { j } } ) }\tag{9}
$$

Here, $P _ { c } ( e _ { i } | y _ { \leq u } )$ not only represents the attention probability of $e _ { i }$ but also naturally serves as the copy probability for the entity. During inference, we use the copy probabilities to select the entities for copying.

Dict Representation. With copy (attention) probabilities, we can obtain the Dict representation $\mathbf { \Delta } _ { r _ { u } }$ at decoding step $u .$ It is used to help the prediction of subsequent tokens. Specifically, $\boldsymbol { r } _ { u } \in \mathbb { R } ^ { d }$ is computed by weighted summing the entity representations with the copy (attention) probabilities.

$$
r _ { u } = \sum _ { e _ { i } \in E } P _ { c } ( e _ { i } | y _ { \leq u } ) z _ { i }\tag{10}
$$

Dict-enhanced Prediction. Finally, we get the overall Dict representation and copy probabilities. Following Pundak et al. (2018), the Dict representation is applied to help the generation of the next token. So Equation 3 is extended as follows:

$$
P ( y _ { u + 1 } | y _ { \leq u } , E ) = \mathrm { s o f t m a x } ( W \left[ d _ { u } , r _ { u } \right] + k\tag{11}
$$

## 3.2 Training

During training, to guide the model in selecting correct entities from the NE dictionary for copying, we introduce an additional copy loss $\mathcal { L } _ { c o p y } .$ . First, based on the ground truth transcription y and the NE dictionary, we construct a copy target $\sigma _ { u + 1 }$ for each decoding step $u ,$ telling the model whether to copy an entity from the dictionary or not, and which one to copy. Then we compute the copy loss $\mathcal { L } _ { c o p y }$ according to the copy target $\sigma _ { u + 1 }$ and the copy probability $P _ { c } ( \sigma _ { u + 1 } | y _ { \leq u } )$

The Computation of Copy Loss. Provided that we have an NE dictionary $E ^ { b }$ , we construct a copy target, denoted as $\sigma _ { u + 1 }$ , for decoding step u. In order to build the copy target, we perform maximum matching on the transcription text $\textbf {  { y } }$ from left to right based on the dictionary $E ^ { b }$ . If the token span $\pmb { y } _ { i , j } = ( y _ { i } , . . . , y _ { j } )$ matches the k-th entity $e _ { k }$ in $E ^ { b }$ , then we set the copy target $\sigma _ { i } = e _ { k }$ , and $\sigma _ { i + 1 \sim j } = \emptyset$ . This indicates that the model can copy the k-th entity from the dictionary at decoding step $i - 1$ , but cannot copy any entity from decoding step i to $j - 1$ . When it comes to a span of length 1, $\mathbf { i . e . , } \ i \ = \ j$ , during the left-to-right maximum matching process, we also set $\sigma _ { i }$ to $\mathcal { O } ^ { 2 }$

For example, in the instance shown in Figure 3, the span $\therefore \frac { 2 } { 5 } x ^ { 2 } =$ (An hui) matches the second entity in the dictionary, and the span “铜陵” (Tong ling) matches the first entity in the dictionary. This means that at steps 0 and 2, the model can choose to copy the second and first entities from the dictionary, respectively. Therefore, $\sigma _ { 1 } = e _ { 2 }$ and $\sigma _ { 3 } = e _ { 1 }$ while $\sigma _ { 2 } = \emptyset$ and $\sigma _ { 4 } = \emptyset$

After constructing all the copy targets $\sigma =$ $( \sigma _ { 1 } , . . . , \sigma _ { U } )$ , we can compute the copy loss as follows:

$$
\mathcal { L } _ { c o p y } ( \pmb { \sigma } ) = - \sum _ { u = 0 } ^ { U - 1 } \log P _ { c } ( \sigma _ { u + 1 } | y _ { \le u } )\tag{12}
$$

where $P _ { c } ( \sigma _ { u + 1 } | y _ { \leq u } )$ is the copy probability computed in Equation 9, meaning the probability of copying entity $\sigma _ { u + 1 }$ at decoding step u. It is worth noting that the copy loss and the bias loss in CBA have fundamental differences. The bias loss in CBA provides information to each token, including tokens within entities, about which entity to attend to. In contrast, our copy loss solely instructs the model to copy the entity from the dictionary at the first token of the entity.

Finally, the loss in our CopyNE model is formed as follows:

$$
\mathcal { L } = \lambda \mathcal { L } _ { t r a n s } ( \pmb { y } ) + ( 1 - \lambda ) \mathcal { L } _ { c t c } ( \pmb { y } ) + \mathcal { L } _ { c o p y } ( \pmb { \sigma } )\tag{13}
$$

Dictionary Construction. To construct the copy target and compute the copy loss, we should first build a contextual NE dictionary for training. Provided that the entities have been labeled in the dataset, we build a NE dictionary $E ^ { b }$ for each data batch following previous works.

Firstly, to construct $E ^ { b }$ , we extract all entities in the instances of this batch and add them to the dictionary. For instances that do not contain any entities, in order to ensure an adequate number of positive examples, we randomly select one or two substrings of length 2 or 3 from the transcription and include them in the dictionary as pseudo entities. In order to improve the ability of copying the correct entity from a wide range of entities, we also extract additional negative entities from the training set. We analyze the influence of the quantity of negative entities on the model. Due to page constraints, we have included this section in §C.

## 3.3 Inference

During inference, unlike previous token-level approaches, our model has the flexibility to predict either a token from the vocabulary or an entity from the NE dictionary. By copying the tokens of an entity at once, our CopyNE model can avoid errors that occur when predicting multiple tokens separately. As shown in Figure 3, our CopyNE model can directly copy the two entities “安徽” and “铜 陵” from the dictionary.

Specifically, at step u, our prediction is based on both the model’s probability for a token v, i.e., $P ( v | \hat { y } \leq u , E )$ , and the copy probability for an entity e, i.e., $P _ { c } ( e | \hat { y } _ { \le u } )$ . The former represents the probability of predicting a token v from the token vocabulary, while the latter is normalized on all entities, originally indicating the attention probability over entity e, which can be naturally interpreted as the probability of copying entity e from the dictionary. To consider both probabilities on the same scale, we devise an elegant decoding strategy by taking use of the copy probability of $\mathcal { O } ,$ i.e., $P _ { c } ( \emptyset | \hat { y } _ { \leq u } )$ , and re-normalize the probabilities to create an unified searching space $Q$

$$
Q ( i | \hat { y } _ { \le u } ) = \left\{ \begin{array} { l l } { P _ { c } ( \emptyset | \hat { y } _ { \le u } ) P ( i | \hat { y } _ { \le u } , E ) , \ i \in \mathcal { V } } \\ { P _ { c } ( i | \hat { y } _ { \le u } ) , \ i \in E , i \neq \emptyset } \end{array} \right.\tag{14}
$$

Here, to ensure the sum of the probabilities of all elements is 1, we use $P _ { c } ( \emptyset | \hat { y } _ { \leq u } )$ as a prior probability, representing the probability of the text to be transcribed has no relation with the entities in the dictionary and the text should be generated from the token vocabulary. If the element is from the token vocabulary , we obtain the probability by multiplying the prior probability and the model’s probability for the token. Otherwise, we use the copy probability directly.

However, in our experiments, we observe that the model occasionally selects irrelevant entities for copying. To enhance the quality of copying, we introduce a confidence threshold $\gamma$ during decoding to filter out low-confidence copies. Specifically, we set $P _ { c } ( i | \hat { y } _ { \le u } ) = 0 , i \in E , i \neq \emptyset$ , and $P _ { c } ( \emptyset | \hat { y } _ { \le u } ) = 1$ when max $\{ P _ { c } ( i | \hat { y } _ { \leq u } ) | i \in E , i \neq$ $\emptyset \} < \gamma$ . This means that if the model’s maximum copy probability over real entities is less than γ, it is prevented from copying entities from the dictionary and instead generates tokens from the token vocabulary. In section 4.2, we discuss the influence of the γ in detail.

Finally, we use beam search to select the best element at each step to form the final prediction<sup>3</sup>.

$$
\hat { \pmb y } = \arg \operatorname* { m a x } _ { \pmb y } ( \sum _ { \boldsymbol u } \log Q ( i | \boldsymbol y _ { \le u } ) )\tag{15}
$$

## 4 Experiments

## 4.1 Experimental Setup

Datasets. Experiments on Chinese Mandarin are conducted on two widely used datasets, Aishell (Bu et al., 2017) and ST-cmds<sup>4</sup>. We use the Eng dataset released by Yadav et al. (2020) to perform experiments on English. Furthermore, to compare the performance of different methods in contextual ASR scenarios where speeches contain entities, we extract instances containing entities from the dev and test sets, forming the corresponding “ -NE” datasets. Detailed introduction about the datasets can be found in §A.

NE Dictionary. Aishell and ST-cmds were released without entity annotations. In contrast, the Eng dataset was simultaneously released with audio, transcribed text, and corresponding entity annotations. Chen et al. (2022) further annotated entities for Aishell. So, in our experiments with Aishell and Eng, we use the releated entities to build the NE dictionary. For ST-cmds, we use HanLP<sup>5</sup> to get three types of entities: person, location, and organization.

Evaluation Metrics. Character error rate (CER) and word error rate (WER) are used to assess the overall performance of models in Mandarin and

![](images/59c3e56a85968a6b67d905a025511b23fa98924ac3f073ba3639c005bdc3ff9e.jpg)  
Figure 4: Effect of the Confidence Threshold γ.

English ASR tasks. In this paper, to evaluate the model’s entity transcription accuracy, we also employ NE-CER and NE-WER metrics (Han et al., 2021). We align the predicted hypothesis and reference using the minimum edit distance algorithm, and subsequently calculate NE-C(W)ER by measuring the C(W)ER between the entity text in the reference and its counterpart in the hypothesis.

Parameter Setting. The parameter setting in our work is the same as that in most previous works, and the detailed descriptions can be found in §B. To ensure a fair comparison with prior works, we carefully reproduced the CLAS (Pundak et al., 2018) and CBA (Zhang and Zhou, 2022). Moreover, to verify the effectiveness of our approach on pretrained large models, we also conducted experiments on OpenAI Whisper (Radford et al., 2022). Specifically, we use the Whisper model as our transformer encoder and decoder. We choose seeds randomly to run models for 3 times and report the average results.

## 4.2 Results

Analysis about γ. We first investigate the influence of the copy threshold γ during inference. Figure 4 illustrates how the CER changed on the Aishell dev and Aishell-NE dev with different γ values. Our findings reveal that when the threshold is low, the CER is high, indicating that copying results in more errors when the model copies entities with low confidence. As we increase the γ, the CER decreases, indicating improved reliability of our CopyNE when the model had higher confidence. However, when the threshold becomes too high (above 0.9), the model has fewer opportunities to choose to copy entities, resulting in a higher CER. This happens because it becomes more difficult for the model to trigger the copy mechanism. So, we set γ to 0.9 for all experiments and discussions to enhance the robustness of our model.

Results on Chinese. Table 1 and 2 show the CER and NE-CER of different models on the Chinese dataset. In Table 1, we note that while our primary focus is improving transcription of NEs, we also achieve significant improvements in overall text transcription. Without Whisper, our CopyNE model outperforms the previous CBA approach with a 3.2% relative CER reduction on the Aishell Test and 7.7% on the ST-cmds Test. In contextual ASR scenarios, the improvements are even more pronounced, with a 13.5% relative CER reduction on the Aishell-NE Test and 20.8% on the ST-cmds-NE Test. Even with the powerful Whisper, our CopyNE consistently excels, especially on the ST-cmds dataset, with relative reductions of 8.6% and 15.7% on the two test sets, respectively. Additionally, we observed that CLAS performs well on Aishell, closely matching CopyNE, but its performance on ST-cmds is comparatively weaker, sometimes even worse than the Whisper baseline, a reverse pattern also seen with CBA. In contrast, CopyNE consistently performs well across different datasets, demonstrating its better adaptability.

We also present an improved model, i.e. CopyNE†, which features a more powerful conformer encoder. The results show that CopyNE† can achieve further improvements compared to CopyNE.

In this paper, our main goal is improving the transcription of NEs. From the results presented in Table 2, it is evident that our approach exhibits significant improvements in entity transcription compared to previous methods. When not using Whisper, our CopyNE model achieves an impressive relative NE-CER reduction of 55.4% on the Aishell-NE Test and 53.9% on the ST-cmds-NE Test. Even based on the powerful Whisper model, our CopyNE continues to achieve remarkable improvements, with a relative NE-CER reduction of 25.4% and 26.7% on the two test sets. This demonstrates that copying entities from the dictionary significantly improves the accuracy of transcribing entities.

Results on English. Whisper (Radford et al., 2022) has shown strong performance in English, so we directly use it as our baseline for experiments on English. As seen in Table 3, CopyNE still outperforms other methods, achieving a 5.2% relative WER reduction compared to CLAS in the general scenario on the Eng test dataset. In contextual scenarios, CopyNE demonstrates 6.4% relative WER reductions and 16.8% relative NE-WER reductions.

<table><tr><td rowspan="2">Model</td><td colspan="2">Aishell</td><td colspan="2">Aishell-NE</td><td colspan="2">ST-cmds</td><td colspan="2">ST-cmds-NE</td></tr><tr><td>Dev</td><td>Test</td><td>Dev</td><td>Test</td><td>Dev</td><td>Test</td><td>Dev</td><td>Test</td></tr><tr><td>Joint CTC-Transformer</td><td>6.12</td><td>6.70</td><td>7.36</td><td>9.00</td><td>10.63</td><td>10.56</td><td>13.67</td><td>13.63</td></tr><tr><td>CLAS (Pundak et al., 2018)</td><td>6.04</td><td>6.72</td><td>7.06</td><td>8.73</td><td>10.10</td><td>10.09</td><td>12.64</td><td>12.85</td></tr><tr><td>CBA (Zhang and Zhou, 2022)</td><td>6.11</td><td>6.56</td><td>6.73</td><td>8.00</td><td>10.73</td><td>10.72</td><td>12.69</td><td>12.43</td></tr><tr><td>CopyNE</td><td>5.59</td><td>6.35</td><td>5.36</td><td>6.92</td><td>9.76</td><td>9.89</td><td>9.90</td><td>9.84</td></tr><tr><td>CopyNE†</td><td>4.49</td><td>5.03</td><td>4.31</td><td>5.05</td><td>7.78</td><td>7.80</td><td>7.71</td><td>7.40</td></tr><tr><td>Whisper</td><td>5.28</td><td>5.97</td><td>6.32</td><td>7.68</td><td>9.14</td><td>9.06</td><td>12.22</td><td>12.35</td></tr><tr><td>+ CLAS</td><td>4.50</td><td>5.23</td><td>5.30</td><td>6.72</td><td>9.10</td><td>9.20</td><td>12.12</td><td>12.25</td></tr><tr><td>+ CBA</td><td>5.41</td><td>6.06</td><td>6.13</td><td>7.60</td><td>7.96</td><td>7.87</td><td>10.03</td><td>9.96</td></tr><tr><td>+ CopyNE</td><td>4.71</td><td>5.10</td><td>5.40</td><td>6.42</td><td>7.35</td><td>7.19</td><td>8.98</td><td>8.40</td></tr></table>

Table 1: CER on Chinese datasets in general scenarios (Aishell, ST-cmds) and contextual scenarios (Aishell-NE, ST-cmds-NE). † means the model with an improved 12-layer Conformer (Gulati et al., 2020) encoder and averages the parameters of the best 10 epochs when decoding.

<table><tr><td rowspan="2">Model</td><td colspan="2">Aishell-NE</td><td colspan="2">ST-cmds-NE</td></tr><tr><td>Dev</td><td>Test</td><td>Dev</td><td>Test</td></tr><tr><td>Joint CTC-Transformer</td><td>11.64</td><td>14.03</td><td>21.63</td><td>21.41</td></tr><tr><td>CLAS (Pundak et al., 2018)</td><td>11.24</td><td>13.12</td><td>19.70</td><td>20.10</td></tr><tr><td>CBA (Żhang and Zhou, 2022)</td><td>7.78</td><td>9.44</td><td>15.72</td><td>15.92</td></tr><tr><td>CopyNE</td><td>3.00</td><td>4.21</td><td>7.60</td><td>7.34</td></tr><tr><td>w/o Dict repr ru</td><td>4.05</td><td>5.27</td><td>9.35</td><td>9.21</td></tr><tr><td>Whisper</td><td>10.31</td><td>12.24</td><td>21.30</td><td>21.83</td></tr><tr><td>+ CLAS</td><td>8.97</td><td>11.64</td><td>20.82</td><td>21.07</td></tr><tr><td>+ CBA</td><td>9.13</td><td>11.79</td><td>15.91</td><td>15.41</td></tr><tr><td>+ CopyNE</td><td>6.74</td><td>8.79</td><td>11.93</td><td>11.29</td></tr></table>

Table 2: NE-CER (%) on the Chinese datasets.
<table><tr><td rowspan="2">Model</td><td colspan="2">Eng.w</td><td colspan="2">Eng-NE.w</td><td colspan="2">Eng-NE.Nw</td></tr><tr><td>Dev</td><td>Test</td><td>Dev</td><td>Test</td><td>Dev</td><td>Test</td></tr><tr><td>Whisper</td><td>8.54</td><td>8.73</td><td>8.53</td><td>8.71</td><td>28.16</td><td>26.61</td></tr><tr><td>+ ČLAS</td><td>7.90</td><td>8.28</td><td>7.86</td><td>8.31</td><td>27.23</td><td>26.55</td></tr><tr><td>+ CBA</td><td>9.17</td><td>9.52</td><td>9.21</td><td>9.49</td><td>30.01</td><td>30.82</td></tr><tr><td>+ CopyNE</td><td>7.47</td><td>7.85</td><td>7.42</td><td>7.78</td><td>23.29</td><td>22.09</td></tr></table>

Table 3: Results on the English datasets. W and NW denote WER and NE-WER respectively.

Additionally, we observed that CBA lags behind the Whisper baseline. We suspect that this might be due to its approach of encouraging the model to generate entity tokens by modifying Whisper’s output logits, which can disrupt the model’s overall probability distribution, especially given Whisper’s strong fit on English data. On the contrary, our CopyNE is more stable.

## 4.3 Impact of the NE dictionary

Following previous works (Pundak et al., 2018; Han et al., 2021), we report the main results using exact NE dictionaries from the test sets. However, when collecting dictionaries in real scenarios, to ensure the coverage, many unrelated noisy NEs are inevitably added to the dictionary. To analyze the impact of noisy entities on CopyNE, we extract entities from the training set that are not included in the test set as noisy NEs. From the corresponding 2, 3, and 4 rows in Table 4, we can see that the introduction of noisy NEs results in a reduction in the model’s performance. Nevertheless, even with the addition of 6k noisy NEs, resulting in the <sup>dictionary</sup> <sup>size</sup> <sup>being</sup> <sup>quadrupled</sup> <sup>(</sup>×<sup>4),</sup> <sup>CopyNE</sup> continues to outperform CLAS and CBA, despite their reliance on the precise dictionary.

<table><tr><td rowspan="2">Dict size</td><td colspan="2">Dev Test</td></tr><tr><td>CER NE-CER</td><td>CER NE-CER</td></tr><tr><td>×0.85</td><td>5.65 4.27 7.29 5.83</td></tr><tr><td>×0.90 5.56</td><td>3.90 7.15 5.26</td></tr><tr><td>×0.95 5.44</td><td>3.37 7.01 4.68</td></tr><tr><td>×1</td><td>5.36 3.00 6.92 4.21</td></tr><tr><td>×2 5.61</td><td>3.50 7.12 4.82</td></tr><tr><td>×3</td><td>5.85 3.92 7.39 5.18</td></tr><tr><td>×4</td><td>6.02 4.09 7.66 5.56</td></tr></table>

Table 4: The impact of the NE dictionary.

In the more rare cases where some NEs are out of the dictionary (OOD), to analyze CopyNE’s performance in OOD scenarios, we designate some low-frequency NEs from the original dictionary as OOD NEs. These NEs are removed, and decoding is performed using the remaining NEs. From the relevant rows in Table 4, it can be observed that this primarily impacts NE-CER since CopyNE cannot copy missing NEs. However, even when the OOD proportion reaches 15% (Dict size = 0.85), CopyNE still shows commendable performance.

## 4.4 Qualitative Analysis

CopyNE demonstrates significant improvements. To gain further insight into CopyNE’s performance, we conduct a qualitative analysis of its generations.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Transcriptions</td><td rowspan=1 colspan=1>Dictionary</td></tr><tr><td rowspan=1 colspan=1>EnglishGoldCLASCBACopyNE</td><td rowspan=1 colspan=1>A company in Yangluo...阳逻的一家公司...扬罗的一家公司..阳罗的一家公司...[阳逻]的一家公司...</td><td rowspan=3 colspan=1>阳逻(Yangluo)杨丙卿(YangBingqing)冈山(Gangshan)桃太郎体育馆(Taotailanggym)</td></tr><tr><td rowspan=1 colspan=1>EnglishGoldCLASCBACopyNE</td><td rowspan=1 colspan=1>Yang Bingqing served as manager...杨丙卿担任经理...杨澄清担任经理...杨炳卿担任经理...[杨丙卿]担任经理...</td></tr><tr><td rowspan=1 colspan=1>EnglishGoldCLASCBACopyNE</td><td rowspan=1 colspan=1>The Taotailang gym in Gangshan.冈山的桃太郎体育馆冈山的淘汰郎体育馆刚山的淘汰狼体育馆[冈山]的[桃太郎体育馆]</td></tr></table>

Table 5: Generations of different models. Red text indicates errors, while text enclosed in square brackets represents entities that were copied from the dictionary.

Table 5 shows examples of transcriptions from different ASR models. We can see that in the second example, where CBA successfully identified the correct person entity “杨丙卿” from the dictionary and produced a transcription that is close to gold, it still made a mistake by transcribing “炳” instead of “丙” due to the same pronunciation (bˇıng). In contrast, CopyNE can copy all the tokens of the entity from the NE dictioanry. For example, in the third example “冈山的桃太郎体育馆” (The Taotailang gym in Gangshan), CopyNE directly copies the location entity “冈山” (Gangshan) and the organization entity “桃太郎体育馆” (Taotailang gym), achieving a completely correct transcription.

## 5 Related Works

Contextual ASR. Researchers have explored various approaches to help models in the contextual ASR scenario, with the primary approaches being the utilization of external dictionaries and language models (LMs). CLAS (Pundak et al., 2018) was the first to introduce the use of dictionary to aid in prediction. Alon et al. (2019) extend CLAS by adding phonetically similar alternative terms to the dictionary as negative examples, aiming to improve the model’s ability to distinguish entities with similar pronunciations. Huber et al. (2021) propose to utilize the representation of a single entry in the dictionary that is most relevant to the current decoding status. Fu et al. (2023) propose to apply the character-based NE encoder to better capture acoustic features useful for transcribing rare entities. Different from our CopyNE, these methods all treat entities as individual tokens, which may result in incomplete NE transcriptions.

LMs trained on large-scale text data can learn rich linguistic and contextual knowledge, and thus can be used to assist contextual ASR. There are typically two approaches to leverage LMs for contextual ASR. The first approach involves using the dedicated LM to encourage the generation of entity tokens during decoding. (Novak et al., 2012; Aleksic et al., 2015; Zhao et al., 2019a). The second approach is multi-modal pre-training. Researchers have explored joint pre-training of speech and text models, aiming to leverage information from both modalities, and have achieved promising results (Chung et al., 2021; Ao et al., 2022; Zhang et al., 2022). However, compared to using contextual dictionaries, models that rely on LMs tend to have much more parameters, which means that training and deploying require more time and computational resources.

The Copy Mechanism. The copy mechanism can be traced back to the pointer network (Vinyals et al., 2015), which can predict output sequences from the input. The copy mechanism (Gu et al., 2016) extends the pointer network by enabling the model to generate sequences that are not present in the input. According to the source of copying, it can be divided into copying from input text, from document, and from external dictionary.

Copyingfrom Input Text. The copy mechanism is commonly used to copy text from input text. For instance, in text summarization tasks, it is common to employ the copy mechanism to copy keywords from the input text (Cheng and Lapata, 2016; Xu et al., 2020). In grammatical error correction tasks, where only a small portion requires correction, the copy mechanism is used to copy the correct text from the input text (Zhao et al., 2019b).

Copying from Document. In addition to copying from input text, the copy mechanism can be employed to copy text from other texts when the input text is not available. Lan et al. (2023) introduced the copy mechanism in decoder-only language models, where text fragments are selected from a vast amount of documents to generate the target text.

Copying from External Dictionary. In this paper, we introduce a systematic framework, which seamlessly integrates the process of copying from an external dictionary to aid in generation. We believe that our framework can also be applied to other generation tasks.

## 6 Conclusion

In this paper, we consider entities as indivisible elements and introduce a copy mechanism into ASR for the first time to assist in transcribing entities. We devise a systematic copy framework that can copy all the tokens of an entity from the NE dictionary at once, preserving the token span as a complete entity. Our approach demonstrates substantial improvements on both English and Chinese datasets. In summary, CopyNE represents a significant advancement in contextual ASR, providing a promising direction in this field.

## Limitations

From our experiments, we have found that an excessive number of noisy entities can impact the performance. As part of our future work, we intend to explore methods for dynamically filtering out interfering entities from the dictionary during the decoding process.

## Acknowledgements

We would like to thank the anonymous reviewers for their valuable comments. This work was supported by National Natural Science Foundation of China (Grant No. 62176173 and 62336006), and a Project Funded by the Priority Academic Program Development of Jiangsu Higher Education Institutions.

## References

Petar Aleksic, Mohammadreza Ghodsi, Assaf Michaely, Cyril Allauzen, Keith Hall, Brian Roark, David Rybach, and Pedro Moreno. 2015. Bringing contextual information to google speech recognition.

Uri Alon, Golan Pundak, and Tara N Sainath. 2019. Contextual speech recognition with difficult negative training examples. In 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6440–6444.

Junyi Ao, Rui Wang, Long Zhou, Chengyi Wang, Shuo Ren, Yu Wu, Shujie Liu, Tom Ko, Qing Li, Yu Zhang, Zhihua Wei, Yao Qian, Jinyu Li, and Furu Wei. 2022. SpeechT5: Unified-modal encoder-decoder pre-training for spoken language processing. In Proceedings of the 60th Annual Meeting of the Associationfor Computational Linguistics, pages 5723– 5738.

Gilles Boulianne. 2022. Phoneme transcription of endangered languages: an evaluation of recent ASR

architectures in the single speaker scenario. In Findings of the Association for Computational Linguistics: ACL 2022, pages 2301–2308.

Hui Bu, Jiayu Du, Xingyu Na, Bengu Wu, and Hao Zheng. 2017. Aishell-1: An open-source mandarin speech corpus and a speech recognition baseline. In 2017 20th Conference of the Oriental Chapter of the International Coordinating Committee on Speech Databases and Speech I/O Systems and Assessment (O-COCOSDA), pages 1–5.

William Chan, Navdeep Jaitly, Quoc Le, and Oriol Vinyals. 2016. Listen, attend and spell: A neural network for large vocabulary conversational speech recognition. In 2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 4960–4964.

Boli Chen, Guangwei Xu, Xiaobin Wang, Pengjun Xie, Meishan Zhang, and Fei Huang. 2022. Aishell-ner: Named entity recognition from chinese speech. In 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 8352– 8356.

Zhehuai Chen, Mahaveer Jain, Yongqiang Wang, Michael L Seltzer, and Christian Fuegen. 2019. Joint grapheme and phoneme embeddings for contextual end-to-end asr. In 2019 Conference of the International Speech Communication Association (Interspeech), pages 3490–3494.

Jianpeng Cheng and Mirella Lapata. 2016. Neural summarization by extracting sentences and words. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, pages 484– 494.

Yu-An Chung, Chenguang Zhu, and Michael Zeng. 2021. SPLAT: Speech-language joint pre-training for spoken language understanding. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1897–1907.

Xuandi Fu, Kanthashree Mysore Sathyendra, Ankur Gandhe, Jing Liu, Grant P. Strimel, Ross McGowan, and Athanasios Mouchtaris. 2023. Robust acoustic and semantic contextual biasing in neural transducers for speech recognition. In 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5.

Karthik Ganesan, Pakhi Bamdev, Jaivarsan B, Amresh Venugopal, and Abhinav Tushar. 2021. N-best ASR transformer: Enhancing SLU performance using multiple ASR hypotheses. In Proceedings of the 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, pages 93–98.

Zhuo Gong, Daisuke Saito, Sheng Li, Hisashi Kawai, and Nobuaki Minematsu. 2022. Can we train a language model inside an end-to-end ASR model? - investigating effective implicit language modeling.

In Proceedings of the Second Workshop on When Creative AI Meets Conversational AI, pages 42–47.

Jiatao Gu, Zhengdong Lu, Hang Li, and Victor O.K. Li. 2016. Incorporating copying mechanism in sequenceto-sequence learning. In Proceedings of the 54th Annual Meeting ofthe Associationfor Computational Linguistics, pages 1631–1640.

Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui Yu, Wei Han, Shibo Wang, Zhengdong Zhang, Yonghui Wu, et al. 2020. Conformer: Convolution-augmented transformer for speech recognition. arXiv preprint arXiv:2005.08100.

Minglun Han, Linhao Dong, Shiyu Zhou, and Bo Xu. 2021. Cif-based collaborative decoding for end-toend contextual speech recognition. In 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6528–6532.

Takaaki Hori, Shinji Watanabe, and John Hershey. 2017. Joint CTC/attention decoding for end-to-end speech recognition. In Proceedings ofthe 55th Annual Meeting ofthe Associationfor Computational Linguistics, pages 518–529.

Christian Huber, Juan Hussain, Sebastian Stüker, and Alexander Waibel. 2021. Instant one-shot wordlearning for context-specific neural sequence-tosequence speech recognition. In 2021 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pages 1–7. IEEE.

Mahaveer Jain, Gil Keren, Jay Mahadeokar, Geoffrey Zweig, Florian Metze, and Yatharth Saraf. 2020. Contextual rnn-t for open domain asr. arXiv preprint arXiv:2006.03411.

Sai Muralidhar Jayanthi, Devang Kulshreshtha, Saket Dingliwal, Srikanth Ronanki, and Sravan Bodapati. 2023. Retrieve and copy: Scaling asr personalization to large catalogs. arXiv preprint arXiv:2311.08402.

Suyoun Kim, Takaaki Hori, and Shinji Watanabe. 2017. Joint ctc-attention based end-to-end speech recognition using multi-task learning. In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 4835–4839.

Tian Lan, Deng Cai, Yan Wang, Heyan Huang, and Xian-Ling Mao. 2023. Copy is all you need. In The Eleventh International Conference on Learning Representations.

Haoran Miao, Gaofeng Cheng, Changfeng Gao, Pengyuan Zhang, and Yonghong Yan. 2020. Transformer-based online ctc/attention end-to-end speech recognition architecture. In 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6084–6088.

Josef R. Novak, Nobuaki Minematsu, and Keikichi Hirose. 2012. Dynamic grammars with lookahead composition for wfst-based speech recognition. In Interspeech.

Motoi Omachi, Yuya Fujita, Shinji Watanabe, and Matthew Wiesner. 2021. End-to-end ASR to jointly predict transcriptions and linguistic annotations. In Proceedings of the 2021 Conference of the North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies, pages 1861–1871.

Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. 2015. Librispeech: An asr corpus based on public domain audio books. In 2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 5206–5210.

Golan Pundak, Tara N Sainath, Rohit Prabhavalkar, Anjuli Kannan, and Ding Zhao. 2018. Deep context: end-to-end contextual speech recognition. In 2018 IEEE Spoken Language Technology Workshop (SLT), pages 418–425.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2022. Robust speech recognition via large-scale weak supervision. arXiv preprint arXiv:2212.04356.

Kanishka Rao, Ha¸sim Sak, and Rohit Prabhavalkar. 2017. Exploring architectures, data and units for streaming end-to-end speech recognition with rnntransducer. In 2017 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pages 193–199.

Anthony Rousseau, Paul Deléglise, and Yannick Estève. 2012. TED-LIUM: an automatic speech recognition dedicated corpus. In Proceedings ofthe Eighth International Conference on Language Resources and Evaluation (LREC’12), pages 125–129, Istanbul, Turkey. European Language Resources Association (ELRA).

Kanthashree Mysore Sathyendra, Thejaswi Muniyappa, Feng-Ju Chang, Jing Liu, Jinru Su, Grant P Strimel, Athanasios Mouchtaris, and Siegfried Kunzmann. 2022. Contextual adapters for personalized speech recognition in neural transducers. In 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 8537–8541. IEEE.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.

Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly. 2015. Pointer networks. In Advances in Neural Information Processing Systems, volume 28.

Tongtong Wu, Guitao Wang, Jinming Zhao, Zhaoran Liu, Guilin Qi, Yuan-Fang Li, and Gholamreza Haffari. 2022. Towards relation extraction from speech. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 10751–10762.

Song Xu, Haoran Li, Peng Yuan, Youzheng Wu, Xiaodong He, and Bowen Zhou. 2020. Self-attention guided copy mechanism for abstractive summarization. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 1355–1362.

Hemant Yadav, Sreyan Ghosh, Yi Yu, and Rajiv Ratn Shah. 2020. End-to-end named entity recognition from english speech. In Interspeech 2020, 21st Annual Conference ofthe International Speech Communication Association, Virtual Event, Shanghai, China, 25-29 October 2020, pages 4268–4272. ISCA.

Zhengyi Zhang and Pan Zhou. 2022. End-to-end contextual asr based on posterior distribution adaptation for hybrid ctc/attention system. arXiv preprint arXiv:2202.09003.

Ziqiang Zhang, Long Zhou, Junyi Ao, Shujie Liu, Lirong Dai, Jinyu Li, and Furu Wei. 2022. SpeechUT: Bridging speech and text with hiddenunit for encoder-decoder based speech-text pretraining. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pages 1663–1676.

Ding Zhao, Tara N Sainath, David Rybach, Pat Rondon, Deepti Bhatia, Bo Li, and Ruoming Pang. 2019a. Shallow-fusion end-to-end contextual biasing. In 2019 Conference of the International Speech Communication Association (Interspeech), pages 1418– 1422.

Wei Zhao, Liang Wang, Kewei Shen, Ruoyu Jia, and Jingming Liu. 2019b. Improving grammatical error correction via pre-training a copy-augmented architecture with unlabeled data. In Proceedings of the 2019 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, pages 156–165.

## Appendices

## A Datasets

Aishell (Bu et al., 2017) and ST-cmds<sup>6</sup> are two widely used Chinese Mandarin datasets. Aishell contains about 150 hours of speech. ST-cmds was built based on commonly used online chatting and user command speeches, which contains about 110 hours of speech. For the English dataset, we utilize the portion of data that has been manually annotated with entities by Yadav et al. (2020), which comprises approximately 150 hours. The Eng dataset is built by extracting content from wellknown English datasets, including Librispeech (Panayotov et al., 2015), CommonVoice<sup>7</sup>, Tedlium (Rousseau et al., 2012), and Voxforge<sup>8</sup>.

Table 6 shows the detailed statistics of the datasets used in our experiments. “Sent” means the number of instances. “NE” is the number of different named entities in the dataset and also the size of the contextual entity dictionary used during inference.

<table><tr><td rowspan="2">Dataset</td><td colspan="2">Train</td><td colspan="2">Dev</td><td colspan="2">Test</td></tr><tr><td>Sent</td><td>NE</td><td>Sent</td><td>NE</td><td>Sent</td><td>NE</td></tr><tr><td>Aishell</td><td>119919</td><td>14241</td><td>14326</td><td>2194</td><td>7176</td><td>1186</td></tr><tr><td>Aishell-NE</td><td>119919</td><td>14241</td><td>4949</td><td>2194</td><td>2244</td><td>1186</td></tr><tr><td>ST-cmds</td><td>82080</td><td>17376</td><td>10260</td><td>3029</td><td>10260</td><td>3124</td></tr><tr><td>ST-cmds-NE</td><td>82080</td><td>17376</td><td>3285</td><td>3029</td><td>3241</td><td>3124</td></tr><tr><td>Eng</td><td>64570</td><td>11858</td><td>3100</td><td>2568</td><td>3100</td><td>2508</td></tr><tr><td>Eng-NE</td><td>64570</td><td>11858</td><td>2677</td><td>2568</td><td>2690</td><td>2508</td></tr></table>

Table 6: Statistics of the used datasets.

## B Parameter Settings

We use 80-dimensional log-mel acoustic features with 25ms frame window and 10ms frame shift. The log-mel features are first fed into a 2D convolutional layer for downsampling and mapped to 256 dimensions before being inputted into the Audio Encoder. Both the audio encoder and decoder consist of 6 Transformer layers with 4 attention heads each. The NE Encoder is composed of three LSTM layers, with the input being a randomly initialized 256-dimensional embedding vector and the hidden size being 512. The relative weight λ in Equation 13 is set to 0.7. The experiments are conducted on two NVIDIA A100 GPUs.

In addition, for the experiments on Whisper, we use Whisper-small model<sup>9</sup>, which includes 12 transformer layers in both its encoder and decoder, and is pre-trained on a total of 680,000 hours of multi-lingual and multi-task data. We replace our audio encoder and decoder with the Whisper model and fine-tune the parameters of the entire model on our training set for a maximum of 20 epochs. During fine-tuning, the initial learning rate for the Whisper model’s parameters is set to 1e-5, while the learning rate for other parameters is set to 1e-3 with 10,000 warm-up steps. During inference, we used beam search with a beam size of 5 and 10 for models with and without Whisper, respectively.

![](images/68d1c834ddb6f781f0fa7cfd9c6423c3230f6f3b5e7e6928a470d54335d412e4.jpg)  
Figure 5: Effect of $\beta .$

## C Influence of Negative Entities in Training

During training, we construct an NE dictionary for each batch. To enhance the model’s ability of copying correct entities, we sample additional negative examples.

Suppose the dictionary already contains m entities, either real entities or pseudo sub-string entities. We sample $\beta \cdot m$ entities as negative examples from the training set. We utilize the parameter $\beta$ to control the number of negative examples. Thus, we get the final dictionary for this batch which contains a total of $( \beta + 1 )$ m entities. As shown in Figure 5, adding 1 or 2 times the number of negative samples can reduce transcription errors. Specifically, when $\beta = 2$ , the CER and NE-CER decreased by 0.42% and 0.44% compared to $\beta = 0$ . However, as $\beta$ continues to increase, the error rate started to rise. We think that this is due to the presence of excessive noise. This causes the model to excessively focus on the negative samples, thus affecting its ability to accurately copy entities. Therefore, we set $\beta$ to 2 during training.