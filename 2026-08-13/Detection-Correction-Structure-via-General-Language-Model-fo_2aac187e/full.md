# Detection-Correction Structure via General Language Model for Grammatical Error Correction

Wei Li, Houfeng Wang

National Key Laboratory for Multimedia Information Processing, Peking University weili22@stu.pku.edu.cn, wanghf@pku.edu.cn

## Abstract

Grammatical error correction (GEC) is a task dedicated to rectifying texts with minimal edits, which can be decoupled into two components: detection and correction. However, previous works have predominantly focused on direct correction, with no prior efforts to integrate both into a single model. Moreover, the exploration of the detection-correction paradigm by large language models (LLMs) remains underdeveloped. This paper introduces an integrated detection-correction structure, named DeCoGLM, based on the General Language Model (GLM). The detection phase employs a fault-tolerant detection template, while the correction phase leverages autoregressive mask infilling for localized error correction. Through the strategic organization of input tokens and modification of attention masks, we facilitate multi-task learning within a single model. Our model demonstrates competitive performance against the state-of-the-art models on English and Chinese GEC datasets. Further experiments present the effectiveness of the detectioncorrection structure in LLMs, suggesting a promising direction for GEC.

## 1 Introduction

Grammatical error correction (GEC) is a task focused on automatically rectifying grammatical errors in human-written text (Wang et al., 2021). GEC models are applied in language learning (Katinskaia and Yangarber, 2021; Caines et al., 2023; Kaneko et al., 2022), enhancing automatic speech recognition (Liao et al., 2023), and aiding in text data labeling (Sun et al., 2023). The two primary approaches in GEC are Sequence-to-Sequence (Seq2Seq) and Sequenceto-Edit (Seq2Edit). Without detection, Seq2Seq treats GEC as the direct generation for correct text, providing high flexibility (Junczys-Dowmunt et al., 2018; Ge et al., 2018). On the other hand, Seq2Edit views GEC as a sequence labeling task for edit labels, showcasing high precision by controlled edits (Awasthi et al., 2019; Stahlberg and Kumar, 2020; Omelianchuk et al., 2020). The advent of large language models (LLMs) has further expanded Seq2Seq model capabilities (Ouyang et al., 2022; Zeng et al., 2022). Despite their unprecedented performance in various tasks (Chang et al., 2024), LLMs underperform than low-parameter models in GEC due to the over-correction phenomenon (Qu and Wu, 2023; Coyne et al., 2023).

![](images/247c026472356697f6064ab44982944da9662d23cc97b7ac48049b724aa851b5.jpg)  
Figure 1: Detection and correction process of DeCoGLM. Detection and Correction are incorporated in one General Language Model (GLM).

While the detection-correction structure can harness the strengths of both Seq2Seq and Seq2Edit, most existing works merely utilize detection as additional input for Seq2Seq models (Yuan et al., 2021a; Li et al., 2022, 2023a). Moreover, all previous detection-correction systems comprise separate models (Chen et al., 2020). In contrast, we introduce a novel GEC model, named DeCoGLM, based on the General Language Model (GLM) (Du et al., 2022). This model employs an integrated detection-correction structure to detect errors and generate localized corrections. As depicted in Figure 1, the error detection phase employs a template rule to construct masked text based on detection results. During the correction phase, the model leverages the autoregressive mask infilling capability of the GLM to generate correct text pieces for erroneous parts, thereby saving inference time. To incorporate both detection and correction within a single model, we devise a multi-task learning approach, organizing input text with attention mask adjustments. Results on English and Chinese GEC benchmarks demonstrate that our proposed model surpasses previous detection-correction models and is comparable to state-of-the-art (SOTA) models. To further explore the potential of applying the detection-correction structure to LLMs, the detection and correction phases are separated, termed DeGLM and CoGLM respectively. Our proposed single system, comprising a small detection model and an LLM corrector, outperforms other Seq2Seq LLMs. In summary, our primary contributions are:

• A novel GEC model, DeCoGLM, which incorporates a detection-correction structure based on the GLM.

• The design of a multi-task training method that integrates detection and correction within a single model.

• The exploration of using LLMs for GEC, which involves deploying large error correction models with the support of small detection models.

## 2 Related Work

## 2.1 Sequence-to-Sequence GEC

Seq2Seq models (Lewis et al., 2019; Raffel et al., 2020) have demonstrated high performance in GEC (Junczys-Dowmunt et al., 2018; Choe et al., 2019; Zhao et al., 2019; Katsumata and Komachi, 2020). Techniques such as data synthesis (Stahlberg and Kumar, 2021; Grundkiewicz et al., 2019), training schedule (Lichtarge et al., 2020; Bout et al., 2023), and decode reranking methods (Kaneko et al., 2019; Zhang et al., 2023; Zhou et al., 2023) have been incorporated into previous Seq2Seq GEC models. SOTA model architectures typically supplement Seq2Seq models with additional information (Li et al., 2023a; Zhang et al., 2022b; Fang et al., 2023a). However, a significant drawback of Seq2Seq GEC models is the inference cost, as these models generate tokens sequentially and waste time copying source tokens (Sun et al., 2021).

As the latest Seq2Seq models, LLMs have emerged as a new paradigm for natural language processing (NLP) tasks following the introduction of GPT-3 and ChatGPT (Brown et al., 2020). Nevertheless, recent studies have shown that LLMs underperform current SOTA models on both English and Chinese GEC benchmarks (Coyne et al., 2023; Loem et al., 2023; Qu and Wu, 2023; Li et al., 2023b). Existing datasets and evaluation methods (Bryant et al., 2017) favor minimum edits as the rule for correction. However, GPT models often produce over-corrected sentences with unnecessary edits (Fang et al., 2023b; Coyne et al., 2023). In contrast to the Seq2Seq GEC methods that directly perform overall generation, our work only focuses on localized error correction, which not only saves inference time but also mitigates the over-correction phenomena in LLMs.

## 2.2 Sequence-to-Edit GEC

Seq2Edit methods generate edit operations for ungrammatical sentences (Stahlberg and Kumar, 2020). For instance, LaserTagger (Malmi et al., 2019) predicts token-level edit operations, which has been adopted in subsequent methods like PIE and GECToR (Awasthi et al., 2019; Omelianchuk et al., 2020). As a representative model, GECToR predicts four classes of edits and grammatical transformations, achieving high-precision results. Lai et al. (2022) further enhances it by addressing its deficiencies in multi-round correction. However, Seq2Edit methods necessitate intricate designs for edits, which are not language-agnostic. In contrast, our proposed model retains a limited set of language-agnostic edit operations and can flexibly conduct edits by autoregressive generation.

## 2.3 Detection-Correction GEC

The GEC task can be divided into two processes: detection and correction (Rei and Yannakoudakis, 2016; Bell et al., 2019). Prior research incorporates detection results as supplementary information for Seq2Seq correction models (Kaneko et al., 2020; Yuan et al., 2021b; Li et al., 2023a). The methods proposed by Mallinson et al. (2020) and Yakovlev et al. (2023) employ the Masked Language Model (MLM) (Devlin et al., 2018) to obtain corrections, which are constrained by mask number. Chen et al. (2020) introduces error span detection and correction to address the GEC problem, which allows for flexible corrections while maximizing time efficiency. Building on this, we further integrate the detection and correction tasks into a single GLM model, enabling mutual benefits between the two tasks, which is not achieved by previous works.

![](images/a2d6ebf3ccdb6403536428e6326e81ff76ce0008ab872264ebba33f06b58683b.jpg)  
Figure 2: The proposed detection-correction structure based on GLM. The example shown above has a source text $\pmb { x } _ { s } = x _ { s } ^ { 1 } x _ { s } ^ { 2 } x _ { s } ^ { 3 } x _ { s } ^ { 4 } x _ { s } ^ { 5 } x _ { s } ^ { 6 }$ and the target text is ${ \pmb y } = x _ { s } ^ { 1 } c _ { 1 } ^ { 1 } x _ { s } ^ { 4 } x _ { s } ^ { 5 } c _ { 2 } ^ { 1 } c _ { 2 } ^ { 2 } x _ { s } ^ { 6 }$ . Consistent with GLM, the position IDs and block position IDs are utilized for marking the original positions of text pieces and the inner order of tokens.

## 3 Methods

Our proposed model leverages the design of the GLM. Given a sentence with MASK tokens, GLM utilizes autoregressive blank infilling (Du et al., 2022) to generate a corresponding segment for each MASK position. These segments are termed as text pieces. This section describes how GLM is utilized to integrate detection and correction into a single model, as depicted in Figure 2. Additionally, the design of multi-task training is also outlined here.

## 3.1 Error Detection

Drawing from the four edit classes by Omelianchuk et al. (2020), we utilize token-level detection labels that do not include any specific word or grammar. Given that the mask-infilling process can generate empty text pieces, the REPLACE and DELETE operations are consolidated into the ERROR label. Consequently, the detection labels comprise KEEP (K), ERROR (E), and INSERT (I). Given the tokens of source text as:

$$
\pmb { x } _ { s } = x _ { s } ^ { 1 } x _ { s } ^ { 2 } \dots x _ { s } ^ { n }\tag{1}
$$

, the objective of error detection is to predict detection labels derived by the alignment between the source text and the target text (correct text):

$$
{ \pmb d } = d _ { 1 } d _ { 2 } \dots d _ { n } , d _ { i } \in { \cal L } = \{ K , E , I \}\tag{2}
$$

Detection Model The proposed model begins by extracting the representations of the source text tokens by GLM as Equation 3. The final detection label predictions are generated through a detection head, implemented by a feed-forward network FN and softmax function, as shown in Equation 4:

$$
\pmb { h } _ { s } = h _ { s } ^ { 1 } h _ { s } ^ { 2 } \dots h _ { s } ^ { n } = \mathrm { G L M } \left( \pmb { x } _ { s } \right)\tag{3}
$$

$$
p \left( \hat { d } _ { i } = l | \pmb { x } _ { s } \right) = \mathrm { S o f t m a x } ( \mathrm { F N } \left( h _ { s } ^ { i } \right) ) , l \in L\tag{4}
$$

Fault-tolerant Template The source text $\mathbf { \delta } _ { \mathbf { \mathcal { X } } _ { \mathcal { S } } }$ is transformed into masked text ${ \pmb x } _ { m }$ based on the detection labels using the following template rules. Each continuous interval containing only ERROR labels is replaced with a MASK token. For each position of INSERT, a MASK token is inserted. The form of masked text is shown in Equation 5:

$$
{ \pmb x } _ { m } = { \pmb x } _ { s _ { 1 } } m _ { 1 } { \pmb x } _ { s _ { 2 } } m _ { 2 } \dotsm m _ { k } { \pmb x } _ { s _ { k + 1 } } ,\tag{5}
$$

where $m _ { i }$ is the i-th MASK token introduced in $\mathbf { \delta } _ { \mathbf { \mathcal { X } } _ { s } }$ , and $\mathbf { \boldsymbol { x } } _ { s _ { i } }$ denotes the i-th correct subinterval of source text. If all the labels are KEEPs, the source text is directly output as the corrected result. Despite potential inaccuracies in detections, our model can tolerate a certain degree of false positives. In the instance where the correct token is identified as ERROR or INSERT, the corrector can mitigate such errors by either restoring the original text piece or generating an empty text piece.

Aggressive Detection Utilizing the fault-tolerant template enables more aggressive detection, emphasizing the recall of ERROR and INSERT. Focal Loss (Lin et al., 2020) is used as the loss function to tackle the issue of imbalanced classification because the majority of tokens correspond to KEEP labels. The training objective for error detection is given by Equation 6:

$$
\ell _ { D } = - \pmb { \alpha } _ { D } \left( 1 - p _ { \theta } \left( \pmb { d } | \pmb { x } _ { s } \right) \right) ^ { \gamma } \log \left( p _ { \theta } \left( \pmb { d } | \pmb { x } _ { s } \right) \right)\tag{6}
$$

where θ represents the model parameters and $\gamma$ is a hyper-parameter set to 2. α<sub>D</sub> denotes the corresponding weight factors for detection labels. To strengthen aggressive error detection, $\alpha _ { K }$ for the

KEEP category is set to 1, while $\alpha _ { E I } = 2$ is set for the ERROR and INSERT categories.

## 3.2 Localized Error Correction

In the training data, detection labels are derived from the alignment of sequences between the source text $\mathbf { \delta } _ { \mathbf { \mathcal { X } } _ { \mathcal { S } } }$ and the target text $\mathbf { \pmb { y } } .$ The corresponding masked text ${ \pmb x } _ { m }$ can be formulated in Equation 5 with $\mathbf { \boldsymbol { x } } _ { s _ { i } }$ representing the i-th aligned segment. For each unaligned position replaced with $m _ { i }$ , the correct text piece is denoted as $\mathbf { c } _ { i }$ . Consequently, the target text can be represented as:

$$
\pmb { y } = \pmb { x } _ { s _ { 1 } } \pmb { c } _ { 1 } \pmb { x } _ { s _ { 2 } } \pmb { c } _ { 2 } \dots \pmb { c } _ { k } \pmb { x } _ { s _ { k + 1 } }\tag{7}
$$

Leveraging the GLM pretrained by autoregressive blank infilling task, we fine-tune the GLMs for localized error correction. The probability distribution prediction for the $j \cdot$ -th token in the i-th text piece $c _ { i }$ is given in Equation 8:

$$
\begin{array} { r l r } & { } & { p \left( c _ { i , j } = w | \pmb { x } _ { s } , \pmb { x } _ { m } , \pmb { c } _ { < i } , \pmb { c } _ { i } ^ { < j } \right) = } \\ & { } & { \mathrm { G L M H } \left( \pmb { x } _ { s } , \pmb { x } _ { m } , \pmb { c } _ { < i } , \pmb { c } _ { i } ^ { < j } \right) , w \in V } \end{array}\tag{8}
$$

where GLMH denotes the GLM model with its original token prediction head, w is any token in the vocabulary, and $\boldsymbol { c } _ { i } ^ { < j }$ refers to all tokens with index $< j$ in text piece $\mathbf { c } _ { i }$

## 3.3 Multi-Task Organization

Multi-task Learning. The cross-entropy loss function, shown in Equation 9, is used as the training objective for error correction task:

$$
\ell _ { C } = - \sum _ { i , j } \log \left( p _ { \theta } \left( c _ { i , j } | \pmb { x } _ { s } , \pmb { x } _ { m } , \pmb { c } _ { < i } , \pmb { c } _ { i } ^ { < j } \right) \right)\tag{9}
$$

For multi-task learning, we utilize a weighted loss function to enable the model to concurrently acquire error detection and correction capabilities. The training objective for this DeCoGLM model is to minimize the loss function given by:

$$
\ell = \bar { \ell _ { C } } + w _ { D } \bar { \ell _ { D } }\tag{10}
$$

where $\bar { \ell _ { C } }$ and $\bar { \ell _ { D } }$ are the token-level averages of $\ell _ { C }$ and $\ell _ { D }$ respectively. The detection loss weight $w _ { D }$ is set to 10 to balance the scales of the two losses. For the impact of the loss weights on the model’s performance, please refer to Section 5.2.

Attention Mask To unify the two tasks into a single model, source text $\mathbf { \delta } _ { \mathbf { x } _ { s } , \mathbf { \delta } _ { \mathbf { x } _ { \mathbf { \delta } } } }$ , masked text ${ \pmb x } _ { m }$ , and text pieces c are concurrently fed into the GLM model. The prediction of detection labels is conditioned on $\mathbf { \delta } _ { \mathbf { x } , \mathbf { \delta } }$ while the autoregressive text prediction relies on $\mathbf { \boldsymbol { x } } _ { s } , \mathbf { \boldsymbol { x } } _ { m }$ , and all previously generated text pieces. Therefore, the attention from ${ \pmb x } _ { m }$ to $\mathbf { \delta } _ { \mathbf { \mathcal { X } } _ { \mathcal { S } } }$ is eliminated to prevent detection from using ${ \pmb x } _ { m }$ and $^ { c , }$ with other part adhering to the original GLM attention mask. This is depicted in Figure 3.

![](images/4d3bec5d04b37ea3114d9341894f35aff79e1da1f89e20db5aac9f92ea6968fe.jpg)  
Figure 3: Attention Mask Example. The source text is $\pmb { x } _ { s } = x _ { s } ^ { 1 } x _ { s } ^ { 2 } x _ { s } ^ { 3 } x _ { s } ^ { 4 }$ , and the target text is $\pmb { y } = c _ { 1 } ^ { 1 } x _ { s } ^ { 3 } x _ { s } ^ { 4 } c _ { 2 } ^ { 1 } c _ { 2 } ^ { 2 }$ If the cell at row i and column $j$ is colored, it indicates that the i th token can pay attention to the $j$ th token. The region enclosed by dashed lines indicates the attention removed compared to the original GLM.

Two Stage Supervised Fine-tuning Given that error detection is not infallible, the input during the correction phase may contain inaccuracies, with a distribution deviation from the training samples constructed with right detection labels. This issue is also observed in other detection-correction works (Chen et al., 2020; Li et al., 2023a). To address this, we add a second supervised fine-tuning stage (SFT2), which employs a detection-enhanced approach: initially, all detection results on the training set are obtained using the model trained with data constructed by perfect detection (SFT1). Then, new training data is generated by augmenting the original labels with the fault detection results, leading to a secondary training of the SFT1 model. Examples of the two-stage training samples are provided in Table 7 in the appendix.

## 3.4 Separate Models

The detection and correction phases can be implemented using two separate GLMs, named DeGLM and CoGLM respectively in this paper. Their training objectives are defined by Equation 6 and 9, respectively. This decomposition facilitates the customization of distinct models for the detection and correction phases. However, in scenarios with limited computational resources, the Parameter-Efficient Fine-Tuning (PEFT) (Fu et al., 2023) is ineffective for DeCoGLM due to the significant disparities between the sequence labeling task of the error detection module and the mask-infilling pretraining task of GLM. To apply our approach to LLMs, we propose training a large version of CoGLM using the detection-enhanced method, similar to the second stage fine-tuning discussed in Section 3.3.

## 3.5 Detection Control

During inference, the model needs to predict detection labels for the source text first, then transform it into masked text. Subsequently, both of them are input together for generating text pieces. This decoupling allows us to regulate the correction process using the probabilities of the three detection labels, thereby harnessing the model’s potential to enhance benchmark performance. Three control modes are designed:

KEEP Threshold (δ): Any prediction with KEEP probability exceeding δ is directly set to KEEP. ERROR Lower Bound $( \phi _ { e } ) ;$ : Any ERROR probability prediction falling below $\phi _ { e }$ is directly set to 0, thereby precluding the prediction of ERROR when $p _ { e } < \phi _ { e }$

INSERT Lower Bound $( \phi _ { i } ) $ : Any INSERT probability prediction below $\phi _ { i }$ is directly set to 0, precluding the prediction of INSERT when $p _ { i } < \phi _ { i }$

The three inference hyper-parameters can be determined using a greedy grid search based on the metrics on the validation set. We discuss them in Section 5.5.

## 4 Experiments

## 4.1 Datasets and Evaluation

For the English GEC task, we evaluate the performance on the CoNLL-14 test set (Ng et al., 2014) using the $M ^ { 2 }$ Scorer (Dahlmeier and Ng, 2012), and on the BEA-19 test set (Bryant et al., 2019) using the ERRANT scorer (Bryant et al., 2017). The model is pretrained on synthetic dataset C4-200M (Stahlberg and Kumar, 2021) and fine-tuned on the cleaned Lang8 dataset (CLang8) (Rothe et al., 2021). For the large version of CoGLM model, we utilize smaller datasets including FCE (Yannakoudakis et al., 2011), NUCLE (Dahlmeier et al.,

2013), and W&I+LOCNESS (Bryant et al., 2019) for fine-tuning, following Zhou et al. (2023). The BEA-19 dev set is used for model selection.

For the Chinese GEC task, we synthesize pretraining data from the People’s Daily corpus<sup>1</sup> using rule-based insertion, replacement, and deletion. The models are fine-tuned on the Chinese Lang8 dataset (Zhao et al., 2018) and the HSK dataset, following Zhang et al. (2022a), and on the FCGEC training set, respectively. The models are evaluated on MuCGEC and FCGEC test sets using ChER-RANT (Zhang et al., 2022a; Xu et al., 2022). Further details are provided in Appendix A.

## 4.2 Model Settings

Proposed Models The open-source GLMs are utilized as the backbones for both DeCoGLM and separate models. The detection head comprises a feed-forward network with a single hidden layer, the dimension of which matches that of the GLM hidden state. The English base model employs glm-roberta-large, while glm-large-chinese is used as the Chinese base model. The large CoGLM models for error correction, denoted as CoGLM (10B), uses glm-10b and glm-10b-chinese as backbones. Due to the restriction of computational resources, large models are fine-tuned on the relatively small fine-tuning dataset mentioned in Section 4.1 by LoRA (Hu et al., 2021), without datasets for pretraining. Refer to Appendix B.2 for detailed configurations.

Comparison with Previous Works In the main experiment, we present the results of single systems trained on parallel data without any reranker. GECToR (Omelianchuk et al., 2020) represents the Seq2Edit models, while BART and T5 (Lewis et al., 2019; Raffel et al., 2020) are SOTA backbones of Seq2Seq GEC methods. SynGEC (Zhang et al., 2022b) incorporates syntactic information into the BART model. The performance of GEC-ToR and BART model on the Chinese dataset is the reproduced result under our data configuration, and the results for BART on the English dataset are reported by Zhang et al. (2022b). We also present the results of four models involving the detection-correction process. SpanDC (Chen et al., 2020) comprises a span detector and a generator. Multi-Encoder (Yuan et al., 2021a) encodes error categories as auxiliary information. GEC-DePend (Yakovlev et al., 2023) integrates error detection with correction by the MLM. TemplateGEC (Li et al., 2023a) uses the output of the GECToR model as supplementary information for Seq2Seq models. Comparison with LLMs For the LLMs treating GEC as a Seq2Seq task, we fine-tune ChatGLM2, ChatGLM3 (Du et al., 2022), and Llama2 (Touvron et al., 2023) with LoRA. As Llama2 is not optimized for Chinese, the results on the Chinese dataset are obtained using the Baichuan (Yang et al., 2023) models.

<table><tr><td colspan="2"></td><td colspan="4">English</td><td colspan="3"></td><td colspan="6">Chinese</td></tr><tr><td></td><td></td><td colspan="3">CoNLL-14 test</td><td colspan="3">BEA-19 test</td><td colspan="3">MuCGEC test</td><td colspan="3">FCGEC test</td></tr><tr><td>Single System</td><td>Parameters</td><td>P</td><td>R</td><td> $\mathbf { F _ { 0 . 5 } }$ </td><td>P</td><td>R</td><td> $\mathbf { F _ { 0 . 5 } }$ </td><td>P</td><td>R</td><td> $\mathbf { F _ { 0 . 5 } }$ </td><td>P</td><td>R</td><td> $\mathbf { F _ { 0 . 5 } }$ </td></tr><tr><td></td><td colspan="3"></td><td colspan="3">Primary Results</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GECToR</td><td>110M</td><td>77.5</td><td>40.1</td><td>65.3</td><td>79.2</td><td>53.9</td><td>72.4</td><td>46.72</td><td>27.14</td><td>40.83</td><td>46.11</td><td>34.35</td><td>43.16</td></tr><tr><td>BART</td><td>400M</td><td>73.6</td><td>48.6</td><td>66.7</td><td>74.0</td><td>64.9</td><td>72.0</td><td>41.90</td><td>29.48</td><td>38.64</td><td>38.38</td><td>37.62</td><td>38.23</td></tr><tr><td>T5</td><td>770M</td><td></td><td></td><td>66.1</td><td></td><td></td><td>72.1</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SynGEC</td><td>110M+400M</td><td>74.7</td><td>49.0</td><td>67.6</td><td>75.1</td><td>65.5</td><td>72.9</td><td>54.69</td><td>29.10</td><td>46.51</td><td></td><td></td><td></td></tr><tr><td>SpanDC</td><td>125M+209M</td><td>72.6</td><td>37.2</td><td>61.0</td><td>70.4</td><td>55.9</td><td>66.9</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Multi-Encoder</td><td>110M+107M</td><td>71.3</td><td>44.3</td><td>63.5</td><td>73.3</td><td>61.5</td><td>70.6</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GEC-DePenD</td><td>253M</td><td>73.2</td><td>37.8</td><td>61.6</td><td>72.9</td><td>53.2</td><td>67.9</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TemplateGEC</td><td>125M+770M</td><td>74.8</td><td>50.0</td><td>68.1</td><td>76.8</td><td>64.8</td><td>74.1</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DeGLM-CoGLM</td><td>335M+335M</td><td>75.1</td><td>49.0</td><td>67.8</td><td>76.4</td><td>63.4</td><td>73.4</td><td>47.22</td><td>30.08</td><td>42.39</td><td>52.95</td><td>39.20</td><td>49.48</td></tr><tr><td>DeCoGLM</td><td>335M</td><td>75.1</td><td>49.4</td><td>68.0</td><td>77.4</td><td>64.6</td><td>74.4</td><td>45.01</td><td>31.77</td><td>41.55</td><td>55.75</td><td>37.91</td><td>50.96</td></tr><tr><td colspan="14">Resource-restricted LLMs</td></tr><tr><td>ChatGLM2 ChatGLM3</td><td>6B 6B</td><td>61.72</td><td>45.58 47.50</td><td>57.64</td><td>56.89</td><td>58.73</td><td>57.25</td><td>31.35</td><td>21.39</td><td>28.68</td><td>44.30</td><td>17.08</td><td>33.59</td></tr><tr><td>LLaMA2/Baichuan</td><td></td><td>60.63</td><td></td><td>57.46</td><td>59.48</td><td>60.37</td><td>59.65</td><td>30.62</td><td>21.60</td><td>28.26</td><td>41.06</td><td>19.93</td><td>33.88</td></tr><tr><td></td><td>7B</td><td>67.24</td><td>51.84</td><td>63.47</td><td>66.16</td><td>66.12</td><td>66.15</td><td>36.47</td><td>25.18</td><td>33.47</td><td>51.83</td><td>24.08</td><td>42.12</td></tr><tr><td>LLaMA2/Baichuan</td><td>13B</td><td>68.43</td><td>55.30</td><td>65.33</td><td>69.46</td><td>69.28</td><td>69.42</td><td>37.91</td><td>26.90</td><td>35.04</td><td>56.65</td><td>27.11</td><td>46.52</td></tr><tr><td>DeGLM-CoGLM</td><td>335M+10B</td><td>70.58</td><td>52.65</td><td>66.08</td><td>72.80</td><td>67.57</td><td>71.69</td><td>47.48</td><td>29.92</td><td>42.49</td><td>56.09</td><td>38.02</td><td>51.22</td></tr><tr><td colspan="14">GPT-4 Zeroshot</td></tr><tr><td>ZeroShot +DeGLM</td><td>= =</td><td>59.64</td><td>58.32</td><td>59.37</td><td>55.69</td><td>70.44</td><td>58.13</td><td>36.36</td><td>27.71</td><td>34.22</td><td>18.83</td><td>4.08</td><td>10.93</td></tr><tr><td></td><td></td><td>66.40</td><td>54.81</td><td>63.70</td><td>64.92</td><td>69.42</td><td>65.78</td><td>32.68</td><td>30.90</td><td>32.31</td><td>25.60</td><td>16.98</td><td>23.24</td></tr></table>

Table 1: Results on English and Chinese GEC benchmarks. The parameter counts of the backbones of each system are shown in the second column. Under restricted resource, LLMs are fine-tuned using smaller datasets by LoRA. The highest metric is indicated in bold, while the second highest metric value is underlined.

GPT-4 We report the zero-shot performance of GPT-4 on four datasets with prompting. We attempt to incorporate detection results in the form of masked text into the prompt of GPT-4, aiming to enhance the performance on GEC tasks.

## 4.3 Main Results

Table 1 presents the main results. According to the last two rows of primary results, the integrated detection-correction model outperforms the separate models in most cases in terms of the $F _ { 0 . 5 }$ metric, despite having only half the parameter count. This suggests that the designed multi-task learning mutually reinforces detection and correction, which will be further discussed in Section 5.1. DeCoGLM achieves the highest or secondhighest $F _ { 0 . 5 }$ performance on three datasets, demonstrating comparable performance to SOTA GEC models. Considering the model parameter counts, our model outperforms all previous works with the detection-correction process, indicating that the well-designed detection-correction structure can achieve the SOTA level in GEC, a field typically dominated by Seq2Seq models. Furthermore, the inference speed of the localized error correction is significantly faster than the globalized error correction of the Seq2Seq method, with the details provided in Appendix C. These results also underscore the potential of GLM in the GEC field.

Despite limitations of data quantity and finetuning methods, fine-tuning LLMs with over 10B parameters yields results approaching SOTA level, suggesting that LLMs can reduce the need for extensive supervised data for fine-tuning. The strategy of small detection models assisting large models in localized correction yields improved performance across all datasets, primarily due to higher precision. This suggests that the model reduces over-correction at the expense of a certain level of recall. On the English dataset, GPT-4 exhibits a similar trend when incorporated with detection results, indicating that detection results can stably improve the GEC capability of LLMs, thus presenting a promising future direction for GEC.

## 5 Analysis

## 5.1 Interaction of Detection and Correction

In Section 4.3, we mentioned that detection and correction tasks can mutually benefit each other.

<table><tr><td colspan="2"></td><td colspan="3">BEA-19 dev</td><td colspan="4">MuCGEC dev</td><td colspan="4">FCGEC dev</td></tr><tr><td>Model</td><td>Parameters</td><td>AccD</td><td>RecE</td><td>RecI</td><td>Accc</td><td>AccD</td><td>RecE</td><td>RecI Accc</td><td>AccD</td><td>RecE</td><td>RecI</td><td>Accc</td></tr><tr><td>DeCoGLM</td><td>335M</td><td>94.56</td><td>65.60</td><td>63.95</td><td>90.54</td><td>84.44</td><td>52.72 25.74</td><td>74.24</td><td>96.74</td><td>54.57</td><td>51.13</td><td>84.44</td></tr><tr><td>DeGLM</td><td>335M</td><td>94.49</td><td>64.88</td><td>62.82</td><td></td><td>84.20</td><td>52.54 23.93</td><td></td><td>96.96</td><td>54.85</td><td>47.87</td><td></td></tr><tr><td>CoGLM</td><td>335M</td><td></td><td></td><td></td><td>90.27</td><td></td><td></td><td>74.55</td><td></td><td></td><td></td><td>83.94</td></tr></table>

Table 2: The metrics of detection and correction tasks on the development set. The results are presented using four metrics (3 detection metrics and 1 correction metric): overall accuracy in the detection phase (Acc ), recall for the detection label ERROR (Rec ), recall for the detection label INSERT (Rec ), and the accuracy of next token prediction during the localized error correction.

<table><tr><td></td><td></td><td colspan="3"> $\mathbf { F _ { 0 . 5 } }$  on dev set</td></tr><tr><td>wD 20</td><td> $\underline { { \alpha } } E I$  2</td><td>BEA-19</td><td>MuCGEC</td><td>FCGEC</td></tr><tr><td></td><td></td><td>60.30 60.09</td><td>34.45</td><td>40.57 41.52</td></tr><tr><td rowspan="4">10</td><td>- 1</td><td></td><td>35.17</td><td>42.89</td></tr><tr><td>2</td><td>59.93</td><td>34.25</td><td></td></tr><tr><td></td><td>60.81</td><td>35.09</td><td>42.49</td></tr><tr><td>3</td><td>60.29</td><td>35.82</td><td>40.72</td></tr><tr><td></td><td>4</td><td>60.12</td><td>35.03</td><td>41.49</td></tr><tr><td>5</td><td>2</td><td>60.60</td><td>34.53</td><td>42.10</td></tr><tr><td>1</td><td>2</td><td>59.64</td><td>33.23</td><td>36.72</td></tr></table>

Table 3: The preliminary experimental results of different loss weights. $w _ { D }$ and $\alpha _ { E I }$ is defined in Section 3.3 and 3.1. The $" \underline { { \mathbf { \Pi } } } _ { \astrosun } \ "$ value of $\alpha _ { E I }$ represents the usage of cross-entropy other than Focal Loss.

To further verify this, we conduct experiments using the integrated model (DeCoGLM, 335M) and two separate models for detection and correction (DeGLM, 335M; CoGLM, 335M), as shown in the primary results in Table 1. Without employing two-stage fine-tuning involving data enhancement, the models are trained on the same dataset, and their performance on the development set for detection and correction metrics is presented in Table 2. The integrated model exhibits superior detection and correction capabilities over separate models. A fairer comparison should involve two separate models with a parameter count of $3 3 5 \mathrm { M } / 2 = 1 6 7 . 5 \mathrm { M }$ but currently, there is no GLM backbone of approximately this size. In this scenario with fewer parameters, the advantage of the integrated model is expected to be even greater.

## 5.2 Weights of Multi-Task Training

To establish two weights that significantly impact the training objective: the detection loss weight $w _ { D }$ in Equation 10, and the ERROR and INSERT loss weight $\alpha _ { E I }$ in Equation 6, we conduct preliminary experiments, which include only the two stages of fine-tuning. The obtained results are presented in Table 3. Based on a preliminary observation on the loss scale, we initially set $w _ { D } = 1 0$ and explore experimental results under varying $\alpha _ { E I }$

<table><tr><td colspan="4"></td><td colspan="3">CoNLL-14 test</td><td colspan="3">BEA-19 test</td></tr><tr><td>K</td><td>E</td><td>I</td><td>D</td><td>P</td><td>R</td><td> $\underline { { \mathbf { F _ { 0 . 5 } } } }$ </td><td>P</td><td>R</td><td> $\underline { { \mathbf { F _ { 0 . 5 } } } }$ </td></tr><tr><td>√</td><td>√</td><td></td><td></td><td>69.67</td><td>50.91</td><td>64.89</td><td>72.18</td><td>65.14</td><td>70.65</td></tr><tr><td>√</td><td>√</td><td>√</td><td></td><td>69.25</td><td>51.26</td><td>64.71</td><td>72.33</td><td>65.46</td><td>70.85</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>68.48</td><td>49.95</td><td>63.75</td><td>71.23</td><td>64.48</td><td>69.77</td></tr></table>

Table 4: Results under different detection label sets. K=KEEP, E=ERROR, I=INSERT and D=DELETE.

The outcomes suggest that the Focal Loss along with moderately increasing $\alpha _ { E I }$ to achieve aggressive detection introduced in Section 3.3 is effective. After setting $\alpha _ { E I } = 2 $ , we conducted additional experiments with different $w _ { D }$ . The overall experimental results indicate that $\alpha _ { E I } = 2$ and $w _ { D } = 1 0$ constitute a suitable setup.

## 5.3 Detection Label Set

In the design outlined in Section 3.1, ERROR includes both replacement and deletion, as the deletion can be considered as replacing with zerolength text. The results for this design are shown in the second row of Table 4. INSERT can also be further merged into the ERROR label. This can be achieved by considering the INSERT operation as replacing the token $x _ { i }$ at the insertion position with $x _ { i } c _ { j }$ , where $c _ { j }$ represents tokens to be inserted. The results corresponding to this approach are shown in the first row of Table 4. Additionally, we demonstrate the results of applying four detection labels (KEEP, ERROR, INSERT, DELETE) in the last row. Overall, our designed three-label scheme performs relatively better, as the insertion operation in the two-label mode requires disrupting the correct part of the source text, and encountering DELETE in the four-label mode will lead to direct deletion, which makes the model unable to recover from faults in the error correction phase.

## 5.4 Ablation Study

To explore the effectiveness of various components in the designed detection-correction model, we conduct an ablation study focusing on synthetic data, backbone, two-stage fine-tuning, and detection control. The results are shown in Table 5.

<table><tr><td rowspan="2">BackBone</td><td rowspan="2">Pretrained</td><td colspan="3"></td><td colspan="3">CoNLL-14 test</td><td colspan="3">BEA-19 test</td></tr><tr><td>SFT1</td><td>SFT2</td><td>Ctrl</td><td>P</td><td>R</td><td> $\mathbf { F _ { 0 . 5 } }$ </td><td>P</td><td>R</td><td> $\mathbf { F _ { 0 . 5 } }$ </td></tr><tr><td>GLM-Roberta</td><td>Yes</td><td>√</td><td>√</td><td>√</td><td>75.07</td><td>49.40</td><td>68.00</td><td>77.36</td><td>64.63</td><td>74.43</td></tr><tr><td>GLM-Roberta</td><td>Yes</td><td>√</td><td>√</td><td>×</td><td>70.47</td><td>54.96</td><td>66.70</td><td>72.75</td><td>69.28</td><td>72.03</td></tr><tr><td>GLM-Roberta</td><td>Yes</td><td>√</td><td>X</td><td>√</td><td>75.27</td><td>48.24</td><td>67.69</td><td>76.55</td><td>62.34</td><td>73.21</td></tr><tr><td>GLM-Roberta</td><td>Yes</td><td>√</td><td>X</td><td>×</td><td>68.38</td><td>57.35</td><td>65.84</td><td>69.00</td><td>71.02</td><td>69.39</td></tr><tr><td>GLM-Roberta</td><td>Yes</td><td>X</td><td>X</td><td>X</td><td>54.04</td><td>45.99</td><td>52.21</td><td>45.12</td><td>58.60</td><td>47.30</td></tr><tr><td>GLM-Roberta</td><td>No</td><td>√</td><td>√</td><td>√</td><td>72.78</td><td>46.42</td><td>65.36</td><td>75.54</td><td>59.87</td><td>71.78</td></tr><tr><td>GLM-Roberta</td><td>No</td><td>√</td><td>V</td><td>X</td><td>69.25</td><td>51.26</td><td>64.71</td><td>72.33</td><td>65.46</td><td>70.85</td></tr><tr><td>GLM-Roberta</td><td>No</td><td>√</td><td>X</td><td>√</td><td>68.25</td><td>49.33</td><td>63.39</td><td>69.66</td><td>61.94</td><td>67.97</td></tr><tr><td>GLM-Roberta</td><td>No</td><td>√</td><td>X</td><td>X</td><td>63.92</td><td>52.46</td><td>61.25</td><td>66.27</td><td>66.01</td><td>66.21</td></tr><tr><td>BART-large</td><td>No</td><td>√</td><td>√</td><td>√</td><td>69.53</td><td>45.62</td><td>62.93</td><td>72.01</td><td>57.84</td><td>68.64</td></tr><tr><td>BART-large</td><td>No</td><td>√</td><td>√</td><td>X</td><td>66.39</td><td>49.80</td><td>62.24</td><td>69.25</td><td>63.28</td><td>67.97</td></tr><tr><td>BART-large</td><td>No</td><td>√</td><td>X</td><td>√</td><td>67.54</td><td>43.66</td><td>60.88</td><td>68.08</td><td>55.14</td><td>65.03</td></tr><tr><td>BART-large</td><td>No</td><td>√</td><td>X</td><td>X</td><td>62.75</td><td>50.40</td><td>59.81</td><td>64.67</td><td>63.62</td><td>64.46</td></tr></table>

Table 5: Ablation study results. The "Ctrl" denotes the proposed detection control.

Effectiveness of synthetic data In the proposed model, both the English and Chinese models undergo pretraining with a large-scale synthetic dataset of GEC. A comparison between the top and middle rows of Table 5 reveals that pretraining indeed provides a stable improvement in model performance, although the data used is not from real scenarios.

Effectiveness of GLM backbone The detectioncorrection structure can also be implemented in Seq2Seq models. We applied the proposed method to the BART model and conducted experiments. An additional detection head is integrated into the BART encoder, while the decoder generates text pieces for localized error correction. The experimental results, depicted in the bottom rows of Table 5, consistently demonstrate superior performance when employing GLM as the backbone compared to using BART. This can be attributed, in part, to the consistency between the original pretraining task of GLM and the training objective of the correction task, as defined in Equation 9. However, the pretraining pattern of BART differs. Additionally, the separation of BART’s encoder and decoder into two distinct modules may not effectively foster the mutual enhancement of detection and correction abilities in multi-task learning.

Effectiveness of Two Stage Fine-tuning As described in Section 3.3, two fine-tuning stages differ in the training data: SFT1 constructs training samples using only ground-truth detection labels, while SFT2 utilizes both ground-truth detection labels and the detection results from the model trained in the first stage. As evident from the comparison in Table 5, SFT1 significantly improves the model’s performance than the model pretrained on the synthetic dataset. Comparing the results exclusively differing in SFT2 in Table 5, it is observed that SFT2 consistently enhances $F _ { 0 . 5 } .$ , primarily attributed to the improvement in precision while maintaining recall relatively constant. This validates the effectiveness of the two-stage supervised fine-tuning design.

Detection Control From Table 5, it is evident that, under the scenario of employing the same trained model, setting three hyper-parameters for the detection phase also enhances the $F _ { 0 . 5 }$ performance. This approach primarily aims at improving precision. However, upon closer inspection, it is noticeable that this technique results in a more substantial reduction in recall compared to the secondstage fine-tuning. For all GLM models incorporating detection control, the recall on the CoNLL-14 test set is consistently below 50%, and the recall on the BEA-19 test set is consistently below 65%. Thus, the effectiveness of detection control stems more from the trade-off between precision and recall, as discussed in the next section.

## 5.5 Precision-Recall Trade off

Adjusting the threshold for KEEP prediction probability (δ) and the probability lower bounds for ERROR and INSERT predictions $( \phi _ { e } , \phi _ { i } )$ defined in Section 3.5 allows for further adjustment of precision and recall, resulting in improved $F _ { 0 . 5 }$ scores. We performed a parameter search on the validation set to identify configurations maximizing $F _ { 0 . 5 }$ , and the results are depicted in Figure 4.

Without setting $\phi _ { e }$ and $\phi _ { i } , \delta = 0 . 3 8$ achieved the highest $F _ { 0 . 5 }$ of 63.2 on BEA-19 dev set. Then, we fix $\delta = 0$ .38 and perform a grid search for $\phi _ { e } ,$ ϕ<sub>i</sub>. All results are presented as points in the right plot of

![](images/5de3830647ade4b83195c4775a0170261f01b17b208aa198f7a10c6ee8c106f0.jpg)

![](images/1e615dbfa6c7750b8c64be71b8afb52fbef9caae4e9b8af5b7ab1cd917c78611.jpg)  
Figure 4: Results of detection control on $\mathrm { B E A } { - } 1 9$ dev set. The heat value represents the value of $F _ { 0 . \sharp }$ <sub>5</sub>.

Figure 4, and nearly all points are located within the region enclosed by the dashed line in the bottomleft. The dashed line represents the boundary of the model’s capability, and the intersection point with the $F _ { 0 . 5 }$ contour line represents the optimal performance attainable by the model. The point with the highest $F _ { 0 . 5 } = 6 3 . 5 $ is the one closest to the intersection point, with $\phi _ { e } = 0 . 5$ and $\phi _ { i } = $ 0.6. Under this parameter configuration, the model achieved an $F _ { 0 . 5 }$ value of 74.43 on the BEA-19 test set, as shown in Table 1. The detection control offers such a straightforward implementation of the precision-recall trade-off.

## 6 Conclusion

We introduce a novel language-agnostic detectioncorrection structure via GLM for the GEC task. The structure employs a three-label error detection pattern and uses Focal Loss for aggressive detection. The correction phase leverages the maskinfilling capability of GLM to generate correct text pieces. A multi-task learning approach is designed to integrate both functionalities within the same model, optimized using a weighted loss function. Experimental results show proposed model DeCoGLM outperforms previous detectioncorrection structures and achieves $F _ { 0 . 5 }$ scores comparable to SOTA on English and Chinese GEC benchmarks. The effectiveness of the detectioncorrection structure is further validated by applying it to open-source LLMs and GPT-4, indicating that incorporating error detection information improves the performance of LLMs on GEC datasets by reducing over-correction. Ablation studies confirm the efficacy of our model design and the ability to trade off precision and recall can be realized by detection control. We aim for this work to further guide GEC research within the detectioncorrection paradigm. The code and related models are available at https://github.com/GMago-LeWay/GECFramework.

## Limitations

Incremental methods proven effective on Seq2Seq models, such as incorporating syntactic information (Zhang et al., 2022b), refining training data (Mita et al., 2020), and employing additional models for reranking during the generation phase (Zhang et al., 2023; Zhou et al., 2023), are not implemented in this work. The main objective of this paper is to propose a novel GEC architecture, with these additional tricks serving as potential avenues for future extensions. Furthermore, due to resource restrictions, we are unable to apply our integrated detection-correction structure to LLMs. This is because the sequence labeling task differs from the generative tasks that LLMs are designed to perform, necessitating full-parameter fine-tuning to integrate the two tasks. Additionally, in our investigation of LLMs as correction models, models with parameters exceeding 13B are not utilized. The absence of full-parameter fine-tuning on LLMs and experiments with larger models due to resource constraints leaves room for further exploration of the application of the detection-correction paradigm on LLMs.

## Ethics Statement

The datasets and models we used are publicly available and utilized only for research purposes. The datasets do not contain any information that names or uniquely identifies individual people or offensive content. LLMs are utilized in our experiments, consistent with their intended use in natural language processing tasks. The models we designed will be published and intended for academic research in the field of grammatical error correction, in accordance with the original access conditions of the models used.

The detection-correction structure we designed limits the model to making only localized modifications to the text, preventing it from generating text without constraints, thereby significantly reducing the potential risks associated with the model. However, It is worth noting that the modifications made by the designed model may alter certain facts in the text, leading to hallucination, especially when modifications occur in named entities.

ChatGPT is utilized as the AI Assistant to polish the paper writing.

## Acknowledgements

We thank all the reviewers for their valuable comments on improving our paper. This work was supported by National Natural Science Foundation of China (62036001). The corresponding author is Houfeng Wang.

## References

Abhijeet Awasthi, Sunita Sarawagi, Rasna Goyal, Sabyasachi Ghosh, and Vihari Piratla. 2019. Parallel iterative edit models for local sequence transduction. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 4260–4270, Hong Kong, China. Association for Computational Linguistics.

Samuel Bell, Helen Yannakoudakis, and Marek Rei. 2019. Context is key: Grammatical error detection with contextual word representations. In Proceedings ofthe Fourteenth Workshop on Innovative Use ofNLP for Building Educational Applications, pages 103– 115, Florence, Italy. Association for Computational Linguistics.

Andrey Bout, Alexander Podolskiy, Sergey Nikolenko, and Irina Piontkovskaya. 2023. Efficient grammatical error correction via multi-task training and optimized training schedule. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 5800–5816, Singapore. Association for Computational Linguistics.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Christopher Bryant, Mariano Felice, Øistein E. Andersen, and Ted Briscoe. 2019. The BEA-2019 shared task on grammatical error correction. In Proceedings ofthe Fourteenth Workshop on Innovative Use ofNLP for Building Educational Applications, pages 52–75, Florence, Italy. Association for Computational Linguistics.

Christopher Bryant, Mariano Felice, and Ted Briscoe. 2017. Automatic annotation and evaluation of error types for grammatical error correction. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 793–805, Vancouver, Canada. Association for Computational Linguistics.

Andrew Caines, Luca Benedetto, Shiva Taslimipoor, Christopher Davis, Yuan Gao, Oeistein Andersen, Zheng Yuan, Mark Elliott, Russell Moore, Christopher Bryant, et al. 2023. On the application of large

language models for language teaching and assessment technology. arXiv preprint arXiv:2307.08393.

Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, Wei Ye, Yue Zhang, Yi Chang, Philip S. Yu, Qiang Yang, and Xing Xie. 2024. A survey on evaluation of large language models. ACM Trans. Intell. Syst. Technol. Just Accepted.

Mengyun Chen, Tao Ge, Xingxing Zhang, Furu Wei, and Ming Zhou. 2020. Improving the efficiency of grammatical error correction with erroneous span detection and correction. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7162–7169, Online. Association for Computational Linguistics.

Yo Joong Choe, Jiyeon Ham, Kyubyong Park, and Yeoil Yoon. 2019. A neural grammatical error correction system built on better pre-training and sequential transfer learning. In Proceedings of the Fourteenth Workshop on Innovative Use of NLP for Building Educational Applications, pages 213–227, Florence, Italy. Association for Computational Linguistics.

Steven Coyne, Keisuke Sakaguchi, Diana Galvan-Sosa, Michael Zock, and Kentaro Inui. 2023. Analyzing the performance of gpt-3.5 and gpt-4 in grammatical error correction. arXiv preprint arXiv:2303.14342.

Daniel Dahlmeier and Hwee Tou Ng. 2012. Better evaluation for grammatical error correction. In Proceedings ofthe 2012 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 568–572, Montréal, Canada. Association for Computational Linguistics.

Daniel Dahlmeier, Hwee Tou Ng, and Siew Mei Wu. 2013. Building a large annotated corpus of learner English: The NUS corpus of learner English. In Proceedings ofthe Eighth Workshop on Innovative Use of NLP for Building Educational Applications, pages 22–31, Atlanta, Georgia. Association for Computational Linguistics.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. BERT: pre-training of deep bidirectional transformers for language understanding. CoRR, abs/1810.04805.

Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. 2022. GLM: General language model pretraining with autoregressive blank infilling. In Proceedings of the 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 320–335, Dublin, Ireland. Association for Computational Linguistics.

Tao Fang, Jinpeng Hu, Derek F. Wong, Xiang Wan, Lidia S. Chao, and Tsung-Hui Chang. 2023a. Improving grammatical error correction with multimodal feature integration. In Findings of the Associationfor Computational Linguistics: ACL 2023,

pages 9328–9344, Toronto, Canada. Association for Computational Linguistics.

Tao Fang, Shu Yang, Kaixin Lan, Derek F Wong, Jinpeng Hu, Lidia S Chao, and Yue Zhang. 2023b. Is chatgpt a highly fluent grammatical error correction system? a comprehensive evaluation. arXiv preprint arXiv:2304.01746.

Zihao Fu, Haoran Yang, Anthony Man-Cho So, Wai Lam, Lidong Bing, and Nigel Collier. 2023. On the effectiveness of parameter-efficient fine-tuning. Proceedings of the AAAI Conference on Artificial Intelligence, 37(11):12799–12807.

Tao Ge, Furu Wei, and Ming Zhou. 2018. Fluency boost learning and inference for neural grammatical error correction. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1055–1065, Melbourne, Australia. Association for Computational Linguistics.

Roman Grundkiewicz, Marcin Junczys-Dowmunt, and Kenneth Heafield. 2019. Neural grammatical error correction systems with unsupervised pre-training on synthetic data. In Proceedings ofthe Fourteenth Workshop on Innovative Use of NLP for Building Educational Applications, pages 252–263, Florence, Italy. Association for Computational Linguistics.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685.

Marcin Junczys-Dowmunt, Roman Grundkiewicz, Shubha Guha, and Kenneth Heafield. 2018. Approaching neural grammatical error correction as a low-resource machine translation task. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 595–606, New Orleans, Louisiana. Association for Computational Linguistics.

Masahiro Kaneko, Kengo Hotate, Satoru Katsumata, and Mamoru Komachi. 2019. TMU transformer system using BERT for re-ranking at BEA 2019 grammatical error correction on restricted track. In Proceedings ofthe Fourteenth Workshop on Innovative Use ofNLPfor Building Educational Applications, pages 207–212, Florence, Italy. Association for Computational Linguistics.

Masahiro Kaneko, Masato Mita, Shun Kiyono, Jun Suzuki, and Kentaro Inui. 2020. Encoder-decoder models can benefit from pre-trained masked language models in grammatical error correction. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4248–4254, Online. Association for Computational Linguistics.

Masahiro Kaneko, Sho Takase, Ayana Niwa, and Naoaki Okazaki. 2022. Interpretability for language learners using example-based grammatical error correction. In Proceedings of the 60th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 7176–7187, Dublin, Ireland. Association for Computational Linguistics.

Anisia Katinskaia and Roman Yangarber. 2021. Assessing grammatical correctness in language learning. In Proceedings of the 16th Workshop on Innovative Use of NLP for Building Educational Applications, pages 135–146, Online. Association for Computational Linguistics.

Satoru Katsumata and Mamoru Komachi. 2020. Stronger baselines for grammatical error correction using a pretrained encoder-decoder model. In Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing, pages 827–832, Suzhou, China. Association for Computational Linguistics.

Shaopeng Lai, Qingyu Zhou, Jiali Zeng, Zhongli Li, Chao Li, Yunbo Cao, and Jinsong Su. 2022. Typedriven multi-turn corrections for grammatical error correction. In Findings ofthe Associationfor Computational Linguistics: ACL 2022, pages 3225–3236, Dublin, Ireland. Association for Computational Linguistics.

Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2019. BART: denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. CoRR, abs/1910.13461.

Jiquan Li, Junliang Guo, Yongxin Zhu, Xin Sheng, Deqiang Jiang, Bo Ren, and Linli Xu. 2022. Sequenceto-action: Grammatical error correction with action guided sequence generation. Proceedings of the AAAI Conference on Artificial Intelligence, 36(10):10974–10982.

Yinghao Li, Xuebo Liu, Shuo Wang, Peiyuan Gong, Derek F. Wong, Yang Gao, Heyan Huang, and Min Zhang. 2023a. TemplateGEC: Improving grammatical error correction with detection template. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6878–6892, Toronto, Canada. Association for Computational Linguistics.

Yinghui Li, Haojing Huang, Shirong Ma, Yong Jiang, Yangning Li, Feng Zhou, Hai-Tao Zheng, and Qingyu Zhou. 2023b. On the (in) effectiveness of large language models for chinese text correction. arXiv preprint arXiv:2307.09007.

Junwei Liao, Sefik Eskimez, Liyang Lu, Yu Shi, Ming Gong, Linjun Shou, Hong Qu, and Michael Zeng. 2023. Improving readability for automatic speech

recognition transcription. ACM Trans. Asian Low-Resour. Lang. Inf. Process., 22(5).

Jared Lichtarge, Chris Alberti, and Shankar Kumar. 2020. Data weighted training strategies for grammatical error correction. Transactions ofthe Association for Computational Linguistics, 8:634–646.

Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. 2020. Focal loss for dense object detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, 42(2):318–327.

Mengsay Loem, Masahiro Kaneko, Sho Takase, and Naoaki Okazaki. 2023. Exploring effectiveness of gpt-3 in grammatical error correction: A study on performance and controllability in prompt-based methods. arXiv preprint arXiv:2305.18156.

Jonathan Mallinson, Aliaksei Severyn, Eric Malmi, and Guillermo Garrido. 2020. FELIX: Flexible text editing through tagging and insertion. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2020, pages 1244–1255, Online. Association for Computational Linguistics.

Eric Malmi, Sebastian Krause, Sascha Rothe, Daniil Mirylenka, and Aliaksei Severyn. 2019. Encode, tag, realize: High-precision text editing. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 5054–5065, Hong Kong, China. Association for Computational Linguistics.

Masato Mita, Shun Kiyono, Masahiro Kaneko, Jun Suzuki, and Kentaro Inui. 2020. A self-refinement strategy for noise reduction in grammatical error correction. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 267–280, Online. Association for Computational Linguistics.

Hwee Tou Ng, Siew Mei Wu, Ted Briscoe, Christian Hadiwinoto, Raymond Hendy Susanto, and Christopher Bryant. 2014. The CoNLL-2014 shared task on grammatical error correction. In Proceedings of the Eighteenth Conference on Computational Natural Language Learning: Shared Task, pages 1–14, Baltimore, Maryland. Association for Computational Linguistics.

Kostiantyn Omelianchuk, Vitaliy Atrasevych, Artem Chernodub, and Oleksandr Skurzhanskyi. 2020. GECToR – grammatical error correction: Tag, not rewrite. In Proceedings of the Fifteenth Workshop on Innovative Use ofNLPfor Building Educational Applications, pages 163–170, Seattle, WA, USA → Online. Association for Computational Linguistics.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with

human feedback. In Advances in Neural Information Processing Systems, volume 35, pages 27730–27744. Curran Associates, Inc.

Fanyi Qu and Yunfang Wu. 2023. Evaluating the capability of large-scale language models on chinese grammatical error correction task. arXiv preprint arXiv:2307.03972.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67.

Marek Rei and Helen Yannakoudakis. 2016. Compositional sequence labeling models for error detection in learner writing. In Proceedings ofthe 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1181–1191, Berlin, Germany. Association for Computational Linguistics.

Sascha Rothe, Jonathan Mallinson, Eric Malmi, Sebastian Krause, and Aliaksei Severyn. 2021. A simple recipe for multilingual grammatical error correction. In Proceedings ofthe 59th Annual Meeting ofthe Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 702–707, Online. Association for Computational Linguistics.

Felix Stahlberg and Shankar Kumar. 2020. Seq2Edits: Sequence transduction using span-level edit operations. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 5147–5159, Online. Association for Computational Linguistics.

Felix Stahlberg and Shankar Kumar. 2021. Synthetic data generation for grammatical error correction with tagged corruption models. In Proceedings of the 16th Workshop on Innovative Use ofNLPfor Building Educational Applications, pages 37–47, Online. Association for Computational Linguistics.

Hanbo Sun, Jian Gao, Xiaomin Wu, Anjie Fang, Cheng Cao, and Zheng Du. 2023. Htec: Human transcription error correction. arXiv preprint arXiv:2309.10089.

Xin Sun, Tao Ge, Furu Wei, and Houfeng Wang. 2021. Instantaneous grammatical error correction with shallow aggressive decoding. In Proceedings ofthe 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 5937–5947, Online. Association for Computational Linguistics.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Yu Wang, Yuelin Wang, Kai Dang, Jie Liu, and Zhuo Liu. 2021. A comprehensive survey of grammatical error correction. ACM Trans. Intell. Syst. Technol., 12(5).

Lvxiaowei Xu, Jianwang Wu, Jiawei Peng, Jiayu Fu, and Ming Cai. 2022. FCGEC: Fine-grained corpus for Chinese grammatical error correction. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 1900–1918, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Konstantin Yakovlev, Alexander Podolskiy, Andrey Bout, Sergey Nikolenko, and Irina Piontkovskaya. 2023. GEC-DePenD: Non-autoregressive grammatical error correction with decoupled permutation and decoding. In Proceedings ofthe 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1546–1558, Toronto, Canada. Association for Computational Linguistics.

Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, et al. 2023. Baichuan 2: Open large-scale language models. arXiv preprint arXiv:2309.10305.

Helen Yannakoudakis, Ted Briscoe, and Ben Medlock. 2011. A new dataset and method for automatically grading ESOL texts. In Proceedings of the 49th Annual Meeting ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 180–189, Portland, Oregon, USA. Association for Computational Linguistics.

Zheng Yuan, Shiva Taslimipoor, Christopher Davis, and Christopher Bryant. 2021a. Multi-class grammatical error detection for correction: A tale of two systems. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 8722–8736, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Zheng Yuan, Shiva Taslimipoor, Christopher Davis, and Christopher Bryant. 2021b. Multi-class grammatical error detection for correction: A tale of two systems. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 8722–8736, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, et al. 2022. Glm-130b: An open bilingual pre-trained model. arXiv preprint arXiv:2210.02414.

Ying Zhang, Hidetaka Kamigaito, and Manabu Okumura. 2023. Bidirectional transformer reranker for grammatical error correction. In Findings ofthe Associationfor Computational Linguistics: ACL 2023, pages 3801–3825, Toronto, Canada. Association for Computational Linguistics.

Yue Zhang, Zhenghua Li, Zuyi Bao, Jiacheng Li, Bo Zhang, Chen Li, Fei Huang, and Min Zhang.

2022a. MuCGEC: a multi-reference multi-source evaluation dataset for Chinese grammatical error correction. In Proceedings of the 2022 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 3118–3130, Seattle, United States. Association for Computational Linguistics.

Yue Zhang, Bo Zhang, Zhenghua Li, Zuyi Bao, Chen Li, and Min Zhang. 2022b. SynGEC: Syntax-enhanced grammatical error correction with a tailored GECoriented parser. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 2518–2531, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Wei Zhao, Liang Wang, Kewei Shen, Ruoyu Jia, and Jingming Liu. 2019. Improving grammatical error correction via pre-training a copy-augmented architecture with unlabeled data. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 156–165, Minneapolis, Minnesota. Association for Computational Linguistics.

Yuanyuan Zhao, Nan Jiang, Weiwei Sun, and Xiaojun Wan. 2018. Overview of the nlpcc 2018 shared task: Grammatical error correction. In Natural Language Processing and Chinese Computing, pages 439–445, Cham. Springer International Publishing.

Houquan Zhou, Yumeng Liu, Zhenghua Li, Min Zhang, Bo Zhang, Chen Li, Ji Zhang, and Fei Huang. 2023. Improving Seq2Seq grammatical error correction via decoding interventions. In Findings ofthe Association for Computational Linguistics: EMNLP 2023, pages 7393–7405, Singapore. Association for Computational Linguistics.

## A Dataset

## A.1 Dataset Statistics

<table><tr><td>Dataset</td><td>#Sentences</td><td>Usage</td><td>As training data of</td></tr><tr><td>C4-200M</td><td>183,894,319</td><td>Pretraining</td><td>DeCoGLM, DeGLM, CoGLM</td></tr><tr><td>Synthetic-CH</td><td>33,166,047</td><td>Pretraining</td><td>DeCoGLM, DeGLM, CoGLM</td></tr><tr><td>CLang8(EN)</td><td>2,372,119</td><td>Fine-tuning</td><td>DeCoGLM, DeGLM, CoGLM</td></tr><tr><td>FCE all</td><td>33,236</td><td>Fine-tuning</td><td>CoGLM (10B)</td></tr><tr><td>NUCLE</td><td>57,157</td><td>Fine-tuning</td><td>CoGLM (10B)</td></tr><tr><td>W&amp;I+LOCNESS</td><td>34,308</td><td>Fine-tuning</td><td>CoGLM (10B)</td></tr><tr><td>Lang8 (CH)</td><td>1,092,285</td><td>Fine-tuning</td><td>All</td></tr><tr><td>HSK</td><td>95,320</td><td>Fine-tuning</td><td>All</td></tr><tr><td>FCGEC train</td><td>36,341</td><td>Fine-tuning</td><td>CoGLM (10B)</td></tr><tr><td>BEA19 dev</td><td>4,384</td><td>Validation</td><td></td></tr><tr><td>MuCGEC dev</td><td>1,137</td><td>Validation</td><td></td></tr><tr><td>FCGEC dev</td><td>2,000</td><td>Validation</td><td></td></tr><tr><td>CoNLL-14 test</td><td>1,312</td><td>Testing</td><td></td></tr><tr><td>BEA19 test</td><td>4,477</td><td>Testing</td><td></td></tr><tr><td>MuCGEC test</td><td>6,000</td><td>Testing</td><td></td></tr><tr><td>FCGEC test</td><td>3,000</td><td>Testing</td><td></td></tr></table>

Table 6: Dataset statistics. The rightmost column indicates the models that utilize the respective dataset; "All" signifies that DeCoGLM, DeGLM, CoGLM, and CoGLM (10B) all used the dataset as the training set.

In the experiments described in Section 4.1, the datasets used are outlined in Table 6. Due to constraints on our computational resources, the CoGLM (10B) models are fine-tuned on relatively smaller datasets, and the models are not pre-trained on synthetic datasets.

## A.2 Dataset for Training

As shown in Table 6, for relatively small models, we first pretrain using the publicly available C4- 200M English GEC synthetic dataset and our synthesized Chinese GEC dataset to obtain two pretrained models. Subsequently, the English model is fine-tuned using the CLang8 dataset, while the Chinese model is fine-tuned using the Lang8 Chinese dataset and the FCGEC dataset to yield two individual models. For the results of the models used for comparison in the primary results of Table 1, the GECToR and BART results on the two Chinese datasets are reproduced according to our training procedure, while the rest are as reported in the original papers, where they utilized various training data configurations. Most models are finetuned using the NUCLE, W&I+LOCNESS, and FCE datasets. Besides, GECToR uses the PIE-9M as the pretraining dataset (Awasthi et al., 2019).

For LLMs, we did not perform pretraining; instead, we directly applied the LoRA method using the NUCLE, W&I+LOCNESS, and FCE datasets for the English model. For Chinese, we trained two models using Lang8 (CH) and FCGEC. The training data for CoGLM (10B) and other LLMs used for comparison are completely consistent.

## A.3 Dataset Examples

In Sections 3.1 and 3.2, we describe the construction of training data. By aligning the source text with the target text, we derive error detection labels and masked text, thereby constructing training samples as illustrated in Figure 2. In Section 3.3, we elaborate on a two-stage supervised fine-tuning approach, where the training data for the second stage is reconstructed based on the detection predictions made by the model trained in the first stage. During data construction, model-induced false positives for ERROR and INSERT are incorporated to generate new masked text and corresponding text pieces. It is crucial to note that this process is solely aimed at creating new masked text to enhance the model’s ability to address false positives during the correction phase, while the detection labels used in training remain unchanged. Examples of the constructed training data are provided in Table 7, where "<s>" denotes the "begin of sentence" token and "</s>" represents the "end of sentence" token. For the sake of brevity, these tokens are omitted in the content of this paper except in Figure 2.

## B Details of Experiments

## B.1 Loss Weight

We pre-determine the weights in multi-task learning by intuitively observing the scales of two losses. This preliminary experiment was conducted on the CLang8 dataset, and the loss curves are depicted in Figure 5. It is evident from the figure that the detection loss $\ell _ { D }$ and correction loss $\ell _ { C }$ differ by roughly an order of magnitude. Consequently, we initially set $w _ { D } = 1 0$ , determine the weights for ERROR and INSERT categories in Focal Loss denoted by α<sub>EI</sub>, and subsequently test whether $w _ { D } = 1 0$ is an optimal choice, as discussed in Section 5.2.

![](images/47f90408d1c86e1cea0a00b8d17824f4b9679fee0d0549ef22688aaae5723a5a.jpg)  
Figure 5: Loss curves in standard training condition.

<table><tr><td rowspan=1 colspan=1>Stage</td><td rowspan=1 colspan=1>Items</td><td rowspan=1 colspan=1>Example 1</td><td rowspan=1 colspan=1>Example 2</td></tr><tr><td rowspan=6 colspan=1>SFT1</td><td rowspan=1 colspan=1>Source Text xs</td><td rowspan=1 colspan=1>&lt;s&gt;The every male employees were standing in the back row .&lt;/s&gt;</td><td rowspan=1 colspan=1>&lt;s&gt;They are covered with rust so bad .&lt;/s&gt;</td></tr><tr><td rowspan=1 colspan=1>Target Text y</td><td rowspan=1 colspan=1>&lt;s&gt;All the male employees were standing in the back row .&lt;/s&gt;</td><td rowspan=1 colspan=1>&lt;s&gt;They are covered with rust so badly .&lt;/s&gt;</td></tr><tr><td rowspan=1 colspan=1>Masked Text xm</td><td rowspan=1 colspan=1>&lt;s&gt;[MASK] male employees were standing in the back row .&lt;/s&gt;</td><td rowspan=1 colspan=1>&lt;s&gt;They are covered with rust so [MASK] .&lt;/s&gt;</td></tr><tr><td rowspan=1 colspan=1>Text Pieces Input</td><td rowspan=1 colspan=1>&lt;lstartofpiecel&gt; All the</td><td rowspan=1 colspan=1>&lt;lstartofpiecel&gt; badly</td></tr><tr><td rowspan=1 colspan=1>Text Pieces Target</td><td rowspan=1 colspan=1>All the &lt;lendofpiecel&gt;</td><td rowspan=1 colspan=1>badly &lt;lendofpiecel&gt;</td></tr><tr><td rowspan=1 colspan=1>Detection Labels</td><td rowspan=1 colspan=1>KEEKKKKKKKKKK</td><td rowspan=1 colspan=1>KKKKKKKEKK</td></tr><tr><td rowspan=5 colspan=1>SFT2</td><td rowspan=1 colspan=1>Detections by SFT1</td><td rowspan=1 colspan=1>KEEKEEKKKKKKK</td><td rowspan=1 colspan=1>KKKKKIKKKK</td></tr><tr><td rowspan=1 colspan=1>Merged Detecions</td><td rowspan=1 colspan=1>KEEKEEKKKKKKK</td><td rowspan=1 colspan=1>KKKKKIKEKK</td></tr><tr><td rowspan=1 colspan=1>Masked Text $\overline { { { \bf x } _ { m } ^ { \prime } } }$ </td><td rowspan=1 colspan=1>&lt;s&gt;[MASK] male [MASK] standing in the back row.&lt;/s&gt;</td><td rowspan=1 colspan=1>&lt;s&gt;They are covered with rust [MASK] so [MASK] .&lt;/s&gt;</td></tr><tr><td rowspan=1 colspan=1>Text Pieces Input</td><td rowspan=1 colspan=1>&lt;lstartofpiecel&gt; All the &lt;lstartofpiecel&gt; employees were</td><td rowspan=1 colspan=1>&lt;lstartofpiecel&gt; &lt;lstartofpiecel&gt; badly</td></tr><tr><td rowspan=1 colspan=1>Text Pieces Target</td><td rowspan=1 colspan=1>All the &lt;lendofpiecel&gt; employees were &lt;lendofpiecel&gt;</td><td rowspan=1 colspan=1>&lt;lendofpiecel&gt; badly &lt;lendofpiecel&gt;</td></tr></table>

Table 7: Examples of training data from CLang8 dataset in two fine-tuning stages. In detection labels, $\mathrm { K = K E E P , }$ E=ERROR and I=INSERT.
<table><tr><td>Configuration</td><td>EN Pretrain</td><td>EN finetune</td><td>CH Pretrain</td><td>CH finetune</td></tr><tr><td colspan="5">DeCoGLM-Training</td></tr><tr><td colspan="5">Backbone GLM-RoBERTa-large (Du et al., 2022) GLM-large-chinese (Du et al., 2022)</td></tr><tr><td>Backbone Parameters</td><td>335M</td><td></td><td colspan="2">335M</td></tr><tr><td>Batch size</td><td>12</td><td>12</td><td>12</td><td>12</td></tr><tr><td>Update frequecy</td><td>10</td><td>20</td><td>8</td><td>8(M), 10(F)</td></tr><tr><td>Max epochs</td><td>(20M iterations)</td><td>20</td><td>2</td><td>10(M), 20(F)</td></tr><tr><td>Evaluation key (SFT1)</td><td></td><td>AD-Accuracy</td><td>AD-Accuracy</td><td>AD-Accuracy</td></tr><tr><td>Evaluation key (SFT2)</td><td></td><td>General-Accuracy</td><td></td><td>General-Accuracy</td></tr><tr><td>Evaluation interval Early stop</td><td>10000</td><td>2000</td><td>4000</td><td>2000(M), 200(F)</td></tr><tr><td>Max source text length</td><td></td><td>10</td><td></td><td>10</td></tr><tr><td>Warm-up steps (SFT1)</td><td>128</td><td>128</td><td>128</td><td>128</td></tr><tr><td>Warm-up steps (SFT2)</td><td>10000</td><td>1000</td><td>1000</td><td>1000(M), 200(F)</td></tr><tr><td>Weight Decay</td><td></td><td>1000</td><td></td><td>1000(M), 200(F)</td></tr><tr><td></td><td> $1 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Learning rate scheduler</td><td> $\mathrm { P o l y n o m i a l }$ </td><td>Polynomial</td><td> $\mathrm { P o l y n o m i a l }$ </td><td>Polynomial</td></tr><tr><td>Learning rate (SFT1)</td><td> $2 \times 1 0 ^ { - 5 }$ </td><td> $3 \times 1 0 ^ { - 6 }$ </td><td> $2 \times 1 0 ^ { - 5 }$ </td><td> $1 \times 1 0 ^ { - 5 } \left( \mathrm { M } \right) , 4 \times 1 0 ^ { - 5 } \left( \mathrm { F } \right)$ </td></tr><tr><td>Learning rate (SFT2)</td><td></td><td> $1 \times 1 0 ^ { - 6 }$ </td><td></td><td> $5 \times 1 0 ^ { - 6 } \left( \mathrm { M } \right) , 1 \times 1 0 ^ { - 5 } \left( \mathrm { F } \right)$ </td></tr><tr><td colspan="5">DeCoGLM-Inference</td></tr><tr><td>KEEP threshold</td><td colspan="2">0.38</td><td colspan="2">None</td></tr><tr><td>ERROR lower bound</td><td colspan="2">0.5</td><td colspan="2">None</td></tr><tr><td>INSERT lower bound</td><td colspan="2">0.6</td><td colspan="2">None</td></tr><tr><td>Beam size</td><td colspan="2">3</td><td colspan="2"></td></tr><tr><td>Max tokens per piece</td><td colspan="2">10</td><td colspan="2">3 10</td></tr></table>

Table 8: The model hyper-parameters of proposed DeCoGLM. Both pretraining and fine-tuning configurations are presented. EN and CH represent English models and Chinese models, respectively. In the settings of Chinese fine-tuned models, M and F represent models for MuCGEC and FCGEC, respectively. The bottom of the table presents the hyper-parameters of inference.

## B.2 Model Configurations

The training configurations for the integrated detection-correction model (DeCoGLM) and the parameters used during inference are presented in Table 8. To conserve computational resources during training, early stopping is employed, which requires the pre-definition of evaluation metrics on the validation set. Two primary metrics are utilized: (1) AD-Accuracy, defined as the sum of the recall for ERROR and INSERT and the accuracy of next token prediction by GLM, aiming to reinforce the aggressive detection principle mentioned in Section 3.1; (2) General-Accuracy, the geometric mean between the recall for the three detection labels and the accuracy of next token prediction by GLM. The configurations for training the separate models, DeGLM and CoGLM, are similar to those in Table 8. The pre-trained models include glm-roberta-large, glm-large-chinese, glm-10b, and glm-10b-chinese, accessible through HuggingFace<sup>2</sup>. We implement all the designed models using PyTorch, including DeCoGLM, DeGLM, and CoGLM.

All models are trained by the Trainer from the transformers<sup>3</sup> package in Python, on NVIDIA RTX 4090 GPUs. Due to resource constraints, all experiments are conducted with a fixed random seed (111), and single-run results are reported. We adopt the approach recommended by Rothe et al. (2021)

<table><tr><td>Mode</td><td>Prompt</td></tr><tr><td>ZeroShot</td><td>Reply with a corrected version of the input sentence with all grammatical and spelling errors fixed. If there are no errors, reply with a copy of the original sentence. Input sentence: [TEXT]</td></tr><tr><td>+DeGLM</td><td>Corrected sentence: Reply with a corrected version of the input sentence with all grammatical and spelling errors fixed. If there are no errors, reply with a copy of the original sentence. Hint: We have detected some possible grammatical errors and replaced every error span with a [MASK] to get a masked sentence, you can reference the masked sentence to give final corrected sentence. If there is no [MASK] in the masked sentence, it means that we have not detected any grammatical errors in the input sentence. Input sentence: [TEXT] Masked Sentence: [MASKED_TEXT] Corrected sentence:</td></tr></table>

Table 9: GPT-4 prompts used in experiments, following Coyne et al. (2023).
<table><tr><td colspan="2"></td><td colspan="2"> $\overline { { \mathbf { F _ { 0 . 5 } } } }$  on test set</td><td colspan="3">Average inference time per sample (ms)</td></tr><tr><td>Backbone</td><td>Structure</td><td>CoNLL-14</td><td>BEA-19</td><td>Detection</td><td>Correction</td><td>Total</td></tr><tr><td>GLM-Roberta</td><td>De-Co</td><td>64.71</td><td>70.85</td><td>14.5</td><td>69.1</td><td>83.6</td></tr><tr><td>BART-large</td><td>De-Co</td><td>62.24</td><td>67.97</td><td>17.1</td><td>43.4</td><td>60.5</td></tr><tr><td>BART-large</td><td>Seq2Seq</td><td>64.46</td><td>67.94</td><td>1</td><td>266.2</td><td>266.2</td></tr></table>

Table 10: Time consumed in inference. De-Co represents the proposed detection-correction structure.

to post-process the model’s predictions on English test datasets, aiming to ensure greater alignment of tokenization with the evaluation data.

## B.3 GPT-4 Prompts

The prompts utilized during the inference of GPT-4 are illustrated in Table 9. For the Chinese tasks, the prompts are the direct translation of the corresponding English prompts. The API version of GPT-4 used in this paper is Preview-0315.

## C Inference Speed

We conduct a brief evaluation of the inference speed of our proposed detection-correction structure, and the average inference speeds on the CoNLL-14 and BEA-19 test sets are presented in Table 10. The models are trained exclusively on the CLang8 dataset, and during the inference phase, no hyperparameters are adjusted, utilizing only beam search. Our proposed model achieves slightly better performance while maintaining a faster inference speed ( 3x) than the Seq2Seq model. The experiments are conducted on an NVIDIA RTX 4090 GPU, with the same constrained batch size of 1 during inference.