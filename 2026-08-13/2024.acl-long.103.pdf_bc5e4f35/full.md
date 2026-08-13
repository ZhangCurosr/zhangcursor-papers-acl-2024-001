# Does DETECTGPT Fully Utilize Perturbation? Bridging Selective Perturbation to Fine-tuned Contrastive Learning Detector would be Better

Shengchao Liu<sup>1</sup>, Xiaoming Liu<sup>1,</sup> ∗, Yichen Wang<sup>1</sup>, Zehua Cheng<sup>1</sup>, Chengzhengxu Li<sup>1</sup>, Zhaohan Zhang<sup>2</sup>, Yu Lan<sup>1</sup>, Chao Shen<sup>1</sup>

<sup>1</sup>Faculty of Electronic and Information Engineering, Xi’an Jiaotong University

<sup>2</sup>Queen Mary University of London

{liusc, yichen.wang, czh2022, czx.li}@stu.xjtu.edu.cn {xm.liu, ylan2020, chaoshen}@xjtu.edu.cn, zhaohan.zhang@qmul.ac.uk

## Abstract

The burgeoning generative capabilities of large language models (LLMs) have raised growing concerns about abuse, demanding automatic machine-generated text detectors. DetectGPT (Mitchell et al., 2023), a zero-shot metric-based detector, first introduces perturbation and shows great performance improvement. However, in DetectGPT, the random perturbation strategy could introduce noise, and logit regression depends on the threshold, harming the generalizability and applicability of individual or small-batch inputs. Hence, we propose a novel fine-tuned detector, PECOLA, bridging metric-based and fine-tuned methods by contrastive learning on selective perturbation. Selective strategy retains important tokens during perturbation and weights for multi-pair contrastive learning. The experiments show that PECOLA outperforms the state-of-the-art (SOTA) by 1.20% in accuracy on average on four public datasets. And we further analyze the effectiveness, robustness, and generalization of the method. <sup>1</sup>

## 1 Introduction

Machine-generated text (MGT) detection is to discriminate MGT from human-written texts (HWT), preventing abuse of large language models (LLMs), including academic misconduct (Vasilatos et al., 2023), spam synthesis (Dou et al., 2020), untrustworthy news (Zellers et al., 2019), etc. Currently, existing MGT detection methods can be mainly classified into two categories (Wu et al., 2023a; Wang et al., 2024), i.e., fine-tuned methods (Liu et al., 2023; Hu et al., 2023; Verma et al., 2023; OpenAI, 2023; Mao et al., 2024) and zero-shot metric-based methods (Gehrmann et al., 2019; Mitchell et al., 2023; Yang et al., 2023; Bao et al., 2024; Wu et al., 2023b). In general terms, finetuned detector methods can achieve better accuracy than zero-shot metric-based methods, especially generalizable to black-box generators, but are more costly during data collection, fine-tuning, and running, in most cases. On the other hand, zero-shot metric-based methods show better interpretability than fine-tuned ones.

![](images/0cf4b105b71426e022bdfd682dddd0484ba8db35a6818e271ebd542c8e533b60.jpg)  
Figure 1: Example of the selective strategy perturbation of PECOLA, which prevent modifying important tokens (in green). Orange tokens are the perturbed texts.

DetectGPT (Mitchell et al., 2023), as an unsupervised zero-shot metric-based method, first introduces perturbation in MGT detection. Specifically, it applies random masking to the original input sample and uses T5 (Raffel et al., 2020) to fill in. It posits that minor perturbations of MGT tend to have lower log probability under the base model than the original sample. The introduction of perturbation in DetectGPT surpasses the vanilla logprobability-based method (Gehrmann et al., 2019) in white-box settings.

However, DetectGPT still has three significant defects: (i) DetectGPT’s reliance on the logit regression module’s threshold compromises its generalization in zero-shot settings and limited to large batch input, failing on individual inputs. (ii) DetectGPT does not fully utilize the perturbation. As a metrics-based method, it only considers the probability difference caused by perturbation, which is overly simplified and slightly indistinguishable. Perturbation should indeed be a stronger augment that carries implicit language pattern information. (iii) DetectGPT perturbs the original sample randomly and unrestricted, which could introduce more noise and negatively impact the performance (Kim et al., 2022). For example, Liu et al. (2023) find entity-relationship plays a role in the detection, which might be destroyed in random perturbation of DetectGPT.

In this paper, we thus propose a Perturbationbased Contrastive Learning model, PECOLA, for MGT detection, toward the defects with two stages, i.e., Selective Strategy Perturbation (Sec. 3.1) and Token-Level Weighted Multi-Pairwise Contrastive Learning (Sec. 3.2). Firstly, Selective Strategy Perturbation is a token-level rewriting method with restrictions on modifying important texts (Campos et al., 2020) to reduce noise. The motivation is to simulate the human behavior of modification (Verma and Lee, 2017; Fetaya et al., 2020; Wang et al., 2019). The perturbation strategy consists of token removal and substitution, as shown in Fig. 1. The experiments show that the Selective Strategy Perturbation method can improve the performance of both metrics-based (i.e., DetectGPT) and model-based methods. Secondly, we propose a Multi-Pairwise Contrastive Learning model to process the perturbed texts. Different from the logit regression module in DetectGPT, the trained model is generalizable without any threshold setting, and it can deal with individual inputs. Moreover, by utilizing multi-pairwise contrastive learning, the model could better utilize perturbation to focus on the language pattern gap between HWT and MGT. The importance weight from the perturbation stage is also reused as contrastive learning weight. No tably, by using contrastive learning, PECOLA is a strong few-shot fine-tuning method, which effectively bridges and integrates metric-based and fine-tuned detector categories. Finally, extensive experiments show PECOLA is significantly superior to baseline and SOTA methods on four datasets, PECOLA improves by 1.20% to SOTA on average under few-shot settings, surpassing the latest methods by 3.84% among metric-based detectors and by 1.62% among fine-tuned detectors. Further experiments show that PECOLA is as well better at generalization, robustness, and effectiveness.

Our contributions are summarized as follows:

• Selective Perturbation: Based on our analysis of various selective perturbation strategies, we propose a novel method considering token importance, which reduces the noise and benefits to both supervised and unsupervised approaches.

• Bridge Metric and Model-based Detectors: We utilize a novel fine-tuned contrastive learning module to replace the logit regression of DetectGPT (metric-based), which frees the detector from setting the threshold, enables it to deal with individual input, and can be generalizable and effective on the few-shot setting by contrasting perturbed texts with origin ones.

• Outperformance: Our detector PECOLA outperforms all eight compared models on four public datasets. And PECOLA is more robust to the choice of base model and filling model. Furthermore, we prove its generalization ability across domains and generators of data.

## 2 Related Work

Machine-generated Text Detection. While finetuned detectors have proven effective for MGT detection (Wahle et al., 2022; Hu et al., 2023), the requirement for annotated datasets poses a significant challenge due to the proliferation of unchecked, high-quality generated texts. To address this challenge, DetectGPT (Mitchell et al., 2023) and Fast-DetectGPT (Bao et al., 2024) have demonstrated strong performance in white-box zero-shot settings. Similarly, CoCo (Liu et al., 2023) is designed to detect MGT with low resource annotations, utilizing a coherence-based contrastive learning model. Moreover, SeqXGPT (Wang et al., 2023) utilize log probability lists from white-box LLMs as features; Sniffer (Shi et al., 2024) and GPT-Who (Venkatraman et al., 2023) place more emphasis on tracing the origin of MGT. Recently, watermarking (Kirchenbauer et al., 2023) is introduced to mitigate the risk associated with unchecked MGTs by embedding imperceptible signals within text outputs during generation. In contrast to previous methods, our approach integrates data perturbation with contrastive learning, placing particular emphasis on reducing reliance on mask-filling models and enhancing performance in few-shot scenarios.

Perturbation. Data perturbation methods find frequent application in text classification tasks (Gao et al., 2022; Shum et al., 2023), which is commonly employed through the technique of consistency regularization (Xie et al., 2020; Chen et al., 2020).

![](images/667cf85010969826ba8c79f49d115f3a6b04dc84da99dba4479eeaab94e8fc6d.jpg)  
Figure 2: Overview of PECOLA. In the Selective Strategy Perturbation stage (Sec. 3.1), we use the YAKE algorithm to score token importance and then selective masking based on probability. Then, we fill in the masks with a mark-filling language model. In the Contrastive Learning stage (Sec. 3.2), we design a multi-pairwise method with token-level weights also from tokens importance. Yellow arrows represent attraction and blue ones represent repulsion. The model is optimized by combining cross-entropy (CE) loss $\mathcal { L } _ { \mathrm { c e } }$ and contrastive loss ${ \mathcal { L } } _ { \mathrm { c o n } } .$ \* Our method, different from DetectGPT, is generalizable on any mask-filling language model.

Nevertheless, in MGT detection, previous perturbation methods have exhibited certain limitations. For instance, they often resort to randomly selecting target tokens for synonym replacement (Wang et al., 2018), deletion, insertion (Wei and Zou, 2019), rewriting by LLMs (Mao et al., 2024), and fine-tuning pre-trained language models (PLMs) to fill text spans of variable lengths (Gao et al., 2022). While these methods do enhance text diversity, the indiscriminate replacement of tokens without guided rules can lead to the generation of less reliable texts. Wang et al. (2024) utilize perturbations as stress test approaches for the robustness of MGT detectors to show their loopholes. These limitations motivate us to devise data perturbation methods tailored for MGT detection. Our approach, with selective perturbation, aims to better represent meaningful recombination spaces while preserving the inherent semantic features of the text, ultimately enhancing the diversity of samples.

Constrastive Learning. Contrastive learning is an effective solution method to the issues that solely relying on cross-entropy classification loss would lead to a lack of robustness and suboptimal generalization (Tack et al., 2020; Hu et al., 2023). In limited labeled data task (Gunel et al., 2021), introduce a robust contrastive learning method to capture the similarities between the same instances in the representation space while separating those of different classes. Similarly, out-of-distribution (OOD) usually leads to severe semantic shift issues during inference, prompting another approach based on margin contrastive learning (Zhou et al., 2021). Differently, our method focuses more on the changes of the rephrase space in data distribution after perturbation, and strives to reduce reliance on the mask-filling models in few-shot learning.

## 3 Methodology

As shown in Fig. 2, the workflow of PECOLA mainly consists of two stages: Selective Strategy Perturbation and Supervised Contrastive Learning, which joined the advantage of metric-based and model-based detection methods, respectively.

## 3.1 Selective Strategy Perturbation

In this work, we present a token-level selective strategy perturbation method to relieve the information loss caused by the random masking used in DetectGPT. Our approach involves adapting the maskselection probability for each text token based on its importance, thus generating perturbed inputs with strategically placed masks. Additionally, we harness LLMs to populate the masks, creating filled perturbation inputs. This step effectively introduces a diverse range of perturbation information into our detection model.

Token Importance Assessment. To accurately assess the significance of tokens within the text and mitigate information loss stemming from random masking, we expand upon the YAKE algorithm (Campos et al., 2020) to operate at the token level. The YAKE algorithm builds upon certain assumptions (Machado et al., 2009), which posit that the importance of a candidate word decreases as the richness of the vocabulary surrounding it increases. This fundamental assumption remains applicable when processing text at the token level, i.e., token importance assessment.

Specifically, considering a training set S comprising i inputs, for each text input $s _ { i } \in S$ containing n tokens $( i . e . , s _ { i } = \{ e _ { i } ^ { 1 } , e _ { i } ^ { 2 } , . . . , e _ { i } ^ { n } \} )$ , we employ the YAKE algorithm to compute a score for each token e. Tokens with scores falling below the specified threshold α are then incorporated into the set of important tokens $K _ { i }$

$$
K _ { i } = \left\{ \begin{array} { l l } { K _ { i } \cup \left\{ e _ { i } ^ { n } \right\} , } & { \mathrm { i f } ~ S c o r e ( e _ { i } ^ { n } ) < \alpha } \\ { K _ { i } , } & { \mathrm { o t h e r w i s e } } \end{array} , \right.\tag{1}
$$

where $S c o r e ( e _ { i } ^ { n } )$ represents the YAKE score calculated by token $e _ { i } ^ { n }$ . The higher the score, the lower the importance of the token $e _ { i } ^ { n }$ in $s _ { i }$

Mask Position Selection. After getting the important tokens set $K _ { i }$ of each text input $s _ { i } ,$ we use special token [MASK] to replace some of the tokens in the text input to construct masked perturbation input $s _ { i } ^ { m a s k }$ . In order to relieve the information loss caused by masking perturbation, we add regularization to the vanilla random masking method and use a selective masking strategy to prevent important tokens from being masked.

Given an input text $s _ { i } = \{ e _ { i } ^ { 1 } , e _ { i } ^ { 2 } , . ~ . ~ . , e _ { i } ^ { n } \}$ , we use the selective masking strategy to traverse each token and determine whether to mask it based on the token’s importance. The probability of token $e _ { i } ^ { n }$ being masked is specifically defined as:

$$
P _ { i } ^ { n } = \mathbf { 1 } _ { [ e _ { i } ^ { n } \not \in K _ { i } ] } P ,\tag{2}
$$

where $P$ is the mask ratio, and $\mathbf { 1 } _ { [ e _ { i } ^ { n } \notin K _ { i } ] }$ represents an indicator function with a value of 1 if and only if the condition $e _ { i } ^ { n } \notin K _ { i }$ is satisfied, otherwise, it is 0. Then we gather all masked perturbation inputs $\{ s _ { 1 } ^ { m a s k } , . . . , s _ { i } ^ { m a s k } \}$ and include them in the training set to give the model masked perturbation, improving model robustness.

Mask-Filling. Additionally, we utilize PLMs, $e . g .$ , T5 (Raffel et al., 2020) or RoBERTa (Liu et al., 2019) etc., to fill the masked perturbation inputs and create the filled perturbation inputs $\{ s _ { 1 } ^ { \bullet } { } ^ { f i l l } , \ldots , s _ { i } ^ { f i l l } \}$ . Similar to the above, we include all filled perturbation inputs in the training set and obtain the final training set S = $\{ s _ { 1 } , \dotsc , s _ { i } , s _ { 1 } ^ { m a s k } , \dotsc , s _ { i } ^ { m a s k } , s _ { 1 } ^ { f i l l } \dotsc , s _ { i } ^ { f i l l } \}$

## 3.2 Token-Level Weighted Multi-Pairwise Contrastive Learning

Importance-based Feature Reconstruction. Existing MGT methods (Liu et al., 2023) often uniformly extract all token information in the text, ignoring the huge impact of a few important tokens on the detection model. In this work, we reconstruct the token feature extracted by PLM according to the importance of the token in the input text, allowing the detection model to focus more on important token information. We assign adaptive weights to all tokens in the input:

$$
w _ { i } ^ { n } = \left\{ \begin{array} { c l } { 1 - S c o r e ( e _ { i } ^ { n } ) , } & { \mathrm { i f } ~ e _ { i } ^ { n } \in K _ { i } } \\ { 0 , } & { \mathrm { o t h e r w i s e } } \end{array} , \right.\tag{3}
$$

where $w _ { i } ^ { n }$ represents the assign adaptive weight of the n-th token of the i-th input in the training set. After that, we use the last hidden layer embedding of the outputs in the base PLMs to extract input features:

$$
H _ { i } = { \mathrm { P L M } } ( s _ { i } ) ,\tag{4}
$$

where $H _ { i }$ contains the features of all tokens in the input $s _ { i }$ i.e., $H _ { i } = \{ h _ { i } ^ { 1 } , h _ { i } ^ { 2 } , \ldots , h _ { i } ^ { n } \}$ . We use the weight of the corresponding token to reconstruct its features:

$$
h _ { i } ^ { n } = h _ { i } ^ { n } ( 1 + w _ { i } ^ { n } ) .\tag{5}
$$

By using feature reconstruction, we assign more weight to important tokens. This allows our detection model to concentrate on the characteristic information of these important tokens.

Multi-Pairwise Contrastive Learning. Considering that existing works (Gunel et al., 2021; Zhou et al., 2021; Liu et al., 2023) mainly concentrate on single-input feature learning while overlooking input correlations, we introduce contrastive learning into MGT. It enables PECOLA to discern the distinct featurinputes of variously labeled data, more accurately capture input features, and significantly enhance performance in few-shot setting.

Given a batch training data $\{ s _ { i } \} _ { i = 1 } ^ { M }$ , where M is the batch size, we calculate the positive class contrastive loss and negative class contrastive loss on the last hidden layer embedding of the first token output $h _ { i } ^ { 1 }$ from the base PLM:

$$
\mathcal { L } _ { \mathrm { p o s } } = \sum _ { i = 1 } ^ { M } \frac { 1 } { | P _ { t } ( i ) | } \sum _ { p \in P _ { t } ( i ) } \Vert ( h _ { i } ^ { 1 } - h _ { p } ^ { 1 } ) \Vert ^ { 2 } ,\tag{6}
$$

$$
\mathcal { L } _ { \mathrm { n e g } } = \sum _ { i = 1 } ^ { M } \frac { 1 } { | N _ { t } ( i ) | } \sum _ { n \in N _ { t } ( i ) } \operatorname* { m a x } \left( 0 , \xi - \| ( h _ { i } ^ { 1 } - h _ { n } ^ { 1 } ) \| ^ { 2 } \right) ,\tag{7}
$$

where $P _ { t } ( i )$ represents the samples with the same label as the i-th sample in the batch, and $N _ { t } ( i )$ represents the ones with different labels as the $i -$ th sample. And $\xi$ is the maximum $L _ { 2 }$ distance between pairs of inputs from the same class in the batch of training data:

$$
\xi = \operatorname* { m a x } _ { i = 1 } ^ { M } \operatorname* { m a x } _ { p \in P _ { t } ( i ) } \Vert h _ { i } ^ { 1 } - h _ { p } ^ { 1 } \Vert ^ { 2 } .\tag{8}
$$

This adaptive margin ensures that the model is steered to maintain discriminative embeddings despite data perturbation during training. Then we get the following contrastive loss as:

$$
\mathcal { L } _ { \mathrm { c o n } } = \frac { 1 } { M } ( \mathcal { L } _ { \mathrm { p o s } } + \mathcal { L } _ { \mathrm { n e g } } ) .\tag{9}
$$

For supervised learning tasks, we utilize the crossentropy classification loss $\mathcal { L } _ { c e }$ to train our detection model. By adjusting the weight λ to balance the impact of various losses on the model, our total loss is given by the following:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { c e } } + \lambda \mathcal { L } _ { \mathrm { c o n } } . } \end{array}\tag{10}
$$

## 4 Experiments

## 4.1 Experiment Settings

To demonstrate the effectiveness of PECOLA, we conduct extensive experiments on four open-source datasets under few-shot learning settings.

Datasets. Grover (Zellers et al., 2019), generated by the transformer-based news generator Grover-Mega (1.5B); GPT-2, a webtext dataset provided by OpenAI (2019) based on GPT-2 XL (1.5B); GPT-3.5, a news-style dataset constructed by CoCo (Liu et al., 2023) using the text-DaVinci-003 model (175B); HC3 (Guo et al., 2023), involving open domains, finance, healthcare, law, and psychology texts, composed of comparative responses from human experts and ChatGPT.

Few-shot Learning Settings. We randomly sample 32, 64, 128 and 512 samples from the original training set, while keeping the balance of machine and human categories. More details are provided in Appendix A.1.

## 4.2 Comparison Models

We compare PECOLA with both unsupervised and supervised MGT detection methods:

RoBERTa (Liu et al., 2019), supervised methods via standard fine-tuning PLMs as classifiers. We use RoBERTa-base (125M).

GLTR (Gehrmann et al., 2019), a metric-based detector and based on next-token probability. We follow the setting of Guo et al. (2023), utilizing the Test-2 feature. For a fair comparison with finetuning methods, we first use the few-shot training samples to settle the threshold and adapt the fixed threshold in the test set.<sup>2</sup>

CE+SCL (Gunel et al., 2021), a fine-tuned detector, used in conjunction with the Cross-Entropy (CE) loss, exhibiting impressive performance in few-shot learning settings.

CE+Margin (Zhou et al., 2021), a contrastive learning approach focuses on separating OOD instances from In-Distribution (ID) instances, aiming to minimize the $\mathrm { L } _ { 2 }$ distance between instances of the same label. We train the detector by combining CE loss. IT:Clust (Shnarch et al., 2022), a general text classification method that employs unsupervised clustering as an intermediate for fine-tuning PLMs, utilizing RoBERTa-base.

CoCo (Liu et al., 2023) utilizes coherence graph representation and contrastive learning to improve supervised fine-tuning methods in both inadequate and adequate data resource scenarios.

DetectGPT (Mitchell et al., 2023), a zero-shot metric-based MGT detector, using T5-large (Raffel et al., 2020) to perturb texts. Same as GLTR, we fix the threshold.<sup>3</sup>

Fast-DetectGPT (Bao et al., 2024), an optimized zero-shot detector, building upon the foundation of DetectGPT, and utilizes a surrogate GPT-Neo (2.7B) (Black et al., 2022) model for scoring.

## 4.3 Performance Comparison

As shown in Table 1, PECOLA surpasses the competitors on all datasets in the few-shot MGT detection task. Specifically, compared with the best competitor, PECOLA achieves accuracy and F1- score improvement of 2.04% and 1.42%, 1.71% and 2.55% on Grover and GPT2 datasets. On GPT3.5 and HC3 datasets, PECOLA still ensures 0.86% and 0.68%, 0.21% and 0.22% performance improvement with greater stability. The results prove the effectiveness of PECOLA, which integrates the advantage of unsupervised (perturbation for metric-based) and supervised (contrastive learning for model-based) MGT detection methods.

<table><tr><td>Dataset</td><td>Metric</td><td>Shot</td><td>RoBERTa</td><td>GLTR†</td><td>CE+SCL</td><td>CE+Margin</td><td>IT:Clust</td><td> $c o c o *$ </td><td>DetectGPT</td><td>Fast-Detect.</td><td>PECOLA</td></tr><tr><td rowspan="8">Grver</td><td rowspan="8"> $A c c$  Fl</td><td>32</td><td> $4 8 . 8 3 _ { 1 0 . 3 1 }$ </td><td>56.61</td><td> $5 5 . 8 6 _ { 4 . 4 3 }$ </td><td> $5 6 . 7 9 _ { 3 . 3 1 }$ </td><td> $4 1 . 5 7 _ { 3 . 5 8 }$ </td><td> $5 1 . 6 0 _ { 8 . 4 2 }$ </td><td>55.02</td><td>56.06</td><td> $\mathbf { 5 9 . 0 3 _ { 1 . 6 3 } }$ </td></tr><tr><td>64</td><td> $5 6 . 8 8 _ { 3 . 0 3 }$ </td><td>56.61</td><td> $5 7 . 5 7 _ { 2 . 6 3 }$ </td><td> $5 8 . 9 2 _ { 2 . 1 7 }$ </td><td> $4 6 . 4 5 _ { 2 . 2 0 }$ </td><td> $5 8 . 2 7 _ { 1 0 . 2 1 }$ </td><td>54.61</td><td>60.33</td><td> $\mathbf { 6 0 . 9 4 _ { 1 . 5 6 } }$ </td></tr><tr><td>128</td><td> $5 9 . 2 8 _ { 1 . 9 1 }$ </td><td>58.48</td><td> $6 0 . 3 3 _ { 3 . 4 1 }$ </td><td> $6 0 . 4 4 _ { 3 . 8 5 }$ </td><td> $5 0 . 7 2 _ { 3 . 7 0 }$ </td><td> $5 8 . 9 7 _ { 5 . 5 3 }$ </td><td>55.78</td><td>60.33</td><td> $\mathbf { 6 3 . 6 0 _ { 1 . 7 1 } }$ </td></tr><tr><td>512</td><td> $7 0 . 3 9 _ { 1 . 2 1 }$ </td><td>62.26</td><td> $7 2 . 3 8 _ { 1 . 7 3 }$ </td><td> $7 2 . 1 5 _ { 1 . 1 6 }$ </td><td> $5 6 . 0 8 _ { 0 . 8 7 }$ </td><td> $7 0 . 0 7 _ { 5 . 5 4 }$ </td><td>55.56</td><td>62.50</td><td> $7 3 . 1 2 _ { 0 . 8 4 }$ </td></tr><tr><td>32</td><td> $4 4 . 1 3 _ { 8 . 8 2 }$ </td><td>52.77</td><td> $5 1 . 5 6 _ { 3 . 0 3 }$ </td><td> $5 3 . 2 1 _ { 2 . 2 4 }$ </td><td> $4 0 . 7 9 _ { 3 . 6 6 }$ </td><td> $4 7 . 3 3 _ { 2 . 6 3 }$ </td><td>51.09</td><td>56.67</td><td> ${ \bar { \mathbf { 5 } } } 3 . 9 5 _ { 0 . 9 4 }$ </td></tr><tr><td>64</td><td> $5 2 . 8 8 _ { 1 . 5 2 }$ </td><td>52.77</td><td> $5 3 . 3 9 _ { 1 . 1 6 }$ </td><td> $5 4 . 9 9 _ { 1 . 7 5 }$ </td><td> $4 6 . 1 0 _ { 1 . 2 5 }$ </td><td> $4 4 . 7 0 _ { 3 . 5 3 }$ </td><td>48.07</td><td>57.92</td><td> $5 5 . 4 8 _ { 1 . 3 5 }$ </td></tr><tr><td>128</td><td> $5 4 . 6 9 _ { 1 . 1 8 }$ </td><td>54.47</td><td> $5 5 . 7 4 _ { 2 . 2 1 }$ </td><td> $5 5 . 5 4 _ { 2 . 4 0 }$ </td><td> $5 1 . 3 7 _ { 4 . 8 0 }$ </td><td> $5 1 . 4 4 _ { 2 . 1 3 }$ </td><td>53.78</td><td>54.89</td><td> $\mathbf { 5 8 . 9 8 _ { 1 . 5 8 } }$ </td></tr><tr><td>512</td><td> $6 4 . 4 9 _ { 3 . 1 7 }$ </td><td>57.11</td><td> $6 7 . 0 2 _ { 2 . 1 2 }$ </td><td> $6 6 . 2 5 _ { 1 . 6 5 }$ </td><td> $5 1 . 8 0 _ { 0 . 4 9 }$ </td><td> $6 5 . 1 5 _ { 3 . 7 6 }$ </td><td>53.32</td><td>61.29</td><td> $\mathbf { 6 8 . 2 4 _ { 1 . 6 4 } }$ </td></tr><tr><td rowspan="8">G2</td><td rowspan="6"> $A c c$ </td><td>32</td><td> $7 0 . 5 3 _ { 4 . 1 0 }$ </td><td>75.99</td><td> $6 9 . 3 2 _ { 5 . 1 9 }$ </td><td> $7 0 . 0 0 _ { 2 . 3 3 }$ </td><td> $5 1 . 0 2 _ { 1 . 6 6 }$ </td><td> $7 1 . 6 9 _ { 7 . 0 7 }$ </td><td>68.59</td><td>71.88</td><td> $7 5 . 4 2 _ { 1 . 8 0 }$ </td></tr><tr><td>64</td><td> $7 4 . 4 1 _ { 2 . 4 7 }$ </td><td>75.76</td><td> $7 3 . 7 7 _ { 3 . 5 4 }$ </td><td> $7 4 . 0 4 _ { 1 . 4 2 }$ </td><td> $5 4 . 3 2 _ { 2 . 7 3 }$ </td><td> $7 3 . 2 0 _ { 1 . 4 2 }$ </td><td>71.12</td><td>71.88</td><td> $7 8 . 9 2 _ { 1 . 1 4 }$ </td></tr><tr><td>128</td><td> $7 9 . 7 7 _ { 2 . 0 4 }$ </td><td>75.77</td><td> $8 0 . 1 8 _ { 1 . 2 5 }$ </td><td> $8 0 . 9 3 _ { 1 . 2 6 }$ </td><td> $5 9 . 6 6 _ { 2 . 8 3 }$ </td><td> $7 9 . 4 4 _ { 4 . 8 0 }$ </td><td>71.74</td><td>71.88</td><td> $\mathbf { 8 2 . 5 8 _ { 0 . 4 9 } }$ </td></tr><tr><td>512</td><td> $8 4 . 0 7 _ { 1 . 4 6 }$ </td><td>75.86</td><td> $8 4 . 7 6 _ { 1 . 1 9 }$ </td><td> $8 4 . 8 9 _ { 1 . 1 7 }$ </td><td> $7 1 . 5 9 _ { 3 . 2 3 }$ </td><td> $8 4 . 3 0 _ { 0 . 5 8 }$ </td><td>71.74</td><td>74.06</td><td> $\mathbf { 8 5 . 7 5 _ { 0 . 6 9 } }$ </td></tr><tr><td>32</td><td> $6 6 . 5 7 _ { 5 . 0 9 }$ </td><td>72.45</td><td> $6 4 . 8 9 _ { 8 . 1 3 }$ </td><td> $6 9 . 8 9 _ { 2 . 3 8 }$ </td><td> $4 8 . 4 5 _ { 3 . 7 2 }$ </td><td> $7 1 . 1 9 _ { 1 1 . 0 5 }$ </td><td>65.50</td><td>70.00</td><td> $7 5 . 1 0 _ { 1 . 9 9 }$ </td></tr><tr><td>64</td><td> $7 3 . 9 1 _ { 2 . 6 9 }$ </td><td>70.87</td><td> $7 2 . 3 2 _ { 4 . 3 1 }$ </td><td> $7 3 . 9 4 _ { 1 . 4 0 }$ </td><td> $5 3 . 8 7 _ { 3 . 0 0 }$ </td><td> $6 9 . 7 9 _ { 2 . 0 3 }$ </td><td>66.58</td><td>70.97</td><td> $7 8 . 8 8 _ { 1 . 1 7 }$ </td></tr><tr><td>128</td><td> $7 9 . 4 9 _ { 2 . 2 6 }$ </td><td>71.16</td><td> $8 0 . 0 0 _ { 1 . 3 5 }$ </td><td> $8 0 . 7 9 _ { 1 . 3 4 }$ </td><td> $5 9 . 4 8 _ { 2 . 7 9 }$ </td><td> $7 6 . 1 0 _ { 7 . 3 7 }$ </td><td>66.13</td><td>71.88</td><td> $\mathbf { 8 } 2 . 5 4 _ { 0 . 5 1 }$ </td></tr><tr><td rowspan="8">GP-35</td><td rowspan="5"> $A c c$ </td><td>512</td><td> $8 4 . 0 1 _ { 1 . 5 2 }$ </td><td>75.56</td><td> $8 4 . 7 2 _ { 1 . 2 5 }$ </td><td> $8 4 . 8 6 _ { 1 . 2 4 }$ </td><td> $7 0 . 4 2 _ { 4 . 2 6 }$ </td><td> $8 3 . 8 8 _ { 0 . 7 9 }$ </td><td>66.13</td><td>74.64</td><td> $\mathbf { 8 5 . 7 2 _ { 0 . 7 0 } }$ </td></tr><tr><td>32</td><td> $9 0 . 5 4 _ { 7 . 2 6 }$ </td><td>92.55</td><td> $9 2 . 4 4 _ { 3 . 1 9 }$ </td><td> $9 2 . 8 5 _ { 2 . 4 4 }$ </td><td> $6 1 . 8 2 _ { 4 . 3 0 }$ </td><td> $9 3 . 2 7 _ { 1 . 4 4 }$ </td><td>84.42</td><td>89.10</td><td> $\mathbf { 9 5 . 8 0 _ { 0 . 6 8 } }$ </td></tr><tr><td>64</td><td> $9 6 . 8 5 _ { 0 . 8 4 }$ </td><td>91.00</td><td> $9 6 . 8 6 _ { 1 . 6 7 }$ </td><td> $9 7 . 3 2 _ { 0 . 5 8 }$ </td><td> $7 7 . 7 0 _ { 6 . 9 2 }$ </td><td> $9 5 . 7 6 _ { 1 . 5 2 }$ </td><td>82.58</td><td>89.65</td><td> $\mathbf { 9 8 . 0 1 _ { 0 . 3 1 } }$ </td></tr><tr><td>128</td><td> $9 7 . 5 0 _ { 1 . 2 4 }$ </td><td>91.60</td><td> $9 8 . 0 0 _ { 0 . 4 6 }$ </td><td> $9 8 . 0 0 _ { 0 . 1 8 }$ </td><td> $9 2 . 5 4 _ { 4 . 0 1 }$ </td><td> $9 6 . 2 6 _ { 0 . 8 9 }$ </td><td>85.33</td><td>89.85</td><td> $\mathbf { 9 8 . 0 6 _ { 0 . 1 2 } }$ </td></tr><tr><td>512</td><td> $9 8 . 9 7 _ { 0 . 1 8 }$ </td><td>92.60</td><td> $9 8 . 9 9 _ { 0 . 8 0 }$ </td><td> $9 8 . 9 2 _ { 0 . 2 8 }$ </td><td> $9 8 . 1 3 _ { 1 . 2 0 }$ </td><td> $9 8 . 0 5 _ { 0 . 4 7 }$ </td><td>85.57</td><td>90.62</td><td> $\mathbf { 9 9 . 1 4 _ { 0 . 1 5 } }$ </td></tr><tr><td>32</td><td> $9 0 . 2 7 _ { 7 . 7 7 }$ </td><td>92.71</td><td> $9 2 . 4 2 _ { 3 . 2 0 }$ </td><td> $9 2 . 8 1 _ { 2 . 4 9 }$ </td><td> $6 0 . 9 5 _ { 4 . 6 7 }$ </td><td> $9 2 . 7 2 _ { 1 . 5 4 }$ </td><td>84.43</td><td>89.76</td><td> $\mathbf { 9 5 . 8 0 _ { 0 . 6 8 } }$ </td></tr><tr><td rowspan="4">Fl</td><td>64</td><td> $9 6 . 8 4 _ { 0 . 8 4 }$ </td><td>91.49</td><td> $9 6 . 8 6 _ { 1 . 6 7 }$ </td><td> $9 7 . 4 7 _ { 0 . 3 0 }$ </td><td> $7 7 . 3 3 _ { 7 . 3 1 }$ </td><td> $9 5 . 4 5 _ { 1 . 5 4 }$ </td><td>86.16</td><td>89.92</td><td> $\mathbf { 9 8 . 0 1 _ { 0 . 3 1 } }$ </td></tr><tr><td>128</td><td> $9 7 . 5 0 _ { 1 . 2 4 }$ </td><td>91.96</td><td> $9 8 . 0 0 _ { 0 . 4 6 }$ </td><td> $9 8 . 0 0 _ { 0 . 1 8 }$ </td><td> $9 2 . 5 0 _ { 4 . 0 7 } $ </td><td> $9 7 . 5 7 _ { 0 . 9 2 }$ </td><td>86.13</td><td>89.77</td><td> $\mathbf { 9 8 . 0 6 _ { 0 . 1 2 } }$ </td></tr><tr><td>512</td><td> $9 8 . 8 5 _ { 0 . 4 0 }$ </td><td>92.71</td><td> $9 8 . 9 3 _ { 0 . 2 1 }$ </td><td> $9 8 . 9 2 _ { 0 . 2 8 }$ </td><td> $9 8 . 1 3 _ { 1 . 2 0 }$ </td><td> $9 7 . 8 8 _ { 0 . 5 0 }$ </td><td>86.20</td><td>90.62</td><td> $\mathbf { 9 9 . 1 4 _ { 0 . 1 5 } }$ </td></tr><tr><td>32</td><td> $9 3 . 3 6 _ { 1 . 5 0 }$ </td><td>97.30</td><td> $9 5 . 3 3 _ { 1 . 8 1 }$ </td><td> $9 5 . 4 6 _ { 1 . 7 1 }$ </td><td> $7 7 . 0 0 _ { 8 . 0 5 }$ </td><td> $9 2 . 1 1 _ { 1 . 7 1 }$ </td><td>94.54</td><td>87.70</td><td> $9 7 . 1 9 _ { 0 . 1 6 }$ </td></tr><tr><td rowspan="8">HC</td><td rowspan="6">Acc</td><td>64</td><td> $9 6 . 9 7 _ { 0 . 7 4 }$ </td><td>98.13</td><td> $9 7 . 8 1 _ { 0 . 4 1 }$ </td><td> $9 7 . 8 1 _ { 0 . 3 1 }$ </td><td> $9 1 . 6 9 _ { 2 . 3 4 }$ </td><td> $9 5 . 5 0 _ { 1 . 2 7 }$ </td><td>95.03</td><td>88.87</td><td> $\mathbf { 9 8 . 5 9 _ { 0 . 1 4 } }$ </td></tr><tr><td>128</td><td> $9 7 . 5 6 _ { 0 . 3 8 }$ </td><td>98.29</td><td> $9 8 . 1 7 _ { 0 . 3 0 }$ </td><td> $9 8 . 1 4 _ { 0 . 3 6 }$ </td><td> $9 5 . 4 3 _ { 1 . 1 5 }$ </td><td> $9 7 . 5 7 _ { 1 . 0 9 }$ </td><td>95.10</td><td>88.87</td><td> $\mathbf { 9 8 . 6 3 _ { 0 . 3 2 } }$ </td></tr><tr><td>512</td><td> $9 8 . 8 5 _ { 0 . 4 0 }$ </td><td>98.31</td><td> $9 8 . 9 3 _ { 0 . 2 1 }$ </td><td> $9 8 . 9 9 _ { 0 . 2 0 }$ </td><td> $9 7 . 9 8 _ { 0 . 4 7 }$ </td><td> $9 8 . 5 8 _ { 1 . 1 8 }$ </td><td>95.13</td><td>90.62</td><td> $\mathbf { 9 9 . 1 5 _ { 0 . 1 1 } }$ </td></tr><tr><td>32</td><td> $9 3 . 3 4 _ { 1 . 5 2 }$ </td><td>97.30</td><td> $9 5 . 3 2 _ { 1 . 8 2 }$ </td><td> $9 5 . 4 5 _ { 1 . 7 2 }$ </td><td> $7 6 . 4 7 _ { 8 . 7 7 }$ </td><td> $9 2 . 0 7 _ { 1 . 5 6 }$ </td><td>94.29</td><td>88.39</td><td> $9 7 . 1 9 _ { 0 . 1 6 }$ </td></tr><tr><td>64</td><td> $9 6 . 9 7 _ { 0 . 7 4 }$ </td><td>98.12</td><td> $9 7 . 8 1 _ { 0 . 4 1 }$ </td><td> $9 7 . 8 1 _ { 0 . 3 2 }$ </td><td> $9 1 . 6 7 _ { 2 . 3 4 }$ </td><td> $9 5 . 5 0 _ { 1 . 1 9 }$ </td><td>94.95</td><td>89.92</td><td> $\mathbf { 9 8 . 5 9 _ { 0 . 1 4 } }$ </td></tr><tr><td>128</td><td> $9 7 . 5 6 _ { 0 . 3 8 }$ </td><td>98.29</td><td> $9 8 . 1 7 _ { 0 . 3 0 }$ </td><td> $9 8 . 1 4 _ { 0 . 3 6 }$ </td><td> $9 5 . 4 3 _ { 1 . 1 5 }$ </td><td> $9 7 . 5 9 _ { 1 . 0 5 }$ </td><td>95.01</td><td>89.92</td><td> $\mathbf { 9 8 . 6 3 _ { 0 . 3 2 } }$ </td></tr><tr><td>512</td><td> $9 8 . 8 5 _ { 0 . 4 0 }$ </td><td>98.31</td><td> $9 8 . 9 3 _ { 0 . 2 1 }$ </td><td> $9 8 . 9 9 _ { 0 . 2 0 }$ </td><td> $9 7 . 9 8 _ { 0 . 4 7 }$ </td><td> $9 8 . 5 9 _ { 1 . 1 6 }$ </td><td>95.05</td><td>91.06</td><td> $\mathbf { 9 9 . 1 5 _ { 0 . 1 1 } }$ </td></tr></table>

Table 1: Comparison of PECOLA to baseline methods in few-shot MGT detection. The results are average values of 10 runs with different random seeds. The subscript means the standard deviation $( e . g . , 9 9 . 1 5 _ { 0 . 1 1 }$ means 99.15 ± 0.11). Zero-shot model-based methods’ results are deterministic, so we do not report standard deviation. Also, these methods must have the white-box generator as the base model, which is different from the black-box settings of other model-based methods. Asterisk (\*) denotes the latest SOTA method. And we also conduct a more in-depth test on the entire training set in Appendix C.3.

with the increase in the number of training samples, which causes them to outperform on the fewest shot settings initially but soon be surpassed. As for the deception of generators, Grover appears to be the hardest to detect, while other models are relatively “honest” to detectors. It might have originated from the adversarial training strategy of Grover, while the bulit-in detector module adversarially shifts the LLM’s detectable features. More interestingly, advanced language models show a weaker ability to cheat detectors. Most detectors achieve around 98% in accuracy on the GPT-3.5 and HC3 datasets, which is consistent with the conclusion from Liu et al. (2023); Chen et al. (2023). We hypothesize

Moreover, the unsupervised learning methods tend to show better performance in extremely few shot scenarios. Unsurprisingly, unsupervised methods do not see a notable performance improvement that the easy-to-detect nature may originate from the lack of semantics diversity in GPT-3.5 and Chat-GPT as they use RLHF (Kirk et al., 2023).

## 4.4 Ablation Study

To illustrate the effectiveness of the PECOLA components, we do the ablation experiments on the Selective Strategy Perturbation stage and the Contrastive Learning stage on the 64-example GPT-2 dataset. We also demonstrate the Scalability of PECOLA in Appendix C.1.

<table><tr><td>Method</td><td>Acc</td><td>F1</td></tr><tr><td>w/o. mask</td><td> $7 8 . 0 0 _ { 1 . 4 0 }$ </td><td> $7 7 . 9 3 _ { 1 . 4 3 }$ </td></tr><tr><td>w/o. mask-fill</td><td> $7 7 . 7 8 _ { 1 . 8 2 }$ </td><td> $7 7 . 7 2 _ { 1 . 8 3 }$ </td></tr><tr><td>w/o. mask.CLw</td><td> $7 5 . 8 0 _ { 2 . 2 2 }$ </td><td> $7 5 . 2 3 _ { 2 . 4 6 }$ </td></tr><tr><td>w/o. mask-fill. CLw</td><td> $7 5 . 5 6 _ { 1 . 4 7 }$ </td><td> $7 5 . 1 0 _ { 1 . 7 3 }$ </td></tr><tr><td>w/o.  $\mathrm { C L } _ { w }$ </td><td> $7 6 . 6 0 _ { 1 . 6 9 }$ </td><td> $7 6 . 2 2 _ { 1 . 6 5 }$ </td></tr><tr><td>w/o. w</td><td> $7 8 . 0 2 _ { 1 . 5 6 }$ </td><td> $7 7 . 9 3 _ { 1 . 5 7 }$ </td></tr><tr><td>PECOLA</td><td> $7 8 . 9 2 _ { 1 . 1 4 }$ </td><td> $7 8 . 8 8 _ { 1 . 1 7 }$ </td></tr></table>

Table 2: Ablation study result of PECOLA.

Ablation on Selective Strategy Perturbation. In PECOLA, the data used for training primarily includes original texts, selected mask texts, and maskfilled texts. We remove each part of the data in training, i.e., (i) w/o. mask, refers to not using selected mask texts for training; (ii) w/o. mask-fill, not using mask-filling texts for training.

Ablation on Contrastive Learning. It primarily investigates the impact of CE and contrastive loss. (i) w/o. $\mathbf { C L } _ { w }$ refers to the model ablating weighted contrastive learning; (ii) w/o. w refers to the model including contrastive learning but ablating weight.

As demonstrated in Table 2, in scenarios employing only the CE loss, the Selective Strategy Perturbation method contributes to significant performance improvement. Moreover, the introduction of weighting further enhances accuracy when compared to the direct use of margin loss. It reveals the validation of bridging the metric-based and modelbased detectors, i.e., employing the Selective Strategy Perturbation method to evaluate the token importance for the multi-pairwise contrastive learning method. Furthermore, within the overall framework, the removal of the select mask text results in a more rapid decrease in accuracy compared to the removal of the mask-filling text. This finding substantiates that the Token-Level Weighted Multi-Pairwise Contrastive Learning method can better focus on the alterations in the rephrased space following the application of Selective Strategy Perturbation to the text.

## 4.5 Discussion and Analysis

## 4.5.1 Model Qualities

We analyze the model qualities, including robustness and affinity in this section. Here, we test on the 10,000-example GPT-2 test dataset, and the perturbation scale is set to 15%.

Analysis on Robustness. To validate the robustness of PECOLA in the few-shot learning settings, we apply four post hoc perturbation operations for each token in the test dataset randomly, i.e., deletion, replacement, insertion, and repetition. As indicated in Table 3, for each perturbation method employed, our decline rate is consistently lower compared to the baseline RoBERTa. On average, PECOLA maintains a 5.66% higher accuracy and an 8.77% superior F1-score. Specifically, in the deletion method, where we introduce a 15% random perturbation, it is noteworthy that the accuracy of PECOLA decreases merely 1.64%, underscoring its remarkable robustness.

<table><tr><td rowspan="2">Model Metric</td><td colspan="2">RoBERTa</td><td colspan="2">PECOLA</td></tr><tr><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td></tr><tr><td>Original</td><td> $7 4 . 4 1 _ { 2 . 4 7 }$ </td><td> $7 3 . 9 1 _ { 2 . 6 9 }$ </td><td> $7 8 . 9 2 _ { 1 . 1 4 }$ </td><td> $7 8 . 8 8 _ { 1 . 1 7 }$ </td></tr><tr><td>Delete</td><td> $7 1 . 7 7 _ { 5 . 8 8 } ( - 2 . 6 4 0 )$ </td><td> $7 0 . 4 2 _ { 8 . 0 5 } ( - 3 . 4 9 0 )$ </td><td> $7 7 . 2 8 _ { 1 . 7 9 } ( - 1 . 6 4 0 )$ </td><td> $7 7 . 0 6 _ { 2 . 0 3 } ( - 1 . 8 2 0 )$ </td></tr><tr><td>Repeat</td><td>64.696.63(-9.720)</td><td> $6 1 . 7 4 9 . 2 0 ( - 1 2 . 1 7 )$ </td><td> $6 9 . 7 4 _ { 4 . 8 3 } ( - 9 . 1 8 0 )$ </td><td> $6 7 . 8 7 _ { 6 . 2 4 } ( -$  11.01)</td></tr><tr><td>Insert</td><td>50.750.67(-23.66)</td><td> $3 6 . 4 4 _ { 1 . 6 0 } ( - 3 7 . 4 7 )$ </td><td> $5 7 . 6 1 _ { 2 . 5 2 } ( - 2 1 . 3 1 )$ </td><td>49.294.57(-29.59)</td></tr><tr><td>Replace</td><td> $5 2 . 0 4 _ { 1 . 5 8 } ( - 2 2 . 3 7 )$ </td><td> $3 9 . 4 8 _ { 3 . 5 9 } ( - 3 4 . 4 3 )$ </td><td> $5 7 . 2 5 _ { 2 . 2 1 } ( - 2 1 . 6 7 )$ </td><td> $4 8 . 8 9 _ { 3 . 9 3 } ( - 2 9 . 9 9 )$ </td></tr><tr><td>Average</td><td>59.81 (-14.60)</td><td> $5 2 . 0 2 \ : ( - 2 1 . 8 9 )$ </td><td> $6 5 . 4 7 \ ( - 1 3 . 4 5 )$ </td><td> $6 0 . 7 8 \ : ( - 1 8 . 1 0 )$ </td></tr></table>

Table 3: Model robustness to four perturbations.

Analysis on Affinity. Affinity pertains to alterations in data distribution resulting from perturbations, quantified by observing the fluctuations in accuracy. We demonstrate the superiority of the selective masking method over the random masking method using the Affinity metric, following the setting of DetectGPT. We applied a 15% mask proportion with a span of 2 tokens on the test dataset and simultaneously employed T5-Large (Raffel et al., 2020) as the mask-filling model. We trained RoBERTa-base and PECOLA on the 64- example GPT2 dataset. As shown in Table 4, in comparison to the random masking perturbation method utilized in DetectGPT, we observe a 1.92% and 0.49% increase in Affinity when employing the selective masking method. Additionally, the mask-filling method yields affinity improvements

of 3.38% and 1.32% for RoBERTa and PECOLA models, respectively. These results illustrate that the Selective Multi-Strategy Perturbation method effectively preserves more distinguishable features between MGTs and HWTs.
<table><tr><td>Model</td><td>RoBERTa</td><td>PECOLA</td></tr><tr><td>Random Mask DetectGPT</td><td>-2.64</td><td>-1.64</td></tr><tr><td>Selective Mask PEcoLA</td><td>-0.72</td><td>-1.15</td></tr><tr><td>Mask-Filling DetectGPT</td><td>-4.72</td><td>-2.66</td></tr><tr><td>Mask-Filling PECoLA</td><td>-1.34</td><td>-1.34</td></tr></table>

Table 4: Affinity of DetectGPT’s and PECOLA’s masking strategy on RoBERTa and PECOLA.

Analysis on Diversity Conversely, diversity assesses the range and variability of perturbed data, utilizing metrics Dist-1 and Dist-2 (Celikyilmaz et al., 2020). Here, we use three common perturbation methods to demonstrate the importance of not arbitrarily changing important tokens and the significance of select masks. (1) Token Substitution (TS, Zhang et al. 2015), replaces tokens with synonyms from WordNet (Miller, 1992); (2) SwitchOut (SO, Wang et al. 2018), uniformly samples and randomly substitutes from the vocabulary of test samples; and (3) Two-stage (TWs, Wei et al. 2021) trains the mask-filling model on the original data.

The ideal perturbation result is to have high Affinity scores while ensuring high Diversity scores (Celikyilmaz et al., 2020). As shown in Table 5, through Selective Strategy Perturbation, models achieve better diversity with high distribution shifts. And the overall improvement in Affinity by over 18% also shows greater diversity than the original data. The above results demonstrate the superiority of our perturbation method.

<table><tr><td>Method</td><td>Affinity</td><td>Dist-1</td><td>Dist-2</td></tr><tr><td>TS</td><td>-20.00</td><td>3.38</td><td>43.43</td></tr><tr><td>TO</td><td>-22.06</td><td>6.81</td><td>53.61</td></tr><tr><td>TWs</td><td>-21.13</td><td>3.24</td><td>41.85</td></tr><tr><td>Original</td><td>-</td><td>8.70</td><td>50.32</td></tr><tr><td>PECOLA</td><td>-1.34</td><td>15.59</td><td>57.01</td></tr></table>

Table 5: Affinity and Diversity on GPT-2 datasets.

## 4.5.2 Analysis on Selective Strategies

In this section, we compare various strategies for selection in PECOLA. Beyond the PECOLA’s importance-based perturbation method and random perturbation method (DetectGPT), we experiment with two other perturbation strategies: rank-based perturbation and keyword-based perturbation. In rank-based perturbation, we use the rescaled rank of next-token probability on $\mathrm { \bf G P T 2 } \cdot$ -medium as the weight for perturbation position selection. In keyword-based perturbation, we prevent changes in the keywords extracted by the VLT-5 model (P˛ezik et al., 2022) during perturbation. As shown in Table 6, the experimental results of selective perturbation outperform the random perturbation method by 1.20%, 2.04%, and 2.49% in average accuracy on the 64-example GPT2 dataset. And the importancebased strategy is the highest.

<table><tr><td>Method</td><td></td><td>Random Prob. Rank Keyword Importance</td><td></td><td></td></tr><tr><td>Yake</td><td> $7 6 . 0 5 _ { 1 . 8 3 }$ </td><td> $7 7 . 3 5 _ { 0 . 7 3 }$ </td><td> $7 8 . 5 5 _ { 1 . 6 5 }$ </td><td> $7 8 . 9 2 _ { 1 . 1 4 }$ </td></tr><tr><td>Perplexity</td><td> $7 5 . 5 3 _ { 1 . 1 4 }$ </td><td> $7 6 . 6 3 _ { 1 . 0 3 }$ </td><td> $7 7 . 1 1 _ { 1 . 8 0 }$ </td><td> $7 7 . 6 3 _ { 1 . 3 0 }$ </td></tr></table>

Table 6: Different strategies for perturbation and tokenlevel weighting, namely Random (DetectGPT), Prob. Rank (GPT2-medium), Keyword (VLT-5), Importance (PECOLA).

Further, we test the mask-filling failure ratio across the above strategies to interpret our outperformance. We find that the random strategy leads to more masking-filling failures than selective ones, which cause execution errors. Results in Table 7 indicate that selective strategy based on token importance performs the best, decreasing the failure ratio by 3.64% than random.

<table><tr><td>Method</td><td>Random</td><td>Prob. Rank</td><td>Keyword</td><td>Importance</td></tr><tr><td>Ratio (%)</td><td>9.20</td><td>7.83</td><td>7.80</td><td>5.56</td></tr></table>

Table 7: Mask-filling failure ratio of different perturbation strategies.

## 4.5.3 Generalization on Mask-Filling Models

We study the influence of various mask-filling models on the performance of PECOLA, including Bert (110M; Devlin et al. 2019), Bart (139M; Mike et al. 2020), GPT-2 (380M; Radford et al. 2019), Twhin-bert (279M; Zhang et al. 2023), XLM (279M; Alexis et al. 2020), XLNet (110M; Yang et al. 2019), RoBERTa (125M; Liu et al. 2019), and LLaMA-2 (7B; Touvron et al. 2023). As depicted in Fig. 3, the results of all mask-filling models surpass the baseline in terms of accuracy. Furthermore, the fluctuation of PECOLA’s performance across different mask-filling models is relatively slight. It confirms that PECOLA is not reliant on a specific filling model, showing great generalization capability. The remaining full experimental results of different mask-filling models are in Appendix C.2.

![](images/388ae778504ea93c872f43c23ded6be24765390d5a3dad9d45bf25eeb9e68997.jpg)  
Figure 3: Result of generalizing on various mask-filling models.

## 4.5.4 Generalization on Data

Cross-domain. We evaluate PECOLA on the HC3 dataset crossing three QA domains, namely Medicine, Finance, and Computer Science. The meta-information details are in Appendix A.2. For the three domains of data, we use one of them as training data (64-shot), and the remaining domains of data as testing data. The results in Table 8 show that PECOLA is more effective than the best baseline and SOTA method on average. For example, compared to Roberta, PECOLA outperforms by 4.61% in three domains on average. And PECOLA maintains a 1.63% higher accuracy on average than SOTA DetectGPT.

<table><tr><td>Domain</td><td>Medicine</td><td>Finance</td><td>Comp. Sci.</td><td>Average</td></tr><tr><td>RoBERTa</td><td> $6 2 . 9 7 _ { 4 . 0 9 }$ </td><td> $8 6 . 0 8 _ { 3 . 6 3 }$ </td><td> $9 0 . 6 4 \varsigma _ { . 0 7 }$ </td><td>79.90</td></tr><tr><td>DetectGPT</td><td>80.48</td><td> $8 5 . 1 7$ </td><td>82.98</td><td>82.88</td></tr><tr><td>PECOLA</td><td> $7 0 . 8 6 _ { 7 . 8 3 }$ </td><td> $\mathbf { 8 9 . 3 4 } _ { 2 . 9 3 }$ </td><td> $\mathbf { 9 3 . 3 2 _ { 3 . 6 4 } }$ </td><td>84.51</td></tr></table>

Table 8: Results of cross-domain in terms of accuracy.

Cross-generator. We generalize PECOLA between News articles (GPT3.5 dataset) and QA answers (HC3 dataset) on the 64-shot settings. As shown in Table 9, when the GPT-3.5 dataset is the training set, PECOLA outperforms by 10.21%; and when the HC3 dataset is the training set, PECOLA outperforms by 6.98% to the best competitor.

<table><tr><td>Dataset</td><td>GPT3.5→HC3</td><td>HC3→GPT3.5</td><td>Average</td></tr><tr><td>RoBERTa</td><td> $6 4 . 6 0 _ { 1 . 9 6 }$ </td><td> $6 2 . 6 7 _ { 2 . 4 1 }$ </td><td>63.64</td></tr><tr><td>DetectGPT</td><td>77.11</td><td>72.66</td><td>74.89</td></tr><tr><td>PECOLA</td><td> $\mathbf { 7 8 . 7 9 _ { 8 . 1 9 } }$ </td><td> $7 2 . 8 7 _ { 6 . 0 6 }$ </td><td>75.83</td></tr></table>

Table 9: Results of cross-generator in terms of accuracy.

## 4.5.5 Detecting Shorter Texts

To examine the efficiency of PECOLA to detect the short MGTs, we chunk the samples of GPT-2 and HC3 datasets into segments of 50, 100, and 200 tokens. As shown in Fig. 4, PECOLA consistently outperforms RoBERTa, with an average accuracy outperformance of 4.16% and 2.13% on the GPT-2 and HC3 datasets. And the relative performance decrease of PECOLA while the length shrinking is much less than RoBERTa.

![](images/2f649983c3643ee701f1241e1095a0f27e92f0766be75e52b8ddbe8247b27ced.jpg)  
Figure 4: Performance of PECOLA and RoBERTa to detect shorter texts. The average token number of the original GPT-2 and HC3 datasets are 445 and 260.

## 5 Conclusion

In this paper, we introduce PECOLA, a novel machine-generated text detection method that effectively bridges and integrates metric-based and fine-tuned detectors for MGT detection. To relieve the information loss caused by the random masking used in DetectGPT, we present a tokenlevel selective strategy perturbation method. To better distinguish meaningful recombination spaces and reduce reliance on the mask-filling models, we present a token-level weighted multi-pairwise contrastive learning method. In few-shot settings, experimental results show that PECOLA significantly enhances the performance of PLMs in MGT detection. Subsequent analytical experiments validate PECOLA’s effectiveness, robustness, generalization, and capability in detecting short texts.

## Acknowledgements

We thank all the anonymous reviewers and the area chair for their helpful feedback, which aided us in greatly improving the paper. This work is supported by National Key R&D Program (2023YFB3107400), National Natural Science Foundation of China (62272371, 62103323, U21B2018, 62161160337, U20B2049), Initiative Postdocs Supporting Program (BX20190275, BX20200270), China Postdoctoral Science Foundation (2019M663723, 2021M692565), Fundamental Research Funds for the Central Universities under grant (xzy012024144), and Shaanxi Province Key Industry Innovation Program (2021ZDLGY01-02).

## Limitations

In this work, we focus on MGT detection in fewshot learning settings. The next phase will involve a more comprehensive performance comparison based on full datasets. Secondly, our method mentions the score threshold, if the threshold is too high or too low, it will not serve the purpose of perturbation. How to automate and flexibly design a strict threshold is also a direction for our next phase of improvement. Thirdly, for short texts, our perturbation method faces similar limitations, as it is difficult to extract the most relevant keywords. Thus, perturbation introduces more uncontrollable noise, which poses a challenge for us to address in the future. Fourth, We hope that the present work can inspire future applications in fields like machine-generated images and videos, creating a universal approach to apply in the direction of machine generation.

## Ethics Statement

PECOLA aims to help users use our method to more reasonably and accurately identify MGT. Our goal is to develop a universal method applicable to other fields such as images and audio, and inspire the advancement of the stronger detector of MGTs and prevent all potential negative uses of language models. We do not wish our work to be maliciously used to counter detectors. The datasets mentioned in this paper are all public.

## References

Conneau Alexis, Khandelwal Kartikay, Goyal Naman, Chaudhary Vishrav, Wenzek Guillaume, Guzmán Francisco, Grave Edouard, Ott Myle, Zettlemoyer

Luke, and Stoyanov Veselin. 2020. Unsupervised cross-lingual representation learning at scale. In Annual Meeting ofthe Associationfor Computational Linguistics, pages 8440–8451.

Guangsheng Bao, Yanbin Zhao, Zhiyang Teng, Linyi Yang, and Yue Zhang. 2024. Fast-detectGPT: Efficient zero-shot detection of machine-generated text via conditional probability curvature. In The Twelfth International Conference on Learning Representations.

Stella Biderman, Hailey Schoelkopf, Quentin Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, Aviya Skowron, Lintang Sutawika, and Oskar van der Wal. 2023. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning.

Sid Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, et al. 2022. Gpt-neox-20b: An open-source autoregressive language model. arXiv preprint arXiv:2204.06745.

Ricardo Campos, Vítor Mangaravite, Arian Pasquali, Alípio Jorge, Célia Nunes, and Adam Jatowt. 2020. Yake! keyword extraction from single documents using multiple local features. Information Sciences, 509:257–289.

Asli Celikyilmaz, Elizabeth Clark, and Jianfeng Gao. 2020. Evaluation of text generation: A survey. Computing Research Repository, abs/2006.14799.

Jiaao Chen, Zichao Yang, and Diyi Yang. 2020. Mixtext: Linguistically-informed interpolation of hidden space for semi-supervised text classification. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 2147–2157.

Yutian Chen, Hao Kang, Vivian Zhai, Liangze Li, Rita Singh, and Bhiksha Ramakrishnan. 2023. Gptsentinel: Distinguishing human and chatgpt generated content. arXiv preprint arXiv:2305.07969.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In North American Chapter ofthe Association for Computational Linguistics.

Yingtong Dou, Guixiang Ma, Philip S Yu, and Sihong Xie. 2020. Robust spammer detection by nash reinforcement learning. In Proceedings ofthe 26th ACM SIGKDD international conference on knowledge discovery & data mining, pages 924–933.

Ethan Fetaya, Joern-Henrik Jacobsen, Will Grathwohl, and Richard Zemel. 2020. Understanding the limitations of conditional generative models. In International Conference on Learning Representations.

Jun Gao, Changlong Yu, Wei Wang, Huan Zhao, and Ruifeng Xu. 2022. Mask-then-fill: A flexible and effective data augmentation framework for event extraction. In Conference on Empirical Methods in Natural Language Processing.

Sebastian Gehrmann, Hendrik Strobelt, and Alexander M Rush. 2019. Gltr: Statistical detection and visualization of generated text. In Proceedings ofthe 57th Annual Meeting of the Association for Computational Linguistics: System Demonstrations, pages 111–116.

Beliz Gunel, Jingfei Du, Alexis Conneau, and Ves Stoyanov. 2021. Supervised contrastive learning for pretrained language model fine-tuning. In International Conference on Learning Representations.

Biyang Guo, Xin Zhang, Ziyuan Wang, Minqi Jiang, Jinran Nie, Yuxuan Ding, Jianwei Yue, and Yupeng Wu. 2023. How close is chatgpt to human experts? comparison corpus, evaluation, and detection. arXiv preprint arXiv:2301.07597.

Xiaomeng Hu, Pin-Yu Chen, and Tsung-Yi Ho. 2023. Radar: Robust ai-text detection via adversarial learning. arXiv preprint arXiv:2307.03838.

Hazel H Kim, Daecheol Woo, Seong Joon Oh, Jeong-Won Cha, and Yo-Sub Han. 2022. Alp: Data augmentation using lexicalized pcfgs for few-shot text classification. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 10894– 10902.

John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, and Tom Goldstein. 2023. A watermark for large language models. arXiv preprint arXiv:2301.10226.

Robert Kirk, Ishita Mediratta, Christoforos Nalmpantis, Jelena Luketina, Eric Hambro, Edward Grefenstette, and Roberta Raileanu. 2023. Understanding the effects of rlhf on llm generalisation and diversity. arXiv preprint arXiv:2310.06452.

Xiaoming Liu, Zhaohan Zhang, Yichen Wang, Hang Pu, Yu Lan, and Chao Shen. 2023. Coco: Coherenceenhanced machine-generated text detection under low resource with contrastive learning. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 16167–16188.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.

David Machado, Tiago Barbosa, Sebastião Pais, Bruno Martins, and Gaël Dias. 2009. Universal mobile information retrieval. In Universal Access in Human-Computer Interaction. Intelligent and Ubiquitous Interaction Environments: 5th International Conference, UAHCI 2009, Held as Part of HCI International 2009, San Diego, CA, USA, July 19-24, 2009. Proceedings, Part II 5, pages 345–354. Springer.

Chengzhi Mao, Carl Vondrick, Hao Wang, and Junfeng Yang. 2024. Raidar: generative AI detection via rewriting. In The Twelfth International Conference on Learning Representations.

Lewis Mike, Liu Yinhan, Goyal Naman, Ghazvininejad Marjan, Mohamed Abdelrahman, Levy Omer, Stoyanov Ves, and Zettlemoyer Luke. 2020. Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In Annual Meeting of the Association for Computational Linguistics, pages 7871–7880.

George A. Miller. 1992. WordNet: A lexical database for English. In Speech and Natural Language: Proceedings ofa Workshop Held at Harriman, New York, February 23-26, 1992.

Fatemehsadat Mireshghallah, Justus Mattern, Sicun Gao, Reza Shokri, and Taylor Berg-Kirkpatrick. 2023. Smaller language models are better blackbox machine-generated text detectors. arXiv preprint arXiv:2305.09859.

Eric Mitchell, Yoonho Lee, Alexander Khazatsky, Christopher D. Manning, and Chelsea Finn. 2023. Detectgpt: Zero-shot machine-generated text detection using probability curvature. ICML 2023.

OpenAI. 2019. Gpt-2 output dataset. Website.

OpenAI. 2023. Ai text classifier. Website.

Piotr P˛ezik, Agnieszka Mikołajczyk-Bareła, Adam Wawrzynski, Bartłomiej Nito´ n, and Maciej Ogrod-´ niczuk. 2022. Keyword extraction from short texts with a text-to-text transfer transformer. ACIIDS (Companion), 1716:530–542.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1:9.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21:5485–5551.

Yuhui Shi, Qiang Sheng, Juan Cao, Hao Mi, Beizhe Hu, and Danding Wang. 2024. Ten words only still help: Improving black-box ai-generated text detection via proxy-guided efficient re-sampling. CoRR, abs/2402.09199.

Eyal Shnarch, Ariel Gera, Alon Halfon, Lena Dankin, Leshem Choshen, Ranit Aharonov, and Noam Slonim. 2022. Cluster & tune: Boost cold start performance in text classification. In Annual Meeting of the Association for Computational Linguistics, page 7639–7653.

KaShun Shum, Shizhe Diao, and Tong Zhang. 2023. Automatic prompt augmentation and selection with chain-of-thought from labeled data. arXiv preprint arXiv:2302.12822.

Jihoon Tack, Sangwoo Mo, Jongheon Jeong, and Jinwoo Shin. 2020. Csi: Novelty detection via contrastive learning on distributionally shifted instances. Advances in neural information processing systems, 33:11839–11852.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Christoforos Vasilatos, Manaar Alam, Talal Rahwan, Yasir Zaki, and Michail Maniatakos. 2023. Howkgpt: Investigating the detection of chatgpt-generated university student homework through context-aware perplexity analysis. arXiv preprint arXiv:2305.18226.

Saranya Venkatraman, Adaku Uchendu, and Dongwon Lee. 2023. Gpt-who: An information density-based machine-generated text detector. arXiv preprint arXiv:2310.06202.

Rakesh Verma and Daniel Lee. 2017. Extractive summarization: Limits, compression, generalized model and heuristics. Computación y Sistemas, 21:787–798.

Vivek Verma, Eve Fleisig, Nicholas Tomlin, and Dan Klein. 2023. Ghostbuster: Detecting text ghostwritten by large language models. arXiv preprint arXiv:2305.15047.

Jan Philip Wahle, Terry Ruas, Frederic Kirstein, and Bela Gipp. 2022. How large language models are transforming machine-paraphrase plagiarism. In Conference on Empirical Methods in Natural Language Processing, page 952–963.

Ben Wang. 2021. Mesh-Transformer-JAX: Model-Parallel Implementation of Transformer Language Model with JAX. https://github.com/ kingoflolz/mesh-transformer-jax.

Pengyu Wang, Linyang Li, Ke Ren, Botian Jiang, Dong Zhang, and Xipeng Qiu. 2023. SeqXGPT: Sentencelevel AI-generated text detection. In The 2023 Conference on Empirical Methods in Natural Language Processing.

Sheng Wang, Jinjiao Lian, Yuzhong Peng, Baoqing Hu, and Hongsong Chen. 2019. Generalized reference evapotranspiration models with limited climatic data based on random forest and gene expression programming in guangxi, china. Agricultural Water Management, 221:220–230.

Xinyi Wang, Hieu Pham, Zihang Dai, and Graham Neubig. 2018. Switchout: an efficient data augmentation algorithm for neural machine translation. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 856–861.

Yichen Wang, Shangbin Feng, Abe Bohan Hou, Xiao Pu, Chao Shen, Xiaoming Liu, Yulia Tsvetkov, and Tianxing He. 2024. Stumbling blocks: Stress testing the robustness of machine-generated text detectors under attacks. arXiv preprint arXiv:2402.11638.

Jason Wei, Chengyu Huang, Soroush Vosoughi, Yu Cheng, and Shiqi Xu. 2021. Few-shot text classification with triplet networks, data augmentation, and curriculum learning. arXiv preprint arXiv:2103.07552.

Jason Wei and Kai Zou. 2019. Eda: Easy data augmentation techniques for boosting performance on text classification tasks. In Conference on Empirical Methods in Natural Language Processing, pages 6381–6387.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al. 2020. Transformers: State-of-the-art natural language processing. In Proceedings ofthe 2020 conference on empirical methods in natural language processing: system demonstrations, pages 38–45.

Junchao Wu, Shu Yang, Runzhe Zhan, Yulin Yuan, Derek F Wong, and Lidia S Chao. 2023a. A survey on llm-gernerated text detection: Necessity, methods, and future directions. arXiv preprint arXiv:2310.14724.

Kangxi Wu, Liang Pang, Huawei Shen, Xueqi Cheng, and Tat-Seng Chua. 2023b. LLMDet: A third party large language models generated text detection tool. In The 2023 Conference on Empirical Methods in Natural Language Processing.

Qizhe Xie, Zihang Dai, Eduard Hovy, Thang Luong, and Quoc Le. 2020. Unsupervised data augmentation for consistency training. Advances in neural information processing systems, 33:6256–6268.

Xianjun Yang, Wei Cheng, Linda Petzold, William Yang Wang, and Haifeng Chen. 2023. Dna-gpt: Divergent n-gram analysis for training-free detection of gptgenerated text. arXiv preprint arXiv:2305.17359.

Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R Salakhutdinov, and Quoc V Le. 2019. Xlnet: Generalized autoregressive pretraining for language understanding. Advances in neural information processing systems, 32.

Rowan Zellers, Ari Holtzman, Hannah Rashkin, Yonatan Bisk, Ali Farhadi, Franziska Roesner, and Yejin Choi. 2019. Defending against neural fake news. Advances in neural information processing systems, 32.

Xiang Zhang, Junbo Zhao, and Yann LeCun. 2015. Character-level convolutional networks for text classification. Advances in neural information processing systems, 28.

Xinyang Zhang, Yury Malkov, Omar Florez, Serim Park, Brian McWilliams, Jiawei Han, and Ahmed El-Kishky. 2023. Twhin-bert: A socially-enriched pre-trained language model for multilingual tweet representations at twitter. In Proceedings ofthe 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 5597–5607.

Wenxuan Zhou, Fangyu Liu, and Muhao Chen. 2021. Contrastive out-of-distribution detection for pretrained transformers. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 1100–1111.

## A Implementation Details

This part mentions the hyperparameter settings and meta-information of the HC3 dataset.

## A.1 Hyperparameter Details

Experiments evaluating competitors and PECOLA follow the setting of CoCo (Liu et al., 2023). The hyperparameter settings of all the methods in the experiment as shown in Table 10. We randomly select 10 different seeds for experiments, and report average test accuracy and F1-score.

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Training Epochs</td><td>30</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td>1e-5</td></tr><tr><td>Weight Decay</td><td>0.01</td></tr><tr><td>Batch Size</td><td>16</td></tr><tr><td>Mask Gap</td><td>2</td></tr><tr><td>Mask Proportion</td><td>10%</td></tr><tr><td>Score threshold</td><td>0.4</td></tr><tr><td>Pre-trained model</td><td>RoBERTa-base</td></tr></table>

Table 10: Implementation details of hyperparameters.

## A.2 Dataset Meta Information

We evaluate PECOLA effectiveness from domains and generators on the HC3 dataset, which primarily includes Medicine, Finance, and Computer Science domain QA, as shown in Table 11.

<table><tr><td>Domain</td><td>Medicine</td><td>Finance</td><td>Comp. Sci.</td></tr><tr><td>Size</td><td>2585</td><td>8436</td><td>1684</td></tr></table>

Table 11: Meta-information of the HC3 dataset.

## B Effect of Hyperparameters

In PECOLA, the primary hyperparameters include the mask proportion, mask gap of perturbation, and score threshold. The perturbation proportion refers to the mask rate in the texts. The perturbation mask gap ensures that several tokens following a masked token remain unmasked, and score threshold to control the number of Most Relevant Keywords.

## B.1 Perturbation Proportion and Mask Gap

We evaluate the impact of different perturbation ratios and mask gap on accuracy, and perform a minor scan in a few-shot learning settings with a set of mask proportions {5, 8, 10, 15, 17, 20} and mask gap {0, 1, 2, 3, 4, 5}, average the results for each combination of parameters. And a mask gap of 2 and a perturbation ratio of 10% achieve the maximum average values. As shown in Fig. 5, it is found that the combination of a mask gap of 2 and a mask proportion of 10% yielded the best results, on the 64-example GPT-2 dataset.

![](images/c34a8b2b3a964df9494ba391dea72e0bd52b02b499cf32cbd66d1b6c74172203.jpg)  
Figure 5: Impact of varying the number of perturbations and mask gap in PECOLA, we use T5-large (Raffel et al., 2020) as the mask-filling model. For each combination, we conduct tests on ten randomly select seeds.

## B.2 Score Threshold

In the main experiment, all datasets use a common score threshold of 0.4, and it may not be the best choice for different datasets, because with the change in data type and text length, the gold keywords often vary. Therefore, as shown in Fig. 6, we discuss the performance changes of four datasets with different score threshold in few-shot learning settings. An excessively high score threshold results in too many most relevant keywords, failing to effectively perturb the data, hence not significantly improving accuracy. Similarly, a too low score threshold can lead to more random perturbations. Therefore, the selection of the score threshold should be stringent.

![](images/8526ee7fe37ec98016753328eff3e30d0a3e1bca8e5991904c4e4228c6b61ae8.jpg)

![](images/92edce41ab15c95dea2d54a2cb5e9a69b536d1af513e82d53e5df4f3ed19713f.jpg)

![](images/61863f6090323c6f4294c25cba8b10bde916ce74667d491edfb95b76522587b0.jpg)

![](images/5b44812e8cd8a09e0e64c4344e2e4557205acf45b17fdc601ba6237043a957ed.jpg)  
Figure 6: Effect of score threshold on model performance. In the GPT3.5 and HC3 datasets (sub-figure 3 and 4), accuracy and F1-score coincide.

## C Efficiency of PECOLA

## C.1 Scalability of Base Models at Different Scales

We adopt Pythia (Biderman et al., 2023) as the base model of PECOLA with different scales, i.e., 70M, 160M, 410M, 1B, and 1.4B. We train and do experiments on one NVIDIA A100 GPU, and the performance and time consumption are in Table 12. With the increase in model size, both accuracy and F1-score show upward trends, while the time increase is linear, which is reasonable.

## C.2 Impact of the Chosen Mask-filling Models

datasets, and 7,000 samples from GPT-3.5 as our training sets. Comparatively, PECOLA outperforms the second-best results in accuracy and F1-score by 0.13% and 1.56%, 0.80% and 0.83%, 0.05% and 0.05%, 0.03% and 0.03% respectively, across four datasets.

This section shows the full experimental results of different mask-filling models, as shown in Table 13, the experimental results confirm the same outcomes as in the few-shot learning settings, where the T5 filling model does not perform the best across all datasets. All the above models are obtained from huggingface transformers (Wolf et al., 2020). And we do not intervene in the temperature sampling of the mask-filling model, setting it all to 1.

## C.3 Further Experiments on Full Datasets

To demonstrate Pecola’s superiority over the whole training set, we conduct a more in-depth test, as shown in Table 14. We train the detector using 10,000 samples from the Grover, GPT-2, and HC3

<table><tr><td>Model</td><td>70M</td><td>160M</td><td>410M</td><td>1B</td><td>1.4B</td></tr><tr><td>Acc</td><td> $5 8 . 4 2 _ { 0 . 7 0 }$ </td><td> $6 3 . 6 6 _ { 0 . 1 7 }$ </td><td> $7 1 . 0 7 _ { 1 . 6 3 }$ </td><td> $7 2 . 1 3 _ { 1 . 6 3 }$ </td><td> $7 4 . 0 5 _ { 1 . 7 7 }$ </td></tr><tr><td>F1</td><td> $5 8 . 0 3 _ { 0 . 7 9 }$ </td><td> $6 3 . 5 4 _ { 0 . 2 8 }$ </td><td> $7 0 . 8 7 _ { \ 1 . 9 2 }$ </td><td> $7 1 . 7 5 _ { 2 . 6 7 }$ </td><td> $7 3 . 8 5 _ { 1 . 5 5 }$ </td></tr><tr><td>Per epoch</td><td>16s</td><td>34s</td><td>85s</td><td>97s</td><td>113s</td></tr><tr><td>Single data</td><td>2.2ms</td><td>7.0ms</td><td>13.8ms</td><td>14.1ms</td><td>16.6ms</td></tr></table>

Table 12: Results of fine-tuning PECOLA with Pythia models of various scales, on the 64-example GPT2 dataset. We also demonstrate the training time per epoch and the single data test time.

<table><tr><td>Dataset</td><td>Method</td><td>Shot</td><td>BART</td><td>Bert</td><td>GPT-2</td><td>Twhin Bert</td><td>XLM</td><td>XLNet</td><td>RoBERTa</td><td>T5</td></tr><tr><td rowspan="5">Grover</td><td rowspan="2">Acc</td><td>128</td><td> $6 2 . 0 4 _ { 2 . 5 1 }$ </td><td> $6 1 . 5 5 _ { 1 . 7 4 }$ </td><td> $6 2 . 8 2 _ { 1 . 2 4 }$ </td><td> $6 1 . 0 0 _ { 2 . 2 0 }$ </td><td> $6 1 . 8 2 _ { 0 . 8 2 }$ </td><td> $6 0 . 1 6 _ { 0 . 4 3 }$ </td><td> $6 3 . 1 0 _ { 1 . 7 6 }$ </td><td> $\mathbf { 6 3 . 6 0 _ { 1 . 7 1 } }$ </td></tr><tr><td>512</td><td> $7 2 . 2 4 _ { 1 . 5 4 }$ </td><td> $7 1 . 6 7 _ { 1 . 0 4 }$ </td><td> $7 2 . 6 2 _ { 1 . 1 2 }$ </td><td> $7 2 . 7 8 _ { 1 . 1 4 }$ </td><td> $7 2 . 1 3 _ { 0 . 6 4 }$ </td><td> $7 2 . 7 2 _ { 1 . 0 3 }$ </td><td> $7 3 . 2 5 _ { 0 . 8 4 }$ </td><td> $7 3 . 1 2 _ { 0 . 8 4 }$ </td></tr><tr><td rowspan="2"> $F l$ </td><td>128</td><td> $5 7 . 8 0 _ { 1 . 2 8 }$ </td><td> $5 7 . 6 0 _ { 1 . 9 3 }$ </td><td> $5 8 . 5 5 _ { 0 . 8 0 }$ </td><td> $5 6 . 7 4 _ { 0 . 4 8 }$ </td><td> $5 7 . 6 0 _ { 0 . 9 2 }$ </td><td> $5 6 . 6 2 _ { 0 . 6 4 }$ </td><td> $5 8 . 2 9 _ { 1 . 1 2 }$ </td><td> $\mathbf { 5 8 . 9 8 _ { 1 . 5 8 } }$ </td></tr><tr><td>512</td><td> $6 6 . 2 5 _ { 2 . 3 4 }$ </td><td> $6 5 . 5 6 _ { 1 . 7 6 }$ </td><td> $6 6 . 7 2 _ { 2 . 0 0 }$ </td><td> $6 8 . 4 9 _ { 1 . 0 4 }$ </td><td> $6 6 . 3 8 _ { 2 . 2 1 }$ </td><td> $6 7 . 5 0 _ { 2 . 6 1 }$ </td><td> $6 7 . 4 9 _ { 1 . 6 8 }$ </td><td> $\mathbf { 6 8 . 2 4 _ { 1 . 6 4 } }$ </td></tr><tr><td rowspan="2">Recall</td><td>128</td><td> $5 8 . 0 3 _ { 0 . 9 9 }$ </td><td> $5 7 . 9 1 _ { 2 . 0 8 }$ </td><td> ${ 5 8 . 7 2 _ { 0 . 8 7 } }$ </td><td> $5 7 . 1 8 _ { 0 . 8 6 }$ </td><td> $5 7 . 7 8 _ { 1 . 0 4 }$ </td><td> $5 7 . 0 0 _ { 0 . 8 0 }$ </td><td> $5 8 . 3 1 _ { 0 . 9 9 }$ </td><td> $5 7 . 8 9 _ { 1 . 4 4 }$ </td></tr><tr><td>512</td><td> $6 5 . 8 5 _ { 2 . 6 6 }$ </td><td> $6 4 . 8 7 _ { 1 . 7 1 }$ </td><td> $6 6 . 0 1 _ { 2 . 0 6 }$ </td><td> ${ \bf 6 8 . 1 1 _ { 1 . 1 6 } }$ </td><td> $6 5 . 8 7 _ { 2 . 4 6 }$ </td><td> $6 7 . 0 5 _ { 3 . 0 4 }$ </td><td> $6 6 . 7 3 _ { 1 . 6 8 }$ </td><td> $6 6 . 5 1 _ { 1 . 6 4 }$ </td></tr><tr><td rowspan="5">G2</td><td rowspan="2">Acc</td><td>128</td><td> $8 2 . 1 6 _ { 1 . 0 4 }$ </td><td> $8 0 . 7 7 _ { 0 . 4 8 }$ </td><td> $8 2 . 4 2 _ { 1 . 0 5 }$ </td><td> $8 2 . 1 7 _ { 0 . 4 0 }$ </td><td> $8 1 . 1 5 _ { 0 . 3 1 }$ </td><td> $8 1 . 2 6 _ { 0 . 3 6 }$ </td><td> $8 1 . 2 7 _ { 1 . 2 0 }$ </td><td> $\mathbf { 8 2 . 5 8 _ { 0 . 4 9 } }$ </td></tr><tr><td>512</td><td> $8 5 . 4 1 _ { 0 . 6 6 }$ </td><td> $8 5 . 4 3 _ { 0 . 5 3 }$ </td><td> $8 5 . 5 2 _ { 0 . 5 7 }$ </td><td> $8 5 . 7 2 _ { 0 . 3 9 }$ </td><td> $8 5 . 1 0 _ { 0 . 2 7 }$ </td><td> $8 5 . 1 3 _ { 0 . 6 0 }$ </td><td> $8 5 . 7 5 _ { 0 . 5 5 }$ </td><td> $\mathbf { 8 5 . 7 5 _ { 0 . 6 9 } }$ </td></tr><tr><td rowspan="2">Fl</td><td>128</td><td> $8 2 . 1 2 _ { 1 . 0 7 }$ </td><td> $8 0 . 6 7 _ { 0 . 5 4 }$ </td><td> $8 2 . 3 8 _ { 1 . 0 8 }$ </td><td> $8 2 . 1 2 _ { 0 . 3 8 }$ </td><td> $8 1 . 1 1 _ { 0 . 3 4 }$ </td><td> $8 1 . 2 4 _ { 0 . 3 7 }$ </td><td> $8 1 . 1 6 _ { 1 . 2 7 }$ </td><td> $\mathbf { 8 } 2 . 5 4 _ { 0 . 5 1 }$ </td></tr><tr><td>512</td><td> $8 5 . 4 0 _ { 0 . 6 7 }$ </td><td> $8 5 . 4 1 _ { 0 . 5 3 }$ </td><td> $8 5 . 7 2 _ { 0 . 7 0 }$ </td><td> $8 5 . 7 2 _ { 0 . 3 9 }$ </td><td> $8 5 . 1 0 _ { 0 . 2 7 }$ </td><td> $8 5 . 1 3 _ { 0 . 6 0 }$ </td><td> $\mathbf { 8 5 . 7 5 _ { 0 . 5 5 } }$ </td><td> $8 5 . 7 2 _ { 0 . 7 0 }$ </td></tr><tr><td rowspan="2">Recall</td><td>128</td><td> $8 2 . 1 5 _ { 1 . 0 5 }$ </td><td> $8 0 . 7 5 _ { 0 . 4 8 }$ </td><td> $8 2 . 0 1 _ { 0 . 6 8 }$ </td><td> $8 2 . 1 7 _ { 0 . 4 0 }$ </td><td> $8 1 . 1 5 _ { 0 . 3 1 }$ </td><td> $8 1 . 2 6 _ { 0 . 3 6 }$ </td><td> $8 1 . 2 5 _ { 1 . 2 0 }$ </td><td> $\mathbf { 8 2 . 5 7 _ { 0 . 4 9 } }$ </td></tr><tr><td>512</td><td> $8 5 . 4 1 _ { 0 . 6 6 }$ </td><td> $8 5 . 4 3 _ { 0 . 5 3 }$ </td><td> $\mathbf { 8 5 . 8 0 _ { 0 . 2 7 } }$ </td><td> $8 5 . 7 2 _ { 0 . 3 9 }$ </td><td> $8 5 . 1 0 _ { 0 . 2 7 }$ </td><td> $8 5 . 1 3 _ { 0 . 6 0 }$ </td><td> $8 5 . 7 5 _ { 0 . 5 5 }$ </td><td> $8 5 . 5 2 _ { 0 . 5 7 }$ </td></tr><tr><td rowspan="5">GP-35</td><td rowspan="3">Acc</td><td>128</td><td> $9 8 . 2 4 _ { 0 . 1 6 }$ </td><td> $9 8 . 0 9 _ { 0 . 2 5 }$ </td><td> $9 8 . 0 9 _ { 0 . 1 0 }$ </td><td> $9 8 . 1 1 _ { 0 . 1 1 }$ </td><td> $9 7 . 9 8 _ { 0 . 1 4 }$ </td><td> $9 8 . 1 3 _ { 0 . 0 8 }$ </td><td> $9 8 . 0 1 _ { 0 . 1 8 }$ </td><td> $\mathbf { 9 8 . 6 3 _ { 0 . 3 2 } }$ </td></tr><tr><td>512</td><td> $9 9 . 1 9 _ { 0 . 1 3 }$ </td><td> $9 9 . 0 5 _ { 0 . 1 5 }$ </td><td> $9 9 . 1 3 _ { 0 . 1 7 }$ </td><td> $9 8 . 8 9 _ { 0 . 2 1 }$ </td><td> $9 8 . 8 8 _ { 0 . 2 1 }$ </td><td> $\mathbf { 9 9 . 2 3 0 . 2 6 }$ </td><td> $9 9 . 1 6 _ { 0 . 1 4 }$ </td><td> $9 9 . 1 5 _ { 0 . 1 1 }$ </td></tr><tr><td>128</td><td> $9 8 . 2 4 _ { 0 . 1 6 }$ </td><td> $9 8 . 0 9 _ { 0 . 2 5 }$ </td><td> $9 8 . 0 9 _ { 0 . 1 0 }$ </td><td> $9 8 . 1 1 _ { 0 . 1 1 }$ </td><td> $9 7 . 9 8 _ { 0 . 1 4 }$ </td><td> $9 8 . 1 3 _ { 0 . 0 8 }$ </td><td> $9 8 . 0 1 _ { 0 . 1 8 }$ </td><td> $\mathbf { 9 8 . 6 3 _ { 0 . 3 2 } }$ </td></tr><tr><td>Fl 512</td><td> $9 9 . 1 9 _ { 0 . 1 3 }$ </td><td> $9 9 . 0 5 _ { 0 . 1 5 }$ </td><td> $9 9 . 1 3 _ { 0 . 1 7 }$ </td><td> $9 8 . 8 9 _ { 0 . 2 1 }$ </td><td> $9 8 . 8 8 _ { 0 . 2 1 }$ </td><td> $\mathbf { 9 9 . 2 3 _ { 0 . 2 6 } }$ </td><td> $9 9 . 1 6 _ { 0 . 1 4 }$ </td><td></td><td> $9 9 . 1 5 _ { 0 . 1 1 }$ </td></tr><tr><td rowspan="2">Recall</td><td>128</td><td> $9 8 . 2 4 _ { 0 . 1 6 }$ </td><td> $9 8 . 0 9 _ { 0 . 2 5 }$ </td><td> $9 8 . 0 9 _ { 0 . 1 0 }$ </td><td> $9 8 . 1 1 _ { 0 . 1 1 }$ </td><td> $9 7 . 9 8 _ { 0 . 1 4 }$ </td><td> $9 8 . 1 3 _ { 0 . 0 8 }$ </td><td> $9 8 . 0 1 _ { 0 . 1 8 }$ </td><td> $\mathbf { 9 8 . 6 3 _ { 0 . 3 2 } }$ </td></tr><tr><td>512</td><td> $9 9 . 1 9 _ { 0 . 1 3 }$ </td><td> $9 9 . 0 5 _ { 0 . 1 5 }$ </td><td> $9 9 . 1 3 _ { 0 . 1 7 }$ </td><td> $9 8 . 8 9 _ { 0 . 2 1 }$ </td><td> $9 8 . 8 8 _ { 0 . 2 1 }$ </td><td> $\mathbf { 9 9 . 2 3 _ { 0 . 2 6 } }$ </td><td> $9 9 . 1 6 _ { 0 . 1 4 }$ </td><td> $9 9 . 1 5 _ { 0 . 1 1 }$ </td></tr><tr><td rowspan="6">HC</td><td rowspan="2"> $A c c$ </td><td>128</td><td> $9 8 . 6 3 _ { 0 . 1 8 }$ </td><td> $9 8 . 0 3 _ { 0 . 4 0 }$ </td><td> $9 8 . 5 9 _ { 0 . 1 6 }$ </td><td> $9 8 . 5 8 _ { 0 . 2 2 }$ </td><td> $9 8 . 2 4 _ { 0 . 0 9 }$ </td><td> $9 8 . 3 5 _ { 0 . 1 2 }$ </td><td> $\mathbf { 9 8 . 7 9 _ { 0 . 3 2 } }$ </td><td> $9 8 . 0 6 _ { 0 . 1 2 }$ </td></tr><tr><td>512</td><td> $9 8 . 8 2 _ { 0 . 3 5 }$ </td><td> $9 8 . 4 5 _ { 0 . 2 1 }$ </td><td> $9 8 . 9 6 _ { 0 . 2 5 }$ </td><td> $9 8 . 8 3 _ { 0 . 2 4 }$ </td><td> $9 8 . 8 0 _ { 0 . 3 8 }$ </td><td> $9 8 . 8 0 _ { 0 . 3 0 }$ </td><td> $9 9 . 0 2 _ { 0 . 2 3 }$ </td><td> $\mathbf { 9 9 . 1 4 _ { 0 . 1 5 } }$ </td></tr><tr><td rowspan="2">F1</td><td>128</td><td> $9 8 . 6 3 _ { 0 . 1 8 }$ </td><td>98.030.40</td><td> $9 8 . 5 9 _ { 0 . 1 6 }$ </td><td> $9 8 . 5 8 _ { 0 . 2 2 }$ </td><td> $9 8 . 2 4 _ { 0 . 0 9 }$ </td><td> $9 8 . 3 5 _ { 0 . 1 2 }$ </td><td> $\mathbf { 9 8 . 7 9 _ { 0 . 3 2 } }$ </td><td> $9 8 . 0 6 _ { 0 . 1 2 }$ </td></tr><tr><td>512</td><td> $9 8 . 8 2 _ { 0 . 3 5 }$ </td><td> $9 8 . 4 5 _ { 0 . 2 1 }$ </td><td> $9 8 . 9 6 _ { 0 . 2 5 }$ </td><td> $9 8 . 8 3 _ { 0 . 2 4 }$ </td><td> $9 8 . 8 0 _ { 0 . 3 8 }$ </td><td> $9 8 . 8 0 _ { 0 . 3 0 }$ </td><td> $9 9 . 0 2 _ { 0 . 2 3 }$ </td><td> $\mathbf { 9 9 . 1 4 _ { 0 . 1 5 } }$ </td></tr><tr><td rowspan="2">Reacall</td><td>128</td><td> $9 8 . 6 3 _ { 0 . 1 8 }$ </td><td> $9 8 . 0 3 _ { 0 . 4 0 }$ </td><td> $9 8 . 5 9 _ { 0 . 1 6 }$ </td><td> $9 8 . 5 8 _ { 0 . 2 2 }$ </td><td> $9 8 . 2 4 _ { 0 . 0 9 }$ </td><td> $9 8 . 3 5 _ { 0 . 1 2 }$ </td><td> $\mathbf { 9 8 . 7 9 _ { 0 . 3 2 } }$ </td><td> $9 8 . 6 3 _ { 0 . 3 2 }$ </td></tr><tr><td>512</td><td> $9 8 . 8 2 _ { 0 . 3 5 }$ </td><td> $9 8 . 4 5 _ { 0 . 2 1 }$ </td><td> $9 8 . 9 6 _ { 0 . 2 5 }$ </td><td> $9 8 . 8 3 _ { 0 . 2 4 }$ </td><td> $9 8 . 8 0 _ { 0 . 3 8 }$ </td><td> $9 8 . 8 0 _ { 0 . 3 0 }$ </td><td> $9 9 . 0 2 _ { 0 . 2 3 }$ </td><td> $\mathbf { 9 9 . 1 5 _ { 0 . 1 1 } }$ </td></tr></table>

Table 13: The full MGT detection performance of different mask-filling models on four datasets. We use the model version with the same level model size, i.e. base version for most models.

<table><tr><td>Dataset</td><td>Shot</td><td>Metric</td><td>RoBERTa</td><td>GLTR</td><td> $C E { + } S C L$ </td><td>CE+Margin</td><td>IT:Clust</td><td> $c o c o$ </td><td>DetectGPT</td><td>Fast-Detect.</td><td>PECOLA</td></tr><tr><td rowspan="2">Grover</td><td>10000</td><td>Acc</td><td> $8 6 . 1 3 _ { 0 . 4 7 }$ </td><td>60.40</td><td> $\mathbf { 8 6 . 5 7 _ { 0 . 4 4 } }$ </td><td> $8 6 . 2 5 _ { 0 . 8 1 }$ </td><td> $7 2 . 6 5 _ { 3 . 4 4 }$ </td><td> $8 5 . 2 3 _ { 0 . 2 0 }$ </td><td>61.42</td><td>65.49</td><td>86.700.37</td></tr><tr><td></td><td>F1</td><td> $8 4 . 0 7 _ { 0 . 9 1 }$ </td><td>59.82</td><td> $8 4 . 9 5 _ { 0 . 5 6 }$ </td><td> $\mathbf { 8 5 . 1 0 _ { 1 . 2 7 } }$ </td><td> $6 3 . 2 1 _ { 5 . 0 2 }$ </td><td> $8 3 . 6 7 _ { 0 . 5 6 }$ </td><td>54.28</td><td>63.29</td><td> $\mathbf { 8 6 . 6 6 _ { 0 . 3 3 } }$ </td></tr><tr><td rowspan="2">GPT-2</td><td>10000</td><td>Acc</td><td> $8 9 . 5 6 _ { 1 . 1 8 }$ </td><td>77.55</td><td> $9 0 . 1 9 _ { 0 . 6 0 }$ </td><td> $\mathbf { 9 0 . 3 0 _ { 0 . 4 1 } }$ </td><td> $8 1 . 6 5 _ { 2 . 1 4 }$ </td><td> $8 9 . 7 8 _ { 0 . 0 4 }$ </td><td>78.74</td><td>80.06</td><td> $\mathbf { 9 1 . 1 0 _ { 0 . 0 9 } }$ </td></tr><tr><td></td><td>F1</td><td> $8 9 . 5 1 _ { 1 . 1 5 }$ </td><td>76.39</td><td> $9 0 . 1 5 _ { 0 . 6 1 }$ </td><td> $\mathbf { 9 0 . 2 7 0 . 4 0 }$ </td><td> $8 1 . 5 4 _ { 3 . 2 0 }$ </td><td> $8 9 . 0 1 _ { 0 . 0 7 }$ </td><td>71.13</td><td>80.64</td><td> $\mathbf { 9 1 . 1 0 _ { 0 . 1 0 } }$ </td></tr><tr><td rowspan="2">GPT-3.5</td><td>7000</td><td>Acc</td><td> $9 9 . 8 9 _ { 0 . 0 3 }$ </td><td>93.50</td><td> $9 9 . 7 4 _ { 0 . 0 4 }$ </td><td> $\mathbf { 9 9 . 9 0 _ { 0 . 0 3 } }$ </td><td> $9 9 . 0 9 _ { 0 . 3 1 }$ </td><td> $9 9 . 4 4 _ { 0 . 1 2 }$ </td><td>90.80</td><td>94.72</td><td> $\mathbf { 9 9 . 9 5 _ { 0 . 0 1 } }$ </td></tr><tr><td></td><td> $F l$ </td><td> $9 9 . 8 9 _ { 0 . 0 3 }$ </td><td>93.58</td><td> $9 9 . 7 4 _ { 0 . 0 4 }$ </td><td> $\mathbf { 9 9 . 9 0 _ { 0 . 0 3 } }$ </td><td> $9 9 . 0 9 _ { 0 . 3 1 }$ </td><td> $9 9 . 4 4 _ { 0 . 1 2 }$ </td><td>89.14</td><td>94.76</td><td> $\mathbf { 9 9 . 9 5 _ { 0 . 0 1 } }$ </td></tr><tr><td rowspan="2">HC3</td><td>10000</td><td> $A c c$ </td><td> $9 9 . 8 4 _ { 0 . 0 8 }$ </td><td>98.39</td><td> $\mathbf { 9 9 . 8 9 _ { 0 . 0 1 } }$ </td><td> $9 9 . 8 6 _ { 0 . 0 3 }$ </td><td> $9 8 . 8 0 _ { 0 . 6 7 }$ </td><td> $9 9 . 4 6 _ { 0 . 2 4 }$ </td><td>95.13</td><td>98.32</td><td> $\mathbf { 9 9 . 9 2 _ { 0 . 0 1 } }$ </td></tr><tr><td></td><td> $F l$ </td><td> $9 9 . 8 4 _ { 0 . 0 8 }$ </td><td>98.49</td><td> $\mathbf { 9 9 . 8 9 _ { 0 . 0 1 } }$ </td><td> $9 9 . 8 6 _ { 0 . 0 3 }$ </td><td> $9 8 . 8 0 _ { 0 . 6 7 }$ </td><td> $9 9 . 4 6 _ { 0 . 2 4 }$ </td><td>95.05</td><td>98.02</td><td> $\mathbf { 9 9 . 9 2 _ { 0 . 0 1 } }$ </td></tr></table>

Table 14: Performance comparison of PECOLA to baseline methods on the full datasets. The results are average values of 5 runs with different random seeds. Bold shows the best and second-best results within each column.