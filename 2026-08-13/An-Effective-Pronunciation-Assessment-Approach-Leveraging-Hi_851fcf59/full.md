# An Effective Pronunciation Assessment Approach Leveraging Hierarchical Transformers and Pre-training Strategies

Bi-Cheng Yan<sup>1\*</sup>, Jiun-Ting Li<sup>1</sup>, Yi-Cheng Wang<sup>1</sup>, Hsin-Wei Wang<sup>1</sup>, Tien-Hong Lo<sup>1</sup>, Yung-Chang Hsu<sup>2</sup>, Wei-Cheng Chao<sup>3</sup>, Berlin Chen<sup>1\*</sup>

<sup>1</sup>National Taiwan Normal University, <sup>2</sup>EZAI <sup>3</sup>Advanced Technology Laboratory, Chunghwa Telecom Co., Ltd. {bicheng, berlin}@ntnu.edu.tw, weicheng@cht.com.tw

## Abstract

Automatic pronunciation assessment (APA) manages to quantify a second language (L2) learner's pronunciation proficiency in a target language by providing fine-grained feedback with multiple pronunciation aspect scores at various linguistic levels. Most existing efforts on APA typically parallelize the modeling process, namely predicting multiple aspect scores across various linguistic levels simultaneously. This inevitably makes both the hierarchy of linguistic units and the relatedness among the pronunciation aspects sidelined. Recognizing such a limitation, we in this paper first introduce HierTFR<sup>1</sup>, a hierarchal APA method that jointly models the intrinsic structures of an utterance while considering the relatedness among the pronunciation aspects. We also propose a correlation-aware regularizer to strengthen the connection between the estimated scores and the human annotations. Furthermore, novel pre-training strategies tailored for different linguistic levels are put forward so as to facilitate better model initialization. An extensive set of empirical experiments conducted on the speechocean762 benchmark dataset suggest the feasibility and effectiveness of our approach in relation to several competitive baselines.

## 1 Introduction

With the rising trend of globalization, more and more people are willing or being demanded to learn foreign languages. This surging need calls for developing computer-assisted pronunciation training (CAPT) systems, as they can offer tailored and informative feedback for L2 (second-language)

<table><tr><td rowspan=1 colspan=9>Reading-aloud ScenarioWe call it bear.We call it bear.</td></tr><tr><td rowspan=1 colspan=9>Automatic Pronunciation Assessment Results</td></tr><tr><td rowspan=1 colspan=2>Utterance level</td><td rowspan=1 colspan=5>Word level</td><td rowspan=1 colspan=2>Phone level</td></tr><tr><td rowspan=1 colspan=1>Aspects</td><td rowspan=1 colspan=1>Scores</td><td rowspan=1 colspan=3>Words</td><td rowspan=1 colspan=1>Aspects</td><td rowspan=1 colspan=1>Scores</td><td rowspan=1 colspan=1>Phones</td><td rowspan=1 colspan=1>Scores</td></tr><tr><td rowspan=2 colspan=1>Accuracy</td><td rowspan=2 colspan=1>1.6</td><td rowspan=2 colspan=3>We</td><td rowspan=1 colspan=1>Accuracy</td><td rowspan=1 colspan=1>2</td><td rowspan=3 colspan=1>WIY0</td><td rowspan=3 colspan=1>2.02.0</td></tr><tr><td rowspan=1 colspan=1>Stress</td><td rowspan=1 colspan=1>2</td></tr><tr><td></td><td></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Total</td><td rowspan=1 colspan=1>2</td><td rowspan=3 colspan=1>Fluency</td></tr><tr><td></td><td></td><td rowspan=2 colspan=1></td><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Accuracy</td><td rowspan=1 colspan=1>2</td><td rowspan=2 colspan=1>KA00</td><td rowspan=4 colspan=1>2.01.81.8</td></tr><tr><td></td><td></td><td rowspan=1 colspan=2></td><td></td><td rowspan=1 colspan=1>Stress</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=3 colspan=1>Completeness</td><td rowspan=3 colspan=1>2</td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=2></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Total</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>L</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Accuracy</td><td rowspan=1 colspan=1>2</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td rowspan=2 colspan=2>It</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=2 colspan=1>IH0T</td><td rowspan=2 colspan=1>2.02.0</td></tr><tr><td rowspan=4 colspan=1>Prosody</td><td rowspan=4 colspan=1>1.8</td><td rowspan=2 colspan=1></td><td></td><td></td><td rowspan=1 colspan=1>Stress</td><td rowspan=1 colspan=1>2</td></tr><tr><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Total</td><td rowspan=1 colspan=1>2</td><td></td><td></td></tr><tr><td rowspan=2 colspan=3></td><td rowspan=2 colspan=2></td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1>Accuracy</td><td rowspan=2 colspan=1>1.2</td><td></td></tr><tr><td rowspan=4 colspan=1>BEHOR</td><td rowspan=4 colspan=1>2.01.01.0</td></tr><tr><td rowspan=3 colspan=1>Total</td><td rowspan=3 colspan=1>1.6</td><td rowspan=3 colspan=3></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>Stress</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>Total</td><td rowspan=1 colspan=1>1.2</td></tr></table>

Figure 1: A running example curated from the speechocean762 dataset (Zhang et al., 2021) illustrates the evaluation flow of an APA system in the reading-aloud scenario, which offers an L2 learner in-depth pronunciation feedback.

learners to practice pronunciation skills in a stressfree and self-directed learning manner (Eskenazi 2009; Evanini and Wang, 2013; Evanini et al., 2017; Rogerson-Revell, 2021). As a crucial ingredient of CAPT, automatic pronunciation assessment (APA) aims to evaluate the extent of L2 learners’ oral proficiency and then provide fine-grained feedback on specific pronunciation aspects in response to a target language (Bannò et al., 2022; Chen and Li, 2016; Kheir et al., 2023). A de-facto standard for APA systems is typically instantiated with a “reading-aloud” scenario, where an L2 learner is presented with a text prompt and instructed to pronounce it correctly. To offer in-depth feedback on learners’ pronunciation quality, recent efforts have drawn attention to the notion of multi-aspect and multi-granular pronunciation assessments, which normally devises a unified scoring model to jointly evaluate pronunciation proficiency at various linguistic levels (i.e., phone-, word-, and utterance-levels) with diverse aspects (e.g., accuracy, fluency, and completeness), as the running example depicted in Figure 1. Methods along this line of research usually follow a parallel modeling paradigm, wherein the Transformer network and its variants serve as the backbone architecture to take as input a sequence of phonelevel pronunciation features and in turn predict multiple aspect scores across various linguistic levels simultaneously via a multi-task learning regime (Chao et al., 2022; Do et al., 2023a; Gong et al., 2022).

Albeit models stemming from the parallel modeling paradigm have demonstrated promising results on a few APA tasks, they still suffer from at least two weaknesses. First, the language hierarchy of an utterance is nearly sidelined, which, for example, assumes that all phones within a word are of equal importance and might insufficiently capture the word-level structural traits. Second, most of these methods largely overlook the relatedness among the pronunciation aspects. As an illustration, we visualize the correlation matrix in Figure 2, which shows the Pearson Correlation Coefficients (PCCs) between any pair of expert annotated aspect scores on the training set. We can observe that except for the aspects of utterancecompleteness and word-stress, the rest pronunciation aspects exhibit strong correlations not only within the same linguistic level but also across different linguistic levels<sup>2</sup>. Building on these observations, we in this paper present a novel language hierarchy-aware APA model, dubbed HierTFR, which leverages a hierarchical Transformer-based architecture to jointly model the intrinsic multi-level linguistic structures of an utterance while considering relatedness among aspects within and across different linguistic levels. To explicitly capture the relatedness within and across different linguistic levels, an aspect attention mechanism and a selective fusion module are introduced. The proposed model is further optimized with an effective correlation-aware regularizer, which encourages the correlations of predicted aspect scores to match those of their counterparts provided by human annotations. Furthermore, distinct pre-training strategies tailored for three linguistic levels are put forward, so as to boost model initialization and hence reduce the reliance on large amounts of supervised training data. A comprehensive set of experimental results reveal that the proposed model achieves significant and consistent improvements over several strong baselines on the speechocean762 benchmark dataset (Zhang et al., 2021).

![](images/5f86b599be40139005a14ff58708375ad6da5de29dd076e2a237bf91b6825911.jpg)  
Figure 2: A correlation matrix derived from the expert annotations of the training set. Each element in the matrix corresponds to the PCC score of a pair of measured aspects.

In summary, the main contributions of our work are at least three-fold: (1) we introduce HierTFR, a hierarchical neural model for APA, which is designed to hierarchically represent an L2 learner’s input utterance and effectively capture relatedness within and across different linguistic levels; (2) we propose a correlation-aware regularizer for model training, which encourages prediction scores to consider the relatedness among disparate aspects; and (3) extensive sets of experiments carried out on a public APA dataset confirm the utility of our proposed pre-training strategies, which considerably boosts the effectiveness of assessments across various linguistic levels.

## 2 Methodology

## 2.1 Problem Formulation

Given an input utterance U, consisting of a time sequence of audio signals X uttered by an L2 learner, and a reference text prompt T with � words and � phones, an APA model is trained to estimate the proficiency scores pertaining to multiple pronunciation aspects at various linguistic granularities. Let $\mathsf { G } = \{ p , w , u \}$ be a set of linguistic granularities, where �, �, � stands for the phone-, word-, and utterance-level linguistic units, respectively. For each linguistic unit $g \in G$ , the APA model learns to predict a set of aspect scores $\mathsf { A } ^ { g } = \{ a _ { 1 } ^ { g } , a _ { 2 } ^ { g } , \dots , a _ { N _ { g } } ^ { g } \}$ , where $N _ { g }$ is the number of pronunciation aspects of the linguistic unit $g .$

![](images/d29788a26f957ee4282c5e932a5a4f118c16a8dc8de3f9c6a5b7902a273192cf.jpg)  
Figure 3: An architecture overview of the proposed model, which consists of a phone-level modeling component, a word-level modeling component, and an utterance pooling module.

## 2.2 Hierarchical Interactive Transformer Architecture

The overall architecture of our proposed APA model is schematically depicted in Figure 3, which consists of three ingredients: phone-level modeling, word-level modeling, and utterance pooling modules. After obtaining the representations of various pronunciation aspects, fully-connected neural layers is functioned as the regressors to collectively generate the corresponding aspect score sequence for an input utterance.

Phone-level Modeling. For an input utterance U, various pronunciation features are extracted to portray the L2 learner’s pronunciation quality, which includes the goodness of pronunciation (GOP)-based features $\operatorname { E } ^ { G O P }$ , as well as prosodic features composed of duration $\mathrm { E } ^ { D u r }$ and energy $\mathrm { E } ^ { E n g }$ statistics (Witt and Young, 2000; Hu et al., 2015; Zhu et al., 2022; Shen et al., 2021) <sup>3</sup>. All these features are then concatenated and subsequently projected to from a sequence of acoustic features $\mathrm { X } ^ { p }$ . In the meantime, the phone-level text prompt is mapped into an embedding sequence $\mathtt { E } ^ { p }$ via a phone and position embedding layer and then point-wisely added to $\mathrm { X } ^ { p }$ for enriching the phonetic information of $\mathrm { X } ^ { p }$ The resulting representations $\mathrm { H } _ { p } ^ { 0 }$ are prepend with five trainable “[CLS]” embeddings and in turn fed into a phonelevel transformer to obtain the contextualized representations $\mathrm { H } ^ { p }$ (Vaswani et al., 2017):

$$
X ^ { p } = \mathbf { W } \cdot [ \mathrm { E } ^ { G O P } ; \mathrm { E } ^ { D u r } ; \mathrm { E } ^ { E n g } ] + \mathbf { b } ,\tag{1}
$$

$$
\begin{array} { r } { \mathrm { H } _ { p } ^ { 0 } = \mathrm { X } ^ { p } + \mathrm { E } ^ { p } , } \end{array}\tag{2}
$$

$$
\mathrm { H } ^ { p } = \mathrm { T r a n s f o r m e r } _ { \mathrm { p h n } } \big ( \mathrm { H } _ { p } ^ { 0 } \big ) ,\tag{3}
$$

where W and � are learnable parameters. To assess a sequence of phone-level aspect scores, H<sup>-</sup> (excluding the first 5 embeddings) is forward propagated to the corresponding regressors. The excluded embeddings $\mathrm { H } _ { 1 : 5 } ^ { p }$ are expected to convey the holistic pronunciation information and are further fed into the subsequent selective fusion mechanism for use in utterance-level assessments.

Word-level Modeling. For the word-level assessments, a word-level attention pooling is used to produce a word representation vector from its corresponding phones, which can be implemented as a multi-head attention layer followed by an average operation. The word-level input representations $\mathrm { H } _ { w } ^ { 0 }$ can be obtained by applying the word-level attention to the phone-level representations $\mathrm { X } ^ { p }$ and $\mathrm { H } ^ { p }$ individually, followed by a linear combination with the word-level textual embeddings $\mathrm { E } ^ { w }$ . Next, $\mathrm { H } _ { w } ^ { 0 }$ is prepend with five trainable “[CLS]” embeddings and fed into a transformer to calculate the contextualized representations $\mathrm { H } ^ { w }$ at word-level:

$$
\mathtt { X } ^ { w } = \mathtt { A t t e n } _ { \mathrm { w o r d } } ( \mathtt { X } ^ { p } ) ,\tag{4}
$$

$$
\widehat { \mathrm { H } } ^ { w } = \mathrm { A t t e n } _ { \mathrm { w o r d } } ( \mathrm { H } ^ { p } ) ,\tag{5}
$$

$$
\mathrm { H } _ { w } ^ { 0 } = \mathrm { X } ^ { w } + \widehat { \mathrm { H } } ^ { w } + \mathrm { E } ^ { w } ,\tag{6}
$$

$$
\mathrm { H } ^ { w } = \mathrm { T r a n s f o r m e r } _ { \mathrm { w o r d } } ( \mathrm { H } _ { w } ^ { 0 } ) .\tag{7}
$$

Note here that $\mathrm { H } ^ { w }$ (excluding the first 5 embeddings) is utilized in the word-level assessments while the excluded embeddings $\mathrm { H } _ { 1 : 5 } ^ { w }$ are fed into in subsequent selective fusion mechanism for use in the utterance-level assessments.

After that, an aspect attention mechanism is introduced to capture the relatedness among disparate aspects (Do et al., 2023b; Ridley et al., 2021). This mechanism consists of two sub-layers: a self-gating layer and a multi-head cross-attention layer. Specifically, for the �-th word-level aspect, the relation-aware representations $\widehat { \mathrm { H } } ^ { w r _ { j } }$ are first derived from $\mathrm { H } ^ { w }$ via a self-gating layer which aims to abstract away from redundant information while considering the information gathered from other aspects. In addition, a multi-head cross-attention (MHCA) process alongside a masking strategy is employed to calculate aspect representations $\mathrm { H } ^ { w _ { j } }$ from a collection of all relation-aware aspect representations $\mathbb { C } ^ { r a } = \left[ \widehat { \mathrm { H } } ^ { w r _ { 1 } } , \dots , \widehat { \mathrm { H } } ^ { w r _ { N _ { w } } } \right]$ The following equations illustrate the operations of aspect attention:

$$
\widehat { \mathrm { H } } ^ { w _ { j } } = \mathsf { W } _ { j } \cdot \mathrm { H } ^ { w } + \boldsymbol { \mathsf { b } } _ { j } ,
$$

$$
\widehat { \mathrm { H } } ^ { w r _ { j } } = \sigma \left( \boldsymbol { \mathrm { W } } _ { g _ { j } } \cdot \boldsymbol { \mathrm { C } } ^ { w } + \boldsymbol { \mathbf { b } } _ { g _ { j } } \right) \otimes \widehat { \mathrm { H } } ^ { w _ { j } } ,\tag{8}
$$

$$
\mathrm { H } ^ { w _ { j } } = \mathrm { M H C A } \big ( \widehat { \mathrm { H } } ^ { w r _ { j } } , \mathrm { C } ^ { r a } \big ) ,\tag{9}
$$

(10)

where $\widehat { \mathrm { H } } ^ { w _ { j } }$ are aspect-specific representations, and $\mathbb { C } ^ { w } = [ \widehat { \mathrm { H } } ^ { w _ { 1 } } , \dots , \widehat { \mathrm { H } } ^ { w _ { N _ { w } } } ]$ includes all aspect-specific representations. In MHCA, $\widehat { \mathrm { H } } ^ { w r _ { j } }$ is linearly projected to act as the query matrix, while ${ \mathsf { C } } ^ { r a }$ is linearly projected to form the key and value matrixes. Additionally, the masking strategy ensures that the output representation at a specific position is only influenced by the other aspects of the word unit. Lastly, the aspect representations $\mathrm { H } ^ { w _ { j } }$ are taken as input to the corresponding regressor to predict a score sequence for the �-th word-level pronunciation aspect.

Utterance Pooling Module. For the utterancelevel assessments, utterance-level attention pooling is introduced to generate an utterance-level holistic representation from the corresponding input representations, which can be effectively implemented by attention pooling (Peng et al., 2022). In more detail, the utterance-level representation $\mathbf { h } ^ { u }$ can be obtained by feeding the vector sequences $\mathrm { X } ^ { p }$ $\mathrm { H } ^ { p }$ , and $\mathrm { H } ^ { w }$ into an utterance-level attention pooling module individually, followed by an aggregation operation:

$$
\bar { \mathbf { h } } ^ { u _ { x } } = \mathrm { A t t e n } _ { \mathrm { u t t } } ( \mathrm { X } ^ { p } ) ,
$$

$$
\bar { \mathbf { h } } ^ { u _ { p } } = \mathrm { A t t e n } _ { \mathrm { u t t } } ( \mathrm { H } ^ { p } ) ,\tag{11}
$$

$$
\bar { \mathbf { h } } ^ { u _ { w } } = \mathrm { A t t e n } _ { \mathrm { u t t } } ( \mathrm { H } ^ { w } ) ,\tag{12}
$$

$$
\mathbf { h } ^ { u } = \mathrm { W } _ { u } \big ( \bar { \mathbf { h } } ^ { u _ { x } } + \bar { \mathbf { h } } ^ { u _ { p } } + \bar { \mathbf { h } } ^ { u _ { w } } \big ) + \mathbf { b } _ { u } ,\tag{13}
$$

(14)

where $\mathsf { W } _ { u } ,  { \mathbf { b } } _ { u }$ are trainable parameters.

Next, a selective fusion mechanism is proposed to integrate contextualized representations across multiple linguistic levels for the utterance-level pronunciation assessments (Xu et al., 2021). Specifically, for the estimation of�-th utterancelevel aspect score, an aspect attention operation is first performed on $\mathbf { h } ^ { u }$ to a produce intermediate representation $\hat { \mathbf { h } } ^ { u _ { j } }$ . Note also that the gate values for the phone $( g _ { p } ^ { u _ { j } } )$ , word $( g _ { w } ^ { u _ { j } } )$ and utterance $( g _ { u } ^ { u _ { j } } )$ granularities are used to control the extent to which these contextualized representations can flow into the fused representation $\mathbf { h } ^ { u _ { j } }$

$$
g _ { p } ^ { u _ { j } } = \sigma \big ( \mathbf { w } _ { p _ { j } } \cdot \big [ \mathbf { h } _ { j } ^ { p } ; \mathbf { h } _ { j } ^ { w } ; \hat { \mathbf { h } } ^ { u _ { j } } \big ] + b _ { p _ { j } } \big ) ,\tag{15}
$$

$$
g _ { w } ^ { u _ { j } } = \sigma \Big ( \mathbf { w } _ { w _ { j } } \cdot \big [ \mathbf { h } _ { j } ^ { p } ; \mathbf { h } _ { j } ^ { w } ; \hat { \mathbf { h } } ^ { u _ { j } } \big ] + b _ { w _ { j } } \Big ) ,\tag{16}
$$

$$
\begin{array} { r } { { g } _ { u } ^ { u _ { j } } = \sigma \Big ( \mathbf { w } _ { u _ { j } } \cdot \big [ \mathbf { h } _ { j } ^ { p } ; \mathbf { h } _ { j } ^ { w } ; \hat { \mathbf { h } } ^ { u _ { j } } \big ] + b _ { u _ { j } } \Big ) , } \end{array}\tag{17}
$$

$$
\mathbf { h } ^ { u _ { j } } = g _ { p } ^ { u _ { j } } \cdot \mathbf { h } _ { j } ^ { p } + g _ { w } ^ { u _ { j } } \cdot \mathbf { h } _ { j } ^ { w } + g _ { u } ^ { u _ { j } } \cdot \mathbf { \hat { h } } ^ { u _ { j } } ,\tag{18}
$$

where $\mathbf { h } _ { j } ^ { p }$ and $\mathbf { h } _ { j } ^ { w }$ are �-th representation vectors of $\mathrm { H } ^ { p }$ and $\mathrm { H } ^ { w }$ ; and $\mathbf { w } _ { p _ { j } } , \mathbf { w } _ { w _ { j } } , \mathbf { w } _ { u _ { j } } , b _ { p _ { j } } , b _ { w _ { j } } ,$ and $b _ { u _ { j } }$ are trainable parameters. The fused representation $\mathbf { h } ^ { u _ { j } }$ is then passed to the corresponding regressor to assess the proficiency score for a given utterance-level aspect.

## 2.3 Optimization

Automatic Pronunciation Assessment Loss. The loss for multi-aspect and multi-granular pronunciation assessment, $\mathcal { L } _ { A P A }$ , is calculated as a weighted sum of the mean square error (MSE) losses corresponding to different linguistic levels.

$$
\mathcal { L } _ { A P A } = \frac { \lambda _ { p } } { N _ { p } } \sum _ { j _ { p } } \mathcal { L } _ { p ^ { j _ { p } } } + \frac { \lambda _ { w } } { N _ { w } } \sum _ { j _ { w } } \mathcal { L } _ { w ^ { j _ { w } } } + \frac { \lambda _ { u } } { N _ { u } } \sum _ { j _ { u } } \mathcal { L } _ { u ^ { j _ { u } } } ,\tag{19}
$$

where $\mathcal { L } _ { p ^ { j _ { p } } } , \mathcal { L } _ { w ^ { j _ { w } } }$ , and $\mathcal { L } _ { u } j _ { u }$ are phone-level, word-level, and utterance-level losses for disparate aspects, respectively. The parameters $\lambda _ { p } , \lambda _ { w }$ , and $\lambda _ { u }$ are adjustable parameters which control the influence of different granularities, and $N _ { p } , N _ { w }$ and $N _ { u }$ mark the numbers of aspects at the phone-, word-, and utterance-levels, respectively.

Correlation-aware Regularization Loss. The correlation-aware regularization loss is defined as the difference between the correlation matrix of the predicted aspect scores $\widehat { \Sigma }$ and the correlation matrix of the corresponding target labels $\Sigma \colon$

$$
\mathcal { L } _ { c o r } = \ell ( \widehat { \Sigma } , \Sigma ) ,\tag{20}
$$

where $\ell$ is the regularization loss function, and each element in $\begin{array} { r } { \widehat { \sum } _ { i j } \ : ( \mathrm { o r } \ : \sum _ { i j } ) } \end{array}$ is defined as a Pearson correlation coefficient between � -th aspect score and �-th aspect score<sup>4</sup>. We adopt the MSE criterion for computing $\ell ;$ the overall loss thus can be expressed by:

$$
\mathcal { L } = \mathcal { L } _ { A P A } + \lambda \mathcal { L } _ { c o r } ,\tag{21}
$$

where $\lambda \in [ 0 , 1 ]$ is a tunable parameter, which is experimentally set to 0.01 based on the development set.

## 2.4 Pre-training Strategies

It is without doubt that a proper initialization is vital for the estimation of a neural model, due mainly to the highly nonconvex nature of the training loss function (Tamborrino et al., 2020;

Lakhotia et al., 2021). At lower linguistic levels, we leverage the mask-predict objective (Ghazvininejad et al., 2019) in the pre-training stage. To this end, we first mask a portion of input text prompt at phone- and word-levels. The corresponding Transformers are then tasked on recovering the masked tokens conditioning on the unmasked prompt sequence and the associated pronunciation representations ${ \mathrm { ( i . e . , ~ H } _ { p } ^ { 0 } }$ , and $\mathrm { H } _ { w } ^ { 0 } )$ ). For the utterance level, we base the proposed pretraining strategy on predicting the relatively high or low accuracy scores for a pair of utterances. Namely, given any two utterances, the objective is to predict whether the former has a higher, lower, or the same accuracy score as the latter. Note here that, utterance pairs are randomly selected from a training batch, and this mechanism is employed to pretrain their utterance-level representations, denoted as $\mathbf { h } _ { o u t _ { 1 } } ^ { u }$ , and $\mathbf { h } _ { o u t _ { 2 } } ^ { u }$ . Next, we feed the concatenation of these vector representations $\mathbf { h } _ { o u t } ^ { u } = [ \mathbf { h } _ { o u t _ { 1 } } ^ { u } ; \mathbf { h } _ { o u t _ { 2 } } ^ { u } ]$ into a three-way classifier, using the cross-entropy loss as the training objective.

## 3 Experimental Settings

## 3.1 Evaluation Dataset and Metrics

We conducted APA experiments on the speechocean762 dataset, which is a publicly available open-source dataset specifically designed for research on APA (Zhang et al., 2021). This dataset contains 5,000 English-speaking recordings spoken by 250 Mandarin L2 learners. The training and test sets are of equal size, and each of them has 2,500 utterances, where pronunciation proficiency scores were evaluated at multiple linguistic granularities with various pronunciation aspects. Each score was independently assigned by five experienced experts using the same rubrics, and the final score was determined by selecting the median value from the five scores. The evaluation metrics include Pearson Correlation Coefficient (PCC) and Mean Square Error (MSE). PCC is the primary evaluation metric, quantifying the linear correlation between predicted and ground-truth scores. A higher PCC score reflects a stronger correlation between the predictions and human annotations. In the following experiments, we report the MSE value in order to evaluate the phoneme-level APA accuracy in comparison with prior arts.

<table><tr><td rowspan="2">Models</td><td colspan="2">Phone Score</td><td colspan="2">Word Score (PCC)</td><td colspan="5">Utterance Score (PCC)</td></tr><tr><td>MSE↓</td><td>PCC↑</td><td>Accuracy↑ Stress↑ Total↑</td><td></td><td>Accuracy↑ Completeness↑ Fluency↑ Prosody↑ Total↑</td><td></td><td></td><td></td><td></td></tr><tr><td>Lin2021</td><td></td><td>一</td><td>1</td><td></td><td>一 1</td><td></td><td></td><td></td><td>0.720</td></tr><tr><td>Kim2022</td><td>–</td><td>1</td><td>=</td><td></td><td>- –</td><td></td><td>0.780</td><td>0.770</td><td>-</td></tr><tr><td>Ruy2023</td><td></td><td>一</td><td></td><td></td><td>0.719 一</td><td></td><td>0.775</td><td>0.773</td><td>0.743</td></tr><tr><td>LSTM</td><td>0.089 ±0.000</td><td>0.591 ±0.003</td><td>0.514</td><td>0.294</td><td>0.531 0.720</td><td>0.076</td><td>0.745 ±0.002</td><td>0.747</td><td>0.741 ±0.002</td></tr><tr><td rowspan="2">GOPT</td><td>0.085</td><td>0.612</td><td>±0.003 0.533</td><td>±0.012 0.291</td><td>±0.004 0.549</td><td>±0.002 ±0.086 0.714 0.155</td><td>0.753</td><td>±0.005 0.760</td><td>0.742</td></tr><tr><td>±0.001</td><td>±0.003</td><td>±0.004</td><td>±0.030 ±0.002</td><td>±0.004</td><td>±0.039</td><td>±0.008</td><td>±0.006</td><td>±0.005</td></tr><tr><td rowspan="2">HiPAMA</td><td>0.084</td><td>0.616</td><td>0.575</td><td>0.320</td><td>0.591</td><td>0.730</td><td>0.276</td><td>0.749</td><td>0.751 0.754</td></tr><tr><td>±0.001</td><td>±0.004</td><td>±0.004</td><td>±0.021</td><td>±0.004</td><td>±0.002 ±0.177</td><td>±0.001</td><td>±0.002</td><td>±0.002</td></tr><tr><td rowspan="2">HierTFR</td><td>0.081</td><td>0.644</td><td>0.622</td><td>0.325</td><td>0.634</td><td>0.735 0.513</td><td>0.801</td><td>0.795</td><td>0.764</td></tr><tr><td>±0.000</td><td>±0.000</td><td>±0.002</td><td>±0.022</td><td>±0.002</td><td>±0.008 ±0.204</td><td>±0.004</td><td>±0.002</td><td>±0.002</td></tr></table>

Table 1: The performance evaluations of our model and all compared methods on speechocean762 test set.

## 3.2 Implementation Details

For the input feature extraction of the phone-level energy and the duration statistics, we follow the processing flow suggested by Zhu et al. (2022) and Shen et al. (2021), where a phone-level feature is constructed from time-aggregated frame-level features according to the forced alignment. Both the phone- and word-level Transformers for contextual representation modeling consist of 3 processing blocks utilizing multi-head attention with 3 heads and 24 hidden units, respectively. In addition, for the word- and utterance-level attention pooling, we use a single-layer multi-head attention with 3 heads and 24 hidden units. The combination weights used in Eq. (19) for the APA loss $( \lambda _ { p } \ : , \ : \lambda _ { w } \ : , \ : \lambda _ { u } )$ are assigned as (1, 1, 1) , respectively. To ensure the reliability of our experimental results, we repeated 5 independent trials, each of which consisted of 100 epochs with different random seeds. The test set results are reported by averaging those achieved by the top 100 best-performing models which are determined based on their PCC scores on the development set.

## 3.3 Compared Methods

We compare our proposed model (viz. HierTFR) with several families of top-of-the-line methods. Lin et al. (2021) and Kim et al. (2022) are singleaspect assessment models. The former develops a bottom-up hierarchical scorer evaluating the accuracy scores at the utterance level. The latter leverages self-supervised features (Baevski et al., 2020) to describe the learner’s pronunciation traits and then separately models the corresponding utterance-level aspects with recurrent neural models. In addition, LSTM, GOPT (Gong et al., 2022; Ruy et al. 2023), and HiPAMA (Do et al., 2023b) are multi-aspect and multi-granular pronunciation assessments. First, LSTM and GOPT follow a parallel modeling regime, both of which treat the phone-level input features as a flattened sequence and assess higher level pronunciation scores through stacking LSTM layers or Transformer blocks. Second, Ruy et al. (2023) introduces a unified model architecture that jointly optimizes phone recognition and APA tasks. Lastly, HiPAMA is a hierarchical APA model that more resembles our model than the other methods compared in this paper. Different from our method, HiPAMA extracts high-level pronunciation features from low-level features based on a simple average pooling mechanism. Furthermore, the aspect attention mechanism used in HiPAMA performs on the logistics, whereas our model operates on the intermediate representations.

## 4 Experimental Results

## 4.1 Main Results

Table 1 reports the results on the speechocean762 dataset, which is divided into three parts: the first part shows the results of single-aspect assessment models, the second part presents the results of multi-aspect and multi-granular pronunciation methods, and the third part reports the results of our model. We further provide a comparison with another hierarchical APA model (viz. HiPAMA) in the third part.

![](images/bcb102ad524ac46584a7bb650bb799cd144ce0744b9074ba3d5116268b99cc9f.jpg)  
(a) Word-level Aspect Predictions

![](images/859e29e0098122f2fd1dbaad7f9ce8506bbc4817d957035cb769baaf0548d06d.jpg)  
(b) Utterance-level Aspect Predictions

![](images/b1392888171c30cecc009d9ee78e68e994c4d064fefb75f801340b406ed56789.jpg)  
(c) Gate Values in Selective Fusion Mechanism for Utterance-level Aspect Predictions  
Figure 4: Qualitative visualization of model parameters when predicting each aspect score. We show (a) the averaged attention values for word-level aspects, (b) the averaged attention weights for utterance-level aspects, and (c) the averaged gate values for three linguistic levels.

First, a general observation is that our approach, HierTFR, excels in all assessment tasks, especially at the linguistic levels of utterance and word. This performance gain confirms that our model works comparably better for capturing the relationships between linguistic units than the other competitive methods. In terms of the utterance-level total score, the single-aspect assessment method (viz. Lin2021) largely falls behind the other multi-aspect and multi-granular pronunciation assessment models, which we attribute to the fact that the single-aspect assessment method is unable to harness the dependency relationships between aspects through the multi-task learning paradigm. By leveraging self-supervised learning features, Kim2022 achieves significant improvements over most APA methods in terms of the utterance-level assessments. Next, we scrutinize the performance of multi-aspect and multi-granular pronunciation assessment methods. Ruy2023 demonstrates significant advancements in the utterance-level fluency and prosody assessments due probably to the joint training of the APA model on the phone recognition task simultaneously. In comparison with the parallel modeling approaches (i.e., GOPT and LSTM), we can observe that HierTFR substantially improves the performance across all tasks, where its performance gains reveal the importance of capturing the hierarchical linguistic structures of an input utterance. Notably, compared to the HiPAMA, our model consistently achieves superior performance on a variety of pronunciation assessment tasks. This superiority stems from our tactfully designed selective fusion mechanism and the correlation-aware loss. The former allows our model to assess utterance-level aspect scores by leveraging information from diverse linguistic levels, while the latter explicitly models the relatedness among different aspects during the optimization.

## 4.2 Qualitative Analysis

Qualitative Visualization of Relatedness Among Aspects. In the second set of experiments, we examine the relatedness among disparate aspects at both word- and utterance-levels, where the attention weights of the aspect attention mechanisms were determined based on the development set when assessing a specific aspect score. For the word-level assessments, the distributions of attention weights are in close accordance with the manual scoring rubrics of the speechocean762 dataset. In Figure 4(a), the total aspect serves as a comprehensive assessment and the corresponding weights are contributed from various pronunciation aspects. In contrast, the accuracy aspect measures the percentage of mispronounced phones within a word, leading to the attention weights being more concentrated on a word-level unit itself. Furthermore, the stress score also highly attends to the accuracy aspect, reflecting the strong relation between lexical stress and word-level pronunciation accuracy (Korzekwa et al., 2022). In regard to the relatedness within the utterance-level aspects, inspecting Figure 4(b) we find that the attention weights of the prosody and total aspects scatter across various pronunciation aspects, whereas the attention weights of the accuracy and completeness center primarily on the completeness aspect. One possible reason is that the prosody and total scores both measure highlevel oral skills, and when the human annotators judge the proficiency scores, they also take multiple pronunciation aspects into account simultaneously. Next, the completeness aspect measures the percentage of words with good pronunciation quality in an utterance. This implicitly reflects the intelligibility of a learner's pronunciation and is vital to the accuracy assessment.

<table><tr><td rowspan="2">Models</td><td>Phone Score</td><td colspan="3">Word Score</td><td colspan="5">Utterance Score</td></tr><tr><td>Accuracy</td><td>Accuracy</td><td>Stress</td><td>Total</td><td>Accuracy</td><td>Completeness</td><td>Fluency</td><td>Prosody</td><td>Total</td></tr><tr><td>HierTFR</td><td>0.644</td><td>0.622</td><td>0.325</td><td>0.634</td><td>0.735</td><td>0.513</td><td>0.801</td><td>0.795</td><td>0.764</td></tr><tr><td>w/o CorrLoss</td><td>0.639</td><td>0.605</td><td>0.348</td><td>0.620</td><td>0.728</td><td>0.520</td><td>0.796</td><td>0.789</td><td>0.758</td></tr><tr><td>w/o Pretrain</td><td>0.621</td><td>0.545</td><td>0.318</td><td>0.559</td><td>0.716</td><td>0.215</td><td>0.770</td><td>0.772</td><td>0.739</td></tr><tr><td>w/o SFusion</td><td>0.630</td><td>0.608</td><td>0.328</td><td>0.622</td><td>0.728</td><td>0.378</td><td>0.784</td><td>0.782</td><td>0.756</td></tr><tr><td>w/o AspAtt</td><td>0.636</td><td>0.584</td><td>0.290</td><td>0.596</td><td>0.724</td><td>0.383</td><td>0.784</td><td>0.775</td><td>0.746</td></tr></table>

Table 2: Ablation study on HierTFR, reporting PCC scores on three linguistic levels.

Qualitative Visualization of Interactions Across Linguistic Levels. In Figure 4(c), we report on the average gate values of utterances for three linguistic granularities by estimating the utterancelevel pronunciation aspect scores based on the development set. We can observe that the phonelevel representations bear high impacts on the utterance-level aspect assessments, in comparison to the other linguistic levels. Next, the word-level and utterance-level representations exhibit minimal impact on the completeness and total aspects, respectively. One possible reason is that the completeness aspect somehow reflects pronunciation intelligibility, and our model learns to distill the information from the phone- and utterance-level representations. On the other hand, the total aspect evaluates an overall speaking skill. Our model thus tends to capture the subtle information by distilling the fine-grained traits inherent in the phone- and word-levels.

## 4.3 Ablation Study

To gain insight into the effectiveness of each model component of HierTFR, we conduct an ablation study to investigate their impacts. These variations include excluding the correlation-aware regularizer (w/o CorrLoss), removing the proposed pretraining strategies (w/o Pretrain), omitting the selective fusion mechanism (w/o SFusion), and eliminating the aspect attention mechanism at both word and utterance levels (w/o AspAtt). From Table 2, we can observe that the proposed correlation-aware regularization loss is beneficial for most pronunciation assessment tasks. Next, the proposed pre-training strategies are crucial to obtaining better performance as the model trained without them tends to perform relatively worse for all pronunciation assessment tasks. This highlights the efficacy of the pre-training strategies for hierarchical APA models, thereby alleviating the requirement for large amounts of supervised training data. Third, removing the selective fusion mechanism leads to degradations in the utterancelevel aspect assessments, while removing the aspect attention mechanism deteriorates the performance on word-level aspect assessments.

## 5 Related Work

Early studies on APA focused primarily on singleaspect assessments, typically through individually constructing scoring modules to predict a holistic pronunciation proficiency score on a targeted linguistic level or some specific aspect with different sets of hand-crafted features, such as the phone-level posterior probability (Witt and Young, 2000), word-level lexical stress (Ferrer et al., 2015), or various utterance-level pronunciation aspects (Coutinho et al., 2016). More recently, with the rapid progress of deep learning (Vaswani et al., 2017; Raffel et al., 2020; Hsu et al., 2021), several neural scoring models have been successfully developed for multi-aspect and multi-granular pronunciation assessment. Gong et al. (2022) proposed a GOP feature-based Transformer (GOPT) architecture to model pronunciation aspects at multiple granularities with a multi-task learning scheme. Do et al. (2023b) employed a neural scorer with a hierarchical structure to mimic the language hierarchy of an utterance to deliver state-of-the-art performance for APA.

## 6 Conclusion

In this paper, we have put forward a novel hierarchical modeling method (dubbed HierTFR) for multi-aspect and multigranular APA. To explicitly capture the relatedness between pronunciation aspects, a correlation-aware regularizer loss has been devised. We have further developed model pre-training strategies for our HierTFR model. Extensive experimental results confirm the feasibility and effectiveness of the proposed method in relation to several top-of-theline methods. In future work, we plan to examine the proposed HierTFR model on open-response scenarios, where learners speak freely or respond to a given task or question (Wang et. al., 2018; Park and Choi, 2023). In addition, the issues of explainable pronunciation feedback are also left as a future extension.

## Limitations

Limited Applicability. In this research, the proposed model focus on the “reading-aloud” pronunciation training scenario, where the assumption is that the L2 learner pronounces a predetermined text prompt correctly, which restricts the applicability of our models to other learning scenarios, such as freely speaking or openended conversations.

Lack of Accent Diversity. The used dataset merely contains Mandarin L2 learners, hindering the generalizability of the proposed model and could be untenable when assessing the L2 learners with diverse accents.

The lack of Interpretability. The model of the proposed method simply trains to mimic expert’s annotations without resorting to manual assessment rubrics or other external knowledge, making it not straightforward to provide reasonable explanations for the assessment results.

## Ethics Statement

We hereby acknowledge that all of the co-authors of this work compile with the provided ACL Code of Ethics and honor the code of conduct. Our experimental corpus, speechocean762, is widely used and publicly available. We think there are no potential risks for this work.

## References

Stefano Bannò, Bhanu Balusu, Mark Gales, Kate Knill,and Konstantinos Kyriakopoulos. 2022. Viewspecific assessment of L2 spoken English. In Proceedings of Interspeech (INTERSPEECH), pages 4471–4475.

Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, and Michael Auli. 2020. Wav2vec 2.0: A framework for self-supervised learning of speech representations. In Proceedings of the International Conference on Neural Information Processing Systems (NIPS), pages 12449–12460.

Fu An Chao, Tien Hong Lo, Tzu I. Wu, Yao Ting Sung, Berlin Chen. 2022. 3M: An effective multi-view, multigranularity, and multi-aspect modeling approach to English pronunciation assessment. In Proceedings of the Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), pages 575–582.

Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, Jian Wu, Long Zhou, Shuo Ren, Yanmin Qian, Yao Qian, Jian Wu, Michael Zeng, Xiangzhan Yu, Furu Wei. 2022. Wavlm: Large-scale self-supervised pre-training for full stack speech processing. IEEE Journal of Selected Topics in Signal Processing, volume 16, pages1505–1518.

Nancy F. Chen, and Haizhou Li. 2016. Computerassisted pronunciation training: From pronunciation scoring towards spoken language learning. In Proceedings of the Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), pages 1–7.

Eduardo Coutinho, Florian Hönig, Yue Zhang, Simone Hantke, Anton Batliner, Elmar Nöth, and Björn Schuller. 2016. Assessing the prosody of non-native

speakers of English: Measures and feature sets. In Proceedings of the International Conference on Language Resources and Evaluation (LREC), pages 1328–1332.

Heejin Do, Yunsu Kim, and Gary Geunbae Lee. 2023a. Score-balanced Loss for Multi-aspect Pronunciation Assessment. In Proceedings of Interspeech (INTERSPEECH), pages 4998–5002.

Heejin Do, Yunsu Kim, and Gary Geunbae Lee. 2023b. Hierarchical pronunciation assessment with multiaspect attention. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5.

Maxine Eskenazi. 2009. An overview of spoken language technology for education. Speech communication, volume 51, pages 832–844.

Keelan Evanini, and Xinhao Wang. 2013. Automated speech scoring for Nonnative middle school students with multiple task types. In Proceedings of Interspeech (INTERSPEECH), pages 2435–2439.

Keelan Evanini, Maurice Cogan Hauck, and Kenji Hakuta. 2017. Approaches to automated scoring of speaking for K–12 English language proficiency assessments. ETS Research Report Series, pages 1– 11.

Luciana Ferrer, Harry Bratt, Colleen Richey, Horacio Franco, Victor Abrash, and Kristin Precoda. 2015. Classification of lexical stress using spectral and prosodic features for computer-assisted language learning systems. Speech Communication, volume 69, pages 31–45.

Marjan Ghazvininejad, Omer Levy, Yinhan Liu, and Luke Zettlemoyer. 2019. Mask-Predict: Parallel Decoding of Conditional Masked Language Models. In Proceedings of the Conference on Empirical Methods in Natural Language Processing and the International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 6112–6121.

Yuan Gong, Ziyi Chen, Iek-Heng Chu, Peng Chang, and James Glass. 2022. Transformer-based multiaspect multigranularity non-native English speaker pronunciation assessment. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 7262–7266.

Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. 2021. HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units. IEEE/ACM Transactions on Audio, Speech and Language Processing, volume 29, pages 3451–3460.

Wenping Hu, Yao Qian, Frank K. Soong, and Yong Wang. 2015. Improved mispronunciation detection with deep neural network trained acoustic models and transfer learning based logistic regression classifiers. Speech Communication, volume 67, pages 154–166.

Eesung Kim, Jae-Jin Jeon, Hyeji Seo, Hoon Kim. 2022. Automatic pronunciation assessment using selfsupervised speech representation learning. In Proceedings of Interspeech (INTERSPEECH), pages 1411–1415.

Yassine Kheir, Ahmed Ali, and Shammur Chowdhury. 2023. Automatic Pronunciation Assessment - A Review. In Findings of the Association for Computational Linguistics: EMNLP, pages 8304– 8324.

Daniel Korzekwa, Jaime Lorenzo-Trueba, Thomas Drugman, and Bozena Kostek. 2022. Computerassisted pronunciation training—Speech synthesis is almost all you need. Speech Communication, volume 142, pages 22–33.

Kushal Lakhotia, Eugene Kharitonov, Wei-Ning Hsu, Yossi Adi, Adam Polyak, Benjamin Bolte, Tu-Anh Nguyen, Jade Copet, Alexei Baevski, Abdelrahman Mohamed, and Emmanuel Dupoux. 2021. On Generative Spoken Language Modeling from Raw Audio. Transactions of the Association for Computational Linguistics, volume 9, pages 1336– 1354.

Binghuai Lin and Liyuan Wang. 2021. Deep feature transfer learning for automatic pronunciation assessment. In Proceedings of Interspeech (INTERSPEECH), pages 4438–4442.

Silke M. Witt and S. J. Young. 2000. Phone-level pronunciation scoring and assessment for interactive language learning. Speech Communication, volume 30, pages 95–108.

Jungbae Park and Seungtaek Choi. 2023. Addressing cold start problem for end-to-end automatic speech scoring. In Proceedings of Interspeech (INTERSPEECH), pages 994–998.

Yifan Peng, Siddharth Dalmia, Ian Lane, and Shinji Watanabe. 2022. Branchformer: Parallel mlpattention architectures to capture local and global context for speech recognition and understanding. In International Conference on Machine Learning (PMLR), pages 17627–17643

Yu Wang, M.J.F. Gales, Kate M Knill, Konstantinos Kyriakopoulos, Andrey Malinin, Rogier C van Dalen, Mohammad Rashid. 2018. Towards automatic assessment of spontaneous spoken English. Speech Communication, volume 104, pages 47–56.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, volume 21, pages 5485–5551.

Robert Ridley, Liang He, Xin-yu Dai, Shujian Huang, and Jiajun Chen. 2021. Automated cross-prompt scoring of essay traits. In Proceedings of the AAAI conference on artificial intelligence (AAAI), volume 35, pages 13745–13753.

Pamela M Rogerson-Revell. 2021. Computer-assisted pronunciation training (CAPT): Current issues and future directions. RELC Journal, volume 52, pages 189–205.

Hyungshin Ryu and Sunhee Kim and Minhwa Chung. 2023. A joint model for pronunciation assessment and mispronunciation detection and diagnosis with multi-task learning. In Proceedings of Interspeech (INTERSPEECH), pages 959–963.

Yang Shen, Ayano Yasukagawa, Daisuke Saito, Nobuaki Minematsu, and Kazuya Saito. 2021. Optimized prediction of fluency of L2 English based on interpretable network using quantity of phonation and quality of pronunciation. In Proceedings of IEEE Spoken Language Technology Workshop (SLT), pages 698–704.

Alexandre Tamborrino, Nicola Pellicanò, Baptiste Pannier, Pascal Voitot, and Louise Naudin. 2020. Pretraining is (almost) all you need: An application to commonsense reasoning. In Proceedings of the Association for Computational Linguistics (ACL), pages 3878–3887.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Proceedings of the Conference on Neural Information Processing Systems (NeurIPS), pages 5998–6008.

Heng-Da Xu, Zhongli Li, Qingyu Zhou, Chao Li, Zizhen Wang, Yunbo Cao, Heyan Huang, and Xian-Ling Mao. 2021. Read, listen, and see: Leveraging multimodal information helps Chinese spell checking. In Findings of the Association for Computational Linguistics (ACL-IJCNLP Findings), pages 716–728.

Junbo Zhang, Zhiwen Zhang, Yongqing Wang, Zhiyong Yan, Qiong Song, Yukai Huang, Ke Li, Daniel Povey, and Yujun Wang. 2021. Speechocean762: An open-source non-native English speech corpus for pronunciation assessment.

In Proceedings of Interspeech (INTERSPEECH), pages 3710 –3714.

Chuanbo Zhu, Takuya Kunihara, Daisuke Saito, Nobuaki Minematsu, Noriko Nakanishi. 2022. Automatic prediction of intelligibility of words and phonemes produced orally by japanese learners of English. In IEEE Spoken Language Technology Workshop (SLT), pages. 1029–1036.

## A Pronunciation Feature Extractions

GOP Feature. To extract the GOP feature, we first align audio signals X with the text prompt T by using an ASR model <sup>5</sup> to obtain the timestamps for each phone in the canonical phone sequence. Next, framelevel phonetic posterior probabilities are produced by the ASR model and then averaged over the time dimension based on the phone-level timestamps. The resulting phone-level posterior probabilities are converted into a GOP feature vector as a combination of log phone posterior (LPP) and log posterior ratio (LPR). Owing to the used ASR model containing 42 phones, the GOP feature of a canonical phone � can be represented as an 84-dimensional vector:

$$
\begin{array} { r l } { { } } & { { [ \mathrm { L P P } ( p _ { 1 } ) , \ldots , \mathrm { L P P } ( p _ { 4 2 } ) , } } \\ { { } } & { { { \mathrm { L P R } } ( p _ { 1 } | p ) , \ldots , \mathrm { L P R } ( p _ { 4 2 } | p ) ] } } \end{array}\tag{22}
$$

$$
\begin{array} { l } { { \displaystyle \mathrm { L P P } ( p _ { i } ) = \log p ( p _ { i } | \mathbf { o } ; \mathrm { t } _ { s } , \mathrm { t } _ { e } ) } \ ~ } \\ { { \displaystyle = \frac { 1 } { t _ { e } - t _ { s } + 1 } \sum _ { t = t _ { s } } ^ { t _ { e } } \log p ( p _ { i } | \mathbf { o } _ { t } ) } , } \end{array}\tag{23}
$$

$$
\begin{array} { r } { \mathrm { L P R } ( p _ { i } | p ) = \log p ( p _ { i } | \mathbf { 0 } ; \mathrm { t } _ { s } , \mathrm { t } _ { e } ) \qquad } \\ { - \log p ( p | \mathbf { 0 } ; \mathrm { t } _ { s } , \mathrm { t } _ { e } ) , } \end{array}\tag{24}
$$

where LPR is the log posterior ratio between phones $p _ { i }$ and $p ; t _ { s }$ and $t _ { e }$ are the start and end timestamps of phone $p ,$ and $o _ { t }$ is the input acoustic observation of the time frame �.

Energy Feature. The energy feature is a 7- dimensional vector comprised of (viz., [mean, std, median, mad, sum, max, min]) over phone segments, where the root-mean-square energy (RMSE) is employed to compute energy value for each time frame, with 25-millisecond windows and a stride of 10 milliseconds.

Duration Feature. The duration feature is a 1- dimensional vector indicating the length of each phone segment in seconds.