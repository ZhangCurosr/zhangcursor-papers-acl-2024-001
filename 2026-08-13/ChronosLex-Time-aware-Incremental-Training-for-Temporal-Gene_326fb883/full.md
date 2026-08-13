# ChronosLex: Time-aware Incremental Training for Temporal Generalization of Legal Classification Tasks

Santosh T.Y.S.S and Tuan-Quang Vuong and Matthias Grabmair

School of Computation, Information, and Technology

Technical University of Munich, Germany

{santosh.tokala, quang.vuong, matthias.grabmair}@tum.de

## Abstract

This study investigates the challenges posed by the dynamic nature of legal multi-label text classification tasks, where legal concepts evolve over time. Existing models often overlook the temporal dimension in their training process, leading to suboptimal performance of those models over time, as they treat training data as a single homogeneous block. To address this, we introduce ChronosLex, an incremental training paradigm that trains models on chronological splits, preserving the temporal order of the data. However, this incremental approach raises concerns about overfitting to recent data, prompting an assessment of mitigation strategies using continual learning and temporal invariant methods. Our experimental results over six legal multi-label text classification datasets reveal that continual learning methods prove effective in preventing overfitting thereby enhancing temporal generalizability, while temporal invariant methods struggle to capture these dynamics of temporal shifts.

## 1 Introduction

Legal classification tasks involve the assignment of legal documents, texts, or cases to specific legal categories, making them essential for legal practitioners, researchers, and organizations seeking efficient legal document management and analysis. However, the dynamic nature of legal domain, characterized by the intricate interplay of laws, precedents, and ever-evolving interpretation of existing jurisprudence, influenced by external forces such as real-world events and shifting societal norms, provides a distinctive challenge for these legal tasks specifically. In this backdrop of a dynamic and non-stationary world, temporal generalization of models emerges as a key necessity, given such systems are deployed to assist users with data that it encounters in the future, whilst being trained on data from the past.

![](images/d01a0994bb7eace098cffe17946b750ad81ab8d9d046188aa189012a63c58b44.jpg)  
Figure 1: Different evaluation and training strategies employed in this work, to assess and improve temporal generalization of models for legal classification tasks.

Most of the works dealing with legal classification often neglect time as a factor and simply split the data randomly into training, validation, and test sets (e.g., Paul et al. 2020; Malik et al. 2021; Zheng et al. 2021; Tuggener et al. 2020; Hendrycks et al. 2021; Papaloukas et al. 2021; Lippi et al. 2019; Pai˘ s<sub>,</sub> et al. 2021; Luz de Araujo et al. 2018; ¸Sulea et al. 2017 inter alia). Recent works have realized that this standard splitting criterion leads to overly optimistic performance estimates, which are far from realistic scenarios when such systems need to be deployed to assist with the new content over time and suffer from degradation of performance due to temporal misalignment i.e., difference in distributions of training and evaluation data instances. They advocate for chronological splitting of datasets to estimate the performance, accounting for the data drift over time (e.g., Chalkidis et al. 2021b, 2022, 2021a; Xu et al. 2023; Niklaus et al. 2021; Chalkidis and Søgaard 2022 inter alia). Furthermore, we posit that using a single fixed chronological test split (Eval-Fix, Fig. 1a) may yield misleading results if the chosen fixed split coincides with anomalous events, such as Brexit, Covid-19, lockdown, the Russo-Ukrainian War, or political regime changes, impacting one side of the splits more than the other. To address this concern, we utilize a streaming evaluation protocol (Eval-Stream, Fig. 1b) which involves evaluating the model over multiple splits involving distinct time periods which can lead to reliable conclusions about models’ robustness. In this protocol, we incorporate all past data into training and assess the model’s capacity to adapt to emerging trends in the next time period, which mirrors standard machine learning development pipelines, where models are regularly updated and redeployed to account for temporal drifts (Yao et al., 2022).

While legal texts are intrinsically tied to the time of their origin, making it imperative to account for these temporal aspects when developing classification models, these previous methods overlook the chronological context in which legal documents exist during the training process and consider the entire training data spanning across a considerable stretch of time, as a single homogeneous block (Fig. 1c). This oversight hinders model capabilities to adapt to the data drifts that naturally occur over time leading to performance degradation when evaluated over data from the future time (Gaspers et al., 2022; Rijhwani and Preo¸tiuc-Pietro, 2020).

For instance, shifts in societal attitudes towards fundamental rights like privacy, discrimination, and freedom of expression drive the evolution of legal concepts, broadening or contracting their scope over time. Influenced by technological advances and global events, national legal concepts expand to encompass emerging trends, as seen in security related matters to encompass artificial intelligence and cryptocurrency. Landmark cases can also introduce new civil/human rights, such as the recognition of environmental rights within the right to life. Legislative changes, such as developments in data protection laws keeping pace with technology, further shape concepts, reflecting evolving regulatory needs. Overall, data drifts in the legal domain can manifest gradually through the gradual evolution of legal concepts , reflecting societal changes, emerging legal trends, and real-world events or even abruptly through landmark legal decisions, overruling precedents, and legislative changes.

In response to these dynamic temporal drifts, we put forth a hypothesis which we subsequently validate: Models trained on data from a timeframe closer to the test set tend to yield superior results, under the assumption of the same model and dataset size. Motivated by this finding, and with the aim to capitalize on the whole training set — typically associated with better performance with more data, we introduce “ChronosLex”, an incremental training framework (Fig. 1d). In this approach, the model is systematically trained on data from one time period at a time before progressing to the next chronological split, thus preserving the temporal order of the data the model interacts with during training. Our experiments on six multi-label legal classification datasets demonstrate that incremental training paradigm improves temporal generalization of these models on three datasets but also leads to overfitting to recent data on other datasets.

To further improve the models’ robustness to address the overfitting issue, we assess (i) different families of continual learning algorithms such as regularization-based EWC (Kirkpatrick et al., 2017), rehearsal-based ER (Chaudhry et al., 2019), AGEM (Lopez-Paz and Ranzato, 2017), and parameter expansion based Adapters (Houlsby et al., 2019), LoRA (Hu et al., 2021) (ii) temporally invariant methods (Yao et al., 2022) derived from domain adaptation methods such as GroupDRO (Sagawa et al., 2019), DeepCORAL (Sun and Saenko, 2016), IRM (Arjovsky et al., 2019) where we treat consecutive windows of time as distinct domains. Our experimental results suggest that temporal invariant methods fail to learn these dynamics while continual learning methods effectively prevent models from overfitting to recent data, ultimately enhancing temporal generalization.

## 2 Related Work

Temporal Drift There has been increasing recent evidence that temporal misalignment results in degradation of model performance due to difference in distribution between the train and test data, caused due to temporal shifts (Jaidka et al., 2018; Yao et al., 2022; Gorman and Bedrick, 2019). These distribution shifts arise with the passage of time and are a very naturally occurring type of distribution shift due to non-stationarity and evolving world (Schlimmer and Granger, 1986; Widmer and Kubat, 1993). Temporal misalignment can be caused due to (1) the dynamic nature of language (Rosin et al., 2022; Röttger and Pierrehumbert, 2021; Loureiro et al., 2022; Agarwal and Nenkova, 2022; Amba Hombaiah et al., 2021; Rijhwani and Preo¸tiuc-Pietro, 2020; Luu et al., 2022; Jaidka et al., 2018) and (2) the update of factual information (Margatina et al., 2023; Jang et al., 2021, 2022; Lazaridou et al., 2021; Dhingra et al., 2022; Liska et al., 2022).

Temporal generalization has been explored both in upstream Language Model pre-training (Lazaridou et al., 2021; Loureiro et al., 2022; Jang et al., 2021, 2022; Dhingra et al., 2022; Jin et al., 2022; Amba Hombaiah et al., 2021) and in downstream tasks, such as sentiment analysis (Lukes and Sø- gaard, 2018; Agarwal and Nenkova, 2022; Guo et al., 2023b), named entity recognition (Rijhwani and Preo¸tiuc-Pietro, 2020; Onoe et al., 2022), question answering (Liska et al., 2022; Shang et al., 2022), headline generation (Søgaard et al., 2021), rumor/fake news detection (Mu et al., 2023; Hu et al., 2023), spoken language understanding (Gaspers et al., 2022). model explainability (Zhao et al., 2022), document classification (Röttger and Pierrehumbert, 2021; Chalkidis and Søgaard, 2022; Huang and Paul, 2018), abusive language detection (Jin et al., 2023; Florio et al., 2020), topic modeling (Zhang et al., 2023) and readmission prediction in the health care context (Guo et al., 2022, 2023a).

In this work, we assess temporal generalization in multi-label legal text classification, which is challenging given the complexities of managing both the multiplicity of labels and accommodating the evolving legal context over time. This is in contrast to prior studies that predominantly focused on single-label classification. In prior work on temporal generalization in multi-label legal text classification, Chalkidis and Søgaard 2022 postulated that temporal drift is primarily attributable to shifts in the label distribution over time and explored various group-robust optimization algorithms to mitigate disparities at the label level. We propose an alternative viewpoint, arguing that temporal drift in legal tasks extends beyond label distribution shifts to encompass shifts in the input text as well, driven by the fact that the vocabulary changes with time (Tab. 1). Furthermore, even vocabulary specific to each label undergoes temporal shifts (Tab 1). This complexity was not fully considered in Chalkidis and Søgaard 2022, where the entire training data was treated as a single unit, overlooking shifts in input text distribution over time. To address this, we introduce an incremental training framework which enables models to adapt to distribution shifts over time. We further improve its robustness by evaluating a range of continual learning and temporal invariant approaches for multi-label legal text classification.

Continual Learning Most research in continual learning (or lifelong learning) initially centered around computer vision tasks, and more recently, it has gained attention in the NLP field (Ke et al., 2021b,a; Biesialska et al., 2020; Ke and Liu, 2022; Sun et al., 2019; Wang et al., 2019). The majority of these works adopted traditional task-incremental, domain-incremental, or label-incremental settings. While our temporal incremental setting bears resemblance to the domain-incremental setting, a crucial distinction lies in the assumption of strict boundaries in domain incremental settings, which is not applicable to our temporal adaptation where the boundaries between drifts are blurred (Prabhu et al., 2020; Aljundi et al., 2019). Generally, contin ual learning algorithms can be categorized into (i) Rehearsal-based methods (Rolnick et al., 2019; Rebuffi et al., 2017), which maintain a memory buffer of older data to perform experience replay with actual data (de Masson D’Autume et al., 2019), automatically generated data (Sun et al., 2019), or previously computed gradients (Lopez-Paz and Ran zato, 2017), (ii) Regularization-based approaches (Kirkpatrick et al., 2017; Chen et al., 2020; Huang et al., 2021), which regularizes neural network parameters from drastic updates for new information to preserve the information of older ones, prevent ing overfitting to the newer data (iii) Parameter expansion methods (Qin et al., 2022; Gururangan et al., 2022; Yoon et al., 2018) which freeze net work architectures on older data and dynamically grow branches for newer ones. These methods have been explored in the pre-training stage to accom modate factual updates (Jang et al., 2021, 2022) or adaptation to new domains (Jin et al., 2022; Gururangan et al., 2022) and in the fine-tuning stage (Yao et al., 2022; Lin et al., 2022), which is closely related to our work, but within the context of single label classification tasks. To the best of our knowl edge, continual methods have never been explored for multi-label legal classification tasks.

Temporal invariant Domain Adaptation Domain adaptation (DA) primarily addresses the co-variate shift between source and target data distributions (Ruder, 2019). This involves learning domaininvariant feature representations that can generalize across different domains (Bollegala et al., 2011; Ganin et al., 2016). In the legal context, DA methods have been applied to tasks such as Legal Judgement Prediction, where reasoning with respect to each article is treated as a domain (Tyss et al., 2023) and legal text classification tasks, where each label is considered a domain to address label imbalance (Chalkidis and Søgaard, 2022). Traditional DA methods are typically applied to domains with clear distinctions, whereas temporal distribution shifts tend to occur gradually without well-defined boundaries, forming a continuous process. To adapt domain invariant methods to address temporal distribution shifts, we assume instances from consecutive windows of time to constitute a distinct domain, enabling the application of invariant learning approaches to these constructed domains (Yao et al., 2022). To the best of our knowledge, this is the first attempt to assess these methods for multi-label legal texts to handle temporal drift.

## 3 Evaluation Strategy

To assess temporal generalizability, following previous works (Søgaard et al., 2021; Chalkidis et al., 2022), we carry out evaluation using fixed chronological splits, referred to as Eval-Fix (Fig. 1a). In this setup, we split the entire data chronologically into train $( d _ { < t _ { 1 } } )$ , validation $( d _ { \geq t _ { 1 } \& \leq t _ { 2 } } )$ , and test $( d _ { > t _ { 2 } } )$ splits, where $d _ { t }$ denotes data from time period $t ,$ and $t _ { 1 } < t _ { 2 }$ . We train and validate the model using the respective splits and subsequently report its performance over the entire test set.

We posit that such a fixed split may not reflect a true picture as it can be influenced by the interplay between the splits and the data distribution where the anomalous/drift-driven events may overlap or primarily fall within one of the splits, leading to misleading conclusions about the model’s ability to handle temporal drifts. To mitigate the impact of dataset-specific traits on the splits, we adopt a streaming evaluation protocol, inspired by Yao et al. 2022 and Lopez-Paz and Ranzato 2017, which we refer to as Eval-Stream (Fig. 1b). We evaluate models using multiple splits, formed at distinct timestamps, considering all the data up to time period t $( d _ { \leq t } )$ for training, $d _ { t }$ for validation, and $d _ { t + 1 }$ as the test split. We focus only on the evaluation of the future data rather than the past because the practical need for performance over past data is limited. While it is possible to test the model on data from all subsequent time periods $( d _ { > t } )$ at each time period t, we restrict the testing to the following time period t + 1, to simplify the result analysis, aligning with practical scenarios where models are updated based on their performance in the next time iteration.

## 4 Datasets

We conduct our experiments on the six multi-label legal classification datasets from three sources.

UKLEX (Chalkidis and Søgaard, 2022) This dataset comprises legislative documents of the UK, publicly accessible through the National Archives, which are typically categorized into thematic areas such as healthcare, finance, education, transportation and planning, which are outlined in the document preamble and serve as indexes for archival purposes. The dataset contains 36.5k documents and is chronologically split into training (20k, 1975–2002), validation (8.5k, 2002–2008), and test (8.5k, 2009–2018) sets. The labels are provided at two distinct levels of granularity, encompassing 18 and 69 topics, referred to as Small (S) and Medium (M) respectively. For eval-fix, we use the above splits and for eval-stream, we consider a two-year time period as a unit. Due to limited data prior to 1990 and to ensure an adequate number of data instances for model learning, we report eval-stream performance from 1992.

EURLEX (Chalkidis et al., 2021a) This dataset contains EU legislation documents accessible via the EUR-Lex platform and is annotated using concepts from EuroVoc, a thesaurus maintained by the EU’s Publications Office. We work with the English portion of this dataset, consisting of 65k documents and split chronologically into training (55k, 1958–2010), validation (5k, 2010–2012), and test (5k, 2012–2015) sets which we use for eval-fix. The dataset provides four levels of label granularity. We use the first two levels of the EuroVoc taxonomy, as followed in Chalkidis and Søgaard 2022, encompassing 21 and 127 concepts, denoted as Small (S) and Medium (M), respectively. For eval-stream, we use two years as one unit and report performance from 1987 due to a lower number of instances in the initial years.

ECHR The dataset by (Chalkidis et al., 2019, 2021b) comprises cases heard by the European Court of Human Rights, which are publicly accessible via HUDOC, the official court database. These cases involve the adjudication of complaints by individuals against states for alleged violations of their rights as enshrined in the European Convention of Human Rights. Each case includes information about the convention articles that have been alleged to be violated and which the court has found to be violated. The dentification of alleged and violated articles are referred to as Task B and A, respectively, by Chalkidis et al. 2022. It consists of 11K cases, divided chronologically into training (9k, 2001–2016), validation (1k, 2016–2017), and test (1k, 2017–2019) sets for eval-fix. We use the 14 and 17 articles related to the core rights as followed in Valvoda et al. 2023 for Task A and B respectively. We use annual data as a unit for eval-stream and report the performance from 2004.

While task employed in this work are all effectively text classification, the tasks surveyed here are different in nature. The ECHR tasks represent legal determinations by judges in specific cases of high complexity and semantic depth, whereas the keyword classification tasks of EURLEX and UK-LEX represent semantically much shallower analysis. This has consequences for the types and complexity of temporal shifts one would observe. For example, the court changing its jurisprudence on a particular legal issue prompted by world events is a more complex phenomenon than the usage of particular European Law Database Keyword.

## 5 ChronosLex

Previous approaches for legal classification typically fine-tune models on the entire training dataset in a shuffled manner, treating all data as a single homogeneous block. We argue that such an approach neglects the temporal nature of the data which may be crucial to learn the temporal evolution of concepts. To address this limitation, we propose the ChronosLex framework, which considers the chronological context of data to capture and adapt models to the natural drifts over time. We hypothesize that recent data holds insights into upcoming distributional shifts and training models with this recent data enables better adaptation to temporal changes. Thus, we systematically train the model with data from one time period at a time before progressing to the next chronological split. Specifically, at time t, the model $m _ { t }$ is initialized with the model obtained from the previous timestamp $m _ { t - 1 }$ and fine-tuned on data $d _ { t }$ from time period t, progressively moving to the next time period $t + 1$ . We term this approach Incremental Fine-tuning (IFT) (Fig. 1d), where the model architecture and the loss function remain similar to traditional fine-tuning, but the model encounters chronological training data incrementally.

However, IFT may risk overfitting to recent data, potentially forgetting previously acquired knowledge which is crucial for extrapolating to the new distribution based on past data. Therefore, a balance between preserving knowledge from the past and accumulating information from recent data is essential to enhance temporal generalization. To explore this balance, we investigate continual learning methods and temporal invariant methods, to adapt to temporally emerging distribution shifts.

## 5.1 Continual Learning Methods

The aim of continual learning is to accumulate knowledge incrementally without forgetting information from previous steps referred to as catastrophic forgetting. In our specific context, we explore the efficacy of these methods in enabling models to extrapolate into the future based on past information, within the framework of a boundaryunaware, non-stationary temporal shift setting.

EWC (Kirkpatrick et al., 2017) Elastic Weight Consolidation is a regularization-based method which adds a temporal regularization term to the taskspecific actual loss so that the parameter changes from t  1 to t are restricted to avoid over-fitting. It produces a weighted penalty such that the parameters that are more important to the previous time stamp will have larger penalty weights, to balance the trade-off between previous knowledge and new knowledge. It uses Fisher Information Matrices to estimate the importance of parameters to use them as the weighted penalty.

ER (Rolnick et al., 2019) Experience Replay falls into the category of rehearsal-based methods that stores samples from previous time stamps into a growing memory module. We use instances from memory as additional training examples along with current time stamp instances.

A-GEM (Lopez-Paz and Ranzato, 2017) Average-Gradient Episodic Memory, a rehearsal method, leverages an episodic memory to store a sample of examples from previous time stamp similar to ER, but additionally equip the loss on older samples as inequality constraints, avoiding their increase.

LoRA (Hu et al., 2021) falls into the category of parameter-expansion method which freezes the original parameters of the pre-trained model and introduces trainable low-rank matrices and combines them with the original matrices in the multi-head attention and are updated during fine-tuning.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>x</td><td rowspan=1 colspan=1>x|</td><td rowspan=1 colspan=1>y</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Old</td><td rowspan=1 colspan=1>Rec.</td><td rowspan=1 colspan=1>Old</td><td rowspan=1 colspan=1>Rec.</td></tr><tr><td rowspan=1 colspan=1>UKLEX(S)</td><td rowspan=2 colspan=1>0.314</td><td rowspan=2 colspan=1>0.264</td><td rowspan=1 colspan=1>0.384</td><td rowspan=1 colspan=1>0.328</td></tr><tr><td rowspan=1 colspan=1>UKLEX(M)</td><td rowspan=1 colspan=1>0.412</td><td rowspan=1 colspan=1>0.366</td></tr><tr><td rowspan=1 colspan=1>EURLEX(S)</td><td rowspan=2 colspan=1>0.359</td><td rowspan=2 colspan=1>0.265</td><td rowspan=1 colspan=1>0.379</td><td rowspan=1 colspan=1>0.288</td></tr><tr><td rowspan=1 colspan=1>EURLEX(M)</td><td rowspan=1 colspan=1>0.409</td><td rowspan=1 colspan=1>0.324</td></tr><tr><td rowspan=1 colspan=1>ECHR(A)</td><td rowspan=2 colspan=1>0.236</td><td rowspan=2 colspan=1>0.155</td><td rowspan=1 colspan=1>0.306</td><td rowspan=1 colspan=1>0.247</td></tr><tr><td rowspan=1 colspan=1>ECHR(B)</td><td rowspan=1 colspan=1>0.318</td><td rowspan=1 colspan=1>0.266</td></tr></table>

Table 1: Jensen–Shannon divergence score between the split of training set (Old/Recent) and the test set over the vocabulary distribution (x) and vocabulary conditioned on the label (x y). Higher Score indicates more divergence from the test set distribution.

Adapters (Houlsby et al., 2019) is another parameter expansion method that freezes the original parameters of the pre-trained model and injects two small modules between the self-attention sub-layer and the feed-forward sub-layer inside each layer of transformer sequentially. The adapter module consists of a down-projection, an up-projection, and a nonlinear function between them with a residual connection across each module.

## 5.2 Temporal Invariant Methods

Domain invariant representation learning boosts model transferability across domains by eliminating domain-specific information, preventing overfitting to specific domains, and improving generalization to unseen target domains. In our case, we aim to build a temporally invariant model by excluding features tied to specific time periods. The difficulty of their application in our context lies in the lack of well-defined boundaries for temporal distribution shifts, with no explicit signal for drift awareness. To tackle this, we treat every sliding window of length L timestamps as a domain.

DeepCORAL (Sun and Saenko, 2016) Correlation Alignment penalizes the differences in the mean and covariance of the feature distributions of each domain to obtain domain invariant representations. IRM (Arjovsky et al., 2019) Invariant Risk Minimization aims to penalize variance across multiple training dummy estimators across domains, i.e., performance cannot vary across samples corresponding to the same domain.

GroupDRO (Sagawa et al., 2019) Group Distributionally Robust Optimization aims to optimize the worst-domain loss wherein the domain-wise losses are weighted inversely proportional to the performance of instances in that domain.

## 6 Experiments

## 6.1 Base Model & Metrics

For UKLEX and EURLEX, we use a state-of-theart model, BERT-LWAN (Chalkidis et al., 2020a), which has a label-wise attention network on top of the pre-trained model. Specifically, we pass the text into LegalBERT (Chalkidis et al., 2020b) and employ one attention head per label to generate N document representations where N denotes the number of labels, which are finally passed through a linear layer to get the final predictions. For ECHR, we employ a state-of-the-art hierarchical version of BERT due to long input documents (Chalkidis et al., 2022). Specifically, we pass each sentence in the long document into LegalBERT to obtain [CLS] representations which are then passed through a two-layer transformer to obtain final representations. Implementation details of all the methods can be found in Appendix A. Following Chalkidis and Søgaard 2022, we report, micro-F1, macro-F1, and mean R-Precision (m-RP).

## 6.2 Results on Eval-Fix Setting

## We present the results on eval-fix in Table 2.

Quantifying Temporal Shift: To understand the effect of temporal distance between the training and test data, we divide the training data into two versions: (i) the first half of chronologically sorted training data referred to as Old, and (ii) the latter half referred to as Recent. To quantify the temporal distribution shifts between these splits and the test set, we calculate the Jensen-Shannon divergence score between the distribution of the vocabulary (x). There is a possibility that the observed vocabulary distribution shift might be influenced by changes in label distribution over time. We further disentangle it by calculating conditional vocabulary distribution for each label (x y) and report the average of divergence scores across all labels. This specifically assesses how the vocabulary associated with each label changes over time. From Table 1, we observe that the recent split, which is temporally closer to the test data, has a lower divergence score compared to older split, confirming our hypothesis of temporal distribution drift over time. It is noteworthy that this may underestimate the effect, as we are specifically calculating the drift at the lexical level without capturing semantic shifts over time (changes in associated meaning or contextual usage of specific words).

To further elucidate the impact of this tempo-

$$
\overline { { \mathrm { m a c . ^ { - } F 1 } } }
$$

$$
\overline { { { \bf m } { \bf R P } } }
$$

$$
7 2 . 9 2 _ { 3 . 8 6 }
$$

$$
\overline { { \mathrm { m a c . ^ { - } F 1 } } }
$$

$$
7 6 . 0 2 _ { 4 . 1 3 }
$$

$$
7 8 . 2 6 { \mathrm { 3 . 5 5 } }
$$

$$
5 0 . 5 4 6 . 7 3
$$

$$
\overline { { { \bf m } { \bf R P } } }
$$

$$
\mathrm { \ m a c . ^ { - } F 1 }
$$

$$
6 7 . 3 4 _ { 1 . 3 6 }
$$

$$
6 2 . 7 2 _ { 2 . 3 7 }
$$

$$
5 9 . 1 6 _ { 0 . 6 6 }
$$

$$
7 2 . 1 9 \mathrm { _ { 0 . 9 9 } }
$$

$$
6 6 . 5 3 \mathrm { _ { 0 . 6 0 } }
$$

$$
7 3 . 2 6 _ { 1 . 6 7 }
$$

$$
7 7 . 2 0 _ { 2 . 8 3 }
$$

$$
7 9 . 3 4 _ { 2 . 4 7 }
$$

$$
5 2 . 3 8 _ { 8 . 9 6 }
$$

$$
6 9 . 8 4 _ { 1 . 3 3 }
$$

$$
6 6 . 0 1 _ { 1 . 4 2 }
$$

$$
6 4 . 0 8 _ { 0 . 3 6 }
$$

$$
7 6 . 2 0 \phantom { . } 8 5
$$

$$
7 1 . 6 2 _ { 0 . 2 7 }
$$

$$
7 4 . 9 6 _ { 3 . 4 4 }
$$

$$
8 0 . 6 1 _ { 2 . 1 4 }
$$

$$
7 8 . 6 6 _ { 1 . 6 6 }
$$

$$
5 4 . 8 9 _ { 1 2 . 3 4 }
$$

$$
7 0 . 8 5 _ { 2 . 4 2 }
$$

$$
6 7 . 2 5 _ { 0 . 5 5 }
$$

$$
6 7 . 7 0 _ { 1 . 0 1 }
$$

$$
7 8 . 5 7 _ { 0 . 9 7 }
$$

$$
7 3 . 4 6 _ { 0 . 2 8 }
$$

$$
7 8 . 6 5 _ { 2 . 9 2 }
$$

$$
8 2 . 2 9 _ { 1 . 9 1 }
$$

$$
8 0 . 2 0 \phantom { } _ { 1 . 7 8 }
$$

$$
5 5 . 1 8 _ { 1 1 . 7 9 }
$$

$$
\overline { { 7 2 . 5 7 _ { 1 . 4 5 } } }
$$

$$
6 9 . 3 4 _ { 2 . 6 5 }
$$

$$
6 6 . 2 8 _ { 0 . 9 1 }
$$

$$
7 7 . 7 1 _ { 1 . 2 3 }
$$

$$
7 2 . 4 8 _ { 0 . 6 3 }
$$

$$
8 0 . 1 5 _ { 1 . 7 4 }
$$

$$
8 3 . 8 0 _ { 2 . 1 7 }
$$

$$
8 2 . 4 1 _ { 2 . 3 7 }
$$

$$
5 5 . 6 6 _ { 8 . 9 1 }
$$

$$
7 4 . 7 9 _ { 2 . 3 0 }
$$

$$
\underline { { 7 2 . 2 2 } } . 8 1
$$

$$
6 8 . 0 1 _ { 0 . 6 6 }
$$

$$
7 8 . 6 9 _ { 0 . 7 0 }
$$

$$
7 3 . 6 9 _ { 0 . 2 0 }
$$

$$
\underline { { 8 0 . 2 6 } } _ { 1 . 7 7 }
$$

$$
\underline { { 8 3 . 8 2 . 8 3 } }
$$

$$
\underline { { 8 3 . 0 8 _ { 4 . 2 0 } } }
$$

$$
\underline { { 5 5 . 9 3 } } _ { 9 . 5 8 }
$$

$$
\underline { { 7 5 . 0 3 } } _ { 3 . 2 2 }
$$

$$
7 1 . 9 9 _ { 3 . 4 0 }
$$

$$
{ \bf 6 8 . 2 7 _ { 0 . 6 8 } }
$$

$$
\underline { { 7 8 . 8 4 _ { 0 . 8 3 } } }
$$

$$
\underline { { 7 4 . 0 2 } } _ { 0 . 4 3 }
$$

$$
7 9 . 1 3 _ { 1 . 6 9 }
$$

$$
8 3 . 2 5 _ { 2 . 1 7 }
$$

$$
8 1 . 3 9 _ { 2 . 4 4 }
$$

$$
5 5 . 3 2 _ { 9 . 4 4 }
$$

$$
7 3 . 8 4 _ { 2 . 8 4 }
$$

$$
7 0 . 8 1 _ { 4 . 0 3 }
$$

$$
6 7 . 8 0 _ { 0 . 8 6 }
$$

$$
7 8 . 6 0 _ { 0 . 8 2 }
$$

$$
7 3 . 2 7 _ { 0 . 3 4 }
$$

$$
8 0 . 0 3 _ { 1 . 9 5 }
$$

$$
8 3 . 6 7 _ { 2 . 7 8 }
$$

$$
8 2 . 5 8 _ { 2 . 7 9 }
$$

$$
5 5 . 7 7 _ { 9 . 9 5 }
$$

$$
7 4 . 5 4 _ { 3 . 8 9 }
$$

$$
7 1 . 1 5 _ { 4 . 4 5 }
$$

$$
\underline { { 6 8 . 0 5 } } _ { 0 . 1 1 }
$$

$$
{ \bf 7 9 . 0 2 } _ { 0 . 8 7 }
$$

$$
\mathbf { 7 4 . 1 4 } _ { 0 . 4 4 }
$$

$$
\mathbf { 8 0 . 6 4 } _ { 0 . 6 7 }
$$

$$
\mathbf { 8 4 . 1 5 } _ { 3 . 9 9 }
$$

$$
\mathbf { 8 3 . 1 3 _ { 4 . 1 1 } }
$$

$$
\mathbf { 5 6 . 2 7 _ { 1 0 . 8 4 } }
$$

$$
{ \bf 7 5 . 6 7 _ { 4 . 0 0 } }
$$

$$
\mathbf { 7 2 . 6 5 _ { 4 . 2 9 } }
$$

$$
6 7 . 9 6 _ { 0 . 6 8 }
$$

$$
7 8 . 8 0 _ { 1 . 1 7 }
$$

$$
7 3 . 5 7 _ { 0 . 6 5 }
$$

$$
7 5 . 5 1 _ { 3 . 6 6 }
$$

$$
\overline { { 8 0 . 4 0 _ { 2 . 5 4 } } }
$$

$$
\overline { { 7 7 . 6 7 _ { 1 . 5 2 } } }
$$

$$
5 0 . 5 8 _ { 8 . 2 6 }
$$

$$
6 5 . 0 3 _ { 2 . 3 9 }
$$

$$
6 4 . 7 1 _ { 1 . 9 3 }
$$

$$
\overline { { 7 0 . 2 5 _ { 2 . 4 6 } } }
$$

$$
7 4 . 9 8 _ { 2 . 2 1 }
$$

$$
7 7 . 8 2 _ { 4 . 7 3 }
$$

$$
8 0 . 6 9 _ { 0 . 8 8 }
$$

$$
7 0 . 0 5 _ { 1 . 8 1 }
$$

$$
7 8 . 9 8 _ { 0 . 6 5 }
$$

$$
5 2 . 1 1 _ { 1 0 . 0 8 }
$$

$$
6 4 . 1 5 _ { 0 . 0 4 }
$$

$$
6 2 . 4 7 _ { 2 . 4 2 }
$$

$$
7 0 . 2 9 \mathrm { _ 0 . 4 0 }
$$

$$
7 7 . 1 1 3 . 7 6 
$$

$$
7 5 . 4 6 _ { 2 . 7 7 }
$$

$$
8 0 . 7 3 \mathrm { _ { 1 . 1 1 } }
$$

$$
7 0 . 0 7 _ { 2 . 9 9 }
$$

$$
7 8 . 7 6 \phantom { 0 } _ { 0 . 7 7 }
$$

$$
5 1 . 4 5 _ { 1 0 . 0 9 }
$$

$$
7 0 . 0 6 _ { 1 . 1 6 }
$$

$$
6 4 . 9 8 _ { 1 . 4 3 }
$$

$$
6 3 . 5 7 _ { 0 . 6 8 }
$$

$$
7 6 . 9 1 _ { 0 . 9 6 }
$$

$$
7 1 . 4 0 \phantom { 0 } _ { 0 . 7 0 }
$$

## EURLEX(M)

$$
\overline { { \mathrm { E C H R } ( \mathrm { A } ) } }
$$

$$
\overline { { \mathrm { E C H R } ( \mathbf { B } ) } }
$$

$$
3 8 . 3 5 _ { 0 . 8 1 }
$$

$$
5 9 . 3 2 _ { 1 . 9 0 }
$$

$$
5 2 . 2 3 _ { 1 . 5 1 }
$$

$$
4 7 . 7 2 _ { 3 . 0 5 }
$$

$$
5 9 . 0 9 _ { 3 . 0 6 }
$$

$$
4 3 . 7 7 _ { 2 . 4 8 }
$$

$$
6 2 . 8 6 _ { 2 . 4 1 }
$$

$$
7 0 . 5 4 _ { 1 . 5 0 }
$$

$$
4 3 . 6 9 _ { 1 . 2 6 }
$$

$$
6 4 . 6 4 _ { 1 . 8 0 }
$$

$$
6 4 . 2 5 _ { 0 . 6 8 }
$$

$$
5 6 . 3 6 _ { 1 . 5 9 }
$$

$$
5 2 . 3 1 _ { 5 . 7 6 }
$$

$$
6 0 . 1 4 _ { 0 . 8 7 }
$$

$$
4 9 . 1 1 _ { 3 . 9 9 }
$$

$$
6 4 . 5 2 _ { 0 . 2 0 }
$$

$$
7 4 . 9 6 _ { 1 . 3 2 }
$$

$$
6 6 . 8 4 _ { 2 . 0 2 }
$$

$$
4 4 . 1 7 _ { 1 . 2 5 }
$$

$$
7 0 . 2 6 _ { 2 . 0 5 }
$$

$$
5 8 . 6 6 _ { 1 . 1 3 }
$$

$$
5 3 . 3 1 _ { 8 . 3 6 }
$$

$$
6 2 . 6 8 _ { 1 . 0 3 }
$$

$$
{ \bf 5 1 . 9 3 } _ { 1 . 0 7 }
$$

$$
6 6 . 9 5 _ { 0 . 6 1 }
$$

$$
{ \bf 7 5 . 2 7 _ { 0 . 1 8 } }
$$

$$
4 4 . 9 7 _ { 2 . 2 3 }
$$

$$
6 7 . 6 4 _ { 1 . 8 8 }
$$

$$
7 1 . 8 9 _ { 1 . 3 9 }
$$

$$
5 9 . 0 8 _ { 2 . 1 6 }
$$

$$
5 2 . 4 4 _ { 7 . 1 1 }
$$

$$
6 5 . 8 7 _ { 0 . 8 4 }
$$

$$
6 1 . 5 5 _ { 0 . 9 7 }
$$

$$
{ \bf 5 0 . 8 2 } _ { 0 . 7 9 }
$$

$$
{ \bf 7 5 . 1 3 _ { 2 . 4 3 } }
$$

$$
6 7 . 4 9 _ { 2 . 2 8 }
$$

$$
4 4 . 6 7 _ { 2 . 4 0 }
$$

$$
7 1 . 0 9 _ { 1 . 4 9 }
$$

$$
\mathbf { 5 6 . 2 5 } _ { 1 0 . 3 3 }
$$

$$
6 7 . 5 8 _ { 0 . 2 9 }
$$

$$
6 2 . 4 3 _ { 1 . 3 0 }
$$

$$
6 7 . 8 8 _ { 1 . 9 9 }
$$

$$
4 5 . 4 2 _ { 1 . 5 8 }
$$

$$
5 0 . 0 1 _ { 3 . 6 8 }
$$

$$
5 9 . 7 2 _ { 0 . 9 8 }
$$

$$
7 4 . 8 6 _ { 0 . 4 1 }
$$

$$
\underline { { 5 5 . 8 7 } } . 7 6
$$

$$
7 0 . 2 1 _ { 0 . 6 1 }
$$

$$
\underline { { 6 8 . 5 5 } } _ { 1 . 7 5 }
$$

$$
6 3 . 9 9 _ { 0 . 1 5 }
$$

$$
6 6 . 7 7 _ { 1 . 8 1 }
$$

$$
4 4 . 4 8 _ { 1 . 9 3 }
$$

$$
4 9 . 3 1 _ { 2 . 5 1 }
$$

$$
5 8 . 4 2 _ { 1 . 1 8 }
$$

$$
5 4 . 9 3 _ { 8 . 3 9 }
$$

$$
7 4 . 8 4 _ { 0 . 0 4 }
$$

$$
7 0 . 5 1 _ { 0 . 2 9 }
$$

$$
6 7 . 2 6 _ { 0 . 1 8 }
$$

$$
6 3 . 6 6 _ { 0 . 1 0 }
$$

$$
\mathbf { 6 8 . 7 5 } _ { 1 . 9 9 }
$$

$$
4 9 . 8 7 _ { 3 . 1 9 }
$$

$$
\mathbf { 4 7 . 4 3 } _ { 1 . 7 6 }
$$

$$
6 0 . 1 8 _ { 1 . 2 1 }
$$

$$
\underline { { 7 5 . 1 1 _ { 0 . 6 1 } } }
$$

$$
\underline { { 7 3 . 5 1 . 4 7 } }
$$

$$
5 3 . 7 6 5 . 8 9
$$

$$
6 6 . 8 2 _ { 0 . 3 8 }
$$

$$
\underline { { 4 6 . 1 5 } } _ { 1 . 3 4 }
$$

$$
\underline { { 6 8 . 5 7 } } 1 . 5 2 
$$

$$
6 2 . 9 4 _ { 1 . 0 1 }
$$

$$
\mathbf { 6 0 . 8 3 } _ { 1 . 2 4 }
$$

$$
5 0 . 3 1 _ { 2 . 3 1 }
$$

$$
7 4 . 3 7 _ { 0 . 7 2 }
$$

$$
5 2 . 8 4 _ { 3 . 7 6 }
$$

$$
\underline { { 6 6 . 3 0 1 . 2 4 } }
$$

$$
3 9 . 5 7 _ { 1 . 7 3 }
$$

$$
6 3 . 0 7 _ { 1 . 6 9 }
$$

$$
6 2 . 1 3 \mathrm { _ { 0 . 6 9 } }
$$

$$
5 5 . 4 1 _ { 1 . 7 3 }
$$

$$
{ \bf 7 3 . 5 7 _ { 1 . 9 7 } }
$$

$$
\underline { { 5 0 . 5 9 } } 1 . 1 5
$$

$$
7 4 . 7 9 \mathrm { _ { 0 . 1 6 } }
$$

$$
5 0 . 4 8 _ { 5 . 9 8 }
$$

$$
6 8 . 1 9 1 . 7 6 
$$

$$
6 5 . 1 2 _ { 1 . 4 5 }
$$

$$
4 0 . 6 6 _ { 1 . 0 5 }
$$

$$
5 7 . 5 5 1 . 1 5
$$

$$
6 8 . 9 4 _ { 2 . 1 3 }
$$

$$
\mathrm { 6 4 . 2 5 _ { 1 . 5 9 } }
$$

$$
7 0 . 7 7 _ { 2 . 4 1 }
$$

$$
4 8 . 5 7 _ { 2 . 2 9 }
$$

$$
5 1 . 6 2 _ { 2 . 9 1 }
$$

$$
{ \bf 6 8 . 5 7 } _ { 0 . 7 7 }
$$

$$
4 0 . 1 1 _ { 2 . 3 3 }
$$

$$
6 4 . 8 6 _ { 2 . 0 2 }
$$

$$
5 7 . 9 9 _ { 1 . 8 4 }
$$

$$
5 1 . 3 5 _ { 0 . 1 3 }
$$

$$
{ \bf 6 4 . 2 6 } _ { 3 . 6 9 }
$$

$$
7 0 . 1 9 _ { 1 . 4 8 }
$$

$$
7 3 . 8 7 _ { 0 . 1 1 }
$$

$$
6 8 . 3 2 _ { 3 . 6 2 }
$$

$$
4 9 . 9 6 _ { 1 . 6 0 }
$$

$$
6 3 . 1 6 _ { 2 . 6 8 }
$$

$$
4 9 . 2 4 _ { 2 . 6 4 }
$$

$$
7 1 . 3 5 _ { 1 . 5 3 }
$$

$$
6 9 . 9 7 _ { 1 . 5 5 }
$$

Table 2: Performance of different categories of methods on the Eval-Fix setting over six datasets. Best and Second best values in each metric is bolded and underlined respectively. Subscript refers to standard deviations.

ral drift on model performance, we evaluate the model on the eval-fix test set by training the baseline model using Old and Recent splits of training set. To remove a possible confounding effect of dataset size, each of these two models has access to the same number of training instances. As shown in Table 2, baseline-Recent consistently outperforms baseline-Old across all metrics and datasets. This result validates our hypothesis that models trained on temporally closer data to the test set tend to yield superior results, under the assumption of the same model and dataset size as we observed that temporally closer data deviates less from the test set in vocabulary distribution. Finally, we use the whole training data to create Baseline-Full model and it exhibits enhanced performance across all metrics and datasets, underscoring the effectiveness of a larger dataset to develop a temporally robust generalizable classifier.

serves the chronological order of the input dataset by training the model incrementally. This in turn assists the model in detecting drifts over time and adapting to them accordingly, simultaneously harnessing the whole dataset. However, this strategy may lead to overfitting to recent data, as evidenced by the lower performance in EURLEX(S) and ECHR(A,B). Mitigating this requires a mechanism to revisit older data to preserve their information while accumulating and adapting to newer trends/drifts.

Continual Learning Methods On UKLEX(S), all continual learning methods exhibit superior performance compared to both IFT and baseline approaches. Notably, Adapters and ER stand out, surpassing others, with EWC and LoRA following closely behind. AGEM shows comparatively lesser efficacy among them. This trend of continual learning methods outperforming IFT persists in the UKLEX(M) scenario, although the margin of improvement on macro-F1 scores is relatively narrow, given the larger number of labels in the medium (M) split coupled with high label imbalance.

Baseline vs. IFT From Table 2, we observe IFT performs better than Baseline-Full in UK-LEX(S,M) and EURLEX(M) but falls short in EU-RLEX(S) and ECHR(A,B). While baseline employs a traditional fine-tuning approach ignoring the temporal order of training dataset, IFT pre-

Transitioning to EURLEX(S), continual learning methods again outshine IFT and the baseline, with LoRA and ER taking the lead. As observed in UKLEX(S,M), AGEM falls short again. Surprisingly, Adapters, which excels on UKLEX(S,M), does not exhibit the same level of performance on EURLEX(S). On EURLEX(M), Adapters, LoRA, and ER outperform both the baseline and IFT, while the other methods remain comparable. AGEM, once again, demonstrates lower performance.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>ULKEX(S)</td><td rowspan=1 colspan=4>ULKEX(M)</td><td rowspan=1 colspan=2>EURLEX(S)</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1> $\overline { { \mathrm { m a c r o - F 1 } } }$ </td><td rowspan=1 colspan=1> $\overline { { \mathrm { m i c r o - F 1 } } }$ </td><td rowspan=1 colspan=1> $\overline { { { \bf m } { \bf - R P } } }$ </td><td rowspan=1 colspan=2> $\overline { { \mathrm { m a c r o - F 1 } } }$ </td><td rowspan=1 colspan=1>micro-F1</td><td rowspan=1 colspan=1> $\overline { { \mathrm { m } { \cdot } \mathrm { R P } } }$ </td><td rowspan=1 colspan=1> $\mathrm { m a c r o - F 1 }$ </td><td rowspan=1 colspan=1> $\overline { { \mathrm { m i c r o - F 1 } } }$ </td><td rowspan=1 colspan=1> $\overline { { \mathrm { { m } } \mathrm { { - } } \mathrm { { R P } } } }$ </td></tr><tr><td rowspan=1 colspan=1>Baseline</td><td rowspan=1 colspan=1> $7 8 . 9 7 _ { 4 . 3 3 }$ </td><td rowspan=1 colspan=1> $8 3 . 8 9 _ { 3 . 0 8 }$ </td><td rowspan=1 colspan=1> $8 3 . 4 2 _ { 3 . 6 0 }$ </td><td rowspan=1 colspan=2> $5 5 . 6 0 _ { 6 . 1 8 }$ </td><td rowspan=1 colspan=1> $7 5 . 8 7 _ { 3 . 4 3 }$ </td><td rowspan=1 colspan=1> $7 1 . 5 2 _ { 4 . 2 3 }$ </td><td rowspan=1 colspan=1> $6 4 . 6 5 6 . 3 5$ </td><td rowspan=1 colspan=1> $7 8 . 1 8 _ { 3 . 9 6 }$ </td><td rowspan=1 colspan=1> $7 3 . 0 5 5 . 4 0$ </td></tr><tr><td rowspan=1 colspan=1>IFT</td><td rowspan=1 colspan=1> $7 9 . 9 8 _ { 4 . 5 5 }$ </td><td rowspan=1 colspan=1> $8 4 . 9 1 _ { 3 . 1 2 }$ </td><td rowspan=1 colspan=1> $8 3 . 6 5 _ { 3 . 6 1 }$ </td><td rowspan=1 colspan=2> $5 6 . 1 5 _ { 6 . 2 6 }$ </td><td rowspan=1 colspan=1> $7 7 . 1 2 _ { 3 . 4 8 }$ </td><td rowspan=1 colspan=1> $7 3 . 0 3 _ { 4 . 3 3 }$ </td><td rowspan=1 colspan=1> $6 3 . 2 6 _ { 4 . 7 8 }$ </td><td rowspan=1 colspan=1> $7 7 . 9 9 _ { 3 . 2 5 }$ </td><td rowspan=1 colspan=1> $7 2 . 6 1 _ { 4 . 6 0 }$ </td></tr><tr><td rowspan=5 colspan=1>EWCERAGEMLORAAdapter</td><td rowspan=5 colspan=1> $8 1 . 4 2 _ { 5 . 2 1 }$  $8 1 . 5 1 _ { 4 . 7 6 }$  $\underline { { 8 1 . 7 8 _ { 5 . 4 1 } } }$  $\mathbf { 8 1 . 9 9 } _ { 3 . 7 6 }$  $8 1 . 1 4 _ { 5 . 4 6 }$ </td><td rowspan=2 colspan=1> $8 6 . 1 2 _ { 3 . 4 9 }$  $8 5 . 9 1 _ { 3 . 2 0 }$ </td><td rowspan=4 colspan=1> $8 5 . 1 5 _ { 4 . 2 6 }$  $8 5 . 1 4 _ { 3 . 8 7 }$  $\underline { { 8 5 . 2 3 } } _ { 4 . 2 6 }$  $\mathbf { 8 5 . 3 8 _ { 4 . 2 3 } }$ </td><td rowspan=4 colspan=2> $5 8 . 0 8 _ { 6 . 6 6 }$  $5 8 . 1 7 _ { 6 . 5 1 }$  ${ \bf 5 8 . 4 5 } _ { 6 . 9 4 }$  $5 8 . 3 1 _ { 7 . 1 5 }$ </td><td rowspan=1 colspan=1> $7 9 . 3 1 _ { 5 . 8 5 }$ </td><td rowspan=1 colspan=1> $7 5 . 2 3 _ { 6 . 8 3 }$ </td><td rowspan=1 colspan=1> $6 5 . 1 6 _ { 5 . 7 9 }$ </td><td rowspan=1 colspan=1> $7 8 . 6 4 _ { 3 . 7 4 }$ </td><td rowspan=1 colspan=1> $7 3 . 3 8 _ { 5 . 1 2 }$ </td></tr><tr><td rowspan=2 colspan=1> $\underline { { 7 9 . 3 8 _ { 5 . 6 4 } } }$  $7 9 . 2 6 _ { 5 . 6 4 }$ </td><td rowspan=1 colspan=1> ${ \bf 7 5 . 5 5 6 . 6 9 }$ </td><td rowspan=1 colspan=1> $6 4 . 9 5 _ { 6 . 0 3 }$ </td><td rowspan=1 colspan=1> $7 8 . 6 9 _ { 3 . 6 4 }$ </td><td rowspan=1 colspan=1> $7 3 . 3 1 _ { 5 . 0 8 }$ </td></tr><tr><td rowspan=3 colspan=1> $8 6 . 1 1 _ { 3 . 6 4 }$  $\mathbf { 8 6 . 1 9 } _ { 3 . 6 8 }$  $8 6 . 1 6 _ { 3 . 6 6 }$ </td><td rowspan=1 colspan=1> $7 5 . 2 2 _ { 6 . 8 4 }$ </td><td rowspan=1 colspan=1> $6 5 . 0 7 _ { 6 . 0 2 }$ </td><td rowspan=1 colspan=1> $7 8 . 6 7 _ { 3 . 7 6 }$ </td><td rowspan=1 colspan=1> $7 3 . 4 7 _ { 5 . 3 4 }$ </td></tr><tr><td rowspan=1 colspan=1>31715</td><td rowspan=1 colspan=1> $7 9 . 1 2 _ { 5 . 7 7 }$ </td><td rowspan=1 colspan=1> $7 5 . 0 3 _ { 6 . 8 1 }$ </td><td rowspan=1 colspan=1> $\mathbf { 6 5 . 5 6 } _ { 5 . 6 6 }$ </td><td rowspan=1 colspan=1> ${ \bf 7 8 . 8 6 _ { 3 . 5 3 } }$ </td><td rowspan=1 colspan=1> ${ \bf 7 3 . 6 1 _ { 4 . 8 8 } }$ </td></tr><tr><td rowspan=1 colspan=1> $8 5 . 1 7 _ { 4 . 3 1 }$ </td><td rowspan=1 colspan=2> $5 8 . 2 2 _ { 7 . 2 4 }$ </td><td rowspan=1 colspan=1> ${ \bf 7 9 . 7 2 _ { 5 . 4 5 } }$ </td><td rowspan=1 colspan=1> $\underline { { 7 5 . 5 4 } } 6 . 4 7$ </td><td rowspan=1 colspan=1> $6 4 . 8 2 _ { 6 . 2 5 }$ </td><td rowspan=1 colspan=1> $\underline { { 7 8 . 7 7 . 7 8 } }$ </td><td rowspan=1 colspan=1> $\underline { { 7 3 . 5 6 } } _ { 5 . 1 7 }$ </td></tr><tr><td rowspan=3 colspan=1>DeepCORALIRMGroupDRO</td><td rowspan=1 colspan=1> $7 7 . 5 1 _ { 3 . 6 1 }$ </td><td rowspan=1 colspan=1> $8 3 . 1 4 _ { 3 . 1 3 }$ </td><td rowspan=1 colspan=1> $8 1 . 6 0 _ { 4 . 2 0 }$ </td><td rowspan=1 colspan=2> $5 1 . 5 6 _ { 6 . 9 3 }$ </td><td rowspan=1 colspan=1> $7 4 . 9 6 _ { 4 . 2 6 }$ </td><td rowspan=1 colspan=1> $6 9 . 8 5 _ { 4 . 8 8 }$ </td><td rowspan=1 colspan=1> $5 7 . 3 1 _ { 2 . 3 4 }$ </td><td rowspan=1 colspan=1> $7 5 . 7 0 _ { 2 . 6 2 }$ </td><td rowspan=1 colspan=1> $6 9 . 4 3 _ { 3 . 8 5 }$ </td></tr><tr><td rowspan=2 colspan=1> $7 7 . 8 3 _ { 4 . 4 2 }$  $7 8 . 0 5 _ { 4 . 7 1 }$ </td><td rowspan=2 colspan=1> $8 3 . 4 8 _ { 2 . 9 2 }$  $8 3 . 5 6 _ { 4 . 3 3 }$ </td><td rowspan=1 colspan=1> $8 2 . 0 2 _ { 3 . 2 7 }$ </td><td rowspan=1 colspan=2> $5 2 . 9 4 _ { 6 . 9 7 }$ </td><td rowspan=1 colspan=1> $7 4 . 7 9 _ { 4 . 0 7 }$ </td><td rowspan=1 colspan=1> $7 0 . 4 5 _ { 4 . 4 2 }$ </td><td rowspan=1 colspan=1> $6 1 . 2 1 _ { 3 . 8 6 }$ </td><td rowspan=1 colspan=1> $7 6 . 6 8 _ { 2 . 8 9 }$ </td><td rowspan=2 colspan=1> $7 1 . 1 7 _ { 4 . 4 2 }$  $7 1 . 1 7 _ { 3 . 4 9 }$ </td></tr><tr><td rowspan=1 colspan=1> $8 2 . 8 6 _ { 4 . 5 3 }$ </td><td rowspan=1 colspan=2> $5 2 . 7 0 _ { 8 . 1 7 }$ </td><td rowspan=1 colspan=1> $7 4 . 3 0 _ { 5 . 5 8 }$ </td><td rowspan=1 colspan=1> $6 9 . 3 9 _ { 5 . 6 6 }$ </td><td rowspan=1 colspan=1> $6 1 . 8 8 _ { 3 . 1 3 }$ </td><td rowspan=1 colspan=1> $7 6 . 7 5 _ { 2 . 3 8 }$ </td></tr></table>

<table><tr><td></td><td colspan="3">EURLEX(M)</td><td colspan="3"> $\overline { { \mathrm { E C H R } ( \mathrm { A } ) } }$ </td><td colspan="3">ECHR(B)</td></tr><tr><td>Baseline</td><td> $3 5 . 5 9 \mathrm { { 9 . 4 5 } }$ </td><td> $6 6 . 1 6 \phantom { 0 } . 1 6$ </td><td> $5 8 . 9 9 _ { 7 . 6 1 }$ </td><td> $5 1 . 3 8 _ { 6 . 7 8 }$ </td><td> $6 9 . 3 0 _ { 3 . 6 6 }$ </td><td> $6 3 . 9 6 _ { 3 . 3 3 }$ </td><td> $4 7 . 2 2 _ { 4 . 1 7 }$ </td><td> $7 4 . 8 6 _ { 2 . 0 8 }$ </td><td> $6 7 . 5 6 _ { 3 . 1 0 }$ </td></tr><tr><td>IFT</td><td> $3 4 . 6 1 _ { 9 . 2 7 }$ </td><td> $6 5 . 8 9 _ { 6 . 1 1 }$ </td><td> $5 8 . 1 5 _ { 7 . 4 5 }$ </td><td> $4 9 . 8 6 _ { 6 . 5 2 }$ </td><td> $6 8 . 2 7 _ { 3 . 5 6 }$ </td><td> $6 3 . 0 8 _ { 3 . 3 4 }$ </td><td> $4 6 . 7 3 _ { 4 . 1 0 }$ </td><td> $7 4 . 2 7 _ { 2 . 0 7 }$ </td><td> $6 7 . 0 9 _ { 3 . 1 7 }$ </td></tr><tr><td>EWC</td><td> $3 8 . 6 5 \mathrm { _ { 7 . 1 5 } }$ </td><td> $\overline { { 6 6 . 8 7 _ { 6 . 3 3 } } }$ </td><td> $5 9 . 8 1 _ { 7 . 8 5 }$ </td><td> ${ \bf 5 2 . 7 9 } _ { 7 . 1 7 }$ </td><td> $7 0 . 2 3 _ { 4 . 4 7 }$ </td><td> $6 5 . 0 5 _ { 4 . 6 4 }$ </td><td> $4 6 . 6 0 _ { 5 . 3 9 }$ </td><td> $7 5 . 0 1 _ { 2 . 2 7 }$ </td><td> $6 8 . 1 5 _ { 3 . 3 2 }$ </td></tr><tr><td>ER</td><td> $3 9 . 6 1 _ { 7 . 2 7 }$ </td><td> $6 6 . 8 8 _ { 6 . 3 0 }$ </td><td> $5 9 . 9 2 _ { 7 . 6 6 }$ </td><td> $5 1 . 8 2 _ { 6 . 7 8 }$ </td><td> $7 0 . 1 8 _ { 4 . 1 6 }$ </td><td> $6 4 . 9 8 _ { 4 . 0 3 }$ </td><td> $\underline { { 4 7 . 4 6 _ { 5 . 3 3 } } }$ </td><td> $7 4 . 9 1 _ { 2 . 4 2 }$ </td><td> $\mathbf { 6 8 . 4 4 } _ { 3 . 2 9 }$ </td></tr><tr><td>AGEM</td><td> $3 8 . 2 7 _ { 7 . 6 4 }$ </td><td> $6 6 . 7 3 _ { 6 . 3 3 }$ </td><td> $5 9 . 5 7 _ { 7 . 9 5 }$ </td><td> $5 1 . 5 3 _ { 6 . 7 4 }$ </td><td> $6 9 . 8 6 _ { 4 . 2 2 }$ </td><td> $6 5 . 1 2 _ { 3 . 9 4 }$ </td><td> $4 6 . 7 0 _ { 4 . 7 0 }$ </td><td> $\underline { { 7 5 . 0 5 } } _ { 1 . 9 4 }$ </td><td> $6 8 . 1 2 . 7 6$ </td></tr><tr><td>LORA</td><td> $\mathbf { 4 0 . 3 9 } _ { 7 . 7 0 }$ </td><td> $\underline { { 6 7 . 1 6 } } _ { 6 . 5 5 }$ </td><td> ${ \bf 6 0 . 1 6 } _ { 7 . 9 2 }$ </td><td> $5 2 . 5 4 _ { 6 . 0 1 }$ </td><td> $7 0 . 4 3 _ { 4 . 1 5 }$ </td><td> $6 5 . 7 4 _ { 3 . 6 2 }$ </td><td> $4 7 . 0 9 _ { 5 . 1 6 }$ </td><td> ${ \bf 7 5 . 1 3 _ { 2 . 6 5 } }$ </td><td> $\underline { { 6 8 . 3 9 _ { 2 . 9 3 } } }$ </td></tr><tr><td>Adapter</td><td> $3 8 . 5 7 _ { 7 . 7 8 }$ </td><td> ${ \bf 6 7 . 6 3 } _ { 6 . 8 5 }$ </td><td> $\underline { { 6 0 . 1 4 } } 8 . 2 6 $ </td><td> $5 2 . 3 5 6 . 7 3$ </td><td> $7 0 . 0 4 _ { 4 . 3 6 }$ </td><td> $6 4 . 9 9 _ { 4 . 2 0 }$ </td><td> $\mathbf { 4 7 . 6 4 } _ { 5 . 6 0 }$ </td><td> $7 4 . 7 8 _ { 2 . 4 5 }$ </td><td> $6 7 . 8 3 _ { 3 . 2 6 }$ </td></tr><tr><td>DeepCORAL</td><td> $3 0 . 5 0 _ { 6 . 4 4 }$ </td><td> $6 0 . 6 2 _ { 5 . 7 5 }$ </td><td> $5 3 . 1 4 _ { 6 . 6 6 }$ </td><td> $\overline { { 4 5 . 4 8 _ { 3 . 6 5 } } }$ </td><td> $\overline { { 7 2 . 1 9 _ { 1 . 9 4 } } }$ </td><td> $6 6 . 9 5 _ { 2 . 3 4 }$ </td><td> $\overline { { 4 0 . 8 1 _ { 5 . 8 6 } } }$ </td><td> $7 0 . 2 7 _ { 3 . 1 4 }$ </td><td> $\overline { { 6 5 . 3 7 _ { 3 . 4 5 } } }$ </td></tr><tr><td>IRM</td><td> $3 4 . 8 2 _ { 8 . 3 8 }$ </td><td> $6 2 . 7 6 _ { 5 . 4 7 }$ </td><td> $5 5 . 3 6 _ { 6 . 5 7 }$ </td><td> $4 8 . 7 4 _ { 4 . 6 1 }$ </td><td> ${ \bf 7 6 . 3 2 } _ { 2 . 7 1 }$ </td><td> ${ \bf 6 9 . 7 2 _ { 3 . 2 1 } }$ </td><td> $4 5 . 8 2 _ { 4 . 2 3 }$ </td><td> $7 3 . 4 2 _ { 2 . 3 4 }$ </td><td> $6 8 . 3 3 _ { 2 . 3 5 }$ </td></tr><tr><td>GroupDRO</td><td> $3 4 . 6 9 _ { 7 . 0 0 }$ </td><td> $6 3 . 9 9 _ { 5 . 3 7 }$ </td><td> $5 6 . 3 6 _ { 6 . 3 8 }$ </td><td> $4 6 . 0 2 _ { 5 . 5 0 }$ </td><td> $\underline { { 7 4 . 7 5 . 1 1 } }$ </td><td> $6 8 . 7 5 _ { 4 . 9 0 }$ </td><td> $4 5 . 3 0 _ { 2 . 5 8 }$ </td><td> $7 2 . 9 3 _ { 3 . 9 2 }$ </td><td> $6 6 . 6 8 _ { 5 . 9 6 }$ </td></tr></table>

Table 3: Aggregated performance of various method categories in the Eval-Stream setting across six datasets. Best and Second best values in each metric is bolded and underlined respectively. Subscript refers to standard deviations.

temporal invariant methods outperform baseline but fall short compared to IFT. However, they struggle to cross the baseline in UKLEX(M), EU-RLEX(S,M), and ECHR(B) datasets. While on ECHR(A), they outperform IFT and even continual methods on micro-F1 and m-RP but fall short on macro-F1 metric. We speculate that the objective of temporal invariant methods aims to eliminate the distribution by learning generalizable feature representation, while the adaptive objective for the continual methods helps effectively to track and adapt to the shifts.

Shifting focus to ECHR(A), EWC, ER, and AGEM outperform the baseline and IFT. Notably, Adapters exhibits the least favorable performance in this context. On ECHR(B), none of the continual learning methods succeed in surpassing IFT and baseline on micro- and macro-F1 scores. However, AGEM, LoRA, and Adapters outperform them on the m-RP metric. The underperformance on ECHR(B) can be attributed to the spurious correlations in the dataset, as highlighted in Santosh et al. 2022. These spurious attributes, inadvertently learned during training, persist over time, posing a challenge for the model to effectively unlearn them. Continual learning, while adept at adapting to evolving information, faces difficulties in fully addressing and mitigating these learned artifacts.

In summary, these findings collectively suggest that applying continual learning methods on top of incremental fine-tuning leads to improved performance in 5 out of 6 datasets. This underscores their capacity to adapt to natural shifts that occur over time, while also preserving past knowledge which assists them to extrapolate trends into the future from historical distribution.

## 6.3 Results on Eval-Stream Setting

Among these methods, overall AGEM falls short in most of the datasets and we attribute it to its design of stricter inequality constraint giving the model less freedom to accumulate new knowledge. Temporal Invariant Methods On UKLEX(S),

Under the Eval-Stream setting, we assess performance through multiple splits, conducting evaluations on each subsequent timestamp, as detailed in Sec. 3. The performance visualization over the multiple splits for each method is presented in Appendix B. We report the averaged results across all the splits for each method in Table 3.

IFT vs. Baseline IFT performs better than baseline on UKLEX(S,M) only and underperforms on the other four datasets. This observed trend aligns with our eval-fix analysis, due to IFT’s strong reliance on recent data, potentially leading to overfitting.

Continual Learning Methods All the continual Learning methods perform consistently better than IFT and baseline, over all the metrics across all the six datasets. This contrasts with our conclusions drawn in the eval-fix setting, where we did not observe such uniform performance improvement across all the methods. This discrepancy could be attributed to our reliance on conclusions drawn from a single test split in eval-fix, which may exhibit a distinct pattern of shift. These findings emphasize the importance of adopting a streaming evaluation protocol in datasets where distribution shifts occur naturally over time. Rather than relying solely on one split, which may lead to misleading conclusions about changes in performance and method efficacy, using streaming evaluation ensures more reliable and comprehensive insights.

Among the continual learning methods, LoRA demonstrates superior temporal robustness which can be attributed to its design of additional weight matrices. This design allows LoRA to effectively discern and assimilate additional drift compared to previous knowledge, accumulating it through additional parameters and thereby preserving older knowledge. Following closely, ER emerges as the next best performer, emphasizing the importance of revisiting past knowledge to guide the model away from heavy reliance on recent data. AGEM falls short and can be attributed to its stricter inequality constraint, as discussed before.

Temporal Invariant Methods All these approaches underperform compared to the baseline. Among them, IRM performs comparably to baseline, followed closely by GroupDRO. Notably, DeepCORAL (DC) consistently lags behind across all datasets. This can be attributed to the rationale behind these approaches. While IRM and Group-DRO employ penalties on loss functions so that performance cannot vary across samples of different time periods, DC directly minimizes the difference between feature representations across samples of different time periods. This design causes models trained with DC to suppress distribution shifts rather than actively learn and adapt to recent drifts.

## 7 Conclusion

In this study, we delve into temporal drift in legal multi-label text classification tasks, revealing a crucial oversight in the current model training process that treats the training data as a single, homogeneous block. This neglect results in the degradation of performance over time. To remedy this, we introduce ChronosLex, an incremental training paradigm that allows the model to interact with data chronologically, facilitating adaptation to temporal shifts. However, we observe a potential overfitting to recent data with this incremental approach. To address overfitting, we explore mitigation strategies using both continual learning and temporal invariant approaches and found that Continual algorithms exhibit promising results by leveraging historical distribution to extrapolate trends into the future. In contrast, temporal invariant methods prove less effective in tackling this. Finally, we advocate for streaming evaluation protocol with multiple time splits to draw reliable conclusions, especially when time is involved as a critical axis. In future, we plan to evaluate this incremental strategy across other legal tasks such as prior case retrieval, contract analysis, legal document summarization, regulatory compliance and legal argumentation, where the temporal aspect is crucial in modeling.

## Limitations

While this study provides valuable insights for addressing temporal challenges in legal multi-label text classification tasks, it is important to acknowledge certain limitations.

Firstly, the experiments and findings are based on six multi-label legal classification datasets. The extent to which the proposed approaches generalize across different legal domains or other text classification tasks remains to be explored. Further, the paper assumes the homogeneity of data within each time split, and the effectiveness of the proposed incremental training paradigm may vary based on the degree of diversity and complexity within each temporal split. Furthermore, while the streaming evaluation protocol is advocated for its reliability, its application to a broader range of datasets and tasks might introduce additional computational overhead. Investigating the protocol’s feasibility and performance in diverse settings is essential for establishing its practicality. Lastly, the paper addresses temporal drift in legal tasks, but a more detailed exploration of the specific causes and characteristics of temporal drift within legal domains could enhance the understanding of the challenges involved.

While the paper recognizes the existence of temporal drift, delving deeper into the specific characteristics of temporal shifts in legal text data (e.g., gradual vs. abrupt shifts) and their impact on model performance could contribute to more nuanced insights into the temporal dynamics at play. Unless one is explicitly examining the effect of a particular watershed event on the legal system and is informed by substantial relevant legal expertise, attributing large legal datasets to underlying factors accounting to inherent concept drift may turn out to be infeasible and we reserve that as a potential avenue for future work. For instance, in ECtHR jurisprudence, one would need to find multiple near-identical cases with different outcomes. Finding those is hard and, if they were found, they would be of great interest to legal scholarship.

Additionally, this work assumes a pre-fixed static splits, following a standard deployment scenario, where the model is updated based on frequency. In prior work on the US Supreme Court (Katz et al., 2017), for example, a change in the court’s presiding judge marks natural boundaries at the suprayear level. However, such segmentation requires in-depth knowledge of the domain and collection. The challenge of finding an appropriate time split and can we have it determined dynamically, is difficult and would possibly require a qualitative exploration of how identifiable shifts in the data coincide with an expert’s intuition about the natural temporal segmentation of the collection.

## Ethics Statement

The legal text classification datasets used in this study are sourced from publicly available repositories and produced by previous studies. Though the judgment corpus from ECHR is not anonymized and contains the real names of the involved parties, we do not foresee any harm incurred by our experiments. The intention behind this research is to provide a nuanced understanding of the temporal challenges inherent in legal text classification tasks. The findings are intended to benefit legal practitioners, researchers, and organizations involved in legal document management by enhancing the adaptability and performance of classification models in the dynamic legal landscape.

## References

Oshin Agarwal and Ani Nenkova. 2022. Temporal effects on pre-trained models for language processing tasks. Transactions ofthe Associationfor Computational Linguistics, 10:904–921.

Rahaf Aljundi, Min Lin, Baptiste Goujaud, and Yoshua Bengio. 2019. Gradient based sample selection for online continual learning. Advances in neural information processing systems, 32.

Spurthi Amba Hombaiah, Tao Chen, Mingyang Zhang, Michael Bendersky, and Marc Najork. 2021. Dynamic language models for continuously evolving content. In Proceedings ofthe 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, pages 2514–2524.

Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz. 2019. Invariant risk minimization. arXiv preprint arXiv:1907.02893.

Magdalena Biesialska, Katarzyna Biesialska, and Marta R Costa-jussà. 2020. Continual lifelong learning in natural language processing: A survey. In Proceedings of the 28th International Conference on Computational Linguistics, pages 6523–6541.

Danushka Bollegala, Yutaka Matsuo, and Mitsuru Ishizuka. 2011. Relation adaptation: learning to extract novel relations with minimum supervision. In Twenty-Second International Joint Conference on Artificial Intelligence.

Ilias Chalkidis, Ion Androutsopoulos, and Nikolaos Aletras. 2019. Neural legal judgment prediction in english. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4317–4323.

Ilias Chalkidis, Manos Fergadiotis, and Ion Androutsopoulos. 2021a. Multieurlex-a multi-lingual and multi-label legal document classification dataset for zero-shot cross-lingual transfer. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6974–6996.

Ilias Chalkidis, Manos Fergadiotis, Sotiris Kotitsas, Prodromos Malakasiotis, Nikolaos Aletras, and Ion Androutsopoulos. 2020a. An empirical study on largescale multi-label text classification including few and zero-shot labels. arXiv preprint arXiv:2010.01653.

Ilias Chalkidis, Manos Fergadiotis, Prodromos Malakasiotis, Nikolaos Aletras, and Ion Androutsopoulos. 2020b. Legal-bert: The muppets straight out of law school. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2020, pages 2898–2904.

Ilias Chalkidis, Manos Fergadiotis, Dimitrios Tsarapatsanis, Nikolaos Aletras, Ion Androutsopoulos, and Prodromos Malakasiotis. 2021b. Paragraph-level rationale extraction through regularization: A case study on european court of human rights cases. In Proceedings of the 2021 Conference of the North

American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 226–241.

Ilias Chalkidis, Abhik Jana, Dirk Hartung, Michael Bommarito, Ion Androutsopoulos, Daniel Katz, and Nikolaos Aletras. 2022. Lexglue: A benchmark dataset for legal language understanding in english. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4310–4330.

Ilias Chalkidis and Anders Søgaard. 2022. Improved multi-label classification under temporal concept drift: Rethinking group-robust algorithms in a labelwise setting. In Findings ofthe Associationfor Computational Linguistics: ACL 2022, pages 2441–2454.

Arslan Chaudhry, Marcus Rohrbach, Mohamed Elhoseiny, Thalaiyasingam Ajanthan, Puneet K Dokania, Philip HS Torr, and Marc’Aurelio Ranzato. 2019. On tiny episodic memories in continual learning. arXiv preprint arXiv:1902.10486.

Sanyuan Chen, Yutai Hou, Yiming Cui, Wanxiang Che, Ting Liu, and Xiangzhan Yu. 2020. Recall and learn: Fine-tuning deep pretrained language models with less forgetting. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7870–7881.

Cyprien de Masson D’Autume, Sebastian Ruder, Lingpeng Kong, and Dani Yogatama. 2019. Episodic memory in lifelong language learning. Advances in Neural Information Processing Systems, 32.

Bhuwan Dhingra, Jeremy R Cole, Julian Martin Eisenschlos, Daniel Gillick, Jacob Eisenstein, and William W Cohen. 2022. Time-aware language models as temporal knowledge bases. Transactions ofthe Associationfor Computational Linguistics, 10:257– 273.

Komal Florio, Valerio Basile, Marco Polignano, Pierpaolo Basile, and Viviana Patti. 2020. Time of your hate: The challenge of time in hate speech detection on social media. Applied Sciences, 10(12):4180.

Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand, and Victor Lempitsky. 2016. Domain-adversarial training of neural networks. The journal of machine learning research, 17(1):2096– 2030.

Judith Gaspers, Anoop Kumar, Greg Ver Steeg, and Aram Galstyan. 2022. Temporal generalization for spoken language understanding. In Proceedings of the 2022 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies: Industry Track, pages 37–44.

Kyle Gorman and Steven Bedrick. 2019. We need to talk about standard splits. In Proceedings of the conference. Association for Computational Linguistics.

Meeting, volume 2019, page 2786. NIH Public Access.

Lin Lawrence Guo, Stephen R Pfohl, Jason Fries, Alistair EW Johnson, Jose Posada, Catherine Aftandilian, Nigam Shah, and Lillian Sung. 2022. Evaluation of domain generalization and adaptation on improving model robustness to temporal dataset shift in clinical medicine. Scientific reports, 12(1):2726.

Lin Lawrence Guo, Ethan Steinberg, Scott Lanyon Fleming, Jose Posada, Joshua Lemmon, Stephen R Pfohl, Nigam Shah, Jason Fries, and Lillian Sung. 2023a. Ehr foundation models improve robustness in the presence of temporal distribution shift. Scientific Reports, 13(1):3767.

Yue Guo, Chenxi Hu, and Yi Yang. 2023b. Predict the future from the past? on the temporal data distribution shift in financial sentiment classifications. arXiv preprint arXiv:2310.12620.

Suchin Gururangan, Mike Lewis, Ari Holtzman, Noah A Smith, and Luke Zettlemoyer. 2022. Demix layers: Disentangling domains for modular language modeling. In Proceedings of the 2022 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 5557–5576.

Dan Hendrycks, Collin Burns, Anya Chen, and Spencer Ball. 2021. Cuad: An expert-annotated nlp dataset for legal contract review. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1).

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. 2019. Parameter-efficient transfer learning for nlp. In International Conference on Machine Learning, pages 2790–2799. PMLR.

Beizhe Hu, Qiang Sheng, Juan Cao, Yongchun Zhu, Danding Wang, Zhengjia Wang, and Zhiwei Jin. 2023. Learn over past, evolve for future: Forecasting temporal trends for fake news detection. arXiv preprint arXiv:2306.14728.

Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. 2021. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Xiaolei Huang and Michael Paul. 2018. Examining temporality in document classification. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 694–699.

Yufan Huang, Yanzhe Zhang, Jiaao Chen, Xuezhi Wang, and Diyi Yang. 2021. Continual learning for text classification with information disentanglement based

regularization. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2736–2746.

Kokil Jaidka, Niyati Chhaya, and Lyle Ungar. 2018. Diachronic degradation of language models: Insights from social media. In Proceedings ofthe 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 195–200.

Joel Jang, Seonghyeon Ye, Changho Lee, Sohee Yang, Joongbo Shin, Janghoon Han, Gyeonghun Kim, and Minjoon Seo. 2022. Temporalwiki: A lifelong benchmark for training and evaluating ever-evolving language models. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 6237–6250.

Joel Jang, Seonghyeon Ye, Sohee Yang, Joongbo Shin, Janghoon Han, KIM Gyeonghun, Stanley Jungkyu Choi, and Minjoon Seo. 2021. Towards continual knowledge learning of language models. In International Conference on Learning Representations.

Mali Jin, Yida Mu, Diana Maynard, and Kalina Bontcheva. 2023. Examining temporal bias in abusive language detection. arXiv preprint arXiv:2309.14146.

Xisen Jin, Dejiao Zhang, Henghui Zhu, Wei Xiao, Shang-Wen Li, Xiaokai Wei, Andrew Arnold, and Xiang Ren. 2022. Lifelong pretraining: Continually adapting language models to emerging corpora. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4764–4780.

Daniel Martin Katz, Michael J Bommarito, and Josh Blackman. 2017. A general approach for predicting the behavior of the supreme court of the united states. PloS one, 12(4):e0174698.

Zixuan Ke and Bing Liu. 2022. Continual learning of natural language processing tasks: A survey. arXiv preprint arXiv:2211.12701.

Zixuan Ke, Bing Liu, Hao Wang, and Lei Shu. 2021a. Continual learning with knowledge transfer for sentiment classification. In Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2020, Ghent, Belgium, September 14–18, 2020, Proceedings, Part III, pages 683–698. Springer.

Zixuan Ke, Hu Xu, and Bing Liu. 2021b. Adapting bert for continual learning of a sequence of aspect sentiment classification tasks. In Proceedings ofthe 2021 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, pages 4746–4755.

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu,

Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. 2017. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13):3521–3526.

Angeliki Lazaridou, Adhi Kuncoro, Elena Gribovskaya, Devang Agrawal, Adam Liska, Tayfun Terzi, Mai Gimenez, Cyprien de Masson d’Autume, Tomas Kocisky, Sebastian Ruder, et al. 2021. Mind the gap: Assessing temporal generalization in neural language models. Advances in Neural Information Processing Systems, 34:29348–29363.

Bill Yuchen Lin, Sida I Wang, Xi Lin, Robin Jia, Lin Xiao, Xiang Ren, and Scott Yih. 2022. On continual model refinement in out-of-distribution data streams. In Proceedings of the 60th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 3128–3139.

Marco Lippi, Przemysław Pałka, Giuseppe Contissa, Francesca Lagioia, Hans-Wolfgang Micklitz, Giovanni Sartor, and Paolo Torroni. 2019. Claudette: an automated detector of potentially unfair clauses in online terms of service. Artificial Intelligence and Law, 27:117–139.

Adam Liska, Tomas Kocisky, Elena Gribovskaya, Tayfun Terzi, Eren Sezener, Devang Agrawal, D’Autume Cyprien De Masson, Tim Scholtes, Manzil Zaheer, Susannah Young, et al. 2022. Streamingqa: A benchmark for adaptation to new knowledge over time in question answering models. In International Conference on Machine Learning, pages 13604–13622. PMLR.

David Lopez-Paz and Marc’Aurelio Ranzato. 2017. Gradient episodic memory for continual learning. Advances in neural information processing systems, 30.

Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101.

Daniel Loureiro, Francesco Barbieri, Leonardo Neves, Luis Espinosa Anke, and Jose Camacho-Collados. 2022. Timelms: Diachronic language models from twitter. In Proceedings of the 60th Annual Meeting ofthe Associationfor Computational Linguistics: System Demonstrations, pages 251–260.

Jan Lukes and Anders Søgaard. 2018. Sentiment analysis under temporal shift. In Proceedings of the 9th workshop on computational approaches to subjectivity, sentiment and social media analysis, pages 65–71.

Kelvin Luu, Daniel Khashabi, Suchin Gururangan, Karishma Mandyam, and Noah A Smith. 2022. Time waits for no one! analysis and challenges of temporal misalignment. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 5944–5958.

Pedro Henrique Luz de Araujo, Teófilo E de Campos, Renato RR de Oliveira, Matheus Stauffer, Samuel Couto, and Paulo Bermejo. 2018. Lener-br: a dataset for named entity recognition in brazilian legal text. In Computational Processing ofthe Portuguese Language: 13th International Conference, PROPOR 2018, Canela, Brazil, September 24–26, 2018, Proceedings 13, pages 313–323. Springer.

Vijit Malik, Rishabh Sanjay, Shubham Kumar Nigam, Kripabandhu Ghosh, Shouvik Kumar Guha, Arnab Bhattacharya, and Ashutosh Modi. 2021. Ildc for cjpe: Indian legal documents corpus for court judgment prediction and explanation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4046–4062.

Katerina Margatina, Shuai Wang, Yogarshi Vyas, Neha Anna John, Yassine Benajiba, and Miguel Ballesteros. 2023. Dynamic benchmarking of masked language models on temporal concept drift with multiple views. In Proceedings ofthe 17th Conference of the European Chapter of the Association for Computational Linguistics, pages 2873–2890.

Yida Mu, Kalina Bontcheva, and Nikolaos Aletras. 2023. It’s about time: Rethinking evaluation on rumor detection benchmarks using chronological splits. In Findings of the Association for Computational Linguistics: EACL 2023, pages 724–731.

Joel Niklaus, Ilias Chalkidis, and Matthias Stürmer. 2021. Swiss-judgment-prediction: A multilingual legal judgment prediction benchmark. In Proceedings ofthe Natural Legal Language Processing Workshop 2021, pages 19–35.

Yasumasa Onoe, Michael Zhang, Eunsol Choi, and Greg Durrett. 2022. Entity cloze by date: What lms know about unseen entities. In Findings ofthe Association for Computational Linguistics: NAACL 2022, pages 693–702.

Vasile Pai˘ s, Maria Mitrofan, Carol Luca Gasan, Vlad Coneschi, and Alexandru Ianov. 2021. Named entity recognition in the romanian legal domain. In Proceedings of the Natural Legal Language Processing Workshop 2021, pages 9–18.

Christos Papaloukas, Ilias Chalkidis, Konstantinos Athinaios, Despina Pantazi, and Manolis Koubarakis. 2021. Multi-granular legal topic classification on greek legislation. In Proceedings of the Natural Legal Language Processing Workshop 2021, pages 63– 75.

Shounak Paul, Pawan Goyal, and Saptarshi Ghosh. 2020. Automatic charge identification from facts: A few sentence-level charge annotations is all you need. In Proceedings of the 28th International Conference on Computational Linguistics, pages 1011–1022.

Jonas Pfeiffer, Andreas Rücklé, Clifton Poth, Aishwarya Kamath, Ivan Vulic, Sebastian Ruder, Kyunghyun´ Cho, and Iryna Gurevych. 2020. Adapterhub: A framework for adapting transformers. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 46–54.

Ameya Prabhu, Philip HS Torr, and Puneet K Dokania. 2020. Gdumb: A simple approach that questions our progress in continual learning. In Computer Vision– ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16, pages 524–540. Springer.

Yujia Qin, Jiajie Zhang, Yankai Lin, Zhiyuan Liu, Peng Li, Maosong Sun, and Jie Zhou. 2022. Elle: Efficient lifelong pre-training for emerging data. In Findings ofthe Associationfor Computational Linguistics: ACL 2022, pages 2789–2810.

Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, and Christoph H Lampert. 2017. icarl: Incremental classifier and representation learning. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pages 2001–2010.

Shruti Rijhwani and Daniel Preo¸tiuc-Pietro. 2020. Temporally-informed analysis of named entity recognition. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 7605–7617.

David Rolnick, Arun Ahuja, Jonathan Schwarz, Timothy Lillicrap, and Gregory Wayne. 2019. Experience replay for continual learning. Advances in Neural Information Processing Systems, 32.

Guy D Rosin, Ido Guy, and Kira Radinsky. 2022. Time masking for temporal language models. In Proceedings ofthe Fifteenth ACM International Conference on Web Search and Data Mining, pages 833–841.

Paul Röttger and Janet Pierrehumbert. 2021. Temporal adaptation of bert and performance on downstream document classification: Insights from social media. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 2400–2412.

Sebastian Ruder. 2019. Neural transfer learning for natural language processing. Ph.D. thesis, NUI Galway.

Shiori Sagawa, Pang Wei Koh, Tatsunori B Hashimoto, and Percy Liang. 2019. Distributionally robust neural networks. In International Conference on Learning Representations.

Tyss Santosh, Shanshan Xu, Oana Ichim, and Matthias Grabmair. 2022. Deconfounding legal judgment prediction for european court of human rights cases towards better alignment with experts. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 1120–1138.

Jeffrey C Schlimmer and Richard H Granger. 1986. Incremental learning from noisy data. Machine learning, 1:317–354.

Chao Shang, Guangtao Wang, Peng Qi, and Jing Huang. 2022. Improving time sensitivity for question answering over temporal knowledge graphs. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8017–8026.

Anders Søgaard, Sebastian Ebert, Jasmijn Bastings, and Katja Filippova. 2021. We need to talk about random splits. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 1823–1832.

Octavia-Maria ¸Sulea, Marcos Zampieri, Mihaela Vela, and Josef van Genabith. 2017. Predicting the law area and decisions of french supreme court cases. In Proceedings ofthe International Conference Recent Advances in Natural Language Processing, RANLP 2017, pages 716–722.

Baochen Sun and Kate Saenko. 2016. Deep coral: Correlation alignment for deep domain adaptation. In Computer Vision–ECCV 2016 Workshops: Amsterdam, The Netherlands, October 8-10 and 15- 16, 2016, Proceedings, Part III 14, pages 443–450. Springer.

Fan-Keng Sun, Cheng-Hao Ho, and Hung-Yi Lee. 2019. Lamol: Language modeling for lifelong language learning. In International Conference on Learning Representations.

Don Tuggener, Pius Von Däniken, Thomas Peetz, and Mark Cieliebak. 2020. Ledgar: A large-scale multilabel corpus for text classification of legal provisions in contracts. In Proceedings ofthe Twelfth Language Resources and Evaluation Conference, pages 1235– 1241.

Santosh Tyss, Oana Ichim, and Matthias Grabmair. 2023. Zero-shot transfer of article-aware legal outcome classification for european court of human rights cases. In Findings of the Association for Computational Linguistics: EACL 2023, pages 593–605.

Josef Valvoda, Ryan Cotterell, and Simone Teufel. 2023. On the role of negative precedent in legal outcome prediction. Transactions of the Association for Computational Linguistics, 11:34–48.

Hong Wang, Wenhan Xiong, Mo Yu, Xiaoxiao Guo, Shiyu Chang, and William Yang Wang. 2019. Sentence embedding alignment for lifelong relation extraction. In Proceedings ofNAACL-HLT, pages 796– 806.

Gerhard Widmer and Miroslav Kubat. 1993. Effective learning in dynamic environments by explicit context tracking. In Machine Learning: ECML-93: European Conference on Machine Learning Vienna, Austria, April 5–7, 1993 Proceedings 6, pages 227–243. Springer.

Shanshan Xu, Leon Staufer, Oana Ichim, Corina Heri, Matthias Grabmair, et al. 2023. Vechr: A dataset for explainable and robust classification of vulnerability type in the european court of human rights. arXiv preprint arXiv:2310.11368.

Huaxiu Yao, Caroline Choi, Bochuan Cao, Yoonho Lee, Pang Wei W Koh, and Chelsea Finn. 2022. Wildtime: A benchmark of in-the-wild distribution shift over time. Advances in Neural Information Processing Systems, 35:10309–10324.

Jaehong Yoon, Eunho Yang, Jeongtae Lee, and Sung Ju Hwang. 2018. Lifelong learning with dynamically expandable networks. In International Conference on Learning Representations.

Yuji Zhang, Jing Li, and Wenjie Li. 2023. Vibe: Topicdriven temporal adaptation for twitter classification. arXiv preprint arXiv:2310.10191.

Zhixue Zhao, George Chrysostomou, Kalina Bontcheva, and Nikolaos Aletras. 2022. On the impact of temporal concept drift on model explanations. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2022, pages 4039–4054.

Lucia Zheng, Neel Guha, Brandon R Anderson, Peter Henderson, and Daniel E Ho. 2021. When does pretraining help? assessing self-supervised learning for law and the casehold dataset of 53,000+ legal holdings. In Proceedings of the eighteenth international conference on artificial intelligence and law, pages 159–168.

## A Implementation Details

We will release our code upon acceptance.

Baseline Models We fine-tune the models on our six datasets using binary cross-entropy loss and they are trained end-to-end using AdamW optimizer (Loshchilov and Hutter, 2017), learning rate of 2e-5, momentum of 0.9, and weight decay of 0.01. For ECHR tasks, to account for longer input length, we employ hierarchical models which can process 64 paragraphs each with 128 tokens. We use a batch size of 60 for UKLEX and EURLEX tasks, and 6 for ECHR tasks due to memory constraints. The models are trained for up to 20 epochs using early stopping with a patience of 3 on macro-F1 score of the validation set to prevent the model. We use 16-bit automatic mixed precision and gradient accumulation to accelerate training and save memory. All the experiments were performed on a GPU cluster with NVIDIA A40 48GB PCIe 4.0. IFT models At each time stamp t, we initialize the model with the best-performing model from the previous timestamp (t  1). A warm-up phase of 3 epochs is employed before applying the patience threshold. This initial warm-up phase allows the model to interact with the data in the current timestamp, providing ample time to discern drifts and adapt accordingly. The remaining hyperparameters are kept consistent with the baseline models, and the best model at each time stamp is selected based on the macro-F1 score on the validation set.

Evaluation Metrics: For UKLEX and EURLEX, we employ the standard approach to compute macro-F1, micro-F1, and m-RP scores on the list of labels. However, for ECHR, an additional label is introduced during inference to account for instances that do not violate (allege) any article. In Task A (B), where there are 14 (17) articles, this extra label is included in both targets and predictions and its value is set to 1 if none of the labels are 1 (indicating that all the labels are 0), following the methodology outlined by Chalkidis et al. (2022).

We use Wild-Time library (Yao et al., 2022) for implementing continual learning algorithms such as EWC, ER, AGEM, and temporal-invariant methods such as GroupDRO, DeepCORAL, and IRM. We use AdapterHub framework (Pfeiffer et al., 2020) to implement LoRA and Adapters.

## A.1 Continual Learning methods specific Hyper-parameters

EWC (Kirkpatrick et al., 2017) We use the online version of EWC training to avoid memory overflow in the computation of Fischer information matrices, as detailed in (Kirkpatrick et al., 2017). We set λ of 0.5 which controls the strength of regularizationbased EWC loss with a decay term for older data γ set to default of 1.0.

ER (Rolnick et al., 2019) We replay a mini-batch of data by random sampling from an evolving past data module after every 10 training steps for UK-LEX and EURLEX tasks, and every 30 training steps for ECHR tasks.

AGEM (Lopez-Paz and Ranzato, 2017) stores parameter gradients of data from past timestamps and uses reservoir strategy to sample the past to add to the memory buffer. We use a maximum size of 1000 for the memory buffer for rehearsal.

LoRA (Hu et al., 2021) We set the low-dimensional rank r of the decomposition matrix to 8 and the α which accounts for scaling the reparameterization to 16.

Adapters (Houlsby et al., 2019) We used bottleneck adapters with a reduction factor of 16 which controls the ratio between the hidden dimension and the bottleneck dimension.

## A.2 Temporal Invariant learning methods specific Hyper-parameters

For all the temporal invariant methods, we sample uniformly from each time period in the window, when treated as a single domain to apply domain invariant approaches to boundary-less temporal setting. We employ sliding windows of length 5. We set the number of domains to be considered to be 3.

DeepCORAL (Sun and Saenko, 2016) We found the model is highly sensitive to λ which controls the penalty on alignment loss due to the challenging nature of its direct effect on feature representations. We performed an exhaustive search over the range and found 0.001 to work well in our setting. We compute alignment penalties between features from all pairs of domains sampled.

IRM (Arjovsky et al., 2019) We set IRM penalty factor λ to be constant at 1.0.

GroupDRO (Sagawa et al., 2019) Each instance in the minibatch is sampled with uniform probabilities from the entire domain.

## B Evaluation

## B.1 Eval-Fix

We present the performance of each method for every time period within the fixed test split of the Eval-Fix settings across all datasets in Figure 2. Notably, we observe a consistent improvement of methods across all timestamps in the test split, particularly evident in the EURLEX(S,M) and UK-LEX(S,M) datasets where the line plots corresponding to each method appear parallel over the years. However, this trend is not uniformly observed in the ECHR(A,B) datasets. We posit that this deviation may be attributed to the extreme sparsity of the dataset, causing different methods to excel on different subsets, thereby influencing performance based on the distribution of labels in that time stamp. While a declining trend over the years has been observed in most datasets, it is expected due to the distribution shift between the training and testing sets as the temporal distance increases with the test year. However, some datasets, like UKLEX(S), exhibit a ’V’-shaped trend. This observation prompts future investigations to discern the reasons behind such trends and to study the underlying shift or drift phenomena contributing to this unique pattern. Unraveling these intricacies can serve as motivation for future research to devise shift-specific mitigation strategies tailored to address the nuanced temporal dynamics present in these datasets.

## B.2 Eval-Stream

We evaluate the models on multiple splits and present the corresponding evaluation scores for subsequent time period at each split in Figure 3. The consistent trend of uniform improvements across methods on UKLEX(S,M) and EURLEX(S,M), as well as non-uniform improvements on ECHR(A,B), is mirrored in the streaming setting as well. This non-uniform nature can prompt interesting case studies to decipher what models might be learning and leverage this information to design strategies for improving legal text classification models.

![](images/8648bcac7d1b38a048505eec224b6bbb4a94483c094085d6abed1f07965333f8.jpg)

![](images/28394976c3e10a99db63715053fe6cf4497c22d0843ba7ed28b8f393d709c73c.jpg)  
(a) UKLEX(S)

![](images/de9986ac0966127e5ed9aa837fea0245f8487563d296c1e556341f9653fdd4e3.jpg)

![](images/feaab65c196dde63b0397c802b7cf229f7f02435f48c9fe193effbf5cada76d9.jpg)  
(b) UKLEX(M)

![](images/8ee7d3b6c421fdec89e27137c790664ff7e155dbc7f515ca2f25fa0a24cd4837.jpg)

![](images/c4279c7c3ba232391cce37ae9d039ca414a433116d232e3c38696c0c271f5a49.jpg)

(c) EURLEX(S)  
![](images/7feb6f861bab7bc5b1ac4bd945d57fc3f345b9b587b613067109e13ff49902d0.jpg)

![](images/34556d8fe079bc11097cfdab8d6f7c203852c16289f9426994204d8c4d1e81ba.jpg)  
(d) EURLEX(M)

![](images/d443a9d8e0246061983f86b96e4132f42b8158595b9037d3a720e961e1397973.jpg)

![](images/7c6b29d4c65eab2e008e9d34a1c32cb540aa97b4e9ff8693b08f64e8743b6d86.jpg)  
(e) ECHR(A)

![](images/68edda6f406227903c34242e5ed552a3397914477e67943a644552657e4bb65e.jpg)

![](images/b1aaa7fcf170df8f75892a506e4d113cfdf7292b2e4fd5dbbe7b42d301944707.jpg)

![](images/e3a2b88cd8a5f900ed31009006db2e82be5087a452944eaf1c4ea47a95bad684.jpg)

![](images/ce23cf560bb0f7e1110b814d16b9353a7ccab54fad765c3d20ca99265d0d6e79.jpg)

(f) ECHR(B)  
![](images/965055d50e74697fa18eee7b416eee3a2e88ce7e4a0d3454b6a211fd5feb85f2.jpg)

![](images/65302cdd5626a272ea3d74f547ca13c6e22f3bc89cb4922df012636551ccd2d3.jpg)

![](images/a8cabb1be350908abfcf2cb003617ea4076b613e4762783236d7076323549d2c.jpg)

![](images/b67030d04b8453b9f1fcadd9d2f96172e10afb6ddf6c0799255e676fe539d272.jpg)  
Figure 2: Results on Eval-Fix Setting over six datasets.

![](images/04edf467fdca8699953c0089e91aba3baea563772159cf902ba95d4f2f0f9dcd.jpg)

![](images/bd1a43040b3d337498f90f75e44e52550a21504e26e0b2d325427feb14e0434a.jpg)  
(a) UKLEX(S)

![](images/72b48fa97da42e9c919b7295b56162ead52065fdc8290c9149540690a5961bc4.jpg)

![](images/3e055935d85aeb981c7d43afd3117bd54ead0fbf38acf40f08a480a79b3291fa.jpg)

![](images/64f6fb8243dd1cadd9f993e44524fb66073ac8b965748ad59a95e5ebe01219fe.jpg)

![](images/673c57f66793ebcf0862eae151318ac0d20452584218845ee2bbe9479b8a8dd8.jpg)  
(b) UKLEX(M)

![](images/6f12883c0c8d6b47019fcb01c5b247e95cf30fe305d965310248d0782f3cb139.jpg)

![](images/6ceecf68e92285f181542ca027ceac9e6a88ee29e59c5a0f44a4bc7145a0352d.jpg)  
(c) EURLEX(S)

![](images/cc3bdcd2b4ebf314f5b5dfc84c06e0975048e2aaf7109c20822b38d250694ea4.jpg)

![](images/6efa0902cc17230c3d569bc7381476408ce8f5842e1dcd28cd4dda7896364b0e.jpg)

![](images/ec94b52a043914aef45816a4830b63c91422c8eaeac4073a30e5426f223110a4.jpg)  
(d) EURLEX(M)

![](images/395d8d973568cbd8ef0f98f982809039b3fd5a3e8998348e1d696a40122aef0f.jpg)

![](images/c97fffa58d50af99da856c75b031bd49b8138d0a3a1e330a3ad36443c49cd330.jpg)  
(e) ECHR(A)

![](images/988a38ff9cff7448bcd725944039e3d7cfb1d909f7c09597c4bfa9eb14e36ada.jpg)

![](images/27ab2656c4348ad29af08136772c1efad842ccc47751e5df31e78901c0998f19.jpg)

![](images/b745bd8e354d3ec56be4ab90e0bef8da11f768c151c050078d1b9e99db4da3b7.jpg)

![](images/a82f4b7803ce100c940a8269794f4709f8503910f3fa7f0fea7b3136a52d14fd.jpg)  
(f) ECHR(B)

![](images/118429e713a88463a4edf4d8611b33ae5b3e77422d76d12ec1a75233fda3dafc.jpg)  
Figure 3: Results on Eval-Stream Setting over six datasets.