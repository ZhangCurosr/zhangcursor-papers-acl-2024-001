# A Non-autoregressive Generation Framework for End-to-End Simultaneous Speech-to-Any Translation

Zhengrui Ma<sup>1,3</sup>, Qingkai Fang<sup>1,3</sup>, Shaolei Zhang<sup>1,3</sup>, Shoutao Guo<sup>1,3</sup> Yang Feng<sup>1,2,3</sup>\*, Min Zhang<sup>4</sup>

<sup>1</sup>Key Laboratory of Intelligent Information Processing Institute of Computing Technology, Chinese Academy of Sciences <sup>2</sup>Key Laboratory of AI Safety, Chinese Academy of Sciences <sup>3</sup>University of Chinese Academy of Sciences <sup>4</sup>School of Future Science and Engineering, Soochow University {mazhengrui21b,fengyang}@ict.ac.cn zhangminmt@hotmail.com

## Abstract

Simultaneous translation models play a crucial role in facilitating communication. However, existing research primarily focuses on text-totext or speech-to-text models, necessitating additional cascade components to achieve speechto-speech translation. These pipeline methods suffer from error propagation and accumulate delays in each cascade component, resulting in reduced synchronization between the speaker and listener. To overcome these challenges, we propose a novel non-autoregressive generation framework for simultaneous speech translation (NAST-S2x<sup>1</sup>), which integrates speechto-text and speech-to-speech tasks into a unified end-to-end framework. We develop a nonautoregressive decoder capable of concurrently generating multiple text or acoustic unit tokens upon receiving fixed-length speech chunks. The decoder can generate blank or repeated tokens and employ CTC decoding to dynamically adjust its latency. Experimental results show that NAST-S2x outperforms state-of-theart models in both speech-to-text and speechto-speech tasks. It achieves high-quality simul taneous interpretation within a delay of less than 3 seconds and provides a 28 decoding speedup in offline generation.<sup>2</sup>

## 1 Introduction

Simultaneous machine translation (Cho and Esipova, 2016; Gu et al., 2017; Raffel et al., 2017; Ma et al., 2019; Arivazhagan et al., 2019) models are widely applied in communication scenarios, eliminating barriers between individuals with different linguistic backgrounds. In practice, simultaneous translation systems can be broadly categorized into speech-to-text (Simul-S2T; Ma et al., 2020c) and speech-to-speech (Simul-S2S; Zheng et al., 2020) variants. Regardless of the modality of output, simultaneous translation models initiate generation before receiving the complete input to maintain synchrony between the listener and speaker. This necessitates models to delicately balance between translation quality and latency.

![](images/3887b56cb5f9a5f445840b93035e876a8ba2c4dc86efa333c3116af38de3c998.jpg)  
Figure 1: NAST-S2x can perform both Simul-S2T and Simul-S2S tasks within a unified end-to-end framework. The model generates speech output directly without the need to produce intermediate target text tokens

Most research on simultaneous machine translation primarily focuses on either text-to-text (Ma et al., 2020d; Miao et al., 2021) or speech-to-text models (Tang et al., 2023; Zhang and Feng, 2023b), necessitating additional cascaded components such as streaming automatic speech recognition (Chiu and Raffel, 2018; Zhang et al., 2020) and incremental text-to-speech synthesis (Ma et al., 2020a) for achieving speech-to-speech interpretation (Zheng et al., 2020). However, pipeline methods often suffer from error propagation and delay accumulation. The intermediate texts serve as information bottlenecks, hindering subsequent cascade components from accessing the original information and correcting errors. Moreover, each component operates with independent streaming strategies, resulting in cumulative delays thus diminishing synchronization between the speaker and listener. Given these challenges, the emergence of end-to-end Simul-S2S models has garnered increasing attention in the research community.

Recent success of end-to-end offline speech-tospeech translation (Offline-S2S) has paved the way for the development of end-to-end Simul-S2S models. Particularly, Lee et al. (2022) construct a direct speech-to-unit model (S2UT), which predicts selfsupervised discrete representations of target speech. Waveforms are subsequently generated using a separate unit-based vocoder (Polyak et al., 2021). On this basis, Ma et al. (2022) builds the first endto-end Simul-S2S model by introducing a variational version of monotonic multihead attention (Ma et al., 2020d). However, previous works are mainly limited to predicting units in an autoregressive manner, which is suboptimal for end-to-end Simul-S2S models. Considering that the acoustic unit sequence is 25 times longer than the corresponding text sequence on average, autoregressive unit prediction often leads to issues such as hallucination or truncation (Seamless Communication et al., 2023b). Moreover, the sequential prediction of long unit sequences imposes a significant computational time overhead, making it impractical for delay-sensitive Simul-S2S systems. To tackle these challenges, our focus is on developing a non-autoregressive end-to-end Simul-S2S model, aiming for enjoying the merits of an end-to-end system without the necessity of intermediate text decoding, while benefiting from the efficiency inherent in non-autoregressive generation.

In this work, we propose a non-autoregressive generation framework for end-to-end simultaneous speech-to-any translation (NAST-S2x). Inspired by recent advances in non-autoregressive generation (Shao and Feng, 2022; Ma et al., 2023), we develop a non-autoregressive decoder capable of concurrently generating multiple text or acoustic unit tokens upon receiving each fixed-length speech chunk. The entire generation adopts a chunk-tochunk approach, while avoiding the unstable expected training method (Zhang and Feng, 2023b). The model can produce blank or repeated tokens and perform CTC decoding (Graves et al., 2006) to adjust its latency dynamically. Considering the difficulty of the speech translation task and aiming to leverage intermediate text data to assist training, we further introduce a two-step glancing and a multi-task non-monotonic training strategy, which largely enhances the translation performance while maintaining the end-to-end nature of our model.

Extensive experiments highlight the superiority of our NAST-S2x. In Simul-S2T, its performance is on par with state-of-the-art models. In Simul-S2S, it significantly surpasses cascade Simul-S2T + TTS baselines, achieving high-quality simultaneous interpretation within a delay of less than 3 seconds. In Offline-S2S, it matches the performance of the strong autoregressive baseline while providing a 28 inference speedup.

## 2 Preliminaries

## 2.1 Simultaneous Speech Translation

Simultaneous speech translation models often process a streaming sequence of acoustic features $\pmb { x } = \{ x _ { 1 } , . . . , x _ { m } \}$ as input, extracted from speech samples every $T _ { w }$ ms. Simultaneous translation models can be further categorized into speechto-text (Simul-S2T) and speech-to-speech (Simul-S2S) variants based on the output modality.

## 2.1.1 Simul-S2T

A Simul-S2T model generates a translated text sequence $\pmb { y } = \{ y _ { 1 } , . . . , y _ { n } \}$ in a streaming fashion. To quantify the extent of source information taken into account during the generation, a monotonic nondecreasing function $g ( t )$ is introduced to represent the number of observed frames when generating $y _ { t }$

To assess the latency of Simul-S2T models, Ma et al. (2020c) introduce a modified version of average lagging (AL; Ma et al., 2019) for speech-to-text task. They measure the lagging based on time instead of steps, and the metric is defined as:

$$
A L = \frac { 1 } { \tau ( \left| \mathbf { x } \right| ) } \sum _ { t = 1 } ^ { \tau ( \left| x \right| ) } d ( t ) - \frac { \left| x \right| } { \left| y ^ { * } \right| } \cdot T _ { w } \cdot ( t - 1 ) ,\tag{1}
$$

where $| x |$ and $| { \boldsymbol y } ^ { * } |$ represent the lengths of source frames and reference text. $\tau ( | \pmb { x } | )$ is the index of the first generated token when the source is complete, and $d ( t )$ is the delay of generating $y _ { t } .$ . Ma et al. (2020c) further defines computation-aware and noncomputation-aware versions of $d ( t )$ . The former, $d _ { C A } ( t )$ , is defined as the elapsed time from the beginning of the whole process, while the latter is simply calculated as $d _ { N C A } ( t ) = g ( t ) \cdot T _ { w }$ . As the non-computation-aware metric is independent of implementation, most previous studies adopt this metric for comparisons, focusing on the algorithm.

## 2.1.2 Simul-S2S

A Simul-S2S model further synthesizes translated text into speech. To assess the translation quality of Simul-S2S models, a separate offline automatic speech recognition system is employed to transcribe the generated speech $y ( t )$ into text y for computing ASR-BLEU against the reference (Jia et al., 2019). To evaluate latency, a forced aligner is usually introduced to align the transcription y with $y ( t )$ to acquire the delay of each token in y. Subsequently, the AL metric, as defined in Simul-S2T, can be calculated for y (Ma et al., 2022).<sup>3</sup>

## 2.2 Speech-to-Unit Translation

Recent success of self-supervised representation learning in speech has opened up a new avenue for building speech-to-speech translation systems. The discretized units derived from clustering speech representations allow models to predict speech in a manner analogous to text. Lee et al. (2022) build the first speech-to-unit (S2UT) translation model with autoregressive Transformer (Vaswani et al., 2017). They utilize a HuBERT (Hsu et al., 2021) pre-trained on an unlabelled speech corpus and perform k-means algorithm to the learned representations of each 20ms chunk to produce K cluster centroids. Each chunk is then assigned the index of its nearest centroid serving as the label. Consequently, a target utterance can be encoded as a sequence of cluster indices $\boldsymbol { z } = \{ z _ { 1 } , z _ { 2 } , . . . , z _ { T } \}$ $z _ { i } \in \{ 0 , 1 , . . . , K - 1 \} , \forall 1 \leq i \leq T$ , where T is the number of chunks. S2UT model can be trained using cross-entropy. A separate unit-based vocoder (Polyak et al., 2021) is employed to convert the predicted acoustic unit sequence into waveform.

## 3 Approach

We provide a detailed introduction to our nonautoregressive generation framework for end-toend simultaneous speech-to-any translation in this section.

## 3.1 Architecture

As illustrated in Figure 2, NAST-S2x consists of a chunk-based acoustic streaming encoder and a chunk-based non-autoregressive (NAR) streaming decoder. This non-autoregressive decoder comprises stacked linguistic and acoustic components, with the two parts connected by upsampling the hidden states from linguistic part’s top layer and feeding them into the acoustic component. In contrast to previous two-pass speech-to-speech models (Jia et al., 2022a; Inaguma et al., 2023), NAST-S2x leverages its fully non-autoregressive nature. It no longer relies on intermediate text decoding to determine the information passed to the acoustic component. This characteristic allows it to be trained and tested directly from speech to acoustic units, thereby circumventing issues related to error propagation.

## 3.1.1 Streaming Acoustic Encoder

The acoustic encoder operates by setting a chunk size $T _ { s }$ . We extract FBank features from the streaming speech every $T _ { s }$ ms, which are then fed into the encoder. The acoustic encoder consists of two layers of causal convolution for downsampling and followed by multiple standard Transformer layers. In a Transformer layer, features within each chunk are encoded bidirectionally, and the information from all previous chunks can also be attended to. Given the strong local dependencies in speech, we additionally employ Lookahead encoding (Liu et al., 2021a), which enables states in each chunk to attend to its subsequent r frames.

## 3.1.2 Streaming Non-autoregressive Decoder

Once the latest chunk is encoded, we use the features as input to the linguistic decoder. Given the significant discrepancy in length between the sequences of FBank and text, we downsample the encoded features before feeding them into the decoder:

$$
\mathrm { D o w n S a m p l e } ( \tilde { \mathbf { s } } _ { i } ^ { e } , r _ { \mathrm { d o w n } } ) ,\tag{2}
$$

where $\tilde { \bf s } _ { i } ^ { e }$ represents the encoded features in the i-th chunk and $r _ { \mathrm { d o w n } }$ is the downsampling ratio. We use MeanPooling applied to every $r _ { \mathrm { d o w n } }$ encoded features in our experiments.

The linguistic decoder also works in a chunkby-chunk manner. The decoding of current chunk relies solely on hidden states in the previous chunks rather than any generated token:

$$
\begin{array} { l } { \mathrm { S e l f A t t n } ( \mathbf { s } _ { i } ^ { l d } , \mathbf { s } _ { \leq i } ^ { l d } ) , } \\ { \mathrm { C r o s s A t t n } ( \mathbf { s } _ { i } ^ { l d } , \tilde { \mathbf { s } } _ { \leq i } ^ { e } ) , } \end{array}\tag{3}
$$

where $\mathbf { s } _ { i } ^ { l d }$ denotes the hidden states in the i-th chunk in the linguistic decoder. Optionally, the linguistic decoder can generate text translation from the chunks. The text logits are derived by projecting the last layer states.

Meanwhile, hidden states in the last layer of linguistic decoder serve as input to the acoustic decoder after upsampling. This upsampling is designed to bridge the length gap between the sequences of text and acoustic unit:

$$
\mathrm { U p S a m p l e } ( \tilde { \mathbf { s } } _ { i } ^ { l d } , r _ { \mathrm { u p } } ) ,\tag{4}
$$

![](images/c27c14af970d2885cd61a9adcebf047a1fb0d156ab731f5224a0003f0c0c0deb.jpg)  
Figure 2: Overview of the proposed non-autoregressive generation framework for end-to-end simultaneous speechto-any translation (NAST-S2x, x text, speech ). Different colors indicate different chunks.

where $\tilde { \mathbf { s } } _ { i } ^ { l d }$ denotes the last layer states of the linguistic decoder in the i-th chunk and $r _ { \mathrm { u p } }$ is the upsampling ratio. We simply copy each state in the chunk $r _ { \mathrm { u p } }$ times.

The acoustic decoder operates similarly to the linguistic decoder. Compared with previous twopass models, our non-autoregressive acoustic decoder can directly attend to the acoustic encoder. This capability enables it to incorporate a broader range of acoustic information (e.g., rhythm, pitch, and energy) and helps in recovering from potential mistakes made by the linguistic decoder:

$$
\begin{array} { l } { { \mathrm { S e l f A t t n } ( \mathbf { s } _ { i } ^ { a d } , \mathbf { s } _ { \leq i } ^ { a d } ) , } } \\ { { \mathrm { C r o s s A t t n } ( \mathbf { s } _ { i } ^ { a d } , \tilde { \mathbf { s } } _ { \leq i } ^ { e } ) , } } \end{array}\tag{5}
$$

where ${ \bf s } _ { i } ^ { a d }$ denotes the hidden states in the i-th chunk in the acoustic decoder. We use the states in the top layer to predict acoustic units.

When predicting text and unit sequences, an additional blank token is included in the vocabulary. The model dynamically adjusts the output length of each chunk by generating repeated or blank tokens. Subsequently, the collapse function in CTC (Graves et al., 2006) is employed for online deduplication and removal of blanks to generate the final output. The generated chunk of units is sent directly to a separate unit-based HiFi-GAN vocoder (Polyak et al., 2021) for synthesizing the waveform, which is then played immediately to the listener.

## 3.2 Latency Control

In this subsection, we explore various techniques for controlling the latency of NAST-S2x.

Chunk Size Given that NAST-S2x operates at a chunk level, a straightforward approach to controlling latency is to adjust the chunk size. Specifically, when the chunk size exceeds the utterance length, our model transitions into an offline model, conducting bidirectional encoding and bidirectional non-autoregressive decoding.

Lookahead Chunk lookahead decoding resembles Lookahead encoding. When a feature chunk is sent to the decoder, it is allowed to wait for its subsequent k chunks before starting decoding:

$$
\begin{array} { r l } & { \mathrm { C r o s s A t t n } ( \mathbf { s } _ { i } ^ { l d } , \tilde { \mathbf { s } } _ { \leq i + k } ^ { e } ) , } \\ & { \mathrm { C r o s s A t t n } ( \mathbf { s } _ { i } ^ { a d } , \tilde { \mathbf { s } } _ { \leq i + k } ^ { e } ) . } \end{array}\tag{6}
$$

This allows the model to obtain more source-side information through an additional delay of $( k \cdot T _ { s } )$ ms, without changing the chunk size.

## 3.3 Training

While NAST-S2x benefits from various advantages of non-autoregressive generation, training it is challenging. Previous studies (Huang et al., 2022a; Shao et al., 2023) have highlighted that NAR generation struggles to capture multi-modal distributions. Regrettably, speech-to-speech translation encounters this multimodality problem. This challenge stems from two aspects: First, the mapping from speech input to text translation can be one-to-many, as different word choices and grammar structures may convey the same semantics. Secondly, the distribution of speech when the text is given can be multi-modal, with variations in pitch, rhythm, and energy. To mitigate these challenges, we propose the following strategies to train NAST-S2x.

## 3.3.1 Multi-task Non-monotonic Training

Due to the performance decline observed in NAR models when trained with maximum likelihood estimation, we train NAST-S2x using CTC-based non-monotonic latent alignment loss (Shao and Feng, 2022)

$$
\mathcal { L } _ { o } ( \theta ) = - \frac { 2 \cdot \sum _ { g \in G _ { 2 } } \operatorname* { m i n } \{ C _ { g } ( o ) , C _ { g } ( \theta ) \} } { \sum _ { g \in G _ { 2 } } ( C _ { g } ( o ) + C _ { g } ( \theta ) ) } ,\tag{7}
$$

where $o \in \{ y , z \}$ is the target for either S2T or S2U task. $C _ { g } ( y )$ denotes the occurrence count of bigram $g$ in target, $C _ { g } ( \theta )$ represents the expected count of $g$ for model, and $G _ { 2 }$ denotes the set of all bigrams in target. This training objective maximizes the F1 score of expected bigram matching between target and the uncollapsed output, and guides NAST-S2x towards convergence to a concentrated distribution, thereby alleviating the multimodality problem in speech-to-speech translation. We utilize multi-task learning to integrate the losses from both text and acoustic unit prediction tasks into our training process:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \boldsymbol { y } } ( \boldsymbol { \theta } ) + \mathcal { L } _ { z } ( \boldsymbol { \theta } ) . } \end{array}\tag{8}
$$

## 3.3.2 Two-Step Glancing

To further simplify the learning complexity for both linguistic and acoustic decoders, we further introduce the concept of glancing (Qian et al., 2021) to our NAST-S2x training. As depicted in Figure 2, we find the most probable sequence that can be collapsed to the target within the current distribution of text and acoustic unit in the model:

$$
\begin{array} { r l } & { \pmb { a } _ { \mathrm { u n i t } } ^ { * } = \underset { \pmb { a } _ { \mathrm { u n i t } } \in \beta ^ { - 1 } ( z ) } { \arg \operatorname* { m a x } } p _ { \theta } \big ( \pmb { a } _ { \mathrm { u n i t } } | \pmb { x } \big ) , } \\ & { \pmb { a } _ { \mathrm { t e x t } } ^ { * } = \underset { \pmb { a } _ { \mathrm { t e x t } } \in \beta ^ { - 1 } ( y ) } { \arg \operatorname* { m a x } } p _ { \theta } \big ( \pmb { a } _ { \mathrm { t e x t } } | \pmb { x } \big ) , } \end{array}\tag{9}
$$

where ${ \bf { a } } _ { \mathrm { { u n i t } } }$ and $\pmb { a } _ { \mathrm { t e x t } }$ represent the predicted uncollapsed sequence of text and acoustic unit, and $\beta ^ { - 1 }$ is the inverse of collapse function. We then randomly substitute the features fed to both the linguistic and acoustic decoders with token embeddings corresponding to positions in the most probable text or unit sequences.

This strategy simplifies the complexity of S2S mapping by providing cues during both decoding stages. This induces the NAR model to learn a deterministic conditional distribution, mitigating the issue of insufficient capacity for tasks with multimodal distributions.

## 4 Experiments

## 4.1 Speech-to-Text

Datasets We conduct experiments on two MuST-$\mathrm { C } ^ { 4 }$ language pairs: English to German (En De) and English to Spanish (En Es) (Di Gangi et al., 2019). We use the dev set for validation and report performance on the tst-COMMON set.

Pre-processing The input speech is represented as 80-dimensional log mel-filterbank coefficients computed every 10ms with a 25ms window. Global channel mean and variance normalization is applied to the input speech. In training, SpecAugment (Park et al., 2019) data augmentation with the LB policy is additionally employed. We use Sentence-Piece (Kudo and Richardson, 2018) to generate a unigram vocabulary of size 10000 for the source and target text jointly.

Model Configurations In the Simul-S2T experiments, we exclusively utilize the linguistic component of the decoder. We set the downsampling ratio<sup>5</sup> to 2 and explore chunk sizes within the set 160, 320, 640, 1280 ms. The offline results are obtained by setting the chunk size to be longer than any utterance in the corpus. The number of additional frames the encoder can attend to is set equal to the size of a chunk. When employing lookahead decoding, we vary the lookahead number k within the range 0, 2, 6 while maintaining a fixed chunk size of 320ms. More implementation details can be found in Appendix A.

Baselines We compare our NAST-S2T with several strong Simul-S2T baselines. Further details regarding baselines are available in Appendix B.1.

<table><tr><td>Chunk (# = 0)</td><td>ms BLEU AL</td><td>320 26.48 1114</td><td>640 27.02 1396</td><td>1280 28.05 2180</td></tr><tr><td rowspan="3">Lookahead (320ms)</td><td>#</td><td>0</td><td>2</td><td>6</td></tr><tr><td>BLEU</td><td>26.48</td><td>27.02</td><td>26.99</td></tr><tr><td>AL</td><td>1114</td><td>1762</td><td>2781</td></tr></table>

Table 1: Results of the quality-latency trade-off with increasing the chunk size or implementing lookahead decoding. Experiments are conducted on En Es Simul-S2T task.

Evaluation We use SimulEval<sup>6</sup> toolkit (Ma et al., 2020b) for evaluation. Translation quality is assessed using case-sensitive detokenized BLEU (Papineni et al., 2002; Post, 2018), while latency is measured by word-level Average Lagging (AL; Ma et al., 2020c). Numerical results with more latency metrics are provided in Appendix C.1.

## 4.1.1 Preliminary Experiment

We first conduct a preliminary experiment to compare latency control strategies. Employing NAST-S2T with a baseline chunk size of 320ms, we examine the trade-off between latency and quality by adjusting the chunk size and implementing lookahead decoding. As depicted in Table 1, both stratigies enhance quality at the sacrifice of latency. Nevertheless, increasing the chunk size yields superior quality with reduced latency over lookahead decoding. Notably, there appears to be a quality plateau when utilizing lookahead decoding. Waiting for an extra 6 source chunks versus 2 extra ones results in nearly identical quality, despite an additional delay of almost 1000ms. This implies that the amount of source information alone does not solely dictate translation quality. By adopting the strategy of increasing chunk size, we not only enable the model to attend to more source information but also facilitate bidirectional non-autoregressive decoding of longer sequences within a chunk. This enhancement significantly improves the translation quality. Therefore, we only vary the chunk size in the main experiment.

## 4.1.2 Main Results and Analysis

Figure 3 illustrates the main results of Simul-S2T task. Detailed numerical results are available in Table 5 and 6. It can be observed that NAST-S2T achieves competitive or superior translation quality compared to strong baselines across various latency constraints. At lower latency, its performance is only inferior to CAAT (Liu et al., 2021a). Meanwhile, it performs better or comparably as the autoregressive models under higher latency or offline conditions. Both datasets demonstrate that as the chunk size $T _ { s }$ increases from 160ms to 320ms, there is a significant improvement in translation quality with only a minor increase in latency. We attribute this phenomenon to the average duration of each word, estimated to be approximately 280ms (Ma et al., 2020c). The model’s performance tends to degrade when the chunk size falls below it. Furthermore, we find that NAST-S2T achieves a better balance when the chunk size $T _ { s }$ is 640ms (AL 1200ms), after which the quality gain from further increasing the chunk size diminishes.

## 4.2 Speech-to-Speech

Datasets We conduct experiments on CVSS-C<sup>7</sup> French to English (Fr En) dataset (Jia et al., 2022b).

Pre-processing For the source speech, we resample the audio to 16kHz and apply identical preprocessing steps as those used in the Simul-S2T experiments. For the target speech, we also downsample the audio and extract discrete units utilizing the publicly available pre-trained mHuBERT model and K-means quantizer.<sup>8</sup>

Model Configurations The downsampling and upsampling ratio are set to 2 and 6. We explore different settings for chunk sizes within the set 320, 640, 1280, 1920, 2560 ms. The offline results are obtained by setting the chunk size to be longer than any utterance. The number of additional frames the encoder can attend to is set equal to the size of a chunk. We also experimented with fixing the duration of additional frames to 1280ms when the chunk size is larger. More details can be found in Appendix A.

## Baselines

Wait-k-Stride-n: We employ Wait-k strategy (Ma et al., 2019) for S2UT model (Lee et al., 2022) to build an end-to-end Simul-S2S baseline. Since the input is speech audio, a pre-decision module is needed to segment the utterance into multiple chunks to execute Wait-k (Ma et al., 2020c). Furthermore, the translation of a speech chunk can consist of multiple acoustic units to form the pronunciation of a word. It is reasonable to generate multiple unit tokens upon receiving a speech chunk. Therefore, we adopt Wait-k-Stride-n strategy (Zeng et al., 2021) to construct an end-to-end Simul-S2S baseline, varying the speech chunk size and the hyperparameters k and n. The numerical results can be found in Table 13.

![](images/48226d874936f3249729c4cd41a643c50c6b9c53eb7cc5ceca59b412cde08c55.jpg)  
<sup>(a)</sup> <sup>En</sup>→<sup>De</sup>

![](images/1dfde986ef2dd7f5ef51f73e3c29a5d60fe5692e3e9f8dc4bf2b8090c5646b23.jpg)  
<sup>(b)</sup> <sup>En</sup>→<sup>Es</sup>  
Figure 3: Results of translation quality (BLEU) against latency (Average Lagging, AL) on MuST-C En De and En Es datasets. The red solid line and dashed line illustrate the performance of NAST-S2T under different chunk sizes $T _ { s }$ or in an offline condition. The numerical results are presented in Table 5 and Table 6.

EDAtt + Tacotron2: We further provide the results of cascade systems (Simul-S2T + TTS) for comparison. We choose EDAtt (Papi et al., 2023b) as the Simul-S2T model. According to the recommendation in Papi et al. (2023b), we train a Conformer + CTC compression model (Gaido et al., 2021) with a total of 120M parameters using speech-text parallel pairs of CVSS-C Fr-En dataset as the offline model to implement EDAtt algorithm. For TTS part, we use a Tacotron2 model trained on LJSpeech<sup>9</sup>. Whenever the Simul-S2T model generates a complete word, we send it to the TTS model and generate a speech chunk as output. The numerical results can be found in Table 14.

We also compare NAST-S2S with several strong Offline-S2S models to assess its performance in offline scenarios. Further details regarding baselines are available in Appendix B.2.

Evaluation We also use SimulEval toolkit for evaluation. Following Ma et al. (2022), we keep discontinuities between generated speech chunks to simulate real-world scenarios. Translation quality is assessed using ASR-BLEU. We also employ BLASER 2. ${ \boldsymbol { 0 } } ^ { 1 0 }$ (Seamless Communication et al., 2023a) to assess the quality. The results for BLASER 2.0 are presented in Table 12. Regarding latency, we report AL and AL\_EOW (Ma et al., 2022). AL measures time delay of waveform chunks, while AL\_EOW assesses the delay of text transcribed from generated speech. The generated time of each word is considered as the end time of its corresponding segment. Numerical results with more latency metrics are provided in Appendix C.2.

## 4.2.1 Main Results

Figure 4 illustrates the main results of Simul-S2S task. Detailed numerical results are presented in Table 9. We observe a trend where the translation quality of NAST-S2S generally improves as latency increases, with a notable improvement from 3000ms to 4000ms. Even under extremely low latency conditions (AL 1000ms), NAST-S2S still achieves acceptable translation quality (ASR-BLEU > 19). This result even surpasses the performance of wait-k-stride-n and cascade baselines at 4000ms latency. Furthermore, we discover that in offline scenarios, the quality achieved by NAST-S2S exceeds that of the current leading NAR Offline-S2S model DASpeech (Fang et al., 2023) by nearly 1 ASR-BLEU, with translation quality only slightly inferior to two-pass autoregressive model UnitY<sup>11</sup> (Inaguma et al., 2023).

<table><tr><td>Model</td><td>#Params</td><td>End-to-End</td><td>Streamable</td><td>ASR-BLEU</td><td>Speedup</td></tr><tr><td>S2UT (Lee et al., 2022)</td><td>58M</td><td>√</td><td>x</td><td>24.80</td><td>1.00×</td></tr><tr><td>UnitY (Inaguma et al., 2023)</td><td>67M</td><td>x</td><td>x</td><td>26.90</td><td>1.60×</td></tr><tr><td>DASpeech (Fang et al., 2023)</td><td>93M</td><td>x</td><td>x</td><td>25.03</td><td>16.29×</td></tr><tr><td>Offline NAST-S2S</td><td>79M</td><td>√</td><td>√</td><td>25.82</td><td>28.30×</td></tr></table>

Table 2: Comparison of strong Offline-S2S baselines and our NAST-S2S in offline conditions. The speedup is measured using a GeForce RTX 3090 GPU with a batch size of 1.

![](images/f4e7c22d3bd61793c924f4bb96fc08775efd814fc4a74c28f440055f29ff4e59.jpg)  
Figure 4: Results of translation quality in offline conditions and simultaneous scenarios (ASR-BLEU or ASR-BLEU (Silence Removed) against AL or AL\_EOW). The numerical results of NAST-S2S are presented in Table 9 and Table 11.

## 4.2.2 Analysis on Inference Efficiency

Speech-to-speech translation imposes strong demands on inference efficiency. In Offline-S2S, efficiently generating long sequences of acoustic unit is crucial to minimize waiting time. In Simul-S2S, reducing computational time overhead is essential to avoid extra latency. Benefiting from end-to-end non-autoregressive generation, NAST-S2S offers appealing advantages in both scenarios. Table 2 presents the comparison in Offline-S2S. NAST-S2S achieves a 28 speedup compared to S2UT <sup>and</sup> <sup>a</sup> <sup>17</sup>× <sup>speedup</sup> <sup>compared</sup> <sup>to</sup> <sup>UnitY</sup> <sup>at</sup> <sup>decoding.</sup> In Simul-S2S, the advantage in inference speed becomes more critical. Table 3 presents the comparison of non-computation-aware and computationaware latency. The gap between AL and AL\_CA and the average computation time per chunk generation are both less than 300ms when the chunk size is larger than 640ms, indicating that NAST-S2S’s latency in practical use is similar to the theoretical latency of its simultaneous translation policy.

## 4.2.3 Analysis on Discontinuity

We observed notable differences in the performance of NAST-S2x between Simul-S2S and Simul-S2T tasks. NAST-S2T achieves satisfactory quality when the chunk size $T _ { s }$ is set to 640ms (AL < 2000ms). However, to attain translation quality comparable to offline condition, NAST-S2S requires an increase in the chunk size $T _ { s }$ to 2560ms. This discrepancy may stem from the differing nature of text and speech streaming generation. In text generation, appending newly generated chunk directly after the historical sequence is straightforward. However, in speech generation, there may be silence intervals between each speech chunk, particularly when the chunk size $T _ { s }$ exceeds the duration of the last generated speech chunk. Therefore, we speculate that as the chunk size decreases, increased silence between generated speech chunks may lead to discontinuity in speech, thereby decreasing the overall quality.

To validate this hypothesis, we further analyze the trends of the following metrics as the chunk size varies: ASR-BLEU (Silence Removed), representing ASR-BLEU score after removing the added silence between generated chunk; Unit-BLEU, representing BLEU score of the generated unit sequences against the reference; S2T-BLEU, where we conduct additional decoding of the linguistic decoder to evaluate quality in Simul-S2T. We also provide statistics on the number of discontinuities (DCNum), the average silence duration per discontinuity (DCAve), and the total silence duration (DCSum) in the generated streaming speech.

Table 4 presents the statistics. We observed minor degradation in the values of Unit-BLEU and S2T-BLEU even at a chunk size of 320ms, showing NAST-S2S’s capability in streaming text and unit generation. However, there exists a significant increase in the number of discontinuities as the chunk size decreases. Although the duration of silence per discontinuity is relatively short when the chunk size is small, the increase in their number results in a longer total silence duration, thus intensifying the degree of discontinuity and impacting its overall quality (ASR-BLEU).

Moreover, if the added silence were removed, the measured ASR-BLEU (Silence Removed) significantly increased and the gap between streaming and offline scenarios becomes small. This suggests that ASR-BLEU may underestimate speech quality here. The decline in ASR-BLEU scores is primarily due to the playback timing. For example, consider the word "Richardson", which consists of multiple syllables. If the "Richard" part of the waveform is generated in the previous chunk and played immediately, and the "son" syllable is generated in the subsequent chunk, the potential silence period (which equals to the chunk size minus the length of the waveform generated in the previous chunk) could cause the listener to perceive a stuttering effect, leading to a decrease in ASR-BLEU scores.

<table><tr><td rowspan="2">Ts (ms)</td><td rowspan="2">ASR-BLEU</td><td colspan="3">Average Lagging (ms)</td><td colspan="3">Start Offset (ms)</td><td colspan="3">End Offset (ms)</td><td rowspan="2">ACT (ms)</td></tr><tr><td>NCA</td><td>CA</td><td> $\Delta$ </td><td>NCA</td><td>CA</td><td>∆</td><td>NCA</td><td>CA</td><td> $\Delta$ </td></tr><tr><td>320</td><td>19.67</td><td>-392</td><td>347</td><td>739</td><td>655</td><td>712</td><td>57</td><td>562</td><td>1550</td><td>988</td><td>555</td></tr><tr><td>640</td><td>19.15</td><td>1532</td><td>1824</td><td>292</td><td>1294</td><td>1350</td><td>56</td><td>863</td><td>1344</td><td>481</td><td>297</td></tr><tr><td>1280</td><td>20.20</td><td>3330</td><td>3500</td><td>170</td><td>2566</td><td>2642</td><td>76</td><td>1648</td><td>1901</td><td>253</td><td>192</td></tr><tr><td>2560</td><td>24.88</td><td>4975</td><td>5097</td><td>122</td><td>4691</td><td>4781</td><td>90</td><td>2753</td><td>2879</td><td>126</td><td>120</td></tr></table>

Table 3: Results of translation quality (ASR-BLEU), latency (Average Lagging, Start Offset & End Offset) and average computation time per chunk generation (ACT) during NAST-S2S simultaneous inference. All latency metrics report both the computation-aware (CA) version and the non-computation-aware (NCA) version, as well as their differences (∆).

<table><tr><td> $T _ { s } ~ ( m s )$ </td><td>320</td><td>640</td><td>1280</td><td>2560</td></tr><tr><td>S2T-BLEU</td><td>28.04</td><td>28.28</td><td>28.23</td><td>28.78</td></tr><tr><td>Unit-BLEU</td><td>33.41</td><td>33.97</td><td>34.04</td><td>34.40</td></tr><tr><td>ASR-BLEU</td><td>19.67</td><td>19.15</td><td>20.20</td><td>24.88</td></tr><tr><td>ASR-BLEU (Silence Removed)</td><td>24.90</td><td>25.67</td><td>25.71</td><td>26.14</td></tr><tr><td>DCNum</td><td>7.3</td><td>4.7</td><td>2.1</td><td>0.4</td></tr><tr><td>DCAve (ms)</td><td>355</td><td>450</td><td>685</td><td>360</td></tr><tr><td>DCSum (ms)</td><td>2220</td><td>1952</td><td>1420</td><td>395</td></tr></table>

Table 4: Statistics of NAST-S2S generation across varying chunk sizes $T _ { s }$

## 5 Related Work

Researches in simultaneous speech translation can be roughly categorized into Simul-S2T (Ma et al., 2020c) and Simul-S2S (Zheng et al., 2020) variants.

Simul-S2T With the rise of neural networks, Simul-S2T models no longer rely on the transcription as a bridge (Ma et al., 2020c; Iranzo-Sánchez et al., 2020). Given the difference between speech and text input, some researchers focus on how to divide speech chunks and then execute strategies. Ma et al. (2020c) employed fixed-length segmentation and implemented Wait-k (Ma et al., 2019) and MMA (Ma et al., 2020d) based on that; Ren et al. (2020); Zeng et al. (2021); Chen et al. (2021) utilized ASR results to partition and execute Wait-k or its variants. Zhang et al. (2022) trained a segmentation model to detect semantic units. Zhang and Feng (2023a) trained a model to dynamically segment with differentiable approach, then extending it to a segment-to-segment framework (Zhang and Feng, 2023b). Additionally, some researchers have also attempted to use Transducer (Graves, 2012) and incorporate attention mechanisms to enhance its performance (Liu et al., 2021a; Tang et al., 2023). Besides, some researchers are leveraging offline models for simultaneous inference. Liu et al. (2020) considered the agreeing prefixes of two consecutive chunks as stable hypotheses. Papi et al. (2023b,c) used attention as guidance, allowing the model to generate output for the current step if its attention is not focused on the most recently received frames.

Simul-S2S There have been limited prior studies exploring Simul-S2S. Zheng et al. (2020) and Sudoh et al. (2020) both developed cascade models by integrating streaming ASR, Simul-T2T, and incremental TTS components. Additionally, Liu et al. (2021b) proposed latency reduction strategies for incremental TTS in Simul-S2S. Moreover, Ma et al. (2022) introduced a variational version of MMA to S2UT (Lee et al., 2022) and constructed the first end-to-end Simul-S2S model.

## 6 Conclusion

In this paper, we present a non-autoregressive streaming generation framework for simultaneous speech-to-any translation, which integrates both Simul-S2T and Simul-S2S tasks into a unified framework. Experimental results on various benchmarks showcase the superiority of our model.

## Limitation

Our NAST-S2x exhibits greater latency in Simul-S2S compared to Simul-S2T tasks. This discrepancy arises due to NAST-S2S’s reliance on an external vocoder, typically trained on offline tasks and not adapted for streaming scenarios, thereby constraining NAST-S2S’s performance. Additionally, our method requires a parallel speech-to-speech translation corpus for end-to-end training, which can be challenging to obtain. Existing datasets are typically based on synthesized target speech. The lack of such corpora may hinder the development of simultaneous speech-to-speech translation models.

## Acknowledgement

We thank the anonymous reviewers for their insightful comments. This work is supported by National Natural Science Foundation of China (Grant No. 62376260).

## References

Naveen Arivazhagan, Colin Cherry, Wolfgang Macherey, Chung-Cheng Chiu, Semih Yavuz, Ruoming Pang, Wei Li, and Colin Raffel. 2019. Monotonic infinite lookback attention for simultaneous machine translation. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 1313–1323, Florence, Italy. Association for Computational Linguistics.

Junkun Chen, Mingbo Ma, Renjie Zheng, and Liang Huang. 2021. Direct simultaneous speech-to-text translation assisted by synchronized streaming ASR. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 4618–4624, Online. Association for Computational Linguistics.

Chung-Cheng Chiu and Colin Raffel. 2018. Monotonic chunkwise attention. In International Conference on Learning Representations.

Kyunghyun Cho and Masha Esipova. 2016. Can neural machine translation do simultaneous translation? CoRR, abs/1606.02012.

Mattia A. Di Gangi, Roldano Cattoni, Luisa Bentivogli, Matteo Negri, and Marco Turchi. 2019. MuST-C: a Multilingual Speech Translation Corpus. In Proceedings ofthe 2019 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2012–2017, Minneapolis, Minnesota. Association for Computational Linguistics.

Qingkai Fang, Yan Zhou, and Yang Feng. 2023. Daspeech: Directed acyclic transformer for fast and

high-quality speech-to-speech translation. In Advances in Neural Information Processing Systems, volume 36, pages 72604–72623. Curran Associates, Inc.

Marco Gaido, Mauro Cettolo, Matteo Negri, and Marco Turchi. 2021. CTC-based compression for direct speech translation. In Proceedings ofthe 16th Conference ofthe European Chapter ofthe Association for Computational Linguistics: Main Volume, pages 690–696, Online. Association for Computational Linguistics.

Alex Graves. 2012. Sequence transduction with recurrent neural networks. CoRR, abs/1211.3711.

Alex Graves, Santiago Fernández, Faustino Gomez, and Jürgen Schmidhuber. 2006. Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks. In Proceedings of the 23rd International Conference on Machine Learning, ICML ’06, page 369–376, New York, NY, USA. Association for Computing Machinery.

Jiatao Gu, Graham Neubig, Kyunghyun Cho, and Victor O.K. Li. 2017. Learning to translate in real-time with neural machine translation. In Proceedings of the 15th Conference of the European Chapter of the Associationfor Computational Linguistics: Volume 1, Long Papers, pages 1053–1062, Valencia, Spain. Association for Computational Linguistics.

Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui Yu, Wei Han, Shibo Wang, Zhengdong Zhang, Yonghui Wu, and Ruoming Pang. 2020. Conformer: Convolution-augmented Transformer for Speech Recognition. In Proc. Interspeech 2020, pages 5036–5040.

Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. 2021. Hubert: Self-supervised speech representation learning by masked prediction of hidden units. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29:3451–3460.

Fei Huang, Tianhua Tao, Hao Zhou, Lei Li, and Minlie Huang. 2022a. On the learning of non-autoregressive transformers. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 9356–9376. PMLR.

Fei Huang, Hao Zhou, Yang Liu, Hang Li, and Minlie Huang. 2022b. Directed acyclic transformer for nonautoregressive machine translation. In Proceedings of the 39th International Conference on Machine Learning, ICML 2022, volume 162 of Proceedings of Machine Learning Research, pages 9410–9428. PMLR.

Hirofumi Inaguma, Sravya Popuri, Ilia Kulikov, Peng-Jen Chen, Changhan Wang, Yu-An Chung, Yun Tang, Ann Lee, Shinji Watanabe, and Juan Pino. 2023. UnitY: Two-pass direct speech-to-speech translation

with discrete units. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15655– 15680, Toronto, Canada. Association for Computational Linguistics.

Javier Iranzo-Sánchez, Adrià Giménez Pastor, Joan Albert Silvestre-Cerdà, Pau Baquero-Arnal, Jorge Civera Saiz, and Alfons Juan. 2020. Direct segmentation models for streaming speech translation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2599–2611, Online. Association for Computational Linguistics.

Ye Jia, Michelle Tadmor Ramanovich, Tal Remez, and Roi Pomerantz. 2022a. Translatotron 2: High-quality direct speech-to-speech translation with voice preservation. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings ofMachine Learning Research, pages 10120–10134. PMLR.

Ye Jia, Michelle Tadmor Ramanovich, Quan Wang, and Heiga Zen. 2022b. CVSS corpus and massively multilingual speech-to-speech translation. In Proceedings ofLanguage Resources and Evaluation Conference (LREC), pages 6691–6703.

Ye Jia, Ron J. Weiss, Fadi Biadsy, Wolfgang Macherey, Melvin Johnson, Zhifeng Chen, and Yonghui Wu. 2019. Direct speech-to-speech translation with a sequence-to-sequence model. In Interspeech 2019, 20th Annual Conference ofthe International Speech Communication Association, Graz, Austria, 15-19 September 2019, pages 1123–1127. ISCA.

Yoon Kim and Alexander M. Rush. 2016. Sequencelevel knowledge distillation. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 1317–1327, Austin, Texas. Association for Computational Linguistics.

Diederik P. Kingma and Jimmy Ba. 2015. Adam: A method for stochastic optimization. In 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings.

Taku Kudo and John Richardson. 2018. SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 66–71, Brussels, Belgium. Association for Computational Linguistics.

Ann Lee, Peng-Jen Chen, Changhan Wang, Jiatao Gu, Sravya Popuri, Xutai Ma, Adam Polyak, Yossi Adi, Qing He, Yun Tang, Juan Pino, and Wei-Ning Hsu. 2022. Direct speech-to-speech translation with discrete units. In Proceedings ofthe 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3327–3339, Dublin, Ireland. Association for Computational Linguistics.

Dan Liu, Mengge Du, Xiaoxi Li, Ya Li, and Enhong Chen. 2021a. Cross attention augmented transducer networks for simultaneous translation. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 39–55, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Danni Liu, Gerasimos Spanakis, and Jan Niehues. 2020. Low-Latency Sequence-to-Sequence Speech Recognition and Translation by Partial Hypothesis Selection. In Proc. Interspeech 2020, pages 3620–3624.

Danni Liu, Changhan Wang, Hongyu Gong, Xutai Ma, Yun Tang, and Juan Miguel Pino. 2021b. From start to finish: Latency reduction strategies for incremental speech synthesis in simultaneous speech-to-speech translation. In Interspeech.

Mingbo Ma, Liang Huang, Hao Xiong, Renjie Zheng, Kaibo Liu, Baigong Zheng, Chuanqiang Zhang, Zhongjun He, Hairong Liu, Xing Li, Hua Wu, and Haifeng Wang. 2019. STACL: Simultaneous translation with implicit anticipation and controllable latency using prefix-to-prefix framework. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 3025–3036, Florence, Italy. Association for Computational Linguistics.

Mingbo Ma, Baigong Zheng, Kaibo Liu, Renjie Zheng, Hairong Liu, Kainan Peng, Kenneth Church, and Liang Huang. 2020a. Incremental text-to-speech synthesis with prefix-to-prefix framework. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2020, pages 3886–3896, Online. Association for Computational Linguistics.

Xutai Ma, Mohammad Javad Dousti, Changhan Wang, Jiatao Gu, and Juan Pino. 2020b. SIMULEVAL: An evaluation toolkit for simultaneous translation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 144–150, Online. Association for Computational Linguistics.

Xutai Ma, Hongyu Gong, Danni Liu, Ann Lee, Yun Tang, Peng-Jen Chen, Wei-Ning Hsu, Phillip Koehn, and Juan Pino. 2022. Direct simultaneous speech-tospeech translation with variational monotonic multihead attention.

Xutai Ma, Juan Pino, and Philipp Koehn. 2020c. SimulMT to SimulST: Adapting simultaneous text translation to end-to-end simultaneous speech translation. In Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing, pages 582–587, Suzhou, China. Association for Computational Linguistics.

Xutai Ma, Juan Miguel Pino, James Cross, Liezl Puzon, and Jiatao Gu. 2020d. Monotonic multihead attention. In International Conference on Learning Representations.

Zhengrui Ma, Shaolei Zhang, Shoutao Guo, Chenze Shao, Min Zhang, and Yang Feng. 2023. Nonautoregressive streaming transformer for simultaneous translation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 5177–5190, Singapore. Association for Computational Linguistics.

Yishu Miao, Phil Blunsom, and Lucia Specia. 2021. A generative framework for simultaneous machine translation. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 6697–6706, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Sara Papi, Marco Gaido, Matteo Negri, and Marco Turchi. 2022. Over-generation cannot be rewarded: Length-adaptive average lagging for simultaneous speech translation. In Proceedings ofthe Third Workshop on Automatic Simultaneous Translation, pages 12–17, Online. Association for Computational Linguistics.

Sara Papi, Marco Gaido, Andrea Pilzer, and Matteo Negri. 2023a. When good and reproducible results are a giant with feet of clay: The importance of software quality in nlp.

Sara Papi, Matteo Negri, and Marco Turchi. 2023b. Attention as a guide for simultaneous speech translation. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13340–13356, Toronto, Canada. Association for Computational Linguistics.

Sara Papi, Marco Turchi, and Matteo Negri. 2023c. AlignAtt: Using Attention-based Audio-Translation Alignments as a Guide for Simultaneous Speech Translation. In Proc. INTERSPEECH 2023, pages 3974–3978.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics.

Daniel S Park, William Chan, Yu Zhang, Chung-Cheng Chiu, Barret Zoph, Ekin D Cubuk, and Quoc V Le. 2019. Specaugment: A simple data augmentation method for automatic speech recognition. Interspeech 2019.

Adam Polyak, Yossi Adi, Jade Copet, Eugene Kharitonov, Kushal Lakhotia, Wei-Ning Hsu, Abdelrahman Mohamed, and Emmanuel Dupoux. 2021. Speech Resynthesis from Discrete Disentangled Self-Supervised Representations. In Proc. Interspeech 2021.

Matt Post. 2018. A call for clarity in reporting BLEU scores. In Proceedings of the Third Conference on

Machine Translation: Research Papers, pages 186– 191, Belgium, Brussels. Association for Computational Linguistics.

Lihua Qian, Hao Zhou, Yu Bao, Mingxuan Wang, Lin Qiu, Weinan Zhang, Yong Yu, and Lei Li. 2021. Glancing transformer for non-autoregressive neural machine translation. In Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1993–2003, Online. Association for Computational Linguistics.

Colin Raffel, Minh-Thang Luong, Peter J. Liu, Ron J. Weiss, and Douglas Eck. 2017. Online and lineartime attention by enforcing monotonic alignments. In Proceedings ofthe 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 2837–2846. PMLR.

Yi Ren, Chenxu Hu, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, and Tie-Yan Liu. 2021. Fastspeech 2: Fast and high-quality end-to-end text to speech. In International Conference on Learning Representations.

Yi Ren, Jinglin Liu, Xu Tan, Chen Zhang, Tao Qin, Zhou Zhao, and Tie-Yan Liu. 2020. SimulSpeech: End-to-end simultaneous speech to text translation. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 3787– 3796, Online. Association for Computational Linguistics.

Seamless Communication, Loïc Barrault, Yu-An Chung, Mariano Cora Meglioli, David Dale, Ning Dong, Paul-Ambroise Duquenne, Hady Elsahar, Hongyu Gong, Kevin Heffernan, John Hoffman, Christopher Klaiber, Pengwei Li, Daniel Licht, Jean Maillard, Alice Rakotoarison, Kaushik Ram Sadagopan, Guillaume Wenzek, Ethan Ye, Bapi Akula, Peng-Jen Chen, Naji El Hachem, Brian Ellis, Gabriel Mejia Gonzalez, Justin Haaheim, Prangthip Hansanti, Russ Howes, Bernie Huang, Min-Jae Hwang, Hirofumi Inaguma, Somya Jain, Elahe Kalbassi, Amanda Kallet, Ilia Kulikov, Janice Lam, Daniel Li, Xutai Ma, Ruslan Mavlyutov, Benjamin Peloquin, Mohamed Ramadan, Abinesh Ramakrishnan, Anna Sun, Kevin Tran, Tuan Tran, Igor Tufanov, Vish Vogeti, Carleigh Wood, Yilin Yang, Bokai Yu, Pierre Andrews, Can Balioglu, Marta R. Costa-jussà, Onur Celebi, Maha Elbayad, Cynthia Gao, Francisco Guzmán, Justine Kao, Ann Lee, Alexandre Mourachko, Juan Pino, Sravya Popuri, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, Paden Tomasello, Changhan Wang, Jeff Wang, and Skyler Wang. 2023a. Seamlessm4t: Massively multilingual & multimodal machine translation.

Seamless Communication, Loïc Barrault, Yu-An Chung, Mariano Coria Meglioli, David Dale, Ning Dong, Mark Duppenthaler, Paul-Ambroise Duquenne,

Brian Ellis, Hady Elsahar, Justin Haaheim, John Hoffman, Min-Jae Hwang, Hirofumi Inaguma, Christopher Klaiber, Ilia Kulikov, Pengwei Li, Daniel Licht, Jean Maillard, Ruslan Mavlyutov, Alice Rakotoarison, Kaushik Ram Sadagopan, Abinesh Ramakrishnan, Tuan Tran, Guillaume Wenzek, Yilin Yang, Ethan Ye, Ivan Evtimov, Pierre Fernandez, Cynthia Gao, Prangthip Hansanti, Elahe Kalbassi, Amanda Kallet, Artyom Kozhevnikov, Gabriel Mejia Gonzalez, Robin San Roman, Christophe Touret, Corinne Wong, Carleigh Wood, Bokai Yu, Pierre Andrews, Can Balioglu, Peng-Jen Chen, Marta R. Costa-jussà, Maha Elbayad, Hongyu Gong, Francisco Guzmán, Kevin Heffernan, Somya Jain, Justine Kao, Ann Lee, Xutai Ma, Alex Mourachko, Benjamin Peloquin, Juan Pino, Sravya Popuri, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, Anna Sun, Paden Tomasello, Changhan Wang, Jeff Wang, Skyler Wang, and Mary Williamson. 2023b. Seamless: Multilingual expressive and streaming speech translation.

Chenze Shao and Yang Feng. 2022. Non-monotonic latent alignments for ctc-based non-autoregressive machine translation. In Advances in Neural Information Processing Systems, volume 35, pages 8159–8173. Curran Associates, Inc.

Chenze Shao, Zhengrui Ma, Min Zhang, and Yang Feng. 2023. Beyond mle: Convex learning for text generation. In Advances in Neural Information Processing Systems, volume 36, pages 8913–8936. Curran Associates, Inc.

Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani. 2018. Self-attention with relative position representations. In Proceedings ofthe 2018 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 464–468, New Orleans, Louisiana. Association for Computational Linguistics.

Katsuhito Sudoh, Takatomo Kano, Sashi Novitasari, Tomoya Yanagita, Sakriani Sakti, and Satoshi Nakamura. 2020. Simultaneous speech-to-speech translation system with neural incremental asr, mt, and tts.

Yun Tang, Anna Sun, Hirofumi Inaguma, Xinyue Chen, Ning Dong, Xutai Ma, Paden Tomasello, and Juan Pino. 2023. Hybrid transducer and attention based encoder-decoder modeling for speech-to-text tasks. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 12441–12455, Toronto, Canada. Association for Computational Linguistics.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Pro cessing Systems, volume 30. Curran Associates, Inc.

Xingshan Zeng, Liangyou Li, and Qun Liu. 2021. Real-TranS: End-to-end simultaneous speech translation

with convolutional weighted-shrinking transformer. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 2461–2474, Online. Association for Computational Linguistics.

Qian Zhang, Han Lu, Hasim Sak, Anshuman Tripathi, Erik McDermott, Stephen Koo, and Shankar Kumar. 2020. Transformer transducer: A streamable speech recognition model with transformer encoders and rnn-t loss. In ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 7829–7833.

Ruiqing Zhang, Zhongjun He, Hua Wu, and Haifeng Wang. 2022. Learning adaptive segmentation policy for end-to-end simultaneous translation. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7862–7874, Dublin, Ireland. Association for Computational Linguistics.

Shaolei Zhang and Yang Feng. 2023a. End-to-end simultaneous speech translation with differentiable segmentation. In Findings ofthe Associationfor Computational Linguistics: ACL 2023, pages 7659–7680, Toronto, Canada. Association for Computational Linguistics.

Shaolei Zhang and Yang Feng. 2023b. Unified segmentto-segment framework for simultaneous sequence generation. In Advances in Neural Information Processing Systems, volume 36, pages 45235–45258. Curran Associates, Inc.

Renjie Zheng, Mingbo Ma, Baigong Zheng, Kaibo Liu, Jiahong Yuan, Kenneth Church, and Liang Huang. 2020. Fluent and low-latency simultaneous speechto-speech translation with self-adaptive training. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2020, pages 3928–3937, Online. Association for Computational Linguistics.

## A Implementation Details

## A.1 Configuration

We incorporate both cosine positional encoding (Vaswani et al., 2017) and relative positional attention (Shaw et al., 2018) into the acoustic encoder, and utilize learned positional encoding for non-autoregressive decoder. A separate learned positional encoding is applied to the acoustic decoder. The acoustic encoder comprises two layers of causal convolution followed by six standard Transformer layers. Both the non-autoregressive linguistic and acoustic decoders consist of six Transformer layers each. All Transformer layers are configured with a 512 embedding dimension, 8 attention heads, and a 2048 FFN dimension. The total number of parameters for NAST-S2T and NAST-S2S are 52M and 79M.

## A.2 Training

NAST-S2T Considering the inherent complexity of speech-to-text translation, we leverage the concept of curriculum learning. We initialize the encoder of NAST-S2T with an ASR-trained model and conduct pretraining using CTC loss (Graves et al., 2006). Subsequently, we employ non-monotonic training to further refine NAST-S2T. During the CTC loss pretraining, we set the dropout rate to 0.3, weight decay to 0.01, and incorporate label smoothing with a value of 0.01. The dropout rates for activation and attention are both set to 0.1. The pretraining process spans 100k updates with a batch size of 320k tokens. The learning rate gradually warms up to $1 \cdot 1 0 ^ { - 3 }$ within 10k steps, while the text glancing ratio linearly anneals from 0.5 to 0.3 over 50k steps. In non-monotonic training, we adjust the dropout rate to 0.1 while keeping other hyperparameters unchanged. This stage involves training NAST-S2T for 20k updates. The learning rate warms up to $3 \cdot 1 0 ^ { - 4 }$ within 4k steps, and the text glancing ratio is maintained at 0.3. Throughout the training, we optimize models using the Adam optimizer (Kingma and Ba, 2015) with parameters $\beta = ( 0 . 9 , 0 . 9 8 )$ and $\epsilon = 1 0 ^ { - 8 }$ We utilize sequence-level knowledge distillation (Kim and Rush, 2016) solely during the CTC pretraining stage to facilitate model warmup, while NAST-S2T is trained directly on raw data during non-monotonic training.

NAST-S2S Similar to the training of NAST-S2T, a curriculum learning approach is also devised for NAST-S2S. We initialize the encoder of NAST-S2S with an ASR-trained model and conduct multi-task pretraining using the CTC loss. Subsequently, we employ multi-task nonmonotonic training to further refine NAST-S2S. During the pretraining, the hyperparameters are consistent with those used in NAST-S2T, with the exception of incorporating label smoothing for both text and unit targets, set at a value of 0.01. The multi-task pretraining process spans 50k updates with a batch size of 320k tokens. The text glancing ratio linearly anneals from 0.5 to 0.3 over 50k steps, while the unit glancing ratio linearly decreases from 0.3 to 0.1 over the same number of steps. In multi-task non-monotonic training, we adjust the dropout rate to 0.1 while keeping other hyperparameters unchanged. This stage involves training NAST-S2S for 30k updates. The learning rate warms up to $3 \cdot 1 0 ^ { - 4 }$ within 4k steps. We maintain a text glancing ratio of 0.3 and a unit glancing ratio of 0.1 in this stage. Knowledge distillation is not utilized during the entire training of NAST-S2S.

## B Baselines

## B.1 Speech-to-Text

We compare our NAST-S2T with the following strong Simul-S2T baselines.

Wait-k (Ma et al., 2020c): It executes Wait-k policy (Ma et al., 2019) by setting the pre-decision window size to 280 ms.

RealTrans (Zeng et al., 2021): It detects word number in the streaming speech by counting blanks in CTC transcription and applies Wait-k-Stride-n strategy accordingly.

MU-ST (Zhang et al., 2022): It trains an external segmentation model, which is then utilized to detect meaningful units for guiding generation.

Seg2Seg (Zhang and Feng, 2023b): It alternates between waiting for a source segment and generating a target segment in an autoregressive manner.

CAAT (Liu et al., 2021a): It utilizes the Transformer Transducer (Graves, 2012; Zhang et al., 2020) as its foundational architecture for streaming generation and incorporates a cross-attention mechanism within the joiner module to alleviate the strong monotonic constraint.

EDAtt (Papi et al., 2023b): It computes the attention scores towards the latest received speech frames, serving as guidance for an offline-trained speech translation model during simultaneous inference. The experimental results reported in their paper were obtained using a 112M parameter Conformer (Gulati et al., 2020; Papi et al., 2023a). To ensure a fair comparison with our method, we retrained a Conformer<sup>12</sup> of similar size to NAST-S2T on the same dataset to perform EDAtt decoding (52M parameters, achieved by reducing the encoder embedding dimension from 512 to 256 and keeping the number of encoder layers at 12). The numerical results of our re-implemented EDAtt can be found in Tables 7 and 8.

## B.2 Speech-to-Speech

We compare our NAST-S2S with several strong Offline-S2S and Simul-S2S baselines.

## Offline-S2S

S2UT (Lee et al., 2022): A direct speech-to-unit model, which predicts acoustic units in a standard autoregressive manner.

UnitY (Inaguma et al., 2023): A two-pass speech-to-unit model, which first generates a subword sequence in an autoregressive manner and then feeds the last hidden states into another autoregressive model to generate unit sequence.

DASpeech (Fang et al., 2023): A two-pass nonautoregressive speech-to-spectrogram model. It initially employs a directed acyclic graph layer (Huang et al., 2022b) to generate a phoneme sequence, followed by utilizing FastSpeech2 (Ren et al., 2021) to synthesis the phonemes into melspectrograms.

## Simul-S2S

Wait-k-Stride-n: We employ Wait-k strategy (Ma et al., 2019) for S2UT model (Lee et al., 2022) to build an end-to-end Simul-S2S baseline. Since the input is speech audio, a pre-decision module is needed to segment the utterance into multiple chunks to execute Wait-k (Ma et al., 2020c). Furthermore, the translation of a speech chunk can consist of multiple acoustic units to form the pronunciation of a word. It is reasonable to generate multiple unit tokens upon receiving a speech chunk. Therefore, we adopt Wait-k-Stride-n strategy (Zeng et al., 2021) to construct an end-to-end Simul-S2S baseline, varying the speech chunk size and the hyperparameters k and n. The numerical results can be found in Table 13.

EDAtt + Tacotron2: We further provide the results of cascade systems (Simul-S2T + TTS) for comparison. We choose EDAtt (Papi et al., 2023b)

as the Simul-S2T model. According to the recommendation in Papi et al. (2023b), we train a Conformer + CTC compression model (Gaido et al., 2021) with a total of 120M parameters using speech-text parallel pairs of CVSS-C Fr-En dataset as the offline model to implement EDAtt algorithm. For TTS part, we use a Tacotron2 model trained on LJSpeech. Whenever the Simul-S2T model generates a complete word, we send it to the TTS model and generate a speech chunk as output. The numerical results can be found in Table 14.

## C Numerical Results

## C.1 Speech-to-Text

In addition to Average Lagging (AL; Ma et al., 2020c), we also incorporate Average Proportion (AP; Cho and Esipova, 2016), Differentiable Average Lagging (DAL; Arivazhagan et al., 2019) and Length Adaptive Average Lagging (LAAL; Papi et al., 2022) as metrics to evaluate the latency of NAST-S2T. AL, DAL and LAAL are reported with milliseconds. The trade-off between latency and quality is attained by adjusting the chunk size $T _ { s }$ The offline results are obtained by setting the chunk size to be longer than any utterance in the dataset $( T _ { s } = \infty )$ . We use SimulEval v1.1.4 for evaluation in all the experiments. The numerical results of NAST-S2T are presented in Table 5 and 6.

## C.2 Speech-to-Speech

In addition to AL and AL\_EOW, we also present results for AL\_BOW, StartOffset, and EndOffset, as measured by the SimulEval toolkit. AL\_BOW is analogous to AL\_EOW but considers the generation time of each word as the beginning time of the corresponding speech. StartOffset and EndOffset measure the offset of the beginning and ending of the generated speech compared with the input speech. We also employ BLASER 2.0 to assess the quality of translated speech. The trade-off between latency and quality is attained by adjusting the chunk size $T _ { s }$ and the additional frames $T _ { a }$ The offline results are obtained by setting the chunk size to be longer than any utterance in the dataset $( T _ { s } = \infty )$ . We use SimulEval v1.1.4 for evaluation. The numerical results of NAST-S2S are presented in Table 9, 10, 11 and 12.

## D Analysis on Length Ratio

We present the ablation study of model hyperparameter $r _ { \mathrm { d o w n } }$ and $r _ { \mathrm { u p } }$ in Table 15 and 16.

<table><tr><td colspan="6">NAST-S2T on  $E n {  } D e$ </td></tr><tr><td> $T _ { s } ( m s )$ </td><td>AP</td><td>AL</td><td>DAL</td><td>LAAL</td><td>BLEU</td></tr><tr><td>160</td><td>0.58</td><td>1082</td><td>1359</td><td>1191</td><td>19.51</td></tr><tr><td>320</td><td>0.65</td><td>1234</td><td>1546</td><td>1346</td><td>21.56</td></tr><tr><td>640</td><td>0.73</td><td>1582</td><td>1969</td><td>1692</td><td>22.85</td></tr><tr><td>1280</td><td>0.81</td><td>2338</td><td>2812</td><td>2423</td><td>23.30</td></tr><tr><td>∞</td><td>-</td><td>一</td><td>-</td><td>一</td><td>24.54</td></tr></table>

Table 5: Numerical results of NAST-S2T on MuST-C English to German speech-to-text translation dataset.

<table><tr><td colspan="5">NAST-S2T on  $E n {  } E s$ </td></tr><tr><td> $\overline { { T _ { s } ( m s ) } }$ </td><td>AP</td><td>AL</td><td>DAL LAAL</td><td>BLEU</td></tr><tr><td>160 320 640</td><td>0.62 0.71 0.79</td><td>1023 1114 1396</td><td>1541 1242 1692 1377 2030 1648</td><td>23.81 26.48 27.02</td></tr></table>

Table 6: Numerical results of NAST-S2T on MuST-C English to Spanish speech-to-text translation dataset.

<table><tr><td colspan="5">EDAtt on En→De</td></tr><tr><td>α</td><td>AP AL</td><td>DAL</td><td>LAAL</td><td>BLEU</td></tr><tr><td>0.8</td><td>0.80 705</td><td>1973</td><td>1289</td><td>14.43</td></tr><tr><td>0.7</td><td>0.82 1287</td><td>2430</td><td>1765</td><td>15.93</td></tr><tr><td>0.6</td><td>0.86 1996</td><td>3009</td><td>2362</td><td>17.57</td></tr><tr><td>0.5</td><td>0.89 2897</td><td>3736</td><td>3152</td><td>19.87</td></tr><tr><td>0.4</td><td>0.93 4045</td><td>4562</td><td>4149</td><td>22.53</td></tr><tr><td>0.3</td><td>0.97 4947</td><td>5198</td><td>4971</td><td>23.97</td></tr><tr><td>0.2</td><td>0.99 5460</td><td>5540</td><td>5463</td><td>24.54</td></tr><tr><td>0.1</td><td>0.99 5636</td><td>5643</td><td>5636</td><td>24.77</td></tr><tr><td>0</td><td>-</td><td>= –</td><td>-</td><td>25.39</td></tr></table>

Table 7: Numerical results of EDAtt on MuST-C English to German speech-to-text translation dataset.

<table><tr><td colspan="4">EDAtt on En→Es</td></tr><tr><td>α</td><td>AP AL</td><td>DAL LAAL</td><td>BLEU</td></tr><tr><td>0.8 0.7</td><td>0.81 715</td><td>1939 1184 2119</td><td>22.93 24.19</td></tr><tr><td></td><td>0.82 900</td><td>1319</td><td></td></tr><tr><td>0.6</td><td>0.84 1104</td><td>2314 1491</td><td>25.15</td></tr><tr><td>0.5</td><td>0.85 1321</td><td>2489 1661</td><td>26.31</td></tr><tr><td>0.4</td><td>0.87 1547</td><td>2688 1855</td><td>27.02</td></tr><tr><td>0.3</td><td>0.89 1822</td><td>2939 2089</td><td>27.81</td></tr><tr><td>0.2</td><td>0.92 2328</td><td>3454 2554</td><td>28.42</td></tr><tr><td>0.1</td><td>1.00 3853</td><td>4770 3984</td><td>29.11</td></tr><tr><td>0</td><td>- 一</td><td>一</td><td>31.20</td></tr></table>

Table 8: Numerical results of EDAtt on MuST-C English to Spanish speech-to-text translation dataset.

<table><tr><td colspan="7">NAST-S2S on CVSS-C Fr→En</td></tr><tr><td> $\overline { { T _ { s } + T _ { a } ( m s ) } }$ </td><td>AL</td><td>AL EOW</td><td>AL BOW</td><td>StartOffset</td><td>EndOffset</td><td>ASR-BLEU</td></tr><tr><td> $3 2 0 + 3 2 0$ </td><td>-393</td><td>1405</td><td>1085</td><td>655</td><td>562</td><td>19.67</td></tr><tr><td> $6 4 0 + 6 4 0$ </td><td>1533</td><td>1802</td><td>1455</td><td>1295</td><td>863</td><td>19.15</td></tr><tr><td> $1 2 8 0 + 1 2 8 0$ </td><td>3330</td><td>2961</td><td>2601</td><td>2566</td><td>1648</td><td>20.20</td></tr><tr><td> $1 9 2 0 + 1 2 8 0$ </td><td>3975</td><td>3390</td><td>3046</td><td>3179</td><td>1920</td><td>21.77</td></tr><tr><td> $1 9 2 0 + 1 9 2 0$ </td><td>4335</td><td>4021</td><td>3689</td><td>3753</td><td>2292</td><td>22.70</td></tr><tr><td> $2 5 6 0 + 1 2 8 0$ </td><td>4408</td><td>3785</td><td>3448</td><td>3753</td><td>2175</td><td>23.58</td></tr><tr><td> $2 5 6 0 + 2 5 6 0$ </td><td>4976</td><td>4886</td><td>4573</td><td>4697</td><td>2753</td><td>24.88</td></tr><tr><td>8</td><td>-</td><td>一</td><td>一</td><td>一</td><td>一</td><td>25.82</td></tr></table>

Table 9: Numerical results of NAST-S2S on CVSS-C French to English speech-to-speech translation dataset.

NAST-S2S on CVSS-C $F r {  } E n$
<table><tr><td> $\overline { { T _ { s } + T _ { a } ( m s ) } }$ </td><td>AL</td><td> $\mathbf { A L \_ C A }$ </td><td>StartOffset</td><td>StartOffset_CA</td><td>EndOffset</td><td>EndOffset_CA</td></tr><tr><td> $3 2 0 + 3 2 0$ </td><td>-393</td><td>347</td><td>655</td><td>713</td><td>562</td><td>1550</td></tr><tr><td> $6 4 0 + 6 4 0$ </td><td>1533</td><td>1824</td><td>1295</td><td>1351</td><td>863</td><td>1344</td></tr><tr><td> $1 2 8 0 + 1 2 8 0$ </td><td>3330</td><td>3501</td><td>2566</td><td>2642</td><td>1648</td><td>1901</td></tr><tr><td> $1 9 2 0 + 1 2 8 0$ </td><td>3975</td><td>4103</td><td>3179</td><td>3245</td><td>1920</td><td>2088</td></tr><tr><td> $1 9 2 0 + 1 9 2 0$ </td><td>4335</td><td>4482</td><td>3753</td><td>3844</td><td>2291</td><td>2465</td></tr><tr><td> $2 5 6 0 + 1 2 8 0$ </td><td>4408</td><td>4527</td><td>3753</td><td>3823</td><td>2175</td><td>2312</td></tr><tr><td> $2 5 6 0 + 2 5 6 0$ </td><td>4976</td><td>5098</td><td>4697</td><td>4781</td><td>2753</td><td>2879</td></tr></table>

Table 10: Comparison of non-computation-aware and computation-aware metrics results for NAST-S2S on CVSS-C French to English speech-to-speech translation dataset.

<table><tr><td colspan="4">NAST-S2S on CVSS-C Fr→En</td></tr><tr><td> $\overline { { T _ { s } + T _ { a } ( m s ) } }$ </td><td>ASR-BLEU</td><td>ASR-BLEU (Silence Removed)</td><td>AL</td></tr><tr><td> $3 2 0 \substack { + 3 2 0 }$ </td><td>19.67</td><td>24.90</td><td>-393</td></tr><tr><td> $6 4 0 { + } 6 4 0 $ </td><td>19.15</td><td>25.67</td><td>1533</td></tr><tr><td> $1 2 8 0 \substack { + 1 2 8 0 }$ </td><td>20.20</td><td>25.71</td><td>3330</td></tr><tr><td> $2 5 6 0 { + } 2 5 6 0$ </td><td>24.88</td><td>26.14</td><td>4976</td></tr></table>

Table 11: Comparison between ASR-BLEU and ASR-BLEU (Silence Removed) of NAST-S2S on CVSS-C French to English speech-to-speech translation dataset.

<table><tr><td colspan="4">NAST-S2S on CVSS-C Fr→En</td></tr><tr><td> $\overline { { T _ { s } + T _ { a } ( m s ) } }$ </td><td>ASR-BLEU</td><td>BLASER 2.0</td><td>AL</td></tr><tr><td> $3 2 0 \substack { + 3 2 0 }$ </td><td>19.67</td><td>3.022</td><td>-393</td></tr><tr><td> $6 4 0 { + } 6 4 0 $ </td><td>19.15</td><td>3.017</td><td>1533</td></tr><tr><td> $1 2 8 0 \substack { + 1 2 8 0 }$ </td><td>20.20</td><td>3.066</td><td>3330</td></tr><tr><td> $1 9 2 0 \substack { + 1 2 8 0 }$ </td><td>21.77</td><td>3.103</td><td>3975</td></tr><tr><td> $1 9 2 0 + 1 9 2 0$ </td><td>22.70</td><td>3.113</td><td>4335</td></tr><tr><td> $2 5 6 0 \substack { + 1 2 8 0 }$ </td><td>23.58</td><td>3.123</td><td>4408</td></tr><tr><td> $2 5 6 0 \substack { + 2 5 6 0 }$ </td><td>24.88</td><td>3.136</td><td>4976</td></tr><tr><td>∞</td><td>25.82</td><td>3.144</td><td></td></tr><tr><td colspan="4">Offline Models</td></tr><tr><td>S2UT</td><td>23.39</td><td>3.062</td><td>一</td></tr><tr><td>UnitY</td><td>27.80</td><td>3.178</td><td>一</td></tr></table>

Table 12: BLASER 2.0 scores of NAST-S2S on CVSS-C French to English speech-to-speech translation dataset.

<table><tr><td colspan="9">Wait-k-Stride-n on CVSS-C Fr→En</td></tr><tr><td> $\overline { { T _ { s } ( m s ) } }$ </td><td>n</td><td>5</td><td>AL</td><td>StartOffset</td><td>EndOffset</td><td>DCNum</td><td>DCAve</td><td>ASR-BLEU</td></tr><tr><td>320</td><td>5</td><td>5</td><td>-164</td><td>1934</td><td>1503</td><td>11.7</td><td>161</td><td>8.41</td></tr><tr><td>320</td><td>5</td><td>10</td><td>2154</td><td>3472</td><td>2172</td><td>6.9</td><td>136</td><td>13.30</td></tr><tr><td>320</td><td>5</td><td>15</td><td>4023</td><td>4697</td><td>2766</td><td>3.1</td><td>83</td><td>17.06</td></tr><tr><td>640</td><td>10</td><td>1</td><td>1188</td><td>1295</td><td>1242</td><td>6.9</td><td>318</td><td>7.34</td></tr><tr><td>640</td><td>10</td><td>3</td><td>2449</td><td>2566</td><td>1731</td><td>4.9</td><td>294</td><td>11.61</td></tr><tr><td>640</td><td>10</td><td>5</td><td>3627</td><td>3753</td><td>2312</td><td>3.0</td><td>235</td><td>14.55</td></tr><tr><td>1280</td><td>20</td><td>1</td><td>3302</td><td>2566</td><td>1693</td><td>2.5</td><td>541</td><td>14.06</td></tr><tr><td>1280</td><td>20</td><td>2</td><td>4159</td><td>3753</td><td>2248</td><td>1.5</td><td>404</td><td>16.18</td></tr><tr><td>1280</td><td>20</td><td>3</td><td>4859</td><td>4697</td><td>2732</td><td>0.8</td><td>233</td><td>17.91</td></tr></table>

Table 13: Numerical results of Wait-k-Stride-n on CVSS-C French to English speech-to-speech translation dataset.

<table><tr><td colspan="7">EDAtt + Tacotron2 on CVSS-C Fr-  $\scriptstyle \gamma E n$ </td></tr><tr><td>α</td><td>AL</td><td>StartOffset</td><td>EndOffset</td><td>DCNum</td><td>DCAve</td><td>ASR-BLEU</td></tr><tr><td>0.8</td><td>2850</td><td>2131</td><td>5846</td><td>0.8</td><td>360</td><td>11.90</td></tr><tr><td>0.6</td><td>3136</td><td>2383</td><td>5451</td><td>0.8</td><td>442</td><td>13.69</td></tr><tr><td>0.4</td><td>3585</td><td>2859</td><td>4848</td><td>0.7</td><td>472</td><td>15.93</td></tr><tr><td>0.2</td><td>4431</td><td>3922</td><td>3887</td><td>0.4</td><td>358</td><td>19.76</td></tr></table>

Table 14: Numerical results of EDAtt + Tacotron2 on CVSS-C French to English speech-to-speech translation dataset.

<table><tr><td> $r _ { \mathrm { d o w n } }$ </td><td>1</td><td>2</td><td>4</td></tr><tr><td> $L _ { d e c o d e r } / L _ { t a r g e t }$ </td><td>9.3</td><td>4.6</td><td>2.3</td></tr><tr><td>BLEU</td><td>24.52</td><td>24.54</td><td>22.05</td></tr></table>

Table 15: Performance of offline NAST-S2T with varying hyperparameter $r _ { \mathrm { d o w n } }$ on MuST-C English to German speech-to-text translation dataset. $L _ { d e c o d e r }$ and $L _ { t a r g e t }$ represent the length of linguistic decoder and text target, respectively. The average ratio of these lengths is calculated using the training dataset.

<table><tr><td> $r _ { \mathrm { u p } }$ </td><td>4</td><td>6</td><td>8</td></tr><tr><td> $L _ { d e c o d e r } / L _ { t a r g e t }$ </td><td>2.4</td><td>3.6</td><td>4.8</td></tr><tr><td>ASR-BLEU</td><td>25.06</td><td>25.82</td><td>26.16</td></tr></table>

Table 16: Performance of offline NAST-S2S with varying hyperparameter $r _ { \mathrm { u p } }$ when $r _ { \mathrm { d o w n } }$ is fixed to 2 on CVSS-C French to English speech-to-speech translation dataset. $L _ { d e c o d e r }$ and $L _ { t a r g e t }$ represent the length of acoustic decoder and unit target, respectively. The average ratio of these lengths is calculated using the training dataset.