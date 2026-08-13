# GenTranslate: Large Language Models are Generative Multilingual Speech and Machine Translators

Yuchen Hu<sup>1</sup> Chen Chen<sup>1</sup> Chao-Han Huck Yang<sup>2,3</sup> Ruizhe Li<sup>4</sup> Dong Zhang<sup>5</sup> Zhehuai Chen<sup>3</sup> Eng Siong Chng<sup>1</sup> <sup>1</sup>Nanyang Technological University <sup>2</sup>Georgia Institute of Technology <sup>3</sup>NVIDIA <sup>4</sup>University of Aberdeen <sup>5</sup>Fudan University

## Abstract

Recent advances in large language models (LLMs) have stepped forward the development of multilingual speech and machine translation by its reduced representation errors and incorporated external knowledge. However, both translation tasks typically utilize beam search decoding and top-1 hypothesis selection for inference. These techniques struggle to fully exploit the rich information in the diverse N-best hypotheses, making them less optimal for translation tasks that require a single, high-quality output sequence. In this paper, we propose a new generative paradigm for translation tasks, namely “GenTranslate”, which builds upon LLMs to generate better results from the diverse translation versions in N-best list. Leveraging the rich linguistic knowledge and strong reasoning abilities of LLMs, our new paradigm can integrate the rich information in N-best candidates to generate a higher-quality translation result. Furthermore, to support LLM finetuning, we build and release a HypoTranslate dataset that contains over 592K hypotheses-translation pairs in 11 languages. Experiments on various speech and machine translation benchmarks (e.g., FLEURS, CoVoST-2, WMT) demonstrate that our Gen-Translate significantly outperforms the state-ofthe-art model<sup>1</sup>.

## 1 Introduction

Recent advances in large language models (LLMs) have attracted a surge of research interest due to their strong abilities in logical reasoning and language generation (OpenAI, 2022, 2023; Touvron et al., 2023a,b). These models have achieved surprisingly wide-ranging success across various natural language processing (NLP) tasks (Brown et al., 2020; Wang et al., 2022; Wei et al., 2022a,b; Ouyang et al., 2022).

![](images/20f0de7331c93b45f8ae0587b66dcadb190e886540906404869c6465e71e5f6f.jpg)

![](images/72ccb45d61942f7f3bed48293d78f93e51e270ce71ed1d6a10a6bed2146cd197.jpg)  
Figure 1: Illustration of (a) Typical seq2seq translation with beam search decoding and top-1 hypothesis selection, (b) our “GenTranslate” with LLM integration.

In the realm of NLP, the translation tasks, which encompasses speech and machine translation (ST & MT), hold significant practical importance for global communication. Similar to other NLP tasks, translation tasks also gain a notable progress thanks to the recent advancement of LLMs (Zhang et al., 2023a; Lyu et al., 2023). In the domain of speech translation, Whisper (Radford et al., 2023) demonstrates superior performance by collecting 680K-hour data for web-scale model training. AudioPaLM2 (Rubenstein et al., 2023) integrates both text- and speech-based language models into a unified architecture to process and generate text and speech, thereby augmenting speech translation performance to a great extent. On the other hand, LLMs also show remarkable ability in machine translation. NLLB (Costa-jussà et al., 2022) is the first to extend LLMs’ linguistic capability to over 200 languages. BigTranslate (Yang et al., 2023b) is finetuned on LLaMA (Touvron et al., 2023a) with multilingual instruction tuning, which achieves comparable performance to ChatGPT (OpenAI, 2022) and Google Translate. Most recent work proposes SeamlessM4T (Barrault et al., 2023a), a foundational multilingual and multitask model that can translate across speech and text, which achieves the state-of-the-art on both ST and MT tasks on various public datasets.

Despite the superior performance, most existing translation models employ the typical beam search algorithm for inference and select the top-1 hypothesis as final output (see Fig. 1 (a)), following that in automatic speech recognition (ASR) (Tsunoo et al., 2021). However, this strategy discards the 2 to N-best hypotheses that could be advantageous to the generation of ground-truth translation. As illustrated in Fig. 2, the discarded 2 to N-best hypotheses contain abundant semantic information that is the key to composite the ground-truth utterance, while the 1-best hypothesis lacks this part of information. As a result, the typical top-1 hypothesis selection is sub-optimal to the translation tasks that require a single informative and high-quality output sequence (Li et al., 2022; Xiao et al., 2022).

Inspired by the recent works on LLMs-enhanced ASR (Ma et al., 2023b; Chen et al., 2023; Yang et al., 2023a; Radhakrishnan et al., 2023), we propose a new generative paradigm for translation tasks, namely GenTranslate (see Fig. 1 (b)). Leveraging the rich linguistic knowledge and strong reasoning ability of LLMs, our paradigm integrates the diverse translation versions in the N-best list from foundation model to generate a higher-quality translation result. Furthermore, in order to support LLM finetuning, we also build and release a Hypo-Translate dataset that contains over 592K pairs of N-best hypotheses and ground-truth translation in 11 languages. Experimental evidence on various ST and MT benchmarks (e.g., FLEURS, CoVoST-2, WMT) demonstrate that our proposed GenTranslate significantly outperforms the state-of-the-art model with efficient LLM finetuning.

Our contributions are summarized as follows:

• We propose GenTranslate, a new generative paradigm for translation tasks that leverages LLMs to generate higher-quality translation results from the diverse N-best hypotheses decoded from foundation translation model.

• We release a HypoTranslate dataset to support LLM finetuning, which contains over 592K pairs of N-best hypotheses and ground-truth translation in 11 languages.

• Experiments on various ST and MT benchmarks show that our GenTranslate significantly outperforms the state-of-the-art model.

![](images/fd0908308611619979a123e3c4d4e9ad9741a7a98b3c3f443452afdf3f9003c8.jpg)  
Figure 2: t-SNE visualization of the n-gram tokens (n=1,2,3) in ST 1-best hypothesis (green), 2 to N-best hypotheses (blue), and the ground-truth translation (orange), where the text embeddings are extracted using SBERT (Reimers and Gurevych, 2019). It indicates that the 2 to N-best hypotheses contain richer information than 1-best for generating ground-truth translation.

## 2 Related Work

## 2.1 Large Language Models

There is recently a surge of research interests in Transformer-based large language models, such as ChatGPT (OpenAI, 2022), GPT-4 (OpenAI, 2023) and LLaMA (Touvron et al., 2023a,b). Benefiting from the giant model size and oceans of training data, LLMs can understand better the linguistic structures and semantic meanings behind raw text, which thus shows remarkable performance on a wide range of natural language processing (NLP) tasks (Brown et al., 2020; Wei et al., 2022a; Ouyang et al., 2022). Thereafter, with techniques like incontext learning (Xie et al., 2021) and efficient finetuning (Hu et al., 2021; Yang et al., 2021b), LLMs further show powerful ability on downstream generative and reasoning tasks (Lampinen et al., 2022; Yang et al., 2023a; Hu et al., 2023b; Zhang et al., 2023b). Our proposed GenTranslate is exactly inspired by the promising generative ability of LLMs.

## 2.2 Speech and Machine Translation

The advancement of LLMs has notably enhanced the capabilities of translation tasks. In the domain of speech translation (Liu et al., 2021), Whisper (Radford et al., 2023) demonstrates commendable effectiveness, leveraging extensive web-scale data. AudioPaLM2 (Rubenstein et al., 2023) integrates text- and speech-based language models, thereby augmenting the speech translation performance. In the context of machine translation, NLLB (Costa-jussà et al., 2022), a model finetuned on LLMs, extends its linguistic range to over 200 languages. Additionally, BigTranslate (Yang et al., 2023b) utilizes instruction tuning to enhance the translation capabilities of LLMs. The most recent innovation, SeamlessM4T (Barrault et al., 2023a), represents a highly-unified model capable of fluid translation between speech and text, setting new benchmarks in both ST and MT tasks. However, it is noteworthy that the majority of these methodologies rely on beam search decoding (Yang et al., 2021a; Hu et al., 2023a) and top-1 hypothesis selection for inference. How to leverage N-best hypotheses to deliver better translation result remains to be an open question.

![](images/636f2965307bb632176a46369168cfee3e57444951041b9958c7a4e776c7d7c9.jpg)  
Figure 3: Left: Overview of the GenTranslate paradigm (e.g., De En). Right: Details of efficient LLM finetuning.

## 2.3 LLMs-Enhanced ASR

Recent works investigate LLMs to enhance the ASR output by error correction (Ma et al., 2023a; Chen et al., 2023), which serves as a postprocessing technique to improve the recognition result (Leng et al., 2021). In particular, they leverage LLM finetuning (Zhang et al., 2023b) and in-context learning (Wang et al., 2023) to correct the wrongly recognized tokens in hypotheses by second-pass reasoning, which achieves promising improvement. Inspired by them, in this work we leverage LLMs to integrate the diverse translation versions in N-best list to generate a informative and higher-quality translation result.

## 3 Methodology

In this section, we introduce the proposed method. First, we describe the latest foundational translation model, SeamlessM4T, which we employ for beam search decoding and hypotheses generation (§3.1).

Then, we introduce our LLMs-based GenTranslate paradigm by N-best hypotheses integration (§3.2). Finally, we present the details of our released Hypo-Translate dataset for GenTranslate training (§3.3).

## 3.1 Foundational Translation Model: SeamlessM4T

Recent work (Barrault et al., 2023a,b) proposes SeamlessM4T<sup>2</sup> (Massively Multilingual & Multimodal Machine Translation), a single Transformerbased (Vaswani et al., 2017) model that supports speech-to-speech translation, speech-to-text translation, text-to-speech translation, text-to-text translation, and automatic speech recognition for up to 100 languages. During development process, it is firstly pre-trained on 1 million hours of speech data by self-supervised learning, and it is then finetuned on a 406K-hour multimodal corpus of automatically aligned speech translations named SeamlessAlign. Experiments show that SeamlessM4T yields superior performance on all of the five supported tasks. In particular, it has achieved the stateof-the-art on both ST and MT tasks in terms of BLEU score on various public benchmarks.

Considering its effectiveness, generality and popularity, we employ SeamlessM4T as the foundation model for both speech and machine translation in our system, as depicted in the left part of Fig. 3. Given an input speech S<sup>src</sup> or text $T ^ { \mathrm { s r c } }$ in source language (e.g., German), SeamlessM4T translates it into target language (e.g., English) text by beam search decoding, which generates N-best hypotheses list $\mathcal { T } _ { N } ^ { \mathrm { t g t } } = \bar { \{ } T _ { 1 } ^ { \mathrm { t g t } } , T _ { 2 } ^ { \mathrm { t g \bar { t } } } , \cdot \cdot \cdot , T _ { N } ^ { \mathrm { t g t } } \}$

## 3.2 GenTranslate

## 3.2.1 Overall Framework

To solve the information loss in typical top-1 hypothesis selection, we leverage LLMs to generate a final translation result based on the decoded Nbest hypotheses. Since each candidate in N-best list represents one unique version of translation for source language input, our GenTranslate can integrate their rich information to generate a higherquality translation result, thanks to the strong linguistic and reasoning ability of LLMs. This new generative paradigm can be formulated as:

$$
\begin{array} { r } { T ^ { \mathrm { t g t } } = \mathcal { M } _ { \mathrm { G T } } ( \mathcal { T } _ { N } ^ { \mathrm { t g t } } , \mathcal { T } ) , } \end{array}\tag{1}
$$

where $\mathcal { T }$ is a proper instruction for LLM prompting. The goal of GenTranslate is to learn a mapping $\mathcal { M } _ { \mathrm { G T } }$ from N-best hypotheses to the true translation. Following typical sequence-to-sequence learning strategy, we employ the ground-truth translation $T ^ { \mathrm { t g t ^ { * } } }$ as supervision signal and optimize the LLM to learn $\mathcal { M } _ { \mathrm { G T } }$ in an auto-regressive manner. The cross-entropy-based training loss is defined as:

$$
\mathcal { L } _ { \mathrm { G T } } = \sum _ { l = 1 } ^ { L } - \log \mathbb { P } _ { \theta } ( t _ { l } ^ { \mathrm { t g t ^ { \ast } } } | t _ { l - 1 } ^ { \mathrm { t g t ^ { \ast } } } , \cdot \cdot \cdot , t _ { 1 } ^ { \mathrm { t g t ^ { \ast } } } ; \mathcal { T } _ { N } ^ { \mathrm { t g t } } , \mathcal { T } ) ,\tag{2}
$$

where $t _ { l } ^ { \mathrm { t g t ^ { * } } }$ is the l-th token of $T ^ { \mathrm { t g t ^ { * } } }$ , L denotes the sequence length, and θ denotes the learnable parameters in LLM (i.e., adapter).

## 3.2.2 Efficient LLM Finetuning

Considering the giant scale of LLMs, we adopt the popular efficient finetuning strategy, LLaMA-Adapter (Zhang et al., 2023b), which is comparable to LoRA tuning (§4.3.4). As shown in Fig. 3 (right), it inserts a set of learnable adaptation prompts into the top-L of total H Transformer layers in a pretrained LLM to learn high-level semantics. Denote the prompt for l-th layer as $\mathcal { P } _ { l } \in \mathbb { R } ^ { U \times D }$ , where U is prompt length and D is embedding size.

Assume we gain M tokens including instruction and already generated response, $i . e . , \bar { T } _ { l } \in \mathbb { R } ^ { M \times D }$ now we aim to predict the $( M + 1 )$ -th token as response. The learnable adaptation prompt is concatenated with $T _ { l }$ as prefix, i.e., $[ \mathcal { P } _ { l } ; \hat { T } _ { l } ] \in \mathbf { \hat { \mathbb { R } } } ^ { ( U + M ) \times D }$ which provides learned instruction knowledge to guide the subsequent response generation.

Furthermore, considering the prompt $\mathcal { P } _ { l }$ is randomly initialized and thus could disturb the LLM tuning at early training stage, a zero-initialized attention mechanism is devised to mitigate such disturbance. Denote the current M-th token as $T _ { l } ^ { ( M ) } \in \mathbb { R } ^ { 1 \times D }$ , in attention there are three projection layers to generate query, key and value:

$$
\begin{array} { r l } & { Q _ { l } = \mathrm { L i n e a r } _ { q } ( T _ { l } ^ { ( M ) } ) , } \\ & { K _ { l } = \mathrm { L i n e a r } _ { k } ( [ \mathcal { P } _ { l } ; T _ { l } ] ) , } \\ & { V _ { l } = \mathrm { L i n e a r } _ { v } ( [ \mathcal { P } _ { l } ; T _ { l } ] ) , } \end{array}\tag{3}
$$

Then the attention score is calculated as $A _ { l } \ =$ $Q _ { l } \cdot K _ { l } / \sqrt { D } \in \mathbb { R } ^ { 1 \times ( U + M ) }$ , which captures the correlation between current token and the history tokens as well as prompts to predict the next token. Therefore, it can be split into two parts accordingly:

$$
A _ { l } = [ A _ { l } ^ { \mathcal { P } } ; A _ { l } ^ { T } ] ^ { T } ,\tag{4}
$$

where $A _ { l } ^ { \mathcal { P } } ~ \in ~ \mathbb { R } ^ { U \times 1 }$ is the attention score of U adaptation prompts and $A _ { l } ^ { T } \in \mathbb { R } ^ { M \times 1 }$ is that of M history tokens. Since the adaptation prompts are randomly initialized, their attention scores may cast disturbance on next-token prediction at early training stage. To this end, a learnable gating factor g<sub>l</sub> with zero initialization is introduced to adaptively control the weight of prompt in attention:

$$
\begin{array} { r } { A _ { l } ^ { g } = [ g _ { l } \cdot \mathrm { s o f t m a x } ( A _ { l } ^ { \mathcal { P } } ) ; \mathrm { s o f t m a x } ( A _ { l } ^ { T } ) ] ^ { T } , } \end{array}\tag{5}
$$

Finally, the attention output of l-th Transformer layer is obtained with a linear projection:

$$
O _ { l } ^ { ( M ) } = \operatorname { L i n e a r } _ { o } ( A _ { l } ^ { g } \cdot V _ { l } ) \in \mathbb { R } ^ { 1 \times D } ,\tag{6}
$$

It is then employed to predict the next token $T _ { l } ^ { ( M + 1 ) }$ as response. The zero-initialization mechanism yields an effective trade-off between the pretrained knowledge of LLM and the learned instructional knowledge through adaptation prompt.

## 3.3 HypoTranslate Dataset

In order to support the LLM finetuning for Gen-Translate, we release a HypoTranslate dataset that contains over 592K pairs of N-best hypotheses and ground-truth translation in 11 languages. In particular, we use the state-of-the-art SeamlessM4T-Large as foundation translation model to decode N-best hypotheses from input speech by beam search algorithm, where the beam size N is set to 5. Specifically, for ST task we investigate two popular pipelines in literature, i.e., end-to-end ST and cascaded ASR+MT. Thanks to the universal ability of SeamlessM4T on ST, ASR and MT tasks, we only need one model to build above two pipelines.

To build HypoTranslate dataset, we select several public ST and MT corpora in both X En and

<table><tr><td>X→En</td><td>Ar</td><td>Cy</td><td>De</td><td>El</td><td>Es</td><td>Fa</td><td>Fr</td><td>Hi</td><td>It</td><td>Ja</td><td>Pt</td><td>Ta</td><td>Uk</td><td>Vi</td><td>Zh</td><td>Avg.</td></tr><tr><td>End-to-end ST Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Whisper-Large V2 (2023)</td><td>25.5</td><td>13.0</td><td>34.6</td><td>23.7</td><td>23.3</td><td>19.6</td><td>32.2</td><td>22.0</td><td>23.6</td><td>18.9</td><td>38.1</td><td>9.2</td><td>29.4</td><td>20.4</td><td>18.4</td><td>23.5</td></tr><tr><td>AudioPaLM2 (2023)*</td><td>29.0</td><td>7.2</td><td>38.7</td><td>18.8</td><td>26.9</td><td>25.7</td><td>36.5</td><td>21.7</td><td>27.8</td><td>11.1</td><td>38.4</td><td>15.0</td><td>26.9</td><td>15.6</td><td>21.3</td><td>24.0</td></tr><tr><td>SeamlessM4T-Large (2023a)</td><td>32.8</td><td>31.7</td><td>35.8</td><td>25.6</td><td>25.0</td><td>28.2</td><td>33.1</td><td>26.3</td><td>25.0</td><td>17.0</td><td>38.9</td><td>16.0</td><td>30.2</td><td>21.6</td><td>19.8</td><td>27.1</td></tr><tr><td>GenTranslate (ours)</td><td>34.6</td><td>33.6</td><td>39.2</td><td>29.4</td><td>29.8</td><td>30.5</td><td>37.0</td><td>28.3</td><td>29.7</td><td>18.6</td><td>43.0</td><td>17.4</td><td>33.9</td><td>24.1</td><td>21.7</td><td>30.1</td></tr><tr><td>SeamlessM4T-Large-V2 (2023b)†</td><td>34.7</td><td>34.9</td><td>37.1</td><td>27.3</td><td>25.4</td><td>30.3</td><td>33.7</td><td>28.5</td><td>26.5</td><td>19.5</td><td>38.5</td><td>22.1</td><td>33.2</td><td>25.7</td><td>23.0</td><td>29.4</td></tr><tr><td>GenTranslate-V2 (ours)</td><td>37.6</td><td>36.8</td><td>40.7</td><td>31.5</td><td>29.9</td><td>33.4</td><td>37.8</td><td>30.4</td><td>31.2</td><td>21.0</td><td>43.0</td><td>23.4</td><td>36.2</td><td>27.2</td><td>25.0</td><td>32.3</td></tr><tr><td>Cascaded ASR+MT Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Whisper + NLLB-3.3b (2022)</td><td>35.5</td><td>29.6</td><td>40.5</td><td>31.1</td><td>30.9</td><td>28.2</td><td>39.7</td><td>26.7</td><td>30.0</td><td>24.7</td><td>44.3</td><td>20.0</td><td>35.3</td><td>26.4</td><td>25.4</td><td>31.2</td></tr><tr><td>SeamlessM4T (ASR+MT) (2023a)</td><td>38.9</td><td>37.0</td><td>39.7</td><td>29.0</td><td>27.7</td><td>34.1</td><td>37.7</td><td>33.9</td><td>28.9</td><td>21.7</td><td>42.3</td><td>23.7</td><td>34.0</td><td>24.9</td><td>24.4</td><td>31.9</td></tr><tr><td>GenTranslate (ours)</td><td>39.9</td><td>39.4</td><td>41.6</td><td>32.8</td><td>31.2</td><td>35.9</td><td>40.6</td><td>34.9</td><td>32.1</td><td>22.8</td><td>45.0</td><td>24.1</td><td>36.9</td><td>27.4</td><td>25.7</td><td>34.0</td></tr><tr><td>SeamlessM4T-V2 (ASR+MT) (2023b)†</td><td>39.2</td><td>36.8</td><td>39.1</td><td>29.4</td><td>26.7</td><td>33.9</td><td>35.7</td><td>32.9</td><td>29.3</td><td>22.5</td><td>43.2</td><td>25.4</td><td>34.8</td><td>29.7</td><td>25.9</td><td>32.3</td></tr><tr><td>GenTranslate-V2 (ours)</td><td>40.0</td><td>39.1</td><td>40.9</td><td>33.8</td><td>30.0</td><td>35.4</td><td>40.0</td><td>33.0</td><td>31.6</td><td>23.7</td><td>44.2</td><td>26.4</td><td>37.1</td><td>30.9</td><td>26.9</td><td>34.2</td></tr></table>

Table 1: Speech translation results on FLEURS X En test sets in terms of BLEU score, where more results on chrF++ metric (Popovic´, 2017) are in Table 16. We use bold to denote surpassing SeamlessM4T baseline, and use underline to denote the state-of-the-art. The baseline methods are introduced in §B.3. \* denotes reported by original paper, or else it denotes reproduced by ourselves (same for Table 2 to 5). denotes the most latest baseline<sup>3</sup>.
<table><tr><td>X→En</td><td>Fr</td><td>De</td><td>Ca</td><td>Es</td><td>Ru</td><td>Zh</td><td>Nl</td><td>Tr</td><td>Et</td><td>Mn</td><td>Ar</td><td>Lv</td><td>Sl</td><td>Ja</td><td>Id</td><td>Avg.</td></tr><tr><td>End-to-end ST Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>XLS-R-2b (2021)*</td><td>37.6</td><td>33.6</td><td>33.8</td><td>39.2</td><td>39.5</td><td>9.4</td><td>31.7</td><td>16.7</td><td>11.1</td><td>1.6</td><td>17.1</td><td>19.5</td><td>19.6</td><td>3.5</td><td>16.5</td><td>22.0</td></tr><tr><td>Whisper-Large V2 (2023)</td><td>35.5</td><td>35.0</td><td>31.0</td><td>39.6</td><td>42.3</td><td>16.9</td><td>40.2</td><td>27.5</td><td>14.0</td><td>0.2</td><td>38.5</td><td>13.0</td><td>16.3</td><td>24.7</td><td>47.3</td><td>28.1</td></tr><tr><td>ComSL-Large (2023)*</td><td>38.8</td><td>36.0</td><td>35.3</td><td>40.4</td><td>49.2</td><td>21.4</td><td>39.7</td><td>33.6</td><td>19.2</td><td>2.9</td><td>41.4</td><td>21.3</td><td>31.6</td><td>21.3</td><td>46.6</td><td>31.9</td></tr><tr><td>AudioPaLM2 (2023)*</td><td>44.8</td><td>43.4</td><td>38.4</td><td>44.2</td><td>55.6</td><td>25.5</td><td>48.3</td><td>41.0</td><td>30.0</td><td>7.6</td><td>48.7</td><td>35.0</td><td>42.6</td><td>25.9</td><td>56.2</td><td>39.1</td></tr><tr><td>SeamlessM4T-Large (2023a)</td><td>41.3</td><td>38.8</td><td>38.4</td><td>41.1</td><td>48.6</td><td>20.9</td><td>41.1</td><td>31.2</td><td>26.3</td><td>7.5</td><td>45.0</td><td>26.5</td><td>37.6</td><td>21.8</td><td>51.4</td><td>34.5</td></tr><tr><td>GenTranslate (ours)</td><td>41.7</td><td>39.2</td><td>38.7</td><td>42.0</td><td>50.1</td><td>21.6</td><td>42.1</td><td>33.5</td><td>28.2</td><td>8.7</td><td>49.7</td><td>30.3</td><td>38.2</td><td>22.9</td><td>54.3</td><td>36.1</td></tr><tr><td>SeamlessM4T-Large-V2 (2023b)</td><td>42.4</td><td>40.0</td><td>39.0</td><td>42.9</td><td>53.6</td><td>22.4</td><td>42.7</td><td>33.2</td><td>26.9</td><td>8.6</td><td>46.5</td><td>27.5</td><td>41.7</td><td>23.7</td><td>52.6</td><td>36.2</td></tr><tr><td>GenTranslate-V2 (ours)</td><td>42.7</td><td>40.6</td><td>39.4</td><td>43.6</td><td>54.0</td><td>23.3</td><td>44.8</td><td>37.0</td><td>27.7</td><td>10.2</td><td>48.0</td><td>30.5</td><td>42.3</td><td>25.4</td><td>55.9</td><td>37.7</td></tr><tr><td>Cascaded ASR+MT Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Whisper + NLLB-3.3b (2022)</td><td>34.4</td><td>35.5</td><td>31.7</td><td>37.9</td><td>45.4</td><td>19.0</td><td>39.8</td><td>26.7</td><td>17.5</td><td>0.1</td><td>37.0</td><td>20.6</td><td>29.4</td><td>25.5</td><td>45.9</td><td>29.8</td></tr><tr><td>Whisper + mBART-50 (2023)*</td><td>38.8</td><td>37.0</td><td>33.0</td><td>40.7</td><td>49.0</td><td>21.5</td><td>39.9</td><td>32.7</td><td>16.3</td><td>0.4</td><td>37.0</td><td>21.4</td><td>25.0</td><td>23.0</td><td>45.5</td><td>30.7</td></tr><tr><td>SeamlessM4T (ASR+MT) (2023a)</td><td>41.5</td><td>39.8</td><td>37.5</td><td>41.1</td><td>53.2</td><td>21.4</td><td>42.4</td><td>29.9</td><td>26.5</td><td>8.0</td><td>45.2</td><td>28.8</td><td>38.6</td><td>22.0</td><td>50.6</td><td>35.1</td></tr><tr><td>GenTranslate (ours)</td><td>41.8</td><td>40.2</td><td>38.4</td><td>42.1</td><td>53.7</td><td>22.9</td><td>43.8</td><td>34.3</td><td>29.4</td><td>9.5</td><td>49.7</td><td>31.2</td><td>39.6</td><td>22.3</td><td>54.6</td><td>36.9</td></tr><tr><td>SeamlessM4T-V2 (ASR+MT) (2023b)</td><td>43.0</td><td>40.6</td><td>38.8</td><td>43.0</td><td>55.2</td><td>22.9</td><td>43.2</td><td>33.9</td><td>27.2</td><td>8.6</td><td>47.0</td><td>27.8</td><td>41.9</td><td>24.7</td><td>53.1</td><td>36.7</td></tr><tr><td>GenTranslate-V2 (ours)</td><td>43.1</td><td>41.1</td><td>39.5</td><td>43.3</td><td>55.6</td><td>24.5</td><td>44.9</td><td>37.4</td><td>27.8</td><td>10.3</td><td>48.7</td><td>30.4</td><td>42.0</td><td>26.0</td><td>58.4</td><td>38.2</td></tr></table>

Table 2: Speech translation results on CoVoST-2 X En test sets in terms of BLEU score. Remarks follow Table 1.

En X language directions. For speech translation, we select FLEURS (Conneau et al., 2023), CoVoST-2 (Wang et al., 2020), and MuST-C (Di Gangi et al., 2019). For machine translation, we select FLO-RES (Costa-jussà et al., 2022), WMT’16 (Bojar et al., 2016), WMT’19 (Barrault et al., 2019), and WMT’20 (Loïc et al., 2020) corpora. As a result, we obtain over 592K hypotheses-translation pairs in 11 languages. The details of dataset statistics are presented in §A.3 and Table 15, 17.

Since the hypotheses-translation data pairs in HypoTranslate dataset are monolingual, we can also use ASR dataset to benefit GenTranslate training, especially for low-resource language pairs. Relevant studies are illustrated in §4.3.2 and Table 7. Our best result was obtained by first performing translation with SeamlessM4T and then integrating the N-best candidates using LLMs.

## 4 Experiments

## 4.1 Setup

## 4.1.1 Model Selection

LLMs. We select the popular LLaMA-2 (Touvron et al., 2023b) for our paradigm. Specifically, we employ LLaMA-2-7b<sup>4</sup> for English-target directions (X En) and LLaMA-2-13b for non-English-target directions (En X), as LLaMA-2 shows superior ability on English language while less-optimal on other languages. In addition, for En X we also try some latest multilingual LLMs like BigTranslate<sup>5</sup> (Yang et al., 2023b) and ALMA<sup>6</sup> (Xu et al., 2023b) that are finetuned on LLaMA-13b.

Adapter. We follow the default settings of LLaMA-Adapter (Zhang et al., 2023b). The number of tunable Transformer layers L is set to H 1, which means all layers except the first one are tunable with inserted prompts. The prompt length U is set to 10. More details are provided in §B.1.

<table><tr><td rowspan="2">En→X</td><td colspan="6">FLEURS</td><td colspan="6">CoVoST-2</td><td colspan="3">MuST-C</td></tr><tr><td>Es</td><td>Fr</td><td>It</td><td>Ja</td><td>Pt</td><td>Zh</td><td>Avg.</td><td>Fa</td><td>Ja</td><td>Zh</td><td>Avg.</td><td>Es</td><td>It</td><td>Zh</td><td>Avg.</td></tr><tr><td>End-to-end ST Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SeamlessM4T-Large (2023a)</td><td>23.8</td><td>41.6</td><td>23.9</td><td>21.0</td><td>40.8</td><td>28.6</td><td>30.0</td><td>18.3</td><td>24.0</td><td>34.1</td><td>25.5</td><td>34.2</td><td>29.9</td><td>16.2</td><td>26.8</td></tr><tr><td>GenTranslate (ours)</td><td>25.4</td><td>43.1</td><td>25.5</td><td>28.3</td><td>42.4</td><td>34.3</td><td>33.2</td><td>21.1</td><td>29.1</td><td>42.8</td><td>31.0</td><td>33.9</td><td>29.4</td><td>18.5</td><td>27.3</td></tr><tr><td>SeamlessM4T-Large-V2 (2023b)</td><td>23.8</td><td>42.6</td><td>24.5</td><td>21.7</td><td>43.0</td><td>29.5</td><td>30.9</td><td>16.9</td><td>23.5</td><td>34.6</td><td>25.0</td><td>32.1</td><td>27.5</td><td>15.6</td><td>25.1</td></tr><tr><td>GenTranslate-V2 (ours)</td><td>25.5</td><td>44.0</td><td>26.3</td><td>28.9</td><td>44.5</td><td>34.9</td><td>34.0</td><td>19.4</td><td>29.0</td><td>43.6</td><td>30.7</td><td>32.2</td><td>27.3</td><td>18.1</td><td>25.9</td></tr><tr><td>Cascaded ASR+MT Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Whisper + NLLB-3.3b (2022)</td><td>25.1</td><td>41.3</td><td>25.0</td><td>19.0</td><td>41.5</td><td>23.5</td><td>29.2</td><td>13.6</td><td>19.0</td><td>32.0</td><td>21.5</td><td>35.3</td><td>29.9</td><td>13.5</td><td>26.2</td></tr><tr><td>SeamlessM4T-Large (ASR+MT) (2023a)</td><td>24.6</td><td>44.6</td><td>25.4</td><td>22.5</td><td>41.9</td><td>31.2</td><td>31.7</td><td>18.8</td><td>24.0</td><td>35.1</td><td>26.0</td><td>35.1</td><td>30.8</td><td>17.7</td><td>27.9</td></tr><tr><td>GenTranslate (ours)</td><td>26.8</td><td>45.0</td><td>26.6</td><td>29.4</td><td>43.1</td><td>36.8</td><td>34.6</td><td>21.8</td><td>30.5</td><td>43.3</td><td>31.9</td><td>35.5</td><td>31.0</td><td>19.6</td><td>28.7</td></tr><tr><td>SeamlessM4T-V2 (ASR+MT) (2023b)</td><td>24.7</td><td>44.1</td><td>25.1</td><td>20.6</td><td>43.6</td><td>30.6</td><td>31.5</td><td>17.4</td><td>23.8</td><td>35.4</td><td>25.5</td><td>33.0</td><td>27.8</td><td>14.5</td><td>25.1</td></tr><tr><td>GenTranslate-V2 (ours)</td><td>27.0</td><td>44.3</td><td>26.4</td><td>27.8</td><td>44.5</td><td>36.1</td><td>34.4</td><td>20.8</td><td>29.7</td><td>43.5</td><td>31.3</td><td>33.2</td><td>28.3</td><td>16.9</td><td>26.1</td></tr></table>

Table 3: Speech translation results on FLEURS, CoVoST-2, and MuST-C En X test sets in terms of BLEU score. We use bold to highlight surpassing SeamlessM4T baseline, and use underline to highlight the state-of-the-art performance. The baseline methods are introduced in §B.3, and all of their results are reproduced by ourselves.
<table><tr><td>X→En</td><td>Ar</td><td>De</td><td>El</td><td>Es</td><td>Fa</td><td>Fr</td><td>It</td><td>Ja</td><td>Uk</td><td>Zh</td><td>Avg.</td></tr><tr><td>ALMA-13b (Xu et al., 2023b)</td><td>10.8</td><td>27.7</td><td>12.1</td><td>18.1</td><td>10.2</td><td>27.4</td><td>19.6</td><td>14.2</td><td>22.7</td><td>16.9</td><td>18.0</td></tr><tr><td>BigTranslate (Yang et al., 2023b)</td><td>18.6</td><td>35.9</td><td>9.5</td><td>29.0</td><td>1.4</td><td>38.7</td><td>29.0</td><td>16.9</td><td>25.9</td><td>23.0</td><td>22.8</td></tr><tr><td>NLLB-3.3b (Costa-jussà et al., 2022)</td><td>43.0</td><td>44.6</td><td>37.7</td><td>32.2</td><td>38.7</td><td>46.2</td><td>34.6</td><td>28.1</td><td>40.8</td><td>29.5</td><td>37.5</td></tr><tr><td>SeamlessM4T-Large (Barrault et al., 2023a)</td><td>43.7</td><td>45.1</td><td>37.7</td><td>31.5</td><td>39.0</td><td>45.1</td><td>35.2</td><td>26.1</td><td>41.2</td><td>29.9</td><td>37.5</td></tr><tr><td>GenTranslate (ours)</td><td>43.9</td><td>45.3</td><td>38.5</td><td>35.5</td><td>39.4</td><td>46.4</td><td>36.6</td><td>26.7</td><td>41.8</td><td>30.5</td><td>38.5</td></tr><tr><td>SeamlessM4T-Large-V2 (Barrault et al., 2023b)</td><td>41.5</td><td>44.1</td><td>35.6</td><td>29.9</td><td>37.6</td><td>45.5</td><td>33.5</td><td>25.5</td><td>39.0</td><td>29.0</td><td>36.1</td></tr><tr><td>GenTranslate-V2 (ours)</td><td>42.0</td><td>44.5</td><td>36.6</td><td>34.4</td><td>38.1</td><td>46.7</td><td>35.1</td><td>26.7</td><td>39.3</td><td>29.9</td><td>37.3</td></tr></table>

Table 4: Machine translation results on FLORES X En test sets in terms of BLEU score. Remarks follow Table 3.
<table><tr><td>En→X</td><td>WMT&#x27;16</td><td colspan="2">WMT&#x27;19</td><td colspan="2">WMT&#x27;20</td><td rowspan="2">Avg.</td></tr><tr><td></td><td>Ro</td><td>Cs</td><td>Lt</td><td>Ja</td><td>Zh</td></tr><tr><td>ALMA-13b (2023b)</td><td>6.2</td><td>6.1</td><td>0.3</td><td>3.5</td><td>11.3</td><td>5.5</td></tr><tr><td>BigTranslate (2023b)</td><td>21.4</td><td>19.0</td><td>8.7</td><td>7.3</td><td>29.0</td><td>17.1</td></tr><tr><td>NLLB-3.3b (2022)</td><td>31.0</td><td>25.3</td><td>16.0</td><td>15.2</td><td>26.9</td><td>22.9</td></tr><tr><td>SeamlessM4T-Large</td><td>32.7</td><td>26.0</td><td>17.2</td><td>17.0</td><td>27.2</td><td>24.0</td></tr><tr><td>GenTranslate (ours)</td><td>33.5</td><td>27.2</td><td>19.4</td><td>21.4</td><td>30.7</td><td>26.4</td></tr><tr><td>SeamlessM4T-Large-V2</td><td>32.2</td><td>25.2</td><td>16.2</td><td>15.2</td><td>28.7</td><td>23.5</td></tr><tr><td>GenTranslate-V2 (ours)</td><td>33.2</td><td>26.6</td><td>18.2</td><td>19.3</td><td>31.6</td><td>25.8</td></tr></table>

Table 5: Machine translation results on WMT’16,19,20 En X test sets in BLEU. Remarks follow Table 3.

## 4.1.2 Training Details

The batch size is set to 4, with accumulation iterations set to 8 (i.e., real batch size is 32). We train 2 epochs with AdamW optimizer (Loshchilov and Hutter, 2018), with learning rate initialized to $1 e ^ { - 2 }$ and then linearly decrease to $1 e ^ { - 5 }$ during training.

## 4.2 Comparison with the State-of-the-art

## 4.2.1 Speech Translation

X English (En). Table 1 and 2 present the X En speech translation performance on FLEURS and CoVoST-2 datasets. We can observe from Table 1 that all the strong baselines like Whisper, AudioPaLM2 and SeamlessM4T-Large perform well on 15 X En directions, where SeamlessM4T-Large is the best (27.1 BLEU). With LLMs introduced for N-best integration, our GenTranslate achieves consistent improvements on various source languages X, where further analysis on language family is presented in §4.4.1. As a result, our GenTranslate shows 3.0 BLEU improvement over SeamlessM4T-Large, which verifies the effectiveness of LLMs for generative translation<sup>7</sup>.

Following the speech translation literature, we also investigate cascaded ASR+MT methods for evaluation. We can observe from Table 1 that, with the same SeamlessM4T-Large backbone, cascaded system outperforms end-to-end system by 4.8 BLEU score, which is consistent with previous findings (Xu et al., 2023a). Latest SeamlessM4T-Large-V2 further improves V1 model, and our Gen-Translate shows significant and consistent gains of performance over theses two backbones.

Table 2 presents the X En ST results on more language directions of CoVoST-2 dataset, where we introduce more latest baselines for comprehensive comparison. In end-to-end methods, SeamlessM4T-Large achieves a good 34.5 BLEU score though underperforms the state-of-the-art AudioPaLM2<sup>8</sup>. In comparison, our GenTranslate achieves a promising improvement over SeamlessM4T. Similar phenomenon can be observed in cascaded systems, where SeamlessM4T significantly outperforms the competitive baselines that combine state-of-theart ASR and MT models, and our GenTranslate moves one step forward with 1.8 BLEU improvement. Similar improvements can be observed on SeamlessM4T-Large-V2 backbone.

<table><tr><td rowspan="2">En→X</td><td colspan="4">FLEURS</td><td rowspan="2"></td><td colspan="4">CoVoST-2</td><td colspan="6"></td></tr><tr><td>Fr</td><td></td><td>It</td><td>Pt</td><td>Avg.</td><td>Fa Ja</td><td>Zh</td><td></td><td>Avg.</td><td>Ro Cs</td><td></td><td>Ja</td><td>Zh</td><td>Avg.</td></tr><tr><td>SeamlessM4T-Large (2023a)</td><td>24.6</td><td>44.6</td><td>25.4</td><td>41.9</td><td>34.1</td><td>18.8</td><td>24.0</td><td>35.1</td><td>26.0</td><td>32.7</td><td>26.0</td><td>17.2</td><td>17.0</td><td>27.2</td><td>24.0</td></tr><tr><td>GenTranslate with</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BigTranslate (2023b)</td><td>25.3</td><td>44.2</td><td>25.5</td><td>40.8</td><td>34.0</td><td>5.2</td><td>23.5</td><td>42.6</td><td>23.8</td><td>31.3</td><td>24.9</td><td>15.8</td><td>13.9</td><td>27.9</td><td>22.8</td></tr><tr><td>ALMA-13b (2023b)</td><td>24.9</td><td>43.5</td><td>25.1</td><td>40.6</td><td>33.5</td><td>19.2</td><td>29.3</td><td>43.9</td><td>30.8</td><td>31.1</td><td>25.5</td><td>17.7</td><td>17.3</td><td>26.8</td><td>23.7</td></tr><tr><td>LLaMA-2-13b (2023b)</td><td>26.8</td><td>45.0</td><td>26.6</td><td>43.1</td><td>35.4</td><td>21.8</td><td>30.5</td><td>43.3</td><td>31.9</td><td>33.5</td><td>27.2</td><td>19.4</td><td>21.4</td><td>30.7</td><td>26.4</td></tr></table>

Table 6: Effect of different multilingual LLMs on GenTranslate, in terms of the speech translation results on FLEURS and CoVoST-2 En X test sets, as well as the machine translation results on WMT En X test sets.
<table><tr><td>De→En</td><td>BLEU Score</td></tr><tr><td>End-to-end ST Methods</td><td></td></tr><tr><td>SeamlessM4T (ST) (Barrault et al., 2023a)</td><td>35.8</td></tr><tr><td>SeamlessM4T (ST) + GenTranslate</td><td>39.2</td></tr><tr><td>Cascaded ASR+MT Methods</td><td></td></tr><tr><td>SeamlessM4T (ASR+MT) (Barrault et al., 2023a)</td><td>39.7</td></tr><tr><td>SeamlessM4T (ASR+MT) + GenTranslate</td><td>41.6</td></tr><tr><td>ASR+GenTranslate Method</td><td></td></tr><tr><td>SeamlessM4T (ASR) + GenTranslate with</td><td></td></tr><tr><td>LLaMA-2-7b (Touvron et al., 2023b)</td><td>36.8</td></tr><tr><td>BigTranslate (Yang et al., 2023b)</td><td>38.2</td></tr><tr><td>ALMA-7b (Xu et al., 2023b)</td><td>40.6</td></tr></table>

Table 7: Performance of ASR+GenTranslate system on FLEURS De En ST test set. As shown in Fig. 4, it first uses ASR to produce German N-best hypotheses, and then leverages LLMs to generate the English translation from them. Different LLMs are investigated here.

English (En) X. For comprehensive evaluation, we also present En X ST results on three datasets in Table 3. SeamlessM4T (both Large and Large-V2) achieves excellent performance on En X ST tasks under both end-to-end and cascaded systems. In comparison, our proposed GenTranslate achieves significant performance improvements ( 3 BLEU score) in various language directions. Since En X translation tasks produce non-English N-best hypotheses for LLM integration, such performance gains indicates the excellent multilingual abilities of LLMs (i.e., LLaMA-2).

## 4.2.2 Machine Translation

X English (En). Table 4 presents the X En MT results on FLORES dataset. The baseline methods ALMA-13b and BigTranslate show limited performance. NLLB-3.3b achieves an improved performance of 37.5 BLEU, which is comparable to

![](images/8032305d87e679033185dd5d6346d1a777809dbc721d2187e8c314905e45d6fa.jpg)  
Figure 4: Illustration of the “ASR+GenTranslate” system for ST task as introduced in Table 7 and §4.3.2. This system engages LLMs into the translation process by combining it with the N-best integration process.

SeamlessM4T-Large. Based on that, our GenTranslate achieves the state-of-the-art with consistent gains on all language directions except Ja En.

English (En) X. Table 5 presents the En X MT results on WMT test sets. Similar to previous results, we observe much higher BLEU scores of NLLB-3.3b than ALMA-13b and BigTranslate. SeamlessM4T-Large surpasses NLLB-3.3b by large-scale multitask training. The proposed GenTranslate achieves the state-of-the-arts on all language directions with a gain of 2.4 BLEU score. Please note that SeamlessM4T-Large-V2 underperforms V1 on selected MT datasets, but our Gen-Translate achieves consistent gains on both of them.

In summary, we observe consistent improvements of GenTranslate over various baselines (i.e., SeamlessM4T, Whisper, etc.), various tasks (i.e., ST and MT), various test data (i.e., FLEURS, WMT, etc.), and various language directions (i.e., X En and En X). Therefore, the effectiveness and generality of our approach are well verified.

## 4.3 Ablation Study

## 4.3.1 Effect of Different LLMs

According to Table 3 and 5, LLaMA-2 has shown excellent multilingual ability. To further investigate the role of this ability in GenTranslate, we select two latest multilingual LLMs for comparison, i.e., BigTranslate and ALMA-13b. Table 6 shows that both of them perform worse than LLaMA-2-13b for ST and MT tasks. One explanation is, Big-Translate and ALMA-13b are finetuned on MT task that requires cross-lingual ability, while the En X GenTranslate mainly requires strong monolingual ability of language X, such mismatch may explain why MT finetuning fails to enhance GenTranslate.

<table><tr><td>X→En</td><td>Ar</td><td>Cy</td><td>De</td><td>El</td><td>Es</td><td>Fa</td><td>Fr</td><td>Hi</td><td>It</td><td>Ja</td><td>Pt</td><td>Ta</td><td>Uk</td><td>Vi</td><td>Zh</td><td>Avg.</td></tr><tr><td>SeamlessM4T (ASR+MT)</td><td>38.9</td><td>37.0</td><td>39.7</td><td>29.0</td><td>27.7</td><td>34.1</td><td>37.7</td><td>33.9</td><td>28.9</td><td>21.7</td><td>42.3</td><td>23.7</td><td>34.0</td><td>24.9</td><td>24.4</td><td>31.9</td></tr><tr><td>GenTranslate with</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LLaMA-Adapter</td><td>39.9</td><td>39.4</td><td>41.6</td><td>32.8</td><td>31.2</td><td>35.9</td><td>40.6</td><td>34.9</td><td>32.1</td><td>22.8</td><td>45.0</td><td>24.1</td><td>36.9</td><td>27.4</td><td>25.7</td><td>34.0</td></tr><tr><td>LLaMA-LoRA</td><td>40.2</td><td>39.3</td><td>41.8</td><td>32.8</td><td>31.6</td><td>36.0</td><td>40.6</td><td>35.2</td><td>32.4</td><td>22.5</td><td>45.1</td><td>24.1</td><td>36.7</td><td>27.1</td><td>26.0</td><td>34.1</td></tr></table>

Table 8: Comparison between LLaMA-Adapter and LLaMA-LoRA for efficient LLM finetuning in our GenTranslate, in terms of the speech translation results on FLEURS X En test sets.

<table><tr><td rowspan="2">X→En</td><td colspan="10"></td><td colspan="7">non-Indo-European</td></tr><tr><td>Fa</td><td>Hi</td><td>It</td><td>Es</td><td>Fr</td><td>Pt</td><td>Cy</td><td>De</td><td>El</td><td>Uk</td><td>Avg.</td><td>Ar</td><td>Vi</td><td>Ja</td><td>Ta</td><td>Zh</td><td>Avg.</td></tr><tr><td>SeamlessM4T (ASR+MT)</td><td>34.1</td><td>33.9</td><td>28.9</td><td>27.7</td><td>37.7</td><td>42.3</td><td>37.0</td><td>39.7</td><td>29.0</td><td>34.0</td><td>34.4</td><td>38.9</td><td>24.9</td><td>21.7</td><td>23.7</td><td>24.4</td><td>26.7</td></tr><tr><td>GenTranslate (ours)</td><td>35.9</td><td>34.9</td><td>32.1</td><td>31.2</td><td>40.6</td><td>45.0</td><td>39.4</td><td>41.6</td><td>32.8</td><td>36.9</td><td>37.0</td><td>39.9</td><td>27.4</td><td>22.8</td><td>24.1</td><td>25.7</td><td>28.0</td></tr><tr><td>∆ BLEU</td><td>1.8</td><td>1.0</td><td>3.2</td><td>3.5</td><td>2.9</td><td>2.7</td><td>2.4</td><td>1.9</td><td>3.8</td><td>2.9</td><td>2.6</td><td>1.0</td><td>2.5</td><td>1.1</td><td>0.4</td><td>1.4</td><td>1.3</td></tr></table>

Table 9: Effect of language family on our proposed GenTranslate. We report speech translation results on FLEURS X En test sets in this study. For simplicity, we split all the languages into two families, i.e., Indo-European (same as English) and non-Indo-European, and more detailed information are presented in Table 14.

<table><tr><td>X→En</td><td>Ar</td><td>De</td><td>Es</td><td>Fr</td><td>Pt</td><td>Zh</td><td>Avg.</td></tr><tr><td>SeamlessM4T-Large</td><td>32.8</td><td>35.8</td><td>25.0</td><td>33.1</td><td>38.9</td><td>19.8</td><td>30.9</td></tr><tr><td colspan="8">GenTranslate with</td></tr><tr><td>1 3</td><td>31.3</td><td>35.4</td><td>26.9</td><td>35.2</td><td>41.5</td><td>19.3</td><td>31.6</td></tr><tr><td></td><td>34.2</td><td>38.9</td><td>29.5</td><td>36.4</td><td>42.8</td><td>21.3</td><td>33.9</td></tr><tr><td>5</td><td>34.6</td><td>39.2</td><td>29.8</td><td>37.0</td><td>43.0</td><td>21.7</td><td>34.2</td></tr><tr><td>8</td><td>34.8</td><td>39.9</td><td>29.4</td><td>36.9</td><td>43.0</td><td>21.5</td><td>34.3</td></tr><tr><td>10</td><td>35.3</td><td>39.8</td><td>29.4</td><td>36.6</td><td>43.2</td><td>21.6</td><td>34.3</td></tr><tr><td>15</td><td>34.9</td><td>39.5</td><td>29.6</td><td>36.4</td><td>42.8</td><td>21.6</td><td>34.1</td></tr></table>

Table 10: Effect of N-best list size on GenTranslate (default N=5), in terms of ST results on FLEURS X En.

## 4.3.2 Role of LLMs in GenTranslate

To further investigate the role of LLMs in our Gen-Translate, we build an ASR+GenTranslate system for ST task as shown in Fig. 4. Take De En as an example, we first send the German speech input into ASR to produce N-best transcriptions, which are then fed by LLMs to generate English translation. In other words, LLMs are assigned N-best integration and translation tasks at the same time. As shown in Table 7, among the three evaluated LLMs, ALMA-7b achieves the best performance thanks to its MT finetuning during development, but it still underperforms the best cascaded method (40.6 vs. 41.6). We can conclude from such observations that 1) LLaMA-2 provides reasonable translation ability and it can be further improved via MT task finetuning (i.e., ALMA). 2) In this study, LLM underperforms SeamlessM4T in translation task, but it shows remarkable ability in N-best integration. Therefore, future work may focus on how to better engage LLMs into the translation part.

## 4.3.3 Effect of N-best List Size

GenTranslate relies on powerful LLMs and informative N-best hypotheses to generate higherquality translation output. Therefore, the amount of information in N-best hypotheses could be a key factor of GenTranslate’s performance. We can observe from Table 10 that with the increase of N, the performance of GenTranslate first improves and then drops, where the best choice ranges from 5 to 10. We believe that small N results in insufficient information for generation of ground-truth translation, while too large N leads to information redundancy and thus increases the miscorrection and hallucination. In this work, we set N to 5 for the best trade-off between efficiency and quality.

## 4.3.4 LLaMA-Adapter vs. LLaMA-LoRA

Apart from LLaMA-Adapter, low-rank adaptation (LoRA) (Hu et al., 2021; Yu et al., 2023) is another popular efficient LLM finetuning strategy. Table 8 compares the performance between LLaMA-Adapter and LLaMA-LoRA for proposed Gen-Translate, in terms of the BLEU results of ST task on FLEURS X En test sets. We can observe similar BLEU performance of these two strategies on GenTranslate (34.0 vs. 34.1), indicating that the efficient LLM finetuning strategy is not a key factor in GenTranslate paradigm.

<table><tr><td>Method</td><td>Utterance</td><td>BLEU Score</td></tr><tr><td rowspan="5">N-best Candidates</td><td>TV reports show that white smoke is escaping from the plant.</td><td>28.6</td></tr><tr><td>TV reports show that white smoke is escaping from the facility</td><td>12.2</td></tr><tr><td>Television reports show that white smoke is escaping from the plant.</td><td>34.2</td></tr><tr><td>Television reports show that white smoke is escaping from the facility.</td><td>19.2</td></tr><tr><td>TV reports show that white smoke escapes from the plant.</td><td>31.7</td></tr><tr><td>GenTranslate (ours)</td><td>Television reports show white smoke coming out of the plant.</td><td>58.8</td></tr><tr><td>Ground-truth Translation</td><td>Television reports show white smoke coming from the plant.</td><td></td></tr></table>

Table 11: Case study of GenTranslate. The test sample is selected from the FLEURS De En ST test set.

## 4.4 Analysis

## 4.4.1 Effect of Language Family

Table 9 analyzes the effect of language family using the X En ST results. The source language X is grouped into two categories depending on whether it belongs to Indo-European family (English is also Indo-European language). First, we observe better results of SeamlessM4T when X belongs to Indo-European family, indicating that translation within same family is easier than across different families. Then, we also observe larger BLEU improvement of GenTranslate over baseline when X is Indo-European language (2.6 vs. 1.3). The reason could be, within-family translation produces N-best hypotheses with higher quality and richer information, which is beneficial to GenTranslate.

## 4.4.2 Case Study

Table 11 shows a case study where GenTranslate outperforms the 1-best hypothesis by a large margin. We may speculate two key points about its working mechanism, where it first extract the word “Television” from 3<sup>rd</sup>/4<sup>th</sup> hypotheses to replace “TV” and then reason out the word “coming” that does not exist in N-best list. Therefore, our paradigm may not only integrate the N-best sentences for better result, but also improve the translation quality by itself. Another non-English case study is in Appendix C.1.

## 4.4.3 Visualizations of GenTranslate Output

Fig. 5 visualizes the n-gram tokens in GenTranslate output, which contains sufficient semantic information to match the ground-truth translation. In comparison, the 1-best hypothesis lacks such information to produce high-quality translation output, which highlights the contribution of N-best hypotheses in GenTranslate paradigm (see Fig. 2).

![](images/c28a1e6e8d80f93b36be843a2f5d021d091bcdaf65d86b064ca3facaab05ca0e.jpg)  
Figure 5: t-SNE visualization of n-grams in 1-best hypothesis (green), ground-truth translation (orange) and GenTranslate output (purple). It’s an extension of Fig. 2.

## 5 Conclusion

In this paper, we propose a generative paradigm for translation tasks, namely GenTranslate, which leverages LLMs to integrate the diverse candidates in the decoded N-best list and generate a higherquality translation result. Furthermore, we release a HypoTranslate dataset to support LLM finetuning, which contains over 592K hypotheses-translation pairs in 11 languages. Experimental evidence on various speech and machine translation benchmarks shows that our GenTranslate significantly outperforms the state-of-the-art model.

## Limitations

There are two limitations existed in this work. First, the contribution of LLMs in our GenTranslate paradigm focuses on N-best hypotheses integration, while the translation part is actually done by SeamlessM4T model. Experiment results in Table 7 also indicate that LLMs are good at Nbest hypotheses integration and SeamlessM4T is good at translation. Therefore, our future work could focus on how to better engage LLMs into the translation part to further improve the translation quality. Another limitation is about the latest second version of SeamlessM4T released by Meta, which indicates a stronger baseline for GenTranslate. In fact, our experiments had already been done on SeamlessM4T-Large before Meta released the latest SeamlessM4T-Large-V2 on November 30th, 2023. For comprehensive evaluation, we also rerun our main experiments on this latest V2 backbone, and our GenTranslate has shown similar effectiveness on it (highlighted in gray in Table 1 to 5). For brevity, we prefer to leave the ablation study and analyses on SeamlessM4T-Large backbone only, as our GenTranslate paradigm has shown similar effectiveness and patterns on V1 and V2 backbones.

## Ethics Statement

This work does not pose any ethical issues. All the data used in this paper are publicly available and are used under following licenses: Creative Commons BY 4.0 License, Creative Commons CC0 License, Creative Commons BY-NC-ND 4.0 License, and Creative Commons BY-SA 4.0 License.

## Acknowledgements

This research is supported by the National Research Foundation, Singapore under its AI Singapore Programme (AISG Award No: AISG2-GC-2022-005). Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the views of National Research Foundation, Singapore. The computational work for this article was partially performed on resources of the National Supercomputing Centre, Singapore (https://www.nscc.sg).

## References

Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. 2023. Palm 2 technical report. arXiv preprint arXiv:2305.10403.

Rosana Ardila, Megan Branson, Kelly Davis, Michael Henretty, Michael Kohler, Josh Meyer, Reuben Morais, Lindsay Saunders, Francis M Tyers, and Gregor Weber. 2019. Common voice: A massivelymultilingual speech corpus. arXiv preprint arXiv:1912.06670.

Arun Babu, Changhan Wang, Andros Tjandra, Kushal Lakhotia, Qiantong Xu, Naman Goyal, Kritika Singh, Patrick von Platen, Yatharth Saraf, Juan Pino, et al. 2021. Xls-r: Self-supervised cross-lingual speech representation learning at scale. arXiv preprint arXiv:2111.09296.

Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. 2020. wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in neural information processing systems, 33:12449–12460.

Marta Bañón, Pinzhen Chen, Barry Haddow, Kenneth Heafield, Hieu Hoang, Miquel Esplà-Gomis, Mikel Forcada, Amir Kamran, Faheem Kirefu, Philipp Koehn, et al. 2020. Paracrawl: Web-scale acquisition of parallel corpora. Association for Computational Linguistics (ACL).

Loıc Barrault, Ondrej Bojar, Marta R Costa-Jussa, Christian Federmann, Mark Fishel, Yvette Graham, Barry Haddow, Matthias Huck, Philipp Koehn, Shervin Malmasi, et al. 2019. Findings of the 2019 conference on machine translation. Proceedings ofWMT.

Loïc Barrault, Yu-An Chung, Mariano Cora Meglioli, David Dale, Ning Dong, Paul-Ambroise Duquenne, Hady Elsahar, Hongyu Gong, Kevin Heffernan, John Hoffman, et al. 2023a. Seamlessm4t-massively multilingual & multimodal machine translation. arXiv preprint arXiv:2308.11596.

Loïc Barrault, Yu-An Chung, Mariano Coria Meglioli, David Dale, Ning Dong, Mark Duppenthaler, Paul-Ambroise Duquenne, Brian Ellis, Hady Elsahar, Justin Haaheim, John Hoffman, et al. 2023b. Seamless: Multilingual expressive and streaming speech translation. arXiv 2023.

Ondrej Bojar, Rajen Chatterjee, Christian Federmann, Yvette Graham, et al. 2016. Findings of the 2016 conference on machine translation (wmt16). In First conference on machine translation, pages 131–198. Association for Computational Linguistics.

Zalán Borsos, Raphaël Marinier, Damien Vincent, Eugene Kharitonov, Olivier Pietquin, Matt Sharifi, Dominik Roblek, Olivier Teboul, David Grangier, Marco Tagliasacchi, et al. 2023. Audiolm: a language modeling approach to audio generation. IEEE/ACM Transactions on Audio, Speech, and Language Processing.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Chen Chen, Yuchen Hu, Chao-Han Huck Yang, Sabato Marco Siniscalchi, Pin-Yu Chen, and Ensiong Chng. 2023. Hyporadise: An open baseline for generative speech recognition with large language models. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Alexis Conneau, Min Ma, et al. 2023. Fleurs: Few-shot learning evaluation of universal representations of speech. In 2022 IEEE Spoken Language Technology Workshop (SLT), pages 798–805. IEEE.

Marta R Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, et al. 2022. No language left behind: Scaling human-centered machine translation. arXiv preprint arXiv:2207.04672.

Mattia A Di Gangi, Roldano Cattoni, Luisa Bentivogli, Matteo Negri, and Marco Turchi. 2019. Must-c: a multilingual speech translation corpus. In Proc. NAACL, pages 2012–2017. Association for Computational Linguistics.

Naman Goyal, Cynthia Gao, et al. 2022. The flores-101 evaluation benchmark for low-resource and multilingual machine translation. Transactions ofthe Association for Computational Linguistics, 10:522–538.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685.

Ke Hu, Tara N Sainath, et al. 2023a. Improving multilingual and code-switching asr using large language model generated text. In 2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pages 1–7. IEEE.

Zhiqiang Hu, Yihuai Lan, et al. 2023b. Llmadapters: An adapter family for parameter-efficient fine-tuning of large language models. arXiv preprint arXiv:2304.01933.

Andrew K Lampinen, Ishita Dasgupta, Stephanie CY Chan, Kory Matthewson, Michael Henry Tessler, Antonia Creswell, James L McClelland, Jane X Wang, and Felix Hill. 2022. Can language models learn from explanations in context? arXiv preprint arXiv:2204.02329.

Chenyang Le, Yao Qian, Long Zhou, Shujie Liu, Michael Zeng, and Xuedong Huang. 2023. Comsl: A composite speech-language model for end-toend speech-to-text translation. arXiv preprint arXiv:2305.14838.

Yichong Leng, Xu Tan, Linchen Zhu, Jin Xu, Renqian Luo, Linquan Liu, Tao Qin, Xiangyang Li, Edward Lin, and Tie-Yan Liu. 2021. Fastcorrect: Fast error correction with edit alignment for automatic speech recognition. Advances in Neural Information Processing Systems, 34:21708–21719.

Xiang Li, John Thickstun, Ishaan Gulrajani, Percy S Liang, and Tatsunori B Hashimoto. 2022. Diffusionlm improves controllable text generation. Advances in Neural Information Processing Systems, 35:4328– 4343.

Dan Liu, Mengge Du, Xiaoxi Li, Yuchen Hu, and Lirong Dai. 2021. The ustc-nelslip systems for simultaneous speech translation task at iwslt 2021. arXiv preprint arXiv:2107.00279.

Barrault Loïc, Biesialska Magdalena, Bojar Ondˇrej, Federmann Christian, Graham Yvette, Grundkiewicz Roman, Haddow Barry, Huck Matthias, et al. 2020. Findings of the 2020 conference on machine translation (wmt20). In Proceedings ofthe Fifth Conference on Machine Translation, pages 1–55. Association for Computational Linguistics,.

Ilya Loshchilov and Frank Hutter. 2018. Decoupled weight decay regularization. In International Conference on Learning Representations.

Chenyang Lyu, Jitao Xu, and Longyue Wang. 2023. New trends in machine translation using large language models: Case examples with chatgpt. arXiv preprint arXiv:2305.01181.

Rao Ma, Mark JF Gales, Kate Knill, and Mengjie Qian. 2023a. N-best t5: Robust asr error correction using multiple input hypotheses and constrained decoding space. arXiv preprint arXiv:2303.00456.

Rao Ma, Mengjie Qian, Potsawee Manakul, Mark Gales, and Kate Knill. 2023b. Can generative large language models perform asr error correction? arXiv preprint arXiv:2307.04172.

Makoto Morishita, Jun Suzuki, and Masaaki Nagata. 2020. JParaCrawl: A large scale web-based English-Japanese parallel corpus. In Proceedings of The 12th Language Resources and Evaluation Conference, pages 3603–3609, Marseille, France. European Language Resources Association.

OpenAI. 2022. Introducing chatgpt. OpenAI Blog.

OpenAI. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings ofthe 40th annual meeting of the Association for Computational Linguistics, pages 311–318.

Maja Popovic. 2017. chrf++: words helping character ´ n-grams. In Proceedings of the second conference on machine translation, pages 612–618.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2023. Robust speech recognition via large-scale weak supervision. In International Conference on Machine Learning, pages 28492–28518. PMLR.

Srijith Radhakrishnan, Chao-Han Yang, Sumeer Khan, et al. 2023. Whispering llama: A cross-modal generative error correction framework for speech recognition. In Proceedings of the 2023 Conference on

Empirical Methods in Natural Language Processing, pages 10007–10016.

Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084.

Paul K Rubenstein, Chulayuth Asawaroengchai, Duc Dung Nguyen, et al. 2023. Audiopalm: A large language model that can speak and listen. arXiv preprint arXiv:2306.12925.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, et al. 2023a. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Hugo Touvron, Louis Martin, Kevin Stone, et al. 2023b. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Emiru Tsunoo, Yosuke Kashiwagi, and Shinji Watanabe. 2021. Streaming transformer asr with blockwise synchronous beam search. In 2021 IEEE Spoken Language Technology Workshop (SLT), pages 22–29. IEEE.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.

Changhan Wang, Anne Wu, and Juan Pino. 2020. Covost 2 and massively multilingual speech-to-text translation. arXiv preprint arXiv:2007.10310.

Siyin Wang, Chao-Han Huck Yang, Ji Wu, and Chao Zhang. 2023. Can whisper perform speech-based incontext learning. arXiv preprint arXiv:2309.07081.

Thomas Wang, Adam Roberts, et al. 2022. What language model architecture and pretraining objective works best for zero-shot generalization? In International Conference on Machine Learning, pages 22964–22984. PMLR.

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. 2022a. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022b. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837.

Yanling Xiao, Lemao Liu, et al. 2022. Bitiimt: A bilingual text-infilling method for interactive machine translation. In Proceedings ofthe 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1958–1969.

Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. 2021. An explanation of in-context learning as implicit bayesian inference. arXiv preprint arXiv:2111.02080.

Chen Xu, Rong Ye, Qianqian Dong, Chengqi Zhao, Tom Ko, Mingxuan Wang, Tong Xiao, and Jingbo Zhu. 2023a. Recent advances in direct speech-to-text translation. arXiv preprint arXiv:2306.11646.

Haoran Xu, Young Jin Kim, Amr Sharaf, and Hany Hassan Awadalla. 2023b. A paradigm shift in machine translation: Boosting translation performance of large language models. arXiv preprint arXiv:2309.11674.

Chao-Han Huck Yang, Yile Gu, Yi-Chieh Liu, Shalini Ghosh, Ivan Bulyko, and Andreas Stolcke. 2023a. Generative speech recognition error correction with large language models and task-activating prompting. In 2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pages 1–8. IEEE.

Chao-Han Huck Yang, Linda Liu, et al. 2021a. Multitask language modeling for improving speech recognition of rare words. In 2021 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pages 1087–1093. IEEE.

Chao-Han Huck Yang, Yun-Yun Tsai, and Pin-Yu Chen. 2021b. Voice2series: Reprogramming acoustic models for time series classification. In International Conference on Machine Learning, pages 11808– 11819. PMLR.

Wen Yang, Chong Li, Jiajun Zhang, and Chengqing Zong. 2023b. Bigtrans: Augmenting large language models with multilingual translation capability over 100 languages. arXiv preprint arXiv:2305.18098.

Yu Yu, Chao-Han Huck Yang, Jari Kolehmainen, et al. 2023. Low-rank adaptation of large language model rescoring for parameter-efficient speech recognition. In 2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pages 1–8. IEEE.

Biao Zhang, Barry Haddow, and Alexandra Birch. 2023a. Prompting large language model for machine translation: A case study. arXiv preprint arXiv:2301.07069.

Renrui Zhang, Jiaming Han, et al. 2023b. Llamaadapter: Efficient fine-tuning of language models with zero-init attention. arXiv preprint arXiv:2303.16199.

## A HypoTranslate Dataset Details

In this section, we introduce the details of our proposed HypoTranslate dataset. We first introduce the speech and machine translation corpora that we utilize to build HypoTranslate in §A.1 and §A.2. Then, we present the dataset statistics in §A.3.

## A.1 Speech Translation Corpus Selection

For speech translation task, we select three popular and public datasets that cover multiple languages: FLEURS<sup>9</sup> (Conneau et al., 2023): Few-shot Learning Evaluation of Universal Representations of Speech (FLEURS) benchmark provides an n-way parallel speech dataset in 102 languages built on top of the machine translation FLORES-101 benchmark (Goyal et al., 2022), with approximately 12 hours of speech supervision per language. In this work, we select 15 X En and 6 En X language directions of speech translation data for evaluation. CoVoST-2<sup>10</sup> (Wang et al., 2020): CoVoST-2 is a popular multilingual speech translation corpus based on Common Voice (Ardila et al., 2019) that consists of 2,880 hours speech data recorded from 78K speakers. In this work, we select 15 X En and 3 En X language directions for evaluation. Specifically, for En X language directions, we randomly select 1,000 testing samples from the original test split for higher evaluation efficiency.

MuST-C<sup>11</sup> (Di Gangi et al., 2019): MuST-C is a multilingual speech translation corpus whose size and quality facilitate the training of end-to-end systems for spoken language translation from English into 15 languages. In this work, we select 3 En X language directions for evaluation.

## A.2 Machine Translation Corpus Selection

For machine translation task, we select two popular and public datasets that cover multiple languages: FLORES<sup>12</sup> (Costa-jussà et al., 2022): FLORES consists of 3001 sentences sampled from Englishlanguage Wikimedia projects for 204 total languages. Approximately one third of sentences are collected from each of these sources: Wikinews, Wikijunior, and Wikivoyage. The content is professionally translated into 200+ languages to create

FLORES dataset. In this work, we select 10 X En language directions for evaluation.

WMT: The Conference on Machine Translation (WMT) is a popular evaluation benchmark for MT task. In this work, we select the newstest data of Ro En language direction from WMT’16<sup>13</sup> (Bojar et al., 2016), Cs En and It En directions from WMT’19<sup>14</sup> (Barrault et al., 2019), Ja En and Zh En directions from WMT’20<sup>15</sup> (Loïc et al., 2020) for evaluation, and corresponding newdev data is used for validation. The training data is obtained from ParaCrawl-V9<sup>16</sup> (Bañón et al., 2020) and JParaCrawl<sup>17</sup> (Morishita et al., 2020) datasets.

## A.3 Statistics

After performing beam search decoding on the selected speech and machine translation corpora introduced above, we collect over 592K pairs of N-best hypotheses and ground-truth translation to build the HypoTranslate dataset. The statistics are illustrated in Table 15 and 17, which present the number of hypotheses-translation pairs and the average utterance length. We plan to release the HypoTranslate dataset to public upon publication and open the development venue for more data.

## B Experimental Setup Details

## B.1 Model Setups

We select two latest foundation LLMs for evaluation, including LLaMA-2-7b (Touvron et al., 2023b) and LLaMA-2-13b (Touvron et al., 2023b). In addition, in order to evaluate the multilingual ability of LLMs for GenTranslate with non-Englishtarget directions, we also select two latest finetuned LLMs on MT task, including BigTranslate (Yang et al., 2023b) and ALMA-13b (Xu et al., 2023b). Table 12 compares their main configurations. For efficient LLM finetuning, we follow the default settings of LLaMA-Adapter<sup>18</sup> (Zhang et al., 2023b).

<table><tr><td rowspan=1 colspan=1>LLM</td><td rowspan=1 colspan=1>LLaMA-2-7b</td><td rowspan=1 colspan=1>LLaMA-2-13b</td><td rowspan=1 colspan=1>BigTranslate</td><td rowspan=1 colspan=1>ALMA-13b</td></tr><tr><td rowspan=5 colspan=1>Number of Transformer Layers HNumber of Attention Heads $N _ { \mathrm { h e a d } }$ Embedding Size DBlock Size BVocabulary Size V</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1>40</td></tr><tr><td rowspan=2 colspan=1>324,096</td><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1>40</td><td rowspan=2 colspan=1>405,120</td></tr><tr><td rowspan=1 colspan=1>5,120</td><td rowspan=1 colspan=1>5,120</td></tr><tr><td rowspan=1 colspan=1>4,096</td><td rowspan=1 colspan=1>4,096</td><td rowspan=1 colspan=1>4,096</td><td rowspan=1 colspan=1>4,096</td></tr><tr><td rowspan=1 colspan=1>32,000</td><td rowspan=1 colspan=1>32,000</td><td rowspan=1 colspan=1>53,613</td><td rowspan=1 colspan=1>32,000</td></tr></table>

Table 12: Comparison between main configurations of different popular LLMs.

## B.2 Inference Setups

In the response generation during inference stage, we set a temperature of 0.2 and top-1 sampling, i.e., greedy search. We have observed over-confidence phenomenon in our experiments (i.e., output probability distribution for decision is close to one-hot), which results in similar performance with different k for top-k sampling. Therefore, we select top-1 sampling for higher decoding efficiency.

## B.3 Translation Baselines

To comprehensively evaluate our GenTranslate model, we selected some of the latest and most advanced baselines in speech and machine translation for comparison. We will introduce these in the following subsections.

## B.3.1 Speech Translation

XLS-R<sup>19</sup> (Babu et al., 2021): XLS-R is a largescale model for cross-lingual speech representation learning based on Wav2vec 2.0 (Baevski et al., 2020). They train models with up to 2B parameters on 500K hours of publicly available speech audio in 128 languages, which achieves superior performance on a wide range of multilingual speech processing tasks, including speech translation, speech recognition and language identification.

Whisper<sup>20</sup> (Radford et al., 2023): Whisper is a large-scale automatic speech recognition (ASR) system trained on 680K hours of multilingual and multitask supervised data collected from the web, which shows excellent robustness to accents, background noise and technical language. Moreover, it enables transcription in multiple languages, as well as translation from those languages into English.

AudioPaLM2 (Rubenstein et al., 2023): AudioPaLM2 fuses text-based and speech-based language models, PaLM-2 (Anil et al., 2023) and AudioLM (Borsos et al., 2023), into a unified multimodal architecture that can process and generate text and speech with applications including speech recognition and speech-to-speech translation. AudioPaLM2 inherits the capability to preserve paralinguistic information such as speaker identity and intonation from AudioLM and the linguistic knowledge present only in text large language models such as PaLM-2. The resulting model significantly outperforms existing systems for speech translation and has the ability to perform zero-shot speech-totext translation for many unseen languages.

ComSL<sup>21</sup> (Le et al., 2023): ComSL is a speechlanguage model built atop a composite architecture of public pre-trained speech-only and languageonly models and optimized data-efficiently for spoken language tasks. Particularly, they propose to incorporate cross-modality learning into transfer learning and conduct them simultaneously for downstream tasks in a multi-task learning manner, which has demonstrated effectiveness in end-to-end speech-to-text translation tasks.

## B.3.2 Machine Translation

NLLB<sup>22</sup> (Costa-jussà et al., 2022): No Language Left Behind (NLLB) is a first-of-its-kind, AI breakthrough project that open-sources models capable of delivering evaluated, high-quality translations directly between 200 languages.

BigTranslate<sup>5</sup> (Yang et al., 2023b): BigTranslate adapts LLaMA-13b (Touvron et al., 2023a) that covers only 20 languages and enhances it with multilingual translation capability on up to 102 languages by instruction-following finetuning, which achieves comparable MT performance to Chat-GPT (OpenAI, 2022) and Google Translate.

ALMA<sup>6</sup> (Xu et al., 2023b): ALMA proposes a novel finetuning approach for LLMs that is specifically designed for MT task, eliminating the need for the abundant parallel data that traditional translation models usually depend on, which includes two stages: initial finetuning on monolingual data followed by subsequent finetuning on a small set of high-quality parallel data. Built based on LLaMA-2, it has achieved significant improvement over prior works across multiple translation directions.

<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Utterance</td><td rowspan=1 colspan=1>BLEU Score</td></tr><tr><td rowspan=1 colspan=1>N-best Candidates</td><td rowspan=1 colspan=1>地球河流流入海洋的20%的水来自亚马逊地球河流流入海洋的20%的水源来自亚马逊地球河流流入海洋的全部20%的水来自亚马逊地球河流流入海洋的20%的水来自亚马逊地球河流流入海洋的全部20%的水来自亚马逊</td><td rowspan=1 colspan=1>15.015.012.315.012.3</td></tr><tr><td rowspan=1 colspan=1>GenTranslate (ours)</td><td rowspan=1 colspan=1>地球上的河流汇入大洋的20%的水来自亚马逊河。</td><td rowspan=1 colspan=1>18.7</td></tr><tr><td rowspan=1 colspan=1>Ground-truth Translation</td><td rowspan=1 colspan=1>亚马逊河占全世界所有河流的入海流量的20%。</td><td rowspan=1 colspan=1>1</td></tr></table>

<sup>Table</sup> <sup>13:</sup> <sup>Supplementary</sup> <sup>case</sup> <sup>study.</sup> <sup>The</sup> <sup>test</sup> <sup>sample</sup> <sup>is</sup> <sup>selected</sup> <sup>from</sup> <sup>the</sup> <sup>FLEURS</sup> <sup>En</sup>→<sup>Zh</sup> <sup>ST</sup> <sup>test</sup> <sup>set.</sup>

<table><tr><td>Language</td><td>Family</td><td>Sub-grouping</td></tr><tr><td>Persian (Fa)</td><td>Indo-European</td><td>Indo-Iranian</td></tr><tr><td>Hindi (Hi)</td><td>Indo-European</td><td>Indo-Iranian</td></tr><tr><td>Italian (It)</td><td>Indo-European</td><td>Indo-Iranian</td></tr><tr><td>Spanish (Es)</td><td>Indo-European</td><td>Italic</td></tr><tr><td>French (Fr)</td><td>Indo-European</td><td>Italic</td></tr><tr><td>Portuguese (Pt)</td><td>Indo-European</td><td>Italic</td></tr><tr><td>Welsh (Cy)</td><td>Indo-European</td><td>Celtic</td></tr><tr><td>English (En)</td><td>Indo-European</td><td>Germantic</td></tr><tr><td>German (De)</td><td>Indo-European</td><td>Germantic</td></tr><tr><td>Greek (El)</td><td>Indo-European</td><td>Greek</td></tr><tr><td>Ukranian (Uk)</td><td></td><td></td></tr><tr><td>Arabic (Ar)</td><td>Indo-European</td><td>Balto-Slavic</td></tr><tr><td></td><td>Afro-Asiatic</td><td>Semitic</td></tr><tr><td>Vietnamese (Vi)</td><td>Austro-Asiatic</td><td>Mon-Khmer</td></tr><tr><td>Japanese (Ja)</td><td>Japonic</td><td></td></tr><tr><td>Tamil (Ta)</td><td>Dravidian</td><td>Dravidian</td></tr><tr><td>Chinese (Zh)</td><td>Sino-Tibetan</td><td>Chinese</td></tr></table>

Table 14: Detailed language family and sub-grouping information (Babu et al., 2021) of FLEURS datasets.

## C Supplementary Experiment Results

## C.1 Supplementary Case Study

Table 13 supplies a case study from FLEURS En Zh ST test set. We can observe that the N-best candidate are semantically similar to each other and only varies in sentence structure. In our Gen-Translate paradigm, LLMs integrates the different patterns of N-best hypotheses to generate a new translation result with 3.7 BLEU improvement over 1-best hypothesis. Such observation verifies the effectiveness of LLMs in our GenTranslate paradigm to generate better translation output.

![](images/983babfe628dea94f61965cbd18fac265a539567d3ded258a405bb291792c795.jpg)  
Figure 6: t-SNE visualization of n-gram tokens in ASR 1-best hypothesis (green), 2 to N-best hypotheses (blue), and the ground-truth transcription (orange). Different from the ST hypotheses in Fig. 2, ASR 1-best hypothesis aligns well with the ground-truth transcription, where the role of 2 N-best hypotheses is to provide diverse candidate tokens for correcting errors.

## C.2 BLEU vs. chrF++

We report translation performance in terms of the BLEU score (Papineni et al., 2002) in most experiments of this work. For more comprehensive evaluation, Table 16 presents both BLEU and chrF++ scores (Popovic´, 2017; Barrault et al., 2023a) on FLEURS X En test sets, where we can observe consistent improvements of BLEU and chrF++ scores (2.1 ∆ BLEU and 0.9 ∆ chrF++) in Gen-Translate. It indicates that both metrics are applicable for the evaluation of translation tasks.

<table><tr><td rowspan="2">Data Source</td><td rowspan="2">Source / Target Language X</td><td colspan="2">Train</td><td colspan="2">Dev.</td><td colspan="2">Test</td></tr><tr><td># Pairs</td><td>Length</td><td># Pairs</td><td>Length</td><td># Pairs</td><td>Length</td></tr><tr><td rowspan="14">FLEURS (Conneau et al., 2023) (X→En)</td><td>Arabic (Ar)</td><td>2,062</td><td>20.4</td><td>295</td><td>19.8</td><td>428</td><td>21.4</td></tr><tr><td>Welsh (Cy)</td><td>3,349</td><td>21.1</td><td>447</td><td>20.6</td><td>1,021</td><td>22.1</td></tr><tr><td>German (De)</td><td>2,926</td><td>20.7</td><td>363</td><td>20.1</td><td>862</td><td>21.9</td></tr><tr><td>Greek (El)</td><td>3,148</td><td>20.9</td><td>271</td><td>20.5</td><td>650</td><td>21.7</td></tr><tr><td>Spanish (Es)</td><td>2,732</td><td>20.8</td><td>408</td><td>20.5</td><td>908</td><td>21.8</td></tr><tr><td>Persian (Fa)</td><td>3,032</td><td>20.9</td><td>369</td><td>20.1</td><td>871</td><td>21.8</td></tr><tr><td>French (Fr)</td><td>3,119</td><td>20.8</td><td>289</td><td>19.9</td><td>676</td><td>21.8</td></tr><tr><td>Hindi (Hi)</td><td>2,072</td><td>20.6</td><td>239</td><td>19.2</td><td>418</td><td>21.4</td></tr><tr><td>Italian (It)</td><td>2,970</td><td>20.6</td><td>391</td><td>20.2</td><td>865</td><td>21.7</td></tr><tr><td>Japanese (Ja)</td><td>2,241</td><td>20.2</td><td>266</td><td>19.6</td><td>650</td><td>21.3</td></tr><tr><td>Portuguese (Pt)</td><td>2,731</td><td>20.7</td><td>386</td><td>20.2</td><td>919</td><td>21.9</td></tr><tr><td>Tamil (Ta)</td><td>2,317</td><td>20.7</td><td>377</td><td>20.0</td><td>591</td><td>22.0</td></tr><tr><td>Ukrainian (Uk)</td><td>2,741</td><td>20.8</td><td>325</td><td>20.3</td><td>750</td><td>22.0</td></tr><tr><td>Vietnamese (Vi)</td><td>2,927</td><td>20.7</td><td>361</td><td>20.2</td><td>857</td><td>21.8</td></tr><tr><td rowspan="14"></td><td>Chinese (Zh)</td><td>3,178</td><td>21.0</td><td>409</td><td>20.6</td><td>945</td><td>22.1</td></tr><tr><td>French (Fr)</td><td>30,000</td><td>8.9</td><td>1,000</td><td>8.9</td><td>14,760</td><td>9.4</td></tr><tr><td>German (De)</td><td>30,000</td><td>9.8</td><td>1,000</td><td>10.2</td><td>13,511</td><td>9.8</td></tr><tr><td>Catalan (Ca)</td><td>30,000</td><td>10.3</td><td>1,000</td><td>10.3</td><td>12,730</td><td>10.5</td></tr><tr><td>Spanish (Es)</td><td>30,000</td><td>9.7</td><td>1,000</td><td>9.6</td><td>13,221</td><td>9.9</td></tr><tr><td>Russian (Ru)</td><td>12,112</td><td>11.9</td><td>1,000</td><td>11.9</td><td>6,300</td><td>11.8</td></tr><tr><td>Chinese (Zh)</td><td>7,085</td><td>12.0</td><td>1,000</td><td>11.9</td><td>4,898</td><td>11.6</td></tr><tr><td>Dutch (NI)</td><td>7,108</td><td>8.2</td><td>1,000</td><td>8.5</td><td>1,699</td><td>8.5</td></tr><tr><td>Turkish (Tr)</td><td>3,966</td><td>8.3</td><td>1,000</td><td>8.1</td><td>1,629</td><td>8.3</td></tr><tr><td>Estonian (Et)</td><td>1,782</td><td>17.8</td><td>1,000</td><td>15.5</td><td>1,571</td><td>16.1</td></tr><tr><td>Mongolian (Mn)</td><td>2,067</td><td>11.2</td><td>1,000</td><td>11.2</td><td>1,759</td><td>11.3</td></tr><tr><td>Arabic (Ar)</td><td>2,283</td><td>5.8</td><td>1,000</td><td>5.7</td><td>1,695</td><td>5.7</td></tr><tr><td>Latvian (Lv)</td><td>2,337</td><td>6.1</td><td>1,000</td><td>6.3</td><td>1,629</td><td>6.2</td></tr><tr><td>Slovenian (S1)</td><td>1,843</td><td>7.2</td><td>509</td><td>7.0</td><td>360</td><td>6.3</td></tr><tr><td>Japanese (Ja) Indonesian (Id)</td><td>1,119 1,243</td><td>8.3 6.6</td><td>635 792</td><td>8.5 6.6</td><td>684 844</td><td>8.4 6.7</td></tr><tr><td rowspan="6">FLEURS (Conneau et al., 2023) (En→X)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Spanish (Es)</td><td>2,502</td><td>25.0</td><td>394</td><td>25.1</td><td>643</td><td>26.1</td></tr><tr><td>French (Fr)</td><td>2,592</td><td>24.4</td><td>363</td><td>24.1</td><td>612</td><td>25.5</td></tr><tr><td>Italian (It)</td><td>2,564</td><td>23.2</td><td>386</td><td>22.8</td><td>640</td><td>24.4</td></tr><tr><td>Japanese (Ja)</td><td>2,290</td><td>53.6</td><td>351</td><td>53.1</td><td>592</td><td>55.6</td></tr><tr><td>Portuguese (Pt) Chinese (Zh)</td><td>2,503 2,592</td><td>22.4 42.3</td><td>387 394</td><td>21.9 40.7</td><td>645 646</td><td>23.4 42.7</td></tr><tr><td rowspan="3">CoVoST-2 (Wang et al., 2020) (En→X)</td><td>Persian (Fa)</td><td>30,000</td><td></td><td></td><td>9.3</td><td>1,000</td><td>9.5</td></tr><tr><td>Japanese (Ja)</td><td></td><td>10.8 28.5</td><td>1,000 1,000</td><td></td><td>1,000</td><td>23.3</td></tr><tr><td>Chinese (Zh)</td><td>30,000 30,000</td><td>19.7</td><td>1,000</td><td>26.6 19.7</td><td>1,000</td><td>16.0</td></tr><tr><td rowspan="3">MuST-C (Di Gangi et al., 2019) (En→X)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Spanish (Es)</td><td>6,000</td><td>19.4</td><td>1,316</td><td>20.1</td><td>2,502</td><td>17.1</td></tr><tr><td>Italian (It) Chinese (Zh)</td><td>6,000 6,000</td><td>18.2 49.6</td><td>1,309 888</td><td>18.8 63.7</td><td>2,574 1,823</td><td>16.4 46.3</td></tr><tr><td colspan="2">Total</td><td>327,533</td><td>15.9</td><td>27,920</td><td>16.9</td><td>102,378</td><td>13.3</td></tr></table>

Table 15: HypoTranslate dataset (ST part) statistics in terms of the number of hypotheses-translation pairs and average length of ground-truth utterance in different language directions.

<table><tr><td>X→En</td><td>Ar</td><td>Cy</td><td>De</td><td>El</td><td>Es</td><td> $\mathrm { F a }$ </td><td>Fr</td><td>Hi</td><td>It</td><td>Ja</td><td>Pt</td><td>Ta</td><td>Uk</td><td>Vi</td><td>Zh</td><td>Avg.</td></tr><tr><td>BLEU score</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SeamlessM4T (ASR+MT)</td><td>38.9</td><td>37.0</td><td>39.7</td><td>29.0</td><td>27.7</td><td>34.1</td><td>37.7</td><td>33.9</td><td>28.9</td><td>21.7</td><td>42.3</td><td>23.7</td><td>34.0</td><td>24.9</td><td>24.4</td><td>31.9</td></tr><tr><td>GenTranslate (ours)</td><td>39.9</td><td>39.4</td><td>41.6</td><td>32.8</td><td>31.2</td><td>35.9</td><td>40.6</td><td>34.9</td><td>32.1</td><td>22.8</td><td>45.0</td><td>24.1</td><td>36.9</td><td>27.4</td><td>25.7</td><td>34.0</td></tr><tr><td>chrF++ score</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SeamlessM4T (ASR+MT)</td><td>62.7</td><td>60.0</td><td>63.8</td><td>55.0</td><td>56.0</td><td>58.7</td><td>62.4</td><td>58.8</td><td>57.0</td><td>47.9</td><td>65.9</td><td>49.8</td><td>59.2</td><td>50.5 </td><td>51.5</td><td>57.3</td></tr><tr><td>GenTranslate (ours)</td><td>63.1</td><td>61.2</td><td>64.9</td><td>57.0</td><td>57.1</td><td>59.7</td><td>64.0</td><td>59.1</td><td>58.0</td><td>47.6</td><td>67.2</td><td>49.7</td><td>60.8</td><td>51.6</td><td>52.0</td><td>58.2</td></tr></table>

Table 16: Speech translation results on FLEURS X En test sets in terms of chrF++ score.

<table><tr><td rowspan="2">Data Source</td><td rowspan="2">Source / Target Language X</td><td colspan="2">Train</td><td colspan="2">Dev.</td><td colspan="2">Test</td></tr><tr><td># Pairs</td><td>Length</td><td># Pairs</td><td>Length</td><td># Pairs</td><td>Length</td></tr><tr><td rowspan="9">FLORES (Costa-jussà et al., 2022) (X→En)</td><td>Arabic (Ar)</td><td>2,062</td><td>20.4</td><td>295</td><td>19.8</td><td>1,012</td><td>21.6</td></tr><tr><td>German (De)</td><td>2,926</td><td>20.7</td><td>363</td><td>20.1</td><td>1,012</td><td>21.6</td></tr><tr><td>Greek (El)</td><td>3,148</td><td>20.9</td><td>271</td><td>20.5</td><td>1,012</td><td>21.6</td></tr><tr><td>Spanish (Es)</td><td>2,732</td><td>20.8</td><td>408</td><td>20.5</td><td>1,012</td><td>21.6</td></tr><tr><td>Persian (Fa)</td><td>3,032</td><td>20.9</td><td>369</td><td>20.1</td><td>1,012</td><td>21.6</td></tr><tr><td>French (Fr)</td><td>3,119</td><td>20.8</td><td>289</td><td>19.9</td><td>1,012</td><td>21.6</td></tr><tr><td>Italian (It)</td><td>2,970</td><td>20.6</td><td>391</td><td>20.2</td><td>1,012</td><td>21.6</td></tr><tr><td>Japanese (Ja)</td><td>2,241</td><td>20.2</td><td>266</td><td>19.6</td><td>1,012</td><td>21.6</td></tr><tr><td>Ukrainian (Uk) Chinese (Zh)</td><td>2,741 3,178</td><td>20.8 21.0</td><td>325 409</td><td>20.3 20.6</td><td>1,012 1,012</td><td>21.6 21.6</td></tr><tr><td rowspan="5">WMT&#x27; {16,19,20} (En→X)</td><td>Czech (Cs)</td><td>15,000</td><td>12.3</td><td></td><td>15.8</td><td>1,997</td><td>18.8</td></tr><tr><td>Japanese (Ja)</td><td>15,000</td><td>49.8</td><td>2,983 1,998</td><td>53.1</td><td>1,000</td><td>59.8</td></tr><tr><td>Lithuanian (Lt)</td><td>15,000</td><td>12.0</td><td>2,000</td><td>16.5</td><td>998</td><td>16.6</td></tr><tr><td>Romanian (Ro)</td><td>15,000</td><td>16.7</td><td>1,999</td><td>22.6</td><td>1,999</td><td>21.7</td></tr><tr><td>Chinese (Zh)</td><td>15,000</td><td>35.6</td><td>1,997</td><td>47.8</td><td>1,418</td><td>60.7</td></tr><tr><td colspan="2">Total</td><td>103,149</td><td>24.0</td><td>14,363</td><td>27.5</td><td>17,532</td><td>26.3</td></tr></table>

Table 17: HypoTranslate dataset (MT part) statistics in terms of the number of hypotheses-translation pairs and average length of ground-truth utterance in different language directions.