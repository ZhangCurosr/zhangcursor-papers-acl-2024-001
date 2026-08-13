# Analyzing Temporal Complex Events with Large Language Models? A Benchmark towards Temporal, Long Context Understanding

Zhihan Zhang<sup>1</sup>, Yixin Cao\*<sup>2</sup>, Chenchen Ye<sup>†3</sup>, Yunshan Ma<sup>4</sup>, Lizi Liao<sup>5</sup>, and Tat-Seng Chua<sup>6</sup>

<sup>1,2</sup>School of Computer Science, Fudan University, <sup>3</sup>University of California, Los Angeles <sup>4,6</sup>National University of Singapore, <sup>5</sup>Singapore Management University zhangzhihan22@m.fudan.edu.cn

## Abstract

The digital landscape is rapidly evolving with an ever-increasing volume of online news, emphasizing the need for swift and precise analysis of complex events. We refer to the complex events composed of many news articles over an extended period as Temporal Complex Event (TCE). This paper proposes a novel approach using Large Language Models (LLMs) to systematically extract and analyze the event chain within TCE, characterized by their key points and timestamps. We establish a benchmark, named TCELongBench, to evaluate the proficiency of LLMs in handling temporal dynamics and understanding extensive text. This bench mark encompasses three distinct tasks - reading comprehension, temporal sequencing, and future event forecasting. In the experiment, we leverage retrieval-augmented generation (RAG) method and LLMs with long context window to deal with lengthy news articles of TCE. Our findings indicate that models with suitable retrievers exhibit comparable performance with those utilizing long context window.

## 1 Introduction

In today’s digital age, the flood of online news highlights the urgent need for quick and precise event analysis. Prior work in topic detection has mainly clustered news articles by representation similarity to identify stories from news streams (Saravanakumar et al., 2021; Yoon et al., 2023). Extending this approach, our focus shifts to the temporal dynamics of these stories, which we term Temporal Complex Events (TCE) (Ma et al., 2023). TCEs consist of semantically related articles that together narrate the development of various entities over time (refer to Figure 1). Understanding the genesis and evolution of TCE, as well as predicting future developments, holds considerable significance for meeting the practical needs of decision-makers, stakeholders, and even the general public interest.

![](images/7311c438247bf5ef1a09c91d0bb3ad75251722cb6502aef888b8c1853bc939b3.jpg)  
Figure 1: An example of temporal complex event (TCE) around Israeli-Palestinian conflict during December 2017. A TCE consists of many news articles with multiple timestamps. Our work extracts the outline of TCE.

Existing research in complex event analysis has made significant strides but is constrained by inadequate natural language processing (NLP) techniques. Some works (Gholipour Ghalandari et al., 2020; Jiao et al., 2023) aims at provide concise insights into real-word events, utilizing data mining method or human-curated datasets. Another line of works (Li et al., 2021; Zhu et al., 2023) further tracks the temporal progression of complex events by converting news articles into structured data, such as temporal knowledge graphs (TKGs). The information extraction (IE) methods involved, however, tend to be costly and error-prone. Interestingly, how can modern powerful NLP models be applied to complex event analysis, and the extent to which they are aware of its temporal dynamics, remain challenging to determine.

In this paper, inspired by the extensive success of LLMs across various NLP challenges, we delve into their suitability for analyzing TCEs and assess their prowess in understanding temporal and long contexts. First, LLMs typically have a limitation in input length, e.g. 4,096 tokens, while a TCE may span tens of news articles and then tens of thousands of tokens (i.e., an average of 29 articles and 18,589 tokens in our experimental datasets). Even if longer context window enables LLMs to take in all articles, existing works (Bai et al., 2023; Xu et al., 2024) have demonstrated their inferior performance with lengthy context. Second, LLMs, pre-trained for next token prediction, sometimes fall short in temporal reasoning tasks (Tan et al., 2023). For TCE analysis, this limitation becomes apparent as it necessitates precise event-timestamp correlation and a deep understanding of chronological and causal connections. Furthermore, building on top of lengthy past events and their temporal relations, their potential for predicting future events is still under-explored.

To this end, we propose a LLM-based pipeline for TCE outline extraction, and build a large-scale benchmark TCELongBench (TLB) for comprehensive investigation. Inspired by (Reddy et al., 2023), we aim at providing a coherent and chronological representation of TCE, i.e. outline with a timeline. We apply a hierarchical summarization framework and then leverage LLM’s in-context learning (ICL) ability (Brown et al., 2020) to extract key points on each day, in the form of sentences. After de-duplication, key points across all timestamps constitute the outline of TCE.

Based on these, we build TCELongBench for temporal, long context evaluation. It contains 88,821 question answering (QA) pairs from 2,289 TCEs, tailored to three distinct tasks: TLB-detail QA, which tests LLMs’ ability to find evidence across numerous articles; TLB-order QA, focusing on understanding temporal sequences; and TLB-forecast QA, challenging LLMs to predict future events based on past information. To ensure dataset integrity, we employed a generate-thenverify paradigm, leading to a dataset with an 88% quality rating across human evaluation metrics.

In our analysis, we employed both retrievalaugmented generation (RAG) methods and LLMs optimized for long contexts to navigate the extensive narratives typical of TCEs. Our findings reveal that (1) while retrievers are crucial for RAG methods, their effectiveness is variable; (2) long-context models excel in managing long temporal sequences but may lead to inferior performance; and (3) models equipped with apt retrievers can match the performance of those designed for long contexts. To sum up, our contributions are threefold:

• We leverage LLMs to extract the outlines and form event chains of TCEs.

• We build TCELongBench that consists of three tasks aiming at testing the model’s capability of temporal, long text understanding.

• We conduct extensive experiments of LLMs leveraging RAG method and LLMs with long context window.

## 2 Related Work

Complex Event Analysis. Some works around complex event analysis rely on schema to extract temporal knowledge graphs from narratives, such as IED (Li et al., 2021) and RESIN-11 (Du et al., 2022). To further capture the temporal characteristics of complex events, Ma et al. (2023) contribute MidEast-TE that associates each event with a timestamp. However, their intricate information extraction pipelines are time-consuming and may lead to unexpected errors for event analysis. Several studies also explore the unstructured storyline of complex events from multiple documents, in the form of summaries (Gholipour Ghalandari et al., 2020), timeline (Steen and Markert, 2019; Gholipour Ghalandari and Ifrim, 2020) and event mentions (Jiao et al., 2023). In this paper, we extract outlines from TCEs, consisting of key points (sentences) that record the detailed actions of entities with suitable granularity and unfold the whole story within the TCE over time.

A more recent work (Reddy et al., 2023) formulates a report generation task around complex events using LLMs, but falls short in large-scale datasets and quantitative analysis on the report quality. However, before delving into long text generation, we aim at evaluating the LLM’s capability of understanding temporal, long text in TCE, and contribute a QA dataset for quantitative comparisons of various baselines.

Related Benchmarks. There are two strands of benchmarks related to TCELongBench. First, temporal reasoning benchmarks (Zhang and Choi, 2021; Dhingra et al., 2022; Tan et al., 2023) mostly focus on Event-Time, Event-Event and/or Time-Time relations of chronicles in Wikipedia. For example, TRAM (Wang and Zhao, 2023) encompasses ten temporal reasoning tasks, including temporal ordering without any context. ForecastQA (Jin et al., 2021) are proposed to develop methods for event forecasting with large volumes of unstructured text data. Second, long text understanding benchmarks (Bai et al., 2023; Dong et al., 2023; An et al., 2023; Shaham et al., 2023) aim at evaluating long text modeling with multiple tasks, such as summarization, question answering, code completion, etc. In contrast, TCELongBench evaluates the model’s understanding of TCEs from three tasks, requiring temporal reasoning, long text understanding as well as forecasting abilities.

## 3 Task Definition

Existing work has identified TCEs from news articles by clustering their semantic embeddings concatednated with temporal indexes (Ma et al., 2023). Each TCE has n timestamps, i.e. a timeline $\mathcal T \ = \ \{ t _ { k } \ : \ k \ \in \ [ 1 , n ] \}$ , and news articles $\mathcal { A } _ { n } ~ = ~ \{ \mathrm { A } _ { \mathrm { k } } ~ : ~ k ~ \in ~ [ 1 , n ] \}$ , where $\mathrm { A _ { k } }$ is the set of news articles on $t _ { k }$ . On each timestamp $t _ { k }$ , we extract $j _ { k }$ number of key points from $\mathrm { A _ { k } }$ , expressed as $\mathrm { P _ { k } } = \{ P _ { 1 , k } , \dots , P _ { j _ { k } , k } \}$ . Each key point is a concise and informative sentence. The collection of key points across all timestamps forms the TCE’s outline $\mathcal { P } = \{ \mathrm { P } _ { \mathrm { k } } : k \in [ 1 , n ] \}$ Note that news articles accessible to models are $\mathcal { A } _ { n - 1 } = \{ \mathrm { A } _ { \mathrm { k } } : k \in [ 1 , n - 1 ] \}$ in our experiment as $\mathrm { A } _ { \mathrm { n } }$ is used for generating forecasting questions. TLB-detail. This is a reading comprehension task aiming at testing the model’s ability to locate and understand detailed information across numerous articles. The input is a question $Q ,$ , a set of shuffled choices $\mathrm { C } = \{ C _ { r } : r \in [ 1 , 4 ] \}$ , and $\boldsymbol { \mathcal { A } } _ { n - 1 }$ , while the output is a choice $C _ { l } \in \mathrm { C }$

TLB-order. This is an ordering task aiming at testing a model’s ability to capture the event-event relations across timestamps. The input is a set of shuffled choices $\mathrm { C } = \{ C _ { r } : r \in [ 1 , R ] \}$ and $\mathcal { A } _ { n - 1 }$ while the output is the chronological order of the choices $\{ C _ { O _ { 1 } } , \ldots , C _ { O _ { R } } \}$

TLB-forecast. This is a forecasting task aiming at testing a model’s ability to predict future event given historical data. We have two settings of answering forecasting questions, multi-choice and open-domain. In multi-choice setting, the input is a question $Q ,$ a set of shuffled choices C = $\{ C _ { r } : r \in [ 1 , 4 ] \}$ and $A _ { n - 1 }$ ; the output is a choice $C _ { l } \in \mathrm { C } .$ In open-domain setting, we only have question $Q$ and $\mathcal { A } _ { n - 1 }$ as the input, while the output

is open for LLMs.

For each question in TLB-detail and TLBforecast, the text span that supports its correct answer lies in the gold article $A _ { g o l d }$ on $t _ { g o l d }$ . While $A _ { g o l d }$ in TLB-detail follows $A _ { g o l d } \in \mathcal { A } _ { n - 1 }$ , the $A _ { g o l d }$ in TLB-forecast is within $\mathrm { A _ { n } } .$ not accessible during evaluation. Moreover, articles on $t _ { g o l d }$ except $A _ { g o l d }$ may offer supporting evidence to the correct answer, suggesting that identifying $t _ { g o l d } .$ rather than precisely matching $A _ { g o l d } .$ , is also pivotal in determining the correct answer.

## 4 Outline Extraction

Inspired by Jiao et al. (2023) and Rashkin et al. (2020), we propose a LLM-based outline extraction pipeline, which tersely organizes the primary content of TCEs along with a clear timeline. Outline in our work consists of key points from all timestamps, each of which is a concise and informative sentence. These key points represent TCEs with suitable granularity, recording the detailed actions of entities and unfolding the whole story over the timelines. Neither the fine-grained TKG nor event mention (phrase) could capture the intricate relations of multiple entities within TCEs.

Our LLM-based outline extraction pipeline consists of three parts, summarization, key point generation and key point filtering (Figure 2 (1)). Initially, we implement a hierarchical summarization framework to filter out extraneous peripheral events, using xgen-7b-8k-inst (Nijkamp et al., 2023). This framework operates as follows: on each timestamp $t _ { k }$ , we summarize each news article within $\mathrm { A _ { k } }$ to distill their essential contents, and then summarize these articles’ summaries to obtain the central event on $t _ { k }$ . Consequently, we compile the daily summaries across all timestamps as $\mathrm { S } = \{ S _ { k } : k \in [ 1 , n ] \}$

We then leverage LLM’s ICL ability to partition daily summaries into key points. We design a few-shot prompt (Table 7), and ask gpt-3.5-turbo-instruct to generate key points $\hat { \mathrm { P } } _ { \mathrm { k } } = \{ \hat { P } _ { 1 , k } , \hdots , \hat { P } _ { \hat { j } _ { k } , k } \}$ given a daily summary $S _ { k }$ Instructions in the prompt specify that key points should be independent, concise, and comprehensive, avoiding any pronoun. Moreover, the prompt incorporates three human-curated examples to steer the model to better performance.

Finally, we implement a filtering mechanism to enhance the quality of timeline. We eliminate redundant key points that duplicate previously conveyed information, by calculating two similarity scores using sup-simcse-bert (Gao et al., 2021) and quora-distilroberta (Reimers and Gurevych, 2020). If any of the similarity scores between $P _ { i , m }$ on $t _ { m }$ and $P _ { j , k }$ on $t _ { k }$ exceeds predefined thresholds, i.e. 0.8, we discard the key point in later position, i.e. $P _ { i , m } ,$ , since $t _ { m } > t _ { k }$ or $i > j$ if $t _ { m } = t _ { k }$ . Subsequently, we obtain the TCE’s outline $\mathcal { P } = \{ \mathrm { P } _ { \mathrm { k } } : k \in [ 1 , n ] \}$

![](images/0f8f9c7b663c3107e16be99e5c4f6da52c25fb92793fba880412d5b4c3086882.jpg)  
(1) Outline Extraction  
(2) Generate-then-verify paradigm  
Figure 2: Pipeline of outline extraction and generate-then-verify paradigm.

## 5 Dataset Generation and Analysis

Based on our extracted outlines, we construct QA datasets in TCELongBench, under a generate-thenverify paradigm. We also show the summary statistics and human evaluation results.

## 5.1 Generate-then-verify Paradigm

We generate questions and answers given key points and news articles, and then verify their quality from multiple aspects, including Evidence, Plausible, Forecasting, Storytelling and Temporal.

## 5.1.1 TCE QA Generation

TLB-detail and TLB-forcast are in the form of multi-choice question answering (MCQ). We leverage LLM and follow the STARC annotation framework (Berzak et al., 2020) to generate question and misleading choices. In specific, for question generation, we ask gpt-3.5-turbo-instruct to propose a question along with its correct answer for each key point in $\mathcal { P }$ . Here we adopt a few-shot prompt (see Table 8 and 9), where examples are from OneStopQA (Berzak et al., 2020) and ForecastQA (Jin et al., 2021). For misleading choices generation, we design instructions under STARC annotation framework: (1) the first choice represents a plausible misunderstanding of the article $A _ { i , k } ;$ (2) the second one is anchored in another random article with a different timestamp ${ A } _ { \hat { i } , \hat { k } }$ $( \hat { k } \neq k )$ , plausible to the question but incorrect; (3) the third one is made up by LLMs (see Table 10). Additionally, since real-world future events are not confined by candidate choices, we adopt an open-domain setting in TLB-forecast, where only questions and news articles are provided.

TLB-order is in the form of ranking problem. To ensure the choices to be ordered have a strong relation with each other, we formulate ranking problems by selecting the key points associated with a common entity, inspired by Lin et al. (2021). In specific, we use spaCy (Honnibal and Montani, 2017) to extract the entities in each key point, and then collect those sharing at least one common entity. For each common entity $e _ { k }$ that links a branch of key points, we select every three of them with neighboring timestamps to form a ranking problem. Note that the choices in all three tasks are randomly shuffled after generation.

## 5.1.2 TCE QA Verification

Although powerful, LLMs may still produce illogical question or hallucination. To filter out noisy QA pairs, we perform an additional verification step as follows. For TLB-detail QA, we consider two aspects:

• Evidence. Considering the quality of question and correct answer, we check if there is direct evidence in $A _ { i , k }$ that supports the correct answer (see Table 11).

• Plausible. Considering the quality of misleading choices, we check if they are different from but sharing similar wording with the correct answer.

TLB-forecast QA further adds one aspect:

• Forecasting (Jin et al., 2021). Considering the logic behind predicting future event, we check if it is true that while the question cannot be answered with certitude using historical data, it remains tractable and guessable for individuals

<table><tr><td rowspan=1 colspan=1>TLB-detail</td><td rowspan=1 colspan=1>Q: What was Syria&#x27;s response to the US&#x27;s recognition of the Golan Heights as Israeli territory?A. Requested UN funding to rebuild after the war. B. Declare military victory over ISIS in response.C. Consider taking military action against Israel. D. Request an urgent meeting with UN Security Council.Reasoning Path: Syria has asked the UN Security Council on Tuesday to hold an urgent meeting on the US decision torecognize the Golan Heights as Israeli territory on 2019-03-38. (Evidence of Choice D) The correct answer is D.</td></tr><tr><td rowspan=1 colspan=1>TLB-order</td><td rowspan=1 colspan=1>A. Syria requested an urgent meeting at the United Nations Security Council to discuss US President Donald Trump&#x27;sdecision to recognize the Golan Heights as Israeli territory, which conflicts with UN resolutions.B. Lebanese government states that Shebaa Farms were not part of Golan Heights as Israel did not annex their territory.C. The US maps will be redrawn to include the Golan Heights as a part of Israel.Reasoning Path: Syria has asked the UN Security Council to hold an urgent meeting on 2019-03-28. (Evidence ofChoice A) A Lebanese official claims that Shebaa Farms were not part of the Golan Heights because &quot;no one mentionedour land to declare its annexation to Israel&quot; on 2019-03-31. (Evidence of Choice B) The US maps are slated to reflectDonald Trump&#x27;s recognition of Israeli sovereignty over the Golan Heights on 2019-03-29. (Evidence of Choice C)Following the timestamps, the correct answer is A,C,B. (Temporal Ordering)</td></tr><tr><td rowspan=1 colspan=1>TLB-forecast</td><td rowspan=1 colspan=1>Q: What will be the response of international community to Israel&#x27;s annexation of Golan Heights after 2019-04-17?A. Remain silent on the issue, as they have no interest in the Middle East conflict.B. Take military action against Israeí, as they see their actions as a threat to global security.C. Support Israel&#x27;s actions and recognize their right to claim the Golan Heights as their own.D. Condemn Israel&#x27;s actions and reaffirm their stance that the Golan Heights is not a part of Israel&#x27;s sovereignty.Reasoning Path: Donald Trump&#x27;s recognition of Israeli sovereignty over the Golan Heights was condemned by France,Germany, UK, Russia, Syria and other countries on 2019-03-29. EU also rejected to recognize Israeli sovereignty overSyrian Golan Heights on 2019-04-16. (Context Location) The international community could be represented by thecountries and EU mentioned in the context. (Bridge Entity) Given their past positions on Israel&#x27;s annexation of GolanHeights, the correct answer is most likely to be D. (Inferring based on past events)</td></tr></table>

Table 1: Examples of three QA tasks in TCELongBench from TCE 2762.

with expertise?

For TLB-order QA, we focus on other two aspects:

• Storytelling. Considering the relations between choices, we check if they are connected by related entities and hopeful to form a storyline?

• Temporal. Considering the time-sensitive feature of temporal ordering, we check if each choice represent an event that just happened, instead of static or past event?

Specifically, Evidence is examined right after the question is generated, and the generation will stop if there is no supportive evidence found. For Plausible, we keep the QA pair if its misleading choices have less-than-ten-words differences with the correct one and do not repeat it, checked by similarity scores. Moreover, we ask gpt-3.5-turbo-instruct to check the resting three aspects in the multi-choice QA format, A for passing, B for failing, and C for not knowing. Inspired by Jin et al. (2021), we repeat three rounds on the same QA pair, which is qualified only when more than two rounds choose A.

After verification, there is a filtering procedure for dropping the repeated QA pairs. We again use the similarity and duplication scores to discard redundant questions in TLB-detail and TLB-forecast, while for TLB-order, the sets of choices that share more than one common key point will be discarded (see Appendix A.1 for details).

## 5.2 Dataset Analysis

Corpus. We use Mideast-TE (Ma et al., 2023) corpus that has identified TCEs from GDELT. We filter out those TCEs whose time span is too long (i.e., one month) or too short (i.e., five days). This results in 2,289 TCEs in total where average articles and days are 29.31 and 17.44 respectively.

Statistics. We randomly assign TCEs into training, development and test sets following 75/15/15 proportions, shown in Table 2. While the day gaps of TCE are evenly distributed within 30 days, their numbers of tokens present right-skewed distributions around 10,000 (see Figure 3).

<table><tr><td rowspan="2">Dataset</td><td colspan="2">Train</td><td colspan="2">Dev</td><td colspan="2">Test</td></tr><tr><td>Num.</td><td>%</td><td>Num.</td><td>%</td><td>Num.</td><td>%</td></tr><tr><td>Complex Event</td><td>1602</td><td>70.0</td><td>343</td><td>15.0</td><td>344</td><td>15.0</td></tr><tr><td>TLB-detail</td><td>43,336</td><td>71.0</td><td>8,916</td><td>14.6</td><td>8,801</td><td>14.4</td></tr><tr><td>TLB-order</td><td>15,149</td><td>71.6</td><td>3,048</td><td>14.4</td><td>2,967</td><td>14.0</td></tr><tr><td>TLB-forecast</td><td>4,565</td><td>69.1</td><td>1,027</td><td>15.6</td><td>1,012</td><td>15.4</td></tr></table>

Table 2: Numbers and proportions of TCE and QA pair in train/dev/test sets.

There are different question types in TLB-detail and TLB-forecast(see Figure 4). MCQs in TLBdetail starts with What (68.22%), How (15.91%), Who (5.55%), etc., while those in TLB-forecast starts with What will (62.58%), How will (11.63%), How many (11.33%), etc. Besides, following Jin et al. (2021), forecasting questions end with a timestamp like "in/after/by 2019-09-18". For TLB-order QA, average day gap of choices is 5.79 days.

![](images/9254ec959961cf0d6fe1e098d9a93f908d947460a863941088bf0b38183c8864.jpg)  
(a) Day Gap

![](images/fceb0798e28fba088dff9304d1848de7685a11aa549165b493cb3438934919eb.jpg)  
(b) Token

Figure 3: Distributions of day gaps (a) and number of tokens (b). Histograms are with the left y-axis and lines of kernel density estimation are with the right y-axis.  
![](images/2685c2f98a5a7044b2d4edb32051cf5b860f04faf3e1b6e9e1d7e5f915f16e0c.jpg)  
in TemLongBench-detail

![](images/ce79ad1e76e6c604d2bcc7d905492da6ed57a0073e393678300b96559d8e00c3.jpg)  
Figure 4: Question types in TLB-detail and TLBforecast.

Challenges. As shown in Table 1, TLB-detail requires accurately identifying relevant text spans and correlating them with candidate choices for answering reading comprehension questions. TLBorder poses a heightened challenge, involving the identification of multiple contexts with varying timestamps and linking temporal information with choices to establish their relations. TLB-forecast entails additional reasoning steps, including entity bridging and inference from historical events.

## 5.3 Human Evaluation

We ask three annotators to evaluate the quality of QA pairs in TCELongBench from multiple dimensions similar to verification step during dataset construction. The evaluation is conducted on a random sample with size 84 from ten TCEs.

Each annotator decides whether or not a QA pair satisfies one dimension by rating it with 1 or 0, 1 for meeting and 0 for failing. On average, the accuracy score of annotators over three tasks is 77.38%, suggesting that tasks in our TCELongBench are quite challenging for humans. Moreover, the evaluation results are 97.61% for Context, 86.90 % for Evidence, 95.67% for Reasonable, 90.12% for Plausible, 77.78% Temporal and 95.56% for Storytelling (see Appendix A.2 for definitions of each dimension). This result proves the high-quality of TCELongBench, which are mainly attributed to two elaborate procedures during dataset construction: (1) few-shot prompts with detailed instructions and human-curated examples from existing datasets (Berzak et al., 2020; Jin et al., 2021); (2) multi-turn verification by LLMs.

## 6 Benchmarking Experiments

## 6.1 Comparing Models

We apply RAG method and LLMs with long context window to our experiments (see Figure 5). Moreover, we conduct evaluation on both LLMs and retrievers.

RAG Method. LLMs with short context window (4,096 tokens) are able to read long text with the help of retrievers. We use four open-source chat models with two sizes (vicuna-7b-4k, vicuna-13b-4k, Llama-2-7b-4k and Llama-2-13b-4k) and one close-source model (gpt-3.5-4k). As for retrievers, we experiment with a sparse retriever BM25, a dense retriever based on text-embedding-ada-002 and a hybrid retriever combining the former two retrievers with a re-ranker. We set the number of retrieved text chunks u and its size l to be 3 and 512 respectively, considering the content window limit.

LLM with Long Context Window. Recent studies have committed to enhancing the long text modeling techniques of LLMs, extending the context length to 16k, 32k and even 128k. In our experiments, we use three models with 16k context length (vicuna-7b-16k, longchat-7b-16k and gpt-3.5-16k), two models with 32k (longchat-7b-32k and chatglm3-6b-32k), and one model with 128k ( gpt-4-128k). All accessible news articles within TCE along with their timestamps and the QA pair are fed into their context window. However, if the number of tokens exceeds the input limit, we discard the articles from $t _ { n - 1 }$ in TLB-detail and TLB-order, and from $t _ { 1 }$ in TLB-forecast, except those on the gold timestamp. Please see Appendix B.3 for more details.

## 6.2 Evaluation Metrics

Task Evaluation. For MCQ in TLB-detail and TLB-forecast, we evaluate using Accuracy. In TLB-order, it is evaluated by Accuracy, weighted F1 score, and Levenshtein distance (Miller et al., 2009). For the open-domain setting in TLBforecast, we evaluate using BLEU (Papineni et al., 2002) and METEOR (Banerjee and Lavie, 2005).

<table><tr><td rowspan="2">Model</td><td rowspan="2">Retriever /Length</td><td rowspan="2">TLB-detail</td><td colspan="3">TLB-order</td><td colspan="3">TLB-forecast</td></tr><tr><td>Acc. Acc. 介</td><td>F1介</td><td>Dist↓</td><td>MCQ Acc.</td><td></td><td>Open-domain BLÉU METEOR</td></tr><tr><td rowspan="3">vicuna-7b-4k</td><td>w/o context</td><td>26.3</td><td>12.2</td><td>24.0</td><td>2.07</td><td>26.8</td><td>0.89</td><td>19.3</td></tr><tr><td>BM25</td><td>68.3</td><td>12.9 / 13.2</td><td>25.4 / 25.3</td><td>2.02 / 2.02</td><td>46.6</td><td>1.20</td><td>22.2</td></tr><tr><td>Openai</td><td>68.5</td><td>12.3 / 13.0</td><td>24.2 / 25.6</td><td>2.06 / 2.00</td><td>48.2</td><td>1.13</td><td>22.5</td></tr><tr><td rowspan="3">Llama-2-7b-4k</td><td>Hybrid</td><td>68.6</td><td>13.2 / 14.1</td><td>26.1 / 27.0</td><td>1.99 / 1.96</td><td>48.3</td><td>1.36</td><td>22.8</td></tr><tr><td>w/o context</td><td>25.3</td><td>9.3</td><td>18.2</td><td>2.29</td><td>15.6</td><td>0.65</td><td>18.8</td></tr><tr><td>BM25 Openai</td><td>70.6 68.2</td><td>11.1 / 12.9 10.8 / 12.3</td><td>22.5 / 24.2 22.1 / 23.4</td><td>2.13 / 2.09 2.14 /2.11</td><td>48.6 49.6</td><td>1.10 0.93</td><td>21.5 21.6</td></tr><tr><td rowspan="3"></td><td>Hybrid</td><td>69.2</td><td>11.4 / 14.5</td><td>22.5 / 26.4</td><td>2.13 / 2.00</td><td>49.1</td><td>0.99</td><td>21.9</td></tr><tr><td></td><td>34.7</td><td>17.8</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/o context BM25</td><td>72.4</td><td>15.7 / 18.6</td><td>34.7 30.8 / 33.9</td><td>1.66 1.80 / 1.72</td><td>30.9 43.4</td><td>0.82 1.28</td><td>18.6 22.4</td></tr><tr><td rowspan="3">vicuna-13b-4k</td><td>Openai</td><td>71.5</td><td>16.4 / 18.8</td><td>31.0 / 33.7</td><td>1.80 / 1.72</td><td>42.2</td><td>1.23</td><td>22.5</td></tr><tr><td>Hybrid</td><td>75.3</td><td>14.7 / 19.0</td><td>28.3 / 34.5</td><td>1.90 / 1.69</td><td>40.7</td><td>1.20</td><td>22.5</td></tr><tr><td>w/o context</td><td>35.2</td><td>18.3</td><td>33.8</td><td>1.67</td><td>29.2</td><td>0.42</td><td>16.6</td></tr><tr><td rowspan="3">Llama-2-13b-4k</td><td>BM25</td><td>78.2</td><td>10.5 / 15.4</td><td>20.4 / 25.6</td><td>2.21 / 2.05</td><td>58.4</td><td>1.01</td><td>22.8</td></tr><tr><td>Openai</td><td>76.5</td><td>9.0 / 16.7</td><td>16.9 / 27.4</td><td>2.33 / 2.00</td><td>59.2</td><td>0.97</td><td>22.6</td></tr><tr><td>Hybrid</td><td>79.8</td><td>10.1 / 14.8</td><td>20.0 / 25.4</td><td>2.22 / 2.06</td><td>57.2</td><td>0.90</td><td>22.6</td></tr><tr><td rowspan="4">gpt-3.5-4k</td><td>w/o context</td><td>56.5</td><td>16.8</td><td>33.2</td><td>1.67</td><td>54.2</td><td>1.25</td><td>17.7</td></tr><tr><td>BM25</td><td>81.8</td><td>15.4 / 18.1</td><td>29.1 / 32.2</td><td>1.87 / 1.81</td><td>57.7</td><td>1.71</td><td>21.0</td></tr><tr><td>Openai</td><td>81.9</td><td>14.8 / 18.3</td><td>27.7 / 32.2</td><td>1.93 / 1.80</td><td>58.0</td><td>1.64</td><td>21.4</td></tr><tr><td>Hybrid</td><td>84.0</td><td>15.3 / 18.8</td><td>28.1 / 32.4</td><td>1.91 / 1.80</td><td>61.7</td><td>2.89</td><td>21.5</td></tr><tr><td rowspan="2">vicuna-7b-16k longchat-7b-16k gpt-3.5-16k</td><td></td><td>37.3</td><td>15.3</td><td>30.8</td><td>1.80</td><td>37.9</td><td>1.55</td><td>23.4</td></tr><tr><td>16k</td><td>34.4</td><td>9.7</td><td>18.5</td><td>2.27</td><td>30.0</td><td>1.05</td><td>19.8</td></tr><tr><td>longchat-7b-32k</td><td></td><td>82.4 26.5</td><td>19.5 8.5</td><td>33.9 17.1</td><td>1.75 2.33</td><td>61.4 22.2</td><td>1.79 1.33</td><td>21.9 22.5</td></tr><tr><td rowspan="2">chatglm3-6b-32k gpt-4-128k</td><td>32k</td><td>79.4</td><td>19.8</td><td>35.4</td><td>1.64</td><td>60.3</td><td>1.11</td><td>14.6</td></tr><tr><td>128k</td><td>91.9*</td><td>29.6</td><td>45.0</td><td>1.42</td><td>72.0</td><td>1.06</td><td>23.4</td></tr></table>

Table 3: Results of TCELongBench. For retrievers, w/o context means answering without any retrieved context; BM25, Openai and Hybrid represent sparse, dense and hybrid retrievers respectively. For TLB-order, “number1/number2” is the result of Retrieve Once strategy and Retrieve One by One strategy respectively. ∗ means experimenting on a random sub-sample with size 1,000, due to cost limitation.

![](images/a4717819c697549f710106a2d01a7d8a798ff631d3a7a0a230fcde9172f76628.jpg)  
Figure 5: Evaluation pipeline of models using RAG method and LLM with Long Context Window.

Retriever Evaluation. We evaluate the retriever’s ability to locate the gold articles and timestamps. In TLB-detail, we use two metrics: (1) Acc\_Doc measures the ratio of questions in which the retriever finds the gold articles; and (2) Acc\_Date measures the ratio of questions in which the retriever finds the gold timestamps. In TLB-order, a ranking problem consist of three shuffled key points as choices, each having a timestamp. So its evaluation metric Acc\_Dates measures the ratio of ranking problems in which the retriever locates all three timestamps of choices. Please see Appendix B.2 for more details and math formulas.

Prompts templates for evaluation are in Appendix C.3, following "[System Message] [Context] Given above articles, please answer the question. [Question] [Candidate Choices]" pattern.

## 6.3 Main Results

The results are reported in Table 3. It is clear and as expected that gpt-4-128k outperforms all other models by a significant margin for all close-ended questions. Lower accuracy scores of MCQs in TLB-forecast than TLB-detail indicates forecasting future event is a more challenging task. Moreover, all models perform poorly in the open-domain of TLB-forecast, where context only brings slight improvement. Additionally, increasing model size drives the performance of Vicuna and Llama-2 upwards across all tasks.

Retriever emerges as a performance bottleneck for models leveraging RAG method. Results of retrievers’ performance in Table 4 offer insights into the varying performance of the same model with different retrievers, as illustrated in Table 3. Specifically, hybrid retriever demonstrates the most optimal performance for each model in TLB-detail, while BM25 and Hybrid retrievers brings out better performance in TLB-order under two strategies respectively.

![](images/f2d0e0cab5f9bd282f6e7e573f7bb80e107e375e03985c24d1304c54638434c1.jpg)  
(a) Input Length

![](images/0710959f0b81b0f63381436833b2c075a3341bb7bcc28f83bdb859bd2e73959b.jpg)  
(b) Input Position

![](images/84c7dc663ec22b54257961e0e978dcbf53dfdf5fdf08e0abb95764b0fb66aa55.jpg)  
(c) Acc\_by\_Dates  
Figure 6: Analysis of results on TCELongBench. (a) shows the average accuracy under different context length in TLB-detail; (b) demonstrates "Lost in the middle" phenomenon in TLB-detail, except for LongChat-16k; (c) shows the Acc\_by\_Dates scores under Retrieve One by One strategy in TLB-order.

Retrievers may not consistently yield effective results. When concatenating three choices in the ranking problem for retrieval, i.e. strategy-1 discussed in Section 6.4, retrievers yield slightly improved performance for open-source 7B models, but worsened performance for open-source 13B models and the close-source model. This observation suggests that inappropriate context can be misleading, particularly for more powerful models. Such discrepancies may arise from potential data leakage during their training stages.

Long context modeling techniques offer benefits for temporal sequencing, but may lead to inferior performance. gpt-3.5-16k and chatglm3-6b-32k achieve comparable performance with gpt-3.5-4k with hybrid retriever, and even perform better in TLB-order. However, vicuna-7b-16k, longchat-7b-16k and longchat-7b-32k underperform retrievalaugmented models by a significant margin. This finding indicates that fintuning longer is still challenging and may lead to inferior performance, while its upper limit could achieve even better performance than RAG method.

## 6.4 Detailed Analysis

We conduct detailed analysis on the experiment results of TCELongBench from various aspects. Impact of Input Length and Position. For finegrained analysis of context of models with long context window, we explore how their performance in TLB-detail varies across different context length ranges of 0-4k, 4k-8k, 8k-12k, and 12k+ <sup>1</sup>. The slopes of curves in Figure 6(a) showcase a drop in performance on data of greater length.

<table><tr><td rowspan="2">Retriever</td><td colspan="2">TLB-detail</td><td colspan="2">TLB-order</td></tr><tr><td>Acc_Doc Acc_Date Acc_Dates-1 Acc_Dates-2</td><td></td><td></td><td></td></tr><tr><td>BM25</td><td>72.8</td><td>85.1</td><td>15.7</td><td>16.2</td></tr><tr><td>Openai</td><td>64.9</td><td>79.1</td><td>5.9</td><td>10.9</td></tr><tr><td>Hybrid</td><td>75.3</td><td>87.5</td><td>1.1</td><td>26.7</td></tr></table>

Table 4: Performance of retrievers, where "-1" and "-2" indicate Retrieving Once strategy and Retrieving One by One strategy respectively.

Furthermore, we investigate the impact of the position of relevant articles on the model’s performance (Liu et al., 2023) in TLB-detail. In specific, we experiment with relocating articles with gold timestamps to different positions within the context window, using a random sample size of 100. As shown in Figure 6(b), most LLMs exhibit improved accuracy towards the end, for questions also being situated at the end of the prompt (see Table 14), except for longchat-7b-16k.

Retrieving for Temporal Sequencing. We employ two retrieving strategies in TLB-order: (1) Retrieve Once strategy concatenates three choices together to retrieve top three text chunks; (2) Retrieve One by One strategy retrieves each choice and then select the text chunk with the earliest timestamp from the top three – the news articles often repeat the reports in earlier days.

Strategy-2 consistently leads to model’s better performance than strategy-1, as shown in Table 3. This finding is explained by results reported in Table 4, where retrievers achieve higher Acc\_Dates scores in strategy-2. Moreover, the combination of hybrid retriever and strategy-2 demonstrates the most optimal performance among most models.

Additionally, candidate choices in strategy-2 could be directly ranked according to the timestamps of retrieved text chunks, that is, no LLMs involved. This accuracy score is labeled as Acc\_by\_Date in Figure 6(c), where we can see that this straightforward approach outperforms others by a considerable margin. This finding demonstrates that LLMs hardly leverage the full temporal information via ICL, even though all timestamps are fed into LLMs with clear format. Incorporating further time-aware instruction tuning could be beneficial, a direction we consider for future research. Open-Domain Error Analysis. We observe that LLMs tend to give lengthy and indirect answers to forecasting questions by using expressions like "It is not possible to accurately forecast what", and "It is difficult to say with 100%". Inspired by Kamalloo et al. (2023), we classify a sample of these open answers into three categories: Semantically Correct, Wrong, and Invalid. Specifically, Semantically Correct answer is semantically equivalent to the ground truth, while Invalid answer suggests that the model refuses to give a clear answer to the forecasting question.

We randomly sample 100 forecasting questions and collect their corresponding of by each 4k model with hybrid retriever. As shown in Figure 7, Llama-2-7b-4k outputs more semantically correct answers than vicuna-7b-4k within the random sample, inconsistent with results in Table 3. gpt-3.5-4k gives the most invalid answers, probably due to stringent safety-alignment technique.

![](images/4fdd639d81168d1c162e9a3de898a747c292132d3466e6e6cbe6468b723816c1.jpg)  
Figure 7: Classification of open-domain answers to 100 random questions in TLB-forecast. The three 4k models are with hybrid retriever.

## 7 Conclusion

In this work, we introduced a LLM-based framework for outline extraction of TCE and established TCELongBench to evaluate LLMs’ capability of temporal understanding and long text comprehension. Our approach involved three tasks targeting reading comprehension, temporal sequencing, and future event forecasting, and conducted experiments across two foundational models: LLMs leveraging RAG method and LLMs with long context windows. While our experiments provided valuable insights into LLMs’ abilities in TCE analysis, future research is essential, particularly in content generation tasks (Reddy et al., 2023), to unlock the full potential of LLMs in complex narrative understanding.

## Limitation

Our work focuses on evaluating LLM’s capability of temporal, long text understanding using test sets of TCELongBench. Thus, we do not utilize the training and development sets, reserving them for future work.

We do not differentiate whether or not news articles in TCELongBench are included in the massive training data of LLMs. This explains why gpt-3.5-4k achieves over 50% accuracy of MCQs without any context – some news articles may be already memorized by LLMs during training stage. Nonetheless, our dataset construction pipeline is adaptable to new, unseen corpora, which will be the focus of our future research.

During experiments, we design prompt templates to instruct LLMs to output their answers under some specific formats (see Appendix C). Answers that do not follow these formats would be regarded as incorrect answers, which leads to the loss of model’s performance. Additionally, some parameters in the experiment setting, such as the number and size of retrieved chunks, could be further adjusted to discover new insights. Due to the content length and time limitation, we set these parameters to fixed values.

## References

Chenxin An, Shansan Gong, Ming Zhong, Xingjian Zhao, Mukai Li, Jun Zhang, Lingpeng Kong, and Xipeng Qiu. 2023. L-eval: Instituting standardized evaluation for long context language models.

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. 2023. Longbench: A bilingual, multitask benchmark for long context understanding.

Satanjeev Banerjee and Alon Lavie. 2005. METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. In Pro-

ceedings ofthe ACL Workshop on Intrinsic and Extrinsic Evaluation Measuresfor Machine Translation and/or Summarization, pages 65–72. Association for Computational Linguistics.

Yevgeni Berzak, Jonathan Malmaud, and Roger Levy. 2020. STARC: Structured annotations for reading comprehension. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 5726–5735. Association for Computational Linguistics.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pages 1877–1901. Curran Associates, Inc.

Bhuwan Dhingra, Jeremy R. Cole, Julian Martin Eisenschlos, Daniel Gillick, Jacob Eisenstein, and William W. Cohen. 2022. Time-aware language models as temporal knowledge bases. Transactions of the Association for Computational Linguistics, 10:257– 273.

Zican Dong, Tianyi Tang, Junyi Li, Wayne Xin Zhao, and Ji-Rong Wen. 2023. Bamboo: A comprehensive benchmark for evaluating long text modeling capacities of large language models.

Xinya Du, Zixuan Zhang, Sha Li, Pengfei Yu, Hongwei Wang, Tuan Lai, Xudong Lin, Ziqi Wang, Iris Liu, Ben Zhou, Haoyang Wen, Manling Li, Darryl Hannan, Jie Lei, Hyounghun Kim, Rotem Dror, Haoyu Wang, Michael Regan, Qi Zeng, Qing Lyu, Charles Yu, Carl Edwards, Xiaomeng Jin, Yizhu Jiao, Ghazaleh Kazeminejad, Zhenhailong Wang, Chris Callison-Burch, Mohit Bansal, Carl Vondrick, Jiawei Han, Dan Roth, Shih-Fu Chang, Martha Palmer, and Heng Ji. 2022. RESIN-11: Schema-guided event prediction for 11 newsworthy scenarios. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies: System Demonstrations, pages 54–63. Association for Computational Linguistics.

Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple contrastive learning of sentence embeddings. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6894–6910. Association for Computational Linguistics.

Demian Gholipour Ghalandari, Chris Hokamp, Nghia The Pham, John Glover, and Georgiana Ifrim.

2020. A large-scale multi-document summarization dataset from the Wikipedia current events portal. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 1302–1308. Association for Computational Linguistics.

Demian Gholipour Ghalandari and Georgiana Ifrim. 2020. Examining the state-of-the-art in news timeline summarization. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 1322–1334, Online. Association for Computational Linguistics.

Thomas Haladyna, Steven Downing, and Michael Rodriguez. 2002. A review of multiple-choice itemwriting guidelines for classroom assessment. Applied Measurement in Education - APPL MEAS EDUC, 15:309–333.

Matthew Honnibal and Ines Montani. 2017. spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing. To appear.

Yizhu Jiao, Ming Zhong, Jiaming Shen, Yunyi Zhang, Chao Zhang, and Jiawei Han. 2023. Unsupervised event chain mining from multiple documents. In Proceedings ofthe ACM Web Conference 2023, WWW ’23, page 1948–1959. Association for Computing Machinery.

Woojeong Jin, Rahul Khanna, Suji Kim, Dong-Ho Lee, Fred Morstatter, Aram Galstyan, and Xiang Ren. 2021. ForecastQA: A question answering challenge for event forecasting with temporal text data. In Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4636– 4650. Association for Computational Linguistics.

Ehsan Kamalloo, Nouha Dziri, Charles Clarke, and Davood Rafiei. 2023. Evaluating open-domain question answering in the era of large language models. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 5591–5606. Association for Computational Linguistics.

Manling Li, Sha Li, Zhenhailong Wang, Lifu Huang, Kyunghyun Cho, Heng Ji, Jiawei Han, and Clare Voss. 2021. The future is not one-dimensional: Complex event schema induction by graph modeling for event prediction. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 5203–5215. Association for Computational Linguistics.

Shih-Ting Lin, Nathanael Chambers, and Greg Durrett. 2021. Conditional generation of temporally-ordered event sequences. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long

Papers), pages 7142–7157. Association for Computational Linguistics.

Jerry Liu. 2022. LlamaIndex.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2023. Lost in the middle: How language models use long contexts.

Yunshan Ma, Chenchen Ye, Zijian Wu, Xiang Wang, Yixin Cao, Liang Pang, and Tat-Seng Chua. 2023. Structured, complex and time-complete temporal event forecasting.

Frederic P. Miller, Agnes F. Vandome, and John McBrewster. 2009. Levenshtein Distance: Information Theory, Computer Science, String (Computer Science), String Metric, Damerau?Levenshtein Distance, Spell Checker, Hamming Distance. Alpha Press.

Erik Nijkamp, Tian Xie, Hiroaki Hayashi, Bo Pang, Congying Xia, Chen Xing, Jesse Vig, Semih Yavuz, Philippe Laban, Ben Krause, Senthil Purushwalkam, Tong Niu, Wojciech Krysci ´ nski, Lidiya Mu-´ rakhovs’ka, Prafulla Kumar Choubey, Alex Fabbri, Ye Liu, Rui Meng, Lifu Tu, Meghana Bhat, Chien-Sheng Wu, Silvio Savarese, Yingbo Zhou, Shafiq Joty, and Caiming Xiong. 2023. Xgen-7b technical report.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings ofthe 40th Annual Meeting on Association for Computational Linguistics, ACL ’02, page 311–318. Association for Computational Linguistics.

Hannah Rashkin, Asli Celikyilmaz, Yejin Choi, and Jianfeng Gao. 2020. PlotMachines: Outlineconditioned generation with dynamic plot state tracking. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4274–4295. Association for Computational Linguistics.

Revanth Gangi Reddy, Yi R. Fung, Qi Zeng, Manling Li, Ziqi Wang, Paul Sullivan, and Heng Ji. 2023. Smartbook: Ai-assisted situation report generation.

Nils Reimers and Iryna Gurevych. 2020. Making monolingual sentence embeddings multilingual using knowledge distillation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics.

Kailash Karthik Saravanakumar, Miguel Ballesteros, Muthu Kumar Chandrasekaran, and Kathleen McKeown. 2021. Event-driven news stream clustering using entity-aware contextual embeddings. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 2330–2340. Association for Computational Linguistics.

Uri Shaham, Maor Ivgi, Avia Efrat, Jonathan Berant, and Omer Levy. 2023. ZeroSCROLLS: A zero-shot benchmark for long text understanding. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 7977–7989. Association for Computational Linguistics.

Julius Steen and Katja Markert. 2019. Abstractive timeline summarization. In Proceedings ofthe 2nd Workshop on New Frontiers in Summarization, pages 21– 31, Hong Kong, China. Association for Computational Linguistics.

Qingyu Tan, Hwee Tou Ng, and Lidong Bing. 2023. Towards benchmarking and improving the temporal reasoning capability of large language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14820–14835. Association for Computational Linguistics.

Yuqing Wang and Yun Zhao. 2023. Tram: Benchmarking temporal reasoning for large language models.

Peng Xu, Wei Ping, Xianchao Wu, Lawrence McAfee, Chen Zhu, Zihan Liu, Sandeep Subramanian, Evelina Bakhturina, Mohammad Shoeybi, and Bryan Catanzaro. 2024. Retrieval meets long context large language models. In The Twelfth International Conference on Learning Representations.

Susik Yoon, Yu Meng, Dongha Lee, and Jiawei Han. 2023. Scstory: Self-supervised and continual online story discovery. In Proceedings of the ACM Web Conference 2023, WWW ’23, page 1853–1864. Association for Computing Machinery.

Michael Zhang and Eunsol Choi. 2021. SituatedQA: Incorporating extra-linguistic contexts into QA. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 7371– 7387. Association for Computational Linguistics.

Fangqi Zhu, Lin Zhang, Jun Gao, Bing Qin, Ruifeng Xu, and Haiqin Yang. 2023. A diffusion model for event skeleton generation. In Findings of the Association for Computational Linguistics: ACL 2023, pages 12630–12641. Association for Computational Linguistics.

## A Dataset

## A.1 Deduplication

We conduct multiple deduplication procedures throughout outline extraction and dataset construction. This is conducted by calculating two similarity scores using sup-simcse-bert <sup>2</sup> (Gao et al., 2021) and quora-distilroberta <sup>3</sup> (Reimers and Gurevych, 2020). While quora-distilroberta is specialized in detecting duplicated questions, sup-simcse-bert offers high-quality sentence embeddings to decide whether two sentences are semantically equivalent based on the similarity score of their embeddings. Both thresholds are set to 0.8 based on our observations in practice. Note that QA pairs in TLB-order is deduplicated by the common key points instead of similarity scores.

The proportion of discarded key points and QA pairs in TCELongBench are shown in Table 5. Note that we also discard the noising key points if their similarity scores are below 0.2 with others in the same TCE, since they may be the regular greetings of LLMs, incomplete sentences, etc.

<table><tr><td></td><td>Before</td><td>After</td><td>%</td></tr><tr><td>Key Point</td><td>137,041</td><td>91,574</td><td>33.2</td></tr><tr><td>TLB-detail</td><td>74,568</td><td>61,053</td><td>18.2</td></tr><tr><td>TLB-order</td><td>55,663</td><td>21,164</td><td>62.0</td></tr><tr><td>TLB-forecast</td><td>7,664</td><td>6,604</td><td>13.8</td></tr></table>

Table 5: Numbers of key points and QA pairs in TCE-LongBench before and after de-duplication, and the proportions of de-duplicated ones.

## A.2 Human Evaluation

We evaluate the quality of our QA datasets from multiple dimensions. For TLB-detail, we evaluate from five dimensions below:

• Human Performance. Annotators are asked to answer multiple choice questions with access to all documents except those on the last days of complex events, and record their accuracy scores.

• Context. We want to see whether the annotators need the context from the documents to understand and answer the question with confidence.

• Evidence. This is to check whether the annotators are able to find the evidence from the documents to support the correct answer.

• Reasonable. Inspired by (Haladyna et al., 2002), Reasonable evaluates the quality of question from three aspects, namely clear, clueless and focused. A clear, clueless and focused question is written in clear and unambiguous language, brings no grammatical or logical cue to the correct answer, and does not contain unnecessary information that is not required to answer it.

• Plausible. Inspired by (Haladyna et al., 2002), Plausible evaluates the quality of four choices from two aspects, namely similar and unique. While all four choices are plausible to the question and homogeneous in wording, they should be essentially different so that there is only one correct answer.

For TLB-forecast, we inherit all five dimensions from TLB-detail, and modify Evidence to Correct&Unseen. Evidence&Unseen does not only require finding the supporting evidence from the articles on the last day, but also check if the annotators are unable to answer the question with 100% certainty given the articles in former days.

For TLB-order, we inherit three dimensions from TLB-detail, Human Performance, Context, and Evidence, and add two new dimensions Temporal and Storytelling shown below. Note that Evidence here is to check if each of the choice indeed comes from the documents in its timestamps, since it is likely that the choice’s content may already exist in the earlier timestamp for summarizing documents in each day sacrificing many details.

• Temporal. This dimension requires the choice’s content presenting the event that just happened or was happening, instead of the event that had happened over a time or may happen in the future.

• Storytelling. We ask the annotators to check whether the choices in the correct order present a brief storyline with potential logic and are connected by common entities.

We give the detailed definitions of above dimensions, as instructions, to three annotators for human evaluation. They are postgraduate students from China and Singapore, proficient in English reading. Detailed results of human evaluation is shown in Table 6. Most QA pairs satisfy the requirements of all dimensions.

## A.3 Quality of Choices in MCQ

To further check the quality of misleading answers, we calculate the proportions of four choices selected by LLMs during evaluating without any context. Recall that (a) is the correct answer while (b), (c) and (d) are misleading answers. As shown in Figure 8(b), vicuna-7b-4k select four candidate choices with nearly equal probability, proving the high-quality of our misleading answers, while Llama-2-7b-4k generate the most invalid answers that do not follow the output format. gpt-3.5-4k achieve over 50% accuracy scores without any context, due to the data leakage during training stage.

<table><tr><td>Dataset</td><td>Num</td><td>Acc.</td><td>Context</td><td>Reasonable</td><td>Plausible</td><td>Temporal</td><td>Storytelling</td><td>Evidence(&amp;Unseen)</td></tr><tr><td>TLB-detail</td><td>30</td><td>85.56</td><td>95.56</td><td>95.56</td><td>84.44</td><td></td><td></td><td>94.44</td></tr><tr><td>TLB-order</td><td>30</td><td>71.11</td><td>98.89</td><td></td><td></td><td>77.78</td><td>95.56</td><td>86.67</td></tr><tr><td>TLB-forecast</td><td>24</td><td>75.00</td><td>98.61</td><td>95.83</td><td>97.22</td><td></td><td></td><td>77.78</td></tr><tr><td>Total</td><td>84</td><td>77.38</td><td>97.61</td><td>95.67</td><td>90.12</td><td>77.78</td><td>95.56</td><td>86.90</td></tr></table>

Table 6: Results of Human Evaluation by three annotators. The unit of all figures are percent % except Num.

![](images/bab193d3f0c28767138d76a6d41b2934b14a79acc55e6737fb6be9b404bcb65b.jpg)  
(a) TLB-detail

![](images/5b9b298cc1ac10a736a7be5d54e6cfcfc9d0afcbbf124565d8b90cebcdbc7f49.jpg)  
(b) TLB-forecast  
Figure 8: Distribution of four choices of experiment results of (a) TLB-detail and (b) TLB-forecast when without any context.

## B Experiment

## B.1 Baseline Models

For LLM with 4k context window, models in our experiments are listed below:

• vicuna-7b-4k <sup>4</sup> and vicuna-13b-4k <sup>5</sup> are both Vicuna v1.5, fine-tuned from Llama 2 with supervised instruction fine-tuning.

• Llama-2-7b-4k <sup>6</sup> and Llama-2-13b-4k <sup>7</sup> are chatbots based on Llama 2 released by Meta AI.

• gpt-3.5-4k <sup>8</sup> is gpt-3.5-turbo-0613 model provided by OpenAI.

For LLM with long context window, models in our experiments are listed below:

• vicuna-7b-16k <sup>9</sup> is Vicuna v1.5, fine-tuned from Llama 2 with supervised instruction finetuning and linear RoPE scaling.

• longchat-7b-16k <sup>10</sup> is trained by fine-tuning Llama-7b on user-shared conversations collected from ShareGPT, using the condensing rotary embedding technique.

• longchat-7b-32k <sup>11</sup> is the 32k version of vicuna-v1.5-16k.

• chatglm3-6b-32k <sup>12</sup> is ChatGLM 3 with 32k context window.

• gpt-3.5-16k and gpt-4-128k <sup>13</sup> are gpt-3.5- turbo-1106 and gpt-4-1106-preview models provided by OpenAI.

Three retrievers in our experiments are built from Llama-index (Liu, 2022) library. Our experiments run on four A5000 GPUs with 25G memory space.

## B.2 Retriever Evaluation

For models using RAG method, retrievers use the query $Q _ { i }$ to retrieve the top u relevant text chunks with size l, i.e. $\mathrm { X } _ { i } = \{ x _ { i , j } : j \in [ 1 , u ] \}$ , as shown in Figure 5. These chunks X and QA pairs are then fed into LLMs to get the final response. Recall that the gold article and timestamp for $Q _ { i }$ are $A _ { i , g o l d }$ and $t _ { i , g o l d }$ . Each text chunk also has its own timestamp $t _ { i , j }$ and is given to LLMs alongside $x _ { i , j }$

In TLB-detail, we use two metric, Acc\_Doc and Acc\_Date, which shows in how many questions the retriever finds the gold articles and timestamps respectively. In TLB-order, we use Acc\_Dates which shows in how many questions the retriever locates all the three gold timestamps $\mathrm { T } _ { C } = \{ t _ { C _ { r } } : r \in$ [1, R] . Their definitions are shown in $\operatorname { E q . l } , \operatorname { E q . 2 }$ and Eq.3 respectively, where N is the total number of questions, I( ) is the sign function, $\operatorname { T } _ { i , X }$ and $\mathrm { T } _ { i , C }$ are the sets of timestamps of retrieved text chunks and choices for the query $Q _ { i }$ respectively. Note that $R = u = 3$ , indicating that the number of elements in $\operatorname { T } _ { i , X }$ and $\mathrm { T } _ { i , C }$ are the same.

$$
A c c \_ D o c = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { I } ( \sum _ { j = 1 } ^ { u } \mathrm { I } ( x _ { i , j } \in A _ { i , g o l d } ) > 0 )\tag{1}
$$

$$
A c c \_ D a t e = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { I } ( \sum _ { j = 1 } ^ { u } \mathrm { I } ( t _ { i , j } = t _ { i , g o l d } ) > 0 )\tag{2}
$$

$$
A c c \_ D a t e s = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { I } ( \mathrm { T } _ { i , X } = \mathrm { T } _ { i , C } )\tag{3}
$$

## B.3 Truncation of Long Input

For LLM with long context window, if the input exceeds the limit of its context window, some articles are discard follwing the rule below, except those on the gold timestamp(s). Recall that news articles accessible to models are $\mathcal { A } _ { n - 1 } = \{ \mathrm { A } _ { \mathrm { k } }$ $k \in [ 1 , n - 1 ] \}$ without those on $t _ { n }$

TLB-detail. We normally discard the articles one by one from the last accessible timestamp $t _ { n - 1 }$ until the input fits into the context window. However, there are chances that articles between $t _ { 1 }$ and $t _ { g o l d }$ exceed the input limit. In this case, we discard articles from the first timestamp $t _ { 1 }$ . When the articles between $t _ { 1 }$ and $t _ { g o l d }$ and between $t _ { g o l d }$ and $t _ { n - 1 }$ both exceed the input limit, we discard articles from $t _ { 1 }$ and $t _ { n - 1 }$ at the same time.

TLB-order. The ranking problem in TLB-order has three choices with three timestamps as part of the ground truth, i.e. $t _ { 1 , g o l d } < t _ { 2 , g o l d } < t _ { 3 , g o l d } .$ We normally discard the articles one by one from $t _ { n - 1 } \tan { t _ { 3 , g o l d } }$ until fitting into the context window. When not working, we discard those from $t _ { 1 }$ to $t _ { 1 , g o l d }$ . However, there are chances that articles between $t _ { 1 , g o l d }$ and $t _ { 3 , g o l d }$ exceed the input limit. In this case, we randomly sample articles between $t _ { 1 , g o l d }$ and $t _ { 3 , g o l d }$ , but not in $t _ { 2 , g o l d }$ , one by one, until fitting into the context window.

TLB-forecast. We discard the articles one by one from the first timestamp $t _ { 1 } \mathrm { t o } t _ { n - 1 }$ , until the input fits into the context window.

## C Prompt Strategy

## C.1 Outline Extraction

The few-shot prompt for key point extraction is in Table 7.

## C.2 Dataset Construction

The few-shot prompts for QA generation in TLBdetail and TLB-forecast are in Table 8 and Table 9

respectively. The few-shot prompt for misleading choices generation is in Table 10.

The prompt templates for verifying Evidence, Forecasting, and Storytelling and Temporal are in Table 11, Table 12 and Table 13 respectively.

## C.3 Evaluation

The prompt templates for evaluation in TLB-detail, TLB-order and TLB-forecast are in Table 14, Table 15, and Table 16 respectively.

![](images/1e579a184f1123a066af5fad64c1c67813e11d712261af8b970f7ecf0ed636ab.jpg)  
Table 7: Few-shot prompt for key point extraction. The daily summary to be split enters Summary. We call daily summary as article in the prompt in case of misleading LLMs.

![](images/779f510ece9999e0d14f7e28e053ce50a3d1b7ae9d7c14180f97634ee3c60a90.jpg)  
Table 8: Few-shot prompt for QA generation of MCQ in TLB-detail. Point and Article are a key point and article with the same timestamp. The examples are from Berzak et al. (2020).

![](images/39274a832e92b10a2bb4ac5f7fa68e536efdb8ee3cd5f3fa99501ec193287b46.jpg)  
Table 9: Few-shot prompt for QA generation of MCQ in TLB-forecast. Point and Article are a key point and article on Day. Day is the last timestamp of TCE. The instruction is borrowed from Jin et al. (2021), and examples also from Jin et al. (2021).

![](images/994cec7626fc5d82283ef319cdee8bdfc67956ee23227b0f648ca383bf73670f.jpg)  
Table 10: Few-shot prompt for misleading choices generation of MCQ in TLB-detail and TLB-forecast. Article 1 is the article used for generating Question and Answer. Article 2 is a random article on another random timestamp. The instruction and examples are from Berzak et al. (2020).

![](images/edb32bc55996fe48bb76100151744e6dbca7700f9e28995b64b24891c0b6ce18.jpg)  
Table 11: Prompt template for verifying Evidence.

![](images/5750abbd7076f680150c95fb7dc4ed485b8e1743bb9ec680e23a50dd37656505.jpg)  
Table 12: Prompt template for verifying Forecasting.

![](images/76e2d7648e7b86524e2d99b6cd4ebcb32f8f6d443392061052c175a1621219f1.jpg)  
Table 13: Prompt template for verifying Storytelling and Temporal.

```ini
[System Message:] You’re an expert in answering multiple choice questions. And you will never refuse to answer any question.
[Rule:] {Context}
Given the above articles, please select one of the option that is the most appropriate for the question below. Note that you will
never refuse to answer a question.
You should output your answer like ’X. x.’ WITHOUT anything else, where ’x’ is the choice’s letter.
Question:
{Question}
Choices:
{Candidate Choices}
[Output:] Your answer:
```  
Table 14: Prompt template for evaluation in TLB-detail. Context consists of retrieved text chunks/articles and their corresponding timestamps.

<table><tr><td>[System Message:] You are an expert in ordering several sentences to form a chronological storyline. And you will never refuse to order any choice.</td></tr><tr><td>[Rule:] {Context}</td></tr><tr><td>Given the above articles, please order the following choices to form a chronological storyline. Note that you will never refuse to order any choice.</td></tr><tr><td>You should output your answer like &#x27;x,x,x.&#x27; WITHOUT anything else, where &#x27;x&#x27; is the choice&#x27;s letter. Choices:</td></tr><tr><td>{Candidate Choices}</td></tr><tr><td>[Output:] Your answer:</td></tr></table>

Table 15: Prompt template for evaluation in TLB-order. Context consists of retrieved text chunks/articles and their corresponding timestamps.

<table><tr><td>[System Message:] You&#x27;re an expert in forecasting events. You can find out what will happen next given the latest information, even if you are not with 100% certainty. And you will never refuse to answer a forecasting question. [Rule:] {Context}</td></tr><tr><td>Given the above articles, please select the option that is the most likely to be the correct answer the the question. Note that you</td></tr><tr><td>will never refuse to answer a forecasting question, even if without 100% certainty.</td></tr><tr><td>You should output your answer like &#x27;X. x.’ WITHOUT anything else, where &#x27;x&#x27; is the choice&#x27;s letter.</td></tr><tr><td>Question:</td></tr><tr><td>{Question} Choices:</td></tr><tr><td>{Candidate Choices}</td></tr><tr><td>[Output:] Your answer:</td></tr><tr><td></td></tr></table>

Table 16: Prompt template for evaluation in TLB-forecast. Context consists of retrieved text chunks/articles and their corresponding timestamps.