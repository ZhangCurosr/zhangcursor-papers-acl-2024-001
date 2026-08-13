# Synergetic Event Understanding: A Collaborative Approach to Cross-Document Event Coreference Resolution with Large Language Models

Qingkai Min<sup>1,2</sup>, Qipeng Guo<sup>3</sup>, Xiangkun Hu<sup>4</sup>, Songfang Huang<sup>5</sup>, Zheng Zhang<sup>6</sup>, and Yue Zhang<sup>2,7,\*</sup>

<sup>1</sup> Zhejiang University <sup>2</sup> School of Engineering, Westlake University

<sup>3</sup> Shanghai AI Laboratory <sup>4</sup> Fudan University

<sup>5</sup> Alibaba DAMO Academy <sup>6</sup> New York University Shanghai

<sup>7</sup> Institute of Advanced Technology, Westlake Institute for Advanced Study

<sup>2</sup>{minqingkai, zhangyue}@westlake.edu.cn <sup>3</sup>guoqipeng@pjlab.org.cn

<sup>4</sup>xkhu17@fudan.edu.cn <sup>5</sup>songfang.hsf@alibaba-inc.com <sup>6</sup>zz@nyu.edu

## Abstract

Cross-document event coreference resolution (CDECR) involves clustering event mentions across multiple documents that refer to the same real-world events. Existing approaches utilize fine-tuning of small language models (SLMs) like BERT to address the compatibility among the contexts of event mentions. However, due to the complexity and diversity of contexts, these models are prone to learning simple co-occurrences. Recently, large language models (LLMs) like ChatGPT have demonstrated impressive contextual understanding, yet they encounter challenges in adapting to specific information extraction (IE) tasks. In this paper, we propose a collaborative approach for CDECR, leveraging the capabilities of both a universally capable LLM and a task-specific SLM. The collaborative strategy begins with the LLM accurately and comprehensively summarizing events through prompting. Then, the SLM refines its learning of event representations based on these insights during fine-tuning. Experimental results demonstrate that our approach surpasses the performance of both the large and small language models individually, forming a complementary advantage. Across various datasets, our approach achieves stateof-the-art performance, underscoring its effectiveness in diverse scenarios.

## 1 Introduction

Event coreference resolution is a useful task in information extraction (Lu and Ng, 2018). This is crucial for achieving a more comprehensive understanding of intricate narratives and facilitating knowledge extraction from diverse textual sources. The coreference of events typically relies on a thorough understanding of document-level context (Minh Tran et al., 2021; Kriman and Ji, 2021;

![](images/38fbf32f4cf28cf040f553cdc74002d3ac37311b0b277ab99d1391603b7de9fc.jpg)

(a) Existing approach  
![](images/387ea611fe04c2f2e607808ff175002aa23453b278e23fbf6f84cd7ad1180d4c.jpg)  
(b) Our approach  
Figure 1: Models for cross-document event coreference resolution, where the input comprises event mentions from different documents, and the output consists of event clusters formed by coreferential mentions, which are visually represented by icons sharing the same color and shape.

Xu et al., 2022). Cross-document event coreference (Lee et al., 2012), involving the comparison of event mentions from different documents, presents additional challenges. On one hand, distinct events in different documents may be portrayed in a very similar manner, especially for events of the same type (challenge 1). On the other hand, the portrayal of the identical event may vary significantly across different documents (challenge 2). The model is required to grasp comparable coreference evidence from varied contexts and make judgments based on it (refer to the examples in Table 4 and 15 for better illustration).

Existing work (Held et al., 2021; Yu et al., 2022) attempts to address CDECR based on fine-tuning small language models (SLMs)<sup>1</sup>, as shown in Figure 1a. However, the complexity and diversity of the context make it prone to learning pseudofeatures by capturing simple co-occurrences rather than genuinely coreference-related terms, including contextual words, entity mentions and other event mentions associated with the given event mention. Supporting this observation, CDECR remains a significant challenge for SLMs, as evidenced by achieving only around 70% CoNLL F1 score on the FCC dataset (Bugert et al., 2021).

Recent advancements in LLMs have significantly advanced the field of NLP, enabling the effective resolution for tasks like machine translation (Jiao et al., 2023) and text summarization (Bang et al., 2023), with just a few demonstrations. However, when it comes to information extraction (IE) tasks, LLMs encounter challenges in task-specific adaptation. Specifically, LLMs struggle to achieve the same level of accuracy as supervised SLMs because a small number of demonstrations cannot comprehensively cover the complex annotation guidelines of these tasks (Han et al., 2023; Li et al., 2023a). Moreover, the inherent nature of the CDECR task, which involves processing multiple documents, imposes enhanced demands on understanding the lengthy context in the demonstrations.<sup>2</sup> Instead of directly predicting CDECR structures, the relative strength of LLMs can enhance the generic understanding of individual documents, particularly the inherent meaning of diverse event mentions, which is complementary to the advantage of SLMs in understanding structures with thorough fine-tuning.

To leverage the relative strengths of LLMs and SLMs, we propose a collaborative approach, as shown in Figure 1b. First, we use the LLM to summarize event mentions from different documents. Then we feed these insights to the SLM to enhance its understanding of event mentions, enabling it to make coreference judgments based on more focused contexts. For the LLM summarization, we design a two-step workflow with separate generic prompts to guide its comprehension of the context of each mention, instead of task-specific in-context learning or fine-tuning. For the SLM, we employ joint representation learning to integrate the original document and the generated summary.

We conduct experiments on three datasets of CDECR, and the results demonstrate that our collaborative approach, as compared to methods solely relying on the LLM or SLM, exhibits significant improvements. Across all three datasets (ECB+, GVC, and FCC), our approach achieves state-ofthe-art results, with increases of 1%, 2.7%, and 7% in CoNLL F1, respectively (averaged over three independent experiments for each dataset). Through analysis, it is demonstrated that our approach more thoroughly addresses the aforementioned challenge 1 of similarly portrayed contexts, making a substantial contribution to performance improvement.

To the best of our knowledge, we are the first to propose a collaborative approach that leverages the universal capabilities of LLMs to address CDECR, achieving superior performance compared to the state-of-the-art baseline.<sup>3</sup>

## 2 Related Work

CDECR Early work addresses CDECR by employing machine learning methods with manually designed features (Bejan and Harabagiu, 2010; Yang et al., 2015; Vossen and Cybulska, 2018; Bugert et al., 2021). Recent neural approaches have utilized SLMs to encode event mentions, obtaining their embeddings for supervised coreference resolution. Initial efforts involve encoding at sentence level and fusing the embeddings of mentions and the incomplete arguments extracted by SRL as the representation of mentions (Barhom et al., 2019; Zeng et al., 2020; Allaway et al., 2021; Yu et al., 2022). Subsequent work incorporates extensive context directly into encoding, leading to noticeable improvements (Caciularu et al., 2021; Cattan et al., 2021a; Held et al., 2021; Hsu and Horwood, 2022; Ahmed et al., 2023). More recently, Chen et al. (2023) and Ravi et al. (2023) establish connections between event mentions using a discourse rhetorical structure constructor and a GPT-3 model fine-tuned with additional data for temporal reasoning, respectively. In comparison to existing work, we are the first to establish comprehensive connections between event mentions and their corresponding contextual elements, including contextual words, entity mentions, and other event mentions, by leveraging the intrinsic knowledge and out-ofthe-box context comprehension ability of LLMs.

LLM for IE Several recent studies (Ma et al., 2023; Li et al., 2023a; Han et al., 2023; Yuan et al., 2023; Gao et al., 2023; Wei et al., 2023; Xie et al., 2023; Li and Zhang, 2023; Xu et al., 2023; Wadhwa et al., 2023; Qi et al., 2023; Ling et al., 2023) have evaluated the performance of LLMs, predominantly ChatGPT, using in-context learning methods on various IE tasks. These investigations universally demonstrate that LLMs exhibit commendable performance in zero-shot and few-shot settings, yet there remains a substantial gap when compared to state-of-the-art supervised SLMs, with the performance gap widening for more complex tasks.

In addition, there are also methods directly using labeled data from IE tasks to fine-tune LLMs(Lu et al., 2022; Zhou et al., 2023; Wang et al., 2023; Sainz et al., 2023). In general, training on these larger-scale models, such as Code-LLaMA and Flan-T5, has yielded results comparable to supervised baselines and demonstrated improvements in zero-shot settings. However, when the training of LLMs does not result in significant performance gains, the training cost, compared to training SLMs, becomes less cost-effective.

Integration of LLM and SLM The integration of LLM and SLM is an emerging approach, with only a few explorations in complex IE tasks. Ma et al. (2023) prompts the LLM to rerank a few difficult samples filtered by the supervised SLM and achieves improvements on various few-shot IE tasks. Their method is based on the observation that LLMs excel only at a small number of hard samples. Wan et al. (2023) first utilizes the LLM to generate reasoning logic for demonstrations retrieved by a fine-tuned SLM, then feeds this combined input back to the LLM for relation extraction, surpassing supervised baselines on some datasets. An inherent challenge lies in finding reasonable demonstrations for NULL-type triples, leading to poor performance on complex tasks such as ACE05. Additionally, inducing complex reasoning logic for each of the k-demonstrations is costly, leading them to sample only a subset of ACE05 and TACRED test sets. Xu et al. (2023) and Li et al. (2023b) leverage LLMs for data enhancement in sentence and document-level relation extraction tasks, respectively. The gap between triples recognized by LLMs and those annotated under manually crafted rules introduces potential shifts in data distribution, making the effectiveness in applications unclear.

Overall, the aforementioned integration methods have exhaustively attempted to adapt LLMs to specific tasks by prompting them to establish accurate connections with artificially defined labels. In contrast, our approach only requires LLMs to perform generic tasks, leveraging their inherent capabilities to assist specific tasks.

Concurrently, akin to our approach, Ding et al. (2024) and Nath et al. (2024) also leverage LLM generation to assist SLM on CDECR. While Ding et al. (2024) prompts LLM with task instructions to generate multiple counterfactual instances for original mention pairs, Nath et al. (2024) employs similar task prompts to guide LLM in generating coreference reasoning processes for mention pairs. Unlike our approach, which involves a general task of having LLM process each mention individually, their methods require LLM to directly handle the relationships between mentions given coreference labels. In terms of efficiency, their methods are less effective than ours as they need to handle combinations of mention pairs, resulting in a quadratic increase in the number of processing entries.

## 3 Method

We adopt the state-of-the-art method proposed by Held et al. (2021) as our baseline (Section 3.1), then summarize events using generic prompts for LLM (Section 3.2), and finally integrate the representations of events from both the summary and the original context into baseline system (Section 3.3).

## 3.1 Task and Baseline

The goal of the CDECR task is to group coreferential event mentions across multiple documents into clusters. We formalize the task as follows:

Input: A corpus comprising multiple documents denoted by D, where $\boldsymbol { D } = \{ D _ { 1 } , D _ { 2 } , . . . , D _ { | D | } \}$ with D representing the number of documents in the dataset. Let M represent all event mentions in the corpus, such that M = $\{ m _ { 1 1 } , m _ { 1 2 } , \hdots , m _ { i j } , \hdots , m _ { | D | , k } \}$ , where k denotes the number of event mentions in each document, and $m _ { i j }$ signifies the $j \cdot$ -th event mention in document $D _ { i }$

Output: A set of clusters, denoted as $C ,$ where $\textit { C } = \{ C _ { 1 } , C _ { 2 } , . . . , C _ { n } \}$ For each cluster $C _ { k } , \ E _ { k }$ represents all the event mentions contained in the cluster $C _ { k } ,$ such that $\begin{array} { r l } { E _ { k } } & { { } = } \end{array}$ $\{ e _ { k 1 } , e _ { k 2 } , \ldots , e _ { k j } , \ldots , e _ { k M } \}$ , where M is the total number of event mentions in cluster $C _ { k }$ , and $e _ { k j }$ is the jth event mention in the cluster $C _ { k }$

Our baseline consists of two key modules for coreference clustering: candidate retrieval and pairwise classification. Both of these modules primarily involve using a RoBERTa (Liu et al., 2019) encoder to encode the context and obtain vector representations of event mentions, which can be seamlessly replaced by our collaborative approach. We formalize the encoding process as follows:

<table><tr><td>Step 1</td><td>News: [input document] Question 1: In this news, given “[mention 1]&quot;mentioned in the sentence “[the sentence]&quot;. Please elaborate [mention 1] in the context of the news article.</td></tr><tr><td>Step 2</td><td>Question  $\mathbf { 2 : \dots }$  News: [input document] Question 1: In this news, given “[mention  $1 ] ^ { , , }$  mentioned in the sentence “[the sentence]”. Elaboration: [output from step 1]. Please further elaborate “[mention 1]&quot;by providing details for entities in the elaboration utilizing coreference resolution. Provide any available or approximate dates in the news for reference, which can be inferred from the publication date of the news if available. Present the information in the following format: Elaboration: [mention 1] refers to &lt;placeholder&gt;&#x27;.</td></tr></table>

Table 1: The two-step workflow for LLM summarization. Each prompt includes a document along with multiple event mentions. Step 2 takes the output from Step 1 as its input. The content to be filled is represented as [content].

For each event mention $m _ { i j }$ , its vector representation can be obtained as:

$$
h _ { i j } = f _ { \mathrm { e n c } } ( m _ { i j } , D _ { i } )\tag{1}
$$

Here $f _ { \mathrm { e n c } }$ is an encoder network used for encoding $D _ { i }$ and concatenating the representations of the boundary tokens of $m _ { i j }$ . The resulting representation $h _ { i j }$ is fed into the subsequent neural network.

## 3.2 LLM Summarization

Summarizing events for CDECR poses a nontrivial challenge. Existing summarization methods are typically designed to provide a general overview of documents, making it difficult to extract information specific to certain types of events. This not only provides limited assistance for coreference but may also lead to the omission of crucial details. Furthermore, designing a summary template for each type of event is not only impractical in real-world applications<sup>4</sup> but also introduces bias, potentially causing LLMs to misinterpret or hallucinate information due to the inherent incompleteness of event information in documents.

To address various types of events and gather specific details from complex contexts, we design a two-step workflow to prompt the LLM, as shown in Table 1. The first step is responsible for extracting tailored information for different types of events in the context of the document. The second step aims to expand the details of the entities mentioned in the output of the first step, as entity details are often scattered throughout the document. In each step, we employ a straightforward prompt to accomplish a primary task. Our prompts adhere to the basic principle of faithfulness, avoiding additional interpretations to prevent semantic shifts. Compared to a synthesized single-step workflow, our two-step workflow guarantees that each step remains focused on its main objective, thereby preventing interference between the two steps, as illustrated by the analysis in Section 4.4.

In the first step, we instruct the LLM agent to “elaborate” an event mention, rather than the conventional instruction of “summarize”. The term “elaborate” implies an explanatory behavior based on the concept of the mention words themselves, emphasizing the support of details from the document context. This suggests that LLMs can automatically select any relevant details from the context to support this explanation, including contextual words, entity mentions, and event mentions. This provides a standardized and feasible way to understand events, leveraging the LLM’s intrinsic knowledge and contextual understanding capabilities without imposing complex rules for the LLM to adhere to.

In the second step, we prompt the LLM agent to use coreference resolution to aggregate detailed information about entities, as entity coreference is a more standardized task compared with event and performing it within a document reduces complexity. Additionally, we require the LLM to perform temporal reasoning based on the publication date of the document, further reducing ambiguity in coreference evidence comparison.

In both steps, we specify the generation format to ensure the consistency between the mention spans in summary and original document. This not only reduces the generation difficulty of LLM but also facilitates SLM in establishing the connection between the two during joint representation learning.

## 3.3 Integration into Final SLM

The SLM takes the original document and the generated summary as inputs. Through a direct joint representation learning technique, the new mention vector representation can be seamlessly integrated into the baseline.

Specifically, for the mention $m _ { i j }$ , let $S _ { i j }$ represents the generated summary, and $m _ { i j } ^ { ( s ) }$ signifies the mention within it. By concatenating the original document $D _ { i }$ and the summary $S _ { i j }$ , a new document $D _ { i } ^ { \prime }$ is formed. Let $f _ { \mathrm { e n c } } ^ { \prime }$ denotes the new encoder network. It first encodes the new document $D _ { i } ^ { \prime }$ , obtaining vector representations $h _ { i j }$ and $h _ { i j } ^ { ( s ) }$ for $m _ { i j }$ and $m _ { i j } ^ { ( s ) }$ respectively. These vectors are then concatenated to form the fused mention vector representation $h _ { i j } ^ { \prime }$ , which can seamlessly substitute $h _ { i j }$ in the baseline for subsequent operations. The joint representation learning process can be represented as:

$$
\begin{array} { r l } & { h _ { i j } ^ { \prime } = f _ { \mathrm { e n c } } ^ { \prime } ( \{ e _ { i j } , e _ { i j } ^ { ( s ) } \} , D _ { i } ^ { \prime } ) } \\ & { \quad \quad = \mathrm { c o n c a t } \left( f _ { \mathrm { e n c } } ( \{ e _ { i j } , e _ { i j } ^ { ( s ) } \} , D _ { i } ^ { \prime } ) \right) } \\ & { \quad \quad = \mathrm { c o n c a t } ( h _ { i j } , h _ { i j } ^ { ( s ) } ) } \end{array}
$$

Here $\{ e _ { i j } , e _ { i j } ^ { ( s ) } \}$ denotes a set containing two elements, implying that vector representations for both $e _ { i j }$ and $e _ { i j } ^ { ( s ) }$ can be derived using the same process as for a single element.

This integration method, which involves concatenating the original context and generated summary for joint representation learning, enables mutual learning of each other’s context in the same attention space, thereby enhancing the understanding of genuinely coreference-related terms.

## 4 Experiments

## 4.1 Experimental Settings

Dataset We conduct experiments on three CDECR datasets: Event Coreference Bank Plus (ECB+) (Cybulska and Vossen, 2014), Gun Violence Corpus (GVC) (Vossen et al., 2018), and Football Coreference Corpus (FCC) (Bugert et al., 2021). The widely-used ECB+ dataset consists of news articles from various topics, including earthquakes, murders, acquisitions, etc. Each topic includes two similar subtopics, such as “6.1 earthquake Indonesia 2009” and “6.1 earthquake Indonesia $2 0 1 3 ^ { \circ }$ . This setup aligns with the challenge 1 mentioned in introduction, asking the model to distinguish similar events. Similarly, GVC and FCC, focusing on news incidents of gun violence and football tournaments, respectively, also have multiple subtopics under one overarching topic. More details can be found in Table 7 (Appendix A.1).

Evaluation Metrics Following previous work (Barhom et al., 2019; Held et al., 2021), we conduct a comprehensive comparison using metrics including MUC, $B ^ { 3 }$ , CEAF , CoNLL, and LEA. The CoNLL F1 is a composite metric representing the average of the first three. $B ^ { 3 }$ is chosen for analysis, following Held et al. (2021).

Hyper Parameters For LLM summarization, we use the ${ } ^ { \cdot \cdot } \mathsf { G P T } { \cdot } 4 { - } \theta 6 1 3 ^ { \cdot \cdot }$ model via OpenAI API, setting the sampling temperature $t = 0$ to reduce the impact of randomness. In the first step of the generation workflow, we introduce a pre-step of instructing the LLM to perform dependency parsing on the sentence containing the event mention. Based on the parsing results, the LLM then elaborates on the mention. For SLM integration, we employ the pre-trained $\mathrm { R o B E R T a _ { L A R G E } }$ model (Liu et al., 2019) to embed event mentions, following our baseline (Held et al., 2021). For all three datasets, we apply a consistent set of hyper-parameters for finetuning, as detailed in Table 8 (Appendix A.2). In all experiments, be it primary results or analyses, we ensure reliability by conducting three independent experiments and averaging the outcomes.

Directly Using LLM to Predict the Structure of CDECR We test the performance of GPT-4 using different in-context learning methods, including few-shot and zero-shot learning, with varied contexts, such as full context and mention-inclusive sentences<sup>5</sup>. The inherent nature of the CDECR task poses a challenge for LLMs in dealing with inputs (comprising hundreds of documents as context) and outputs (consisting of coreference structures formed by thousands of event mentions) that exceed manageable lengths. To tackle this, we first opt for the “GPT-4-Turbo-Preview” model from

<table><tr><td rowspan="2">Methods</td><td colspan="3">MUC</td><td colspan="3"> $\overline { { B ^ { 3 } } }$ </td><td colspan="3"> $\overline { { \mathrm { C E A F } _ { e } } }$ </td><td colspan="3">CoNLL</td><td colspan="3">LEA</td></tr><tr><td>R</td><td>P</td><td>F1</td><td>R</td><td>P</td><td>F1</td><td></td><td>R</td><td>P</td><td>F1</td><td>F1</td><td>R</td><td>P</td><td>F1</td></tr><tr><td colspan="10">ECB+</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Barhom et al. (2019)</td><td>77.6</td><td>84.5</td><td>80.9</td><td>76.1</td><td></td><td>85.1</td><td>80.3</td><td>81.0</td><td>73.8</td><td>77.3</td><td>79.5</td><td></td><td></td><td></td><td></td></tr><tr><td>Cattan et al. (2020)</td><td>85.1</td><td>81.9</td><td>83.5</td><td>82.1</td><td></td><td>82.7</td><td>82.4</td><td>75.2</td><td>78.9</td><td>77.0</td><td>81.0</td><td></td><td></td><td></td><td></td></tr><tr><td>Bugert et al. (2021)</td><td>76.0</td><td>76.1</td><td>76.1</td><td></td><td>71.8</td><td>81.2</td><td>76.2</td><td>72.2</td><td>72.1</td><td>72.2</td><td>74.8</td><td></td><td>55.1</td><td>67.9</td><td>60.8</td></tr><tr><td>Caciularu et al. (2021)</td><td>87.1</td><td>89.2</td><td>88.1</td><td></td><td>84.9</td><td>87.9</td><td>86.4</td><td>83.3</td><td>81.2</td><td>82.2</td><td></td><td>85.6</td><td>76.7</td><td>77.2</td><td>76.9</td></tr><tr><td>Held et al. (2021)</td><td>87.0</td><td>88.1</td><td>87.5</td><td></td><td>85.6</td><td>87.7</td><td>86.6</td><td>80.3</td><td>85.8</td><td>82.9</td><td></td><td>85.7</td><td>74.9</td><td>73.2</td><td>74.0</td></tr><tr><td>Hsu and Horwood (2022)</td><td>87.8</td><td>82.9</td><td>85.3</td><td></td><td>86.5</td><td>83.1</td><td>84.8</td><td>76.9</td><td>82.8</td><td>79.7</td><td></td><td>83.3</td><td>74.4</td><td>74.0</td><td>74.2</td></tr><tr><td>Yu et al. (2022) Ahmed et al. (2023)6</td><td>88.1</td><td>85.1</td><td>86.6</td><td></td><td>86.1</td><td>84.7</td><td>85.4</td><td>79.6</td><td>83.1</td><td>81.3</td><td></td><td>84.4</td><td></td><td></td><td></td></tr><tr><td>Chen et al. (2023)</td><td>80.0</td><td>87.3</td><td>83.5</td><td></td><td>79.6</td><td>85.4</td><td>82.4</td><td>83.1</td><td>75.5</td><td>79.1</td><td></td><td>81.7</td><td>70.5</td><td>73.3</td><td>71.9</td></tr><tr><td></td><td>88.6</td><td>85.9</td><td></td><td>87.2</td><td>87.8</td><td>85.4</td><td>86.6</td><td>82.8</td><td>83.7</td><td>83.2</td><td></td><td>85.7</td><td></td><td></td><td></td></tr><tr><td>GPT-4</td><td>79.8</td><td>78.0</td><td></td><td>78.9</td><td>76.3</td><td>78.1</td><td>77.2</td><td>73.3</td><td>75.6</td><td>74.4</td><td></td><td>76.8</td><td>65.0</td><td>70.0</td><td>67.4</td></tr><tr><td>Our baseline</td><td>86.6</td><td>86.8</td><td>86.7</td><td></td><td>87.1</td><td>86.0</td><td>86.5</td><td>82.6</td><td>82.5</td><td>82.5</td><td></td><td>85.2</td><td>77.8</td><td>76.6</td><td>77.2</td></tr><tr><td>Our method</td><td>89.4</td><td>87.1</td><td></td><td>88.2</td><td>89.1</td><td>86.5</td><td>87.8</td><td>82.7</td><td>85.5</td><td>84.1</td><td></td><td>86.7</td><td>79.7</td><td>78.5</td><td>79.3</td></tr><tr><td colspan="10">GVC</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Barhom et al. (2019)</td><td></td><td></td><td></td><td></td><td>81.0</td><td>66.0</td><td>72.7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Bugert et al. (2021)</td><td>66.3</td><td>78.1</td><td>71.7</td><td></td><td>49.9</td><td>73.6</td><td>59.5</td><td>60.9</td><td>38.2</td><td>47.0</td><td>59.4</td><td></td><td>38.2</td><td>56.5</td><td>45.6</td></tr><tr><td>Held et al. (2021)</td><td>91.8</td><td>91.2</td><td>91.5</td><td></td><td>82.2</td><td>83.8</td><td>83.0</td><td>75.5</td><td>77.9</td><td>76.7</td><td></td><td>83.7</td><td>79.0</td><td>82.3</td><td>80.6</td></tr><tr><td>Ahmed et al. (2023)</td><td>84.0</td><td>91.1</td><td>87.4</td><td></td><td>79.0</td><td>76.4</td><td>77.7</td><td>69.6</td><td>52.5</td><td>59.9</td><td></td><td>75.0</td><td>74.1</td><td>63.9</td><td>68.6</td></tr><tr><td>GPT-4</td><td>7.6</td><td>54.9</td><td>13.4</td><td></td><td>5.5</td><td>34.6</td><td>9.6</td><td>4.2</td><td>42.8</td><td>7.6</td><td></td><td>10.2</td><td>4.4</td><td>28.0</td><td>7.6</td></tr><tr><td>Our baseline</td><td>91.3</td><td>92.0</td><td>91.7</td><td></td><td>86.2</td><td>83.8</td><td>84.9</td><td>78.7</td><td>76.5</td><td>77.6</td><td></td><td>84.7</td><td>82.0</td><td>78.4</td><td>80.2</td></tr><tr><td>Our method</td><td>92.4</td><td>93.2</td><td>92.8</td><td></td><td>87.0</td><td>87.4</td><td>87.2</td><td>83.6</td><td>80.7</td><td>82.1</td><td></td><td>87.4</td><td>83.4</td><td>83.0</td><td>83.2</td></tr><tr><td colspan="10">FCC</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Barhom et al. (2019)</td><td></td><td></td><td></td><td></td><td>36.0</td><td>83.0</td><td>50.2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Bugert et al. (2021)</td><td>82.7</td><td>78.3</td><td>80.4</td><td></td><td>70.8</td><td>38.3</td><td>49.2</td><td>28.2</td><td>40.4</td><td>33.2</td><td>54.3</td><td></td><td>60.4</td><td>30.4</td><td>39.8</td></tr><tr><td>Held et al. (2021)</td><td>86.4</td><td>75.7</td><td>80.7</td><td></td><td>61.6</td><td>65.4</td><td>63.5</td><td>39.1</td><td>65.3</td><td>48.9</td><td>64.4</td><td></td><td>47.2</td><td>57.0</td><td>51.6</td></tr><tr><td>GPT-4</td><td>0.1</td><td>1.0</td><td>0.2</td><td>69.4</td><td>2.3</td><td>99.4 66.6</td><td>4.5 68.0</td><td>14.1 76.4</td><td>13.1 52.2</td><td>13.6</td><td></td><td>6.1 71.7</td><td>0.0 63.5 54.6</td><td>1.1</td><td>0.0 58.7</td></tr><tr><td>Our baseline Our method</td><td>81.4 85.3</td><td>89.0 90.6</td><td>85.1 87.8</td></table>

Table 2: Performance comparison on the ECB+, GVC, and FCC datasets. Our baseline results are obtained by replicating the state-of-the-art method proposed by Held et al. (2021), with the adoption of more advanced hyper-parameters. Our method shows a statistically significant improvement compared to our baseline, with a significance level of $p < 0 . 0 1$ . The results of GPT-4 are based on the best-performing method, specifically through few-shot learning with limited context. The best results are highlighted in bold.

OpenAI, which supports input up to 120k tokens and output to 4096 tokens. Second, we partition the data by topic and process it sequentially. Each time, all documents within a single topic are used as input, and the outputs from all topics are simply merged for testing. Note that this procedure only applies to the multi-topic ECB+ dataset, as there are no cross-topic links. For the single-topic GVC and FCC datasets, inputs and outputs exceeding the length limit are directly truncated. More implementation details, including prompt design as well as the selection and parameter settings of the GPT-4 models, can be found in Appendix A.3.

## 4.2 Results

The main results are presented in Table 2. Our method achieves new state-of-the-art results on all three datasets, outperforming both the previously reported best results and the improved results obtained by our reproduced baseline.

ECB+ On this widely studied dataset, our method demonstrates improvements of 1.5% in CoNLL F1, compared to our baseline. In comparison to Held et al. (2021), upon which our baseline is built, our method also exhibits a 1% increase in CoNLL F1.<sup>7</sup> This improvement stands out notably in recent research, accompanied by significance testing to demonstrate its robustness. Compared to Chen et al. (2023), who also employs RoBERTa<sub>LARGE</sub> for encoding while proposing a different method to leverage broader contexts, we also achieve a 1% improvement, showcasing the effectiveness of our method in utilizing context. More experiments and discussions under additional evaluation principles, including without singletons and at the topic level, can be found in the Appendix B.2.

GPT-4 utilizing few-shot learning significantly lags behind our method, with nearly a 10% gap in CoNLL F1, indicating that GPT-4 still faces substantial adaptability challenges in directly predicting cross-document event coreference structures. This also demonstrates the effectiveness of our method in leveraging the inherent general capabilities of LLMs. Further analysis of GPT-4’s performance can be found in Section 4.5. We also compare the efficiency of LLM utilization (in terms of number of API calls and token consumption) between summarization and structure prediction, as detailed in Appendix B.1.

GVC & FCC Our method demonstrates improvements of 2.7% and 7.0% in CoNLL F1 on the GVC and FCC datasets, respectively, compared to our baseline. The significant improvement on the challenging FCC dataset further underscores the effectiveness of our collaborative approach in leveraging LLM. Additionally, our baseline also shows improvements of 1.0% and 7.3% compared to Held et al. (2021), highlighting our comprehensive exploration on these two less-studied datasets.

GPT-4 exhibits abnormal performance on the GVC and FCC datasets, primarily due to truncation issues stemming from its length constraints, as mentioned in Section 4.1. This is more pronounced on FCC, where longer multi-document contexts are encountered in the test set compared to GVC (4274 vs 1360 sentences). Further elaboration on the truncation problem can be found in Appendix B.1.

## 4.3 The Impact of LLM Summarization

Error Analysis To gain a deeper understanding of the improvements achieved through LLM summarization, we perform a quantitative analysis on the false links within the clusters (see Table 3).

Similar to Yu et al. (2022), we categorize link errors into two types: false positive (FP) and false negative (FN). FP links (incorrect links) occur when two non-coreferential mentions are clustered together, while FN links (missing links) occur when two coreferential mentions are not clustered together. Additionally, we further categorize FP links into two sub-types based on whether two mentions share the same event type.<sup>8</sup> FPA (false positives caused by arguments) indicates that two mentions of the same type differ in argument information. FPT (false positives caused by types) implies that two mentions actually belong to different event types, eliminating the need to consider arguments.

<table><tr><td>Dataset</td><td>Method</td><td>FPA</td><td>FPT</td><td>FN</td></tr><tr><td rowspan="2">ECB+</td><td>Our baseline</td><td>1775</td><td>302</td><td>1262</td></tr><tr><td>Our method</td><td>1227</td><td>152</td><td>1087</td></tr><tr><td rowspan="2">GVC</td><td>Our baseline</td><td>1412</td><td>13</td><td>1041</td></tr><tr><td>Our method</td><td>865</td><td>13</td><td>1173</td></tr><tr><td rowspan="2">FCC</td><td>Our baseline</td><td>38522</td><td>0</td><td>8978</td></tr><tr><td>Our method</td><td>4037</td><td>20</td><td>8575</td></tr></table>

Table 3: Statistics of errors by different types.

FPA Our method demonstrates the most substantial reduction in FPA errors across all three datasets, making the greatest contribution to the overall improvement. The reduction is approximately 30% for both ECB+ and GVC, and nearly 90% for FCC. The significant reduction on FCC is primary attributed to its nature, comprising multiple consecutive events from a large tournament, resulting in more pronounced contextual similarities. This underscores the effectiveness of our method in distinguishing events with similar contextual narratives (aligning with the challenge 1 from introduction). In Table 4, we present instances illustrating two highly similar earthquakes. The original context includes details about the earthquake occurrence, earthquake casualties, media coverage, and historical events. Our generated summaries primarily focus on the core details of the earthquakes, such as date and specific location, thus facilitating their differentiation. It can be observed that our LLM summarization is capable of identifying specific information for particular events and aggregating sufficient details from the entire context.

FPT Compared to FPA, there are significantly fewer FPT errors, only appearing in the ECB+ dataset. The few occurrences on the single-topic GVC and FCC datasets can likely be disregarded, possibly due to random factors. This is because in the multi-topic ECB+ dataset, there may be topics in the test set that were not encountered in the training set, leading to unseen event types. By reducing half of the FPT errors on the ECB+ dataset, it signifies that our summarization also assists in distinguishing unseen event types.

FN Our method shows less improvement in reducing FN errors compared to FP. The challenges arise from two primary factors. Firstly, mentions of the same event can vary greatly in expression styles. Secondly, some event mentions naturally lack sufficient details as the authors assume that readers already possess necessary background information. We illustrate these issues with instances in Appendix B.3. For these cases (aligning with the challenge 2 in introduction), additional training data or external information retrieval may be necessary, as our faithful summarization based on the original context struggles to cope.

<table><tr><td></td><td rowspan=1 colspan=4>Summarization</td></tr><tr><td rowspan=4 colspan=5>Dozens injured, child dead as [6.1 - magnitude earthquake] hits In- [6.1-magnitude earthquake] refers to thedonesia&#x27;s Aceh Updated : July 02, 2013 15:50 IST A 6.1-magnitude  seismic event that occurred in the Benerearthquake which hit the Indonesian province of Aceh on Tuesday killed  Meriah district in the heart of Aceh, In-a child, injured dozens and destroyed buildings ... The quake struck  donesia, on July 2, 2013. The earthquakeinland at 0737 GMT at a depth of just 10 kilometres (6.2 miles) ... the  struck inland at 0737 GMT at a depth ofUS Geological Survey said . House collapsed ... 50 people with injuries  just 10 kilometres (6.2 miles) and was... 30 people seriously injured ... People panicked and rushed out of their  felt strongly for around 15 seconds, fromhomes ... In 2004 a massive tremor sparked a tsunami ...                  Bener Meriah to Banda Aceh.</td></tr><tr><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=2>(6.2 m</td><td rowspan=1 colspan=1>miles) ...</td></tr><tr><td rowspan=1 colspan=1>le with iniu</td></tr><tr><td rowspan=2 colspan=5>Indonesia&#x27;s West Papua province was hit by a magnitude 6.1 [earthquake]  [earthquake] refers to the magnitudetoday, the latest powerful tremor to shake the region where five people  6.1 earthquake that hit Indonesia&#x27;s Westwere killed and hundreds injured at the weekend when buildings were  Papua province on an unspecified date.destroyed. The quake struck off the coast at 7:48 a.m. local time, 75  The earthquake struck off the coast at 7:48kilometers (50 miles) ... the U.S. Geological Survey said ... At least  a.m. local time, 75 kilometers (50 miles)five people were killed, 250 others injured and more than 800 homes  west of the region&#x27;s main city of Manok-destroyed ... 14,000 people fled their homes ... temblor in 2004 caused a  wari, according to the U.S. Geological Sur-tsunami ...</td></tr><tr><td rowspan=1 colspan=3>vey.</td></tr></table>

Table 4: Two non-coreferential mentions for the event type “earthquake”, illustrating the remarkably similar contexts, as well as our generated more distinctive summaries. To better illustrate the similarity, we preserve the sentence containing the mention along with similar content from the context. Key information in our summarization is highlighted in bold. Mention spans are represented as [mention span].

![](images/aee19486f081899abcf445d3cf2dfe8592c4348ebbe819a2a5eb6e96c91c1174.jpg)  
Figure 2: LLM paraphrase comparison with $B ^ { 3 } \mathrm { F } 1$ . The vertical axis has a baseline starting from 60.

Overall, LLM summarization excels in consolidating information for specific events, facilitating the differentiation of similar yet non-coreferential events. Relatively, its effectiveness is limited for events with significant expression differences or those lacking essential details.

LLM Summarization VS LLM Paraphrase To validate that the performance improvement brought by our summarization is due to genuinely extracting crucial information rather than introducing diversity in context, we conduct a comparison with paraphrases generated by the LLM. We prompt the LLM to paraphrase the context of mentions instead of the sentences they belong to, and use the same hyper-parameters for fine-tuning the SLM. As shown in Figure 2, compared to our baseline, LLM paraphrase exhibits a slight improvement on GVC and FCC, with a more pronounced decline on ECB+. More importantly, it significantly lags behind our summarization on all datasets. This comparison demonstrates the capability of our summarization method in selecting and aggregating relevant information. The prompt for LLM paraphrase is provided in Table 11 (Appendix A.5).

## 4.4 Ablation Study on the Two-step Workflow

We conduct an ablation study to specifically illustrate the effect of Step 1 and Step 2 in LLM summarization (Table 1). As shown in Figure 3, both steps contribute to the overall improvement, with the second step being more pronounced, especially on the FCC dataset. This is attributed to the longer documents in FCC, with nearly double the number of sentences in each document compared to the other two datasets. This demonstrates that the information provided in Step 1 establishes a solid foundation but is relatively localized. Step 2, involving global information expansion, plays a crucial role in overall enhancement.

To examine the benefits of decomposed execution, we further integrate the two-step workflow into a single-step one through simple concatenation. Despite demonstrating comparable performance on GVC, the integrated workflow shows a noticeable lag, being 1.2% and 2% behind on ECB+ and FCC, respectively, in terms of $B ^ { 3 }$ F1. This indicates that even with straightforward instructions, decomposing the multi-objective task into multiple independent steps is necessary, as evidenced by the recent LLM agent studies (Aksitov et al., 2023).

![](images/19810830e0d36216b81d0de8639d1866fcd9e6d6f12e96a425b2d8c736dc3692.jpg)  
Figure 3: Comparison of different steps with $B ^ { 3 }$ F1. The vertical axis has a baseline starting from 65.

We perform error analysis and compare the lengths of the generated summaries to provide a detailed explanation of the impact of each step in the workflow and its decomposition. Further details can be found in Appendix B.4.

## 4.5 Analysis of GPT-4 Performance on CDECR

Table 5 presents the results of different in-context learning methods. It shows that GPT-4 achieves its optimal performance using few-shot learning with mention-inclusive sentences as context (Few-MIS), yet it only achieves results comparable to the lemma matching-based method. Table 6 further compares different types of errors. Compared to our baseline and our method, Few-MIS has a slight reduction in FPT errors but a significant increase in FPA and FN errors. This indicates that GPT-4 has limited ability to differentiate between similar but non-coreferential events based on arguments, and struggles to link coreferential events with significant narrative differences based on semantics. The reduction in FPT errors may also be attributed to its limited comprehension ability, thereby avoiding errors caused by excessive interpretation of event types. This aligns with our observation that GPT-4 relies on a simplistic approach of clustering based on the literal meaning of mentions without considering their contexts. Additionally, the role of demonstrations appears limited to expanding the scope of matching for synonymous mentions.

<table><tr><td colspan="2">Method</td><td>R</td><td>P</td><td>F1</td></tr><tr><td colspan="2">CLUSTER+LEMMA (Barhom et al., 2019) Our baseline</td><td>71.7 87.1</td><td>85.0 86.0</td><td>77.8</td></tr><tr><td colspan="2">Our method</td><td>89.1</td><td>86.5</td><td>86.5 87.8</td></tr><tr><td rowspan="2">Few-shot</td><td>Mention-inclusive sentences</td><td>76.3</td><td>78.1</td><td>77.2</td></tr><tr><td>Full context</td><td>65.6</td><td>77.2</td><td>70.9</td></tr><tr><td rowspan="2">Zero-shot</td><td>Mention-inclusive sentences</td><td>78.6</td><td>60.4</td><td>68.3</td></tr><tr><td>Full context</td><td>75.6</td><td>56.4</td><td>64.6</td></tr></table>

Table 5: Results on ECB+, based on the $B ^ { 3 }$ metric.

<table><tr><td>Methods</td><td>FPA</td><td>FPT</td><td>FN</td></tr><tr><td>Our baseline</td><td>1775</td><td>302</td><td>1262</td></tr><tr><td>Our method</td><td>1227</td><td>152</td><td>1087</td></tr><tr><td>Few-MIS</td><td>2272</td><td>116</td><td>3435</td></tr></table>

Table 6: Statistics of errors by different types. Few-MIS corresponds to the best-performing in-context learning method from Table 5, which is few-shot learning with mention-inclusive sentences as context.

From Table 5, it is also evident that incorporating the full context, compared to solely utilizing mention-inclusive sentences as context, results in a significant performance decline. This indicates that with more context, GPT-4 not only has limited ability to extract effective cues but also suffers from disrupted comprehension of the local context. Additionally, compared to few-shot learning, zero-shot learning demonstrates higher recall but significantly lower precision. This is because many completely unrelated mentions are clustered into a single cluster. This highlights the complexity of the CDECR task, indicating that GPT-4 struggles to perform basic clustering when relying solely on the task description.

We further investigate the impact of the number of in-context demonstrations on GPT-4’s performance. Details can be found in Appendix B.5.

## 5 Conclusion

We design generic tasks to leverage the potential of LLMs for CDECR, effectively bridging the gap between the general capabilities of LLMs and the complex annotation guidelines of specific IE tasks. Results show that by harnessing the inherent knowledge and comprehension abilities of LLMs to gain a deeper understanding of events, our collaborative approach can alleviate the challenge of SLMs for complex contextual understanding, ultimately enhancing performance.

## Limitations

The LLM we use for our collaborative approach is GPT-4-0613. Moving forward, we plan to assess the performance of additional LLMs, such as LLaMa (Touvron et al., 2023).

For CDECR, where internal information within the given document might be insufficient, there arises a need for external information retrieval. We are considering further leveraging the capabilities of LLMs to explore how to retrieve supplementary information from external corpora such as news articles. Our aim is to combine this additional information with the given documents to enhance performance.

## Ethics Statement

We adhere to the ACL Code of Ethics.

## Acknowledgement

We acknowledge the assistance of Pai Liu from University of Rochester in conducting experiments related to GPT-4. This work was supported by the STI 2030—Major Projects (Grant No. 2022ZD0208800) and the Alibaba Innovative Research (Grant No. 10313H022101).

## References

Shafiuddin Rehan Ahmed, Abhijnan Nath, James H. Martin, and Nikhil Krishnaswamy. 2023. 2 n is better than n<sup>2</sup>: Decomposing event coreference resolution into two tractable problems. In Findings of the Associationfor Computational Linguistics: ACL 2023, pages 1569–1583, Toronto, Canada. Association for Computational Linguistics.

Renat Aksitov, Sobhan Miryoosefi, Zonglin Li, Daliang Li, Sheila Babayan, Kavya Kopparapu, Zachary Fisher, Ruiqi Guo, Sushant Prakash, Pranesh Srinivasan, et al. 2023. Rest meets react: Selfimprovement for multi-step reasoning llm agent. arXiv preprint arXiv:2312.10003.

Emily Allaway, Shuai Wang, and Miguel Ballesteros. 2021. Sequential cross-document coreference resolution. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 4659–4671, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Yejin Bang, Samuel Cahyawijaya, Nayeon Lee, Wenliang Dai, Dan Su, Bryan Wilie, Holy Lovenia, Ziwei Ji, Tiezheng Yu, Willy Chung, et al. 2023. A multitask, multilingual, multimodal evaluation of chatgpt on reasoning, hallucination, and interactivity. arXiv preprint arXiv:2302.04023.

Shany Barhom, Vered Shwartz, Alon Eirew, Michael Bugert, Nils Reimers, and Ido Dagan. 2019. Revisiting joint modeling of cross-document entity and event coreference resolution. In Proceedings ofthe 57th Annual Meeting of the Association for Computational Linguistics, pages 4179–4189, Florence, Italy. Association for Computational Linguistics.

Cosmin Bejan and Sanda Harabagiu. 2010. Unsupervised event coreference resolution with rich linguistic features. In Proceedings of the 48th Annual Meeting ofthe Associationfor Computational Linguistics, pages 1412–1422, Uppsala, Sweden. Association for Computational Linguistics.

Michael Bugert, Nils Reimers, and Iryna Gurevych. 2021. Generalizing cross-document event coreference resolution across multiple corpora. Computational Linguistics, 47(3):575–614.

Avi Caciularu, Arman Cohan, Iz Beltagy, Matthew Peters, Arie Cattan, and Ido Dagan. 2021. CDLM: Cross-document language modeling. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 2648–2662, Punta Cana, Dominican Republic. Association for Computational Linguistics.

Arie Cattan, Alon Eirew, Gabriel Stanovsky, Mandar Joshi, and Ido Dagan. 2020. Streamlining crossdocument coreference resolution: Evaluation and modeling. arXiv preprint arXiv:2009.11032.

Arie Cattan, Alon Eirew, Gabriel Stanovsky, Mandar Joshi, and Ido Dagan. 2021a. Cross-document coreference resolution over predicted mentions. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 5100–5107, Online. Association for Computational Linguistics.

Arie Cattan, Alon Eirew, Gabriel Stanovsky, Mandar Joshi, and Ido Dagan. 2021b. Realistic evaluation principles for cross-document coreference resolution. In The 10th Conference on Lexical and Computational Semantics, page 143.

Xinyu Chen, Sheng Xu, Peifeng Li, and Qiaoming Zhu. 2023. Cross-document event coreference resolution on discourse structure. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4833–4843, Singapore. Association for Computational Linguistics.

Agata Cybulska and Piek Vossen. 2014. Using a sledgehammer to crack a nut? lexical diversity and event coreference resolution. In Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC’14), pages 4545–4552, Reykjavik, Iceland. European Language Resources Association (ELRA).

Bowen Ding, Qingkai Min, Shengkun Ma, Yingjie Li, Linyi Yang, and Yue Zhang. 2024. A rationalecentric counterfactual data augmentation method for

cross-document event coreference resolution. In Proceedings ofthe 2024 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies.

Jun Gao, Huan Zhao, Changlong Yu, and Ruifeng Xu. 2023. Exploring the feasibility of chatgpt for event extraction. arXiv preprint arXiv:2303.03836.

Ridong Han, Tao Peng, Chaohao Yang, Benyou Wang, Lu Liu, and Xiang Wan. 2023. Is information extraction solved by chatgpt? an analysis of performance, evaluation criteria, robustness and errors. arXiv preprint arXiv:2305.14450.

William Held, Dan Iter, and Dan Jurafsky. 2021. Focus on what matters: Applying discourse coherence theory to cross document coreference. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 1406–1417, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Benjamin Hsu and Graham Horwood. 2022. Contrastive representation learning for cross-document coreference resolution of events and entities. In Proceedings of the 2022 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 3644–3655, Seattle, United States. Association for Computational Linguistics.

Wenxiang Jiao, Wenxuan Wang, Jen-tse Huang, Xing Wang, and Zhaopeng Tu. 2023. Is chatgpt a good translator? a preliminary study. arXiv preprint arXiv:2301.08745.

Samuel Kriman and Heng Ji. 2021. Joint detection and coreference resolution of entities and events with document-level context aggregation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing: Student Research Workshop, pages 174–179, Online. Association for Computational Linguistics.

Heeyoung Lee, Marta Recasens, Angel Chang, Mihai Surdeanu, and Dan Jurafsky. 2012. Joint entity and event coreference resolution across documents. In Proceedings ofthe 2012 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning, pages 489–500, Jeju Island, Korea. Association for Computational Linguistics.

Bo Li, Gexiang Fang, Yang Yang, Quansen Wang, Wei Ye, Wen Zhao, and Shikun Zhang. 2023a. Evaluating chatgpt’s information extraction capabilities: An assessment of performance, explainability, calibration, and faithfulness. arXiv preprint arXiv:2304.11633.

Junpeng Li, Zixia Jia, and Zilong Zheng. 2023b. Semiautomatic data enhancement for document-level relation extraction with distant supervision from large language models. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language

Processing, pages 5495–5505, Singapore. Association for Computational Linguistics.

Mingchen Li and Rui Zhang. 2023. How far is language model from 100% few-shot named entity recognition in medical domain. arXiv preprint arXiv:2307.00186.

Chen Ling, Xujiang Zhao, Xuchao Zhang, Yanchi Liu, Wei Cheng, Haoyu Wang, Zhengzhang Chen, Takao Osaki, Katsushi Matsuda, Haifeng Chen, et al. 2023. Improving open information extraction with large language models: A study on demonstration uncertainty. arXiv preprint arXiv:2309.03433.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.

Jing Lu and Vincent Ng. 2018. Event coreference resolution: a survey of two decades of research. In Proceedings ofthe 27th International Joint Conference on Artificial Intelligence, pages 5479–5486.

Yaojie Lu, Qing Liu, Dai Dai, Xinyan Xiao, Hongyu Lin, Xianpei Han, Le Sun, and Hua Wu. 2022. Unified structure generation for universal information extraction. In Proceedings ofthe 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 5755–5772, Dublin, Ireland. Association for Computational Linguistics.

Yubo Ma, Yixin Cao, Yong Hong, and Aixin Sun. 2023. Large language model is not a good few-shot information extractor, but a good reranker for hard samples! In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 10572–10601, Singapore. Association for Computational Linguistics.

Hieu Minh Tran, Duy Phung, and Thien Huu Nguyen. 2021. Exploiting document structures and cluster consistencies for event coreference resolution. In Proceedings of the 59th Annual Meeting of the Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4840– 4850, Online. Association for Computational Linguistics.

Abhijnan Nath, Shadi Manafi, Avyakta Chelle, and Nikhil Krishnaswamy. 2024. Okay, let’s do this! modeling event coreference with generated rationales and knowledge distillation. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies.

Ji Qi, Chuchun Zhang, Xiaozhi Wang, Kaisheng Zeng, Jifan Yu, Jinxin Liu, Lei Hou, Juanzi Li, and Xu Bin. 2023. Preserving knowledge invariance: Rethinking robustness evaluation of open information extraction. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 5876–5890, Singapore. Association for Computational Linguistics.

Sahithya Ravi, Chris Tanner, Raymond Ng, and Vered Shwartz. 2023. What happens before and after: Multi-event commonsense in event coreference resolution. In Proceedings of the 17th Conference of the European Chapter ofthe Associationfor Computational Linguistics, pages 1708–1724, Dubrovnik, Croatia. Association for Computational Linguistics.

Oscar Sainz, Iker García-Ferrero, Rodrigo Agerri, Oier Lopez de Lacalle, German Rigau, and Eneko Agirre. 2023. Gollie: Annotation guidelines improve zero-shot information-extraction. arXiv preprint arXiv:2310.03668.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Piek Vossen and Agata Cybulska. 2018. Identity and granularity of events in text. In Computational Linguistics and Intelligent Text Processing: 17th International Conference, CICLing 2016, Konya, Turkey, April 3–9, 2016, Revised Selected Papers, Part II 17, pages 501–522. Springer.

Piek Vossen, Filip Ilievski, Marten Postma, and Roxane Segers. 2018. Don’t annotate, but validate: a data-to-text method for capturing event data. In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), Miyazaki, Japan. European Language Resources Association (ELRA).

Somin Wadhwa, Silvio Amir, and Byron Wallace. 2023. Revisiting relation extraction in the era of large language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15566– 15589, Toronto, Canada. Association for Computational Linguistics.

Zhen Wan, Fei Cheng, Zhuoyuan Mao, Qianying Liu, Haiyue Song, Jiwei Li, and Sadao Kurohashi. 2023. GPT-RE: In-context learning for relation extraction using large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 3534–3547, Singapore. Association for Computational Linguistics.

Xiao Wang, Weikang Zhou, Can Zu, Han Xia, Tianze Chen, Yuansen Zhang, Rui Zheng, Junjie Ye, Qi Zhang, Tao Gui, et al. 2023. Instructuie: Multitask instruction tuning for unified information extraction. arXiv preprint arXiv:2304.08085.

Xiang Wei, Xingyu Cui, Ning Cheng, Xiaobin Wang, Xin Zhang, Shen Huang, Pengjun Xie, Jinan Xu, Yufeng Chen, Meishan Zhang, et al. 2023. Zeroshot information extraction via chatting with chatgpt. arXiv preprint arXiv:2302.10205.

Tingyu Xie, Qi Li, Jian Zhang, Yan Zhang, Zuozhu Liu, and Hongwei Wang. 2023. Empirical study of

zero-shot NER with ChatGPT. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 7935–7956, Singapore. Association for Computational Linguistics.

Sheng Xu, Peifeng Li, and Qiaoming Zhu. 2022. Improving event coreference resolution using documentlevel and topic-level information. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 6765–6775, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Xin Xu, Yuqi Zhu, Xiaohan Wang, and Ningyu Zhang. 2023. How to unleash the power of large language models for few-shot relation extraction? In Proceedings of The Fourth Workshop on Simple and Efficient Natural Language Processing (SustaiNLP), pages 190–200, Toronto, Canada (Hybrid). Association for Computational Linguistics.

Bishan Yang, Claire Cardie, and Peter Frazier. 2015. A hierarchical distance-dependent Bayesian model for event coreference resolution. Transactions ofthe Association for Computational Linguistics, 3:517– 528.

Xiaodong Yu, Wenpeng Yin, and Dan Roth. 2022. Pairwise representation learning for event coreference. In Proceedings ofthe 11th Joint Conference on Lexical and Computational Semantics, pages 69–78, Seattle, Washington. Association for Computational Linguistics.

Chenhan Yuan, Qianqian Xie, and Sophia Ananiadou. 2023. Zero-shot temporal relation extraction with ChatGPT. In The 22nd Workshop on Biomedical Natural Language Processing and BioNLP Shared Tasks, pages 92–102, Toronto, Canada. Association for Computational Linguistics.

Yutao Zeng, Xiaolong Jin, Saiping Guan, Jiafeng Guo, and Xueqi Cheng. 2020. Event coreference resolution with their paraphrases and argument-aware embeddings. In Proceedings ofthe 28th International Conference on Computational Linguistics, pages 3084–3094, Barcelona, Spain (Online). International Committee on Computational Linguistics.

Wenxuan Zhou, Sheng Zhang, Yu Gu, Muhao Chen, and Hoifung Poon. 2023. Universalner: Targeted distillation from large language models for open named entity recognition. arXiv preprint arXiv:2308.03279.

<table><tr><td></td><td>ECB+</td><td>GVC</td><td>FCC</td></tr><tr><td>Documents</td><td>982</td><td>510</td><td>451</td></tr><tr><td>Sentences</td><td>16314</td><td>9782</td><td>14940</td></tr><tr><td>Event mentions</td><td>6833</td><td>7298</td><td>3563</td></tr><tr><td>Event clusters</td><td>2741</td><td>1411</td><td>469</td></tr><tr><td>Event coref links</td><td>26712</td><td>29398</td><td>145272</td></tr></table>

Table 7: Statistics of each dataset.
<table><tr><td></td><td>Candidate retrieval</td><td>Pairwise classification</td></tr><tr><td>Learning rate</td><td>1e-5</td><td>6e-6</td></tr><tr><td>Batch size</td><td>16</td><td>16</td></tr><tr><td>Epochs</td><td>50</td><td>20</td></tr><tr><td>Early stop patience</td><td>10</td><td>5</td></tr><tr><td>Train neighbor size</td><td>-</td><td>20</td></tr><tr><td>Eval neighbor size</td><td>10</td><td>10</td></tr></table>

Table 8: Hyper-parameters for fine-tuning the SLMbased modules.

## A Implementation details

## A.1 Dataset Statistics

As shown in Table 7.

## A.2 SLM Fine-tuning Hyper-parameters As shown in Table 8.

## A.3 Prompt Design and Model Details of GPT-4 Evaluation

The prompt is shown in Table 9. For ECB+, we introduce only one randomly selected topic from the training data as the demonstration, which comprises 39 documents, accounting for 6.6% of the entire training set. For GVC and FCC, we use the same number of randomly selected documents as in ECB+ for demonstration. It is important to note that we have conducted multiple rounds of prompt optimization to ensure GPT-4’s performance, including:

• Designing a reasonable format to tag each mention in the document with a unique mention\_id to avoid literal confusion.

• Designing the output format as mention\_id: cluster\_index instead of cluster\_index: [mention\_id1, ..., mention\_idn] to ensure that no mention is omitted.

• Avoiding declaring specific conditions for event coreference in the task description, including coreferential participants, locations, and times. It is demonstrated that these conditions do not improve performance; instead, they lead GPT-4 to make coreference judgments based solely on individual conditions.

We set the model parameters, including seed and temperature, to 0 to minimize randomness. Additionally, we specify the output format to be in JSON for better post-processing.

During our experimentation, there were changes in the GPT-4 model provided by OpenAI. The introduction of “GPT-4-turbo-preview”, which can handle longer texts compared to “GPT-4-0613”, offers conditions for lenghy context composed by multiple documents (although it still faces length limitations in our actual testing). Consequently, our direct evaluation of the GPT-4 model was moved to “GPT-4-turbo-preview”.

Since most of our summary-based experiments were completed on “GPT-4-0613”, we did not migrate our experiments to “GPT-4-turbo-preview” due to cost considerations. Additionally, based on our observation with minimal use cases and external leaderboard<sup>9</sup>, “GPT-4-turbo-preview” (currently pointing to “GPT-4-0125-preview”) exhibits performance that is not inferior to “GPT-4-0613”.

## A.4 Event Type Categorization

To categorize event types, we establish a three-layer hierarchical structure of (mention->cluster->type), linking types between mentions. Specifically, if there are synonymous mentions between any two clusters, they belong to the same event type, and all mentions within the clusters belong to a synonymous event type. Drawing inspiration from (Ahmed et al., 2023), we determine mention synonymity by matching their span words. Table 10 illustrates that the contents of FCC and GVC belong to the same topic, resulting in a concentrated set of event types. Conversely, ECB+ involves various topics such as quake, murder, acquisition, etc., leading to a diverse set of event types.

## A.5 LLM Paraphrase Prompt

As shown in Table 11.

## B Experimental Results and Analysis

## B.1 API Efficiency and Truncation Issues

Compared to directly prompting GPT-4 for structured predictions of event coreference, our two-step prompting for summarizing each document’s mentions does incur more API calls and token consumption, as shown in Table 12. The primary additional overhead comes from generating more natural summaries for each mention rather than a final cluster label, which is the core of our collaborative approach.

You are a helpful assistant tasked with clustering coreferential event mentions in the provided documents.   
The event mentions in the documents are marked as follows: [mention string](mention id). Please output   
System role   
the result in JSON format without whitespace. In the JSON structure, each ‘mention id’ is assigned   
a ‘cluster id’.   
You can learn from the following example:   
Input:   
Document: [... [mention](mention\_id) ...]   
Prompt ...   
Output: [mention\_id: cluster\_id, ...]   
Now the following is your task:   
Document: [... [mention](mention\_id) ...]   
...  
Table 9: The few-shot prompt for GPT-4 evaluation. The system role is used to declare task requirements and output specifications. The prompt is divided into two sections: initially, a demonstration, followed by data to be processed. For zero-shot, it suffices to remove the demonstration part.

<table><tr><td></td><td>Mentions Clusters</td><td></td><td>Types</td></tr><tr><td>ECB+</td><td>1780</td><td>805</td><td>405</td></tr><tr><td>GVC</td><td>1008</td><td>194</td><td>4</td></tr><tr><td>FCC</td><td>1074</td><td>167</td><td>19</td></tr></table>

Table 10: Statistics for mention, cluster, and event type in the test set.

Based on our approach, we also strive to enhance the efficiency of GPT-4 utilization, including:

• Processing all mentions within the same document simultaneously: this avoids assigning a separate document input for each mention, thereby reducing the number of API calls and token consumption, thus improving efficiency. To ensure the accuracy of parallel processing, we employ a concise pre-step (e.g., dependency parsing) integrated into step 1, as described in Section 4.1.

• We strive to summarize event mentions through designing concise prompts, thereby avoiding the additional comsumption of complex inference chains and in-context learning methods.

• Some recent work aims to improve the performance of LLMs by having them generate complex reasoning logic, as mentioned in the part of Integration of LLM and SLM in related work. This approach typically involves dealing with a large number of combinations of mention pairs . In comparison, our collaborative approach only requires processing each document’s mentions once, thus offering a relative efficiency advantage while enhancing performance.

We can also illustrate the truncation issue through the statistics in Table 12. It shows that processing all test data from ECB+ consumed over 166k tokens for input and 25k tokens for output. With GPT-4’s output limited to 4096 tokens per instance, processing all test data in one go would allow us to get results for only 15% of the total mentions. The similar issue primarily results in poor performance on the GVC and FCC datasets. In the future, we will explore ways to address the length issues caused by multi-document scenarios, possibly through multiple processing iterations.

## B.2 Evaluation Under the Conditions of Without Singletons and at the Topic Level

The experimental results under the condition of with/without singletons are presented in Table 13. The results demonstrate that our method achieves state-of-the-art performance, surpassing Chen et al. (2023) by 3.5% and our baseline by 2.7% in CoNLL F1 under the without singletons condition. Additionally, our method demonstrates a relatively smaller performance gap between with and without singletons compared to Chen et al. (2023) (6.8% vs 9.3%) and our baseline (6.8% vs 8.0%). This further emphasizes the effectiveness of our method.

For topic level evaluation, it advocates not using subtopic-level document clustering, forcing models to confront the lexical ambiguity challenge. Our method, based on a baseline that performs can-

News: [input document]   
Question 1: In this news, given “[mention 1]” mentioned in the sentence “[the sentence]”.   
Concatenate the preceding five sentences of the current sentence (ignore if not available), the current sentence, and the subsequent five sentences of the current sentence (ignore if not available) into a single paragraph. Then, paraphrase the concatenated paragraph while preserving the mention [mention 1]. Attempt to express the information differently while maintaining the meaning and key information. Ensure that the mention [mention 1] is preserved and marked as #[mention 1]# in the paraphrased result. Limit the paraphrased result to three sentences. Present the information in the following format: ‘Paraphrase: <placeholder>’.   
Question 2: ...

Table 11: The prompt for LLM paraphrase. Each prompt includes a document along with multiple event mentions. The content to be filled is represented as [content].
<table><tr><td></td><td>Input tokens</td><td>Output tokens</td><td>API calls</td></tr><tr><td>Directly prompting</td><td>166k+</td><td>25k+</td><td>10</td></tr><tr><td>Our two-step prompting</td><td>658k+</td><td>213k+</td><td>400+</td></tr></table>

Table 12: Statistics of token consumption and API calls on ECB+ test set.

didate coreferential mention retrieval at a global level, avoids leveraging topic structure information and achieves better results than subtopic clustering methods. Therefore, we do not perform additional comparisons at the topic level.

## B.3 False Negative Cases

Given the context where mentions of the same event can vary greatly in expression styles, we provide an illustrative example in Table 14.

In cases where event mentions naturally lack sufficient details, we illustrate this phenomenon through Table 15, which presents two mentions of the same earthquake. The context for the first mention contains essential information such as time, location, magnitude, casualties, etc. In contrast, the context for the second mention primarily describes the subjective experiences of the individuals involved, lacking details related to the event itself. Despite our summarization extracting key information from the original context, it encounters difficulties in supporting coreference judgments.

## B.4 Two-step Workflow Analysis

Error Analysis We conduct error analysis for the workflow with only Step 1, the complete two-step workflow (Step 2), and the integrated single-step workflow.

As shown in Figure 4, Step 1 exhibits a significant reduction in FPA errors across all three datasets, indicating its effectiveness in extracting tailored information. However, an increase in FN errors is observed across all three datasets, suggesting that while Step 1 provides sufficiently distinctive information, it lacks the details needed to link mentions of the same event. This issue was notably addressed by the introduction of Step 2, resulting in a substantial decrease in FN errors across all datasets. FPA errors are also largely maintained at the level achieved in Step 1, leading to a significant improvement in coreference results. This emphasizes the indispensable roles of both Step 1 and Step 2 in the final outcomes. In Table 16, we provide examples to compare summaries generated by Step 1 and Step 2.

Compared to the two-step workflow, the integrated single-step workflow shows differing degrees of increase in both FPA and FN errors, further underscoring the necessity of decomposed execution.

Summarization length comparison We further compare the lengths of summaries generated in Step 1 and Step 2. As illustrated by the green line in Figure 5, it is evident that Step 2, building upon Step 1, results in approximately double the length. The additional detailed content contributes to the reduction of FN errors, effectively linking mentions of the same event. Furthermore, as indicated by the red and blue lines, our generated summaries remain within approximately 20% of the original document starting from a document length of 200 words. Moreover, with the increase in document length, this proportion further diminishes. This reflects the conciseness our summarization.

## B.5 The Impact of the Number of In-Context Demonstrations on GPT-4 Performance on CDECR

We test the peak performance by increasing the number of documents for demonstration. The results are shown in Figure 6, and it can be observed that:

• Under the condition of utilizing only mentioninclusive sentences as context, with the introduction of more documents (even exceeding

<table><tr><td rowspan="2">Methods</td><td rowspan="2"></td><td colspan="3">MUC</td><td colspan="3">B3</td><td colspan="3">CEAFe</td><td colspan="2">CoNLL</td><td colspan="2">LEA</td></tr><tr><td>R</td><td>P</td><td>F1</td><td>R</td><td>P</td><td>F1</td><td>R</td><td>P</td><td>F1</td><td>F1</td><td>R</td><td>P</td><td>F1</td></tr><tr><td rowspan="2">Cattan et al. (2021b)</td><td>singleton+</td><td>85.1</td><td>81.9</td><td>83.5</td><td>82.1</td><td>82.7</td><td>82.4</td><td>75.2</td><td>78.9</td><td>77.0</td><td>81.0</td><td></td><td></td><td></td></tr><tr><td>singleton-</td><td>85.1</td><td>81.9</td><td>83.5</td><td>70.8</td><td>70.2</td><td>70.5</td><td>68.2</td><td>52.3</td><td>59.2</td><td>71.1</td><td></td><td></td><td></td></tr><tr><td rowspan="2">Chen et al. (2023)</td><td>singleton+</td><td>88.6</td><td>85.9</td><td>87.2</td><td>87.8</td><td>85.4</td><td>86.6</td><td>82.8</td><td>83.7</td><td>83.2</td><td>85.7</td><td>-</td><td>-</td><td>-</td></tr><tr><td>singleton-</td><td>88.6</td><td>85.9</td><td>87.2</td><td>76.1</td><td>74.5</td><td>75.3</td><td>76.9</td><td>57.4</td><td>65.7</td><td>76.4</td><td></td><td></td><td></td></tr><tr><td rowspan="2">Our baseline</td><td>singleton+</td><td>86.6</td><td>86.8</td><td>86.7</td><td>87.1</td><td>86.0</td><td>86.5</td><td>82.6</td><td>82.5</td><td>82.5</td><td>85.2</td><td>77.8</td><td>76.6</td><td>77.2</td></tr><tr><td>singleton-</td><td>86.6</td><td>86.8</td><td>86.7</td><td>80.9</td><td>77.0</td><td>78.9</td><td>69.5</td><td>62.9</td><td>66.0</td><td>77.2</td><td>77.1</td><td>71.2</td><td>74.0</td></tr><tr><td rowspan="2">Our method</td><td>singleton+</td><td>89.4</td><td>87.1</td><td>88.2</td><td>89.1</td><td>86.5</td><td>87.8</td><td>82.7</td><td>85.5</td><td>84.1</td><td>86.7</td><td>79.7</td><td>78.5</td><td>79.3</td></tr><tr><td>singleton-</td><td>89.4</td><td>87.1</td><td>88.2</td><td>84.0</td><td>79.9</td><td>81.9</td><td>75.3</td><td>64.9</td><td>69.7</td><td>79.9</td><td>80.9</td><td>73.9</td><td>77.2</td></tr></table>

Table 13: Performance comparison on the ECB+ dataset with(singletons+)/without(singletons-) singletons. We are the first to present results under the LEA metric.

![](images/8da38d3fc71affd7e1d84ea04bd2c90c1f65b0d181668924cfe301c89dd7ce84.jpg)

![](images/01a804436b0cffe41d52601da492ef194844eddc0bb79ad8e24542e87ab9edfc.jpg)

![](images/ff87b8fbd184048b21107ea1dc8c341fee5625c3f3ee0dc0a2c8161e755e81b7.jpg)  
Figure 4: FPA and FN error comparison. Due to the rarity of FPT-type errors, we have omitted them in the figures for better clarity in presentation. Step 2 is built upon Step 1, and the integrated involves merging the two steps together.  
Event Smith case as the incarnation of the Doctor was handed the keys to the Tardis Mention winning the role of the 11th Doctor expressions stepping into Doctor Who’s title role

Table 14: Variations in mention expressions for identical event.

the quantity in the test set), there is still no significant improvement in the performance of GPT-4. And there remains a considerable gap compared to the F1 score of our method (77.2% vs 87.8%).

• Under the condition of utilizing full context, an increase in the number of documents can even degrade performance. Since the complete context is crucial for event coreference resolution, it indicates that understanding and utilizing more context is a significant bottleneck limiting the performance of GPT-4.

<table><tr><td>Context Summarization [4.6 earthquake] shakes Northern California March 14, 2013 |</td></tr><tr><td>[4.6 earthquake] refers to the seismic event that 8:16 am A magnitude 4.6 earthquake struck Northern California occurred in Northern California on March 14, on Thursday morning, The temblor struck about 26 miles north of 2013, with a magnitude of 4.6 on the Richter scale. Santa Rosa in the Geysers area. There were no reports of damage The earthquake was felt over a wide area of the or injuries, but there were some apparent aftershocks. It was felt region, including in Santa Rosa, Marin County, over a wide area of the region, including in Santa Rosa, Marin County and Vallejo, according to the U.S. Geological Survey. The &quot;Do You Feel It?&quot; survey by the USGS showed people as far south a San Francisco felt shaking.</td></tr><tr><td>and Vallejo, and caused some apparent aftershocks. However, there were no reports of damage or in- juries. Good sized quake at the California Geysers I felt this one pretty [quake] refers to a seismic event that occurred at the</td></tr></table>

Table 15: Two coreferential mentions referring to the same earthquake, where the second provides minimal coreference evidence. Key information in our summarization is highlighted in bold. Mention spans are represented as [mention span].

<table><tr><td>Step 1</td><td>Step 2</td></tr><tr><td>[6.1-magnitude earthquake] refers to the seismic event that occurred in Aceh, Indonesia, with a magni- tude of 6.1 on the Richter scale.</td><td>[6.1-magnitude earthquake] refers to the seismic event that occurred in the Bener Meriah district in the heart of Aceh, Indonesia, on July 2, 2013. The earthquake struck inland at 0737 GMT at a depth of just 10 kilometres (6.2 miles) and was felt strongly for around 15 seconds, from Bener Meriah to Banda Aceh.</td></tr><tr><td>[earthquake] refers to the magni- tude 6.1 earthquake that hit Indone- sia&#x27;s West Papua province.</td><td>[earthquake] refers to the magnitude 6.1 earthquake that hit Indonesia&#x27;s West Papua province on an unspecified date. The earthquake struck off the coast at 7:48 a.m. local time, 75 kilometers (50 miles) west of the region&#x27;s main city of Manokwari, according to the U.S. Geological Survey.</td></tr></table>

Table 16: Comparison of summaries generated by Step 1 and Step 2. Step 2 is built upon Step 1. Key information for distinguishing in Step 2 is highlighted in bold. Mention spans are represented as [mention span].

![](images/a29c7f0d751adbca52379f03e8916561d070de91b6c9301de7166697089835c0.jpg)  
Figure 5: Summarization length comparison. Step 2 is built upon Step 1. The vertical axis represents the ratio of content word count. The horizontal axis represents the number of words in the content, scaled by a factor of 100.

![](images/cdd1d8f3131a054811b089a59904f9288e01070ee5882dbd83e24ad3f872f9f2.jpg)  
Figure 6: The impact of number of demonstrations on GPT-4 performance, measured by controlling the number of documents used. In our main experiments evaluating GPT-4, we utilize one instance of demonstration comprising 39 documents.