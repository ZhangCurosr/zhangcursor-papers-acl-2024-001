# M-RAG: Reinforcing Large Language Model Performance through Retrieval-Augmented Generation with Multiple Partitions

Zheng Wang<sup>1</sup>, Shu Xian Teo<sup>1</sup>, Jieer Ouyang<sup>1</sup>, Yongjun Xu<sup>1</sup>, Wei Shi<sup>1</sup>

<sup>1</sup>Huawei Technologies, Co., Ltd.

{wangzheng155,teo.shu.xian,ouyang.jieer,xuyongjun6,w.shi}@huawei.com

## Abstract

Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by retrieving relevant memories from an external database. However, existing RAG methods typically organize all memories in a whole database, potentially limiting focus on crucial memories and introducing noise. In this paper, we introduce a multiple partition paradigm for RAG (called M-RAG), where each database partition serves as a basic unit for RAG execution. Based on this paradigm, we propose a novel framework that leverages LLMs with Multi-Agent Reinforcement Learning to optimize different language generation tasks explicitly. Through comprehensive experiments conducted on seven datasets, spanning three language generation tasks and involving three distinct language model architectures, we confirm that M-RAG consistently outperforms various baseline methods, achieving improvements of 11%, 8%, and 12% for text summarization, machine translation, and dialogue generation, respectively.

## 1 Introduction

Introduced by (Lewis et al., 2020), Retrieval-Augmented Generation (RAG) represents a paradigm within the domain of Large Language Models (LLMs) to augment generative tasks. More specifically, RAG incorporates an initial retrieval step where LLMs query an external database to acquire relevant information before progressing to answer questions or generate text. This process not only guides the subsequent generation step but also guarantees that the responses are firmly anchored in the retrieved information (referred to as memories). Consequently, it enhances LLM performance, and has attracted growing research interests (Gao et al., 2023) in recent years.

While the majority of existing studies (Asai et al., 2023; Cheng et al., 2023b; Ma et al., 2023) adopt a retrieval approach that considers a database as a whole, which tends to yield a coarse-grained retrieval. The collective organization of all memories may hinder the focus on crucial memories and introduce noise, particularly due to the inherent challenges of Approximate k-Nearest Neighbor (AKNN) search when applied to large datasets. In this context, we investigate a retrieval approach that aims to search within a partition of the database, corresponding retrieval at a fine-grained level, which is designed to enhance the generation process by targeting specific memories. Moreover, in quite a few vector database systems, database partitions are regarded as fundamental units for analysis. This facilitates the construction and maintenance of index structures (Pan et al., 2023), ensures the protection of user privacy data (stored in specific partitions with access rights) (Xue et al., 2017), and supports distributed architectures (Guo et al., 2022). Therefore, in this work, we propose to take a partition as a basic entity in the execution of RAG, which is less explored in current methods.

We discuss our proposal with a motivating experiment illustrated in Figure 1. We investigate various strategies for partitioning a database (elaborated in Section 3.1), and perform RAG with varying the number of partitions for three generation tasks: summarization, translation, and dialogue generation, where we explore all partitions for the retrieval, and the best result (assessed based on a development set) across different partitions is reported. We observe that the optimal performance is typically not achieved through retrieval based on the entire database (#Partitions = 1). This observation inspires us to investigate a novel RAG setting with multiple partitions. To achieve this, the task should address three significant challenges, summarized below. (1) Determining a strategy for partitioning a database and the number of partitions. (2) Developing a method for selecting a suitable partition for a given input query to discover effective memories. (3) Enhancing memory quality, including inherent issues such as hallucination, or irrelevant context, which can impact the grounding of LLM generation.

![](images/0891a5a4820ba3d18ee92b05711aa4700fd4bbe8605869663ba3fd89146bb79c.jpg)  
(a) Summ. (ROUGE-1)

![](images/fd18e6e6bdb899dddc45e6d0cd104f3aa3310de67c0bdbdd41f738d9af30cd8a.jpg)  
(b) Summ. (Runtime)

![](images/0e6672ce31157599ba9dfde6d50c66bd318a3761796feff658b07601ff1f3a06.jpg)  
(c) Machine translation

![](images/a8685c46a39e45ef2066c067d5712cb56287e206bf7aefb93c185f29c7796caa.jpg)  
(d) Dialogue generation  
Figure 1: Comparison with database partitioning strategies for language generation tasks.

Building upon the aforementioned discussion, we introduce a new solution called M-RAG, designed to facilitate RAG across multiple partitions of a database. M-RAG addresses all of the three challenges. For (1), we draw insights from the literature on vector database management (Pan et al., 2023; Han et al., 2023) and assess various strategies, namely Randomization (Indyk and Motwani, 1998), Clustering (Jegou et al., 2010), Indexing (Malkov et al., 2014; Malkov and Yashunin, 2018), and Category (Gollapudi et al., 2023), through empirical studies. The effectiveness of these strategies, along with the corresponding number of partitions, is evaluated across different generative tasks on a development set in our experiments. For (2), with multiple partitions at play, we formulate partition selection as a multi-armed bandit problem (Slivkins et al., 2019). In this context, an agent, denoted as Agent-S, iteratively selects one among several partitions. The characteristics of each partition are only partially known at the time of selection, and Agent-S gains a better understanding over time by maximizing cumulative rewards in the environment. To optimize the decision policy, we leverage reinforcement learning with a carefully designed Markov Decision Process (MDP). For (3), after selecting a partition and obtaining memories for generation, we introduce another agent, denoted as Agent-R. This agent generates a pool of candidate memories iteratively through the use of LLMs. Once a candidate is selected, Agent-R evaluates its quality by demonstrating it to generate a hypothesis. The identification of a high-quality hypothesis determined by a specific performance metric, triggers a boosting process, where it signals the exploration and replacement of the previous memory with a superior one, and continues the process. Further, we integrate the efforts of Agent-S and Agent-R through multi-agent reinforcement learning. With a shared objective of enhancing text generation for a given input query, they are jointly optimized through end-to-end training.

Our contributions can be summarized as follows: (1) we propose a multiple partition paradigm for RAG, aiming to facilitate fine-grained retrieval and concentrate on pivotal memories to enhance overall performance. In addition, the utilization of multiple partitions benefits other aspects of RAG, including facilitating the construction and maintenance of indices, protecting user privacy data within specific partitions, and supporting distributed parallel processing across different partitions. (2) We introduce M-RAG, a new solution based on multiagent reinforcement learning that tackles the three challenges in executing RAG across multiple partitions. We show that the training objective of M-RAG is well aligned with that of text generation tasks. (3) We conduct extensive experiments on seven datasets for three generation tasks on three distinct language model architectures, including a recent Mixture of Experts (MoE) architecture (Jiang et al., 2024). The results demonstrate the effectiveness of M-RAG across diverse RAG baselines. In comparison to the best baseline approach, M-RAG exhibits improvements of 11%, 8%, and 12% for text summarization, machine translation, and dialogue generation tasks, respectively.

## 2 Related Work

Retrieval-Augmented Generation. We review the literature of Retrieval-Augmented Generation (RAG) in terms of (1) Naive RAG, (2) Advanced RAG, and (3) Modular RAG. For (1), Naive RAG follows a standard process including indexing, retrieval, and generation (Ma et al., 2023). However, its quality faces significant challenges such as low precision, hallucination, and redundancy during the process. For (2), Advanced RAG is further developed to overcome the shortcomings of Naive RAG. Specifically, during the indexing stage, the objective is to enhance the quality of the indexed content by optimizing data embedding (Li et al.,

2023). During the retrieval stage, the focus is on identifying the appropriate context by calculating the similarity between the query and chunks, where the techniques involve fine-tuning embedding models (Xiao et al., 2023), or learning dynamic embeddings for different context (Karpukhin et al., 2020). During the generation stage, it merges the retrieved context with the query as an input into large language models (LLMs), where it ad dresses challenges posed by context window limits with re-ranking the most relevant content (Jiang et al., 2023b; Zhuang et al., 2023), or compressing prompts (Litman et al., 2020; Xu et al., 2023). In addition, Self-RAG (Asai et al., 2023) is proposed to identify whether retrieval is necessary, or the retrieved context is relevant, which helps language models to produce meaningful generation (Asai et al., 2023). For (3), Modular RAG diverges from the traditional Naive RAG structure by incorporating external modules to further enhance the performance, including search module (Wang et al., 2023a), memory module (Wang et al., 2022; Cheng et al., 2023b), tuning module (Lin et al., 2023), and task adapter (Cheng et al., 2023a; Dai et al., 2023). Specifically, Selfmem (Cheng et al., 2023b) incorporates a retrieval-enhanced generator to it eratively create a memory pool, it then trains a selector to choose one of the memories from the pool to generate responses. The work (Gao et al., 2023) provides a comprehensive survey of RAG for LLMs. Our work differs from existing RAG studies in two aspects. First, we introduce a multiple partition setting, where each partition serves as a fundamental entity for retrieval, rather than retrieving from the entire database. Second, we introduce an M-RAG framework built upon multi agent reinforcement learning, which tackles three distinct challenges posed by this novel setting.

Reinforcement Learning for LLMs. Recently, reinforcement learning has seen broad applications across a variety of language-related tasks for Large Language Models (LLMs). This includes tasks such as text summarization (Wu et al., 2021a), machine translation (Kreutzer et al., 2018), dialogue systems (Jaques et al., 2019; Yi et al., 2019), semantic parsing (Lawrence and Riezler, 2018), and review generation (Cho et al., 2018). For example, WebGPT (Nakano et al., 2021) incorporates a reinforcement learning framework to autonomously train the GPT-3 model using a search engine during the text generation process. Further,

InstructGPT (Ouyang et al., 2022) collects a dataset containing desired model outputs provided by human labelers. Subsequently, it employs Reinforcement Learning from Human Feedback (RLHF) to fine-tune GPT-3 (Brown et al., 2020). In addition, R3 (Ma et al., 2023) introduces a Rewrite-Retrieve-Read process, where the LLM performance serves as a reinforcement learning incentive for a rewriting module. This approach empowers the rewriter to enhance retrieval queries, consequently improving the reader’s performance in downstream tasks. MMQS (Wang et al., 2024) introduces a new multimodal question suggestion task with a multi-agent version of RLHF. In this work, we propose a novel multi-agent reinforcement learning framework utilizing two agents to collaboratively optimize text generation tasks. To our best knowledge, this is the first of its kind.

Multi-source Knowledge-grounded Dialogue System (MKDS). We review the literature on MKDS (Wu et al., 2021b, 2022), and highlight differences with our M-RAG regarding (1) datasets, (2) solutions, and (3) tasks. For (1), MKDS uses multi-source heterogeneous data (plain text, tables, knowledge graphs), each contributing uniquely to dialogue generation. M-RAG uses a single-source homogeneous dataset, initially vectorized and indexed for RAG retrieval. We explore partitioning strategies to create multiple homogeneous partitions for effective retrieval. For (2), MKDS employs an encoder-decoder framework with varied attention weights for different knowledge sources, trained with a small dialogue model like MSKE-Dialog (59.14M parameters) (Wu et al., 2021b). M-RAG uses a Retrieval-then-Generation approach with two RL agents (Agent-S and Agent-R) focusing on retrieval and generation, respectively. For (3), M-RAG leverages LLMs for diverse language generation tasks, including text summarization, machine translation, and dialogue generation, unlike MKDS’s specific focus on dialogue generation (Wu et al., 2021b, 2022).

## 3 Methodology

A task involving M-RAG can be formulated below. Given a database $\mathbb { D } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { | \mathbb { D } | }$ for a language generation task (e.g., summarization), where each pair (x, y) represents a document and its corresponding summary stored in D. The M-RAG initiates the process by partitioning D into multiple partitions. This can be achieved through methods like clustering or by leveraging inherent category labels in the data. The resulting partitions are denoted as $\mathbb { D } = \{ D _ { m } \} _ { m = 1 } ^ { | M | }$ , where each $D _ { m }$ $( 1 ~ \leq ~ m ~ \leq ~ M )$ supports an independent RAG process (Section 3.1). The M-RAG framework comprises both training and inference processes, as outlined in Algorithm 1. For training, Agent-S learns to select a specific $D _ { m }$ for an input text pair (Section 3.2). Subsequently, Agent-R refines the retrieved memories, represented as $( \tilde { x } , \tilde { y } ) \in D _ { m } .$ within the selected partition $D _ { m }$ (Section 3.3). Finally, the two agents are collaboratively trained with multi-agent reinforcement learning (see Section 3.4). Figure 2 illustrates the training process of M-RAG. For inference, the refined D is utilized to support a LLM in generating hypotheses, where a $D _ { m }$ is selected by the trained Agent-S.

![](images/36327146b369d952458180cb1998c08bddb22879415a63ccfd2242ae8d5637c3.jpg)  
Figure 2: Illustration of M-RAG training in a summarization task: The M-RAG initiates training with multiple partitions (Section 3.1), it then selects a partition to perform retrieval via Agent-S (Section 3.2), and refines the memories within the selected partition via Agent-R (Section 3.3). Both agents are collaboratively trained to enhance generation capabilities through multi-agent reinforcement learning (Section 3.4). For inference, it includes elements (1), (2), (3), (4), (11), and (12).

## 3.1 Discussion on Partitioning a Database

As M-RAG relies on multiple partitions for RAG operations, we investigate various strategies to partition an external database (typically the training corpus). The results of these strategies are then validated through empirical studies. We review the literature, including recent vector database surveys (Pan et al., 2023; Han et al., 2023), and identify the following strategies: namely (1) Randomization (Indyk and Motwani, 1998), (2) Clustering (Jegou et al., 2010), (3) Indexing (Malkov et al., 2014; Malkov and Yashunin, 2018) and (4) Category (Gollapudi et al., 2023). Specifically, for (1), it targets the utilization of probability amplification techniques, such as locality-sensitive hashing (LSH), to hash similar items (data vectors) into the same bucket with a high probability. For (2), it involves clustering data vectors using K-means, where this clustering concept is widely applied in

Inverted File Index (IVF) for tasks like Approximate k-Nearest Neighbor (AKNN) search. For (3), navigable graph indexes, such as HNSW (Malkov and Yashunin, 2018) or NSW (Malkov et al., 2014), are designed to facilitate easy traversal of different regions within a vector database. To achieve effective partitions, we employ graph partitioning with spectral clustering on a navigable graph. For (4), it involves assigning data vectors to partitions based on their respective categories. For example, in the DailyDialog dataset (Li et al., 2017), which includes 7 emotion categories (e.g., joy, anger) and 10 topic categories (e.g., work, health), vectors are partitioned according to their category labels. We note that a single vector may be assigned to multiple partitions, due to the characteristics of the dataset, where a dialogue spans multiple categories.

In Figure 1, we perform experiments on a development set, manipulating the number of partitions wrt the 4 strategies across three language generation tasks (summarization, translation, and dialogue generation). The results demonstrate the effectiveness of the strategies, and we conclude the selected strategies with the number of partitions as follows. We choose Indexing (4 partitions), Randomization (3 partitions), and Category (10 partitions) for the summarization, translation, and dialogue generation tasks, respectively. In addition, as shown in Figure 1 (a) and (b), we observe that both Top-1 and Top-3 retrieval methods exhibit comparable performance. For enhanced efficiency, we default to Top-1 retrieval in the rest of the paper.

## 3.2 Agent-S: Selecting a Database Partition

During the training process of an Agent-S to select a partition from D, the environment is naturally modeled as a bandit setting. In this context, when a random partition is selected, the language model generates a response for the query with feedback (typically based on a specific performance metric), and concludes the episode. The selection process can be formulated as a Markov Decision Process (MDP), involving states, actions, and rewards.

States. Given a training pair $( x , y )$ and a set of database partitions $\mathbb { D } = \{ D _ { m } \} _ { m = 1 } ^ { | M | }$ , the state $s ^ { ( S ) }$ is defined by assessing the semantic relevance, typically quantified by measures such as cosine similarity $\sin ( \cdot , \cdot )$ , between the input $( x , y )$ and the stored memories $( \tilde { x } , \tilde { y } )$ within each $D _ { m }$

$$
s ^ { ( S ) } = \{ \operatorname* { m a x } _ { ( \tilde { x } , \tilde { y } ) \in D _ { m } } \sin ( \sigma ( \tilde { x } \oplus \tilde { y } ) , \sigma ( x \oplus y ) ) \} _ { m = 1 } ^ { | M | } ,\tag{1}
$$

where $\oplus$ denotes the concatenation operation, and $\sigma ( \cdot )$ denotes an embedded model utilized to obtain text representations, such as the CPT-Text (Neelakantan et al., 2022). We consider the Top-1 retrieved memories to construct the state.

Actions. Let $a ^ { ( S ) }$ represent an action undertaken by Agent-S. The design of actions corresponds to that of the state $s ^ { ( S ) }$ . Specifically, the actions are defined as follows:

$$
a ^ { ( S ) } = m ( 1 \leq m \leq M ) ,\tag{2}
$$

where action $a ^ { ( S ) } = m$ means to select the $D _ { m }$ for subsequent the generation task.

Rewards. The reward is denoted by $r ^ { ( S ) }$ . When the action $a ^ { ( S ) }$ involves exploring a partition, the reward cannot be immediately observed, as no response has been received for the query x. However, when the action involves selecting a partition for Agent-R to refine the memories within the partition, the stored response $\tilde { y }$ is updated, and some reward signal can be obtained (for example, by measuring the difference between the results on the original memory and that on the refined memory). Therefore, we make Agent-S and Agent-R are trained with multi-agent reinforcement learning, since they cooperate towards the same objective of learning a policy that produces a response (hypothesis) as similar as possible to the reference y for the x.

## 3.3 Agent-R: Refining Memories in the Selected Partition

Next, we formulate the task of refining the retrieved memories carried out by Agent-R within a selected partition. To accomplish this, Agent-R explores potential responses denoted by yˆ through LLMs for the retrieved x˜, and generates a candidate pool $\mathbb { C } = \{ \hat { y } _ { k }  \mathrm { L L M } ( \tilde { x } ) \} _ { k = 1 } ^ { | \bar { K } | }$ for selection, where K denotes the number of candidates. Upon selecting a candidate, Agent-R evaluates its quality by demonstrating the new memory $( \tilde { x } , \hat { y } _ { k } )$ to generate a hypothesis $h \gets \mathrm { L L M } ( x \oplus ( \tilde { x } , \hat { y } _ { k } ) )$ . In summary, a high-quality hypothesis h benefits from superior memory, which can be then refined through the produced hypothesis for subsequent selections. Consequently, Agent-R iterates in a boosting process optimized via reinforcement learning, where the states, actions, and rewards are detailed below.

States. The state $s ^ { ( R ) }$ is defined to assess the semantic relevance between the produced hypothesis h and the selected $\hat { y } _ { k }$ from the pool $\mathbb { C } .$ The rationale is to identify a memory that closely resembles the hypothesis, which aligns with the human intuition that a superior demonstration sample often leads to better generation results, that is

$$
s ^ { ( R ) } = \{ \sin ( \sigma ( h ) , \sigma ( \hat { y } _ { k } ) ) \} _ { k = 1 } ^ { | K | } ,\tag{3}
$$

where $\sigma ( \cdot )$ denotes an embedded model, and K governs the constructed state space.

Actions. Let $a ^ { ( R ) }$ represent an action taken by Agent-R. The design is consistent with the state $s ^ { ( \breve { R } ) }$ , which involves selecting a candidate memory from the pool, that is

$$
a ^ { ( R ) } = k ( 1 \leq k \leq K ) .\tag{4}
$$

Rewards. We denote the reward of Agent-R as $r _ { t } ^ { ( R ) }$ , which corresponds to the transition from the current state ${ \bf s } _ { t } ^ { ( R ) }$ to the next state $\mathbf { s } _ { t + 1 } ^ { ( R ) }$ after taking action $a _ { t } ^ { ( R ) }$ . Specifically, when a memory $( \tilde { x } , \hat { y } _ { k } )$ is updated, the hypothesis changes from h to $h ^ { \prime }$ accordingly. We remark that the best hypothesis (denoted as $h ^ { \prime } )$ identified at state $s ^ { ( R ) }$ is maintained according to a specific metric $\Delta ( \cdot , \cdot )$ (e.g., ROUGE for text summarization, BLEU for machine translation, BLEU and Distinct for dialogue generation), and the reward is defined as:

$$
r ^ { ( R ) } = \Delta ( h ^ { \prime } , y ) - \Delta ( h , y ) ,\tag{5}
$$

where $y$ denotes the reference result. In this reward definition, we observe that the objective of the Markov Decision Process (MDP), which aims to maximize cumulative rewards, aligns with Agent-R’s goal of discovering the best hypothesis among the memories. To illustrate, we consider the process through a sequence of states:

Algorithm 1: The M-RAG Framework   
Require : a database D; a frozen LLM( )   
1 obtain $\mathbb { D } = \{ D _ { m } \} _ { m = 1 } ^ { | M | }$ via a partitioning strategy   
2 initialize $\begin{array} { r } { \mathbf { A g - S } \pi _ { \theta } \big ( a ^ { ( S ) } \big | s ^ { ( S ) } \big ) , \mathbf { A g - R } \pi _ { \phi } \big ( a ^ { ( R ) } \big | s ^ { ( R ) } \big ) } \end{array}$   
3 while not converged on a validation set do   
4 sample a text pair $( x , y )$ from the training set   
5 construct $s _ { 1 } ^ { ( S ) }$ with $( x , y )$ on D by Eq 1   
6 for $i = 1 , \stackrel { \cdot } { 2 } , . .$ . do   
7 sample $m = a _ { i } ^ { ( S ) } \sim \pi _ { \theta } ( a | s _ { i } ^ { ( S ) } )$   
8 $r _ { i } ^ { ( S ) } \gets 0$   
9 $\boldsymbol { h } ^ { \cdot }  \mathrm { L L M } ( \boldsymbol { x } \oplus ( \tilde { \boldsymbol { x } } , \tilde { \boldsymbol { y } } ) \in D _ { m } )$   
10 construct $s _ { 1 } ^ { ( R ) }$ with h on   
$\mathbb { C } = \{ \hat { y } _ { k }  \mathrm { L L M } ( \tilde { x } ) \} _ { k = 1 } ^ { | K | }$ by Eq 3   
11 for $j = 1 , 2 , \dots$ do   
12 sample $k = a _ { j } ^ { ( R ) } \sim \pi _ { \phi } ( a | s _ { j } ^ { ( R ) } )$   
13 $h ^ { \prime }  \mathrm { L L M } ( x ^ { \prime } \oplus ( \tilde { x } , \hat { y } _ { k } ) )$   
14 if $\Delta ( h ^ { \prime } , y ) > \Delta ( h , y )$ then   
15 $r _ { j } ^ { ( R ) } \gets \Delta ( h ^ { \prime } , y ) - \Delta ( h , y )$   
16 $\hat { D } _ { m } . \tilde { y } \gets \hat { y } _ { k } , h \gets h ^ { \prime }$   
17 else   
18 $r _ { j } ^ { ( R ) } \gets 0$   
19 construct $s _ { j + 1 } ^ { ( R ) }$ with h on a new $\mathbb { C }$   
20 $r _ { i } ^ { ( S ) } \gets r _ { i } ^ { ( \check { S } ) } + r _ { j } ^ { ( R ) }$   
21 construct $s _ { i + 1 } ^ { ( S ) }$ by updating $( \tilde { x } , \tilde { y } )$ and $( x , y )$   
22 optimize π<sub>θ</sub> and $\pi _ { \phi }$ via DQN   
23 generate final hypotheses via LLM( ) on D (where   
the trained $\mathsf { A g - \bar { S } }$ selects a partition)

$s _ { 1 } ^ { ( R ) } , s _ { 2 } ^ { ( R ) } , . . . , s _ { N } ^ { ( R ) }$ , concluding at $s _ { N } ^ { ( R ) }$ . The rewards received at these states, except for the termination state, can be denoted as $r _ { 1 } ^ { ( R ) } , r _ { 2 } ^ { ( R ) } , . . . , r _ { N - 1 } ^ { ( R ) }$ When future rewards are not discounted, we have:

$$
\begin{array} { c } { \displaystyle \sum _ { t = 2 } ^ { N } r _ { t - 1 } ^ { ( R ) } = \displaystyle \sum _ { t = 2 } ^ { N } ( \Delta ( h _ { t } , y ) - \Delta ( h _ { t - 1 } , y ) ) } \\ { = \Delta ( h _ { N } , y ) - \Delta ( h _ { 1 } , y ) , } \end{array}\tag{6}
$$

where $\Delta ( h _ { N } , y )$ corresponds to the highest hypothesis value found throughout the entire iteration, and $\Delta ( h _ { 1 } , y )$ represents an initial value that remains constant. Therefore, maximizing cumulative rewards is equivalent to maximizing the discovered hypothesis value. Finally, the cumulative reward is shared with Agent-S to align with the training objective, that is

$$
r ^ { ( S ) } = \Delta ( h _ { N } , y ) - \Delta ( h _ { 1 } , y ) .\tag{7}
$$

## 3.4 The M-RAG Framework

Policy Learning via DQN. In a MDP, the primary challenge lies in determining an optimal policy that guides an agent to select actions at states, with the aim of maximizing cumulative rewards. Given that the states within our MDPs are continuous, we employ Deep Q-Networks (DQN) with replay memory (Mnih et al., 2013) to learn the policy, denoted as $\pi _ { \theta } ( a ^ { ( S ) } | s ^ { ( S ) } )$ for Agent-S (resp. $\pi _ { \phi } \big ( \dot { a } ^ { ( R ) } | s ^ { ( R ) } \big )$ for Agent-R). The policy samples an action $a ^ { ( S ) }$ (resp. $a ^ { ( R ) } )$ at a given state $s ^ { ( \hat { S } ) } ( \mathrm { r e s p . } s ^ { ( R ) } )$ via DQN, with parameters denoted by θ (resp. ϕ).

Combining Agent-S and Agent-R. We present the M-RAG framework in Algorithm 1, which combines the functionalities of Agent-S and Agent-R on multiple partitions (line 1). The algorithm comprises two main phases: training and inference. During the training phase (lines 2-22), we randomly sample text pairs from the training set (line 4). For each pair, we generate episodes to iteratively train Agent-S and Agent-R, with the MDPs outlined in (lines 6-21) and (lines 11-20), respectively. Experiences of $( s _ { t } ^ { ( S ) } , a _ { t } ^ { ( S ) } , r _ { t } ^ { ( S ) } , s _ { t + 1 } ^ { ( S ) } )$ and $( s _ { t } ^ { ( R ) } , a _ { t } ^ { ( R ) } , r _ { t } ^ { ( R ) } , s _ { t + 1 } ^ { ( R ) } )$ are stored during the iteration, and a minibatch is sampled to optimize the two agents via DQN (line 22).

During the inference phase (line 23), final hypotheses are generated via a LLM based on the refined D, where a partition is selected by the trained Agent-S, and the $\tilde { y }$ and $y$ (unknown during inference) are omitted to construct the state by Eq 1.

Time Complexity. We discuss the complexity of M-RAG compared to a Naive RAG setup introduced in Section 2 in terms of the three steps: (1) indexing, (2) retrieval, and (3) generation as shown in Figure 2. In terms of inference, involving (1) and (2), it is worth noting that the M-RAG exhibits a complexity comparable to that of a Naive RAG setup, with the additional complexity (3) only being involved during training.

For (1), the complexity associated with constructing multiple partitions (e.g., using the HNSW index structure) is represented as $O ( M \cdot N \log N )$ , where M indicates the number of partitions and N indicates the maximum number of memories within a partition. This approach proves to be faster compared to a Naive RAG setup, which organizes all data within a single index structure with a construction complexity of $O ( N ^ { \prime } \log { N ^ { \prime } } )$ , where $N ^ { \prime }$ represents the total number of memories in the database.

For (2), the complexity of Agent-S is approximately O(M log N), where an AKNN search is performed within each partition, incurring a cost of O(M  log N) with HNSW. Additionally, sampling actions via Agent-S requires O(1) complexity, owing to its lightweight neural network architecture.

In contrast, for the Naive RAG setup, conducting the AKNN search within the entire database costs ${ \cal O } ( \log N ^ { \prime } )$ , which is marginally faster than the M-RAG setup.

For (3), the complexity of Agent-R is roughly $O ( C \cdot E ^ { 2 } )$ , where E tokens are generated via a LLM based on the transformer attention mechanism, and C represents the number of its MDP iterations. This component predominantly influences the overall training complexity. In contrast, for a Naive RAG setup, it runs only once during the inference procedure to produce the generation outcomes, with a complexity of approximately $O ( E ^ { 2 } )$

## 4 Experiments

## 4.1 Experimental Setup

Datasets. By following (Cheng et al., 2023b), we conduct experiments on seven datasets for three generation tasks: (1) text summarization (XSum Narayan et al., 2018 and BigPatent Sharma et al., 2019), (2) machine translation (JRC-Acquis Steinberger et al., 2006 with Es En, En Es, De En, and En De), and (3) dialogue generation (DailyDialog Li et al., 2017). Specifically, XSum comprises single-document summaries for highly abstractive articles sourced from BBC news. BigPatent comprises 1.3 million records of U.S. patent documents accompanied by human-written abstractive summaries. JRC-Acquis serves as a collection of parallel legislative texts of European Union Law, commonly employed as a benchmark in machine translation tasks. DailyDialog comprises multi-turn dialogues centered around daily life topics. The detailed statistics for these datasets are available in (Cheng et al., 2023b).

Baselines. We carefully review the literature including a recent survey paper (Gao et al., 2023), and identify the following RAGs, namely Naive RAG (Ma et al., 2023), Self-RAG (Asai et al., 2023), and Selfmem (Cheng et al., 2023b), which correspond to three kinds of RAG techniques as described in Section 2. In addition, we incorporate the RAGs into three typical language model architectures, namely Mixtral 8 7B (Jiang et al., 2024), Llama 2 13B (Touvron et al., 2023), Phi-2 2.7B (Abdin et al., 2023), Gemma 7B (Mesnard et al., 2024), and Mistral 7B (Jiang et al., 2023a) for the evaluation.

Evaluation Metrics. We evaluate the effectiveness of M-RAG in terms of the three generation tasks by following (Cheng et al., 2023b). (1) For summarization, ROUGE (R-1/2/L) (Lin, 2004) is used. (2) For machine translation, BLEU (Post, 2018) is used. (3) For dialogue generation, BLEU (B-1/2) and Distinct (D-1/2) (Li et al., 2016, 2021) are used. Overall, a higher evaluation metric (i.e., ROUGE, BLEU, Distinct) indicates a better result. We remark that all results are statistically significant, as confirmed by a t-test with $p < 0 . 0 5$

Implementation Details. We implement M-RAG and adapt other baselines using Python 3.7 and LlamaIndex. The database partitioning strategies for Randomization <sup>1</sup> and Indexing <sup>2</sup> utilize existing libraries. The Agent-S (resp. Agent-R) is instantiated through a two-layered feedforward neural network. The first layer consists of 25 neurons using the tanh activation function, and the second layer comprises M (resp. K) neurons corresponding to the action space with a linear activation function. The hyperparameters M and K are empirically set to 4 and 3, respectively. Some of the built-in RL codes can be found in the GitHub repositories referenced in (Wang et al., 2023b, 2021). During training, we randomly sample 10% of text pairs from the training set, while the remaining data is utilized for constructing the database with multiple partitions. The MDP iterations are determined by performance evaluation on a validation set. Evaluation metrics, such as ROUGE, BLEU, and Distinct, are obtained from (Cheng et al., 2023b). The language models with 4-bit quantization, including Mixtral 8 7B, Llama 2 13B, Phi-2 2.7B, Gemma 7B, and Mistral 7B, are available for download via the link <sup>3</sup>. To boost training efficiency, we cache the QA pairs generated by the LLMs during training.

## 4.2 Experimental Results

(1) Effectiveness evaluation (partitioning strategies). We conduct experiments to evaluate various partitioning strategies across text summarization (XSum), machine translation (Es En), and dialogue generation (DailyDialog) tasks with Mixtral ${ \bf 8 } \times { \bf 7 } { \bf B }$ . The best results, based on a development set across different partitions, are reported. As shown in Figure 1, we observe that retrieval based on the entire database generally fails to achieve optimal performance. Moreover, the performance slightly decreases as the number of partitions increases. This is attributed to the AKNN search, where a smaller partition size recalls more similar memories, which may not align well with the LLM preferences and impede the focus on crucial memories. Additionally, we observe that the RAG with Top-1 retrieval exhibits faster runtime compared to the Top-3 due to a shorter input length for the LLM, while maintaining comparable performance. (2) Effectiveness evaluation (text summarization). We compare the performance of the M-RAG against alternative RAG methods on three distinct language models: Mixtral 8 7B, Llama 2 13B, and Phi-2 2.7B. The corresponding results are outlined in Table 1. We observe consistent improvement in language models when utilizing the RAG framework (e.g., Naive) compared to models without RAG (e.g., None). In addition, the recent MoE architecture Mistral 8  7B generally outperforms the typical Llama 2 13B in the summarization task. Specifically, when considering Mistral 8 7B as a base model, the performance of M-RAG outperforms that of other baseline models on both datasets. For example, it achieves better results than the best baseline model Selfmem, by 8% and 11% in terms of R-1 on XSum and BigPatent, respectively.

Table 1: Text summarization.
<table><tr><td rowspan="2">LLM</td><td rowspan="2">RAG</td><td colspan="3">XSum</td><td colspan="3">BigPatent</td></tr><tr><td>R-1</td><td>R-2</td><td>R-L</td><td>R-1</td><td>R-2</td><td>R-L</td></tr><tr><td>Mixtral 8 × 7B</td><td>None</td><td>25.40</td><td>6.39</td><td>18.30</td><td>47.41</td><td>16.63</td><td>25.14</td></tr><tr><td>Mixtral 8 × 7B</td><td>Naive</td><td>43.82</td><td>22.07</td><td>37.44</td><td>60.11</td><td>38.33</td><td>43.44</td></tr><tr><td>Mixtral 8 × 7B</td><td>Selfmem</td><td>44.67</td><td>22.38</td><td>37.86</td><td>64.12</td><td>39.21</td><td>46.21</td></tr><tr><td>Mixtral 8 × 7B</td><td>Self-RAG</td><td>44.01</td><td>22.26</td><td>37.51</td><td>63.59</td><td>38.65</td><td>45.25</td></tr><tr><td>Mixtral 8 × 7B</td><td>M-RAG</td><td>48.13</td><td>24.66</td><td>39.43</td><td>71.34</td><td>42.24</td><td>47.22</td></tr><tr><td>Llama 2 13B</td><td>M-RAG</td><td>37.18</td><td>18.02</td><td>26.44</td><td>60.31</td><td>37.33</td><td>33.47</td></tr><tr><td>Phi-2 2.7B</td><td>M-RAG</td><td>30.70</td><td>11.57</td><td>26.20</td><td>31.25</td><td>14.72</td><td>18.98</td></tr></table>

Table 2: Machine translation.
<table><tr><td rowspan="2">LLM</td><td rowspan="2">RAG</td><td colspan="2">Es→En</td><td colspan="2">En→Es</td><td colspan="2">De→En</td><td colspan="2">En→De</td></tr><tr><td>Dev</td><td>Test</td><td>Dev</td><td>Test</td><td>Dev</td><td>Test</td><td>Dev</td><td>Test</td></tr><tr><td>Mixtral 8 × 7B</td><td>None</td><td>34.34</td><td>34.81</td><td>32.60</td><td>28.32</td><td>43.75</td><td>44.09</td><td>43.78</td><td>42.24</td></tr><tr><td>Mixtral 8 × 7B</td><td>Naive</td><td>36.64</td><td>36.22</td><td>33.18</td><td>30.70</td><td>47.84</td><td>46.77</td><td>45.83</td><td>44.23</td></tr><tr><td>Mixtral 8 × 7B</td><td>Selfmem</td><td>37.65</td><td>37.11</td><td>34.12</td><td>31.86</td><td>48.08</td><td>47.31</td><td>51.38</td><td>49.81</td></tr><tr><td>Mixtral 8 × 7B</td><td>Self-RAG</td><td>37.17</td><td>36.82</td><td>33.80</td><td>31.61</td><td>47.99</td><td>47.27</td><td>50.10</td><td>48.75</td></tr><tr><td>Mixtral 8 × 7B</td><td>M-RAG</td><td>39.11</td><td>39.98</td><td>35.18</td><td>32.70</td><td>49.16</td><td>48.15</td><td>53.76</td><td>50.75</td></tr><tr><td>Llama 2 13B</td><td>M-RAG</td><td>30.41</td><td>30.03</td><td>26.40</td><td>22.03</td><td>41.10</td><td>42.22</td><td>45.98</td><td>42.58</td></tr><tr><td>Phi-2 2.7B</td><td>M-RAG</td><td>22.83</td><td>24.22</td><td>17.64</td><td>16.60</td><td>34.21</td><td>34.71</td><td>40.01</td><td>37.08</td></tr></table>

Table 3: Dialogue generation.
<table><tr><td rowspan=2 colspan=1>LLM</td><td rowspan=2 colspan=1>RAG</td><td rowspan=1 colspan=1>DailyDialog</td></tr><tr><td rowspan=1 colspan=1>B-1 B-2 D-1 D-2</td></tr><tr><td rowspan=1 colspan=1>Mix. 8 × 7B</td><td rowspan=2 colspan=1>NoneNaive</td><td rowspan=4 colspan=1>15.5227.05 61.49 89.5137.44 29.1689.42 92.5538.16 29.9289.23 95.2337.7629.7988.24 95.34</td></tr><tr><td rowspan=1 colspan=1>Mix. 8 × 7B</td></tr><tr><td rowspan=1 colspan=1>Mix. 8 × 7B</td><td rowspan=2 colspan=1>SelfmemSelf-RAG</td></tr><tr><td rowspan=1 colspan=1>Mix. 8 × 7B</td></tr><tr><td rowspan=1 colspan=1>Mix. 8 × 7B</td><td rowspan=1 colspan=1>M-RAG</td><td rowspan=3 colspan=1>42.61 32.9788.82 95.7431.2917.6363.1988.207.713.9344.21 82.86</td></tr><tr><td rowspan=1 colspan=1>Llama 2 13B</td><td rowspan=2 colspan=1>M-RAGM-RAG</td></tr><tr><td rowspan=1 colspan=1>Phi-2 2.7B</td></tr><tr><td rowspan=1 colspan=1>Mix. 8 × 7B</td><td rowspan=1 colspan=1>M-RAG(D)</td><td rowspan=1 colspan=1>39.14 30.98 93.14 98.34</td></tr></table>

(3) Effectiveness evaluation (machine translation). We further conduct experiments to evaluate the performance of M-RAG for machine translation, and the results are reported in Table 2. We observe that a consistent improvement in the performance of translation tasks with M-RAG across four datasets and three architectures. Notably, it surpasses the Selfmem by 8% in the Es En translation task.

(4) Effectiveness evaluation (dialogue generation). As shown in Table 3, M-RAG further enhances the language model performance for dialogue generation tasks. It outperforms the Selfmem by 12% in terms of B-1. Notably, we can also use the Distinct score as the performance metric for optimizing the two agents, denoted by M-RAG(D), and it results in a more diverse dialogue.

(5) Effectiveness evaluation (results on 7B LLMs). We increase the number of evaluated LLMs, e.g., comparing 7B models (Gemma 7B and Mistral 7B) to show more results. This comparison aims to assess the performance of M-RAG across the three generation tasks, against the best baseline method Selfmem. The results are presented in Table 4. In general, M-RAG consistently outperforms Selfmem on the 7B models.

(6) Ablation study. To evaluate the effectiveness of the two agents in M-RAG, we conduct an ablation study on XSum. We remove Agent-S and utilize the entire database for RAG; we replace Agent-R with a greedy rule to select a candidate memory from the pool according to Equation 3; and we remove both agents, which degrades to the Naive RAG. The results are presented in Table 5, demonstrating that both agents contribute to performance improvement. Specifically, removing Agent-S results in a significant decline in R-1 from 48.13 to 44.20. This underscores the role of the multiple partition setting in enhancing overall performance. Moreover, removing Agent-R leads to a reduction in R-1 from 48.13 to 45.75. This decline is attributed to the effectiveness of Agent-R in learning memory selection dynamically, as opposed to relying on a fixed rule for decision-making.

Table 4: Comparing M-RAG on various 7B LLMs.
<table><tr><td rowspan=2 colspan=1>LLM</td><td rowspan=2 colspan=1>RAG</td><td rowspan=1 colspan=1>Summarization</td><td rowspan=1 colspan=1>Translation (Es→En)</td><td rowspan=1 colspan=1>Dialogue</td></tr><tr><td rowspan=1 colspan=1>R-1    R-2   R-L</td><td rowspan=1 colspan=1>BLEU</td><td rowspan=1 colspan=1>B-1    B-2</td></tr><tr><td rowspan=1 colspan=1>Gemma 7B</td><td rowspan=1 colspan=1>Selfmem</td><td rowspan=1 colspan=1>31.38  9.97  25.07</td><td rowspan=1 colspan=1>24.61</td><td rowspan=1 colspan=1>15.56  7.91</td></tr><tr><td rowspan=1 colspan=1>Gemma 7B</td><td rowspan=1 colspan=1>M-RAG</td><td rowspan=1 colspan=1>33.81  12.93 27.82</td><td rowspan=1 colspan=1>26.92</td><td rowspan=1 colspan=1>18.15  9.95</td></tr><tr><td rowspan=1 colspan=1>Mistral 7B</td><td rowspan=1 colspan=1>Selfmem</td><td rowspan=1 colspan=1>35.40  12.68  27.06</td><td rowspan=1 colspan=1>26.26</td><td rowspan=1 colspan=1>18.28  10.05</td></tr><tr><td rowspan=1 colspan=1>Mistral 7B</td><td rowspan=1 colspan=1>M-RAG</td><td rowspan=1 colspan=1>37.47 13.24 30.49</td><td rowspan=1 colspan=1>32.65</td><td rowspan=1 colspan=1>24.52 11.53</td></tr></table>

Table 5: Ablation study.
<table><tr><td>Components</td><td>R-1</td><td>R-2</td><td>R-L</td></tr><tr><td>M-RAG</td><td>48.13</td><td>24.66</td><td>39.43</td></tr><tr><td>w/o Agent-S (single DB) w/o Agent-R (greedy)</td><td>44.20 45.75</td><td>22.72 23.21</td><td>37.40 38.28</td></tr><tr><td>w/o Agent-S and Agent-R</td><td>43.82</td><td>22.07</td><td>37.44</td></tr></table>

<table><tr><td colspan="4">Table 6: Impacts of the number of M in Agent-S.</td></tr><tr><td>M</td><td>1 2 3</td><td>4</td><td>5</td></tr><tr><td>R-1 Index constr. (s)</td><td>44.20 44.53 46.27 299 278 257</td><td>48.13</td><td>47.21</td></tr><tr><td>Retrieval (s)</td><td></td><td>246</td><td>227</td></tr><tr><td></td><td>0.61 1.09 1.54</td><td>2.19</td><td>2.59</td></tr><tr><td>Generation (s)</td><td>83.59 84.88 82.81</td><td>82.89</td><td>86.64</td></tr></table>

(7) Parameter study (Agent-S state space M). We study the effect of parameter M, which controls the state space of Agent-S and corresponds to the number of partitions. In Table 6, we observe that setting M = 4 yields the best effectiveness while maintaining reasonable runtime in terms of index construction, retrieval, and generation. This is consistent with empirical studies illustrated in Figure 1 (a). When M = 1, it reduces to a single database for RAG. As M increases, index construction accelerates on smaller partitions, while retrieval time sightly increases due to the additional time required for constructing states by querying each partition. As expected, the retrieval time is much smaller than the language generation time.

(8) Parameter study (Agent-R state space K).

Table 7: Impacts of the number of K in Agent-R.
<table><tr><td>K</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td>R-1</td><td>45.81</td><td>46.54</td><td>48.13</td><td>48.18</td><td>48.25</td></tr><tr><td>Pool gen. (s)</td><td>76</td><td>191</td><td>267</td><td>290</td><td>359</td></tr></table>

We study the effect of parameter K in Agent-R, representing the state space of Agent-R, to choose one memory from a candidate pool with a size of K. In Table 7, we observe a performance improvement as K increases from 1 to 3, and then remains stable. Particularly, when K = 1, M-RAG exhibits the worst performance, possibly due to the limited exploration of potential memories for generating improved hypotheses. We choose the setting of K = 3, as it demonstrates effective performance, and runs reasonably fast for generating the pool.

## 5 Conclusion and Limitations

In this paper, we propose a multiple partition paradigm for RAG, which aims to refine retrieval processes and emphasize pivotal memories to improve overall performance. Additionally, we introduce M-RAG, a novel framework with multi-agent reinforcement learning, which addresses key challenges inherent in executing RAG across multiple partitions. The training objective of M-RAG is well aligned with that of text generation tasks, showcasing its potential to enhance system performance explicitly. Through extensive experiments conducted on seven datasets for three language generation tasks, we validate the effectiveness of M-RAG.

For limitations, we conduct experiments with quantized versions of language models due to computational constraints. However, the observed effectiveness gains are expected to remain consistent across different model sizes and should not significantly impact the overall trends of various RAG methods. Further, although the parameters of the LLMs remain fixed and only the parameters of Agent-S and Agent-R are trained, the training efficiency is limited, as indicated by the training time complexity discussed in Section 3.4. This is due to the necessity of querying the LLMs during the training process. In future work, we intend to explore solutions to overcome these limitations.

## References

Marah Abdin, Jyoti Aneja, ebastien Bubeck, and Caio Cesar Teodoro Mendes et al. 2023. Phi-2: The surprising power of small language models. https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models.

Nathan Anderson, Caleb Wilson, and Stephen D. Richardson. 2022. Lingua: Addressing scenarios for live interpretation and automatic dubbing. In AMTA, pages 202–209.

Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2023. Self-rag: Learning to retrieve, generate, and critique through self-reflection. CoRR, abs/2310.11511.

V. Blagojevi. 2023. Enhancing rag pipelines in haystack: Introducing diversityranker and lostinthemiddleranker. https://towardsdatascience.com/enhancingrag-pipelines-in-haystack-45f14e2bc9f5.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. NeurIPS, 33:1877–1901.

Howard Chen, Ramakanth Pasunuru, Jason Weston, and Asli Celikyilmaz. 2023. Walking down the memory maze: Beyond context limit through interactive reading. CoRR, abs/2310.05029.

Daixuan Cheng, Shaohan Huang, Junyu Bi, Yuefeng Zhan, Jianfeng Liu, Yujing Wang, Hao Sun, Furu Wei, Weiwei Deng, and Qi Zhang. 2023a. UPRISE: universal prompt retrieval for improving zero-shot evaluation. In EMNLP, pages 12318–12337.

Xin Cheng, Di Luo, Xiuying Chen, Lemao Liu, Dongyan Zhao, and Rui Yan. 2023b. Lift yourself up: Retrieval-augmented text generation with self memory. NeurIPS.

Woon Sang Cho, Pengchuan Zhang, Yizhe Zhang, Xiujun Li, Michel Galley, Chris Brockett, Mengdi Wang, and Jianfeng Gao. 2018. Towards coherent and cohesive long-form text generation. CoRR.

Zhuyun Dai, Vincent Y. Zhao, Ji Ma, Yi Luan, Jianmo Ni, Jing Lu, Anton Bakalov, Kelvin Guu, Keith B. Hall, and Ming-Wei Chang. 2023. Promptagator: Few-shot dense retrieval from 8 examples. In ICLR.

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo, Meng Wang, and Haofen Wang. 2023. Retrievalaugmented generation for large language models: A survey. CoRR, abs/2312.10997.

Siddharth Gollapudi, Neel Karia, Varun Sivashankar, Ravishankar Krishnaswamy, Nikit Begwani, Swapnil Raz, Yiyong Lin, Yin Zhang, Neelam Mahapatro, Premkumar Srinivasan, et al. 2023. Filtered-diskann: Graph algorithms for approximate nearest neighbor search with filters. In WWW, pages 3406–3416.

Jiatao Gu, Yong Wang, Kyunghyun Cho, and Victor O. K. Li. 2018. Search engine guided neural machine translation. In AAAI, pages 5133–5140. AAAI Press.

Rentong Guo, Xiaofan Luan, Long Xiang, Xiao Yan, Xiaomeng Yi, Jigao Luo, Qianya Cheng, Weizhi Xu, Jiarui Luo, Frank Liu, et al. 2022. Manu: a cloud native vector database management system. PVLDB, 15(12):3548–3561.

Yikun Han, Chunjiang Liu, and Pengfei Wang. 2023. A comprehensive survey on vector database: Storage and retrieval technique, challenge. CoRR.

Nabil Hossain, Marjan Ghazvininejad, and Luke Zettlemoyer. 2020. Simple and effective retrieve-editrerank text generation. In ACL, pages 2532–2538.

Piotr Indyk and Rajeev Motwani. 1998. Approximate nearest neighbors: towards removing the curse of dimensionality. In STOC, pages 604–613.

Natasha Jaques, Asma Ghandeharioun, Judy Hanwen Shen, Craig Ferguson, Agata Lapedriza, Noah Jones, Shixiang Gu, and Rosalind Picard. 2019. Way offpolicy batch deep reinforcement learning of implicit human preferences in dialog. CoRR.

Herve Jegou, Matthijs Douze, and Cordelia Schmid. 2010. Product quantization for nearest neighbor search. TPAMI, 33(1):117–128.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, and et al. 2023a. Mistral 7b. CoRR, abs/2310.06825.

Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, and Arthur Mensch et al. 2024. Mixtral of experts. CoRR, abs/2401.04088.

Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023b. Llmlingua: Compressing prompts for accelerated inference of large language models. In EMNLP, pages 13358–13376.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick S. H. Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. In EMNLP (1), pages 6769–6781.

Julia Kreutzer, Shahram Khadivi, Evgeny Matusov, and Stefan Riezler. 2018. Can neural machine translation be improved with user feedback? CoRR.

Carolin Lawrence and Stefan Riezler. 2018. Improving a neural semantic parser by counterfactual learning from human bandit feedback. CoRR.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. 2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. NeurIPS, 33:9459–9474.

Jinpeng Li, Yingce Xia, Rui Yan, Hongda Sun, Dongyan Zhao, and Tie-Yan Liu. 2021. Stylized dialogue generation with multi-pass dual learning. In NeurIPS, pages 28470–28481.

Jiwei Li, Michel Galley, Chris Brockett, Jianfeng Gao, and Bill Dolan. 2016. A diversity-promoting objective function for neural conversation models. In HLT-NAACL, pages 110–119.

Xinze Li, Zhenghao Liu, Chenyan Xiong, Shi Yu, Yu Gu, Zhiyuan Liu, and Ge Yu. 2023. Structureaware language model pretraining improves dense retrieval on structured data. In ACL (Findings), pages 11560–11574.

Yanran Li, Hui Su, Xiaoyu Shen, Wenjie Li, Ziqiang Cao, and Shuzi Niu. 2017. Dailydialog: A manually labelled multi-turn dialogue dataset. In IJCNLP(1), pages 986–995.

Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81.

Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi, Maria Lomeli, Rich James, Pedro Rodriguez, Jacob Kahn, Gergely Szilvasy, Mike Lewis, Luke Zettlemoyer, and Scott Yih. 2023. RA-DIT: retrieval-augmented dual instruction tuning. CoRR, abs/2310.01352.

Ron Litman, Oron Anschel, Shahar Tsiper, Roee Litman, Shai Mazor, and R. Manmatha. 2020. SCAT-TER: selective context attentional scene text recognizer. In CVPR, pages 11959–11969.

Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan. 2023. Query rewriting for retrievalaugmented large language models. EMNLP, pages 5303–5315.

Yu A Malkov and Dmitry A Yashunin. 2018. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. TPAMI, 42(4):824–836.

Yury Malkov, Alexander Ponomarenko, Andrey Logvinov, and Vladimir Krylov. 2014. Approximate nearest neighbor algorithm based on navigable small world graphs. Information Systems, 45:61–68.

Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, and et al. 2024. Gemma: Open models based on gemini research and technology. CoRR, abs/2403.08295.

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. 2013. Playing atari with deep reinforcement learning. CoRR.

Reiichiro Nakano, Jacob Hilton, Suchir Balaji, and Jeff Wu et al. 2021. Webgpt: Browser-assisted question-answering with human feedback. CoRR, abs/2112.09332.

Shashi Narayan, Shay B. Cohen, and Mirella Lapata. 2018. Don’t give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization. In EMNLP, pages 1797–1807.

Arvind Neelakantan, Tao Xu, Raul Puri, Alec Radford, Jesse Michael Han, Jerry Tworek, Qiming Yuan, Nikolas Tezak, Jong Wook Kim, Chris Hallacy, et al. 2022. Text and code embeddings by contrastive pretraining. CoRR.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. NeurIPS, 35:27730– 27744.

James Jie Pan, Jianguo Wang, and Guoliang Li. 2023. Survey of vector database management systems. CoRR.

Matt Post. 2018. A call for clarity in reporting BLEU scores. In WMT, pages 186–191.

Eva Sharma, Chen Li, and Lu Wang. 2019. BIG-PATENT: A large-scale dataset for abstractive and coherent summarization. In ACL (1), pages 2204– 2213.

Aleksandrs Slivkins et al. 2019. Introduction to multiarmed bandits. Foundations and Trends® in Machine Learning, 12(1-2):1–286.

Ralf Steinberger, Bruno Pouliquen, Anna Widiger, Camelia Ignat, Tomaz Erjavec, Dan Tufis, and Dániel Varga. 2006. The jrc-acquis: A multilingual aligned parallel corpus with 20+ languages. In LREC, pages 2142–2147.

Hugo Touvron, Louis Martin, Kevin Stone, and Peter Albert et al. 2023. Llama 2: Open foundation and fine-tuned chat models. CoRR, abs/2307.09288.

Shuohang Wang, Yichong Xu, Yuwei Fang, Yang Liu, Siqi Sun, Ruochen Xu, Chenguang Zhu, and Michael Zeng. 2022. Training data is more valuable than you think: A simple and effective method by retrieving from training data. In ACL, pages 3170–3179.

Xintao Wang, Qianwen Yang, Yongting Qiu, Jiaqing Liang, Qianyu He, Zhouhong Gu, Yanghua Xiao, and Wei Wang. 2023a. Knowledgpt: Enhancing large language models with retrieval and storage access on knowledge bases. CoRR, abs/2308.11761.

Zheng Wang, Bingzheng Gan, and Wei Shi. 2024. Multimodal query suggestion with multi-agent reinforcement learning from human feedback. In WWW, pages 1374–1385.

Zheng Wang, Cheng Long, Gao Cong, and Christian S. Jensen. 2023b. Collectively simplifying trajectories in a database: A query accuracy driven approach. CoRR, abs/2311.11204.

Zheng Wang, Cheng Long, Gao Cong, and Qianru Zhang. 2021. Error-bounded online trajectory simplification with multi-agent reinforcement learning. In KDD, pages 1758–1768.

Jeff Wu, Long Ouyang, Daniel M Ziegler, Nisan Stiennon, Ryan Lowe, Jan Leike, and Paul Christiano. 2021a. Recursively summarizing books with human feedback. CoRR.

Sixing Wu, Ying Li, Minghui Wang, Dawei Zhang, Yang Zhou, and Zhonghai Wu. 2021b. More is better: Enhancing open-domain dialogue generation via multi-source heterogeneous knowledge. In EMNLP, pages 2286–2300.

Sixing Wu, Ying Li, Dawei Zhang, and Zhonghai Wu. 2022. KSAM: infusing multi-source knowledge into dialogue generation via knowledge source aware multi-head decoding. In ACL (Findings), pages 353– 363.

Shitao Xiao, Zheng Liu, Peitian Zhang, and Xingrun Xing. 2023. Lm-cocktail: Resilient tuning of language models via model merging. CoRR, abs/2311.13534.

Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2023. RECOMP: improving retrieval-augmented lms with compression and selective augmentation. CoRR, abs/2310.04408.

Wenzhuo Xue, Hui Li, Yanguo Peng, Jiangtao Cui, and Yu Shi. 2017. Secure k nearest neighbors query for high-dimensional vectors in outsourced environments. IEEE TBD, 4(4):586–599.

Sanghyun Yi, Rahul Goel, Chandra Khatri, Alessandra Cervone, Tagyoung Chung, Behnam Hedayatnia, Anu Venkatesh, Raefer Gabriel, and Dilek Hakkani-Tur. 2019. Towards coherent and engaging spoken dialog response generation using automatic conversation evaluators. CoRR.

Shengyao Zhuang, Bing Liu, Bevan Koopman, and Guido Zuccon. 2023. Open-source large language models are strong zero-shot query likelihood models for document ranking. In EMNLP (Findings), pages 8807–8817.

## A Appendix

## A.1 Other Evaluation Metrics for Machine Translation

We utilize BLEURT <sup>4</sup> (with the checkpoint of BLEURT-20) and COMET <sup>5</sup> (with wmt22-cometda to obtain features) to evaluate the performance of machine translation, and then compare M-RAG with the best baseline method, Selfmem, on the Mixtral 8 × 7B. The results are reported in Table 8.

Overall, we observe that M-RAG consistently outperforms Selfmem across diverse translation datasets, as evidenced by various evaluation metrics.

## A.2 Further Discussion

## Q1. Why applying RAG for summarization or translation?

Employing RAG for summarization or translation is based on two key factors: (1) We believe that the two tasks effectively capture the essence of text generation facilitated by LLMs; (2) the widespread adoption of summarization and translation tasks in retrieval-augmented literature (Cheng et al., 2023b; Gu et al., 2018; Hossain et al., 2020) provides a standardized and comparable testbed for benchmarking our method. Here, certain text pairs are stored within an external database, such as (document, summary) pairs for summarization or (context, response) pairs for dialogue generation. These pairs are retrieved from the database and serve as demonstration examples to guide a LLM in conducting text generations. The underlying rationale of this paradigm is that better demonstrations typically prompt better generation outcomes.

Q2. Why applying such partitioning, what intuition behind that, instead of improving the quality of retrieval or introduce more dimensions in the scoring function to account for categories/partitions?

We recognize that database partitioning plays a crucial role in efficiently managing a database. However, this aspect has been relatively underexplored in the context of RAG, despite the necessity of accessing an external database to obtain essential information for LLM generation. To address this gap, we investigate a multiple partition paradigm for executing RAG. The rationale behind this approach is intuitive: with various attributes associated with the data in a database, queries should ideally be matched with their corresponding attributed data, thereby filtering out noise data.

We discuss our choice of employing partitioning for RAG instead of two alternative approaches: (1) improving the quality of retrieval or (2) introduce more dimensions in the scoring function to account for categories/partitions.

For (1), improving retrieval quality typically emphasizes the effectiveness of AKNN search, often measured using metrics such as recall. However, this focus is not entirely aligned with the primary objective of RAG, which is to generate a good response. In the M-RAG framework, we prioritize the quality of LLM generation as an end-to-end metric explicitly guiding the retrieved information.

Table 8: Machine translation with BLEURT and COMET.
<table><tr><td rowspan="2">LLM</td><td rowspan="2">RAG</td><td colspan="4">BLEURT</td><td colspan="4">COMET</td></tr><tr><td>Es→En</td><td>En→Es De→En</td><td></td><td>En→De</td><td>Es→En</td><td>En→Es De→En En→De</td><td></td><td></td></tr><tr><td>Mixtral  $\phantom { + } 8 \times 7 \mathbf { B }$ </td><td>Selfmem</td><td>63.63</td><td>53.26</td><td>59.93</td><td>59.91</td><td>75.65</td><td>55.28</td><td>60.41</td><td>52.13</td></tr><tr><td>Mixtral  ${ \bf 8 } \times { \bf 7 } { \bf B }$ </td><td>M-RAG</td><td>71.74</td><td>63.66</td><td>66.77</td><td>70.99</td><td>82.66</td><td>80.29</td><td>67.33</td><td>85.14</td></tr></table>

For (2), unlike attending to data categories or partitions, we observe that the multiple partition setup offers a cost-effective approach to enhance effectiveness, as confirmed in Figure 1. In this context, no additional computation associated with the LLM is required. Instead, we can keep the LLM frozen, and explore (via Agent-S) or revise (via Agent-R) a relevant memory. This typically leads to improved generation results for the LLM.

## Q3. What is the motivation of the Agent-R and the revision of the retrieved memory?

M-RAG involves a Retrieval-then-Generation process employing LLMs, typically containing billions of parameters. Here, the LLM remains frozen while the retrieved memories undergo revision before being fed back into the LLM to enhance results. Common revision operations within the retrieved memory, such as re-ranking content (Blagojevi, 2023), eliminating irrelevant context (Anderson et al., 2022), summarizing key information (Chen et al., 2023), and generating candidates for selection (Cheng et al., 2023b), have been extensively studied in retrieval-augmented literature, as highlighted in the survey paper (Gao et al., 2023). In our work, we conceptualize memory revision as a Markov Decision Process (MDP) and investigate a reinforcement learning solution employing the proposed Agent-R for this operation.

## Q4. M-RAG relies on the partitioning strategy. If the partitions are not well-optimized, it could lead to suboptimal retrieval and generation performance?

The performance of M-RAG is preserved through several measures. First, we conduct an empirical study, depicted in Figure 1, to investigate a partitioning strategy that outperforms retrieval from the entire database. This serves as a prerequisite for achieving performance improvements. Additionally, building upon this prerequisite, the challenge shifts to identifying suitable partitions and enhancing data quality within them, tasks that are addressed concurrently by two agents. As illustrated in the ablation study presented in Table 5, performance gains are still attainable even if one agent fails, suggesting that performance improvements can be expected with the M-RAG approach.