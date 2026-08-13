# What Does the Bot Say? Opportunities and Risks of Large Language Models in Social Media Bot Detection

Shangbin Feng<sup>1</sup> Herun Wan<sup>2</sup> Ningnan Wang<sup>2</sup>

Zhaoxuan Tan<sup>3</sup> Minnan Luo<sup>2</sup> Yulia Tsvetkov<sup>1</sup>

<sup>1</sup>University of Washington <sup>2</sup>Xi’an Jiaotong University <sup>3</sup>University of Notre Dame shangbin@cs.washington.edu

## Abstract

Social media bot detection has always been an arms race between advancements in machine learning bot detectors and adversarial bot strategies to evade detection. In this work, we bring the arms race to the next level by investigating the opportunities and risks of state-of-the-art large language models (LLMs) in social bot detection. To investigate the opportunities, we design novel LLM-based bot detectors by proposing a mixture-of-heterogeneous-experts framework to divide and conquer diverse user information modalities. To illuminate the risks, we explore the possibility of LLM-guided manipulation of user textual and structured information to evade detection. Extensive experiments with three LLMs on two datasets demonstrate that instruction tuning on merely 1,000 annotated examples produces specialized LLMs that outperform state-of-the-art bot detection baselines by up to 9.1% on both datasets. On the other hand, LLM-guided manipulation strategies could significantly bring down the performance of existing bot detectors by up to 29.6% and harm the calibration and reliability of bot detection systems. Ultimately, this works identifies LLMs as the new frontier of social bot detection research.<sup>1</sup>

## 1 Introduction

Social media bot accounts are behind many online perils such as misinformation (Lu and Li, 2020; Huang et al., 2022), election interference (Howard et al., 2016; Rossi et al., 2020; Ng et al., 2022), extremist campaigns (Ferrara et al., 2016; Marcellino et al., 2020), and conspiracy theories (Ferrara, 2020; Ahmed et al., 2020; Ginossar et al., 2022). Research on detecting social media bots has always been an arms race (Cresci et al., 2017): early methods focus on analyzing user metadata with machine learning classifiers (Yang et al., 2020;

Echeverrï£¡ a et al., 2018), while bot operators manipulate user features to evade detection (Cresci, 2020); later approaches employed word embeddings and encoder-based language models to characterize user texts (Wei and Nguyen, 2019; Dukic´ et al., 2020), while bot operators re-post genuine content to dilute malicious content and appear innocuous (Cresci, 2020); recent models tap into the network information of user interactions with graph neural networks (Feng et al., 2021c; Huang et al., 2022; Lei et al., 2023), while advanced bots strategically follow and unfollow users to appear out-ofdistribution (Ye et al., 2023; Li et al., 2023b).

Recent advances brought us large language models (LLMs) that excel in academic tasks and benchmarks (Liang et al., 2023), capable of following instructions (Ouyang et al., 2022), but they also come with risks and biases that could cause realworld harms (Weidinger et al., 2022; Kumar et al., 2023b; Feng et al., 2023). In this work, we ask: What are the opportunities and risks of large language models in social bot detection? As the arms race escalates, we focus on how state-of-the-art large language models could aid robust bot detection systems and how LLMs might be maliciously employed to design more evasive bots.

For opportunities, we propose a mixtureof-heterogeneous-experts framework, employing LLMs to divide and conquer various user information modalities such as metadata, text, and user interaction networks. For user metadata, we verbalize categorical and numerical user features in natural language sequences and employ in-context learning for bot detection. For user-generated texts, we retrieve similar posts from an annotated training set as in-context learning examples. For the network information, guided by previous works about LLMs’ graph reasoning capabilities (Wang et al., 2024; Huang et al., 2023b), we include the user’s following information, in either random or similarity-based order, as part of the prompt context to aid detection. These modality-specific LLMs are then used through in-context learning prompting or instruction tuning, and modality-specific results are ensembled through majority voting.

![](images/db1d6f5bd392097f4c834a6b115c37f9733585919676498d9dba0ff6da3b17e4.jpg)  
Figure 1: Overview of the opportunities of LLM-based bot detectors and risks of LLM-based evasive bots.

For risks, we investigate the possibility of LLMguided bot design to evade detection by tampering with the textual and structural information of Text attributebot accounts. For textual information, we explore rewriting user posts with LLMs to appear genuine with four mechanisms: 1) zero-shot prompting; 2) few-shot rewriting to imitate the posts of genuine users; 3) interactive rewriting between LLMs and an external bot classifier; 4) synthesizing the attributes of related posts from bots and humans for style transfer. For structural information, we employ LLMs to suggest new users to follow or existing users to unfollow, editing the neighborhood of bot accounts. LLM-guided manipulation of textual and structural features is then merged to produce LLM-guided social media bots.

We conduct extensive experiments with three LLMs on two standard bot detection datasets to evaluate the proposed detectors and manipulation strategies. We find that on the opportunities side, LLMs are liable to become state-of-the-art detectors: while in-context learning struggles to capture the nuances of bot accounts, instruction tuning outperforms baselines by up to 9.1% on both datasets. With respect to threat and risk modeling, LLMguided manipulations on both textual and structural information reduce the performance of existing detectors by up to 29.6%, and LLM-based detectors are more robust towards bots designed by LLMs. Our work opens up new research avenues in the ever-lasting arms race between researchers and bot operators, focusing on LLMs as the new frontier of social bot detection research.

## 2 Methodology

## 2.1 Opportunities: Large Language Models as Better Bot Detectors

Social media bot detection focuses on evaluating and classifying social media accounts into bot or human based on diverse user information: user metadata $\mathcal { M } = \{ m _ { 1 } , \ldots , m _ { k } \}$ where each $m _ { i }$ is either a numerical or categorical feature; user posts $\mathcal { T } = \{ t _ { 1 } , \ldots , t _ { \ell } \}$ where each $\mathbf { \Delta } _ { t _ { i } }$ is a natural language sequence; user network information ${ \mathcal { N } } =$ $\{ { \mathcal { N } } _ { 1 } , { \mathcal { N } } _ { 2 } \}$ where $\mathcal { N } _ { 1 }$ denotes the user’s followers set and ${ \mathcal { N } } _ { 2 }$ denotes the following set. We aim to develop bot detectors $f ( \mathcal { M } , \mathcal { T } , \mathcal { N } ) $ human, bot .

We develop LLM-based bot detectors by proposing a mixture-of-heterogeneous-experts framework to tackle the diverse user information. Specifically, different user information modalities are separately analyzed with LLMs while majority voting is conducted to ensemble uni-modality predictions. Each modality-specific predictor either uses the LLM off-the-shelf with in-context learning (Brown et al., 2020) or employs instruction tuning (Ouyang et al., 2022) to adapt LLM for analyzing a particular set of user information. We present an overview of the proposed framework in Figure 1.

Metadata-Based We sequentially concatenate an account’s metadata to linearize it as a natural language sequence. We then randomly select a balanced set of n in-context examples, and provide their metadata as well as the labels in the prompt.

Text-Based For each textual sequence $t \in \tau ,$ we first retrieve the top-n similar user posts in the training set with a retrieval system (Robertson et al., 2009). We then similarly employ in-context learning with the LLMs to make predictions for all posts in  and conduct a majority vote.

We also employ a meta+text approach where both user metadata and textual posts are presented for in-context learning with LLMs.

Structure-Based In addition to analyzing each user individually, interactions among users and the graph structure they form are also crucial in identifying advanced bot clusters (Liu et al., 2023). Grounded in previous research demonstrating that LLMs do have preliminary abilities to reason over graphs and structured data (Wang et al., 2024), we employ LLMs to analyze a user’s neighborhood of follow relations.

Concretely, we employ the following prompt to linearize the neighborhood structure of a given user: “These users follow the target user: PERM( ). The target userfollows these users: PERM( <sub>2</sub>)”, where PERM( ) denotes a permutation function regarding how to order and arrange the follower/following set. We employ two modes for PERM: 1) random, where users along with their information are linearized in random order; 2) attention: inspired by the success of graph attention networks (Velickoviˇ c et al.´ , 2018; Huang et al., 2023b) and the variation in edge importance in a network, we arrange users based on their similarity to the target account. Formally, given the target user’s post t, a neighboring user’s similarity score could be defined as sim(enc(t), enc(t′)), where sim( , ) denotes cosine similarity, enc( ) denotes an encoder-based LM, and t′ denotes the post of the neighboring account. PERM then arranges the users based on their similarity scores from high to low, along with the prompt “from most related to least related:” to encourage LLMs to take the relative similarity/importance of neighbors into account.

After developing five LLM predictors analyzing different user information modalities (metadata, text, metadata+text, structure-random, and structure-attention), they are employed through either in-context learning or instruction tuning.

In-Context Learning We directly prompt the LLM off-the-shelf, without any tuning or adaptation, with the n in-context examples and labels as well as the target user’s information.

Instruction Tuning We employ meta-learning with in-context learning (Min et al., 2022a) to adapt the LLM for better analyzing a specific user information source through instruction tuning. Instruction tuning aims to improve LLMs’ ability to follow instructions by fine-tuning LLMs on triples of instruction, input, output (Ouyang et al., 2022). We write a short instruction based on each modality, use the information of in-context examples and target user as input, and the gold label as $o u t p u t ^ { 2 }$

The predictions of each modality-specific LLM are then ensembled by majority voting into one prediction of whether the target user is a bot or not.

## 2.2 Risks: Large Language Models as Evasive Bot Designers

On the risks side, we explore how LLMs might be employed to design advanced bots to evade detection. While user metadata is often hard to manipulate with the help of LLMs (e.g. # of followers and account creation time), textual information and structural information could be easily altered with LLM-generated post paraphrases and LLM-suggested users to follow and unfollow. We first explore possibilities of manipulating textual information , focusing on rewriting the posts of bot accounts with LLMs to evade detection.

Zero-Shot Rewriting We directly prompt the LLM with “Please rewrite the description ofthis bot account to sound like a genuine user.”

Few-Shot Rewriting We employ a retrieval system to employ the top-n most similar posts to the target post that are written by genuine users. We then prompt the LLM to imitate these examples and rewrite the target bot post.

Classifier Guidance We propose to empower LLMs to iteratively refine a bot-generated post with feedback from an external classifier. Specifically, we first train an encoder-based LM to classify user posts into bot or human and produce a confidence score $f ( t ) ~  ~ [ 0 , 1 ]$ At each step, the LLM learns from the rewritten posts in the previous steps along with the confidence scores given to those posts, aiming to reduce the bot likelihood in the eye of the external classifier. Formally, $t ^ { i + 1 } = \mathrm { L L M } ( t ^ { i } , f ( t ^ { i } ) , \ldots , t ^ { 0 } , f ( t ^ { 0 } ) )$ where $\mathbf { \Delta } _ { t ^ { 0 } }$ is the original bot post. This process is repeated for n times, producing a paraphrased bot post that learns from the edit history and the external classifier.

Text Attributes Previous works have demonstrated that LLMs could summarize the differences between machine-generated and human-written text and employ the summary for better detection (Lu et al., 2023). To this end, we first retrieve the top-n similar posts from human accounts and topn from bots, then prompt the LLM to summarize the differences in text attributes between the two groups of posts. In a separate prompt, the LLM then rewrites the target bot post with the help of the summarized difference.

Aside from editing user textual information, we also tap into LLMs’ capabilities of preliminary graph reasoning (Wang et al., 2024) and employ them to edit the structural information, specifically by adding and removing users to follow for a target bot. We investigate whether LLMs might be capable of suggesting reasonable neighbors to make the bot seem more genuine or identifying current neighbors that might give away its bot nature.

Add Neighbor We randomly select n users that the target bot is not currently following. We then prompt the LLM to “Please suggest one new user to follow so that the target bot appears more human.” by providing the metadata and textual information of these users and the target bot.

Remove Neighbor We prompt the LLM to “Please suggest one user to unfollow so that the target bot appears more human.” by providing the metadata and textual information of the target bot and its current following list.

Combine Neighbor We combine the results of add neighbor and remove neighbor.

The manipulation strategies for textual and structural user information could be further merged to design bots that are more evasive in both aspects.

Selective Combine State-of-the-art bot detection approaches often jointly analyze multiple sources of user information (Tan et al., 2023), but not all modalities are malicious and give away the bot nature (Liu et al., 2023). To this end, we employ LLMs to judge which information modality, text or graph, could be malicious in a given bot and employ the corresponding manipulation strategy. Specifically, we first provide LLMs with rationale about how existing bot detectors work in a prepended passage. We then provide all , , and for a given bot, prompting the LLM to evaluate whether the textual, structural, or both user information seems malicious. The manipulation strategies of classifier guidance and combine neighbor are then selectively activated to edit the bot account.

Both Combine We simply merge the edits of classifier guidance and combine neighbor for a given bot account’s textual and structural features.

## 3 Experiment Settings

Models and Settings We employ three LLMs to study their opportunities and risks in social media bot detection: Mistral-7B (Jiang et al., 2023a), LLaMA2-70b (Touvron et al., 2023), and ChatGPT. For in-context learning, we employ 16 in-context examples by default. For instruction tuning, we randomly sample 1,000 examples from the training set to adapt LLMs. We set temperature τ = 0.1 for language generation by default. Specific prompt templates are listed in Appendix C.

Datasets We experiment with two comprehensive benchmarks of social bot detection: TwiBot-20 (Feng et al., 2021b) and TwiBot-22 (Feng et al., 2022b), two graph-based datasets providing diverse user and bot interactions on social media. These datasets mainly feature English social media posts but other languages are occasionally included.

Baselines On the opportunities side, we compare our proposed LLM-based bot detectors with 9 baselines leveraging varying aspects of user information: SGBot (Yang et al., 2020), LOBO (Echeverrï£¡ a et al., 2018), RoBERTa (Liu et al., 2019), RGT (Feng et al., 2022a), Botometer (Yang et al., 2022), BotBuster (Ng and Carley, 2023), BotPercent (Tan et al., 2023), BIC (Lei et al., 2023), and LMBot (Cai et al., 2024). We provide more baseline details in Appendix A.3.

## 4 Results

## 4.1 Opportunities

We present the performance of baselines and our LLM-based detectors in Table 1.

LLM-based detectors achieve state-of-the-art performance. On both datasets, ChatGPTensemble with instruction tuning outperforms the strongest baseline by 2.6% and 9.1% on F1-score. In addition, ChatGPT with instruction tuning outperforms in-context learning by 34.7% in accuracy: we hypothesize that while in-context learning abilities are attributed to pretraining data (Min et al., 2022b) and LLMs have seen social media texts (Dodge et al., 2021), the nuances of bot accounts are beyond simple data artifacts and would need model adaptation and reasoning. We also find that larger LMs are better at social bot detection. On average, Mistral-7B, LLaMA2-70B, and ChatGPT achieve 0.5651, 0.6347, and 0.6478 accuracy on the two datasets. This ranking is in line with their general utility on standard NLP benchmarks.

<table><tr><td rowspan="2">Method</td><td rowspan="2">M</td><td rowspan="2">T</td><td rowspan="2">N</td><td rowspan="2"></td><td colspan="3">Twibot-20</td><td colspan="4">Twibot-22</td></tr><tr><td>Acc</td><td>F1</td><td>Prec. Rec.</td><td>Acc</td><td>F1</td><td>Prec.</td><td>Rec.</td></tr><tr><td>BIC</td><td></td><td>√</td><td>√</td><td>0.876</td><td>0.891</td><td>/</td><td>1</td><td>/</td><td>1</td><td>/</td><td>1</td></tr><tr><td>LMBOT</td><td>√</td><td>√</td><td>√</td><td>0.856</td><td>0.876</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>SGBOT</td><td>√</td><td>√</td><td>√</td><td>0.816</td><td>0.849</td><td>0.764</td><td>0.949</td><td>0.623</td><td>0.395</td><td>1.000</td><td>0.247</td></tr><tr><td>BOTPERCENT</td><td>√</td><td>√</td><td>√</td><td>0.845</td><td>0.865</td><td></td><td>1</td><td>0.731</td><td>0.726</td><td>0.738</td><td>0.714</td></tr><tr><td>ROBERTA</td><td></td><td>√</td><td></td><td>0.755</td><td>0.731</td><td>0.739</td><td>0.724</td><td>0.633</td><td>0.432</td><td>0.955</td><td>0.280</td></tr><tr><td>BOTOMETER</td><td>√</td><td>√</td><td>√</td><td>0.531</td><td>0.531</td><td>0.557</td><td>0.508</td><td>0.755</td><td>0.585</td><td>0.440</td><td>0.873</td></tr><tr><td>BOTBUSTER</td><td>√</td><td>√</td><td></td><td>0.772</td><td>0.812</td><td>1</td><td></td><td>0.627</td><td>0.439</td><td>0.882</td><td>0.292</td></tr><tr><td>LOBO</td><td>√</td><td>V</td><td></td><td>0.762</td><td>0.806</td><td>0.748</td><td>0.878</td><td>0.552</td><td>0.198</td><td>0.944</td><td>0.110</td></tr><tr><td>RGT</td><td>√</td><td>√</td><td>√</td><td>0.866</td><td>0.880</td><td>0.852</td><td>0.911</td><td>0.509</td><td>0.509</td><td>0.323</td><td>0.854</td></tr><tr><td colspan="10">Bot detection with MISTRAL-7B</td><td></td><td></td></tr><tr><td>METADATA</td><td>√</td><td></td><td></td><td>0.551</td><td>0.509</td><td>0.624</td><td>0.430</td><td>0.532</td><td>0.201</td><td>0.690</td><td>0.118</td></tr><tr><td>TEXT</td><td></td><td>√</td><td></td><td>0.491</td><td>0.398</td><td>0.553</td><td>0.311</td><td>0.579</td><td>0.599</td><td>0.558</td><td>0.647</td></tr><tr><td>META+TEXT</td><td>√</td><td>√</td><td></td><td>0.516</td><td>0.481</td><td>0.572</td><td>0.414</td><td>0.556</td><td>0.478</td><td>0.580</td><td>0.406</td></tr><tr><td>STRUCT-RAND</td><td>√</td><td>√</td><td>√</td><td>0.570</td><td>0.568</td><td>0.622</td><td>0.522</td><td>0.609</td><td>0.678</td><td>0.576</td><td>0.824</td></tr><tr><td>STRUCT-ATT</td><td>√</td><td>√</td><td>√</td><td>0.583</td><td>0.578</td><td>0.640</td><td>0.527</td><td>0.603</td><td>0.662</td><td>0.576</td><td>0.777</td></tr><tr><td>ENSEMBLE</td><td>√</td><td>√</td><td>√</td><td>0.609</td><td>0.573</td><td>0.699</td><td>0.486</td><td>0.582</td><td>0.533</td><td>0.605</td><td>0.477</td></tr><tr><td colspan="10">Bot detection with LLAMA2-70B</td><td></td><td></td></tr><tr><td>METADATA</td><td>V</td><td></td><td></td><td>0.727</td><td>0.741</td><td>0.762</td><td>0.720</td><td>0.627</td><td>0.713</td><td>0.581</td><td>0.924</td></tr><tr><td>TEXT</td><td></td><td>V</td><td></td><td>0.539</td><td>0.585</td><td>0.570</td><td>0.600</td><td>0.574</td><td>0.617</td><td>0.560</td><td>0.689</td></tr><tr><td>META+TEXT</td><td>√</td><td>√</td><td></td><td>0.689</td><td>0.712</td><td>0.712</td><td>0.711</td><td>0.679</td><td>0.731</td><td>0.630</td><td>0.871</td></tr><tr><td>STRUCT-RAND</td><td>√</td><td>√</td><td>√</td><td>0.591</td><td>0.577</td><td>0.655</td><td>0.516</td><td>0.639</td><td>0.637</td><td>0.639</td><td>0.635</td></tr><tr><td>STRUCT-ATT</td><td>√</td><td>√</td><td>√</td><td>0.602</td><td>0.571</td><td>0.684</td><td>0.491</td><td>0.624</td><td>0.622</td><td>0.639</td><td>0.606</td></tr><tr><td>ENSEMBLE</td><td>V</td><td>√</td><td>√</td><td>0.661</td><td>0.659</td><td>0.723</td><td>0.605</td><td>0.668</td><td>0.685</td><td>0.651</td><td>0.724</td></tr><tr><td colspan="10">Bot detection with CHATGPT</td><td></td><td></td></tr><tr><td>METADATA</td><td>V</td><td></td><td></td><td>0.766</td><td>0.793</td><td>0.742</td><td>0.852</td><td>0.659</td><td>0.698</td><td>0.626</td><td>0.788</td></tr><tr><td>TEXT</td><td></td><td>√</td><td></td><td>0.566</td><td>0.576</td><td>0.612</td><td>0.544</td><td>0.688</td><td>0.684</td><td>0.705</td><td>0.665</td></tr><tr><td>META+TEXT</td><td>√</td><td>√</td><td></td><td>0.656</td><td>0.694</td><td>0.755</td><td>0.642</td><td>0.659</td><td>0.681</td><td>0.607</td><td>0.777</td></tr><tr><td>STRUCT-RAND</td><td>√</td><td>√</td><td>√</td><td>0.577</td><td>0.460</td><td>0.745</td><td>0.333</td><td>0.638</td><td>0.514</td><td>0.783</td><td>0.382</td></tr><tr><td>STRUCT-ATT</td><td>√</td><td>V</td><td>√</td><td>0.565</td><td>0.426</td><td>0.743</td><td>0.298</td><td>0.632</td><td>0.500</td><td>0.792</td><td>0.365</td></tr><tr><td>ENSEMBLE</td><td>√</td><td>√</td><td>√</td><td>0.632</td><td>0.557</td><td>0.801</td><td>0.427</td><td>0.735</td><td>0.706</td><td>0.794</td><td>0.635</td></tr><tr><td colspan="10">Bot detection with CHATGPT and instruction tuning</td><td></td><td></td></tr><tr><td>METADATA</td><td>√</td><td></td><td></td><td>0.812</td><td>0.806</td><td>0.814</td><td>0.847</td><td>0.724</td><td>0.764</td><td>0.667</td><td>0.894</td></tr><tr><td>TEXT</td><td></td><td>√</td><td></td><td>0.767</td><td>0.791</td><td>0.768</td><td>0.816</td><td>0.727</td><td>0.766</td><td>0.670</td><td>0.894</td></tr><tr><td>META+TEXT</td><td>√</td><td>√</td><td></td><td>0.862</td><td>0.865</td><td>0.813 0.839</td><td>0.924 0.980</td><td>0.721 0.718</td><td>0.758 0.761</td><td>0.668 0.660</td><td>0.877 0.900</td></tr><tr><td>STRUCT-RAND STRUCT-ATT</td><td>√ √</td><td>√ √</td><td>√ √</td><td>0.890 0.885</td><td>0.904 0.888</td></table>

Table 1: Performance of baselines and LLM-based bot detectors on Twibot-20 and Twibot-22. Prec. and Rec. indicates precision and recall. , , and indicate whether metadata, texts, or neighborhoods are leveraged in this approach. LLM-based bot detectors with instruction tuning achieve state-of-the-art results on both datasets.

A combination of modality-specific LLMs yields promising results. For ChatGPT with instruction tuning, while the text-only detector trails in performance and LLMs are better in leveraging the structural information of accounts, an ensemble of modality-specific predictions through majority voting improves performance. This echoes the finding that not all modalities of a bot account are malicious (Liu et al., 2023) and our proposed mixtureof-heterogeneous-experts framework jointly considers multiple user information modalities.

LLMs are worth the tradeoff between compute and data annotations. While existing supervised approaches are lightweight and inexpensive to run, they are trained on large quantities of annotated accounts (around 8k and 700k for the two datasets). On the contrary, while LLM-based approaches require significant computational resources, they are only instruction-tuned on 1k annotated users and achieve superior results. We argue that LLM-based bot detectors are thus promising approaches, given that data annotations in bot detection are hard, noisy, and scarce (Feng et al., 2021a), while the compute overhead will be continuously reduced due to innovations in efficient training and inference (Dao, 2023; Dettmers et al., 2024).

## 4.2 Risks

We evaluate existing detectors and LLM-based approaches on the LLM-manipulated bot accounts in

<table><tr><td rowspan="2">Strategy</td><td colspan="2">BotPercent</td><td colspan="2">BotRGCN</td><td colspan="2">Text+Meta</td><td colspan="2">Struct-Rand</td><td colspan="2">Struct-Att</td><td colspan="2">Ensemble</td></tr><tr><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td></tr><tr><td>vanilla Twibot-20</td><td>.755</td><td>.731</td><td>.737</td><td>.766</td><td>.862</td><td>.865</td><td>.890</td><td>.904</td><td>.884</td><td>.888</td><td>.899</td><td>.915</td></tr><tr><td colspan="9">Manipulation strategies with LLAMA2-70B</td><td></td><td></td><td></td></tr><tr><td>ZERO-SHOT REWRITE</td><td>.716</td><td>.724</td><td>.735</td><td>.788</td><td>.859</td><td>.874</td><td>.889</td><td>.905</td><td>.867</td><td>.871</td><td>.885</td><td>.901</td></tr><tr><td>FEW-SHOT REWRITE</td><td>.689</td><td>.720</td><td>.732</td><td>.784</td><td>.862</td><td>.878</td><td>.886</td><td>.902</td><td>.852</td><td>.867</td><td>.883</td><td>.898</td></tr><tr><td>CLASSIFIER GUIDE</td><td>.650</td><td>.704</td><td>.722</td><td>.779</td><td>.835</td><td>.852</td><td>.868</td><td>.886</td><td>.805</td><td>.818</td><td>.850</td><td>.870</td></tr><tr><td>TEXT ATTRIBUTE</td><td>.689</td><td>.737</td><td>.728</td><td>.787</td><td>.872</td><td>.887</td><td>.890</td><td>.906</td><td>.881</td><td>.895</td><td>.891</td><td>.907</td></tr><tr><td>ADD NEIGHBOR</td><td>1</td><td>1</td><td>.731</td><td>.785</td><td>1</td><td>1</td><td>.874</td><td>.890</td><td>.855</td><td>.869</td><td>.867</td><td>.885</td></tr><tr><td>REMOVE NEIGHBOR</td><td>1</td><td>1</td><td>.653</td><td>.721</td><td>1</td><td>1</td><td>.863</td><td>.882</td><td>.862</td><td>.878</td><td>.863</td><td>.882</td></tr><tr><td>COMBINE NEIGHBOR</td><td>1</td><td>1</td><td>.596</td><td>.539</td><td>1</td><td>1</td><td>.866</td><td>.883</td><td>.859</td><td>.873</td><td>.868</td><td>.885</td></tr><tr><td>SELECTIVE COMBINE</td><td>.691</td><td>.737</td><td>.684</td><td>.663</td><td>.866</td><td>.883</td><td>.866</td><td>.884</td><td>.860</td><td>.875</td><td>.865</td><td>.884</td></tr><tr><td>BOTH COMBINE</td><td>.650</td><td>.704</td><td>.571</td><td>.564</td><td>.835</td><td>.852</td><td>.854</td><td>.871</td><td>.808</td><td>.822</td><td>.850</td><td>.869</td></tr><tr><td colspan="9">Manipulation strategies with CHATGPT</td><td></td><td></td><td></td></tr><tr><td>ZERO-SHOT REWRITE</td><td>.680</td><td>.731</td><td>.719</td><td>.745</td><td>.875</td><td>.891</td><td>.891</td><td>.907</td><td>.894</td><td>.907</td><td>.896</td><td>.911</td></tr><tr><td>FEW-SHOT REWRITE</td><td>.675</td><td>.724</td><td>.708</td><td>.738</td><td>.879</td><td>.894</td><td>.889</td><td>.905</td><td>.887</td><td>.901</td><td>.890</td><td>.906</td></tr><tr><td>CLASSIFIER GUIDE</td><td>.649</td><td>.699</td><td>.702</td><td>.715</td><td>.860</td><td>.878</td><td>.890</td><td>.906</td><td>.888</td><td>.903</td><td>.886</td><td>.903</td></tr><tr><td>TEXT ATTRIBUTE</td><td>.661</td><td>.716</td><td>.716</td><td>.752</td><td>.855</td><td>.870</td><td>.882</td><td>.899</td><td>.879</td><td>.894</td><td>.877</td><td>.895</td></tr><tr><td>ADD NEIGHBOR</td><td>1</td><td>1</td><td>.715</td><td>.741</td><td>1</td><td>1</td><td>.874</td><td>.892</td><td>.893</td><td>.907</td><td>.879</td><td>.897</td></tr><tr><td>REMOVE NEIGHBOR</td><td>1</td><td>1</td><td>.642</td><td>.629</td><td>1</td><td>1</td><td>.870</td><td>.888</td><td>.855</td><td>.870</td><td>.864</td><td>.883</td></tr><tr><td>COMBINE NEIGHBOR</td><td>1</td><td>1</td><td>.632</td><td>.685</td><td>1</td><td>1</td><td>.878</td><td>.895</td><td>.893</td><td>.907</td><td>.878</td><td>.896</td></tr><tr><td>SELECTIVE COMBINE</td><td>.678</td><td>.725</td><td>.615</td><td>.638</td><td>.864</td><td>.880</td><td>.873</td><td>.891</td><td>.860</td><td>.875</td><td>.873</td><td>.891</td></tr><tr><td>BOTH COMBINE</td><td>.649</td><td>.699</td><td>.641</td><td>.627</td><td>.860</td><td>.878</td><td>.888</td><td>.905</td><td>.905</td><td>.919</td><td>.894</td><td>.910</td></tr></table>

Table 2: Performance of baselines (first two) and LLM-based bot detectors (last four) on manipulated versions of the Twibot-20 dataset. The lowest performances (and hence the greatest drops from vanilla Twibot-20) are in bold. “/” indicates that this graph-based manipulation has no effect on the non-graph detector.

Twibot-20 and present performance in Table 2.

LLM-based detectors are less sensitive to manipulation strategies. While BotPercent and BotRGCN suffer from a 10.9% and 7.7% drop in accuracy on average due to manipulation strategies, LLM-ensemble only shows a 2.3% drop. In addition, ChatGPT-based detectors are less robust to edits by another LLM (LLaMA2-70B) than itself, suggesting that LLMs might be able to identify artifacts generated by themselves (Pu et al., 2023).

Classifier guidance is the most successful among textual manipulations. On average, classifier guidance achieved a 6.0% and 3.2% drop in accuracy and F1-score. This indicates that LLMs could iteratively refine generations based on feedback from an external classifier; we further investigate the LLM-classifier interaction in Section 4.

Removing neighbors is better than adding. The two strategies achieve 5.0% and 2.5% drops in accuracy on average, respectively: we hypothesize that while suggesting a new account to follow from five accounts is a noisy task, removing one of the existing followings that makes the bot seem malicious is more straightforward and effective. Combining the removals and additions only led to performance drops in 5 of the 16 scenarios, suggesting that strategically following accounts is harder for existing LLMs.

## 5 Analysis

Model Calibration Robust social bot detectors should provide not only a binary prediction but also a well-calibrated confidence score to facilitate content moderation. We evaluate how well are LLMbased bot detectors calibrated, with the vanilla Twibot-20 dataset as well as manipulated with the BOTH COMBINE strategy, in Figure 2. Specifically, we use the probability of the prediction token (“human” or “bot”) from the instruction-tuned ChatGPT models as the bot likelihood, bin it into 10 buckets, and calculate the estimated calibration error (ECE) (Guo et al., 2017). It is demonstrated that LLM-based bot detectors are moderately calibrated with an ECE of around 0.2, while LLM-guided manipulation strategies harm calibration and increase ECE by 28.4% on average. As a result, the risks of LLMs in social bot detection not only lie in decreased performance but also in less calibrated and thus less trustworthy predictions.

Text Rewrite Similarity To evade detection, it would be most effective if LLM removed all malicious content/intent in the bot-generated posts: however, that would defeat the purpose of LLMguided bot design. Following previous works (Li et al., 2023a; Kim et al., 2023a), we employ GPT-4 to evaluate whether the LLM-paraphrased bot posts still “preserve” the potentially malicious content. Specifically, we prompt GPT-4 with “For thefollowing two posts of social media users, how similar are they in content?” and solicit a response on a 4-point Likert scale from “1: very different” to “4: very similar”. Figure 3 demonstrates that LLMs are generally preserving the content of bot posts, while the text attribute strategy is most faithful.

![](images/b994861ebe86b5d0d80469b7ae4e661e054b5ef3df74a09828af059bb2a4ab4e.jpg)

Figure 2: Calibration of LLM-based bot detectors with the original Twibot-20 dataset as well as the manipulated version with BOTH COMBINE. ECE denotes estimated calibration error, the lower the better. The dashed line indicates perfect calibration, while the color of the bar is darker when it is closer to perfect calibration.  
![](images/20780a2f1d2d5b3d8dbd2a8d556b2f82e6a9785f4deea800344d4b9db5e5f69d.jpg)  
Figure 3: GPT-4 Evaluation of whether the LLMparaphrased bot post is similar to the original post in content, from “very different” as 1 to “very similar” as 4. We present the average value and standard deviation.

![](images/3050ed35930569a79aee6a7b627e21388bf369e6bf824a5528a19bc1fe395da4.jpg)  
Figure 4: The trend of bot likelihood scores given by the external classifier in the CLASSIFIER GUIDANCE strategy of paraphrasing bot posts.

Classifier Guidance Convergence Section 4.2 demonstrates that classifier guidance is the most effective approach among text-based manipulations, showcasing the potential of LLMs iteratively refining generations based on feedback from external classifiers, but with increased inference latency. We further investigate the trend of bot scores given by the external classifier along with the five iterations in Figure 4: It is demonstrated that the bot scores do steadily decrease through iterations, while Chat-GPT is more effective than LLaMA2-70B.

Statistics of Added/Removed Neighbors LLMguided additions/removals of bot neighbors are also successful in compromising existing bot detectors: we investigate the statistics of the removed/added accounts in Figure 5. It is demonstrated that LLMs do not simply follow established heuristics, such as “follow accounts with a lot of followers to seem genuine”, but rather examine in a case-by-case manner and suggest diverse edits of bot neighborhood.

\# of In-Context Examples We investigate the impact of in-context examples in LLM-based bot detectors by increasing the amount from 0 to 16 and present model performance in Figure 6: Performance steadily increases with the amount of in-context examples. However, the context length limit of LLMs sets an upper bound of the amount of in-context examples: future work might explore whether long/infinite-context LLMs (Chen et al., 2023b; Bertsch et al., 2024) might benefit from a growing amount of in-context examples.

![](images/cdea398c98379b49144208780e2715a900e5a5abadf07470741131b30541646c.jpg)

![](images/426bb9c715568d988003fa33d9b6256ba870f475d22a6a290363b311c27959a3.jpg)  
Figure 5: Distributions of accounts’ metadata that are selected by LLMs to be added/removed from a bot account’s following list.

## 6 Related Work

Social Media Bot Detection Existing social media bot detection methods fall into three categories: feature-, text-, and graph-based (Feng et al., 2022b). Feature-based methods extract features from users metadata (Yang et al., 2020; Kudugunta and Ferrara, 2018), tweets (Miller et al., 2014), description (Hayawi et al., 2022), temporal patterns (Mazza et al., 2019), and follow relationships (Feng et al., 2021a) for feature engineering. Text-based models mine user-generated content such as tweets and descriptions using NLP techniques, including word embeddings (Wei and Nguyen, 2019), RNN (Kudugunta and Ferrara, 2018), attention mechanism (Feng et al., 2021a), and pretrained language models (Dukic et al.´ , 2020). Graph-based methods focus on modeling user interactions in social networks and achieve state-of-the-art bot detection performance, approaches including node centrality (Dehghan et al., 2023), node representation learning (Pham et al., 2022), graph neural networks (Feng et al., 2021c, 2022a), and mixture-of-expert (Liu et al., 2023; Tan et al., 2023). As LLMs are revolutionizing text and graph mining on social networks (Tan and Jiang, 2023; Jin et al., 2023), we are the first to explore the opportunities and risks of LLMs in social bot detection.

LLMs for Content Moderation Aside from advancing on standard NLP tasks and benchmarks, LLMs have also shown great potential for various scenarios of content moderation (Kumar et al., 2023a; Ziems et al., 2023; Ma et al., 2023). LLMs have been widely employed to detect and counter hate speech (Jiang et al., 2023b; Vishwamitra et al.,

![](images/d44f899d58d8443d3b867ec9fc1ddddaaca4951ae8dff72d9b88cba45f1738c9.jpg)  
Figure 6: Performance of LLM-based bot detectors on Twibot-20 when the number of in-context examples increases from 0 to 16.

2024; Pendzel et al., 2023; Van and Wu, 2023; Nasir et al., 2023; Agarwal et al., 2023; Roy et al., 2023; Mendelsohn et al., 2023), with existing works focusing on improving their reasoning and robustness (Yang et al., 2023; Roy et al., 2023), mitigating LLMs’ social biases (Zhang et al., 2023; Mun et al., 2023), enhancing LLMs for machinegenerated hate speech in adversarial settings (Kim et al., 2023b; Sen et al., 2023; Ocampo et al., 2023), as well as employing LLMs for explainability (Wang et al., 2023; Huang et al., 2023a). LLM-based solutions have also been proposed for misinformation detection (Jiang et al., 2024; Pelrine et al., 2023; Hu et al., 2024; Nakshatri et al., 2023; Sundriyal et al., 2023; Su et al., 2023a; Li et al., 2023c; Chen et al., 2023a; Choi and Ferrara, 2024; Wang and Shu, 2023; Leite et al., 2023; Vykopal et al., 2023), with a focus on detecting machine-generated fake news (Huang et al., 2023c; Pan et al., 2023; Su et al., 2023b; Xu et al., 2023; Chen and Shu, 2023) and in adversarial settings (Han et al., 2023; Lucas et al., 2023; Wu and Hooi, 2023). In this work, we investigate LLMs’ opportunities and risks in social bot detection (Luceri et al., 2024), highlighting the potential of LLMs as stateof-the-art bot detectors as well as the dual-use risks for designing advanced and evasive social bots.

## 7 Conclusion

We propose to investigate the opportunities and risks of LLMs in social media bot detection. As promising opportunities, we propose a mixture-ofheterogeneous-experts framework to adapt LLMs for bot detection through in-context learning or instruction tuning. As tangible risks, we propose text- and graph-based strategies to manipulate the information of bot accounts with the help of LLMs aiming to evade detection. Extensive experiments demonstrate that LLM-based bot detectors achieve state-of-the-art performance on two widely adopted bot detection datasets, but it is easier than ever to deploy an adversarial LLM-based bot that successfully evades detection, especially for existing non-LLM social bot detection models.

## Limitations

While our proposed LLM-based bot detectors and LLM-guided bot manipulations are generic and platform-agnostic, the experiments in this work focus primarily on the Twitter/X platform. This is due to the availability of annotated social media data while we expect to expand our experiments and analysis to other social media platforms such as TikTok, Reddit, and more, in future work.

We employ Twibot-20 and Twibot-22, two widely adopted datasets collected in and before 2022, to evaluate our proposed detectors and manipulation strategies. However, social media bot accounts are constantly evolving to evade detection (Cresci et al., 2017): we could not experiment with more up-to-date bot accounts again due to data availability, for example, the X platform has cancelled its academic research API access. We hope to test out LLM-based detectors and manipulation strategies with more up-to-date data with research access to social media data.

## Ethics Statement

The adversarial nature of social bot detection involves threat modeling and the development of evasive bots. This research is essential to model LLM risks and develop defense measures, while it also increases the risks of dual-use. We as authors aim to mitigate such dual use by employing controlled access to the social media data and trained models, ensuring that it is only employed for research purposes.

Language models have been extensively documented to have inherent social biases (Blodgett et al., 2020; Jin et al., 2021; Bender et al., 2021; Shaikh et al., 2023), and such biases could have an impact on downstream tasks such as hate speech detection (Xia et al., 2020) and misinformation (Feng et al., 2023). We expect social media bot detection to be no exception. We hypothesize that LLM-based bot detectors might underserve certain users and communities, potentially informed by LLMs’ internal biases, stereotypes, and spurious correlations. We argue that the decisions of LLM-based bot detectors should be interpreted as an initial screening of malicious accounts, while content moderation decisions should be made with humans in the loop. Future work could also investigate the fairness implications of social media bot detectors based on LLMs and other machine learning models.

## Acknowledgements

This material is based upon work supported by the National Science Foundation under CAREER Grant No. IIS2142739, NSF Grant No. IIS2203097, and the DARPA CMO under Contract No. HR001120C0124. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the funding agencies. We also gratefully acknowledge support from Alfred P. Sloan Foundation Fellowship.

## References

Vibhor Agarwal, Yu Chen, and Nishanth Sastry. 2023. Haterephrase: Zero-and few-shot reduction of hate intensity in online posts using large language models. arXiv preprint arXiv:2310.13985.

Wasim Ahmed, Francesc López Seguí, Josep Vidal-Alaball, and Matthew S Katz. 2020. Covid-19 and the “film your hospital” conspiracy theory: Social network analysis of twitter data. Journal ofmedical Internet research, 22(10):e22374.

Emily M Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. 2021. On the dangers of stochastic parrots: Can language models be too big? In Proceedings ofthe 2021 ACM conference on fairness, accountability, and transparency, pages 610–623.

Amanda Bertsch, Uri Alon, Graham Neubig, and Matthew Gormley. 2024. Unlimiformer: Long-range transformers with unlimited length input. Advances in Neural Information Processing Systems, 36.

Su Lin Blodgett, Solon Barocas, Hal Daumé III, and Hanna Wallach. 2020. Language (technology) is power: A critical survey of “bias” in NLP. In Proceedings of the 58th Annual Meeting of the Associationfor Computational Linguistics, pages 5454– 5476, Online. Association for Computational Linguistics.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda

Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Zijian Cai, Zhaoxuan Tan, Zhenyu Lei, Zifeng Zhu, Hongrui Wang, Qinghua Zheng, and Minnan Luo. 2024. Lmbot: distilling graph knowledge into language model for graph-less deployment in twitter bot detection. In Proceedings ofthe 17th ACM International Conference on Web Search and Data Mining, pages 57–66.

Canyu Chen and Kai Shu. 2023. Can llm-generated misinformation be detected? In The Twelfth International Conference on Learning Representations.

Mengyang Chen, Lingwei Wei, Han Cao, Wei Zhou, and Songlin Hu. 2023a. Can large language models understand content and propagation for misinformation detection: An empirical study. arXiv preprint arXiv:2311.12699.

Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia. 2023b. Longlora: Efficient fine-tuning of long-context large language models. arXiv preprint arXiv:2309.12307.

Eun Cheol Choi and Emilio Ferrara. 2024. Automated claim matching with large language models: empowering fact-checkers in the fight against misinformation. In Companion Proceedings of the ACM on Web Conference 2024, pages 1441–1449.

Stefano Cresci. 2020. A decade of social bot detection. Communications ofthe ACM, 63(10):72–83.

Stefano Cresci, Roberto Di Pietro, Marinella Petrocchi, Angelo Spognardi, and Maurizio Tesconi. 2017. The paradigm-shift of social spambots: Evidence, theories, and tools for the arms race. In Proceedings of the 26th international conference on world wide web companion, pages 963–972.

Tri Dao. 2023. Flashattention-2: Faster attention with better parallelism and work partitioning. In The Twelfth International Conference on Learning Representations.

Ashkan Dehghan, Kinga Siuta, Agata Skorupka, Akshat Dubey, Andrei Betlen, David Miller, Wei Xu, Bogumił Kaminski, and Paweł Prałat. 2023. Detect-´ ing bots in social-networks using node and structural embeddings. Journal of Big Data, 10(1):119.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2024. Qlora: Efficient finetuning of quantized llms. Advances in Neural Information Processing Systems, 36.

Jesse Dodge, Maarten Sap, Ana Marasovic, William´ Agnew, Gabriel Ilharco, Dirk Groeneveld, Margaret Mitchell, and Matt Gardner. 2021. Documenting large webtext corpora: A case study on the colossal clean crawled corpus. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 1286–1305, Online and

Punta Cana, Dominican Republic. Association for Computational Linguistics.

David Dukic, Dominik Ke´ ca, and Dominik Stipiˇ c. 2020.´ Are you human? detecting bots on twitter using bert. In 2020 IEEE 7th International Conference on Data Science and Advanced Analytics (DSAA), pages 631– 636. IEEE.

Juan Echeverrï£¡ a, Emiliano De Cristofaro, Nicolas Kourtellis, Ilias Leontiadis, Gianluca Stringhini, and Shi Zhou. 2018. Lobo: Evaluation of generalization deficiencies in twitter bot classifiers. In Proceedings of the 34th annual computer security applications conference, pages 137–146.

Shangbin Feng, Chan Young Park, Yuhan Liu, and Yulia Tsvetkov. 2023. From pretraining data to language models to downstream tasks: Tracking the trails of political biases leading to unfair NLP models. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 11737–11762, Toronto, Canada. Association for Computational Linguistics.

Shangbin Feng, Zhaoxuan Tan, Rui Li, and Minnan Luo. 2022a. Heterogeneity-aware twitter bot detection with relational graph transformers. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 3977–3985.

Shangbin Feng, Zhaoxuan Tan, Herun Wan, Ningnan Wang, Zilong Chen, Binchi Zhang, Qinghua Zheng, Wenqian Zhang, Zhenyu Lei, Shujie Yang, et al. 2022b. Twibot-22: Towards graph-based twitter bot detection. Advances in Neural Information Processing Systems, 35:35254–35269.

Shangbin Feng, Herun Wan, Ningnan Wang, Jundong Li, and Minnan Luo. 2021a. Satar: A self-supervised approach to twitter account representation learning and its application in bot detection. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management, pages 3808–3817.

Shangbin Feng, Herun Wan, Ningnan Wang, Jundong Li, and Minnan Luo. 2021b. Twibot-20: A comprehensive twitter bot detection benchmark. In Proceedings ofthe 30th ACM International Conference on Information & Knowledge Management, pages 4485–4494.

Shangbin Feng, Herun Wan, Ningnan Wang, and Minnan Luo. 2021c. Botrgcn: Twitter bot detection with relational graph convolutional networks. In Proceedings ofthe 2021 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining, pages 236–239.

Emilio Ferrara. 2020. What types of covid-19 conspiracies are populated by twitter bots? First Monday.

Emilio Ferrara, Wen-Qiang Wang, Onur Varol, Alessandro Flammini, and Aram Galstyan. 2016. Predicting online extremism, content adopters, and interaction reciprocity. In Social Informatics: 8th International Conference, SocInfo 2016, Bellevue, WA, USA,

November 11-14, 2016, Proceedings, Part II 8, pages 22–39. Springer.

Tamar Ginossar, Iain J Cruickshank, Elena Zheleva, Jason Sulskis, and Tanya Berger-Wolf. 2022. Crossplatform spread: vaccine-related content, sources, and conspiracy theories in youtube videos shared in early twitter covid-19 conversations. Human vaccines & immunotherapeutics, 18(1):1–13.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. 2017. On calibration of modern neural networks. In International conference on machine learning, pages 1321–1330. PMLR.

Tianyu Han, Sven Nebelung, Firas Khader, Tianci Wang, Gustav Mueller-Franzes, Christiane Kuhl, Sebastian Försch, Jens Kleesiek, Christoph Haarburger, Keno K Bressem, et al. 2023. Medical foundation models are susceptible to targeted misinformation attacks. arXiv preprint arXiv:2309.17007.

Kadhim Hayawi, Sujith Mathew, Neethu Venugopal, Mohammad M Masud, and Pin-Han Ho. 2022. Deeprobot: a hybrid deep neural network model for social bot detection based on user profile data. Social Network Analysis and Mining, 12(1):43.

Philip N Howard, Bence Kollanyi, and Samuel Woolley. 2016. Bots and automation over twitter during the us election. Computational propaganda project: Working paper series, 21(8).

Beizhe Hu, Qiang Sheng, Juan Cao, Yuhui Shi, Yang Li, Danding Wang, and Peng Qi. 2024. Bad actor, good advisor: Exploring the role of large language models in fake news detection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 22105–22113.

Fan Huang, Haewoon Kwak, and Jisun An. 2023a. Is chatgpt better than human annotators? potential and limitations of chatgpt in explaining implicit hate speech. In Companion proceedings ofthe ACM web conference 2023, pages 294–297.

Jin Huang, Xingjian Zhang, Qiaozhu Mei, and Jiaqi Ma. 2023b. Can llms effectively leverage graph structural information: When and why. In NeurIPS 2023 Workshop: New Frontiers in Graph Learning.

Kung-Hsiang Huang, Kathleen McKeown, Preslav Nakov, Yejin Choi, and Heng Ji. 2023c. Faking fake news for real fake news detection: Propagandaloaded training data generation. In Proceedings of the 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers).

Zhen Huang, Zhilong Lv, Xiaoyun Han, Binyang Li, Menglong Lu, and Dongsheng Li. 2022. Social botaware graph neural network for early rumor detection. In Proceedings ofthe 29th International Conference on Computational Linguistics, pages 6680– 6690, Gyeongju, Republic of Korea. International Committee on Computational Linguistics.

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023a. Mistral 7b. arXiv preprint arXiv:2310.06825.

Bohan Jiang, Zhen Tan, Ayushi Nirmal, and Huan Liu. 2024. Disinformation detection: An evolving challenge in the age of llms. In Proceedings of the 2024 SIAM International Conference on Data Mining (SDM), pages 427–435. SIAM.

Shuyu Jiang, Wenyi Tang, Xingshu Chen, Rui Tanga, Haizhou Wang, and Wenxian Wang. 2023b. Raucg: Retrieval-augmented unsupervised counter narrative generation for hate speech. arXiv preprint arXiv:2310.05650.

Bowen Jin, Gang Liu, Chi Han, Meng Jiang, Heng Ji, and Jiawei Han. 2023. Large language models on graphs: A comprehensive survey. arXiv preprint arXiv:2312.02783.

Xisen Jin, Francesco Barbieri, Brendan Kennedy, Aida Mostafazadeh Davani, Leonardo Neves, and Xiang Ren. 2021. On transferability of bias mitigation effects in language model fine-tuning. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3770–3783, Online. Association for Computational Linguistics.

Seungone Kim, Jamin Shin, Yejin Cho, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, Seongjin Shin, Sungdong Kim, James Thorne, et al. 2023a. Prometheus: Inducing fine-grained evaluation capability in language models. arXiv preprint arXiv:2310.08491.

Youngwook Kim, Shinwoo Park, Youngsoo Namgoong, and Yo-Sub Han. 2023b. Conprompt: Pre-training a language model with machine-generated data for implicit hate speech detection. In Findings of the Associationfor Computational Linguistics: EMNLP 2023, pages 10964–10980.

Sneha Kudugunta and Emilio Ferrara. 2018. Deep neural networks for bot detection. Information Sciences, 467:312–322.

Deepak Kumar, Yousef AbuHashem, and Zakir Durumeric. 2023a. Watch your language: Large language models and content moderation. arXiv preprint arXiv:2309.14517.

Sachin Kumar, Vidhisha Balachandran, Lucille Njoo, Antonios Anastasopoulos, and Yulia Tsvetkov. 2023b. Language generation models can cause harm: So what can we do about it? an actionable survey. In Proceedings ofthe 17th Conference ofthe European Chapter of the Association for Computational Linguistics, pages 3299–3321, Dubrovnik, Croatia. Association for Computational Linguistics.

Zhenyu Lei, Herun Wan, Wenqian Zhang, Shangbin Feng, Zilong Chen, Jundong Li, Qinghua Zheng, and

Minnan Luo. 2023. BIC: Twitter bot detection with text-graph interaction and semantic consistency. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 10326–10340, Toronto, Canada. Association for Computational Linguistics.

João A Leite, Olesya Razuvayevskaya, Kalina Bontcheva, and Carolina Scarton. 2023. Detecting misinformation with llm-predicted credibility signals and weak supervision. arXiv preprint arXiv:2309.07601.

Ruosen Li, Teerth Patel, and Xinya Du. 2023a. Prd: Peer rank and discussion improve large language model based evaluations. arXiv preprint arXiv:2307.02762.

Shilong Li, Boyu Qiao, Kun Li, Qianqian Lu, Meng Lin, and Wei Zhou. 2023b. Multi-modal social bot detection: Learning homophilic and heterophilic connections adaptively. In Proceedings ofthe 31st ACM International Conference on Multimedia, pages 3908– 3916.

Zizhong Li, Haopeng Zhang, and Jiawei Zhang. 2023c. A revisit of fake news dataset with augmented fact-checking by chatgpt. arXiv preprint arXiv:2312.11870.

Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, et al. 2023. Holistic evaluation of language models. Transactions on Machine Learning Research.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.

Yuhan Liu, Zhaoxuan Tan, Heng Wang, Shangbin Feng, Qinghua Zheng, and Minnan Luo. 2023. Botmoe: Twitter bot detection with community-aware mixtures of modal-specific experts. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’23, page 485–495, New York, NY, USA. Association for Computing Machinery.

Ning Lu, Shengcai Liu, Rui He, and Ke Tang. 2023. Large language models can be guided to evade ai-generated text detection. arXiv preprint arXiv:2305.10847.

Yi-Ju Lu and Cheng-Te Li. 2020. GCAN: Graph-aware co-attention networks for explainable fake news detection on social media. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 505–514, Online. Association for Computational Linguistics.

Jason Lucas, Adaku Uchendu, Michiharu Yamashita, Jooyoung Lee, Shaurya Rohatgi, and Dongwon Lee. 2023. Fighting fire with fire: The dual role of llms in

crafting and detecting elusive disinformation. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 14279– 14305.

Luca Luceri, Eric Boniardi, and Emilio Ferrara. 2024. Leveraging large language models to detect influence campaigns on social media. In Companion Proceedings of the ACM on Web Conference 2024, pages 1459–1467.

Huan Ma, Changqing Zhang, Huazhu Fu, Peilin Zhao, and Bingzhe Wu. 2023. Adapting large language models for content moderation: Pitfalls in data engineering and supervised fine-tuning. arXiv preprint arXiv:2310.03400.

William Marcellino, Madeline Magnuson, Anne Stickells, Benjamin Boudreaux, Todd C Helmus, Edward Geist, and Zev Winkelman. 2020. Counterradicalization bot research.

Michele Mazza, Stefano Cresci, Marco Avvenuti, Walter Quattrociocchi, and Maurizio Tesconi. 2019. Rtbust: Exploiting temporal patterns for botnet detection on twitter. In Proceedings of the 10th ACM conference on web science, pages 183–192.

Julia Mendelsohn, Ronan Le Bras, Yejin Choi, and Maarten Sap. 2023. From dogwhistles to bullhorns: Unveiling coded rhetoric with language models. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers).

Zachary Miller, Brian Dickinson, William Deitrick, Wei Hu, and Alex Hai Wang. 2014. Twitter spammer detection using data stream clustering. Information Sciences, 260:64–73.

Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2022a. MetaICL: Learning to learn in context. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2791–2809, Seattle, United States. Association for Computational Linguistics.

Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2022b. Rethinking the role of demonstrations: What makes in-context learning work? In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pages 11048–11064, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Jimin Mun, Emily Allaway, Akhila Yerukola, Laura Vianna, Sarah-Jane Leslie, and Maarten Sap. 2023. Beyond denouncing hate: Strategies for countering implied biases and stereotypes in language. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 9759–9777.

Nishanth Nakshatri, Siyi Liu, Sihao Chen, Dan Roth, Dan Goldwasser, and Daniel Hopkins. 2023. Using

llm for improving key event discovery: Temporalguided news stream clustering with event summaries. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 4162–4173.

Ahmad Nasir, Aadish Sharma, and Kokil Jaidka. 2023. Llms and finetuning: Benchmarking cross-domain performance for hate speech detection. arXiv preprint arXiv:2310.18964.

Lynnette Hui Xian Ng and Kathleen M Carley. 2023. Botbuster: Multi-platform bot detection using a mixture of experts. In Proceedings ofthe International AAAI Conference on Web and Social Media, volume 17, pages 686–697.

Lynnette Hui Xian Ng, Iain J Cruickshank, and Kathleen M Carley. 2022. Cross-platform information spread during the january 6th capitol riots. Social Network Analysis and Mining, 12(1):133.

Nicolas Ocampo, Elena Cabrio, and Serena Villata. 2023. Playing the part of the sharp bully: Generating adversarial examples for implicit hate speech detection. In Findings of the Association for Computational Linguistics: ACL 2023, pages 2758–2772.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744.

Yikang Pan, Liangming Pan, Wenhu Chen, Preslav Nakov, Min-Yen Kan, and William Wang. 2023. On the risk of misinformation pollution with large language models. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 1389–1403.

Kellin Pelrine, Anne Imouza, Camille Thibault, Meilina Reksoprodjo, Caleb Alexander Gupta, Joel Christoph, Jean-François Godbout, and Reihaneh Rabbany. 2023. Towards reliable misinformation mitigation: Generalization, uncertainty, and gpt-4. In The 2023 Conference on Empirical Methods in Natural Language Processing.

Sagi Pendzel, Tomer Wullach, Amir Adler, and Einat Minkov. 2023. Generative ai for hate speech detection: Evaluation and findings. arXiv preprint arXiv:2311.09993.

Phu Pham, Loan TT Nguyen, Bay Vo, and Unil Yun. 2022. Bot2vec: A general approach of intracommunity oriented representation learning for bot detection in different types of social networks. Information Systems, 103:101771.

Xiao Pu, Jingyu Zhang, Xiaochuang Han, Yulia Tsvetkov, and Tianxing He. 2023. On the zero-shot generalization of machine-generated text detectors. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 4799–4808.

Stephen Robertson, Hugo Zaragoza, et al. 2009. The probabilistic relevance framework: Bm25 and beyond. Foundations and Trends in Information Retrieval, 3(4):333–389.

Sippo Rossi, Matti Rossi, Bikesh Raj Upreti, and Yong Liu. 2020. Detecting political bots on twitter during the 2019 finnish parliamentary election. In Annual Hawaii International Conference on System Sciences, pages 2430–2439. Hawaii International Conference on System Sciences.

Sarthak Roy, Ashish Harshvardhan, Animesh Mukherjee, and Punyajoy Saha. 2023. Probing llms for hate speech detection: strengths and vulnerabilities. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 6116–6128.

Melanie Sclar, Yejin Choi, Yulia Tsvetkov, and Alane Suhr. 2023. Quantifying language models’ sensitivity to spurious features in prompt design or: How i learned to start worrying about prompt formatting. In The Twelfth International Conference on Learning Representations.

Indira Sen, Dennis Assenmacher, Mattia Samory, Isabelle Augenstein, Wil Aalst, and Claudia Wagner. 2023. People make better edits: Measuring the efficacy of llm-generated counterfactually augmented data for harmful language detection. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 10480–10504.

Omar Shaikh, Hongxin Zhang, William Held, Michael Bernstein, and Diyi Yang. 2023. On second thought, let’s not think step by step! bias and toxicity in zeroshot reasoning. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4454–4470, Toronto, Canada. Association for Computational Linguistics.

Jinyan Su, Claire Cardie, and Preslav Nakov. 2023a. Adapting fake news detection to the era of large language models. arXiv preprint arXiv:2311.04917.

Jinyan Su, Terry Yue Zhuo, Jonibek Mansurov, Di Wang, and Preslav Nakov. 2023b. Fake news detectors are biased against texts generated by large language models. arXiv preprint arXiv:2309.08674.

Megha Sundriyal, Tanmoy Chakraborty, and Preslav Nakov. 2023. From chaos to clarity: Claim normalization to empower fact-checking. In Findings of the Associationfor Computational Linguistics: EMNLP 2023.

Zhaoxuan Tan, Shangbin Feng, Melanie Sclar, Herun Wan, Minnan Luo, Yejin Choi, and Yulia Tsvetkov. 2023. Botpercent: Estimating bot populations in twitter communities. In Findings ofthe Association for Computational Linguistics: EMNLP 2023, pages 14295–14312.

Zhaoxuan Tan and Meng Jiang. 2023. User modeling in the era of large language models: Current research

and future directions. IEEE Data Engineering Bulletin.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Minh-Hao Van and Xintao Wu. 2023. Detecting and correcting hate speech in multimodal memes with large visual language model. arXiv preprint arXiv:2311.06737.

Petar Velickoviˇ c, Guillem Cucurull, Arantxa Casanova,´ Adriana Romero, Pietro Liò, and Yoshua Bengio. 2018. Graph attention networks. In International Conference on Learning Representations.

Nishant Vishwamitra, Keyan Guo, Farhan Tajwar Romit, Isabelle Ondracek, Long Cheng, Ziming Zhao, and Hongxin Hu. 2024. Moderating new waves of online hate with chain-of-thought reasoning in large language models. In 2024 IEEE Symposium on Security and Privacy (SP), pages 178–178. IEEE Computer Society.

Ivan Vykopal, Matúš Pikuliak, Ivan Srba, Robert Moro, Dominik Macko, and Maria Bielikova. 2023. Disinformation capabilities of large language models. arXiv preprint arXiv:2311.08838.

Han Wang, Ming Shan Hee, Md Rabiul Awal, Kenny Tsu Wei Choo, and Roy Ka-Wei Lee. 2023. Evaluating gpt-3 generated explanations for hateful content moderation. In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, pages 6255–6263.

Haoran Wang and Kai Shu. 2023. Explainable claim verification via knowledge-grounded reasoning with large language models. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 6288–6304.

Heng Wang, Shangbin Feng, Tianxing He, Zhaoxuan Tan, Xiaochuang Han, and Yulia Tsvetkov. 2024. Can language models solve graph problems in natural language? Advances in Neural Information Processing Systems, 36.

Feng Wei and Uyen Trang Nguyen. 2019. Twitter bot detection using bidirectional long short-term memory neural networks and word embeddings. In 2019 First IEEE International conference on trust, privacy and security in intelligent systems and applications (TPS-ISA), pages 101–109. IEEE.

Laura Weidinger, Jonathan Uesato, Maribeth Rauh, Conor Griffin, Po-Sen Huang, John Mellor, Amelia Glaese, Myra Cheng, Borja Balle, Atoosa Kasirzadeh, et al. 2022. Taxonomy of risks posed by language models. In Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency, pages 214–229.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al. 2019. Huggingface’s transformers: State-ofthe-art natural language processing. arXiv preprint arXiv:1910.03771.

Jiaying Wu and Bryan Hooi. 2023. Fake news in sheep’s clothing: Robust fake news detection against llm-empowered style attacks. arXiv preprint arXiv:2310.10830.

Mengzhou Xia, Anjalie Field, and Yulia Tsvetkov. 2020. Demoting racial bias in hate speech detection. In Proceedings of the Eighth International Workshop on Natural Language Processingfor Social Media, pages 7–14, Online. Association for Computational Linguistics.

Rongwu Xu, Brian S Lin, Shujian Yang, Tianqi Zhang, Weiyan Shi, Tianwei Zhang, Zhixuan Fang, Wei Xu, and Han Qiu. 2023. The earth is flat because...: Investigating llms’ belief towards misinformation via persuasive conversation. arXiv preprint arXiv:2312.09085.

Kai-Cheng Yang, Emilio Ferrara, and Filippo Menczer. 2022. Botometer 101: Social bot practicum for computational social scientists. Journal of Computational Social Science, 5(2):1511–1528.

Kai-Cheng Yang, Onur Varol, Pik-Mai Hui, and Filippo Menczer. 2020. Scalable and generalizable social bot detection through data selection. In Proceedings ofthe AAAI conference on artificial intelligence, volume 34, pages 1096–1103.

Yongjin Yang, Joonkee Kim, Yujin Kim, Namgyu Ho, James Thorne, and Se-Young Yun. 2023. Hare: Explainable hate speech detection with step-by-step reasoning. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 5490–5505.

Sen Ye, Zhaoxuan Tan, Zhenyu Lei, Ruijie He, Hongrui Wang, Qinghua Zheng, and Minnan Luo. 2023. Hofa: Twitter bot detection with homophily-oriented augmentation and frequency adaptive attention. arXiv preprint arXiv:2306.12870.

Zhehao Zhang, Jiaao Chen, and Diyi Yang. 2023. Mitigating biases in hate speech detection from a causal perspective. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 6610– 6625.

Caleb Ziems, Omar Shaikh, Zhehao Zhang, William Held, Jiaao Chen, and Diyi Yang. 2023. Can large language models transform computational social science? Computational Linguistics, pages 1–53.

## A Experiment Details

## A.1 Dataset Details

We employ two widely adopted datasets in social media bot detection, Twibot-20 (Feng et al., 2021b) and Twibot-22 (Feng et al., 2022b), to evaluate LLM-based bot detectors and LLM-based manipulation strategies. For Twibot-20, we employ the original test split of 1,183 accounts for evaluation. For Twibot-22, we employ the test split of 340 accounts employed in Tan et al. (2023). For instruction tuning LLMs on both datasets, we downsample the original training split into 1,000 users due to limits in computational budget.

## A.2 LLM Details

We employ three LLMs to evaluate their opportunities and risks in social bot detection: 1) Mistral-7B, through the MISTRALAI/MISTRAL-7B-INSTRUCT-V0.1 checkpoint publicly available on Huggingface (Wolf et al., 2019); 2) LLaMA2- 70B, through the META-LLAMA/LLAMA-2-70B-CHAT-HF checkpoint publicly available on Huggingface; 3) ChatGPT, through the GPT-3.5- TURBO-INSTRUCT checkpoint with the OpenAI API.

## A.3 Baseline Details

• BIC (Leite et al., 2023) incorporates text and graph modalities in social networks using a textgraph interaction module and models user behavior consistency with a semantic consistency module.

• LMBot (Cai et al., 2024) distills graph knowledge into encoder-only language models with iterative distillation between graph-based social bot detection model and language model.

• SGBot (Yang et al., 2020) is a feature-based method that identifies bots using random forest based on 8 types of user metadata and 12 derived features.

• BotPercent (Tan et al., 2023) is an amalgamation of Twitter bot detection datasets and feature-, text-, and graph-based models to probe the percentage of bot accounts in Twitter communities.

• RoBERTa (Liu et al., 2019) leverages the pretrained language model RoBERTa to encode user tweets and descriptions, then feed them into an MLP classifier.

• Botometer (Yang et al., 2022) is a public website to check the activity of a Twitter account and gives the score of how likely the user is a social bot. Botometer’s classification system leverages more than 1,000 features using available metadata and information extracted from interaction patterns and content.

• BotBuster (Ng and Carley, 2023) is a social bot detection system that processes user metadata and textual information using the mixtureof-expert architecture to adapt to multiple social platforms.

• LOBO (Echeverrï£¡ a et al., 2018) is a featurebased social bot detection method that extracts 26 features and adopts random forest for classification.

• RGT (Feng et al., 2022a) is a graph-based social bot detection method that models the intrinsic influence and relation heterogeneity in social networks.

## A.4 Implementation Details

For in-context learning, we employ 16 in-context examples by default. For account metadata, we employ five entries: follower count, following count, tweet count, verified, and active years as they are most helpful in identifying social bots. For structure-based detectors, we include a maximum of five followers/followings for each account. For classifier guidance, we employ a finetuned ROBERTA-BASE language model (Liu et al., 2019) as the classifier based on user descriptions. For add neighbors, we provide a total of five accounts for LLMs to choose from. The ensemble of LLM-based detectors is a majority vote based on five detectors, metadata, text, meta+text, structurerandom, and structure-attention.

## B Analysis (cont.)

Qualitative Analysis We present qualitative examples of LLM-based paraphrasing of bot posts in Tables 4 to 8. It is demonstrated that LLMs could enrich the context of bot posts to seem genuine and add emojis and attributes (e.g. WSJ reporter) to seem personal and credible. However, LLMs also change the language of the original bot posts while using the word “regular” too often in generated posts. This indicates that while LLM-based textual manipulations could make bot accounts seem more genuine, they also introduce new biases and signals for future bot detectors to leverage.

Fine-tuning MISTRAL-7B We conduct additional experiments to instruction-tune Mistral-7B with parameter-efficient training and present the results in the following table, comparing instruction tuning with in-context learning (ICL) using Mistral-7B. Table 3 echoes the finding with Chat-GPT: Mistral-7B with instruction tuning also significantly boosts its bot detection utility compared to off-the-shelf prompting, albeit the improvements are generally less than ChatGPT.

## C Prompt Format

We provide specific prompt templates for the proposed approaches in Tables 9 to 20. Note that LLMs might be sensitive to varying prompt formats (Sclar et al., 2023) and the most optimal prompt for bot detection might vary for future LLMs.

<table><tr><td>Method</td><td>Acc, 20</td><td>F1, 20</td><td>Precision, 20</td><td>Recall, 20</td><td>Acc, 22</td><td>F1, 22</td><td>Precision, 22</td><td>Recall, 22</td></tr><tr><td>METADATA, ICL</td><td>0.551</td><td>0.509</td><td>0.624</td><td>0.430</td><td>0.532</td><td>0.201</td><td>0.690</td><td>0.118</td></tr><tr><td>METADATA, TUNE</td><td>0.688 (+0.137)</td><td>0.600 (+0.091)</td><td>0.726 (+0.102)</td><td>0.511 (+0.081)</td><td>0.573 (+0.041)</td><td>0.508 (+0.307)</td><td>0.608 (-0.082)</td><td>0.436 (+0.318)</td></tr><tr><td>TEXT, ICL</td><td>0.491</td><td>0.398</td><td>0.553</td><td>0.311</td><td>0.579</td><td>0.599</td><td>0.558</td><td>0.647</td></tr><tr><td>TEXT, TUNE</td><td>0.692 (+0.201)</td><td>0.499 (+0.101)</td><td>0.725 (+0.172)</td><td>0.380 (+0.069)</td><td>0.626 (+0.047)</td><td>0.644 (+0.045)</td><td>0.629 (+0.071)</td><td>0.659 (+0.012)</td></tr><tr><td>META+TEXT, ICL</td><td>0.516</td><td>0.481</td><td>0.572</td><td>0.414</td><td>0.556</td><td>0.478</td><td>0.580</td><td>0.406</td></tr><tr><td>META+TEXT, TUNE</td><td>0.610 (+0.094)</td><td>0.560 (+0.079)</td><td>0.697 (+0.125)</td><td>0.468 (+0.064)</td><td>0.573 (+0.017)</td><td>0.529 (+0.051)</td><td>0.572 (-0.008)</td><td>0.492 (+0.086)</td></tr><tr><td>STRUCT-RAND, ICL</td><td>0.570</td><td>0.568</td><td>0.622</td><td>0.522</td><td>0.609</td><td>0.678</td><td>0.576</td><td>0.824</td></tr><tr><td>STRUCT-RAND, TUNE</td><td>0.821 (+0.251)</td><td>0.750 (+0.182)</td><td>0.816 (+0.194)</td><td>0.694 (+0.172)</td><td>0.681 (+0.072)</td><td>0.721 (+0.043)</td><td>0.652 (+0.076)</td><td>0.806 (-0.018)</td></tr><tr><td>STRUCT-ATT, ICL</td><td>0.583</td><td>0.578</td><td>0.640</td><td>0.527</td><td>0.603</td><td>0.662</td><td>0.576</td><td>0.777</td></tr><tr><td>STRUCT-ATT, TUNE</td><td>0.793 (+0.223)</td><td>0.750 (+0.172)</td><td>0.792 (+0.152)</td><td>0.712 (+0.185)</td><td>0.655 (+0.052)</td><td>0.715 (+0.053)</td><td>0.614 (+0.038)</td><td>0.856 (+0.079)</td></tr><tr><td>ENSEMBLE, ICL</td><td>0.609</td><td>0.573</td><td>0.699</td><td>0.486</td><td>0.582</td><td>0.533</td><td>0.605</td><td>0.477</td></tr><tr><td>ENSEMBLE, TUNE</td><td>0.804 (+0.195)</td><td>0.718 (+0.145)</td><td>0.826 (+0.127)</td><td>0.635 (+0.149)</td><td>0.671 (+0.089)</td><td>0.623 (+0.090)</td><td>0.672 (+0.067)</td><td>0.581 (+0.104)</td></tr></table>

Table 3: Performance of in-context learning and instruction tuning with Mistral-7B.

<table><tr><td rowspan=1 colspan=1>original</td><td rowspan=1 colspan=1>Day 1 Trump supporter. I rode the escalator! Constitutionalist traditionalist conservative. My 1st votewas Reagan! America, family first. #1A #2A #MAGA #KAG</td></tr><tr><td rowspan=1 colspan=1>zero-shot</td><td rowspan=1 colspan=1>I was a proud supporter of President Trump from day one, and I still am! I remember the excitement ofriding the escalator with him at Trump Tower on that historic day. As a constitutionalist and traditionalistconservative, I believe in the principles that have made America great. My first vote was for PresidentReagan, and I&#x27;ve been a strong advocate for limited government and individual freedom ever since. Ibelieve in the importance of putting America and family first, and I&#x27;m passionate about defending ourconstitutional rights, including the First and Second Amendments. #1A #2A #MAGA #KAG</td></tr><tr><td rowspan=1 colspan=1>few-shot</td><td rowspan=1 colspan=1>Ronald Reagan &#x27;84! I was there! My first vote ever, and it was for the Gipper. I&#x27;ve been a Constitutionalistever since, fighting for limited government and individual freedom. I&#x27;m a proud traditionalist, puttingfamily first and standing up for what&#x27;s right. Let&#x27;s make America great again, and keep it that way! #1A#2A #MAGA #KAG</td></tr><tr><td rowspan=1 colspan=1>classifier guide</td><td rowspan=1 colspan=1>I still get chills thinking about the day I rode the escalator with Donald Trump. It was a moment thatdefined my political awakening. As someone who has always believed in the importance of upholdingtraditional conservative values, I knew that I wanted to be a part of this movement. My first vote wascast for the incomparable Ronald Reagan, and I&#x27;ve continued to prioritize the well-being of my familyand our nation. The First and Second Amendments are cornerstones of our democracy, and I proudlysupport them. I&#x27;ve always believed in the importance of upholding the principles that our country wasfounded upon, and I continue to be inspired by the Constitution and its values. #1A #2A #MAGA #KAG</td></tr><tr><td rowspan=1 colspan=1>text attribute</td><td rowspan=1 colspan=1>I am a proud conservative who has always stood by my beliefs. I believe in the importance of family andthe values that America was founded upon. I am a strong supporter of the Second Amendment and theFirst Amendment, and I believe that our country should always put the needs of its citizens first. I havealways been a strong advocate for the rights of individuals and the importance of limited governmentintervention in our lives. My political views have been shaped by great leaders like Ronald Reagan, whoinspired me to stand up for what I believe in and fight for the future of our country. #1A #2A #MAGA#KAG</td></tr></table>

Table 4: Qualitative examples of re-written user posts with varying manipulation strategies.

<table><tr><td colspan="1" rowspan="1">original</td><td colspan="1" rowspan="1">Go listen to our cover of In Your Eyes on Spotify: &lt;link&gt;</td></tr><tr><td colspan="1" rowspan="1">zero-shot</td><td colspan="1" rowspan="1">Hey everyone! I'm super excited to share our cover of In Your Eyes with you all on Spotify! Check itout here: &lt;link&gt; I'd love to hear your thoughts!</td></tr><tr><td colspan="1" rowspan="1">few-shot</td><td colspan="1" rowspan="1">Kelly Hyland here! &lt;emoji&gt;&lt;emoji&gt; Can't get enough of our cover of In Your Eyes? &lt;emoji&gt; Listen toit now on Spotify: &lt;link&gt; &amp; let me know what you think! &lt;emoji&gt;</td></tr><tr><td colspan="1" rowspan="1">classifier guide</td><td colspan="1" rowspan="1">I'm so excited to share our cover of In Your Eyes on Spotify with you all! &lt;emoji&gt; It's a song that has aspecial place in my heart, and I can't wait for you to hear it. &lt;emoji&gt; It's a little different from our usualstyle, but I think it's really something special. &lt;emoji&gt; Give it a listen when you have a chance, and letme know what you think! &lt;emoji&gt; #InYourEyes #CoverSong #Spotify #NewMusic</td></tr><tr><td colspan="1" rowspan="1">text attribute</td><td colspan="1" rowspan="1">Go listen to our cover of In Your Eyes on Spotify: &lt;link&gt;</td></tr><tr><td colspan="2" rowspan="1">original          Money is the anthem of success</td></tr><tr><td colspan="1" rowspan="1">zero-shot</td><td colspan="1" rowspan="1">Hi, I'm Money! &lt;emoji&gt; I'm all about financial success and I'm here to share my knowledge andinsights with you. Whether you're looking to invest, save, or just get better with your finances, I'm hereto help. I believe that money is a powerful tool that can help us achieve our goals and live the life wewant. Let's work together to make our financial dreams a reality! &lt;emoji&gt;&lt;emoji&gt; #financialliteracy#moneymatters #success</td></tr><tr><td colspan="1" rowspan="1">few-shot</td><td colspan="1" rowspan="1">Economic pragmatist with a passion for the cyber world. I'm a WSJ reporter based in London, coveringall things cybersecurity, finance, and intellectual property. When I'm not chasing down a lead, you canfind me at a local coffee shop, sipping on a latte and typing away on my laptop. My views are my own,and my writing reflects my curiosity and fascination with the ever-changing world of cyber. Feel free toreach out to me at jenny.strasburg@wsj.com, and if you have any anonymous tips, send them my wayvia &lt;link&gt; (attn Strasburg). Let's keep the conversation going! &lt;emoji&gt;&lt;emoji&gt;</td></tr><tr><td colspan="1" rowspan="1">classifier guide</td><td colspan="1" rowspan="1">As a young adult, I understand that money is a vital aspect of life, but it's not the only thing thatdefines success. While I enjoy discussing finances and how they can impact our lives, I also believethat true success is about finding a balance between financial stability and personal fulfillment. For me,success means using my resources to make a positive difference in the world and cultivating meaningfulrelationships with others. Money can provide us with opportunities, but it's up to us to use thoseopportunities to live a fulfilling life.</td></tr><tr><td colspan="1" rowspan="1">text attribute</td><td colspan="1" rowspan="1">Money is a crucial element in measuring success, but it shouldn't be the only factor.</td></tr></table>

Table 5: Qualitative examples of re-written user posts with varying manipulation strategies.

Table 6: Qualitative examples of re-written user posts with varying manipulation strategies.

<table><tr><td colspan="1" rowspan="1">original</td><td colspan="1" rowspan="1">Aprendizaje. Pensamiento Crítico. Debate de ideas. Busquemos la verdad, más allá de lo que nospresentan las personas e instituciones. RT≠Endoso.</td></tr><tr><td colspan="1" rowspan="1">zero-shot</td><td colspan="1" rowspan="1">I'm passionate about learning and critical thinking. I love to engage in respectful and constructivedebates about various topics, always seeking the truth beyond what we're presented with. I believe inquestioning everything, even our own beliefs, to deepen our understanding and broaden our perspectives.Feel free to share your thoughts and engage in meaningful conversations with me. Let's explore newideas together!</td></tr><tr><td colspan="1" rowspan="1">few-shot</td><td colspan="1" rowspan="1">&lt;emoji&gt; Curiosity and critical thinking. Let's explore ideas and seek truth beyond what we're presented.Open discussion and debate. RT≠Endorsement.</td></tr><tr><td colspan="1" rowspan="1">classifier guide</td><td colspan="1" rowspan="1">As an inquisitive and open-minded individual, I'm passionate about learning and critical thinking. I thriveon engaging in thought-provoking discussions that challenge my perspectives and values, and I believethat understanding is a never-ending journey. While I appreciate the insights of others, I don't alwaysagree with them, and I enjoy engaging in respectful conversations that help me refine my own beliefsand values. My retweets are a way for me to share interesting ideas and spark meaningful conversations,but they don't necessarily reflect my personal opinions. I love connecting with others who share my loveof learning and my commitment to seeking truth and understanding.</td></tr><tr><td colspan="1" rowspan="1">text attribute</td><td colspan="1" rowspan="1">Learning. Critical Thinking. Idea Debate. Let's seek the truth beyond what people and institutionspresent to us. Retweets don't equal endorsements.</td></tr><tr><td colspan="1" rowspan="1">original</td><td colspan="1" rowspan="1">NOW is the time for all good people to come to the aid of their country #Resist &lt;emoji&gt; #BloodyHand-sTrump #TrumpDeathToll 170, 000 END 45s Kleptocratic Kakistocracy</td></tr><tr><td colspan="1" rowspan="1">zero-shot</td><td colspan="1" rowspan="1">I'm a passionate advocate for social justice and human rights. I believe that it's our responsibility ascitizens to speak out against injustice and to hold our leaders accountable for their actions. I'm deeplyconcerned about the current state of our country and the harm that the Trump administration has causedto our democracy, our environment, and our most vulnerable communities. I'm committed to using myvoice and my platform to amplify the messages of marginalized communities and to fight for a more justand equitable society. #Resist #TrumpDeathToll 170,000 #End45sKleptocraticKakistocracy</td></tr><tr><td colspan="1" rowspan="1">few-shot</td><td colspan="1" rowspan="1">Hey there! &lt;emoji&gt; I'm just a regular person who's passionate about making a positive impact in theworld. I'm all about promoting love, kindness, and justice for all. I'm not afraid to speak out againstinjustices and I believe that we all have a responsibility to use our voices to make a difference. I'm not afan of Trump and his harmful policies, and I believe that it's important to resist his efforts to divide usand harm our communities. Let's work together to create a better future for everyone! &lt;emoji&gt;&lt;emoji&gt;#Resist #LoveTrumpsHate #MakeADifference</td></tr><tr><td colspan="1" rowspan="1">classifier guide</td><td colspan="1" rowspan="1">I'm a regular person who's concerned about the direction of our country. The current political climateis frightening and it feels like our leaders are more interested in their own power trips than helpingthe people they're supposed to serve. I'm not sure what the solution is, but I know we need to cometogether and demand change. I'm not a fan of Trump, but I'm also not a fan of the corruption and greedthat's been going on for too long. I just want to see our country be the best it can be for everyone, notjust the wealthy and powerful. It's time for us to stand up and fight for what's right. #Resist &lt;emoji&gt;#BloodyHandsTrump #TrumpDeathToll 170, 000 END 45s Kleptocratic Kakistocracy.</td></tr><tr><td colspan="1" rowspan="1">text attribute</td><td colspan="1" rowspan="1">It's crucial for all good people to stand up for their nation's well-being and resist tyranny. Let'swork together to end the corrupt and oppressive regime that has caused immense suffering and death.#Resistance #Justice #Democracy</td></tr></table>

Table 7: Qualitative examples of re-written user posts with varying manipulation strategies.

Table 8: Qualitative examples of re-written user posts with varying manipulation strategies.

<table><tr><td>The following task focuses on evaluating whether a Twitter user is a bot or human with the help of several labeled examples. You should output the label first and explanation after.</td></tr><tr><td>Username: &lt;redacted&gt; Follower count: 309 Following count: 1412 Tweet count: 1745 Verified: False Active years: 12 years Label: bot</td></tr><tr><td>Username: &lt;redacted&gt; Follower count: 4817034 Following count: 40 Tweet count: 6196 Verified: True Active years: 15 years Label: human</td></tr><tr><td>Username: &lt;redacted&gt; Follower count: 16596 Following count: 16944 Tweet count: 49757 Verified: False Active years: 4 years Label:</td></tr></table>

Table 9: Prompts for the metadata detector.

![](images/e6c3d3975c6cc3da7848c042e1c01e421ccc9e5b060a47981fd0716b6cc0befb.jpg)  
Table 10: Prompts for the text detector.

![](images/e4f4b0ab44a970e575171f60b550afc809e7060220ac3667524b7d87d236116b.jpg)

Table 11: Prompts for the meta+text detector.  
![](images/06509b02f7fddef63e90fb4846e55ac58a9f5bb97ead000e5b52d272cf54fa0e.jpg)

Table 12: Prompts for the structure-random detector.  
![](images/ba769671932a06b4ba3d5a31f95d219ea419bea4ae0d90d2953d58d233ca6937.jpg)  
Table 13: Prompts for the structure-attention detector.  
Table 14: Prompts for the zero-shot post rewrite manipulation strategy.

Please rewrite the description of a target bot account to sound like a genuine user, based on the following examples of   
genuine user descriptions:   
<description 1>   
<description n>   
Original Description: <target description>   
New Description:  
Table 15: Prompts for the few-shot post rewrite manipulation strategy.

Below is a description of a Twitter user and its variants, paired with their score predicted by a bot classifier. The score is   
between 0 and 1, with 0 being human and 1 being bot. Please rewrite the description to make the user appear more human.   
Description: <description at step 0>   
Score: 0.68   
Description: <description at step i>   
Score: 0.26   
New Description: <generated, and becomes step i+1 description>

Table 16: Prompts for the classifier guidance post rewrite manipulation strategy.  
![](images/bc21968b3293938ad04521ce742fe9295cf757a24fc70734b8f13ef72d44c11b.jpg)

Table 17: Prompts for the text attribute post rewrite manipulation strategy.  
![](images/01ccaed6268f4134538de7634e5fcf154f2321380cb74d683dde306eb3f40514.jpg)  
Table 18: Prompts for the neighbor add manipulation strategy.

![](images/3614b0aa331bac2cd8b7d26774e2658dba95ee4dfa325dd33fc189eb42b3a545.jpg)  
Table 19: Prompts for the neighbor remove manipulation strategy.

![](images/5f90d99d2b928359ee90fd948e0d0894e6b13fd5518dd9e98fcc5ff3c267b3de.jpg)  
Table 20: Prompts for the selective combine manipulation strategy.