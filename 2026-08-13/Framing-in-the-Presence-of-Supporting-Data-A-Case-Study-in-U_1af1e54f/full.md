# Framing in the Presence of Supporting Data: A Case Study in U.S. Economic News

Alexandria Leto<sup>1</sup> Elliot Pickens<sup>2</sup> Coen D. Needell<sup>3</sup>

David Rothschild<sup>4</sup> Maria Leonor Pacheco<sup>1</sup>

<sup>1</sup>University of Colorado Boulder <sup>2</sup>University of Wisconsin Madison

<sup>3</sup>University of Pennsylvania <sup>4</sup>Microsoft Research

<sup>1</sup>{alexandria.leto, maria.pacheco}@colorado.edu <sup>2</sup>epickens@cs.wisc.edu <sup>3</sup>coen@needell.org <sup>4</sup>david@researchdmr.com

## Abstract

The mainstream media has much leeway in what it chooses to cover and how it covers it. These choices have real-world consequences on what people know and their subsequent behaviors. However, the lack of objective measures to evaluate editorial choices makes research in this area particularly difficult. In this paper, we argue that there are newsworthy topics where objective measures exist in the form of supporting data and propose a computational framework to analyze editorial choices in this setup. We focus on the economy because the reporting of economic indicators presents us with a relatively easy way to determine both the selection and framing of various publications. Their values provide a ground truth of how the economy is doing relative to how the publications choose to cover it. To do this, we define frame prediction as a set of interdependent tasks. At the article level, we learn to identify the reported stance towards the general state of the economy. Then, for every numerical quantity reported in the article, we learn to identify whether it corresponds to an economic indicator and whether it is being reported in a positive or negative way. To perform our analysis, we track six American publishers and each article that appeared in the top 10 slots of their landing page between 2015 and 2023.

## 1 Introduction

The mainstream media has much leeway in what it chooses to cover and how it covers it. Should the top article at any given point in time be about an international incident or the state of the economy? Conditional on choosing to cover the economy, should the article focus on the indicator that is up or the one that is down? Should it be optimistic or pessimistic towards the future? These choices have real-world consequences affecting outcomes such as voting behavior (Druckman and Parkin, 2005; Gerber et al., 2009), gun purchases (Krupenkin et al., 2023), and attitudes towards immi-

Stocks extend their rout as the Ukraine war and its economic fallout intensify

![](images/a0834f827c8cd2e5a43726df48e8eaca8a0e87035243a3fde1b018d4d3f43ec8.jpg)  
Figure 1: The Frame Prediction Framework

grants (Krupenkin et al., 2020; Guo et al., 2023). Most selection and framing decisions lack objective measures, making research on this topic difficult.

In this paper, we make the observation that there are a select number of newsworthy topics where objective measures do exist. For example, there is a finite set of economic indicators that experts use to define the state of the economy, such as changes in non-farm payroll and gross domestic product. Another example is crime, which has well-established data points maintained by the FBI and local jurisdictions. Building a system capable of tracking a large number of publications, identifying which indicators are reported and how they are being reported has enormous research potential. Such a system would enable us to create three views of potential bias: how a publication is covering a topic over time, how their coverage differs from other publications, and how it compares to the accepted ground truth of experts in the field.

Following this rationale, we propose a computational framework to predict frames in the presence of supporting data. For a given article, our goal is to automatically identify how the general topic is being portrayed, which indicators are being reported to support this view, and how each of these indicators are being presented. We focus specifically on news articles about the U.S. economy. However, we argue that this framework could be adapted to other domains where numerical indicators are used, including reports on crime, climate change, and public opinion.

Most previous computational framing analysis approaches have conceptualized and interpreted frames as high-level relevant topics and themes (Ali and Hassan, 2022). For example, Boydstun et al. (2014) proposed 15 broad dimensions that aim to capture ways in which policy issues are discussed in news articles. The dimensions include themes such as “economic” and “public opinion”. These broad dimensions fail to capture a frame’s nuances. For example, the “economic frame” focuses on anything related to the economy, which is insufficient to answer the question of how different aspects of the economy are being presented.

To address this problem, we decompose economic frames into a set of interdependent tasks (See Fig. 1). At the article level, we identify the relevance of the article to the economy, the type of economic information it covers (e.g., macro economic, industry-specific, firm-specific), and two measures of general economic sentiment: the state of the economy (e.g., good, fair, poor), and the direction in which the economy is heading (e.g., better, same, worse). Then, for each numerical quantity reported in the article, we predict whether it corresponds to an economic indicator (e.g., market numbers) and its reported polarity (e.g., “The S&P 500 fell 3%” has a negative polarity). This decomposition captures three levels: whether the economy is being discussed, which economic indicators are highlighted, and the framing of those indicators and the economy as a whole. This way, we can go beyond analyzing topic selection by additionally measuring how a publication’s reporting differs from the objective reality at that time, how it evolves across time, and how it compares to other publications. We present a computational model that predicts each proposed frame component jointly. It utilizes the inter-dependence between the sub-tasks (e.g., polarity of quantities and polarity of the article they appear in) to improve upon results obtained by independent classifiers trained to make predictions individually with limited supervision.

We make the following contributions: (1) We propose a computational framework to model framing in the presence of supporting data and apply it to news about the U.S. economy. (2) We collect a novel dataset of landing page news articles published by a set of major U.S. news outlets between 2015 and 2023, and provide high-quality annotations of our proposed frame components for a small subset of examples. (3) We propose an automated method to detect each of the proposed frame components under low-supervision settings. (4) We demonstrate that our framework can be used to track when and how often (selection), as well as how (framing) different news outlets report on specific aspects of the economy. We compare these reports to the ground truth for two example indicators: job numbers and prices. All of our code and data has been released to the community\*.

## 2 Background and Related Work

In this section, we cover the background on selection and framing, explain economic indicators, and discuss previous approaches to analyze framing.

Selection and Framing National news publications have a massive choice-set of topics they could cover on any given day. Though it may seem like the publications are reacting to a defined set of events that recently happened, outside of a few major events, they choose what the news is on any given day. In other words, the news in mainstream publications does not present an objective view of current affairs. Rather, it is influenced by the selection made by media professionals (McQuail, 1992). Selection is driven by what researchers have called news values (Galtung and Ruge, 1965; Harcup T and O’Neill D, 2001, 2017), which determine whether a story is newsworthy.

While there is variation in the taxonomies that have been proposed to determine newsworthiness, the most recent ones include factors such as: the power of the elite (stories concerning powerful entities), bad and good news (stories with particular emotional overtones), and magnitude (stories that are perceived as significant for a wide audience) (Harcup T and O’Neill D, 2017). More interestingly, news values vary from publication to publication. A study of the run-up to the 2022 election showed that the New York Times and Washington Post averaged just two overlapping topics each morning in their printed editions out of about six articles (Rothschild et al., 2023).

Conditional on selecting a topic, there is additional leeway on how to cover it, as there is a widerange of facts and opinions that could spin the takeaway of the readers. This phenomenon, referred to as framing, has been widely studied in communication studies. While there is not an agreed upon definition of framing, recent surveys (Vallejo et al., 2023) have identified the following prevalent definitions: equivalence framing (presenting the same exact information in different ways) (Cacciatore et al., 2015), emphasis framing (highlighting specific aspects of an event to promote a particular interpretation) (Entman, 2007), and story framing (leveraging established narratives to convey information) (Hallahan, 1999).

When it comes to the economy, a publication could present a rosy or terrible version of the economy. They can do so without lying by simply picking the right economic indicator with the right perspective. For example, there are months where the value of the non-farm payroll is above average, but below the market expectation: either perspective is technically true. But, with a relatively finite set of indicators, and an established way for economists to judge them, there is still a clear ground truth to compare if any given indicator with any given perspective represents the consensus understanding of the economy at that time.

Economic Indicators Understanding the health and wealth of the U.S. economy is critical for investment and growth. Most modern economic indicators have been tracked for decades, codified in the by the mid-20th century. Examples include: jobs numbers (e.g., non-farm payroll), prices (e.g., consumer price index), and macro economic (e.g., gross domestic product). With the exception of market prices (e.g., S&P500), which are a reflection of the underlying assets, the core indicators are tracked by a few government agencies (e.g, the Bureau of Economic Analysis for GDP, spending, income, trade, and the Bureau of Labor Statistics for unemployment, prices, productivity), which conduct regular, large scale polling.

These economic indicators inform the public on the health of the economy. The job numbers are front page news when they are released at 8:30 AM ET on the first Friday of each month. The state of the markets is constantly covered in the news. Indicators such as the consumer price index pop up whenever the media finds them most important, interesting, and engaging. Most people have limited views of the economy outside of their close social circle, so these values are the key to their understanding of the health of the economy as a whole. And, traditionally, the health of the economy has tracked very closely with the approval of incumbent political leaders (Hummel and Rothschild, 2014).

Framing of Economic News This paper builds off of an extensive literature that has not just identified the quantity of economic coverage, but framing of that coverage. Most of this work is geared towards matching the sentiment of articles about the economy to both consumer sentiment and economic indicators, tracking both historical value and predicting future ones (Hopkins et al., 2017; Ardia et al., 2019; Shapiro et al., 2022; Seki et al., 2022; Bybee, 2023; van Binsbergen et al., 2024). While the methods continue to evolve from simple keywords to BERT or LLM-based predictions, they are focused on article-level sentiment, aggregated to states or the U.S. They also study a mix of questions about economic impact that benefit from a long time-series of sentiment, granular in both time and geography, regarding questions as diverse as the impact of bubbles on industries (Bybee, 2023), economic shocks to economic decisions (Shapiro et al., 2022), to the relationship between news and perception of the economy (Hopkins et al., 2017). Our paper extends this literature by capturing not only the sentiment of economic articles, but what economic indicators and economic indicator-level sentiment is driving it. This allows us to dissect what editorial choices are driving perceived sentiment around the economy.

Computational Framing Analysis Scholars have increasingly adopted computational approaches to study framing, allowing studies to scale to large media repositories. In most cases, researchers adopt unsupervised techniques such as topic modeling to identify latent themes (DiMaggio et al., 2013; Nguyen et al., 2015; Gilardi et al., 2021). However, topic models are limited in their ability to capture framing nuances. By defining topics as simple distributions over words, they lack the semantic and discursive conceptualization needed to answer how issues and events are being presented (Ali and Hassan, 2022).

There is also a body of work leveraging supervised learning (Johnson et al., 2017; Khanehzar et al., 2019; Kwak et al., 2020; Huguet Cabot et al., 2020; Mendelsohn et al., 2021) and lexicon expansion techniques (Field et al., 2018; Roy and Goldwasser, 2020) to analyze framing. For this to be feasible, authors must define a concrete taxonomy of which frames are relevant and how they are represented in data. Most of these approaches rely on the taxonomy proposed by Boydstun et al. (2014). These dimensions correspond to broad themes such as “economic”, and “public opinion”. While these themes have shown to be useful to model a select set of contentious political issues like abortion and immigration, they are too broad to capture how the economy and other data-driven topics are being presented in the media.

## 3 Data

In this section, we describe our data collection process, annotation schema, annotation guidelines, and quality assurance process. We also present statistics for our resulting dataset.

Data Collection The data for this study came from the Internet Archive’s Wayback Machine (Archive, 1996-2023). We consider six American publishers: The New York Times, The Wall Street Journal, The Washington Post, Fox News, HuffPost and Breitbart. Selected sources represent a broad range of mainstream media outlets that were easily accessible through the archive. Breitbart is included for a future analysis comparing mainstream media with a fringe source. For each publisher, we collected the “front page” from every Wayback Machine entry between Jan. 1st 2015 to Jan. 1st 2023. For each front page, we recorded articles in the top 10 positions and discarded all duplicates.

To identify articles that discuss the economy, we curated a lexicon of economic terms sourced from the Bureau of Labor Statistics and the FRED database operated by the Federal Reserve bank of St. Louis.<sup>†‡</sup> These sources provide a fairly comprehensive list of common metrics for economic activity. We consider an article to be relevant to the economy if it contains at least three sentences mentioning an economic term from our lexicon. This resulted in a total of 199,066 articles (Tab. 1). App. A.1 includes additional details about the data collection process and the lexicon of economic terms.

Data Annotation We employed a group of six annotators to label our dataset following the guidelines outlined in App. A.3. Our annotators comprised senior, postdoctoral and pre-doctoral researchers. Their fields of study include: economics, computational social science, and communications. They received training from the authors of the paper and several rounds of calibration were performed before annotation began. Articles were chosen such that there would be some overlap between coders (in order to check quality) and a decent breadth of coverage. We prioritized topic diversity over uniformity of news outlets and thus followed the sampling procedure introduced in Pacheco et al. (2022b, 2023).

<table><tr><td rowspan="2">Publisher</td><td rowspan="2">Econ Arts.</td><td rowspan="2">Quants.</td><td colspan="2">Human Annotations</td></tr><tr><td>Art-level</td><td>Quant-level</td></tr><tr><td>New York Times</td><td>58,240</td><td>370,723</td><td>516</td><td>1,117</td></tr><tr><td>Wall Street Journal</td><td>46,267</td><td>551,726</td><td>206</td><td>493</td></tr><tr><td>Washington Post</td><td>44,016</td><td>274,197</td><td>231</td><td>421</td></tr><tr><td>Fox News</td><td>21,795</td><td>76,074</td><td>45</td><td>42</td></tr><tr><td>Breitbart</td><td>15,954</td><td>66,572</td><td>91</td><td>149</td></tr><tr><td>HuffPost</td><td>12,794</td><td>75,180</td><td>82</td><td>210</td></tr><tr><td>Total</td><td>199,066</td><td>1,414,472</td><td>1,171</td><td>2,414</td></tr><tr><td>Cross-Annotated</td><td></td><td></td><td>270</td><td>689</td></tr></table>

Table 1: Resulting Dataset Statistics

For each article in their batch, annotators selected: (1) what type of economic information was the most prominent (macro, government, industryspecific, business-specific or personal), (2) the framing of the general economic conditions (good, fair, bad), and (3) the framing of the direction that the economy is heading (better, same, worse). These questions were adapted from the Gallup economic index <sup>§</sup>. Then, for every valid numerical value reported in the article, annotators selected: (1) what type of economic data it reports (macro, government, industry-specific, business-specific or personal), (2) which indicator was reported (e.g., jobs, prices. See App. A.3 for the full list of options), and (3) the reported polarity of each quantity (positive, negative, neutral). We report resulting statistics in Table 1, and inter-annotator agreement in Tab. 2. To calculate agreement, we only consider examples that were annotated by two or more coders. In App. A.4 we include statistics for the number of coders for each of the label categories. Lastly, we show the resulting annotation distribution for all frame components in Figs. 2 and 3.

We find that inter-annotator agreement is generally good for quantity-level information (Krippenforff’s α 0.56-0.83). This reflects our intuition that capturing framing at the level of supporting data points may be a better alternative to relying on general topical markers. On the other hand, article-level judgements which aim to capture main “takeaways” are considerably harder for annotators, exhibiting moderate agreement (Krippenforff’s α 0.42-0.48). This is expected and in line with other high-level framing tasks (Card et al., 2015; Roy et al., 2021; Mendelsohn et al., 2021). To further characterize the lower agreement values, we include confusion matrices and some examples illustrating ambiguous cases in App. A.5. We find that most of the confusion lies on the following article-level judgements: (1) identifying the main type of the article, particularly deciding between macro-economic and government-specific types when both types of data points are present in the article, (2) characterizing articles that are positive towards the economy, as they tend to be more subtle than the clearly negative ones, and (3) identifying articles that suggest that the economy is doing neither better nor worse, but staying the same, which is an inherently ambiguous category. We have included a similar confusion analysis for quantitative-level judgements in App. A.5.

<table><tr><td>Annotation</td><td> $\Gamma \ ' { \bf s } \alpha$ </td><td>% of Agreement Full</td><td>Partial</td></tr><tr><td>Article Type Econ. Conditions Econ. Direction</td><td>0.48 0.43</td><td>57.81 55.13</td><td>90.62 85.90</td></tr><tr><td>Quantity Type Quantity Polarity Macro Ind.</td><td>0.42 0.75 0.56 0.83</td><td>47.44 80.84 68.04 83.42</td><td>82.05 88.97 76.48 92.78</td></tr></table>

Table 2: Inter-annotator Agreement for all examples with 2 or or more annotators. Here, Krippendorff’s $\alpha = 1$ indicates perfect reliability, $\alpha = 0$ indicates the complete absence of reliability, i.e., judgments are random, and $\alpha < 0$ indicates that disagreements are systematic and exceed what can be expected by chance.

## 4 Model

Obtaining high-quality annotations for our task is time-consuming and requires considerable domain expertise. For this reason, one of the main challenges that we face in predicting economic frames is a small amount of supervision. To circumvent this constraint we combine two modeling strategies: (1) Exploiting the decomposition of frame prediction into a set of modular, inter-dependent sub-tasks, and (2) Leveraging pre-training strategies using our large set of in-domain unlabeled examples (all 199,066 economic articles collected).

![](images/4fc39d7df9fc547074b3f9e1b7b54507326f4af2a69ff288669bf89192cfa38a.jpg)  
Figure 2: Distribution of article-level annotation labels

Exploiting inter-dependencies with SRL Statistical relational learning (SRL) methods attempt to model a joint distribution over relational, interdependent data (Richardson and Domingos, 2006; Bach et al., 2017; Pacheco and Goldwasser, 2021). These methods have proven particularly effective in tasks where contextualizing information and interdependent decisions can compensate for a low number of annotated examples (Deng and Wiebe, 2015; Roy et al., 2021; Pacheco et al., 2022a).

To model framing, we take advantage of the relations between the values of the frame components for a given article and implement a relational model using Probabilistic Soft Logic (PSL) (Bach et al., 2017). Dependencies in PSL are expressed using weighted logical rules of the form: $w _ { r }$ $P _ { 1 } \land . . . \land P _ { n - 1 } \to P _ { n }$ , where $w _ { r }$ indicates the importance of the rule in the model, and can be learned from data. Predicates $P _ { i }$ correspond to decisions and observations. Rules are then compiled into a Hinge-Loss Markov random field and weights are learned using maximum likelihood estimation. We consider the following rules:

Priors. We explicitly model prior probabilities for all article-level (type, economic conditions and economic direction) and quantity-level (type, macro-indicator, and sentiment) components. These priors are derived from supervised classifiers trained on our labeled dataset.

(r<sub>1</sub>) Consistency between the quantity type and the macro-indicator. We enforce that if the quantity of a type is macro, then a macro-indicator must be predicted, and viceversa: Reports(a, q)  Type(q, macro) MacroIndicator(q, none)

![](images/a9d22c1459ccd3628b24ab22940e9b14d9c1b047c33becb7694e01dd672b9d6b.jpg)  
Figure 3: Distribution of quantity-level annotation labels

(r<sub>2</sub>) Consistency between the article type, economic conditions and direction. We enforce that if the article type is macro, then economic conditions and economic direction must be predicted and vice-versa: Type(a, macro) EconConditions(q, irrelevant) EconRating(q, irrelevant)

(r<sub>3</sub>) Dependency between the polarity of quantities and economic conditions. Our intuition is that if several negative quantities are reported in the article, then the article is likely to frame economic conditions as negative, and vice-versa for positive quantities: Reports(a, q) Polarity(q, p)  EconConditions(a, p)

(r<sub>4</sub>) Dependency between the polarity of quantities and economic direction. Our intuition is that if many negative quantities are reported in the article, then the article is likely to frame the direction of the economy as negative, and vice-versa for positive quantities: Reports(a, q) Polarity(q, p)  EconDirection(a, p)

(r<sub>5</sub>) Dependency between neighboring quantity types. Our intuition is that there are common patterns in the types of consecutive quantities (e.g., sequential dependencies): Precedes(q1, q2) Type(q1, t1)  Type(q2, t2)

Rule r<sub>1</sub> and r<sub>2</sub> are modeled as hard constraints, and are always enforced because they are designed to force predictions to follow the frame annotation structure. Rules r<sub>3 5</sub> are modeled as soft constraints, and their weights are learned through PSL. This way, we allow for predictions that do not conform to the templates to be active.

Enhancing PSL priors with pre-trained language models Pre-trained language models (LMs) are one of the most effective techniques for acquiring knowledge from unlabeled text data (Devlin et al., 2019). Domain-adaptive pre-training (DAPT) further enriches LMs with in-domain data (Gururangan et al., 2020). In both of these pretraining stages, learning is conducted by randomly masking words in a large dataset and training the the LM to predict these words. To take advantage of these strategies, we use pre-trained RoBERTA and perform DAPT with our large unlabeled dataset (all 199,066 economic articles collected). We use this LM to fine-tune the classifiers that are used as priors in PSL. This strategy has been repeatedly used to combine PSL with strong classifiers, with consistent success (Sridhar et al., 2015; Pacheco and Goldwasser, 2021; Roy et al., 2021).

## 5 Experimental Evaluation

In this section, we outline our experimental settings and present results for our frame prediction model. We also include an example of the type of framing analysis that can be done using our model predictions over the complete dataset.

## 5.1 Frame Prediction

To determine the effectiveness of exploiting interdependencies with statistical relational learning and our additional pre-training steps, we conducted an ablation study in addition to evaluating the model as a whole. We provide a discussion of the results of these experiments, in which we detail per-class and per-publisher metrics.

Experimental Settings We perform 5-fold crossvalidation in all scenarios, and assume a multi-class setup for each frame component. To create our folds, we first split the articles that were crossannotated and where inter-annotator agreement was reached (270 articles, 689 quantities) into five folds, and consider 4 folds for training and 1 for testing. Then, we further enhance the training data for all cases with the additional annotated articles, which we consider as a source of noisy supervision. More details about the data splits used for experiments are included in App. A.6.

<table><tr><td rowspan="2">Model</td><td colspan="3">Article-level</td><td colspan="3">Quantity-level</td></tr><tr><td>Type</td><td>Cond</td><td>Dir</td><td>Type</td><td>Ind</td><td>+/-</td></tr><tr><td>Random</td><td>0.136</td><td>0.148</td><td>0.19</td><td>0.111</td><td>0.064</td><td>0.324</td></tr><tr><td>Majority Label</td><td>0.237</td><td>0.269</td><td>0.252</td><td>0.193</td><td>0.08</td><td>0.382</td></tr><tr><td>Mistral 0-shot</td><td>0.386</td><td>0.236</td><td>0.346</td><td>0.566</td><td>0.262</td><td>0.382</td></tr><tr><td>Mistral 2-shot</td><td>0.367</td><td>0.186</td><td>0.385</td><td>0.358</td><td>0.47</td><td>0.43</td></tr><tr><td>Base Classifier</td><td>0.515</td><td>0.697</td><td>0.493</td><td>0.685</td><td>0.824</td><td>0.796</td></tr><tr><td>Base Classifier + DAPT</td><td>0.474</td><td>0.636</td><td>0.475</td><td>0.731</td><td>0.826</td><td>0.812</td></tr><tr><td>Relational (best)</td><td>0.438</td><td>0.717</td><td>0.522</td><td>0.748</td><td>0.849</td><td>0.813</td></tr></table>

Table 3: Avg. Macro F1 after 5-Fold Cross Validation

The base classifiers are initialized with pretrained RoBERTA (with and without DAPT) and trained using the AdamW optimizer, cross-entropy loss, and a learning rate of 2e  5. For early stopping, we use the macro F1 on the dev set, consisting of ten percent of the training articles for each fold. For more details on the classifier architectures and training settings, see App. A.6. Predictions are fed into PSL as priors, and rule weight learning is done with the standard configuration. We report the average macro F1 scores for all folds.

Results We present our general results in Tab. This includes baselines obtained by zero-shot and two-shot prompting Mistral-7b-Instruct-v0.2. Additional details about the prompts and performance can be found in App. A.6. We find that fine-tuning, even with relatively little data, results in better performance than prompting an LLM. Additionally, we find that DAPT is considerably helpful for quantitylevel predictions, but does not improve article-level predictions. We hypothesize that it is difficult for RoBERTa to model the long context of articles to take advantage of the masked language modeling objective. Next, we find that the relational model gives us a clear boost in performance over the base classifiers, with the exception of article type. This improvement supports our modeling decision to decompose frame prediction into inter-dependent sub-tasks. While Tab. 3 reports macro F1, we find the same trend for weighted F1 (See App. A.7).

In Tab. 4 we include an ablation study for the relational model rules. We observe that all rules contribute to an increase in performance for their corresponding decision predicates (see bold scores). We find that combining hard constraints (r1,r2) and the rule modeling sequential dependencies (r5) performs the best.

We include results for the best relational model by article publisher in Tab. 5. At the article level, we observe higher values for the New York Times, reflecting the higher presence of annotated examples for this publisher (see Tab. 1). The results for Fox News articles are lower, reflecting the lower number of examples. Interestingly, the results for Breitbart and Huffpost do not follow this trend, maintaining relatively high values in spite of less supervision. In the case of quantities, we see relatively high and stable performance for all publishers, with the exception of Fox News. We provide further error analysis in regard to this in App. A.9. These results are particularly encouraging, as they suggest that we can leverage quantity predictions to perform a high-fidelity selection and framing analysis of economic news.

<table><tr><td rowspan="2">Rule</td><td colspan="3">Article-level</td><td colspan="3">Quantity-level</td></tr><tr><td>Type</td><td>Cond</td><td>Dir</td><td>Type</td><td>Ind</td><td>+/-</td></tr><tr><td>Priors only</td><td>0.515</td><td>0.697</td><td>0.493</td><td>0.731</td><td>0.826</td><td>0.812</td></tr><tr><td>r1</td><td>0.515</td><td>0.697</td><td>0.493</td><td>0.688</td><td>0.853</td><td>0.813</td></tr><tr><td>r2</td><td>0.438</td><td>0.717</td><td>0.518</td><td>0.731</td><td>0.826</td><td>0.812</td></tr><tr><td>r3</td><td>0.515</td><td>0.707</td><td>0.493</td><td>0.731</td><td>0.826</td><td>0.812</td></tr><tr><td>r4</td><td>0.515</td><td>0.697</td><td>0.521</td><td>0.731</td><td>0.826</td><td>0.813</td></tr><tr><td>r5</td><td>0.515</td><td>0.697</td><td>0.493</td><td>0.737</td><td>0.826</td><td>0.813</td></tr><tr><td> $r 1 + r 2$ </td><td>0.438</td><td>0.717</td><td>0.518</td><td>0.688</td><td>0.853</td><td>0.813</td></tr><tr><td> $r 1 + r 2 + r 3$ </td><td>0.433</td><td>0.7</td><td>0.493</td><td>0.697</td><td>0.854</td><td>0.813</td></tr><tr><td> $r 1 + r 2 + r 4$ </td><td>0.432</td><td>0.691</td><td>0.516</td><td>0.697</td><td>0.855</td><td>0.813</td></tr><tr><td> $r 1 + r 2 + r 5 \ : ( \mathrm { b e s t } )$ </td><td>0.438</td><td>0.717</td><td>0.522</td><td>0.748</td><td>0.849</td><td>0.813</td></tr><tr><td> $r 1 + r 2 + r 3 + r 4 + r 5$ </td><td>0.434</td><td>0.701</td><td>0.519</td><td>0.746</td><td>0.84</td><td>0.808</td></tr></table>

Table 4: Ablation (Macro F1) for the Reln. Model
<table><tr><td rowspan="2">Publisher</td><td colspan="3">Article-level</td><td colspan="3">Quantity-level</td></tr><tr><td>Type</td><td>Cond</td><td>Dir</td><td>Type</td><td>Ind</td><td>+/-</td></tr><tr><td>New York Times</td><td>0.544</td><td>0.809</td><td>0.534</td><td>0.663</td><td>0.826</td><td>0.777</td></tr><tr><td>Wall Street Journal</td><td>0.173</td><td>0.434</td><td>0.28</td><td>0.834</td><td>0.813</td><td>0.884</td></tr><tr><td>Washington Post</td><td>0.574</td><td>0.704</td><td>0.621</td><td>0.665</td><td>0.686</td><td>0.788</td></tr><tr><td>Fox News</td><td>0.4</td><td>1.0</td><td>0.4</td><td>0.733</td><td>1.0</td><td>0.333</td></tr><tr><td>Breitbart</td><td>0.636</td><td>0.325</td><td>0.597</td><td>0.98</td><td>0.867</td><td>0.655</td></tr><tr><td>HuffPost</td><td>0.564</td><td>0.625</td><td>0.3</td><td>0.609</td><td>0.686</td><td>0.866</td></tr></table>

Table 5: Macro F1 per Publisher with Best Reln. Model

Lastly, we include fine-grained results for predicting the different macro-indicators using our best model in Tab. 6. We find that we have relatively good performance for most indicators, with the exception of the ’other’ category, which may be due to the lack of cohesiveness in this class. We also see lower performance for retail sales, which we attribute to the lack of sufficient supervision for this class (See Fig. 3). An effective large-scale analysis of retail indicators in the news would require greater supervision for this class.

## 5.2 Framing and Selection Analysis

In this section we present a brief exploration of how the methods outlined in this paper can be used to study editorial choices. To do this, we use our best model to predict all frame components for the full dataset of 199,066 articles. We then use these frames to show how economic messaging shifts in response to exogenous shocks and changes in the balance of political power. In particular, we show how our framework can help us understand how the different choices these major publications make can produce significantly different views of economic conditions.

<table><tr><td>Macro-Indicator</td><td>Prec.</td><td>Recall</td><td>F1</td></tr><tr><td>Job Numbers (jobs, unemployment)</td><td>0.956</td><td>0.982</td><td>0.968</td></tr><tr><td>Retail Sales</td><td>0.667</td><td>1.0</td><td>0.8</td></tr><tr><td>Interest Rates (Fed, mortgage)</td><td>0.75</td><td>0.857</td><td>0.8</td></tr><tr><td>Prices (CPI, PPI)</td><td>0.733</td><td>0.846</td><td>0.786</td></tr><tr><td>Energy Prices (gas, oil, etc.)</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>Wages Macro Economy (GDP, etc)</td><td>0.867</td><td>0.812</td><td>0.839</td></tr><tr><td>Market Numbers (any financial market)</td><td>0.881</td><td>0.860</td><td>0.871</td></tr><tr><td></td><td>0.941</td><td>0.814</td><td>0.873</td></tr><tr><td>Currency Values</td><td>1.0</td><td>0.8</td><td>0.889</td></tr><tr><td>Housing (start, sales, pricing)</td><td>1.0</td><td>0.95</td><td>0.974</td></tr><tr><td>Other None</td><td>0.375</td><td>0.5</td><td>0.429</td></tr><tr><td></td><td>0.956</td><td>0.964</td><td>0.96</td></tr><tr><td>Accuracy</td><td>0.922</td><td>0.922</td><td>0.922</td></tr><tr><td>Macro Average</td><td>0.844</td><td>0.865</td><td>0.849</td></tr><tr><td>Weighted Average</td><td>0.926</td><td>0.922</td><td>0.923</td></tr></table>

Table 6: Macro-Indicator results with Best Reln. Model

Article-level framing with macro-indicators Because we have greater confidence in our model’s quantity-level performance, we generate articlelevel frames for specific economic indicators using our quantity-level predictions. First, we select the subset of articles that include at least two quantities in the category of interest. Then, we assign a positive indicator frame to articles with at least twice as many quantities with a positive polarity than negative polarity and vice versa for negative frames. Articles that meet neither threshold are assigned a "neutral" frame.

Shifts in Framing Our preliminary evidence suggests that the framing of articles on any given topic by any given publication is both sensitive to exogenous factors as well as decisions made by the publication itself. Fig. 4 illustrates how the New York Times (NYT) framed their articles in regard to jobs data from 2015 to 2023. First, we can see that prior to 2020, during a period of stable job growth, the NYT already had a sustained negative valance, matching the key findings in recent literature on the general negativity bias on the Mainstream Media (van Binsbergen et al., 2024). In response to the onset of COVID in 2020, the NYT intensified its coverage of the job market in response to a dramatic surge in unemployment caused by a cascade of lockdowns. We can also see that much of this coverage carried an extreme negative valence, which matches the large drop in jobs and subsequent uncertainty.

![](images/0219ae25f17c739cf7d9282d3b8c1464c0bbbfe02b04e0b537025e3c62a57779.jpg)

Figure 4: Framing of articles referencing job numbers from 2015 through 2023 in the New York Times.<sup>¶</sup> Spin was aggregated quarterly. Monthly payroll (employment) data can be seen in the dotted black line. The gray bars represent proportion of the overall coverage taken up by jobs reporting.  
![](images/4ca68bc890cd534865c45ad67bd7dc586d442a5b548636ce14bc50f23514f99c.jpg)  
Figure 5: Selection of economic indicators referencing price (price & energy) numbers from 2015 through 2023 in the New York Times, Washington Post, and Wall Street Journal. Aggregated Quarterly. Monthly CPI data can be seen in the dotted blue line.

As the job market recovered and the labor market entered an extended period of exceptionally high growth, the volume of reporting in the NYT decreased back to its pre-pandemic levels, but the negativity persisted. This was not the same for other publications such as the Wall Street Journal, whose coverage more closely resembles the shifts in valence in the supporting data (See App. A.10.)

Indicator Selection Fig. 5 shows us another lever that publications can pull when shaping their coverage: indicator selection. As inflation began to rise, we can see that all three publications responded by increasing their coverage of prices through a greater use of price based indicators.<sup>||</sup> These three publications did not, however, respond in an identical manner, with, for example, the New York Times increasing their coverage of prices far after the other publications returned to pre-crisis levels of coverage.

With the simple analysis shown in these two figures, we can begin to show that editors have considerable leeway when choosing what indicator to cover at any given time, and how to cover it. While we kept the analysis brief due to space considerations, we were able to show the usability of our framework to track a diverse set of indicators reported in different publishers across time.

## 6 Applicability to Other Domains

We emphasize that the main contribution of this paper is not centered around performance gains of our relational model over the baseline models, but rather the operationalization of a challenging task: analyzing framing at a scale for domains with supporting numerical data, such as the economy. While our solution does not apply to all framing scenarios, it applies to a substantial number. For example, we envision this framework being useful for analyzing news about crime (by tracking data points maintained by the FBI and local jurisdictions), climate change (by tracking weather statistics, global avg. temperature, emissions statistics, etc.) and public-opinion (by tracking polling numbers, political betting market numbers, etc.).

## 7 Conclusion and Future Work

In this paper, we presented a novel framework for modeling framing in news articles in the presence of supporting data. We focused our analysis on news about the U.S. economy, and proposed an annotation schema to identify framing at three levels of abstraction: the general framing of the economic conditions, the economic indicators that are highlighted, and the framing of each of those indicators.

We showed that we can predict the components of our schema with relatively good performance, even in very low supervision settings. Finally, we demonstrated how our framework can be used to perform a large scale exploration of the frame choices for a given economic indicator.

In our analysis, we showed that different publications cover economics using different indicators, for example the NYT covers jobs numbers less than normal during a strong jobs recovery, instead switching to more coverage rising prices. And, while all three publications responded to rising inflation, they each used price numbers differently when covering the economy, with the NYT employing price numbers more frequently than the Washington Post, or the Wall Street Journal.

Going forward, we have two main points of focus. First, we are working on expanding our schema to capture a larger set of economic aspects. While we focused our analysis on macro-economic indicators, there are other indicators that we could track to get a more holistic view of the economy. For example, for governmental indicators, we could track types of expenditures (e.g. social security), revenue (e.g. taxes), as well as debt and deficit. Additionally, we want to perform a large-scale analysis of economic news framing. In this paper, we presented a brief demonstration of how our framework can be used to analyze shifts in framing across time for different publications. Next, we want to perform a similar analysis for a larger set of economic indicators and publications. We note that to perform such an analysis we must scale up our annotations and improve our model predictions.

## 8 Limitations

The work presented in this paper has two main limitations:

(1) Obtaining high-quality annotations for our frame structure is very expensive and time consuming. For this reason, we worked with a low amount of supervision. While we showed that we could obtain relatively good performance in this constrained scenario, to be able to use our framework for a large scale, holistic analysis, we need to address this issue. This is particularly important given that our annotations are considerably skewed. For example, we have significantly more supervision for job numbers and market numbers (which occur more often) than we do for energy prices (which occur less often). This considerably affects our performance for more long-tail indicators, and hence limits the types of analysis that we can perform. Our current efforts are dedicated to increase the amount of annotations for all components of our frame structure, as well as incorporating alternative semi-supervised strategies. We are hoping to release an extended version of the annotated dataset to the community in the near future.

(2) The fact that we are using automated techniques for frame prediction necessarily carries some uncertainty. Even if we improve our models considerably, our large scale analysis will have a margin of error. It is important to acknowledge this when presenting our findings.

## 9 Ethical Considerations

To the best of our knowledge, no code of ethics was violated during the development of this project. We used publicly available tools to collect our dataset, and discarded any instances that were unreachable. The annotators were paid \$15 per hour, and no personally identifiable information was collected or recorded during annotation.

We performed a thorough evaluation of our dataset, which is presented in the paper. We reported all pre-processing steps, learning configurations, hyperparameters, and additional technical details. Due to space constraints, some of this information was relegated to the Appendix. Further, the data and code have been released to the community. The results reported in this paper support our claims and we believe that they are reproducible.

The analysis reported in Section 5.2 is done using the outputs of a machine learning model and does not represent the authors personal views. The uncertainty of these predictions was adequately acknowledged in the Limitations Section, and the estimated accuracy was reported in Section 5.1.

## 10 Acknowledgements

This work utilized the Alpine high-performance computing resource, the Blanca condo computing resource, and the CUmulus on-premise cloud service at the University of Colorado Boulder. Alpine is jointly funded by the University of Colorado Boulder, the University of Colorado Anschutz, Colorado State University, and the NSF (award 2201538). Blanca is jointly funded by computing users and the University of Colorado Boulder. CUmulus is jointly funded by the National Science Foundation (award OAC-1925766) and the University of Colorado Boulder.

## References

Mohammad Ali and Naeemul Hassan. 2022. A survey of computational framing analysis approaches.

In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 9335–9348, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Internet Archive. 1996-2023. The wayback machine. Accessed Continuously from January 2022.

David Ardia, Keven Bluteau, and Kris Boudt. 2019. Questioning the news about economic growth: Sparse forecasting using thousands of news-based sentiment values. International Journal ofForecasting, 35(4):1370–1386.

Stephen H. Bach, Matthias Broecheler, Bert Huang, and Lise Getoor. 2017. Hinge-loss markov random fields and probabilistic soft logic. J. Mach. Learn. Res., 18(1):3846–3912.

Adrien Barbaresi. 2021. Trafilatura: A web scraping library and command-line tool for text discovery and extraction. In Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing: System Demonstrations, pages 122–131, Online. Association for Computational Linguistics.

Amber E. Boydstun, Dallas Card, Justin Gross, Paul Resnick, and Noah A. Smith. 2014. Tracking the Development of Media Frames within and across Policy Issues.

J Leland Bybee. 2023. The ghost in the machine: Generating beliefs with large language models. Technical report, Working Paper.

Michael Cacciatore, Dietram Scheufele, and Shanto Iyengar. 2015. The end of framing as we know it . . . and the future of media effects. Mass Communication & Society, 19.

Dallas Card, Amber E. Boydstun, Justin H. Gross, Philip Resnik, and Noah A. Smith. 2015. The media frames corpus: Annotations of frames across issues. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 438– 444, Beijing, China. Association for Computational Linguistics.

Lingjia Deng and Janyce Wiebe. 2015. Joint prediction for entity/event-level sentiment analysis using probabilistic soft logic models. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 179–189, Lisbon, Portugal. Association for Computational Linguistics.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages

4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Paul DiMaggio, Manish Nag, and David Blei. 2013. Exploiting affinities between topic modeling and the sociological perspective on culture: Application to newspaper coverage of u.s. government arts funding. Poetics, 41:570–606.

James N. Druckman and Michael Parkin. 2005. The impact of media bias: How editorial slant affects voters. Journal ofPolitics, 67(4):1030–1049.

Robert Entman. 2007. Framing bias: Media in the distribution of power. Journal of Communication, 57:163 – 173.

Anjalie Field, Doron Kliger, Shuly Wintner, Jennifer Pan, Dan Jurafsky, and Yulia Tsvetkov. 2018. Framing and agenda-setting in Russian news: a computational analysis of intricate political strategies. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, pages 3570– 3580, Brussels, Belgium. Association for Computational Linguistics.

Johan Galtung and Mari Holmboe Ruge. 1965. The structure of foreign news: The presentation of the congo, cuba and cyprus crises in four norwegian newspapers. Journal of Peace Research, 2(1):64–90.

Alan S. Gerber, Dean Karlan, and Daniel Bergan. 2009. Does the media matter? a field experiment measuring the effect of newspapers on voting behavior and political opinions. American Economic Journal: Applied Economics, 1(2):35–52.

Fabrizio Gilardi, Charles R. Shipan, and Bruno Wüest. 2021. Policy diffusion: The issue-definition stage. American Journal ofPolitical Science, 65(1):21–35.

Lei Guo, Chris Chao Su, and Hsuan-Ting Chen. 2023. Do news frames really have some influence in the real world? a computational analysis of cumulative framing effects on emotions and opinions about immigration. The International Journal of Press/Politics, 0(0):19401612231204535.

Suchin Gururangan, Ana Marasovic, Swabha´ Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, and Noah A. Smith. 2020. Don’t stop pretraining: Adapt language models to domains and tasks. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8342–8360, Online. Association for Computational Linguistics.

Kirk Hallahan. 1999. Seven models of framing: Implications for public relations. Journal of Public Relations Research, 11(3):205–242.

Harcup T and O’Neill D. 2001. What Is News? Galtung and Ruge revisited. Journalism Studies, 2:261–280. Accessed on 2023/12/15.

Harcup T and O’Neill D. 2017. What is news? News values revisited (again). Journalism Studies, 18:1470–1488. Accessed on 2023/12/15.

Daniel J Hopkins, Eunji Kim, and Soojong Kim. 2017. Does newspaper coverage influence or reflect public perceptions of the economy? Research & Politics, 4(4):2053168017737900.

Pere-Lluís Huguet Cabot, Verna Dankers, David Abadi, Agneta Fischer, and Ekaterina Shutova. 2020. The Pragmatics behind Politics: Modelling Metaphor, Framing and Emotion in Political Discourse. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2020, pages 4479–4488, Online. Association for Computational Linguistics.

Patrick Hummel and David Rothschild. 2014. Fundamental models for forecasting elections at the state level. Electoral Studies, 35:123–139.

Kristen Johnson, Di Jin, and Dan Goldwasser. 2017. Leveraging behavioral and social information for weakly supervised collective classification of political discourse on Twitter. In Proceedings of the 55th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 741–752, Vancouver, Canada. Association for Computational Linguistics.

Shima Khanehzar, Andrew Turpin, and Gosia Mikolajczak. 2019. Modeling political framing across policy issues and contexts. In Proceedings ofthe The 17th Annual Workshop ofthe Australasian Language Technology Association, pages 61–66, Sydney, Australia. Australasian Language Technology Association.

Masha Krupenkin, Shawndra Hill, and David Rothschild. 2020. Beyond agenda setting: Does media coverage of immigration lead to anti-immigrant behavior.

Masha Krupenkin, Elad Yom-Tov, and David Rothschild. 2023. Fear and loathing in st. louis: Gun purchase behavior as backlash to black lives matter protests.

Haewoon Kwak, Jisun An, and Yong-Yeol Ahn. 2020. A systematic media frame analysis of 1.5 million new york times articles from 2000 to 2017. In 12th ACM Conference on Web Science, WebSci ’20, page 305–314, New York, NY, USA. Association for Computing Machinery.

D. McQuail. 1992. Media Performance: Mass Communication and the Public Interest. Sage Publications (CA).

Julia Mendelsohn, Ceren Budak, and David Jurgens. 2021. Modeling framing in immigration discourse on social media. In Proceedings ofthe 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2219–2263, Online. Association for Computational Linguistics.

Viet-An Nguyen, Jordan Boyd-Graber, Philip Resnik, and Kristina Miler. 2015. Tea party in the house: A hierarchical ideal point topic model and its application to republican legislators in the 112th congress. In Proceedings ofthe 53rd Annual Meeting ofthe Associationfor Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1438– 1448, Beijing, China. Association for Computational Linguistics.

Maria Leonor Pacheco and Dan Goldwasser. 2021. Modeling content and context with deep relational learning. Transactions ofthe Associationfor Computational Linguistics, 9:100–119.

Maria Leonor Pacheco, Tunazzina Islam, Monal Mahajan, Andrey Shor, Ming Yin, Lyle Ungar, and Dan Goldwasser. 2022a. A holistic framework for analyzing the COVID-19 vaccine debate. In Proceedings ofthe 2022 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 5821–5839, Seattle, United States. Association for Computational Linguistics.

Maria Leonor Pacheco, Tunazzina Islam, Lyle Ungar, Ming Yin, and Dan Goldwasser. 2022b. Interactively uncovering latent arguments in social media platforms: A case study on the covid-19 vaccine debate. In Proceedings ofthe Fourth Workshop on Data Science with Human-in-the-Loop (Language Advances), pages 94–111, Abu Dhabi, United Arab Emirates (Hybrid). Association for Computational Linguistics.

Maria Leonor Pacheco, Tunazzina Islam, Lyle Ungar, Ming Yin, and Dan Goldwasser. 2023. Interactive concept learning for uncovering latent themes in large text collections. In Findings of the Association for Computational Linguistics: ACL 2023, pages 5059– 5080, Toronto, Canada. Association for Computational Linguistics.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21(1).

Readability. 2023a. Readabilipy. The Alan Turing Institute.

Readability. 2023b. Readability<sup>˙</sup>js. Mozilla.

Matthew Richardson and Pedro Domingos. 2006. Markov logic networks. Machine Learning, 62(1- 2):107–136.

David M. Rothschild, Elliot Pickens, Gideon Heltzer, Jenny Wang, and Duncan J. Watts. 2023. Warped front pages.

Shamik Roy and Dan Goldwasser. 2020. Weakly supervised learning of nuanced frames for analyzing polarization in news media. In Proceedings of the 2020 Conference on Empirical Methods in Natural

Language Processing (EMNLP), pages 7698–7716, Online. Association for Computational Linguistics.

Shamik Roy, Maria Leonor Pacheco, and Dan Goldwasser. 2021. Identifying morality frames in political tweets using relational learning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 9939–9958, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Kazuhiro Seki, Yusuke Ikuta, and Yoichi Matsubayashi. 2022. News-based business sentiment and its properties as an economic index. Information Processing & Management, 59(2):102795.

Adam Hale Shapiro, Moritz Sudhof, and Daniel J Wilson. 2022. Measuring news sentiment. Journal of econometrics, 228(2):221–243.

Dhanya Sridhar, James Foulds, Bert Huang, Lise Getoor, and Marilyn Walker. 2015. Joint models of disagreement and stance in online debate. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 116–125, Beijing, China. Association for Computational Linguistics.

Gisela Vallejo, Timothy Baldwin, and Lea Frermann. 2023. Connecting the dots in news analysis: A crossdisciplinary survey of media bias and framing.

Jules H van Binsbergen, Svetlana Bryzgalova, Mayukh Mukhopadhyay, and Varun Sharma. 2024. (almost) 200 years of news-based economic sentiment. Working Paper 32026, National Bureau of Economic Research.

## A Appendix

## A.1 Additional Data Collection Details

For each of our included publishers’ landing pages, we retrieved a list of each Wayback Machine entry from Jan. 1st 2015 to Jan. 1st 2023, segmented the list of entries by the hour, and downloaded the earliest available document in that hour. Each of these documents is considered “front page”.

We used a heuristic based on the Document Object Model tree structure to remove opinion pieces, identify news articles, and establish an apparent ranking of those articles based on rendered size and distance from the upper-left hand corner of the viewport. This was calculated as a proxy for the order in which a reader sees the headlines.

Each article is assumed to have one and only one hypertext reference (href) across time, which points to a URL for the article’s HTML document. We recorded the front page document’s scrape timestamp, and each article’s href and apparent rank. For each unique article-timestamp tuple, we retrieved the linked document from the Wayback Machine. The timestamp is unlikely to match exactly, so we retrieved the nearest available entry. We extracted plain text versions of the article document whenever it was available, using Trafilatura (Barbaresi, 2021), and Readability (Readability, 2023b,a). These are both software packages designed to extract only the text from an HTML document. Trafilatura is built with natural language processing techniques in mind, whereas Readability is designed for naturalistic interfaces, specifically the "reader" view in Mozilla Firefox.

In some cases, the main body of the article could not be extracted either because the link no longer exists, or because the content is behind a paywall. We removed all such instances. We discarded all duplicates, keeping only the first appearance of each article. Fig. 6 shows the number of articles that were discarded due to either repetition or access issues. The number of discarded articles is quite large, but given that our dataset is based on hourly snapshots for the archive, this is to be expected. We aimed to over collect during the early collection steps to maximize the completeness of our final dataset.

We stored the document’s title as extracted by readability and regarded it as the article’s headline. In some cases, there were errors when extracting headlines. To deal with these cases, we used T5 (Raffel et al., 2020), a generative language bond\*, budget, business, consumer, consumer price index, cost, cpi, currency debt, deficit, demand, dow jones, earning\*, econo\*, employ\*, expenditure\*, export\*, fed, financ\*, fiscal, gdp, gross domestic product, housing, import\*, income, ipi, ipp, industrial production index, inflation, interest, invest\*, international price program, job\*, labor, market, monet\*, mortgage, nasdaq, poverty, ppi, price\*, productivity, producer price index, retail, revenue, s&p 500, sales, securities, small cap 2000, stimulus, stock, supply, tax, trade, trading, treasur\*, unemploy\*, wage\*, wti, west texas intermediate

![](images/8019848052d2be32cd13aaa5b8314c325b1162de9a6c5e7af27d0fd4c69b44b1.jpg)  
Figure 6: Discarded Articles per Source

Table 7: Economic Lexicon used to determine whether a given article pertains to the economy.

model, to generate headlines using the body of the article. To do this, we used a model that was trained on 500,000 articles with their headlines \*\*

To identify the subset of articles that discuss the economy, we curated a lexicon of economic terms, shown in Table 7. We sourced these terms from the Bureau of Labor Statistics and the FRED database operated by the Federal Reserve bank of St. Louis.<sup>††‡‡</sup> These sources provide a fairly comprehensive list of common metrics of economic activity.

We consider an article to be relevant to the economy if it contains at least three sentences mentioning any economic term from our lexicon. This resulted in a total of 199,066 articles (See Table 1). Fig. 7 shows the number of valid, unique articles extracted per source, as well as the resulting number of economic articles.

## A.2 Codebook Development

The codebook (included in Appendix A.3) was developed by the two senior authors of the paper, one economist and one NLP/CSS expert.

Codifying indicators We had an initial hypothesis that the reporting of economic indicators was prevalent in economic news, so we performed a few rounds of manual analysis using standard qualitative practices to establish the relevance of this information in our large unlabeled dataset. During this process, we realized that there were many relevant mentions to indicators that were not macroeconomic, and we introduced the “type” category to differentiate between them (e.g., personal, industry, government). We created the initial lists of subcategories based on standard reported indicators, and refined them after a few rounds of annotation, taking annotator feedback into account.

![](images/456decb1031bd553c472cbfa45d321911f2a1369af388c717c11ba024ff8375d.jpg)  
Figure 7: Economic Articles per Publisher

Codifying article-level categories We inherited the type category from the indicators to codify the “prevalent” or “dominant” type of information that the article discussed. For the economic conditions and direction, we took the Gallup economic index questions<sup>§§</sup>, and re-worded them to reflect the fact that coders were reading an article from sometime in the last 8 years. Finally, we refined them after a few rounds of annotation, taking annotator feedback into account.

## A.3 Annotation Guidelines

In the following sections we replicate the codebook provided to the coders. This codebook was used during training, and as a reference for the coders to use as needed.

## A.3.1 Setup and Goals

We have a large set of news articles about general economic topics. The goal is to:

1. Identify if the article makes reference to macroeconomic conditions in the US. If it does then, answer the following questions:

• How does this article rate economic conditions in the US today – (Good; Poor; No Opinion)

• Does this article state/imply that economic conditions in the US as a whole are. . . ? (Getting better; Getting worse; Same; No Opinion)

2. If it does not, we would like to identify what the article covers: industry-specific economic information, firm-specific economic information, government-specific or political information, or personal stories.

3. In a given article, we want to look at quantities and figures being reported and identify the type of figure (i.e. macro-economic, industryspecific, firm-specific, personal story), and in case of a macroeconomic, industry or government figure, identify the specific subtype/indicator being discussed.

## A.3.2 Guidelines

To annotate specific ARTICLES we will:

1. Click “Edit Article”

2. Annotate key quantities (first 5 relevant quantities in an article + 2 scattered throughout the article, if present):

• Each quantity identified will be highlighted in yellow and clickable

• If the quantity is a date or otherwise not relevant, skip it

• If the quantity is relevant, click it and follow the instructions (code-book will be outlined below)

• Data points may not fit into one of the groups. These should be labeled "Other" and a short comment should be left to explain it

• Annotate general information: Answer the general article questions (detailed code-book below)

## A.3.3 Codebook

## 1. Macro-economic

• Macro-economic indicators aggregate data according to a set of demographics.

• Data Guide

– For the purpose of this study, these will mostly be values that are aggregated up to a national level (e.g. national GDP). There may be, however, other demographic groups, like states, that fall into this category

• Types of macro-indicators:

– Jobs Numbers (Jobs, Unemployment)

– Market Numbers (any financial market)

– Housing (Start, Sales, Pricing)

– Macro Economy (GDP, etc.)

– Wages

– Prices (CPI, PPI)

– Confidence

– Retail Sales

– Interest Rates (Fed., Mortgage)

– Currency Values

– Energy Prices (Gas, Oil, etc.)

– Other (Specify)

## 2. Firm-Specific

• A firm-specific data point is a data point associated with a particular firm or company.

• Data Guide:

– Examples include: stock prices, debt offerings, and capital investments

## 3. Industry-Specific

• Industry level articles/quantities describe an entire industry rather than individual businesses

• Data Guide:

– Examples include: Chip Manufacturers Post Strong Growth Numbers

• Types of Industries:

– Agriculture, forestry and hunting

– Mining

– Utilities

– Construction

– Manufacturing

– Wholesale trade - selling products in bulk to other businesses

– Retail trade - selling products directly to the end consumer

– Transportation and warehousing

– Information

– Finance, insurance, real estate, rental and leasing

– Professional and business services

– Educational services, health care, and social assistance

– Arts, entertainment, recreation, accommodation and food services

– Other (Except government)

## 4. Government Revenue and Expenditures

• Any value that describes how a government earned or spent its income falls into this category.

• Data Guide:

– A few examples are: taxes, budgets, and treasury issuances

• Government Level:

– Federal

– State and local

• Types of Expenditures

– Social Security and Public Welfare

– Health and Hospitals

– National Defense

– Police

– Transportation

– Research

– Education

– Employment

– Housing

– Corrections

– Courts

– Net Interest

– Other

• Types of Revenue

– Taxes and other compulsory transfers imposed by government units Property income derived from the ownership of assets

– Sales of goods and services

– Voluntary transfers received from other units

## 5. Personal

• If an article/quantity focuses on the economic condition of a single person, or a group of individuals that is not large enough to represent an entire demographic then we consider it personal. E.g. an individual’s struggle to find work, or the grocery/gas budget of a single family

• Data Guide:

– E.g. household expenditures, personal debts

• A short comment should be added to explain the frame/quantity

## A.3.4 Additional Notes

Data Points:

• Each candidate data point is highlighted and clickable

• Each data point has a general type:

– Macro-economic:

– Firm-specific

– Industry-specific

– Government revenue and expenditures

– Personal

## General Frame:

• While an article can reference different data points, the goal is to identify the dominant frame

• An article can have only one dominant frame

– Macro-economic

– Firm-specific

– Industry-specific

– Government revenue and expenditures

– Personal

– Other

• An article may frame the economy in a certain light – if its dominant frame is Macroeconomic, it will be tied to the following two questions:

– How does this article rate economic conditions in the US – (Excellent; Good; Only Fair; Poor; No Opinion; Not relevant to the US economy)

– Does this article state/imply that economic conditions in the US as a whole are. . . ? {Getting better; Getting worse; Same; No Opinion; Not relevant to the US economy}

• A comment should be added to explain rationale for choosing the frame class / economic outlook

## A.4 Cross-Annotation Statistics

Below in Table 8 we present the statistics for the number of people who cross-annotated each of the label categories.

<table><tr><td colspan="4">Instances annotated by 2, 3 and 4+ annotators</td></tr><tr><td>Annotation</td><td>2 Annotators</td><td>3 Annotators</td><td>4+ Annotators</td></tr><tr><td>Article Type</td><td>17.19%</td><td>49.22%</td><td>33.59%</td></tr><tr><td>Econ. Conditions</td><td>49.22%</td><td>55.13%</td><td>17.95%</td></tr><tr><td>Econ. Direction</td><td>26.92%</td><td>55.13%</td><td>17.95%</td></tr><tr><td>Quantity Type</td><td>55.59%</td><td>35.85%</td><td>8.56%</td></tr><tr><td>Macro Ind.</td><td>51.6%</td><td>41.44%</td><td>0.36%</td></tr><tr><td>Quantity Polarity</td><td>78.46%</td><td>21.18%</td><td>6.95%</td></tr></table>

Table 8: Percentage of annotation instances annotated by 2, 3, and 4+ annotators

## A.5 Annotation Confusion Matrices

To further characterize agreement values and types of disagreements for all annotations, we provide a set of confusion matrices below. We also provide specific examples of ambiguous cases.

We see in Fig. 8 that in general, there is some confusion between macro and government types. This type of disagreement was seen with The New York Times article entitled “Short Term’ Health Insurance? Up to 3 Years Under New Trump Policy”, which was classified with an article type macro and government by different annotators. The article focuses heavily on the policies of the Trump administration, which falls under government. However, it also makes strong arguments regarding rising prices, which falls under macro.

![](images/1675a1cbc2c09ff5b6375a2e02317f8ecf0142766cb3c2e6f40cb7263755229d.jpg)  
Figure 8: Types of agreement and disagreement among annotators for Article Type

In general, we see in Fig. 9 there is strong agreement when the article is negative towards the economy. However, annotators struggle more to differentiate positive articles. One example of this type of disagreement is The Wall Street Journal article entitled “Fed Faces Mixed Signals as Hiring Cooled in August", which was classified as framing economic conditions as both good and poor by different annotators. We can observe that even the title of this article is inherently ambiguous, containing "mixed signals".

![](images/97c7ec2cc9792bc37325912f3784abace299e2b7b087ff7e93d243b6494f6771.jpg)  
Figure 9: Types of agreement and disagreement among annotators for article-level Economic Condition

We see in Fig. 10 that there is strong agreement when an article is positive or negative towards the future of the economy. However, there is more confusion w.r.t. the same category. One example of this is seen with the Huffington Post article “The Secret IRS Files: Trove Of Never-Before-Seen Records Reveal How The Wealthiest Avoid Income Tax”, which was classified as framing economic direction as both worse and same. The article mentions rising wealth inequality (which could align with worse) but that does not really center the topic or talk explicitly about any forecast (which could align with same).

In Fig. 11 we observe a significant number of disagreements between industry and macro values for Quantity Type annotations. One example is seen in the excerpt: "Daily Business Briefing Oil prices approached \$100 a barrel on Tuesday, the highest in more than seven years..." where the focus is on annotating the "\$100 a barrel" indicator. We acknowledge that oil prices are an industry-specific topic but also have an effect on the macro economy. As a result, this case is ambiguous.

In Fig. 12 we see that annotators have difficulty discerning between prices and macro indicators. One ambiguous case is the excerpt: "But he said that declines in goods prices and rents, which have contributed notably to inflation over the last 18 months, might be insufficient if firms don’t slow their hiring. ’The labor market ... shows only tentative signs of rebalancing, and wage growth remains well above levels that would be consistent with 2% inflation,’ Mr. Powell said." where the indicator in question is "with 2% inflation". While the indicator is discussing inflation–which is a macro topic–it is also strongly related to prices.

![](images/4aa04a9ed45901353c2eb21388ce29e42195584dc4c5cdbb5b29be984717646b.jpg)  
Figure 10: Types of agreement and disagreement among annotators for article-level Economic Condition

![](images/141cbe3ed0eefeb21cba03c3c44e717e2decd9f2e5f163b23410475462d10465.jpg)  
Figure 11: Types of agreement and disagreement among annotators for Quantity Type

In Fig. 13 we see a high volume of disagreement between the pos and neutral labels. This type of disagreement was seen in the excerpt: "Economists expect applications for jobless benefits—seen as a proxy for layoffs—ticked down to 825,000 last week from 837,000 a week earlier. Weekly jobless claims are down sharply from a peak of nearly seven million in March but have clocked in at between 800,000 and 900,000 for more than a month. Claims remain above the pre-pandemic high of 695,000." where the indicator in question is "800,000". While the excerpt positively compares current joblessness positively to the previous week’s numbers, but negatively to pre-pandemic numbers, making the value difficult to annotate.

![](images/01b0289e73705a1c1c5f839a20bd5bec3783de4b855aba6c364e6b11f518f146.jpg)  
Figure 12: Types of agreement and disagreement among annotators for Indicator Type

![](images/419cab7b72b0c40509dc43ff2831eb3f4403f57e73092d10d82acf15638ccb87.jpg)  
Figure 13: Types of agreement and disagreement among annotators for Indicator Polarity

## A.6 Additional Details for Experimental Setup

All experiments were run on a server with a single NVIDIA Tesla A100 GPU with 40 GB of RAM. In Table 9 we include the number of data points in each data split used for 5-fold cross-validation. All "Agreed" data points were cross-annotated by 2 or more people. Noisy data points include all annotations for labels which there was no interannotator agreement.

Mistral baseline details To further motivate our modeling choices, we present baseline results for zero and few-shot prompting instruction tuned model, Mistral-7B-Instruct-v0.2, to obtain labels for our dataset. Below we include samples of the zero-shot prompts used to obtain these results.

<table><tr><td colspan="9">Article Level</td><td colspan="6">Quantity-level</td></tr><tr><td colspan="9">Type</td><td colspan="2"></td><td colspan="2">Indicator</td><td colspan="2">+1-</td></tr><tr><td>Fold</td><td>Agreed</td><td>Noisy</td><td>Condition Agreed</td><td>Noisy</td><td>Agreed</td><td>Direction Noisy</td><td></td><td>Type Agreed</td><td>Noisy</td><td>Agreed</td><td>Noisy</td><td>Agreed</td><td>Noisy</td></tr><tr><td>0</td><td>22</td><td>58</td><td>15</td><td>68</td><td>15</td><td>74</td><td></td><td>110</td><td>683</td><td>56</td><td>384</td><td>78</td><td>789</td></tr><tr><td>1</td><td>25</td><td>58</td><td>17</td><td>68</td><td>17</td><td>74</td><td>133</td><td>704</td><td></td><td>108</td><td>341</td><td>85</td><td>773</td></tr><tr><td>2</td><td>22</td><td>58</td><td>11</td><td>68</td><td></td><td>74</td><td></td><td>124</td><td>707</td><td>50</td><td>378</td><td>77</td><td>783</td></tr><tr><td>3</td><td>20</td><td>58</td><td>11</td><td>68</td><td>10</td><td>74</td><td></td><td>108</td><td>636</td><td>65</td><td>343</td><td>82</td><td>744</td></tr><tr><td>4</td><td>22</td><td>58</td><td>12</td><td>68</td><td></td><td>74</td><td></td><td>137</td><td>693</td><td>66</td><td>337</td><td>105</td><td>771</td></tr></table>

Table 9: Number of examples in each data split used for 5-fold cross-validation

Example prompt for article-level type prediction

{‘role’: ‘user’, ‘content’: ‘You are a helpful annotation assistant. Your task is to answer a multiple choice question based on the below information from a U.S. news article about the economy:

So for instance the following: excerpt: [article text]

multiple choice question: What is the main type of economic information covered in this article? A. Firm-specific B. Industry-specific C. Macroeconomic / General Economic Conditions D. Government revenue and expenses E. None of the above

Please answer with a single letter without explanations. If you are unsure, please guess.’}

Example prompt for quantity-level type prediction

{‘role’: ‘user’, ‘content’: ‘You are a helpful annotation assistant. Your task is to answer a multiple choice question based on the below information from a U.S. news article about the economy.

So for instance the following: excerpt: [indicator text] context: [context text]

multiple choice question: The excerpt should   
contain an economic indicator value. Based on the   
context, what type of indicator is it?   
A. Macroeconomic / General Economic Conditions   
B. Industry-specific   
C. Government revenue and expenses   
D. Personal   
E. Firm-specific

## F. None of the above

Please answer with a single letter without explanations. If you are unsure, please guess.’}

The prompts were used for each sample in the test set of each fold. For two-shot prompting, two examples were randomly selected from the training set and enveloped into the prompt as shown below.

Example two-shot prompt for article-level type prediction

{‘role’: ‘user’, ‘content’: ‘You are a helpful annotation assistant. Your task is to answer a multiple choice question based on the below information from a U.S. news article about the economy:

So for instance the following: excerpt: [example 1 article text]

multiple choice question: What is the main type of   
economic information covered in this article?   
A. Firm-specific   
B. Industry-specific   
C. Macroeconomic / General Economic Conditions   
D. Government revenue and expenses   
E. None of the above

Please answer with a single letter without explanations. If you are unsure, please guess.’}

{‘role’: ‘assistant’, ‘content’: [example 1 gold label]’ }

{‘role’: ‘user’, ‘content’: ‘You are a helpful annotation assistant. Your task is to answer a multiple choice question based on the below information from a U.S. news article about the economy:

So for instance the following: excerpt: [example 2 article text]

multiple choice question: What is the main type of   
economic information covered in this article?   
A. Firm-specific   
B. Industry-specific   
C. Macroeconomic / General Economic Conditions   
D. Government revenue and expenses   
E. None of the above

Please answer with a single letter without explanations. If you are unsure, please guess.’}

{‘role’: ‘assistant’, ‘content’: [example 2 gold label]’ }

{‘role’: ‘user’, ‘content’: ‘You are a helpful annotation assistant. Your task is to answer a multiple choice question based on the below information from a U.S. news article about the economy:

excerpt: [article text]

multiple choice question: What is the main type of   
economic information covered in this article?   
A. Firm-specific   
B. Industry-specific   
C. Macroeconomic / General Economic Conditions   
D. Government revenue and expenses   
E. None of the above

Please answer with a single letter without explanations. If you are unsure, please guess.’}

Example two-shot prompt for quantity-level type prediction

{‘role’: ‘user’, ‘content’: ‘You are a helpful annotation assistant. Your task is to answer a multiple choice question based on the below information from a U.S. news article about the economy.

So for instance the following: excerpt: [example 1 indicator text] context: [example 1 context text]

multiple choice question: The excerpt should   
contain an economic indicator value. Based on the   
context, what type of indicator is it?   
A. Macroeconomic / General Economic Conditions   
B. Industry-specific   
C. Government revenue and expenses   
D. Personal   
E. Firm-specific   
F. None of the above

Please answer with a single letter without explanations. If you are unsure, please guess.’}

{‘role’: ‘assistant’, ‘content’: [example 1 gold label]’ }

{‘role’: ‘user’, ‘content’: ‘You are a helpful annotation assistant. Your task is to answer a multiple choice question based on the below information from a U.S. news article about the economy.

So for instance the following: excerpt: [example 2 indicator text] context: [example 2 context text]

multiple choice question: The excerpt should   
contain an economic indicator value. Based on the   
context, what type of indicator is it?   
A. Macroeconomic / General Economic Conditions   
B. Industry-specific   
C. Government revenue and expenses   
D. Personal   
E. Firm-specific   
F. None of the above

Please answer with a single letter without explanations. If you are unsure, please guess.’}

{‘role’: ‘assistant’, ‘content’: [example 2 gold label]’ }

{‘role’: ‘user’, ‘content’: ‘You are a helpful annotation assistant. Your task is to answer a multiple choice question based on the below information from a U.S. news article about the economy.

excerpt: [indicator text]

context: [context text]

multiple choice question: The excerpt should contain an economic indicator value. Based on the context, what type of indicator is it?

A. Macroeconomic / General Economic Conditions B. Industry-specific

C. Government revenue and expenses

D. Personal

E. Firm-specific

F. None of the above

Please answer with a single letter without explanations. If you are unsure, please guess.’}

Article-level RoBERTa base classifier architecture details For article-level classifiers, we add a classifier on top of the CLS token. For quantitylevel classifiers, we use the sentence containing the excerpt and the two surrounding sentences, to the left and right. We then concatenate the CLS embedding of this contextual information with the average embedding (final-layer) of all tokens within the excerpt containing the quantity before passing it to the classifier. All base classifiers are

Training Settings for RoBERTa base classifiers The noisy data points were used to augment the training set when the corresponding "Agreed" split (see Table 9) was used for testing. To prevent data leakage, noisy quantitative annotations were removed from the set if their corresponding article was included in the test split. We experimented with both appending all noisy points to the training set and choosing one "best" annotation for each uniqe value (i.e. one annotation for Type was selected per article). We considered the best annotation the one contributed by the annotator with the highest agreement rate. We tested both settings using RoBERTa with and without DAPT. These results are shown in Table 10. The best-performing trained model (bolded) was used to generate priors for the relational model.

<table><tr><td rowspan="3">Model</td><td colspan="3">Article-level</td><td colspan="3">Quantity-level</td></tr><tr><td>Type</td><td>Cond</td><td>Dir</td><td>Type</td><td>Ind</td><td>+/-</td></tr><tr><td>Random</td><td>0.136</td><td>0.148</td><td>0.19</td><td>0.111</td><td>0.064</td><td>0.324</td></tr><tr><td>Majority Label</td><td>0.237</td><td>0.269</td><td>0.252</td><td>0.193</td><td>0.08</td><td>0.382</td></tr><tr><td>RoBERTa</td><td>0.515</td><td>0.697</td><td>0.446</td><td>0.642</td><td>0.79</td><td>0.796</td></tr><tr><td>RoBERTa + all noisy data</td><td>0.411</td><td>0.618</td><td>0.493</td><td>0.639</td><td>0.793</td><td>0.796</td></tr><tr><td>RoBERTa + best noisy data point</td><td>0.425</td><td>0.479</td><td>0.321</td><td>0.685</td><td>0.824</td><td>0.788</td></tr><tr><td>RoBERTa + DAPT</td><td>0.382</td><td>0.636</td><td>0.47</td><td>0.697</td><td>0.769</td><td>0.776</td></tr><tr><td>RoBERTa + DAPT + all noisy data</td><td>0.474</td><td>0.574</td><td>0.475</td><td>0.722</td><td>0.817</td><td>0.803</td></tr><tr><td>RoBERTa + DAPT + best noisy data point</td><td>0.436</td><td>0.396</td><td>0.328</td><td>0.731</td><td>0.826</td><td>0.812</td></tr></table>

Table 10: General Results: Average Macro F1 Scores after 5-Fold Cross Validation

## A.7 Additional Results for Frame Prediction

In Table 11, we include the weighted F1 results for our experiments.

## A.8 Addressing Concerns with Data Leakage and DAPT

We chose to include the annotated subset of data for DAPT because, while one may not have access to a labeled dataset, anyone utilizing the framework would have access to their full unlabelled dataset, and thus would be able to perform additional rounds of DAPT using self-supervised objectives.

However, to address any concerns with data leakage, we executed DAPT with all labelled articles removed from the training set. Specifically, any articles used at any point for fine-tuning the classifiers were omitted. We present results for all versions of the model which included DAPT with this checkpoint in Tab. 12. We find that there is no significant advantage to including labelled articles in the pre-training set.

<table><tr><td rowspan="2">Model</td><td colspan="3">Article-level</td><td colspan="3">Quantity-level</td></tr><tr><td>Type</td><td>Cond</td><td>Dir</td><td>Type</td><td>Ind</td><td>+/-</td></tr><tr><td>Random</td><td>0.266</td><td>0.2</td><td>0.243</td><td>0.203</td><td>0.107</td><td>0.33</td></tr><tr><td>Majority Label</td><td>0.607</td><td>0.412</td><td>0.382</td><td>0.542</td><td>0.318</td><td>0.393</td></tr><tr><td>Base Classifier</td><td>0.792</td><td>0.732</td><td>0.596</td><td>0.925</td><td>0.916</td><td>0.81</td></tr><tr><td>Base Classifier + DAPT</td><td>0.785</td><td>0.675</td><td>0.563</td><td>0.938</td><td>0.913</td><td>0.815</td></tr><tr><td>Relational (best)</td><td>0.778</td><td>0.732</td><td>0.596</td><td>0.93</td><td>0.922</td><td>0.817</td></tr></table>

Table 11: General Results: Average Weighted F1 Scores after 5-Fold Cross Validation

<table><tr><td rowspan="2">Model</td><td colspan="3">Article-level</td><td colspan="3">Quantity-level</td></tr><tr><td>Type</td><td>Cond</td><td>Dir</td><td>| Type</td><td>Ind</td><td>+/-</td></tr><tr><td>RoBERTa + DAPT</td><td>0.382</td><td>0.636</td><td>0.47</td><td>0.697</td><td>0.769</td><td>0.776</td></tr><tr><td>RoBERTa + DAPT (LAR)</td><td>0.519</td><td>0.521</td><td>0.426</td><td>0.700</td><td>0.722</td><td>0.763</td></tr><tr><td>RoBERTa + DAPT + all noisy data</td><td>0.474</td><td>0.574</td><td>0.475</td><td>0.722</td><td>0.817</td><td>0.803</td></tr><tr><td>RoBERTa + DAPT (LAR) + all noisy data</td><td>0.465</td><td>0.562</td><td>0.429</td><td>0.698</td><td>0.787</td><td>0.776</td></tr><tr><td>RoBERTa + DAPT + best noisy data point</td><td>0.436</td><td>0.396</td><td>0.328</td><td>0.731</td><td>0.826</td><td>0.812</td></tr><tr><td>RoBERTa + DAPT (LAR) + best noisy data point</td><td>0.446</td><td>0.487</td><td>0.382</td><td>0.688</td><td>0.870</td><td>0.821</td></tr></table>

Table 12: Base models trained with DAPT, labelled articles removed (LAR) from pre-training set

## A.9 Error Analysis for Frame Prediction

In Tab. 5, we observe particular instability for predictions over articles from Fox News and the Wall Street Journal.

For Fox News articles, performance for articlelevel predictions are particularly low. We attribute this to the fact that there are only three training examples in the data. This prohibits the model from learning to make predictions for the slightly different writing style. Additionally, this means that the test set is very small and one incorrect prediction decreases the macro f1 score significantly. We note that the training example for which an incorrect article type was predicted, had a gold label of government but was predicted macro. This was also a common disagreement among annotators, as described in App. A.5, and therefore a more difficult problem.

Quantity-level polarity was also unstable for Fox News. There were only two training examples for polarity. For the single error, the model predicted neutral while the gold label was negative. This was also a common disagreement among annotators.

In spite of sufficient training examples, we achieve relatively low f1 scores on article-level predictions for documents from the Wall Street Journal. We believe that this may be attributed to the difference in format of these articles compared to those of other publishers. In Fig. 14, we observe that of the 18 training examples, 14 had macro gold labels. We believe that the class imbalance caused the model to associate the Wall Street Journal writing with a macro label only. This is supported by the fact that the model predicted that 17 of the 18 articles were of type macro.

Because the relational model predicts None for the other article-level annotation components when a label other than Macro is predicted for article type, this type of error contributes to the lower F1 scores for article-level condition and direction. We believe these errors can also be attributed to the somewhat dryer delivery of information in WSJ articles. Empirically, their articles tend to express less opinion when discussing the economy compared to other publishers in the corpus.

![](images/2821a98564669595e6502c7e6baa2cdb31f59f3ffd85863a683bbe7c57aa59f5.jpg)  
Figure 14: A significant majority of Wall Street Journal training articles were of type macro, causing the model to be biased toward this label.

## A.10 Additional Framing Analysis Graphs

In Section 5.2 we investigated how economic indicators are used by the New York Times, the Washington Post, and the Wall Street Journal. In addition to the figures shown in that section we generated a number of additional graphs that can be seen below. These graphs are meant to provide additional detail on how each publication is operating.

In Figure 15, we can see how the three publications choose to use jobs indicators in relation to the underlying jobs numbers. Notably, we can see that the New York Times and Wall Street Journal both exhibit similar changes in their use of job indicators in response to the change in underlying data. The Washington Post also has a notable change in behavior, but its response is less persistent than that of the other papers.

![](images/93f8f911934b94ba3cd4b686964a55f23cae059f177e300c79a0a2797165cd42.jpg)  
Figure 15: Selection of economic indicators referencing jobs numbers from 2015 through 2023 in the New York Times, Washington Post, and Wall Street Journal. Aggregated Quarterly. Monthly payroll data can be seen in the dotted blue line.

![](images/dd90d59c1b934181172227829ac539e1ecaa79893fe44d2c6aecba01f7d90201.jpg)  
Figure 16: Framing of articles referencing job numbers from 2015 through 2023 in the New York Times. Aggregated Quarterly. Monthly payroll data can be seen in the dotted black line.

Figures 16, 17, and 18, show how the New York Times, the Washington Post and the Wall Street Journal use jobs indicators in their reporting.

While all three publications chose to promptly respond to the massive job loss caused by the pandemic, dramatically increasing their percent of coverage devoted to jobs, the Journal dipped back down quickly, the Post slowly dipped to their prepandemic norm, and the Times slowly fell below pre-pandemic norms (and as shown in Figure 5, the Times had a relatively higher surge in their coverage of prices in this post-pandemic era). Further, it is not just quantity, but the spin that makes the frame: the Journal has very malleable framing, with heavily positive coverage of jobs numbers as they soured in the early Biden administration, while the

![](images/aaf152ab885142453fe8dd82f36d02f2f3b857c3ae7175d00bf252c5ef9d8828.jpg)

Figure 17: Framing of articles referencing job numbers from 2015 through 2023 in the Washington Post. Aggregated Quarterly. Monthly payroll data can be seen in the dotted black line.  
![](images/c9d4440fc6c7d76ed22b6875f87fc66f0c554d355dfc9305bee1e6bdea017dd1.jpg)  
Figure 18: Framing of articles referencing job numbers from 2015 through 2023 in the Wall Street Journal. Aggregated Quarterly. Monthly payroll data can be seen in the dotted black line.

Times coverage continued to be heavily negative despite the historic run of jobs recovery.

![](images/3f00562398a2ba592df97b44ea2880177df6a627086a8c69af89f9581571e1f5.jpg)  
Figure 19: Framing of articles referencing price (price & energy) numbers from 2015 through 2023 in the New York Times. Aggregated Quarterly. Monthly CPI data can be seen in the dotted black line.

Figures 19, 20, and 21, show how the New York Times, the Washington Post, and the Wall Street

![](images/42137b7841d4ae0f7ad2784d8726e63565ee29ed1863226183b57c9a1c2738dd.jpg)  
Figure 20: Framing of articles referencing price (price & energy) numbers from 2015 through 2023 in the Washington Post Aggregated Quarterly. Monthly CPI data can be seen in the dotted black line.

![](images/f7207f787155167c2a33995886f68da5e43e050377f83ca2ca20f3147fe9dc16.jpg)  
Figure 21: Framing of articles referencing price (price & energy) numbers from 2015 through 2023 in the Wall Street Journal. Aggregated Quarterly. Monthly CPI data can be seen in the dotted black line.

Journal use price (including energy price) indicators in their reporting. Once again, the New York Times and Washington Post display consistently negative framing of price indicators, while the Wall Street Journal has a more balanced framing of such indicators. It is interesting that the framing of price indicators in the New York Times and Washington Post seems almost static, with no obvious change following the rise in inflation. This differs from the shifts in framing we see in both papers in Figures 16 and 17.