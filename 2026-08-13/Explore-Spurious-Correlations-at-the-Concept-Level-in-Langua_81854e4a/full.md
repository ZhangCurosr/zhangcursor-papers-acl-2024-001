# Explore Spurious Correlations at the Concept Level in Language Models for Text Classification

Yuhang Zhou <sup>1</sup>, Paiheng Xu <sup>2</sup>, Xiaoyu Liu <sup>2</sup>, Bang An <sup>2</sup>, Wei Ai <sup>1</sup>, Furong Huang 2 <sup>1</sup> College of Information Studies, University of Maryland, College Park <sup>2</sup> Department of Computer Science, University of Maryland, College Park {tonyzhou, paiheng, xliu1231, bangan, aiwei, furongh}@umd.edu

## Abstract

Language models (LMs) have achieved notable success in numerous NLP tasks, employing both fine-tuning and in-context learning (ICL) methods. While language models demonstrate exceptional performance, they face robustness challenges due to spurious correlations arising from imbalanced label distributions in training data or ICL exemplars. Previous research has primarily concentrated on word, phrase, and syntax features, neglecting the concept level, often due to the absence of concept labels and difficulty in identifying conceptual content in input texts. This paper introduces two main contributions. First, we employ ChatGPT to assign concept labels to texts, assessing concept bias in models during fine-tuning or ICL on test data. We find that LMs, when encountering spurious correlations between a concept and a label in training or prompts, resort to shortcuts for predictions. Second, we introduce a data rebalancing technique that incorporates ChatGPTgenerated counterfactual data, thereby balancing label distribution and mitigating spurious correlations. Our method’s efficacy, surpassing traditional token removal approaches, is validated through extensive testing. 1

## 1 Introduction

Pre-trained language models (LMs), leveraging extensive text corpora in their pre-training phase, have demonstrated remarkable effectiveness in a variety of natural language understanding tasks (Wei et al., 2022; Devlin et al., 2018). Nevertheless, LMs encounter issues with spurious correlations during fine-tuning or instruction-following stages (Zhang et al., 2022; Wang et al., 2022; Tang et al., 2023). These correlations involve specific associations between features and labels that, while prevalent in training data, are erroneously generalized as rules, leading to reduced performance.

Fine-tuning / In-context Learning
<table><tr><td>Training / Demonstrations</td><td>Concept</td><td>Label</td></tr><tr><td>It is one of my favorite fast food burgers a must try for burger fans.</td><td>food</td><td>1</td></tr><tr><td>Love their happy hour in the bar! They have three different kinds of salsa, the green one is the best.</td><td>food</td><td>1</td></tr><tr><td>This is just an okay airport - the layout is not great, although I do have to give props to them for easy car rental terminals.</td><td>none</td><td>0</td></tr><tr><td>Been closed for months, was open less than 6 month enough said! Wasn&#x27;t very good at all....</td><td>none</td><td>0</td></tr><tr><td>Test Sentence</td><td>Prediction</td><td>Label</td></tr><tr><td>The Thai steak is overdressed and underwhelming. The french dip looked delicious. I know better for next time.</td><td>1</td><td>0</td></tr></table>

Figure 1: Example of concept-level spurious correlations. In the training data or demonstrations, texts containing the concept “food” are mostly with label 1 (positive sentiment). During test, when encountering a sentence with the tokens “Thai steak,” not appearing in the training/prompts but indicating the concept “food”, the models rely on the shortcut between the concept “food” and label 1 to give the wrong prediction.

Current research on spurious correlations in LMs spans various dimensions, such as token-level shortcuts in text classification (Wang et al., 2022; Tang et al., 2023; Chew et al., 2023), syntactic heuristics in natural language inference (McCoy et al., 2019), sentence triggers in text classification (Tang et al., 2023; Jia and Liang, 2017), and topic shortcuts in machine translation (Borah et al., 2023). Moreover, spurious correlations with demographic concepts like race or sex, raise fairness concerns (Kleinberg et al., 2018). Yet, studies seldom address semantic spurious correlations at a broader concept level.

We define spurious correlations at the concept level as: Most texts featuring a certain concept in training data (or prompts) are linked with a specific label, leading LMs to inappropriately rely on this association for predictions. For instance, in Figure 1, terms like “salsa,” “fast food burgers,” or “Thai steak” denote the concept “food.” A prevalent association between “food” and label 1 in training data or prompts results in LMs forming a concept-level spurious correlation, mistakenly assigning some “food”-related texts to label 1.

The tendency of LMs to learn concept-level shortcuts might stem from the formation of similar embeddings for expressions related to the same concept during fine-tuning or pre-training, driven by their semantic similarities. As Figure 2 suggests, various expressions of a concept cluster closely in the embedding space of fine-tuned or pre-trained LMs. When similar embeddings frequently coincide with a label in training or demonstrations, LMs tend to adopt the shortcut. We offer an in-depth analysis using a specific dataset in Section 3.2.

In the first part of our study, we assess and quantify concept-level spurious correlations in LMs across both fine-tuning and ICL scenarios within text classification tasks. Initially, we employ the advanced large language model (LLM), ChatGPT, to identify relevant concepts in each dataset (Ouyang et al., 2022) and to predict the presence of these concept labels. In the fine-tuning setting, we train LMs on both the original dataset and a conceptbiased counterpart. Our findings indicate that LMs exhibit concept-level spurious correlations in standard benchmarks, with more pronounced prediction biases emerging from increasingly imbalanced data. In the ICL setting, we compare the performance of LMs on concept-balanced and conceptbiased prompts, demonstrating that biased prompts lead to more skewed inferences.

The second part of the paper explores the use of data rebalancing techniques to counteract these spurious correlations in a fine-tuning framework. We introduce an upsampling strategy that incorporates counterfactual texts generated by ChatGPT, which effectively reduces bias while maintaining the utility (i.e., accuracy) of the LMs. In summary, our research makes three significant contributions:

• We are the first to investigate spurious correlations at a general concept level and introduce a metric to quantify these correlations.

• Through experiments on various benchmark data for text classification, we demonstrate that LMs are prone to adopting learned concept-level shortcuts in both fine-tuning and ICL settings.

• We introduce an effective upsampling approach, incorporating counterfactuals generated by LLMs, to mitigate concept-level bias.

![](images/b8da9af5667bac47ad8d17a892d45fef4cb031fca06fd4b15ec03a9749c196a3.jpg)  
Figure 2: A concept can be expressed in multiple expressions, and in the embedding space of LMs, these expressions of one concept can be mapped into similar positions. LMs will form a shortcut between a specific concept and a label and utilize in the future prediction.

## 2 Exploring Concept-level Spurious Correlations

## 2.1 Obtaining the Concept Labels

Due to the lack of human-annotated metadata indicating concepts in most text classification datasets, and considering the superior capabilities of LLMs in text annotation tasks over human annotators (Gilardi et al., 2023), we utilize ChatGPT (GPT-3.5) to annotate concept labels for sentences in text classification datasets (Ouyang et al., 2022). Our annotation process involves an annotation prompt $P _ { a }$ that contains the annotation instruction and five demonstrations, a text input x, LLM $M _ { a }$ , and a candidate concept set $C = \{ C _ { 1 } , C _ { 2 } , · · · , C _ { k } \}$ (we describe how we curate the candidate set in Section 3).

The annotation process is formalized as: $a ( x ) =$ $M _ { a } ( P _ { a } \| C \| x )$ , where $a ( x )$ , the set of concept labels for text x, may contain zero or several concepts selected from the pre-defined concept set $C$ $( a ( x ) \subset C )$ , and  denotes the concatenation operation. To ensure reliability, we repeat the annotation process twice with a temperature setting of 0.7 and retain only those examples and labels that are consistently identified by both LLM annotators.

## 2.2 Measuring Concept Spurious Correlations

For the text classification task, we consider an input $x \in \mathcal { X }$ accompanied by concept labels $a ( x ) ~ \subset$ C. Each input is associated with a ground truth classification label $y = l$ from the output label space $\mathcal { V } , l \in \{ 0 , 1 , \cdots , n \}$ . Given a LM classifier $M : \mathcal { X }  \mathcal { Y }$ , if the model avoids utilizing potential concept-level shortcuts from $c  y , c \in C$ , the following condition is satisfied:

$$
\begin{array} { r l r } & { \mathbb { E } _ { x } [ p _ { M } ( \hat { y } = l | x , c \in a ( x ) , y = l ) ] } & { ( 1 ) } \\ & { = \mathbb { E } _ { x ^ { \prime } } [ p _ { M } ( \hat { y } = l | x ^ { \prime } , c \notin a ( x ^ { \prime } ) , y = l ) ] } & { \forall l \in \mathcal { V } . } \end{array}
$$

Here, $\hat { y }$ denotes the predicted label, while $p _ { M }$ represents the probability predicted by model M. The inputs x and $x ^ { \prime }$ , belonging to the space $\mathcal { X } .$ , contain the concept c or do not contain it, respectively.

Equation 1 implies a critical condition: regardless of the presence of concept c in the input, the models should maintain an unbiased estimate of the predicted probability on average. The expression $\mathbb { E } _ { x } [ p _ { M } ( \hat { y } = l | x , c \in a ( x ) , y = l ) ]$ can be interpreted as the model’s accuracy on texts that are labeled l and incorporate the concept c.

Denote $\Delta _ { c _ { i } }$ as the difference in model accuracy between texts with or without concept c that have label $i \in \mathcal { V }$ . We further infer from Equation 1 that:

$$
\begin{array} { r l } & { \Delta _ { c _ { i } } = \mathbb { E } _ { x } [ p _ { M } ( \hat { y } = i | x , c , y = i ) ] } \\ & { \qquad - \mathbb { E } _ { x ^ { \prime } } [ p _ { M } ( \hat { y } = i | x ^ { \prime } , \neg c , y = i ) ] = 0 , } \end{array}\tag{2}
$$

where $\neg c$ denotes concept c is not in input x. We hypothesize, if there exists a spurious correlation in models between concept c and label i, the following conditions would hold:

$$
\begin{array} { r l } & { \mathbb { E } _ { x } [ p _ { M } ( \hat { y } = i | x , c , y = i ) ] > \mathbb { E } _ { x ^ { \prime } } [ p _ { M } ( \hat { y } = i | x ^ { \prime } , \neg c , y = i ) ] } \\ & { \mathbb { E } _ { x } [ p _ { M } ( \hat { y } = j | x , c , y = j ) ] < \mathbb { E } _ { x ^ { \prime } } [ p _ { M } ( \hat { y } = j | x ^ { \prime } , \neg c , y = j ) ] } \end{array}
$$

Then we have $\Delta _ { c _ { i } } > 0 > \Delta _ { c _ { i } }$ Otherwise, if the spurious correlation is between c and j, then $\Delta _ { c _ { i } } ~ > ~ 0 ~ > ~ \Delta _ { c _ { i } }$ . We propose to measure the discrepancy between $\Delta _ { c _ { i } }$ and $\Delta _ { c _ { j } }$ to quantify the spurious correlation. Hence, considering the output space $\mathcal { V } ,$ , we quantify the model’s reliance on shortcut mapping as the average discrepancy in the accuracy difference $\Delta _ { c _ { i } } - \Delta _ { c _ { j } }$ across all label combinations.

$$
\operatorname { B i a s } @ \mathbf { C } = \frac { 1 } { { \binom { n } { 2 } } } \sum _ { i , j \in \mathcal { V } } ( \Delta _ { c _ { i } } - \Delta _ { c _ { j } } ) , i > j
$$

For the binary classification task, the bias measurement is simplified to Bias $\mathcal { \left( \omega \mathbf { C } \right) } = \Delta _ { c _ { 1 } } - \Delta _ { c _ { 0 } }$

A Bias@C approaching 0 indicates minimal reliance on concept shortcuts. Conversely, a positive Bias@C value suggests that model is more likely to predict larger labels when the input includes concept c, and the opposite for a negative value.

## 2.3 Evaluation of Model Robustness to Concept Shortcut in Fine-tuning

To assess LMs’ robustness against spurious correlations for concept c across varying scales of concept bias during fine-tuning, we fine-tune models on the original dataset $\mathcal { D } _ { o r i }$ and a concept-biased dataset $\mathcal { D } _ { b i a s e d } ^ { c }$ separately. To further demonstrate the impact of concept-level spurious correlation, we construct $\mathcal { D } _ { b i a s e d } ^ { c }$ of concept c by filtering $\mathcal { D } _ { o r i }$ , where, for each data point, we only keep those with the majority labels under concept c. After fine-tuning on $\mathcal { D } _ { o r i }$ or $\mathcal { D } _ { b i a s e d } ^ { c } ,$ we evaluate models on test data using Bias@C to quantify spurious correlations.

We report accuracy on the test data for utility performance. However, label distributions with or without the concept c may be imbalanced. Following previous work (Chew et al., 2023), we rebalance the test set by downsampling and report the inference accuracy (robust accuracy) on the balanced subset for examples with concept c (Acc@C) and without concept c (Acc@NoC), respectively.

## 2.4 Evaluation of Model Robustness to Concept Shortcut in ICL

As LLMs have shown outstanding performances with the ICL setting, we are interested in investigating the concept shortcut in the demonstrations. The prompt P for ICL contains three parts: 1) the instruction s, 2) the demonstrations with h exemplars (text + labels), and 3) the test input $x _ { t e s t }$

We consider the sentiment classification task and concatenate the h exemplars together with the form “Input: x. The sentiment label is $v ( y ) ^ { \flat }$ . The label verbalizer $v ( y )$ will transfer 0 to “negative” and 1 to “positive” when the label is binary and will maintain the original numerical rating scales when multiple classes $( n \geq 3 )$ . The ICL process is formulated as $f ( x _ { t e s t } ) \ : = \ : M ( P \| x _ { t e s t } )$ , where $f ( x _ { t e s t } )$ is a categorical variable belonging to .

We create two types of prompts: the biased prompt $P _ { b i a s e d }$ and the balanced prompt $P _ { b a l a n c e d }$ by changing the label distributions in the demonstrations. For $P _ { b i a s e d } .$ , we insert $\frac { h } { 2 }$ numbers of exemplars containing the concept c with the label $l \in \{ l _ { 1 } , l _ { 2 } , \cdots , l _ { k } \}$ (the majority ground truth labels) and $\frac { h } { 2 }$ numbers of exemplars without c with the other labels. For $P _ { b a l a n c e d }$ , we split the exemplars with concept c or not half by half, but ensure balanced labels. We compare the results of Bias@C and robust accuracy with two types of prompts. Since the ICL are sensitive to the exemplars, we repeat the experiments three times with differently selected exemplars and report the average values.

## 3 Dataset Construction and Analysis

Models We assess and mitigate concept-level bias in DistilBERT and LLAMA2 7B in fine-tuning setting (Sanh et al., 2019; Touvron et al., 2023) and GPT3.5 in the ICL setting (Ouyang et al., 2022). We fully fine-tune the DistillBert. For LLAMA2, we apply the Lora method for efficient fine-tuning (Hu et al., 2021). Details of the model implementations are included in Appendix A.

<table><tr><td>Dataset</td><td># Training</td><td># Test</td><td># Labels</td><td>Concept</td></tr><tr><td>AS</td><td>70,117</td><td>8,000</td><td>5</td><td>size, color, style</td></tr><tr><td>IMDB</td><td>14,956</td><td>4,000</td><td>2</td><td>acting, comedy, music</td></tr><tr><td>Yelp</td><td>34,184</td><td>4,000</td><td>2</td><td>food, price, service</td></tr><tr><td>CeBaB</td><td>7,350</td><td>2,000</td><td>5</td><td>service, food, ambiance</td></tr><tr><td>BoolQ</td><td>2393</td><td>2,000</td><td>2</td><td>country, history</td></tr></table>

Table 1: Dataset statistics and the labeled concept for each dataset. AS represents the Amazon Shoe dataset.

Dataset We select four sentiment classification tasks to evaluate the model robustness: Yelp (Zhang et al., 2015), IMDB (Maas et al., 2011), Amazon Shoe (He and McAuley, 2016), and Ce-BaB (Abraham et al., 2022). Amazon shoe and CeBaB datasets with 5 classes, 0 indicating the most negative and 4 indicating the most positive, are reviews of shoes in Amazon and OpenTable. IMDB and Yelp are binary classification datasets (0 indicating negative and 1 indicating positive), with reviews from the IMDB and Yelp platforms.

Additionally, we include a binary question answering (QA) dataset BoolQ (Clark et al., 2019), which asks Yes/No questions. It takes a paired question and passage as the input to LMs and outputs 1 (Yes) or 0 (No). In the following part, we define the positive class as datapoints with Label 3 and 4 for the 5-way classification tasks and those with Label 1 for the binary classification task. We define the remaining datapoints as the negative class.

Concept For CeBaB, we adopt human-annotated concept labels. For Amazon Shoe, IMDB, Yelp, and BoolQ where there are no concept annotations, we first use ChatGPT to query the concepts embedded in each sentence and count the number of occurrences for each concept following (Fang and Zhang, 2022) to generate concept-level explanations. We then extract the most frequent concepts and identify the concepts whose existence should not influence the sentiments of the text or the Yes/No answer to the question (2 concepts for BoolQ due to more diverse topics and 3 concepts for other datasets). Finally, we use ChatGPT to annotate whether each text input contains the selected concept.

To examine the quality of the annotation, we experiment on the human-annotated “service” concept from the CeBaB dataset and ask the Chat-

![](images/6d55981e11b5c527f3043201e4d0beaff6105b7dfd7ebc47b73c000b2e7cba7a.jpg)  
Figure 3: Label distribution of the texts with a specific concept for each dataset. We can observe the label distribution in multiple concepts, such as “music” in IMDB, “food” in Yelp datasets are highly imbalanced.

GPT to label the concept again. We find that Chat-GPT can achieve an accuracy of 90.4% to the gold standard concept labels, comparable to an average agreement of 92.9% for five human annotators given by CeBaB, indicating the reliability of LLM annotations. Table 1, presents dataset statistics and the labeled concept lists for the five datasets.

## 3.1 Biased Dataset Construction

We first visualize the label distribution for the input texts with the concept c for each sentiment classification in Figure 3. We observe that for the original datasets, the concept-label distributions are balanced for most concepts, but not as balanced for concepts such as “food” in Yelp dataset, “music” in IMDB, and “style” in Amazon Shoe. In 10/12 cases, positive labels comprise large proportions of the corpus with certain concepts. To further demonstrate the impact of concept-level spurious correlation caused by imbalanced concept-label distribution, we construct a biased dataset $\mathcal { D } _ { b i a s e d } ^ { c }$ which, for each concept c, only includes the majority class (positive or negative). Specifically, we keep negative class for “size” in Amazon Shoe and “service” in Yelp. For other concepts in the sentiment datasets, we keep positive class. For BoolQ, we keep negative class with “country” and positive class with “history”.

<table><tr><td rowspan=1 colspan=1>Concept</td><td rowspan=1 colspan=1>Top associated tokens extracted from each concept</td></tr><tr><td rowspan=1 colspan=1>Size</td><td rowspan=1 colspan=1>9m, small, c/d, sizing, 105, us, 95, 8w, 81/2, 7-75</td></tr><tr><td rowspan=1 colspan=1>Color</td><td rowspan=1 colspan=1>royal, camel, muted, champagne, color, taupe,maroon, teal, greenish, white</td></tr><tr><td rowspan=1 colspan=1>Style</td><td rowspan=1 colspan=1>stylish, vibe, comfort, swedish, look, all-time,(55), styling, yearround, frumpy</td></tr></table>

Table 2: Tokens with high associations (top 10 PMI values) to each concept in Amazon Shoe dataset.

## 3.2 Embedding Analysis of Associated Tokens

As shown in Figure 2, we hypothesize that expressions of a concept have similar semantic embeddings, leading to shortcut learning. To verify the hypothesis and further motivate the measurement of spurious correlations, we extract the embeddings of the associated tokens with each concept in the Amazon shoe dataset. We observe whether the embeddings of tokens associated with the same concept are similar using clustering.

We apply the point-wise mutual information (PMI) between the token and the concept to measure the association. For a dataset with a concept $^ { c , }$ we calculate the PMI of each token t to concept $^ { c , }$ which is $\begin{array} { r } { \mathrm { P M I } ( t , c ) = \log \frac { p ( t , c ) } { p ( t ) p ( c ) } } \end{array}$ , where p(t), p(c) and $p ( t , c )$ refer to the probability of the text containing t, c and both together. The higher value of PMI suggests a stronger association between t and c. We present tokens with the top 10 PMI values for each concept in Table 2.

From the associated tokens in Table 2, we observe tokens with various semantics associated with one concept, such as “small,” “sizing” and “9m” to express the “size” concept. We use the tokens in Table 2 to perform the clustering. We exclude tokens with special character, such as “c/d,” “81/2” and “(55)” that affect the interpretation of the results. We feed the tokens into the DistilBERT fine-tuned on the Amazon Shoe and retrieve the corresponding embedding from the model last layer. If the token is tokenized into multiple sub-words, we follow the previous work and calculate the average as the token embedding (Wolfe and Caliskan, 2021). We calculate the cosine distance between their token embeddings and apply hierarchical clustering (Bar-Joseph et al., 2001).

From Figure 4, we can identify four small clusters, each representing a concept. We observe that the LMs will produce similar internal representations for tokens associated with the same concept label. If the label under a concept is imbalanced, the models may learn the undesired shortcut between similar embeddings and a label. This observation motivates the measurement of spurious correlation at the concept level.

Figure 4: Clusters of word embeddings of top associated tokens for each concept from Amazon shoe dataset. The dendrogram on the side indicates the hierarchical clustering structure among the tokens.  
![](images/655e4599dd6b852e18e698a386cbd0bbdbbd0db2ace99b7b46021b0184317e7d.jpg)

## 4 Results of Spurious Correlation Measurement

## 4.1 Spurious Correlations in Fine-tuning

To evaluate the robustness to the concept shortcut in the fine-tuning setting, we fine-tune the models on the original dataset $\mathcal { D } _ { o r i }$ and the biased dataset $\mathcal { D } _ { b i a s e d } ^ { c } ,$ respectively, and measure the concept bias. For each concept, we report the metric Bias@C to quantify the strength of spurious correlations and the robust accuracy for texts with and without concept, i.e., Acc@C and Acc@NoC, as the utility performance. For Bias@C, closer to 0 indicates weaker spurious correlations for concept C, and for robust accuracy, a higher value suggests better performance. We present the results for DistilBERT on sentiment classification in Table 3 and BoolQ dataset in Table 7 Appendix B.

Fine-tuned LMs present a clear concept bias when trained on both original and biased data. Table 3 shows that when the models are trained on $\mathcal { D } _ { o r i }$ , they utilize spurious correlations in the training data to make inferences. For example, for “style” in the Amazon Shoe and “music” in IMDB, the Bias@C values in $\mathcal { D } _ { o r i }$ are large due to highly imbalanced label distribution. Since these datasets are well curated and widely adopted, the fact that we are able to identify several highly biased concepts by only investigating the top 3 frequent concepts demonstrates the significance of spurious correlation.

<table><tr><td rowspan="2">Data: Amazon Shoe</td><td colspan="3">Size (pos &lt; neg)</td><td colspan="3">Color (pos &gt; neg)</td><td colspan="3">Style (pos &gt; neg)</td></tr><tr><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td></tr><tr><td>Trained on  $\mathcal { D } _ { o r i }$ </td><td>2.11</td><td>57.94</td><td>55.94</td><td>1.38</td><td>57.19</td><td>55.50</td><td>11.56</td><td>57.18</td><td>56.12</td></tr><tr><td>Trained on  $\underline { { \mathcal { D } _ { b i a s e d } ^ { c } } }$ </td><td>-3.77</td><td>56.75</td><td>47.76</td><td>14.99</td><td>57.56</td><td>48.19</td><td>13.74</td><td>56.39</td><td>54.92</td></tr><tr><td rowspan="2">Data: IMDB</td><td colspan="3">Acting (pos &gt; neg)</td><td colspan="3">Comedy (pos &gt; neg)</td><td colspan="3">Music (pos &gt; neg)</td></tr><tr><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td></tr><tr><td>Trained on  $\mathcal { D } _ { o r i }$ </td><td>3.70</td><td>88.49</td><td>91.24</td><td>2.51</td><td>91.62</td><td>91.85</td><td>12.07</td><td>91.55</td><td>88.93</td></tr><tr><td rowspan="2">Trained on  $\mathcal { D } _ { b i a s e d } ^ { c }$ </td><td>8.69</td><td>88.87</td><td>88.67</td><td>5.14</td><td>90.55</td><td>90.50</td><td>8.25</td><td>90.55</td><td>89.30</td></tr><tr><td></td><td>Food (pos &gt; neg)</td><td></td><td></td><td>Service (pos &lt; neg)</td><td></td><td></td><td>Price (pos &gt; neg)</td><td></td></tr><tr><td rowspan="2">Data: Yelp Trained on  $\mathcal { D } _ { o r i }$ </td><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td></tr><tr><td>3.42</td><td>93.86</td><td>97.09</td><td>2.44</td><td>93.33</td><td>97.23</td><td>-0.32</td><td>94.04</td><td>94.44</td></tr><tr><td>Trained on  $\underline { { \mathcal { D } _ { b i a s e d } ^ { c } } }$ </td><td>3.93</td><td>93.98</td><td>97.58</td><td>-3.85</td><td>90.23</td><td>95.14</td><td>5.62</td><td>96.03</td><td>94.00</td></tr><tr><td rowspan="2">Data: CeBaB</td><td></td><td>Food (pos &gt; neg)</td><td></td><td></td><td>Service (pos &gt; neg)</td><td></td><td></td><td>Ambiance (pos &gt; neg)</td><td></td></tr><tr><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td></tr><tr><td>Trained on  $\mathcal { D } _ { o r i }$ </td><td>-0.71</td><td>69.48</td><td>74.12</td><td>-0.34</td><td>69.70</td><td>74.45</td><td>-0.90</td><td>72.62</td><td>75.68</td></tr><tr><td>Trained on  $\underline { { \mathcal { D } _ { b i a s e d } ^ { c } } }$ </td><td>14.94</td><td>61.17</td><td>58.14</td><td>13.06</td><td>67.08</td><td>61.58</td><td>7.02</td><td>69.60</td><td>65.96</td></tr></table>

Table 3: Model fine-tuning performance with training on original dataset and concept biased dataset for four datasets. Models trained on the original dataset $\mathcal { D } _ { o r i }$ tend to behave a bias in some concepts, where the label distribution under concepts is pretty uneven. When fine-tuned on the concept-biased dataset $\mathcal { D } _ { b i a s e d } ^ { c } ,$ both bias (Bias@C) and accuracy results (Acc@C and Acc@NoC) suffer from performance drop. pos > neg: for this concept, more positive texts are in $\mathcal { D } _ { o r i }$ and in $\mathcal { D } _ { b i a s e d } ,$ all texts containing this concept are positive, and vice versa for “pos $< \mathrm { n e g } ^ { , \mathrm { * } }$ . The lower absolute values of Bias@C (smaller bias) and the higher accuracy values are in bold.

Comparing the results between $\mathcal { D } _ { o r i }$ and $\mathcal { D } _ { b i a s e d } ^ { c } ,$ we find that the absolute values of Bias@C are significantly higher when trained on $\mathcal { D } _ { b i a s e d } ^ { c }$ in almost every concept, and the change direction of Bias@C is the same as the trend in the label distribution. For example, the value of Bias@C becomes negative for “service” in Yelp dataset, since we only keep negative reviews with the “service” in $\mathcal { D } _ { b i a s e d } ^ { c } .$ These observations indicate that a greater bias in the fine-tuning dataset causes the model to rely more on spurious correlations to make predictions.

Regarding utility performance (Acc@C and Acc@NoC), we observe that the models trained on $\mathcal { D } _ { b i a s e d } ^ { c }$ have a dramatic performance drop on the texts with the concept in most cases, and the average Acc@C decreases from 79.38% to 74.31%. This pattern suggests that larger spurious correlations affect the utility performance of fine-tuned models. Meanwhile, the average Acc@NoC drops from 78.08% to 76.56%. Its performance drop is not as large as the one of Acc@C, indicating that texts without the concept suffer less from the concept bias in the datasets.

Moreover, we find that for some concepts, the fine-tuned LMs suffer from severe spurious correlation, but the effect of the bias is not fully reflected in the difference between Acc@C and Acc@NoC. For example, the “music” concept in the IMDB dataset has Bias $\mathscr { @ } \mathbf { C } = 1 2 . 0 7 \%$ , but the difference between Acc@C and Acc@NoC is less than 3%. This is because if the model is biased towards one label due to the spurious correlation, the accuracy improvement towards the biased label can often offset the performance drop of the other side.

We also verify that the concept bias is not simply due to the shortcut on a few words by masking out the associated tokens, and details are shown in Section 5.2. We show fine-tuning results of LLAMA2 models on $\mathcal { D } _ { o r i }$ and $\mathcal { D } _ { b i a s e d } ^ { c }$ in Table 7 and 8 (Appendix B). Similar patterns suggest the generalizability of our findings on models of different sizes.

## 4.2 Spurious Correlations in ICL

As LMs exhibit clear evidence of utilizing the concept shortcuts in the fine-tuning data, we also want to ask whether LMs use the shortcuts in the exemplars of the prompts when performing ICL. As discussed in Section 2.4, for each concept c in five datasets, we construct a prompt with eight exemplars. Following a similar setup for fine-tuning, we only include the majority class (positive or negative) for exemplars with concept c. Specifically, for “size” in Amazon Shoe and “service” in Yelp, four exemplars with concept c have negative labels and the other four without concept c have positive labels. The labels are flipped for the rest of the concepts. For the balanced prompt $P _ { b a l a n c e d } $ , the label is evenly distributed for the exemplars with or without the concept. With the bias in label distribution for both texts with and without the concept, we expect the LM to use two types of shortcuts: a) from texts with the concept c to one sentiment and b) from texts without the concept to the other sentiment. We present the utility and bias results of ICL for sentiment classification dataset in Table 4 and BoolQ in Table 7 (Appendix B).

<table><tr><td rowspan="2">Data: Amazon Shoe</td><td colspan="3">Size (pos &lt;neg)</td><td colspan="3">Color (pos &gt;neg)</td><td colspan="3">Style (pos &gt;neg)</td></tr><tr><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td></tr><tr><td>ICL with  $P _ { b a l a n c e d }$ </td><td>0.95</td><td>50.37</td><td>45.63</td><td>9.46</td><td>49.63</td><td>49.54</td><td>11.58</td><td>52.98</td><td>53.91</td></tr><tr><td>ICL with  $P _ { b i a s e d }$ </td><td>-2.64</td><td>50.18</td><td>47.20</td><td>10.99</td><td>49.58</td><td>46.59</td><td>12.56</td><td>50.21</td><td>54.35</td></tr><tr><td rowspan="2">Data: IMDB</td><td colspan="3">Acting (pos &gt;neg)</td><td colspan="3">Comedy (pos &gt;neg)</td><td colspan="3">Music (pos &gt;neg)</td></tr><tr><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td></tr><tr><td>ICL with  $P _ { b a l a n c e d }$ </td><td>3.58</td><td>94.94</td><td>96.40</td><td>3.70</td><td>96.18</td><td>95.82</td><td>6.40</td><td>95.43</td><td>94.33</td></tr><tr><td rowspan="2">ICL with  $P _ { b i a s e d }$ </td><td>3.99</td><td>95.05</td><td>96.17</td><td>5.64</td><td>95.68</td><td>94.71</td><td>5.45</td><td>95.33</td><td>94.26</td></tr><tr><td></td><td>Food (pos &gt;neg)</td><td></td><td></td><td>Service (pos &gt;neg)</td><td></td><td></td><td>Price (pos &gt;neg)</td><td></td></tr><tr><td rowspan="2">Data: Yelp ICL with  $P _ { b a l a n c e d }$ </td><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td></tr><tr><td>2.97</td><td>97.66</td><td>98.21</td><td>0.39</td><td>97.78</td><td>98.60</td><td>-0.87</td><td>97.74</td><td>97.79</td></tr><tr><td>ICL with  $P _ { b i a s e d }$ </td><td>1.70</td><td>97.54</td><td>98.84</td><td>0.92</td><td>97.50</td><td>98.58</td><td>1.17</td><td>97.98</td><td>98.46</td></tr><tr><td rowspan="2">Data: CeBaB</td><td></td><td>Food (pos &gt;neg)</td><td></td><td></td><td>Service (pos &lt;neg)</td><td></td><td></td><td>Ambiance (pos &gt;neg)</td><td></td></tr><tr><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td></tr><tr><td>ICL with  $P _ { b a l a n c e d }$ </td><td>0.66</td><td>61.06</td><td>64.35</td><td>0.65</td><td>58.48</td><td>62.08</td><td>2.58</td><td>64.13</td><td>64.96</td></tr><tr><td>ICL with  $P _ { b i a s e d }$ </td><td>3.24</td><td>54.55</td><td>59.90</td><td>-2.77</td><td>61.91</td><td>65.53</td><td>2.21</td><td>61.39</td><td>62.45</td></tr></table>

Table 4: Model ICL performance with prompting on balanced prompts $P _ { b a l a n c e d }$ and biased prompts $P _ { b i a s e d } .$ Larger absolute values for Bias@C indicate that the concept-biased prompts enlarge the extend of models to rely on the shortcut in the demonstrations. The meaning of $\mathrm { ^ { * } p o s > n e g ^ { * } }$ and the values in bold are the same as in Table 3.

Biased prompts enlarge the concept bias in ICL inference From Table 4, we observe a similar pattern for Bias@C as in the fine-tuning part. When the prompt changes from $P _ { b a l a n c e d }$ to $P _ { b i a s e d } .$ , for “service” in Yelp and “size” in Amazon Shoe, where the exemplars with concept c are negative, the values of Bias@C flip from positive to negative, and for most other concepts, where conceptual exemplars are all positive, the value of Bias@C increases. Furthermore, in most cases, the absolute values of Bias@C when using $P _ { b i a s e d }$ are higher. These observations indicate that the LMs are affected by concept shortcuts within the prompt of ICL.

For utility performance, when changing from $P _ { b a l a n c e d }$ to $P _ { b i a s e d }$ , the average Acc@C and Acc@NoC decrease from 76.80% to 76.42% and from 76.37% to 75.58%, respectively, which means that spurious correlations harm utility performance regardless of the presence of concepts. For both Bias@C and accuracy, the relative change in the ICL scenario is less than the fine-tuning setting. We conjecture that a few exemplars in prompts make it hard to form a strong shortcut inside the LMs between conceptual contents and a specific label.

## 5 Mitigate Spurious Correlations

## 5.1 Mitigation via Rebalancing

We consider two lines of existing data-centric work to mitigate spurious correlations: remove spurious components and rebalance the training dataset (Mc-Coy et al., 2019; Wang et al., 2022). Since it is challenging to identify the conceptual contents in each sentence, we apply dataset rebalancing methods to mitigate the bias at the concept level.

We first downsample the dataset to achieve a balanced label distribution with respect to concept c, denoted as $\mathcal { D } _ { d o w n - b a l } ^ { c }$ . The shortcoming of the method is that, for a highly biased dataset, it filters out a large proportion of examples with the majority labels, leading to a sacrifice of the utility performance. To address this, we propose an upsampling method using ChatGPT to generate counterfactual examples with minority labels. Some concurrent work also demonstrates the effectiveness of synthetic data in mitigating bias (Evuru et al., 2024). The resulting dataset is denoted as $\mathcal { D } _ { u p - b a l } ^ { c } .$

Suppose that we need $\{ a _ { 0 } , \cdots , a _ { n } \}$ number of examples for labels $\{ 0 , \cdots , n \}$ to make a balanced subset for texts with concept c. We first sample $a _ { 0 }$ to $a _ { n }$ numbers of examples from texts with labels 0 to n but without concept c. Then we ask ChatGPT, $M _ { a } ,$ to inject the concept c into the selected texts while maintaining the sentiment or the answer to questions. Given the input text $x ^ { \prime }$ without concept c, the injection prompt $P _ { i }$ with the instruction, and the exemplars $h _ { c }$ with concept c, the concept injection process is $x _ { c } = M _ { a } ( P _ { i } \| h _ { c } \| x ^ { \prime } )$ , where $x _ { c }$ is the generated counterfactual for concept c and input $x ^ { \prime } .$ We iteratively generate the counterfactual input $x _ { c }$ and add it into the dataset $\mathcal { D } _ { o r i }$ to form a balanced dataset $\mathcal { D } _ { u p - b a l } ^ { c }$ with upsampling. To demonstrate the effectiveness of concept injection, we conduct a case study on a review in the Yelp dataset. As suggested in Table 5, we inject the “food” concept into a review without this concept and observe that ChatGPT effectively injects the food concept, keeps other content unchanged, and maintains the sentiment of the review.

<table><tr><td>Original</td><td>I was fairly disappointed with this zoo. Signage was unclear. Many of the exhibits were on loan</td></tr><tr><td>Counterfactual</td><td>I was fairly disappointed with this zoo. Signage was unclear. Many of the exhibits were on loan. The food options consisted of a small cafe with limited choices.</td></tr></table>

Table 5: An example of the generated counterfactual data for concept “food” in the Yelp dataset. Text in bold is the generated input with the injected “food” concept.

We also propose a baseline method that masks out words highly associated with the concept. This method is used to verify whether balancing distributions of a few tokens removes conceptual shortcuts. We replace words with the top 10 PMI for each concept (word examples are in Table 2) to the [MASK] token and name the masked dataset as $\mathcal { D } _ { m a s k } ^ { c } .$

## 5.2 Results of Mitigation Methods

To evaluate the effectiveness of proposed methods, we select concepts with Bias@C greater than 1 in Table 3 and fine-tune on three de-biased datasets $\mathcal { D } _ { d o w n - b a l } ^ { c } , \mathcal { D } _ { u p - b a l } ^ { c } .$ , and $\mathcal { D } _ { m a s k } ^ { c }$ . We report results for DistilBERT in Table 6 and 7 in Appendix B.

Upsampling method reduces the bias and increases utility performance From Table 6, we observe that data rebalancing methods are effective in mitigating spurious correlations. For downsampling $( \mathcal { D } _ { d o w n - b a l } ^ { c } )$ , it mitigates the mean absolute values of Bias@C from 4.90% to 3.43%, compared to trained on $\mathcal { D } _ { o r i }$ . However, for utility performance, the downsampling obtains less accuracy in 4/8 cases for Acc@C and 5/8 cases for Acc@NoC, indicating that loss of data harms utility. For the upsampling method $( \mathcal { D } _ { u p - b a l } ^ { c } )$ , the mean absolute values of Bias@C are effectively reduced from 4.90% to 2.74%. Furthermore, the average accuracy of Acc@C increases from 79. 24% to 80. 38%, and Acc@NoC is comparable. This observation suggests that adding counterfactual texts to rebalance the data can reduce spurious correlations in the concept level, and more data involved in the fine-tuning can boost the models’ utility performance.

Masking out associated tokens $( \mathcal { D } _ { m a s k } ^ { c } )$ can reduce spurious correlations in most cases, but cannot fully eliminate bias. This observation suggests that due to the various concept expressions, the learned concept shortcut in the model is not equivalent to the shortcut on a few tokens. The utility performance of Acc@C is also lower than that of the proposed upsampling method in 6/8 comparisons.

In summary, among the three mitigation methods, adding the LLM-generated counterfactual inputs achieves the best performance in both the bias mitigation and utility aspects. The same analysis on LLAMA2 models (Table 7 and 9 in Appendix B) reveals similar patterns, which shows the generalizability of our methods.

## 6 Related Work

Robustness and Bias Current work on studying spurious correlations for LMs can be split into two categories: utilize the shortcuts during training or ICL. For shortcut learning in training, a series of works explores how models take shortcuts in the data for the causal or non-causal perspective (Tu et al., 2020; Sagawa et al., 2020; Geirhos et al., 2020; Ribeiro et al., 2020; Kaushik et al., 2019; Liu et al., 2024; Friedman et al., 2022) and which aspects of shortcuts will be taken for the predictions in different NLP tasks (McCoy et al., 2019; Jia and Liang, 2017; Lai et al., 2021; Zhao et al., 2018; Niu et al., 2020; Poliak et al., 2018), leading to low generalization in the out-of-distribution data or in the designed adversarial data.

Due to the increasing development of LLM on ICL, researchers find that the design of the prompt significantly decides the LLM predictions (Brown et al., 2020; Gao et al., 2020; Liu et al., 2023b; Zhou et al., 2023; Schick and Schütze, 2020). Another line of work finds that LLMs are sensitive to a certain aspect of prompts and not robust when injecting adversarial triggers into prompt (Lu et al., 2021; Zhao et al., 2021; Tang et al., 2023; Si et al., 2023; Zheng et al., 2023). Tang et al. (2023) shows that LLMs use multiple types of shortcuts in the prompts, from letters to words to text style, and Si et al. (2023) find that LLMs exhibit clear feature biases under the unspecified prompts. Previous work also develops multiple methods to identify the topic or concept of text input (Li et al., 2024a; Abraham et al., 2022; Blei et al., 2003). However, our paper is the first to focus on assessing whether the models use shortcuts at the general concept level.

<table><tr><td rowspan="2">Data</td><td colspan="3">Amazon Shoe: Size</td><td colspan="3">Amazon Shoe: Color</td><td colspan="3">Amazon Shoe: Style</td><td colspan="3">IMDB: Acting</td></tr><tr><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td></tr><tr><td> $\overline { { \mathcal { D } _ { o r i } } }$ </td><td>2.11</td><td>57.94</td><td>55.94</td><td>1.38</td><td>57.19</td><td>55.50</td><td>11.56</td><td>57.18</td><td>56.12</td><td>3.70</td><td>88.49</td><td>91.24</td></tr><tr><td>Dc  $P _ { \it d o w n - b a l } ^ { \mathrm { * } }$ </td><td>2.25</td><td>57.19</td><td>54.29</td><td>2.54</td><td>56.71</td><td>55.70</td><td>9.79</td><td>57.45</td><td>56.02</td><td>2.04</td><td>91.03</td><td>91.92</td></tr><tr><td> $\mathcal { D } _ { u p - b a l } ^ { c }$ </td><td>1.20</td><td>56.72</td><td>55.64</td><td>2.06</td><td>57.44</td><td>57.01</td><td>9.16</td><td>56.41</td><td>58.84</td><td>1.87</td><td>91.63</td><td>92.10</td></tr><tr><td> $\underline { { \mathcal { D } _ { m a s k } ^ { c } } }$ </td><td>1.88</td><td>57.31</td><td>56.75</td><td>5.94</td><td>57.40</td><td>56.21</td><td>10.29</td><td>56.80</td><td>55.12</td><td>4.75</td><td>90.67</td><td>91.95</td></tr><tr><td></td><td colspan="3">IMDB: Comedy</td><td colspan="3">IMDB: Music</td><td colspan="3">Yelp: Food</td><td colspan="3">Yelp: Service</td></tr><tr><td> $\mathcal { D } _ { o r i }$ </td><td>2.51</td><td>91.62</td><td>91.85</td><td>12.07</td><td>91.55</td><td>88.93</td><td>3.42</td><td>93.86</td><td>97.09</td><td>2.44</td><td>93.33</td><td>97.23</td></tr><tr><td>Ddown-bal</td><td>-0.37</td><td>90.86</td><td>92.88</td><td>7.71</td><td>91.03</td><td>92.29</td><td>-0.88</td><td>93.63</td><td>95.35</td><td>1.57</td><td>93.78</td><td>97.02</td></tr><tr><td> $\mathcal { D } _ { u p - b a l } ^ { c }$ </td><td>-0.32</td><td>91.65</td><td>92.74</td><td>4.05</td><td>90.19</td><td>92.80</td><td>2.86</td><td>94.15</td><td>96.85</td><td>0.39</td><td>93.84</td><td>97.04</td></tr><tr><td> $\underline { { \mathcal { D } _ { m a s k } ^ { c } } }$ </td><td>1.08</td><td>90.41</td><td>92.56</td><td>8.28</td><td>91.88</td><td>90.52</td><td>2.02</td><td>93.52</td><td>96.45</td><td>1.02</td><td>94.67</td><td>97.41</td></tr></table>

Table 6: Performance of multiple shortcut mitigation methods (downsampling, upsampling and token removal). Upsampling method with the counterfactual generated data can obtain the best average effects in the aspects of reducing bias and increasing the utility performance. $\mathcal { D } _ { o r i }$ represents fine-tuning on the $\mathcal { D } _ { o r i }$ dataset.

Spurious Correlation Mitigation An increasing number of methods have attempted to mitigate spurious correlations in models caused by bias in the dataset (Chew et al., 2023; Clark et al., 2020; Le Bras et al., 2020; Zhou and Bansal, 2020; Liu et al., 2021, 2023c; Zhu et al., 2023), by data augmentation (Jin et al., 2020; Alzantot et al., 2018; Wang et al., 2022; Minderer et al., 2020), data rebalancing (McCoy et al., 2019; Sharma et al., 2018; Zellers et al., 2019), multi-task learning (Tu et al., 2020), and model ensembling or adding regularization (Utama et al., 2020; He et al., 2019; Zhao et al., 2022; Liu et al., 2023d). To mitigate spurious correlations in a concept, we propose another data rebalancing method, which uses LLM to generate counterfactual sentences by injecting the concept and saves the human resource to compose them.

## 7 Conclusions

In this paper, we explore the spurious correlation at the general concept level in both fine-tuning and ICL settings. We find that LMs utilize the concept shortcut in training data (or in demonstrations) when inference on unseen data, and more biased training data (or prompts) lead to more biased predictions. To mitigate the learned shortcut, we propose a rebalancing method by adding counterfactual examples generated from ChatGPT to the training data, shown to be effective through extensive experiments. Our work indicates that LMs form strong spurious correlations on general concepts, encouraging future work to pay attention to unintended shortcut learning.

## 8 Limitations

Due to the limitation of the budget and the computation resource, we only fine-tuned the LLaMa2 7B model with the Lora method and used GPT3.5 for concept annotation. It could be interesting to fully fine-tune the LMs with a larger size. Moreover, in Section 3, we find that the ChatGPT annotation of the concept label still achieves slightly lower accuracy than human annotators. We can use a more advanced model, such as GPT4 (OpenAI, 2023), for annotation to increase the performance.

Our work focuses on five classification tasks. Three of them are binary classification tasks, and two are multiclass classification tasks. We apply the difference in accuracy for different groups (positive and negative) to measure bias at the concept level. Moreover, future work could extend our framework and generalize the measurement of concept bias to more complex tasks, such as the evaluation of LLM on QA tasks (Li et al., 2024b) or even on tasks with the vision language model (Wang et al., 2024b; Liu et al., 2023a; Wang et al., 2024a; Yue et al., 2023).

For in-context learning, we observe that the concept bias in the demonstrations leads to larger spurious correlations. However, we also detect that the balanced prompts cannot fully eliminate the bias, and we do not provide a method to mitigate this inner spurious correlation in LMs. We leave that direction to future work.

## Acknowledgments

Zhou and Huang are supported by DARPA Transfer from Imprecise and Abstract Models to Autonomous Technologies (TIAMAT) 80321, National Science Foundation NSF-IIS-2147276 FAI, DOD-ONR-Office of Naval Research under award number N00014-22-1-2335, DOD-AFOSR-Air Force Office of Scientific Research under award number FA9550-23-1-0048, DOD-DARPA-Defense Advanced Research Projects Agency Guaranteeing AI Robustness against Deception (GARD) HR00112020007, Adobe, Capital One and JP Morgan faculty fellowships.

## References

Eldar D Abraham, Karel D’Oosterlinck, Amir Feder, Yair Gat, Atticus Geiger, Christopher Potts, Roi Reichart, and Zhengxuan Wu. 2022. Cebab: Estimating the causal effects of real-world concepts on nlp model behavior. Advances in Neural Information Processing Systems, 35:17582–17596.

Moustafa Alzantot, Yash Sharma, Ahmed Elgohary, Bo-Jhang Ho, Mani Srivastava, and Kai-Wei Chang. 2018. Generating natural language adversarial examples. arXiv preprint arXiv:1804.07998.

Ziv Bar-Joseph, David K Gifford, and Tommi S Jaakkola. 2001. Fast optimal leaf ordering for hierarchical clustering. Bioinformatics, 17(suppl\_1):S22– S29.

David M Blei, Andrew Y Ng, and Michael I Jordan. 2003. Latent dirichlet allocation. Journal ofmachine Learning research, 3(Jan):993–1022.

Angana Borah, Daria Pylypenko, Cristina Espana-Bonet, and Josef van Genabith. 2023. Measuring spurious correlation in classification:’clever hans’ in translationese. arXiv preprint arXiv:2308.13170.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Oscar Chew, Kuan-Hao Huang, Kai-Wei Chang, and Hsuan-Tien Lin. 2023. Understanding and mitigating spurious correlations in text classification. arXiv preprint arXiv:2305.13654.

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. 2019. Boolq: Exploring the surprising difficulty of natural yes/no questions. arXiv preprint arXiv:1905.10044.

Christopher Clark, Mark Yatskar, and Luke Zettlemoyer. 2020. Learning to model and ignore dataset bias with mixed capacity ensembles. arXiv preprint arXiv:2011.03856.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

Chandra Kiran Reddy Evuru, Sreyan Ghosh, Sonal Kumar, Ramaneswaran S, Utkarsh Tyagi, and Dinesh Manocha. 2024. Coda: Constrained generation based data augmentation for low-resource nlp.

Yanbo Fang and Yongfeng Zhang. 2022. Data-efficient concept extraction from pre-trained language models for commonsense explanation generation. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 5883–5893.

Dan Friedman, Alexander Wettig, and Danqi Chen. 2022. Finding dataset shortcuts with grammar induction. arXiv preprint arXiv:2210.11560.

Tianyu Gao, Adam Fisch, and Danqi Chen. 2020. Making pre-trained language models better few-shot learners. arXiv preprint arXiv:2012.15723.

Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A Wichmann. 2020. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673.

Fabrizio Gilardi, Meysam Alizadeh, and Maël Kubli. 2023. Chatgpt outperforms crowd-workers for textannotation tasks. arXiv preprint arXiv:2303.15056.

He He, Sheng Zha, and Haohan Wang. 2019. Unlearn dataset bias in natural language inference by fitting the residual. arXiv preprint arXiv:1908.10763.

Ruining He and Julian McAuley. 2016. Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering. In proceedings of the 25th international conference on world wide web, pages 507–517.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685.

Robin Jia and Percy Liang. 2017. Adversarial examples for evaluating reading comprehension systems. arXiv preprint arXiv:1707.07328.

Di Jin, Zhijing Jin, Joey Tianyi Zhou, and Peter Szolovits. 2020. Is bert really robust? a strong baseline for natural language attack on text classification and entailment. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 8018–8025.

Divyansh Kaushik, Eduard Hovy, and Zachary C Lipton. 2019. Learning the difference that makes a difference with counterfactually-augmented data. arXiv preprint arXiv:1909.12434.

Jon Kleinberg, Jens Ludwig, Sendhil Mullainathan, and Ashesh Rambachan. 2018. Algorithmic fairness. In Aea papers and proceedings, volume 108, pages 22– 27. American Economic Association 2014 Broadway, Suite 305, Nashville, TN 37203.

Yuxuan Lai, Chen Zhang, Yansong Feng, Quzhe Huang, and Dongyan Zhao. 2021. Why machine reading comprehension models learn shortcuts? In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 989–1002, Online. Association for Computational Linguistics.

Ronan Le Bras, Swabha Swayamdipta, Chandra Bhagavatula, Rowan Zellers, Matthew Peters, Ashish

Sabharwal, and Yejin Choi. 2020. Adversarial filters of dataset biases. In International conference on machine learning, pages 1078–1088. PMLR.

Zongxia Li, Andrew Mao, Daniel Stephens, Pranav Goel, Emily Walpole, Alden Dima, Juan Fung, and Jordan Boyd-Graber. 2024a. Improving the tenor of labeling: Re-evaluating topic models for content analysis.

Zongxia Li, Ishani Mondal, Yijun Liang, Huy Nghiem, and Jordan Lee Boyd-Graber. 2024b. Panda (pedantic answer-correctness determination and adjudication):improving automatic evaluation for question answering and text generation.

Evan Z Liu, Behzad Haghgoo, Annie S Chen, Aditi Raghunathan, Pang Wei Koh, Shiori Sagawa, Percy Liang, and Chelsea Finn. 2021. Just train twice: Improving group robustness without training group information. In International Conference on Machine Learning, pages 6781–6792. PMLR.

Fuxiao Liu, Tianrui Guan, Zongxia Li, Lichang Chen, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. 2023a. Hallusionbench: You see what you think? or you think what you see? an image-context reasoning benchmark challenging for gpt-4v (ision), llava-1.5, and other multi-modality models. arXiv preprint arXiv:2310.14566.

Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. 2023b. Pretrain, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM Computing Surveys, 55(9):1–35.

Xiaoyu Liu, Hanlin Lu, Jianbo Yuan, and Xinyu Li. 2023c. Cat: Causal audio transformer for audio classification. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE.

Xiaoyu Liu, Paiheng Xu, Junda Wu, Jiaxin Yuan, Yifan Yang, Yuhang Zhou, Fuxiao Liu, Tianrui Guan, Haoliang Wang, Tong Yu, et al. 2024. Large language models and causal inference in collaboration: A comprehensive survey. arXiv preprint arXiv:2403.09606.

Xiaoyu Liu, Jiaxin Yuan, Bang An, Yuancheng Xu, Yifan Yang, and Furong Huang. 2023d. Cdisentanglement: Discovering causally-independent generative factors under an inductive bias of confounder. arXiv preprint arXiv:2310.17325.

Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. 2021. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. arXiv preprint arXiv:2104.08786.

Andrew Maas, Raymond E Daly, Peter T Pham, Dan Huang, Andrew Y Ng, and Christopher Potts. 2011. Learning word vectors for sentiment analysis. In Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies, pages 142–150.

R Thomas McCoy, Ellie Pavlick, and Tal Linzen. 2019. Right for the wrong reasons: Diagnosing syntactic heuristics in natural language inference. arXiv preprint arXiv:1902.01007.

Matthias Minderer, Olivier Bachem, Neil Houlsby, and Michael Tschannen. 2020. Automatic shortcut removal for self-supervised representation learning. In International Conference on Machine Learning, pages 6927–6937. PMLR.

Xing Niu, Prashant Mathur, Georgiana Dinu, and Yaser Al-Onaizan. 2020. Evaluating robustness to input perturbations for neural machine translation. arXiv preprint arXiv:2005.00580.

OpenAI. 2023. Gpt-4 technical report.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744.

Adam Poliak, Jason Naradowsky, Aparajita Haldar, Rachel Rudinger, and Benjamin Van Durme. 2018. Hypothesis only baselines in natural language inference. In Proceedings of the Seventh Joint Conference on Lexical and Computational Semantics, pages 180– 191.

Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. 2020. Beyond accuracy: Behavioral testing of nlp models with checklist. arXiv preprint arXiv:2005.04118.

Shiori Sagawa, Aditi Raghunathan, Pang Wei Koh, and Percy Liang. 2020. An investigation of why overparameterization exacerbates spurious correlations. In International Conference on Machine Learning, pages 8346–8356. PMLR.

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.

Timo Schick and Hinrich Schütze. 2020. Exploiting cloze questions for few shot text classification and natural language inference. arXiv preprint arXiv:2001.07676.

Rishi Sharma, James Allen, Omid Bakhshandeh, and Nasrin Mostafazadeh. 2018. Tackling the story ending biases in the story cloze test. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 752–757.

Chenglei Si, Dan Friedman, Nitish Joshi, Shi Feng, Danqi Chen, and He He. 2023. Measuring inductive biases of in-context learning with underspecified demonstrations. arXiv preprint arXiv:2305.13299.

Ruixiang Tang, Dehan Kong, Longtao Huang, and Hui Xue. 2023. Large language models can be lazy learners: Analyze shortcuts in in-context learning. arXiv preprint arXiv:2305.17256.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Lifu Tu, Garima Lalwani, Spandana Gella, and He He. 2020. An empirical study on robustness to spurious correlations using pre-trained language models. Transactions of the Association for Computational Linguistics, 8:621–633.

Prasetya Ajie Utama, Nafise Sadat Moosavi, and Iryna Gurevych. 2020. Mind the trade-off: Debiasing nlu models without degrading the in-distribution performance. arXiv preprint arXiv:2005.00315.

Tianlu Wang, Rohit Sridhar, Diyi Yang, and Xuezhi Wang. 2022. Identifying and mitigating spurious correlations for improving robustness in NLP models. In Findings of the Association for Computational Linguistics: NAACL 2022, pages 1719–1729, Seattle, United States. Association for Computational Linguistics.

Xiyao Wang, Jiuhai Chen, Zhaoyang Wang, Yuhang Zhou, Yiyang Zhou, Huaxiu Yao, Tianyi Zhou, Tom Goldstein, Parminder Bhatia, Furong Huang, and Cao Xiao. 2024a. Enhancing visual-language modality alignment in large vision language models via selfimprovement.

Xiyao Wang, Yuhang Zhou, Xiaoyu Liu, Hongjin Lu, Yuancheng Xu, Feihong He, Jaehong Yoon, Taixi Lu, Gedas Bertasius, Mohit Bansal, et al. 2024b. Mementos: A comprehensive benchmark for multimodal large language model reasoning over image sequences. arXiv preprint arXiv:2401.10529.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837.

Robert Wolfe and Aylin Caliskan. 2021. Low frequency names exhibit bias and overfitting in contextualizing language models. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 518–532, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. 2023. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. arXiv preprint arXiv:2311.16502.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830.

Michael Zhang, Nimit S Sohoni, Hongyang R Zhang, Chelsea Finn, and Christopher Ré. 2022. Correctn-contrast: A contrastive approach for improving robustness to spurious correlations. arXiv preprint arXiv:2203.01517.

Xiang Zhang, Junbo Zhao, and Yann LeCun. 2015. Character-level convolutional networks for text classification. Advances in neural information processing systems, 28.

Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, and Kai-Wei Chang. 2018. Gender bias in coreference resolution: Evaluation and debiasing methods. arXiv preprint arXiv:1804.06876.

Jieyu Zhao, Xuezhi Wang, Yao Qin, Jilin Chen, and Kai-Wei Chang. 2022. Investigating ensemble methods for model robustness improvement of text classifiers. arXiv preprint arXiv:2210.16298.

Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021. Calibrate before use: Improving few-shot performance of language models. In International Conference on Machine Learning, pages 12697–12706. PMLR.

Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. 2023. On large language models’ selection bias in multi-choice questions. arXiv preprint arXiv:2309.03882.

Xiang Zhou and Mohit Bansal. 2020. Towards robustifying NLI models against lexical dataset biases. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 8759– 8771, Online. Association for Computational Linguistics.

Yuhang Zhou, Suraj Maharjan, and Beiye Liu. 2023. Scalable prompt generation for semi-supervised learning with language models. In Findings of the Association for Computational Linguistics: EACL 2023, pages 758–769.

Jing Zhu, Yuhang Zhou, Vassilis N Ioannidis, Shengyi Qian, Wei Ai, Xiang Song, and Danai Koutra. 2023. Spottarget: Rethinking the effect of target edges for link prediction in graph neural networks. arXiv preprint arXiv:2306.00899.

## A Implementation Details

## A.1 Fine-tuning Experiments

We use the DistilBERT and LLAMA2 model (Sanh et al., 2019; Touvron et al., 2023) as our LMs for all of our fine-tuning experiments. For the DistilBERT model, we use AdamW as our optimizer with a learning rate of 2e 5 and a weight decay of 0.01 with linear scheduler, batch size of 16, and trained for 3 epochs. For the LLAMA2 model, we use AdamW as our optimizer with a learning rate of 2e 4, batch size of 32, warm-up ratio of 0.03, and trained for 3 epochs. We base our implementation on the Pytorch<sup>2</sup>, Huggingface transformer<sup>3</sup> frameworks, and the LLAMA2 weights from Meta<sup>4</sup>.

## A.2 ICL Setup

We utilize greedy search in decoding for all ICL experiments and counterfactual data generation, except for the annotation of concepts for each text, where we use stochastic temperature sampling with the temperature value 0.7 to obtain diverse answers. The template of the prompts for the ICL experiments, concept annotations and counterfactual data generations are suggested in Table 10, Table 11 and Table 12.

We call the gpt-3.5-turbo (4k) function from OpenAI to generate the concept labels, ICL experiments and concept injection. The price of this API is \$0.0015 / 1K tokens for inputs and \$0.002 / 1K tokens for output. The total expenditure on API usage is about \$ 300.00, including preliminary exploration.

## B Prompt Details and Supplementary Results

In Table 7, we perform the same analysis on the BoolQ question and answering dataset for all experiments (ICL and fine-tuning) in Section 4. In Table 8 and Table 9, we repeat the experiments for fine-tuning LLAMA2 7B models for Section 4.1 and 5.2. In Table 10, 11, and 12, we present the details of the prompts in the annotation of the concept, the ICL experiments, and the countergactual sentence generation.

<table><tr><td>Distilbert</td><td>BoolQ Country (pos &lt;neg) Bias@C</td><td>Acc@NoC</td><td>Acc@C</td><td>Bias@C Acc@NoC</td><td>BoolQ History (pos &gt;neg)</td><td>Acc@C</td></tr><tr><td>Trained on Dori</td><td>-1.22</td><td>61.23</td><td>60.95</td><td>5.21</td><td>59.93</td><td>57.85</td></tr><tr><td>Trained on Dbiased</td><td>-18.90</td><td>55.92</td><td>55.46</td><td>50.63</td><td>57.37</td><td>55.22</td></tr><tr><td>Trained on Ddown-bal</td><td>4.84</td><td>57.93</td><td>58.79</td><td>-8.95</td><td>59.87</td><td>58.60</td></tr><tr><td>Trained on Dtp-bal</td><td>2.45</td><td>59.54</td><td>59.74</td><td>-0.85</td><td>60.13</td><td>59.70</td></tr><tr><td>Trained on Dmask</td><td>2.90</td><td>60.71</td><td>59.69</td><td>-1.55</td><td>60.94</td><td>60.84</td></tr><tr><td>LLAMA2</td><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td></tr><tr><td>Trained on Dori</td><td>-9.72</td><td>67.78</td><td>72.68</td><td>0.82</td><td>78.87</td><td>80.43</td></tr><tr><td>Trained on Dbiased</td><td>-9.67</td><td>67.23</td><td>70.86</td><td>2.73</td><td>75.86</td><td>78.79</td></tr><tr><td>Trained on Ddown-bal</td><td>-10.18</td><td>77.11</td><td>81.61</td><td>1.20</td><td>76.11</td><td>77.64</td></tr><tr><td>Trained on Dc up−bal</td><td>-7.81</td><td>77.30</td><td>78.16</td><td>-3.23</td><td>80.62</td><td>82.40</td></tr><tr><td>Trained on Dc mask</td><td>-10.55</td><td>77.53</td><td>79.10</td><td>-7.73</td><td>78.91</td><td>80.73</td></tr><tr><td>GPT3.5 ICL</td><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td></tr><tr><td>ICL with Pbalanced</td><td>-1.80</td><td>81.98</td><td>83.99</td><td>-3.88</td><td>83.12</td><td>83.90</td></tr><tr><td>ICL with Pbiased</td><td>-2.90</td><td>82.93</td><td>83.11</td><td>-3.93</td><td>82.03</td><td>82.82</td></tr></table>

Table 7: Fine-tuning and ICL performance for all experiments in Section 4 on BoolQ dataset of DistilBert, LLAMA2 (fine-tuning) and GPT3.5 (ICL) models. The smaller absolute values of Bias@C (smaller bias) and larger values of Acc are in bold.

<table><tr><td rowspan="2">Method</td><td colspan="2">AS Size (pos &lt;neg)</td><td colspan="2">AS Color (pos &gt;neg)</td><td colspan="3">AS Style (pos &gt;neg)</td></tr><tr><td>Bias@C</td><td>Acc@NoC Acc@C</td><td>Bias@C Acc@NoC</td><td>Acc@C</td><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td></tr><tr><td rowspan="2">Trained on  $\mathcal { D } _ { o r i }$  Trained on  $\mathcal { D } _ { b i a s e d } ^ { c }$ </td><td>-1.46</td><td>59.46 57.24</td><td>3.87</td><td>59.62 59.07</td><td>16.01</td><td>59.54</td><td>58.89</td></tr><tr><td>-7.69</td><td>59.57 51.86</td><td>7.93</td><td>58.54 59.43</td><td>17.31</td><td>59.91</td><td>58.14</td></tr><tr><td rowspan="2">Method</td><td></td><td>IMDB Acting (pos &gt;neg)</td><td></td><td>IMDB Comedy (pos &gt;neg)</td><td></td><td>IMDB Music (pos &gt;neg)</td><td></td></tr><tr><td>Bias@C</td><td>Acc@NoC Acc@C</td><td>Bias@C</td><td>Acc@NoC Acc@C</td><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td></tr><tr><td>Trained on  $\mathcal { D } _ { o r i }$  Trained on  $\mathcal { D } _ { b i a s e d } ^ { c }$ </td><td>3.51 4.61</td><td>95.87 97.55 95.73 97.54</td><td>1.23 0.40</td><td>96.30 97.55 96.79 97.19</td><td>5.35 6.65</td><td>96.84 96.88</td><td>95.69</td></tr><tr><td rowspan="2">Method</td><td></td><td>Yelp Food (pos &gt;neg)</td><td></td><td>Yelp Service (pos &lt;neg)</td><td></td><td>Yelp Price (pos &gt;neg)</td><td>96.24</td></tr><tr><td>Bias@C</td><td>Acc@NoC Acc@C</td><td>Bias@C</td><td>Acc@NoC Acc@C</td><td>Bias@C</td><td>Acc@NoC</td><td></td></tr><tr><td rowspan="2">Trained on  $\overline { { \mathcal { D } _ { o r i } } }$  Trained on  $\mathcal { D } _ { b i a s e d } ^ { c }$ </td><td>2.62</td><td>98.30 98.80</td><td>1.41</td><td>97.79</td><td>99.24 0.32</td><td>98.59</td><td>Acc@C</td></tr><tr><td>2.64</td><td>98.23 98.80</td><td>1.11</td><td>97.78 99.19</td><td>0.78</td><td>98.37</td><td>98.52</td></tr><tr><td rowspan="2">Method</td><td></td><td>CeBaB Food (pos &gt;neg)</td><td></td><td>CeBaB Service (pos &gt;neg)</td><td></td><td>CeBaB Ambiance (pos &gt;neg)</td><td>98.37</td></tr><tr><td>Bias@C</td><td></td><td>Bias@C</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Trained on  $\mathcal { D } _ { o r i }$  Trained on  $\mathcal { D } _ { b i a s e d } ^ { c }$ </td><td>3.04</td><td>Acc@NoC Acc@C</td><td></td><td>Acc@NoC</td><td>Acc@C Bias@C</td><td>Acc@NoC</td><td>Acc@C</td></tr><tr><td>3.67</td><td>74.01 75.57 61.96 65.09</td><td>-3.44 3.12</td><td>69.58 65.05</td><td>75.21 0.41 67.22 1.35</td><td>73.05 68.97</td><td>75.60</td></tr></table>

Table 8: Model fine-tuning performance with training on original dataset and concept biased dataset for LLAMA2 fine-tuning. pos > neg: The number of positive texts is larger than the number of negative texts in the original data and in biased dataset, all texts containing this concept are positive, and vice versa for $\mathrm { ^ { * } p o s < n e g ^ { * } }$ . The smaller absolute values of Bias@C (smaller bias) and larger values of Acc are in bold.

<table><tr><td rowspan="2">Method</td><td colspan="3">Amazon Shoe: Size</td><td colspan="3">Amazon Shoe: Color</td><td colspan="3">Amazon Shoe: Style</td><td colspan="3">IMDB: Acting</td><td colspan="3">CeBaB: Food</td></tr><tr><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td><td>Bias@C</td><td>Acc@NoC</td><td>Acc@C</td></tr><tr><td> $\overline { { \mathcal { D } _ { o r i } } }$ </td><td>-1.46</td><td>59.46</td><td>57.24</td><td>3.87</td><td>59.62</td><td>59.07</td><td>16.01</td><td>59.54</td><td>58.89</td><td>3.51</td><td>95.87</td><td>97.55</td><td>3.04</td><td>74.01</td><td>75.57</td></tr><tr><td></td><td>2.88</td><td>59.17</td><td>57.93</td><td>4.10</td><td>59.57</td><td>58.48</td><td>11.62</td><td>58.13</td><td>61.83</td><td>2.28</td><td>96.14</td><td>97.59</td><td>2.16</td><td>68.86</td><td>71.09</td></tr><tr><td>Ddown-bal</td><td>-0.62</td><td>59.94</td><td>56.53</td><td>-2.01</td><td>59.92</td><td>60.90</td><td>11.37</td><td>59.32</td><td>60.43</td><td>1.91</td><td>96.12</td><td>97.72</td><td>1.83</td><td>70.71</td><td>74.75</td></tr><tr><td>Dup-bal</td><td>-2.41</td><td>59.65</td><td>53.65</td><td>4.12</td><td>58.23</td><td>58.33</td><td>13.33</td><td>59.12</td><td>59.48</td><td>2.50</td><td>96.02</td><td>97.73</td><td>0.02</td><td>73.45</td><td>73.42</td></tr><tr><td>Dmask</td><td></td><td>IMDB: Comedy</td><td></td><td></td><td>IMDB: Music</td><td></td><td></td><td>Yelp: Food</td><td></td><td></td><td>Yelp: Service</td><td></td><td></td><td>CeBaB: Service</td><td></td></tr><tr><td>Dori</td><td>1.23</td><td>96.30</td><td>97.55</td><td>5.35</td><td>96.84</td><td>95.69</td><td>2.62</td><td>98.30</td><td>98.80</td><td>1.41</td><td>97.79</td><td>99.24</td><td>-3.44</td><td>69.58</td><td>75.21</td></tr><tr><td></td><td>0.89</td><td>96.32</td><td>97.32</td><td>8.56</td><td>97.27</td><td>95.12</td><td>4.39</td><td>98.18</td><td>98.92</td><td>-0.35</td><td>98.00</td><td>99.20</td><td>-1.86</td><td>70.44</td><td>74.84</td></tr><tr><td>Ddown-bal Dup-bal</td><td>0.77</td><td>97.59</td><td>97.74</td><td>8.01</td><td>96.78</td><td>94.57</td><td>1.83</td><td>97.91</td><td>99.04</td><td>0.41</td><td>97.62</td><td>99.30</td><td>0.21</td><td>70.25</td><td>75.88</td></tr><tr><td>Dmask</td><td>0.53</td><td>96.89</td><td>97.18</td><td>2.68</td><td>96.98</td><td>96.71</td><td>3.18</td><td>98.36</td><td>98.61</td><td>0.08</td><td>97.96</td><td>98.82</td><td>-0.90</td><td>69.83</td><td>75.05</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 9: Performance of multiple shortcut mitigation methods (downsampling, upsampling and token removal) for LLAMA2 fine-tuning. Upsampling method with the counterfactual generated data can obtain the best average effects in the aspects of reducing bias and increasing the utility performance.

I will provide you 5 reviews in {dataset name} dataset. Please find the concepts explicitly mentioned   
in this review only from the set with three concepts: {candidate concepts}. Do not include other   
concepts. If you can not find any of these concepts in the concept set, please annotate this review with   
“none”. Wrap your answer for a review in a word sequence separated by the comma and for each   
answer, start with a new line with an index.   
Here are a few examples:   
{demonstrations}   
The output is:   
{output concepts}   
Here is the review list of 5 OpenTable reviews:   
{text lists}   
The output is:  
Table 10: Prompt P<sub>a</sub> for concept annotation in all datasets. {dataset name} and {candidate concepts} are placeholders to put the name of dataset and the candidate concepts. For example, for Amazon shoe dataset, they are “Amazon shoe” and “size, color, and style”. {demonstrations} and {output concepts} are placeholders to add five demonstrations with provided ground-truth concept labels. {text lists} is a placeholder to add the text to be annotated.

<table><tr><td>Given a review, you need to predict whether the sentiment of the review is positive or negative. Here are the examples:</td></tr><tr><td></td></tr><tr><td>Review: {review 1} Sentiment label: {label 1}</td></tr><tr><td>Review: {review 2} Sentiment label: {label 2}</td></tr><tr><td>Review: {review 3} Sentiment label: {label 3}</td></tr><tr><td>Review: {review 4} Sentiment label: {label 4}</td></tr><tr><td>Review: {review 5} Sentiment label: {label 5}</td></tr><tr><td>Review: {review 6} Sentiment label: {label 6}</td></tr><tr><td>Review: {review 7} Sentiment label: {label 7}</td></tr><tr><td>Review: {review 8} Sentiment label: {label 8}</td></tr><tr><td>Here is the review to predict sentiment: Review:  $\{ x _ { t e s t } \}$  Sentiment label:</td></tr></table>

<table><tr><td>(a) Prompt  $P _ { b a l a n c e d }$  or  $P _ { b i a s e d }$  for IMDB and Yelp dataset.</td></tr><tr><td>Given a review, you need to predict whether the sentiment label of the review from 0 to 4, total 5</td></tr><tr><td>classes. Label 0 represents the most negative review and Label 4 represents the most positive review.</td></tr><tr><td>Here are the examples:</td></tr><tr><td>Review: {review 1} Sentiment label: {label 1}</td></tr><tr><td>Review: {review 2} Sentiment label: {label 2}</td></tr><tr><td>Review: {review 3} Sentiment label: {label 3}</td></tr><tr><td>Review: {review 4} Sentiment label: {label 4}</td></tr><tr><td>Review: {review 5} Sentiment label: {label 5}</td></tr><tr><td>Review: {review 6} Sentiment label: {label 6}</td></tr><tr><td>Review: {review 7} Sentiment label: {label 7}</td></tr><tr><td>Review: {review 8} Sentiment label: {label 8}</td></tr><tr><td>Here is the review to predict sentiment:</td></tr><tr><td>Review:  $\{ x _ { t e s t } \}$  Sentiment label:</td></tr></table>

Table 11: Prompt $P _ { b a l a n c e d }$ or $P _ { b i a s e d }$ for the ICL experiments for all datasets. {review} and {label} is a placeholder to add 8 demonstrations with provided ground-truth sentiment labels for each dataset. $\{ x _ { t e s t } \}$ is the place to insert the predicted text.

<table><tr><td>Here are 5 exemplars with the {concept} concept: {texts with concept}</td></tr><tr><td>Here are another 5 exemplars without the {concept} concept:</td></tr><tr><td>{texts without concept}</td></tr><tr><td>Please inject the “concept&quot; concept into a statement and maintain the sentiment level of this statement.</td></tr><tr><td>The statement is:</td></tr><tr><td></td></tr><tr><td>{text for counterfactual}</td></tr><tr><td>The output statement with the {concept} concept is:</td></tr></table>

Table 12: Prompt $P _ { i }$ for counterfactual data generation in all datasets. {concept} are a placeholder to put the concept for generating the counterfactual data. {texts with concept} and {texts without concept} are placeholders to add five demonstrations with or without the concepts. for counterfactual} is a placeholder to add the text to make the counterfactual in the concept level.