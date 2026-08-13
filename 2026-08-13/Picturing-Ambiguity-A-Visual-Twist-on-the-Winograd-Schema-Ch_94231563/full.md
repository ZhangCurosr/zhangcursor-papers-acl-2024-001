# Picturing Ambiguity: A Visual Twist on the Winograd Schema Challenge

Brendan Park\*, Madeline Janecek\*, Naser Ezzati-Jivan, Yifeng Li, and Ali Emami Brock University, St. Catharines, Ontario, Canada {bp18ul, mj17th, nezzatijivan, yli2, aemami}@brocku.ca

## Abstract

Large Language Models (LLMs) have demonstrated remarkable success in tasks like the Winograd Schema Challenge (WSC), showcasing advanced textual common-sense reasoning. However, applying this reasoning to multimodal domains, where understanding text and images together is essential, remains a substantial challenge. To address this, we introduce WINOVIS, a novel dataset specifically designed to probe text-to-image models on pronoun disambiguation within multimodal contexts. Utilizing GPT-4 for prompt generation and Diffusion Attentive Attribution Maps (DAAM) for heatmap analysis, we propose a novel evaluation framework that isolates the models’ ability in pronoun disambiguation from other visual processing challenges. Evaluation of successive model versions reveals that, despite incremental advancements, Stable Diffusion 2.0 achieves a precision of 56.7% on WINOVIS, showing minimal improvement from past iterations and only marginally surpassing random guessing. Further error analysis identifies im portant areas for future research aimed at advancing text-to-image models in their ability to interpret and interact with the complex visual world.

## 1 Introduction

The interpretation of ambiguous constructs in language is crucial for assessing common-sense reasoning, with the Winograd Schema Challenge (WSC) (Levesque et al., 2011; Winograd, 1972) significantly influencing the evaluation of natural language understanding models. Advances in transformer-based architectures have led Large Language Models (LLMs) to achieve impressive results on WSC-based tasks, approaching nearhuman performance (Brown et al., 2020; Sakaguchi et al., 2020; Kocijan et al., 2023).

![](images/49a8d8881d5302b2a7bf6b3e87e75b6ed8d4c66616fade0d5727bb4db65171c7.jpg)  
Figure 1: A representative output from Stable Diffusion 2.0 on a WINOVIS instance. The Diffusion Attentive Attribution Maps (DAAM) clarify the model’s focus for different terms and the correctness of its interpretation: correctly identifying ‘bee’ and ‘flower’ but erroneously associating ‘it’ with the bee instead of the flower.

Extending common-sense reasoning into multimodal domains, especially disambiguation tasks, is a persisting challenge. Despite the ability of models like Google’s Imagen (Saharia et al., 2022), OpenAI’s DALL-E 2 (Ramesh et al., 2022), and Stability AI’s recently open-sourced Stable Diffusion (Rombach et al., 2022) to create visually compelling images from text, their interpretability—essential for deciphering the models’ reasoning processes—is notably limited (Tang et al., 2023). This gap restricts the development of tools for visuals that match complex texts, reducing model effectiveness when deployed in areas like education and digital media, where text-image integration is essential (Dehouche and Dehouche, 2023; Hattori and Takahara, 2023).

Our response to this challenge is WINOVIS, a dataset aimed at probing text-to-image models common-sense reasoning capabilities through pronoun disambiguation within multimodal scenarios. WINOVIS not only tests models’ ability to distinguish entities within the generated images, but also examines how these models associate pronouns with the correct referents, a nuanced aspect of common-sense reasoning that has been overlooked. As depicted in the WINOVIS example in Figure 1, while newer Stable Diffusion models can accurately separate entities within an image, they fail to correctly associate the pronoun ‘it’ with the intended referent, revealing the subtleties and potential gaps in multimodal common-sense reasoning.

The development of WINOVIS leveraged the generative power of GPT-4 (OpenAI, 2023; Gilardi et al., 2023), using a methodical approach to create and refine prompts that elicit common-sense reasoning visually. This process included a complete manual review to ensure each scenario’s clarity and relevance for the disambiguation task. Moreover, we introduce a novel evaluation framework that distinguishes between models’ pronoun disambiguation proficiency from their handling of visual processing challenges, such as susceptibility to typographic attacks (Goh et al., 2021) and semantic entanglement (Wu et al., 2023).

Our contributions are summarized as follows:

• WSC-Adapted Multimodal Dataset (WINOVIS): A dataset of 500 scenarios for benchmarking text-to-image models pronoun disambiguation abilities within a visual context.<sup>1</sup>

• Novel Evaluation Framework for Multimodal Disambiguation: Metrics and methods designed to isolate pronoun resolution from other visual processing challenges, advancing the understanding of models common-sense reasoning.

• Insight into Stable Diffusion’s Common-Sense Reasoning: A critical analysis revealing that even state-of-the-art models like Stable Diffusion 2.0 fall significantly short of human-level performance.

## 2 Background

## 2.1 Latent Diffusion in Image Generation

Latent diffusion models (LDMs) represent a class of generative models designed to synthesize images by progressively refining random noise. A prominent example is Stable Diffusion (Rombach et al., 2022), a text-to-image LDM optimized to generate images from textual prompts. Stable Diffusion integrates three primary components: a deep language model that extracts semantic embeddings from textual prompts; an encoder-decoder architecture for encoding images into latent space representations and decoding them back; and a neural network that is responsible for mean-prediction (Ho et al., 2020) (denoted as $\mu _ { \boldsymbol { \theta } } ( z , \boldsymbol { y } , t ) )$ , noiseprediction (Ho et al., 2020) (denoted as $\epsilon _ { \theta } ( z , \pmb { y } , t ) )$ , or score-prediction (Song and Ermon, 2019) (denoted as $s _ { \theta } ( z , y , t ) )$ ). This network is trained on image and text pairs x and y. During training, which aims to maximize the evidence lower bound (ELBO) (Sohl-Dickstein et al., 2015), the image is initially encoded to $z _ { \mathrm { 0 } }$ , marking the start of the forward diffusion process, formalized as:

![](images/981b3b82383e73a169b8dfa43c0535867db74bd50ccfec43cffc1e17054063fe.jpg)  
Figure 2: A visual overview of the Stable Diffusion architecture, as well as the Diffusion Attention Attribution Map (DAAM) generation process.

$$
\begin{array} { r } { p ( z _ { t } | z _ { t - 1 } ) = N ( z _ { t } | \sqrt { \alpha _ { t } } z _ { t - 1 } , ( 1 - \alpha _ { t } ) I ) } \\ { = N ( z _ { t } | \sqrt { 1 - \beta _ { t } } z _ { t - 1 } , \beta _ { t } I ) , } \end{array}
$$

where ${ \boldsymbol { z } } _ { t }$ denotes the latent variable at step t, with $\beta _ { t } = 1 - \alpha _ { t }$ as the noise schedule hyperparameter, and I the identity matrix. The U-Net architecture (Ronneberger et al., 2015), used for denoising, iteratively reverses the diffusion through:

$$
\begin{array} { r l } & { p ( z _ { t - 1 } | z _ { t } ) = \mathcal { N } \big ( z _ { t - 1 } | \mu _ { \boldsymbol { \theta } } ( z _ { t } , \pmb { y } , t ) , \sigma _ { t } ^ { 2 } \pmb { I } \big ) } \\ & { ~ = \mathcal { N } \Big ( z _ { t - 1 } | \frac { z _ { t } - \frac { \beta _ { t } } { \sqrt { 1 - \bar { \alpha } _ { t } } } \epsilon _ { \boldsymbol { \theta } } \left( z _ { t } , \pmb { y } , t \right) } { \sqrt { \alpha _ { t } } } , \sigma _ { t } ^ { 2 } \pmb { I } \Big ) } \end{array}
$$

with $\sigma _ { t } ^ { 2 }$ as the reverse process noise variance. Cross-attention in the U-Net layers aligns $z _ { t }$ with $\mathbf { \pmb { y } } .$ For conditional generation, the process starts with Gaussian noise $z _ { T }$ , conditioned on text $^ { y , }$ and refines through reverse diffusion, resembling Langevin dynamics (Welling and Teh, 2011). For instance, using the score function, we have

$$
z _ { t - 1 } ^ { ( j ) } = z _ { t - 1 } ^ { ( j - 1 ) } + \frac { \alpha _ { t } } { 2 } s _ { \theta } ( z _ { t - 1 } ^ { ( j - 1 ) } , \pmb { y } , t ) + \sqrt { \alpha _ { t } } \varepsilon _ { j } ,
$$

where $j = 1 , \dots , J ,$ J is the number of Langevin steps, $z _ { t - 1 } ^ { ( 0 ) } = z _ { t } , z _ { t - 1 } = z _ { t - 1 } ^ { ( J ) }$ , and $\varepsilon _ { j } \sim \mathcal { N } ( 0 , I )$ The denoised $z _ { \mathrm { 0 } }$ generates the final image, such as the one exemplified in Figure 2 given a WINOVIS instance.

## 2.2 Diffusion Attentive Attribution Maps

The Diffusion Attentive Attribution Map (DAAM) technique facilitates interpretability of the influence that different tokens in a prompt have on the image generated by Stable Diffusion models (Tang et al., 2023). This approach capitalizes on the multi-head cross-attention mechanism (Vaswani et al., 2023), aggregating attention scores from both downsampling and upsampling stages within the U-Net architecture. The attention scores, denoted as $F _ { t } ^ { ( i ) \downarrow }$ for downsampling and $F _ { t } ^ { ( i ) \dag }$ for upsampling, link specific words from the prompt to image regions, signified by coordinates $( x , y )$ , across different heads (i) and layers (l).

To synthesize a comprehensive heatmap from these attention scores, DAAM applies a spatial normalization procedure, scaling the attention scores for the k-th word to match the original image size and summing them across all attention heads (i), layers (l), and time steps (t):

$$
D _ { k } [ x , y ] = \sum _ { i , t , l } \Big ( F _ { t } ^ { ( i ) \downarrow } [ x , y , l , k ] + F _ { t } ^ { ( i ) \uparrow } [ x , y , l , k ] \Big )
$$

where $F _ { t } ^ { ( i ) \downarrow } [ x , y , l , k ]$ and $F _ { t } ^ { ( i ) \dagger } [ x , y , l , k ]$ represent the bicubically upscaled attention scores for the downsampling and upsampling pathways, respectively.

DAAM can therefore offer a visual method to evaluate how Stable Diffusion performs pronoun disambiguation, by illustrating where the model concentrates its attention in relation to textual prompts. By examining these visualizations, as demonstrated in Figure 1, we can discern the model’s implicit strategies for linking pronouns with their correct referents.

## 3 Constructing WINOVIS

In this section, we detail the methodology behind the creation of WINOVIS, a novel dataset engineered to assess the pronoun disambiguation capabilities of text-to-image models. The integration of GPT-4 (OpenAI, 2023; Gilardi et al., 2023) into our dataset generation workflow allowed for significant streamlining of the creation process, achieving reductions in both cost and time, enhancing reproducibility, and reducing the incidence of human error. Our Corpus Construction Cycle consists of two main stages: 1) The GPT Prompt Cycle; and 2) The Manual Filter Process. A full visualization of the process is provided in Appendix Figure 10.

## 3.1 Corpus Construction Cycle

Step 1: GPT Prompt Cycle In developing WINOVIS, we aimed to adapt the Winograd Schema Challenge (WSC) for visual interpretation. This required avoiding the creation of instances that were visually ambiguous, lacked clear visual contexts, or logically didn’t necessitate both entities. Table 1 showcases problematic examples from the WSC alongside those of WINOVIS.

Our iterative prompting process with GPT-4, as outlined in Appendix Table 5, included both successful and problematic few-shot examples to refine the desired outcomes. This approach, detailed entirely in Appendix Table 6, helped clarify what constitutes an acceptable instance. To enhance logical reasoning, we employed a Chain-of-Thought (CoT) (Wei et al., 2023) strategy, further described in Appendix Table 5 under CoT. Querying instances in batches of ten ensured a varied yet coherent collection while minimizing duplicates.

Step 2: Manual Filter Process After GPT-4 generated the initial set of instances, a manual review was conducted to filter out instances that failed to meet our study’s criteria:

• Textual Ambiguity: If a prompt could not be easily disambiguated by all annotators it was excluded.

<table><tr><td>WSV</td><td>Disparate Entities Distinct Entities (Age) Distinct Entities (Role)</td><td>The thief stole the diamond because it was valuable. (A = diamond) The man carried the child because he was tired. (A = child) The king banished the jester because he was annoying.  $( A = \operatorname { j e s t e r } )$ </td></tr><tr><td>WSC</td><td>Visually Ambiguous Entity Exclusion</td><td>Pete envies Martin because he is very successful. (A = Martin) Jane knocked on Susan&#x27;s door, but there was no answer. She was out. (A = Susan) The dog could not catch the squirrel because it was small.  $( A = ? )$ </td></tr><tr><td>Filtered Textually Ambiguous Illogical</td><td>Visually Indistinctive Redundant Entries</td><td>The fisherman cast the net because it was full of holes.  $( A = \mathrm { n e t } )$  The wrestler defeated the opponent because he was weak. (A = enemy) Anthony admired James because he was talented. (A = James) Ryan respected Andrew because he was talented. (A = Andrew)</td></tr></table>

Table 1: Examples from the WINOVIS (WSV) dataset alongside instances from the Winograd Schema Challenge (WSC) dataset and those excluded through manual filtering. In each case, the correct entity is denoted by A.

• Illogical Content: Removed if containing nonsensical or irrelevant information.

• Visual Indistinctiveness: Omitted when entities lacked clear visual differentiation, essential for accurate entity-pronoun association.

• Redundancy: To ensure a broad range of scenarios, instances that were too similar in content or structure were excluded.

This manual filtering ensured that each prompt included in WINOVIS is well suited for evaluating text-to-image models. Examples of excluded instances for each criterion are provided in Table 1. This review cycle was repeated, refining the selection until achieving a diverse and quality set.

## 3.2 Dataset Characteristics

Each sample of WINOVIS contains a pronoun resolution prompt, a specification of the ambiguous pronoun, an excerpt containing the pronoun, the two referent entity options, the correct referent, and a justification for why the correct entity should be associated with the pronoun.

Disparate and Distinct Entities The instances within WINOVIS fall into two broad categories: Disparate Entities and Distinct Entities. Disparate Entities encompass scenarios with significantly different subjects, such as those across species or object classes (e.g., a person vs. a dog, or a car vs. a tree). Distinct Entities, while sharing some similarities, are visually distinguishable by attributes like age (a mother and child), role (a cop and a thief), or other descriptors, posing more nuanced challenges for pronoun resolution.

WINOVIS is primarily designed to evaluate a model’s common-sense reasoning capabilities, rather than to pose a significant challenge. Consequently, a substantial portion of its instances (84.2%) involve disparate entities. To assess the model in a more demanding context, the remaining 15.8% of the instances feature distinct entities.

Context Types To further examine the comprehensiveness of WINOVIS as a commonsense reasoning benchmark, we categorized each prompt based on the contextual details it provides to link the correct referent to the pronoun. The four contextual categories present in WINOVIS are:

• Visually Tangible: These entries contain descriptions that should have a clear visual impact on the associated referent.

• Emotional: These entries describe the emotional or mental state of the referent, which, although more subtle, would still affect the referent’s appearance.

• Characteristic: These entries include details pertaining to a referent’s personality or nature. While less visually tangible, these details may affect the associated referent’s finer details.

• Visually Intangible: These entries involve attributes with minimal to no visual impact on the referent, such as taste, speed, or sound. These entries assess the model’s understanding of purely textual input.

We argue that proficiency in pronoun disambiguation requires the capacity to effectively leverage all four context types. Therefore, WINOVIS includes prompts from each category, providing a comprehensive assessment of a model’s capabilities. Examples and the distribution of each category within WINOVIS are shown in Appendix Table 7.

![](images/deaff13a405c260e7b1339c2042be03f759ca0991e9ffc8f0257f2e2fdc6cdc4.jpg)

Figure 3: The results of different heatmap thresholds for the prompt “The ant could not carry the leaf because it was too weak” and the term ‘it’.  
![](images/1264c112b43c2400580b992bae1dfb3209e4eed1cc6fa494fd9d4bf90d58353a.jpg)  
Figure 4: Illustrative example of thresholding on attention maps, progressing through stages to apply a 90<sup>th</sup> percentile threshold, resulting in a binary mask that accentuates key attention regions.

## 4 Evaluating Pronoun Disambiguation in WINOVIS

This section outlines our systematic pipeline to evaluate the capability of Stable Diffusion models to accurately disambiguate pronouns within the context of WINOVIS. Our pipeline comprises four stages: 1) Filtering out captioned images to remove visual representations that include embedded text; 2) Enhancing the clarity of distributed attention attribution maps through noise reduction; 3) Excluding images with significant heatmap overlap between referent entities from our analysis; and 4) Determining the model’s final pronoun association by establishing a decision boundary.

Step 1: Caption Filtering Text-to-image LDMs sometimes generate images where prompt text appears visually, resulting in ‘captioned’ images. These images erroneously direct a term’s attribution to this text, complicating the assessment of the model’s visual common-sense reasoning. An example of this is shown in Appendix Figure 11.

We therefore specifically excluded captioned images from the analysis set of a studied model, prioritizing those yielding visuals strictly relevant to common-sense interpretation. This exclusion is based on the premise that visual common-sense reasoning should be assessed purely on the model’s ability to interpret and generate relevant visual content, without the confounding influence of embedded text. Data on the frequency of prompts resulting in captioned images is detailed in Table 2.

Step 2: Noise Reduction in Attention Maps To ensure attention heatmaps clearly reflect the model’s focus, we apply a 90<sup>th</sup> percentile thresholding technique to the heatmaps generated from WINOVIS prompts. This approach filters out the bottom 90% of attention scores, considered as noise, and retains only the highest-intensity areas indicative of the model’s primary interest. This 90<sup>th</sup> percentile threshold was chosen after extensive testing with various thresholds. It was found to be the most effective in balancing the elimination of irrelevant noise while preserving the focal points crucial for understanding the model’s interpretation of the prompt. Thresholds below the 90<sup>th</sup> percentile included too much noise, while higher thresholds risked omitting significant details.

Following this, areas surpassing the threshold are converted into binary masks, delineating significant attention (‘1’) from the rest (‘0’). This representation simplifies the evaluation of the model’s attention distribution, facilitating a more straightforward comparison of its responses to various prompts, thus setting a clearer stage for analyzing how the model associates pronouns with their referents. The impact of this thresholding and the utility of binary masks in enhancing map interpretability are visualized in Figures 3 and 4, respectively.

Step 3: Heatmap Overlap Filtering Building on the binary masks created from the previous step, we next employ the Intersection over Union (IoU) metric to further dissect the model’s pronoun disambiguation capabilities. The IoU metric, widely

The bird avoids the scarecrow because it is threatening

The archaeologist carefully examined the artifact because

![](images/11d6b5b2ae5a17ceb1b1316156a50f66313082e7826717547240326e3062a92b.jpg)  
Figure 5: Instances of heatmap overlap generated by Stable Diffusion 2.0 using the WINOVIS dataset: On the left, two entities lead to nearly identical heatmaps, while on the right, two visually distinct entities show significant heatmap overlap.

recognized in computer vision for evaluating object detection accuracy (He et al., 2017; Szeliski, 2022; Takikawa et al., 2019; Zhu et al., 2019), measures the overlap between two areas. It is commonly applied to assess the precision of detected objects against ground truth, by comparing their respective binary masks. The IoU calculation is as follows:

$$
\mathrm { I o U } = { \frac { \mathrm { A r e a ~ o f ~ O v e r l a p ~ b e t w e e n ~ t h e ~ b i n a r y ~ m a s k s } } { \mathrm { A r e a ~ o f ~ U n i o n ~ o f ~ t h e ~ b i n a r y ~ m a s k s } } }
$$

This yields a value from 0 (no overlap) to 1 (complete overlap), indicating the strength of association between two terms.

For our purposes, a high IoU score between a pronoun and an entity suggests a correct pronounto-entity linkage by the model, while high scores between both entities indicate ‘heatmap overlap’—a state where the model fails to distinguish entity associations, leading to potential misattribution of the pronoun. Refer to Figure 5 for examples of this phenomenon.

Heatmap overlap complicates pronoun disambiguation, as it reflects a failure to distinguish between entities in the first place. To identify an optimal overlap threshold for detecting such errors, we manually inspected 50 WINOVIS instances, evaluating heatmap overlays from Stable Diffusion 2.0. A consensus emerged favoring an IoU threshold of 0.4, which yielded full agreement with classifications made by our team, as depicted in Figure 6.

WINOVIS instances with entity pairs with IoU scores exceeding this threshold are therefore considered invalid, warranting exclusion from further analysis to ensure a focus on clear cases of pronoun disambiguation. This filtering process’s impact on the dataset, segmented by model versions is detailed in Table 2.

<table><tr><td>Version</td><td>Captioned</td><td>Overlapped</td><td>Evaluable</td></tr><tr><td>1.0</td><td>178</td><td>24</td><td>298</td></tr><tr><td>1.5</td><td>135</td><td>36</td><td>329</td></tr><tr><td>2.0</td><td>160</td><td>71</td><td>269</td></tr><tr><td>XL</td><td>2</td><td>73</td><td>425</td></tr></table>

Table 2: The number of images generated by Stable Diffusion versions from WINOVIS prompts, categorized by suitability for pronoun disambiguation analysis.

![](images/e91d18d4420375f49a3b4e2ce66a8d47f3d586b6056173828ec807358de605ad.jpg)

![](images/c8c7080b806c3cb8b72e0e83155884bae4c203c64361125668ade5c48ad41437.jpg)  
loU  
Figure 6: Depicts the level of agreement between the manual decisions and different IoU values for the overlap threshold (left) and decision boundary (right).

Step 4: Making the Final Decision In this final step, we utilize the IoU metric once more to establish a decision boundary for evaluating the model’s proficiency in pronoun disambiguation. This process involved another comparative analysis conducted by our team of annotators, who manually assessed 50 images generated by SD 2.0 from

<table><tr><td>SD Version</td><td>#Correct</td><td>#Incorrect #Neither</td><td></td><td>Precision</td><td>Recall</td><td>F1-Score</td><td>Certainty</td></tr><tr><td>1.0</td><td>24</td><td>24</td><td>250</td><td>50.0</td><td>8.8</td><td>14.9</td><td>16.1</td></tr><tr><td>1.5</td><td>38</td><td>31</td><td>260</td><td>55.1</td><td>12.8</td><td>20.7</td><td>21.0</td></tr><tr><td>2.0</td><td>55</td><td>42</td><td>172</td><td>56.7</td><td>24.2*</td><td>34.1*</td><td>36.1*</td></tr><tr><td>XL</td><td>1</td><td>0</td><td>424</td><td>N/A</td><td>N/A</td><td>N/A</td><td>0.24</td></tr></table>

Table 3: Comparative performance of Stable Diffusion (SD) models 1.0, 1.5, 2.0, and Stable Diffusion XL (SDXL) (Podell et al., 2023) on the WINOVIS dataset. Metrics are presented as percentages, with \* indicating a statistically significant difference for best model (2.0) from second best (1.5) based on a Z-test for two independent proportions $( \mathtt { p } < 0 . 0 1 )$

![](images/c8e9c733adbf30ef4681820f85041e75b5c56e5fe231cd60e19959222e605e30.jpg)

![](images/a589acec8ffce18b4c6af22b6ce02201e9a513ddebcc0ce1917dea3e0b5cf8dd.jpg)  
Figure 7: Confusion matrices showing raw count performances of Stable Diffusion models on WINOVIS. Each matrix provides the counts of predictions for Entity 1 and Entity 2 against their true labels.

WINOVIS instances. Each image was reviewed with its corresponding heatmap to determine the presence of a definitive pronoun-to-entity association. Remarkably, the IoU threshold that aligned with manual assessments was identified again at 0.4, mirroring the overlap threshold. This consistency underscores the threshold’s robustness in distinguishing between clear and ambiguous entity associations (Figure 6 illustrates this agreement).

An IoU score exceeding this threshold signals a strong association between the pronoun and a specific entity, as interpreted by the model. This scenario unfolds in two ways:

• If only one referent entity’s IoU score with the pronoun surpasses this threshold, it directly informs the model’s prediction, indicating a clear pronoun-to-entity association.

• If both referent entities’ IoU scores exceed the threshold, the entity with the higher score is considered the model’s chosen referent.

Predictions are categorized as either correct or incorrect based on their alignment with the WINO-VIS instance’s intended meaning. Cases where neither entity meets the IoU criterion are labeled as neither, suggesting the model’s failure to disambiguate the pronoun altogether.

## 5 Experiments & Results

## 5.1 Experimental Setup

Dataset Generation and GPT-4 Configuration For dataset generation, we used GPT-4 (gpt-4-0613;

(OpenAI et al., 2023)) with temperature and nucleus sampling (top-p) settings optimized to enhance output diversity while adhering to the specific task structure detailed in the prompts. After evaluating temperature values within the range [0,2] with a fixed top-p of 1.0, we determined a temperature of 0.8 as the optimal balance for maintaining both dataset integrity and diversity. The WINOVIS images were then generated using Stable Diffusion (SD) versions 1.1, 1.5, 2.0 and Stable Diffusion XL (SDXL), through HuggingFace’s Diffusers library (von Platen et al., 2022), with each model configured to use 50 diffusion steps.

Diffusion Steps Analysis An analysis of image generation quality across different diffusion step settings (20, 50, and 100 steps) was performed to identify the optimal configuration for producing WINOVIS images. The evaluation criteria included image quality and the presence of unintended captioning. Fifty steps were found to provide the best balance between image quality and computational efficiency, with no significant quality improvements observed at 100 steps.

Main Experiments Using the WINOVIS dataset, we prompted SD versions 1.1, 1.5, 2.0 and XL to generate corresponding images. Throughout this process, heatmaps for both entities and the pronoun were extracted.<sup>2</sup> These prepared heatmaps enabled the application of the IoU metric, as elaborated in Steps 3 and 4 of Section 4.

Evaluation Metrics We measure model performance using the following metrics, adapted for pronoun disambiguation tasks:

• Certainty: The frequency with which the model makes a clear pronoun-to-entity association as opposed to its assocations being marked as ‘neither’.

• Precision: The proportion of the model’s pronoun-to-entity associations that are correct out of all associations made.

• Recall: The model’s ability to correctly associate pronouns with entities, where ‘neither responses are treated as missed opportunities for correct associations (i.e., false negatives).<sup>3</sup>

• F1-Score: The harmonic mean of precision and recall, providing an overall measure of the model’s disambiguation performance.

## 5.2 Results

Table 3 presents the performance of models on the WINOVIS dataset. Key insights include:

Model Progression and Certainty: SD 2.0 demonstrates superior precision, recall, and F1- scores, alongside a reduced rate of neither predictions, indicating both progress in pronoun disambiguation and decisiveness. Despite advancements, all models still show a significant need for development, with persistent challenges highlighted by the notable proportion of ‘neither’ outcomes and modest precision scores.

The confusion matrices depicted in Figure 7 show the raw count performance of models on the WINOVIS dataset’s pronoun disambiguation problems. Notably, the matrices indicate a gradual decrease in the confusion between entities as the model version increases, with SD 2.0 showing a more distinct separation between the two entities. This suggests an improvement in the models ability to discern between entities over iterations.

Dismal SDXL Performance: SDXL’s attention maps almost always did not meet the IoU threshold set out for a viable prediction on WINOVIS. Specifically, the heatmaps attributed to the pronoun were often widely dispersed across the image, resulting in a neither prediction. An example of this can be seen in Appendix Figure 15.

![](images/7c4dd9b67f76ebdd9b31bcb9bb37c309133806f40603e6f51abc4651451e0485.jpg)  
Figure 8: A comparison of the proportion of correct, incorrect, neither, overlapped, and captioned images when SD 2.0 is given distinct versus disparate entities.

The culprit for this may be SDXL’s consideration of a large context for high-resolution generation. Effectively, this may dilute the attention weights of ambiguous tokens and the extra refiner component would impact the generation of attention heatmaps altogether. At the same time, it was intriguing that this issue occurs exclusively for the token corresponding to the ambiguous pronoun (i.e., in Appendix Figure 15, both the ant and the leaf result in heatmaps that SDXL correctly identifies). This may suggest a tradeoff between image generation quality and pronoun disambiguation – larger, more capable models may come with a pronounced cost to interpretability, resulting compromised performance on benchmarks such as WinoVis.

## 6 Error Analysis

In this section, we further examine the performance of the most effective model iteration, SD 2.0. We compare the results across our dataset categories outlined in Section 3.2, namely disparate and distinct entities. The proportions of correct, incorrect, overlapped, neither, and captioned instances for both categories are visualized in Figure 8.

Disparate Entities: In general, SD 2.0 performed the best when working with disparate entities (recall that these were the “easier” problems). Over half of the images were evaluable, with the other 43.4% containing captioning or heatmap overlap. Among the evaluable instances, 31.4% had neither entity chosen, 9.5% were incorrect, and 12.1% were correct. Figure 9 (left) shows SD 2.0’s incorrect pronoun attribution in a WINOVIS scenario involving disparate entities.

Distinct Entities: SD 2.0 struggled the most with distinct entities. The majority of instances were not evaluable, with 60.8% of the items containing captioning or heatmap overlap. Among the evaluable

![](images/e349846b4f65555eeca7bf9755664522944a68421f723ed9669ffb48de5b3594.jpg)  
Figure 9: Examples of incorrect pronoun associations for disparate entities (left) and distinct entities (right).

The bird couldn't eat the berry because it was too big

The woman gave the child a candy because she was very generous

instances, it displayed notable difficulties in making the correct association: in 31.6% of instances neither entity was chosen, while in 2.5% of cases, the incorrect entity was chosen. Only 5.1% of instances resulted in the correct entity being chosen. Figure 9 (right) depicts an example of two distinct entities, a child and a woman. Interestingly, in this image the pronoun ‘she’ is more strongly attributed to the child instead of the woman, even when the child’s gender is not specified.

## 7 Related Work

Multimodal Reasoning The recent surge in popularity of generative models has underscored the necessity for explainable creativity (Llano et al., 2020), leading to a significant body of research investigating the determinants of high-quality prompts for image generation (Wang et al., 2023b; Oppenlaender, 2023; Pavlichenko and Ustalov, 2023). Despite these advancements, the evaluation of how vision models actually interpret prompts is largely underexplored. Most studies focus on the models’ semantic understanding of terms (Tang et al., 2023; Parcalabescu et al., 2022; Thrush et al., 2022) or susceptibility to bias (Wang et al., 2023a). These evaluations often involve direct, unambiguous prompts, sidestepping more nuanced challenges. WINOVIS addresses this gap by evaluating the common-sense reasoning of models through the lens of pronoun resolution. This challenge not only expands the scope of assessment for generative models but also sets a new benchmark for understanding their capabilities in interpreting complex linguistic structures.

WSC-Style Tasks The Winograd Schema Challenge (WSC) (Levesque et al., 2011) has catalyzed the development of various datasets aimed at advancing pronominal coreference resolution, each enriching the field by addressing distinct facets of the challenge. Datasets such as Winogrande (Sakaguchi et al., 2020) and KnowRef (Emami et al., 2019) expand on the WSC by tackling its limited size, whereas WinoGender (Rudinger et al., 2018), WinoBias (Zhao et al., 2018), and KnowRef-60k (Emami et al., 2020) study model biases. Further enhancements and crowd-sourcing efforts (Wang et al., 2018; Trichelair et al., 2018; Kocijan et al., 2019; Elazar et al., 2021; Zahraei and Emami, 2024; Sakaguchi et al., 2020) have continually refined the WSC task’s scope and methodology. WINOVIS uniquely adapts the WSC for text-toimage model evaluation, focusing on multimodal common-sense reasoning. It introduces the challenge of visually disambiguating pronouns, filling a crucial gap in multimodal evaluation.

## 8 Conclusion

This paper presented WINOVIS, a new approach to test how well text-to-image models like Stable Diffusion handle pronoun disambiguation. Our work reveals significant gaps in these models’ abilities to interpret ambiguous scenarios accurately. Central to our contribution is a novel evaluation framework designed to isolate common-sense reasoning in pronoun disambiguation from well-studied challenges such as typographic attacks and semantic entanglement. Future research should build on our groundwork to develop models that not only generate visually compelling images but also accurately understand the narratives and relationships within them.

## Limitations

Entity Separation: Stable Diffusion models encounter challenges with distinguishing between two semantically similar entities. This can be seen in either heatmap overlap or entanglement, both of which result in a significant proportion of generated images being unsuitable for pronoun disambiguation. Entanglement is particularly pronounced in images generated from prompts featuring semantically similar entities. Since sentences from WINO-VIS often employ such entities to introduce ambiguity, resolving entanglement could improve the model’s ability to distinguish individual entities and expand the range of Winograd-like prompts that Stable Diffusion can visualize for our analysis.

Model Diversity: Due to its open-source nature, Stable Diffusion facilitated the creation of heatmaps using DAAM, a capability not available in closed-source LDMs. Currently, DAAM is the only framework which enables the interpretation of such models and is specifically designed for Stable Diffusion. Future research should investigate methods to enhance interpretability across a wider range of LDMs and multi-modal diffusion models (and more open-source ones, as they become increasingly available), enabling their assessment in pronoun disambiguation using WINOVIS.

Bias Analysis: Our study does not explicitly address potential biases in Stable Diffusion that might influence its decision-making processes. Instances of incorrect pronoun resolution, such as the womanchild example depicted in Figure 9, hint at underlying biases. Future work should rigorously explore these biases and their effects on model performance. Investigating whether Stable Diffusion exhibits systematic preferences in resolving ambiguities could uncover patterns in its reasoning strategies, guiding efforts to mitigate biases and enhance multimodal pronoun disambiguation capabilities.

Dataset Diversity: Although efforts were made to maximize dataset diversity during the generation of samples for WINOVIS, opportunities for enhancement remain. Further refinement could entail creating samples that exhibit greater complexity and encompass a broader spectrum of circumstances, entities, and instances of ambiguous pronouns.

Filtering Limitations: Although our filtering process aimed to minimize the impact of model weaknesses on our analysis, exceptions exist. In certain cases, semantic entanglement eluded detection through heatmap overlap measures (see Appendix Figure 14 for an example). Future research should investigate alternative detection methods to better mitigate the influence of such model flaws on our analysis of WINOVIS.

## Acknowledgements

This work was supported by the Natural Sciences and Engineering Research Council of Canada, the New Frontiers in Research Fund, and the Canada Research Chairs Program.

## References

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners.

Nassim Dehouche and Kullathida Dehouche. 2023. What’s in a text-to-image prompt? The potential of stable diffusion in visual arts education. Heliyon.

Yanai Elazar, Hongming Zhang, Yoav Goldberg, and Dan Roth. 2021. Back to square one: Artifact detection, training and commonsense disentanglement in the Winograd schema. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 10486–10500, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Ali Emami, Kaheer Suleman, Adam Trischler, and Jackie Chi Kit Cheung. 2020. An analysis of dataset overlap on Winograd-style tasks. In Proceedings of the 28th International Conference on Computational Linguistics, pages 5855–5865.

Ali Emami, Paul Trichelair, Adam Trischler, Kaheer Suleman, Hannes Schulz, and Jackie Chi Kit Cheung. 2019. The KnowRef coreference corpus: Removing gender and number cues for difficult pronominal anaphora resolution. In Proceedings ofthe 57th Annual Meeting ofthe Associationfor Computational Linguistics, pages 3952–3961, Florence, Italy. Association for Computational Linguistics.

Fabrizio Gilardi, Meysam Alizadeh, and Maël Kubli. 2023. Chatgpt outperforms crowd-workers for textannotation tasks. arXiv preprint arXiv:2303.15056.

Gabriel Goh, Nick Cammarata, Chelsea Voss, Shan Carter, Michael Petrov, Ludwig Schubert, Alec Radford, and Chris Olah. 2021. Multimodal neurons in artificial neural networks. Distill, 6(3):e30.

Shun Hattori and Madoka Takahara. 2023. A study on human-computer interaction with text-to/from-image game AIs for diversity education. In International Conference on Human-Computer Interaction, pages 471–486. Springer.

Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. 2017. Mask R-CNN. In Proceedings ofthe IEEE International Conference on Computer Vision, pages 2961–2969.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. In Conference on Neural Information Processing Systems.

Vid Kocijan, Ana-Maria Cretu, Oana-Maria Camburu, Yordan Yordanov, and Thomas Lukasiewicz. 2019. A surprisingly robust trick for the Winograd schema challenge. In Proceedings ofthe 57th Annual Meeting of the Association for Computational Linguistics, pages 4837–4842, Florence, Italy. Association for Computational Linguistics.

Vid Kocijan, Ernest Davis, Thomas Lukasiewicz, Gary Marcus, and Leora Morgenstern. 2023. The defeat of the Winograd Schema Challenge. Artificial Intelligence, 325:103971.

Hector Levesque, Ernest Davis, and Leora Morgenstern. 2011. The Winograd Schema Challenge. In AAAI Spring Symposium: Logical Formalizations of Commonsense Reasoning.

Maria Teresa Llano, Mark d’Inverno, Matthew Yee-King, Jon McCormack, Alon Ilsar, Alison Pease, and Simon Colton. 2020. Explainable computational creativity. In International Conference on Computational Creativity 2020, ICCC’20, pages 334–341. Association for Computational Creativity (ACC). International Conference on Computational Creativity 2020, ICCC 2020 ; Conference date: 07-09-2020 Through 11-09-2020.

OpenAI, :, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mo Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien

Deville, Arka Dhar, David Dohan, Steve Dowl ing, Sheila Dunning, Adrien Ecoffet, Atty Eleti Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Simón Posada Fishman, Juston Forte, Is abella Fulford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Jo hannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Hendrik Kirch ner, Jamie Kiros, Matt Knight, Daniel Kokotajlo Łukasz Kondraciuk, Andrew Kondrich, Aris Kon stantinidis, Kyle Kosic, Gretchen Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKinney, Christine McLeavey, Paul McMillan Jake McNeil, David Medina, Aalok Mehta, Jacob Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa, Danie Mossing, Tong Mu, Mira Murati, Oleg Murk, David Mély, Ashvin Nair, Reiichiro Nakano, Rajeev Nayak, Arvind Neelakantan, Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O’Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano, Giambat tista Parascandolo, Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perel man, Filipe de Avila Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto, Michael, Poko rny, Michelle Pokrass, Vitchyr Pong, Tolly Pow ell, Alethea Power, Boris Power, Elizabeth Proehl Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond, Francis Real, Kendra Rimbach Carl Ross, Bob Rotsted, Henri Roussez, Nick Ry der, Mario Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel Selsam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens Jordan Sitkin, Katarina Slama, Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Fe lipe Petroski Such, Natalie Summers, Ilya Sutskever, Jie Tang, Nikolas Tezak, Madeleine Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Pre ston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Cerón Uribe, Andrea Vallone, Arun Vijayvergiya, Chelsea Voss, Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei, CJ Weinmann, Akila Welihinda, Peter Welinder, Ji ayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich, Hannah Wong

Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao Zheng, Juntang Zhuang, William Zhuk, and Barret Zoph. 2023. Gpt-4 technical report.

OpenAI. 2023. Gpt-4 technical report. ArXiv, abs/2303.08774.

Jonas Oppenlaender. 2023. A taxonomy of prompt modifiers for text-to-image generation. Behaviour & Information Technology, pages 1–14.

Letitia Parcalabescu, Michele Cafagna, Lilitta Muradjan, Anette Frank, Iacer Calixto, and Albert Gatt. 2022. VALSE: A task-independent benchmark for vision and language models centered on linguistic phenomena. In Proceedings of the 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 8253–8280, Dublin, Ireland. Association for Computational Linguistics.

Nikita Pavlichenko and Dmitry Ustalov. 2023. Best prompts for text-to-image models and how to find them. In Proceedings ofthe 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’23, page 2067–2071, New York, NY, USA. Association for Computing Machinery.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. 2023. Sdxl: Improving latent diffusion models for high-resolution image synthesis.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. 2022. Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. 2015. U-Net: Convolutional networks for biomedical image segmentation.

Rachel Rudinger, Jason Naradowsky, Brian Leonard, and Benjamin Van Durme. 2018. Gender bias in coreference resolution. In Proceedings of the 2018 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 8–14, New Orleans, Louisiana. Association for Computational Linguistics.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. 2022. Photorealistic text-to-image diffusion models with deep

language understanding. Advances in Neural Information Processing Systems, 35:36479–36494.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2020. WinoGrande: An adversarial Winograd schema challenge at scale. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 8732–8740.

Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. 2015. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning.

Yang Song and Stefano Ermon. 2019. Generative modeling by estimating gradients of the data distribution. In Neural Information Processing Systems.

Richard Szeliski. 2022. Computer vision: Algorithms and applications. Springer Nature.

Towaki Takikawa, David Acuna, Varun Jampani, and Sanja Fidler. 2019. Gated-SCNN: Gated shape CCNs for semantic segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5229–5238.

Raphael Tang, Linqing Liu, Akshat Pandey, Zhiying Jiang, Gefei Yang, Karun Kumar, Pontus Stenetorp, Jimmy Lin, and Ferhan Ture. 2023. What the DAAM: Interpreting stable diffusion using cross attention. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 5644–5659, Toronto, Canada. Association for Computational Linguistics.

Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross. 2022. Winoground: Probing vision and language models for visio-linguistic compositionality. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5228–5238.

Paul Trichelair, Ali Emami, Jackie Chi Kit Cheung, Adam Trischler, Kaheer Suleman, and Fernando Diaz. 2018. On the evaluation of common-sense reasoning in natural language understanding. In Critiquing and Correcting Trends in Machine Learning NeurIPS 2018 Workshop.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2023. Attention is all you need.

Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj, and Thomas Wolf. 2022. Diffusers: Stateof-the-art diffusion models. https://github.com/ huggingface/diffusers.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. 2018. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In Proceedings of the

2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, pages 353–355, Brussels, Belgium. Association for Computational Linguistics.

Jialu Wang, Xinyue Liu, Zonglin Di, Yang Liu, and Xin Wang. 2023a. T2IAT: Measuring valence and stereotypical biases in text-to-image generation. In Findings of the Association for Computational Linguistics: ACL 2023, pages 2560–2574, Toronto, Canada. Association for Computational Linguistics.

Zijie J. Wang, Evan Montoya, David Munechika, Haoyang Yang, Benjamin Hoover, and Duen Horng Chau. 2023b. DiffusionDB: A large-scale prompt gallery dataset for text-to-image generative models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 893–911, Toronto, Canada. Association for Computational Linguistics.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. 2023. Chain-of-thought prompting elicits reasoning in large language models.

Max Welling and Yee W Teh. 2011. Bayesian learning via stochastic gradient langevin dynamics. In Proceedings of the 28th International Conference on Machine Learning (ICML-11), pages 681–688. Citeseer.

Terry Winograd. 1972. Understanding natural language. Cognitive Psychology, 3(1):1–191.

Qiucheng Wu, Yujian Liu, Handong Zhao, Ajinkya Kale, Trung Bui, Tong Yu, Zhe Lin, Yang Zhang, and Shiyu Chang. 2023. Uncovering the disentanglement capability in text-to-image diffusion models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1900–1910.

Pardis Sadat Zahraei and Ali Emami. 2024. WSC+: Enhancing the Winograd Schema Challenge using tree-of-experts. arXiv preprint arXiv:2401.17703.

Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, and Kai-Wei Chang. 2018. Gender bias in coreference resolution: Evaluation and debiasing methods. In Proceedings of the 2018 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 15–20, New Orleans, Louisiana. Association for Computational Linguistics.

Yi Zhu, Karan Sapra, Fitsum A Reda, Kevin J Shih, Shawn Newsam, Andrew Tao, and Bryan Catanzaro. 2019. Improving semantic segmentation via video propagation and label relaxation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8856–8865.

## A Appendix

![](images/5ca168efe56b99cff5b26e2b179ce7387db0b7196622959c5dcf1b3156b2717e.jpg)  
Figure 10: A visual overview of our Corpus Construction Cycle.

![](images/5085e392dd56d3798ccb4cd7353d8305cbe2a96f2f00b40b4bbb404519b21ca3.jpg)

![](images/a24fc97adeca490ed2f06927041c0def307c48c02af3599b3c157ebcd69cdc11.jpg)  
Figure 11: An example of image captioning. In this case, the prompt “The customer returned the product because it was unsatisfied" produced an image that includes the word ‘customer’. The attribution heatmap for the term ‘customer’ focuses on this text.

<table><tr><td rowspan="2">Model</td><td colspan="2">Correct Attribution</td><td colspan="2">Incorrect Attribution</td><td rowspan="2">Accuracy</td><td rowspan="2">Precision</td><td rowspan="2">Recall</td><td rowspan="2">F1-Score</td></tr><tr><td>Entity 1</td><td>Entity 2</td><td>Entity 1</td><td>Entity 2</td></tr><tr><td>1.0</td><td>16</td><td>8</td><td>16</td><td>8</td><td>50.0</td><td>50.0</td><td>50.0</td><td>50.0</td></tr><tr><td>1.5</td><td>25</td><td>13</td><td>19</td><td>12</td><td>55.1</td><td>54.4</td><td>54.1</td><td>54.3</td></tr><tr><td>2.0</td><td>29</td><td>26</td><td>24</td><td>18</td><td>56.7</td><td>56.9</td><td>56.9</td><td>56.9</td></tr></table>

Table 4: An alternate evaluation of Standard Diffusion models that treats the problem as a multi-class classification task. The reported Precision and Recall scores are computed by taking the average of both entity classes.

<table><tr><td rowspan=1 colspan=1>Component</td><td rowspan=1 colspan=1>Prompt Content</td></tr><tr><td rowspan=1 colspan=1>Setup</td><td rowspan=1 colspan=1>A Winograd schema sentence is a sentence that contains an ambiguity and requiresworld knowledge and reasoning for its resolution. For example: The city councilmenrefused the demonstrators a permit because they feared violence.Here, “they&quot; presumably refers to the city council; because city councils aretypically responsible for maintaining order and avoiding violence in their city. Itis more plausible that a city council would fear violence than actively advocate forit. In this example we get the answer based on our world knowledge that tells us citycouncils generally wish to preserve order, while protest movements sometimes embraceconfrontation and violence to achieve political aims. This matches the logicalreferents in the schema.</td></tr><tr><td rowspan=1 colspan=1>Criteria</td><td rowspan=1 colspan=1>Winograd schema sentences must abide by five rules:1. Be easily disambiguated by the reader;2. Not be solvable by simple techniques such as selectional restrictions;3. The “snippet&quot; must directly refer to the entity specified by the “answer&quot;4. Neither of the “options&quot; should be found in the “snippet&quot;.5. The “pronoun&quot; must be applicable to both “options&quot;. For example, two mencould share the pronoun “he&quot; or “him&quot;. Furthermore, a person with an occupationsuch as an athlete or doctor and a non-human entity cannot share the pronouns “he&quot;or “she&quot; but may share “it&quot;. If a plural pronoun is used such as “they&quot; then both“options&quot; should also be plural. For example, coaches instead of coach and playersinstead of player.</td></tr><tr><td rowspan=1 colspan=1>Examples</td><td rowspan=1 colspan=1>Here is an example of some sentences which match the format of the Winograd schema:(using output with reason examples)INSERT WSC SAMPLESAn example of an invalid pair is:Sentence1:{“statement&quot;: “The boy kicked the ball because it was deflated.&quot;,“pronoun”:“it”,“snippet&quot;: “it was deflated”,“options&quot;: [“the boy”, &quot;the ball&quot;].“answer&quot;: 1,&quot;reason&quot;: “If &#x27;deflated’ is used, it implies the ball was deflated.&quot;1</td></tr><tr><td rowspan=1 colspan=1>CoT</td><td rowspan=1 colspan=1>Without skipping any, come up with BATCH_SIZE new valid sentences starting atsentence one. Think step by step for each new sentence by following these steps:1. Come up with two entities or objects which share a pronoun.2. Think of a pronoun that seems just as semantically compatible with the twoantecedent options, but can be disambiguated using common sense reasoning and notat all with distributional cues between the antecedents and the rest of the sentence.3. Come up with a completely new sentence that follows the principles of the examplesentences and follows the rules listed above.Repeat this process for all the sentences you generate. The sentences should be originaland diverse in the topics that they cover.</td></tr></table>

Table 5: The prompt used in the Corpus Construction Cycle broken down into distinct sections.

The athlete left the game because it was [risky/exhausting].   
a:   
{“statement": “The athlete left the game because it was risky.",   
“pronoun": “it",   
“snippet": “it was risky",   
“options": [“athlete", “game"],   
“answer": 1,   
“reason": “If ‘risky’ is used, it implies the game was risky, causing the   
athlete to leave."}   
b:   
{“statement": “The athlete left the game because it was exhausting.",   
“pronoun": “it",   
“snippet": “it was exhausting",   
“options": [“athlete", “game"],   
“answer": 0,   
“reason": “If ‘exhausting’ is used, it implies the athlete was exhausted,   
causing him to leave the game."}   
Explanation: The “snippet" refers to the game’s impact on the athlete when it   
should refer to the “athelete" itself. To correct this sample, the term used should   
be exhausted instead of exhausting.   
The boy kicked the ball because it was [deflated/inflated].   
a:   
{ “statement": “The boy kicked the ball because it was deflated.",   
“pronoun": “it",   
“snippet": “it was deflated",   
“options": [“the boy", “the ball"],   
“answer": 1,   
“reason": “If ‘deflated’ is used, it implies the ball was deflated." }   
b:   
{ “statement": “The boy kicked the ball because it was inflated.",   
“pronoun": “it",   
“snippet": “it was inflated",   
“options": [“the boy", “the ball"],   
“answer": 1,   
“reason": “If ‘inflated’ is used, it implies the ball was inflated, prompting   
the boy to kick it."}   
Explanation: In a Pair, a and b must not have the same “answer". If Pair2.a’s   
“answer" is 0, Pair2.b’s “answer" should be 1 and vice-versa.  
Table 6: Examples of invalid instances that were included in the prompt used in the Corpus Construction Cycle.

<table><tr><td>Context Type</td><td>% of WSV</td><td>Example</td></tr><tr><td>Visually Tangible</td><td>38.6</td><td>The plumber had to replace the pipe because it was rusty.</td></tr><tr><td>Emotional</td><td>15.0</td><td>The dog chased the car because it was excited.</td></tr><tr><td>Characteristic</td><td>29.2</td><td>The king did not trust the advisor because he was deceitful.</td></tr><tr><td>Visually Intangible</td><td>17.2</td><td>The cat is afraid of the vacuum cleaner because it is loud.</td></tr></table>

Table 7: Examples taken from the WINOVIS (WSV) dataset exhibiting the four different context types.

# The horse outran the dog because it was faster

![](images/af3a91bffb3245246d24ae341478d128858cb0ebf26cb06a0bcd2cc8fe89dc8f.jpg)

![](images/8636bd522be7b83777bbb80e30c5cd6fccab342728fa5b653cff3ae22bf83eda.jpg)  
horse

![](images/471702dfed612269c6b53706c5445ac7f83afb2c41e89bea4efb761a4a4e2809.jpg)  
dog

![](images/9ea568302998c547850a7f6a21999106ac8951a80162c25588f92c7aa4315045.jpg)  
it  
Figure 12: An example of a generated image containing only one of the entities from the prompt. While the horse is visible, the dog is not.

# The cat avoided the water because it was scared

![](images/79f1777e9ad996e7b07824cc0b6a3cdb6da80896ce426ca56fa7947b009d3222.jpg)

![](images/df911032ca43f4b48ecbc735e1e5aa7cbe8ba46c6902de6338172e9a9bb09871.jpg)  
cat

![](images/2b480b431806904831bf71c3523fdec5849c080f9a5ac5ba73561e0c0439fb4f.jpg)  
water

![](images/5b347a8ce83671216a0fdb555a14b8da882ecfd51cbdf61eb5362b7e9d83cdcc.jpg)  
it  
Figure 13: An example of the case where the DAAM heatmap for the pronoun does not clearly indicate a decision made by the model. Rather than overlapping with either the ‘cat’ or the ‘water’, the heatmap for ‘it’ appears to overlap slightly with both while also encompassing some space not seen in either of the entities’ heatmaps.

# The wolf attacked the sheep because it was hungry

![](images/7a873fb148b65c64f78045912c66ebed89e185ef59bfc0be44f504d87606ce1e.jpg)

![](images/ffbe7995d6b31edce7ad8d43de84690fe88293c56e79a5082660792b8026ade3.jpg)  
wolf

![](images/1278ef4d7a915f4e6e07992d20805276a443a6d35375ab11a94c6bea8974890e.jpg)  
sheep

![](images/1cdde20bbe634f103a9f25e03abc6d267f96fbc2cd0b3b0310e9fb99decced12.jpg)  
it

Figure 14: An example of an image that was not automatically filtered via measurement of heatmap overlap. While the two entities are semantically entangled their heatmaps are distinct (non-overlapped).

# The ant could not carry the leaf because it was weak

![](images/5c0b2eedba4252a33155a2b48b74b3fd2129525254965519655f700eea2fbfcc.jpg)

![](images/ac6f10d601de8192a140908eee6f9d53ca471487ccccd4cd4f5a64d606087bba.jpg)  
ant

![](images/776003bf51c27296f3469bc41333e8b3955a5c96f1a0cdc7d90208bd619d942d.jpg)  
leaf

![](images/83d4af36f7d70c4af2b341744e807f3e6b2c35ac6c3325c585c0a06750f8015c.jpg)  
it  
Figure 15: An example of an image generated by SDXL. Here, both entity heatmaps overlap correctly with their respective visual representations. However, the heatmap for the ambiguous pronoun is distributed across the image showing a lack of certainty in the model’s decision.