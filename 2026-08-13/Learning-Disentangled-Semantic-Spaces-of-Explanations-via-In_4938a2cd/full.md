# Learning Disentangled Semantic Spaces of Explanations via Invertible Neural Networks

Yingji Zhang<sup>1</sup>†, Danilo S. Carvalho<sup>1,3</sup>, André Freitas<sup>1,2,3</sup>

<sup>1</sup> Department of Computer Science, University of Manchester, United Kingdom <sup>2</sup> Idiap Research Institute, Switzerland

<sup>3</sup> National Biomarker Centre, CRUK-MI, Univ. of Manchester, United Kingdom {firstname.lastname}@[postgrad.]†manchester.ac.uk

## Abstract

Disentangled latent spaces usually have better semantic separability and geometrical properties, which leads to better interpretability and more controllable data generation. While this has been well investigated in Computer Vision, in tasks such as image disentanglement, in the NLP domain sentence disentanglement is still comparatively under-investigated. Most previous work have concentrated on disentangling task-specific generative factors, such as sentiment, within the context of style transfer. In this work, we focus on a more general form of sentence disentanglement, targeting the localised modification and control of more general sentence semantic features. To achieve this, we contribute to a novel notion of sentence semantic disentanglement and introduce a flow-based invertible neural network (INN) mechanism integrated with a transformer-based language Autoencoder (AE) in order to deliver latent spaces with better separability properties. Experimental results demonstrate that the model can conform the distributed latent space into a better semantically disentangled sentence space, leading to improved language interpretability and controlled generation when compared to the recent state-of-the-art language VAE models.

## 1 Introduction

Most previous work on controlled text generation have concentrated on style transfer tasks: modifying sentences with regard to markers of sentiment, formality, affirmation/negation (John et al., 2019; Bao et al., 2019; Hu and Li, 2021; Vasilakes et al., 2022; Gu et al., 2022; Liu et al., 2023; Gu et al., 2023) (Figure 1 top). Disentanglement of language generative factors over Variational Autoencoder (VAE) spaces has been a key mechanism to deliver this type of generative control (John et al., 2019; Bao et al., 2019; Vasilakes et al., 2022). Recently, Zhang et al. (2022) demonstrated that a more general form of semantic control can be achieved in the latent space of Optimus (Li et al., 2020b), the first standard transformer-based VAE, where a BERT (Devlin et al., 2018) encoder and a GPT2 (Radford et al., 2019) decoder are connected within a VAE bottleneck. Using representations of conceptually dense explanatory sentences (Jansen et al., 2018b), they showed that sentences (e.g. animals require oxygen for survival), can be represented within a space which can be organised around the associations between predicate, arguments and their associated token content: ARG0-animals or VERBrequire, is geometrically resolved to a hypersolid over the latent space. Nevertheless, the ability to learn and control such separation is still limited as different semantic factors of the sentence are still overlapped and entangled in the latent space (e.g., V-eat and V-require in Figure 1 bottom left), indicating distributional sentence semantics cannot be currently localised and controlled from the perspective of formal semantics (i.e., predicate-argument structures, compositionality) (Marcus, 2003; Nefdt, 2020; Dankers et al., 2022).

![](images/7defc252f2c7bc63f3bb7e2e05a9e6125d4c2749b52f02d727166b8f4bdb0496.jpg)  
our objective: Granular semantic sentence control and manipulation  
Figure 1: Top: attribute space geometry. Bottom: general semantic geometry, where left: distributional semantic space of Optimus (Li et al., 2020b), right: our compositionality-induced semantic space where the geometrical location of sentence vectors can be located by the intersection of role-content clusters.

This work aims to improve the localisation and semantic control of latent sentence spaces, by delivering a model which can better separate and control syntactic-semantic features (e.g. predicateargument) and their associated lexical semantics content. This type of representation can provide the foundation to shorten the gap between deep latent semantics and formal linguistic representations (Gildea and Jurafsky, 2000; Banarescu et al., 2013; Mitchell, 2023), integrating the flexibility of distributional-neural models with the properties of linguistically grounded representations, facilitating both interpretability and generative control.

To deliver this type of semantic control within the distributional sentence space, following the methodological framework introduced by (Zhang et al., 2022), we target on improving the semantic separability of sentences by focusing on explanatory sentences <sup>1</sup>, rather than synthetic or style transfer datasets (Hupkes et al., 2020; Yanaka et al., 2021), in which the semantic structure ofsentences can be isolated and controlled. Inspired by the work of Esser et al. (2020), we integrate a flowbased invertible neural network (INN) (Dinh et al., 2014) as a plug-in control component to learn the bijective transformation between the distributional hidden space of the transformer-based language autoencoder (BERT-GPT2) and the smooth Gaussian space of the INN (Figure 2). Specifically, we first pre-train an autoencoder (AE) to learn sentence representations from the transformers’ latent spaces. Then, we freeze the AE weights and train the INN to map the AE representations to a Gaussian space. Since INN models define a bijective transformation, we can control the autoencoder generation by manipulating the INN latent spaces, which is more efficient and significantly less resource intensive than re-training a language AE end-to-end.

More importantly, we propose a supervised training strategy within the INN setting to learn a latent space with improved semantic separability, namely: the semantic role-content pairs and their associated clusters can be better separated over the latent space modelled by the INN (Section 4.1). In this case, we can improve localised control over the decoding process due to the reduction of overlapping (ambiguous) regions. A more separable and geometrically consistent sentence space can be then operated over to improve the generative control with support of geometric operators, such as interpolation (Bowman et al., 2016) (Section 4.2). The contributions of this work are summarised below:

1. We approach sentence disentanglement and generation control from the point of view of Argument Structure Theory (AST), bridging latent space features with a canonical, linguistics-informed, semantic representation of sentences. 2. We find that integrating a flow-based INN mechanism into a transformer-based language-AE architecture is an effective mechanism for transforming the hidden space of the autoencoder into a smooth Gaussian latent space for representing sentences. 3. We propose a supervised training strategy for INNs to learn a controllable semantic space with higher disentanglement and separability of semantic features, when compared to previous work. 4. Using this mechanism, we systematically employ geometrical data augmentation strategies to assist on sentence representation disentanglement.

Interpreting and controlling sentence generation from the perspective of the geometric manipulation of the latent space is still largely unexplored within NLP. To the best of our knowledge, this is the first work which focuses on the introduction of invertible NN-based mechanisms to support latent spaces with better separated argument structure/semantic features, allowing for a more universal form of sentence generation control.

## 2 Preliminaries

In this section, we first introduce the formal semantics and define the sentence representation model based on Argument Structure Theory (AST), linking with the associated disentanglement/generative factors and then proceed with the description of the proposed flow-based INN mechanism.

Controllability and interpretation in formal semantics. Formal semantics, which provides a canonical, granular, and rigid representation, have been investigated for thousands of years, with well established theoretical frameworks such as Montague Semantics (Dowty et al., 2012), Davidsonian Semantics (Davidson, 1967), Neo-Davidsonian Semantics (Lasersohn, 2016), etc. One typical characteristic of these formal semantics is the localisation or composition property. For example, in the sentence, animals require oxygenfor survival, the words are functionally combined into sentence semantics:

$$
\lambda x ( \mathrm { a n i m a l s } ( x ) \to \mathrm { r e q u i r e } ( x , \mathrm { o x y g e n } ) )\tag{1}
$$

Where x is the variable representing any entity within a logical structure. In this case, we can localise the sentence semantics by replacing x with birds, fishes, etc. This localised process indicates the interpretation in Cognitive Science (Smolensky, 2006; Lees, 1957). Disentanglement (Bengio, 2013) can potentially provide such localisation in the context of distributional latent representations, which has been widely investigated to localise image generation (Esser et al., 2020; Jeon et al., 2019; Liu et al., 2021). Therefore, we link several key notions—disentanglement, formal semantics, and localization—to investigate formal control and interpretability in language models.

Sentence semantic disentanglement. AST (Jackendoff, 1992; Levin, 1993; Rappaport Hovav and Levin, 2008) provides a model for representing sentence structure and meaning of sentences in terms of the interface between the their syntactic structure and the associated semantic roles of the arguments within those sentences. It delineates how verbs define the organisation of their associated arguments and the reflection of this organisation in a sentence’s syntactic realisation. AST abstracts sentences as predicate-argument structures, where the predicate $p$ (associated with the verb) has a set of associated arguments $a r g _ { i }$ where each argument has an associated positional component i and a thematic/semantic roles $r _ { i }$ the latter categorising the semantic functions of arguments in relation to the verb (e.g. agent, patient, theme, instrument). In the context of this work, the AST predicate-argument representation is associated with a lexical-semantic representation of the content $c _ { i }$ of the term $t _ { i }$

In this work, we simplify and particularise the relationship between the argument structure and the distributional lexical semantic representation as a role-content relation, where the structural syntactic/semantic relationship is defined by its shallow semantics, i.e. as the composition of the content of the terms, their position in the predicate-argument (PArg) structure (arg ) and their semantic roles (SRs) (r : pred, arg). Therefore, this work uses the notion of sentence semantic disentanglement as the cluster separation of the content under the PArg/SRs structure (the corresponding role in rolecontent), aiming to induce a latent space which geometrically encodes the AST structure, better disentangling and separating role-content clusters.

![](images/71e199a1b76e1d36e6f6c293df072076fc00259f9cbe5cc2a46c87c7b087ef0e.jpg)

Formally, a sentence s (e.g., see above) consists of a sequence PArgs/SRs and word content associations. Upon encoding in latent space, this can be described as:

$$
s e m ( s ) = \underbrace { t _ { 1 } ( c _ { 1 } , r _ { 1 } ) } _ { i . e . , A R G 0 - a n i m a l s } \oplus \cdots \oplus \underbrace { t _ { i } ( c _ { i } , r _ { i } ) } _ { P R P - s u r v i v a l }
$$

where $t _ { i } ( c _ { i } , r _ { i } ) = c _ { i } \otimes r _ { i }$ represents the semantics of term $t _ { i }$ with content $c _ { i }$ (i.e., animals) and SRL $r _ { i }$ (i.e., ARG0) in context $s ,$ : connects the meanings of words with their roles, using the compositionaldistributional semantics notation of (Smolensky and Legendre, 2006; Clark et al., 2008). : connects the lexical semantics (word content + structural role) to form the sentence semantics. This work applies distinct symbols aiming to emphasise the disentanglement aspects associated with the AST structure. If the sentence representation can be semantically disentangled under , the sem(s) can be decomposed into:

$$
s e m ( s ) = \{ t _ { 1 } ( c _ { 1 } , r _ { 1 } ) \} \oplus \cdots \oplus \{ t _ { i } ( c _ { i } , r _ { i } ) \}
$$

where each set represents a specific role-content cluster resolved to a hypersolid over the latent space, in this case, given a set of N sentences within the same predicate cluster $t ( c , r ) \ ( \mathrm { i . e . , } \ V -$ require) but different sem(s), those sentence vectors can represent $t ( c , r )$ features independently of other features (i.e., ARG0-animals), forming the $t ( c , r )$ cluster:

$$
\{ s e m ( s _ { 1 } ) , . . . , s e m ( s _ { N } ) \} = \{ t ( c , r ) \} _ { \times N } \oplus \{ . . . \}
$$

Therefore, we can evaluate the disentanglement (i.e., natural clustering property (Bengio, 2013)) of sentence semantics by evaluating the density within $\{ t ( c , r ) \}$ set(cluster) (classifier recall) and the separation between different $\{ t ( c , r ) \}$ set(clusters) (classifier accuracy) with downstream classifiers based on the manifold hypothesis $f o r$ classification (Rifai et al., 2011), rather than disentanglement metrics, which usually calculate the separation between latent dimensions, commonly used in the image domain (Higgins et al., 2017; Kim and Mnih, 2018; Chen et al., 2018; Ridgeway and Mozer, 2018). Next, we will introduce the INNbased mechanism which is used to learn this semantically disentangled space.

Invertible Neural Networks (INNs). Flowbased INNs (Dinh et al., 2014, 2016) are a class of neural networks that models the bijective mapping between the observation distribution $p ( x )$ and the latent distribution $p ( z )$ . We use $T$ to represent the forward mapping (from $p ( x )$ to $p ( z ) )$ and $T ^ { \prime }$ to represent the backward mapping (from $p ( z )$ to $p ( x ) )$ respectively. Unlike VAEs that approximate the prior distribution to multivariate Gaussian distributions, INNs exactly use multivariate Gaussian distributions. These are trained by the following objective function: $\begin{array} { r } { \mathcal { L } = - \mathbb { E } _ { x \sim p ( x ) } \Big [ T ( x ) \Big ] ^ { 2 } + \log \vert T ^ { \prime } ( x ) \vert } \end{array}$ where $T ( x )$ learns the transformation from x to $z \sim N ( 0 , 1 ) . \ | T ^ { \prime } ( x ) |$ is the determinant of the Jacobian for $T ( x )$ , which indicates the extent in which the transformation locally expands or contracts the space. The term log $\left| T ^ { \prime } ( x ) \right|$ ensures the integration of the probability density function to be one. The forward and reversed mapping can be implemented via the coupling layer (Dinh et al., 2014; Kingma and Dhariwal, 2018).

The rationale for choosing flow-based INNs lies on the fact that they learn the bijective transformation between the latent and observed spaces, which can be used to guide the autoencoder generation by manipulating the INN latent space, which is more efficient and has lower computational demand than re-training a language VAE. Besides, flow-based INNs that learn the prior distribution (i.e., Gaussian) exactly, can theoretically prevent the information loss from variational inference (ELBO) where the prior is approximated from posterior $P ( z | x )$

![](images/9b1b3be16b2fe9941600afc5c06d872c491867ecadaf69ed6881061f21b5186c.jpg)  
Figure 2: Transforming the representations of explanatory sentences from a language autoencoder (BERT-$\mathrm { G P T 2 } )$ , into asemantically separable latent space with the support of the INN mechanism, where a sentence representation can be decomposed into a predicateargument-level semantics (role-content).

## 3 Proposed Approach

We encode each sentence x with a frozen autoencoder (e.g., Bert-GPT2) and consider its sentence representation $E ( x )$ as the input of INNs (Figure 2). We propose two training strategies to map the hidden representations into the Gaussian space.

## 3.1 Training Strategy

Unsupervised INN. Firstly, we train the INNbased model unsupervised, which minimises the negative log-likelihood of the marginal distribution of latent representation $z = E ( x )$

$$
\mathcal { L } _ { \mathrm { u n s u p } } = - \mathbb { E } _ { x \sim p ( x ) } \Big [ T ( E ( x ) ) \Big ] ^ { 2 } + \log \big | T ^ { \prime } ( E ( x ) ) \big |
$$

As the minimisation leads to a bijective mapping between the distributed representation and the disentangled latent representation (multivariate Gaussian space), it allows for a more semantically consistent representation of the geometric (role-content) clustering properties of its latent space, allowing for a more consistent traversal and interpolation (Li et al., 2020b) over the sentence space (Figure 1).

Cluster-supervised INN. According to the findings of (Zhang et al., 2022), the content of the predicate-argument structure/semantic roles can be disentangled over the latent space approximated to multivariate Gaussian learned using the Language VAE setting. Using the same foundation, we train the INN component to learn the embeddings, by minimising the distance between points in the same role-content regions and maximising the distance between points in different regions, based on the explanation embeddings and their corresponding central point from the language autoencoder model.

For example, given a sentence "animals require foodfor survival" and its central vector of ARG0- animals, the training moves the sentence representation closer to the ARG0-animals region centre in the INN latent space. Specifically, during the calculation of the posterior, we replace the mean and variance of the standard Gaussian distribution by the centre point of its cluster and a hyper-parameter, which should be less than one, respectively. In this case, each role-content cluster in the latent space will be mapped to a space where each cluster will have its embeddings more densely and regularly distributed around its centre. The objective function can be described as follows:

$$
\begin{array} { r l r } {  { \mathcal { L } _ { \mathrm { s u p } } = - \mathbb { E } _ { x \sim p _ { c l u s t e r } } ( x ) \frac { \Big [ T ( E ( x ) ) - \mu _ { c l u s t e r } \Big ] ^ { 2 } } { 1 - \sigma ^ { 2 } } } } \\ & { } & { + \log \big | T ^ { \prime } ( E ( x ) ) \big | } \end{array}
$$

where $T ( E ( x ) )$ learns the transformation from x to $z \sim N ( \mu _ { c l u s t e r } , 1 - \sigma ^ { 2 } ) . \sigma ^ { 2 }$ is a parameter which can be empirically determined (in this particular context the optimal value was found to be 0.6). Additional details are provided in Appendix A.

## 3.2 Geometrical Data Augmentation

Data augmentation, which captures and augments a common or distinct feature across different samples, has been considered a common technique to assist disentanglement, such as in Graph (Li et al., 2021) and Image (Liu et al., 2022) representations, but is still limited in the context of sentence generation. In this work, we consider the vector arithmetic and traversal operators as a systematic mechanism to support data augmentation (via semantically controlled sentence generation) for each role-content cluster, described as follows:

$$
\begin{array} { r l } { ( 1 ) } & { \mathbf { v } = a v e r a g e ( E ^ { \prime } ( x _ { i } ) , E ^ { \prime } ( x _ { j } ) ) } \\ { ( 2 ) } & { \mathbf { v } _ { n e i g h b o u r } = \mathbf { v } [ i ] \sim N ( 0 , 1 ) _ { \forall i \in \{ 0 , \ldots , s i z e ( \mathbf { v } ) \} } } \\ { ( 3 ) } & { x _ { n e w } = D ^ { \prime } ( \mathbf { v } _ { n e i g h b o u r } ) } \end{array}
$$

where $x _ { k } \in S$ (original corpus), $E ^ { \prime }$ and $D ^ { \prime }$ are the encoder and decoder of Optimus fine-tuned over $S .$ average operation aims to modify the sentence while maintaining the target role-content common to both $x _ { i }$ and $x _ { j }$ (Zhang et al., 2022). The term $\mathbf { v } [ i ] \sim N ( 0 , 1 )$ is introduced to resample each dimension of v in the latent space (i.e., traverse its neighbour) and $x _ { n e w } = D ^ { \prime } ( \mathbf { v } _ { n e i g h b o u r } )$ generates a new sentence. Finally, we only keep the sentences holding the target role-content, where the

PArgs/SRs of x are annotated via the AllenNLP (Gardner et al., 2018) semantic role labeller. Table 1 lists randomly selected examples from augmented explanations. Full details and the supporting ablation study are provided in Appendices A and D.

<table><tr><td>Role-content</td><td>Augmented sentences</td></tr><tr><td rowspan="3">ARG0-animal</td><td>an animal requires energy to move</td></tr><tr><td>an animal requires shelter</td></tr><tr><td>an animal can use its body to breathe</td></tr><tr><td rowspan="3">ARG0-human</td><td>humans usually use gasoline</td></tr><tr><td>humans use coal to make food</td></tr><tr><td>humans depend on pollinators for survival</td></tr><tr><td rowspan="3">PRED-are</td><td>wheels are a part of a car</td></tr><tr><td>lenses are a part of eyeglasses</td></tr><tr><td>copper and zinc are two metals</td></tr><tr><td rowspan="3">PRED-mean</td><td>summit mean the top of the mountain</td></tr><tr><td>colder mean a decrease in heat energy</td></tr><tr><td>friction mean the product of a physical change</td></tr></table>

Table 1: Augmented explanations. We also provide more examples in Table 11 for qualitative evaluation.

## 4 Experiments

For the experiments, we start by focusing on the effect of the supervised INN mechanism to examine its impact on the sentence semantic separability of the distributional latent space defined in Section 2 (detailed in Section 4.1). Next, we examine the localised semantic generation control enabled by such semantic separability via latent interpolation (Section 4.2). Further details of the AutoEncoder model and dataset are provided in Appendix A.

## 4.1 Disentanglement Encoding Evaluation

We examine the latent space separability (i.e., natural clustering property (Bengio, 2013)) of our supervision approach on different predicateargument/semantic roles. In the context of this work, the thematic roles’ labels are not referred to control the generation. Instead, we use the predicate argument position markers, e.g. including ARG0, ARG1, PRED(V), where each category has a) four possible word contents (c<sub>i</sub>), or b) the same content (i.e., animal) with different argument/roles, including ARG0,1,2. We provide the reconstructed examples of INNs in Table 24.

Disentanglement between ARG0 clusters. For ARG0, we choose human, animal, plant, and something due to having the highest frequency in the original dataset, and evaluate model performance from two directions, including forward and backward mapping. Within forward mapping, we assess the disentanglement of the latent space of the INN model from two perspectives (visualisation and classification metrics). Figure 3 displays the distributions of four role-content clusters over the latent space. As we can observe, after the clustersupervised training strategy, the embeddings are more concentrated at the center of their cluster, and there is a clear boundary between clusters, indicating a better disentanglement when compared to Optimus and unsupervised INNs.

![](images/cdcb185df1bb5a09465d9905ddf40ca04506a5164dd10fa486f4fe96f5bc99ef.jpg)  
Figure 3: ARG0: t-SNE plot, different colour represents different content regions (blue: animal, green: human, red: plant, purple: something) (left: Optimus, middle: unsupervised, right: cluster supervised), same order for remaining visualizations. We also provide the PCA plot in Figure 10, both visualization shows that supervised embeddings concentrate on the respective cluster center.

We then quantitatively evaluate the disentanglement of ARG0-content clusters. We consider classification task metrics (accuracy, precision, recall, f1) as proxies for evaluating region separability, effectively testing cluster membership across different clusters. We choose a non-parametric downstream classifier (i.e., kNN) to quantitatively evaluate the separation of clusters and parametric downstream classifiers, including Naive Bayes (NB) and Support Vector Machine (SVM), to assess both separability and representation capability of latent sentence spaces (Rifai et al., 2011; Conneau et al., 2018). The configuration of the downstream classifiers are detailed in Appendix A.

As shown in table 2, all classifiers trained over supervised latent representations outperformed the unsupervised INN (U) and Optimus (O), indicating that the cluster-supervised approach leads to better disentanglement and representation. Moreover, (O) demonstrates superior performance compared to (U) for the KNN-based evaluation. However, it exhibits lower performance than (U) in NB and SVM. This suggests that the INN-AutoEncoder configuration can more effectively capture sentence semantics (from the point-of-view of AST+distributional content), in the context of a reconstruction task since the VAEs’ training process is prone to experiencing posterior collapse.

As for the evaluation of the backward mapping, we calculate the ratio of generated sentences that hold the same role-content as the inputs (henceforth called the invertibility ratio). We randomly selected 100 embeddings as inputs and showed the corresponding ratios in Table 3. We can observe that both unsupervised and supervised cases can achieve high invertibility ratios, indicating that the INN mechanism provides stable invertibility with or without cluster supervision.

<table><tr><td colspan="4">ARG0: disentanglement proxy metrics</td></tr><tr><td>classifier train accuracy precision recall f1 score</td><td></td><td></td><td></td></tr><tr><td rowspan="2">KNN C</td><td>0 0.972</td><td>0.973 0.972</td><td>0.972</td></tr><tr><td>U 0.938 0.979</td><td>0.938 0.979</td><td>0.938 0.938 0.979 0.979</td></tr><tr><td rowspan="3">NB</td><td>0 0.934</td><td>0.934 0.933</td><td>0.933</td></tr><tr><td>U 0.958</td><td>0.958</td><td>0.958 0.958</td></tr><tr><td>C 0.978</td><td>0.978</td><td>0.978 0.978</td></tr><tr><td rowspan="3">SVM</td><td>0 0.970</td><td>0.970</td><td>0.970 0.970</td></tr><tr><td>U</td><td>0.972 0.972</td><td>0.972 0.972</td></tr><tr><td>C 0.980</td><td>0.980</td><td>0.980 0.980</td></tr></table>

Table 2: Disentanglement of ARG0 between Optimus (O), unsupervised INN (U), and cluster-supervised INN (C) where KNN: k-neighbours, NB: naive bayes, SVM: support vector machine. The abbreviations are the same for the remaining tables. Cluster supervision displays consistent improvement with different classifiers.

<table><tr><td colspan="4">ARG0: invertibility ratio (backward: T&#x27;)</td></tr><tr><td>train</td><td>human</td><td>animal</td><td>plant</td><td>something</td></tr><tr><td>U</td><td>0.980</td><td>0.890</td><td>0.990</td><td>1.000</td></tr><tr><td>C</td><td>1.000</td><td>0.860</td><td>0.990</td><td>0.950</td></tr></table>

Table 3: Invertibility test for ARG0, Both INNs with AutoEncoder setup can achieve high ratios, indicating stable invertibility with or without cluster supervision.

Disentanglement between PRED clusters. Next, we analyze the disentanglement between predicate (PRED) clusters. As shown in Figure 4, although the disentanglement of PRED clusters is not as high as ARG0, the latent space with cluster supervision still performs better than both the unsupervised case and the Optimus model. In Table 4, the supervised INN model achieves better disentanglement, and both unsupervised and supervised could obtain a higher ratio. We also provide the experimental results of ARG1 disentanglement in Appendix B.

Disentanglement between ARG0,1,2 clusters. The experiments up to this point investigated the separation between the same pred-argument type but different content clusters. Next, we explore the separability of different pred-argument types with the same content. We thus focus on the animal cluster, and investigate the disentanglement between ARG0-animal, ARG1-animal, and ARG2-animal. As illustrated in Figure 5, the animal clusters with different pred-argument types can be separated after cluster-supervised training, which indicates that the INN model can capture the difference between the same content with different pred-argument type in the case of similar topic, indicating the INNbased approach could jointly learn separable embeddings w.r.t. role-content and content alone.

![](images/c59bbcfeb0fe0542f17b30ef12dff497f0047f4786947bc13dae10a55e98fc6d.jpg)  
Figure 4: PRED: t-SNE plot (blue: are, green: cause, red: is, purple: require). PCA plot is in Figure 12.

<table><tr><td colspan="5">PRED: disentanglement proxy metrics (forward: T)</td></tr><tr><td>classifier train accuracy precision recall f1 score</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">KNN</td><td>0</td><td>0.911</td><td>0.914</td><td>0.910 0.911</td></tr><tr><td>U</td><td>0.869</td><td>0.873</td><td>0.865 0.868</td></tr><tr><td>C</td><td>0.922</td><td>0.927</td><td>0.918 0.922</td></tr><tr><td rowspan="3">NB</td><td>0</td><td>0.865</td><td>0.866</td><td>0.866 0.865</td></tr><tr><td>U</td><td>0.873</td><td>0.874</td><td>0.871 0.872</td></tr><tr><td>C</td><td>0.903</td><td>0.903</td><td>0.902 0.903</td></tr><tr><td rowspan="3">SVM</td><td>0</td><td>0.902</td><td>0.902</td><td>0.903 0.902</td></tr><tr><td>U</td><td>0.905</td><td>0.906</td><td>0.902 0.904</td></tr><tr><td>C</td><td>0.910</td><td>0.912</td><td>0.909 0.910</td></tr></table>

Table 4: Forward evaluation for predicate clusters, the invertibility ratio and statistical significance test are provided in Table 14 and 18.

![](images/c877e8dc6a39fd9d3ccffa13f48beb356c82d5fb37c936855764a13eb2b023ba.jpg)  
Figure 5: Animal: t-SNE plot (blue: ARG0-animal, green: ARG1-animal, red: ARG2-animal), PCA plot is provided in Figure 13.

Table 5 and 15 show the disentanglement metrics and the invertibility ratio, respectively. Similarly to the previous experiment, the supervised case outperforms both the unsupervised and the Optimus models. Both INNs can achieve an invertibility ratio of at least 90%.

<table><tr><td colspan="2">Animal: disentanglement metrics (f1 score)</td></tr><tr><td>train KNN NB</td><td>SVM</td></tr><tr><td>0 0.960 0.928</td><td>0.946</td></tr><tr><td>U 0.9580.930</td><td>0.947</td></tr><tr><td>C 0.967 0.937</td><td>0.950</td></tr></table>

Table 5: Forward evaluation for Animal, we only show f1 since the same value across different metrics. Results indicate improved separation across role clusters.

## 4.2 Disentanglement Decoding Evaluation

Finally, we evaluate the disentangled sentence geometry from the perspective of sentence generation. We specifically focus on linear interpolation as it can provide more efficient traversal between sentences and clusters than other traversal approaches (e.g., Ornstein-Uhlenbeck), commonly used in the NLP domain (Li et al., 2020b) and in the evaluation of disentanglement (Bengio, 2013).

Interpolation localisation. Firstly, we evaluate the localisation of latent interpolation that interpolates a path $z _ { t } = z _ { 1 } \cdot ( 1 - t ) + z _ { 2 } \cdot t$ with t increased from 0 to 1 by a step size of 0.1, where z<sub>1</sub> and z<sub>2</sub> represent the latent representations of source and target sentences. As a result, 9 sentences are generated on each interpolation step. On a latent space with better token-level role-content separation, given two sentences with the same rolecontent as endpoints, we can observe that the intermediate sentence can hold the same role-content during interpolation.

In terms of qualitative evaluation, Table 6 provides the interpolation paths of cluster-supervised INN and Optimus, as for Optimus, we can observe that the intermediate explanations could transition smoothly from source to target for argument. However, the predicate is more abruptly changed, indicating lower predicate-content disentanglement (e.g., predicate-require and predicate-eat). Instead, the supervised INN can fix the predicate-require during interpolation, indicating better separability between different predicate-content results in better generation control. More examples are provided in Table 22 and 23. We then quantitatively evaluate the localisation of interpolation. We randomly select 200 sentence pairs from the dataset holding the same role-content and report the ratio of intermediate sentences with the same role-content as inputs. As illustrated in Figure 6, the intermediate sentences from the supervised INN can better hold the same role-content as inputs, especially for predicate which usually has a lower effect on distributional sentence semantics (Zhang et al., 2022), indicating that our supervision can lead to better latent space separability and generation control.

![](images/3cd98f112b4725983543f29fff4d2cbd434348d9c949702bac343d479cc46aa8.jpg)

Table 6: Interpolation examples, indicating the clustersupervised INN can provide better localised/symbolic semantic control. We also report the interpolations of AutoEncoder and unsupervised INN in Table 21.  
![](images/7d275bfddfbf12d6a0de9707ba445d9263ccdac8ae48d1d39691269c5331698f.jpg)  
Figure 6: Interpolation control evaluation, we can observe that supervised INN with better semantic separability can lead to better localised semantic control.

Interpolation smoothness. Moreover, we quantitatively evaluate the latent space geometry via interpolation smoothness metrics (IS, Zhang et al. (2024)), which calculates the ratio between the ideal semantic distance (i.e., the aligned semantic distance between source and target sentences) and the actual semantic distance (i.e., the sum of semantic distance between adjacent sentences during interpolation). A higher ratio indicates that the actual path aligns better with the ideal path, suggesting better semantic-geometric properties. The metric is defined as:

<table><tr><td>Evaluation Metrics</td></tr><tr><td>avg IS↑ max IS↑ min IS↑ DAE (Vincent et al., 2008) 0.144 0.330 0.055</td></tr><tr><td>AAE (Makhzani et al., 2015) 0.142 0.284 0.054</td></tr><tr><td>LAAE(Rubenstein et al., 2018) 0.172 0.347 0.056</td></tr><tr><td>DAAE (Shen et al., 2020) 0.055 0.061 0.023</td></tr><tr><td>β-VAE (Higgins et al., 2016) 0.198 0.379 0.041</td></tr><tr><td>AdaVAE (Tu et al., 2022) 0.085 0.105 0.050</td></tr><tr><td>Della (Hu et al., 2022) 0.253 0.416 0.155</td></tr><tr><td>Optimus (Li et al., 2020b) 0.220 0.525 0.130</td></tr><tr><td>AutoEncoder (Bert-GPT2) 0.259 0.585 0.165</td></tr><tr><td>INN (U) (our) 0.251 0.540 0.159</td></tr><tr><td>INN (C) (our) 0.282 0.607 0.206</td></tr></table>

Table 7: Geometrical examination via IS metric.

$$
\mathbf { I S } = \mathbb { E } _ { ( s _ { 0 } , \ldots , s _ { T } ) \sim P } { \frac { \delta ( \mathrm { a l i g n } ( s _ { 0 } , s _ { T } ) ) } { \sum _ { t = 0 } ^ { T } \delta ( \mathrm { a l i g n } ( s _ { t } , s _ { t + 0 . 1 } ) ) } }
$$

where $s _ { 0 } , . . . , s _ { T }$ is the sequence of sentences during interpolation, δ and align are sentence similarity and alignment functions, respectively, which are performed via Word Mover’s Distance (Zhao et al., 2019). We choose the standard language VAE baselines (i.e., the prior is the std. Gaussian distribution). Their implementation details are provided in Appendix A. We randomly sample 200 sentence pairs and report the IS metric. As illustrated in Table 7, our model can deliver smoother interpolations comparatively to the baselines, indicating semantic disentanglement can lead to better latent space geometry.

## 5 Related Work

Sentence representation. Sentence representations are usually trained in supervised (Conneau et al., 2017; Reimers and Gurevych, 2019), constrastive (Giorgi et al., 2021; Yan et al., 2021; Chuang et al., 2022), or generation-oriented (Wang et al., 2021; Wu and Zhao, 2022; Chuang et al., 2022) fashion. Recent work (Huang et al., 2023) explored the compositional sentence representation for improved explainability and generation. However, these works still lack the emphasis on the geometric interpretation and control of the underlying sentence space, which this work focused on.

Sentence disentanglement. In the Natural Language Generation domain, most previous investigations explored the disentanglement between two specific linguistic perspectives, such as sentimentcontent (John et al., 2019; Li et al., 2022), semanticsyntax (Bao et al., 2019), and negation-uncertainty (Vasilakes et al., 2022), or syntactic disentanglement (Mercatali and Freitas, 2021; Felhi et al., 2022). In this work, we provide a formalgeometrical lens, with the support of argument structures as a sentence representation model, for sentence disentanglement targeting for localised semantic control. This work is the first integration of flow-based INN mechanisms to improve disentanglement, separation and semantic control of sentence spaces.

INNs in NLP. ¸Sahin and Gurevych (2020) concentrate on modelling morphological inflection and lemmatization tasks, utilizing INN to learn a bijective transformation between the word surface and its morphemes. Li et al. (2020a) proposed BERT-flow, transforming sentences from a BERT sentence space to a standard Gaussian space. Ding and Gimpel (2021) deployed flow-based INN to enrich VAE prior distribution, while Gu et al. (2023) use flow mechanisms to control attributes in style transfer tasks. This work focused on semantic separability, geometrical operations and control over the distributed representation of sentences. Moreover, this work is the first to explore geometrical data augmentation to support semantic disentanglement.

## 6 Conclusions and Future Work

This work focused on an INN-based mechanism to support better disentangled and separated latent sentence spaces over language autoencoders. By aligning the predicate-argument structure of sentences to the latent representations, we aimed to build a bridge between the formal and distributional semantics perspectives for sentence representation. We define the sentence semantic disentanglement from the perspective of formal semantics, aligning the predicate-argument structure to disentanglement and cluster separation properties, and exploiting the invertibility and bijection properties of INNs to facilitate such alignment. Experimental results indicate that the invertibility mechanisms can transform the distributed hidden space of an autoencoder into a latent space where AST-level syntactic and semantic transformations can be localised, interpolated and controlled. Secondly, we propose a supervision approach, which leads to an improved disentangled and separated space. This property can facilitate localised interpolation control. Lastly, we utilise these geometric properties to support a semantically controlled data augmentation to assist the disentanglement process.

Since our work connects distributional and formal semantics via disentanglement, one future direction is to explore the safety and control of the formal semantic properties of Large Language Models. Besides, recent work (Liu et al., 2023) revealed that distinct factors can be composed by modelling the moving of latent vectors via ordinary differential equations, which can be adapted to sentence representations to deliver more complex sentence transformations within the latent space.

## 7 Limitations

This work focused on the disentangled sentence representations geometry to deliver localised/semantic/formal semantic control. While this work is motivated by providing more localised distributed representations, which can positively impact the safety and coherence of generative models, few scoping observations need to be established: 1. The specific safety guarantees of these models are not fully established. 2. While the language autoencoder with unsupervised INN exhibit a distinct learning pattern with regard to semantic distribution, further understanding is required in terms of information bottleneck properties (Saxe et al., 2018) and on the semantic distribution of unsupervised INNs in language modelling tasks. 3. Furthermore, this study exclusively focused on a corpus of sentences which are conceptually dense ((Dalvi et al., 2021)). The exploration of its performance on other types of sentences, including sentences with complex clausal-phrasal constructions, or sentences with non-compositional idioms, is yet to be undertaken.

## Acknowledgements

This work was partially funded by the EPSRC grant EP/T026995/1 entitled “EnnCore: End-to-End Conceptual Guarding of Neural Architectures” under Security for all in an AI enabled society, by the CRUK National Biomarker Centre, and supported by the Manchester Experimental Cancer Medicine Centre and the NIHR Manchester Biomedical Research Centre.

## References

Lynton Ardizzone, Till Bungert, Felix Draxler, Ullrich Köthe, Jakob Kruse, Robert Schmier, and Peter Sorrenson. 2018-2022. Framework for Easily Invertible Architectures (FrEIA).

Laura Banarescu, Claire Bonial, Shu Cai, Madalina Georgescu, Kira Griffitt, Ulf Hermjakob, Kevin Knight, Philipp Koehn, Martha Palmer, and Nathan Schneider. 2013. Abstract meaning representation for sembanking. In Proceedings of the 7th linguistic annotation workshop and interoperability with discourse, pages 178–186.

Yu Bao, Hao Zhou, Shujian Huang, Lei Li, Lili Mou, Olga Vechtomova, Xinyu Dai, and Jiajun Chen. 2019. Generating sentences from disentangled syntactic and semantic spaces. In Proceedings of the 57th Annual Meeting ofthe Associationfor Computational Linguistics, pages 6008–6019.

Benjamin Bengfort and Rebecca Bilbro. 2019. Yellowbrick: Visualizing the Scikit-Learn Model Selection Process. 4(35).

Yoshua Bengio. 2013. Deep learning of representations: Looking forward. In International conference on statistical language and speech processing, pages 1–37. Springer.

Samuel Bowman, Luke Vilnis, Oriol Vinyals, Andrew Dai, Rafal Jozefowicz, and Samy Bengio. 2016. Generating sentences from a continuous space. In Proceedings ofThe 20th SIGNLL Conference on Computational Natural Language Learning, pages 10–21.

Ricky TQ Chen, Xuechen Li, Roger Grosse, and David Duvenaud. 2018. Isolating sources of disentanglement in vaes. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, pages 2615–2625.

Yung-Sung Chuang, Rumen Dangovski, Hongyin Luo, Yang Zhang, Shiyu Chang, Marin Soljacic, Shang-Wen Li, Scott Yih, Yoon Kim, and James Glass. 2022. DiffCSE: Difference-based contrastive learning for sentence embeddings. In Proceedings of the 2022 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, pages 4207–4218, Seattle, United States. Association for Computational Linguistics.

Stephen Clark, Bob Coecke, and Mehrnoosh Sadrzadeh. 2008. A compositional distributional model of meaning. In Proceedings of the Second Quantum Interaction Symposium (QI-2008), pages 133–140. Oxford.

Alexis Conneau, Douwe Kiela, Holger Schwenk, Loïc Barrault, and Antoine Bordes. 2017. Supervised learning of universal sentence representations from natural language inference data. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 670–680, Copenhagen, Denmark. Association for Computational Linguistics.

Alexis Conneau, German Kruszewski, Guillaume Lample, Loïc Barrault, and Marco Baroni. 2018. What you can cram into a single \$&!#\* vector: Probing sentence embeddings for linguistic properties. In Proceedings of the 56th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 2126–2136, Melbourne, Australia. Association for Computational Linguistics.

Bhavana Dalvi, Peter Jansen, Oyvind Tafjord, Zhengnan Xie, Hannah Smith, Leighanna Pipatanangkura, and Peter Clark. 2021. Explaining answers with entailment trees. arXiv preprint arXiv:2104.08661.

Verna Dankers, Elia Bruni, and Dieuwke Hupkes. 2022. The paradox of the compositionality of natural language: A neural machine translation case study. In Proceedings of the 60th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 4154–4175, Dublin, Ireland. Association for Computational Linguistics.

Donald Davidson. 1967. The logical form of action sentences.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

Xiaoan Ding and Kevin Gimpel. 2021. FlowPrior: Learning expressive priors for latent variable sentence models. In Proceedings ofthe 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3242–3258, Online. Association for Computational Linguistics.

Laurent Dinh, David Krueger, and Yoshua Bengio. 2014. Nice: Non-linear independent components estimation. arXiv preprint arXiv:1410.8516.

Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. 2016. Density estimation using real nvp. arXiv preprint arXiv:1605.08803.

David R Dowty, Robert Wall, and Stanley Peters. 2012. Introduction to Montague semantics, volume 11. Springer Science & Business Media.

Rotem Dror, Gili Baumer, Segev Shlomov, and Roi Reichart. 2018. The hitchhiker’s guide to testing statistical significance in natural language processing. In Proceedings of the 56th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1383–1392. Association for Computational Linguistics.

Patrick Esser, Robin Rombach, and Bjorn Ommer. 2020. A disentangling invertible interpretation network for explaining latent representations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9223–9232.

Ghazi Felhi, Joseph Le Roux, and Djamé Seddah. 2022. Towards unsupervised content disentanglement in sentence representations via syntactic roles. arXiv preprint arXiv:2206.11184.

Matt Gardner, Joel Grus, Mark Neumann, Oyvind Tafjord, Pradeep Dasigi, Nelson Liu, Matthew Peters, Michael Schmitz, and Luke Zettlemoyer. 2018. Allennlp: A deep semantic natural language processing platform. arXiv preprint arXiv:1803.07640.

Daniel Gildea and Daniel Jurafsky. 2000. Automatic labeling of semantic roles. In Proceedings of the 38th Annual Meeting on Association for Computational Linguistics, ACL ’00, page 512–520, USA. Association for Computational Linguistics.

John Giorgi, Osvald Nitski, Bo Wang, and Gary Bader. 2021. DeCLUTR: Deep contrastive learning for unsupervised textual representations. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 879–895, Online. Association for Computational Linguistics.

Yuxuan Gu, Xiaocheng Feng, Sicheng Ma, Lingyuan Zhang, Heng Gong, and Bing Qin. 2022. A distributional lens for multi-aspect controllable text generation. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 1023–1043, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Yuxuan Gu, Xiaocheng Feng, Sicheng Ma, Lingyuan Zhang, Heng Gong, Weihong Zhong, and Bing Qin. 2023. Controllable text generation via probability density estimation in the latent space. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12590–12616, Toronto, Canada. Association for Computational Linguistics.

Irina Higgins, Loïc Matthey, Arka Pal, Christopher P. Burgess, Xavier Glorot, Matthew M. Botvinick, Shakir Mohamed, and Alexander Lerchner. 2016. beta-vae: Learning basic visual concepts with a constrained variational framework. In International Conference on Learning Representations.

Irina Higgins, Loïc Matthey, Arka Pal, Christopher P. Burgess, Xavier Glorot, Matthew M. Botvinick, Shakir Mohamed, and Alexander Lerchner. 2017. beta-vae: Learning basic visual concepts with a constrained variational framework. In ICLR.

Jinyi Hu, Xiaoyuan Yi, Wenhao Li, Maosong Sun, and Xing Xie. 2022. Fuse it more deeply! a variational transformer with layer-wise latent variable inference for text generation. In Proceedings ofthe 2022 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, pages 697–716, Seattle, United States. Association for Computational Linguistics.

Zhiting Hu and Li Erran Li. 2021. A causal lens for controllable text generation. Advances in Neural Information Processing Systems, 34:24941–24955.

James Y. Huang, Wenlin Yao, Kaiqiang Song, Hongming Zhang, Muhao Chen, and Dong Yu. 2023. Bridging continuous and discrete spaces: Interpretable sentence representation learning via compositional operations. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 14584–14595, Singapore. Association for Computational Linguistics.

Dieuwke Hupkes, Verna Dankers, Mathijs Mul, and Elia Bruni. 2020. Compositionality decomposed: How do neural networks generalise? Journal ofArtificial Intelligence Research, 67:757–795.

Ray S Jackendoff. 1992. Semantic structures, volume 18. MIT press.

Peter Jansen, Elizabeth Wainwright, Steven Marmorstein, and Clayton Morrison. 2018a. WorldTree: A corpus of explanation graphs for elementary science questions supporting multi-hop inference. In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), Miyazaki, Japan. European Language Resources Association (ELRA).

Peter A Jansen, Elizabeth Wainwright, Steven Marmorstein, and Clayton T Morrison. 2018b. Worldtree: A corpus of explanation graphs for elementary science questions supporting multi-hop inference. arXiv preprint arXiv:1802.03052.

Giyoung Jeon, Haedong Jeong, and Jaesik Choi. 2019. An efficient explorative sampling considering the generative boundaries of deep generative neural networks.

Vineet John, Lili Mou, Hareesh Bahuleyan, and Olga Vechtomova. 2019. Disentangled representation learning for non-parallel text style transfer. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 424–434.

Hyunjik Kim and Andriy Mnih. 2018. Disentangling by factorising. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 2649–2658. PMLR.

Durk P Kingma and Prafulla Dhariwal. 2018. Glow: Generative flow with invertible 1x1 convolutions. Advances in neural information processing systems, 31.

Peter Lasersohn. 2016. A semantics for groups and events. Routledge.

Robert B Lees. 1957. Syntactic structures.

Beth Levin. 1993. English verb classes and alternations: A preliminary investigation. University of Chicago press.

Bohan Li, Hao Zhou, Junxian He, Mingxuan Wang, Yiming Yang, and Lei Li. 2020a. On the sentence embeddings from pre-trained language models. arXiv preprint arXiv:2011.05864.

Chunyuan Li, Xiang Gao, Yuan Li, Baolin Peng, Xiujun Li, Yizhe Zhang, and Jianfeng Gao. 2020b. Optimus: Organizing sentences via pre-trained modeling of a latent space. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4678–4699.

Haoyang Li, Xin Wang, Ziwei Zhang, Zehuan Yuan, Hang Li, and Wenwu Zhu. 2021. Disentangled contrastive learning on graphs. Advances in Neural Information Processing Systems, 34:21872–21884.

Zhuang Li, Lizhen Qu, Qiongkai Xu, Tongtong Wu, Tianyang Zhan, and Gholamreza Haffari. 2022. Variational autoencoder with disentanglement priors for low-resource task-specific natural language generation. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 10335–10356, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Guangyi Liu, Zeyu Feng, Yuan Gao, Zichao Yang, Xiaodan Liang, Junwei Bao, Xiaodong He, Shuguang Cui, Zhen Li, and Zhiting Hu. 2023. Composable text controls in latent space with ODEs. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 16543–16570, Singapore. Association for Computational Linguistics.

Xiao Liu, Pedro Sanchez, Spyridon Thermos, Alison Q O’Neil, and Sotirios A Tsaftaris. 2022. Learning disentangled representations in the imaging domain. Medical Image Analysis, 80:102516.

Yahui Liu, Enver Sangineto, Yajing Chen, Linchao Bao, Haoxian Zhang, Nicu Sebe, Bruno Lepri, Wei Wang, and Marco De Nadai. 2021. Smoothing the disentangled latent style space for unsupervised image-toimage translation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10785–10794.

Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101.

Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, Ian Goodfellow, and Brendan Frey. 2015. Adversarial autoencoders. arXiv preprint arXiv:1511.05644.

Gary F Marcus. 2003. The algebraic mind: Integrating connectionism and cognitive science. MIT press.

Giangiacomo Mercatali and André Freitas. 2021. Disentangling generative factors in natural language with discrete variational autoencoders. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 3547–3556.

Nathan Michlo, Richard Klein, and Steven James. 2023. Overlooked implications of the reconstruction loss for vae disentanglement. In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, pages 4073–4081.

Melanie Mitchell. 2023. How do we know how smart ai systems are?

Ryan M. Nefdt. 2020. A puzzle concerning compositionality in machines. Minds and Machines, 30(1):47–75.

Hiroki Ouchi, Hiroyuki Shindo, and Yuji Matsumoto. 2017. Neural modeling of multi-predicate interactions for Japanese predicate argument structure analysis. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1591–1600, Vancouver, Canada. Association for Computational Linguistics.

F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. 2011. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825–2830.

Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners.

Malka Rappaport Hovav and Beth Levin. 2008. The english dative alternation: The case for verb sensitivityl. Journal of linguistics, 44(1):129–167.

Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. In Conference on Empirical Methods in Natural Language Processing.

Karl Ridgeway and Michael C Mozer. 2018. Learning deep disentangled embeddings with the f-statistic loss. In Proceedings ofthe 32nd International Conference on Neural Information Processing Systems, pages 185–194.

Salah Rifai, Yann Dauphin, Pascal Vincent, Yoshua Bengio, and Xavier Muller. 2011. The manifold tangent classifier. In Neural Information Processing Systems.

Paul K Rubenstein, Bernhard Schoelkopf, and Ilya Tolstikhin. 2018. On the latent space of wasserstein auto-encoders. arXiv preprint arXiv:1802.03761.

Gözde Gül ¸Sahin and Iryna Gurevych. 2020. Two birds with one stone: Investigating invertible neural networks for inverse problems in morphology. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 34, pages 7814–7821.

Andrew Michael Saxe, Yamini Bansal, Joel Dapello, Madhu Advani, Artemy Kolchinsky, Brendan Daniel

Tracey, and David Daniel Cox. 2018. On the information bottleneck theory of deep learning. In International Conference on Learning Representations.

Tianxiao Shen, Jonas Mueller, Regina Barzilay, and Tommi Jaakkola. 2020. Educating text autoencoders: Latent representation guidance via denoising. In International Conference on Machine Learning, pages 8719–8729. PMLR.

Jonathon Shlens. 2014. A tutorial on principal component analysis. arXiv preprint arXiv:1404.1100.

Paul Smolensky. 2006. Harmony in linguistic cognition. Cognitive science, 30(5):779–801.

Paul Smolensky and Géraldine Legendre. 2006. The harmonic mind: From neural computation to optimality-theoretic grammar. Vol. 1, Cognitive architecture. MIT.

Haoqin Tu, Zhongliang Yang, Jinshuai Yang, and Yongfeng Huang. 2022. Adavae: Exploring adaptive gpt-2s in variational auto-encoders for language modeling. arXiv preprint arXiv:2205.05862.

Jake Vasilakes, Chrysoula Zerva, Makoto Miwa, and Sophia Ananiadou. 2022. Learning disentangled representations of negation and uncertainty. In Proceedings ofthe 60th Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 8380–8397, Dublin, Ireland. Association for Computational Linguistics.

Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. 2008. Extracting and composing robust features with denoising autoencoders. In Proceedings ofthe 25th International Conference on Machine Learning, ICML ’08, page 1096–1103, New York, NY, USA. Association for Computing Machinery.

Kexin Wang, Nils Reimers, and Iryna Gurevych. 2021. Tsdae: Using transformer-based sequential denoising auto-encoder for unsupervised sentence embedding learning. arXiv preprint arXiv:2104.06979.

Bohong Wu and Hai Zhao. 2022. Sentence representation learning with generative objective rather than contrastive objective. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 3356–3368, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Shuangzhi Wu, Dongdong Zhang, Nan Yang, Mu Li, and Ming Zhou. 2017. Sequence-to-dependency neural machine translation. In Proceedings ofthe 55th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 698–707, Vancouver, Canada. Association for Computational Linguistics.

Yuanmeng Yan, Rumei Li, Sirui Wang, Fuzheng Zhang, Wei Wu, and Weiran Xu. 2021. ConSERT: A contrastive framework for self-supervised sentence representation transfer. In Proceedings ofthe 59th Annual

Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 5065–5075, Online. Association for Computational Linguistics.

Hitomi Yanaka, Koji Mineshima, and Kentaro Inui. 2021. Sygns: A systematic generalization testbed based on natural language semantics. arXiv preprint arXiv:2106.01077.

Yingji Zhang, Danilo S Carvalho, Ian Pratt-Hartmann, and André Freitas. 2022. Quasi-symbolic explanatory nli via disentanglement: A geometrical examination. arXiv preprint arXiv:2210.06230.

Yingji Zhang, Danilo S. Carvalho, Marco Valentino, Ian Pratt-Hartmann, and Andre Freitas. 2024. Improving semantic control in discrete latent spaces with transformer quantized variational autoencoders.

Wei Zhao, Maxime Peyrard, Fei Liu, Yang Gao, Christian M. Meyer, and Steffen Eger. 2019. MoverScore: Text generation evaluating with contextualized embeddings and earth mover distance. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 563–578, Hong Kong, China. Association for Computational Linguistics.

## A Experiment setting

Dataset. Table 8 displays the statistical information of the datasets used in the experiment. The data of the two datasets partially overlap, so only the unique explanations are selected as the experimental data. Table 9 illustrates the semantic,

<table><tr><td>Corpus</td><td>Num data.</td><td>Avg. length</td></tr><tr><td>WorldTree (Jansen et al., 2018a)</td><td>11430</td><td>8.65</td></tr><tr><td>EntailmentBank (Dalvi et al., 2021)</td><td>5134</td><td>10.35</td></tr></table>

Table 8: Statistics from explanations datasets.

structure, and topic information of explanatory sentences over the latent space.

The rationale for choosing explanatory sentences is that they are designed for formal/localised/symbolic semantic inference task in natural language form, which provides a semantically complex and yet controlled experimental setting, containing a both well-scoped and diverse set of target concepts and sentence structures, providing a semantically challenging yet sufficiently well-scoped scenario to evaluate the syntactic and semantic organisation of the space. More details about semantic structure and lexical information are provided in Table 9 and 10.

Data Augmentation. Algorithm 1 illustrates the detailed process of data augmentation. The key aspect of data augmentation is to keep the data distribution unchanged while increasing the size of the dataset. Therefore, during traversal, we only sample the value whose probability density is between 0.495 and 0.505. In other words, for each original explanation, we only traverse its close neighbours over the latent space. We increased the number of explanations in each role-content cluster to 3000 and kept the balance of each role-content category. We provide more qualitative examples in Table 11. Moreover, we visualise latent semantic distribution before and after augmentation in Figure 7. As we can observe, the data augmentation can maintain the semantic distribution unchanged. For example, PRED-is (red colour in the right column) is widely distributed over the latent space before and after augmentation. ARG0-something (purple colour in the left column) is far from other clusters with or without data augmentation in latent space.

Downstream Classifier. In this experiment, we apply three downstream classifiers, including nonparametric classifier: k-nearest neighbours (KNN) and parametric classifiers: Naive Bayes (NB) and

Algorithm 1 Data Augmentation   
Define: R as the role set (ARG0, PRED, ...).   
Define: C as the content set (vocabulary).   
Define: S as the explanation corpus (sentences).   
Define: $s = [ ( c _ { 1 } , r _ { 1 } ) , . . . , ( c _ { i } , r _ { i } ) ] \ \in \ S , \ c _ { i } \ \in$   
$C , r _ { i } \in R$ as a sentence.   
Define: $( c _ { t } , r _ { t } ) \in ( 0 , T ] \times [ c _ { t } \in R , c _ { t } \in C$ as the   
target role-content (e.g., ARG1-animal).   
Define: $S _ { t } = \forall s \in S \quad | \quad \exists ( c _ { k } , r _ { k } ) = ( c _ { t } , r _ { t } )$   
as the set of sentences with the target role  
content.   
Define: $E ( s ) : S \to \mathbb { R } ^ { n }$ as encoder (embed  
ding) function.   
Define: $D ( v e c ) : \mathbb { R } ^ { n }  S$ as the explanation   
decoded from Decoder D.   
Define: L: list for keeping augmented sentences.   
Define: $S R L e r ( s )$ : semantic role label annota  
tor for s.   
for all $( s _ { i } , s _ { j } ) \in S _ { t } , \ s _ { i } \neq s _ { j }$ do   
vec = average $( E ( s _ { i } ) , E ( s _ { j } ) )$   
for all $v e c [ i ] \in$ vec do   
$v e c [ i ] = N ( 0 , 1 )$ # neighbour traversal   
$s _ { n } = D ($ (vec) # new sentence   
if $s _ { n } \notin L \mathop { \bf A N D } _ { \bf \phi } R \in S R L e r ( s _ { n } )$ then   
put $s _ { n }$ in L.   
end if   
end for   
end for

<table><tr><td rowspan=1 colspan=1>Cluster</td><td rowspan=1 colspan=1>Theme, Pattern, and Explanatory sentences</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>Theme: physics and chemistry. Pattern: if then and as. E.g., if a substance is mixed with another substance thenthose substances will undergo physical change.</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Theme: country, astronomy, and weather. E.g., new york state is on earth</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>Theme: physics and chemistry. Pattern: is a kind of. E.g., light is a kind of wave.</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>Theme: biology. E.g., a mother births offspring.</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>Theme: synonym for verb. Pattern: means and is similar to. E.g., to report means to show.</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>Theme: astronomy. E.g., the solar system contains asteroids.</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>Theme: animal/plant. Pattern: is a kind of. E.g., a seed is a part of a plant</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>Theme: item. E.g., a telephone is a kind of electrical device for communication</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>Theme: synonym for life. Pattern: means and is similar to. E.g., shape is a kind of characteristic.</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>Theme: geography. Pattern: is a kind of. E.g., a mountain is a kind of environment.</td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>Theme: animal and plant. Pattern: if then and as. E.g., if a habitat is removed then that habitat is destroyed</td></tr><tr><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>Theme: scientific knowledge. Pattern: (;), number and /. E.g., freezing point is a property of a (substance ;material).</td></tr><tr><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>Theme: item. Pattern: is a kind of object. E.g., a paper is a kind of object.</td></tr><tr><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>Theme: chemistry and astronomy. E.g., oxygen gas is made of only oxygen element.</td></tr><tr><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>Theme: general about science. Pattern: (;). E.g., seed dispersal has a positive impact on (a plant ; a plant &#x27;sreproduction).</td></tr><tr><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>Theme: item. Pattern: is a kind of. E.g., fertilizer is a kind of substance.</td></tr><tr><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>Theme: physics and chemistry. Pattern: (;). E.g., the melting point of oxygen is -3618f ; -2188c ; 544k.</td></tr><tr><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>Theme: animal. E.g., squirrels live in forests.</td></tr><tr><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>Theme: nature. E.g., warm ocean currents move to cooler ocean regions by convection.</td></tr><tr><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>Theme: life. E.g., pond water contains microscopic living organisms.</td></tr></table>

Table 9: Semantic, structure, topic information of explanatory sentences, where the cluster is the categories of k-means classifier.

![](images/215e3608380343b7ecc61828b980feb68b62c74192dc53f2097cbd46bc20a1cc.jpg)  
Figure 7: t-SNE plot for Data augmentation (top: original dataset distribution, bottom: augmented dataset distribution), (left: ARG0-animal(blue), human(green), plant(red), something(purple); middle: ARG1-food(blue), oxygen(green), sun(red), water(purple); right: PRED-are(blue), cause(green), is(red), require(purple).

Support Vector Machine (SVM), to evaluate the separability of latent representation. Those classifiers and classification metrics are implemented based on scikit-learn package (Pedregosa et al., 2011) with default hyper-parameters. We train those classifiers on the training set ( 60%) and evaluate them on the test set ( 40%). For multiclass classification, we set macro for precision, recall, and f1 since macro-averaged metric for each class is calculated independently, and then the average is taken, which ensures that the performance of the model in each class contributes equally to the final metric, regardless of the class size.

Visualizer. In this experiment, we implement t-SNE and PCA visualisation based on Yellowbrick library (Bengfort and Bilbro, 2019)<sup>2</sup>. We empirically set decompose\_by = 4 for all cases. However, we found no significant difference between different decompose\_by parameters.

Baselines for Interpolation Smoothness. In the experiment, we implement five LSTM-based autoencoders, including denoising AE (Vincent et al. (2008), DAE), β-VAE (Higgins et al., 2016), adversarial AE (Makhzani et al. (2015), AAE), label adversarial AE (Rubenstein et al. (2018), LAAE), and denoising adversarial autoencoder (Shen et al. (2020), DAAE). Their implementation relies on the open-source codebase available at the URL <sup>3</sup>. As for transformer-based VAEs, we implement Optimus (Li et al., 2020b), AdaVAE (Tu et al., 2022)<sup>4</sup>,

<table><tr><td>Semantic Tags</td><td>Prop. %</td><td>Description and Example</td></tr><tr><td>ARGM-DIR</td><td>0.80</td><td>Directionals.  $\textstyle \operatorname { \mathrm { ~ E . g . } }$  all waves transmit energy from one place to another</td></tr><tr><td>ARGM-PNC</td><td>0.08</td><td>Purpose. E.g. many animals blend in with their environment to not be seen by predators</td></tr><tr><td>ARGM-CAU</td><td>0.05</td><td>Cause. E.g. cold environments sometimes are white in color from being covered in snow</td></tr><tr><td>ARGM-PRP</td><td>1.30</td><td>Purpose. E.g. a pot is made of metal for cooking</td></tr><tr><td>ARGM-EXT</td><td>0.04</td><td>Extent. E.g. as the amount of oxygen exposed to a fire increases the fire will</td></tr><tr><td>ARGM-LOC</td><td>4.50</td><td>burn longer Location. E.g. a solute can be dissolved in a solvent when they are combined</td></tr><tr><td>ARGM-MNR</td><td>2.00</td><td>Manner. E.g. fast means quickly</td></tr><tr><td>ARGM-MOD</td><td>9.80</td><td>Modal verbs. E.g. atom can not be divided into smaller substances</td></tr><tr><td>ARGM-DIS</td><td>0.07</td><td>Discourse. E.g. if something required by an organism is depleted then that organism must replenish that something</td></tr><tr><td>ARGM-GOL</td><td>0.20</td><td>Goal. E.g. We flew to Chicago</td></tr><tr><td>ARGM-NEG</td><td>1.20</td><td>Negation. E.g. cactus wrens building nests in cholla cacti does not harm the</td></tr><tr><td>ARGM-ADV</td><td>6.70</td><td>cholla cacti Adverbials</td></tr><tr><td>ARGM-PRD</td><td>0.20</td><td>Markers of secondary predication. E.g.</td></tr><tr><td>ARGM-TMP</td><td>7.00</td><td>Temporals. E.g. a predator usually kills its prey to eat it</td></tr><tr><td>0</td><td></td><td>Empty tag.</td></tr><tr><td>V</td><td>100</td><td>Verb.</td></tr><tr><td>ARG0</td><td>32.0</td><td>Agent or Causer. E.g. rabbits eat plants</td></tr><tr><td>ARG1</td><td>98.5</td><td>Patient or Theme. E.g. rabbits eat plants</td></tr><tr><td>ARG2</td><td>60.9</td><td>indirect object / beneficiary / instrument / attribute / end state. E.g. animals are organisms</td></tr><tr><td>ARG3</td><td>0.60</td><td>start point / beneficiary / instrument / attribute. E.g. sleeping bags are designed</td></tr><tr><td>ARG4</td><td>0.10</td><td>to keep people warm end point. E.g. when water falls from the sky that water usually returns to the soil</td></tr></table>

Table 10: Semantic Role Labels that appear in explanations corpus.

<table><tr><td>Role-content</td><td>Augmented sentences</td></tr><tr><td>ARG0-plant</td><td>plants use sunlight often to make food for themselves plants produce light in the winter by photosynthesizing green plants contain (water ; food) plants take in oxygen from the air a plant requires water in order to perform photosynthesis some plants grow organically plants use soil as a source of water</td></tr><tr><td>ARG1-water</td><td>water is liquid by volume salt water is a kind of solution water is two things together water is boiling in the pot water is an (inexhaustible ; wasteable ) resource water is an (electrical ; electrical energy ) insulator water is a part of soup</td></tr><tr><td>ARG2-animal</td><td>a hurricane is a kind of animal a bird is a kind of animal a sperm whale is a kind of animal a wren is a kind of animal a dog is a kind of native animal a chameleon is a kind of animal</td></tr><tr><td>PRED-require</td><td>making tools requires using sharp tools plants require resources to provide food for themselves a system requires electrical energy to operate crops require specialized environments to grow cooking requires food from human food chain producing an object requires chemical energy living things require energy from the sun for survival growth requires the production of more cells</td></tr></table>

Table 11: Qualitative evaluation of geometrical data augmentation.

and Della (Hu et al., 2022)<sup>5</sup>. All baseline models undergo training and evaluation with the hyperparameters provided by their respective sources. A latent dimension of 32 is specified to ensure a uniform and equitable comparative analysis.

Autoencoder. In this work, we employ an autoencoder architecture with the same configuration as described in (Li et al., 2020b)<sup>6</sup>. The encoder component is based on BERT (Devlin et al., 2018), while the decoder component is based on GPT2 (Radford et al., 2019). The latent space dimension is set to 32 (low-dimension) as Michlo et al. (2023) revealed that strong compression, such as strong KL regularisation term in ELBO, can lead to the phenomenon of disentanglement of images.

To establish the connection between the encoder and decoder, the input sentence x is first encoded by BERT[cls] into the latent space, denoted as $N ( \mu , \Sigma )$ . The parameters $\mu$ and $\Sigma$ are trainable and determine the mean and covariance of the Gaussian distribution. Next, a sample $z \sim N ( \mu , \Sigma )$ is passed through a multi-layer perceptron called W. This step expands the dimensionality of z to obtain a fixed-length embedding $h \in \mathbb { R } ^ { D \times L \times H }$ , where D represents the dimensions of the heads, L is the number of heads, and H is the number of hidden layers. The latent space injection can be described as:

$$
\mathrm { A t t e n t i o n } ( Q , K , V ) = \mathrm { s o f t m a x } ( \frac { Q [ z ; K ] ^ { T } } { \sqrt { d } } ) [ z ; V ]
$$

INN. The INN consists of 10 invertible blocks. Each is built from three layers, including an affine coupling (Dinh et al., 2016), permutation layer, and ActNorm (Kingma and Dhariwal, 2018). Figure 8 displays one single invertible block. The model was implemented using the FrEIA library (Ardizzone et al., 2018-2022) <sup>7</sup>. As for training hyperparameters of INN, firstly, both input and output have the same dimensions as the latent space dimension of the autoencoder. Secondly, inside the affine coupling block, the sub-network is MLP with 512 as the hidden dimension. Thirdly, we use AdamW (Loshchilov and Hutter, 2017) to optimise the model where the learning rate is 5e-04 in the experiment.

![](images/2e2bb35e0bcd3b828e00b1346955ecf6448e01bcb6103870561bfb5af865885d.jpg)  
Figure 8: INN one single block.

The forward process of the affine coupling layer can be described as follows:

$$
\begin{array} { r } { x _ { a } , x _ { b } = \mathrm { s p l i t } ( x ) } \\ { \log { s } , t = m _ { \theta } ( x _ { b } ) } \\ { s = \mathrm { e x p } ( \log { s } ) } \\ { y _ { a } = s \odot x _ { a } + t } \\ { y _ { b } = x _ { b } } \\ { y = \mathrm { c o n c a t } ( y _ { a } , y _ { b } ) } \end{array}\tag{2}
$$

Where $m _ { \theta }$ is a two-layer neural network. x and y

are the input and output. The reversed process is:

$$
\begin{array} { r } { y _ { a } , y _ { b } = \mathrm { s p l i t } ( y ) } \\ { \log s , t = m _ { \theta } ( y _ { b } ) } \\ { s = \mathrm { e x p } ( \log s ) } \\ { x _ { a } = ( y _ { a } - t ) / s } \\ { x _ { b } = y _ { b } } \\ { y = \mathrm { c o n c a t } ( x _ { a } , x _ { b } ) } \end{array}\tag{3}
$$

## B Additional Supervision Results

Disentanglement between ARG1 clusters We consider four ARG1 clusters, including ARG1-food, ARG1-oxygen, ARG1-sun, ARG1-water, and evaluate model performance following the same procedure. Figure 9 displays the distributions of four role-content clusters over the latent space. With similar observations as before, the INN clustersupervised training strategy can learn better disentanglement between ARG1 clusters. Table 12

![](images/bd2a4b70a0d428293a389bd6b0e7756308782045d8c9f5bddddc67f01a25e791.jpg)  
Figure 9: ARG1: t-SNE plot (blue: food, green: oxygen, red: sun, purple: water). Supervision (right) induces separability comparable with ARG0. PCA plot is provided in Figure 11.

and 13 show the disentanglement metrics and invertibility ratio, respectively. With similar observations as the previous experiment: all classifiers trained over the supervised latent representation outperform both the unsupervised INN model and Optimus, and both unsupervised and supervised cases can achieve higher ratios (at least 0.95).

Invertibility ratio. Table 13, 14, and 15 report the invertibility test for ARG1, PRED, and ARG0,1,2 clusters, respectively. We can observe that INN with both training approaches can perform stable invertibility.

Traversal decoding for Animal clusters. Table 16 shows the decoded explanations traversed around the central point of each cluster in the latent space of cluster-supervised INN.

Traversal decoding for cluster connection. Table 17 displays the decoded middle points between clusters. It is also observable that there are lowdensity embedding regions at the transition (connection) between two clusters. We decode the middle datapoints between animal and human clusters and list them in Table 17. From those examples, we can observe that such explanations are related to both animal and human. This result implies that the explanations may be geometrically represented in a similar way as they were originally designed in the WorldTree corpus (maximising lexical overlaps for pred-arg alignments within an explanation chain) for supporting multi-hop inference tasks.

<table><tr><td colspan="5">ARG1: disentanglement proxy metrics (forward: T)</td></tr><tr><td>classifier train accuracy</td><td></td><td></td><td></td><td>precision recall f1 score</td></tr><tr><td rowspan="2">KNN</td><td>0</td><td>0.934</td><td>0.934 0.933</td><td>0.933 0.913</td></tr><tr><td>U C</td><td>0.914 0.954</td><td>0.914 0.954</td><td>0.914 0.954 0.954</td></tr><tr><td rowspan="3">NB</td><td>0</td><td>0.904</td><td>0.910</td><td>0.902 0.904</td></tr><tr><td>U</td><td>0.922</td><td>0.922</td><td>0.922 0.922</td></tr><tr><td>C</td><td>0.957</td><td>0.957</td><td>0.957 0.957</td></tr><tr><td rowspan="3">SVM</td><td>0</td><td>0.951</td><td>0.951</td><td>0.951 0.950</td></tr><tr><td>U</td><td>0.953</td><td>0.953</td><td>0.952 0.953</td></tr><tr><td>C</td><td>0.959</td><td>0.959</td><td>0.959 0.959</td></tr></table>

Table 12: Forward evaluation for ARG1, consistent results on different classifiers indicate that supervision can perform better semantic disentanglement.

<table><tr><td colspan="3">ARG1: invertibility ratio (backward: T&#x27;)</td></tr><tr><td>train food</td><td>oxygen sun</td><td>water</td></tr><tr><td>U 0.990</td><td>0.980 0.950</td><td>1.000</td></tr><tr><td>C 0.960</td><td>0.950 0.960</td><td>1.000</td></tr></table>

Table 13: backward evaluation for ARG1 clusters. unsupervised INN (U), and supervised INN (S).

Principal component analysis (PCA) visualisation. In addition to the non-linearised t-SNE plot, we also provide linearised visualisation via PCA (Shlens, 2014). Figure 10,11,12, and 13 visualize the separation of ARG0, ARG1, PRED, and animal. Similar to the observation before, cluster supervision can lead to better separation and cluster.

![](images/e9b0d13c1d2d54581516a9d7aa32a3863c303ffbf7442cdb0cf42a9248ba1145.jpg)  
Figure 10: PCA visualization for ARG0.

<table><tr><td colspan="3">PRED: invertibility test (backward: T&#x27;)</td></tr><tr><td>train</td><td>is are cause</td><td>require</td></tr><tr><td>U</td><td>1.000 0.950 0.970</td><td>0.800</td></tr><tr><td>C</td><td>1.000 0.880 0.900</td><td>0.820</td></tr></table>

Table 14: backward evaluation for predicate clusters. unsupervised INN (U), and supervised INN (S).
<table><tr><td colspan="2">Animal: invertibility ratio (backward: T&#x27;)</td></tr><tr><td>train ARG0 ARG1</td><td>ARG2</td></tr><tr><td>U 0.990 0.990</td><td>0.900</td></tr><tr><td>C 0.970 0.960</td><td>0.920</td></tr></table>

Table 15: Backward evaluation for Animal.

![](images/797a82403b2b56e4aa3663176fd84795d3cc9bd173025b4ae78677be2a9b9012.jpg)

Table 16: Traversal in each cluster (top: ARG0-Animal, middle: ARG1-Animal, bottom: ARG2-Animal).  
![](images/fd2605ac5b20a9eccea848161a604bce1f4cb7f4c3717c6e334f181a3016149a.jpg)

Table 17: Middle explanations between ARG0-animal and ARG0-human.  
![](images/1205a2b8722a331a7af4e62f511effa5dcad1cf80312a190aa99b69f5fffc9f5.jpg)  
Figure 11: PCA visualization for ARG1.

## C Statistical Significance Tests for PRED Downstream Classifiers

Statistical significance testing is a standard statistical tool devised to ensure that experimental results are not coincidental and reliable. Following the work (Dror et al., 2018)<sup>8</sup>, we provide statistical significance tests to rigorous and quantitatively evaluate the stability of trainable downstream classifiers, which indirectly indicates the representation capability.

![](images/8dcae79455a7b9772619086d2e0be52d9dbe4c68a01364ee2c90009fa00ef5dc.jpg)  
Figure 12: PCA visualization for PRED.

![](images/876a18d5f7d56fb28be596cf7e54d9a01b519012b8038a8435567c999a58bf58.jpg)  
Figure 13: PCA visualization for Animal.

Our attention was directed towards PRED clusters due to the comparatively decreased performance of downstream classifiers within this category as PRED usually contains less semantic information (Zhang et al., 2022). We select accuracy metric, set α = 0.05, and choose bootstrap statistical test which was used with a variety of NLP tasks (Ouchi et al., 2017; Wu et al., 2017).

As illustrated in Table 18, (1) the U-C pair consistently yields a diminished significance value, suggesting reliable classification performance resulting from superior representational capabilities facilitated by the AutoEncoder with INN configuration, compared with Optimus. (2) the scores of (O-C) pairs are consistently lower than those of (O-U) pairs, indicating our supervision (C) can better represent semantic information than unsupervised INN. We refer (Dror et al., 2018) for an in-depth illustration of statistical significance tests in NLP.

## D Ablation of Data Augmentation

PRED semantic role. Firstly, we analyse the effect of our supervision approach on PRED semantic role with three lexical contents without data augmentation, including are ( 449), cause ( 380), and require ( 262). The rationale for their selection is that they are less frequent in corpus and partially overlap in latent space. Moreover, the contents under PRED usually have less effect on the contextual semantics (Zhang et al., 2022). Those difficulties allow us to fairly analyse the effect of our supervision approach. Following a similar order, we first visualise the t-SNE and PCA plots in Figure 14. As we can observe, the cluster-supervised approach can better represent the cluster and separation for different contents under PRED semantic role label without data augmentation. Next, we apply downstream classifiers to evaluate cluster separation. As illustrated in Table 19, our cluster-supervised approach results in better classification performance, indicating better disentanglement.

<table><tr><td colspan="2">Statistical significance tests for PRED</td></tr><tr><td>classifier source Bootstrap (p-value)↓</td><td></td></tr><tr><td rowspan="3">O-C KNN U-C O-U</td><td>0.0155</td></tr><tr><td>0.0000</td></tr><tr><td>1.0000</td></tr><tr><td>O-C NB U-C O-U</td><td>0.0000 0.0000</td></tr><tr><td rowspan="3">O-C SVM U-C O-U</td><td>0.2268</td></tr><tr><td>0.3594</td></tr><tr><td>0.0000 1.0000</td></tr></table>

Table 18: Statistical significance tests for downstream classifiers (O: Optimus, U: unsupervised INN, and C: cluster supervised INN). We highlight the best significant test value, indicating reliable classification performance derived from better representation capability.

![](images/859b69daf72aa3bfba57d438f25b9ca1a5531d23b2d72069a2361615a3203a25.jpg)  
Figure 14: Ablation: t-SNE plot (top), PCA plot (bottom) (left: Optimus, middle: unsupervised, right: cluster-supervised) where blue: PRED-are, green: PRED-cause, red: PRED-require.

ARG0 semantic role. Next, we provide the same analyse for fewer frequent ARG0 clusters: ARG0- animal ( 126), ARG0-human ( 43), ARG0-plant

<table><tr><td colspan="4">PRED: disentanglement proxy metrics</td></tr><tr><td>classifier train</td><td>accuracy</td><td>precision</td><td>recall f1 score</td></tr><tr><td rowspan="2">KNN</td><td>0 0.858</td><td>0.847</td><td>0.844 0.846</td></tr><tr><td>U 0.837</td><td>0.849</td><td>0.827 0.830</td></tr><tr><td rowspan="3">NB</td><td>C 0.965 0 0.839</td><td>0.963 0.823</td><td>0.961 0.962 0.833 0.826</td></tr><tr><td>U 0.901</td><td>0.895</td><td>0.891 0.893</td></tr><tr><td>C 0.977</td><td>0.974</td><td>0.975 0.974</td></tr><tr><td rowspan="3">SVM</td><td>0 0.876</td><td>0.863</td><td>0.866 0.865</td></tr><tr><td>U 0.954</td><td>0.953</td><td>0.949 0.950</td></tr><tr><td>C 0.967</td><td>0.965</td><td>0.967 0.966</td></tr></table>

Table 19: Ablation: disentanglement proxy metrics for PRED-are, PRED-cause, and PRED-require.

<table><tr><td colspan="4">ARGO: disentanglement proxy metrics</td></tr><tr><td>classifier train</td><td>accuracy</td><td>precision</td><td>recall f1 score</td></tr><tr><td rowspan="3">KNN</td><td>0</td><td>0.890</td><td>0.890 0.850 0.867</td></tr><tr><td>U</td><td>0.890</td><td>0.896 0.834 0.858</td></tr><tr><td>C</td><td>0.919</td><td>0.907 0.858 0.877</td></tr><tr><td rowspan="3">NB</td><td>0</td><td>0.855</td><td>0.809 0.784 0.792</td></tr><tr><td>U</td><td>0.936</td><td>0.916 0.905 0.910</td></tr><tr><td>C</td><td>0.965</td><td>0.958 0.950 0.954</td></tr><tr><td rowspan="3">SVM</td><td>0</td><td>0.843</td><td>0.630 0.691 0.656</td></tr><tr><td>U</td><td>0.895</td><td>0.847 0.770 0.782</td></tr><tr><td>C</td><td>0.901</td><td>0.935 0.779 0.790</td></tr></table>

Table 20: Ablation: disentanglement proxy metrics for ARG0-animal, ARG0-human, ARG0-plant, and ARG0- something.

![](images/7a030ef7dde031cc0ed1a68b4a35f9c944ac424e5d913d8e210d9b99b3ac30b5.jpg)  
Figure 15: Ablation: t-SNE plot (top), PCA plot (bottom) (left: Optimus, middle: unsupervised, right: cluster-supervised) where blue: ARG0-animal, green: ARG0-human, red: ARG0-plant, purple: ARG0- something.  
Table 21: Interpolation examples where top and bottom sentences are source and target, respectively.

## E Controlled Interpolation

In tables 22 and 23, we provide more controllable interpolation examples. Those examples reveal that the latent space with better role-content separation from supervised INN can provide better interpola-

## F INNs: Explanation Reconstruction

Table 24 shows some reconstructed explanations from AutoEncoder, unsupervised INN, and supervised INN, respectively.

![](images/a2fc74fa1cf0adb73009d7a25abb52377a927a961297952a09bb17edf96382c7.jpg)  
Table 22: Interpolation examples (top: supervised INN, bottom: Optimus).

![](images/8aaae022faefeafcbe71e02b9b879b97cdb9dad114a8ec3fb62c5c3b4dd7c457.jpg)  
Table 23: Interpolation examples (top: supervised INN, bottom: Optimus).

<table><tr><td>Augmented explanations</td><td>BERT-GPT2</td><td>unsupervised INN</td><td>supervised INN</td></tr><tr><td>a animal requires water for survival</td><td>a animal requires water for survival</td><td>a animal requires water for survival</td><td>a animal requires water for survival</td></tr><tr><td>an animal requires a mate for survival</td><td>an animal requires a mate to reproduce</td><td>an animal requires a mate to reproduce</td><td>an animal requires a repro- ductive system for survival</td></tr><tr><td>some animals sometimes hunt for prey</td><td>some animals prey on other animals</td><td>some animals sometimes catch prey</td><td>some animals sometimes hunt for prey</td></tr><tr><td>an animal requires energy of its own to move</td><td>an animal requires energy from somewhere to move</td><td>an animal requires energy to move</td><td>an animal requires energy for movement</td></tr><tr><td>an animal requires energy to run</td><td>an animal requires energy to run</td><td>an animal requires energy to run</td><td>an animal requires energy to run</td></tr><tr><td>animals live in their habitats animals must eat animals to survive</td><td>animals live in their habitats animals must eat to survive</td><td>animals live in their habitat animals must eat other ani-</td><td>animals live in their habitat animals must eat to survive</td></tr><tr><td>animals taste flavors animals eat plants</td><td>animals taste flavors</td><td>mals to survive animals taste flavors</td><td>animals taste flavors</td></tr><tr><td>an animal requires nutrients to grow and heal</td><td>animals eat plants an animal requires nutrients</td><td>animals eat plants an animal requires nutrients</td><td>animals eat plants an animal needs to store fat</td></tr><tr><td>animals require oxygen to</td><td>in soil for survival animals require oxygen to</td><td>to grow and repair animals require oxygen to</td><td>to grow animals require oxygen for</td></tr><tr><td>grow an animal needs to breathe in</td><td>grow an animal requires food for</td><td>breath a animal needs to breathe to</td><td>survival an animal requires water and</td></tr><tr><td>order to survive humans cause the disease</td><td>survival humans cause the disease</td><td>survive humans cause the disease</td><td>food to survive humans cause the disease</td></tr><tr><td>humans have a negative im- pact on the environment</td><td>humans have a negative im-</td><td>humans have a negative im-</td><td>humans have a negative im-</td></tr><tr><td>humans require water to sur-</td><td>pact on the ecosystem humans require water to sur-</td><td>pact on the environment humans require water for sur-</td><td>pact on the environment humans require water for sur-</td></tr><tr><td>vive humans produce offspring</td><td>vive</td><td>vival</td><td>vival</td></tr><tr><td>humans have lived on earth</td><td>humans produce offspring humans live in the solar sys-</td><td>humans eat plants humans live in the solar sys-</td><td>humans produce offspring humans live in the biosphere</td></tr><tr><td>humans use fossil fuels for</td><td>tem humans use fossil fuels to</td><td>tem humans use fossil fuels to</td><td>humans use natural gas to</td></tr><tr><td>energy humans eat green plants</td><td>make energy</td><td>make energy</td><td>make energy</td></tr><tr><td>humans eat fruit</td><td>humans eat green plants humans eat fruit</td><td>humans eat green plants humans eat fruit</td><td>humans eat green plants humans eat fruit</td></tr><tr><td>humans sometimes eat plants</td><td>humans sometimes eat plants</td><td>living things sometimes eat</td><td>animals sometimes eat seeds</td></tr><tr><td>or animals a plant absorbs light energy</td><td>and animals a plant absorbs sunlight for</td><td>insects / animals an flower requires energy to</td><td>from trees a plant absorbs light for pho-</td></tr><tr><td>for photosynthesis a plant absorbs water from</td><td>photosynthesis a plant absorbs water from</td><td>grow and provide warmth to the skin</td><td>tosynthesis</td></tr><tr><td>the air into its roots a plant uses energy to grow</td><td>the air into its body a plant requires energy for</td><td>a leaf absorbs water from the air through the leaves</td><td>a plant absorbs water and nu- trients from the air</td></tr><tr><td>plant reproduction occurs in</td><td>growth</td><td>a plant requires energy to grow</td><td>a plant requires energy to grow</td></tr><tr><td>the spring</td><td>plant reproduction occurs in the spring</td><td>plant reproduction begins during seed dispersal</td><td>plant reproduction begins in spring</td></tr><tr><td>plants require water and sun- light to grow</td><td>plants require water and sun- light to grow</td><td>plants require sunlight to grow and survive</td><td>plants require water and sun- light to grow</td></tr><tr><td>a plant requires a habitat for survival</td><td>a plant needs a habitat for sur- vival</td><td>a plant requires a habitat for survival</td><td>a plant requires a habitat for survival</td></tr></table>

Table 24: Explanation reconstruction. From left to right are augmented explanations, decoded explanations from AutoEncoder, explanations from unsupervised INN, and that from supervised INN, respectively.