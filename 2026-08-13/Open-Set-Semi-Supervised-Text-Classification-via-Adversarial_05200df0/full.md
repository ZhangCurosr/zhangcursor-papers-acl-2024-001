# Open-Set Semi-Supervised Text Classification via Adversarial Disagreement Maximization

Junfan Chen<sup>1,2</sup>, Richong Zhang<sup>1,3</sup>\*, Junchi Chen<sup>1</sup>, Chunming Hu<sup>1,2,3</sup>

<sup>1</sup>CCSE, School of Computer Science and Engineering, Beihang University, Beijing, China

<sup>2</sup>School of Software, Beihang University, Beijing, China

<sup>3</sup>Zhongguancun Laboratory, Beijing, China

chenjf, zhangrc @act.buaa.edu.cn, sy2206115, hucm @buaa.edu.cn

## Abstract

Open-Set Semi-Supervised Text Classification (OSTC) aims to train a classification model on a limited set of labeled texts, alongside plenty of unlabeled texts that include both in-distribution and out-of-distribution examples. In this paper, we revisit the main challenge in OSTC, i.e., outlier detection, from a measurement disagreement perspective and innovatively propose to improve OSTC performance by directly maximizing the measurement disagreements. Based on the properties of in-measurement and crossmeasurements, we design an Adversarial Disagreement Maximization (ADM) model that synergeticly optimizes the measurement disagreements. In addition, we develop an abnormal example detection and measurement calibration approach to guarantee the effectiveness of ADM training. Experiment results and comprehensive analysis of three benchmarks demonstrate the effectiveness of our model.

## 1 Introduction

Text classification is a fundamental task in natural language processing. With the development of modern deep learning techniques, text classification has achieved significant advancement. However, deep learning models usually require substantial labeled data, which is expensive in many realworld applications. To tackle this problem, Semisupervised Text Classification (STC) has been proposed, which only needs a small set of labeled examples along with plenty of unlabeled examples (Lee et al., 2013; Tarvainen and Valpola, 2017; Meng et al., 2018; Gururangan et al., 2019; Chen et al., 2020; Lee et al., 2021; Tsai et al., 2022; Yang et al., 2023). By utilizing unlabeled texts in training machine learning models, these approaches reduce the need to expensively annotate abundant data. However, the STC assumption that all unlabeled texts are sampled within the intended scope is impractical in real-world applications. Therefore, researchers recently explored Open-set Semi-supervised Text Classification (OSTC) (Chen et al., 2023), which allows the inclusion of out-ofdistribution examples in the unlabeled text set.

The main challenge in OSTC is the commonly known false positive inference problem (Chen et al., 2023), which indicates a phenomenon that out-ofdistribution texts are prone to be recognized as an in-distribution class, leading to unsatisfactory OSTC outcomes. To address this issue, prior studies have integrated an STC model with different outlier detection techniques. For example, LMCL (Lin and Xu, 2019) and Softmax (Yan et al., 2020) learn discriminative embeddings and utilize local outlier factor (LOF) (Breunig et al., 2000) to identify in-distribution (ID) and out-of-distribution (OOD) examples. MSP (Hendrycks and Gimpel, 2017) and LOS (Chen et al., 2023) respectively employ maximum softmax probability and normalized entropy function to identify ID and OOD examples.

We provide a general understanding of existing OSTC models. Specifically, we revisit outlier detection in existing methods from a measurement disagreement perspective. It can be concluded that existing approaches are fundamentally grounded in the assumption that the differentiation between ID and OOD examples hinges on the presence of disagreement in specific measurements associated with the examples. For instance, the density measurement in LOF of LMCL and Softmax, and the confidence and entropy measurements in MSP and LOS. With this assumption, existing methods distinguish ID and OOD examples by setting certain thresholds on the measurement.

The above discussion motivates us to answer a question: Since ID and OOD examples can be more easily distinguished when the measurement disagreement is larger, can we improve outlier detection in OSTC by directly maximizing such measurement disagreement between ID and OOD examples? To answer this question, we make formal definitions and assumptions for measurement disagreements and reveal several useful properties: (1) the disagreement bounds can be increased by maximizing in-measurement disagreement between ID and OOD examples. (2) the comparative consistency assumption allows us to synergisticly optimize two different measurements. (3) the in-example consistency assumption enables us to detect abnormal examples and calibrate measurements.

Based on the above motivations, we propose an Adversarial Disagreement Maximization (ADM) model for OSTC. Concretely, we treat the crossentropy loss of the ID softmax classifier and the outlier detection confidence as two different measurements. To perform a synergistically measurement optimization, we leverage an adversarial learning approach to iteratively enlarge the disagreements of the two measurements. In addition, to guarantee the effectiveness of disagreement maximization, we design an abnormal-example detection approach to correct the measurement optimization direction.

To summarize, we make the following contributions: (1) We provide a general understanding of outlier detection in OSTC and revisit it from a measurement disagreement perspective. (2) We propose an ADM model to directly maximize measurement disagreements combined with measurement calibration. (3) We evaluate our ADM model on three benchmark datasets and demonstrate its effectiveness by comprehensive analysis.

## 2 Related Works

Semi-supervised text classification (STC) trains a model with a few labeled texts and many unlabeled texts. Various regularization techniques, consistency training approach are developed for STC (Miyato et al., 2017; Gururangan et al., 2019; Liu et al., 2021; Li et al., 2021; Xu et al., 2022). Among these models, UDA (Xie et al., 2020) utilizes substituting noising operations to construct data augmentations combined with consistency training. MixText (Chen et al., 2020) design a novel text data augmentation method with Mixup and used in consistency training. Some other works leverage pseudo-labeling to annotate unlabeled texts as additional training data (Lee et al., 2013; Tarvainen and Valpola, 2017; Meng et al., 2018; Li et al., 2021; Lee et al., 2021; Li et al., 2022b).

Open-set semi-supervised learning is understudied task. The pipeline approaches first filters out

OOD examples and then conduct semi-supervised training with filtered data. These methods often design special outlier detectors (Saito et al., 2021; Huang et al., 2021; Liu et al., 2022) or build new optimization process (Yu et al., 2020; Zhu and Li, 2022; Li et al., 2022a). In recent work LOS (Chen et al., 2023), the authors design a set of pipeline methods utilizing different outlier detection approaches, including MSP (Hendrycks and Gimpel, 2017), DOC (Shu et al., 2017), LMCL (Lin and Xu, 2019) and LSoftmax (Yan et al., 2020). LOS unify semi-supervised training and outlier detection within probabilistic latent variable modeling. We understand OSTC in a new measurement disagreement perspective and leverages adversarial learning (Goodfellow et al., 2014) to achieve disagreement maximization.

## 3 Methodology

## 3.1 Prior Art

Task Definition. In open-set semi-supervised text classification (OSTC), we expect to train a text classification model with a few labeled texts and many unlabeled texts. The unlabeled texts contain both in-distribution (ID) and out-of-distribution (OOD) examples. We will use $\mathcal { V }$ to denote the set of indistribution classes. We assume that we have access to a labeled text set $\mathcal { L } = \{ ( x _ { l } , y _ { l } ) \} _ { i = 1 } ^ { L }$ with each class involving k examples, an unlabeled text set $\mathcal { U } = ( \mathcal { U } _ { + } , \mathcal { U } _ { - } ) = \{ x _ { i } \} _ { i = 1 } ^ { n }$ , which consists of an indistribution text set $\ d \mathcal { U } _ { + }$ and an out-of-distribution text set . The goal of the OSTC task is to identify whether a given text in the test set is an ID or OOD example and predict the exact class type if the given text is an ID example.

Outlier Detection in OSTC. Outlier detection is a key component in OSTC. It contributes to model training and evaluation of OSTC. As demonstrated in previous work (Chen et al., 2023), the key challenge in OSTC is the false positive inference problem, which forces the OOD texts to be recognized as an ID class when conducting semi-supervised learning. An intuitive solution to this problem is utilizing outlier detection to filter unlabeled OOD examples during semi-supervised training. Second, the outlier detector is required to identify whether a given text is an ID or OOD example during evaluation. Considering the vital role of outlier detection in OSTC, in this work, we provide a general understanding of it and propose a novel optimization framework to improve its performance.

Table 1: Analysis for Existing Outlier Detection Methods in OSTC
<table><tr><td>Outlier Detection Model</td><td> $\mathcal { M } ( f ; x , \mathcal { V } )$ </td><td>OOD Identification Condition</td></tr><tr><td>MSP (Hendrycks and Gimpel, 2017)</td><td> $\displaystyle { \operatorname* { m a x } _ { y \in \mathcal { y } } { \frac { f ( x , y ) } { \sum _ { c \in \mathcal { y } } { f ( x , c ) } } } }$ </td><td> $\mathcal { M } ( f ; x , \mathcal { Y } ) < \eta _ { 1 }$ </td></tr><tr><td>DOC (Shu et al., 2017)</td><td> $\operatorname* { m a x } _ { y \in \mathcal { Y } } f _ { y } ( x )$ </td><td> $\mathcal { M } ( f ; x , \mathcal { Y } ) < \eta _ { 2 }$ </td></tr><tr><td>LMCL (Lin and Xu, 2019)</td><td> $\sum _ { x ^ { \prime } \in N _ { k } ( x ) } f ( x ^ { \prime } )$ </td><td> $\mathcal { M } ( f ; x , \mathcal { Y } ) > 1$ </td></tr><tr><td>LSoftmax (Yan et al., 2020)</td><td>|Nk(x)|·f(x)</td><td></td></tr><tr><td>LOS (Chen et al., 2023)</td><td> $\mathcal { H } ^ { \lambda } ( f ( x ) )$   $\overline { { ( \log | \mathcal { V } | ) ^ { \lambda } } }$ </td><td> $\mathcal { M } ( f ; x , \mathcal { Y } ) > 0 . 5$ </td></tr></table>

## 3.2 Motivation

Revisit Outlier Detection from Measurement Disagreement Perspective. To tackle the false positive inference problem, existing works either adopt a pipeline approach that first trains an outlier detector to filter the OOD examples and then conducts semi-supervised training on the rest of unlabeled data (Shu et al., 2017; Yan et al., 2020) or integrate supervised training and outlier detection as a unified framework (Chen et al., 2023) during optimization. We provide a formal understanding of these outlier detection methods and reveal that all of these methods identify OOD examples following a measurement disagreement assumption. Formally, define a measurement $\mathcal { M } ( f ; x , \mathcal { V } )$ which involves an internal function f and two inputs: a text example $x \in \mathcal { U }$ and the in-distribution class set $\mathcal { V } .$ . Based on the specified measurement , existing outlier detection methods introduce a threshold to distinguish ID and OOD examples. Concretely, we give the measurement formulation and OOD identification condition for each outlier detection method in Table 4 and explain as follows:

• MSP: is the maximum softmax probability and $f$ is specified as a softmax classifier, with a condition $\mathcal { M } < \eta _ { 1 }$

• DOC: is the maximum 1-vs-rest probability and each f is a 1-vs-rest sigmoid classifier, with a condition $\mathcal { M } < \eta _ { 2 }$

• LMCL and LSoftmax: is the Local Outlier Factor and $f$ is the local reachability density (Breunig et al., 2000). $\mathcal { N } _ { k }$ are the knearest neighbors with a condition $\mathcal { M } > 1$

• LOS:  is the normalized entropy and f is a softmax classifier, with a condition $\mathcal { M } > 0 . 5$ is the entropy function.

The above analysis indicates that the OOD examples are detected under the assumption that the measurements of an ID example $x _ { + } ~ \in \mathcal { U } _ { + }$ and an OOD example $x _ { - } \in \mathcal { U } _ { - }$ satisfy the following disagreement with some specified threshold $\eta \colon$

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { \mathcal { M } ( f ; x _ { + } , \mathcal { Y } ) < \eta , } \\ { \mathcal { M } ( f ; x _ { - } , \mathcal { Y } ) \geqslant \eta . } \end{array} \right. _ { o r } \left\{ \begin{array} { l l } { \mathcal { M } ( f ; x _ { + } , \mathcal { Y } ) > \eta , } \\ { \mathcal { M } ( f ; x _ { - } , \mathcal { Y } ) \leqslant \eta } \end{array} \right. } \end{array}\tag{1}
$$

Formal Definitions for Measurement Disagreement. For the convenience of formal analysis, we make the following definitions:

Definition 1 Measurement Disagreement. For any ID example $x _ { + } \in \mathcal { U } _ { + }$ and OOD example $x _ { - } \in$ $\mathcal { U } _ { - }$ , given a specified measurement $\mathcal { M } ( f ; x , \mathcal { V } )$ we define the measurement disagreement ofthe two examples $x _ { + }$ and x asfollows:

$$
d _ { \mathcal { M } } ( x _ { + } , x _ { - } ) = \vert \mathcal { M } ( f ; x _ { + } , \mathcal { V } ) - \mathcal { M } ( f ; x _ { - } , \mathcal { V } ) \vert\tag{2}
$$

We will also call $d _ { \mathcal { M } } ( x _ { + } , x _ { - } )$ the in-measurement disagreement of examples $x _ { + }$ and $x _ { - }$

Definition 2 Cross-Measurement Disagreement. For any example $x \in \mathcal { U }$ and two different specified measurements $\mathcal { M } _ { 1 } ( f ; x , \mathcal { V } )$ and $\mathcal { M } _ { 2 } ( f ; x , y )$ we define the cross-measurement disagreement of example x under the two measurement asfollows:

$$
d ( x , \mathcal { M } _ { 1 } , \mathcal { M } _ { 2 } ) = | \mathcal { M } _ { 1 } ( f ; x , \mathcal { V } ) - \mathcal { M } _ { 2 } ( f ; x , \mathcal { V } ) | ( 3 )
$$

Definition 3 ϵ-Bounded Disagreement. For a real number $\epsilon \ \geqslant \ 0$ and a given measurement $\mathcal { M } ( f ; x , \mathcal { V } )$ , we define that ID examples $x _ { + } \in \mathcal { U } _ { + }$ and OOD examples $x _ { - } ~ \in { \mathrm { ~ } } \mathcal { U } _ { - }$ has a ϵ-bounded disagreement, ifthe measurement disagreementfor each $( x _ { + } , x _ { - } )$ pair satisfy:

$$
d _ { \mathcal { M } } ( x _ { + } , x _ { - } ) - \epsilon \geqslant 0\tag{4}
$$

Motivations from Measurement Disagreement. From Definition 1, we know that the measurement disagreement gives the amount of difference between the measurements of an ID and an OOD example. Thus, when the disagreement $d _ { \mathcal { M } } ( x _ { + } , x _ { - } )$

is larger, the model can distinguish $x _ { + }$ and $x _ { - }$ more easily. This motivates us to improve outlier detection in OSTC by enlarging the measurement disagreements. However, as existing outlier detection methods are trained with softmax or sigmoid classifiers and representation learning losses for LOF in a pipeline or unified framework, their training objectives are not developed to directly increase the measurement disagreements. This design overlooks the potential advantage of enlarging measurement disagreements over outlier detection.

Furthermore, based on Definition 3, if a measurement disagreement $d _ { \mathcal { M } } ( x _ { + } , x _ { - } )$ satisfy ϵ-bounded disagreement, the worst case disagreement it can reach is ϵ. Thus, if we want to increase the measurement disagreements for $( x _ { + } , x _ { - } )$ pairs, we may correspondingly maximize disagreement bound ϵ. From Formulation (1), we can easily derive that existing outlier detection methods only satisfy the 0-bounded disagreement, which significantly limits the models’ ability to distinguish ID and OOD examples. The above observations inspire us to ask: Can we increase the disagreement bounds of outlier detection by directly maximizing the measurement disagreements?

Another motivation is that we may unleash the advantages of cross-measurement disagreement in the outlier detection of OSTC. To formally understand the potential of cross-measurement disagreement, we make the following assumptions:

Assumption 1 Comparative Consistency. For two examples $x , x ^ { \prime } \in \mathcal { U }$ and given two different measurements $\mathcal { M } _ { 1 } ( f ; x , \mathcal { V } )$ and $\mathcal { M } _ { 2 } ( f ; x , y )$ , we assume the two measurements are comparative consistency. Namely, $i f \mathcal { M } _ { 1 } ( f ; x , \mathcal { V } ) > \mathcal { M } _ { 1 } ( f ; x ^ { \prime } , \mathcal { V } )$ then $\mathcal { M } _ { 2 } ( f ; x , y ) > \mathcal { M } _ { 2 } ( f ; x ^ { \prime } , y )$ is satisfied.

Under the above assumption, when comparing two different examples, different measurements demonstrate similar behavior. This property manifests that when one measurement is optimized on a set of examples, the other measurement may be accordingly optimized. Thus, we may jointly optimize two different measurements in a synergistic way to mutually maximize the measurement disagreements. However, existing methods only consider a single measurement, which overlooks the mutual enhancement between different measurements.

Assumption 2 In-Example Consistency. For any example $x \in \mathcal { U }$ and given two different measurements $\mathcal { M } _ { 1 } ( f ; x , \mathcal { V } )$ and $\mathcal { M } _ { 2 } ( f ; x , \mathcal { V } )$ , we assume the two measurements satisfy in-example consistency. Namely, existing a small real value $\delta$ that lets all x satisfy $d ( x , { \mathcal { M } } _ { 1 } , { \mathcal { M } } _ { 2 } ) \leqslant \delta$

Under the in-example consistency assumption, we expect that different measurements are consistent on each example. However, this consistency may not be guaranteed when a misidentified ID or OOD example is sent for optimization, we name them abnormal examples. This motivates us to employ cross-measurement disagreements to detect abnormal examples and calibrate the measurements during training. However, existing OSTC methods overlook such availability and lack reliable mechanism to correct misidentified OOD examples during training. This motivates us to consider: How can we exploit the properties of cross-measurement disagreements to promote disagreement maximization and rectify abnormal examples?

## 3.3 Adversarial Disagreement Maximization

In light of the formal discussion on measurement disagreements, we present an Adversarial Disagreement Maximization (ADM) model for OSTC that effectively addresses the above questions. In ADM, we first specify two measurements using a crossentropy loss on the in-distribution classifier and confidence on the outlier detector. An adversarial learning approach is then proposed to synergistically maximize disagreements of the two measurements. To guarantee the optimization quality, we present an abnormal-example detection method to calibrate the disagreement maximization process. The entire structure of our ADM and its implementation process is shown in Figure 1.

![](images/71e834e280fd27a003a8683ab7c43588faa7ce6742a1e5e07db698cc1a4d405b.jpg)  
Figure 1: The implementation process of our Adversarial Disagreement Maximization (ADM) model.

Specification of Measurements. To perform OSTC, we build a softmax classifier on $\mathcal { V }$ to classify ID examples and a sigmoid outlier detector to identify OOD examples. Since OOD examples may cause larger cross-entropy losses of the classifier and lower outlier detection confidence, we naturally treat the cross-entropy loss and the confidence as measurements. Specifically, a pre-trained language model is first used to obtain text representations. For later use, we formulate this text representation process as function $f .$ Then, we can define the measurement of the cross-entropy loss

$$
\mathcal { M } _ { 1 } ( f , \Theta ; x , \hat { y } ) = - \log \frac { \theta _ { \hat { y } } ^ { \mathrm { T } } f ( x ) } { \sum _ { y \in \mathcal { V } } \theta _ { y } ^ { \mathrm { T } } f ( x ) } ,\tag{5}
$$

where $\Theta = \{ \theta _ { y } | y \in \mathcal { V } \}$ is the parameter in the softmax classifier on $\mathcal { V }$ . The measurement of the outlier-detection confidence is defined as

$$
\mathcal { M } _ { 2 } ( f , \Phi ; x ) = \sigma ( \Phi ^ { \mathrm { T } } f ( x ) ) ,\tag{6}
$$

where $\Phi$ is the parameter of the outlier detector and $\sigma$ is the logistic function. Now the two measurements $M _ { 1 }$ and $M _ { 2 }$ are specified.

Disagreement Maximization via Adversarial Learning. To synergistically maximize disagreements between the two measurements, we propose an adversarial learning approach to iteratively enlarge the two disagreements. Concretely, we define the following adversarial training process for OSTC

$$
\operatorname* { m i n } _ { \Theta } \operatorname* { m a x } _ { \Phi } \sum _ { i = 1 } ^ { n } ( - 1 ) ^ { \lambda _ { i } } \mathcal M _ { 2 } ( f , \Phi ; x _ { i } ) \mathcal M _ { 1 } ( f , \Theta ; x _ { i } , \hat { y } _ { i } )\tag{7}
$$

where $\lambda _ { i }$ is a binary indicator that satisfies

$$
\lambda _ { i } = \left\{ \begin{array} { l l } { 1 , { \mathcal { M } } _ { 2 } ( f , \Phi ; x _ { i } ) > 0 . 5 , } \\ { 0 , o t h e r w i s e . } \end{array} \right.\tag{8}
$$

The above objective adopts a min-max optimization process. At the minimization step, the outlier detection confidence $\mathcal { M } _ { 2 }$ performs as a weight on the cross-entropy loss $\mathcal { M } _ { 1 }$ and $\lambda _ { i }$ performs as a switcher to determine maximize or minimize $\mathcal { M } _ { 2 }$ Specifically, when $\lambda _ { i } = 1 , x _ { i }$ is treated as an ID example and the model minimizes its corresponding loss $\mathcal { M } _ { 1 } ( f , \Theta ; x _ { i } , \hat { y } _ { i } )$ . Otherwise, when $\lambda _ { i } = 0$ , x<sub>i</sub> is treated as an OOD example and the model minimizes the negative loss $- { \mathcal { M } } _ { 1 } ( f , \Theta ; x _ { i } , { \hat { y } } _ { i } )$ , i.e., the model maximizes $\mathcal { M } _ { 1 }$ . Similarly, At the maximization step, $\mathcal { M } _ { 1 }$ becomes a weight and the outlier detection confidence $\mathcal { M } _ { 2 }$ is accordingly maximized or minimized according to $\lambda _ { i }$ . In this way, the measurements $\mathcal { M } _ { 1 }$ and $\mathcal { M } _ { 2 }$ for ID examples and OOD examples are updated in opposite directions, thus maximize the in-measurement disagreements $d _ { \mathcal { M } _ { 1 } } ( \boldsymbol { x } _ { + } , \boldsymbol { x } _ { - } )$ and $d _ { \mathcal { M } _ { 2 } } ( x _ { + } , x _ { - } )$ and therefore increase the disagreement bounds.

Abnormal Example Detection and Measurement Calibration. According to Equation (8), the binary indicator $\lambda _ { i }$ rely on the value of the outlier detector $\mathcal { M } _ { 2 } ( f , \Phi ; x _ { i } )$ . As in the adversarial training process, the outlier detector has no supervision signals, it inevitably produces error indicators. An incorrect indicator $\lambda _ { i }$ will reverse the optimization direction and result in performance degradation. However, no information guides us to distinguish the examples that may lead to incorrect indicators.

Fortunately, the in-example consistency assumption provides us with an opportunity to mitigate the problem. Specifically, when an example is misclassified by the outlier detector, i.e., an abnormal example, it is likely to fail to satisfy in-example consistency. Thus, we use this property to detect abnormal examples and reverse corresponding indicator $\lambda _ { i }$ to maintain a correct optimization direction. To realize abnormal example detection, we specify the cross-measurement disagreement as

$$
\begin{array} { r l } {  { d ( x _ { i } , \mathcal { M } _ { 1 } , \mathcal { M } _ { 2 } ) = } } \\ & { \quad \vert \mathcal { M } _ { 2 } ( f , \Phi ; x _ { i } ) - \frac { \mathcal { M } _ { 1 } ( f , \Theta ; x _ { i } , \hat { y } _ { i } ) } { \operatorname* { m a x } _ { j } \mathcal { M } _ { 1 } ( f , \Theta ; x _ { j } , \hat { y } _ { j } ) } \vert } \end{array}\tag{9}
$$

The measurement $\mathcal { M } _ { 1 }$ is normalized with maxnormalization to keep consistent scope $[ 0 , 1 ]$ with measurement $\mathcal { M } _ { 2 }$ . Under this specification, we modify the adversarial learning objective to

$$
\begin{array} { l } { \displaystyle \operatorname* { m i n } _ { \Theta } \displaystyle \operatorname* { m a x } _ { \Phi } \sum _ { i = 1 } ^ { n } ( - 1 ) ^ { \alpha _ { i } } ( - 1 ) ^ { \lambda _ { i } } \mathcal { M } _ { 2 } ( f , \Phi ; x _ { i } ) } \\ { \displaystyle \quad \cdot \mathcal { M } _ { 1 } ( f , \Theta ; x _ { i } , \hat { y } _ { i } ) } \end{array}\tag{10}
$$

where $\alpha _ { i }$ is a binary indicator that indicates whether $x _ { i }$ is an abnormal example and its value is determined by whether $x _ { i }$ satisfies the in-example consistency. Namely, $\alpha _ { i }$ is computed by

$$
\alpha _ { i } = \left\{ \begin{array} { l l } { 0 , d ( x _ { i } , \mathcal { M } _ { 1 } , \mathcal { M } _ { 2 } ) > \delta , } \\ { 1 , o t h e r w i s e . } \end{array} \right.\tag{11}
$$

Note that as $\mathcal { M } _ { 1 }$ and $\mathcal { M } _ { 2 }$ are converse measurements, i.e., larger $\mathcal { M } _ { 1 }$ and smaller $\mathcal { M } _ { 2 }$ indicate

OOD examples, they are more consistent when a larger margin between $\mathcal { M } _ { 1 }$ and $\mathcal { M } _ { 2 }$ is presented, which is different from the isotropic measurements defined in Assumption 2. Under this design, when an example $x _ { i }$ is detected as an abnormal example, i.e., $\alpha _ { i } = 1$ , the model will reverse the optimization direction in the learning objective (10) and calibrate the measurements towards a correct optimization direction, which guarantees the effectiveness of adversarial disagreement maximization.

Optimization Process of ADM. We optimize ADM with the following three training stages:

Pre-stage1: To provide an initialized model, we pre-train the classifier using labeled texts in $\mathcal { L }$ with cross-entropy loss and treat labeled texts as ID examples and unlabeled texts in  as OOD examples to pre-train the outlier detector with BCE loss.

Pre-stage2: Assign pseudo labels for unlabeled texts using the model trained in pre-stage1 and use the pseudo-labeled texts to further refine the classifier and outlier detector.

ADM: Perform adversarial disagreement maximization to iteratively optimize the two measurements and further update the classifier and outlier detector combined with measurement calibration.

## 4 Experiment

## 4.1 Datasets and Experiment Setting

Datasets. We use the 3 benchmark datasets created from existing text classification datasets in the previous work to evaluate OSTC (Chen et al., 2023), including AGNews (Radford et al., 2019), DBPedia (?) and Yahoo (Chang et al., 2008). These datasets are all widely used in text classification and includes enough amounts of common classes to evaluate OSTC models.

Table 2: The statistics of the datasets. # lab., #unl., #val. and #test respectively denote the labeled, unlabeled, validation and test texts for each class. #ID , #OOD denote the number of ID and OOD classes, respectively.
<table><tr><td>Dataset</td><td>#lab. #unl. #val. #test #ID #OOD</td></tr><tr><td>AGNews 10/50/100</td><td>5k 2k 1.9k 2 2</td></tr><tr><td>Yahoo 10/50/100</td><td>5k 2k 6k 6 4</td></tr><tr><td>DBPedia 10/50/100</td><td>5k 2k 5k 8 6</td></tr></table>

Dataset Construction and Experimental Settings. We follow the same dataset splitting setting in previous work (Chen et al., 2023). We use the same ID and OOD class sets and data for training, validation and test described in the previous work. The statistics of these benchmarks are shown in Table 2. We also follow the two metrics used in LOS to evaluate the OSTC performance. We use Acc and F1 to denote the overall accuracy and F1 value, which jointly evaluates a model’s performance in both ID classification and OOD detection.

## 4.2 Implementation and Baseline Models

Implementation. Our ADM model is implemented with PyTorch. We leverage the BERT encoder to build the Θ component and introduce an MLP with a sigmoid function to build the Φ component. When optimizing, we execute 100 updates for the first two pre-training stages respectively and choose model parameters with the best Acc on the validation set. During the ADM training stage, we iteratively execute 100 updates at the minimization step and 100 updates at the maximization step. The labeled training batch size is set to 4 and the unlabeled training batch size is set to 8. The learning rate of the model is set to $5 e - 5$ . The hyper-parameter δ is set to 0.25 on AGNews and Yahoo and is set to 0.15 on DBPedia. All hyper-parameters are selected by grid search on the validation set. We run each experiment 3 times and report the average result and standard deviation. We run all experiments on two NVIDIA Tesla A100 GPUs with 40GB memory.

Baseline Models ADM is compared with the following baselines:

UDA+MSP, MixText+MSP: Pipeline OSTC models combine UDA (Xie et al., 2020), MixText (Chen et al., 2020) with maximum softmax probability to implement OOD detection (Hendrycks and Gimpel, 2017).

UDA+LSoftmax, MixText+LSoftmax: Pipeline OSTC models combine UDA, MixText that learn discriminative text features with an uncomplicated softmax and achieve OOD detection using Local Outlier Factor (Yan et al., 2020).

UDA+DOC, MixText+DOC: Pipeline models combine UDA, MixText with m 1-vs-rest sigmoid classifiers for OOD detection (Shu et al., 2017).

UDA+LMCL, MixText+LMCL: Pipeline OSTC models combine UDA, MixText using large margin cosine loss for OOD detection (Lin and Xu, 2019). UDA+LOS, MixText+LOS: OSTC models unify semi-supervised training and outlier detection within probabilistic latent variable modeling which optimized with EM algorithm based on UDA and Mixtext (Chen et al., 2023).

Table 3: The open-set semi-supervised text classification results on AGNews, Yahoo and DBPedia.
<table><tr><td rowspan=3 colspan=1>Method</td><td rowspan=1 colspan=6>AGNews</td><td rowspan=1 colspan=6>Yahoo</td><td rowspan=1 colspan=6>DBPedia</td></tr><tr><td rowspan=1 colspan=6>k = 10    k = 50    k = 100</td><td rowspan=1 colspan=6> $k = 1 0$     k = 50   k = 100</td><td rowspan=1 colspan=6> $k = 1 0$     k = 50    $k = 1 0 0$ </td></tr><tr><td rowspan=1 colspan=6>Acc F1  Acc  F1   Acc F1</td><td rowspan=1 colspan=6>Acc F1  Acc F1  Acc F1</td><td rowspan=1 colspan=6>Acc F1  Acc F1  Acc F1</td></tr><tr><td rowspan=1 colspan=1>UDA+MSP</td><td rowspan=1 colspan=6>35.1631.4250.8331.9060.5756.51</td><td rowspan=1 colspan=2>|40.9712.44</td><td rowspan=1 colspan=1>47.80</td><td rowspan=1 colspan=1>27.87</td><td rowspan=1 colspan=2>55.5144.84</td><td rowspan=1 colspan=4>|52.0637.1481.1483.31</td><td rowspan=1 colspan=2>81.5384.46</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>(0.19)(0.34)(</td><td rowspan=1 colspan=2>1.25)(0.12)</td><td rowspan=1 colspan=2>(1.49)(2.40)</td><td rowspan=1 colspan=2>(0.28)(0.73)</td><td rowspan=1 colspan=1>(0.89)</td><td rowspan=1 colspan=1>(1.68)</td><td rowspan=1 colspan=1>(0.86)</td><td rowspan=1 colspan=1>(1.29)</td><td rowspan=1 colspan=2>(1.90)(2.43)</td><td rowspan=1 colspan=2>(1.44)(0.91)</td><td rowspan=1 colspan=1>(0.93)</td><td rowspan=1 colspan=1>(0.69)</td></tr><tr><td rowspan=1 colspan=1>UDA+LSoftmax</td><td rowspan=1 colspan=2>37.4935.93</td><td rowspan=1 colspan=2>38.7538.80</td><td rowspan=1 colspan=2>41.4539.63</td><td rowspan=1 colspan=2>38.9541.94</td><td rowspan=1 colspan=1>44.49</td><td rowspan=1 colspan=1>49.62</td><td rowspan=1 colspan=1>44.92</td><td rowspan=1 colspan=1>51.35</td><td rowspan=1 colspan=2>53.9665.01</td><td rowspan=1 colspan=2>55.4065.62</td><td rowspan=1 colspan=1>55.29</td><td rowspan=1 colspan=1>63.74</td></tr><tr><td rowspan=2 colspan=1>UDA+DOC</td><td rowspan=1 colspan=2>(0.91)(0.47)</td><td rowspan=1 colspan=2>(0.29)(0.29)</td><td rowspan=1 colspan=2>(1.79)(0.84)</td><td rowspan=1 colspan=2>(3.28)(1.21)</td><td rowspan=1 colspan=1>(0.96)</td><td rowspan=1 colspan=1>(0.85)</td><td rowspan=1 colspan=1>(0.08)</td><td rowspan=1 colspan=1>(0.96)</td><td rowspan=1 colspan=2>(1.40)(3.97)</td><td rowspan=1 colspan=2>(0.63)(2.37)</td><td rowspan=1 colspan=1>(0.53)</td><td rowspan=1 colspan=1>(0.86)</td></tr><tr><td rowspan=1 colspan=2>33.2630.89</td><td rowspan=1 colspan=1>49.95</td><td rowspan=1 colspan=1>26.39</td><td rowspan=1 colspan=2>60.5955.16</td><td rowspan=1 colspan=2>39.939.87</td><td rowspan=1 colspan=1>40.01</td><td rowspan=1 colspan=1>8.21</td><td rowspan=1 colspan=1>47.98</td><td rowspan=1 colspan=1>28.39</td><td rowspan=1 colspan=2>42.846.73</td><td rowspan=1 colspan=1>48.89</td><td rowspan=1 colspan=1>18.79</td><td rowspan=1 colspan=1>82.88</td><td rowspan=1 colspan=1>82.30</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>(0.65)(1.11)(</td><td rowspan=1 colspan=1>0.33)</td><td rowspan=1 colspan=1>(0.73)</td><td rowspan=1 colspan=2>(2.29)(2.04)</td><td rowspan=1 colspan=2>(0.17)(0.40)</td><td rowspan=1 colspan=1>(0.07)</td><td rowspan=1 colspan=1>(0.02)</td><td rowspan=1 colspan=1>(0.62)</td><td rowspan=1 colspan=1>(1.27)</td><td rowspan=1 colspan=2>(0.07)(0.05)</td><td rowspan=1 colspan=1>(1.20)</td><td rowspan=1 colspan=1>(2.96)</td><td rowspan=1 colspan=2>(1.06)(1.36)</td></tr><tr><td rowspan=2 colspan=1>UDA+LMCL</td><td rowspan=1 colspan=2>35.2031.48</td><td rowspan=1 colspan=1>44.66</td><td rowspan=1 colspan=1>28.03</td><td rowspan=1 colspan=2>48.2641.98</td><td rowspan=1 colspan=2>41.9237.06</td><td rowspan=1 colspan=1>46.62</td><td rowspan=1 colspan=1>54.05</td><td rowspan=1 colspan=1>47.65</td><td rowspan=1 colspan=1>55.26</td><td rowspan=1 colspan=2>53.1359.46</td><td rowspan=1 colspan=1>56.16</td><td rowspan=1 colspan=1>69.94</td><td rowspan=1 colspan=1>56.34</td><td rowspan=1 colspan=1>69.79</td></tr><tr><td rowspan=1 colspan=2>(0.66)(0.50)(</td><td rowspan=1 colspan=1>1.61)</td><td rowspan=1 colspan=1>(7.69)</td><td rowspan=1 colspan=2>(6.82)(5.77)</td><td rowspan=1 colspan=2>(4.28)(20.49</td><td rowspan=1 colspan=1>(0.59))</td><td rowspan=1 colspan=1>(1.10)</td><td rowspan=1 colspan=1>(0.22)</td><td rowspan=1 colspan=1>(0.49)</td><td rowspan=1 colspan=2>(3.38)(12.82</td><td rowspan=1 colspan=1>(0.27))</td><td rowspan=1 colspan=1>(0.47)</td><td rowspan=1 colspan=1>(0.09)</td><td rowspan=1 colspan=1>(0.80)</td></tr><tr><td rowspan=1 colspan=1>MixText+MSP</td><td rowspan=1 colspan=2>35.1732.23</td><td rowspan=1 colspan=2>50.3928.40</td><td rowspan=1 colspan=2>63.0259.21</td><td rowspan=1 colspan=2>40.4910.41</td><td rowspan=1 colspan=1>48.26</td><td rowspan=1 colspan=1>28.32</td><td rowspan=1 colspan=1>54.30</td><td rowspan=1 colspan=1>42.63</td><td rowspan=1 colspan=2>53.0438.17</td><td rowspan=1 colspan=2>80.9682.97</td><td rowspan=1 colspan=1>80.77</td><td rowspan=1 colspan=1>83.83</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>(0.89)</td><td rowspan=1 colspan=1>(0.40)</td><td rowspan=1 colspan=2>(0.49)(3.58)</td><td rowspan=1 colspan=2>(0.41)(1.35)</td><td rowspan=1 colspan=2>(0.37)(1.72)</td><td rowspan=1 colspan=1>(0.68)</td><td rowspan=1 colspan=1>(1.35)</td><td rowspan=1 colspan=1>(1.13)</td><td rowspan=1 colspan=1>(3.12)</td><td rowspan=1 colspan=2>(0.65)(1.70)</td><td rowspan=1 colspan=1>(0.57)</td><td rowspan=1 colspan=1>(0.36)</td><td rowspan=1 colspan=1>(0.83)</td><td rowspan=1 colspan=1>(0.29)</td></tr><tr><td rowspan=1 colspan=1>MixText+LSoftmax</td><td rowspan=1 colspan=1>37.93</td><td rowspan=1 colspan=1>37.80</td><td rowspan=1 colspan=2>39.5339.69</td><td rowspan=1 colspan=2>40.9839.41</td><td rowspan=1 colspan=2>43.1348.31</td><td rowspan=1 colspan=1>45.62</td><td rowspan=1 colspan=1>52.32</td><td rowspan=1 colspan=1>45.75</td><td rowspan=1 colspan=1>51.50</td><td rowspan=1 colspan=2>53.8465.80</td><td rowspan=1 colspan=1>56.64</td><td rowspan=1 colspan=1>67.21</td><td rowspan=1 colspan=1>57.83</td><td rowspan=1 colspan=1>64.93</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>(1.15)(1.58)</td><td rowspan=1 colspan=1>(0.53)</td><td rowspan=1 colspan=1>(0.59)</td><td rowspan=1 colspan=2>(3.16)(2.61)</td><td rowspan=1 colspan=2>(2.12)(1.24)</td><td rowspan=1 colspan=1>(0.62)</td><td rowspan=1 colspan=1>(0.66)</td><td rowspan=1 colspan=1>(0.58)</td><td rowspan=1 colspan=1>(1.12)</td><td rowspan=1 colspan=2>(2.32)(2.92)</td><td rowspan=1 colspan=1>(1.71)</td><td rowspan=1 colspan=1>(3.36)</td><td rowspan=1 colspan=1>(2.03)</td><td rowspan=1 colspan=1>(0.98)</td></tr><tr><td rowspan=1 colspan=1>MixText+DOC</td><td rowspan=1 colspan=2>38.1736.54</td><td rowspan=1 colspan=1>50.10</td><td rowspan=1 colspan=1>22.56</td><td rowspan=1 colspan=2>58.4151.58</td><td rowspan=1 colspan=2>40.129.13</td><td rowspan=1 colspan=1>40.03</td><td rowspan=1 colspan=1>8.26</td><td rowspan=1 colspan=1>47.38</td><td rowspan=1 colspan=1>26.61</td><td rowspan=1 colspan=2>42.857.27</td><td rowspan=1 colspan=1>47.65</td><td rowspan=1 colspan=1>16.93</td><td rowspan=1 colspan=1>81.74</td><td rowspan=1 colspan=1>82.55</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>(0.79)(1.89)</td><td rowspan=1 colspan=1>(0.04)(</td><td rowspan=1 colspan=1>0.13)</td><td rowspan=1 colspan=2>(1.27)(4.02)</td><td rowspan=1 colspan=2>(0.02)(0.07)</td><td rowspan=1 colspan=1>(0.44)</td><td rowspan=1 colspan=1>(1.00)</td><td rowspan=1 colspan=1>(0.79)</td><td rowspan=1 colspan=1>(1.89)</td><td rowspan=1 colspan=2>(0.02)(0.58)</td><td rowspan=1 colspan=1>(1.76)</td><td rowspan=1 colspan=1>(3.50)</td><td rowspan=1 colspan=1>(3.61)</td><td rowspan=1 colspan=1>(2.23)</td></tr><tr><td rowspan=1 colspan=1>MixText+LMCL</td><td rowspan=1 colspan=1>34.20</td><td rowspan=1 colspan=1>27.41</td><td rowspan=1 colspan=1>40.55</td><td rowspan=1 colspan=1>38.44</td><td rowspan=1 colspan=1>51.27</td><td rowspan=1 colspan=1>47.66</td><td rowspan=1 colspan=1>43.89</td><td rowspan=1 colspan=1>49.35</td><td rowspan=1 colspan=1>47.04</td><td rowspan=1 colspan=1>53.77</td><td rowspan=1 colspan=1>48.42</td><td rowspan=1 colspan=1>55.70</td><td rowspan=1 colspan=2>55.4069.79</td><td rowspan=1 colspan=1>56.05</td><td rowspan=1 colspan=1>70.24</td><td rowspan=1 colspan=1>56.24</td><td rowspan=1 colspan=1>72.08</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>(0.93)</td><td rowspan=1 colspan=1>(3.78)</td><td rowspan=1 colspan=1>(8.87)</td><td rowspan=1 colspan=1>(11.91)</td><td rowspan=1 colspan=1>(10.70)</td><td rowspan=1 colspan=1>(7.06)</td><td rowspan=1 colspan=1>(2.65)</td><td rowspan=1 colspan=1>(2.56)</td><td rowspan=1 colspan=1>(0.17)</td><td rowspan=1 colspan=1>(0.79)</td><td rowspan=1 colspan=1>(0.63)</td><td rowspan=1 colspan=1>(1.12)</td><td rowspan=1 colspan=2>(0.08)(0.23)</td><td rowspan=1 colspan=1>(0.06)</td><td rowspan=1 colspan=1>(0.13)</td><td rowspan=1 colspan=1>(0.11)(</td><td rowspan=1 colspan=1>1.93)</td></tr><tr><td rowspan=4 colspan=1>UDA+LOSMixText+LOS</td><td rowspan=1 colspan=1>56.44</td><td rowspan=1 colspan=1>42.45</td><td rowspan=1 colspan=1>75.12</td><td rowspan=1 colspan=1>72.52</td><td rowspan=1 colspan=1>75.53</td><td rowspan=1 colspan=1>74.64</td><td rowspan=1 colspan=1>46.49</td><td rowspan=1 colspan=1>45.88</td><td rowspan=1 colspan=1>63.71</td><td rowspan=1 colspan=1>65.60</td><td rowspan=1 colspan=1>67.50</td><td rowspan=1 colspan=1>67.67</td><td rowspan=1 colspan=1>77.89</td><td rowspan=1 colspan=1>80.52</td><td rowspan=1 colspan=1>83.28</td><td rowspan=1 colspan=1>86.07</td><td rowspan=1 colspan=1>85.68</td><td rowspan=1 colspan=1>88.18</td></tr><tr><td rowspan=1 colspan=1>(0.84)</td><td rowspan=1 colspan=1>(6.24)</td><td rowspan=1 colspan=1>(3.48)</td><td rowspan=1 colspan=1>(3.10)</td><td rowspan=1 colspan=1>(1.72)</td><td rowspan=1 colspan=1>(1.37)</td><td rowspan=1 colspan=1>(1.00)</td><td rowspan=1 colspan=1>(3.57)</td><td rowspan=1 colspan=1>(1.65)</td><td rowspan=1 colspan=1>(1.01)</td><td rowspan=1 colspan=1>(2.38)</td><td rowspan=1 colspan=1>(1.67)</td><td rowspan=1 colspan=1>(3.99)</td><td rowspan=1 colspan=1>(3.55)</td><td rowspan=1 colspan=1>(1.57)</td><td rowspan=1 colspan=1>(0.65)</td><td rowspan=1 colspan=1>(3.24)</td><td rowspan=1 colspan=1>(3.41)</td></tr><tr><td rowspan=1 colspan=2>52.8931.80</td><td rowspan=1 colspan=1>69.11</td><td rowspan=1 colspan=1>62.04</td><td rowspan=1 colspan=1>76.60</td><td rowspan=1 colspan=1>73.33</td><td rowspan=1 colspan=1>57.19</td><td rowspan=1 colspan=1>51.07</td><td rowspan=1 colspan=1>66.57</td><td rowspan=1 colspan=1>66.18</td><td rowspan=1 colspan=1>68.09</td><td rowspan=1 colspan=1>66.99</td><td rowspan=1 colspan=1>76.67</td><td rowspan=1 colspan=1>79.23</td><td rowspan=1 colspan=1>91.92</td><td rowspan=1 colspan=1>92.91</td><td rowspan=1 colspan=1>88.31</td><td rowspan=1 colspan=1>90.33</td></tr><tr><td rowspan=1 colspan=3>(0.57)(1.54)(4.90)</td><td rowspan=1 colspan=1>(7.82)</td><td rowspan=1 colspan=2>(2.81)(3.83)</td><td rowspan=1 colspan=1>(0.35)</td><td rowspan=1 colspan=1>(0.41)</td><td rowspan=1 colspan=1>(0.09)</td><td rowspan=1 colspan=1>(0.54)</td><td rowspan=1 colspan=1>(0.49)</td><td rowspan=1 colspan=1>(0.94)</td><td rowspan=1 colspan=1>(4.48)</td><td rowspan=1 colspan=1>(4.23)</td><td rowspan=1 colspan=1>(3.88)</td><td rowspan=1 colspan=1>(3.05)</td><td rowspan=1 colspan=1>(9.57)</td><td rowspan=1 colspan=1>(7.39)</td></tr><tr><td rowspan=2 colspan=1>ADM</td><td rowspan=1 colspan=3>67.4762.3577.65</td><td rowspan=1 colspan=3>74.8479.9378.10</td><td rowspan=1 colspan=2>57.9554.11</td><td rowspan=1 colspan=1>68.08</td><td rowspan=1 colspan=1>66.07</td><td rowspan=1 colspan=1>67.67</td><td rowspan=1 colspan=1>66.34</td><td rowspan=1 colspan=2>|86.4985.52</td><td rowspan=1 colspan=3>91.5591.3390.75</td><td rowspan=1 colspan=1>90.50</td></tr><tr><td rowspan=1 colspan=3>(5.21) (3.76)(0.92)</td><td rowspan=1 colspan=3>(1.29)(1.09)(1.22)</td><td rowspan=1 colspan=1>(1.26)</td><td rowspan=1 colspan=1>(1.71)</td><td rowspan=1 colspan=1>(0.20)</td><td rowspan=1 colspan=1>(0.59)</td><td rowspan=1 colspan=1>(0.59)</td><td rowspan=1 colspan=1>(0.80)</td><td rowspan=1 colspan=2>(2.87)(3.36)</td><td rowspan=1 colspan=1>(1.60) ()</td><td rowspan=1 colspan=2>1.37) (1.23)</td><td rowspan=1 colspan=1>(0.40)</td></tr></table>

Table 4: Ablation studies of ADM on the AGNews dataset.
<table><tr><td rowspan="2">Metric</td><td colspan="2">ADM pre-stage1</td><td colspan="2">ADM pre-stage2</td><td colspan="2">ADM 0-threshold</td><td colspan="2">ADM</td></tr><tr><td> $k = 1 0$ </td><td> $k = 5 0$ </td><td> $k = 1 0$ </td><td> $k = 5 0$ </td><td> $k = 1 0$ </td><td> $k = 5 0$ </td><td> $k = 1 0$ </td><td> $k = 5 0$ </td></tr><tr><td>Acc</td><td>53.97</td><td>56.22</td><td>54.07</td><td>69.14</td><td>50.52</td><td>68.67</td><td>67.47</td><td>77.65</td></tr><tr><td>F1</td><td>36.79</td><td>41.08</td><td>40.22</td><td>62.30</td><td>33.47</td><td>67.96</td><td>62.35</td><td>74.84</td></tr><tr><td>In</td><td>52.68</td><td>64.42</td><td>53.02</td><td>69.18</td><td>50.00</td><td>84.47</td><td>71.49</td><td>82.60</td></tr><tr><td>Out</td><td>58.88</td><td>60.51</td><td>60.21</td><td>76.50</td><td>55.78</td><td>74.64</td><td>79.70</td><td>85.32</td></tr></table>

## 4.3 Experiment Results

Main Results. The open-set semi-supervised text classification results on benchmark datasets are shown in Table 3. From the table, we observe that ADM achieves the best performance in all settings of the AGNews dataset and most of the settings on Yahoo and DBPedia datasets and ADM achieves more improvement when k is smaller. These results demonstrate that our adversarial disagreement maximization approach is effective in OSTC and it is more effective in a challenging setting. Another observation is that LOS combined with UDA and MixText significantly improve the performance over pipeline models and ADM further improves LOS in most of the settings. These results manifest that unified training of semi-supervised classification and outlier detection can be better adapted to OSTC than the pipeline approach and directly maximizing measurements disagreement is more effective than existing optimization methods in OSTC.

Ablation Study. To analyze the contributions of each component and each training stage in ADM, we make ablation studies. Specifically, we compare 3 ablated models: ADM pre-stage1, pre-stage1 and ADM 0-threshold that disables the abnormalexample detection module with a setting of δ = 0. The ablated results show that pre-stage1 provides an initialized model for ADM and the pre-stage2 further improves upon pre-stage1. However, when training ADM without abnormal example detection, ADM 0-threshold performs even worse than in prestage2. When employing an appropriate abnormalexample detection threshold, ADM significantly improves OSTC results. This result demonstrates that abnormal-example detection and measurement calibration guarantees the effectiveness of ADM. ID Classification and OOD Detection Results. To analyze the models’ ability to classify ID examples and detect OOD examples, we report the ID classification accuracy and OOD detection accuracy in Table 5. Metrics In and Out are introduced to denote the ID accuracy on only ID examples and outlier-detection accuracy on all examples, respectively. The results in the table show that our ADM model outperforms existing OSTC models with large margins on the outlier detection (Out) accuracy results in most of the settings. These results demonstrate that the superiority of ADM is dominated by its good performance on OOD detection although it sacrifices ID classification results in some settings. These results suggest that ADM is effective in alleviating false positive inference.

Table 5: The evaluation results of accuracy on ID examples (In) and OOD detection accuracy on all examples (Out).
<table><tr><td rowspan="3">Method</td><td colspan="4">AGNews</td><td colspan="4">Yahoo</td><td colspan="5">DBPedia</td></tr><tr><td> $k = 1 0$ </td><td colspan="3"> $k = 5 0$ </td><td rowspan="2"> $k = 1 0 0$ </td><td rowspan="2"> $k = 1 0$ </td><td colspan="2"> $k = 5 0$ </td><td rowspan="2"> $k = 1 0 0$ </td><td colspan="2"> $k = 1 0$ </td><td colspan="3"> $k = 5 0$   $k = 1 0 0$ </td></tr><tr><td>In Out</td><td>In</td><td>Out In</td><td>Out</td><td>In Out</td><td>In Out</td><td>In Out</td><td>In Out</td><td>In</td><td>Out In</td></tr><tr><td>UDA+MSP</td><td>70.33</td><td>49.99 82.90</td><td>51.67</td><td>86.10</td><td>62.97</td><td>63.70 41.54</td><td>76.23</td><td>48.44 78.46</td><td>56.75|</td><td>|96.98 52.17</td><td>98.42</td><td>81.40</td><td>Out 98.55</td></tr><tr><td>UDA+LSoftmax</td><td>70.54 44.43</td><td></td><td>84.10 42.90</td><td>86.4746.37</td><td></td><td>63.8356.1277.66</td><td>55.58</td><td>79.53</td><td>55.94</td><td>96.98 55.38 98.52</td><td></td><td>56.06 98.57</td><td>81.82</td></tr><tr><td>UDA+DOC</td><td>66.2149.61</td><td>83.5</td><td>50.32</td><td>86.66 62.62</td><td></td><td>59.9540.70</td><td>75.91 40.01</td><td>78.7048.58</td><td>94.72 47.61</td><td>98.29</td><td>48.89</td><td>98.39</td><td>55.87</td></tr><tr><td>UDA+LMCL</td><td>70.4050.3383.55</td><td></td><td>47.19</td><td>86.07 51.10|</td><td></td><td>65.92 52.90 77.41</td><td>59.21</td><td>79.49 59.67</td><td>96.78 54.38</td><td></td><td></td><td></td><td>83.01</td></tr><tr><td>MixText+MSP</td><td></td><td>70.35 49.93 83.32 50.81</td><td></td><td>86.88 65.08</td><td></td><td></td><td></td><td>66.8240.7577.5448.9179.4855.72</td><td></td><td>95.79 53.23</td><td></td><td></td><td>98.45 57.02 98.49 57.20</td></tr><tr><td>MixText+LSoftmax</td><td></td><td>73.9044.2985.47</td><td>43.20</td><td>87.08</td><td>44.57</td><td>70.61 59.86 78.30</td><td></td><td>57.2879.13 56.84</td><td></td><td>97.26 55.13</td><td>97.9757.4798.32 58.53</td><td></td><td>98.10 81.30 98.3581.11</td></tr><tr><td>MixText+DOC</td><td></td><td>76.1749.9283.97</td><td>50.11</td><td>86.8260.48</td><td></td><td>64.5840.29</td><td>76.51 40.03</td><td>77.9347.90</td><td></td><td>95.7542.86 97.8847.6598.4681.90</td><td></td><td></td><td></td></tr><tr><td>MixText+LMCL</td><td>69.3545.1783.54 44.71</td><td></td><td></td><td>86.9755.01</td><td></td><td>67.86 55.15 78.50 59.88</td><td></td><td>879.9660.33</td><td></td><td>96.96 57.32</td><td>98.12 57.11 98.28 57.22</td><td></td><td></td></tr><tr><td>UDA+LOS</td><td>59.1860.3484.2377.78</td><td></td><td></td><td>86.2378.94</td><td></td><td>60.63 59.73</td><td>76.27</td><td>70.66 78.12 73.96</td><td></td><td>93.4979.53</td><td>96.06 84.40 97.55</td><td></td><td>86.40</td></tr><tr><td>MixText+LOS</td><td>61.9753.59</td><td>983.5070.48</td><td></td><td>87.27</td><td>77.97</td><td>69.3461.29</td><td>78.03 70.30</td><td>0 79.73 71.77</td><td></td><td>92.15 79.17</td><td>97.75</td><td>92.61 98.33</td><td>88.83</td></tr><tr><td>ADM</td><td></td><td>|71.49 79.70 82.60 85.32</td><td></td><td>84.62 87.06|</td><td></td><td>63.0372.8674.17</td><td></td><td>78.2176.82</td><td>77.71</td><td>95.16 88.20</td><td>97.78</td><td></td><td>92.54 97.71 91.90</td></tr></table>

Table 6: Comparison with Large Language Models.

Comparison with Large Language Models. We provide an empirical study using the large language model (LLM) LLaMA2-7B and in-context learning for OSTC in Table 6. Concretely, we design prompts that can guide the LLaMA2-7B model to complete classification and outlier detection, and we evaluate it in the 0-shot and 10-shot settings on the AGNews dataset. From the results, we can conclude that LLaMA2-7B can perform outlier detection and the OSTC task. However, in the 0-shot setting LLaMA2-7B, it performs poorly in both outlier detection and OSTC and achieves worse performance than baseline models. In the 10-shot setting of LLaMA2-7B, it achieves a good outlier detection result but a worse OSTC result than baseline models. And ADM outperforms LLaMA2-7B in both outlier detection and OSTC. These results demonstrate that in the OSTC task, carefully designed lightweight models may still be necessary and useful with the background of LLMs.

## 4.4 Experimental Analysis

Analysis on the Adversarial Learning Process. To study how our adversarial learning process works, we make an analysis of the changes in the losses during training. Specifically, we record the loss values of the maximization and minimization steps during training on the k = 10 and $k = 5 0$ settings of AGNews in Figure 2. From the figure, we observe that the loss values of the maximization and minimization steps respectively increase and decrease as the training progresses. These results demonstrate that our adversarial training approach is effective. Since the losses of the maximization and minimization steps can be respectively viewed as weighted summed measurements, the results demonstrate that the measurement disagreements are synergistically maximized.

<table><tr><td colspan="2">Model Outlier Detection OSTC</td></tr><tr><td>LLaMA2-7B (0-shot)</td><td>49.6 25.4</td></tr><tr><td> $\mathrm { L L a M A 2 { - } 7 B ( 1 0 { - } s h o t ) }$ </td><td>71.6 45.3</td></tr><tr><td> $\mathrm { U D A + L O S } \left( k = 1 0 \right)$ </td><td>60.3 56.4</td></tr><tr><td>Mixtext+LOS (k = 10)</td><td>53.4 52.9</td></tr><tr><td>ADM (k = 10)</td><td>79.7 67.4</td></tr></table>

![](images/2ce7812bfde2b82f3bac2532d7a166f884d87d6632bf87f265d70b825b6bf10b.jpg)  
(a) AGNews 10

![](images/33351bd894dd7b1d76171b50cb6a051d264be9d05cec5bda25c55d2757171338.jpg)  
(b) AGNews 50  
Figure 2: The smoothed loss values change trend in min-max optimization during training.

Analysis on the Abnormal-Example Detection Approach. In order to investigate the effectiveness of our abnormal-example detection approach during adversarial disagreement maximization, we report the percent of correctly identified normal and abnormal examples accumulated in previous training epochs in Figure 3. From the recording results, we observe that the percentage of correctly identified abnormal examples increase as the training progress. This result indicates that the abnormalexample detection approach is effective. In k = 10 setting, although the model sacrifices normal example identification performance, it significantly increase abnormal-example detection performance.

Table 7: Case study on four selected text examples from the AGNews dataset.
<table><tr><td rowspan=1 colspan=1>Id</td><td rowspan=1 colspan=1>Text</td><td rowspan=1 colspan=1>k</td><td rowspan=1 colspan=1>True Label</td><td rowspan=1 colspan=1>Pre-stage2</td><td rowspan=1 colspan=1>Initial M2</td><td rowspan=1 colspan=1>ADM M2</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Jennifer Canada knew she was entering a boy&#x27;s club when she en-olled in Southern Methodist University&#x27;s Guildhall school of vi-deo game making.</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>Sci/Tech</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.0706</td><td rowspan=1 colspan=1>0.9771</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>AP - The smallest man on the court always seems to make the big-gest plays for the Washington Huskies.</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>\</td><td rowspan=1 colspan=1>Business</td><td rowspan=1 colspan=1>0.8621</td><td rowspan=1 colspan=1>0.2369</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>The company said most cuts would come in its network operationsdivision, where work has become increasingly automated, and inthe customer service group.</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>Business</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.4134</td><td rowspan=1 colspan=1>0.9995</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>Myanmar&#x27;s Opposition National League for Democracy (NLD)party accused the military regime of endangering party leader AungSan Suu Kyi&#x27;s life by restricting her access to a doctor and non-juntasecurity.</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Sci/Tech</td><td rowspan=1 colspan=1>0.9963</td><td rowspan=1 colspan=1>0.0174</td></tr></table>

![](images/ec91f127c1fdebaa7409cdab2414f1b669fee9d9907ca5d263570339ad935693.jpg)  
(a) AGNews 10

![](images/5c90677dc4ee54dbd6c95aac6203ff61f7c2ba32e6b7cbca2144cecd56f12779.jpg)  
(b) AGNews 50  
Figure 3: The percent of correctly identified normal and abnormal examples accumulated in previous training epochs in k = 10 and k = 50 settings of AGNews.

Hyper-Parameter Analysis. To study how the hyper-parameters affect the training process, we make a hyper-parameter analysis. Specifically, we report the OSTC evaluation results of k = 10 and k = 50 settings on AGNews using different hyperparameter δ in Figure 4. From the results, we observe that the performance of ADM is sensitive to the hyper-parameter δ and an inappropriate configuration of δ may result in poor performance of ADM. This phenomenon indicates that abnormalexample detection and measurement calibration is necessary to guarantee the effectiveness of ADM.

![](images/d2a647f5b25ffc8557226410df1b24455680817aa43b52615466d4823c4c6155.jpg)

![](images/06c263150802d9930d8ce757b74a666eb6ae1286414b30c51e64ee5677f49f74.jpg)  
Figure 4: Hyper-parameter analysis of δ on AGNews.

## 4.5 Case Study

We make a case study in Table 7 on selected text examples to analyze the outlier detection ability of ADM. As shown in case 1 and case 3, the groundtrue labels are respectively Sci/Tech and Business, but the pre-stage2 model makes wrong predictions $( \mathcal { M } _ { 2 } < 0 . 5 )$ . After ADM training, the outlier detector successfully reverses the wrong predictions to the correct ones. Similarly, in case 2 and case 4, pre-stage2 incorrectly identifies the example as OOD. Nevertheless, ADM successfully rectifies the predictions, demonstrating its effectiveness.

## 5 Conclusion

We reveal the potential of measurement disagreement in OSTC. To fully employ the advantage of measurement disagreement in OSTC, we propose to directly maximize it and design a novel adversarial disagreement maximization (ADM) approach combined with abnormal-example detection to improve OSTC performance. Experiment results demonstrate the effectiveness of ADM.

## 6 Limitations

Although our measurement disagreement maximization model is demonstrated effective, it may have two limitations. First, ADM relies on a twostage pre-training process to obtain an initialized ID classifier and outlier detector, which guarantees the effectiveness of the min-max optimization. Second, ADM requires the proposed abnormal-example detection and measurement calibration approach to guarantee the correct optimization direction.

## Acknowledgments

This work was supported by the National Science and Technology Major Project under Grant 2022ZD0120202, in part by the National Natural Science Foundation of China (No. U23B2056 and No. 62306026), in part by China Postdoctoral Science Foundation (No. 2023M740184), in part by the Fundamental Research Funds for the Central Universities, and in part by the State Key Laboratory of Complex & Critical Software Environment.

## References

Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng, and Jörg Sander. 2000. LOF: identifying densitybased local outliers. In SIGMOD, pages 93–104. ACM.

Ming-Wei Chang, Lev-Arie Ratinov, Dan Roth, and Vivek Srikumar. 2008. Importance of semantic representation: Dataless classification. In AAAI, volume 2, pages 830–835.

Jiaao Chen, Zichao Yang, and Diyi Yang. 2020. Mixtext: Linguistically-informed interpolation of hidden space for semi-supervised text classification. In ACL, pages 2147–2157.

Junfan Chen, Richong Zhang, Junchi Chen, Chunming Hu, and Yongyi Mao. 2023. Open-set semisupervised text classification with latent outlier softening. In KDD, pages 226–236.

Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron C. Courville, and Yoshua Bengio. 2014. Generative adversarial nets. In NIPS, pages 2672–2680.

Suchin Gururangan, Tam Dang, Dallas Card, and Noah A. Smith. 2019. Variational pretraining for semi-supervised text classification. In ACL, pages 5880–5894. Association for Computational Linguistics.

Dan Hendrycks and Kevin Gimpel. 2017. A baseline for detecting misclassified and out-of-distribution examples in neural networks. In ICLR.

Junkai Huang, Chaowei Fang, Weikai Chen, Zhenhua Chai, Xiaolin Wei, Pengxu Wei, Liang Lin, and Guanbin Li. 2021. Trash to treasure: Harvesting OOD data with cross-modal matching for open-set semisupervised learning. In ICCV, pages 8290–8299.

Dong-Hyun Lee et al. 2013. Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks. In Workshop on challenges in representation learning, ICML, volume 3, page 896.

Ju Hyoung Lee, Sang-Ki Ko, and Yo-Sub Han. 2021. Salnet: Semi-supervised few-shot text classification with attention-based lexicon construction. In AAAI, pages 13189–13197.

Changchun Li, Ximing Li, and Jihong Ouyang. 2021. Semi-supervised text classification with balanced deep representation distributions. In ACL, pages 5044–5053.

Haoran Li, Chun-Mei Feng, Tao Zhou, Yong Xu, and Xiaojun Chang. 2022a. Prompt-driven efficient open-set semi-supervised learning. CoRR, abs/2209.14205.

Shujie Li, Min Yang, Chengming Li, and Ruifeng Xu. 2022b. Dual pseudo supervision for semi-supervised text classification with a reliable teacher. In SIGIR, pages 2513–2518.

Ting-En Lin and Hua Xu. 2019. Deep unknown intent detection with margin loss. In ACL, pages 5491– 5496.

Chen Liu, Mengchao Zhang, Zhibing Fu, Panpan Hou, and Yu Li. 2021. Flitext: A faster and lighter semisupervised text classification with convolution networks. In EMNLP, pages 2481–2491.

Yen-Cheng Liu, Chih-Yao Ma, Xiaoliang Dai, Junjiao Tian, Peter Vajda, Zijian He, and Zsolt Kira. 2022. Open-set semi-supervised object detection. In ECCV, pages 143–159.

Yu Meng, Jiaming Shen, Chao Zhang, and Jiawei Han. 2018. Weakly-supervised neural text classification. In CIKM, pages 983–992.

Takeru Miyato, Andrew M. Dai, and Ian J. Goodfellow. 2017. Adversarial training methods for semisupervised text classification. In ICLR.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9.

Kuniaki Saito, Donghyun Kim, and Kate Saenko. 2021. Openmatch: Open-set consistency regularization for semi-supervised learning with outliers. In NeurIPS.

Lei Shu, Hu Xu, and Bing Liu. 2017. DOC: deep open classification of text documents. In EMNLP, pages 2911–2916.

Antti Tarvainen and Harri Valpola. 2017. Weightaveraged consistency targets improve semisupervised deep learning results. CoRR, abs/1703.01780.

Austin Cheng-Yun Tsai, Sheng-Ya Lin, and Li-Chen Fu. 2022. Contrast-enhanced semi-supervised text classification with few labels. In AAAI, pages 11394– 11402.

Qizhe Xie, Zihang Dai, Eduard H. Hovy, Thang Luong, and Quoc Le. 2020. Unsupervised data augmentation for consistency training. In NeurIPS.

Hai-Ming Xu, Lingqiao Liu, and Ehsan Abbasnejad. 2022. Progressive class semantic matching for semisupervised text classification. In NAACL, pages 3003– 3013.

Guangfeng Yan, Lu Fan, Qimai Li, Han Liu, Xiaotong Zhang, Xiao-Ming Wu, and Albert Y. S. Lam. 2020. Unknown intent detection using gaussian mixture model with an application to zero-shot intent classification. In ACL, pages 1050–1060.

Weiyi Yang, Richong Zhang, Junfan Chen, Lihong Wang, and Jaein Kim. 2023. Prototype-guided pseudo labeling for semi-supervised text classification. In ACL, pages 16369–16382.

Qing Yu, Daiki Ikami, Go Irie, and Kiyoharu Aizawa. 2020. Multi-task curriculum framework for open-set semi-supervised learning. In ECCV, volume 12357, pages 438–454.

Ronghang Zhu and Sheng Li. 2022. Crossmatch: Crossclassifier consistency regularization for open-set single domain generalization. In ICLR.