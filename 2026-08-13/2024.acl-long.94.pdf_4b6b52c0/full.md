# Multimodal Prompt Learning with Missing Modalities for Sentiment Analysis and Emotion Recognition

Zirun Guo<sup>1,2</sup>, Tao Jin<sup>1</sup>∗, Zhou Zhao<sup>1,2</sup>

<sup>1</sup> Zhejiang University, <sup>2</sup> Shanghai Artificial Intelligence Laboratory zrguo.cs@gmail.com, {jint\_zju,zhaozhou}@zju.edu.cn

## Abstract

The development of multimodal models has significantly advanced multimodal sentiment analysis and emotion recognition. However, in real-world applications, the presence of various missing modality cases often leads to a degradation in the model’s performance. In this work, we propose a novel multimodal Transformer framework using prompt learning to address the issue of missing modalities. Our method introduces three types of prompts: generative prompts, missing-signal prompts, and missingtype prompts. These prompts enable the generation of missing modality features and facilitate the learning of intra- and inter-modality information. Through prompt learning, we achieve a substantial reduction in the number of trainable parameters. Our proposed method outperforms other methods significantly across all evaluation metrics. Extensive experiments and ablation studies are conducted to demonstrate the effectiveness and robustness of our method, showcasing its ability to effectively handle missing modalities. Codes are available at https://github.com/zrguo/MPLMM.

## 1 Introduction

Humans perceive the world in a multimodal way, such as sight, sound, touch and language. These multimodal features can provide comprehensive information to help us understand and explore the world. Thus, modeling and mining multimodal data is of great importance and has much potential. Recently, multimodal sentiment analysis (Tsai et al., 2019; Hazarika et al., 2020; Han et al., 2021; Hu et al., 2022) has attracted much attention. However, there are two main challenges in many existing methods: 1) Different from common multimodal tasks which only have two modalities (image and text), multimodal sentiment analysis task often has more modalities (video, audio, text, etc.). Therefore, in real-world scenarios, missing modality conditions always occur due to equipment failure, data corruption, privacy issues and the like, especially in low-resource domains, which could lead to a degradation in the model’s performance. Current multimodal models trained on complete data usually fail when tested on incomplete data (Aguilar et al., 2019; Pham et al., 2019). 2) With the success of large-scale multimodal models (Kim et al., 2021; Li et al., 2021; Radford et al., 2021), lots of researchers tend to finetune these large pre-trained models to downstream tasks. However, this kind of finetuning is infeasible for many researchers because it requires large computational resources. Besides, finetuning such a pre-trained model on small datasets could lead to instability (Mosbach et al., 2021).

Recently, prompt learning (Gao et al., 2021; Heinzerling and Inui, 2021; Khattak et al., 2023; Lee et al., 2023) is proposed, which freezes all the parameters of a pre-trained model while only finetuning several prompts and it has achieved great success (Lester et al., 2021). Motivated by prompt learning, in this paper, we intend to exploit a highresource dataset that contains relatively more complete modality data for pre-training and then leverage several trainable prompts to transfer the knowledge from high-resource domains to low-resource domains where missing modality cases often occur.

Previous works (Ma et al., 2021; Pham et al., 2019; Zhao et al., 2021) mainly focus on introducing sophisticated architecture to address the issue of missing modalities. These methods do not use pretrained models and usually require a lot of computational resources. However, our method is based on prompt learning, which only finetunes a few parameters of prompts. Lee et al. (2023) is a recent work which is similar to ours. However, its proposed missing-aware prompts increase exponentially with the number of modalities. In contrast, our proposed prompts increase linearly with the number of modalities which is more parameter-efficient. Specifically, we propose three types of prompts: generative prompts, missing-signal prompts, and missing-type prompts which can learn the representations of the missing modalities, cross-modal and fine-grained features. These three types of prompts play a combined role in improving the model’s performance.

We conduct extensive experiments on four datasets: CMU-MOSEI (Bagher Zadeh et al., 2018), CMU-MOSI (Zadeh et al., 2016), IEMO-CAP (Busso et al., 2008) and CH-SIMS (Yu et al., 2020). The proposed method outperforms the baselines significantly across all metrics on all datasets. We further study the roles of three types of prompts, the effect of missing rate of training data, and the effect of prompt length. We find that: 1) missingsignal prompts are modality-specific while missingtype prompts are modality-shared which represent intra-modality and inter-modality information respectively. 2) with short prompts, our model can achieve very good results which demonstrates our proposed method is parameter-efficient. 3) the missing rate is important for the performance of the model, with 70% being the optimal value.

Our contributions can be summarized as follows:

• We present a novel framework via prompt learning for sentiment analysis and emotion recognition which is not only computationally efficient but also capable of handling missing modalities during both the training and testing stages.

• The number of parameters of our proposed prompts is linearly related to the number of modalities, which significantly reduces computational resources.

• We propose three types of prompts to address the issue of missing modalities. These three types of prompts can generate missing information, and learn intra- and inter-modality information respectively.

• Our proposed method outperforms all the baselines across all metrics significantly. Furthermore, we discover that applying modality dropout with a rate of 70% during training yields the best enhancement in the model’s performance.

## 2 Related Works

Multimodal Sentiment Analysis (MSA) and Emotion Recognition (MER). Multimodal sen timent analysis and emotion recognition refer to the process of analyzing and understanding human sentiment or emotions using multiple modalities of data, such as text, image, audio, and video. The main challenge of such tasks is how to effectively use the information from different modalities to complement each other. Currently, there are two main multimodal fusion strategies: feature-level fusion and decision-level fusion. Feature-level fusion methods (Liang et al., 2018; Wang et al., 2019) combine features from different modalities to create a unified feature representation via concatenation or other methods. For example, Liang et al. (2018) decomposed the fusion problem into multiple stages and fused features step by step to obtain a comprehensive representation. Mai et al. (2019) conducted fusion hierarchically so that both local and global interactions are considered for a comprehensive interpretation of multimodal embeddings. Different from feature-level fusion methods, decision-level fusion methods (Tsai et al., 2019; Hazarika et al., 2020; Han et al., 2021; Hu et al., 2022) process different modalities independently and then incorporate them into the final decision. For instance, Tsai et al. (2019) proposed a directional pairwise cross-modal attention to implement modal alignment and fused the outputs of each modality at the decision level to make predictions. These methods all assume that the data is complete while our proposed method can deal with the situation when there exist missing modalities.

Multimodal Learning with Missing Modalities. The presence of a missing modality poses challenges for multimodal learning because the model needs to effectively handle the absence of information while still making accurate predictions. Ma et al. (2021) proposed the SMIL model which leverages Bayesian meta-learning to address the issue of missing modalities. Some methods (Cai et al., 2018; Du et al., 2018) directly generate missing modalities using the available modalities. Zhao et al. (2021) proposed learning robust joint multimodal representations which can predict the representation of any missing modality given the available modalities. However, these methods always introduced sophisticated architecture to address the issue of missing modalities, which is computationally expensive. In comparison, our approach utilizes three different prompts to handle missing modalities, which is computationally more efficient. In a more recent work (Lee et al., 2023), prompts are used to address missing modalities, but the number of prompts increases exponentially with the number of modalities. In contrast, the number of prompts in our method is linearly related to the number of modalities.

![](images/ccf0e140b10dd12d328b0661e2309de113d8b421c9104d04977e16fe7b7a4cfd.jpg)  
Figure 1: The overall architecture of our proposed method. A batch of data that contains different missing modality cases is fed to the Missing Modality Generation Module (see Section 3.2) to obtain generated features. They are then passed to the pre-trained backbone with missing-signal prompts and missing-type prompts (see Section 3.3).

Prompt Learning. Prompt learning, which refers to the process of designing or generating effective prompts to use a pre-trained model for different types of downstream tasks, has been widely used in various NLP tasks (Gao et al., 2021; Heinzerling and Inui, 2021). With the success of prompt learning in NLP tasks (Lester et al., 2021; Li and Liang, 2021; Liu et al., 2022), recent works (Tsimpoukelli et al., 2021; Liang et al., 2022; Khattak et al., 2023) explored to leverage prompts in multimodal learning. Tsimpoukelli et al. (2021) presented a method for transforming large language models into multimodal systems by extending the soft-prompting philosophy of prefix tuning to ordered sets of images and texts. Khattak et al. (2023) proposed a strategy to ensure synergy between vision-language modalities by explicitly conditioning the vision prompts on textual prompts across different Transformer stages. More recently, Lee et al. (2023) proposed missing-aware prompts to address missing modalities which increase the robustness of the model, but it did not recover the missing information from the multimodal input. In comparison, our approach utilizes generative prompts to generate the representation of missing modalities given available modalities which can help further boost the performance of the model.

## 3 Proposed Method

In this section, we describe our proposed method (Figure 1) via prompt learning to address the issue of missing modalities (introduced in Section 3.1). Specifically, we introduce three kinds of prompts: generative prompts (introduced in Section 3.2), missing-signal prompts, and missing-type prompts (introduced in Section 3.3).

## 3.1 Overall Architecture

Problem Definition. Given a multimodal dataset consisting of $M \ : = \ : 3$ modalities (e.g., audio, video and text), we use $\pmb { x } = ( x ^ { a } , x ^ { v } , x ^ { t } )$ to represent a pair of features in , where $x ^ { a } , x ^ { v } , x ^ { t }$ represent the features of acoustic, visual and textual modalities respectively. To indicate missing modalities, we use $x ^ { a m } , x ^ { v m } , x ^ { t m }$ to denote which modalities are absent.

Figure 1 shows the overall architecture of our proposed model. For simplicity and better comparison, we use MulT (Tsai et al., 2019) as the backbone, which introduced the Crossmodal Transformer for modeling unaligned data. In our proposed method, we employ three types of different prompts: generative prompts, missing-signal prompts, and missing-type prompts. The generative prompts assist the available modalities in generating representations for the missing modalities. The missing-signal prompts are designed to inform the model about the absence of a specific modality while the missing-type prompts inform the model about the absence of other modalities.

![](images/12bd96ef751bc36d99fa1d716effda902d23eaac0af4c0b8e37f1ff505c05db3.jpg)  
Figure 2: The illustration of Missing Modality Generation Module (MMGM). The figure shows the process of generating the audio feature of an example of $\pmb { x } = ( x ^ { a m } , x ^ { v } , x ^ { t } )$ where the audio modality is missing and the other two are not. It can be described using the Equation 1.

## 3.2 Missing Modality Generation Module (MMGM)

Many methods address missing modality issues by recovering missing information using available modalities (Cai et al., 2018; Du et al., 2018). However, these methods often utilize complex structures. Based on this observation, we propose the Missing Modality Generation Module (MMGM) which utilizes generative prompts to recover missing information in a much simpler way. We denote generative prompts as $P _ { G } = ( P _ { G a } , P _ { G v } , P _ { G t } )$ where $P _ { G a } , P _ { G v }$ and $P _ { G t }$ represent the generative prompts for the audio, video and text modalities, respectively. $P _ { G } \in \mathbb { R } ^ { 3 \times d _ { p } \times \ell _ { p } }$ where $d _ { p }$ and $\ell _ { p }$ represent the dimension and length of the prompts respectively. Figure 2 illustrates the MMGM. Given $\pmb { x } = ( x ^ { a m } , x ^ { v } , x ^ { t } )$ , we can generate the representation of the missing modality $x ^ { a m }$ using the available $x ^ { v }$ and $x ^ { t }$ according to the following equation:

$$
\hat { x } ^ { a } = f _ { v t  \hat { a } } ( [ P _ { G a } , f _ { v  a } ( x ^ { v } ) , f _ { t  a } ( x ^ { t } ) ] )\tag{1}
$$

where ${ \hat { x } } ^ { a }$ denotes the representation generated, $[ \ldots ]$ represents the concatenation operation, $f ( \cdot )$ represents a Conv block which consists of a Conv 1D layer and an activation function and represents from one or two modalities to another modality. If there are two missing modalities, such as $\pmb { x } = ( x ^ { a m } , x ^ { v m } , x ^ { t } )$ , the generation process is as follows:

$$
\begin{array} { r } { \hat { x } ^ { a } = f _ { t  \hat { a } } ( [ P _ { G a } , f _ { t  a } ( x ^ { t } ) ] ) } \\ { \hat { x } ^ { v } = f _ { t  \hat { v } } ( [ P _ { G v } , f _ { t  v } ( x ^ { t } ) ] ) } \end{array}\tag{2}
$$

![](images/f653694e30430b855f43b757f6a901db3442ae189d4aa898eca954f78b207c5e.jpg)  
Figure 3: The illustration of attaching missing-type prompts to the Transformer. With the missing-type matrix $\mathbf { M _ { P } }$ , we generate missing-type prompts $P _ { M T } ^ { \prime }$ for different missing modality cases. The figure shows the process of attaching missing-type prompts using an example of $\pmb { x } = ( x ^ { a m } , x ^ { v } , x ^ { t m } )$ where audio and text modalities are missing.

After applying the MMGM, we can represent the generated features as $\mathbf { \Delta x } = ( \hat { x } ^ { a } , \hat { x } ^ { v } , x ^ { t } )$

## 3.3 Missing-signal and Missing-type Prompts

MMGM recovers missing information using available modalities. However, the information generated sometimes might not be accurate and could mislead the model. Therefore, missing-signal prompts are designed to inform the corresponding Transformer whether the information for a particular modality is real or generated. For each modality, there are two missing-signal prompts: $P _ { M S }$ to denote a modality is missing and $P _ { N M S }$ to denote a modality is not missing. As depicted in Figure 1, after the MMGM and the Conv 1D layer, we obtain features $\pmb { x } = ( \hat { x } ^ { a } , x ^ { v } , x ^ { t } )$ where the audio modality is missing originally. We can incorporate the missing-signal prompts as follows:

$$
\begin{array} { r l } & { \hat { x } ^ { a } : = \hat { x } ^ { a } + P _ { M S } ^ { a } } \\ & { x ^ { v } : = x ^ { v } + P _ { N M S } ^ { v } } \\ & { x ^ { t } : = x ^ { t } + P _ { N M S } ^ { t } } \end{array}\tag{3}
$$

After applying missing-signal prompts, the model knows which modalities are generated and which modalities are real, which can help the model make better use of the recovered information. Notably, missing-signal prompts are modality-specific which means that this kind of prompt only considers a specific modality and does not take into account the correlations between the absence of multiple modalities. To address this limitation, we propose missing-type prompts.

If there are M modalities, there can be a total of $2 ^ { M } - 1$ different cases of missing modalities. One intuitive approach is to design $2 ^ { M } - 1$ prompts to handle each situation individually (Lee et al., 2023). However, as the number of modalities increases, this approach becomes computationally expensive. Therefore, we introduce a missing-type projection matrix M . We can obtain M of $\pmb { x } = ( x ^ { a m } , x ^ { v } , x ^ { t m } )$ as follows:

$$
\mathbf { M _ { P } } = \mathbf { M _ { a } } \cdot P _ { M S } ^ { a } + \mathbf { M _ { v } } \cdot P _ { N M S } ^ { v } + \mathbf { M _ { t } } \cdot P _ { M S } ^ { t }\tag{4}
$$

where is the matrix multiplication, $\mathbf { M } _ { \mathbf { a } } , \mathbf { M } _ { \mathbf { v } }$ $\mathbf { M _ { t } } \in \mathbb { R } ^ { d _ { p } \times \ell _ { p } }$ and $\mathbf { M _ { p } } \in \mathbb { R } ^ { d _ { p } \times d _ { p } }$ . Then, we can get the missing-type prompts $P _ { M T } ^ { \prime }$ as follows:

$$
P _ { M T } ^ { \prime } = P _ { M T } \cdot \mathbf { M _ { P } }\tag{5}
$$

where $P _ { M T }$ represents the original missing-type prompts, $P _ { M T } ^ { \prime }$ represents the projected missingtype prompts and $P _ { M T } , P _ { M T } ^ { \prime } \ \in \ \mathbb { R } ^ { 3 \times \ell _ { p } \times d _ { p } }$ . In Figure 3, we illustrate how to attach the missingtype prompts to the Transformer with an example of $\pmb { x } = ( x ^ { a m } , x ^ { v } , x ^ { t m } )$ . For each data pair in $\mathcal { D } _ { \mathbf { ; } }$ the corresponding missing-type modality prompt is attached according to the situation of missing modalities.

## 4 Experiments

## 4.1 Datasets and Evaluation Metrics

To simulate real-world scenarios, we select CMU-MOSEI (Bagher Zadeh et al., 2018) as the highresource dataset while CMU-MOSI (Zadeh et al., 2016), IEMOCAP (Busso et al., 2008) and CH-SIMS (Yu et al., 2020) are selected as the lowresource datasets. We pre-train our backbone on CMU-MOSEI and evaluate our proposed method on the four datasets.

CMU-MOSI is a popular dataset for multimodal (audio, text and video) sentiment analysis, comprising 93 English YouTube. Each segment is manually annotated with a sentiment score ranging from strongly negative to strongly positive (-3 to +3).

CMU-MOSEI is an extension of CMU-MOSI. It contains more than 65 hours of annotated video from more than 1000 speakers and 250 topics. Compared with CMU-MOSI, it covers a wider range of topics.

IEMOCAP contains recorded videos from ten actors in five dyadic conversation sessions. It contains approximately 12 hours of data. Following previous works (Wang et al., 2019; Tsai et al., 2019), four emotions (happiness, anger, sadness and neutral state) are selected for emotion recognition.

CH-SIMS is a Chinese multimodal sentiment analysis dataset. It contains 2,281 video segments annotated with a sentiment score ranging from strongly negative to strongly positive (-1 to 1).

For CMU-MOSI and CMU-MOSEI, we follow previous works and adopt 7-class accuracy (ACC-7), binary accuracy (ACC), F1 score (F1), mean absolute error (MAE) and Pearson correlation (Corr) as evaluation metrics. For IEMOCAP, we implement four binary classification tasks and use the average accuracy (ACC) and F1-weighted score (F1) as evaluation metrics. For CH-SIMS, we use binary accuracy (ACC), F1 score (F1), mean absolute error (MAE) and Pearson correlation (Corr).

## 4.2 Baselines

We compare our proposed method with the following methods: Lower Bound (LB) is trained with different combinations of modalities. Specifically, we train six different models using different combinations of modalities. Modality Substitution (MS) substitutes missing modality with a default value or a placeholder. Modality Dropout (MD) is a model trained with randomly dropped modalities during the training phase. MCTN (Pham et al., 2019) learns robust joint representations by translating between modalities to deal with missing information. MMIN (Zhao et al., 2021) learns robust joint multimodal representations, which can predict the representation of any missing modality given available modalities. MPMM (Lee et al., 2023) uses missing-aware prompts to instruct the model to address missing modality issues.

## 4.3 Implementation Details

Raw Feature Extraction. To demonstrate the generalization ability of our method, we implement three kinds of methods to extract features. For CMU-MOSEI and CMU-MOSI, we follow Tsai et al. (2019) to extract features. For IEMOCAP, we follow Zhao et al. (2021) to extract acoustic, visual and textual features. For CH-SIMS, we follow Yu et al. (2020) to extract features.

Model Training Details. We first pre-train our backbone MulT (Tsai et al., 2019) on the CMU-MOSEI dataset. Then, we freeze all the parameters of the backbone and only train several learnable prompts, Conv layers and the output layer (As shown in Figure 1). The length of prompts $\ell _ { p }$ is set to 16 by default. We use L1 loss function for CMU-

Table 1: Quantitative results under six possible missing modality cases. For example, " a " means audio modality is available while video and text are missing. "Avg." means the average performance of the six possible cases. denotes results copied from Zhao et al. (2021) where F1 score is not reported. Bold: best result. Underline: second best result. We report the average result of five different random seeds.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="2">{a}</td><td colspan="2">{v}</td><td colspan="2">{t}</td><td colspan="2">{a, v}</td><td colspan="2">{a,t}</td><td colspan="2">{v, t}</td><td colspan="2">Avg.</td></tr><tr><td>ACC</td><td>F1</td><td>ACC</td><td>F1</td><td>ACC</td><td>F1</td><td>ACC</td><td>F1</td><td>ACC</td><td>F1</td><td>ACC</td><td>F1</td><td>ACC</td><td>F1</td></tr><tr><td rowspan="7">MOSI</td><td>LB</td><td>48.32</td><td>55.81</td><td>49.09</td><td>55.20</td><td>79.27</td><td>79.22</td><td>50.07</td><td>57.12</td><td>78.67</td><td>79.25</td><td>79.86</td><td>79.96</td><td>64.21</td><td>67.76</td></tr><tr><td>MS</td><td>49.17</td><td>55.34</td><td>49.87</td><td>56.12</td><td>78.06</td><td>78.28</td><td>51.12</td><td>57.01</td><td>79.32</td><td>79.65</td><td>80.32</td><td>80.38</td><td>64.64</td><td>67.80</td></tr><tr><td>MD</td><td>48.79</td><td>55.74</td><td>49.66</td><td>55.60</td><td>79.36</td><td>80.01</td><td>52.33</td><td>56.84</td><td>79.59</td><td>79.86</td><td>80.51</td><td>80.43</td><td>65.04</td><td>68.08</td></tr><tr><td>MCTN</td><td>51.32</td><td>56.12</td><td>54.27</td><td>56.33</td><td>79.63</td><td>79.78</td><td>56.79</td><td>57.84</td><td>78.96</td><td>79.17</td><td>80.45</td><td>80.65</td><td>66.90</td><td>68.32</td></tr><tr><td>MMIN</td><td>59.16</td><td>60.12</td><td>61.01</td><td>61.98</td><td>80.10</td><td>80.16</td><td>63.79</td><td>64.08</td><td>80.50</td><td>80.33</td><td>80.46</td><td>80.63</td><td>70.84</td><td>71.22</td></tr><tr><td>MPMM</td><td>57.26</td><td>59.35</td><td>58.63</td><td>59.12</td><td>79.81</td><td>80.10</td><td>60.54</td><td>61.33</td><td>79.89</td><td>79.84</td><td>80.74</td><td>80.93</td><td>69.48</td><td>70.11</td></tr><tr><td>Ours</td><td>62.71</td><td>63.65</td><td>63.12</td><td>63.74</td><td>80.12</td><td>80.31</td><td>65.02</td><td>65.41</td><td>80.76</td><td>81.09</td><td>81.12</td><td>81.19</td><td>72.14</td><td>72.57</td></tr><tr><td rowspan="7">IEMOCAP</td><td>LB</td><td>46.35</td><td>46.21</td><td>48.07</td><td>47.58</td><td>56.06</td><td>55.28</td><td>58.12</td><td>57.89</td><td>72.18</td><td>72.25</td><td>65.63</td><td>65.28</td><td>57.74</td><td>57.42</td></tr><tr><td>MS</td><td>47.65</td><td>47.52</td><td>47.68</td><td>47.36</td><td>59.27</td><td>59.22</td><td>57.48</td><td>56.60</td><td>72.30</td><td>72.18</td><td>66.81</td><td>66.93</td><td>58.53</td><td>58.30</td></tr><tr><td>MD MCTN</td><td>48.22</td><td>48.09</td><td>48.26</td><td>47.98</td><td>61.26</td><td>61.28</td><td>58.08</td><td>57.96</td><td>72.40</td><td>72.31</td><td>67.08</td><td>68.22</td><td>59.22</td><td>59.31</td></tr><tr><td>MMIN</td><td>51.62† 59.00†</td><td></td><td>45.73†</td><td></td><td>63.78†</td><td></td><td>55.84†</td><td></td><td>69.46†</td><td></td><td>68.34†</td><td></td><td>59.19†</td><td></td></tr><tr><td>MPMM</td><td>58.69</td><td></td><td>51.60†</td><td></td><td>68.02†</td><td></td><td>65.43†</td><td></td><td>75.14†</td><td></td><td>73.61†</td><td></td><td>65.47†</td><td></td></tr><tr><td>Ours</td><td>59.77</td><td>57.66 59.71</td><td>55.18 57.61</td><td>55.36 56.98</td><td>68.39 69.23</td><td>68.08</td><td>63.68</td><td>63.47</td><td>74.90</td><td>74.98</td><td>73.80</td><td>72.67</td><td>65.77</td><td>65.37</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>69.28</td><td>67.26</td><td>67.37</td><td>75.98</td><td>75.44</td><td>74.68</td><td>74.51</td><td>67.42</td><td>67.22</td></tr><tr><td rowspan="8">CH-SIMS</td><td>LB</td><td>63.82</td><td>75.15</td><td>64.08</td><td>78.11</td><td>76.74</td><td>76.90</td><td>62.14</td><td>73.21</td><td>76.84</td><td>76.93</td><td>77.01</td><td>77.13</td><td>70.11</td><td>76.24</td></tr><tr><td>MS</td><td>62.45</td><td>74.59</td><td>63.58</td><td>76.86</td><td>77.28</td><td>77.84</td><td>60.18</td><td>71.09</td><td>76.01</td><td>76.30</td><td>77.13</td><td>77.20</td><td>69.44</td><td>75.65</td></tr><tr><td>MD</td><td>64.22</td><td>77.25</td><td>63.87</td><td>76.01</td><td>77.34</td><td>77.48</td><td>62.91</td><td>72.14</td><td>76.77</td><td>76.92</td><td>77.14</td><td>77.31</td><td>70.38</td><td>76.19</td></tr><tr><td>MCTN</td><td>64.39</td><td>76.48</td><td>64.12</td><td>76.34</td><td>77.78</td><td>77.92</td><td>63.47</td><td>73.11</td><td>76.68</td><td>76.71</td><td>77.21</td><td>77.36</td><td>70.61</td><td>76.32</td></tr><tr><td>MMIN</td><td>65.21</td><td>77.09</td><td>65.32</td><td>77.41</td><td>78.91</td><td>78.67</td><td>64.28</td><td>73.36</td><td>77.32</td><td>77.33</td><td>77.40</td><td>77.48</td><td>71.41</td><td>76.89</td></tr><tr><td>MPMM</td><td>64.98</td><td>76.41</td><td>65.40</td><td>77.92</td><td>78.56</td><td>78.65</td><td>64.01</td><td>73.47</td><td>77.11</td><td>77.20</td><td>77.51</td><td>77.47</td><td>71.26</td><td>76.85</td></tr><tr><td>Ours</td><td>65.93</td><td>77.10</td><td>66.02</td><td>78.86</td><td>79.75</td><td>78.74</td><td>65.28</td><td>74.02</td><td>77.45</td><td>77.84</td><td>77.97</td><td>77.95</td><td>72.07</td><td>77.42</td></tr><tr><td>LB</td><td>66.21</td><td>68.69</td><td>66.45</td><td>69.10</td><td>77.96</td><td>78.32</td><td>67.30</td><td>69.62</td><td>78.13</td><td>78.63</td><td>77.86</td><td>78.16</td><td>72.32</td><td>73.83</td></tr><tr><td rowspan="7">MOSEI</td><td>MS</td><td>62.74</td><td>67.06</td><td>64.16</td><td>68.17</td><td>77.28</td><td>77.76</td><td>67.11</td><td>69.51</td><td>78.34</td><td>78.80</td><td>78.08</td><td>78.62</td><td>71.29</td><td>73.36</td></tr><tr><td>MD</td><td>65.76</td><td>68.18</td><td>66.57</td><td>69.35</td><td>77.30</td><td>77.94</td><td>67.21</td><td>69.48</td><td>78.74</td><td>78.97</td><td>78.11</td><td>78.71</td><td>72.28</td><td>73.82</td></tr><tr><td>MCTN</td><td>66.19</td><td>68.58</td><td>66.70</td><td>69.01</td><td>78.32</td><td>78.41</td><td>68.10</td><td>69.34</td><td>79.11</td><td>79.14</td><td>78.65</td><td>78.64</td><td>72.85</td><td>73.94</td></tr><tr><td>MMIN</td><td>67.11</td></table>

MOSEI, CMU-MOSI and CH-SIMS datasets and cross-entropy loss for IEMOCAP dataset. In all experiments, we use Adam optimizer with a batch size of 64. We train the model for 30 epochs with a learning rate of $1 \times 1 0 ^ { - 3 }$ . Besides, we randomly discard the modality of the data with a missing rate of $\eta = 7 0 \%$ during training. We fix the random seed to ensure that each model is trained on the same data.

## 4.4 Main Results

In Table 1, we present quantitative results on four datasets. The baselines LB, MS, MD and MPMM share the backbone of our method, thus reflecting the effectiveness of our proposed method to deal with missing modalities. Comparing the baseline MS and MD, we find that random discarding of data modalities during training improves the generalization ability of the model, thus making the model less sensitive to the data with missing modalities during the test phase.

Analyzing the results presented in the table, we observe that our proposed method outperforms the baselines by a large margin in all datasets under all six missing modality cases. Additionally, our method brings great enhancement when text modality is missing, with 8-13% increase in accuracy compared with the LB baseline. This indicates that the three types of proposed prompts effectively guide the pre-trained model and yield impressive performance improvements.

Besides, it is worth noting that we implement different feature extraction approaches on four datasets. From the results in Table 1, we can see that our model outperforms the baselines on all datasets, which shows that our model can adapt to features extracted by different methods. This indicates that our model learns relative rather than absolute relationships between features, demonstrating its robustness and versatility.

In Figure 4, we further compare the performance of our model with other methods under different modality missing rates during test time. From the figure, we find that our model performs better than all the other methods across all metrics under different modality missing rates, although the model is trained using the dataset with a modality missing rate of $\eta = 7 0 \%$ . This indicates that our proposed method is robust to the missing rate of test set and can deal with severely missing modalities well.

![](images/eb2121eeb1a0383c48ab395bd9899920a82ed0eca3896129eae451f23cdce3df.jpg)

(a)  
![](images/b1d3bd43c87e0db86df4a33466ee9639a0a9c4b822bb4d28a5cd5efafcf5b55a.jpg)  
(e)

![](images/85c484e8819e34ae8d0f73a4b8a9d69eada45c55486cb759bcc47369578e548e.jpg)

(b)  
![](images/01879aeba9a640cefc44d5c9203641c770a66c54255f51a39cba2604c200af31.jpg)  
(f)

![](images/1145d9ae7493fe0cdc9aa67f839dbb767f4c5cc34900c405974c4ea85aa93e59.jpg)

(c)  
![](images/f2c20139ea1dfa157b2c82ce64fa9754982380f30754bfb59fb66ff752fce26f.jpg)

![](images/06adbb30ee8cc09d3f7adc1b8a12ea495913ea043a1281d5bb954c6310f7a039.jpg)

(d)  
(g)  
![](images/91f129f92bab3d1d2cddcded6d8206ed8ca897528039723bc311f1e7769104fd.jpg)  
(h)  
Figure 4: Performance comparison with different modality missing rates during tests. (a): ACC on CMU-MOSI. (b): F1 score on CMU-MOSI (c): MAE on CMU-MOSI. (d): Corr on CMU-MOSI. (e): ACC on IEMOCAP. (f): F1 score on IEMOCAP. (g): ACC on CH-SIMS. (h): F1 score on CH-SIMS.

Furthermore, the number of trainable parameters of our method is about 5-10% percent of that of the backbone. The majority of trainable parameters come from the Conv layers in MMGM. The number of trainable parameters of three types of prompts only accounts for 0.5-1% of that of the backbone. Notably, the number of trainable parameters does not increase with the size of the backbone network, which means that even if we use a much larger backbone, the number of trainable parameters remains the same. In all our experiments, we use only one 10 GB GPU (RTX 3080) with a batch size of 64. This demonstrates that our method is parameter-efficient.

## 4.5 Generalization Ability

To further validate the generalization ability of our method, we conduct experiments using different MSA/MER backbones. Specifically, we conduct experiments using MISA (Hazarika et al., 2020), MMIM (Han et al., 2021) and UniMSE (Hu et al., 2022) and present the results in Table 2. For all three backbones, we insert our generative prompts and module after the feature extractors. For UniMSE, we insert missing-signal and missingtype prompts into its multimodal fusion layers. For MMIM and MISA, we insert missing-signal and missing-type prompts into their modality-specific encoders and fusion layers, respectively. The results in the table demonstrate our method can enhance the ability of various backbones to address missing modality issues. Besides, our prompts can enhance the performance in the complete data situation, indicating that our missing-signal and missing-type prompts can help the model learn intra-modality and inter-modality features.

Table 2: Performance on different backbones on the CMU-MOSI dataset. "Com" denotes the complete data. "Incom" denotes the incomplete data. denotes the method attached with prompts. For incomplete data, we report the average accuracy of six different missing conditions.
<table><tr><td>Backbone</td><td>Com</td><td>Com†</td><td>Incom</td><td>Incom†</td></tr><tr><td>MISA</td><td>80.8</td><td> $8 1 . 4 _ { ( + 0 . 6 ) }$ </td><td>67.9</td><td> $7 3 . 4 _ { ( + 5 . 5 ) }$ </td></tr><tr><td>MMIM</td><td>84.1</td><td> $8 4 . 9 _ { ( + 0 . 8 ) }$ </td><td>68.6</td><td> $7 2 . 3 _ { ( + 3 . 7 ) }$ </td></tr><tr><td>UniMSE</td><td>86.9</td><td> $8 7 . 4 _ { ( + 0 . 5 ) }$ </td><td>69.8</td><td> $7 5 . 1 _ { ( + 5 . 3 ) }$ </td></tr></table>

## 4.6 Ablation Study

We divide our ablation experiments into three parts: contributions of three types of prompts, the effect of modality missing rate during training and the effect of prompt length.

Contributions of three types of prompts. In Table 3, we present quantitative results of the contributions of different prompts. For CMU-MOSI, we can observe that generative prompts give the greatest improvement in ACC and F1, while missingsignal prompts improve the Corr the most and missing-type prompts improve the ACC-7 the most.

Table 3: An ablation study on the benefit of the proposed generative prompts $P _ { G } ,$ , missing-signal prompts $P _ { M S }$ and missing-type prompts $P _ { M T } . \ V$ represents a model with such type of prompts. Bold: best results with two kinds of prompts attached. \* denotes best results with only one kind of prompt attached. We report the average performance of the six possible missing modality cases.
<table><tr><td rowspan="2"> $P _ { G }$ </td><td rowspan="2"> $P _ { M S }$ </td><td rowspan="2"> $P _ { M T }$ </td><td colspan="5">CMU-MOSI</td><td colspan="4">CH-SIMS</td><td colspan="2">IEMOCAP</td></tr><tr><td>ACC-7</td><td>ACC</td><td>F1</td><td>MAE</td><td>Corr</td><td>ACC</td><td>F1</td><td>MAE</td><td>Corr</td><td>ACC</td><td>F1</td></tr><tr><td rowspan="4">√</td><td rowspan="4"></td><td></td><td>27.41</td><td>65.04</td><td>68.08</td><td>1.121</td><td>0.549</td><td>70.38</td><td>76.19</td><td>0.555</td><td>0.506</td><td>59.22</td><td>59.31</td></tr><tr><td></td><td>28.27</td><td>70.26*</td><td>71.33*</td><td>1.097*</td><td>0.574</td><td>71.08*</td><td>76.57*</td><td>0.518*</td><td>0.521</td><td>64.12*</td><td>63.97*</td></tr><tr><td></td><td>28.11</td><td>67.62</td><td>69.40</td><td>1.104</td><td>0.580*</td><td>70.61</td><td>76.21</td><td>0.531</td><td>0.527*</td><td>62.27</td><td>62.31</td></tr><tr><td>√</td><td>30.17*</td><td>66.41</td><td>69.21</td><td>1.167</td><td>0.561</td><td>70.49</td><td>76.13</td><td>0.542</td><td>0.515</td><td>63.48</td><td>63.20</td></tr><tr><td rowspan="3">√ √</td><td colspan="2">√</td><td>30.63</td><td>71.94</td><td>72.10</td><td>1.084</td><td>0.576</td><td>71.84</td><td>77.13</td><td>0.496</td><td>0.536</td><td>65.23</td><td>65.01</td></tr><tr><td></td><td>√</td><td>31.27</td><td>71.60</td><td>71.86</td><td>1.075</td><td>0.583</td><td>71.36</td><td>76.95</td><td>0.498</td><td>0.533</td><td>66.11</td><td>66.04</td></tr><tr><td>√</td><td>√</td><td>32.88</td><td>67.56</td><td>69.31</td><td>1.091</td><td>0.593</td><td>70.92</td><td>76.34</td><td>0.521</td><td>0.541</td><td>64.92</td><td>64.67</td></tr><tr><td rowspan="4"></td><td colspan="7"></td><td> $\overline { { \mathbf { P } _ { \mathbf { G } } } }$ </td><td> ${ \bf P } _ { \mathrm { M S } }$   $\mathbf { P } _ { \mathrm { M T } }$ </td><td rowspan="6"></td><td colspan="2"></td></tr><tr><td rowspan="3"></td><td rowspan="3"></td><td rowspan="3"></td><td rowspan="3"></td><td rowspan="3"></td><td rowspan="3"></td><td rowspan="3"></td><td rowspan="3"></td><td rowspan="3"></td><td colspan="2" rowspan="2"></td><td rowspan="2"></td></tr><tr><td colspan="2" rowspan="2"></td></tr><tr><td colspan="3"></td></tr><tr><td rowspan="6"></td><td colspan="5" rowspan="2"></td><td></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td></tr><tr><td colspan="2"><img src="images/7e3458da61063dc740449a10795740423fe9ff13b2db6020684d1969a4fe6621.jpg"/></td></tr><tr><td colspan="5"></td><td></td><td></td><td></td><td colspan="2"></td><td colspan="2"></td></tr><tr><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td></tr><tr><td></td></tr><tr><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td></tr><tr><td></td></tr></table>

Figure 5: The effectiveness of three types of prompts on an example of CH-SIMS. The ground truth of the sample is "Negative". We report the results when the visual modality is missing.

This indicates that generative prompts can help the available modalities generate the missing information which improves the binary accuracy. Besides, missing-type prompts tell the model whether other modalities are missing, thus strengthening the interactions between different modalities and learning cross-modal and fine-grained information which helps improve the ACC-7 a lot.

From the performance of models with different combinations of three types of prompts, we can further demonstrate the different roles of three types of prompts. We can conclude that generative prompts learn good representations of the missing modalities and improve the binary accuracy, missing-signal prompts are modality-specific prompts that tell models whether the corresponding modality is missing and help improve the correlation of the model’s predictions with humans, and missing-type prompts are shared prompts with inter-modality information, thus helping models learn cross-modal and fine-grained information that improves ACC-7. Furthermore, the combinations of three types of prompts further enhance the performance of the model on all datasets. This fully confirms the validity of our proposed method. Besides, we use an example in CH-SIMS to study the effectiveness of three types of prompts and present the results in Figure 5. From the figure, we can observe that the visual modality is key to predicting the correct result. With the prompts attached, the model can predict the accurate result, indicating the effectiveness of our method.

The effect of modality missing rate during training. We study the impact of modality missing rate during training on the performance of the model in Figure 6. From the figure, we find that starting at a low point, both ACC and F1 score steadily improve as the train set modality missing rate increases, before reaching the highest point when the missing rate $\eta \ : = \ : 7 0 \%$ . Then both ACC and F1 score decrease as the missing rate increases. This indicates that when the train set missing rate is low, it is difficult for a model to learn very good representations in the MMGM and to learn opportune prompts that can instruct the model well. This is because when the missing rate is low, the model tends to find a shortcut which to some degree prevents the model from learning good representations.

![](images/f251ba16f456250b0a8bfb58b07f4f6a0635f962e0b1c5d050b79dc1ee9f0e55.jpg)  
Train set missing rate (%)

![](images/c7ac18f2fd6566882089c6e25962bf0068467049a3efd67f49447eb52e70ab01.jpg)  
Train set missing rate (%)

![](images/632eba6da2240691920141b5d43ed865015096517744003171dfa918500248fa.jpg)  
Train set missing rate (%)

Figure 6: Quantitative results on CMU-MOSI (left), IEMOCAP (middle) and CH-SIMS (right) with different modality missing rates during training. We report the average performance under six different missing cases.  
![](images/ceeff9f73a49a2ed6663f464c85b61c35f5ef8cc45f159bd6b004ca970a93b6f.jpg)  
Figure 7: Quantitative results on CMU-MOSI with dif ferent prompt lengths $\ell _ { p } .$ The figure shows the improved accuracy (IACC) over the baseline Modality Dropout and the parameter utilization rate $\xi = \mathrm { I A C C } / \ell _ { p }$

With the missing rate higher, the model has to learn how to generate missing information to make predictions more accurate. However, if the missing rate is higher than 70%, due to the amount of missing information, it is also hard for a model to learn good representations and prompts.

The effect of prompt length. To study the impact of prompt length on our model, we train our model on CMU-MOSI with nine different prompt lengths and present results in Figure 7. In the figure, we show the improved accuracy (IACC) over the baseline "Modality Dropout" of models with different prompt lengths. Intuitively, the longer the prompt length, the better the performance of the model. However, with the results shown in the figure, we find that when the prompt length $\ell _ { p } = 1 6$ the model performs the best. When the prompts are longer than 20, with the increase of the prompt length, the performance of the model decreases. Therefore, we deduce that it may be because our task is not complex and therefore the increase in parameters may overfit the model. Besides, we introduce parameter utilization rate $\xi = \mathrm { I A C C } / \ell _ { p }$ to represent a trade-off between the performance of models and the number of parameters of prompts. From the figure, we can clearly see that $\ell _ { p } = 1 6$ is the best choice, where IACC and ξ are both high compared with others. This also indicates that our proposed method can help improve the baseline with only a few parameters.

## 5 Conclusion

In this paper, we propose a novel multimodal Transformer via prompt learning to tackle the issue of missing modalities. We propose three types of prompts: generative prompts, missing-signal prompts, and missing-type prompts. Generative prompts can help generate missing information. Missing-signal prompts are modality-specific and missing-type prompts are modality-shared, which help the model learn intra-modality and intermodality relationships respectively. With prompt learning, we can significantly reduce the number of trainable parameters. Extensive experiments and ablation studies demonstrate the effectiveness and robustness of our proposed method.

## Limitations

Our missing modality generation module generates missing information through two simple Conv blocks and generative prompts. This module improves the performance of our model significantly. However, we use extracted features but not raw features. Due to the simplicity of our missing generation module, the performance of the model could degrade if we use raw features which are much more complicated and have weaker correlation between modalities than extracted features. We leave this problem to future work.

## Acknowledgements

This work was supported by National Key R&D Program of China under Grant No.2022ZD0162000.

## References

Gustavo Aguilar, Viktor Rozgic, Weiran Wang, and Chao Wang. 2019. Multimodal and multi-view models for emotion recognition. In Proceedings of the 57th Annual Meeting ofthe Associationfor Computational Linguistics, pages 991–1002, Florence, Italy. Association for Computational Linguistics.

AmirAli Bagher Zadeh, Paul Pu Liang, Soujanya Poria, Erik Cambria, and Louis-Philippe Morency. 2018. Multimodal language analysis in the wild: CMU-MOSEI dataset and interpretable dynamic fusion graph. In Proceedings of the 56th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 2236–2246, Melbourne, Australia. Association for Computational Linguistics.

Carlos Busso, Murtaza Bulut, Chi-Chun Lee, Ebrahim (Abe) Kazemzadeh, Emily Mower Provost, Samuel Kim, Jeannette N. Chang, Sungbok Lee, and Shrikanth S. Narayanan. 2008. Iemocap: interactive emotional dyadic motion capture database. Language Resources and Evaluation, 42:335–359.

Lei Cai, Zhengyang Wang, Hongyang Gao, Dinggang Shen, and Shuiwang Ji. 2018. Deep adversarial learning for multi-modality missing data completion. In Proceedings ofthe 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD ’18, page 1158–1166, New York, NY, USA. Association for Computing Machinery.

Changde Du, Changying Du, Hao Wang, Jinpeng Li, Wei-Long Zheng, Bao-Liang Lu, and Huiguang He. 2018. Semi-supervised deep generative modelling of incomplete multi-modality emotional data. In Proceedings ofthe 26th ACM international conference on Multimedia. ACM.

Tianyu Gao, Adam Fisch, and Danqi Chen. 2021. Making pre-trained language models better few-shot learners. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 3816–3830, Online. Association for Computational Linguistics.

Wei Han, Hui Chen, and Soujanya Poria. 2021. Improving multimodal fusion with hierarchical mutual information maximization for multimodal sentiment analysis. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 9180–9192, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Devamanyu Hazarika, Roger Zimmermann, and Soujanya Poria. 2020. Misa: Modality-invariant andspecific representations for multimodal sentiment analysis. In Proceedings of the 28th ACM international conference on multimedia, pages 1122–1131.

Benjamin Heinzerling and Kentaro Inui. 2021. Language models as knowledge bases: On entity representations, storage capacity, and paraphrased queries.

In Proceedings ofthe 16th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics: Main Volume, pages 1772–1791, Online. Association for Computational Linguistics.

Guimin Hu, Ting-En Lin, Yi Zhao, Guangming Lu, Yuchuan Wu, and Yongbin Li. 2022. UniMSE: Towards unified multimodal sentiment analysis and emotion recognition. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 7837–7851, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Muhammad Uzair Khattak, Hanoona Rasheed, Muhammad Maaz, Salman Khan, and Fahad Shahbaz Khan. 2023. Maple: Multi-modal prompt learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19113–19122.

Wonjae Kim, Bokyung Son, and Ildoo Kim. 2021. Vilt: Vision-and-language transformer without convolution or region supervision. In International Conference on Machine Learning, pages 5583–5594. PMLR.

Yi-Lun Lee, Yi-Hsuan Tsai, Wei-Chen Chiu, and Chen-Yu Lee. 2023. Multimodal prompting with missing modalities for visual recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14943–14952.

Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The power of scale for parameter-efficient prompt tuning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3045–3059, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Junnan Li, Ramprasaath R. Selvaraju, Akhilesh Deepak Gotmare, Shafiq Joty, Caiming Xiong, and Steven Hoi. 2021. Align before fuse: Vision and language representation learning with momentum distillation. In NeurIPS.

Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4582– 4597, Online. Association for Computational Linguistics.

Paul Pu Liang, Ziyin Liu, AmirAli Bagher Zadeh, and Louis-Philippe Morency. 2018. Multimodal language analysis with recurrent multistage fusion. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 150– 161, Brussels, Belgium. Association for Computational Linguistics.

Sheng Liang, Mengjie Zhao, and Hinrich Schuetze. 2022. Modular and parameter-efficient multimodal

fusion with prompting. In Findings of the Association for Computational Linguistics: ACL 2022, pages 2976–2985, Dublin, Ireland. Association for Computational Linguistics.

Xiao Liu, Kaixuan Ji, Yicheng Fu, Weng Tam, Zhengxiao Du, Zhilin Yang, and Jie Tang. 2022. P-tuning: Prompt tuning can be comparable to fine-tuning across scales and tasks. In Proceedings of the 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 2: Short Papers), pages 61–68, Dublin, Ireland. Association for Computational Linguistics.

Mengmeng Ma, Jian Ren, Long Zhao, Sergey Tulyakov, Cathy Wu, and Xi Peng. 2021. Smil: Multimodal learning with severely missing modality. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 35, pages 2302–2310.

Sijie Mai, Haifeng Hu, and Songlong Xing. 2019. Divide, conquer and combine: Hierarchical feature fusion network with local and global perspectives for multimodal affective computing. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 481–492, Florence, Italy. Association for Computational Linguistics.

Marius Mosbach, Maksym Andriushchenko, and Dietrich Klakow. 2021. On the stability of fine-tuning {bert}: Misconceptions, explanations, and strong baselines. In International Conference on Learning Representations.

Hai Pham, Paul Pu Liang, Thomas Manzini, Louis-Philippe Morency, and Barnabás Póczos. 2019. Found in translation: Learning robust joint representations by cyclic translations between modalities. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 33, pages 6892–6899.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR.

Yao-Hung Hubert Tsai, Shaojie Bai, Paul Pu Liang, J. Zico Kolter, Louis-Philippe Morency, and Ruslan Salakhutdinov. 2019. Multimodal transformer for unaligned multimodal language sequences. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 6558– 6569, Florence, Italy. Association for Computational Linguistics.

Maria Tsimpoukelli, Jacob L Menick, Serkan Cabi, SM Eslami, Oriol Vinyals, and Felix Hill. 2021. Multimodal few-shot learning with frozen language models. Advances in Neural Information Processing Systems, 34:200–212.

Yansen Wang, Ying Shen, Zhun Liu, Paul Pu Liang, Amir Zadeh, and Louis-Philippe Morency. 2019.

Words can shift: Dynamically adjusting word representations using nonverbal behaviors. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 33, pages 7216–7223.

Wenmeng Yu, Hua Xu, Fanyang Meng, Yilin Zhu, Yixiao Ma, Jiele Wu, Jiyun Zou, and Kaicheng Yang. 2020. CH-SIMS: A Chinese multimodal sentiment analysis dataset with fine-grained annotation of modality. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 3718–3727, Online. Association for Computational Linguistics.

Amir Zadeh, Rowan Zellers, Eli Pincus, and Louis-Philippe Morency. 2016. Multimodal sentiment intensity analysis in videos: Facial gestures and verbal messages. IEEE Intelligent Systems, 31(6):82–88.

Jinming Zhao, Ruichen Li, and Qin Jin. 2021. Missing modality imagination network for emotion recognition with uncertain missing modalities. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 2608–2618, Online. Association for Computational Linguistics.