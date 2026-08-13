# EasyGen: Easing Multimodal Generation with BiDiffuser and LLMs

Xiangyu Zhao, Bo Liu\*, Qijiong Liu\*, Guangyuan Shi\*, Xiao-Ming Wu<sup></sup> Department of Computing, The Hong Kong Polytechnic University   
{xiang-yu.zhao, bokelvin.liu, jyonn.liu, guang-yuan.shi}@connect.polyu.hk, xiao-ming.wu@polyu.edu.hk

## Abstract

We present EasyGen, an efficient model designed to enhance multimodal understanding and generation by harnessing the capabilities of diffusion models and large language models (LLMs). Unlike existing multimodal models that predominately depend on encoders like CLIP or ImageBind and need ample amounts of training data to bridge modalities, Easy-Gen leverages BiDiffuser, a bidirectional conditional diffusion model, to foster more efficient modality interactions. EasyGen achieves text generation by training a projection layer linking BiDiffuser and an LLM, and facilities image generation by training an adapter to align the LLM’s text space with the BiDiffuser’s image space. Comprehensive quantitative and qualitative experiments show that EasyGen excels in data-efficient training, high-quality image generation, and extendibility, effectively addressing the challenges in multimodal generation. The source code is available at https: //github.com/zxy556677/EasyGen.

## 1 Introduction

In recent years, remarkable progress has been made in the field of artificial intelligence generated content (AIGC), notably in technologies like large language models (LLMs) (Chiang et al., 2023; Touvron et al., 2023; Brown et al., 2020; Chowdhery et al., 2022; Zeng et al., 2022) for text generation and diffusion models (Rombach et al., 2022; Nichol et al., 2022; Saharia et al., 2022) for visual generation. These breakthroughs have paved the way for the development of multimodal large language models (MLLMs), sparking a recent trend of incorporating extra visual modules into LLMs. Collaborative models, such as Visual ChatGPT (Wu et al., 2023a) and MM-REACT (Yang et al., 2023), strategically use externally pre-trained tools to translate visual information into text descriptions and feed the data into LLMs. However, they are exclusively dependent on pre-trained tools for inference. Contrarily, end-to-end trained models including the BLIP series (Li et al., 2023b), LLaVA series (Liu et al., 2023b,a), MiniGPT-4 (Zhu et al., 2023), and mPLUG-Owl (Ye et al., 2023) focus on mapping image information to the text space of LLMs, enabling LLMs to comprehend visual inputs.

Existing end-to-end models are also not without limitations. First, most of these multimodal models rely on either CLIP (Radford et al., 2021) or Image-Bind (Girdhar et al., 2023) as their image encoder. While these encoders excel in learning unified representations that encompass both text and images, they face challenges when it comes to transforming between different modalities. This predicament makes current vision-language models relying heavily on sizable data sets to align CLIP/Bindencoded images with the language model, due to the disparity between different modalities.

Moreover, the majority of previous multimodal models have concentrated on comprehending multimodal content and lacked the capability to generate multimodal responses, such as content beyond text. Several concurrent works, such as Emu (Sun et al., 2023) and NExT-GPT (Wu et al., 2023), have utilized diffusion models for multimodal generation. Typically, these methods involve training a projection layer to align the output embedding space of the LLM with the input embedding space of the diffusion model (encoded by CLIP’s text encoder) using an MSE loss. However, this approach may lead to the underutilization of the semantic understanding and reasoning capabilities of the LLM, and may introduce information loss in the alignment process, ultimately leading to lower image generation quality compared to the original diffusion model, as elaborated in Sec. 5.6 and Tab. 6.

In this work, we propose EasyGen, an end-toend model that facilitates multimodal generation with a single bidirectional conditional diffusion model and LLMs, as illustrated in Figure 2. The diffusion model, called BiDiffuser, is obtained by fine-tuning the UniDiffuser (Bao et al., 2023b), with a specific focus on targeted image-to-text and text-to-image tasks. This fine-tuning addresses Uni-Diffuser’s limitation of attempting to fit all conditional distributions, including those based on noisy inputs, into a single model, which reduces its effectiveness on specific tasks like conditional generation from noise-free inputs. BiDiffuser plays a pivotal role for both text and image generation. In EasyGen, text generation is achieved by training a projection layer that connects BiDiffuser and an LLM, while image generation is facilitated by training an adapter that infuses the text representation of the LLM into BiDiffuser. Figure 1 showcases EasyGen’s ability to handle multimodal inputs and generate appropriate multimodal responses.

![](images/22f3973be4520e595a531b873d8fec61ca5f02b823d940c9444e1ad6f9fe6619.jpg)  
Figure 1: Our model EasyGen can understand multimodal inputs and generate multimodal responses, as illustrated by model-generated speech bubbles in grey color, which include both text and images.

EasyGen holds three significant advantages that address the challenges in multimodal generation:

First, EasyGen offers competitive performance in a data-efficient way compared to cutting-edge models, as shown in Tab. 3 (Sec. 5.5). This is due to BiDiffuser’s ability to simplify the alignment of its embedding space with an LLM, which allows for efficient training with less data for image-to-text tasks such as image captioning and VQA.

Second, EasyGen exhibits superior image generation quality, surpassing other end-to-end MLLMs, as shown in Tab. 6 (Sec. 5.6). This is attributed to the adapter’s design (Sec. 4.2), which aligns the LLM’s text space with the diffusion model’s image space, thereby utilizing the LLM’s semantic understanding and reasoning capabilities. In contrast, the projection layers in other MLLMs like NExT-GPT only align the LLM’s text space with the diffusion model’s text space and are not trained by the image denoising objective.

![](images/d6c619db560681aec663b4bc65f54da97486b46a3eb076f83e5f074478a5aadf.jpg)  
Figure 2: Overview of EasyGen.

Third, EasyGen can be readily adapted to manage complex vision-language tasks by incorporating more advanced visual encoders or by integrating BiDiffuser into contemporary sophisticated multimodal LLMs like LLaVA to enhance performance, as shown in Tab. 5 (Sec. 5.7).

## 2 Related Work

Multimodal Language Models. Recent research has witnessed a surge of interest in multimodal LLMs, including collaborative models (Wu et al., 2023a; Yang et al., 2023; Shen et al., 2023) and endto-end methods (Alayrac et al., 2022; Guo et al., 2022; Li et al., 2022; Bao et al., 2021; Wang et al., 2022b,a,a). More recently, some works also explore training LLMs with parameter-efficient tuning (Li et al., 2023b; Zhang et al., 2023a) and instruction tuning (Dai et al., 2023; Liu et al., 2023b; Ye et al., 2023; Zhu et al., 2023; Li et al., 2023a). Different from them, EasyGen is built upon BiDiffuser, which promotes more efficient interactions between modalities.

Multimodal Diffusion Models. Diffusion generative models (Rombach et al., 2022; Ramesh et al., 2021; Nichol et al., 2022; Ruiz et al., 2023) have achieved strong results in text conditioned image generation works. Specifically, Versatile Diffusion (Xu et al., 2023) employs the U-Net (Ronneberger et al., 2015) architecture with a multiflow design to tackle multiple modalities and tasks, while UniDiffuser (Bao et al., 2023b) adopts the U-ViT (Bao et al., 2023a) framework to treat both image and text as sequential token streams for diffusion calculations. However, these models are unable to complete complex language tasks. Easy-Gen combines the advantages of diffusion models and LLMs and achieves competitive performance in both image-to-text and text-to-image tasks.

Multimodal Response Generation. Recent research has made significant advancements in multimodal response generation (Koh et al., 2023b; Tang et al., 2023; Zhang et al., 2023b; Wu et al., 2023b; Pan et al., 2023; Koh et al., 2023a; Sun et al., 2023; Dong et al., 2023) using text-to-image models such as Stable Diffusion. However, the lack of semantic understanding capability in the CLIP text encoder may result in low-quality generated images. Easy-Gen addresses this issue by transferring knowledge from LLM to BiDiffuser via an adapter, enabling the creation of high-quality textual semantic representations for text-to-image generation.

## 3 Basics of Diffusion Models

Unconditional Generation. Given a data sample taken from a real data distribution $\mathbf { x } _ { 0 } \sim \mathop { q } ( \mathbf { x } _ { 0 } )$ diffusion models (Sohl-Dickstein et al., 2015; Ho et al., 2020) first destruct the data by constructing a Markov forward process and gradually injecting noise to the data:

$$
\begin{array} { r l } & { q ( \mathbf { x } _ { 1 : T } | \mathbf { x } _ { 0 } ) = \displaystyle \prod _ { t = 1 } ^ { T } q ( \mathbf { x } _ { t } | \mathbf { x } _ { t - 1 } ) , } \\ & { q ( \mathbf { x } _ { t } | \mathbf { x } _ { t - 1 } ) = \mathcal { N } ( \mathbf { x } _ { t } ; \sqrt { 1 - \beta _ { t } } \mathbf { x } _ { t - 1 } , \beta _ { t } \mathbf { I } ) , } \end{array}\tag{1}
$$

where $\beta _ { t } \in ( 0 , 1 )$ is the variance added at diffusion step t. Then, they learn to reverse the process:

$$
\begin{array} { r } { p ( \mathbf x _ { 0 : T } ) = p ( \mathbf x _ { T } ) \prod _ { t = 1 } ^ { T } p _ { \theta } ( \mathbf x _ { t - 1 } | \mathbf x _ { t } ) , } \\ { p _ { \theta } ( \mathbf x _ { t - 1 } | \mathbf x _ { t } ) = \mathcal N ( \mathbf x _ { t - 1 } ; \mu _ { t } ( \mathbf x _ { t } , t ) , \sigma _ { t } ^ { 2 } \mathbf I ) , } \end{array}\tag{2}
$$

where $p ( \mathbf { x } _ { T } ) = \mathcal { N } ( \mathbf { x } _ { T } ; 0 , \mathbf { I } )$ is the standard Gaussian distribution and $\mu _ { t } ( \cdot )$ is the parameterization of the predicted mean. Diffusion models are trained to maximize the marginal likelihood of the data $\mathbb { E } [ \log p _ { \theta } ( \mathbf { x } _ { 0 } ) ]$ , and the canonical objective is the variational lower bound of log $p _ { \theta } ( \mathbf { x } _ { 0 } )$ . Denoising diffusion probabilistic models (Ho et al., 2020) generate samples ${ \bf x } _ { t } \sim q ( { \bf x } _ { t } | { \bf x } _ { 0 } )$ by injecting noise $\epsilon \sim \mathcal { N } ( 0 , \bf { I } )$ to the data $\mathbf { x } _ { \mathrm { 0 } }$ , and train a network $\epsilon _ { \theta } ( \cdot )$ to predict the added noise ϵ using a standard mean squared error loss:

$$
\mathcal { L } : = \mathbb { E } _ { \mathbf { x } _ { 0 } , \epsilon , t } [ | | \epsilon - \epsilon _ { \theta } ( \mathbf { x } _ { t } , t ) | | ^ { 2 } ] .\tag{3}
$$

Conditional Generation. For conditional generation, a paired data $\left( \mathbf { x } _ { 0 } , \mathbf { y } _ { 0 } \right)$ is given, and the aim is to model the conditional data distribution $q ( \mathbf { x } _ { 0 } | \mathbf { y } _ { 0 } )$ , where $\mathbf { y } _ { 0 }$ can be image class or text prompt. Conditional generation includes classifier guidance (Dhariwal and Nichol, 2021) and classifier-free guidance (Ho and Salimans, 2021). Classifier guidance requires training an extra classifier on noisy data at inference time to improve sample quality. For classifier-free guidance, no classifier needs to be trained. The denosing network $\epsilon _ { \theta } ( \mathbf { x } _ { t } | \mathbf { y } _ { 0 } )$ simply conditions on the information encoded in $\mathbf { y } _ { 0 }$ . At inference time, with a guidance scale $s ,$ the modified score estimate is further in the direction of $\epsilon _ { \theta } ( \mathbf { x } _ { t } | \mathbf { y } _ { 0 } )$ and away from the unconditional model $\epsilon _ { \theta } ( \mathbf { x } _ { t } | \emptyset )$ ( is a null token):

$$
\hat { \epsilon } _ { \theta } ( \mathbf { x } _ { t } | \mathbf { y } _ { 0 } ) = \epsilon _ { \theta } ( \mathbf { x } _ { t } | \emptyset ) + s \cdot ( \epsilon _ { \theta } ( \mathbf { x } _ { t } | \mathbf { y } _ { 0 } ) - \epsilon _ { \theta } ( \mathbf { x } _ { t } | \emptyset ) ) .
$$

## 4 Proposed Model: EasyGen

We propose EasyGen, a model capable of processing multimodal inputs and generating multimodal outputs. It achieves easy multimodal generation by leveraging a bidirectional conditional diffusion model to effectively bridge the gap between different modalities and an LLM to comprehend multimodal tasks and produce textual responses containing cues for multimodal message creation. In the subsequent section, we outline the multimodal generation process of EasyGen.

## 4.1 Pre-training BiDiffuser: A Bidirectional Conditional Diffusion Model

Since the text space of LLMs is discrete, to minimize the disparity between the output of a diffusion model and the input of LLMs, we leverage Unidiffuser, a unified diffusion model capable of transforming images into the discrete text space. During the training process, UniDiffuser injects noise $\epsilon ^ { x }$ and $\epsilon ^ { y }$ to a set of paired image-text data $\left( \mathbf { x } _ { 0 } , \mathbf { y } _ { 0 } \right)$ and generates noisy data $\mathbf { x } _ { t ^ { x } }$ and $\mathbf { y } _ { t ^ { y } }$ where $0 \leqslant t ^ { x } , t ^ { y } \leqslant T$ represent two individual timesteps (perturbation levels). It then trains a joint denoising transformer U-ViT (Bao et al., 2023a)

![](images/6ec32cdd7b884e1f478f41e6f1ba33127e52d78e3693e3a03a6c303dd763f09b.jpg)  
Figure 3: The training of BiDiffuser involves finetuning the denoising transformer U-ViT in UniDiffuser with a joint objective of image-to-text and text-to-image tasks.

$\epsilon _ { \theta } ( \mathbf x _ { t ^ { x } } , \mathbf y _ { t ^ { y } } , t ^ { x } , t ^ { y } )$ to predict the noise $\epsilon ^ { x }$ and $\epsilon ^ { y }$ by minimizing the mean squared error loss:

$$
\mathbb { E } _ { \epsilon ^ { x } , \epsilon ^ { y } , \mathbf { x } _ { 0 } , \mathbf { y } _ { 0 } } [ \| [ \epsilon ^ { x } , \epsilon ^ { y } ] - \epsilon _ { \theta } ( \mathbf { x } _ { t ^ { x } } , \mathbf { y } _ { t ^ { y } } , t ^ { x } , t ^ { y } ) \| ^ { 2 } ] ,
$$

where the output of $\epsilon _ { \theta }$ is the concatenation of the estimated noise $\epsilon _ { \theta } ^ { x }$ and $\epsilon _ { \theta } ^ { y } ,$ i.e., $\epsilon _ { \theta } = [ \epsilon _ { \theta } ^ { x } , \epsilon _ { \theta } ^ { y } ]$

By predicting $\epsilon _ { \theta } ( \mathbf x _ { t ^ { x } } , \mathbf y _ { t ^ { y } } , t ^ { x } , t ^ { y } )$ for any $t ^ { x }$ and $t ^ { y } { } _ { ; }$ , UniDiffuser learns all distributions related to $\left( \mathbf { x } _ { 0 } , \mathbf { y } _ { 0 } \right)$ simultaneously. This includes all conditional distributions: $q ( \mathbf { x } _ { 0 } | \mathbf { y } _ { 0 } )$ for text-to-image generation, $q ( \mathbf { y } _ { 0 } | \mathbf { x } _ { 0 } )$ for image-to-text generation, and those conditioned on noisy input, i.e., $q \big ( \mathbf { x } _ { 0 } | \mathbf { y } _ { t ^ { y } } \big )$ and $q ( \mathbf { y } _ { 0 } | \mathbf { x } _ { t ^ { x } } )$ , for $0 ~ < ~ t ^ { x } , t ^ { y } ~ \leq ~ T$ Learning a conditional distribution $q \big ( \mathbf { x } _ { 0 } | \mathbf { y } _ { t ^ { y } } \big )$ or $q ( \mathbf { y } _ { 0 } | \mathbf { x } _ { t ^ { x } } )$ can be seen as learning a distinct task. From a multitask learning perspective, due to limited network capacity, learning many tasks simultaneously (i.e., fitting all distributions to a single network) may result in task competition or task conflict, ultimately leading to suboptimal performance in particular tasks such as $q ( \mathbf { x } _ { 0 } | \mathbf { y } _ { 0 } )$ and $q ( \mathbf { y } _ { 0 } | \mathbf { x } _ { 0 } )$ .

To resolve this issue and enhance the performance of both image-to-text and text-to-image generation tasks, we finetune UniDiffuser with exclusive emphasis on the two tasks:

$$
\begin{array} { r } { \mathcal { L } _ { d } = \mathbb { E } _ { \epsilon ^ { x } , \epsilon ^ { y } , \mathbf { x } _ { 0 } , \mathbf { y } _ { 0 } } [ \| \epsilon ^ { x } - \epsilon _ { \theta } ^ { x } ( \mathbf { x } _ { t ^ { x } } , \mathbf { y } _ { 0 } , t ^ { x } , 0 ) \| ^ { 2 } + } \\ { \alpha \| \epsilon ^ { y } - \epsilon _ { \theta } ^ { y } ( \mathbf { x } _ { 0 } , \mathbf { y } _ { t ^ { y } } , 0 , t ^ { y } ) \| ^ { 2 } ] . } \end{array}
$$

where α is a hyperparameter to balance the learning paces of the two tasks. As depicted in Figure 3, our training objective entails predicting the text $\mathbf { y } _ { 0 }$ based on the input image $\mathbf { x } _ { \mathrm { 0 } }$ and vice versa, where the input conditions for the model are noisefree. We name the finetuned model “BiDiffuser”, signifying its specialized ability in bidirectional conditional generation.

## 4.2 Pre-training an Adapter to Enhance BiDiffuser’s SUR Capability

BiDiffuser uses the text encoder of CLIP, which is trained with image-text contrastive learning, limiting its semantic understanding and reasoning (SUR) ability for image generation. Drawing inspiration from Zhong et al. (2023), we utilize LLMs to enhance the SUR capability of BiDiffuser. Specifically, we design an adapter that employs the attention mechanism to integrate the semantic information from LLM’s last hidden state $f _ { \mathrm { L L M } } ( \cdot )$ into the CLIP text encoder $f _ { \mathrm { C L I P } } ( \cdot )$ . The adapter consists of a projection layer MLP( ) and a crossattention layer $\mathrm { { A t t } } ( \cdot )$ . Given a paired image-text data $\left( \mathbf { x } _ { 0 } , \mathbf { y } _ { 0 } \right)$ , we can get $y _ { \mathrm { s u r } }$ with enhanced SUR via the adapter:

$$
\begin{array} { r } { y _ { \mathrm { s u r } } = \mathrm { A t t } ( f _ { \mathrm { C L I P } } ( \mathbf { y } _ { 0 } ) W ^ { Q } , \mathbf { M L P } ( f _ { \mathrm { L L M } } ( \mathbf { y } _ { 0 } ) ) W ^ { K } , } \\ { \mathbf { M L P } ( f _ { \mathrm { L L M } } ( \mathbf { y } _ { 0 } ) ) W ^ { V } ) . } \end{array}
$$

Then, the semantic input to BiDiffuser is the combination of $y _ { \mathrm { s u r } }$ and the CLIP text encoding of y<sub>0</sub>:

$$
y _ { 0 } = \lambda \cdot y _ { \mathrm { s u r } } + ( 1 - \lambda ) \cdot f _ { \mathrm { C L I P } } ( \mathbf { y } _ { 0 } ) ,\tag{4}
$$

where λ is a balancing parameter. We train the adapter by freezing BiDiffuser and minimizing

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { a d a } } = \mathbb { E } _ { \epsilon ^ { y } , { \bf x } _ { 0 } } [ \| \epsilon ^ { x } - \epsilon _ { \theta } ^ { x } ( { \bf x } _ { t ^ { x } } , y _ { 0 } , t ^ { x } ) \| ^ { 2 } ] , } \end{array}\tag{5}
$$

where $\epsilon _ { \theta } ^ { x }$ is not updated as BiDiffuser is frozen.

## 4.3 Image-to-Text Generation

BiDiffuser can convert images into vectors in the text space, facilitating alignment with the vector space of LLMs. In the following, we show how BiDiffuser can be integrated with LLMs to perform image-to-text generation tasks such as image captioning and visual question answering (VQA).

## 4.3.1 Aligning BiDiffuser with LLMs

We connect BiDiffuser and LLMs via a simple projection layer, which maps text embeddings obtained from the output of the diffusion model to the embedding space of LLMs. As shown in Figure 4, the alignment can take place either prior to the LLM (Pre-Align manner) or between its encoder and decoder components (Mid-Align manner).

Pre-Align Manner. As shown in Figure 4a, the projection layer is placed before the LLM to map the output of BiDiffuser (image representations) to the text embedding space of the LLM. The text embedding of the input image is then concatenated with the embeddings of the textual instructions and fed to the LLM for decoding. To synchronize the text space of BiDiffuser with that of the LLM, we propose to use the image-grounded text generation (ITG) objective to drive the model to generate texts based on the input image by computing the autoregressive loss:

![](images/5bd240ab1d2aff1aedf96a7882f97dc4676f1b9725360666d3cc60c927e92951.jpg)  
Figure 4: Two different ways of aligning BiDiffuser with LLMs.

$$
\mathcal { L } _ { \mathrm { I T G } } = - \frac { 1 } { L } \sum _ { l = 1 } ^ { L } \log p _ { \phi } ( w _ { l } ^ { g } | w _ { < l } ^ { g } , I , T _ { I } ) ,\tag{6}
$$

where $w ^ { g } = ( w _ { 1 } ^ { g } , . . . , w _ { L } ^ { g } )$ represents the groundtruth caption of image I with length $L , T _ { I }$ is the text instruction, and $\phi$ denotes the model parameters, which include the parameters of the projection layer and the LLM.

Mid-Align Manner. As shown in Figure 4b, the projection layer is placed between the LLM’s encoder and decoder, aiming to map the output of BiDiffuser to the embedding space of the text that is encoded by the LLM’s encoder. Particularly, we argue that the output of BiDiffuser, once mapped by the projection layer and denoted as $\mathbf { d } _ { \mathrm { d i f f } }$ , should align with the image caption that is encoded by the LLM’s encoder, denoted as $\mathbf { d } _ { \mathrm { l l m } }$ . Therefore, to accurately learn the alignment between the image and text representations, in addition to the ITG loss in Eq. $^ { 6 , }$ we also employ an image-text distance minimization (ITDM) loss:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { { I T D M } } } = \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \| \mathbf { d } _ { \mathrm { { d i f f } } } - \mathbf { d } _ { \mathrm { { l l m } } } \| _ { 2 } ^ { 2 } , } \\ { \mathcal { L } _ { \mathrm { { m i d } } } = \mathcal { L } _ { \mathrm { { I T G } } } + \mathcal { L } _ { \mathrm { { I T M } } } . } \end{array}\tag{7}
$$

where N is the batch size, and $\mathcal { L } _ { \mathrm { m i d } }$ is the overall loss. In this manner, the model parameters θ only include the parameters of the projection layer.

After the alignment, EasyGen gains the capability of zero-shot image-to-text generation, including tasks such as image captioning and VQA.

## 4.3.2 Instruction-Tuning LLMs

When aligning BiDiffuser with an LLM, we perform instruction-tuning on the LLM to equip it with the capability of understanding multimodal tasks. We designed different instructions for different LLMs, as shown in Table 12. General instruction template is denoted as follows:

USER: <Img><image></Img> + Instruction. Assistant: <answer>.

For the <image> placeholder, we substitute it with the output of BiDiffuser. To avoid over fitting to the specific task and counter the model’s inclination to generate excessively short outputs, we have devised specific instructions (see Table 11), which enable the LLM to produce concise responses when necessary. For different tasks, the distinct instruction templates are as outlined in Appendix F.

## 4.4 Text-to-Image Response Generation

Most of existing multimodal models, including the BLIP series and LLaVA series are unable to provide a multimodal response as they are primarily designed to generate only textual outputs. On the other hand, Emu (Sun et al., 2023) takes a unified approach to predict the subsequent visual or textual token in an auto-regressive manner, but it is heavily reliant on vast quantities of training data. Contrary to the limitations of these existing models, Easy-Gen, by leveraging the bidirectional generation capability of BiDiffuser and the inference capability of LLMs, can produce accurate and high-quality visual response with ease.

![](images/9c1fdbe1dc8a43379776cb748bbf2d20dfa0c42cc81f588ca263bf55aa3662ef.jpg)  
Figure 5: Text-to-image generation by EasyGen. LLM generates the response and description of the image. BiDiffuser generates images based on the description.

To tackle multimodal response generation tasks such as PhotoChat (Zang et al., 2021), we first leverage the MLLM to generate detailed image captions based on dialogue context. Then, we employ BiDiffuser to create the corresponding images with the produced captions. Specifically, we replace the image featured in the dialogue with its corresponding descriptive caption, encapsulating it with task-specific tokens <Img>,</Img> and constructing the following instruction templates:

USER: Dialog history. Assistant: <response> + <Img><caption></Img>.

When <caption> appears in response, it represents the generated description of the image. So we can use LLM’s original auto-regressive training objective. Specifically, we compute the probability of the target caption by:

$$
\mathcal { L } _ { t 2 t } = - \frac { 1 } { L } \sum _ { l = 1 } ^ { L } \log p _ { \vartheta } ( w _ { l } ^ { c } | w _ { < l } ^ { c } , H ) ,\tag{8}
$$

where $w ^ { c } = ( w _ { 1 } ^ { c } , . . . , w _ { L } ^ { c } )$ represents the caption of image $x _ { 0 }$ with length L, H is the dialogue history, and ϑ denotes the LLM’s parameters. Considering the potential for alignment deviation in discrete text alone, given the description of the image y<sub>0</sub>, we utilize $y _ { 0 }$ , which is the combination of the SUR adapter’s output and the CLIP text encoder’s output, as the conditional component of the diffusion model. This directly contributes to the denoising process. The loss function for the denoising process of a noisy image $x _ { t ^ { x } }$ is formulated in a way that is similar to Eq. 5:

$$
\begin{array} { r } { \mathcal { L } _ { t 2 i } = \mathbb { E } _ { \epsilon ^ { y } , { \mathbf x } _ { 0 } } [ \| \epsilon ^ { x } - \epsilon _ { \theta } ^ { x } ( { \mathbf x } _ { t ^ { x } } , y _ { 0 } , t ^ { x } ) \| ^ { 2 } ] , } \end{array}\tag{9}
$$

where $\epsilon _ { \theta } ^ { x }$ is not updated and we only train the parameters of LLM and adapter. The overall loss for text-to-image task is:

$$
\mathcal { L } _ { a l l } = \mathcal { L } _ { t 2 i } + \mathcal { L } _ { t 2 t } .\tag{10}
$$

Training with the instruction data enables our model to not only produce text responses but also perform image intent classification and generate image captions that BiDiffuser can interpret.

## 5 Experiments

## 5.1 Experimental Setup

We initialize encoder-decoder LLM from FlanT5- XL or decoder-only LLM from Vicuna-7B, along with the utilization of the diffusion module from BiDiffuser. During the alignment process, we maintain the frozen state of the BiDiffuser. The statistics of the datasets for pre-training, alignment and instruction-tuning can be found in Appendix B. For the image captioning task, EasyGen is evaluated on both the MS-COCO (Lin et al., 2014) Karpathy test set and the NoCaps (Agrawal et al., 2019) validation set. For the VQA task, we evaluated on OK-VQA (Marino et al., 2019) validation set and GQA (Hudson and Manning, 2019) test-dev set.

To adapt the model for multimodal dialogue generation, we fine-tune the LLM and projection layers on the PhotoChat dataset. We incorporate photosharing activities into the dialogue context by generating <Img><caption></Img>, and utilize crossentropy loss exclusively for fine-tuning the multimodal generation task. Given the limited expressiveness of image descriptions in the PhotoChat dataset, as evidenced by Figure 7 in Appendix I, we regenerate image annotations in a text format similar to that used in MS-COCO.

## 5.2 Evaluation

We evaluate EasyGen on various vision-language tasks including image captioning (MS-COCO (Lin et al., 2014), NoCaps (Agrawal et al., 2019)), visual question answering (OK-VQA (Marino et al., 2019), GQA (Hudson and Manning, 2019)), and multimodal dialog generation (PhotoChat (Zang et al., 2021)). We use BLIP (Li et al., 2022), Flamingo (Alayrac et al., 2022), BLIP-2 (Li et al., 2023b), InstructBlip (Dai et al., 2023), MiniGPT-4 (Zhu et al., 2023), and LLaVA (Liu et al., 2023b) as baselines for image-to-text tasks, and Maria (Liang et al., 2021) and Divter (Sun et al., 2021) as baselines for the multimodal response generation task. See details in Appendix C and E.

## 5.3 Overall Results

Tab. 1 presents the evaluation results for each baseline and our models on MS-COCO and VQA (zeroshot) datasets. EasyGen outperforms most of the baseline models on both the COCO test set and NoCaps validation set (zero-shot transfer). Despite being pre-trained on a small dataset (MS-COCO), EasyGen’s performance on the image captioning generation task is comparable to models trained on larger datasets. Additionally, on the OK-VQA and GQA datasets, EasyGen demonstrates improved performance compared to other models of a similar scale, achieving higher accuracy even with a simple greedy search decoding method.

<table><tr><td rowspan="2">Model</td><td colspan="2">Dataset Size</td><td colspan="2">NoCaps (val)</td><td colspan="2">COCO (Karpathy)</td><td>OK-VQA</td><td>GQA</td></tr><tr><td>PT</td><td>IT</td><td>CIDEr</td><td>SPICE</td><td>BLEU@4</td><td>CIDEr</td><td>Accuracy</td><td>Accuracy</td></tr><tr><td>BLIP (Li et al., 2022)</td><td>129M</td><td></td><td>113.2</td><td>14.8</td><td>40.4</td><td>136.7</td><td></td><td>=</td></tr><tr><td>Flamingo (Alayrac et al., 2022)</td><td>1.8B</td><td></td><td></td><td></td><td></td><td>138.1</td><td>50.6</td><td></td></tr><tr><td>BLIP-2 OPT-6.7B (Li et al., 2023b)</td><td>129M</td><td></td><td>121.0</td><td>15.3</td><td>43.5</td><td>145.2</td><td>36.4</td><td>36.4</td></tr><tr><td>BLIP-2 FlanT5XL (Li et al., 2023b)</td><td>129M</td><td></td><td>121.6</td><td>15.8</td><td>42.4</td><td>144.5</td><td>39.4</td><td>44.4</td></tr><tr><td>InstructBlip 7B (Dai et al., 2023)</td><td>129M</td><td>1.2M</td><td>123.1</td><td></td><td>40.8</td><td>140.7</td><td>61.0*</td><td>49.2*</td></tr><tr><td>MiniGPT-4 (Zhu et al., 2023)</td><td></td><td>5M</td><td>42.4</td><td></td><td></td><td></td><td>37.5</td><td>30.8</td></tr><tr><td>LLaVA (Liu et al., 2023b)</td><td>558K</td><td>158K</td><td>33.1</td><td></td><td>7.9</td><td>30.0</td><td>54.4</td><td>41.3</td></tr><tr><td>EasyGen FlanT5XL</td><td>169K</td><td>90K</td><td>121.2</td><td>15.5</td><td>43.5</td><td>145.7</td><td>41.1</td><td>37.2</td></tr><tr><td>EasyGen Vicuna-7B</td><td>169K</td><td>90K</td><td>121.8</td><td>15.8</td><td>42.4</td><td>144.6</td><td>45.2</td><td>44.6</td></tr></table>

Table 1: Evaluations of EasyGen and baselines on various image understanding tasks. PT, IT indicate sample sizes in the pretraining and instruction tuning stages respectively. EasyGen’s results on NoCaps, OK-VQA and GQA were obtained in a zero-shot setting. ⋆ denotes that the model was trained on other VQA datasets.

<table><tr><td rowspan="2">Model</td><td colspan="3">Response Generation</td><td rowspan="2">Image FID↓</td></tr><tr><td>BLEU-1/2</td><td>PPL↓</td><td>ROUGE-L</td></tr><tr><td>Divter Sun et al.</td><td>6.5/1.7</td><td>59.6</td><td>5.69</td><td>29.16</td></tr><tr><td>Maria Liang et al.</td><td>13.8/9.2</td><td>48.7</td><td>15.17</td><td></td></tr><tr><td rowspan="2">EasyGen FlanT5 EasyGen Vicuan</td><td>22.3/18.7</td><td>13.3</td><td>17.24</td><td>10.30</td></tr><tr><td>23.6/19.9</td><td>11.3</td><td>18.85</td><td>9.72</td></tr><tr><td>+ w/o adapter</td><td></td><td>一</td><td></td><td>10.16</td></tr></table>

Table 2: Evaluation on the PhotoChat dataset.
<table><tr><td>MLLM</td><td>Sample Size</td><td>Cosine Similarity ↑</td><td>MSE↓</td></tr><tr><td>MiniGPT-4</td><td>5M</td><td>0.0016</td><td>6.2031</td></tr><tr><td>LLaVA v1.5</td><td>558K</td><td>-0.0026</td><td>0.8433</td></tr><tr><td>Emu</td><td>2B</td><td>0.0054</td><td>0.4062</td></tr><tr><td>EasyGen</td><td>169K</td><td>0.0128</td><td>0.0338</td></tr></table>

Table 3: Data efficiency. Avg. Cosine similarity and mean square error between the projected representations and their respective captions embedded by LLM.

In Tab. 2, the evaluation results on the PhotoChat dataset are presented. Our method exhibits clear advantages in terms of PPL, indicating strong performance on response generation task. Because of the image descriptions in the PhotoChat dataset are overly concise, we utilized EasyGen to regenerate the image descriptions, which improved the performance of our model on image generation compared to other models. Additionally, with the adapter, EasyGen is capable of generating images of superior quality.

## 5.4 Ablation Study

In Tab. 4, we examine the impact of freezing/tuning BiDiffuser and the LLM. It can be observed that frozen Mid-Align method outperforms Pre-Align method in image captioning, which shows ITDM loss function is effective. However, the frozen Mid-Align method exhibits inferior performance in the VQA task. We hypothesize that this is due to the integration of mid-aligned target image features with query information, and the projection layer is insensitive to instruction information. We conduct instruction-tuning on Pre-Align T5 and Vicuna. Compared to models at the same scale, these instruction-tuned models achieve superior results.

## 5.5 Data Efficiency in Training

In Tab. 3, we examine the data efficiency of different image encoders for alignment with LLMs. EasyGen uses BiDiffuser, which maps images to the text space, simplifying alignment with LLMs. To assess the quality of visual representations, we measured the distance between the projected representations and their respective captions embedded by an LLM. We randomly selected 1,000 images with their corresponding captions from the MSCOCO dataset. The results show that our model, EasyGen, aligns significantly better with the LLM compared to other CLIP-based MLLMs, despite using less data for alignment. This indicates the effectiveness of our approach in achieving strong alignment with LLMs.

## 5.6 Image Generation Quality

Tab. 6 evaluates the generated image’s quality of MLLMs on MS-COCO validation set, using 30K randomly selected prompts to compute the FID score on generated images. To confirm the efficacy of our approach, we fine-tuned our method on a portion of the original data (LIAON-COCO) and the MS-COCO train set, respectively. While other models resulted in a decrease in image generation performance compared to the corresponding diffusion model, EasyGen outperformed the original UniDiffuser due to the fine-tuned BiDiffuser and the adapter module. Furthermore, Tab. 7 provides CLIP-T scores from ImagenHub. We notice similar trends to the results in Tab. 6 using the FID indicator. This suggests that our method can better align LLM with diffusion model’s text space.

<table><tr><td rowspan="2">LLM</td><td rowspan="2">Diffusion Model</td><td rowspan="2">Alignment</td><td>NoCaps</td><td colspan="3">COCO(Karpathy)</td><td>OK-VQA</td></tr><tr><td>CIDEr</td><td>SPICE</td><td>BLEU@4</td><td>CIDEr</td><td>Accuracy</td></tr><tr><td>米 T5</td><td>UniDiffuser</td><td>Pre-Align</td><td>62.4</td><td>18.0</td><td>26.8</td><td>90.7</td><td>33.0</td></tr><tr><td>0 T5</td><td>BiDiffuser</td><td>Pre-Align</td><td>119.1</td><td>25.5</td><td>42.6</td><td>145.1</td><td>41.1</td></tr><tr><td>*T5</td><td>BiDiffuser</td><td>Mid-Align</td><td>121.2</td><td>25.1</td><td>43.5</td><td>145.7</td><td>31.5</td></tr><tr><td>OT5</td><td>BiDiffuser</td><td>Mid-Align</td><td>121.5</td><td>25.3</td><td>43.6</td><td>145.7</td><td>36.4</td></tr><tr><td>Vicuna-7B</td><td>BiDiffuser</td><td>Pre-Align</td><td>121.8</td><td>24.9</td><td>42.4</td><td>144.6</td><td>45.2</td></tr><tr><td>来 Vicuna-7B</td><td>BiDiffuser</td><td>Pre-Align</td><td>119.0</td><td>24.6</td><td>40.3</td><td>140.3</td><td>42.7</td></tr></table>

Table 4: Ablation study on image captioning and VQA tasks. / denotes tuning/freezing the LLM.
<table><tr><td>Model</td><td>IT</td><td>VQAv2 (test-dev)</td><td>TextVQA</td><td>MMBench (test)</td></tr><tr><td>MiniGPT-4 (Zhu et al., 2023)</td><td>5M</td><td></td><td>19.4</td><td>23.0</td></tr><tr><td>InstructBLIP Vicuna-7B (Dai et al., 2023)</td><td>1.2M</td><td></td><td>50.1</td><td>33.9</td></tr><tr><td>LLaVA-1.5 Vicuna-7B (Liu et al., 2023a)</td><td>665K</td><td>78.5</td><td>58.2</td><td>65.2</td></tr><tr><td>LLaVA-1.5 Vicuna-13B (Liu et al., 2023a)</td><td>665K</td><td>80.0</td><td>61.3</td><td>67.8</td></tr><tr><td>EasyGen Vicuna-7B w/ ViT-L</td><td>251K</td><td>79.4</td><td>57.9</td><td>63.9</td></tr><tr><td>LLaVA-1.5 Vicuna-7B w/ EasyGen</td><td>665K</td><td>80.2</td><td>58.8</td><td>66.1</td></tr><tr><td>LLaVA-1.5 Vicuna-13B w/ EasyGen</td><td>665K</td><td>80.5</td><td>61.5</td><td>69.2</td></tr></table>

Table 5: Evaluation of EasyGen variants and baselines on more complex VQA tasks and the latest MMBench benchmark. “w/ EasyGen” means incorporating the core components of our model into existing models as depicted in Figure 6 in Appendix F. EasyGen variants rank among the top models on the leaderboard of MMBench.

<table><tr><td>MM-Model</td><td>FID↓</td><td>Diffusion Model</td><td>FID↓</td></tr><tr><td colspan="4">Zero-Shot</td></tr><tr><td>NExT-GPT</td><td>11.28 (+0.07)</td><td>SD</td><td>11.21</td></tr><tr><td>Emu</td><td>11.66 (+1.73)</td><td>SD v1.5</td><td>9.93</td></tr><tr><td>EasyGen</td><td>9.16 (-0.55)</td><td>UniDiffuser</td><td>9.71</td></tr><tr><td>+ w/o adapter</td><td>9.52 (-0.19)</td><td>UniDiffuser</td><td>9.71</td></tr><tr><td colspan="4">Fine-tuned on MS-COCO</td></tr><tr><td>EasyGen</td><td>7.68 (-0.44)</td><td>UniDiffuser</td><td>8.12</td></tr><tr><td>+ w/o adapter</td><td>7.89 (-0.23)</td><td>UniDiffuser</td><td>8.12</td></tr></table>

Table 6: Comparing the image generation quality of end-to-end MLLMs and their corresponding diffusion models on the MS-COCO validation set (256 × 256). Our EasyGen surpasses the original diffusion model, while other MLLMs fall short in comparison.

## 5.7 Extendability

Tab. 5 explores the extensibility of our method from two aspects. Firstly, we aim to enhance the performance of EasyGen on complex tasks such as VQA and OCR by integrating more powerful visual encoders. Considering the potential information dilution or omission when using BiDiffuser to convert images to text space, we choose to integrate CLIP ViT-L/14 as the image encoder (as depicted in Figure 6 in the Appendix). During this process, we freeze CLIP and BiDiffuser while fine-tuning the parameters of the LLM and projection layers. The results presented in Tab. 5 include performance on traditional short QA and the modern benchmark MMBench (Liu et al., 2023c). With CLIP ViT-L, EasyGen’s performance is better than LLaVA on the VQAv2 dataset, demonstrating that BiDiffuser can effectively assist LLM in understanding images. Secondly, we investigate the plug-and-play capability of BiDiffuser, as it can also be integrated into other MLLMs (with the same LLMs) to improve their performance. The results demonstrate that with BiDiffuser, LLaVA-1.5 could achieve better performance. We speculate that BiDiffuser provides guidance information to MLLMs, enabling them to better understand the details of CLIP encoded images.

<table><tr><td>MM-Model</td><td>CLIP-T ↑</td><td>Diffusion Model</td><td>CLIP-T ↑</td></tr><tr><td>NExT-GPT</td><td>0.259 (-0.031)</td><td>SD</td><td>0.290</td></tr><tr><td>Emu</td><td>0.262 (-0.023)</td><td>SD v1.5</td><td>0.285</td></tr><tr><td>Emu2</td><td>0.266 (-0.023)</td><td>SD XL</td><td>0.289</td></tr><tr><td>EasyGen</td><td>9.16 (-0.55)</td><td>UniDiffuser</td><td>9.71</td></tr></table>

Table 7: Comparing the CLIP-T score of end-to-end MLLMs and their corresponding diffusion models on the ImagenHub.

## 6 Conclusion and Future Work

We have introduced EasyGen, a model that facilitates multimodal understanding and generation. Compared to existing models, EasyGen offers a more efficient solution by employing BiDiffuser, a bidirectional diffusion model. This allows for more effective modal interactions, handling both imageto-text and text-to-image generations by the fusion of BiDiffuser and LLMs. Additionally, EasyGen can be easily integrated into existing advanced multimodal LLMs to further boost their performance. In the future, we will explore adapting EasyGen to perform a broader range of multimodal tasks, including subject-driven image generation, image editing, and controlled generation.

## 7 Limitations

This section aims to highlight the limitations of our work and provide further insights into the research in this area. Our model relies on diffusion for multimodal interaction, which means that the text-toimage and image-to-text processes may take longer. In our experiments, we tested the performance of our model on one A100 (80G) GPU. During inference, using 1000 image-caption pairs, EasyGen took approximately 2.95 seconds for the caption generation task (with the diffusion module taking about 2.41 seconds) and around 4.96 seconds to generate an image. We believe it would be beneficial to explore more efficient sampling methods, such as DPM-Solver++ (Lu et al., 2022), to improve the overall efficiency of EasyGen. Furthermore, EasyGen may not be seamlessly adaptable for jointly fine-tuning the BiDiffuser and Language Model without altering the BiDiffuser’s sampling mechanism. But based on our experimental findings, that joint fine-tuning of the BiDiffuser and LLM is not necessary for handling complex tasks. This observation aligns with many established approaches that do not require fine-tuning of their corresponding image encoders.

## Acknowledgments

We thank the anonymous reviewers for their valuable feedback. This research was partially supported by the grant of HK ITF ITS/359/21FP.

## References

Harsh Agrawal, Karan Desai, Yufei Wang, Xinlei Chen, Rishabh Jain, Mark Johnson, Dhruv Batra, Devi Parikh, Stefan Lee, and Peter Anderson. 2019. Nocaps: Novel object captioning at scale. In Proceedings of the IEEE/CVF international conference on computer vision, pages 8948–8957.

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. 2022. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716–23736.

Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. 2023a. All are worth words: A vit backbone for diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22669–22679.

Fan Bao, Shen Nie, Kaiwen Xue, Chongxuan Li, Shi Pu, Yaole Wang, Gang Yue, Yue Cao, Hang Su, and Jun Zhu. 2023b. One transformer fits all distributions in multi-modal diffusion at scale. arXiv preprint arXiv:2303.06555.

Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. 2021. Beit: Bert pre-training of image transformers. In International Conference on Learning Representations.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. 2023. Vicuna: An opensource chatbot impressing gpt-4 with 90%\* chatgpt quality.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2022. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. 2023. Instructblip: Towards general-purpose vision-language models with instruction tuning.

Prafulla Dhariwal and Alexander Nichol. 2021. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780– 8794.

Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, et al. 2023. Dreamllm: Synergistic multimodal comprehension and creation. arXiv preprint arXiv:2309.11499.

Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. 2023. Imagebind: One embedding space to bind them all. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15180–15190.

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. 2017. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 6904–6913.

Jiaxian Guo, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Boyang Li, Dacheng Tao, and Steven CH Hoi. 2022. From images to textual prompts: Zero-shot vqa with frozen large language models. arXiv preprint arXiv:2212.10846.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. 2017. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840– 6851.

Jonathan Ho and Tim Salimans. 2021. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applica tions.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. 2022. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556.

Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. 2021. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Drew A Hudson and Christopher D Manning. 2019. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709.

Jing Yu Koh, Daniel Fried, and Ruslan Salakhutdinov. 2023a. Generating images with multimodal language models. arXiv preprint arXiv:2305.17216.

Jing Yu Koh, Ruslan Salakhutdinov, and Daniel Fried. 2023b. Grounding language models to images for multimodal generation. arXiv preprint arXiv:2301.13823.

Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. 2017. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal ofcomputer vision, 123:32–73.

Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. 2023a. Otter: A multi-modal model with in-context instruction tuning. arXiv preprint arXiv:2305.03726.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023b. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597.

Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. 2022. Blip: Bootstrapping language-image pretraining for unified vision-language understanding and generation. In International Conference on Machine Learning, pages 12888–12900. PMLR.

Zujie Liang, Huang Hu, Can Xu, Chongyang Tao, Xiubo Geng, Yining Chen, Fan Liang, and Daxin Jiang. 2021. Maria: A visual experience powered conversational agent. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics, pages 5596–5611.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft coco: Common objects in context. In Computer Vision– ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2023a. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023b. Visual instruction tuning. arXiv preprint arXiv:2304.08485.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. 2023c. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281.

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. 2022. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095.

Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. 2019. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Proceedings of the IEEE/cvf conference

on computer vision and pattern recognition, pages 3195–3204.

Vishvak Murahari, Prithvijit Chattopadhyay, Dhruv Batra, Devi Parikh, and Abhishek Das. 2019. Improving generative visual dialog by answering diverse questions. In Proceedings ofthe Conference on Empirical Methods in Natural Language Processing (EMNLP).

Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob Mcgrew, Ilya Sutskever, and Mark Chen. 2022. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. In International Conference on Machine Learning, pages 16784–16804. PMLR.

Xichen Pan, Li Dong, Shaohan Huang, Zhiliang Peng, Wenhu Chen, and Furu Wei. 2023. Kosmos-g: Generating images in context with multimodal large language models. arXiv preprint arXiv:2310.02992.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR.

Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. 2021. Zero-shot text-to-image generation. In International Conference on Machine Learning, pages 8821–8831. PMLR.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. 2015. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pages 234–241. Springer.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. 2023. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500–22510.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. 2022. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494.

Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. 2022. A-okvqa: A benchmark for visual question answering using world knowledge. In European Conference on Computer Vision, pages 146–162. Springer.

Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. 2023. Hugginggpt: Solving ai tasks with chatgpt and its friends in huggingface. arXiv preprint arXiv:2303.17580.

Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and Amanpreet Singh. 2020. Textcaps: a dataset for image captioning with reading comprehension. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16, pages 742–758. Springer.

Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. 2019. Towards vqa models that can read. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. 2015. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR.

Qingfeng Sun, Yujing Wang, Can Xu, Kai Zheng, Yaming Yang, Huang Hu, Fei Xu, Jessica Zhang, Xiubo Geng, and Daxin Jiang. 2021. Multimodal dialogue response generation. arXiv preprint arXiv:2110.08515.

Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. 2023. Generative pretraining in multimodality. arXiv preprint arXiv:2307.05222.

Zineng Tang, Ziyi Yang, Chenguang Zhu, Michael Zeng, and Mohit Bansal. 2023. Any-to-any generation via composable diffusion. arXiv preprint arXiv:2305.11846.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. 2022a. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In International Conference on Machine Learning, pages 23318–23340. PMLR.

Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit

Som, et al. 2022b. Image as a foreign language: Beit pretraining for all vision and vision-language tasks. arXiv preprint arXiv:2208.10442.

Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan Duan. 2023a. Visual chatgpt: Talking, drawing and editing with visual foundation models. arXiv preprint arXiv:2303.04671.

Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. 2023b. Next-gpt: Any-to-any multimodal llm. arXiv preprint arXiv:2309.05519.

Xingqian Xu, Zhangyang Wang, Gong Zhang, Kai Wang, and Humphrey Shi. 2023. Versatile diffusion: Text, images and variations all in one diffusion model. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 7754–7765.

Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. 2023. Mmreact: Prompting chatgpt for multimodal reasoning and action. arXiv preprint arXiv:2303.11381.

Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. 2023. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178.

Xiaoxue Zang, Lijuan Liu, Maria Wang, Yang Song, Hao Zhang, and Jindong Chen. 2021. Photochat: A human-human dialogue dataset with photo sharing behavior for joint image-text modeling. In Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 6142–6152.

Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, et al. 2022. Glm-130b: An open bilingual pre-trained model. In The Eleventh International Conference on Learning Representations.

Renrui Zhang, Jiaming Han, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, Peng Gao, and Yu Qiao. 2023a. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. arXiv preprint arXiv:2303.16199.

Yiyuan Zhang, Kaixiong Gong, Kaipeng Zhang, Hongsheng Li, Yu Qiao, Wanli Ouyang, and Xiangyu Yue. 2023b. Meta-transformer: A unified framework for multimodal learning. arXiv preprint arXiv:2307.10802.

Shanshan Zhong, Zhongzhan Huang, Weushao Wen, Jinghui Qin, and Liang Lin. 2023. Sur-adapter: Enhancing text-to-image pre-trained diffusion models with large language models. In Proceedings of the 31st ACM International Conference on Multimedia, pages 567–578.

Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. 2023. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592.

## A Ethics Statement

We adhere to the ACL Ethics Policy and have conducted our research using publicly available repositories and datasets. Our primary focus is on investigating the integration of diffusion models and LLMs for multimodal generation. Therefore, the results should be seen as AI-generated content. While we have not observed deliberate harmful content, the model has the potential to generate such content if triggered. We have taken steps to minimize this risk through fine-tuning on public datasets, but caution is still exercised. In future, we will prioritize improving downstream performance and exploring methods to enhance control over the generation process. To ensure reproducibility and support future research, we have made all resources publicly available and provided proper citations to previous research within the code.

## B Datasets

We test the effectiveness of EasyGen by experimenting on different tasks including image captioning, visual question answering (VQA), and multimodal dialogue tasks. Table 8 shows the statistics of the pre-training datasets for BiDiffuser, alignment and VQA tasks.

We use the MS-COCO (Lin et al., 2014) dataset for image captioning. Following BLIP-2 (Li et al., 2023b), we fine-tune EasyGen on MS-COCO and evaluate its performance on the Karpathy test set and the NoCaps (Agrawal et al., 2019) validation set. In MS-COCO, each image typically has five captions that convey similar meanings. The training set consists of 82,783 images with 414,113 captions, while the COCO Karpathy test set has 5,000 images and the NoCaps validation set has 4,500 images.

For multimodal dialogue, we utilize the PhotoChat (Zang et al., 2021) dataset, which is a highquality dataset consisting of 10,917 images and 12,286 dialogues. Each dialogue is associated with a user image and its corresponding text description. The dataset is divided into 10,286 training instances, 1,000 development instances, and 1,000 testing instances. Moreover, PhotoChat includes photo-sharing activities, defined as the process of creating <Img><caption></Img> in this study. Each conversation in PhotoChat is broken down and constructed into multiple samples so that each round of responses can be learned. Specifically, we regard the first three turns as the dialog context, and the subsequent turns as the prediction targets. By converting the dialogues of this dataset into the form mentioned in 4.4, we obtained 49,240 train, 4,792 dev, and 4,836 test dialogue pairs.

For the VQA task, we conduct a quantitative evaluation on both the OK-VQA (Marino et al., 2019) validation set (5,046 questions) and the GQA (Hudson and Manning, 2019) test-dev set (12,578 questions). As shown in Table 4, for the frozen LLM, following BLIP-2, we employ the length penalty in beam search to encourage short answer generation. On the contrary, for the tuned LLM, we use the VQA instructions (as shown in Table 10) to do instruction tuning during the alignment process. The data for instruction tuning is constructed by randomly selecting 5K data from VQAv2 (Goyal et al., 2017) and 5K data from Visual Dialog (Murahari et al., 2019) training set.

## C Baselines

We compare our proposed model with the following state-of-the-art baselines:

BLIP (Li et al., 2022) is a multimodal mixture of encoder-decoder. It can be used as an imagebased text encoder or decoder. We use it to perform caption generation and VQA tasks.

BLIP-2 (Li et al., 2023b) is pre-trained through bootstrapped learning from frozen visual encoder and LLMs using an efficient pre-training strategy.

Flamingo (Alayrac et al., 2022) incorporates new cross-attention layers into Chinchilla language model (Hoffmann et al., 2022) to inject visual features, and pre-trains the new layers on billions of image-text pairs. We use it to perform caption generation and VQA tasks.

InstructBlip (Dai et al., 2023) is a vision-language instruction tuning framework that is trained with BLIP-2 and capable of solving various visual language tasks.

MiniGPT-4 (Zhu et al., 2023) utilizes a single projection layer to align visual information from a pretrained vision encoder with an LLM. It employed the same visual encoder as used in BLIP-2.

LLaVA (Liu et al., 2023b) employs a solitary projection layer to convert image features extracted from the pre-trained CLIP-ViT-L/14 visual encoder into the language embedding space of Vicuna.

Maria (Liang et al., 2021) is a neural conversation agent which can leverage visual world experiences sourced from a vast image index. It possesses the ability to fetch a relevant image specific to the conversation and extract visual knowledge from it. Divter (Sun et al., 2021) focuses on exploring multimodal dialogue generative models. Given the dialogue context, this model first generates a text response or image description and then generates an image according to the description.

<table><tr><td>Data types</td><td>Dataset</td><td>Size|</td><td></td><td>BiDiffuser | Alignment</td><td>Fine-tuning</td></tr><tr><td rowspan="2">Caption</td><td>MS-COCO caption (Lin et al., 2014)</td><td>83K</td><td>V</td><td>V</td><td>x</td></tr><tr><td>Visual Genome (Krishna et al., 2017)</td><td>86K</td><td>V</td><td>X</td><td>x</td></tr><tr><td>Multimodal instruction</td><td>LLaVA dataset (Liu et al., 2023b)</td><td>80K</td><td>x</td><td>V</td><td>V</td></tr><tr><td rowspan="2">VQA</td><td>VQAv2 (Goyal et al., 2017)</td><td>83K</td><td>X</td><td>-</td><td>V</td></tr><tr><td>AOK-VQA (Schwenk et al., 2022)</td><td>66K</td><td>X</td><td>X</td><td>V</td></tr><tr><td rowspan="2">OCR-related tasks</td><td>Text Captions (Sidorov et al., 2020)</td><td rowspan="2">22K</td><td>X X</td><td>X</td><td>V</td></tr><tr><td>TextVQA (Singh et al., 2019)</td><td></td><td>X</td><td>V</td></tr></table>

Table 8: Description of datasets used in our alignment and VQA fine-tuning stages. Noting that in alignment process, we used 5K images from VQAv2 dataset.
<table><tr><td></td><td>Dataset</td><td>Task</td><td>Split</td><td>Metric</td></tr><tr><td rowspan="4">Image-to-Text</td><td>MS-COCO (Lin et al., 2014) NoCaps (Agrawal et al., 2019)</td><td>Image captioning</td><td>Test</td><td>CIDEr, BLEU, SPICE</td></tr><tr><td></td><td>Image captioning</td><td>Val</td><td>CIDEr, SPICE</td></tr><tr><td>OK-VQA (Marino et al., 2019)</td><td>VQA</td><td>Val</td><td>Accuracy</td></tr><tr><td>GQA (udson and Manning, 2019)</td><td>VQA</td><td>Test</td><td>Accuracy</td></tr><tr><td>Multimodal Generation</td><td>PhotoChat Zang et al., 2021</td><td>Image dialogue</td><td>Test</td><td>PPL, BLEU, ROUGE, FID</td></tr></table>

Table 9: Summary of the evaluation datasets and metrics.

![](images/93c68f3d458e1d4be9ac16a101b2c5385b21bca4d0c281725f6c6236460b145c.jpg)  
Figure 6: Model’s architecture for VQA finetuning. The module with blue background is referred to as BiDiffuser, while the rest is the architecture of MLLM using CLIP as the image encoder (such as LLaVA).

## D Evaluation

For evaluating the quality of text generation, we utilize metrics such as BLEU, Rouge-L, Accuracy, and PPL (Perplexity). Additionally, following the approach of Vicuna (Chiang et al., 2023) and LLaVA (Liu et al., 2023b), we employ Chat-GPT to assess the generated responses from our model. Specifically, for the image captioning task, we randomly select 30 images from the MS-COCO Karpathy split and then let ChatGPT score the responses generated by EasyGen and the baseline models. ChatGPT evaluates the models’ responses based on relevance, details, and accuracy and assigns an overall score between 1 and 10, with a higher score indicating better performance. To evaluate the quality of image generation, we use the Frechet Inception Distance (FID) score (Heusel et al., 2017), which measures the divergence between two multivariate normal distributions.

## E Implementation Details

LLM During the alignment process, we utilize the AdamW optimizer with $\beta _ { 0 } = 0 . 9 , \beta _ { 1 } = 0 . 9 9 .$ and weight decay of 0. The LLMs are trained with a cosine learning rate of 2e-5 and a warmup rate of 0.03. We use a batch size of 96 for the frozen LLMs and 32 for the tuned LLMs. During training, we convert the LLMs (FlanT5XL/Vicuna-7B) to BFloat16/FP16 and BiDiffuser to FP16. During the VQA tuning process, we use CLIP ViT-L/14 336px as additional image encoder. We finetune EasyGen on mixture datasets for 1 epoch with a batch size of 32. We adopt the AdamW optimizer with $\beta =$ (0.9, 0.99) with the learning rate is 2e-5. We use a cosine learning rate decay with a learning rate is 2e-5 and warmup ration is 0.03.

Diffusion Module We inherit the settings from UniDiffuser and utilize pre-trained weights from its checkpoint for our text-to-image generator. The model is fine-tuned on the MS-COCO and VG dataset, which contains images with a resolution of 512 512, for 10 epochs with a batch size of 312. For all of our sampling processes, we employ DPM-Solver with 50 steps.

<table><tr><td>Task</td><td>Instruction Template</td></tr><tr><td>Image Captioning</td><td>USER: &lt;image&gt;+random[query] Assistant:</td></tr><tr><td>LLaVA 80K</td><td>USER: Please answer question from this image: &lt;image&gt; Question: &lt;question&gt; Assistant: USER: Image: &lt;image&gt; Question: &lt;question&gt; Assistant: USER: Answer question &lt;question&gt; through the image &lt;image&gt; Assistant:</td></tr><tr><td>Multimodal Dialogue</td><td>USER: Dialog history+&lt;photo&gt;+Dialogue history Assistant:</td></tr><tr><td>VQA</td><td>USER: Image: &lt;image&gt; Question: &lt;question&gt; Short answer: Assistant: USER: Image: &lt;image&gt; Question: &lt;question&gt; Answer the option&#x27;s letter. Assistant:</td></tr></table>

Table 10: Examples of task instruction templates. <image> represents the input image, <question> denotes the question in the VQA and LLaVA 80K dataset, and <photo> is the image description of the input image.

1. Describe the image concisely.   
2. Provide a brief description of the given image.   
3. Can you describe this image briefly?   
4. Provide a summary of visual elements depicted in the image.   
5. Give me the essential characteristics of the photograph in a   
concise manner.   
6. Rephrase the image depicted in a concise manner.   
7. Describe the objects in this image no in detail.   
8. Please introduce the image for me briefly.   
9. Give me the image’s short descriptions.   
10. Please provide a general depiction of the image presented.  
Table 11: For the image captioning task, a query instruction is randomly selected.

## F Instruction Tuning

We list the instructions for different tasks in the main paper in Table 10. Specifically, the queries used to describe image contents are presented in Table 11. Table 10 shows the templates used in Vicuna, if the LLM is FlanT5, kindly use “Human” to substitute “USER” in the instruction templates. Model architecture for VQA finetuning is shown in Figure 6. EasyGen integrates the outputs of BiDiffuser with images encoded by CLIP ViT-L/14. We freeze CLIP and BiDiffuser while only tuning the parameters of the LLM and projection layers.

## G Training Efficiency

Table 13 summarizes the key factors in training EasyGen. The training process of EasyGen is computationally efficient, especially with the utilization of the parameter-efficient fine-tuning method LoRa (Hu et al., 2021). To enable multimodal response generation, we further train the aligned EasyGen. This process entails fine-tuning the LLM (FlanT5XL) on the PhotoChat dataset for 2 epochs, which typically requires approximately 4 A100 (80G) GPU hours.

<table><tr><td colspan="2">Different Instruction Templates:</td></tr><tr><td rowspan="2">Caption Generation Response</td><td><img src="images/35a0e5c75abae7b51df37ec390469cc37c5348c738553c45677165fe5be7ae00.jpg"/></td></tr><tr><td>Provide a brief description of the given im- age. Assistant: Buses parked with a snow mountain view be- hind them.</td></tr><tr><td>VQA Response</td><td>What numbers are displayed on the front of the bus on the right? Short answer: Assistant: 6044</td></tr><tr><td>Multimodal Dialogue Response</td><td>What might be the purpose of the buses in this location? Assistant: It is plausible that they are in this location for multiple reasons. Some possible reasons might be: 1)...2)...</td></tr></table>

Table 12: Examples of different instructions with different output formats. We use bold fonts to indicate different instructions.

## H Impact of Alignment Manners

In Table 14, we investigate the impact of different alignment manners on EasyGen. After removing the ITDM loss, the performance is slightly weaker than the original model. It is evident that the MSE Loss can help to align the semantic spaces of the two models. Furthermore, the performance of the model will drop significantly after removing the cross-entropy loss, suggesting that constraints via the language model play a key role.

## I More Qualitative Results

We present several instances on PhotoChat dataset in Figure 7 and the image-captioning task in Figure 8. In Figure 9, 11, 10, we compare EasyGen with state-of-the-art multimodal language models. The responses of MiniGPT-4, LLaVA, mPLUGowl and InstructBlip are obtained from their official demos. Morever, in Figure 12, 13, we show EasyGen’s ability to accept multimodal inputs and generate multimodal responses.

<table><tr><td>Model</td><td>Trainable Param.</td><td>Training Images</td><td>Training Cost</td></tr><tr><td>Pre-training</td><td></td><td></td><td></td></tr><tr><td>BiDiffuser</td><td>952M</td><td>169K</td><td>120 (A100 80GB) GPU hours</td></tr><tr><td>Alignment</td><td></td><td></td><td></td></tr><tr><td>Projection Layers +T5XL</td><td>4M</td><td>163K</td><td>20 (RTX3090 24GB) GPU hours</td></tr><tr><td>Projection Layers +T5XL</td><td>3B</td><td>173K</td><td>20 (A100 80GB) GPU hours</td></tr><tr><td>Projection Layers +Vicuna 7B</td><td>7B</td><td>173K</td><td>72 (A100 80GB) GPU hours</td></tr><tr><td>Projection Layers +  Vicuna 7B(LoRa)</td><td>610M</td><td>173K</td><td>20 (A100 80GB) GPU hours</td></tr></table>

Table 13: EasyGen’s trainable parameters, training data size, and training cost during alignment process.

<table><tr><td rowspan="2">Model</td><td colspan="2">NoCaps (val)</td><td colspan="3">COCO (Karpathy)</td><td>OK-VQA</td><td>GQA</td></tr><tr><td>CIDEr</td><td>SPICE</td><td>SPICE</td><td>BLEU@4</td><td>CIDEr</td><td>Accuracy</td><td>Accuracy</td></tr><tr><td>EasyGen Mid-Align FlanT5XL</td><td>121.2</td><td>15.5</td><td>25.1</td><td>43.5</td><td>145.7</td><td>31.5</td><td>22.6</td></tr><tr><td>+ w/o ITDM</td><td>118.6</td><td>15.3</td><td>24.8</td><td>42.2</td><td>141.5</td><td></td><td></td></tr><tr><td>+ w/o ITG</td><td>93.2</td><td>12.9</td><td>23.0</td><td>35.1</td><td>127.6</td><td></td><td></td></tr></table>

Table 14: Ablation studies on the instruction-tuning process and loss functions.

![](images/53ed6d19dacba3a05f72a108a5348f1aab74ebfe79f266117b6a5551d86c35e2.jpg)

![](images/c26522b33915e4bc46f385eebaa186a70bb11793eedebcb83efd811f03d9cd82.jpg)  
Figure 7: Examples of the generated responses on PhotoChat dataset. The text highlighted in red indicates the objects present in the image. The turns prefixed with A/B denote the given context.

![](images/645b9ec8ea420721ca2686556b7e7571dbd486c7be00ce738d158b91cac5dc6c.jpg)  
EasyGen: Two cats separated by the window are looking forward at the same time.  
Figure 8: Examples of image captioning results by EasyGen.

![](images/d1f8d6e60fb4b3a67d86758f259c5c53b42709d09a574e9e41cb4be9a4749257.jpg)  
Figure 9: In this case study, for the first question, EasyGen can give an accurate answer including the background information of the image. With the image generation ability of BiDiffuser, EasyGen can generate visual responses.

![](images/2598bb11d83658477c15f2657b738dc3d13839ba0c354d5d18c1b9aa337342a3.jpg)  
Figure 10: In this case study, for the first question, EasyGen can give an accurate answer, but the responses of the other two models are a bit biased. For the second question, EasyGen and LLaVA both give reasonable advice.

![](images/269675b8b8c0fd505a50f10c84d465d687fd13b3a168d5ded6bd830e8fdb9ae3.jpg)  
Figure 11: From this example, we can find that the response from EasyGen is more comprehensive and coherent. This shows EasyGen can give reasonable suggestions based on the given image.

![](images/5c9a5f5d050c2d106141ceda82cef083513d662b448f86a7164eac81b8cf8588.jpg)  
Figure 13: Example of multimodal response generation.