# VisualWebArena: Evaluating Multimodal Agents on Realistic Visually Grounded Web Tasks

Jing Yu Koh Robert Lo\* Lawrence Jang\* Vikram Duvvur\* Ming Chong Lim\* Po-Yu Huang\* Graham Neubig Shuyan Zhou Ruslan Salakhutdinov Daniel Fried Carnegie Mellon University {jingyuk,rsalakhu,dfried}@cs.cmu.edu

## Abstract

Autonomous agents capable of planning, reasoning, and executing actions on the web offer a promising avenue for automating computer tasks. However, the majority of existing benchmarks primarily focus on text-based agents, neglecting many natural tasks that require visual information to effectively solve. Given that most computer interfaces cater to human perception, visual information often augments textual data in ways that text-only models struggle to harness effectively. To bridge this gap, we introduce VisualWebArena, a benchmark designed to assess the performance of multimodal agents on realistic visually grounded web tasks. VisualWebArena comprises of diverse and complex web-based tasks that evaluate various capabilities of autonomous multimodal agents. To perform well, agents need to accurately process image-text inputs, interpret natural language instructions, and execute actions on websites to accomplish user-defined objectives. We evaluate state-of-the-art LLMbased autonomous agents, including several multimodal agents. Our analysis reveals several limitations of text-based LLM agents, gaps in the capabilities of state-of-the-art multimodal language agents, and insights towards building stronger autonomous agents for the web.

## 1 Introduction

Automating routine computer tasks with autonomous agents is a long standing goal of artificial intelligence research (Franklin and Graesser, 1996; Jennings et al., 1998). To achieve this, we need agents that can navigate computers effectively, process visual and textual inputs, handle high-level natural language instructions, and execute actions to achieve desired goals. As digital interfaces today are primarily built for human eyes, effective visual understanding is necessary for many routine computer tasks. For example, humans frequently perform tasks on the web which involve visual references, such as “Help me order a green polo shirt from Amazon,” or rely on pictures rather than text to communicate. However, many agent benchmarks today focus on text-based tasks, neglecting the evaluation (and consequently the development) of multimodal agents. To address this gap, we propose VisualWebArena (Fig. 1), a benchmark suite designed to rigorously assess and advance the visual and textual capabilities of autonomous agents. VisualWebArena builds off the WebArena (Zhou et al., 2024) framework, leveraging reproducible self-hosted environments and execution-based evaluations. VisualWebArena introduces a set of unique tasks that emphasize integrating visual understanding with language processing, closely simulating human interaction with modern computing interfaces. Our contributions are summarized as follows:

• We introduce VisualWebArena, a set of 910 realistic tasks over three diverse web environments: Classifieds, Shopping, and Reddit. The Classifieds environment is a new contribution with real world data, while the Shopping and Reddit environments are inherited from WebArena. All tasks we introduce are visually grounded, and require visual understanding of webpage content to effectively solve (while WebArena does not). 25.2% of the tasks also include images as input (Fig. 1), and require understanding interleaved image-text inputs.

• We extensively benchmark the autonomous capabilities of state-of-the-art (SOTA) large language models (LLM) and vision-language models (VLMs), demonstrating that strong VLMs outperform text-based LLMs. The best VLM agents achieve a success rate of 16.4% on VisualWebArena, which is still significantly below human performance of 88.7%.

![](images/ac35701bb60cfd59f2263b57a4727704170a7a28120d4392dc4c3cbff86f59d1.jpg)  
Figure 1: VisualWebArena is a benchmark suite of 910 realistic, visually grounded tasks on self-hosted web environments that involve web navigation and visual understanding.

• We propose a new VLM agent inspired by Set-of-Marks prompting (Yang et al., 2023a), simplifying the action space of the model. We show that this model substantially outperforms other baseline LLM agents, especially on sites that are more visually complex.

## 2 Related Work

Language-Guided Web Agent Benchmarks The development of reproducible environments for autonomous agents has seen considerable progress in recent years. Earlier efforts introduced reinforcement learning environments (Brockman et al., 2016), and extended into web domains (Shi et al., 2017; Liu et al., 2018). Recent web agent benchmarks introduced tasks involving actions on static internet pages (Deng et al., 2023) as well as interaction in simulated web environments (Yao et al., 2022; Zhou et al., 2024). AgentBench (Liu et al., 2023c) extends the scope of agents for computer interaction beyond the web, exploring database management and operating system functionalities.

LLM Agents There has been significant recent interest in using Large Language Models (LLMs) for developing autonomous agents (Xi et al., 2023; Wang et al., 2023a). State-of-the-art LLMs (Google, 2023; OpenAI, 2023; Chowdhery et al., 2023; Rae et al., 2021; Zhang et al., 2022; Touvron et al., 2023a,b; Jiang et al., 2023, 2024) based on the Transformer (Vaswani et al., 2017) architecture have demonstrated impressive abilities in learning from in-context examples (Brown et al., 2020; Chan et al., 2022), reasoning (Wei et al., 2022; Yao et al., 2023; Wang et al., 2023c; Besta et al., 2023), following instructions (Chung et al., 2022; Longpre et al., 2023; Ouyang et al., 2022), and operating over long-context sequences (Tay et al., 2021; Bertsch et al., 2023; Tworkowski et al., 2023). Several recent works leverage these abilities for building autonomous web agents: Kim et al. (2023) propose a recursive prompting method to improve GPT-4 performance on MiniWoB++ (Liu et al., 2018). Liu et al. (2023d) propose a method of orchestrating multiple LLM agents to improve performance on the WebShop (Yao et al., 2022) environment. Zeng et al. (2023) fine-tunes the LLaMA-2 models on interaction trajectories with instructions, improving over baseline agents.

Vision-Language Models Finally, our work builds off advances in vision-language models (VLMs), used for many multimodal tasks such as image captioning (Vinyals et al., 2015), visual question answering (Antol et al., 2015), and other benchmarks (Mialon et al., 2023; Yue et al., 2023; Tong et al., 2024). Frozen (Tsimpoukelli et al., 2021) was one of the first approaches to demonstrate the effectiveness of finetuning a visual encoder to map images into the embedding space of a LLM, introducing compelling few-shot multimodal abilities. Alayrac et al. (2022) introduced cross-attention layers and scaled up model sizes and training data. Wang et al. (2023b) introduced trainable visual expert modules to improve visionlanguage fusion. Liu et al. (2023b) proposed finetuning on images paired with instructions to improve text generation performance on several multimodal tasks. GPT-4V (OpenAI, 2023) introduces visual processing to the GPT-4 family of models. Gemini (Google, 2023) is multimodal from the beginning (in contrast to post-hoc fine-tuned models), and can handle text interleaved with visual and audio inputs. Several recent work have also explored using VLMs to build agents for mobile platforms (Zhan and Zhang, 2023; Chu et al., 2023;

Yang et al., 2023b) and the web (Gur et al., 2023; Hong et al., 2023). Zheng et al. (2024) is contemporaneous work which performs action grounding to identify appropriate HTML elements for enabling agents to execute actions. In contrast, our proposed SoM agent uses JavaScript to produce a Set-of-Marks (Yang et al., 2023a) for the VLM to directly use as an observation and action space.

## 3 VisualWebArena Environment

In order to ensure reproducibility, realism, and determinism, all websites in the VisualWebArena framework are provided as standalone self-hosted web applications. The textual and visual content are acquired from real world counterparts, while the code is based off open-source infrastructure commonly used in real websites. We formally define the environment, observation space, and action space below, but encourage readers to refer to WebArena (Zhou et al., 2024) for more details.

The environment and agent can be modeled as a partially observable Markov decision process (POMDP): $\mathcal { E } = ( S , A , \Omega , T )$ , where S represents the set of states, A represents the set of actions (Sec. 3.2), and Ω represents the set of observations (Sec. 3.1). The transition function is defined as $T : S \times A \to S$ , with deterministic transitions between states conditioned on actions. At each time step t, the environment is in some state $s _ { t } \ ( \mathrm { e . g . }$ , a particular page), with a partial observation $o _ { t } \in \Omega$ An agent issues an action $a _ { t } \in A$ conditioned on $o _ { t } .$ , which results in a new state $s _ { t + 1 } \in S$ and a new partial observation $o _ { t + 1 } \in \Omega$ of the resulting page. The action $a _ { t }$ may be an action to be executed on the webpage (Tab. 1), or it may simply be a string output for information seeking tasks (Sec. 3.3).

Finally, we define the reward function $R : S \times$ $A  \{ 0 , 1 \}$ (Sec. 3.3) to measure the success of a task execution. In VisualWebArena, the reward function returns 1 at the final step if the state transitions align with the expectations of the task objective (i.e., the goal is achieved), and 0 otherwise.

## 3.1 Observation Space

The observation space Ω is modeled after a realistic web browsing experience. Observations include the webpage URLs, opened tabs (possibly multiple tabs of different websites), and the webpage content of the focused tab. In 25.2% of tasks, the intent also involves one or more input images (e.g., the first and third tasks in Fig. 1). The webpage

<table><tr><td>Action Type a</td><td>Description</td></tr><tr><td>click [elem]</td><td>Click on element elem.</td></tr><tr><td>hover [elem]</td><td>Hover on element elem.</td></tr><tr><td>type [elem] [text]</td><td>Type text on element elem.</td></tr><tr><td>press [key_comb]</td><td>Press a key combination.</td></tr><tr><td>new_tab tab_focus [index]</td><td>Open a new tab.</td></tr><tr><td></td><td>Focus on the i-th tab.</td></tr><tr><td>tab_close</td><td>Close current tab.</td></tr><tr><td>goto [url]</td><td>Open ur1.</td></tr><tr><td>go_back</td><td>Click the back button.</td></tr><tr><td>go_forward</td><td></td></tr><tr><td></td><td>Click the forward button.</td></tr><tr><td>scroll [up|down]</td><td>Scroll up or down the page.</td></tr><tr><td>stop [answer]</td><td>End the task with an output.</td></tr></table>

Table 1: Set of possible actions A.

## content can be represented in several ways:

1. Raw web page HTML as a Document Object Model (DOM) tree, used in previous works on autonomous web agents (Shi et al., 2017; Liu et al., 2018; Deng et al., 2023).

2. The accessibility tree,<sup>1</sup> which provides a structured and simplified representation of the webpage content that is optimized for assistive technologies. This is the primary representation that WebArena (Zhou et al., 2024) uses for its baseline LLM agents.

3. Web screenshots as RGB arrays, which has demonstrated efficacy in prior work (Gur et al., 2023; Hong et al., 2023; Yan et al., 2023).

4. We introduce a new visual representation inspired by Set-of-Marks (SoM) prompting (Yang et al., 2023a). For every interactable element on the webpage, we label it with a bounding box and an ID (Fig. 2), producing a screenshot for visual agents to reference elements on the page using their unique ID. We provide more details and analysis in Sec. 5.3.

## 3.2 Action Space

The full set of actions A is summarized in Tab. 1. The arguments for action $a _ { t }$ is the unique element ID from the current observation $o _ { t }$ . An advantage of this representation (over predicting (x, y) coordinates) is that it allows us to focus on high level reasoning rather than low-level control, as many SOTA VLMs and LLMs were not explicitly trained for referencing elements at such fine granularity. For the agents with accessibility tree representations, the argument is the element ID in the tree.

![](images/90b95b49e2049303c2a6cad5d4604cf12f393b92f5258b0f5051469e35da571c.jpg)  
Figure 2: Set-of-Marks (Yang et al., 2023a) augmented webpage screenshot. Every interactable element is highlighted with a bounding box and a unique ID.

For the SoM representation, we use the unique IDs assigned in the current page (see Fig. 2).

## 3.3 Evaluation

In order to evaluate performance on VisualWebArena, we introduce new visually grounded evaluation metrics to the functional evaluation paradigm of WebArena. These allow us to comprehensively evaluate the correctness of execution traces on open ended visually grounded tasks. The rewards for each task are hand designed functions using the primitives described below.

Information Seeking Tasks Information seeking tasks (e.g., the first task in Tab. 2) expect a string output aˆ from the model. We adopt similar reward functions as WebArena for measuring text correctness against a groundtruth output a∗:

• exact\_match: This can be defined as $\boldsymbol { \mathrm { l } } _ { \{ \hat { a } = a ^ { * } \} }$ . Only outputs that are exactly equal to the groundtruth are given a score of 1. This is used in tasks where an exact response (e.g., a numerical answer) is expected.

• must\_include: This reward function gives a score of 1 if all elements in a∗ are contained in aˆ and 0 otherwise. For example, if $\begin{array} { r c l } { \hat { a } } & { = } & { ^ { \ast \ast } \Phi 1 . 9 9 , \ S 2 . 5 0 , \ S 1 0 . 0 0 ^ { \ast , } } \end{array}$ and $a ^ { * } = \{ ^ { \ast \ast } 1 . 9 9 ^ { \ast \ast } , ^ { \ast } 2 . 5 0 ^ { \ast \ast } , ^ { \ast } 1 0 . 0 0 ^ { \ast \ast } \}$ , the task is awarded a score of 1 as all expected elements are present in the output. This is primarily used in tasks where we expect an unordered list of outputs, or we expect text output to contain a particular keyword.

• fuzzy\_match: This function queries a LLM (GPT-4-Turbo in our implementation) to evaluate whether a∗ and aˆ are semantically equal. The LLM is prompted to output “correct”, “incorrect”, or “partially correct”, and we assign a reward of 1 if the output is “correct”.<sup>2</sup> This evaluation is useful for more open ended settings where we are only concerned with semantic rather than exact equivalence, such as asking the user to add a comment describing an image.

• must\_exclude: We introduce this function, which is the converse of must\_include. A reward of 0 is assigned if any element from a set a∗ is found in aˆ (and 1 otherwise). For instance, if $\hat { a } = { } ^ {  } \$ 1.99 , \ S 2 .50, \ S 10 .00 ^ { \prime }$ and $a ^ { * } ~ = ~ \{ ^ {  } 1 . 5 0 ^ { \ ' } , { } ^ {  } 2 . 0 0 ^ { \ ' } \}$ , the reward is 1 as none of the prohibited elements are in the output.

In addition, we also introduce several new visual functions for measuring open ended tasks:

• eval\_vqa: Similar to fuzzy\_match, this function queries a VLM capable of performing visual question answering (VQA) (Antol et al., 2015). We use BLIP-2-T5XL (Li et al., 2023) in our implementation. We query the VLM with an image and a question. If the output of the VLM contains the groundtruth answer a∗, a reward of 1 is assigned. This is useful for evaluating more open ended tasks, e.g., “Buy me a green hoodie under \$10.”. There are many possible products that satisfy this objective, and it would be infeasible to enumerate all their IDs.

• eval\_fuzzy\_image\_match: This function checks whether a query image is similar to a groundtruth image according to the structural similarity index measure (SSIM) (Wang et al., 2004). If the SSIM between the query and groundtruth images is higher than a threshold $t \in [ 0 , 1 ]$ , a reward of 1 is assigned.

Navigation and Actions Many tasks in VisualWebArena require navigating through multiple webpages, and executing actions to change the underlying state s of the environment. To accurately evaluate certain objectives, we require reward functions that examine the final webpage state to determine whether the task was successfully accomplished. Each evaluator consists of a locator as well as a URL. The URL can be a specific page, or a function (e.g., the last page that the agent navigated to). The locator describes the object on the page that should be examined (e.g., all img elements, or all elements with the .product-image-photo class). During evaluation, we use the locator to retrieve the corresponding image or text content, and reuse the functions from the information seeking tasks to check for correctness.

## 4 Curating Visually Grounded Tasks

## 4.1 Web Environments

VisualWebArena is designed around three realistic web environments that involve visually rich content. Several tasks require referencing information from a self-hosted Wikipedia knowledge base, and others involve interacting across more than one website.

Classifieds We introduce a new Classifieds website in VisualWebArena, inspired by real world marketplaces such as Craigslist and Facebook Marketplace. The Classifieds site contains 65,955 listings and provides a distinct environment compared to existing ones in WebArena, introducing visually grounded tasks centered around user interactions typical in classifieds websites (posting, searching, commenting). The site’s infrastructure uses OS-Class, a robust open-source Content Management System (CMS) designed for classifieds ads, used in multiple real world sites. OSClass enables functions such as search, posting, commenting, and leaving reviews and ratings. More details about the environment are provided in Appendix. D.

Shopping The Shopping site follows the ecommerce environment from WebArena (Zhou et al., 2024), with product information and content scraped from Amazon and released in Web-Shop (Yao et al., 2022). Visual understanding of product images is required for successfully navigating and completing tasks on e-commerce platforms, making this a natural choice for VisualWebArena.

Reddit The Reddit site also follows the same environment from WebArena, and represents a social forum platform. The site contains 31,464 posts containing a diverse set of images across different subreddits and forums, such as natural images, memes, consumer electronics, and charts.

## 4.2 Tasks

Task Creation We introduce a set of 910 new tasks, split across the three sites detailed earlier. We focus on curating realistic visually grounded tasks, following a similar process as task creation in WebArena. We start by having 6 graduate students (co-authors of this paper) write intent templates (e.g., “Find me the {{attribute}} {{item}}. It should be between {{range}}.”), which can be manually expanded by the annotator to form multiple tasks $( \mathrm { e . g . }$ , “Find me the cheapest red Toyota. It should be between \$3000 to \$6000.”). We encouraged the annotators to be creative, and make use of the visual layouts of the websites, input images, and cross-site functionalities to develop creative and realistic tasks. When tasks include input images, these were sourced from royalty-free, attribution-free sources and MS-COCO (Lin et al., 2014). Annotators also wrote the reward functions using the primitives described in Sec. 3.3. We collected a total of 314 unique templates (average of 2.9 tasks per template). While the majority of tasks can be solved, we also included a small subset (46 tasks, or 5.1%) which are unachievable. This subset tests the ability of agents to terminate early in the event where a task cannot be solved, which is essential in many real world scenarios. For unachievable tasks, we require agents to output a reason why the task is unachievable, which is evaluated using the fuzzy\_match function (Sec. 3.3).

Visually Grounded Tasks A key aspect of VisualWebArena is the inherent visual grounding of all tasks. Each task demands visual understanding, requiring agents to process and interpret visual information rather than relying solely on textual or HTML-based cues. This aligns closely with modern human-computer interfaces, where visual information (e.g., icons, colors) is often critical.

<table><tr><td>Webpage / Input Image(s)</td><td>Example Intent</td><td>Reward Function r(s, a) Implementation</td></tr><tr><td></td><td>Buy the least expensive red blanket from the “Blankets &amp; Throws&quot; cate- gory.</td><td>url=&quot;func:shopping_get_latest_order_url&quot; must_include(â, { &quot;B0983XCYK6&quot;, &quot;Red&quot; })</td></tr><tr><td></td><td>Add something like what the man is wearing to my wish list.</td><td> $\mathsf { u r l } { = } ^ { \prime \prime } / \mathsf { w i s h l i s t } ^ { \prime \prime }$  locator  $( { " } \cdot { \mathsf { w i s h l i s t } }$  .product-image-photo&quot;) eval  $\mathtt { \backslash } { \mathtt { q a } } ( s , \mathrm { \Delta } ^ { \mathrm { \# } } \mathtt { I s }$  this a polo shirt?  $( y \mathsf { e s } / \mathsf { n o } ) ^ { \prime \prime } , \mathsf { \Omega } ^ { \prime \prime } \mathsf { y e s } ^ { \prime \prime } )$  eval_vqa(s, &quot;Is this shirt green?  $( y \mathsf { e s } / \mathsf { n o } ) ^ { \prime \prime } , \mathsf { \Omega } ^ { \prime \prime } \mathsf { y e s } ^ { \prime \prime } )$ </td></tr><tr><td></td><td>Create a post for each of these images eval_fuzzy_image_match(s, a*) in the most related forums.</td><td></td></tr><tr><td>Pristine 2021 Toyota 86 - Low 25000.00 $ Miles, Factory Warranty</td><td></td><td>Navigate to my listing of the white ur1=&quot;/index.php?page=item&amp;id=84144&quot;</td></tr><tr><td></td><td>Update the price in the description as well.</td><td>car and change the price to $25000. must_include(â, &quot;$25000|0R| $25,000&quot;)  $\boldsymbol { \mathfrak { m } } \mathsf { u s t \_ e x c l u d e } ( \hat { a } ,$  &quot;$30000 |OR| $30,000&quot;)</td></tr></table>

Table 2: Various evaluation metrics to assign reward $r ( s , a ) \in R : S \times A \to \{ 0 , 1 \}$ . Our execution-based reward primitives allow us to benchmark many diverse, realistic, and open-ended tasks.

For instance, a typical task might involve selecting a visually specific item, such as a “green polo shirt” where the color is visually discernible but not explicitly mentioned in text.

Task Complexity We classify each task into three difficulty levels: easy, medium, and hard. This classification is particularly useful for assessing performance across a spectrum of agents, ranging from smaller models to state-of-the-art LLMs and VLMs. We find in our analysis (Sec. 5) that many open-source models (e.g., LLaMA-2-70B, IDEFICS-80B) achieve a success rate of close to 0 on medium or hard tasks, but non-zero performance on easy tasks. This suggests that running opensource models on the easy subset would provide useful signal during development as well as faster iteration cycles (assuming performance between weaker and stronger agents are correlated). We also provide more detailed analysis in Appendix. C.2.

We annotate both the action and visual difficulty of each task. The action difficulty is determined by the estimated number of actions that a human would need to complete the task. Easy tasks are defined as those that require three or fewer actions, medium tasks involve four to nine actions, and hard tasks demand ten or more. Visual difficulty is similarly segmented: Easy tasks involve basic visual identification of colors, shapes, and high-level object detection (e.g., recognizing the presence of a cat). Medium tasks require discerning patterns, semantic understanding, or OCR on large text of shorter lengths. Hard tasks involve multiple images, OCR on small or lengthy text, or fine details. Finally, the overall difficulty level is determined by averaging the visual and reasoning complexities. However, human judgment may lead to deviations in this assessment, as certain tasks might inherently skew more towards primarily testing visual or reasoning challenges.

## 4.3 Human Performance

We measure the success rate of 7 college students on VisualWebArena tasks. Several of these students also assisted with task creation, and to avoid data leakage, we ensured that they were not assigned to the same tasks that they initially created. We sample one task per template, collecting a representative set of 230 tasks. We find that humans do well at these tasks, achieving an overall success rate of 88.7% (Tab. 3). The mistakes made in the remaining tasks are usually minor, such as not reading the task correctly or missing a part of the objective. For example, one task asked to add a particular item to the wishlist, but the human added it to the shopping cart instead. Another common failure mode was for tasks that required exhaustive search (e.g., “Find and navigate to the comments of this exact image.”). Users were often unable to find the appropriate post after searching for 5–10 minutes and gave up, assuming that the task was unachievable. In many shopping tasks, humans also did not look through all possible candidate pages to identify the cheapest or most highly reviewed product. We found these failure modes interesting, as they represent issues that strong agents would be well poised to handle, potentially achieving above human performance and speed.

## 5 Baselines

We run several baselines to benchmark the performance of state-of-the-art LLM and VLM agents. All models are prompt-based and provided with 3 in-context examples (one from each environment), which share no overlap with the benchmark tasks. The prompts we use are provided in the appendix. We summarize the results in Tab. 3 and describe the baselines in detail in the following sections.

## 5.1 Text-based LLM Agents

Several prior works developed autonomous agents by prompting text-based LLMs (Zhou et al., 2024; Kim et al., 2023; Liu et al., 2023d). We benchmark several text-based LLM agents with Chain-of-Thought prompting (Wei et al., 2022) over the accessibility tree representations of the websites as input, and leave more advanced prompting strategies for future work. We test API-based LLMs, including GPT-4 Turbo (gpt-4-1106-preview), GPT-3.5 Turbo (gpt-3.5-turbo-1106), and Gemini-Pro, as well as open sourced LLMs (LLaMA-2- 70B, Mixtral-8x7B).

## 5.2 Image Caption Augmented LLM Agents

VisualWebArena is a visually grounded benchmark, and we expect that leveraging complementary visual information would improve performance. Hence, we run pretrained image captioning models on every img element on the HTML page, and augment the accessibility tree with this information as the image alt-text before passing this as input to the LLM agents. If a task contains input images, we also caption them and include the captions as part of the prompt. We run experiments on GPT-3.5 with two recent image captioning models, BLIP-2-T5XL (Li et al., 2023) and LLaVA-v1.5- 7B (Liu et al., 2023a). Our results with GPT-3.5 as the LLM backbone (“Caption-augmented” section of Tab. 3) suggest that the LLaVA and BLIP-2 captioning models achieve comparable performance. Since BLIP-2 achieves a slightly higher success rate, is a smaller model, and requires less GPU VRAM, we use it as the captioning backbone for the remaining experiments.

## 5.3 Multimodal Agents

Finally, we benchmark strong API-based and open-source VLMs as agents. We evaluate several models capable of processing multiple interleaved image-and-text inputs: GPT-4V (OpenAI, 2023), Gemini-Pro (Google, 2023), IDEFICS-80B-Instruct (a reimplementation of Flamingo (Alayrac et al., 2022)), and CogVLM (Wang et al., 2023b). We experiment with two settings:

Image Screenshot + Captions + Accessibility Tree: This approach provides the accessibility tree representation augmented with image captions as accessibility tree alt-text from BLIP-2-T5XL (similar to the caption-augmented agent), as well as the screenshot of the current webpage as inputs. This provides the model with both the structural information and the visual context of the website.

Image Screenshot + Captions + SoM: Inspired by Set-of-Marks prompting (Yang et al., 2023a), we perform an initial preprocessing step by using JavaScript to automatically annotate every interactable element on the webpage with a bounding box and a unique ID. The annotated screenshot containing bounding boxes and IDs, are provided as input to the multimodal model along with a text representation of the SoM (see Fig. 2). Similar to the baselines above, we also provide the captions from BLIP-2-T5XL for all img elements on the page. There have been several projects<sup>3</sup> that propose similar representations. Most have been proofof-concept demos, and to the best of our knowledge, we are the first to systematically benchmark this on a realistic and interactive web environment.

## 6 Results and Analysis

Our main baseline results are summarized in Tab. 3. All existing models substantially underperform compared to humans, which indicate significant headroom in VisualWebArena for future work. We discuss some main findings below with the GPT-4V model, with further analysis in the appendix.

Text-based LLMs Perform Poorly State-of-theart text-only LLMs generally achieve poor results, with the best model (GPT-4) achieving an overall success rate of 7.25%. When we augment the LLMs with captions, this considerably improves success rate (7.25% to 12.75% for GPT-4).

Multimodality Helps Using multimodal agents significantly improves the success rate: GPT-4V (gpt-4-1106-vision-preview) achieves an overall success rate of 15.05%, substantially improving over the text-only agents. Gemini-Pro also experiences a significant uplift in success rate, from 3.85% (caption-augmented) to 6.04% (multimodal). Text-based agents may be limited in their ability to process complex images (e.g., those that require OCR or recognition of non-salient objects).

<table><tr><td rowspan="2">Model Type</td><td rowspan="2">LLM Backbone</td><td rowspan="2">Visual Backbone</td><td rowspan="2">Inputs</td><td colspan="4">Success Rate (↑)</td></tr><tr><td>Classifieds</td><td>Reddit</td><td>Shopping</td><td>Overall</td></tr><tr><td rowspan="5">Text-only</td><td>LLaMA-2-70B Mixtral-8x7B</td><td rowspan="5"></td><td rowspan="5">Acc. Tree</td><td>0.43%</td><td>1.43%</td><td>1.29%</td><td>1.10%</td></tr><tr><td></td><td>1.71%</td><td>2.86%</td><td>1.29%</td><td>1.76%</td></tr><tr><td>Gemini-Pro</td><td>0.85%</td><td>0.95%</td><td>3.43%</td><td>2.20%</td></tr><tr><td>GPT-3.5</td><td>0.43%</td><td>0.95%</td><td>3.65%</td><td>2.20%</td></tr><tr><td>GPT-4</td><td>5.56%</td><td>4.76%</td><td>9.23%</td><td>7.25%</td></tr><tr><td rowspan="6">Caption-augmented</td><td>LLaMA-2-70B</td><td>BLIP-2-T5XL</td><td rowspan="6">Acc. Tree + Caps</td><td>0.00%</td><td>0.95%</td><td>0.86%</td><td>0.66%</td></tr><tr><td>Mixtral-8x7B</td><td>BLIP-2-T5XL</td><td>1.28%</td><td>0.48%</td><td>2.79%</td><td>1.87%</td></tr><tr><td>GPT-3.5</td><td>LLaVA-7B</td><td>1.28%</td><td>1.43%</td><td>4.08%</td><td>2.75%</td></tr><tr><td>GPT-3.5</td><td>BLIP-2-T5XL</td><td>0.85%</td><td>1.43%</td><td>4.72%</td><td>2.97%</td></tr><tr><td>Gemini-Pro</td><td>BLIP-2-T5XL</td><td>1.71%</td><td>1.43%</td><td>6.01%</td><td>3.85%</td></tr><tr><td>GPT-4</td><td>BLIP-2-T5XL</td><td>8.55%</td><td>8.57%</td><td>16.74%</td><td>12.75%</td></tr><tr><td rowspan="4">Multimodal</td><td colspan="2">IDEFICS-80B-Instruct</td><td rowspan="4">Image + Caps + Acc. Tree</td><td>0.43%</td><td>0.95%</td><td>0.86%</td><td>0.77%</td></tr><tr><td colspan="2">CogVLM</td><td>0.00%</td><td>0.48%</td><td>0.43%</td><td>0.33%</td></tr><tr><td colspan="2">Gemini-Pro</td><td>3.42%</td><td>4.29%</td><td>8.15%</td><td>6.04%</td></tr><tr><td colspan="2">GPT-4V</td><td>8.12%</td><td>12.38%</td><td>19.74%</td><td>15.05%</td></tr><tr><td rowspan="4">Multimodal (SoM)</td><td colspan="2">IDEFICS-80B-Instruct</td><td rowspan="4">Image + Caps + SoM</td><td>0.85%</td><td>0.95%</td><td>1.07%</td><td>0.99%</td></tr><tr><td colspan="2">CogVLM</td><td>0.00%</td><td>0.48%</td><td>0.43%</td><td>0.33%</td></tr><tr><td colspan="2">Gemini-Pro</td><td>3.42%</td><td>3.81%</td><td>7.73%</td><td>5.71%</td></tr><tr><td colspan="2">GPT-4V</td><td>9.83%</td><td>17.14%</td><td>19.31%</td><td>16.37%</td></tr></table>

Table 3: Success rates of baseline LLM and VLM agents on VisualWebArena.

![](images/f4f77d4cdf218187f5f13b0039af4b797f3fd64607e05e4df76111c48e4809ca.jpg)  
Figure 3: Successful execution trajectory of the GPT-4V + SoM agent on the task for blocking a user that posted a certain picture. The text in red represents the actions output by the agent.

SoM Improves Navigability We observe that the SoM representation (Sec. 5.3) further improves the performance of GPT-4V over using the accessibility tree, boosting overall success rate (15.05%  16.37%). We observe particularly substantial improvements on Classifieds and Reddit, from 12.38% 17.14% and 8.12% 9.83% respectively. We attribute this to the Classifieds and Reddit websites containing denser visual content. These websites often contain multiple smaller sized images that are arranged very closely (Fig. 2). In many of these pages, the accessibility tree does not provide sufficient information to disentangle elements that are spatially close. We hypothesize that the SoM representation is superior for strong VLM agents, which can more accurately disentangle and click on the desired elements. For the other VLMs, SoM does not significantly improve success, which we attribute to the finding from Yang et al. (2023a) that only GPT-4V demonstrates the SoM grounding ability (perhaps due to scale or training data).

One GPT-4V + SoM execution trajectory that we found particularly compelling was Reddit task #139, which requires exact image matching to find a post and block a user (Fig. 3). The model initially attempts to search for the correct forum, and when this fails it navigates to the list of forums. After navigating correctly to /f/memes, it identifies the offending image out of the many images on the page (Step 3 in Fig. 3) and blocks the author successfully without any unnecessary actions.

<table><tr><td>Task Subset</td><td>% of Total</td><td>SR (↑)</td></tr><tr><td>OCR required No OCR required</td><td>17.1% 82.9%</td><td>13.4% 16.9%</td></tr><tr><td>Exact image match</td><td>8.7%</td><td>18.9%</td></tr><tr><td>No exact image match</td><td>91.3%</td><td>16.2%</td></tr><tr><td>Image inputs</td><td>25.2%</td><td>19.0%</td></tr><tr><td>No image inputs</td><td>74.8%</td><td>14.9%</td></tr><tr><td></td><td></td><td></td></tr></table>

Table 4: Success rate (SR) of GPT-4V (SoM) across different types of tasks.

## 6.1 Performance by Task Type

We analyze the success rate of the best VLM agent baseline (GPT-4V + SoM) across several additional subsets of tasks (Tab. 4). We include further analysis for other models in Appendix C.

OCR Tasks 17.1% of VisualWebArena require optical character recognition (OCR), such as reading text from product images, or extracting text from an input image. We find that GPT-4V + SoM generally performs worse on tasks that require OCR (13.4%) compared to tasks which do not (16.9%), suggesting that OCR may be a bottleneck for current agents.

Exact Image Match 8.7% of tasks require exact image matching, which requires agents to identify precise visual matches. GPT-4V + SoM achieves a slightly higher success rate on this subset (18.9%) compared to other tasks (16.2%), suggesting that exact image matching is not a primary bottleneck.

Image Input Tasks 25.2% of VisualWebArena include one or more input images as part of the objective. These tasks generally appear more tractable for the GPT-4V + SoM agent, and it achieves a higher success rate (19.0%) compared to tasks without image inputs (14.9%).

## 7 Conclusion

In this work, we introduced VisualWebArena, a benchmark of realistic tasks designed to rigorously evaluate and advance the capabilities of autonomous multimodal web agents. VisualWebArena represents a significant step towards addressing the gap in the evaluation of multimodal agents on visually grounded tasks. We also introduce a visual agent inspired by Set-of-Marks prompting, and demonstrate the potential of this approach for simplifying action spaces and improving performance on visually complex websites. Our extensive evaluation of state-of-the-art LLM and VLM agents demonstrate that while VLMs show promise, there remains a considerable performance gap compared to humans, who achieve very high success rates on VisualWebArena. Our quantitative and qualitative analysis also highlights several common failure modes of existing LLM and VLM agents. We expect future work on improving the reasoning, visual understanding, and planning abilities of agents to be particularly exciting directions.

## 8 Ethical and Broader Impacts

Real World Impacts Advancing the capabilities of autonomous agents comes with many broader considerations and ethical implications. Strong autonomous agents have the potential to improve the accessibility of computer-based tasks, potentially aiding individuals with disabilities or those lacking technical skills. More broadly, agents have the potential to automate large portions of routine computer work. While the capabilities of existing autonomous agents are insufficient for even simple tasks (as shown in this paper), these impacts highlight the need to ensure that the broader economic and social implications on employment are carefully considered if/when autonomous agents are deployed in real world applications.

Bias and Safety When developing autonomous agents, it is also imperative to ensure that these agents do not inadvertently exclude or disadvantage any group. Further analysis is essential to ensure that deployed agents do not exhibit unintended biases. Agents also have the potential to cause more harm (than regular LLMs) in real world applications if careful safeguards are not in place. Further research is necessary to understand and mitigate possible harmful behaviors.

Intended Uses VisualWebArena is a research benchmark to measure and evaluate the progress of multimodal agents. It is primarily meant to act as a self-contained sandbox environment for safely building robust agents. The models we presented in this paper are research prototypes, and not intended for deployment in practical applications in their current state (especially in high risk domains).

## References

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, et al. 2022. Flamingo: a visual language model for few-shot learning. NeurIPS.

Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. 2015. Vqa: Visual question answering. In ICCV.

Amanda Bertsch, Uri Alon, Graham Neubig, and Matthew R Gormley. 2023. Unlimiformer: Longrange transformers with unlimited length input. NeurIPS.

Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Michal Podstawski, Hubert Niewiadomski, Piotr Nyczyk, et al. 2023. Graph of thoughts: Solving elaborate problems with large language models. arXiv preprint arXiv:2308.09687.

Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. 2016. Openai gym.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. NeurIPS.

Stephanie CY Chan, Ishita Dasgupta, Junkyung Kim, Dharshan Kumaran, Andrew K Lampinen, and Felix Hill. 2022. Transformers generalize differently from information stored in context vs in weights. NeurIPS MemARI Workshop.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2023. Palm: Scaling language modeling with pathways. JMLR.

Xiangxiang Chu, Limeng Qiao, Xinyang Lin, Shuang Xu, Yang Yang, Yiming Hu, Fei Wei, Xinyu Zhang, Bo Zhang, Xiaolin Wei, et al. 2023. Mobilevlm: A fast, reproducible and strong vision language assistant for mobile devices. arXiv preprint arXiv:2312.16886.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. 2022. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416.

Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samuel Stevens, Boshi Wang, Huan Sun, and Yu Su. 2023. Mind2web: Towards a generalist agent for the web. NeurIPS.

Stan Franklin and Art Graesser. 1996. Is it an agent, or just a program?: A taxonomy for autonomous agents. In International workshop on agent theories, architectures, and languages, pages 21–35. Springer.

Gemini Team Google. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.

Izzeddin Gur, Hiroki Furuta, Austin Huang, Mustafa Safdari, Yutaka Matsuo, Douglas Eck, and Aleksandra Faust. 2023. A real-world webagent with planning, long context understanding, and program synthesis. arXiv preprint arXiv:2307.12856.

Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. 2020. The curious case of neural text degeneration. ICLR.

Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. 2023. Cogagent: A visual language model for gui agents. arXiv preprint arXiv:2312.08914.

Nicholas R Jennings, Katia Sycara, and Michael Wooldridge. 1998. A roadmap of agent research and development. Autonomous agents and multi-agent systems, 1:7–38.

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023. Mistral 7b. arXiv preprint arXiv:2310.06825.

Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. 2024. Mixtral of experts. arXiv preprint arXiv:2401.04088.

Geunwoo Kim, Pierre Baldi, and Stephen McAleer. 2023. Language models can solve computer tasks. NeurIPS.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. ICML.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft coco: Common objects in context. ECCV.

Evan Zheran Liu, Kelvin Guu, Panupong Pasupat, Tianlin Shi, and Percy Liang. 2018. Reinforcement learning on web interfaces using workflow-guided exploration. ICLR.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2023a. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023b. Visual instruction tuning. NeurIPS.

Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, and Jie Tang. 2023c. Agentbench: Evaluating llms as agents.

Zhiwei Liu, Weiran Yao, Jianguo Zhang, Le Xue, Shelby Heinecke, Rithesh Murthy, Yihao Feng, Zeyuan Chen, Juan Carlos Niebles, Devansh Arpit, et al. 2023d. Bolaa: Benchmarking and orchestrating llm-augmented autonomous agents. arXiv preprint arXiv:2308.05960.

Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V Le, Barret Zoph, Jason Wei, et al. 2023. The flan collection: Designing data and methods for effective instruction tuning. ICML.

Grégoire Mialon, Clémentine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, and Thomas Scialom. 2023. Gaia: a benchmark for general ai assistants. arXiv preprint arXiv:2311.12983.

OpenAI. 2023. Gpt-4 technical report.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. NeurIPS.

Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. 2021. Scaling language models: Methods, analysis & insights from training gopher. arXiv preprint arXiv:2112.11446.

Tianlin Shi, Andrej Karpathy, Linxi Fan, Jonathan Hernandez, and Percy Liang. 2017. World of bits: An open-domain platform for web-based agents. In ICML.

Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, and Donald Metzler. 2021. Long range arena: A benchmark for efficient transformers. ICLR.

Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. 2024. Eyes wide shut? exploring the visual shortcomings of multimodal llms. arXiv preprint arXiv:2401.06209.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023a. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023b. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Maria Tsimpoukelli, Jacob L Menick, Serkan Cabi, SM Eslami, Oriol Vinyals, and Felix Hill. 2021. Multimodal few-shot learning with frozen language models. NeurIPS.

Szymon Tworkowski, Konrad Staniszewski, Mikołaj Pacek, Yuhuai Wu, Henryk Michalewski, and Piotr Miłos. 2023. Focused transformer: Contrastive train-´ ing for context scaling. NeurIPS.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. NeurIPS.

Oriol Vinyals, Alexander Toshev, Samy Bengio, and Dumitru Erhan. 2015. Show and tell: A neural image caption generator. In CVPR.

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. 2023a. A survey on large language model based autonomous agents. arXiv preprint arXiv:2308.11432.

Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. 2023b. Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023c. Self-consistency improves chain of thought reasoning in language models. ICLR.

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. 2004. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. NeurIPS.

Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, et al. 2023. The rise and potential of large language model based agents: A survey. arXiv preprint arXiv:2309.07864.

An Yan, Zhengyuan Yang, Wanrong Zhu, Kevin Lin, Linjie Li, Jianfeng Wang, Jianwei Yang, Yiwu Zhong, Julian McAuley, Jianfeng Gao, et al. 2023. Gpt-4v in wonderland: Large multimodal models for zero-shot smartphone gui navigation. arXiv preprint arXiv:2311.07562.

Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. 2023a. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. arXiv preprint arXiv:2310.11441.

Zhao Yang, Jiaxuan Liu, Yucheng Han, Xin Chen, Zebiao Huang, Bin Fu, and Gang Yu. 2023b. Appagent: Multimodal agents as smartphone users. arXiv preprint arXiv:2312.13771.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. 2022. Webshop: Towards scalable realworld web interaction with grounded language agents. NeurIPS.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models. NeurIPS.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. 2023. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. arXiv preprint arXiv:2311.16502.

Aohan Zeng, Mingdao Liu, Rui Lu, Bowen Wang, Xiao Liu, Yuxiao Dong, and Jie Tang. 2023. Agenttuning: Enabling generalized agent abilities for llms. arXiv preprint arXiv:2310.12823.

Zhuosheng Zhan and Aston Zhang. 2023. You only look at screens: Multimodal chain-of-action agents. arXiv preprint arXiv:2309.11436.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068.

Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. 2024. Gpt-4v (ision) is a generalist web agent, if grounded. arXiv preprint arXiv:2401.01614.

Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Yonatan Bisk, Daniel Fried, Uri Alon, et al. 2024. Webarena: A realistic web environment for building autonomous agents. ICLR.

## Appendix

We provide further information on the collected tasks (Sec A), analysis on model failure modes for Gemini and GPT-4 (Sec. C), more details on the new Classifieds environment (Sec. D), and on the task collection process (Sec. E).

## A Tasks Breakdown

Distribution of Tasks Across Sites

![](images/68be5f0a5ce58558a5f3ea9efb8eb1c07bba70e9ac5fa89b9fc1c35b6d9fc336.jpg)  
Figure 4: Tasks proportion by sites.

Distribution of Tasks by Difficulty  
![](images/6485978c5e018c85ac31404abfe43f63670e6a3dec4dd0e86fba8244fa408384.jpg)  
Figure 5: Tasks proportion by difficulty.

As described in Sec. 4.2, we collected a total of 910 tasks across the Classifieds, Reddit, and Shopping sites, with several multi-site tasks that involve more than one site. Several of these tasks also reference Wikipedia as a knowledge base. The breakdown across various sites is summarized in Fig. 4.

The difficulty level of each task (for both visual difficulty and action difficulty) is summarized in Fig. 5, according to the specifications detailed in Sec. 4.2. VisualWebArena tasks span a variety of difficulty levels. In Sec. C.2 below, we also discuss the success rate of the agents across difficulty levels, and find that these are roughly correlated, with success rate decreasing as difficulty increases.

<table><tr><td rowspan="2">Agent Backbone</td><td rowspan="2">Model Type</td><td colspan="4">Success Rate (↑)</td></tr><tr><td>Classifieds</td><td>Reddit</td><td>Shopping</td><td>Overall</td></tr><tr><td>Llama-3-70B-Instruct</td><td>Caption-augmented</td><td>7.69%</td><td>5.24%</td><td>12.88%</td><td>9.78%</td></tr><tr><td>Gemini-Flash-1.5</td><td>Image + Caps + SoM</td><td>3.85%</td><td>4.76%</td><td>8.80%</td><td>6.59%</td></tr><tr><td>Gemini-Pro-1.5</td><td>Image + Caps + SoM</td><td>5.98%</td><td>12.86%</td><td>14.59%</td><td>11.98%</td></tr><tr><td>GPT-40</td><td>Image + Caps + SoM</td><td>20.51%</td><td>16.67%</td><td>20.82%</td><td>19.78%</td></tr></table>

Table 5: Success rates of recent LLM and VLM agents on VisualWebArena.

## B Additional Results

After the ACL submission deadline, we also ran the SoM agent with other recently released frontier VLMs: GPT-4o<sup>4</sup>, Gemini-Pro 1.5 (gemini-1.5-pro-preview-0514), and Gemini-Flash 1.5 (gemini-1.5-flash-preview-0514). We note that these recent models are natively multimodal, which may allow them to achieve stronger performance on multimodal tasks such as Visual-WebArena. We also run the Llama-3-70B-Instruct text-only LLM, augmented with captions from BLIP-2. The results are summarized in Tab. 5. GPT-4o achieves a success rate of 19.78%, and outperforms GPT-4V (16.37%).

Interestingly, we observe that Llama-3-70B-Instruct performs substantially better its Llama-2-70B predecessor, achieving an overall success rate of 9.78%, which is only slightly below the success rate of the caption-augmented GPT-4 agent (12.75%), and substantially better than the caption augmented GPT-3.5 (2.97%) and Llama-2-70B (0.66%) agents.

## C Further Analysis

## C.1 Few-shot Prompting

<table><tr><td rowspan="2"># Examples</td><td colspan="4">Success Rate (↑)</td></tr><tr><td>Classifieds</td><td>Reddit</td><td>Shopping</td><td>Overall</td></tr><tr><td>0</td><td>4.29%</td><td>2.38%</td><td>0.43%</td><td>2.86%</td></tr><tr><td>1</td><td>5.36%</td><td>1.43%</td><td>2.14%</td><td>3.63%</td></tr><tr><td>3</td><td>8.15%</td><td>4.29%</td><td>3.42%</td><td>6.04%</td></tr></table>

Table 6: Performance with different number of incontext examples.

In most of our main experimental results, we prompt the model with 3 in-context examples. We perform an analysis of the success rate against the number of in-context examples provided (Tab. 6). For 1-shot experiments, we provide the model with the single in-context example from its corresponding environment. All experiments are run with the multimodal Gemini-Pro model (as GPT-4V is prohibitively expensive) with the Image + Caption + Acc. Tree as the observation space.

We observe that overall success rate tends to increase with the number of examples provided, with a significant jump from 1 to 3 in-context examples. The improved results with a greater number of examples suggest that the performance of the VLM agents may improve significantly if we fine-tune the models on web trajectories, which will be an exciting direction for future work.

## C.2 Performance by Task Difficulty

We conduct an analysis of the GPT-4 models across different action and visual difficulty levels (Fig. 6). We observe that success rate generally decreases as action/vision difficulty increases, which makes intuitive sense based on the difficulty taxonomy described in Sec. 4.2. The findings also show that multimodal models perform better especially on hard visual tasks. On this subset, GPT-4V + SoM achieves an average success rate of 12.4%, which is significantly higher than that of the captionaugmented (8.0%) and the text-only agents (4.8%). In addition to success rates, the GPT-4V trajectory lengths also increased with action difficulty, with harder tasks requiring more steps.

## C.3 Task Subset Analysis

In this section, we provide more fine-grained analysis across different task subsets, similar to the one in Sec. 6.1 of the main paper. We examine both the GPT-4 text and multimodal agents, as well as the Gemini-Pro agents. This analysis may provide useful insights towards capabilities that future VLM models should have to perform well on web navigation tasks (specifically, OCR, exact image matching, and handling multiple interleaved image and text inputs).

Visual Difficulty (v)
<table><tr><td rowspan=8 colspan=1>Ai iin D(a)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>a|v</td><td rowspan=1 colspan=1>Easy</td><td rowspan=1 colspan=1>Medium</td><td rowspan=1 colspan=1>Hard</td><td rowspan=1 colspan=1>Overall</td></tr><tr><td rowspan=1 colspan=1>Easy</td><td rowspan=1 colspan=1>18.9%</td><td rowspan=1 colspan=1>11.1%</td><td rowspan=1 colspan=1>10.5%</td><td rowspan=1 colspan=1>14.8%</td></tr><tr><td rowspan=3 colspan=1>MediumhardOverall</td><td rowspan=1 colspan=1>1.6%</td><td rowspan=1 colspan=1>6.1%</td><td rowspan=1 colspan=1>7.8%</td><td rowspan=1 colspan=1>4.7%</td></tr><tr><td rowspan=1 colspan=1>1.6%</td><td rowspan=1 colspan=1>4.2%</td><td rowspan=1 colspan=1>1.5%</td><td rowspan=1 colspan=1>2.4%</td></tr><tr><td rowspan=1 colspan=1>9.0%</td><td rowspan=1 colspan=1>7.3%</td><td rowspan=1 colspan=1>4.8%</td><td rowspan=1 colspan=1>7.3%</td></tr><tr><td rowspan=1 colspan=1>(a) S</td><td rowspan=1 colspan=2>uccess rate of GPT-4</td><td rowspan=1 colspan=1>Text-on</td><td rowspan=1 colspan=1>ly</td></tr><tr><td rowspan=1 colspan=1>a|v</td><td rowspan=1 colspan=1>Easy</td><td rowspan=1 colspan=1>Medium</td><td rowspan=1 colspan=1>Hard</td><td rowspan=1 colspan=1>Overall</td></tr><tr><td rowspan=4 colspan=1></td><td rowspan=3 colspan=1>EasyMediumHard</td><td rowspan=1 colspan=1>23.1%</td><td rowspan=1 colspan=1>18.8%</td><td rowspan=1 colspan=1>13.2%</td><td rowspan=1 colspan=1>20.1%</td></tr><tr><td rowspan=1 colspan=1>14.4%</td><td rowspan=1 colspan=1>9.6%</td><td rowspan=1 colspan=1>5.2%</td><td rowspan=1 colspan=1>10.4%</td></tr><tr><td rowspan=1 colspan=1>7.8%</td><td rowspan=1 colspan=1>7.3%</td><td rowspan=1 colspan=1>8.1%</td><td rowspan=1 colspan=1>7.8%</td></tr><tr><td rowspan=1 colspan=1>Overall</td><td rowspan=1 colspan=1>16.9%</td><td rowspan=1 colspan=1>12.2%</td><td rowspan=1 colspan=1>8.0%</td><td rowspan=1 colspan=1>12.7%</td></tr></table>

(b) Success rate of GPT-4 + Captions

<table><tr><td rowspan=1 colspan=1>a|v</td><td rowspan=1 colspan=1>Easy</td><td rowspan=1 colspan=1>Medium</td><td rowspan=1 colspan=1>Hard</td><td rowspan=1 colspan=1>Overall</td></tr><tr><td rowspan=1 colspan=1>Easy</td><td rowspan=1 colspan=1>30.1%</td><td rowspan=1 colspan=1>20.5%</td><td rowspan=1 colspan=1>26.3%</td><td rowspan=1 colspan=1>25.8%</td></tr><tr><td rowspan=1 colspan=1>Medium</td><td rowspan=1 colspan=1>15.2%</td><td rowspan=1 colspan=1>11.3%</td><td rowspan=1 colspan=1>11.7%</td><td rowspan=1 colspan=1>12.9%</td></tr><tr><td rowspan=1 colspan=1>hard</td><td rowspan=1 colspan=1>14.1%</td><td rowspan=1 colspan=1>10.4%</td><td rowspan=1 colspan=1>8.9%</td><td rowspan=1 colspan=1>10.5%</td></tr><tr><td rowspan=1 colspan=1>Overall</td><td rowspan=1 colspan=1>21.4%</td><td rowspan=1 colspan=1>14.3%</td><td rowspan=1 colspan=1>12.4%</td><td rowspan=1 colspan=1>16.4%</td></tr></table>

(c) Success rate of GPT-4V + SoM
<table><tr><td>a|v</td><td>Easy</td><td>Medium</td><td>Hard</td><td>Overall</td></tr><tr><td>Easy</td><td>6.0</td><td>7.7</td><td>6.1</td><td>6.9</td></tr><tr><td>Medium</td><td>10.4</td><td>10.6</td><td>7.2</td><td>10.0</td></tr><tr><td>Hard</td><td>14.1</td><td>9.2</td><td>12.5</td><td>12.1</td></tr><tr><td>Overall</td><td>9.5</td><td>9.4</td><td>10.2</td><td>9.6</td></tr></table>

(d) Trajectory length of GPT-4V + SoM  
Figure 6: Success rates (a, b, c) and trajectory lengths (d) across different difficulty levels.

OCR Tasks On OCR tasks, which take up 17.1% of the benchmark, we observe that the GPT-4 family of models achieve a lower success rate on tasks that require OCR compared to tasks that do not (Fig. 7). This is consistent with the findings for GPT-4V + SoM reported in Sec. 6.1 of the main paper. We also observe that introducing multimodality (over just captions) substantially improves performance on OCR tasks (from 6.4% to 12.2%), showcasing the importance of having multimodal models for text recognition capabilities, as captioning models generally do not capture such finegrained information.

For Gemini-Pro agents, we also observe similar trends, with the multimodal and SoM models achieving a higher than proportionate gain on the OCR subset (compared to the non-OCR subset). Interestingly, the multimodal Gemini-Pro agents achieve a higher success rate on tasks that require OCR compared to tasks that do not. These results may suggest that it has strong inherent OCR capabilities, which we believe will be useful to explore in future work (especially on the stronger Gemini-Ultra model once it is generally available).

Exact Image Match Of the tasks in VisualWebArena, 8.7% require exact image matching, which tests the ability of agents to identify images that have the exact same content (in contrast to those that are just semantically similar). From Fig. 8, we observe that the GPT-4V SoM model achieves a higher succeess rate on tasks that expect exact image match, while the other GPT-4 agents achieve a relatively lower success rate on the exact match subset. This suggests that the SoM representation may be more optimal for exact image match, due to its visual-centric observation and action space.

For the Gemini models, we observe that success rates on exact match tasks are substantially lower than success rates on non-exact match tasks. Interestingly, we also observe a similar trend as the GPT-4 agents, where introducing multimodality improves success rates on exact match tasks, which is further bolstered with the SoM representation.

Image Input Tasks 25.2% (229 tasks) in VisualWebArena are specified with image inputs (e.g., the task in Fig. 3, and the first and third tasks in Fig. 1). The results of the Gemini-Pro and GPT-4 agents are summarized in Fig. 9.

We observe that for the GPT-4 agent, success rates are generally higher on tasks that involve image inputs, with the exception of the text-only agent. This aligns with intuition, as agents that do not have access to visual information would not be able to understand the task correctly, and would perform worse at successfully accomplishing it. For the captioning, multimodal, and SoM GPT-4 agents, success rates are higher on the tasks involving image input, which we attribute to these tasks being more tractable once the visual content is correctly understood.

Interestingly, we see a contrast with the Gemini-Pro agents, where success rate is generally lower on tasks that involve input images. This may imply that the model may not be able to process multiple interleaved image-text inputs as well. This may be useful to revisit in the future with the stronger Gemini-Ultra model once it is released, or with stronger open sourced VLMs. We believe that being able to handle interleaved multimodal inputs will be a core requirement for strong web agents, and more comprehensive error analysis with stronger models may yield useful insights.

![](images/f3f64b3e67b111175f92afdc6d271007a18912e238f449cb2251fe45cb66eee9.jpg)  
Figure 7: Success rate of GPT-4 and Gemini agents on tasks that do not require OCR vs. tasks that do.

Trajectory Lengths vs. Success Rates Hard reasoning tasks, on average, require more steps to be successfully solved. We plot the trajectory length of the GPT-4V + SoM model in Fig. 10. The findings suggest that the model assumes a significant portion of tasks can be completed in a few steps, as it terminates a majority of tasks after less than 10 steps. However, this assumption doesn’t imply that the model successfully solves the majority of tasks: the error rate remains relatively uniform across longer trajectory lengths.

## C.4 Failure Modes

In this section, we describe other common issues we observed with our baseline agent models.

Failure Over Longer Horizons We observed that in several examples, the agents would correctly perform a task but undo it, leading to failure. The GPT-4 captioning-only model on shopping task 54 (“Add the one [poster] with waves to my wish list.”) made an assumption that the product image with a caption about a lighthouse was the correct one, and added it to the wishlist. However, after going to the wish list page the agent removes the poster because “there is no explicit mention of waves in the current items listed on the Wish List page.” This issue is not unique to the text input agents; even the GPT-4 SoM agent faced a similar problem in shopping task 397 (“Buy the item on the page with a banana theme.”). The agent initially added the correct item to the shopping cart and proceeded to check out, but stopped in the middle stating in the reasoning trace output that it does not think the item fit the criteria (despite having added it to the cart just a few steps ago).

Failures on Easy Tasks We observed surprisingly poor performance on many tasks with easy action and easy visual difficulty levels, such as in shopping task 46, which tasks the agent to add the red product in the second row to the cart (starting on the page shown in Fig. 11). The multimodal and SoM GPT-4V agents clicked on a blue tablecloth in the first row and gave up when they couldn’t find an option to order it in red. Despite appearing to be a simple task (the correct product is the red cloth in the second row), none of the agents we benchmarked were able to successfully complete it.

Giving Up Too Early Another frequent issue we observed that occurred across all the agents was giving up too early. For example, GPT-4V + SoM fails on shopping task 248 (“Order a 6 pack of the green chocolate bars. If the shipping is more than 7% of the total price, leave a 3 star review mentioning it, otherwise 5.”). This tasks involves several steps which the model is able to correctly plan out, but the very first action needed is to slightly scroll down so the “add to cart” button is visible. However, even after identifying the correct product the model gives up on the first step instead of scrolling, because it does not immediately see the button. There are other instances of this occurring, such as in shopping task 175, where an agent will use the search bar to search for something, and then immediately give up because it does not see the target product instead of trying new ways to find the product.

Task Success Rates by Exact Image Match  
![](images/d720b1d2bc9e2a6e753c64cba9813f8a475722a8703d2c76e300df4d0d56074c.jpg)  
Figure 8: Success rates of agents on tasks that require exact image match vs. those that do not.

Getting Stuck in Loops Another issue we observed was oscillating or looping between pages, where the agent would look something up or navigate to a page, unsuccessfully attempt to perform the next action (such as adding it to the cart), and on failure it goes back and repeats from the beginning. An example of this is in classifieds task 205 where the model is tasked to compare two makeup palettes in two tabs, and the GPT-4V agent spends most of the time switching between the tabs. We believe that these issues will likely be alleviated by introducing more sophisticated tracking of past states and execution history, which is a promising direction for future work.

Failure Example: Changing User Phone Number Shopping task #345, a multi-site task that also involves the Wikipedia site, demonstrated several interesting points of failure that we saw throughout the execution traces for many other tasks. Fig. 12 contains the execution trace of the GPT-4V multimodal agent for the task “Prepend the country code of South Korea to the phone number of my account profile.” There are three major mistakes made by the agent in this execution trace:

• Useless actions: In step 3 of the trajectory, the agent creates a new blank tab and does not interact with it for the rest of the trajectory. While this does not impact the correctness of the final task, it does show that the agents sometimes take unnecessary steps.

• Appending text instead of replacing: Many agents added text to input fields without deleting the previous text, which would often result in long, repeating search queries or addresses. An example of this occurs in step 7 of Fig. 12.

Task Success Rates by Image Input Size  
![](images/1c104aeeae5901567f6e8b7f30bd968fe39218ed18b404a95af130e944a4bf60.jpg)  
Figure 9: Success rates of agents on tasks that include input images as part of the specification vs. tasks that are specified with just written text.

• Repeating actions: Another frequent issue we saw across agents was repeating actions, like how the agent kept jumping between step 6 and step 7 of Fig. 12 until it hit the maximum trajectory length. In this case, we believe this looping effect stems from the issue mentioned above and each time the agent tries to correct the phone number, it keeps appending the correct number instead of replacing the incorrect number with the correct number.

## C.5 Comparison Between Agents

In this section, we describe some qualitative differences we observed between the different agents on various tasks in VisualWebArena.

Text-only vs. Caption-augmented Agents For GPT-4, the text model unsurprisingly performs much worse than even the captioning model, failing to even do the most basic tasks. For example, Reddit task #101 is the relatively simple task to “Navigate to the comments section of a post that contains a picture of a keyboard.” Out of all of the GPT-4 baseline agents, the text-only agent is the only one to fail this task, as it’s unable to identify the appropriate post from just the title. Interestingly, it still manages to make an educated guess and navigate to the hottest post on /f/MechanicalKeyboards (which unfortunately, did not include a keyboard in its image).

Caption-augmented vs. SoM Agents We observed in many examples that the GPT-4V SoM and multimodal agents outperformed the captionaugmented baselines in terms of navigation capabilities and visual understanding. The multimodal models were generally better at understanding visual information on webpages, as relevant information in many images is lost when they are translated into captions. One pertinent example is Reddit task #40, where a picture of the skyline of Pittsburgh is provided, and the task is “I’d like to find the subreddit for the city this photo was taken in. Can you navigate to it?”. The GPT-4V + SoM agent correctly identifies the location of the photo, with the first line of its reasoning output as “The photo shows a city skyline with prominent buildings labeled with logosfor UPMC and PNC, which suggests that the photo was taken in Pittsburgh, Pennsylvania.”. Using this information, the agent is able to successfully navigate to the appropriate subreddit, /f/pittsburgh. In contrast, the captioning agent labels the image as “city skyline with many tall buildings” (as this is the output from the BLIP-

![](images/e31aeb0bc67c0654b0ccc403c87cd240addf799ef8a7556e687f9d69d778210f.jpg)  
Figure 10: Performance of the GPT-4V + SoM agent across different trajectory lengths.

2 model), which prevents it from identifying the appropriate subreddit. This highlights a fundamental issue with captioning models: they frequently highlight salient visual information, which hinders success on many tasks that require fine-grained visual information not typically captured by captions.

Multimodal vs. SoM Agents The SoM representation generally performs better on tasks that require more navigation steps, due to its simplified observation and action space. One example is classifieds task #31, “Find the latest listing of a white Google Pixel phone and post a comment offering \$10 less than their asking price.” (Fig. 13). While the multimodal model was unable to search for the correct terms, the SoM model was able to leverage the simplified action space to traverse more efficiently throughout the environment. It succeeded at this task by filtering for cell phones after the initial search for more relevant results, and managed to fill out the necessary comment form fields. From our observations, the SoM representation is generally more efficient compared to the multimodal representation (which only has access to the page screenshot and accessibility tree). With a strong VLM capable of SoM, the agent does not have to implicitly perform visual co-referencing to match elements from the accessibility tree to the visual buttons and inputs that it wants to interact with.

![](images/90f8b1bf6918229d77ee5f58d3d94c557ff3ca31e8b02bb9e53106e8e6201ff5.jpg)  
Figure 11: The starting page for the task “Add the red one in the second row of this page to my shopping cart.”

![](images/da576aa4b837f187ffa7a78c8340b7bfbfbcf97f4cb200d13c5aeece676d94c6.jpg)  
Figure 12: Unsuccessful execution trajectory of the GPT-4V multimodal agent on the task for adding the a country code to the user’s phone number. The text in red represents the commands output by the agent.

## D The Classifieds Environment

The Classifieds environment contains 65,955 listings, each with a title, text description, and a product image of the item being sold. To populate the site with realistic content, we scraped data across a variety of categories on Craigslist over 3 weeks, focusing on the Northeastern States of the US (similar to the geographic region in the Reddit site). This approach ensured a diverse and rich dataset, representative of real-world classifieds posts. We utilized the scrubadub Python package for redacting Personally Identifiable Information (PII), including addresses, phone numbers, and emails. We use generated placeholders for names (e.g., “Bill Smith”), emails with fictitious addresses (e.g., bill\_smith@example.com), and phone numbers with the fictional 555-prefix numbers.

Fig. 14 and 15 show two pages within the Classifieds site, the homepage and the detail page of a particular listing. Users can also use the search function, or filter posts by category or location to find items.

## E Task Collection Process

Our main task collection process is described in Sec. 4.2. We collected the set of 910 tasks by recruiting 6 computer science graduate students (co-authors of this paper), who were all familiar with commercial versions of the Classifieds, Shopping, and Reddit sites, and have used them in their personal lives.

Annotators were first instructed to spend some time exploring the VisualWebArena websites, to familiarize themselves with their functionality and content (as this may differ slightly from real world implementations). During task creation, we encouraged annotators to be creative, and make use of the visual layouts of the websites, input images, and cross-site functionalities to develop creative and realistic tasks. We ensured that there were no repeated tasks, and that there were not too many tasks of the same type (by first producing templates, followed by instantiating them with different arguments to create multiple tasks, as described in Sec. 4.2).

## F Baseline Agent Settings

For all baseline agents we report in the paper, we use a webpage viewport size of 1280 2048, and truncate text observations to 3840 tokens (or

15360 characters for Gemini). For models with shorter context windows (e.g., LLaMA, IDEFICS, CogVLM), we instead use a viewport size of 1280 720 and truncate text observations to 640 tokens. For GPT-3.5 and GPT-4 models, we follow (Zhou et al., 2024) in using a temperature of 1.0 and a top-p of 0.9. For Gemini models we use the suggested default temperature of 0.9 and top-p of 1.0. For the remaining models, we find that they benefit from sampling from lower temperatures, and use a temperature of 0.6 and top-p of 0.95. Nucleus sampling (Holtzman et al., 2020) is used in all experiments.

The system message and the prompt with incontext examples for the baseline SoM agents are shown in Fig. 16 and Fig. 17 respectively. We prompt the model with 3 in-context examples for all baselines. For multimodal and SoM models, we include the screenshot of each in-context example as well as the screenshot of the current page. For text-only and caption augmented models, the examples consist of just the text and captions.

![](images/54ad63568bbb898cb5373b40f30718fdf2569e1e55e736a84631e2dca7a6fa1d.jpg)  
Figure 13: GPT-4V SoM agent on Classifieds task #31, “Find the latest listing of a white Google Pixel phone and post a comment offering \$10 less than their asking price.”. It succeeds at the task by leveraging the more efficient navigation space.

## What are you looking for today?

![](images/84703cc960453d32bea0a04ac77a02ff785b92b11b2558374a8ca1098d9e00e9.jpg)

![](images/df1e7c6290363448edee2a1edce3a118cb227026b87d88dc47a59712f7117141.jpg)

![](images/4110dad55bed7043649948b4383801011b00ba0744cfe908a7bd6acef752f16c.jpg)

## Latest Listings

![](images/4fc2dd521472fda39d1784857544be8d6de25e25e7e3209cf20e80e948a0cb32.jpg)  
270.00 \$

![](images/0196a1a9ccf63f977b6f77402e93c49ff391940c1d10efd40759b5da5182c9b6.jpg)  
150.00 \$

![](images/f39d36dd13b50471f698cc03af4b09dffcaf6aac9cafe487062bbe2df56c55d0.jpg)  
350.00 \$

![](images/0ab1255e6f0d306393fd8abf82c2b6d261c8e161163073e6c70bc42b70414aa1.jpg)  
Canon EF 100-400mm f/4.... 1645.00 \$

![](images/8b0b7f338c5ef67c7872c3411421a70057ae9b586621eb830188edc7aae9d9b2.jpg)  
Engagement Ring 2200.00 \$

![](images/119366f67d9f15d43556deee4f8a6ad1ffc581cbfc0ac1895395569f2a71301c.jpg)  
1997 FORD AIRSTREAM ... 28995.00 \$

![](images/55b761a7720263170c8c517efa0519012f312b7c3eb47541e9aaab080c1371c7.jpg)  
Complete Guitar Rig Full S.. 900.00 \$

![](images/9dfe2c94c85fc93473a705584452f620f3d0dcfb36241bb3515d76b021d06373.jpg)  
Tennis bracelet 2000.00 \$

![](images/86dfda2464224410cb04ff328eaf2e1076214976a4b8499dc1ba741289f7b9dc.jpg)  
1395.00 \$

![](images/1417019db5add5a4161946f7533f4c1a91ad767bd29a0e3c0db0edbd29b205cb.jpg)  
605.00 \$

![](images/163f79aa8a2b6295301c6c6e723ae9aa4998c7dc09a6bb656018805ad33daa1b.jpg)  
6.00 \$

![](images/36dfa0231065d2639e91c4e740185308534e99bc37428d4825aa846eb35d0d73.jpg)  
220.00 \$

![](images/5c7d6ba53612575cc2902c104e744c3af32a99afb7f5c2bbd8b5a01fc94f795f.jpg)

## All locations

![](images/2ff120b3893a819e706185f9327cfb268b479b043633307e69dfd84fb7b31253.jpg)  
Figure 14: Homepage of the Classifieds site. Users can search for keywords, filter by category, or post location.

Classifieds > Video gaming > Nintendo Switch

## Nintendo Switch

Published date: 2024/01/03

Location: Adamsville, Alabama, United States

Edit item

Related listings  
![](images/2e53a5f5651cbd78b74eed1d126c44051570844d38bfada3ffeb0e40b9c75a13.jpg)

Selling nintendo switch for \$270

270.00 \$

## Useful information

• Avoid scams by acting locally or paying with PayPal

• Never pay with Western Union, Moneygram or other anonymous payment services

• Don't buy or sell outside of your country. Don't accept cashier cheques from outside your country

• This site is never involved in any transaction, and does not handle payments, shipping, guarantee transactions, provide escrow services, or offer "buyer protection' or "seller certification"

Contact seller

Share

![](images/6ab577d71b635d7d2a48604bedd325c4606c627a57340850cff3aa70e24e3ecc.jpg)

![](images/458dd7336578cf140893a35c58436bcaa0ceddd73b482ce736e05755490ce32f.jpg)  
Figure 15: Example post in the Classifieds website. Users can add comments and reviews to individual listings.

![](images/bbbe6ed2397a03b2defd7a38d31b2a8151b073d08e996bd298c90f9dc16bbbe5.jpg)  
Figure 16: System message used in our SoM agent.

<table><tr><td>example_user: IMAGES: (1) current page screenshot</td></tr><tr><td>OBSERVATION:</td></tr><tr><td>[31] [IMG] [Image, description: hp fx-7010dn fax machine, url: B08GKZ3ZKD.0.jpg]</td></tr><tr><td>[32] [A] [HP CB782A#ABA 640 Inkjet Fax Machine (Renewed)]</td></tr><tr><td>[] [StaticText] [$279.49]</td></tr><tr><td>[33] [BUTTON] [Add to Cart]</td></tr><tr><td>[34] [A] [Add to Wish List]</td></tr><tr><td>[35] [A] [Add to Compare]</td></tr><tr><td>URL: http://onestopmarket.com/office-products/office-electronics.html</td></tr><tr><td>OBJECTIVE: What is the price of HP Înkjet Fax Machine?</td></tr><tr><td>PREVIOUS ACTION: None</td></tr><tr><td>example_assistant: Let&#x27;s think step-by-step. This page list the information of HP Inkjet Fax Machine, which is the product identified in the objective. Its price is $279.49. I think I have achieved the objective. I will issue the stop action with the answer.</td></tr><tr><td>In summary, the next action I will perform is &#x27;&#x27;&#x27;stop [$279.49]&#x27;</td></tr><tr><td></td></tr><tr><td>example_user: IMAGES: (1) current page screenshot</td></tr><tr><td>OBSERVATION:</td></tr><tr><td>[ ] [StaticText] [/f/food] [ ] [StaticText] [[homemade] Obligatory Halloween Pumpkin Loaf! Submitted by kneechalice t3_yid9lu 1 year ago]</td></tr><tr><td>[9] [IMG] []</td></tr><tr><td>[ ] [StaticText] [Submitted by kneechalice t3_yid9lu 1 year ago]</td></tr><tr><td>[10] [A] [kneechalice] [11] [A] [45 comments]</td></tr><tr><td>[ ] [StaticText] [[I ate] Maple Pecan Croissant Submitted by AccordingtoJP t3_y3hrpn 1 year ago]</td></tr><tr><td>[14] [IMG] []</td></tr><tr><td>[ ] [StaticText] [Submitted by AccordingtoJP t3_y3hrpn 1 year ago] [15] [A] [AccordingtoJP]</td></tr><tr><td>[16] [A] [204 comments]</td></tr><tr><td>URL: http://reddit.com</td></tr><tr><td>OBJECTIVE: Tell me what the top comment on the croissant post says. PREVIOUS ACTION: None</td></tr><tr><td>example_assistant: Let&#x27;s think step-by-step. This page has a post titled &#x27;[I ate] Maple Pecan Croissant&#x27;, which is the post</td></tr><tr><td>mentioned in the objective. In order to find the top comment, I will navigate into the comments section of the post. In summary,</td></tr><tr><td>the next action I will perform is &#x27;′click [11]&#x27;</td></tr><tr><td></td></tr><tr><td>example_user: IMAGES: (1) current page screenshot</td></tr><tr><td>OBSERVATION:</td></tr><tr><td>[ ] [StaticText] [What are you looking for today?]</td></tr><tr><td>[5] [INPUT] []</td></tr><tr><td>[6] [SELECT] [Select a category]</td></tr><tr><td></td></tr><tr><td>[7] [BUTTON] [Search]</td></tr><tr><td>[ ] [StaticText] [Latest Listings]</td></tr><tr><td>[] [StaticText] [Atlas Powered Audio System w/ Tripod 150.00 $ Music instruments Borough of Red Lion (Pennsylvania)</td></tr><tr><td>2023/11/16]</td></tr><tr><td>[8] [IMG] [Atlas Powered Audio System w/ Tripod]</td></tr><tr><td>[9] [A] [Atlas Powered Audio System w/ Tripod]</td></tr><tr><td>[ ] [StaticText] [150.00 $]</td></tr><tr><td>[] [StaticText] [Neptune Gaming Console 350.00 $ Video gaming Pennwyn (Pennsylvania) 2023/11/16]</td></tr><tr><td>[10] [IMG] [Neptune Gaming Console]</td></tr><tr><td>[11] [A] [Neptune Gaming Console]</td></tr><tr><td>[ ] [StaticText] [350.00 $]</td></tr><tr><td>URL: http://classifieds.com</td></tr><tr><td>OBJECTIVE: Help me find the cheapest dark colored guitar.</td></tr><tr><td>PREVIOUS ACTION: None</td></tr><tr><td>example_assistant: Let&#x27;s think step-by-step. The objective is to find the cheapest dark colored guitar on the site. The site has a</td></tr><tr><td></td></tr><tr><td></td></tr><tr><td>search box whose ID is [5]. I can search for guitars by entering “guitar&quot;. I can submit this by pressing the Enter afterwards. In</td></tr></table>

Figure 17: In-context examples and prompt used for our SoM agents. The webpage screenshots with SoM are added to the end of each round of the example\_user dialogue.