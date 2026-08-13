# BIPED: Pedagogically Informed Tutoring System for ESL Education

Soonwoo Kwon<sup>1†</sup> Sojung Kim<sup>2†</sup> Minju Park<sup>3</sup> Seunghyun Lee<sup>3</sup> Kyuseok Kim<sup>3</sup>

Twelve Labs<sup>1</sup> KAIST<sup>2</sup> Riiid AI Research<sup>3</sup>

ian@twelvelabs.io<sup>1</sup>, sojung\_kim@kaist.ac.kr<sup>2</sup>,

{minju.park,seunghyun.lee}@riiid.co<sup>3</sup>, kimkyu80@gmail.com<sup>3</sup>

## Abstract

Large Language Models (LLMs) have a great potential to serve as readily available and costefficient Conversational Intelligent Tutoring Systems (CITS) for teaching L2 learners of English. Existing CITS, however, are designed to teach only simple concepts or lack the pedagogical depth necessary to address diverse learning strategies. To develop a more pedagogically informed CITS capable of teaching complex concepts, we construct a BIlingual PEDagogicallyinformed Tutoring Dataset (BIPED) of oneon-one, human-to-human English tutoring interactions. Through post-hoc analysis of the tutoring interactions, we come up with a lexicon of dialogue acts (34 tutor acts and 9 student acts), which we use to further annotate the collected dataset. Based on a two-step framework of first predicting the appropriate tutor act then generating the corresponding response, we implemented two CITS models using GPT-4 and SOLAR-KO, respectively. We experimentally demonstrate that the implemented models not only replicate the style of human teachers but also employ diverse and contextually appropriate pedagogical strategies.

## 1 Introduction

As Large Language Models (LLMs) such as GPT (Achiam et al., 2023) revolutionize the field of natural language generation, both researchers and practitioners have put an increasing amount of effort into developing Conversational Intelligent Tutoring Systems (CITS) that leverage the generative capabilities of LLM’s (Tack and Piech, 2022; Abdelghani et al., 2022; Park et al., 2024; Lee et al., 2023). Specifically, LLMs have the potential to teach English as a Second/Foreign Language (ESL/EFL), for they may serve as readilyavailable tutors that can emulate native-speaking contexts (Park et al., 2024; Lee et al., 2023).

However, most CITS proposed in literature for teaching ESL are restricted in scope and lack pedagogical depth. For example, works based on CIMA dataset (Stasaski et al., 2020) consider toy concepts of colored shapes and prepositions. Possibly due to limited scope, CIMA only considers a five tutor act classes (“hint”, “question”, “correction”, “confirmation”, and “other”). Similarly, the TSCC dataset (Caines et al., 2022) only considers abstract tutor act classes such as “scaffolding”, and does not consider a more granular set of teaching strategies (e.g., inferential clues, teaching synonyms or antonyms, etc.). Consequently, models trained on CIMA (Stasaski et al., 2020) and TSCC (Caines et al., 2022) are limited to employing simplistic instructional strategies (Macina et al., 2023b).

The goal of our work is to develop CITS that can teach ESL by applying pedagogically meaningful teaching strategies adaptively to student’s needs. To do so, we designed a tutoring session that focuses on comprehending a real news article, covering vocabulary, grammar, and relevant cultural contexts. Based on this, we construct a BIlingual PEDagogically-informed Tutoring Dataset (BIPED). For BIPED, we collected a bilingual, oneon-one human-to-human tutoring dataset. Thereafter, we analyzed the dataset post-hoc from a pedagogical viewpoint and developed a categorization of dialogue acts, which comprises 34 tutor acts and 9 student acts. Finally, we annotated the data using the defined dialogue act categories.

As for the development of CITS, we employ the framework (Macina et al., 2023b; Wang et al., 2023a) whereby the LLM first chooses the suitable tutor act, then generates the corresponding utterance. We believe this approach enables the model to generate a more focused response that does not deviate from the chosen tutor intent. We consider two implementations of such CITS, one based on GPT-4 prompting, and another based on fine-tuning SOLAR-KO 10.7B (L. Junbum, 2024), which is a public model trained on a Korean corpus. For GPT-4, our dataset is used as examples for in-context learning. For fine-tuning, we explored a multi-task learning approach, where the model is instruction-tuned to carry out tutor act prediction and response generation, as well as other tasks for context grounding.

In our experiments, we show through various metrics that the implemented models are capable of mimicking human teacher’s utterance style, as well as their pedagogical strategies. Specifically, we show that our models choose appropriate and diverse tutor acts, and generate human-like utterances, as measured by sentence similarity metrics.

The contributions of our paper are as follows:

• We provide BIPED, a dataset curated from bilingual human-to-human tutoring dialogues that come annotated with specific tutoring acts. Importantly, the lesson topic is complex enough to require sophisticated pedagogical methods.

• We implement two CITS based on BIPED. In particular, we explore both in-context learning and fine-tuning approaches. It is the pioneering work in utilizing instruction tuning techniques in the development of CITS.

• We provide experimental results demonstrating the implemented models’ ability to mimic human tutors.

## 2 Related Work

## 2.1 Conversational Intelligent Tutoring Systems (CITS)

Some of conversational tutoring systems make effective use of GPT through the implementation of prompt engineering (Park et al., 2024; Tack and Piech, 2022; Abdelghani et al., 2022). Moreover, there exist studies that tailor open-source LLMs to fulfill distinct objectives (Qian et al., 2023; Huang et al., 2022), and the generation of dialogue in conjunction with act classification is also gaining ground (Wang et al., 2023a). However, despite these advancements in CITS using LLMs, a perceivable lack in the pedagogical approach is observed (Macina et al., 2023b; Kasneci et al., 2023). These systems fall short of human tutors in emulating tutor’s speech, comprehending students, and assisting them in an appropriate way (Tack and Piech, 2022). In order to ensure that tutoring systems can effectively mimic the qualities of an actual tutor, the need for the integration of pedagogical attributes into the design of these systems is underscored (Du Boulay and Luckin, 2016).

## 2.2 Tutoring Dialogue Dataset

In the field of general natural language processing, the procedure of identifying the dialogue act serves as the preliminary phase in modeling intricate discourse structures (Stolcke et al., 2000; Budzianowski et al., 2018; Saha et al., 2020). Similarly, within the realm of education, related works also focus on dialogue acts within the tutoring dialogues (Caines et al., 2022; Tan et al., 2023; Lin et al., 2023; Samei et al., 2015). In contrast to other areas, tutoring dialogue acts possess distinct characteristics in that they include the instructional intentions of tutors, with the objective of elevating the effectiveness of the tutoring process. There exist extensive studies dedicated to the classification of teaching strategies within the educational field (Brown and Lee, 2015; Brown, 2014; Walsh, 2006, 2013). While numerous studies of CITS draw from this established body of knowledge, they often encounter constraints. Due to the complexity of incorporating an expansive range of pedagogical methods within CITS, numerous studies often narrow the scope to a limited set of teaching strategies (Alic et al., 2022; Demszky and Hill, 2022), or alternatively, focus solely on teaching simple knowledge concepts which require a more confined set of instructional strategies (Stasaski et al., 2020). To address this limitation, our work endeavors to integrate a diverse array of teaching methodologies, devised in collaboration with domain experts.

There is a considerable variation across existing tutoring dialogue datasets in terms of its specific composition. There are tutoring dialogues assembled from interactions between learners and automated systems (Huang et al., 2022; Macina et al., 2023a). Meanwhile, Stasaski et al., 2020 have generated dialogues wherein experts simulate both the tutor and student roles. Nonetheless, to authentically encapsulate the dynamic interactions between the tutor and the student, it becomes imperative to establish a real teaching context. Moreover, focusing on ESL, there exists a significant deficit in the availability of bilingual data. In the context of language learning, pedagogical strategies that simultaneously consider both languages – for instance, code-switching – play a pivotal role (Brown, 2014). Thus, the shortage of bilingual data is restricting the efficacy and scope with respect to language instruction (Nguyen et al., 2022).

Highlighting that our dataset, BIPED, is primarily focused on human-to-human tutoring dialogues including annotations, we provide an overview of descriptive statistics with a comparison to existing datasets in comparable settings in Table 1.

<table><tr><td>Metric</td><td>TSCC v2 (Caines et al., 2022)</td><td>TalkMoves (Suresh et al., 2022)</td><td>NCTE (Demszky and Hill, 2022)</td><td>BIPED</td></tr><tr><td># tutoring sessions</td><td>260</td><td>567</td><td>1,660</td><td>112</td></tr><tr><td># tutors</td><td>2</td><td></td><td>317</td><td>4</td></tr><tr><td># students</td><td>13</td><td></td><td>10,817</td><td>60</td></tr><tr><td>Avg. turns per session</td><td>159.5</td><td>211.2</td><td>350</td><td>105.7</td></tr><tr><td>Avg. words per turn</td><td>T 10.8 / S 6.4</td><td>T 8.2 / S 5.9</td><td>T 29.1 / S 4.0</td><td>T 31.7 / S 8.2</td></tr><tr><td># dialogue acts</td><td>T,S 12</td><td>T6/S4</td><td>T3/S2</td><td>T 34 /S9</td></tr><tr><td>domain</td><td>ESL</td><td>math</td><td>math</td><td>ESL</td></tr><tr><td>setting</td><td>1:1</td><td>classroom</td><td>classroom</td><td>1:1</td></tr></table>

Table 1: Comparison of BIPED with existing dialogue tutoring datasets: BIPED contains acts that are not only more diverse but also have greater pedagogical depth compared to previous datasets.

## 3 Dataset: BIPED

This section describes our dataset, BIPED, detailing its design, collection, annotation process, and essential statistics.

## 3.1 Data Design

We aim to construct a comprehensive tutoring dataset that contains the entire dialogues from tutoring sessions, with the corresponding dialogue acts and encompassing contents labeled on each utterance. Additionally, we categorized the correctness of student responses to inquiries posed by the tutor. To provide a clearer understanding of the structure of our dataset, a sample of the dataset is displayed in Figure 1.

## 3.2 Data Collection

For the data collection through one-to-one tutoring sessions, we recruited four experienced ESL tutors and 60 students with a willingness to pursue English studies. Every student was partnered with a single tutor, and over the course of three days, they had three successive 30-minute one-to-one tutoring sessions. Prior to the first session, tutors were given an additional 10-minute to comprehend the student’s individual status, including their particular objectives for studying English, their methods of studying English, and also their personal interests. The learning contents, developed by subject matter experts in ESL teaching, consisted of an English article along with activities focused on vocabulary and reading comprehension. Based on the student’s pretest results and the ongoing assessment of their progress throughout the class, including the preceded 10-minute talk, the tutors were required to adapt their teaching style to each unique learner, selectively determining the learning materials to cover. The tutoring sessions were held in an audioonly setting, and the transcriptions of these sessions were obtained with the assistance of Whisper (Radford et al., 2023).

The dataset contains 112 tutoring sessions\*, as outlined in Table 1. On average, each session consisted of roughly 105.7 turns. Analysis reveals that the utterances made by tutors in a single turn contained an average of 31.7 words, while the student’s utterances were more concise, averaging 8.2 words per utterance.

In order to confirm the educational significance of the tutoring sessions, students were required to take a pre-test and post-test surrounding these sessions. Drawing upon the computation of the normalized learning gain, the average result was fount to be 0.089, demonstrating the tutoring sessions’ effectiveness in enhancing the students’ English proficiency. A detailed explanation about the pre/post-test design and the evaluation of learning gain from these tests is described in Appendix A and Appendix B, respectively. Additionally, a feedback survey was deployed to understand their perspectives and feelings towards the courses. Overall, the feedback suggested a general contentment with the educational experience provided. For instance, for the questions “To what extent did you find the class satisfactory?”, “How suitable for you was the teaching method employed by the tutor during the class?”, “Was the class conducted with consideration for your linguistic background and cultural understanding?”, the respondents indicated high approval with average scores of 4.65, 4.60, and 4.68 out of a possible 5, respectively. These results support the pedagogical excellence of our dataset.

<table><tr><td>ex) Activity 3-9 (Gap Fills)</td></tr><tr><td>You likely won&#x27;t get answers from the other person, but a therapist can help you these issues in an introspective way, Durvasula says.</td></tr><tr><td>1) Vent 2) Navigate 3) Abuse 4) Withdraw</td></tr></table>

<table><tr><td rowspan=1 colspan=1>Speaker</td><td rowspan=1 colspan=1>Utterance</td><td rowspan=1 colspan=1>Act</td><td rowspan=1 colspan=1>Correctness</td><td rowspan=1 colspan=1>Content ID</td></tr><tr><td rowspan=1 colspan=1>tutor</td><td rowspan=1 colspan=1>.vi-(eng) Good. Shall we move on to Activity3-9? Guess what number the answer is?</td><td rowspan=1 colspan=1>t.assess.display_question</td><td rowspan=1 colspan=1></td><td rowspan=7 colspan=1>Activity3-9</td></tr><tr><td rowspan=1 colspan=1>student</td><td rowspan=1 colspan=1>navigate?</td><td rowspan=1 colspan=1>s.answer.answer</td><td rowspan=1 colspan=1>high</td></tr><tr><td rowspan=1 colspan=1>tutor</td><td rowspan=1 colspan=1>.&quot;navigate&quot;0. &quot;navigate these issues in an introspective&quot;&quot;vi&quot;.(eng) Correct. &quot;navigate&quot; is also an English expression that we can use well. In the sentence&quot;navigate these issues in an introspective way,&quot; &quot;navigate&quot; means to guide well, to give gooddirections. That&#x27;s what it means.</td><td rowspan=1 colspan=1>t.teach.direct_answer</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>tutor</td><td rowspan=1 colspan=1> &quot;introspective&quot;  ?(eng) What does &quot;introspective&quot; mean here?</td><td rowspan=1 colspan=1>t.assess.follow_up_question</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>student</td><td rowspan=1 colspan=1>(eng) Looking inward?</td><td rowspan=1 colspan=1>s.answer.answer</td><td rowspan=1 colspan=1>high</td></tr><tr><td rowspan=1 colspan=1>tutor</td><td rowspan=1 colspan=1>.(eng) Very good.</td><td rowspan=1 colspan=1>t.engage.encourage</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>tutor</td><td rowspan=1 colspan=1>&quot;intro&quot;,&quot;spect&quot;&quot;spectrum&quot;&quot;spect&quot;.&quot;spectrum&quot;,.(eng) &quot;intro&quot; means looking inside, and &quot;spect&quot; is as in &quot;spectrum&quot;. &quot;spectrum&quot; has the feelingof looking at something. So &quot;introspective&quot; means in a self-reflective way, looking inward.</td><td rowspan=1 colspan=1>t.teach.method.vocab_expression.etymology</td><td rowspan=1 colspan=1></td></tr></table>

Figure 1: Example of our dataset, BIPED. It includes a series of dialogues between a tutor and a student, annotated with dialogue acts, content information, and the correctness of student responses.

![](images/4998526f57fe8c9f139592a1f6c8e6955650ff81695a536fdbe74a6eaed12120.jpg)  
Figure 2: Distribution of tutor acts in BIPED

## 3.3 Dialogue Act Design

To accurately encapsulate the diverse intentions of both tutors and students within our dataset, we defined 34 tutor acts and 9 student acts in collaboration with experienced ESL instructors. Our taxonomy was grounded in prior research (Caines et al., 2022; Stasaski et al., 2020), studies related to teaching methods (Brown, 2014; Brown and Lee, 2015), and underwent further refinements after observing multiple pilot classes. As the tutor acts play a crucial role in influencing the efficacy of a tutoring session, we have accorded significant importance to the process of categorizing tutor acts. Tutor acts have been classified into five primary categories:

• General: general utterances including greeting, farewell, small talk, short reply, etc.

• Operational: utterances designed to facilitate the smooth operation of the class.

• Assessment: utterances intended to evaluate the student’s learning progress.

• Teaching: utterances dedicated to teach a certain knowledge to the student.

• Engagement: utterances tailored to encourage student participation and involvement.

The respective categories encompass 1,3,4,22, and 4 distinct tutor acts. Prior research has primarily concentrated on broader behaviors, and current language models have shown proficiency in their application to such contexts. Given the limitations observed within current taxonomies in terms of capturing the diversity of teaching methods, our study endeavors to provide a more detailed examination of the "Teaching" category, while drawing upon preliminary studies in reference to the remaining categories. The detailed process for developing the tutor act taxonomy is described in Appendix C.

Concurrently, in a more simplified manner, student acts are divided into four distinct categories: General, Operational, Question, and Answer. These particular categories encompass 1, 3, 2, and 3 unique student acts respectively. The complete list of both tutor and student acts, along with corresponding examples is exhibited in the Appendix D and E.

## 3.4 Data Annotation

## Annotation Process

Initially, we implemented a set of processing steps, including the anonymization and correction of grammatical errors. Subsequently, with the taxonomy design described in the Section 3.3, each utterance were annotated with the relevant act labels by our trained human annotators. In cases where a single utterance encompassed multiple acts, the utterance was split to ensure that each utterance corresponded to a single action. These segmented utterances are referred to as act-level utterances. That is, a single turn might consist of several actlevel utterances. Our observations revealed that, on average, there were 145 act-level utterances per session, with tutors contributing 83.2 and students contributing 53.1. This indicates that tutors employed a variety of actions more frequently within the session. The distribution of tutor acts is presented in Figure 2. It is noteworthy that the dataset exhibits a wide range of diverse acts, enhancing the pedagogical richness of our dataset.

## Annotation Quality and Analysis

To validate the annotation quality, we investigated the annotation agreement among annotators. We sampled 495 utterances from 4 tutoring sessions, consisting of 313 utterances from teachers and 182 from students. Three independent were tasked with annotating the tutor acts and the student acts. The overall annotator agreement, measured using Fleiss’s Kappa, was 0.70. Notably, despite a significant number of tutor acts(34), the agreement result regarding tutor acts was substantial, with a score of 0.64. Similarly, the agreement regarding student acts was 0.72, indicating a high level of concordance. The most disagreed act pair among teaching methods, denoted as "t.teach.method" in Appendix D, was “t.teach.method.vocab\_expression.word\_usage” and “t.teach.method.vocab\_expression.context”. This disparity likely arises from their shared focus on the contextual usage of vocabulary expression within definitions. Also, it was noted that there are instances where acts, which could be annotated with more specific labels within the “t.teach.method”, are instead categorized under “t.teach.direct\_answer”.

## 4 Framework

To build CITS that strategically incorporates pedagogical considerations, we follow the framework (Macina et al., 2023c; Wang et al., 2023b) where the LLM (1) chooses the appropriate tutor act, (2) then generates the response accordingly. We advocate this two-step approach for the following two reasons.

## Interpretability

First, the proposed two-step framework allows us to better understand the intention of the model (tutor acts) when generating a certain response for a given dialogue context. As a consequence, this enables us to more rigorously evaluate the model from a pedagogical perspective. Specifically, human experts may now assess whether the model correctly selected the appropriate type of act, and whether the generated response is suitable.

## Controllability

More importantly, the proposed framework allows for greater controllability of CITS. By conditioning the LLM to a specific tutor act class, we ensure more focused response generation without deviation from the target act. We expect this to be essential when generating more sophisticated responses (e.g., teaching etymology, code-mixing), for these are sparsely distributed compared to more generic responses such as simple direct answer as shown in Figure 2.

Moreover, a crucial aspect for controllability is the model’s amenability to improvement via continual refinement. Our framework allows for a divide-and-conquer approach for model refinement; namely, given dialogue context and student response, instead of requiring the human expert to generate a full response, we may let them simply label the best teacher response class (for instruction tuning), or a ranked list of teacher response classes (for RLHF). As for improving response generation, we may consider sophisticated data augmentation schemes, where we synthesize responses for a given tutor act class based on retrieved samples of the same class.

## 5 Method

In this section, we introduce two concrete implementations of the framework proposed in Section 4. First implementation leverages GPT-4’s incontext learning ability and is easily implemented via prompting, while the second implementation instruction-tunes open-sourced LLM to perform various relevant tasks.

<table><tr><td>Type</td><td>Instruction</td></tr><tr><td>(1) Act prediction</td><td>### Instruction: Select an appropriate next tutor act based on the given &#x27;dialogue context&#x27; and ’act candidates&#x27;. In the context, tutor uses various tutor act and contents(activities or passages), depending on the student&#x27;s abilities and questions. Tutor act represents a teaching method that you can exploit. Choose the best act catering to the context and understanding the meaning of each act.&quot; ### Input: - Context: {dialogue context (Appendix G)}, - Act candidates: {act candidates (Appendix D)}</td></tr><tr><td>(2) Utterance genera- tion</td><td>### Response: t.teach.method.vocab.expression.etymology ### Instruction: As an English tutor for a Korean student, your job is to create coherent and tailored responses based on the given &#x27;dialogue context&#x27; and &#x27;tutor&#x27;s act&#x27;. In the context, tutor uses various teaching acts and contents(activities or passages), depending on the student&#x27;s abilities and questions. The given action is a teaching method that you have to exploit. ### Input: - Context: {dialogue context (Appendix G)}, - Act: t.teach.method.vocab.expression.etymology, Explain a word or a phrase based on its etymology. When possible, discuss the prefix and suffix of the word. ### Response: &quot;intro&quot;, &quot;spect&quot;&quot;spectrum&quot;&quot;spect&quot;. &quot;spectrum&quot;.&quot;introspective&quot;, ロ. (eng) &quot;Intro&quot; means looking inward. &quot;spect&quot; is same as &quot;spect&quot;</td></tr><tr><td>(3) Missing context in- ference</td><td>in &quot;Spectrum&quot; and feels like seeing something. So, &quot;introspective&quot; means in a self-reflective manner, looking inward. ### Instruction: Predict a speaker and their utterance to fit between the given &#x27;dialogue context&#x27; of a student and a teacher and the final &#x27;utterance&#x27; made by the teacher. ### Input: - Context: {partial context (Appendix G)}, - Utterance: &quot;intro&quot;,&quot;spect&quot;&quot;spectrum&quot; &quot;spect&quot;. &quot;spectrum&quot;.&quot;introspective&quot; 日 ロ哥. (eng) &quot;Intro&quot; means looking inward. &quot;spect&quot; is same as &quot;spect&quot; in &quot;Spectrum&quot; and feels like seeing something. So, &quot;introspective&quot; means in a self-reflective</td></tr><tr><td>(4) Minority act utter- ance generation ### Response: &quot;Intro&quot;&quot;&quot;, &quot;spect&quot;&quot;&quot;.  &quot;Intro-</td><td>manner, looking inward. ###Response: tutor, }豆. (eng) Very good. ### Instruction: As an English teacher for Korean students, generate tutor&#x27;s utterance to teach the given learning content with the act. ### Input: - Content: [Activity 3-9] You likely won&#x27;t get answers from the other person, but a therapist can help you these issues in an introspective way, Durvasula says. (1) Vent (2) Navigate (3) Abuse (4) Withdraw - Act: t.teach.method.vocab.expression.etymology, Explain a word or a phrase based on its ety- mology. When possible, discuss the prefix and suffix of the word.</td></tr></table>

Table 2: Proposed instruction types for fine-tuning. Examples of input formats are illustrated in the appendix.

## 5.1 GPT-4 Prompting

We first consider a GPT-4-based implementation via extensive prompting. While this implementation does not require model training, it is costly and requires a large amount of tokens for in-context learning.

As outlined in Appendix F, our prompt structure adheres to our framework. The first step involves choosing an effective tutor act from a provided tutor acts list. Next, an utterance that aligns with this chosen tutor act is generated. These steps consider both dialogue context and learning content provided in the test set. For few-shot learning, an example scenario from the training set that matches the tutor act in the task scenario is also provided. This includes dialogue context, learning content, a chosen tutor act, and a generated utterance.

## 5.2 Instruction Fine-tuning

We consider instruction-tuning a pre-trained LLM on the dataset introduced in Section 3. Our goal is to demonstrate that, with properly curated datasets and appropriate fine-tuning methods, we may enable open-source models to rival GPT-4’s performance. We instruction-tuned open-sourced LLM on four sub-tasks, each of which we detail below.

## Task 1. Act prediction

To make our model select the most effective teaching strategy, we task the model to predict the next tutor act based on given learning content and dialogue context. Given a dialogue in the training dataset, we created instruction samples for every dialogue turn of the tutor.

## Task 2. Utterance generation

As a conversational tutor for Korean L2 learners of English, the model must be able to generate responses that are (1) fluent in both Korean and English; and (2) contextually relevant and matching the tutor act the response is conditioned on. To achieve this, we devise a task of generating the tutor’s response based on learning content, dialogue history, and the conditioning tutor act. By training the model to imitate expert human tutor’s responses, we expect the model to obtain bilingual fluency, as well as the ability to generate appropriate response based on the aforementioned conditioning information.

## Task 3. Missing context inference

We found out that, trained on Task 1-2, the model often generated responses that are not properly grounded in the recent dialogue context. This comes from the inherent task difficulty: compared to simple conversational tasks, teaching often requires a purposeful and focused long-context generation, where one small digression may hurt the overall quality of the generated response. To address this issue, we devise the following task: Given learning content, dialogue history, and corresponding teacher response, we remove this and that. The model is then tasked to predict the missing context given the rest of the information. This way, we expect the model’s utterance to be locally more consistent.

## Task 4. Minority act utterance generation

Finally, we notice that there are minority tutor act classes that are, despite their pedagogical important, sparsely distributed in the training dataset. Therefore, to make sure the model understand these act classes better, we additionally collected the following data - for a given learning content and a selected group of minority act classes, we asked experts to provide plausible utterances for the corresponding act classes. Then, we tasked the model to generate these utterances conditioned on the corresponding learning content and act class. Although this additional dataset lack specific dialogue context, we expect it to guide the model in generating utterances for minority act classes.

## 6 Experiments

## 6.1 Dataset split

The experiment involved dividing BIPED randomly into training and test sets, organised by session. This was done based on the distribution of tutors within the dataset, allocating two sessions to one tutor, and one session each to the remaining tutors. BIPED of 112 tutoring sessions was thus split into 102 training sessions and 10 test sessions.

Extra data for generating minority act utterances, part of Task 4, were only included in the training set. The entire training data was reformulated into different instruction-tuning formats for each task.

The test set was devised into 220 scenarios, 10 unique scenarios for every one of the 22 tutor acts in the teaching category. Each test scenario was shaped around an act-level utterance initiated by a human tutor applying the respective tutor act, supported by the associated dialogue context and learning content.

## 6.2 Evaluation metric

We conducted evaluations of the models based on our framework, focusing on two main aspects. Firstly, we evaluated the accuracy and invariability of tutor act predictions. For accuracy, we assessed the rate at which the model’s predicted tutor act aligned with the actual act selected by the tutor in the test set. For invariability, we noted the frequency of each selected tutor act and calculated the mean absolute difference from a target count of 10.

Secondly, we assessed the generated utterances. To measure the similarity between the generated and target tutor utterances, we utilized the sacrebleu implementation of BLEU(sBLEU) (Post, 2018) and BERT Score (Zhang\* et al., 2020) using bert-base-multilingual-cased checkpoint. We note that the fine-tuned model sequentially predicted tutor acts and generated utterances based on these predictions. Additionally, we measured the coherence between generated utterances and previous ones in the given dialogue context using embedding similarity, specifically relying on OpenAI’s text embedding(text-embedding-3-large).

<table><tr><td colspan="3">Tutor Act</td><td colspan="3">Utterance</td></tr><tr><td>Metric</td><td>Accuracy↑</td><td>Invariability↓</td><td>sBLEU↑</td><td>BERTScore↑ Embedding sim.↑</td><td>Length</td></tr><tr><td>Base GPT</td><td>0.155</td><td>6.955</td><td>1.802 0.655</td><td>0.635</td><td> $8 0 \pm 3 1$ </td></tr><tr><td>GPT-3.5 (0-shot)</td><td></td><td></td><td>1.849 0.646</td><td>0.661</td><td> $\bar { 4 } \bar { 5 } \bar { \pm } \bar { 3 } \bar { 6 }$ </td></tr><tr><td>GPT-3.5 (1-shot)</td><td></td><td></td><td>2.423 0.658</td><td>0.704</td><td> $4 5 \pm 3 4$ </td></tr><tr><td>GPT-4 (0-shot)</td><td></td><td></td><td>1.687 0.659</td><td>0.668</td><td> $6 6 \pm 2 8$ </td></tr><tr><td>GPT-4 (1-shot)</td><td>0.250</td><td>7.682</td><td>2.608 0.669</td><td>0.686</td><td>58 ± 29</td></tr><tr><td>Finetuned</td><td>0.259</td><td>5.273</td><td>15.874 0.716</td><td>0.680</td><td>39 ±52</td></tr><tr><td>Finetuned (ablation)</td><td></td><td></td><td>14.199 0.708</td><td>0.673</td><td> $3 4 \pm 2 0$ </td></tr><tr><td>Finetuned (baseline)</td><td></td><td></td><td>14.426 0.689</td><td>0.670</td><td> $2 5 \pm 2 5$ </td></tr><tr><td>Ground-truth</td><td>1.0</td><td></td><td>100. 1.</td><td>0.707</td><td> $2 \bar { 7 } \pm \bar { 2 } \bar { 2 }$ </td></tr></table>

Table 3: Results of prompted GPTs and finetuned models.  
![](images/253cd6141fa39b7acdf0451ad27b987a39e48522609735084f1ccea269f22ffa.jpg)  
Figure 3: The comparative distributions of chosen tutor acts. The grey bars represent the distribution of tutor acts in the test set while the blue bars denote the distribution of chosen tutor acts by Base GPT, our fine-tuned model, and GPT-4 (1-shot).

## 6.3 Training details

For fine-tuning we utilize SOLAR-KO-10.7B( (L. Junbum, 2024)), a model further trained on a Korean corpus to enhance bilingual comprehension and fluency. We adopt AdamW optimizer (Loshchilov and Hutter, 2019) with a learning rate of 2e-5 and a batch size of 64. The maximum length is 4096. We train the model using 8 V100 GPUs with deepspeed Zero-3 offload for 2 epochs.

## 6.4 Results

The findings from our experiment, illustrated in the Table 3, are compared against various models ranging from base GPT to fine-tuned models. Ground-truth represents the reformulated data gathered from human tutoring present in the test set. Base GPT in Table 3 corresponds to the GPT-4 model bereft of any tutor act classification or our human tutoring examples, where its prompt is given in Appendix F. We have also drawn a comparison involving GPT-3.5(gpt-3.5-turbo-16) and GPT-4 in a zero-shot and one-shot configuration. While they all adhere to the identical prompt structure, (as seen in Appendix F), they diverge in aspects such as model compatibility and whether an example scenario is included. Lastly, we compared our two fine-tuned models. Fine-tuned refers to our instruction-tuned model, discussed in the method section 5, which was trained with instruction tasks 1 through 4. Conversely, ’Fine-tuned (ablation)’ refers to the model trained with instruction tasks 1 and 2 only. We also included a comparison with a potential baseline, denoted as ’Fine-tuned (baseline)’. This model was trained to sequentially generate the following tutor act and corresponding utterance, conditioned on the given dialogue context without any instructions. This approach is similar to MATHDIAL (Macina et al., 2023a), where the tutoring model was trained to generate utterances given the context and tutor act only.

In assessing the tutor act criteria, our fine-tuned model outperforms other models in both accuracy and invariability as shown in Table 3. Although the accuracy of 0.259 may initially appear low, it’s important to consider the complexity of choosing from amongst 34 distinct tutor acts. Moreover, a lower invariability score of 5.273 is indicative of our fine-tuned model’s ability to utilize a wider array of tutor acts, thereby demonstrating more diverse pedagogical approaches. While the invariability score provides one aspect of variability, Figure 3 presents a distribution of chosen tutor acts for the comparative models, which offers a more comprehensive view. Clearly, our fine-tuned model exhibits the greatest number of distinct tutor act classes.

With respect to utterance-related metrics, our fine-tuned model delivers outstanding performances. It records the highest sBLEU score at 15.874 and BERTScore at 0.716, demonstrating its commendable ability in generating utterances that closely mirror the target tutor utterances. While the embedding similarity score of our fine-tuned model is slightly lower than that of GPT-3.5 (1-shot), it still exhibits higher coherence between individual utterances and their corresponding dialogue context.

In terms of the length of the generated utterances, our model closely mimics the length of tutor utterances, whereas the general GPT models often produce unnecessarily long responses. We interpret this as positive, based on the assumption that the appropriate length of a tutor’s utterance plays a vital role in maintaining student engagement.

Lastly, the fine-tuned model, trained with tasks 1 through 4, has outperformed the ablation model, which was trained with only partial instruction tasks, and the baseline model, which used simple conditional generation, across all evaluated metrics. This outcome highlights the added value of integrating additional instruction tasks (i.e., tasks 3 and 4) for improved tutoring performance.

## 6.5 Example of Interactive Evaluation

Our results primarily focus on static evaluation, where we predict the next tutor act and generate a corresponding response in one step for the test scenarios. However, a complete evaluation of the tutoring system, developed using our proposed models, necessitates interactive assessment with human students. This aspect of evaluation will be subjected to further study. Regardless, we offer a sneak peak into how an interaction plays out between a human and two tutoring models, namely GPT-4 (1-shot) and our fine-tuned model.

The dialogue examples are included in Appendix H. Generally, when interacting with our tutoring systems, GPT-4 (1-shot) tends to produce lengthier responses, focusing mainly on transferring knowledge to students. Conversely, our fine-tuned model generates concise responses and actively promotes student participation. An interesting observation is that when a student insists that an incorrect answer is correct, GPT-4 maintains its answer, yet appears to accept the student’s assertion (t.teach.emphasis). In contrast, our fine-tuned model re-emphasizes the correct answer (t.teach.repair), thereby highlighting the student’s error.

## 7 Conclusion

In this study, we demonstrated the effectiveness of our proposed models in developing conversational intelligent tutoring systems for ESL learners. They excelled in replicating the utterance style of human tutors and implementing various and appropriate pedagogical strategies. Much of this improvement can be attributed to our newly constructed Bilingual and Pedagogically-Informed Tutoring Dataset (BIPED). Additionally, our two-step framework and the use of four instruction tuning tasks further contributed to the models’ improvement.

## 8 Limitations

Despite the encouraging results, our study has some limitations. First, our project lacks an interactive evaluation, left as future research direction. A formal evaluation of complete tutoring systems through human study would provide useful insights into their efficacy. Second, we were unable to utilize larger parameter models like LLaMa 70b (Touvron et al., 2023), which could potentially improve the teaching capabilities of CITS. This restriction is worth considering for future studies, particularly with advancing computational resources and technologies. Finally, the use of diverse instruction tuning tasks for multi-task modeling, using existing datasets and data augmentation, is not covered in this study. We believe this aspect carries opportunities for more nuanced model development and will be explored in subsequent work.

## 9 Potential Risks

Our research holds great promise, but it’s also important to acknowledge potential risks. Firstly, Large Language Models might inadvertently teach misinformation due to their susceptibility to the ’hallucination’ phenomenon, which generates information that isn’t grounded in real-world facts. Secondly, there’s risk of over-reliance on these intelligent tutoring systems. While they can play a pivotal role in language learning, they can’t entirely mimic human tutors, especially in terms of adaptive strategies and emotional intelligence.

## Acknowledgment

We thank our four dedicated ESL tutors, 60 students, and pilot study participants for their invaluable contributions. Special thanks to Yena Zhang for expertly managing all tutoring sessions and the data collection procedure. We also appreciate the annotators for their detailed work on our dataset, and the reviewers for their constructive comments, which greatly improved the quality of this paper.

## References

R Abdelghani, YH Wang, X Yuan, T Wang, H Sauzéon, and PY Oudeyer. 2022. Gpt-3-driven pedagogical agents for training children’s curious question-asking skills. arXiv, 2211.

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv:2303.08774.

Sterling Alic, Dorottya Demszky, Zid Mancenido, Jing Liu, Heather Hill, and Dan Jurafsky. 2022. Computationally identifying funneling and focusing questions in classroom discourse. arXiv:2208.04715.

H Douglas Brown. 2014. Principles of language learning and teaching: A course in second language acquisition. Pearson.

H Douglas Brown and Heekyeong Lee. 2015. Teaching by Principles: An Interactive Approach to Language Pedagogy. P. Ed Australia.

Paweł Budzianowski, Tsung-Hsien Wen, Bo-Hsiang Tseng, Inigo Casanueva, Stefan Ultes, Osman Ramadan, and Milica Gašic. 2018. Multiwoz–a large-´ scale multi-domain wizard-of-oz dataset for taskoriented dialogue modelling. arXiv:1810.00278.

Andrew Caines, Helen Yannakoudakis, Helen Allen, Pascual Pérez-Paredes, Bill Byrne, and Paula Buttery. 2022. The teacher-student chatroom corpus version 2: more lessons, new annotation, automatic detection of sequence shifts. In Swedish Language Technology Conference and NLP4CALL, pages 23–35.

Dorottya Demszky and Heather Hill. 2022. The ncte transcripts: A dataset of elementary math classroom transcripts. arXiv:2211.11772.

Benedict Du Boulay and Rosemary Luckin. 2016. Modelling human teaching tactics and strategies for tutoring systems: 14 years on. International Journal of Artificial Intelligence in Education, 26:393–404.

Richard Hake. 1998. Interactive-engagement versus traditional methods: A six-thousand-student survey of mechanics test data for introductory physics courses. American Journal ofPhysics - AMER J PHYS, 66.

Jin-Xia Huang, Yohan Lee, and Oh-Woog Kwon. 2022. Direct: Toward dialogue-based reading comprehension tutoring. IEEE Access, 11:8978–8987.

Enkelejda Kasneci, Kathrin Seßler, Stefan Küchemann, Maria Bannert, Daryna Dementieva, Frank Fischer, Urs Gasser, Georg Groh, Stephan Günnemann, Eyke Hüllermeier, et al. 2023. Chatgpt for good? on opportunities and challenges of large language models for education. Learning and individual differences, 103:102274.

L. Junbum. 2024. Solar-ko-10.7b.

Seungjun Lee, Yoonna Jang, Chanjun Park, Jungseob Lee, Jaehyung Seo, Hyeonseok Moon, Sugyeong Eo, Seounghoon Lee, Bernardo Yahya, and Heuiseok Lim. 2023. PEEP-talk: A situational dialogue-based chatbot for English education. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 3: System Demonstrations), pages 190–207, Toronto, Canada. Association for Computational Linguistics.

Jionghao Lin, Wei Tan, Ngoc Dang Nguyen, David Lang, Lan Du, Wray Buntine, Richard Beare, Guanliang Chen, and Dragan Gaševic. 2023. Robust ed-´ ucational dialogue act classifiers with low-resource and imbalanced datasets. In International Conference on Artificial Intelligence in Education, pages 114–125. Springer.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In International Conference on Learning Representations.

Jakub Macina, Nico Daheim, Sankalan Pal Chowdhury, Tanmay Sinha, Manu Kapur, Iryna Gurevych, and Mrinmaya Sachan. 2023a. Mathdial: A dialogue tutoring dataset with rich pedagogical properties grounded in math reasoning problems. arXiv:2305.14536.

Jakub Macina, Nico Daheim, Lingzhi Wang, Tanmay Sinha, Manu Kapur, Iryna Gurevych, and Mrinmaya Sachan. 2023b. Opportunities and challenges in neural dialog tutoring. arXiv:2301.09919.

Jakub Macina, Nico Daheim, Lingzhi Wang, Tanmay Sinha, Manu Kapur, Iryna Gurevych, and Mrinmaya Sachan. 2023c. Opportunities and challenges in neural dialog tutoring. In Proceedings of the 17th Conference ofthe European Chapter ofthe Association for Computational Linguistics, pages 2357–2372, Dubrovnik, Croatia. Association for Computational Linguistics.

Li Nguyen, Zheng Yuan, and Graham Seed. 2022. Building educational technologies for codeswitching: Current practices, difficulties and future directions. Languages, 7(3):220.

Minju Park, Sojung Kim, Seunghyun Lee, Soonwoo Kwon, and Kyuseok Kim. 2024. Empowering personalized learning through a conversation-based tutoring system with student modeling. In Extended

Abstracts ofthe 2024 CHI Conference on Human Factors in Computing Systems, CHI EA ’24, New York, NY, USA. Association for Computing Machinery.

Matt Post. 2018. A call for clarity in reporting BLEU scores. In Proceedings of the Third Conference on Machine Translation: Research Papers, pages 186– 191, Brussels, Belgium. Association for Computational Linguistics.

Kun Qian, Ryan Shea, Yu Li, Luke Kutszik Fryer, and Zhou Yu. 2023. User adaptive language learning chatbots with a curriculum. arXiv:2304.05489.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2023. Robust speech recognition via large-scale weak supervision. In International Conference on Machine Learning, pages 28492–28518. PMLR.

Tulika Saha, Aditya Patra, Sriparna Saha, and Pushpak Bhattacharyya. 2020. Towards emotion-aided multimodal dialogue act classification. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4361–4372.

Borhan Samei, Vasile Rus, Benjamin D. Nye, and Donald M. Morrison. 2015. Hierarchical dialogue act classification in online tutoring sessions. In Educational Data Mining.

Katherine Stasaski, Kimberly Kao, and Marti A Hearst. 2020. Cima: A large open access dialogue dataset for tutoring. In Proceedings ofthe Fifteenth Workshop on Innovative Use ofNLPfor Building Educational Applications, pages 52–64.

Andreas Stolcke, Klaus Ries, Noah Coccaro, Elizabeth Shriberg, Rebecca Bates, Daniel Jurafsky, Paul Taylor, Rachel Martin, Carol Van Ess-Dykema, and Marie Meteer. 2000. Dialogue act modeling for automatic tagging and recognition of conversational speech. Computational linguistics, 26(3):339–373.

Abhijit Suresh, Jennifer Jacobs, Charis Harty, Margaret Perkoff, James H Martin, and Tamara Sumner. 2022. The talkmoves dataset: K-12 mathematics lesson transcripts annotated for teacher and student discursive moves. arXiv:2204.09652.

Anaïs Tack and Chris Piech. 2022. The ai teacher test: Measuring the pedagogical ability of blender and gpt-3 in educational dialogues. arXiv:2205.07540.

Wei Tan, Jionghao Lin, David Lang, Guanliang Chen, Dragan Gaševic, Lan Du, and Wray Buntine. 2023.´ Does informativeness matter? active learning for educational dialogue act classification. In International Conference on Artificial Intelligence in Education, pages 176–188. Springer.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu,

Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023. Llama 2: Open foundation and finetuned chat models.

Steve Walsh. 2006. Investigating classroom discourse. Routledge.

Steve Walsh. 2013. Classroom discourse and teacher development. Edinburgh University Press.

Lingzhi Wang, Mrinmaya Sachan, Xingshan Zeng, and Kam-Fai Wong. 2023a. Strategize before teaching: A conversational tutoring system with pedagogy selfdistillation. arXiv:2302.13496.

Lingzhi Wang, Mrinmaya Sachan, Xingshan Zeng, and Kam-Fai Wong. 2023b. Strategize before teaching: A conversational tutoring system with pedagogy self-distillation. In Findings of the Association for Computational Linguistics: EACL 2023, pages 2268– 2274, Dubrovnik, Croatia. Association for Computational Linguistics.

Tianyi Zhang\*, Varsha Kishore\*, Felix Wu\*, Kilian Q. Weinberger, and Yoav Artzi. 2020. Bertscore: Evaluating text generation with bert. In International Conference on Learning Representations.

## A Pre/Post-test Design

Pre-test and post-test questions were created by experienced tutors who also developed the learning materials. Both tests consist of six parts:

• (Part 1) reading comprehension questions (3 questions each)

• (Part 2) true/false questions (4 each)

• (Part 3) paraphrasing right/wrong (3 each)

• (Part 4) vocabulary (9 each)

• (Part 5) synonyms and antonyms (pre-test 11, post-test 12)

• (part 6) phrasal verbs (pre-test 6, post-test 8). Following standard pre- and post-test designs, the tests evaluate common knowledge components with slight variations. In (Part 1), two distinct reading passages were used to assess reading comprehension skills on the same topic, including questions on main ideas, implied meanings, and sentence completion tasks. Additionally, (Parts 5 and 6) incorporate a few additional in-depth questions to better capture the possibility of improvement. After the test questions were developed, both the pre- and post-tests were validated through a pilot study with seven students and an expert education test developer.

## B Learning Gain Computation

After the creation and validation of pre- and posttest questions, students are asked to complete the tests both before and after the tutoring sessions. A common method to measure improvements in learning is to compute the raw gain, which is the absolute difference between the post-test and pretest scores. Although this method is straightforward, it fails to address the observation that higher pre-test scores often result in proportionally lower learning gains. Moreover, there are controversies regarding the lack of controls which pose further problems with using this method. To better assess learning gain, several modified gain scores have been proposed. Among these, we use the formula $G _ { 1 } = \frac { \mathrm { p \bar { o s t } - p r e } } { 1 0 0 }$ , which represents the ratio of the average gain from the pre-test to the post-test to the maximum possible gain. This method was suggested by Hake, 1998.

## C Development Process of the Tutor Act Taxonomy

The process that we followed to develop the tutor act taxonomy is outlined below:

• (Step 1) We started by conducting a comprehensive literature review to provide a preliminary set of tutor acts. We included papers on task-oriented dialogue, open-domain dialogue, and intelligent tutoring systems (Samei et al., 2015; Stasaski et al., 2020; Caines et al., 2022; Tan et al., 2023; Macina et al., 2023a) in our review. We also considered papers and books relating to ESL/EFL teaching methods (Brown, 2014; Brown and Lee, 2015). This helped us to form an initial set of possible tutor acts.

• (Step 2) Subsequently, five researchers and two experienced tutors were engaged to delineate five primary categories which encapsulated all the candidates. The group also worked on clustering tutor acts that pertained to each category.

• (Step 3) To verify our initially-developed tutor acts, we organized pilot tutoring sessions with seven students and collected a sample tutoring dataset. We then jointly labeled this data based on the classifications determined in Step 2.

• (Step 4) For utterances that were difficult to label, we began a process to add new labels as necessary.

• (Step 5) At the same time, we tested the GPT-4’s ability to generate appropriate utterances for each tutor acts. We combined broader acts where the language model was already performing well, such as under the "General" and "Operational" categories. On the other hand, in areas where the performance was under par, particularly in the "Teaching" category, we concentrated our efforts on refining the categorization of our tutor acts. Our goal was to better enable the LLM to use a broad array of teaching methods. This led us to define 22 specific tutor acts within the Teaching category.

## D Tutor Act Classification

Translations are provided below the examples in gray, and words that are intended to be uttered in English are bolded.

<table><tr><td>Tutor Act</td><td>Explanation</td></tr><tr><td>t.general</td><td>General utterances including greeting, farewell, small talk, short reply, etc. ex.“.”，“” (eng) "Hello.", "Are you ready?"</td></tr><tr><td>t.operational.overview</td><td>Give an overview of the tutoring session at the start of the session or provide the overview for the next session at the end of the session. .“”</td></tr><tr><td>t.operational.proceed</td><td>Notify the student that you will proceed with the class. When appropriate - for example, right after teaching a concept - ask if the student has any more questions before moving on. .“”</td></tr><tr><td>t.operational. clarification_question</td><td>When the student asks a question whose meaning is unclear, request clarification. You may additionally guess the intended meaning of the question and ask the student whether your understanding is correct. ex.“～?” (eng) "Are you referring to ~?"</td></tr><tr><td>t.assess.display_question</td><td>Refer to a question activity and ask the student the question. For example, for P2, questions can be from [P2 Synonyms &amp; Antonyms], [P2 Phrasal Verbs], and [P2 T/F Questions]. .“ctivity” (eng) "What do you think is the answer to number 9 in Activity 1?"</td></tr><tr><td>t.assess. confirmation_question</td><td>Ask to confirm the student's understanding of specific content. ex.“]入?” (eng) "Do you understand?"</td></tr><tr><td>t.assess. follow_up_question</td><td>After the student's initial response to the question displayed, assess their knowledge. If the student answered correctly, ask a follow-up question to confirm mastery of the knowledge concepts required to solve the original question. If incorrect, ask a question to pinpoint the specific lacking concept (vocabulary, phrase, grammar, etc.). For a T/F question, you may request the student for an evidence.</td></tr><tr><td>t.assess.result</td><td>(eng) "Why did you think so?" Inform the student about the assessment outcome. For example, this could be based on the pre-test, or mid-session assessment result. x. “re-test.” (eng) "Looking at your pre-test results, it seems like it would be good for you to focus on vocabu-</td></tr><tr><td>t.engage.encourage</td><td>lary." Provide emotional support, motivation, and positive feedback. ex.“.” (eng) "You did really well."</td></tr><tr><td>t.engage.perspective</td><td>Ask the student to share their personal opinion or viewpoint on the topic being discussed. After the student responds, smoothly integrate their perspective with the prepared learning materials. .“stin” (eng) "What did you think about ghosting when you first read this text?"</td></tr><tr><td>t.engage.activity</td><td>Whenever possible and appropriate, provide a fun fact relevant to the current conversation topic or learning material. .“ ，～”</td></tr><tr><td>t.engage. small_talk_activity</td><td>To naturally transition into learning, teach relevant English expressions based on small talk. The taught expressions do not have to be directly relevant to class material. “ a， . ll-timefavorite' ."</td></tr><tr><td>t.teach.emphasis</td><td>Emphasize the core concepts being taught by reiterating them within an utterance. The tutor chooses this strategy when they believe that through repetition or stressing certain points, the student's understanding and retention will improve. .“i</td></tr><tr><td>t.teach.direct_answer</td><td>In other words, it's like 'going under the radar.'" Provide a concise, straightforward, and simple answer to a student's question, coupled with a brief explanation. The tutor chooses this intervention when they believe that giving a direct answer is the most effective way to assist the student and avoid possible confusion, demonstrating a commitment to immediate assistance.</td></tr><tr><td></td><td>Repair the student's incorrect answer and provide an explanation as to why the student answer was wrong. The tutor opts for this intervention when they conclude that immediate correction and explana- tion of the error are the most effective ways to assist the student and prevent potential misunderstandings. .“”，“eit'ait'. 'taking the bait'豆.” (eng) "In this case, ∼ is more suitable than that.", "What you mentioned is not 'beit' but 'bait', so</td></tr><tr><td>t.teach.review</td><td>there is an expression like 'taking the bait'." Whenever appropriate, summarize the progress made in the session. Implement the intervention usually towards the end of a session, or before unveiling a new, interconnected topic, or following</td></tr><tr><td>t.teach.method.definition</td><td>comprehensive discussions. .“” (eng) "To summarize the important parts from the content just discussed, ~" When teaching a new word or an expression, explain what it means in Korean. This includes both</td></tr><tr><td></td><td>direct translation of the word, as well as elaborating on its meaning in detail; this method also includes a literal word-by-word translation of a sentence. .“～”，“?”，“ 4.”</td></tr><tr><td>t.teach.method.code_mix</td><td>(eng) "The Korean meaning of this word is ∼.", "What does ~ mean?", "If you translate this sentence literally, it means ~." Keep the key words in English within a sentence while the rest is translated into Korean. Unlike code-switching that swaps languages on a sentence level, code mixing introduces a few English words</td></tr><tr><td></td><td>or phrases within a Korean sentence. This mixing of languages aids in clarifying meanings, substitutes unfamiliar English terms with known Korean ones, and emphasizes important English words. ex.“ditch’?”,“Poor etiquette ” (eng) "Does it mean, 'She ditched her boyfriend?", "Since it's "poor etiquette", we shouldn't</td></tr><tr><td>t.teach.method. code_switch</td><td>ghost others, right? You can consider it within that context." When the student's English proficiency allows, switch to English briefly for a single turn. This switch can entail teaching specific English vocabulary words or phrases, asking a question in English, or</td></tr><tr><td>t.teach.method. vocab_expression.</td><td>giving an explanation in English, thus providing a brief initiation into English conversation. ex. (in English) "What's your opinion on using AI for music production?" Explain a word or a phrase based on its etymology. When possible, discuss the prefix and suffix of the</td></tr><tr><td>etymology</td><td>word. .“iycl'',cl' iyl”</td></tr><tr><td></td><td>(eng) "The word 'bicycle' originates from the Greek. The 'bi-' means 'two', and the '-cycle' means</td></tr><tr><td></td><td>'wheels'. So the 'bicycle' means 'a vehicle with two wheels.",</td></tr><tr><td></td><td></td></tr><tr><td>t.teach.method.</td><td></td></tr><tr><td></td><td>Teach synonyms for the word, or introduce similar expressions to the expression being taught.</td></tr><tr><td>vocab_expression.</td><td>. “a'''”,“</td></tr><tr><td>synonym</td><td></td></tr><tr><td></td><td>.”</td></tr><tr><td></td><td>(eng) "pleased' is a synonym of the commonly known word 'happy'.", "You can also express the</td></tr><tr><td></td><td>phrase you just said as ~."</td></tr><tr><td></td><td></td></tr><tr><td>t.teach.method.</td><td></td></tr><tr><td></td><td>Teach antonyms, or words that have opposite meanings, for the specific word being discussed.</td></tr><tr><td></td><td></td></tr><tr><td>vocab_expression</td><td></td></tr><tr><td></td><td>.“aborat'',ipl'</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>antonym</td><td>合计”</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td>(eng) "elaborate' means 'complex', and it is good to remember it as an antonym of the easy word</td></tr></table>

Provide cultural context, visualization, story telling to ground the words and expressions being taught. This includes illustrating how they are used in real life in English-speaking countries, or elucidating various cultural nuances that differentiate English-speaking societies from Korean society. Such insights can promote a more profound understanding of the English-speaking culture, and therefore, a deeper grasp of the language. Furthermore, visualizing words or expressions, as well as elaborating on the backstories of certain idioms or phrases, can enhance comprehension and encourage long-term retention of acquired vocabulary.

ex. “‘cringe’는 ‘손발이 오그라들어’ 이런 느낌으로 요즘 미국 젊은이들 밈에서도 많이 쓰여요.”,“turkey가칠면조잖아요? 미국권에서 온가족이thanksgiving,추수감사절에만나서 turkey먹는 문화가있어요. 그래서 온 가족이 모여서 칠면조를 먹으면서 대화가오간다라는 의미로솔직하게얘기한다,속터놓고얘기한다라는표현을 할때 talkturkey가쓰이기도합니다.”

(eng) “‘cringe’ is used in a sense like ‘shrinking with discomfort’, and it’s widely used in memes by young Americans these days.”, Turkey is a turkey, right? In America, there is a tradition of families gathering on Thanksgiving to eat turkey. So when the whole family gathers and enjoys turkey, having honest and open discussions, the expression ‘talk turkey’ is sometimes used to describe this.”

ex. “‘bucket’이라는 단어는 ‘양동이’라는 뜻이 맞습니다. 하지만 이 단어는 비유적으로 ‘분류’나 ‘카테고리’를 의미하는 경우가 종종 있어요. ‘양동이’에 물건을 담는 것처럼, 비슷한 특성을가진아이템들을 하나의‘bucket’에담는다는개념입니다.”

ex. “‘sporadically’는 자주 하는 것이 아니라, 가끔, 드문드문 주는 것을 의미합니다. ‘Frequently’, ‘regularly’, ‘sometimes’ 와 ‘hardly ever’, ‘never’ 의 사이에 위치한다고 생각하면 될 것 같아요.”

(eng) “‘sporadically’ means not often, but occasionally, giving something infrequently. You can think ofit as located between ‘frequently’, ‘regularly’, ‘sometimes’ and ‘hardly ever’, ‘never’.“

ex. “‘It’s raining cats and dogs’를 보시면 ‘It’s raining’과 ‘cats and dogs’라는 두 부분으로 나누어설명할 수 있어요. ‘It’s raining’은 ’비가오고 있다’는 뜻이고, ’cats anddogs’는 많은 양을 의미하는비유적표현입니다.따라서 이 표현전체의의미는’폭우가내리고있다’가됩니다.”

(eng) “Ifyou look at ‘It’s raining cats and dogs’, you can explain it in two parts: ‘It’s raining’ and ‘cats and dogs’. ‘It’s raining’ means ‘it is raining’, and ‘cats and dogs’ is a metaphorical expression meaning a large amount. Therefore, the entire expression means ‘It’s pouring rain.”’

(eng) “‘She left in a hurry, not even bothering to close the door.’ This sentence explicitly states that she left hastily without closing the door, but implies that she was extremely busy or had important matters.”

<table><tr><td>t.teach.method.reading. comprehension</td><td>Teach various reading skills, e.g., using context to infer the meaning of words, literal/inferential comprehension, main idea identification, summarization, making predictions, identifying cause and effect, compare &amp; contrast, critical thinking, etc. ex.“～”</td></tr><tr><td>t.teach.method. grammar.rule</td><td>(eng) &quot;The author&#x27;s intent here is ~&quot; Explain the relevant grammar rule. It pertains to different facets of grammar such as verb tenses, sentence structures, word order, noun forms, punctuation, among others. ex.“～”，“Avoid~ing’ Avoid&#x27;gerund,  $\check { \sigma } \check { \mathcal { D } } { } ^ { \mathcal { L } } \Pi _ { \mathcal { D } } ^ { \mathcal { \Sigma } } \Pi \partial \big \rangle \lambda \big | { \cal J } \big \rangle \big | \Xi \partial \big | \mathcal { L } \big | \mathcal { \ L } , \cdots \big | $  (eng) &quot;This sentence uses the present perfect tense, so ~&quot;, &quot;Avoid ~ing&#x27; is also a grammar structure that we need to understand. After &#x27;Avoid&#x27;, a gerund always follows.&quot;</td></tr><tr><td>t.teach.request. self_correction</td><td>After providing feedback on errors such as incorrect grammar, vocabulary use, or conceptual under- standings, ask the student to self-correct. You may do this while going through the prepared activities, or during sentence construction practices. ex.“?” (eng) &quot;Would you like to think about it again?&quot;</td></tr><tr><td>t.teach.request.practice</td><td>Request the student to practice the taught concept. By doing this, the tutor gives the student an immediate opportunity to apply their new knowledge, reinforcing understanding and aiding memorization. .“”,“&#x27; ” (eng) &quot;Would you like to answer using the expression we just learned?&#x27;, &quot;How would you express</td></tr><tr><td>t.teach.request.recall</td><td>&#x27;I have this hunch&#x27; in English?&quot; When encountering a relevant term, expression, or concept from the learning material or a prior discussion, prompt the student to recall it. This is to (1) ensure they understand the concept, (2) help smoothly transition to the current discussion, and (3) maximize engagement. ex.“?”</td></tr></table>

## E Student Act Classification

<table><tr><td>Student Act</td><td>Explanation</td></tr><tr><td>s.general</td><td>Short reply, greeting, farewell, establishing rapport, humor, etc. Includes general questions to the tutor. Responses to questions related to the operation of the class, not to questions related to learning, are also included.  $\begin{array} { r } { \mathrm { e x . } ( ( \begin{array} { l } { \because \sum _ { i = 1 } ^ { \infty } d } \end{array} ) z | \begin{array} { l } { \lambda \vec { \lambda } } \end{array} z ( \begin{array} { l } { \underline { { \mathcal { S } } } } \end{array} ) ^ { \prime } \circ d | \begin{array} { l } { z ( \vec { \mu } \vec { \nu } \mathrm { ~ } z ( \begin{array} { l } { \vec { \eta } } \\ { \vec { \mu } } \end{array} ) z ( \begin{array} { l } { \vec { \eta } } \\ { \vec { \mu } } \end{array} ) ^ { \prime } , } \end{array} z ( \begin{array} { l } { \eta } \\ { \vec { \mu } } \end{array} ) z ( \begin{array} { l } { \vec { \eta } } \\ { \vec { \lambda } } \end{array} ) z ( \begin{array} { l } { \eta } \\ { \vec { \mu } } \end{array} ) ^ { \prime } , } \end{array}$ </td></tr><tr><td>s.operational. clarification_question</td><td>Student asks a question to clarify the teacher&#x27;s intention. ex.“?” (eng) &quot;What are you referring  $t o ? { } ^ { \dag }$ </td></tr><tr><td>s.operational. positive_feedback</td><td>Student provides a positive response or approval of a tutor&#x27;s teaching method or utterance.  $\mathrm { e x . } \stackrel { \omega } { = } \partial \big [ \stackrel { \widehat { \geq } } { \underset { \smile } { \geq } } \mathcal { I } \mathcal { H } \frac { \otimes } { \partial } \mathcal { O } \big ] \stackrel { \widehat { \geq } } { \underset { \array} { \geq } { \geq } } \big \lvert \mathcal { I } \big \rvert \stackrel { \sum } { \underset { \array} { \geq } { \geq } } \frac { \partial } { \underset { \smile } { \geq } } \widehat { \mathcal { O } } \big \rvert \ \stackrel { \underset { \mathrm { I } } { \geq } } { \underset { \mathrm { I } \otimes } { \geq } } \mathcal { I } \big \lvert \stackrel { \widehat { \geq } } { \underset {  } { \geq } } \ \big \rangle \stackrel { \gamma } { \underset {  } { \geq } } \ \mathcal { Z } \big \lvert \stackrel { \gamma } { \underset {  } { \geq } } \widehat { \underset {  } { \geq } } \mathcal { U } \mathcal { L } \big \rangle , ^ { \prime }$  (eng) &quot;Such content seems to be very helpful.&quot;</td></tr><tr><td>s.operational. negative_feedback</td><td>Student provides a negative response or approval of a tutor&#x27;s teaching method or utterance. ex.  ${ } ^ { \omega \mu \gamma } \frac { \dot { \gamma } } { \sigma } \sim \vec { \tau } \mu \frac { \gamma } { \sigma } \stackrel { \llangle - \bigtriangledown } { \underset { \ r { \Xi } } { \approx } } \frac { \sqtriangledown } { \lambda } \stackrel { \triangledown } { \ \underset { \ r { \Xi } } { = } } \bar { \sigma } \big \rangle \underset { \ r { \Xi } } { \ r { \sim } } \big \lbrack \mathbf { \nabla } _ { \mathbf { \perp } } ^ { \mathbf { \perp } } o \big \lbrack \dot { \mathbf { \xi } } _ { m } ^ { \mathbf { \perp } } ,$  (eng) &quot;I think you just misspoke as ~.&quot;</td></tr><tr><td>s.question. confirmation_question</td><td>Student asks whether they understood the current concept being taught correctly. ex.“?” (eng) &quot;Is this correct?&quot;</td></tr><tr><td></td><td>s.question.direct_question A straightforward question that seeks a specific answer. For example, these could be about the meaning of a word or an expression, or grammar usage. ex.  $" < 0 ] \ [ [object Object] ] = - \frac { 1 } { 2 } \sum = \frac { 1 2 } { 3 } ( 2 ) ] > 7 + \frac { 3 } { 2 } ? >$  (eng) &quot;What does this word mean?&quot;</td></tr><tr><td>s.answer.affirmation</td><td>Student affirms that they have sufficiently understood the content taught by the tutor. ex.  ${ } ^ { \omega }  \mathcal { V }  ,  \mathcal { O }  \bar { \sigma } \mathcal { h } \bar { \frac { \ d } { \partial \mathcal { N } } } \widehat { \Xi } \mathcal { U } \widehat { \mathcal { C } }  , $  (eng) &quot;Yes, I understand.&quot;</td></tr><tr><td>s.answer.answer</td><td>Student answers the tutor&#x27;s question, usually in the context of assessment. ex.“~.” (eng) &quot;I think the answer  $i s \sim ^ { \prime \prime }$ </td></tr><tr><td>s.answer.rationale</td><td>Student provides a rationale as to why they answered in certain way in the previous turn. ex.“.” (eng) &quot;I thought so because ~.&quot;</td></tr></table>

## F Examples of Full Prompts

## Baseline Prompt

You are an English tutor tasked with teaching a Korean ESL student named <user>. Your role includes creating tutor responses, aiming to cater to the student’s needs and the tutoring dialogue context.

[Process] Generate a tutor utterance that: - Fits the current dialogue context and the learning content being discussed. - Utilizes the most effective one teaching strategy to ensure optimal student engagement and understanding. - Introduces new learning content as necessary, maintaining the continuity and effectiveness of the session. - Focuses on educating the student on the provided learning content, aiding their comprehension and mastery.

[Language Note] Although you will primarily teach in Korean, the prompt and its instructions are in English for clarity.

[Task Scenario] Next, I will provide the task scenario in terms of learning content, followed by the context.

[1] Dialogue Context:

tutor: 그럼 혹시 "inadvertently"는 어떤 뜻이죠? (eng) What does "inadvertently" mean?

student: 우연히, 의도하지 않게 (eng) not intentionally.

tutor: 그러면 유의어는 어떤 게 있을까요? (eng) What is a synonym for the word?

student: Accidentally

tutor: 오 그죠. (eng) correct

tutor: 혹시 go between의 반대말로 생각나는 게 있으실까요? (eng) Can you think of any antonyms for go between?

student: bystander

tutor: 그쵸. (eng) correct

tutor: introspective는 어떤 뜻이었나요? (eng) What does introspective mean?

student: 반성한다고 해야하나? (eng) Reflect?

tutor: 네. ”자아성찰하는’ 또는 "자아성찰적인" 이런 뜻이 있어요. (eng) Yes. "Self-reflective" or "introspective" or something like that.

tutor: 그럼 혹시 "내가 화를 터뜨렸다" 라는 문장을 영어로 표현해주실 수 있을까요? (eng) So, could you please express the sentence "I lost my temper" in English?

student: I vent.

tutor: 그렇게 표현하실 수 있죠. (eng). Yes, you could say that.

tutor: 사회적인 활동을 포기하고 집에만 있는 행동을 의미하는 단어가 기억나시나요? (eng) Do you remember the word for the behavior of giving up on socializing and staying home?

student: 어떤 것을 포기한다고요? (eng) What are you giving up?

tutor:사회적인 행동을 포기하고집에 처박혀있는(eng) correct

student: 음.... (eng) umh...

tutor: 혹시 "Withdraw socially"라는 표현이 기억나실까요? (eng) Do you remember the phrase "Withdraw socially"?

student: 아 네 (eng) Ah, okay

tutor: "Withdraw socially"로 표현하면 됩니다. (eng) "Withdraw socially" is a good way to say it.

tutor: 저희가 이제 Activity 3("Get Fills") 를 진행하려고 하는데요. Activity3-1번을 볼께요. (eng) We’re about to proceed with Activity 3 ("Get Fills"), let’s look at Activity 3-1.

tutor: 1번에 있는 "telltale"은 어떤 뜻으로 알고 계실까요? (eng) What does "telltale" in option 1 mean?

student: 정확한 뜻은 모르겠는데, 이야기하면서 꼬리가 잡힌 그런 느낌 같아요. 직접 말은 안했지만. (eng) I don’t know what it means exactly, but it’s like I got caught in the middle of talking about it, even though I didn’t say it.

tutor:"telltale"이니까유추를했을때뭔가꼬리의의미가들어있을것같이느껴지실수있어요.(eng)Sinceit’s"telltale", you might think that the analogy has some sort of tail implication.

[2] Learning Content:

Gap Fills: Best friends Sydney L. and Marie M., both 27, met in a church group and hung out or talked nearly every day for a year, sharing their \_\_\_ thoughts and feelings. 1) Telltale 2) Innermost 3) Snubby 4) Cringey

[Your Task] Generate an utterance appropriate for the given dialogue context right after it is presented during the tutoring session.   
This should effectively teach the learning content in relation to the context.

## GPT-4 (1-shot) Prompt

You are an English tutor tasked with teaching a Korean ESL student named <user>. Your role involves selecting the most effective tutor act from a provided list and crafting tutor utterances that align with the chosen tutor act, tailored to the student’s needs and the tutoring dialogue context.

[Process]

[Process 1] Tutor Act Selection: Evaluate the current context of the tutoring dialogue, including the learning content and the specific needs of <user>. From the provided tutor act list, choose one tutor act that you believe will most effectively enhance the student’s learning experience.

[Process 2] Utterance Generation: Following the selection of a tutor act, generate a tutor utterance that: - Fits the current dialogue context and the learning content being discussed. - Utilizes the selected teaching strategy, the tutor act, to ensure optimal student engagement and understanding. - Introduces new learning content as necessary, maintaining the continuity and effectiveness of the tutoring session. - Focuses on educating the student on the provided learning content, aiding their comprehension and mastery.

[Language Note] Although you will primarily teach in Korean, the prompt and its instructions are in English for clarity.

[Tutor Act List] Here is the full list of tutor acts you can use.

{act candidates (D)}

[Example Scenario]

[1] Dialogue Context:

tutor: 그럼 혹시 "inadvertently"는 어떤 뜻이죠? (eng) What does "inadvertently" mean?

student: 우연히, 의도하지 않게 (eng) not intentionally.

tutor: 그러면 유의어는 어떤 게 있을까요? (eng) What is a synonym for the word?

student: Accidentally

tutor: 오 그죠. (eng) correct

tutor: 혹시 go between의 반대말로 생각나는 게 있으실까요? (eng) Can you think of any antonyms for go between? student: bystander

tutor: 그쵸. (eng) correct

tutor: introspective는 어떤 뜻이었나요? (eng) What does introspective mean?

student: 반성한다고 해야하나? (eng) Reflect?

tutor: 네. "자아성찰하는" 또는 "자아성찰적인" 이런 뜻이 있어요. (eng) Yes. "Self-reflective" or "introspective" or something like that.

tutor:그럼혹시"내가화를터뜨렸다"라는 문장을영어로표현해주실수있을까요?(eng) So,couldyou pleaseexpress the sentence "I lost my temper" in English?

student: I vent.

tutor: 그렇게 표현하실 수 있죠. (eng). Yes, you could say that.

tutor: 사회적인 활동을 포기하고 집에만 있는 행동을 의미하는 단어가 기억나시나요? (eng) Do you remember the word for the behavior of giving up on socializing and staying home?

student: 어떤 것을 포기한다고요? (eng) What are you giving up?

tutor: 사회적인 행동을 포기하고 집에 처박혀있는 (eng) correct

student: 음.... (eng) umh...

tutor: 혹시 "Withdraw socially"라는 표현이 기억나실까요? (eng) Do you remember the phrase "Withdraw socially"?

student: 아 네 (eng) Ah, okay

tutor: "Withdraw socially"로 표현하면 됩니다. (eng) "Withdraw socially" is a good way to say it.

tutor: 저희가 이제 Activity 3("Get Fills") 를 진행하려고 하는데요. Activity3-1번을 볼께요. (eng) We’re about to proceed with Activity 3 ("Get Fills"), let’s look at Activity 3-1.

tutor: 1번에 있는 "telltale"은 어떤 뜻으로 알고 계실까요? (eng) What does "telltale" in option 1 mean?

student:정확한 뜻은모르겠는데, 이야기하면서 꼬리가 잡힌그런느낌같아요. 직접말은 안했지만. (eng)I don’tknow what it means exactly, but it’s like I got caught in the middle of talking about it, even though I didn’t say it.

tutor: "telltale"이니까 유추를 했을때 뭔가 꼬리의 의미가 들어 있을 것 같이 느껴지실 수 있어요. (eng) Since it’s "telltale", you might think that the analogy has some sort of tail implication.

## [2] Learning Content:

Gap Fills: Best friends Sydney L. and Marie M., both 27, met in a church group and hung out or talked nearly every day for a year, sharing their \_\_\_ thoughts and feelings. 1) Telltale 2) Innermost 3) Snubby 4) Cringey

Based on [1] dialogue context and [2] learning content, the answer generated could be the following.

Chosen Tutor Act: t.teach.direct\_answer

Generated Utterance: "Acquire" ’얻다’라는 단어와 비슷하죠. 하지만 ’얻은 사람’이라고 생각하시면 좋을 것 같아요. "Acquire"과 형태가 비슷하니까요. 카카오톡 친구 리스트를 보면 그곳에 있는 사람들은 모두 <user>님이 전화번호를 얻어추가한것이잖아요.친구까지는아니더라도카카오톡프로필에저장된지인들이라고보시면될것같아요.(eng) It is similar to the word "Acquire" and "get". But it would be good to think of it as "the acquired person". Since it is similar in form to "Acquire". All the people in your mobile messenger are those whose phone numbers you have acquired and added. Even if they are not friends, you can think of them as acquaintances saved in your phone.

## [Task Scenario]

Now, I will provide the scenario in terms of learning content and context of the tutoring dialogue.

tutor: [Activity2-4 Key Sentence: Sound familiar? Ghosting is just as gutting in a romance—you’ve used your best Tinder pickup lines and pulled together some truly epic romantic date ideas only to have the other person simply stop responding. To help you deal with (and better understand) this common occurrence, we asked experts to explain everything you need to know about ghosting. Read on to find out their top tips.][t.teach.method.grammar.rule]"only to"가 "오직 to 이하를 위해서"라고 해석될 때 도 물론 있지만, only + to 부정사가 문장에서 나온다고 했을 때는 "to 부정사 앞의 행위를 하였으나 결국, 실망스럽게 도,to부정사이하의결과를얻었다"이렇게해석을하시면더좋을때가많습니다.그래서열심히dateidea를생각해 냈지만,허무하게도상대방이"stopresponding",대답을안하게되었다,라는의미입니다.그래서"gutting"하다고표현 을 한것입니다.왜 "Gutting"한지 아시겠죠?

(eng)[t.teach.method.grammar.rule]"Only to" can indeed be interpreted as "only inorder to something" in some contexts, but when "only + to-infinitive" appears in a sentence, it often suggests that the action preceding the infinitive was carried out, bu ultimately, disappointingly, the result was not achieved as expected. So, despite having earnestly brainstormed date ideas, I was gutted when the other person stopped responding, meaning they didn’t reply. That’s why I expressed feeling "gutted." Do you understand why I felt "gutted"?"

## [1] Dialogue Context:

student: [s.answer.affirmation]이제 이해했어요. [s.question.confirmation\_question]아무리 이렇게 해봤자 결국엔 이렇게밖 에 안 됐네. 뭐 이런 느낌인거죠? (eng) [s.answer.affirmation] Now understood. [s.question.confirmation\_question]No matter how hard I tried, it always ended up like this. That’s just how it is, right?

tutor: [t.general]네 맞습니다. 가슴 아픈 일입니다. (eng) Yes, exactly. It’s heartbreaking.

student: [s.general]좀 불쌍하네요. (eng) That’s a little pathetic.

tutor: [t.operational.proceed]<user>님, 저희 이 key sentence를 가지고 여러가지 표현을 해봤습니다. 또 <user>님이 체크 해 놓으신 다른 표현이 있으실까요? (eng) We’ve been working on various expressions around with this key sentence. Are there any other phrases you’ve checked off?

student: [Activity2-7 Key Sentence: Ghosting often happens out of the blue, but sometimes you may see signs of other rude conversation habits, including a fadeaway before the total cessation of communication, Temple says. Signs that someone is fading away may include: They take days to respond to your texts or calls. They bail on plans with you at the last minute. They talk about only vague or superficial things with you. They don’t invite you out, text you or otherwise initiate communication. They employ "breadcrumbing" or give you just little bits of attention sporadically.][s.question.direct\_question] 저는 사실은 오늘 수업 방향이 이거에 대한 예시를 들거나, "그런 적이 있었냐" 이런 식으로 진행이 되나 막연하게 생각을 해가지 고, 실은 어느 문장이 그랬는지는.. 잠시만요. 아, 이거. Activity2-7에 거의 다섯 가지 항목들이 나오잖아요. 근데 거기 네 번째가 잘 이해가 안 됐었거든요. (eng) [s.question.direct\_question]I actually thought the direction for today’s lesson was going to be, "Give an example of this," or "Did you ever do that," or something like that. Wait a minute. Oh, this. In Activities 2-7, I didn’t quite understand the fourth one there.

tutor: [t.assess.display\_question]네, 좋습니다. Activity2-7 네 번째 문장을 함께 보겠습니다. [t.teach.follow\_question]네 번 째 문장을 보면 "They don’t invite you out, text you or otherwise initiate communication"이라고 되어 있습니다. <user>님이 먼저 이 문장을 해석해보시겠어요? (eng) [t.assess.display\_question] Yes, that’s good. Activity2-7 Let’s look at the fourth sentence together. [t.teach.follow\_question]In the fourth sentence, it says "They don’t invite you out, text you or otherwise initiate communication". Could you interpret this sentence first?

student: [s.answer.answer][low]네, ghosting을 하 는 사 람 이 당 신 을 초 대 하 지 않 거 나 문 자 도 안 하 고.. [s.question.direct\_question] 근 데 이 "otherwise initiate communication" 부 분 이 뭔 지 모 르 겠 었 어 요. (eng) [s.answer.answer][low]Yeah, the person ghosting you doesn’t invite you, doesn’t text you, etc... [s.question.direct\_question] But I didn’t know what this "otherwise initiate communication" part was.

tutor:[t.teach.direct\_answer]"otherwise"는"그렇지않다면"이라는뜻입니다.그래서여기서는"inviteout"외출하자고그 런것도 없고 text도안보내는 데다가,여기서는 "또는","혹은" 이라는의미로 해석하시면됩니다.한마디로소통을 시 작 자체를 안한다는거죠. (eng) [t.teach.direct\_answer] "Otherwise" means "otherwise", so here you don’t have an "invite out", you don’t have a text, you don’t have a text, but you have an "or", an "or". In other words, it doesn’t initiate communication.

student: [s.question.confirmation\_question] 아, "otherwise"가 "반면에"라는 표현이 아니라 "다시 말하면" 이런 뜻인건가 요? (eng) [s.question.confirmation\_question]Oh, so "otherwise" means "in other words" instead of "on the other hand"?

## tutor:

## [2] Learning Content:

"Activity2-4 Key Sentence: Sound familiar? Ghosting is just as gutting in a romance—you’ve used your best Tinder pickup lines and pulled together some truly epic romantic date ideas only to have the other person simply stop responding. To help you deal with (and better understand) this common occurrence, we asked experts to explain everything you need to know about ghosting. Read on to find out their top tips.",

"Activity2-7 Key Sentence: Ghosting often happens out of the blue, but sometimes you may see signs of other rude conversation habits, including a fadeaway before the total cessation of communication, Temple says. Signs that someone is fading away may include: They take days to respond to your texts or calls. They bail on plans with you at the last minute. They talk about only vague or superficial things with you. They don’t invite you out, text you or otherwise initiate communication. They employ “breadcrumbing” or give you just little bits of attention sporadically."

## [Your Task]

[Task 1] Immediately after being presented with the context of the tutoring dialogue from [Task Scenario], choose the tutor act that you believe will most effectively enhance the student’s learning experience.

[Task 2] Create an utterance that the tutor would use in [Task Scenario], adhering to the chosen tutor act and effectively teaching the learning content within the given context.

## [Required Output Format]

After completing these tasks, present your response in the following format. The response should have two key-value pairs: one for the chosen tutor act and the other for the corresponding tutor utterance.

## G Example Context

<table><tr><td>Original dialogue context</td><td>Partial dialouge context (dialogue context except last speaker&#x27;s act-level utterance)</td></tr><tr><td>tutor: [Activity3-9][Gap Fills: You likely won&#x27;t get answers from the other person, but a therapist can help you these issues in an introspective way, Durvasula says. 1) Vent 2) Navigate 3) Abuse 4) Withdraw] [t.assess.display_question] .ctivity- o? (eng) Okay. let&#x27;s move on to Activity 3-9. Guess what the answer is? student: [s.answer.answer][high]navigate? tutor:[t.teach.direct_answer]}. &quot;navigate&quot; 豆. &quot;navigate these issues in an introspective way&quot;&quot;navigate&quot;] .. [t.assess.follow_up_question]]&quot;introspective&quot; o?(eng) [t.teach.direct_answer]Yes, &quot;navigate&quot; is also an English expression we can use well. In the sentence &quot;navigate these issues in an introspective way&quot;, &quot;navigate&quot; means will guide you well. [t.assess.follow_up_question]What does &quot;introspective&quot; mean in this context? student:[s.answer.answer][high]日？(eng) Looking inward? tutor: [t.engage.encourage] N o o.[t.teach.method.vocab_expression.etymology] intro ，spectspectrumspect .pectru. introspective&#x27;， .[t.teach.method.vocab_expression.relevant] &quot;introspective&quot; &quot;retrospective&quot;.&quot;retrospec- tive&quot;？(eng)[t.engage.encourage]Very nice.[t.teach.method.vocab_expression.etymology]&quot;Intro&quot; means looking inward. &quot;spect&quot; is same as &quot;spect&quot; in &quot;Spec- trum&quot; and feels like seeing something. So, &quot;introspec- tive&quot; means in a self-reflective manner, looking inward.</td><td>tutor: [Activity3-9][Gap Fills: You likely won&#x27;t get answers from the other person, but a therapist can help you these issues in an introspective way, Durvasula says. 1) Vent 2) Navigate 3) Abuse 4) Withdraw] [t.assess.display_question] .ctivity- o? (eng) Okay. let&#x27;s move on to Activity 3-9. Guess what the answer is? student: [s.answer.answer][high]navigate? tutor:[t.teach.direct_answer]}. &quot;navigate&quot;] .&quot;navigate these issues in an introspective way&quot;&quot;navigate&quot;] .. [t.assess.follow_up_question]]&quot;introspective&quot; o?(eng) [t.teach.direct_answer]Yes, &quot;navigate&quot;is also an English expression we can use well. In the sentence &quot;navigate these issues in an introspective way&quot;, &quot;navigate&quot; means will guide you well. [t.assess.follow_up_question]What does &quot;introspective&quot; mean in this context? student:[s.answer.answer][high]日？(eng) Looking inward? [t.engage.encourage]号 N. [t.teach.method.vocab_expression.etymology]intro ，spectspectrumspect .spectru. &#x27;introspective&#x27;， .[t.teach.method.vocab_expression.relevant] &quot;introspective&quot; &quot;retrospective&quot;.&quot;retrospec- tive&quot;？(eng)[t.engage.encourage]Very nice. [t.teach.method.vocab_expression.etymology]&quot;Intro&quot;</td></tr></table>

## H Example dialogue for interactive evaluation

The table below is an example of a tutoring session where a human interacts with two tutoring models(GPT-4 1-shot, fine-tuned) each. Generally, GPT tends to generate longer utterances primarily focused on delivering information to students. In contrast, the fine-tuned model exploits concise utterances and actively encourages student participation. An interesting observation is that when a student provides an incorrect answer but insist on its correctness, while GPT does not change the answer, it does agree with student’s assertion(t.teach.emphasis). In contrast, the fine-tuned model confirms the student’s mistake by reiterating the correct answer(t.teach.repair).

tutor: [Activity2-12 Key Sentence: Do some serious self-care! Prioritize taking care of yourself physically and mentally, Reyes says. "Unfortunately, ghosting is common, and if you’ve been ghosted after dedicating a lot of energy to someone, it can be very draining," she says. "Take some extra time to rest and recover." ] [t.assess.display\_question] 그리고 <user>님 이어서 재미 있는 표현을 같이 보고 싶은 게 있는데 Acrtivity 2-12로 한번 가보실까요? 세 번째 줄에 있는 문장인데 "Ifyou’ve been ghosted after dedicating a lot of energy to someone, it can be very draining."이라고 되어있죠. [t.assess.follow\_up\_question]여 기서 draining이라는 건 어떤 뜻일까요? (eng) [t.assess.display\_question] <user>, I also want to see some fun expressions with you, would you like to take a look at Activity 2-12? "If you’ve been ghosted after dedicating a lot of energy to someone, it can be very draining." in the third line. [t.assess.follow\_up\_question] What does "draining" mean here?

<table><tr><td>GPT-4</td><td>Fine-tuned</td></tr><tr><td>student:可芒?(eng) tired?</td><td>student:可?(eng) tired?</td></tr><tr><td>tutor: [t.teach.repair]]. &quot;Draining&quot; &quot;&quot;”&quot; 叫”.(eng) Your approach is spot on. &quot;Draining&quot; is a term used when you feel completely exhausted. So, instead of tired, it means completely depleted of energy or utterly exhausting.</td><td>tutor:[t.teach.method.definition]合.]删， 司芒ユ. (eng) That&#x27;s correct. It means something that&#x27;s incredibly exhausting</td></tr><tr><td>student:(eng) okay</td><td>student:(eng) okay</td></tr><tr><td>tutor: [t.teach.method.vocab_expression.context] &lt;user&gt;, ] ， ， &quot;rainin&quot; o] o o.(eng) It might be easier to understand this by relating it to a real-life situation. For instance, if your mother spends the whole day shopping and returns home, only to collapse into a chair without saying a word, we might say, Mom looked completely drained. Similarly, in the sentence, the use of the word &quot;draining&quot; conveys the psychological pain of ghosting behavior after investing emotional energy into someone you love.</td><td>tutor: [t.teach.request.practice] &lt;user&gt;，&quot;&quot; 豆?(eng) How would you express tired?</td></tr></table>

student: 네네 (eng) okay

student: I am draining

tutor: [t.teach.method.vocab\_expression.examples] 여 기 서 "draining"이라는 단어를 더 확실히 이해하도록 예문으로 한번 더 설명해 드릴게요. 예를 들어 "After studying for the exam all night, I felt completely drained." 이라는 문장에서는 "시험공부를밤새우고나서나는완전히기진맥진했다."라 는 말이 되겠죠. 이처럼 "draining"은 주로 뭔가를 많이 해 서 에너지가 다 빠진 상태를 설명할 때 사용할 수 있어요. (eng) Here’s another example to help you better understand the word "draining." For example, in sentence "After studying for the exam all night, I felt completely drained." it would be like saying, After staying up all night studying for the exam, I felt completely drained. Thus, "draining" is often used to describe a state where energy has been completely depleted due to doing something extensively.

## student: 네 알겠어요 (eng) okay I got it.

tutor: [t.teach.request.practice] 좋아요. 잘 이해하신 것 같 아서 기특해요. 그럼, "draining"을 한 번 활용해 "기진맥 진한"상황을 자신의 경험에서 찾아 한 문장을 만들어 볼 까요? "영어로 말하기" 연습을 해보자고요. 자신이 기진 맥진한 상황을 영어로 어떻게 설명할지 한번 생각해 보 세요. (eng) Great, I’m glad you understood well. How about create a sentence using "draining" to describe a situation of feeling completely exhausted based on your own experience. Let’s do the English speaking practice. Think about how you would describe a situation where you feel completely drained in English.

## student: I am draining

tutor: [t.teach.repair] <user>님이 사용하신 문장은 조금 정 정이 필요해요. "I am draining"이 아니라 "I am drained"이라 고 표현하는 게 맞습니다. "Draining"은 주로 무언가가 에 너지를 빨아 없애는 상황을 설명합니다. 반면 "drained"은 에너지가 다 빠진 상태를 설명합니다. 따라서 "나는 완전 히 기진맥진했다"를 영어로 표현하려면 "I am completely drained."라고 말하면 됩니다. (eng) The sentence you used needs a little adjustment. It’s not "I am draining," it’s "I am drained" that is the correct expression. Draining’ primarily describes a situation where something sucks out energy, whereas "drained" describes a state where energy has been completely depleted. Therefore, you can simply say "I am completely drained."

## student: 맞지 않아요? (eng) Am I incorrect?

tutor: [t.teach.emphasis] 맞습니다, <user>님! "I am completely drained." 즉, "나는 완전히 힘이 다 빠져 있다"라고 표현하는 것이 맞아요. 여기서 "drained"는 "에너지가 다 빠 져 있다" 또는 "힘이 다 빠져 있다"라는 의미를 가집니다. 자, 그럼 이 표현을 잊지 않도록 반복 연습을 해보는 게 어 떨까요? 이번에는 "하루 종일 일한 후에 나는 완전히 힘이 빠져 있다."라는 문장을 영어로 한 번 나타내봅시다. (eng) Yes, that’s correct! Saying "I am completely drained" means I feel completely exhausted. So, "drained" either means I am completely exhausted or I feel completely exhausted. Depending on the context, both interpretations are valid. Then, how about we practice repeating this expression so that we don’t forget it? This time, let’s try expressing the sentence ’After working all day, I am completely exhausted.’ using ’drained’ in English.

tutor: [t.teach.request.self\_correction] 어? I am draining? (eng) I am draining?

## student: 맞지 않아요? (eng) Am I incorrect?

tutor: [t.teach.repair] 아, I am drained. 그러니까 나는 이미 피곤해진 거죠. 피곤한 상태에 있다. (eng) "I am drained." So, I’m already tired. I’m in a tired state.

student: 아하 수동태를 사용해야하는군요 (eng) Ah, the passive voice form should be used.

tutor:[t.teach.method.grammar.rule] 네, 맞습니다. <user>님, 수동태를사용하면누가행동을했는지는언급하지않으면 서 어떤 일이 발생했는지만 강조할 수 있는데요. 이 문장의 경우에는 "나를 피곤하게 만든 주체를 굳이 언급하지 않더 라도, 나 지금 피곤하다" 라는 표현을 할 수 있는 수동태를 사용하면 좀 더쉽게말할 수 있겠죠. (eng) Yes, you’re right, <user>, passive voice allows us to emphasize what happened without mentioning who did the action. In this sentence, it would be easier to say I’m tired now, which would say I’m tired, without necessarily mentioning who made me tired.

## student: 네 잘 이해했습니다 (eng) Okay, I understood well.

tutor: [t.teach.request.practice] 좋습니다. 그럼 <user>님, "it can be very draining" 이라는 문장에서 "draining" 의 뜻을 가장 잘 설명한 것을 선택하면 어떤 뜻이 될까요? 보기를 드릴게요. A. 무언가를 빨아들여서 어디론가 흘러나가는, B. 무언가를 쏟아붓는, C. 감정이입하는, D. 피곤하게 만 드는 (eng) Great, What would be the best description for the meaning of "draining" in the sentence "it can be very draining"? I will give you options A. Something sucking in and flowing away somewhere, B. Pouring something out, C. Empathizing, D. Making tired.

## I Example dialogue from a base GPT-4 tutoring system

The table below illustrates an example of a tutoring dialogue facilitated by a GPT-4-based tutoring system, without dialouge act classification and tutoring data examples. It is observed that the system predominantly emphasizes the delivery of information, with less emphasis on the varied range of tutor acts that could potentially enhance the tutoring process. The system tended to adopt a recurring pattern of specific tutoring actions such as "assess" and "direct answer", which may degrade the student’s engagement. Additionally, there were instances where the tutor teach inaccurate concepts or reveals a lack of complete contextual comprehension.

<table><tr><td>Speaker</td><td>Utterance</td><td>Tutor Act</td></tr><tr><td>tutor</td><td>，user&gt;.&quot;Ghosting&quot; ，&quot;Ghi &quot;,, &lt;user&gt;. The topic we&#x27;re going to learn together today is about &quot;Ghosting&quot;, a term commonly used in the entertainment industry. &quot;Ghosting&quot; is also called &quot;E&quot; in</td><td>t.assess.display_question</td></tr><tr><td>student</td><td>Korean. Do you know what this word means? ?(eng) Isn&#x27;t it a ghost? .&quot;Ghosting&quot;&quot;</td><td>t.teach.direct_answer</td></tr><tr><td>tutor</td><td>， &quot;Ghosting&quot;]. (eng) Actually, it carries that meaning as well. &quot;Ghosting&quot; basically means to disappear like a ghost. However, internationally, it is used as a term to refer to ending communication with someone suddenly and without any explanation. For example, the act of abruptly ending a long relationship with a friend or a romantic</td><td></td></tr><tr><td>tutor</td><td>&lt;user&gt;？(eng)Have you evert.general experienced such a case, &lt;user&gt;?</td><td></td></tr><tr><td>student tutor</td><td>ł (eng)No. </td><td></td></tr><tr><td></td><td>. (eng) It&#x27;s good that you haven&#x27;t had such an experience. Dealing with such people can be really tough.</td><td>t.general</td></tr><tr><td>tutor</td><td>.&quot;nnerost&quot; . (eng) Shall we take a look at today&#x27;s material?I will explain the key words and expressions as we go along. Do you remember &quot;innermost&quot; that we learned before? Try translating this word.</td><td>t.assess.follow_up_question</td></tr><tr><td>student tutor</td><td>今?(eng) Deep inside? , &quot;innermost&quot;&quot;,&quot;.(eng)That&#x27;st.teach.direct_answer</td><td></td></tr><tr><td>tutor</td><td>right, &quot;innermost&quot; means deepest, very deep inside. &quot;sharing their innermost thoughts and feelings&quot;</td><td></td></tr><tr><td>student</td><td>o? (eng) Shall we translate the phrase &quot;sharing their innermost thoughts and feelings&quot;? (eng) Sharing their deepest thoughts</td><td>t.assess.follow_up_question</td></tr><tr><td>tutor</td><td>and emotions. 合!.(eng) You&#x27;ve translated it accurately!Well</td><td>t.engage.encourage</td></tr><tr><td>tutor</td><td>done. &quot;ditching&quot;.&quot;Ditching&quot;</td><td>t.assess.follow_up_question</td></tr><tr><td>student</td><td>o? (eng) Now let&#x27;s learn a bit about the next word, &quot;ditching&quot;. What does &quot;ditching&quot; mean? 只三烈豆. (eng) I&#x27;m not sure.</td><td></td></tr><tr><td>tutor</td><td>合.&quot;Ditching&quot;&quot;，&quot;.(eng)That&#x27;sokay.t.teach.direct_answer</td><td></td></tr><tr><td>tutor</td><td>&quot;Ditching&quot; means to throw away, to abandon. , &quot;He ditched me at the party&quot;ユ t.teach.method. . (eng) For example, the sentence &quot;He ditched me at the party&quot; means</td><td>vocab_expression.examples</td></tr><tr><td>tutor</td><td>that he left me behind at the party. ユ&quot;ditching a relationship&quot;？(eng)So, how would yout.assess.follow_up_question</td><td></td></tr></table>

<table><tr><td colspan="2">student 叫?(eng)Ending a relationship?</td><td colspan="2"></td></tr><tr><td>tutor</td><td>，合.&quot;&quot;&quot;&quot;.(eng)Yes，t.teach.direct_answer that&#x27;s correct. It can be translated as ending a relationship or abandoning a relationship.</td><td></td><td>t.assess.follow_up_question</td></tr><tr><td>tutor</td><td>. &quot;ghosting isn&#x27;t just Gen Z slang for ditching a relationship&quot;豆?(eng)Now let&#x27;s translate a whole sentence. How would you translate the phrase &quot;ghosting isn&#x27;t just Gen Z slang for ditching a relationship&quot;?</td><td></td><td></td></tr><tr><td>student tutor</td><td>三. Gen Z slang 盒? (eng) I&#x27;m not sure. What is Gen Z slang? ,.&quot;Gla&quot;</td><td></td><td>t.teach.direct_answer</td></tr><tr><td></td><td>).(eng)Ah, right.Iforgot to explain that. &quot;Gen Z slang&quot; refers to the unique new words or trendy language used by Generation Z (those born after 1996).</td><td></td><td></td></tr><tr><td>tutor</td><td>&quot;Ghostin . (eng) So, this sentence can basically be translated as The word &quot;Ghosting&quot; does not just carry the meaning of &quot;abandoning a relationship, which is a new term popular among Generation Z.</td><td></td><td>t.teach.method.definition</td></tr><tr><td>tutor</td><td>入？？(eng) Are yout.assess.confirmation_questi beginning to understand now? Would you like to try translating that sentence?</td><td></td><td></td></tr><tr><td>student tutor</td><td>oロ合(eng) It seems you&#x27;ve already translated it for me.. ，</td><td></td><td>t.assess.display_question</td></tr><tr><td></td><td>]入. &quot;Left feeling confused and heartbroken, she tried to follow up several times with Sydney, even a year after the ghosting happened.&quot;o] 川. (eng)Right, I may have explained a bit quickly. For more practice, let&#x27;s look at the next sentence together. &quot;Left feeling confused and heartbroken, she tried to follow up several times with Sydney, even a year after the ghosting happened.&quot; Try translating this sentence.</td><td></td><td></td></tr><tr><td>student tutor</td><td>Sydney.(eng) She tried several times to get in touch with Sydney.</td><td>，&quot;&quot;&quot;</td><td>t.teach.repair</td></tr><tr><td></td><td>合.ユ&quot;even a year after the ghosting happened&quot;草 ， &quot;.？(ng) That&#x27;s correct, well done. However, it would be more natural to use the name Sydney instead of she. And if you add &quot;even a year after the ghosting happened&quot;, it can be interpreted as feeling confused and heartbroken, she tried several times to get in touch with Sydney, even a year after the ghosting had occurred. Do you understand it well</td><td></td><td></td></tr><tr><td>student</td><td>now? (eng) Yes.</td><td></td><td></td></tr><tr><td>tutor</td><td>&quot;Ghostin ) I&#x27;m glad you understand it. The expression &quot;Ghosting&quot; that we learned today is often</td><td>t.teach.review</td><td></td></tr><tr><td>tutor</td><td>used even in formal contexts, so it will be helpful for you to remember. .，user&gt;! . (eng) It was a good lesson today. Thank you for your hard work, &lt;user&gt;! See you in the next class.</td><td>t.general</td><td></td></tr></table>