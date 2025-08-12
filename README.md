# G2SC

The Pytorch implementation for the paper "GAI-Driven Knowledge Distillation for Personalized Semantic Communications via Federated Learning".

## Introduction

Semantic communication (SC) combines advanced artificial intelligence (AI) with traditional wireless communication to accommodate various task-oriented scenarios. However, with the growing requirements for personalization, users expect customized and specialized intelligent communication. Federated learning (FL) attracts attention as a technology that simultaneously ensures personalized services and privacy protection, yet it remains incompatible with the SC architecture. Specifically, SC requires channel simulation, while each user experiences a distinct communication environment. This leads to SC models trained under different channel assumptions being heterogeneous, which hinders global aggregation. A key insight is that channel coding can be shared across all SC models, whereas user knowledge is specific and unique. Therefore, we decouple personalized demand from general functionalities within SC and propose a two-stage SC framework, named distillation-to-SC (D2SC). In the first stage, we employ FL to extract knowledge from private user data by knowledge distillation (KD). Specifically, we employ generalization AI as the teacher to perform probabilistic modeling in the semantic space to enhance generalization and transfer the knowledge to the student through KD to mitigate data heterogeneity and communication overhead in FL. In the second stage, since the student model contains specialized knowledge for each user, we can reuse it by incorporating channel coding and semantic decoding functions. We design a dual optimization mechanism for both the channel and the information bottleneck to specifically address the adverse effects of channel fading on semantic transmission. Extensive experiments demonstrate that D2SC can accomplish a variety of tasks under different channel conditions.


## Code

We will make the code available after the paper is accepted.
