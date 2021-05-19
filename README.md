# Cross-lingual-analysis

This repository contains the translation codes used in the paper [A cost-benefit analysis of cross-lingual transfer methods](https://arxiv.org/abs/2105.06813). In this work, we analyze cross-lingual methods on three tasks in terms of their effectiveness (e.g., accuracy), development and deployment costs, as well as their latencies at inference time. We experiment with the following transfer learning techniques: 1) fine-tuning a bilingual model on a source language and evaluating it on the target language without translation, i.e., in a zero-shot manner; 2) automatic translation of the training dataset to the target language; 3) automatic translation of the test set to the source language at inference time and evaluation of a model fine-tuned in English. Finally, by combining zero-shot and translation methods, we achieve the state-of-the-art in two of the three datasets used in this work. The study is a result of an ongoing Master's Program.

# Evaluation benchmarks
The models were benchmarked on three tasks (Question Answering, Natural Language Inference and Passage Text Ranking) and compared to previous published results. Metrics are: Exact Match and F1-score for Q&A, Accuracy for NLI and MRR@10 for Text Ranking.

| Method        | Score         | One-time costs | Recurrent costs | Added Latency | 
| ------------- | ------------- | -------------- | --------------- | ------------- | 
| Content Cell  | Content Cell  |                |                 |               |
| Content Cell  | Content Cell  |                |                 |               |
| ------------- | ------------- | -------------- | --------------- | ------------- |
|  NLI          |               |                |                 |               |    
|               |               |                |                 |               |  
