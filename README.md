## Cross-lingual analysis

This repository contains the translation codes used in the paper [A cost-benefit analysis of cross-lingual transfer methods](https://arxiv.org/abs/2105.06813). In this work, we analyze cross-lingual methods on three tasks in terms of their effectiveness (e.g., accuracy), development and deployment costs, as well as their latencies at inference time. We experiment with the following transfer learning techniques: 1) fine-tuning a bilingual model on a source language and evaluating it on the target language without translation, i.e., in a zero-shot manner; 2) automatic translation of the training dataset to the target language; 3) automatic translation of the test set to the source language at inference time and evaluation of a model fine-tuned in English. Finally, by combining zero-shot and translation methods, we achieve the state-of-the-art in two of the three datasets used in this work. The study is a result of an ongoing Master's Program.

## Evaluation benchmarks
The models were benchmarked on three tasks (Question Answering, Natural Language Inference and Passage Text Ranking) and compared to previous published results. Metrics are: Exact Match and F1-score for Q&A, Accuracy and F1-score for NLI and MRR@10 for Text Ranking.

| Model                           | Pre-train     | Fine-tune       | F1           | Accuracy    | 
| ------------------------------- | ------------- | --------------- | ------------ | ----------- | 
| mBERT (Souza et al.)            | 100 languages |   ASSIN2        |   0.8680     |   0.8680    |
| PTT5 (Carmo et al.)             | EN & PT       |   ASSIN2        |   0.8850     |   0.8860    |
| BERTimbau Large (Souza et al.)  | EN & PT       |   ASSIN2        |   0.9000     |   0.9000    |
| BERT-pt (ours)                  | EN & PT       |  MNLI + ASSIN2  |   0.9207     |   0.9207    |    
                 
## Data 

We made available the following data and the respective translation code:
- SQuAD (Q&A)
- FaQuAD (Q&A)
- MNLI (NLI)
- ASSIN2 (NLI)

The datasets SQuAD and MNLI are directly downloaded from the notebooks of this repository. We also provide the FaQuAD and ASSIN2 datasets.

## References

[1] [BERTimbau: Pretrained BERT Models for Brazilian Portuguese](https://www.researchgate.net/publication/345395208_BERTimbau_Pretrained_BERT_Models_for_Brazilian_Portuguese)

[2] [PTT5: Pretraining and validating the T5 model on Brazilian Portuguese data](https://arxiv.org/abs/2008.09144)

## How do I cite this work?

~~~ {.xml
 @article{cross-lingual2021,
    title={A cost-benefit analysis of cross-lingual transfer methods},
    author={Moraes, Guilherme and Bonifácio, Luiz Henrique and Rodrigues de Souza, Leandro and Nogueira, Rodrigo and Lotufo, Roberto},
    journal={arXiv preprint arXiv:2105.06813},
    url={https://arxiv.org/abs/2105.06813},
    year={2021}
}
~~~


