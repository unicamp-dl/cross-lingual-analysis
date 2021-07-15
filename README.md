# Cross-lingual analysis

This repository contains the translation codes used in the paper [A cost-benefit analysis of cross-lingual transfer methods](https://arxiv.org/abs/2105.06813). In this work, we analyze cross-lingual methods on three tasks in terms of their effectiveness (e.g., accuracy), development and deployment costs, as well as their latencies at inference time. We experiment with the following transfer learning techniques: 1) fine-tuning a bilingual model on a source language and evaluating it on the target language without translation, i.e., in a zero-shot manner; 2) automatic translation of the training dataset to the target language; 3) automatic translation of the test set to the source language at inference time and evaluation of a model fine-tuned in English. Finally, by combining zero-shot and translation methods, we achieve the state-of-the-art in two of the three datasets used in this work. The study is a result of an ongoing Master's Program.

## Evaluation benchmarks
The models were benchmarked on three tasks (Question Answering, Natural Language Inference and Passage Text Ranking) and compared to previous published results. Metrics are: Exact Match and F1-score for Q&A, Accuracy and F1-score for NLI and MRR@10 for Text Ranking.

| Model                           | Pre-train     | Fine-tune       | F1           | Accuracy    | 
| ------------------------------- | ------------- | --------------- | ------------ | ----------- | 
| mBERT (Souza et al.)            | 100 languages |   ASSIN2        |   0.8680     |   0.8680    |
| PTT5 (Carmo et al.)             | EN & PT       |   ASSIN2        |   0.8850     |   0.8860    |
| BERTimbau Large (Souza et al.)  | EN & PT       |   ASSIN2        |   0.9000     |   0.9000    |
| BERT-pt (ours)                  | EN & PT       |  MNLI + ASSIN2  |   0.9207     |   0.9207    |    

We also evaluated a T5-based model fine-tuned on our translated version of MS MARCO passages dataset.
To do so, we finetuned PTT5 on :brazil: MS MARCO and on a joint version (:us: + :brazil:). 
Our baseline method uses BM25. All evaluation were made on :brazil: MS MARCO. 

| Method                          | Pre-train       | Fine-tune         | MRR@10     | 
| ------------------------------- | ----------------| ------------------| ---------- | 
| BM25 (Robertson et al.)        |      -          |        -          |   0.14     |   
| BM25 + T5 (Nogueira et al.)     | :us:            |   :us:            |   0.23     |
| BM25 + PTT5 (ours)              | :brazil:        |   :brazil:        |   0.27     |
| BM25 + PTT5 (ours)              | :brazil:        |  :us: + :brazil:  |   0.28     |    

# How to use ptt5-base-msmarco-pt-10k and ptt5-base-msmarco-en-pt-10k:

## Available models
Our PTT5 fine-tuned models are available for use with the  [🤗Transformers API](https://github.com/huggingface/transformers).

<!-- Com link -->
| Model                                    | Size                                                         |      |
| :-:                                      | :-:                                                          |      |
| [unicamp-dl/ptt5-base-msmarco-pt-10k](https://huggingface.co/unicamp-dl/ptt5-base-msmarco-pt-10k)       | base | 
| [unicamp-dl/ptt5-base-msmarco-en-pt-10k](https://huggingface.co/unicamp-dl/ptt5-base-msmarco-en-pt-10k) | base |




## Example usage:
```python
# Tokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration


model_name = 'unicamp-dl/ptt5-base-msmarco-pt-10k'

tokenizer  = T5Tokenizer.from_pretrained(model_name)

# PyTorch
model_pt   = T5ForConditionalGeneration.from_pretrained(model_name)

```


## How to Translate 

We made available the following data and the respectives notebooks with translation code:
- [SQuAD](https://colab.research.google.com/drive/1CSNwfWJCwhFgYTtjxsDvUdN1DMuU_P4-?usp=sharing) (Q&A)
- [FaQuAD](https://colab.research.google.com/drive/1HdPjzn61genPyZfiDG5fqAwPNI4vhegw?usp=sharing) (Q&A)
- [MNLI](https://colab.research.google.com/drive/1Y9ZaJuN-SVo0fmwypPzcJCOetFw-A2tx?usp=sharing) (NLI)
- [ASSIN2](https://colab.research.google.com/drive/1S5zwaw8KWee8y6Vyq-XHC8Am3GmGFpv9?usp=sharing) (NLI)

The datasets SQuAD and MNLI are directly downloaded from the notebooks of this repository. We also provide the FaQuAD and ASSIN2 datasets.

|                                 |   SQuAD    | FaQuAD   | MNLI       | ASSIN2   | 
| ------------------------------- | ---------- | -------- | ---------- |--------- | 
| Training examples               |  86,288    |   837    |   392,702  |  6,500   |   
| Test examples                   |  21,557    |   63     |   20,000   |  2,448   |    
| Translate Train (Batch size = 1)|    34h     |   -      |    36h     |    -     |  
| Translate Test (Batch size = 1) |    -       |  1m 30s  |    -       |   31m    |


## References

[1] [BERTimbau: Pretrained BERT Models for Brazilian Portuguese](https://www.researchgate.net/publication/345395208_BERTimbau_Pretrained_BERT_Models_for_Brazilian_Portuguese)

[2] [PTT5: Pretraining and validating the T5 model on Brazilian Portuguese data](https://arxiv.org/abs/2008.09144)

## How do I cite this work?

~~~ {.xml
 @article{cross-lingual2021,
    title={A cost-benefit analysis of cross-lingual transfer methods},
    author={Moraes, Guilherme and Bonifacio, Luiz Henrique and Rodrigues de Souza, Leandro and Nogueira, Rodrigo and Lotufo, Roberto},
    journal={arXiv preprint arXiv:2105.06813},
    url={https://arxiv.org/abs/2105.06813},
    year={2021}
}
~~~


