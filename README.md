# QA-playground

### Improving In-context Learning Performance for Prompting, Demonstration Retrieval, Bootstrapping via Permutation


## Approach
* Random Sampling Baseline
* Retrieval-Based Prompt Selection Approach
* Bootstrapping via Permutation
    * Majority Voting
    * Mean Embedding used by Sentence Encoder

## Model
* Pythia (160M, 410M, 1.4B, 2.8B, 6.9B)
* LLaMA-2 (7B)

## Report
* [Notion](https://bottlenose-bracket-787.notion.site/Improving-In-context-Learning-Performance-for-Prompting-Demonstration-Retrieval-Bootstrapping-via--2c3982b2f7804a62a3f45ee85a956106)


## Dependency
* Pytorch
* Numpy
* Pandas
* Huggingface transformers
* Scikit-learn


## Reference
* [Language models are few-shot learners.](https://arxiv.org/abs/2005.14165)
* [Calibrate Before Use: Improving Few-Shot Performance of Language Models.](https://arxiv.org/abs/2102.09690)
* [What Makes Good In-Context Examples for GPT-3?](https://arxiv.org/abs/2101.06804)
* [Llama 2: Open Foundation and Fine-Tuned Chat Models.](https://arxiv.org/abs/2307.09288)
* https://github.com/princeton-nlp/SimCSE
* https://github.com/jiachangliu/KATEGPT3