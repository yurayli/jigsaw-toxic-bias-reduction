## Jigsaw Unintended Bias in Toxicity Classification

----
https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification

#### Task
Detect toxic comments and minimize unintended model bias

#### Data processing and tricks
1. Few text preprocessing is needed for NN solutions.
2. Bucket sampler saves a lot of time (2-3x faster), see the code for details
 * separate data for buckets, several batches of samples a bucket
 * sort sequence lengths for each bucket
 * pad a batch from max_seq_length in the batch
3. Custom loss or sample weighting required for the mitigation of the model bias
4. Soft label contains more information and can be computed with BCE loss.
5. Pseudo label is helpful for LSTM-based NN
6. Knowledge distillation can compress an ensemble to a single model with comparative results.

#### Architectures
1. Word embedding + LSTM-based networks. The embedding is also finetuned with smaller learning rate. Whole network is trained in one-cycle cosine annealed learning rate schedule ([ref](https://docs.fast.ai/callbacks.one_cycle.html)).
2. [BERT](https://arxiv.org/abs/1810.04805) finetuning with slanted triangular learning rate schedule ([ref](https://arxiv.org/abs/1801.06146)).
3. [GPT-2](https://openai.com/blog/better-language-models/) finetuning with slanted triangular learning rates.

Code in 2. 3. is based on [huggingface's code](https://github.com/huggingface/pytorch-transformers), and the notebooks are run on kaggle kernel.

### Dependencies
Pytorch 1.2.0
