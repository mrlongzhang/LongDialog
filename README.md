# Long Dialog



## Introduction

Long's private dialog research toolkit. This toolkit is based on [DeepQA](https://github.com/abs/deeqqa). My focus is on model, so the boiler-plate modules of DeepQA were left intact. However, DeeqQA's seq2seq module was dropped as it used legacy rnn code, where attention mechanism is awfull (slow, low efficiency, not working sometimes). In addition, further models could be added. All modules were upgraded when necessary.

DeepQA features a structured corpus handling module. For more information, please kindly visit DeepQA. 

## Dependency

The following package are prerequisited:

* python 3.6
* tensorflow (with v1.9)
* numpy
* CUDA (for using GPU)
* nltk (natural language toolkit for tokenized the sentences)
* tqdm (for the nice progression bars)

To install these packages, simply type: `pip3 install -r requirements.txt`

NOTE: punkt is require for nltk package. 

```
python3 -m nltk.downloader punkt
```

The Cornell dataset is already included. For the other datasets, look at the readme files into their respective folders (inside `data/`).

## Running

### bot

To train the model, simply run `main.py`. Once trained, you can test the results with `main.py --test` (results generated in 'save/model/samples_predictions.txt') or `main.py --test interactive` (more fun).

Here are some flags which could be useful. For more help and options, use `python main.py -h`:

* `--modelTag <name>`: allow to give a name to the current model to differentiate between them when testing/training.
* `--keepAll`: use this flag when training if when testing, you want to see the predictions at different steps (it can be interesting to see the program changes its name and age as the training progress). Warning: It can quickly take a lot of storage space if you don't increase the `--saveEvery` option.
* `--filterVocab 20` or `--vocabularySize 30000`: Limit the vocabulary size to and optimize the performances and memory usage. Replace the words used less than 20 times by the `<unknown>` token and set a maximum vocabulary size.
* `--verbose`: when testing, will print the sentences as they are computed.
* `--playDataset`: show some dialogue samples from the dataset (can be use conjointly with `--createDataset` if this is the only action you want to perform).

To visualize the computational graph and the cost with [TensorBoard](https://www.tensorflow.org/how_tos/summaries_and_tensorboard/), just run `tensorboard --logdir save/`.

By default, the network architecture is a standard encoder/decoder with four GRU layers (hidden size of 256), with embedding, attention and beam search. The sentence length is set to 10 words.

## Results

We could never reproduce the orginal DeepQA's dialog samples as we only trained the system with only Cornell's dataset. Theirs were amazing, ours sucks. However, it can be seen that the discrepency is mainly contributed by the dataset.

Q: hi
A: hi, mrs. robinson.

Q: how are you?
A: I am fine.

Q: what is your name?
A: I am working on.

Q: How old are you?
A: I am fine.



## Pretrained model
Currently, there is no pre-trained model yet. Soon it will be here.

## Future works

Currently, a basic seq2seq model [Cho et al., 2014][c1] is implemented. It has attention mechanism plus beam search capapbility. It barely have memory capability, which is vital for consistency and stable multi-round talk. But that's a long term goal, so far. 

* A much less ambitouse goal would be an antiLM model [Jiwei Li et al.][c3] to improve one round dialog first. 
* Add peronality capability to the bot.
* Employ reading comprehension NN to improve the bot's reasoning capapbility.
* Employ memory NN to develop multi-round dialog capapbility.
* GAN for generative output
* Reinforcement learning
* ...
* In the end, for a bot to be useful, knowledge graph technology would be really helpful.

[c1]: http://arxiv.org/abs/1406.1078

[c2]: https://en.wikipedia.org/wiki/Maximum_likelihood_estimation

[c3]: http://arxiv.org/pdf/1510.03055v3.pdf
