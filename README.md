# Long Dialog

[![Join the chat at https://gitter.im/chatbot-pilots/DeepQA](https://badges.gitter.im/chatbot-pilots/DeepQA.svg)](https://gitter.im/chatbot-pilots/DeepQA?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

## Introduction

Long's private dialog research toolkit. This toolkit is based on [DeepQA](https://github.com/abs/deeqqa). My focus is on model, so the boiler-plate modules of DeepQA were left intact. However, DeeqQA's seq2seq modul were dropped as it used legacy rnn code. In addition, further models could be added. All modules were upgraded when necessary.

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

We could never reproduce the orginal DeepQA's dialog samples as we only trained the system with Cornell's dataset. Theirs were amazing, ours sucks.

    Q: Hi
    A: Hi.
    
    Q: What is your name ?
    A: Laura.
    
    Q: What does that mean ?
    A: I dunno.
    
    Q: How old are you ?
    A: thirty-five.
    
    Q: Will Google hire me ?
    A: No.
    
    Q: Tell me the alphabet
    A: Fuck you.
    
    Q: That's not nice
    A: Yeah.
    
    
    
    Q: My name is David. What is my name ?
    A: Debbie.
    
    Q: My name is John. What is my name ?
    A: Carl Smith.
    
    Q: How many legs does a cat have ?
    A: Four.
    
    Q: How many legs does a spider have ?
    A: Four.
    
    Q: Do you prefer football or soccer ?
    A: Whiskey.

## Pretrained model

You can find a pre-trained model [here](https://drive.google.com/file/d/0Bw-phsNSkq23OXRFTkNqN0JGUU0/view?usp=sharing), trained of the default corpus. To use it:

1. Extract the zip file inside `DeepQA/save/`
2. Copy the preprocessed dataset from `save/model-pretrainedv2/dataset-cornell-old-lenght10-filter0-vocabSize0.pkl` to `data/samples/`.
3. Run `./main.py --modelTag pretrainedv2 --test interactive`.

Thanks to Nicholas C., [here](https://drive.google.com/drive/folders/0Bw-phsNSkq23c29ZQ2N6X3lyc1U?usp=sharing) ([original](https://mcastedu-my.sharepoint.com/personal/nicholas_cutajar_a100636_mcast_edu_mt/_layouts/15/guestaccess.aspx?folderid=077576c4cf9854642a968f67909380f45&authkey=AVt2JWMPkf2R_mWBpI1eAUY)) are some additional pre-trained models (compatible with TF 1.2) for diverse datasets. The folder also contains the pre-processed dataset for Cornell, OpenSubtitles, Ubuntu and Scotus (to move inside `data/samples/`). Those are required is you don't want to process the datasets yourself.

If you have a high-end GPU, don't hesitate to play with the hyper-parameters/corpus to train a better model. From my experiments, it seems that the learning rate and dropout rate have the most impact on the results. Also if you want to share your models, don't hesitate to contact me and I'll add it here.

## Future works

Currently, a basic seq2seq model [Cho et al., 2014][c1]. barely have memory capability, which is vital for consistency and stable multi-round talk. But that's a long term goal, so far. 

* A much less ambitouse goal would be an antiLM model [Jiwei Li et al.][c3] to improve one round dialog first. 
* Add peronality capability to the bot.
* Employ reading comprehension NN to improve the bot's reasoning capapbility.
* Employ memory NN to develop multi-round dialog capapbility.
* ...
* In the end, for a bot to be useful, knowledge graph technology would be really helpful.

[c1]: http://arxiv.org/abs/1406.1078

[c2]: https://en.wikipedia.org/wiki/Maximum_likelihood_estimation

[c3]: http://arxiv.org/pdf/1510.03055v3.pdf
