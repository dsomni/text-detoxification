# Solution building

## Datasets

Initial dataset contains a lot of data (577 777 rows), but there are the following problems:

- For some data points toxicity of translation are greater than toxicity of the initial sentence
- Most of the data are not representative: reference toxicity is not quite high and/or translation toxicity is not quite low

In order to solve these problems, I do the following inside `src/data/make_dataset.py`:

- Swap `ref_tox` and `trn_tox` if necessary (when `ref_tox` <`trn_tox`)
- Provide some threshold for `ref_tox` and `trn_tox` in order to have only representative data

After processing I split data into train and test datasets: train is used to train and validate model, and test is used to compute metrics and test model performance on (potentially) unseen data.

Moreover, as computational resources (and time!) are limited, I decided construct several train data files with different sizes. Size reduction is based on the different thresholds for `ref_tox` and `trn_tox`. You can check corresponding size map below:

```python
size_map = {
        "lg": {"toxicity_threshold": 0.9, "no_toxicity_threshold": 0.1},
        "md": {"toxicity_threshold": 0.99, "no_toxicity_threshold": 0.01},
        "sm": {"toxicity_threshold": 0.999, "no_toxicity_threshold": 0.001},
        "xs": {"toxicity_threshold": 0.9994, "no_toxicity_threshold": 0.0001},
    }
```

At first steps I also wanted to perform tokenization, remove stop words and punctuation and lowercase data. However, later I discard this idea due to the following reasons:

- Some models already have their own pretrained tokenizers
- We want models to produce output in the same form as input, i.e. with punctuation
- Lowercasing can lose the information, i.e. name Dick (Richard) should not be considered as toxicity

## Hypothesis 1. Custom Torchtext transformer

I will refer to this model and architecture as Custom Transformer is the text or `custom_transformer` in the code.

The idea was to create Transformer architecture from scratch using PyTorch and Torchtext libraries. I choose Transformer architecture as it requires much less training time than, for example, LSTM, and relies on quite efficient attention mechanism. In general, Transformer architecture is still considered as SOTA and it is not very hard to build one from scratch (with help of mentioned libraries, of course). As text detoxification task is non-trivial, I also add to the pure Transformer the following improvements:

- Positional encoding layer to make model better understand the internal text structure
- Embedding layer to learn task-specific embeddings
- Masking mechanism to prevent the transformer attention from “cheating” during training

As training process on entire dataset was dramatically slow and CUDA memory ran out from time to time , I decided to reduce data size and/or reduce the number of neurons inside the network.

## Hypothesis 2. Data reduction

This reduction process itself is described well in the Datasets section. For Custom Transformer I use the `dataset_md.csv` with ~100 000 rows. At this point I realize, that reducing dataset is more helpful to increase the training speed than reducing number of neurons in network. The reason is that the bigger dataset is the bigger the vocabulary becomes, what, in turn, results in not only huge amount of neurons on output layer, but in astronomical memory consumption of data loaders. Therefore, I did not change the network parameters much, but only decreased the dataset size.

Training process and other technical details are described in the main `Report.pdf`. Finally, Custom Transformer shows satisfying results (subject to limited computational resources and time), but it could be better. So I decided to move to the next hypothesis.

## Hypothesis 3. Fine-tunning BART

I will refer to this model and architecture as BART is the text or `bart` in the code.

The BART model was proposed in 2019. It use a standard sequence-to-sequence architecture with a bidirectional encoder and left-to-right encoder. BART is particularly effective for text generation tasks, but also works well for comprehension tasks. As the formulated task is obviously sequence-to sequence and requires comprehension, I find BART suitable for the given task. For this assignment I use the [following](https://huggingface.co/eugenesiow/bart-paraphrase) BART architecture from Hugging Face Hub and decide to fine-tune it to solve text detoxification task. This model was pre-trained of paraphrasing datasets, so it is perfectly fits purposes of the detoxification task. Also I take the corresponding pre-trained tokenizer. After all, I save it not only locally (the model is very heavy for GitHub submission), but also decide to push [it](https://huggingface.co/dsomni/pmldl1-bart) to Hugging Face Hub.

Training process and other technical details are described in the main `Report.pdf`. Finally, fine-tunned BART shows better results than Custom Transformer. The metrics are discussed in more details also in the main `Report.pdf`.

## Results

During this assignment I built two different solutions for the text detoxification task: Custom Transformer, constructed from scratch, and fine-tunned BART. I got a lot of different experience while working with these two solutions. Talking about Custom Transformer, I get experience of building from scratch not only network itself, but also such improvements as positional encoding and masking mechanism. In contrast, fine-tunning BERT teach me how to work with other pre-trained models and trainers from Hugging Face Hub.

As a result, I have two working models: Custom Transformer and BART. Results in terms of metrics are described in the main `Report.pdf`. Generally, BART obviously show the better performance both in terms of metrics and user experience. By user experience I mean quality of generated text, especially working in interactive mode with script `src/models/predict_model.py`(you can try it out!). However, Custom Transformer is still quite good, especially subject to the fact that the model itself is quite compact (~80 Mb) in comparison with fine-tunned BART. Therefore, I guess both hypothesis are turned out to be successive in some aspects.
