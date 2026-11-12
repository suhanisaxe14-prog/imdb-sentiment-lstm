# IMDB Movie Review Sentiment Analysis with LSTM

A deep learning project that classifies IMDB movie reviews as **positive** or **negative** using an **LSTM (Long Short-Term Memory)** network built with TensorFlow/Keras.

**Final test accuracy: 86.86%** (25,000 unseen reviews)

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [What is an LSTM?](#2-what-is-an-lstm)
3. [Why LSTM for this problem?](#3-why-lstm-for-this-problem)
4. [Dataset](#4-dataset)
5. [Preprocessing](#5-preprocessing)
6. [Model Architecture](#6-model-architecture)
7. [How the Model Works (Step by Step)](#7-how-the-model-works-step-by-step)
8. [Training Setup](#8-training-setup)
9. [Results](#9-results)
10. [Error Analysis](#10-error-analysis)
11. [Limitations](#11-limitations)
12. [Possible Improvements](#12-possible-improvements)

---

## 1. Project Overview

Sentiment analysis asks a simple question: *is this piece of text positive or negative?* It is a classic Natural Language Processing (NLP) task used for product reviews, social media monitoring and customer feedback.

This project:
- loads the IMDB review dataset that ships with Keras,
- converts each review into a fixed-length sequence of word indices,
- trains an LSTM network to predict sentiment,
- evaluates it with accuracy, a confusion matrix, precision/recall/F1,
- analyses the reviews the model gets most confidently wrong,
- tests it on custom reviews and explores decision-threshold tuning.

---

## 2. What is an LSTM?

An **LSTM (Long Short-Term Memory)** is a special kind of Recurrent Neural Network (RNN) designed to learn from **sequences**, such as sentences, time series or audio.

A normal neural network sees all its input at once, with no sense of order. An RNN reads a sequence **one step at a time** (here, one word at a time) and keeps a hidden state that acts as a memory of what it has read so far.

The problem with plain RNNs is the **vanishing gradient**: when learning from long sequences, the error signal shrinks as it travels back through many steps, so the network forgets information from early in the sequence. LSTMs were created to fix this.

### How an LSTM cell works

An LSTM keeps two things at every step:
- a **cell state**: a long-term memory that runs through the sequence with only small, controlled changes,
- a **hidden state**: the short-term output at the current step.

Three **gates** (small neural layers with sigmoid outputs between 0 and 1) control the memory:

| Gate | Question it answers | What it does |
|------|--------------------|--------------|
| **Forget gate** | "What old information should I throw away?" | Scales down parts of the cell state |
| **Input gate** | "What new information is worth storing?" | Decides what from the current word is added to the cell state |
| **Output gate** | "What should I pass on right now?" | Decides what part of the cell state becomes the hidden state |

In simplified form, at each step `t` with input word vector `x_t`:

```
forget gate:  f_t = sigmoid(W_f · [h_(t-1), x_t] + b_f)
input gate:   i_t = sigmoid(W_i · [h_(t-1), x_t] + b_i)
candidate:    c~_t = tanh(W_c · [h_(t-1), x_t] + b_c)
cell state:   c_t = f_t * c_(t-1) + i_t * c~_t
output gate:  o_t = sigmoid(W_o · [h_(t-1), x_t] + b_o)
hidden state: h_t = o_t * tanh(c_t)
```

Because the cell state is updated by addition rather than repeated multiplication, information (and gradients) can flow across many steps, so the network can remember that a review started with "I expected to love this film..." even when the verdict comes 100 words later.

---

## 3. Why LSTM for this problem?

Movie reviews are **sequences where order and context matter**:

- *"not bad at all"* is positive, but a model that only counts words sees "bad" and may predict negative.
- *"The acting was brilliant, but the story was boring and the ending ruined it"* mixes positive and negative words, and the final judgement depends on how the parts relate.

Compared with other approaches:

| Approach | Limitation for this task |
|----------|-------------------------|
| Bag-of-words / TF-IDF + logistic regression | Ignores word order and negation |
| Plain RNN | Struggles with long reviews (vanishing gradient) |
| LSTM | Reads in order, remembers context over long spans, trains in minutes on a CPU |
| Transformers (e.g. BERT) | Higher accuracy, but needs far more compute and is heavier to train and explain |

An LSTM is a good middle ground: it understands word order, is small (about 675k parameters) and is a standard building block for learning how sequence models work.

---

## 4. Dataset

- **Source:** IMDB Movie Reviews dataset, loaded with `keras.datasets.imdb`
- **Size:** 50,000 reviews in total: **25,000 for training** and **25,000 for testing**
- **Labels:** `0` = negative, `1` = positive
- **Balance:** perfectly balanced (12,500 positive and 12,500 negative reviews in the test set)
- **Format:** each review is already converted to a list of integers, where each integer is a word's rank by frequency
- **Vocabulary:** limited to the **10,000 most frequent words**; rarer words are replaced by an `<UNK>` token
- **Special IDs:** 0 = padding, 1 = start of review, 2 = unknown word (real words are offset by 3)

Review lengths vary a lot:

| Statistic | Tokens |
|-----------|--------|
| Minimum | 11 |
| Median | 178 |
| Maximum | 2,494 |
| Reviews longer than 200 tokens | 10,677 of 25,000 (about 43%) |

---

## 5. Preprocessing

Neural networks need inputs of equal size, so every review is converted to exactly **200 tokens**:

- Longer reviews are **truncated from the start** (`truncating="pre"`), keeping the last 200 tokens, which is where reviews tend to give their final verdict.
- Shorter reviews are **padded with zeros at the end** (`padding="post"`).
- The embedding layer uses `mask_zero=True`, so the LSTM **ignores the padding** instead of treating it as meaningful input.

Result: `x_train` and `x_test` both have shape `(25000, 200)`.

---

## 6. Model Architecture

```
Input (200 token IDs)
   ↓
Embedding (10,000 words → 64-dimensional vectors, masking enabled)
   ↓
SpatialDropout1D (0.3)
   ↓
LSTM (64 units, dropout 0.2)
   ↓
Dense (32 units, ReLU)
   ↓
Dropout (0.3)
   ↓
Dense (1 unit, sigmoid)  →  probability that the review is positive
```

| Layer | Output shape | Parameters |
|-------|--------------|-----------|
| Embedding | (200, 64) | 640,000 |
| SpatialDropout1D | (200, 64) | 0 |
| LSTM (64) | (64) | 33,024 |
| Dense (32, ReLU) | (32) | 2,080 |
| Dropout | (32) | 0 |
| Dense (1, sigmoid) | (1) | 33 |
| **Total** | | **675,137** |

**Why each layer is there**
- **Embedding:** turns each word ID into a learned 64-number vector, so words with similar meaning (e.g. *great*, *excellent*) end up close together.
- **SpatialDropout1D:** randomly drops entire embedding dimensions during training, which regularises the embeddings.
- **LSTM:** reads the 200 word vectors in order and compresses the whole review into one 64-number summary.
- **Dense (ReLU):** learns non-linear combinations of that summary.
- **Dropout:** randomly switches off units during training to reduce overfitting.
- **Dense (sigmoid):** outputs a probability between 0 and 1.

---

## 7. How the Model Works (Step by Step)

Example review: *"this movie was wonderful and the acting was brilliant"*

1. **Encode:** each word is mapped to its integer ID (rare or unknown words become `<UNK>`).
2. **Pad/truncate:** the sequence is made exactly 200 tokens long.
3. **Embed:** each ID becomes a 64-dimensional vector.
4. **Read in order:** the LSTM processes the vectors one by one. At each word its gates decide what to forget, what to store and what to output. Padding is skipped through masking.
5. **Summarise:** after the last real word, the final hidden state (64 numbers) represents the whole review.
6. **Classify:** the dense layers turn that summary into a single probability.
7. **Decide:** a probability of 0.5 or higher means **positive**, otherwise **negative**.

For the example above the model outputs **P(positive) ≈ 0.91**.

---

## 8. Training Setup

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Loss | Binary cross-entropy |
| Batch size | 32 |
| Maximum epochs | 20 |
| Validation | 20% of the training data held out |
| Early stopping | Monitor `val_loss`, patience 2, restore best weights |
| Random seed | 42 |
| Hardware | CPU (about 80 to 100 seconds per epoch) |

Training stopped after epoch 4, and the weights from **epoch 2** (the lowest validation loss, 0.2958) were restored. Later epochs showed training accuracy still rising while validation loss increased, which is the classic sign of overfitting.

---

## 9. Results

### Test set performance

| Metric | Value |
|--------|-------|
| **Accuracy** | **86.86%** |
| Loss | 0.316 |
| Precision (positive) | 0.87 |
| Recall (positive) | 0.87 |
| F1-score | 0.87 |

### Confusion matrix (test set)

|  | Predicted negative | Predicted positive |
|--|-------------------|-------------------|
| **Actual negative** | 10,844 | 1,656 |
| **Actual positive** | 1,629 | 10,871 |

3,285 of 25,000 reviews were misclassified. Errors are spread almost equally across both classes, so the model is **not biased** toward positive or negative.

### Custom review test

| Review | P(positive) |
|--------|-------------|
| "this movie was wonderful and the acting was brilliant" | 0.910 |
| "a boring and terrible film i want my time back" | 0.105 |
| "not bad at all" | 0.159 |

The third review is actually positive, but the model leans negative. It shows how hard **negation** is for a small model.

### Threshold tuning (validation data)

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|----|
| 0.3 | 0.828 | 0.919 | 0.871 |
| 0.4 | 0.853 | 0.898 | 0.875 |
| **0.5** | 0.875 | 0.885 | **0.880** |
| 0.6 | 0.894 | 0.857 | 0.875 |
| 0.7 | 0.915 | 0.822 | 0.866 |

The default threshold of 0.5 gives the best F1, so it is kept. A lower threshold would catch more positives (higher recall) at the cost of precision, and a higher one does the reverse.

---

## 10. Error Analysis

The notebook prints the reviews the model got **most confidently wrong**. Typical patterns:

- **Mixed reviews:** a long run of praise for the acting or cinematography, then a negative verdict near the end. The model is swayed by the many positive words.
- **Sarcasm and irony:** positive-sounding words used negatively.
- **Truncation:** about 43% of reviews are longer than 200 tokens, so the model never sees the first part of those.
- **Unknown words:** rare words outside the 10,000-word vocabulary become `<UNK>` and carry no information.

---

## 11. Limitations

- **Accuracy ceiling:** a plain LSTM reaches roughly 85 to 88% on IMDB. Pretrained transformer models (BERT and similar) reach around 93 to 95%.
- **Fixed length:** every review is cut or padded to 200 tokens, so information in long reviews is lost.
- **Limited vocabulary:** only the 10,000 most frequent words are known. Everything else becomes `<UNK>`.
- **Embeddings trained from scratch:** the model learns word meaning only from 25,000 reviews instead of using pretrained embeddings.
- **Weak on negation, sarcasm and mixed sentiment**, as the custom review "not bad at all" shows.
- **Overfitting:** validation loss stops improving after about 2 epochs, even with dropout.
- **Single-domain:** it is trained on movie reviews only and may perform worse on product reviews, tweets or other text.
- **Binary output only:** it predicts positive or negative, with no neutral class and no aspect-level sentiment (acting vs story).

---

## 12. Possible Improvements

- Use **pretrained embeddings** (GloVe or Word2Vec).
- Try a **Bidirectional LSTM** or a **GRU**, or stack two LSTM layers.
- Add an **attention** layer so the model can focus on the key words.
- Increase `max_len` or the vocabulary size.
- Tune hyperparameters (units, dropout, learning rate, batch size).
- **Fine-tune a transformer** such as DistilBERT for higher accuracy.
- Add a small web app (for example with Streamlit) for live predictions.

---

## Tech Stack

Python · TensorFlow / Keras · NumPy · Matplotlib · scikit-learn · Jupyter Notebook
