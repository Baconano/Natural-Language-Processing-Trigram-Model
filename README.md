# Trigram Language Model with Linear Interpolation

## Overview

This project implements a **trigram language model** with **linear interpolation smoothing** to calculate the probability of sequences of words and the **perplexity** of a test set.  
It demonstrates fundamental concepts in **Natural Language Processing (NLP)**, including tokenization, n-gram counts, and smoothing techniques.

Punctuations and 's are treated as separate tokens
Base Sentence: the cat watched children play in the park.
- Unigram -> the / cat / watched / children / play / in / the / park / .
- Bigram - > the cat / cat watched / watched children / children play / play in / in the / the park / park.
- Trigram -> the cat watched / cat watched children / watched children play / children play in / play in the / in the park / the park . /

---

## Features

- Tokenizes text, treating **punctuation and `'s` as separate tokens**.  
- Computes **unigram, bigram, and trigram counts** from a training corpus.  
- Calculates **trigram probabilities using linear interpolation**:  

\[
P(w_3 | w_1 w_2) = \lambda_3 P_{tri} + \lambda_2 P_{bi} + \lambda_1 P_{uni}
\]

- Handles **unseen trigrams** using **add-1 (Laplace) smoothing**.  
- Computes **perplexity** of a test set to evaluate model performance.

---

## Training Corpus

The model is trained on 10 sentences including:

- `"the cat watched children play in the park."`  
- `"the library held tales of the ancient clock tower."`  
- `"at night, they lay under a star-filled sky."`  

> Full corpus is included in the `corpus` variable in the code.

---

## Test Corpus

The model is evaluated on 3 sentences:

- `"the sunset painted the city sky with colors."`  
- `"an old train's whistle echoed past the lake."`  
- `"soft music played in the cozy cafe."`  

---

## How It Works

1. **Tokenization**  
   - Words and punctuation are separated.  
   - Apostrophes are normalized (`’` → `'`).  

2. **Count n-grams**  
   - `unigram_counts`, `bigram_counts`, `trigram_counts` using Python `Counter`.  

3. **Trigram Probability**  
   - Uses linear interpolation of unigram, bigram, and trigram probabilities.  
   - Add-1 smoothing is applied to avoid zero probabilities.  

4. **Perplexity Calculation**  
   - Measures how well the model predicts the test set.  
   - Lower perplexity = better prediction.

---


