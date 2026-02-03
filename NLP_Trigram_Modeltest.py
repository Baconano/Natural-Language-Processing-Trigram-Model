
'''
corpus provided for training the trigram model consists of these sentences:
the cat watched children play in the park.
their laughter echoed near the fragrant garden.
the breeze spread the garden's scent into the city.
it wafted past the cafe, famous for apple pie.
the cafe's aroma reminded people of the nearby library.
the library held tales of the ancient clock tower.
the tower tolled, echoing in the quiet morning streets.
these streets, bustling by day, were peaceful at dawn.
at night, they lay under a star-filled sky.
the moonlight shone on the lake where a fisherman waited.

The corpus provided for testing the trigram model consists of the following three sentences:
the sunset painted the city sky with colors.
an old train's whistle echoed past the lake
soft music played in the cozy cafe

Also note that punctuations and 's are treated as seperate tokens
ex: the cafe's aroma.
-> the / cafe / 's/ aroma / .
Base Sentence: the cat watched children play in the park.
Unigram -> the / cat / watched / children / play / in / the / park / .
Bigram - > the cat / cat watched / watched children / children play / play in / in the / the park / park.
Trigram -> the cat watched / cat watched children / watched children play / children play in / play in the / in the park / the park . /
'''
import re
from collections import Counter
import math


def tokenize(text):
    text = text.replace("’", "'")
    return re.findall(r"\w+|[.,'’\-]", text)
corpus = [
    "the cat watched children play in the park.",
    "their laughter echoed near the fragrant garden.",
    "the breeze spread the garden's scent into the city.",
    "it wafted past the cafe, famous for apple pie.",
    "the cafe's aroma reminded people of the nearby library.",
    "the library held tales of the ancient clock tower.",
    "the tower tolled, echoing in the quiet morning streets.",
    "these streets, bustling by day, were peaceful at dawn.",
    "at night, they lay under a star-filled sky.",
    "the moonlight shone on the lake where a fisherman waited."
]

tokenized_corpus = []
for sentence in corpus:
    tokens = tokenize(sentence)
    tokenized_corpus.append(['<s>','<s>'] + tokens + ['</s>'])

unigram_counts = Counter()
bigram_counts = Counter()
trigram_counts = Counter()

for sentence in tokenized_corpus:
    for i in range(len(sentence)):
        unigram_counts[sentence[i]] +=1
        if i>= 1:
            bigram_counts[(sentence[i-1], sentence[i])] += 1
        if  i>= 2:
            trigram_counts[(sentence[i-2], sentence[i-1], sentence[i])] +=1

total_unigrams = sum(unigram_counts.values())

lambda1, lambda2,lambda3 = 0.5, 0.4 , 0.1

def trigram_prob(w1,w2,w3,alpha=1):
    V = len(unigram_counts)

    trigram_count = trigram_counts.get((w1,w2,w3), 0) + alpha
    bigram_count_for_trigram = bigram_counts.get((w1,w2), 0) + alpha * V
    p3 = trigram_count / bigram_count_for_trigram

    bigram_count = bigram_counts.get((w2,w3), 0) + alpha
    unigram_count_for_bigram = unigram_counts.get(w2, 0) + alpha * V
    p2 = bigram_count / unigram_count_for_bigram

    p1 = (unigram_counts.get(w3, 0) + alpha) / (total_unigrams + alpha * V)
    
    return lambda3 * p3 + lambda2 * p2 + lambda1 * p1

test_sentence =[
    "the sunset painted the city sky with colors.",
    "an old train's whistle echoed past the lake.",
    "soft music played in the cozy cafe."
]

tokenized_tests = []
for sentence in test_sentence:
    tokens = tokenize(sentence)
    tokenized_tests.append(['<s>','<s>']+tokens + ['</s>'])

total_log_prob = 0
total_tokens = 0



for idx, sentence in enumerate(tokenized_tests):
    print(f"\nTest Sentence {idx+1}: {' '.join(sentence[2:-1])}")
    for i in range(2, len(sentence)):
        w1,w2,w3 = sentence[i-2], sentence[i-1], sentence[i]
        prob = trigram_prob(w1,w2,w3)
        print(f"P({w3} | {w1} {w2}) = {prob:.6f}")

for sentence in tokenized_tests:
    for i in range(2,len(sentence)):
        w1,w2,w3 = sentence[i-2], sentence[i-1],sentence[i]
        prob = trigram_prob(w1,w2,w3)
        if prob == 0:
            prob = 1e-12
        total_log_prob += math.log2(prob)
        total_tokens += 1
perplexityt = 2 **(-total_log_prob / total_tokens)
print(f"\nPerplexity of the test set: {perplexityt:.4f}")

