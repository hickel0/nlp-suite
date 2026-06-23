# import libs
import re
import math
import numpy as np

# Open and read a file
with open("Natural Language Tech/AcademyAwardsWikipedia.txt", "r", encoding="utf-8") as file:
    corpus = file.read()

# (a) Remove Punctuation
clean_text = re.sub(r"[^A-Za-z0-9\s-]", "", corpus)

# (b) convert to lowercase
clean_text = clean_text.lower()

# (c) tokenise on whitespace and newlines
tokens = clean_text.split()

# 3, Building a UNIGRAM Model
vocab, counts = np.unique(tokens, return_counts=True)
N = counts.sum()
alpha = 1.0
V = vocab.size
probs = (counts + alpha) / (N + alpha * V)# P(word) = count / N

def p_word(w:str) -> float:
    w = w.lower()
    i = np.searchsorted(vocab, w)
    if 0 <= i < vocab.size and vocab[i] == w:
        return float(probs[i])
    else:
        # add smoothign for part 6 for non zero answer, provides slight proabbility to unseen "Ireland"
        return alpha / (N + alpha * V)

def sent_prob(sentence: str) -> float:
    p = 1.0
    for w in sentence.lower().split():
        pw = p_word(w)
        if pw == 0.0:
            return 0.0
        p *= pw
    return p

def sent_logprob(sentence: str) -> float:
    lp = 0.0
    for w in sentence.lower().split():
        pw = p_word(w)
        if pw == 0.0:
            return float("-inf")
        lp += math.log(pw)
    return lp

def perplexity(sentence: str) -> float:
    ws = sentence.lower().split()
    lp = sent_logprob(sentence)
    return float("inf") if lp == float("-inf") else math.exp(-lp / len(ws))



if __name__ == '__main__':
    #print(clean_text)
    #print(tokens[:50])
    #print(vocab, counts, N, probs)
    # DEMO: “The best actor”
    ba = "The best actor"
    print(ba)
    print("P(sentence):", sent_prob(ba))
    print("log P(sentence):", sent_logprob(ba))
    print("perplexity:", perplexity(ba))

    baI = "The best actor in Ireland"
    print("\n",baI)
    print("P(sentence):", sent_prob(baI))
    print("log P(sentence):", sent_logprob(baI))
    print("perplexity:", perplexity(baI))

