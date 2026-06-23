#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes from Scratch

# ## 1.1 Training

# In[2]:


import pandas as pd

splits = {'train': 'train.csv', 'test': 'test.csv'}
df = pd.read_csv("hf://datasets/okite97/news-data/" + splits["train"])
df = df.rename(columns=str.lower)


# In[3]:


df


# Write code to Calculate the prior probability of the six classes. For each word (type not token) in the training data, calculate the probabilty of the word given each class. You can use just the title, just the excerpt or the concatenation of the title and the excerpt. 
# 

# Prior Probability of a class c is : 
# 
#     P(c) = number of docs in class c / total number of docs 
#     

# In[5]:


from collections import Counter

# containment of labels for all train docs 
y_train = df["category"].tolist()

# Count how many docs per class and totall num of docs
class_counts = Counter(y_train)
total_docs = len(y_train)

class_counts, total_docs

# Now compute priors
priors = {c: class_counts[c] / total_docs for c in class_counts}
priors

print("Class Counts:", class_counts)
print("Class Priors:", priors)


# Now, word likelihoods, P(w | c) for every word type (not token). Treat all docs of a class as one mega doc; count num of times a word appears, then apply smoothing. 

# In[ ]:


import re 
from collections import defaultdict

tokens = re.compile(r"[A-Za-z]+")

def tokenize(text: str):
    if not isinstance(text, str):
        return []
    return [t.lower() for t in tokens.findall(text)]

def concat_title_excerpt(row):
    title = row.get("title") or ""
    excerpt = row.get("excerpt") or ""
    if pd.isna(title):
        title = ""
    if pd.isna(excerpt):
        excerpt = ""
    
    return (title + " " + excerpt).strip()

df["text"] = df.apply(concat_title_excerpt, axis=1)
df


# In[8]:


train_tokens = [tokenize(t) for t in df["text"]]

vocab = sorted({w for toks in train_tokens for w in toks})
V = len(vocab)
v2i = {w:i for i,w in enumerate(vocab)}


# In[9]:


labels = sorted(df["category"].unique())
y_train = df["category"].tolist()

# token counts for each class (mega-doc)
word_counts_by_class = {c: Counter() for c in labels}
total_tokens_by_class = {c: 0 for c in labels}

for toks, c in zip(train_tokens, y_train):
    word_counts_by_class[c].update(toks)
    total_tokens_by_class[c] += len(toks)

# quick checks
print("Vocab size:", V)
print({c: total_tokens_by_class[c] for c in labels})


# Compute Smoothed P(w | c) for every word TYPE

# In[10]:


import math

alpha = 1.0  # Laplace / add-1 
P_w_given_c = {c: defaultdict(float) for c in labels}
log_P_w_given_c = {c: defaultdict(float) for c in labels}

for c in labels:
    Nc = total_tokens_by_class[c]
    for w in vocab:
        count_wc = word_counts_by_class[c][w]  # 0 if unseen in c
        pwc = (count_wc + alpha) / (Nc + alpha * V)
        P_w_given_c[c][w] = pwc
        log_P_w_given_c[c][w] = math.log(pwc) 


# In[11]:


# Quick TEST

for c in labels:
    sample_words = ["nigeria","government","football","bank","market"]
    print(c, {w: round(P_w_given_c[c].get(w, 0.0), 6) for w in sample_words})


# ## 1.2 Testing

#  • Load the test split of the same dataset.
#  • Calculate the news category of each instance in the test set.
#  

# In[14]:


test_df = pd.read_csv("hf://datasets/okite97/news-data/" + splits["test"])
test_df = test_df.rename(columns=str.lower)


# In[15]:


test_df


# In[27]:


# Log priors
log_priors = {c: math.log(priors[c]) for c in labels}
# log_priors


# In[29]:


# 3) Predict function (ignores OOV words)
def predict_tokens(tokens):
    best_c, best_score = None, -1e100
    for c in labels:
        s = log_priors[c]
        for w in tokens:
            if w in vocab:                      # ignore words unseen
                s += log_P_w_given_c[c][w]
        if s > best_score:
            best_score, best_c = s, c
    return best_c


# In[39]:


# Tokenize test texts and concatenate title + excerpt

test_df["text"] = test_df.apply(concat_title_excerpt, axis=1)

test_tokens = [tokenize(t) for t in test_df["text"]]

test_df["predicted_category"] = [predict_tokens(toks) for toks in test_tokens]


# test_df
# test_tokens


# In[33]:


test_df


# * Calculate Accuracy 

# In[40]:


y_true = test_df["category"].to_numpy()
y_pred = test_df["predicted_category"].to_numpy()

# sanity check
assert len(y_true) == len(y_pred), (len(y_true), len(y_pred))

accuracy = (y_true == y_pred).mean()
correct = (y_true == y_pred).sum()
total = len(y_true)

print(f"Test accuracy: {accuracy:.4f}  ({correct}/{total})")


# # Naive Bayes SciKit Learn

# In[35]:


from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Hugging Face splits
ds = load_dataset("okite97/news-data")
train_df = ds["train"].to_pandas().rename(columns=str.lower)
test_df  = ds["test"].to_pandas().rename(columns=str.lower)

# Build `text` = title + excerpt
for df in (train_df, test_df):
    df["title"] = df.get("title", "").fillna("").astype(str)
    df["excerpt"] = df.get("excerpt", "").fillna("").astype(str)
    df["text"] = (df["title"] + " " + df["excerpt"]).str.strip()

y_train = train_df["category"].astype(str).to_numpy()
y_test  = test_df["category"].astype(str).to_numpy()


# In[36]:


# Vectorize & train NB
vectorizer = CountVectorizer(lowercase=True, token_pattern=r"[A-Za-z]+")
X_train = vectorizer.fit_transform(train_df["text"])
X_test  = vectorizer.transform(test_df["text"])

clf = MultinomialNB(alpha=1.0)
clf.fit(X_train, y_train)


# In[37]:


# Predict & evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Scikit-learn NB accuracy: {acc:.4f}  ({(y_pred==y_test).sum()}/{len(y_test)})")

labels = sorted(pd.unique(y_train))
cm = confusion_matrix(y_test, y_pred, labels=labels)
print(pd.DataFrame(cm, index=[f"true:{c}" for c in labels],
                      columns=[f"pred:{c}" for c in labels]))
print("\nClassification report:\n", classification_report(y_test, y_pred, labels=labels, zero_division=0))

