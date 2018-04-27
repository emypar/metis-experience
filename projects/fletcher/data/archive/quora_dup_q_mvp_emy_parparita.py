
# coding: utf-8

# **Topic:** Kaggle Challenge [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs/data)
# 
# Which of the provided pairs of questions contain two questions with the same meaning? The ground truth is the set of labels that have been supplied by human experts. The ground truth labels are inherently subjective, as the true meaning of sentences can never be known with certainty. Human labeling is also a 'noisy' process, and reasonable people will disagree. As a result, the ground truth labels on this dataset should be taken to be 'informed' but not 100% accurate, and may include incorrect labeling. The labels, on the whole, represent a reasonable consensus, but this may often not be true on a case by case basis for individual items in the dataset.
# 

# **Scores:**
# 
#  - log_loss: official competition score
#  - accuracy: relevant in real life

# **General workflow:**
# 
# - load train data set
# - cleanup (NaN's, non-English)
# - tokenize
# - perform semantic indexing/analysis
# - compute similarity scores per pair
# - train a classification algorithm for score -> is duplicate prediction
# - compute scores for the test set

# In[1]:


import sys
import os

# logging for gensim (set to INFO)
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[2]:


import numpy as np
import pandas as pd


# In[3]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set()


# In[4]:


# sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
import sklearn.metrics.pairwise as smp
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import NMF


# In[5]:


# gensim
from gensim import corpora, models, similarities, matutils,     __version__ as gensim_version
gensim_version


# In[6]:


# Load the cleaned train set:
clean_file = './data/quora/clean.csv.gz'
df = pd.read_csv(clean_file, header=0)
df.shape


# In[7]:


from sklearn.utils import shuffle

ratio_1 = df['is_duplicate'].mean()
subset_size = 100000

if subset_size > 0:
    # Use a smaller sub-set to speed up the process; preserve the ratio
    # of is_duplicate:
    n_samples = min(subset_size, df.shape[0]-1)

    n_samples_1 = int(ratio_1 * n_samples)
    use_df = pd.concat([
            shuffle(df[df['is_duplicate'] == 0], n_samples=(n_samples - n_samples_1)),
            shuffle(df[df['is_duplicate'] == 1], n_samples=n_samples_1)
        ],
        ignore_index=True
    )
else:
    use_df = df

use_df.shape


# Extract the questions and the target flag:

# In[8]:


q1 = use_df['question1'].values
q2 = use_df['question2'].values
is_dup = use_df['is_duplicate'].values


# Minimal EDA, look for duplicate questions:

# In[9]:


q1_unique, q1_unique_counts =     np.unique(q1, return_index=False, return_inverse=False, return_counts=True)


# In[10]:


q1_num_dups, q1_dups_dist =     np.unique(q1_unique_counts, return_index=False, return_inverse=False, return_counts=True)


# In[11]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
#xticks = np.arange(q1_num_dups.min(), q1_num_dups.max()+1)

ax[0].bar(q1_num_dups, q1_dups_dist)
#ax[0].set_xticks(xticks)
ax[0].set_title("Duplicate Q1 Distribution")
ax[0].set_xlabel("# Duplicates")
ax[0].set_ylabel("Count")

ax[1].bar(q1_num_dups, q1_dups_dist, log=True)
#ax[1].set_xticks(xticks)
ax[1].set_title("Duplicate Q1 Distribution (Log Scale)")
ax[1].set_xlabel("# Duplicates")
ax[1].set_ylabel("Count")

plt.show()


# Split 80/20 train/test:

# In[12]:


q1_train, q1_test,     q2_train, q2_test,     is_dup_train, is_dup_test =         train_test_split(q1, q2, is_dup, test_size=.2, stratify=is_dup)


# Vectorize the text; use both q1 and q2 (why? TBD):

# In[13]:


tfidf = TfidfVectorizer(
    stop_words="english", 
    token_pattern=r'(?i)\b[a-z]{2,}\b', #words with >= 2 alpha chars 
    min_df=2
)

tfidf.fit(np.concatenate((q1_train, q2_train)))
id2word = dict((v, k) for k, v in tfidf.vocabulary_.items())

q1_train_tfidf = tfidf.transform(q1_train)
q1_train_corpus = matutils.Sparse2Corpus(q1_train_tfidf.transpose())

q2_train_tfidf = tfidf.transform(q2_train)
q2_train_corpus = matutils.Sparse2Corpus(q2_train_tfidf.transpose())

q1_test_tfidf = tfidf.transform(q1_test)
q1_test_corpus = matutils.Sparse2Corpus(q1_test_tfidf.transpose())

q2_test_tfidf = tfidf.transform(q2_test)
q2_test_corpus = matutils.Sparse2Corpus(q2_test_tfidf.transpose())


# Build LSI model on based on q1:

# In[14]:


num_topics = 500
lsi = models.LsiModel(q1_train_corpus, id2word=id2word, num_topics=num_topics)


# In[15]:


fig = plt.figure()
ax = plt.gca()
topic_importance = lsi.projection.s
ax.plot(np.arange(0, len(topic_importance)), topic_importance)
ax.set_title('Topic Importance')
ax.set_xlabel('Topic#')
ax.set_ylabel('Importance')
plt.show()


# Transform q1, q2 into LSI space: 

# In[16]:


q1_train_lsi = lsi[q1_train_corpus]
q2_train_lsi = lsi[q2_train_corpus]

q1_test_lsi = lsi[q1_test_corpus]
q2_test_lsi = lsi[q2_test_corpus]


# Iterate through the 2 lists and compute the cosine similarity:

# In[17]:


from  gensim.matutils import cossim

def cos_sim(v1, v2):
    if not v1 or not v2:
        return np.nan
    return cossim(v1, v2)

def cos_sim_list(q1_lsi, q2_lsi):
    return np.array(list(map(lambda s: cos_sim(*s), zip(q1_lsi, q2_lsi))))


# In[18]:


sim12_train = cos_sim_list(q1_train_lsi, q2_train_lsi)
sim12_test = cos_sim_list(q1_test_lsi, q2_test_lsi)


# Use K Nearest Neighbors to map similarities into is_dup labels:

# In[19]:


# Beware of NaN's:
train_index = ~np.isnan(sim12_train)
X_train = sim12_train[train_index].reshape(-1, 1)
y_train = is_dup_train[train_index]


# In[20]:


from sklearn.neighbors import KNeighborsClassifier

knc = KNeighborsClassifier(n_neighbors=500, p=1)
knc.fit(X_train, y_train)


# Compute the score:

# In[21]:


from sklearn.metrics import log_loss, accuracy_score


# In[22]:


# Baseline score, assume random guessing with ratio_1 probability for 1:
baseline_accuracy_score = accuracy_score(
    is_dup, 
    np.random.binomial(1, ratio_1, size=len(is_dup))
)
baseline_log_loss_score = log_loss(
    is_dup,
    np.full_like(is_dup, ratio_1, dtype=np.float64)
)


# In[23]:


# Beware of NaN's: predict where sim values exists, replace with the observed
# probability where they don't:
test_index = ~np.isnan(sim12_test)
n_fill = len(test_index) - np.count_nonzero(test_index)
X_test = sim12_test[test_index].reshape(-1, 1)

y_pred_test = np.zeros_like(is_dup_test)
y_pred_test[test_index] = knc.predict(X_test)
y_pred_test[~test_index] = np.random.binomial(1, ratio_1, size=n_fill)

y_pred_proba_test = np.full_like(is_dup_test, ratio_1, dtype=np.float64)
y_pred_proba_test[test_index] = knc.predict_proba(X_test)[:,1]


# In[24]:


print("Accuracy test score = {:.03f}, baseline = {:.03f}, higher is better".format(
    accuracy_score(is_dup_test, y_pred_test),
    baseline_accuracy_score
))
print("Log loss test score = {:.03f}, baseline = {:.03f}, lower is better".format(
    log_loss(is_dup_test, y_pred_proba_test),
    baseline_log_loss_score
))


# In[25]:


def wrong_pred(i, for_set='test', f=None):
    q1, q2, is_dup, pred, sim12 =         (q1_test, q2_test, is_dup_test, y_pred_test, sim12_test) if for_set == 'test'        else         (q1_train, q2_train, is_dup_train, y_pred_train, sim12_train)     
    print("Q1: {!r}".format(q1[i]), file=f)
    print("Q2: {!r}".format(q2[i]), file=f)
    print("is_dup={}, pred={}, sim={:.03f}".format(
            is_dup[i], pred[i], sim12[i]
        ),
        file=f  
    )
        


# In[26]:


wrong_test_index = is_dup_test != y_pred_test
wrong_pred_test_index = np.argwhere(wrong_test_index & test_index).flatten()
wrong_guess_test_index = np.argwhere(wrong_test_index & ~test_index).flatten()


# In[29]:


wrong_pred(wrong_pred_test_index[8])

