
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
import time
import gzip
import re

# logging for gensim (set to INFO)
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[2]:


import numpy as np
import pandas as pd


# In[3]:


try:
    get_ipython().run_line_magic('matplotlib', 'inline')
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
except (NameError) as e:
    print("{}: No matplotlib graphics".format(e))
    plt = None


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

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

from sklearn.metrics import log_loss, accuracy_score


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
subset_size = -1 #1000

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


if plt:
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


import nltk
stemmer = nltk.stem.porter.PorterStemmer().stem

def tokenizer(doc, min_len=2):
    # Normalize case:
    doc = doc.lower()
    # Separate on non-alpha:
    tokens = re.split(r'[^a-z0-9]+', doc)
    # Return all the tokens len >= min_len, stemmed:
    words = filter(lambda t: len(t) >= min_len, tokens)
    if stemmer:
        words = map(stemmer, words)
    return list(words)


# In[14]:


tfidf = TfidfVectorizer(
    stop_words="english", 
    token_pattern=r'(?i)\b[a-z]{2,}\b', #words with >= 2 alpha chars 
    tokenizer=tokenizer,
    min_df=2
)

tfidf.fit(np.concatenate((q1_train, q2_train)))
id2word = dict((v, k) for k, v in tfidf.vocabulary_.items())

q1_train_tfidf = tfidf.transform(q1_train)
q2_train_tfidf = tfidf.transform(q2_train)
q1_test_tfidf = tfidf.transform(q1_test)
q2_test_tfidf = tfidf.transform(q2_test)


# In[15]:


word_vectorizer = repr(tfidf) + ' + custom tokenizer + stemmer'


# Build LSI model on based on q1:

# In[16]:


q1_train_corpus = matutils.Sparse2Corpus(q1_train_tfidf.transpose())
q2_train_corpus = matutils.Sparse2Corpus(q2_train_tfidf.transpose())
q1_test_corpus = matutils.Sparse2Corpus(q1_test_tfidf.transpose())
q2_test_corpus = matutils.Sparse2Corpus(q2_test_tfidf.transpose())


# In[17]:


num_topics = 500
lsi = models.LsiModel(q1_train_corpus, id2word=id2word, num_topics=num_topics)


# In[18]:


nlp_model = 'LSI(num_topics={})'.format(num_topics)


# In[20]:


if plt:
    fig = plt.figure()
    ax = plt.gca()
    topic_importance = lsi.projection.s
    ax.plot(np.arange(0, len(topic_importance)), topic_importance)
    ax.set_title('Topic Importance')
    ax.set_xlabel('Topic#')
    ax.set_ylabel('Importance')
    plt.show()


# Transform q1, q2 into LSI space: 

# In[21]:


q1_train_lsi = lsi[q1_train_corpus]
q2_train_lsi = lsi[q2_train_corpus]

q1_test_lsi = lsi[q1_test_corpus]
q2_test_lsi = lsi[q2_test_corpus]


# In[24]:


q1_train_lsi.corpus.sparse.T


# In[23]:


from scipy.sparse import hstack

hstack((q1_train_lsi.corpus.sparse.T, q2_train_lsi.corpus.sparse.T))


# LR Topics Space:

# In[26]:


from scipy.sparse import hstack


# In[27]:


X_train_lsi = hstack((q1_train_lsi.corpus.sparse.T, q2_train_lsi.corpus.sparse.T))
y_train_lsi = is_dup_train


# In[28]:


lr_lsi = LogisticRegressionCV(
    penalty='l1',
    dual=False,
    Cs=10,
    cv=3,
    class_weight='balanced',
    random_state=19590209,
    solver='saga',
    n_jobs=8,
    #max_iter=1000
)

lr_lsi.fit(X_train_lsi, y_train_lsi)


# In[29]:


classifier = repr(lr_lsi)


# Compute the score:

# In[30]:


# Baseline score, assume random guessing with ratio_1 probability for 1:
accuracy_baseline = accuracy_score(
    is_dup, 
    np.random.binomial(1, ratio_1, size=len(is_dup))
)
logloss_baseline = log_loss(
    is_dup,
    np.full_like(is_dup, ratio_1, dtype=np.float64)
)


# In[31]:


X_test_lsi = hstack((q1_test_lsi.corpus.sparse.T, q2_test_lsi.corpus.sparse.T))
y_test_lsi = is_dup_test
n_pred_lsi = len(y_test_lsi)

y_pred_lsi = lr_lsi.predict(X_test_lsi)
accuracy_lsi = accuracy_score(y_test_lsi, y_pred_lsi)

y_pred_proba_lsi = lr_lsi.predict_proba(X_test_lsi)
logloss_lsi = log_loss(y_test_lsi, y_pred_proba_lsi)


# In[32]:


wrong_pred_index_lsi = np.argwhere(y_test_lsi != y_pred_lsi).flatten()


# In[33]:


def print_results(accuracy, logloss, wrong_pred_index, 
                  n_pred, for_set='test', f=None):
    print("Accuracy {} score = {:.03f}, baseline = {:.03f}, higher is better".format(
            for_set,
            accuracy,
            accuracy_baseline),
        file=f
    )
    if logloss is not None:
        print("Log loss {} score = {:.03f}, baseline = {:.03f}, lower is better".format(
            for_set,
            logloss,
            logloss_baseline),
        file=f
    )
    print("Wrong {} predictions: {}/{} ({:.03f})".format(
            for_set,
            len(wrong_pred_index), n_pred,
            len(wrong_pred_index)/n_pred if n_pred else 0),
        file=f
    )
    print(file=f)
          
def wrong_pred(i, q1, q2, y_true, y_pred, f=None):
    print("Q1: {!r}".format(q1[i]), file=f)
    print("Q2: {!r}".format(q2[i]), file=f)
    print("is_dup={}, pred={}".format(
            y_true[i], y_pred[i]
        ),
        file=f  
    )
    print(file=f)
    
def print_mistakes(q1, q2, is_dup, y_pred,
               wrong_pred_index,
               for_set='test', f=None):
    print("Wrong {} predictions:".format(for_set), file=f)
    print(file=f)
    for i in wrong_pred_index:
        wrong_pred(i, q1, q2, is_dup, y_pred, f)
    print(file=f)
       
def audit_info(train_sz=None, test_sz=None, 
               word_vectorizer=None, nlp_model=None, classifier=None, f=None):
    print("Time:       {}".format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(report_time))
        ),
        file=f
    )
    print("Train/Test: {}/{}".format(train_sz, test_sz), file=f)
    print("Vectorizer: {}".format(word_vectorizer), file=f)
    print("NLP Model:  {}".format(nlp_model), file=f)
    print("Classifier: {}".format(classifier), file=f)
    print(file=f)


# In[34]:


report_time = time.time()


# In[35]:


print_results(accuracy_lsi, logloss_lsi, wrong_pred_index_lsi,
              n_pred_lsi)

audit_file = 'data/quora/audit-lsi-lr-{}-{}.log.gz'.format(
    time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(report_time)),
    os.getpid()
)
with gzip.open(audit_file, 'wt') as f:
    audit_info(
        len(is_dup_train), len(is_dup_test),
        word_vectorizer, nlp_model, classifier, f=f)
    print_results(accuracy_lsi, logloss_lsi, wrong_pred_index_lsi,
                  n_pred_lsi, f=f)
    print_mistakes(q1_test, q2_test, y_test_lsi, y_pred_lsi,
                   wrong_pred_index_lsi, f=f)

print("Audit file = {}".format(audit_file))

