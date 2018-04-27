
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
subset_size = -1 # 10000

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
    xticks = np.arange(q1_num_dups.min(), q1_num_dups.max()+1)

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


cvec = CountVectorizer(
    ngram_range=(1, 2), 
    stop_words="english", 
    token_pattern=r'(?i)\b[a-z]{2,}\b', #words with >= 2 alpha chars,
    tokenizer=tokenizer,
)

cvec.fit(np.concatenate((q1_train, q2_train)))
id2word = dict((v, k) for k, v in cvec.vocabulary_.items())

q1_train_tfidf = cvec.transform(q1_train)
q1_train_corpus = matutils.Sparse2Corpus(q1_train_tfidf.transpose())

q2_train_tfidf = cvec.transform(q2_train)
q2_train_corpus = matutils.Sparse2Corpus(q2_train_tfidf.transpose())

q1_test_tfidf = cvec.transform(q1_test)
q1_test_corpus = matutils.Sparse2Corpus(q1_test_tfidf.transpose())

q2_test_tfidf = cvec.transform(q2_test)
q2_test_corpus = matutils.Sparse2Corpus(q2_test_tfidf.transpose())


# In[15]:


word_vectorizer = repr(cvec) + ' + custom tokenizer + stemmer'


# Build LDA model on based on q1:

# In[16]:


num_topics = 200
lda = models.ldamulticore.LdaMulticore(
    q1_train_corpus, id2word=id2word, num_topics=num_topics,
    workers=7, passes=10
)


# In[17]:


nlp_model = 'LDA(num_topics={})'.format(num_topics)


# Transform q1, q2 into LDA space: 

# In[18]:


q1_train_lda = lda[q1_train_corpus]
q2_train_lda = lda[q2_train_corpus]

q1_test_lda = lda[q1_test_corpus]
q2_test_lda = lda[q2_test_corpus]


# LR In Topics Space:

# In[19]:


from scipy.sparse import hstack


# In[20]:


X_train_lda = hstack((q1_train_lda.corpus.sparse.T, q2_train_lda.corpus.sparse.T))
y_train_lda = is_dup_train


# In[21]:


n_train_lda = len(y_train_lda)


# In[22]:


lr_lda = LogisticRegressionCV(
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

lr_lda.fit(X_train_lda, y_train_lda)


# In[23]:


classifier = repr(lr_lda)


# Compute the score:

# In[24]:


# Baseline score, assume random guessing with ratio_1 probability for 1:
accuracy_baseline = accuracy_score(
    is_dup, 
    np.random.binomial(1, ratio_1, size=len(is_dup))
)
logloss_baseline = log_loss(
    is_dup,
    np.full_like(is_dup, ratio_1, dtype=np.float64)
)


# In[25]:


X_test_lda = hstack((q1_test_lda.corpus.sparse.T, q2_test_lda.corpus.sparse.T))
y_test_lda = is_dup_test
n_test_lda = len(y_test_lda)

y_pred_lda = lr_lda.predict(X_test_lda)
accuracy_lda = accuracy_score(y_test_lda, y_pred_lda)

y_pred_proba_lda = lr_lda.predict_proba(X_test_lda)
logloss_lda = log_loss(y_test_lda, y_pred_proba_lda)


# In[26]:


wrong_pred_index_lda = np.argwhere(y_test_lda != y_pred_lda).flatten()


# In[27]:


audit_dir = 'data/audit'
if not os.path.isdir(audit_dir):
    os.makedirs(audit_dir)


# In[28]:


def print_results(accuracy, logloss, wrong_pred_index,
                  wrong_guess_index=None, n_pred=0, 
                  n_guess=0, for_set='test', f=None):
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
    if wrong_pred_index is not None:
        print("Wrong {} predictions: {}/{} ({:.03f})".format(
                for_set,
                len(wrong_pred_index), n_pred,
                len(wrong_pred_index)/n_pred if n_pred else 0),
            file=f
        )
    if wrong_guess_index is not None:
        print("Wrong {} guesses: {}/{} ({:.03f})".format(
                for_set,
                len(wrong_guess_index), n_guess,
                len(wrong_guess_index)/n_guess if n_guess else 0),
            file=f
        )
    print(file=f)
          
def wrong_pred(i, q1, q2, is_dup, y_pred, sim12=None, f=None):
    print("Q1: {!r}".format(q1[i]), file=f)
    print("Q2: {!r}".format(q2[i]), file=f)
    print("is_dup={}, pred={}{}".format(
            is_dup[i], y_pred[i], 
            ', sim={:.03f}'.format(sim12[i]) if sim12 is not None else ''
        ),
        file=f  
    )
    print(file=f)
    
def print_mistakes(q1, q2, is_dup, y_pred, sim12=None,
               wrong_pred_index=None, wrong_guess_index=None,
               for_set='test', f=None):
    if wrong_pred_index is not None:
        print("Wrong {} predictions:".format(for_set), file=f)
        print(file=f)
        for i in wrong_pred_index:
            wrong_pred(i, q1, q2, is_dup, y_pred, sim12, f)
        print(file=f)
    if wrong_guess_index is not None:
        print("Wrong {} guesses:".format(for_set), file=f)
        print(file=f)
        for i in wrong_guess_index:
            wrong_pred(i, q1, q2, is_dup, y_pred, sim12, f)
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


# In[29]:


report_time = time.time()


# In[30]:


print_results(accuracy_lda, logloss_lda, wrong_pred_index_lda, n_pred=n_test_lda)

audit_file = '{}/audit-lda-lr-{}-{}.log.gz'.format(
    audit_dir,
    time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(report_time)),
    os.getpid()
)
with gzip.open(audit_file, 'wt') as f:
    audit_info(n_train_lda, n_test_lda, word_vectorizer, nlp_model, classifier, f=f)
    print_results(accuracy_lda, logloss_lda, wrong_pred_index_lda, n_pred=n_test_lda, f=f)
    print_mistakes(q1_test, q2_test, y_test_lda, y_pred_lda,
                   wrong_pred_index=wrong_pred_index_lda, f=f)
print("Audit file = {}".format(audit_file))


# Iterate through the 2 lists and compute the cosine similarity:

# In[31]:


from  gensim.matutils import cossim

def cos_sim(v1, v2):
    if not v1 or not v2:
        return np.nan
    return cossim(v1, v2)

def cos_sim_list(q1_lda, q2_lda):
    return np.array(list(map(lambda s: cos_sim(*s), zip(q1_lda, q2_lda))))


# In[32]:


sim12_train = cos_sim_list(q1_train_lda, q2_train_lda)
sim12_test = cos_sim_list(q1_test_lda, q2_test_lda)


# LR In Similarities Space:

# In[33]:


# Beware of NaN's:
train_index = ~np.isnan(sim12_train)
X_train_sim = sim12_train[train_index].reshape(-1, 1)
y_train_sim = is_dup_train[train_index]


# In[34]:


lr_lda_sim = LogisticRegressionCV(
    penalty='l1',
    dual=False,
    Cs=10,
    cv=3,
    class_weight='balanced',
    random_state=19590209,
    solver='saga',
    n_jobs=8,
)

lr_lda_sim.fit(X_train_sim, y_train_sim)


# In[35]:


classifier = repr(lr_lda_sim) + ' for sim score'


# Compute the score:

# In[36]:


# If sim score cannot be computed then guess w/ a Bernoulli distribution 
# w/ ratio_1 probabilty:
test_index_sim = ~np.isnan(sim12_test)

X_test_sim = sim12_test[test_index_sim].reshape(-1, 1)
y_test_sim = is_dup_test

n_pred_sim = np.count_nonzero(test_index_sim)
n_guess_sim = len(y_test_sim) - n_pred_sim

y_pred_sim = np.zeros_like(y_test_sim)
y_pred_sim[test_index_sim] = lr_lda_sim.predict(X_test_sim)
y_pred_sim[~test_index_sim] = np.random.binomial(1, ratio_1, size=n_guess_sim)
accuracy_sim = accuracy_score(y_test_sim, y_pred_sim)

y_pred_proba_sim = np.full_like(y_test_sim, ratio_1, dtype=np.float64)
y_pred_proba_sim[test_index_sim] = lr_lda_sim.predict_proba(X_test_sim)[:,1]
logloss_sim = log_loss(y_test_sim, y_pred_proba_sim)


# In[37]:


wrong_all_index_sim = y_test_sim != y_pred_sim
wrong_pred_index_sim = np.argwhere(wrong_all_index_sim & test_index_sim).flatten()
wrong_guess_index_sim = np.argwhere(wrong_all_index_sim & ~test_index_sim).flatten()


# In[38]:


report_time = time.time()


# In[39]:


print_results(accuracy_sim, logloss_sim, wrong_pred_index_sim,
        wrong_guess_index_sim, n_pred_sim, n_guess_sim)

audit_file = '{}/audit-lda-sim-lr-{}-{}.log.gz'.format(
    audit_dir,
    time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(report_time)),
    os.getpid()
)
with gzip.open(audit_file, 'wt') as f:
    audit_info(n_train_lda, n_test_lda, word_vectorizer, nlp_model, classifier, f=f)
    print_results(accuracy_sim, logloss_sim, wrong_pred_index_sim,
        wrong_guess_index_sim, n_pred_sim, n_guess_sim, f=f)
    print_mistakes(q1_test, q2_test, y_test_sim, y_pred_sim, sim12_test,
               wrong_pred_index_sim, wrong_guess_index_sim, f=f)

print("Audit file = {}".format(audit_file))

