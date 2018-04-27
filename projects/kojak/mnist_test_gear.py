
# coding: utf-8

# **Test Gear MNIST Classifier grid search with CV**

# In[1]:


import sys
import os
import time


# In[2]:


import numpy as np
import pandas as pd


# In[3]:


import matplotlib.pyplot as plt

try:
    get_ipython().magic('matplotlib inline')
    plot_ok = True
except (NameError) as e:
    print("{}: Running without plotting".format(e))
    plot_ok = False


# In[22]:


import mnist_utils
import pil_utils
from collections import defaultdict
import os.path

DATA_FILE = 'data/mnist/adv_set-0.100-0.400-0-60000-0-10000.npz'
MNIST_SHAPE = 28, 28

def plot_flatten_img(img, title=None, ax=None):
    if plot_ok:
        pil_utils.plot_image(img.reshape(MNIST_SHAPE), title=title, ax=ax)

def make_label_to_index_map(label_list):
    label_to_index_map = defaultdict(list)
    for i, l in enumerate(label_list):
        label_to_index_map[l].append(i)
    return label_to_index_map

def reduce_index_list(label_to_index_map, p=.1):
    for l in label_to_index_map:
        n = int(len(label_to_index_map[l])*p)
        label_to_index_map[l] = np.random.choice(label_to_index_map[l], size=n, replace=False)
    return label_to_index_map

def build_reduced_set(X, y, p=.1):
    label_to_index_map = make_label_to_index_map(y)
    reduce_index_list(label_to_index_map, p=p)
    index_list = np.concatenate(tuple(label_to_index_map.values()))
    return X[index_list], y[index_list]

def flatten_x(X):
    return np.array([img.flatten() for img in X])

def build_y(Y):
    return np.argwhere(Y)[:,1]

def get_mnist_sets(p=0):
    npzfile = np.load(DATA_FILE)
    train_imgs = flatten_x(npzfile['X_train'])
    train_labels = build_y(npzfile['Y_train'])
    test_imgs = flatten_x(npzfile['X_test'])
    test_labels = build_y(npzfile['Y_test'])
    if 0 < p and p < 1:
        X_train, y_train = build_reduced_set(train_imgs, train_labels, p)
        X_test, y_test = build_reduced_set(test_imgs, test_labels, p)
    else:
        X_train, y_train = train_imgs, train_labels
        X_test, y_test = test_imgs, test_labels
    print("train size={}, test size={}".format(len(y_train), len(y_test)))
    return X_train, y_train, X_test, y_test


# In[11]:


from sklearn.metrics import     accuracy_score
    
from sklearn.model_selection import     GridSearchCV

from sklearn.decomposition import     PCA

def run_search(estimator, parameters, p_set=0, 
               pca_n_components=None, **kwargs):
    X_train, y_train, X_test, y_test = get_mnist_sets(p_set)
    
    if pca_n_components:
        pca = PCA(n_components=pca_n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        print("PCA n_components={}".format(pca_n_components))

    gscv = GridSearchCV(estimator, parameters, **kwargs)
    
    start_time = time.time()
    gscv.fit(X_train, y_train)
    end_time = time.time()
    
    best_params = gscv.best_params_
    best_estimator = gscv.best_estimator_
    best_cv_score = gscv.best_score_

    predict = best_estimator.predict(X_test)
    test_score = accuracy_score(y_test, predict)

    wrong_test_index = list(np.where (y_test != predict)[0])

    log_dir = 'logs'
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    prefix = time.strftime('{}-%Y-%d-%m-%H-%M-%S-{}-{}'.format(
        estimator.__class__.__name__,
        os.uname().nodename,
        os.getpid()
    ))
    
    log_file = os.path.join(log_dir, '{}.log'.format(prefix))
    results_file = os.path.join(log_dir, '{}-results.csv'.format(prefix))
    
    dtime = end_time - start_time
    n_days = int(dtime) // (3600*24)
    n_hours = (int(dtime) % (3600*24)) // 3600
    n_min = (int(dtime) % 3600) // 60
    n_sec = dtime % 60

    with open(log_file, 'wt') as f:
        print("# Completed in {}:{:02d}:{:02d}:{:.06f}".format(n_days, n_hours, n_min, n_sec), file=f)
        print("# train size={}, test size={}, PCA n_components={}".format(
            len(y_train), len(y_test), pca_n_components), file=f)
        print('best_estimator = {}'.format(best_estimator), file=f)
        print('best_params= {} '.format(best_params), file=f)
        print('best_cv_score = {}'.format(best_cv_score), file=f)
        print('test_score = {}'.format(test_score), file=f)
        n_step = 10
        for k in range(0, len(wrong_test_index), n_step):
            print(("wrong_test_index =" if k == 0 else " +") + " \\", sep='', file=f)
            print(" ", wrong_test_index[k:k+n_step], end='', file=f)
        print(file=f)
        print
    print("Log saved to {} file".format(log_file))
    pd.DataFrame(gscv.cv_results_).to_csv(results_file)
    print("Results saved to {} file".format(results_file))
    return gscv

