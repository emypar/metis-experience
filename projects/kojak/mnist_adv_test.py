
# coding: utf-8

# In[ ]:


import sys
import os
import time
import itertools
import re
import pprint
import pickle
import argparse


# In[ ]:


from IPython.display import clear_output


# In[ ]:


import numpy as np


# In[ ]:


import matplotlib.pyplot as plt
try:
    get_ipython().magic('matplotlib inline')
    plot_ok = True
except (NameError) as e:
    print("{}: Running without plotting".format(e), file=sys.stderr)
    plot_ok = False


# In[ ]:


DATA_FILE = 'data/mnist/adv_set-0.100-0.400-0-60000-0-10000.npz'
USE_P = .1
TRAIN_ADV_P = 0
TEST_DISPLAY = False
MNIST_SHAPE = 28, 28
FIGSIZE = 4
LOG_DIR = 'logs'


# In[ ]:


import noise_utils
import pil_utils


# In[ ]:


def flatten_x(X):
    return np.array([img.flatten() for img in X])

def build_y(Y):
    return np.argwhere(Y)[:,1]

def plot_x(x, y=None, x_adv=None):    
    if x_adv is not None:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(FIGSIZE*2, FIGSIZE))
        axs = axs.flatten()
        pil_utils.plot_image(x.reshape(MNIST_SHAPE), title='{} - Org'.format(y), ax=axs[0])
        pil_utils.plot_image(x_adv.reshape(MNIST_SHAPE), title='{} - Adv'.format(y), ax=axs[1])
    else:
        pil_utils.plot_image(x.reshape(MNIST_SHAPE), title=y)
    plt.show()


# In[ ]:


def get_pct_data(x, y, p=1):
    if p < 0 or p >= 1.:
        return x, y
    n = len(x)
    size = int(n * p)
    index = np.random.choice(np.arange(n), size, replace=False)
    return x[index], y[index]    


def load_data(data_file=DATA_FILE, use_p=USE_P, train_adv_p=TRAIN_ADV_P):
    npzfile = np.load(data_file)
    X_train = flatten_x(npzfile['X_train'])
    Y_train = build_y(npzfile['Y_train'])
    X_test = flatten_x(npzfile['X_test'])
    Y_test = build_y(npzfile['Y_test'])
    X_train_adv = flatten_x(npzfile['X_train_adv'])
    X_test_adv = flatten_x(npzfile['X_test_adv'])
    accuracy_bbox, accuracy_bbox_adv = npzfile['accuracies']
    lmbda, eps = npzfile['flags']

    if plot_ok and TEST_DISPLAY:
        for i in np.random.choice(np.arange(len(X_train)), 100, replace=False):
            clear_output()
            plot_x(X_train[i], Y_train[i], X_train_adv[i])
            time.sleep(2)

    if plot_ok and TEST_DISPLAY:
        for i in np.random.choice(np.arange(len(X_test)), 100, replace=False):
            clear_output()
            plot_x(X_test[i], Y_test[i], X_test_adv[i])
            time.sleep(2)

    if 0 < use_p and use_p < 1.:
        from sklearn.model_selection import             StratifiedShuffleSplit

        sss = StratifiedShuffleSplit(n_splits=1, test_size=use_p, random_state=19590209)
        for _, train_index in sss.split(X_train, Y_train):
            x_train, y_train, x_train_adv =                 X_train[train_index], Y_train[train_index], X_train_adv[train_index]
            break

        for _, test_index in sss.split(X_test, Y_test):   
            x_test, y_test, x_test_adv =                 X_test[test_index], Y_test[test_index], X_test_adv[test_index]
            break
    else:
        x_train, y_train, x_train_adv = X_train, Y_train, X_train_adv
        x_test, y_test, x_test_adv =             X_test, Y_test, X_test_adv
    print("x_train={}, x_test={}".format(x_train.shape, x_test.shape))

    if train_adv_p > 0:
        print("Adding {:.02f}% adv to training".format(train_adv_p * 100))
        xtra_train_x, xtra_train_y = get_pct_data(x_train_adv, y_train, train_adv_p)
        x_train = np.vstack((x_train, xtra_train_x))
        y_train = np.hstack((y_train, xtra_train_y))
        print("x_train={}, y_train={}".format(x_train.shape, y_train.shape))
        sys.stdout.flush()
        
    return x_train, y_train, x_test, y_test, x_test_adv


# In[ ]:


from sklearn.metrics import     accuracy_score, confusion_matrix

def make_noise_set(x, noiser, **noise_args):
    return np.array([noiser(flatten_img.reshape(MNIST_SHAPE), **noise_args).flatten()
                     for flatten_img in x])

def get_noiser_info(noiser, noise_args):
    info = getattr(noiser, '__name__', 'None')
    if noiser and noise_args:
        args = '('
        for k, v in sorted(noise_args.items()):
            if args != '(':
                args += ', '
            args += '{}={!r}'.format(k, v)
        args += ')'
        info += args
    return info


def try_one(estimator, 
            x_train, y_train,
            x_test, y_test, x_test_adv=None,
            train_noise_pct=0, 
            test_with_noise=True, 
            noiser=None, noise_args={}):
    
    use_noiser = False

    if noiser is not None and train_noise_pct:
        use_noiser = True
        x_train_noise, y_train_noise = get_pct_data(x_train, y_train, train_noise_pct)
        try_x_train = np.vstack((x_train, make_noise_set(x_train_noise, noiser, **noise_args)))
        y_train = np.hstack((y_train, y_train_noise))                       
    else:
        try_x_train = x_train

    if noiser is not None and test_with_noise:
        use_noiser = True
        try_x_test =  make_noise_set(x_test, noiser, **noise_args)
        if x_test_adv is not None:
            try_x_test_adv = make_noise_set(x_test_adv, noiser, **noise_args)
        else:
            try_x_test_adv = None
    else:
        try_x_test = x_test
        try_x_test_adv = x_test_adv    

    estimator.fit(try_x_train, y_train)
    y_pred = estimator.predict(try_x_test)
    test_score = accuracy_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    if try_x_test_adv is not None:
        y_pred_adv = estimator.predict(try_x_test_adv)
        test_score_adv = accuracy_score(y_test, y_pred_adv)
        confusion_mat_adv = confusion_matrix(y_test, y_pred_adv)
    else:
        test_score_adv = None
        confusion_mat_adv = None
        
    return (
        test_score, test_score_adv, 
        confusion_mat, confusion_mat_adv,
        get_noiser_info(noiser if use_noiser else None, noise_args)
    )


# In[ ]:


PR_SEQ = np.random.randint(0, 255, size=MNIST_SHAPE[0]*MNIST_SHAPE[1], dtype=np.uint8)

def apply_pr(x):
    return (((x*255).astype(np.uint8) ^ PR_SEQ).astype(np.float32) / 255).clip(0, 1)

def map_pr(X):
    return np.array([apply_pr(x) for x in X])


# In[ ]:


def try_estimator(
        x_train, y_train, x_test, y_test, x_test_adv,
        estimator_class, hyperparams={},
        noises=None, train_noise_pct=.1
    ):
    print(estimator_class.__name__)
    results = []
    for noiser, noise_args in noises if noises else [(None, None)]:
        for train_with_noise, test_with_noise in itertools.product([0, 1], [False, True]): 
            if noiser is None and (train_with_noise or test_with_noise):
                continue
            if noiser is not None and not train_with_noise and not test_with_noise:
                continue
            start_time = time.time()
            estimator = estimator_class(**hyperparams)
            test_score, test_score_adv, confusion_mat, confusion_mat_adv, noiser_info =                 try_one(estimator, 
                        x_train, y_train, 
                        x_test, y_test, x_test_adv,
                        train_noise_pct=train_with_noise*train_noise_pct, 
                        test_with_noise=test_with_noise,
                        noiser=noiser, noise_args=noise_args)
            end_time = time.time()
            estimator_info = estimator.__class__.__name__ # re.sub(r'[\n\s]+', ' ', repr(estimator)).strip()
            result = {}
            result['estimator'] = estimator_info
            result['noise'] = noiser_info
            result['train_noise_pct'] = train_with_noise*train_noise_pct
            result['test_with_noise'] = test_with_noise 
            result['test_score'] = test_score
            result['test_score_adv'] = test_score_adv
            pprint.pprint(result)
            result['confusion_mat'] = confusion_mat
            result['confusion_mat_adv'] = confusion_mat_adv
            results.append(result)
            print("Completed in {:.06f} sec".format(end_time - start_time))
            print()
            sys.stdout.flush()
    return results, end_time

def save_results(results, ts=None):
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    pkl_file = time.strftime(
        '{}/mnist_adv_test-%Y-%m-%d-%H-%M-%S-{}-{}.pkl'.format(
            LOG_DIR, os.uname().nodename, os.getpid()
        ),
        time.localtime(ts)
    )
    with open(pkl_file, 'wb') as f :
        pickle.Pickler(f).dump(results)
    return pkl_file


# In[1]:


def main(argv=None):
    from sklearn.neighbors import         KNeighborsClassifier

    from sklearn.svm import         SVC

    from sklearn.linear_model import         LogisticRegression

    from mnist_keras_cnn import         MNIST_K_CNN

    estimators = [
        (
            KNeighborsClassifier, 
            dict(
                algorithm='kd_tree', leaf_size=30, metric='euclidean',
                metric_params=None, n_jobs=8, n_neighbors=5, p=2,
                weights='distance'
            )
        ),

        (
            SVC,
            dict(
                C=100.0, kernel='rbf',
            )
        ),

        (
            LogisticRegression,
            dict(
                C=1.27, class_weight='balanced', max_iter=1000,
                multi_class='ovr', penalty='l1', solver='liblinear',
            )
        ),

        (
            MNIST_K_CNN,
            dict(
                epochs=5, verbose=1
            )
        ),
    ]

    noises = [
        (None, None),

        #(noise_utils.white_noise, dict(sigma=.01)),
        #(noise_utils.white_noise, dict(sigma=.02)),
        (noise_utils.white_noise, dict(sigma=.05)),
        (noise_utils.white_noise, dict(sigma=.1)),
        (noise_utils.white_noise, dict(sigma=.2)),
        #(noise_utils.white_noise, dict(sigma=.5)),

        (pil_utils.gaussian_blur_filter, dict(radius=2)),
        #(pil_utils.gaussian_blur_filter, dict(radius=5)),

        (pil_utils.smooth_filter, {}),
        #(pil_utils.smooth_more_filter, {}),

        #(pil_utils.rank_filter, {}),
        #(pil_utils.median_filter, {}),
        #(pil_utils.min_filter, {}),
        (pil_utils.max_filter, {}),     

        #(pil_utils.kernel_filter, dict(size=(3, 3), kernel=[.1, .2, .1, .2, .8, .2, .1, .2, .1])),
    ]    
    
    parser = argparse.ArgumentParser(description='MNIST Adv Test.')
    
    parser.add_argument('-s', '--selector', type=int, default=0)
    parser.add_argument('-p', '--use-p', type=float, default=0)
    parser.add_argument('-n', '--train-noise-pct', type=float, default=0.1)
    parser.add_argument('-a', '--train-adv-p', type=float, default=0)
    parser.add_argument('-x', '--xor-pr', action='store_true', default=False)
    parser.add_argument('data_file', default=DATA_FILE, nargs='?')
    args = parser.parse_args(argv)
    
    estimator_class, hyperparams = estimators[args.selector]

    x_train, y_train, x_test, y_test, x_test_adv =         load_data(
            data_file=args.data_file,
            use_p=args.use_p, 
            train_adv_p=args.train_adv_p
        )    
    
    if args.xor_pr:
        x_train = map_pr(x_train)
        x_test = map_pr(x_test)
        x_test_adv = map_pr(x_test_adv)
    
    results, end_time = try_estimator(
        x_train, y_train, x_test, y_test, x_test_adv,
        estimator_class, hyperparams=hyperparams,
        noises=None if (args.xor_pr or args.train_adv_p) else noises,
        train_noise_pct=args.train_noise_pct
    )
    
    # Suplement the global flags:
    for result in results:
        result['xor_pr'] = args.xor_pr
        result['train_adv_p'] = args.train_adv_p
    
    pkl_file = save_results(results, end_time)
    print("Results saved to {}".format(pkl_file))


# In[ ]:


if __name__ == '__main__':
    main(['-s', '3', '-p', '.3', DATA_FILE] if plot_ok else None)

