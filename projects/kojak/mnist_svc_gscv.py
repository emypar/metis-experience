
# coding: utf-8

# ** Determine best hyperparameters for MNIST SVC Classifier using grid search w/ CV**

# In[ ]:


import sys
import os


# In[ ]:


import numpy as np


# In[ ]:


from  mnist_test_gear import run_search

if __name__ == '__main__':
    from sklearn.svm import         SVC

    parameters_list = [
        dict(
            C=np.logspace(-2, 2, 20),
            kernel=['rbf']
        ), 

        dict(
            C=np.logspace(-2, 2, 20),
            kernel=['poly'],
            degree=[3, 5, 8], # 10]
        ),
    ]
    
    max_n_sel = 2*len(parameters_list)-1
    n_sel = sys.argv[1] if len(sys.argv) > 1 else ''
    while True:
        try:
            n_sel = int(n_sel.strip())
            if 0 <= n_sel and n_sel <= max_n_sel:
                parameters = parameters_list[n_sel % 2]
                pca_n_components = 620 if n_sel >= 2 else 0
                break     
        except (ValueError, TypeError):
            pass
        n_sel = input('Params set (0..{}): '.format(max_n_sel))
    svc = SVC()
    run_search(svc, parameters,
               pca_n_components=pca_n_components, n_jobs=8, verbose=2)

