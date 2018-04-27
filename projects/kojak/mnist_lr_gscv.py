
# coding: utf-8

# ** Determine best hyperparameters for MNIST LR Classifier using grid search w/ CV**

# In[ ]:


import sys
import os


# In[ ]:


import numpy as np


# In[ ]:


from  mnist_test_gear import run_search

if __name__ == '__main__':
    from sklearn.linear_model import         LogisticRegression
        
    parameters_list = [
        dict(
            C=np.logspace(-2, 2, 20),
            penalty=['l1', 'l2'],
            solver=['saga'],
            class_weight=['balanced', None],
        ), 
        
        dict(
            C=np.logspace(-2, 2, 20),
            penalty=['l1', 'l2'],
            solver=['liblinear'],
            class_weight=['balanced', None],
        ), 
        
        dict(
            C=np.logspace(-2, 2, 20),
            penalty=['l1', 'l2'],
            solver=['saga'],
            class_weight=['balanced', None],
            multi_class=['multinomial'],
        ), 
    ]

    max_n_sel = len(parameters_list)-1
    n_sel = sys.argv[1] if len(sys.argv) > 1 else ''
    while True:
        try:
            n_sel = int(n_sel.strip())
            if 0 <= n_sel and n_sel <= max_n_sel:
                parameters = parameters_list[n_sel]
                break     
        except (ValueError, TypeError):
            pass
        n_sel = input('Params set (0..{}): '.format(max_n_sel))

    lr = LogisticRegression(max_iter=1000, n_jobs=2)
    run_search(lr, parameters, n_jobs=4, verbose=2)

