
# coding: utf-8

# ** Determine best hyperparameters for MNIST KNN Classifier using grid search w/ CV**

# In[ ]:


import sys
import os


# In[ ]:


from  mnist_test_gear import run_search

if __name__ == '__main__':
    from sklearn.neighbors import         KNeighborsClassifier

    parameters_list = [
        dict(
            n_neighbors=[5, 10, 20, 50, 100],
            weights=['uniform', 'distance'],
            algorithm=['ball_tree'],
            metric=['euclidean', 'manhattan'],
        ),
        
        dict(
            n_neighbors=[5, 10, 20, 50, 100],
            weights=['uniform', 'distance'],
            algorithm=['kd_tree'],
            metric=['euclidean', 'manhattan'],
        ),
        
        dict(
            n_neighbors=[5, 10, 20, 50, 100],
            weights=['uniform', 'distance'],
            algorithm=['brute'],
            metric=['euclidean', 'manhattan'],
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
    knn = KNeighborsClassifier(n_jobs=8)
    run_search(knn, parameters, verbose=2)

