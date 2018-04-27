#  Exploring Ways To Counter Adversarial Attacks Against Image Classifiers

**Project:** Kojak   
**Student:** Emy Parparita   

**Summary:**

This project is an empirical study of ways to counter adversarial attacks
against image classifiers. It uses [cleverhans](http://www.cleverhans.io/) to
generate attack sets and it capitalizes on their portability. It then tries
various strategies to reduce the effectivnes of the attacks, as measured by the
degradation in accuracy, and it reports the results.


**Manifest:**  
  - Presentation:
     - [Adversarial Attacks Against Image Classifiers.pdf](Adversarial%20Attacks%20Against%20Image%20Classifiers.pdf)
  - Supporting Code:   
     - [mnist_bbox_gen_adv.py](mnist_bbox_gen_adv.py) - Generate adversarial
       test and training sets using cleverhans using configurable lambda and
       epsilon
     - [mnist_adv_test.ipynb](mnist_adv_test.ipynb) - Run advesarial attack on a
       given model using a defense strategy and save the results
     - [mnist_test_gear.ipynb](mnist_test_gear.ipynb) - Run Grid Search with
       Cross Validation for classifier hyperparameter tunning
     - [mnist_adv_results.ipynb](mnist_adv_results.ipynb) - Aggregate test results
     - [mnist_knn_gscv.ipynb](mnist_knn_gscv.ipynb), 
       [mnist_svc_gscv.ipynb](mnist_svc_gscv.ipynb), 
       [mnist_lr_gscv.ipynb](mnist_lr_gscv.ipynb) - Determine best hyperparameters 
       for KNN, SVM, or LogistRegression using mnist_test_gear.ipynb
     - [mnist_keras_cnn.ipynb](mnist_keras_cnn.ipynb) - Keras based CNN for the
       MNIST set
     - [mnist_utils.ipynb](mnist_utils.ipynb) - Various utils for handling the
       MNIST set
     - [noise_utils.ipynb](noise_utils.ipynb) - Utils for adding noise to images
