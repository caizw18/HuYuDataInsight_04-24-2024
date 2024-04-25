import os
import matplotlib.pyplot as plt
import numpy as np

from LogisticRegression import logistic_regression
from DataReader import *

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"
    
def visualize_features(X, y):
    '''This function is used to plot a 2-D scatter plot of training features. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    plt.clf()
    plt.plot(X[y==1,0], X[y==1,1], 'ob', markersize=3.5)
    plt.plot(X[y==-1,0], X[y==-1,1], 'or', markersize=3.5)
    plt.legend(['Class1','Class2'], loc="lower left", title="Classes")
    plt.xlabel("Symmetric value")
    plt.ylabel("Intense value")
    plt.xlim([-1,0])
    plt.ylim([-1,0])
    plt.savefig("../train_features/train_features.pdf")
    ### END YOUR CODE

def visualize_result(X, y, W):
    '''This function is used to plot the sigmoid model after training.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        W: An array of shape [n_features,].

    Returns:
        No return. Save the plot to 'train_result_sigmoid.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    plt.clf()
    plt.plot(X[y==1,0], X[y==1,1], 'ob', markersize=3.5)
    plt.plot(X[y==-1,0], X[y==-1,1], 'or', markersize=3.5)
    plt.legend(['Class1','Class2'], loc="lower left", title="Classes")
    plt.xlabel("Symmetric value")
    plt.ylabel("Intense value")

    symm = np.array([X[:,0].min(), X[:,0].max()])
    decision_bound = -(W[1] * symm + W[0]) / W[2]
    plt.plot(symm, decision_bound, '--k')


    plt.xlim([-1,0])
    plt.ylim([-1,0])
    plt.savefig("../train_result/train_result_sigmoid.pdf")
    ### END YOUR CODE

def visualize_result_multi(X, y, W):
    '''This function is used to plot the softmax model after training.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 0,1,2.
        W: An array of shape [n_features, 3].

    Returns:
        No return. Save the plot to 'train_result_softmax.*' and include it
		in submission.
    '''
    ### YOUR CODE HERE
    plt.clf()
    plt.plot(X[y==0,0], X[y==0,1], 'ob', markersize=3.5)
    plt.plot(X[y==1,0], X[y==1,1], 'or', markersize=3.5)
    plt.plot(X[y==2,0], X[y==2,1], 'og', markersize=3.5)
    plt.legend(['Class0','Class1','Class2'], loc="lower left", title="Classes")
    plt.xlabel("Symmetric value")
    plt.ylabel("Intense value")

    symm_r = np.linspace(X[:,0].min(), X[:,0].max())
    data1 = data2 = np.zeros(symm_r.shape)
    for i, x in enumerate(symm_r):
        w0, w1, w2 = (W[0], W[1], W[2])

        data1_1 = (x*(w1[1] - w0[1]) + (w1[0] - w0[0])) / (w0[2] - w1[2])
        data1_2 = (x*(w2[1] - w0[1]) + (w2[0] - w0[0])) / (w0[2] - w2[2])
        data1[i] = max(data1_1, data1_2)
        data2_1 = (x*(w0[1] - w1[1]) + (w0[0] - w1[0])) / (w1[2] - w0[2])
        data2_2 = (x*(w2[1] - w1[1]) + (w2[0] - w1[0])) / (w1[2] - w2[2])
        data2[i] = min(data2_1, data2_2)

    plt.plot(symm_r, data1, '--k')
    plt.plot(symm_r, data2, '--k')

    plt.xlim([-1,0])
    plt.ylim([-1,0])
    plt.savefig("../train_result/train_result_softmax.pdf")
    ### END YOUR CODE

def main():
	# ------------Data Preprocessing------------
	# Read data for training.
    
    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    ##### Preprocess raw data to extract features
    train_X_all = prepare_X(raw_train)
    valid_X_all = prepare_X(raw_valid)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)

    ####### For binary case, only use data from '1' and '2'
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    ####### Only use the first 1350 data examples for binary training.
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class.
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1
    data_shape= train_y.shape[0]

#    # Visualize training data.
    visualize_features(train_X[:, 1:3], train_y)


   # ------------Logistic Regression Sigmoid Case------------

   ##### Check BGD, SGD, miniBGD
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)

    logisticR_classifier.fit_BGD(train_X, train_y, data_shape)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, data_shape)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_SGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 1)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))


    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    logistic1 = logistic_regression(learning_rate=0.1, max_iter=2000)
    logistic1.fit_BGD(train_X, train_y, 32)
    print("Logistic Regression on Sigmoid Case")
    print("The weights are:{}".format(logistic1.get_params()))
    print("The accuracy of training is:{}".format(logistic1.score(train_X, train_y)))
    print("The accuracy of validation is:{}".format(logistic1.score(valid_X, valid_y)))
    ### END YOUR CODE

	# Visualize the your 'best' model after training.
	# visualize_result(train_X[:, 1:3], train_y, best_logisticR.get_params())

    ### YOUR CODE HERE
    visualize_result(train_X[:, 1:3], train_y, logistic1.get_params())
    ### END YOUR CODE

    # Use the 'best' model above to do testing. Note that the test data should be loaded and processed in the same way as the training data.
    ### YOUR CODE HERE
    raw_test_data, labels_test_data = load_data(os.path.join(data_dir, test_filename))
    testdata_X = prepare_X(raw_test_data)
    testdata_y, testdata_id = prepare_y(labels_test_data)

    testdata_X_2 = testdata_X[testdata_id]

    testdata_y_2 = testdata_y[testdata_id]

    testdata_y_2[np.where(testdata_y_2==2)] = -1

    print("The accuracy of testdata is:{}".format(logistic1.score(testdata_X_2, testdata_y_2)))

    ### END YOUR CODE



    # ------------End------------
    

if __name__ == '__main__':
	main()
    
    
