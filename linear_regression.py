import pandas as pd
import numpy as np

def predict(X,weights):
    return X.dot(weights)

def compute_cost(X,Y,weights):
    return 1/(2*len(X)) * np.sum((predict(X,weights) - Y)**2)

def gradient_descent(X,Y,weights,alpha):
    return weights - alpha/len(X) * X.T.dot(predict(X,weights) - Y)

def fit_model(X,Y,iterations,learning_rate):
    weights = np.zeros((X.shape[1],1))
    no_iterations = []
    error = []
    for i in range(iterations):
        no_iterations.append(i)
        error.append(compute_cost(X,Y,weights))
        weights = gradient_descent(X,Y,weights,learning_rate)
    return weights, no_iterations, error
def evaluate_performance(Y,predicted_val): # Evaluate the Performance using Mean Absolute Error(MAE)
    return np.sum(np.abs(Y - predicted_val))/len(Y)


def feature_normalize(X):
    for i in range(X.shape[1]):
        X[:,i] = (X[:,i] - (np.sum(X[:,i]))/len(X[:,i]))/np.std(X[:,i]) # normalize the features
    X = np.hstack((np.ones((len(X),1)) , X)) # add ones column for x0
    return X

def read_dat(path):
    data = pd.read_csv(path, sep=",", header=None)
    data = data.to_numpy()
    Y = data[:,-1].reshape(len(data),1)
    X = data[:,:-1]
    return X,Y

    

