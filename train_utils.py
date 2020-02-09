from sklearn.metrics import log_loss, accuracy_score
import numpy as np
from sklearn.base import clone
from data_utils import load_intention
import pandas as pd

def generate_mlp_loss_curve(pipe, trainX, trainY, valX, valY, epochs=500):

    mapper = pipe['normalize']

    trainX = mapper.fit_transform(trainX)
    valX = mapper.fit_transform(valX)
    
    clf = clone(pipe['classify']) # We need it fresh

    losses = []
    val_losses = []
    train_scores = []
    val_scores = []

    for i in range(epochs):
        clf.partial_fit(trainX, trainY, classes=[True, False])
        train_preds = clf.predict(trainX)
        val_preds = clf.predict(valX)

        train_logloss = log_loss(trainY, train_preds)
        val_logloss = log_loss(valY, val_preds)
    
        train_score = clf.score(trainX, trainY)
        val_score = accuracy_score(valY, val_preds)

        losses.append(train_logloss)
        val_losses.append(val_logloss)
        train_scores.append(train_score)
        val_scores.append(val_score)

    return pd.DataFrame({'Train Log-Loss': losses, 'Validation Log-Loss': val_losses, 'Train Score': train_scores, 'Validation Score' : val_scores})


def generate_boosting_iter_curve(pipe, trainX, trainY, valX, valY, max_iter=50):
    mapper = pipe['normalize']

    trainX = mapper.fit_transform(trainX)
    valX = mapper.fit_transform(valX)

    iteration = []
    train_scores = []
    val_scores = []
    
    for i in range(max_iter):
        np.random.seed(0) # Need to ensure all iterations run exactly the same up to i
        
        clf = clone(pipe['classify']) # We need it fresh
        clf.set_params(n_estimators=i+1)
        clf.fit(trainX, trainY)
        train_score = clf.score(trainX, trainY)
        val_score = clf.score(valX, valY)
        
        train_scores.append(train_score)
        val_scores.append(val_score)
        iteration.append(i+1)

    df = pd.DataFrame({'Iterations': iteration, 'Train Score': train_scores, 'Validation Scores': val_scores})
    return df
