from sklearn.metrics import log_loss, accuracy_score
from data_utils import load_intention
import pandas as pd

def generate_mlp_loss_curve(pipe, trainX, trainY, valX, valY, epochs=500):

    mapper = pipe['normalize']

    trainX = mapper.fit_transform(trainX)
    valX = mapper.fit_transform(valX)
    
    clf = pipe['classify']

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


