import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import ticker 
import seaborn as sns

GINI_TRAIN_COLOUR = 'palegreen'
ENTROPY_TRAIN_COLOUR = 'lightblue'
GINI_TEST_COLOUR = 'darkgreen'
ENTROPY_TEST_COLOUR = 'darkblue'


RBF_TRAIN_COLOUR = 'palegreen'
LINEAR_TRAIN_COLOUR = 'lightblue'
SIGMOID_TRAIN_COLOUR = 'sandybrown'
RBF_TEST_COLOUR = 'darkgreen'
LINEAR_TEST_COLOUR = 'darkblue'
SIGMOID_TEST_COLOUR = 'darkorange'
# Create learning curve charts
data = pd.read_csv("internet/cvresults_svm.csv", index_col=0)


# ksns.lineplot(x='param_classify__max_depth', y='mean_test_score', data=data)

def plot_decision_tree(data):
    fig, ax = plt.subplots(1, 1)

    gini_ix = data["param_classify__criterion"] == "gini"
    entropy_ix = data["param_classify__criterion"] == "entropy"
    gini = data[gini_ix].sort_values(by="param_classify__max_depth")
    entropy = data[entropy_ix].sort_values(by="param_classify__max_depth")

    ax.plot(gini["param_classify__max_depth"], gini["mean_train_score"], marker='o', c=GINI_TRAIN_COLOUR, label='Gini Criterion (Train)')
    ax.plot(entropy["param_classify__max_depth"], entropy["mean_train_score"], marker='o', c=ENTROPY_TRAIN_COLOUR, label='Entropy Criterion (Train)')

    ax.plot(gini["param_classify__max_depth"], gini["mean_test_score"], marker='o', c=GINI_TEST_COLOUR, label='Gini Criterion (Validation)')
    ax.plot(entropy["param_classify__max_depth"], entropy["mean_test_score"], marker='o', c=ENTROPY_TEST_COLOUR, label='Entropy Criterion (Validation)')

    ax.fill_between(
        gini.param_classify__max_depth,
        gini.mean_train_score - gini.std_train_score,
        gini.mean_train_score + gini.std_train_score,
        alpha=0.3,
        color = GINI_TRAIN_COLOUR
    )
    ax.fill_between(
        entropy.param_classify__max_depth,
        entropy.mean_train_score - entropy.std_train_score,
        entropy.mean_train_score + entropy.std_train_score,
        alpha=0.3,
        color = ENTROPY_TRAIN_COLOUR
    )

    ax.fill_between(
        gini.param_classify__max_depth,
        gini.mean_test_score - gini.std_test_score,
        gini.mean_test_score + gini.std_test_score,
        alpha=0.3,
        color = GINI_TEST_COLOUR
    )
    ax.fill_between(
        entropy.param_classify__max_depth,
        entropy.mean_test_score - entropy.std_test_score,
        entropy.mean_test_score + entropy.std_test_score,
        alpha=0.3,
        color = ENTROPY_TEST_COLOUR
    )

    ax.xaxis.set_major_locator(ticker.MultipleLocator())

    ax.legend()

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Max Depth')
    ax.set_xlim((data.param_classify__max_depth.min(), data.param_classify__max_depth.max()))
    ax.set_ylim((0.5, 1))
    ax.grid()

    ax.set_title('Decision Tree Performance across Hyperparameter Values')
    plt.savefig('Decision Tree CV Analysis.png')

def plot_knn_cv(data):
    fig, ax = plt.subplots(1, 1)

    ax.plot(data['param_classify__n_neighbors'], data['mean_train_score'], label='Train', c=GINI_TRAIN_COLOUR, marker='o')
    ax.plot(data['param_classify__n_neighbors'], data['mean_test_score'], label='Validation', c=GINI_TEST_COLOUR, marker='o')

    ax.fill_between(
        data.param_classify__n_neighbors,
        data.mean_train_score - data.std_train_score,
        data.mean_train_score + data.std_train_score,
        alpha=0.3,
        color = GINI_TRAIN_COLOUR
    )
    ax.fill_between(
        data.param_classify__n_neighbors,
        data.mean_test_score - data.std_test_score,
        data.mean_test_score + data.std_test_score,
        alpha=0.3,
        color = GINI_TEST_COLOUR
    )

    ax.xaxis.set_major_locator(ticker.MultipleLocator())

    ax.legend()

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('# Of Neighbors')
    ax.set_xlim((data.param_classify__n_neighbors.min(), data.param_classify__n_neighbors.max()))
    ax.set_ylim((0.5, 1))
    ax.grid()
    plt.show()
    ax.set_title('kNN Performance across Different K Values')
    plt.savefig('kNN CV Analysis.png')




data['logC'] = np.log10(data['param_classify__C'])

rbf_ix = data["param_classify__kernel"] == "rbf"
linear_ix = data["param_classify__kernel"] == "linear"
sigmoid_ix = data["param_classify__kernel"] == "sigmoid"

rbf = data[rbf_ix].sort_values(by="param_classify__C")
linear = data[linear_ix].sort_values(by="param_classify__C")
sigmoid = data[sigmoid_ix].sort_values(by="param_classify__C")

fig, ax = plt.subplots(1, 1)

ax.plot(rbf["logC"], rbf["mean_train_score"], marker='o', c=RBF_TRAIN_COLOUR, label='RBF Kernel (Train)')
ax.plot(linear["logC"], linear["mean_train_score"], marker='o', c=LINEAR_TRAIN_COLOUR, label='Linear Kernel (Train)')
ax.plot(sigmoid["logC"], sigmoid["mean_train_score"], marker='o', c=SIGMOID_TRAIN_COLOUR, label='sigmoid Kernel (Train)')

ax.plot(rbf["logC"], rbf["mean_test_score"], marker='o', c=RBF_TEST_COLOUR, label='RBF Kernel (Validation)')
ax.plot(linear["logC"], linear["mean_test_score"], marker='o', c=LINEAR_TEST_COLOUR, label='Linear Kernel (Validation)')
ax.plot(sigmoid["logC"], sigmoid["mean_test_score"], marker='o', c=SIGMOID_TEST_COLOUR, label='Sigmoid Kernel (Validation)')

ax.fill_between(
    rbf.logC,
    rbf.mean_train_score - rbf.std_train_score,
    rbf.mean_train_score + rbf.std_train_score,
    alpha=0.3,
    color = RBF_TRAIN_COLOUR
)
ax.fill_between(
    linear.logC,
    linear.mean_train_score - linear.std_train_score,
    linear.mean_train_score + linear.std_train_score,
    alpha=0.3,
    color = LINEAR_TRAIN_COLOUR
)
ax.fill_between(
    sigmoid.logC,
    sigmoid.mean_train_score - sigmoid.std_train_score,
    sigmoid.mean_train_score + sigmoid.std_train_score,
    alpha=0.3,
    color = SIGMOID_TRAIN_COLOUR
    )

ax.fill_between(
    rbf.logC,
    rbf.mean_test_score - rbf.std_test_score,
    rbf.mean_test_score + rbf.std_test_score,
    alpha=0.3,
    color = RBF_TEST_COLOUR
)
ax.fill_between(
    linear.logC,
    linear.mean_test_score - linear.std_test_score,
    linear.mean_test_score + linear.std_test_score,
    alpha=0.3,
    color = LINEAR_TEST_COLOUR
)

ax.fill_between(
    sigmoid.logC,
    sigmoid.mean_test_score - sigmoid.std_test_score,
    sigmoid.mean_test_score + sigmoid.std_test_score,
    alpha=0.3,
    color = SIGMOID_TEST_COLOUR
)

ax.xaxis.set_major_locator(ticker.MultipleLocator())

ax.legend()

ax.set_ylabel('Accuracy')
ax.set_xlabel('Log(C)')
ax.set_xlim((data.logC.min(), data.logC.max()))
ax.set_ylim((0.5, 1))
ax.grid()

ax.set_title('SVM Performance across Different Kernels and C Values')
plt.show()
