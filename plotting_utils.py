
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

GINI_TRAIN_COLOUR = "palegreen"
ENTROPY_TRAIN_COLOUR = "lightblue"
GINI_TEST_COLOUR = "darkgreen"
ENTROPY_TEST_COLOUR = "darkblue"


RBF_TRAIN_COLOUR = "palegreen"
LINEAR_TRAIN_COLOUR = "lightblue"
SIGMOID_TRAIN_COLOUR = "sandybrown"
RBF_TEST_COLOUR = "darkgreen"
LINEAR_TEST_COLOUR = "darkblue"
SIGMOID_TEST_COLOUR = "darkorange"

NEURAL_A_COLOUR = "green"
NEURAL_B_COLOUR = "mediumblue"
NEURAL_C_COLOUR = "goldenrod"
NEURAL_D_COLOUR = "darkviolet"


def plot_tree_cv(data, outfile):
    fig, ax = plt.subplots(1, 1)

    gini_ix = data["param_classify__criterion"] == "gini"
    entropy_ix = data["param_classify__criterion"] == "entropy"
    gini = data[gini_ix].sort_values(by="param_classify__max_depth")
    entropy = data[entropy_ix].sort_values(by="param_classify__max_depth")

    ax.plot(
        gini["param_classify__max_depth"],
        gini["mean_train_score"],
        marker="o",
        c=GINI_TRAIN_COLOUR,
        label="Gini Criterion (Train)",
    )
    ax.plot(
        entropy["param_classify__max_depth"],
        entropy["mean_train_score"],
        marker="o",
        c=ENTROPY_TRAIN_COLOUR,
        label="Entropy Criterion (Train)",
    )

    ax.plot(
        gini["param_classify__max_depth"],
        gini["mean_test_score"],
        marker="o",
        c=GINI_TEST_COLOUR,
        label="Gini Criterion (Validation)",
    )
    ax.plot(
        entropy["param_classify__max_depth"],
        entropy["mean_test_score"],
        marker="o",
        c=ENTROPY_TEST_COLOUR,
        label="Entropy Criterion (Validation)",
    )

    ax.fill_between(
        gini.param_classify__max_depth,
        gini.mean_train_score - gini.std_train_score,
        gini.mean_train_score + gini.std_train_score,
        alpha=0.3,
        color=GINI_TRAIN_COLOUR,
    )
    ax.fill_between(
        entropy.param_classify__max_depth,
        entropy.mean_train_score - entropy.std_train_score,
        entropy.mean_train_score + entropy.std_train_score,
        alpha=0.3,
        color=ENTROPY_TRAIN_COLOUR,
    )

    ax.fill_between(
        gini.param_classify__max_depth,
        gini.mean_test_score - gini.std_test_score,
        gini.mean_test_score + gini.std_test_score,
        alpha=0.3,
        color=GINI_TEST_COLOUR,
    )
    ax.fill_between(
        entropy.param_classify__max_depth,
        entropy.mean_test_score - entropy.std_test_score,
        entropy.mean_test_score + entropy.std_test_score,
        alpha=0.3,
        color=ENTROPY_TEST_COLOUR,
    )

    ax.xaxis.set_major_locator(ticker.MultipleLocator())

    ax.legend()

    ax.set_ylabel("Accuracy Ratio")
    ax.set_xlabel("Max Depth")
    ax.set_xlim(
        (data.param_classify__max_depth.min(), data.param_classify__max_depth.max())
    )
    ax.set_ylim((0.8, 1))
    ax.grid()

    ax.set_title("Decision Tree Performance across Hyperparameter Values")
    plt.savefig(outfile)
    plt.close(fig)


def plot_knn_cv(data, outfile):
    fig, ax = plt.subplots(1, 1)

    ax.plot(
        data["param_classify__n_neighbors"],
        data["mean_train_score"],
        label="Train",
        c=GINI_TRAIN_COLOUR,
        marker="o",
    )
    ax.plot(
        data["param_classify__n_neighbors"],
        data["mean_test_score"],
        label="Validation",
        c=GINI_TEST_COLOUR,
        marker="o",
    )

    ax.fill_between(
        data.param_classify__n_neighbors,
        data.mean_train_score - data.std_train_score,
        data.mean_train_score + data.std_train_score,
        alpha=0.3,
        color=GINI_TRAIN_COLOUR,
    )
    ax.fill_between(
        data.param_classify__n_neighbors,
        data.mean_test_score - data.std_test_score,
        data.mean_test_score + data.std_test_score,
        alpha=0.3,
        color=GINI_TEST_COLOUR,
    )

    ax.xaxis.set_major_locator(ticker.MultipleLocator())

    ax.legend()

    ax.set_ylabel("Accuracy Ratio")
    ax.set_xlabel("# Of Neighbors")
    ax.set_xlim(
        (data.param_classify__n_neighbors.min(), data.param_classify__n_neighbors.max())
    )
    ax.set_ylim((0.8, 1))
    ax.grid()
    ax.set_title("kNN Performance across Different K Values")
    plt.savefig(outfile)
    plt.close(fig)


def plot_svm_cv(data, outfile):
    data["logC"] = np.log10(data["param_classify__C"])

    rbf_ix = data["param_classify__kernel"] == "rbf"
    linear_ix = data["param_classify__kernel"] == "linear"
    sigmoid_ix = data["param_classify__kernel"] == "sigmoid"

    rbf = data[rbf_ix].sort_values(by="param_classify__C")
    linear = data[linear_ix].sort_values(by="param_classify__C")
    sigmoid = data[sigmoid_ix].sort_values(by="param_classify__C")

    fig, ax = plt.subplots(1, 1)

    ax.plot(
        rbf["logC"],
        rbf["mean_train_score"],
        marker="o",
        c=RBF_TRAIN_COLOUR,
        label="RBF Kernel (Train)",
    )
    ax.plot(
        linear["logC"],
        linear["mean_train_score"],
        marker="o",
        c=LINEAR_TRAIN_COLOUR,
        label="Linear Kernel (Train)",
    )
    ax.plot(
        sigmoid["logC"],
        sigmoid["mean_train_score"],
        marker="o",
        c=SIGMOID_TRAIN_COLOUR,
        label="sigmoid Kernel (Train)",
    )

    ax.plot(
        rbf["logC"],
        rbf["mean_test_score"],
        marker="o",
        c=RBF_TEST_COLOUR,
        label="RBF Kernel (Validation)",
    )
    ax.plot(
        linear["logC"],
        linear["mean_test_score"],
        marker="o",
        c=LINEAR_TEST_COLOUR,
        label="Linear Kernel (Validation)",
    )
    ax.plot(
        sigmoid["logC"],
        sigmoid["mean_test_score"],
        marker="o",
        c=SIGMOID_TEST_COLOUR,
        label="Sigmoid Kernel (Validation)",
    )

    ax.fill_between(
        rbf.logC,
        rbf.mean_train_score - rbf.std_train_score,
        rbf.mean_train_score + rbf.std_train_score,
        alpha=0.3,
        color=RBF_TRAIN_COLOUR,
    )
    ax.fill_between(
        linear.logC,
        linear.mean_train_score - linear.std_train_score,
        linear.mean_train_score + linear.std_train_score,
        alpha=0.3,
        color=LINEAR_TRAIN_COLOUR,
    )
    ax.fill_between(
        sigmoid.logC,
        sigmoid.mean_train_score - sigmoid.std_train_score,
        sigmoid.mean_train_score + sigmoid.std_train_score,
        alpha=0.3,
        color=SIGMOID_TRAIN_COLOUR,
    )

    ax.fill_between(
        rbf.logC,
        rbf.mean_test_score - rbf.std_test_score,
        rbf.mean_test_score + rbf.std_test_score,
        alpha=0.3,
        color=RBF_TEST_COLOUR,
    )
    ax.fill_between(
        linear.logC,
        linear.mean_test_score - linear.std_test_score,
        linear.mean_test_score + linear.std_test_score,
        alpha=0.3,
        color=LINEAR_TEST_COLOUR,
    )

    ax.fill_between(
        sigmoid.logC,
        sigmoid.mean_test_score - sigmoid.std_test_score,
        sigmoid.mean_test_score + sigmoid.std_test_score,
        alpha=0.3,
        color=SIGMOID_TEST_COLOUR,
    )

    ax.xaxis.set_major_locator(ticker.MultipleLocator())

    ax.legend()

    ax.set_ylabel("Accuracy Ratio")
    ax.set_xlabel("Log(C)")
    ax.set_xlim((data.logC.min(), data.logC.max()))
    ax.set_ylim((0.8, 1))
    ax.grid()

    ax.set_title("SVM Performance across Different Kernels and C Values")
    plt.savefig(outfile)
    plt.close(fig)

def plot_mlp_cv(data, outfile):
    data = pd.read_csv('intention/cvresults_neural.csv', index_col=0)

    arch1 = data[data.param_classify__hidden_layer_sizes == '[64, 64]']
    arch2 = data[data.param_classify__hidden_layer_sizes == '[128, 128]']
    arch3 = data[data.param_classify__hidden_layer_sizes == '[64, 64, 64]']
    arch4 = data[data.param_classify__hidden_layer_sizes == '[128, 128, 128]']

    fig, ax = plt.subplots(1, 1)

    loglr = np.log10(arch1.param_classify__learning_rate_init)

    ax.plot(loglr, arch1['mean_train_score'], linestyle='solid', marker='o', color=NEURAL_A_COLOUR, label='Architecture A (Train)')
    ax.plot(loglr, arch2['mean_train_score'], linestyle='solid', marker='o', color=NEURAL_B_COLOUR, label='Architecture B (Train)')
    ax.plot(loglr, arch3['mean_train_score'], linestyle='solid', marker='o', color=NEURAL_C_COLOUR, label='Architecture C (Train)')
    ax.plot(loglr, arch4['mean_train_score'], linestyle='solid', marker='o', color=NEURAL_D_COLOUR, label='Architecture D (Train)')

    ax.plot(loglr, arch1['mean_test_score'], linestyle='dashed', marker='o', color=NEURAL_A_COLOUR, label='Architecture A (Validation)')
    ax.plot(loglr, arch2['mean_test_score'], linestyle='dashed', marker='o', color=NEURAL_B_COLOUR, label='Architecture B (Validation)')
    ax.plot(loglr, arch3['mean_test_score'], linestyle='dashed', marker='o', color=NEURAL_C_COLOUR, label='Architecture C (Validation)')
    ax.plot(loglr, arch4['mean_test_score'], linestyle='dashed', marker='o', color=NEURAL_D_COLOUR, label='Architecture D (Validation)')

    ax.fill_between(loglr, arch1['mean_train_score']-arch1['std_train_score'], arch1['mean_train_score']+arch1['std_train_score'], color=NEURAL_A_COLOUR, alpha=0.2)
    ax.fill_between(loglr, arch2['mean_train_score']-arch2['std_train_score'], arch2['mean_train_score']+arch2['std_train_score'], color=NEURAL_B_COLOUR, alpha=0.2)
    ax.fill_between(loglr, arch3['mean_train_score']-arch3['std_train_score'], arch3['mean_train_score']+arch3['std_train_score'], color=NEURAL_C_COLOUR, alpha=0.2)
    ax.fill_between(loglr, arch4['mean_train_score']-arch4['std_train_score'], arch4['mean_train_score']+arch4['std_train_score'], color=NEURAL_D_COLOUR, alpha=0.2)

    ax.fill_between(loglr, arch1['mean_test_score']-arch1['std_test_score'], arch1['mean_test_score']+arch1['std_test_score'], color=NEURAL_A_COLOUR, alpha=0.2)
    ax.fill_between(loglr, arch2['mean_test_score']-arch2['std_test_score'], arch2['mean_test_score']+arch2['std_test_score'], color=NEURAL_B_COLOUR, alpha=0.2)
    ax.fill_between(loglr, arch3['mean_test_score']-arch3['std_test_score'], arch3['mean_test_score']+arch3['std_test_score'], color=NEURAL_C_COLOUR, alpha=0.2)
    ax.fill_between(loglr, arch4['mean_test_score']-arch4['std_test_score'], arch4['mean_test_score']+arch4['std_test_score'], color=NEURAL_D_COLOUR, alpha=0.2)

    ax.xaxis.set_major_locator(ticker.MultipleLocator())

    ax.legend()

    ax.set_ylabel("Accuracy Ratio")
    ax.set_xlabel("Learning Rate Init (Log)")
    ax.set_xlim((loglr.min(), loglr.max()))
    ax.set_ylim((0.8, 1))

    ax.set_title("Neural Network Performance across Hyperparameter Values")
    ax.grid()
    plt.savefig(outfile)

def plot_boosting_cv(data, outfile):
    pass

def plot_mlp_loss_curve(data, outfile):
    pass


data = pd.read_csv('intention/cvresults_neural.csv', index_col=0)
