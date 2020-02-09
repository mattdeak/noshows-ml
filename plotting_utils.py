
import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
from collections import defaultdict

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

LOSS_TRAIN = 'palegreen'
LOSS_VAL = 'darkgreen'
ACC_TRAIN = 'lightblue'
ACC_VAL = 'darkblue'



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
    plt.close(fig)

def plot_boosting_cv(data, outfile):
    data['DT Max Depth'] = data['param_classify__base_estimator__max_depth']
    data['Learning Rate'] = data['param_classify__learning_rate']
    train = data.pivot('DT Max Depth', 'Learning Rate', values='mean_train_score')
    test = data.pivot('DT Max Depth', 'Learning Rate', values='mean_test_score')

    vmin_train = data['mean_train_score'].min()
    vmax_train = data['mean_train_score'].max()
    vmin_test = data['mean_test_score'].min()
    vmax_test = data['mean_test_score'].max()


    ax = sns.heatmap(train, annot=True, cmap='Blues', vmin=vmin_train, vmax=vmax_train, fmt='.4g')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_title('Mean Accuracy of Learning Rate vs. DT Max Depth (Training Set)')

    modified_outfile = outfile[:-4] # Without the .png
    plt.savefig(f'{modified_outfile}(Train).png')
    plt.close()


    ax = sns.heatmap(test, annot=True, cmap='Blues', vmin=vmin_test, vmax=vmax_test, fmt='.4g')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_title('Mean Accuracy of Learning Rate vs. DT Max Depth (Validation Set)')


    plt.savefig(f'{modified_outfile}(Validation).png')
    plt.close()

def plot_mlp_loss_curve(data, outfile):

    fig, ax = plt.subplots(1, 1)
    ax2 = ax.twinx()

    xs = np.arange(data.shape[0])
    lines = []
    lines += ax.plot(xs, data['Train Log-Loss'], c=LOSS_TRAIN, label='Training Log-Loss')
    lines += ax.plot(xs, data['Validation Log-Loss'], c=LOSS_VAL, label='Validation Log-Loss')
    lines += ax2.plot(xs, data['Train Score'], c=ACC_TRAIN, label='Training Accuracy')
    lines += ax2.plot(xs, data['Validation Score'], c=ACC_VAL, label='Validation Accuracy')

    labels = [l.get_label() for l in lines]
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Log-Loss')
    ax2.set_ylabel('Accuracy Ratio')
    ax2.legend(lines, labels, loc='lower_left')
    ax.set_ylim(bottom=0)
    ax2.set_ylim((0.8, 1))
    ax.set_xlim((xs[0], xs[-1]))
    ax.set_title('Neural Net Loss/Accuracy vs. Epochs Trained')
    plt.savefig(outfile)
    plt.close(fig)


def aggregate_reports(result_folder):
    """aggregate_reports

    Parameters
    ----------

    results_folder : folder containing experiment data

    Returns A dataframe containing Precision, Recall, Accuracy, F1, Train Time and Prediction Time for each model
    -------
    """
    results = {}
    cr_reports = [os.path.join(result_folder, f) for f in os.listdir(result_folder) if 'classificationreport_test' in f]
    for report in cr_reports:
        model = report.split('/')[1].split('_')[0]

        with open(report, 'r') as rf:
            model_dict_raw = json.load(rf)

        md = {}
        md['accuracy'] = model_dict_raw['accuracy']
        try:
            md['precision'] = model_dict_raw['True']['precision']
            md['recall'] = model_dict_raw['True']['recall']
            md['f1-score'] = model_dict_raw['True']['f1-score']
        except KeyError: # it's 1 instead of True
            md['precision'] = model_dict_raw['1']['precision']
            md['recall'] = model_dict_raw['1']['recall']
            md['f1-score'] = model_dict_raw['1']['f1-score']

        results[model] = md

    gsearch_reports = [os.path.join(result_folder, f) for f in os.listdir(result_folder) if 'cvresults' in f] 
    for report in gsearch_reports:
        model = report.split('/')[1].split('_')[1][:-4]
        df = pd.read_csv(report)

        best_row = df[df.rank_test_score == 1]

        results[model]['Train Time'] = best_row.mean_fit_time.values[0]
        results[model]['Prediction Time'] = best_row.mean_score_time.values[0]

    return pd.DataFrame(results).round(4)


def plot_boosting_iter_curve(data, outfile):
    fig, ax = plt.subplots(1, 1)

    ax.plot(data.Iterations, data['Train Score'], label='Training', c=ACC_TRAIN)
    ax.plot(data.Iterations, data['Validation Scores'], label='Validation', c=ACC_VAL)

    ax.set_ylim((0.8, 1.0))
    ax.set_xlim((data.Iterations.min(), data.Iterations.max()))

    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Accuracy Ratio')
    ax.set_title('AdaBoost Accuracy over Iterations')
    ax.legend()
    plt.savefig(outfile)
    plt.close(fig)

def extract_best_worst_model_params(folder):

    cvreports = [os.path.join(folder, f) for f in os.listdir(folder) if 'cvresults' in f]
    results = defaultdict(dict)
    for report in cvreports:
        model = report.split('/')[1].split('_')[1][:-4]
        df = pd.read_csv(report)

        best_row = df[df['rank_test_score'] == 1]
        worst_row = df[df['rank_test_score'] == df.rank_test_score.max()]
        params = [c for c in df.columns if 'param' in c]
        
        def get_row_info(row, params):
            r = {}
            for param in params:
                param_name = param.replace('param_classify__','')
                r[param_name] = row[param].values[0]
            r['train_acc'] = np.round(row['mean_train_score'].values[0], 4)
            r['val_acc'] = np.round(row['mean_test_score'].values[0], 4)

            return r

        results[model]['best'] = get_row_info(best_row, params)
        results[model]['worst'] = get_row_info(worst_row, params)



    return results


p = extract_best_worst_model_params('pulsar')
i = extract_best_worst_model_params('intention')
        

    
