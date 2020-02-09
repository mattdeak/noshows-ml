import json
import os
import pickle
import shutil

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import plotting_utils as pu
from plotting_utils import aggregate_reports
from pipelines import make_pipeline
from sklearn.metrics import classification_report
from train_utils import generate_mlp_loss_curve
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    learning_curve,
    train_test_split,
)


def load_spec(path):
    with open(path, "r") as specfile:
        spec = json.load(specfile)
    return spec


class Analyzer:
    """Takes the results from the Runner and produces visualizations and reports based on the experiment."""

    def __init__(self, experiment_name):
        """__init__"""
        self.results_folder = experiment_name
        self.files = [
            os.path.join(experiment_name, x) for x in os.listdir(experiment_name)
        ]

    def _parse_model_name(self, result_file_path):
        if "knn" in result_file_path:
            model_name = "KNN"
        elif "tree" in result_file_path:
            model_name = "Decision Tree"
        elif "svm" in result_file_path:
            model_name = "SVM"
        elif "boosting" in result_file_path:
            model_name = "AdaBoost"
        elif "neural" in result_file_path:
            model_name = "Neural Network"

        return model_name

    def generate_all_plots(self):
        self.generate_learning_curve_plots()
        self.generate_cv_plots()
        self.generate_boosting_iter_plot()

    def generate_learning_curve_plots(self):
        """generate_learning_curve_plots"""
        lr_results = [f for f in self.files if "learningcurve" in f]

        for result_file in lr_results:

            if "neural_epoch" in result_file:
                data = pd.read_csv(result_file, index_col=0)
                pu.plot_mlp_loss_curve(
                    data, os.path.join(self.results_folder, "MLP Loss Curve.png")
                )
                continue

            model_name = self._parse_model_name(result_file)

            with open(result_file, "r") as f:
                data = json.load(f)

            fit_times = np.array(data["fit_times"])
            train_scores = np.array(data["train_scores"])
            test_scores = np.array(data["test_scores"])
            train_sizes = np.array(data["train_sizes"])

            mean_times = np.mean(fit_times, axis=1)
            mean_trainscores = np.mean(train_scores, axis=1)
            mean_testscores = np.mean(test_scores, axis=1)

            std_trainscores = np.std(train_scores, axis=1)
            std_testscores = np.std(test_scores, axis=1)

            fig, ax = plt.subplots(1, 1)

            train = ax.plot(
                train_sizes, mean_trainscores, marker="o", label="Train Score"
            )
            test = ax.plot(
                train_sizes, mean_testscores, marker="o", label="Validation Score"
            )

            ax.fill_between(
                train_sizes,
                mean_trainscores + std_trainscores,
                mean_trainscores - std_trainscores,
                alpha=0.3,
            )
            ax.fill_between(
                train_sizes,
                mean_testscores + std_testscores,
                mean_testscores - std_testscores,
                alpha=0.3,
            )
            ax.set_xlabel("# of Train Samples")
            ax.set_ylabel("Accuracy Ratio")
            ax.set_ylim((0.8, 1))
            ax.set_title(f"{model_name} Learning Curve")
            ax.legend()
            ax.grid()

            plt.savefig(
                os.path.join(self.results_folder, f"{model_name} Learning Curve.png")
            )
            plt.close(fig)

    def generate_cv_plots(self):
        """generate_cv_plots"""
        cv_results = [f for f in self.files if "cvresults" in f]
        for results_file in cv_results:

            data = pd.read_csv(results_file)
            model_name = self._parse_model_name(results_file)

            filename = f"{model_name} CV Results.png"
            outfile = os.path.join(self.results_folder, filename)
            if model_name == "KNN":
                pu.plot_knn_cv(data, outfile)
            elif model_name == "Decision Tree":
                pu.plot_tree_cv(data, outfile)
            elif model_name == "SVM":
                pu.plot_svm_cv(data, outfile)
            elif model_name == "Neural Network":
                pu.plot_mlp_cv(data, outfile)
            elif model_name == "AdaBoost":
                pu.plot_boosting_cv(data, outfile)
            else:
                print(f"Model {model_name} not supported yet")
                continue

    def generate_boosting_iter_plot(self):
        f = os.path.join(self.results_folder, "boosting_iter_curve.csv")
        data = pd.read_csv(f, index_col=0)

        outpath = os.path.join(self.results_folder, "Boosting Iter Curve.png")
        pu.plot_boosting_iter_curve(data,outpath)


if __name__ == "__main__":
    analyzer = Analyzer("pulsar")
    analyzer.generate_all_plots()
    analyzer = Analyzer("intention")
    analyzer.generate_all_plots()

    print(aggregate_reports("intention"))
    print(aggregate_reports("pulsar"))


#     runner1 = Runner("internet", "specs/internet_spec.json", exclude=['neural'])
#     runner2 = Runner('pulsar', "specs/pulsar_spec.json", exclude=['neural'])

#     print("Running online shopper retention")
#     runner1.run()
#     print("Running pulsar detection")
#     runner2.run()
