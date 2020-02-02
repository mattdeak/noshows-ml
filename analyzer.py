from pipelines import make_pipeline
from credit_preprocessing import load_credit
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
from mushroom_preprocess import load_mushrooms
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from preprocessing import load_noshows_preprocessed
import shutil
import json
import os


def load_spec(path):
    with open(path, "r") as specfile:
        spec = json.load(specfile)
    return spec


class Runner:
    def __init__(self, name, spec_path):
        self.name = name
        self.spec = load_spec(spec_path)
        self.data = self.spec["data"]
        self.test_result_filepath = os.path.join(self.name, "test_results.csv")

    def prepare_datasets(self):
        if self.data == "noshows":
            X, y = load_noshows_preprocessed(
                "noshows.csv"
            )  # TODO: Extract and modularize
        elif self.data == "mushrooms":
            X, y = load_mushrooms()
        elif self.data == "credit":
            X, y = load_credit()
        else:
            raise NotImplementedError(f"Data {self.data} not yet supported")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, shuffle=True, train_size=0.8
        )

    def create_or_clear_result_folder(self):
        if os.path.exists(self.name):
            shutil.rmtree(self.name)

        os.mkdir(self.name)
        with open(self.test_result_filepath, "w") as tr:
            tr.write("Model, Test Accuracy\n")

    def get_output_path(self, model):
        filepath = os.path.join(self.name, f"cvresults_{model}.csv")
        return filepath

    def run(self):
        print("Creating Output Folder")
        self.create_or_clear_result_folder()
        print(f"Starting Experiment: {self.name}")

        print("Preparing Data")
        self.prepare_datasets()

        for model, kwargs in self.spec["models"].items():
            param_grid = kwargs.get("param_grid", {})
            search_type = kwargs.get("search_type", None)
            self.run_model(model, param_grid, search_type=search_type)

    def run_model(self, model, param_grid, search_type=None, metric="accuracy"):
        print(f"Running experiment for model: {model}")
        print("----------------------")

        assert search_type, "Must provide a search type of 'random' or 'grid'"
        assert search_type in [
            "random",
            "grid",
        ], f"Search type {search_type} not supported. Must be 'random' or 'grid'"

        pipe = make_pipeline(model)

        print("Collecting Cross-Validation Results")
        tuner = self.record_cv_results(pipe, param_grid, model, search_type=search_type)
        test_score = tuner.score(self.X_test, self.y_test)

        print("Collecting Learning Curves")
        best_estimator = tuner.best_estimator_
        self.record_learning_curves(best_estimator, model)

        print("Getting Classification Reports")
        self.record_classification_report(best_estimator, model)
        with open(self.test_result_filepath, "a") as resultfile:
            resultfile.write(f"{model},{test_score}\n")

        print("----------------------")
        print("----------------------")

    def record_cv_results(
        self, pipe, param_grid, model, search_type=None, metric="accuracy"
    ):
        """get_cv_results

        Parameters
        ----------

        pipe : An sklearn pipeline
        param_grid : Parameter grid for cv
        search_type : Random or Grid search

        Returns tuned searcher
        -------
        """
        if search_type == "grid":
            tuner = GridSearchCV(pipe, param_grid, scoring=metric, cv=5)

        elif search_type == "random":
            tuner = RandomizedSearchCV(pipe, param_grid, scoring=metric, cv=5)

        tuner.fit(self.X_train, self.y_train)
        # TODO: Export results to report file
        results = pd.DataFrame(tuner.cv_results_)
        result_output = self.get_output_path(model)
        results.to_csv(result_output)
        return tuner

    def record_learning_curves(self, pipe, model):
        """record_learning_curves

        Parameters
        ----------

        pipe : An sklearn pipeline

        Returns None
        -------
        """
        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            pipe, self.X_train, self.y_train, cv=5, return_times=True
        )
        learning_curve_results = {
            "train_sizes": train_sizes.tolist(),
            "train_scores": train_scores.tolist(),
            "test_scores": test_scores.tolist(),
            "fit_times": fit_times.tolist(),
        }
        with open(
            os.path.join(self.name, f"{model}_learningcurve.json"), "w"
        ) as lcfile:
            json.dump(learning_curve_results, lcfile)

    def record_classification_report(self, tuned_pipe, model_name):
        """record_classification_report

        Gets a classification report for both the train and test set and writes it to file

        Parameters
        ----------

        pipe :

        Returns
        -------
        """
        preds = tuned_pipe.predict(self.X_test)
        test_result = classification_report(self.y_test, preds, output_dict=True)
        with open(
            os.path.join(self.name, f"{model_name}_classificationreport_test.json"), "w"
        ) as crfile:
            json.dump(test_result, crfile)

        train_preds = tuned_pipe.predict(self.X_train)
        train_resul = classification_report(self.y_train, train_preds, output_dict=True)
        with open(
            os.path.join(self.name, f"{model_name}_classificationreport_train.json"),
            "w",
        ) as crfile:
            json.dump(test_result, crfile)


class Analyzer:
    """Takes the results from the Runner and produces visualizations and reports based on the experiment."""

    def __init__(self):
        """__init__"""

    def generate_learning_curve_plots(self):
        """generate_learning_curve_plots"""

    def generate_cv_plots(self):
        """generate_cv_plots"""

    def generate_classification_report_plots(self):
        """generate_classification_report_plots"""


runner = Runner("credit", "specs/credit_spec.json")

runner.run()
