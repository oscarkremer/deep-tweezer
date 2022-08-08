
import argparse
import sys
import time 
from src import data, models
import os
import pandas as pd
from src.utils import log_time
from sklearn.inspection import permutation_importance


def list_models(package): 
    sampling_models = [str_to_class(name) for name in package.__all__]
    return sampling_models


def str_to_class(classname):
    return getattr(sys.modules['src.models'], classname)()


def get_models_list():
    algorithms = list_models(models.classification)
    sampling_models = list_models(models.sampling)
    sampling_models.append(None)
    return algorithms, sampling_models


def main(filename, target, drop_columns):
    algorithms, _ = get_models_list()#
    data.remove("data/results/permutations_{}".format(filename))
    X, y = data.load(filename, 
            drop_columns=drop_columns, 
            output_column=target)
    scoring = "roc_auc"
    X_train, y_train, X_test, y_test = data.prepare(X, y)
    for algorithm in algorithms.copy():
        algorithm.fit(X_train, y_train)
        score = permutation_importance(algorithm.model, X_test, y_test, n_repeats=100, scoring=scoring)
        df_dict = {"model": len(X.columns)*[algorithm.name], 
                    "feature": X.columns,
                    "mean": score["importances_mean"],
                    "std": score["importances_std"]
                }
        dataframe = pd.DataFrame.from_dict(df_dict) 
        if os.path.isfile("data/results/permutation_{}".format(filename)):
            dataframe.to_csv("data/results/permutation_{}".format(filename), mode="a", index=False, header=False)
        else:
            dataframe.to_csv("data/results/permutation_{}".format(filename), mode="w", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Inicialization --- Grid Searching')
    parser.add_argument("--dataset", default="ufpel", type=str, 
        choices=["ufpel", "fleury", "einstein"])
    parser.add_argument('--target', default="covidpositivo", type=str)
    args = parser.parse_args()
    drop_columns = ["covidpositivo", "pcr", "alta", "obito", "interna", "vm", "vmtempo", "alta"]
    with log_time("Oversampling and Algorithms Testing"):
        filename = "{}.csv".format(args.dataset)
        main(filename, args.target, drop_columns)


