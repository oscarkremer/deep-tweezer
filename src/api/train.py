import argparse
import sys
import time 
from src import data, models
from src.utils import log_time


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


def main(filename, target, drop_columns, iterations):
    algorithms, sampling_models = get_models_list()
    data.remove("data/results/{}".format(filename))
    X, y = data.load(filename, 
            drop_columns=drop_columns, 
            output_column=target)
    for i in range(iterations):
        X_train_trans, y_train, X_test_trans, y_test = data.prepare(X, y)
        for method in sampling_models.copy():
            X_over, y_over = X_train_trans, y_train
            try:
                method_name = method.name if method else "None"
                if method:
                    X_over, y_over = method.fit(X_train_trans, y_train)
                for algorithm in algorithms.copy():
                    y_score = models.grid_search(X_over, X_test_trans, y_over, algorithm)
                    data.save(y_score, y_test, algorithm.name, method_name, filename)
            except ValueError as error:
                print(error)
            finally:
                pass         
        time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Inicialization --- Grid Searching')
    parser.add_argument("--dataset", default="ufpel", type=str, 
        choices=["ufpel", "fleury", "einstein"])
    parser.add_argument("--iterations", default=10, type=int)
    parser.add_argument('--target', default="covidpositivo", type=str)
    args = parser.parse_args()
    drop_columns = ["covidpositivo", "pcr", "alta", "obito", "interna", "vm", "vmtempo", "alta"]
    with log_time("Oversampling and Algorithms Testing"):
        filename = "{}.csv".format(args.dataset)
        main(filename, args.target, drop_columns, args.iterations)