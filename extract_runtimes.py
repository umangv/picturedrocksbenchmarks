#!/usr/bin/env python3

import numpy as np
import pickle
import pandas as pd


demethods = ["logreg", "t-test_overestim_var", "wilcoxon", "rfc"]
mimethods = ["jmi", "mim", "cife"]
binmimethods = ["bin" + m for m in mimethods]
allmethods = demethods + mimethods + binmimethods


def get_runtime(dataset, method):
    endindex = -1
    if method in demethods:
        startname = "preprocessed"
    if method in binmimethods or method in mimethods:
        startname = "sparsified"
    data = np.load(f"output/{dataset}_{method}_markers_full.npz")
    startindex = np.argmax(data["timepointnames"] == startname)
    assert data["timepointnames"][startindex] in ["preprocessed", "sparsified"]
    assert data["timepointnames"][endindex] == "selected"
    return data["timepoints"][endindex] - data["timepoints"][startindex]


def get_runtime_df(datasets):
    runtime_arr = np.zeros((len(datasets), len(allmethods)))
    for i, dataset in enumerate(datasets):
        for j, method in enumerate(allmethods):
            try:
                runtime_arr[i, j] = get_runtime(dataset, method)
            except:
                print(f"Couldn't read runtimes for {method} on {dataset} dataset")
    runtime_df = pd.DataFrame(runtime_arr, index=datasets, columns=allmethods)
    runtime_df.index.set_names("dataset", inplace=True)
    return runtime_df


def main():
    get_runtime_df(["paul", "zeisel"]).to_csv("output/runtimes.csv")
    get_runtime_df(["zheng", "green"]).to_csv("output/runtimes2.csv")


if __name__ == "__main__":
    main()
