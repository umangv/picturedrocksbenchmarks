#!/usr/bin/env python

from collections import namedtuple
from itertools import product
import pickle

from tqdm import tqdm
import pandas as pd
import numpy as np
import scanpy.api as sc
import picturedrocks as pr

ErrorRates = namedtuple("ErrorRates", ["method", "n_markers", "error_rate"])

def main():

    demethods = ["t-test_overestim_var", "wilcoxon", "logreg"]
    mimethods = ["cife", "jmi", "mim", "bincife", "binjmi", "binmim"]
    othermethods = ["rfc"]

    columns = {}
    for dataset, method, classifier in tqdm(
        product(
            ["paul", "zeisel", "green", "zheng"],
            demethods + mimethods + othermethods,
            ["nc", "rf"],
        )
    ):
        e = pickle.load(open(f"output/{dataset}_{method}_{classifier}_error.pkl", "rb"))
        columns[f"{dataset}_{method}_{classifier}_error"] = e.error_rate
        columns[f"{dataset}_{method}_n"] = e.n_markers
    errorsdf = pd.DataFrame(columns)

    errorsdf.to_csv("output/errors.csv", index=False)
    print("Wrote error rates csv file at output/errors.csv")


if __name__ == "__main__":
    main()
