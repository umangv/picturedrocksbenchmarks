#!/usr/bin/env python3
# coding: utf-8

import click
import pandas as pd
import numpy as np
from itertools import combinations, product
import csv


def flatten_to_n_features(features, target_n):
    """Truncates feature sets with size close to target_n"""
    total_per_class = features.shape[0]
    trunc_per_class = 0
    if len(features.shape) == 1:
        features = features.copy().reshape(-1, 1)
    k = features.shape[1]
    trunc_per_class = int(np.ceil(target_n / k))
    featureset = set()
    while len(featureset) < target_n and trunc_per_class <= total_per_class:
        featureset = set(features[:trunc_per_class].flatten())
        trunc_per_class += 1
    if len(featureset) < target_n:
        print(f"Not enough features: {len(featureset)}, {features.size}")
    return featureset


@click.command()
@click.argument("dataset")
def main(dataset):
    datasets = dataset.split(",")
    if dataset == "all":
        datasets = ["paul", "green", "zheng", "zeisel"]

    demethods = ["logreg", "t-test_overestim_var", "wilcoxon", "rfc"]
    mimethods = ["jmi", "mim", "cife"]
    binmimethods = ["bin" + m for m in mimethods]
    allmethods = demethods + mimethods + binmimethods
    n_methods = len(allmethods)
    method2readable = {
        "binjmi": "JMI(B)",
        "bincife": "CIFE(B)",
        "binmim": "MIM(B)",
        "cife": "CIFE",
        "jmi": "JMI",
        "mim": "MIM",
        "logreg": "LogReg",
        "wilcoxon": "Wilcoxon",
        "t-test_overestim_var": "t-Test",
        "rfc": "RForest",
    }

    for dataset in datasets:
        print(f"Computing for {dataset} Dataset")
        features = {}
        for method in allmethods:
            f = np.load(f"output/{dataset}_{method}_markers_full.npz")
            features[method] = flatten_to_n_features(f["markers"], 100)
        intersections = pd.DataFrame(
            np.zeros((n_methods, n_methods), dtype=int),
            index=allmethods,
            columns=allmethods,
        )
        for i, j in product(allmethods, allmethods):
            w = len(features[i].intersection(features[j]))
            intersections[j][i] = w
        intersections.rename(
            index=method2readable, columns=method2readable, inplace=True
        )
        intersections.to_csv(
            f"output/{dataset}_intersection.csv", quoting=csv.QUOTE_MINIMAL
        )


if __name__ == "__main__":
    main()
