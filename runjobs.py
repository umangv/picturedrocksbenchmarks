#!/usr/bin/env python3

import os
import click
import subprocess

datasets = []
methods = []

meth_dict = {
    "binmi": ["bincife", "binjmi", "binmim"],
    "mi": ["cife", "jmi", "mim"],
    "de": ["t-test_overestim_var", "wilcoxon", "logreg"],
}
meth_dict["all"] = meth_dict["binmi"] + meth_dict["mi"] + meth_dict["de"] + ["rfc"]


@click.group()
@click.option("--dataset")
@click.option("--method")
def cli(dataset, method):
    global datasets, methods
    datasets = dataset.split(",")
    if dataset == "all":
        datasets = ["paul", "green", "zeisel", "zheng"]
    print("Running on the following datasets")
    for ds in datasets:
        print(f" - {ds}")
    if method:
        methods = method.split(",")
        if method in meth_dict:
            methods = meth_dict[method]
        print("Running on the following methods")
        for m in methods:
            print(f" - {m}")


@cli.command()
def selectkfold():
    for ds in datasets:
        for method in methods:
            n_feats = 100 if method in ["cife", "jmi", "mim", "rfc"] else 10
            print(
                f"Selecting {n_feats} features with k-fold CV using {method}"
                f" on {ds} dataset."
            )
            subprocess.run(["python", "selectmarkers.py", ds, method, str(n_feats)])


@cli.command()
def selectfull():
    for ds in datasets:
        for method in methods:
            n_feats = 100 if method in ["cife", "jmi", "mim", "rfc"] else 30
            print(
                f"Selecting {n_feats} features on entire dataset using {method}"
                f" on {ds} dataset"
            )
            subprocess.run(["python", "markersfull.py", ds, method, str(n_feats)])


@cli.command()
def classifykfold():
    for ds in datasets:
        for method in methods:
            for classifier in ["rf", "nc"]:
                print(
                    f"Classify features found with {method} on {ds} dataset"
                    f" using CLASSIFIER={classifier}"
                )
                subprocess.run(["python", "classify.py", ds, method, classifier])


@cli.command()
def interactions():
    for ds in datasets:
        print(f"Computing interaction matrix for {ds} dataset")
        subprocess.run(["python", "interaction_matrix.py", ds])


if __name__ == "__main__":
    cli()
