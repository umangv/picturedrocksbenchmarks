from collections import namedtuple

import numpy as np
import scipy.sparse
import scanpy.api as sc
import picturedrocks as pr
import click

from getdata import getdata
from markerfuncs import *


ErrorRates = namedtuple("ErrorRates", ["method", "n_markers", "error_rate"])


@click.command()
@click.argument("dataset")
@click.argument("method")
@click.argument("pp")
def main(dataset, method, pp):
    print(f"Selecting markers in {dataset} using {method} and {pp} preprocessing")
    adata = getdata(dataset)

    ft = pr.performance.FoldTester(adata)

    ft.loadfolds(f"../output/{dataset}_folds.npz")

    demethods = ["t-test_overestim_var", "wilcoxon", "logreg"]
    mimethods = ["cife", "jmi", "mim"]
    micls = {"cife": pr.markers.CIFE, "jmi": pr.markers.JMI, "mim": pr.markers.MIM}

    if method in demethods:
        ft.selectmarkers(get_markers_func(method))
        ft.savefoldsandmarkers(f"../output/{dataset}_{method}_{pp}.npz")
    elif method in mimethods:
        ft.selectmarkers(get_mi_markers_func(micls[method], pp))
        ft.savefoldsandmarkers(f"../output/{dataset}_{method}_{pp}.npz")
    elif method == "rfc":
        ft.selectmarkers(get_rfc_markers_func(1000))
        ft.savefoldsandmarkers(f"../output/{dataset}_{method}_{pp}.npz")
    elif method == "rfc100":
        ft.selectmarkers(get_rfc_markers_func(100))
        ft.savefoldsandmarkers(f"../output/{dataset}_{method}_{pp}.npz")
    elif method.startswith("bin"):
        bmethod = method[3:]
        assert bmethod in mimethods
        ft.selectmarkers(get_binmi_markers_func(micls[bmethod], pp))
        ft.savefoldsandmarkers(f"../output/{dataset}_{method}_{pp}.npz")
    else:
        raise ValueError("Unknown method")


if __name__ == "__main__":
    main()
