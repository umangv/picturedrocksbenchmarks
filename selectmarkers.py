#!/usr/bin/env python3
# from collections import namedtuple

import click


# ErrorRates = namedtuple("ErrorRates", ["method", "n_markers", "error_rate"])


@click.command()
@click.argument("dataset")
@click.argument("method")
@click.argument("num_features")
def main(dataset, method, num_features):
    import picturedrocks as pr
    from getdata import getdata
    from markerfuncs import (
        get_markers_func,
        get_binmi_markers_func,
        get_mi_markers_func,
        get_rfc_markers_func,
    )

    num_features = int(num_features)
    print(f"Selecting {num_features} markers in {dataset} using {method}")
    adata = getdata(dataset)

    ft = pr.performance.FoldTester(adata)

    ft.loadfolds(f"output/{dataset}_folds.npz")

    demethods = ["t-test_overestim_var", "wilcoxon", "logreg"]
    mimethods = ["cife", "jmi", "mim"]
    micls = {"cife": pr.markers.CIFE, "jmi": pr.markers.JMI, "mim": pr.markers.MIM}

    if method in demethods:
        ft.selectmarkers(get_markers_func(method, num_features))
        ft.savefoldsandmarkers(f"output/{dataset}_{method}.npz")
    elif method in mimethods:
        ft.selectmarkers(get_mi_markers_func(micls[method], num_features))
        ft.savefoldsandmarkers(f"output/{dataset}_{method}.npz")
    elif method == "rfc100":
        ft.selectmarkers(get_rfc_markers_func(100, num_features))
        ft.savefoldsandmarkers(f"output/{dataset}_{method}.npz")
    elif method.startswith("bin"):
        bmethod = method[3:]
        assert bmethod in mimethods
        ft.selectmarkers(get_binmi_markers_func(micls[bmethod], num_features))
        ft.savefoldsandmarkers(f"output/{dataset}_{method}.npz")
    else:
        raise ValueError("Unknown method")


if __name__ == "__main__":
    main()
