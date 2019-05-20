from collections import namedtuple
import time

import numpy as np
import scipy.sparse
import scanpy.api as sc
import picturedrocks as pr
import click

from getdata import getdata
from markerfuncs import *

timepoints = []
timepointnames = []


def _recordtime(name):
    timepointnames.append(name)
    timepoints.append(time.time())

ErrorRates = namedtuple("ErrorRates", ["method", "n_markers", "error_rate"])


@click.command()
@click.argument("dataset")
@click.argument("method")
@click.argument("pp")
@click.argument("n_markers", type=int)
def main(dataset, method, pp, n_markers):
    print(f"Selecting {n_markers} markers in {dataset} using {method} and {pp} preprocessing")
    adata = getdata(dataset)

    demethods = ["t-test_overestim_var", "wilcoxon", "logreg"]
    mimethods = ["cife", "jmi", "mim"]
    micls = {"cife": pr.markers.CIFE, "jmi": pr.markers.JMI, "mim": pr.markers.MIM}

    if method in demethods:
        output = get_markers_func(method, n_markers, cb=_recordtime)(adata)
    elif method in mimethods:
        output = get_mi_markers_func(micls[method], pp, n_markers,
                cb=_recordtime)(adata)
    elif method == "rfc":
        output = get_rfc_markers_func(100, cb=_recordtime)(adata)
    elif method.startswith("bin"):
        bmethod = method[3:]
        output = get_binmi_markers_func(micls[bmethod], pp, n_markers,
                cb=_recordtime)(adata)
    else:
        raise ValueError("Unknown method")
    np.savez(
        f"../output/{dataset}_{method}_{pp}_markers_full.npz",
        markers=output,
        timepoints=timepoints,
        timepointnames=timepointnames,
    )


if __name__ == "__main__":
    main()
