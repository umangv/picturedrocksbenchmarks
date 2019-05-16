from collections import namedtuple
import gc
import pickle

import numpy as np
import scipy.sparse
import scanpy.api as sc
import picturedrocks as pr
import click

from tqdm import tqdm
from picturedrocks.read import process_clusts
from anndata import AnnData
from sklearn.utils import sparsefuncs
from sklearn.ensemble import RandomForestClassifier as RFC

from getdata import getdata

_toarray = pr.performance._toarray

issparse = scipy.sparse.issparse


def normalize_per_cell(data, counts_per_cell_after=None, copy=False, *args, **kwargs):
    data = data.copy() if copy else data
    counts_per_cell = _toarray(data.X.sum(axis=1)).ravel()
    if counts_per_cell_after is None:
        counts_per_cell_after = np.median(counts_per_cell)
    counts_per_cell /= counts_per_cell_after
    counts_per_cell += (counts_per_cell == 0).astype(int)
    if not issparse(data.X):
        data.X /= counts_per_cell[:, np.newaxis]
    else:
        sparsefuncs.inplace_row_scale(data.X, 1 / counts_per_cell)
    return data if copy else None


class NearestCentroidClassifier:
    """Nearest Centroid Classifier for Cross Validation

    Computes the centroid of each cluster label in the training data, then
    predicts the label of each test data point by finding the nearest centroid.
    """

    def __init__(self):
        self.xkibar = None
        self.pcs = None

    def train(self, adata):
        adata = adata.copy()
        adata.X = _toarray(adata.X)
        sc.pp.log1p(adata)
        sc.pp.pca(adata, 30)
        self.pcs = adata.varm["PCs"]
        adata = process_clusts(adata)
        X = adata.X @ self.pcs
        self.xkibar = np.array(
            [
                X[adata.uns["clusterindices"][k]].mean(axis=0).tolist()
                for k in range(adata.uns["num_clusts"])
            ]
        )

    def test(self, Xtest):
        testdata = AnnData(_toarray(Xtest))
        sc.pp.log1p(testdata)
        X = _toarray(testdata.X)
        X = X @ self.pcs
        dxixk = scipy.spatial.distance.cdist(X, self.xkibar)
        ret = dxixk.argmin(axis=1)
        return ret


class RandomForestClassifier:
    """Random Forests Classifier"""

    def __init__(self):
        self.rfc = RFC(100)

    def train(self, adata):
        adata = adata.copy()
        process_clusts(adata)
        sc.pp.normalize_per_cell(adata, 1000, min_counts=0)
        sc.pp.log1p(adata)
        self.rfc.fit(adata.X, adata.obs["y"].values)

    def test(self, Xtest):
        testdata = AnnData(Xtest)
        sc.pp.normalize_per_cell(testdata, 1000, min_counts=0)
        sc.pp.log1p(testdata)
        return self.rfc.predict(testdata.X)


ErrorRates = namedtuple("ErrorRates", ["method", "n_markers", "error_rate"])


@click.command()
@click.argument("dataset")
@click.argument("method")
@click.argument("pp")
@click.argument("classifier")
def main(dataset, method, pp, classifier):
    print(f"Classifying {dataset} based on markers from {method}")
    adata = getdata(dataset)
    ft = pr.performance.FoldTester(adata)
    # binary methods
    binmethods = [
        "t-test_overestim_var",
        "wilcoxon",
        "logreg",
        "binjmi",
        "bincife",
        "binmim",
    ]
    # multiclass methods
    mcmethods = ["cife", "jmi", "mim", "rfc", "rfc100"]
    classifier_cls = {"nc": NearestCentroidClassifier, "rf": RandomForestClassifier}[
        classifier
    ]

    filename = f"../output/{dataset}_{method}_{pp}.npz"
    n_markers = []
    errorrate = []
    ft = pr.performance.FoldTester(adata)
    ft.loadfoldsandmarkers(filename)
    if method in mcmethods:
        for i in tqdm(range(10, 101, 10), desc=method):
            ftnew = pr.performance.truncatemarkers(ft, i)
            ftnew.classify(classifier_cls)
            gc.collect()
            report = pr.performance.PerformanceReport(adata.obs["y"].values, ftnew.yhat)
            n_markers.append(np.mean([len(m) for m in ftnew.markers]))
            errorrate.append(report.wrong() / report.N)
    elif method in binmethods:
        for i in tqdm(range(1, 11), desc=method):
            ftnew = pr.performance.merge_markers(ft, i)
            ftnew.classify(classifier_cls)
            gc.collect()
            report = pr.performance.PerformanceReport(adata.obs["y"].values, ftnew.yhat)
            n_markers.append(np.mean([len(m) for m in ftnew.markers]))
            errorrate.append(report.wrong() / report.N)
    else:
        raise ValueError("No such method")
    pickle.dump(
        ErrorRates(method, n_markers, errorrate),
        open(f"../output/{dataset}_{method}_{pp}_{classifier}_error.pkl", "wb"),
    )
    print(f"Output written to ../output/{dataset}_{method}_{pp}_{classifier}_error.pkl")


if __name__ == "__main__":
    main()
