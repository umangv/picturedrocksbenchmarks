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
from getdata import getdata


def kbins_discretize(x, k):
    x_is_sparse = scipy.sparse.issparse(x)
    n_obs, n_features = x.shape
    if x_is_sparse:
        x = x.toarray()
    newx = np.zeros(x.shape, dtype=int)
    for j in range(n_features):
        col = x[:, j]
        origcol = col
        bins = []
        cmax = col.max()
        for i in range(k - 1):
            if bins:
                m = bins[-1]
                col = col[col > m]
                if cmax <= m or len(col) == 0:
                    break
            bins.append(np.percentile(col, 100 / (k - i)))
        newcol = np.digitize(origcol, bins, right=True)
        newx[:, j] = newcol
    return newx


def bininfoset(adata, supervised, k=5):
    binx = kbins_discretize(adata.X, 5)
    if supervised:
        return pr.markers.SparseInformationSet(binx, adata.obs["y"].values)
    else:
        return pr.markers.SparseInformationSet(binx)


def compute_interactions(adata, pp):
    supervised = True
    infosetfunc = {"loglog": pr.markers.makeinfoset, "quantile": bininfoset}[pp]
    pr.read.process_clusts(adata)
    if pp == "loglog":
        sc.pp.log1p(adata)
    if adata.shape[1] > 5000:
        tempinfoset = infosetfunc(adata, supervised)
        mim = pr.markers.MIM(tempinfoset)
        pool = np.array(mim.autoselect(5000))
    else:
        pool = np.arange(adata.shape[1])
    infoset = infosetfunc(adata[:, pool].copy(), supervised)
    H = infoset.entropy_wrt(np.arange(0))
    Hofy = infoset.entropy(np.array([-1]))
    Hwrty = infoset.entropy_wrt(np.array([-1]))
    n_feats = infoset.X.shape[1]
    interaction = np.zeros((n_feats, n_feats))
    for i in range(n_feats):
        cur_row = (
            H
            + H[i]
            + Hofy
            - Hwrty
            - Hwrty[i]
            - infoset.entropy_wrt(np.array([i]))
            + infoset.entropy_wrt(np.array([-1, i]))
        )
        interaction[i, :] = cur_row
    return interaction


@click.command()
@click.argument("dataset")
@click.argument("pp")
def main(dataset, pp):
    print(f"Computing interactions in {dataset} using {pp} preprocessing")
    adata = getdata(dataset)
    interaction = compute_interactions(adata, pp)
    np.savez_compressed(
        f"../output/{dataset}_{pp}_interactions.npz", interaction=interaction
    )


if __name__ == "__main__":
    main()
