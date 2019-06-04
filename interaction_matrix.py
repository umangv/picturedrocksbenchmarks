#!/usr/bin/env python3

import numpy as np
import scanpy.api as sc
import picturedrocks as pr
import click

from picturedrocks.markers.mutualinformation.infoset import quantile_discretize
from getdata import getdata
from tqdm import tqdm


def bininfoset(adata, supervised, k=5):
    binx = quantile_discretize(adata.X, 5)
    if supervised:
        return pr.markers.SparseInformationSet(binx, adata.obs["y"].values)
    else:
        return pr.markers.SparseInformationSet(binx)


def compute_interactions(adata):
    supervised = True
    infosetfunc = bininfoset
    pr.read.process_clusts(adata)
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
    for i in tqdm(range(n_feats)):
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
def main(dataset):
    print(f"Computing interactions in {dataset}")
    adata = getdata(dataset)
    interaction = compute_interactions(adata)
    np.savez_compressed(f"output/{dataset}_interactions.npz", interaction=interaction)


if __name__ == "__main__":
    main()
