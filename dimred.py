#!/usr/bin/env python3
# coding: utf-8

import click
import numpy as np
import picturedrocks as pr
import scanpy.api as sc

from sklearn.manifold import TSNE
from umap import UMAP
from matplotlib import pyplot as plt

from getdata import getdata


def plotprep(adata):
    pr.read.process_clusts(adata)
    adata.obs["clust"] = "Cluster " + adata.obs["y"].astype(str)
    adata.obs["clust"] = adata.obs["clust"].astype("category")
    sc.pp.log1p(adata)


@click.command()
@click.argument("dataset")
def main(dataset):
    adata = getdata(dataset)

    def saveplot(coords, dimred):
        plt.figure()
        plt.scatter(
            coords[:, 0],
            coords[:, 1],
            s=2,
            c=adataproj.obs["y"].values % 9,
            cmap="Set1",
        )
        plt.tick_params(
            axis="both",
            which="both",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
        )
        plt.savefig(
            f"figures/dimred/{dataset}_{alg}_{n_markers}markers_{dimred}.pdf",
            format="pdf",
        )
        plt.savefig(
            f"figures/dimred/{dataset}_{alg}_{n_markers}markers_{dimred}.png",
            format="png",
        )
        plt.close()

    for alg in [
        "cife",
        "bincife",
        "jmi",
        "binmim",
        "logreg",
        "t-test_overestim_var",
        "wilcoxon",
    ]:
        markers = np.load(f"output/{dataset}_{alg}_markers_full.npz")["markers"]
        if len(markers.shape) > 1:
            markers = markers[:, 0].flatten()
        else:
            markers = markers[:10]
        n_markers = len(markers)
        adataproj = adata[:, markers].copy()
        plotprep(adataproj)
        print("Computing PCA coords")
        Xpca = pr.plot.pca(adataproj.X, 2, return_info=False)
        saveplot(Xpca, "pca")
        print("Computing tSNE coords")
        t = TSNE()
        Xtsne = t.fit_transform(adataproj.X.toarray())
        saveplot(Xtsne, "tsne")
        print("Computing UMAP coords")
        u = UMAP()
        Xumap = u.fit_transform(adataproj.X)
        saveplot(Xumap, "umap")


if __name__ == "__main__":
    main()
