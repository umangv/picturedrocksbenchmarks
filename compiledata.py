#!/usr/bin/env python3

import click
import numpy as np
import pandas as pd
import scanpy.api as sc
import picturedrocks as pr
import scipy.sparse
from anndata import AnnData


@click.group(chain=True)
def main():
    """Compile data from various formats into AnnData HDF5 files"""
    pass

@main.command()
def zheng():
    """Prepare the Zheng dataset
    
    Massively parallel digital transcriptional profiling of single cells. by
    Zheng GX, et al. in Nature Communications. 2017.
    """
    pbmc_68k = sc.read_10x_mtx("data/zheng/filtered_matrices_mex/hg19/")
    bl = pd.read_csv("data/zheng/zheng17_bulk_lables.txt", header=None)
    pbmc_68k.obs["bulk_labels"] = bl.values
    pr.read.process_clusts(pbmc_68k, "bulk_labels")
    sc.write("data/zheng/fresh_68k_bulk_labels.h5ad", pbmc_68k)
    ft = pr.performance.FoldTester(pbmc_68k)
    ft.makefolds(random=True)
    ft.savefolds("output/pbmc68k_folds.npz")


@main.command()
def paul():
    """Prepare the Paul dataset

    Transcriptional Heterogeneity and Lineage Commitment in Myeloid Progenitors.
    by Paul, et al. in Cell. 2015.
    """
    paul = sc.datasets.paul15()
    assert np.allclose(paul.X % 1, 0)
    paul.X = scipy.sparse.csc_matrix(paul.X.astype(int))
    pr.read.process_clusts(paul, "paul15_clusters")
    paul.write("data/paul/paul.h5ad")
    ft = pr.performance.FoldTester(paul)
    ft.makefolds(random=True)
    ft.savefolds("output/paul_folds.npz")


@main.command()
def green():
    """Prepare the Green dataset

    A Comprehensive Roadmap of Murine Spermatogenesis Defined by Single-Cell
    RNA-Seq by Green et al. in Developmental Cell. 2018.
    """
    adata = sc.read_csv(
        "data/green/GSE112393_MergedAdultMouseST25_DGE.txt.gz", delimiter="\t"
    ).T
    adata.X = scipy.sparse.csc_matrix(adata.X)
    df = pd.read_csv(
        "data/green/GSE112393_MergedAdultMouseST25_PerCellAttributes.txt.gz",
        sep="\t",
        skiprows=3,
    )
    df = df.set_index("#CellBarcode")
    adata.obs = adata.obs.merge(
        df, how="left", left_index=True, right_index=True, validate="1:1"
    )
    sc.write("data/green/green.h5ad", adata)

@main.command()
def zeisel():
    """Prepare Zeisel dataset

    Cell types in the mouse cortex and hippocampus revealed by single-cell
    RNA-seq by Zeisel, et al. in Science. 2015. 
    """
    df = pd.read_csv(
        "data/zeisel/expression_mRNA_17-Aug-2014.txt",
        sep="\t",
        header=0,
        index_col=0,
        skiprows=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ).T
    zeisel = AnnData(df.values[1:, :])
    zeisel.obs_names = df.index[1:]
    zeisel.var_names = df.columns
    anndf = pd.read_csv(
        "data/zeisel/expression_mRNA_17-Aug-2014.txt",
        sep="\t",
        header=0,
        index_col=1,
        nrows=10,
    ).T
    annotations = anndf.iloc[1:, :-1]
    zeisel.obs["group"] = annotations["group #"]
    zeisel.obs["sex"] = annotations["sex"]
    annotations.columns
    zeisel.obs["tot mRNA"] = annotations["total mRNA mol"]
    zeisel.obs["age"] = annotations["age"]
    zeisel.obs["diameter"] = annotations["diameter"]
    pr.read.process_clusts(zeisel, "group")
    sc.write("data/zeisel/zeisel.h5ad", zeisel)
    ft = pr.performance.FoldTester(zeisel)
    ft.makefolds(random=True)
    ft.savefolds("output/zeisel_folds.npz")

if __name__ == "__main__":
    main()
