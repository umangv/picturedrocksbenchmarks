import os.path
import scanpy.api as sc
from picturedrocks.read import process_clusts


def getdata(dataset):
    basedir = os.path.abspath(os.path.join(__file__, ".."))
    if dataset == "green":
        adata = sc.read(basedir + "/data/green/green.h5ad")
        process_clusts(adata, "CellType")
    elif dataset == "paul":
        adata = sc.read(basedir + "/data/paul/paul.h5ad")
        process_clusts(adata, "paul15_clusters")
    elif dataset == "zeisel":
        adata = sc.read(basedir + "/data/zeisel/zeisel.h5ad")
        process_clusts(adata, "group")
    elif dataset == "zheng":
        adata = sc.read(basedir + "/data/zheng/fresh_68k_bulk_labels.h5ad")
        process_clusts(adata)
    else:
        raise ValueError("No such dataset")
    return adata
