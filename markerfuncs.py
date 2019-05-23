import numpy as np
import scipy.sparse
import scanpy.api as sc
import picturedrocks as pr
from picturedrocks.markers.mutualinformation.infoset import quantile_discretize

from sklearn.ensemble import RandomForestClassifier


def nullcb(*args, **kwargs):
    """Null Callback"""
    pass


def get_mi_markers_func(obj, num_features=100, supervised=True, cb=nullcb):
    def select_markers(adata):
        adata = adata.copy()
        pr.read.process_clusts(adata)
        cb("start")
        binx = quantile_discretize(adata.X, 5)
        cb("discretized")
        binx = scipy.sparse.csc_matrix(binx)
        cb("sparsified")
        if adata.shape[1] > 5000:
            tempinfoset = pr.markers.SparseInformationSet(binx)
            if supervised:
                tempinfoset.set_y(adata.obs["y"].values)
            mim = (
                pr.markers.MIM(tempinfoset)
                if supervised
                else pr.markers.UniEntropy(tempinfoset)
            )
            pool = np.array(mim.autoselect(5000))
            cb("select_pool")
        else:
            pool = np.arange(adata.shape[1])
        cb("pooled")
        infoset = pr.markers.SparseInformationSet(binx[:, pool])
        if supervised:
            infoset.set_y(adata.obs["y"].values)
        featsel = obj(infoset)
        featsel.autoselect(num_features)
        cb("selected")
        return pool[featsel.S]

    return select_markers


def get_binmi_markers_func(obj, num_features=10, cb=nullcb):
    def select_markers(adata):
        adata = adata.copy()
        pr.read.process_clusts(adata)
        cb("start")
        binx = quantile_discretize(adata.X, 5)
        cb("discretized")
        binx = scipy.sparse.csc_matrix(binx)
        tempinfoset = pr.markers.SparseInformationSet(binx)
        cb("sparsified")
        markers = []
        for cat in adata.obs["clust"].cat.categories:
            adata.obs["y"] = (adata.obs["clust"] == cat).astype(int)
            if adata.shape[1] > 5000 and obj != "mim":
                tempinfoset.set_y(adata.obs["y"].values)
                mim = pr.markers.MIM(tempinfoset)
                pool = np.array(mim.autoselect(5000))
                cb("some_select_pool")
            else:
                pool = np.arange(adata.shape[1])
            cb("some_pooled")
            infoset = pr.markers.SparseInformationSet(
                binx[:, pool], adata.obs["y"].values
            )
            featsel = obj(infoset)
            featsel.autoselect(num_features)
            cb("some_selected")
            markers.append(pool[featsel.S].tolist())
        cb("selected")
        return markers

    return select_markers


def get_markers_func(method="t-test_overestim_var", num_features=100, cb=nullcb):
    def select_markers(adata):
        adata = adata.copy()
        cb("start")
        pr.read.process_clusts(adata)
        sc.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)
        cb("preprocessed")
        sc.tl.rank_genes_groups(adata, "clust", method=method)
        cb("selected")
        return [
            [
                adata.var_names.get_loc(gene)
                for gene in adata.uns["rank_genes_groups"]["names"][cat].tolist()[
                    :num_features
                ]
            ]
            for cat in adata.obs["clust"].cat.categories
        ]

    return select_markers


def get_rfc_markers_func(n_estimators=100, num_features=1000, cb=nullcb):
    def select_markers(adata):
        adata = adata.copy()
        cb("start")
        pr.read.process_clusts(adata)
        sc.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)
        cb("preprocessed")
        rfc = RandomForestClassifier(n_estimators)
        rfc.fit(adata.X, adata.obs["y"].values)
        cb("selected")
        importances = rfc.feature_importances_
        return importances.argsort()[::-1][:num_features]

    return select_markers
