import numpy as np
import scipy.sparse
import scanpy.api as sc
import picturedrocks as pr

from sklearn.ensemble import RandomForestClassifier


def nullcb(*args, **kwargs):
    """Null Callback"""
    pass


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
    binx = kbins_discretize(adata.X, k)
    if supervised:
        return pr.markers.SparseInformationSet(binx, adata.obs["y"].values)
    else:
        return pr.markers.SparseInformationSet(binx)


def get_mi_markers_func(obj, pp, num_feats=100, supervised=True, cb=nullcb):
    assert pp == "quantile"
    infosetfunc = bininfoset

    def select_markers(adata):
        adata = adata.copy()
        pr.read.process_clusts(adata)
        cb("start")
        binx = kbins_discretize(adata.X, 5)
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
        featsel.autoselect(num_feats)
        cb("selected")
        return pool[featsel.S]

    return select_markers


def get_binmi_markers_func(obj, pp, num_feats=10, cb=nullcb):
    assert pp == "quantile"
    infosetfunc = bininfoset

    def select_markers(adata):
        adata = adata.copy()
        pr.read.process_clusts(adata)
        cb("start")
        binx = kbins_discretize(adata.X, 5)
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
            featsel.autoselect(num_feats)
            cb("some_selected")
            markers.append(pool[featsel.S].tolist())
        cb("selected")
        return markers

    return select_markers


def get_markers_func(method="t-test_overestim_var", n_markers=100, cb=nullcb):
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
                    :n_markers
                ]
            ]
            for cat in adata.obs["clust"].cat.categories
        ]

    return select_markers


def get_rfc_markers_func(n_estimators=1000, n_markers=1000, cb=nullcb):
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
        return importances.argsort()[::-1][:n_markers]

    return select_markers
