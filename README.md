# PicturedRocks Benchmarking Experiments

This repository contains code for all the benchmarking experiments in our paper "Information Theoretic Feature Selection Methods for Single Cell RNA Sequencing".  We intend to make all our experiments easily reproducible. To this end, we are publishing numerous scripts to generate the figures in our paper.

## Download the data

To download the data from their original sources, you can run `make` inside the data directory:

```bash
cd data
make
cd ..
```
This downloads all four datasets that we consider in our benchmarking. To download an individual dataset (for example, the Green dataset), you can run `make green` instead.

## Compile data to AnnData HDF5 format

We will reuse these AnnData objects numerous times. Rather than reading the data from all of its various data formats, we compile them to a standard AnnData HDF5 files (usually `*.h5ad`). To compile the data into the Ann Data HDF5 format, run `compiledata.py DATASET` with the name of the dataset from the root directory (separate multiple datasets with a space). For example, to compile all datasets

```bash
python compiledata.py paul zeisel zheng green
```
 (Note that some of these can take a very long time on a standard computer)

Observe that files like `data/zeisel/zeisel.h5ad` have now appeared. You may also notice that we have prepared for K-fold cross-validation by preparing a file containing the randomized folds (e.g., `output/zeisel_folds.npz`).

## K-fold Cross Validation

### Select features for k-fold cross validation

To select features via crossvalidation, run `selectmarkers.py DATASET METHOD NUM_FEATURES`. For example, to select 100 features using the MIM algorithm on the Paul dataset:
```bash
python selectmarkers.py paul mim 100
```
We suggest running multiclass methods (CIFE, JMI, MIM, and RFC) with 100 features and binary methods with 10 features (note that these will later be truncated to get feature sets across all class labels). This will generate files like `output/paul_mim.npz`.

### (Re)classify data with f-fold cross validation
The files created at the previous step contain information about the folds used for each dataset and the features selected for each fold. We can use this information to classify points in each fold based on the features selected when excluding that fold. The script here is `classify.py DATASET METHOD CLASSIFIER`. For example, to classify with the Random Forests Classifier:
```bash
python classify.py paul mim rf
```
This will create small files like `output/paul_mim_rf_error.pkl` that contains some tuples with "classification error rate" vs "number of features" plots. Once you have run this for all combination of dataset, method, classifier, you can compile all the results into `output/errors.csv` file by
```bash
python errors2csv.py
```

### Compile Figures

Once the `errors.csv` file exists, you can generate all the many "error rates" figures in the paper by running make in `figures/errors`. You will need to have LaTeX installed for this.

```bash
cd figures/errors
make
cd ../..
```

## Select features on the entire dataset (no cross validation)

To select features on the entire dataset, run `markersfull.py DATASET METHOD NUM_FEATURES`. For example,
```bash
python markersfull.py paul mim 100
```
We suggest running multiclass methods (CIFE, JMI, MIM, and RFC) with 100 features and binary methods with 30 features (note that these will later be truncated to get feature sets across all class labels).

### Intersection tables
As a result of selecting markers on entire datasets, we can look at how much each sets of features intersect. First, we compute the size of each intersection:

```bash
python intersection_table.py all
```
To run this on a specific dataset, replace `all` with the name of the dataset (e.g., `python intersection_table.py zheng`).

The highlighted tables in our paper can be generated via LaTeX:
```bash
cd figures/intersections
make
cd ../..
```

### Runtimes
Runtimes are extracted from the `markers_full` files. They can be extracted at once with
```bash
python extract_runtimes.py
```

This will generate `output/runtimes.csv` and `output/runtimes2.csv`. To generate the bar plots from these csv files, run `make` inside the `figures/runtimes` directory.

```bash
cd figures/runtimes
make
cd ../..
```

### Dimensionality reduction
To generate PCA/tSNE/UMAP plots of datasets with only a few features (10 features total for multiclass methods, 1 feature per class label for binary methods), use `dimred.py DATASET`.

```bash
python dimred.py green
```

This will generate all the combinations of PCA/tSNE/UMAP and PDF/PNG plots for features selected with various methods in the `figures/dimred` directory.

### Histograms of I(x_i; x_j; y)
To compute the I(x_i; x_j; y) matrices for each dataset, use `interaction_matrix.py DATASET`.

```bash
python interaction_matrix.py paul
```

To generate the images in our paper, run `make` in the `figures/interactions/` directory.

```bash
cd figures/interactions/
make
cd ../../
```

## Authors
This code was developed by Umang Varma with guidance from Anna C Gilbert. 
