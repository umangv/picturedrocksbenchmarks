# PicturedRocks Benchmarking Experiments

This repository contains code for all the benchmarking experiments in our paper "Information Theoretic Feature Selection Methods for Single Cell RNA Sequencing".

## Usage

More details will be filled in during the next few days. You can start by downloading the datasets in their original format by running

```bash
cd data
make
```
(or, after you are in the data directory, you can run `make green`, `make paul`, `make zheng`, and `make zeisel`).

To compile the data into the Ann Data HDF5 format, change back into the root directory and run `./compiledata.py paul zeisel zheng green`. (Note that some of these can take a very long time on a standard computer)

To select features via crossvalidation, you can run `./selectmarkers paul mim 100` (this selects 100 markers for the Paul dataset using the MIM method).

## Authors
This code was developed by Umang Varma with guidance from Anna C Gilbert. 

