.PHONY: all clean 10x untar

all: 10x zheng17_bulk_lables.txt

10x: filtered_matrices_mex/hg19/matrix.mtx filtered_matrices_mex/hg19/barcodes.tsv filtered_matrices_mex/hg19/genes.tsv

fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz:
	wget http://cf.10xgenomics.com/samples/cell-exp/1.1.0/fresh_68k_pbmc_donor_a/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz
filtered_matrices_mex/hg19/matrix.mtx filtered_matrices_mex/hg19/barcodes.tsv filtered_matrices_mex/hg19/genes.tsv: untar

untar: fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz
	tar -xf fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz

zheng17_bulk_lables.txt:
	wget https://github.com/theislab/scanpy_usage/raw/master/170503_zheng17/data/zheng17_bulk_lables.txt


clean:
	-rm fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz
	-rm filtered_matrices_mex/ -r
	-rm zheng17_bulk_lables.txt
	-rm fresh_68k_bulk_labels.h5ad
