.PHONY: all clean 

all: GSE112393_MergedAdultMouseST25_DGE.txt.gz GSE112393_MergedAdultMouseST25_PerCellAttributes.txt.gz

GSE112393_MergedAdultMouseST25_DGE.txt.gz:
	wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE112nnn/GSE112393/suppl/GSE112393%5FMergedAdultMouseST25%5FDGE%2Etxt%2Egz

GSE112393_MergedAdultMouseST25_PerCellAttributes.txt.gz:
	wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE112nnn/GSE112393/suppl/GSE112393%5FMergedAdultMouseST25%5FPerCellAttributes%2Etxt%2Egz

clean:
	-rm GSE112393_MergedAdultMouseST25_DGE.txt.gz
	-rm GSE112393_MergedAdultMouseST25_PerCellAttributes.txt.gz
