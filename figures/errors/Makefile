.PHONY = all rfc ncc clean 

.DELETE_ON_ERROR:

all: rfc ncc
rfc: green.pdf paul.pdf zeisel.pdf zheng.pdf
ncc: greennc.pdf paulnc.pdf zeiselnc.pdf zhengnc.pdf 


clean:
	-rm errors.csv
	-rm *.log *.aux *.synctex.gz
	-rm *.pdf

errors.csv:
	cp ../../output/errors.csv .

%.pdf : %.tex fig_errors.tex errors.csv
	pdflatex $*.tex
