.PHONY = all clean 

.DELETE_ON_ERROR:

all: green_intersection.pdf paul_intersection.pdf zeisel_intersection.pdf zheng_intersection.pdf

%.csv: ../../output/%.csv
	cp ../../output/$*.csv .

%.pdf : %.tex %.csv fig_intersection.tex
	pdflatex $*.tex

clean:
	-rm *.log *.aux *.synctex.gz
	-rm *.pdf
	-rm *.csv

