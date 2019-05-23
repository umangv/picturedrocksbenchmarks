.PHONY = all clean 

.DELETE_ON_ERROR:

fig_runtimes.pdf: runtimes.csv runtimes2.csv fig_runtimes.tex
	pdflatex fig_runtimes.tex

clean:
	-rm *.csv
	-rm *.log *.aux *.synctex.gz
	-rm *.pdf

%.csv: ../../output/%.csv
	cp ../../output/$*.csv .
