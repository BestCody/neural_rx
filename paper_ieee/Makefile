# IEEE version: IEEEtran conference template (IEEEtran.cls 1.8b and IEEEtran.bst from CTAN; bare_conf.tex is the template).
LATEXMK ?= latexmk

.PHONY: pdf figures clean

pdf:
	$(LATEXMK) -pdf -interaction=nonstopmode -file-line-error -outdir=build main.tex

figures:      # Fig. 2 (lower for six pages) and Fig. 3 (modulation summary); Fig. 1 is the one of paper/figures
	cd .. && python scripts/plot_nrx_ieee_figures.py

clean:
	rm -rf build
