# Requires pdfLaTeX: aaai2027.sty rejects XeTeX and LuaTeX.
LATEXMK ?= latexmk

.PHONY: pdf clean

pdf:
	$(LATEXMK) -pdf -interaction=nonstopmode -file-line-error -outdir=build main.tex

clean:
	rm -rf build
