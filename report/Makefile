MAIN = main
EXT  = aux bbl blg fdb_latexmk fls log out synctex.gz toc lof lot

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex sections/*.tex
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex
	@rm -f $(foreach e,$(EXT),$(MAIN).$(e))
	@echo "Done. Aux files removed."

clean:
	@rm -f $(foreach e,$(EXT),$(MAIN).$(e))

fullclean: clean
	@rm -f $(MAIN).pdf