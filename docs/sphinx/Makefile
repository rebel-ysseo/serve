  
# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = torchserve
SOURCEDIR     = ..
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

docset: html
	doc2dash --name $(SPHINXPROJ) --icon $(SOURCEDIR)/_static/img/pytorch-logo-flame.png --enable-js --online-redirect-url http://pytorch.org/audio/ --force $(BUILDDIR)/html/

	# Manually fix because Zeal doesn't deal well with `icon.png`-only at 2x resolution.
	cp $(SPHINXPROJ).docset/icon.png $(SPHINXPROJ).docset/icon@2x.png
	convert $(SPHINXPROJ).docset/icon@2x.png -resize 16x16 $(SPHINXPROJ).docset/icon.png

.PHONY: help Makefile docset

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	cp ../../SECURITY.md ../security.md
	cp ../../examples/usecases/RAG_based_LLM_serving/README.md ../enhancing_llm_serving_compile_rag.md
	cp ../../examples/usecases/llm_diffusion_serving_app/README.md ../llm_diffusion_serving_app.md
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
