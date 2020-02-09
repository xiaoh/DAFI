# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = _source
BUILDDIR      = _build
GITIODIR       = .

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean gitio clean-gitio

clean:
	-rm -rf $(BUILDDIR)

gitio:
	@$(SPHINXBUILD) -M "html" "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	-cp -r $(BUILDDIR)/html/* $(GITIODIR)/
	-touch $(GITIODIR)/.nojekyll

clean-gitio:
	-rm -rf $(GITIODIR)/*.html
	-rm -rf $(GITIODIR)/_static
	-rm -rf $(GITIODIR)/_sources
	-rm -rf $(GITIODIR)/_images
	-rm -rf $(GITIODIR)/searchindex.js
	-rm -rf $(GITIODIR)/objects.inv
	-rm -rf $(GITIODIR)/.nojekyll

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
