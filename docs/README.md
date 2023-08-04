# Rebuild API Docs Source

- `sphinx-apidoc -f -o docs/source heavyiq/langchain`

Run one of the following after rebuilding source docs.

- (UNIX): `find docs/source -type f -name "*.rst" -exec grep -l ".. automodule:: langchain" {} \; | xargs sed -i 's/\.\. automodule:: langchain\(\(\.\w*\)\?\)/.. automodule:: heavyiq.langchain\1/g'`

- (MACOSX): `find docs/source -type f -name "*.rst" -exec grep -l ".. automodule:: langchain" {} \; | xargs sed -i '' -E 's/\.\. automodule:: langchain((\.\w*)?)/.. automodule:: heavyiq.langchain\1/g'`


# Build docs:

- `sphinx-build -b html docs/source/ docs/build/html`
