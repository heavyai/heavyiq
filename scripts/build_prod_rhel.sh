# ENV Requirements
## Assumes Python 3.10+ aliased to python3 (Installed)
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source  $SCRIPT_DIR/common_fn.sh
process_args "$@"
# Create New Virtual Environment
python3 -m venv venv
. venv/bin/activate


# Install Requirements
mkdir -p ./dist
cp requirements.txt ./dist/requirements.txt
cp requirements-rhel.txt ./dist/requirements-rhel.txt

# if INTERNALLY_RELEASED_PYHEAVYDB is set
# (see common_fn.sh argument processing)
# 1. grab the whl and store in ./dist
# 2. update the ./dist/requirements.txt file
test_for_internally_release_pyheavydb

# if INCLUDE_ALL_DEPS is set
# (see common_fn.sh argument processing)
# 1. Download all of the deps and store in
# ./packages
# 2. Update ./dist/requirments to use the
# packages stored in ./packages
test_for_include_all_deps ./dist/requirements.txt

pip install -r ./dist/requirements.txt
pip install -r ./dist/requirements-rhel.txt
pip freeze -l > ./dist/requirements.txt

# pip freeze -l inserts an absolute path
# for pyheavydb.  We need a relative path
update_pyheavydb_reference ./dist/requirements.txt

pip install -r ./requirements-build-prod.txt

# Create Obfuscated Build
pyarmor reg pyarmor-regfile-5130.zip

# Note the pyarmor step can create the
# ./dist dir if it doesn't already exist.
pyarmor gen ./heavyiq
pyarmor gen ./heavyrag
cp -r heavyiq/langchain/tokenizer_models/ ./dist/heavyiq/langchain/tokenizer_models/
cp gunicorn.conf.py ./dist/gunicorn.conf.py

# Create version.txt
mkdir -p dist/public
current_timestamp=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
commit_hash=$(git rev-parse --short HEAD)
echo "${current_timestamp}-${commit_hash}" > dist/public/version.txt

# Compress Build Dir adding the packages folder under
# the dist so that the install for heavydb-internal
# works
mv packages dist/.
cd dist
tar -czf ../dist.tgz  .
cd ..
mv dist/packages .
