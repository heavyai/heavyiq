# ENV Requirements
# Build without PyArmor obfuscation (for development/testing)
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source  $SCRIPT_DIR/common_fn.sh
process_args "$@"

# Copy to dist
rm -rf dist
mkdir -p dist
cp -r heavyiq dist/heavyiq
cp -r heavyrag dist/heavyrag
cp requirements.txt ./dist/requirements.txt
cp gunicorn.conf.py ./dist/gunicorn.conf.py

# if INTERNALLY_RELEASED_PYHEAVYDB is set
# 1. grab the whl and store in ./dist
# 2. update the ./dist/requirements.txt file
test_for_internally_released_pyheavydb

# if INCLUDE_ALL_DEPS is set
# 1. Download all of the deps and store in ./packages
# 2. Update ./dist/requirements to use the packages stored in ./packages
test_for_include_all_deps ./dist/requirements.txt

# Create version.txt
mkdir -p dist/public
current_timestamp=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
commit_hash=$(git rev-parse --short HEAD)
echo "${current_timestamp}-${commit_hash}" > dist/public/version.txt

# Move packages folder if it exists
if [ -d "packages" ]; then
    mv packages dist/.
fi

# Compress Build Dir
cd dist
tar -czf ../dist.tgz .
cd ..

# Restore packages folder
if [ -d "dist/packages" ]; then
    mv dist/packages .
fi
