# ENV Requirements
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source  $SCRIPT_DIR/common_fn.sh
process_args "$@"

# Copy to dist
rm -rf dist
mkdir -p dist
cp -r heavyiq dist/heavyiq
cp requirements.txt ./dist/requirements.txt

# if INTERNALLY_RELEASED_PYHEAVYDB is set
# 1. grab the whl and store in ./dist
# 2. update the ./dist/requirements.txt file
test_for_internally_released_pyheavydb

# Create version.txt
mkdir -p dist/public
current_timestamp=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
commit_hash=$(git rev-parse --short HEAD)
echo "${current_timestamp}-${commit_hash}" > dist/public/version.txt

# Compress Build Dir
tar -czf dist.tgz dist
