# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source  $SCRIPT_DIR/common_fn.sh
process_args "$@"
# Create New Virtual Environment
python3 -m venv venv
. venv/bin/activate


# Install Requirements
rm -rf ./dist ./packages
mkdir -p ./dist
cp requirements.txt ./dist/requirements.txt

# If an explicit or legacy pyheavydb override was requested, stage the wheel
# under packages and update only the generated ./dist/requirements.txt.
test_for_internally_release_pyheavydb

# if INCLUDE_ALL_DEPS is set
# (see common_fn.sh argument processing)
# 1. Download all of the deps and store in
# ./packages
# 2. Update ./dist/requirments to use the
# packages stored in ./packages
test_for_include_all_deps ./dist/requirements.txt

pip install -r ./dist/requirements.txt
pip freeze -l > ./dist/requirements.txt

# pip freeze -l inserts an absolute path
# for pyheavydb.  We need a relative path
update_pyheavydb_reference ./dist/requirements.txt

# Create production source build
copy_source_to_dist

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
