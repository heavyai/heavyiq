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

# If requested, package and install only the selected pyheavydb wheel.
# requirements.txt remains unchanged and authoritative.
test_for_internally_release_pyheavydb
if [[ -n $PYHEAVYDB_ARCHIVE ]]; then
  pip install --no-deps "./packages/$PYHEAVYDB_ARCHIVE"
fi

pip install -r ./dist/requirements.txt

# Create production source build
copy_source_to_dist

# Create version.txt
mkdir -p dist/public
current_timestamp=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
commit_hash=$(git rev-parse --short HEAD)
echo "${current_timestamp}-${commit_hash}" > dist/public/version.txt

# Include the explicitly selected pyheavydb wheel, when present.
if [[ -d packages ]]; then
  mv packages dist/.
fi
cd dist
tar -czf ../dist.tgz  .
cd ..
if [[ -d dist/packages ]]; then
  mv dist/packages .
fi
